"""Host side of the one-dispatch OmniVoice backbone step (omni_step.py): weight packing, the
activation buffer, building, running and reading back.

    eng = Engine(model, NL=28, NG=2)    # packs the weights once (cached), builds once (cached)
    logits = eng.run(xc, xu)            # cond / uncond embeddings [len <= 64 NG, 1024] ->
                                        # logits [len, 8256] of each
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omni_step as S  # noqa: E402
from ws_gemm import bfp_decode, bfp_encode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "omni_npu")
TILE, BLK = S.TILE, 72


# ------------------------------------------------------------------ tiles
def bf16_bits(x):
    """float32 -> uint16 bf16 bit patterns (round to nearest even)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def tiles(m):
    """[R, C] float32 (R % 32 == 0, C % 128 == 0) -> [R/32, C/128, 4608] bfp16 tiles
    ([row pair][column block][row block of the pair][72], 8 values of a row per exponent)."""
    R, C = m.shape
    g = bfp_encode(np.ascontiguousarray(m, np.float32).reshape(R // 8, 8, C // 8, 8).transpose(0, 2, 1, 3))
    g = g.reshape(R // 32, 2, 2, C // 128, 16, BLK)          # [rt, p, j, ct, kb, 72]
    return np.ascontiguousarray(g.transpose(0, 3, 1, 4, 2, 5)).reshape(R // 32, C // 128, TILE)


def bfp_encode_t(x):
    """bfp_encode (ws_gemm) of a float32 torch tensor [..., 8] -> uint8 [..., 9], the same
    bits, on torch's threads."""
    import torch

    xb = x.to(torch.bfloat16).to(torch.float32)
    emax = ((xb.view(torch.int32) >> 23) & 0xFF).amax(-1)
    sc = torch.ldexp(torch.ones(emax.shape), (133 - emax).float())
    m = torch.round(xb * sc[..., None])
    over = m.abs().amax(-1) > 127
    emax = torch.where(over, emax + 1, emax)
    sc = torch.where(over, sc * 0.5, sc)
    m = torch.round(xb * sc[..., None])
    small = emax < 8
    m[small] = 0
    emax[small] = 0
    out = torch.empty(x.shape[:-1] + (9,), dtype=torch.uint8)
    out[..., 0] = emax.to(torch.uint8)
    out[..., 1:] = m.to(torch.int8).view(torch.uint8)
    return out


def tiles_t(m):
    """tiles() of a float32 torch tensor [R, C] -> uint8 tensor [R/32, C/128, 4608]."""
    R, C = m.shape
    g = bfp_encode_t(m.reshape(R // 8, 8, C // 8, 8).transpose(1, 2))
    g = g.reshape(R // 32, 2, 2, C // 128, 16, BLK)
    return g.permute(0, 3, 1, 4, 2, 5).reshape(R // 32, C // 128, TILE)


def untile(t):
    """[..., 4608] -> [..., 32, 128] float32."""
    lead = t.shape[:-1]
    g = t.reshape(-1, 2, 16, 2, 8, 9)                        # [n, p, kb, j, row, 9]
    v = bfp_decode(g)                                        # [n, p, kb, j, row, 8]
    v = v.transpose(0, 1, 3, 4, 2, 5).reshape(-1, 32, 128)
    return v.reshape(*lead, 32, 128)


def untile_vt(t):
    """V^T tile [16 d blocks][4 key blocks][72] -> V [32 keys, 128 d]."""
    v = bfp_decode(t.reshape(16, 4, 8, 9))                   # [db, kb, d, key]
    return v.transpose(1, 3, 0, 2).reshape(32, 128)


# ------------------------------------------------------------------ weights
def layer_weights(lay):
    import torch

    with torch.no_grad():
        a, m = lay.self_attn, lay.mlp
        g_in = lay.input_layernorm.weight.float()
        g_post = lay.post_attention_layernorm.weight.float()
        wqkv = (torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).float() * g_in).numpy()
        wo = a.o_proj.weight.float().numpy()
        wg = (m.gate_proj.weight.float() * g_post).numpy()
        wu = (m.up_proj.weight.float() * g_post).numpy()
        wd = m.down_proj.weight.float().numpy()
        gamma = np.concatenate([a.q_norm.weight.float().numpy(), a.k_norm.weight.float().numpy()])
    return wqkv, wo, wg, wu, wd, gamma


def column_stream(wqkv, wo, wg, wu, wd):
    """One layer's weight objects for each column, in the cores' order: [8][480][4608]."""
    tq = tiles(wqkv)                       # [128 n-tiles, 8 k-tiles]
    to = tiles(wo)                         # [32, 16]
    tg, tu = tiles(wg), tiles(wu)          # [96, 8]
    td = tiles(wd)                         # [32, 24]
    out = np.empty((S.NCOL, S.w_objects_per_layer(), TILE), np.uint8)
    for c in range(S.NCOL):
        objs = [tq[:, c]]                                            # u, g -> n-tile 4u+g
        objs.append(to[:, 2 * c:2 * c + 2].reshape(-1, TILE))        # (u, g), ch
        # units alternate gate block i, up block i: (u, g) -> n-tile 4(u >> 1) + g
        gu = [(tu if u & 1 else tg)[4 * (u >> 1) + g, c] for u in range(48) for g in range(4)]
        objs.append(np.stack(gu))
        objs.append(td[:, 3 * c:3 * c + 3].reshape(-1, TILE))
        out[c] = np.concatenate(objs)
    return out


def head_stream(wh):
    """Audio heads [8200 -> 8256, 1024] (final norm folded): [8][258][4608]."""
    wp = np.zeros((S.NHEAD_P, S.D), np.float32)
    wp[:wh.shape[0]] = wh
    t = tiles(wp)                                   # [258, 8]
    return np.ascontiguousarray(t.transpose(1, 0, 2))


def pack_weights(model, NL, path=None):
    """The whole weight buffer (cached on disk): per column, NL layers then the heads."""
    import torch

    key = f"omnivoice_nl{NL}_v2"
    path = path or os.path.join(CACHE, key + ".npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    os.makedirs(CACHE, exist_ok=True)
    wl = S.w_objects_per_layer()
    buf = np.zeros((S.NCOL, NL * wl + S.HEAD_W, TILE), np.uint8)
    llm = model.llm
    for L in range(NL):
        t = time.time()
        wqkv, wo, wg, wu, wd, _ = layer_weights(llm.layers[L])
        buf[:, L * wl:(L + 1) * wl] = column_stream(wqkv, wo, wg, wu, wd)
        print(f"packed layer {L} ({time.time() - t:.1f} s)", flush=True)
    with torch.no_grad():
        wh = (model.audio_heads.weight.float() * llm.norm.weight.float()).numpy()
    buf[:, NL * wl:] = head_stream(wh)
    tmp = path + f".{os.getpid()}.tmp.npy"
    np.save(tmp, buf)
    os.replace(tmp, path)
    return buf


# ------------------------------------------------------------------ activations
def rope_tiles(inv_freq, g=0):
    """Row group g's row blocks 0..3 = positions 64g + 0..31, 64g + 32..63 of each sequence:
    [4][4608] (cos | sin)."""
    out = []
    for rb in range(4):
        pos = np.arange(32) + 32 * (rb & 1) + 64 * g
        ang = pos[:, None].astype(np.float64) * inv_freq[None, :].astype(np.float64)
        m = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)   # [32, 128]
        out.append(tiles(m)[0, 0])
    return np.stack(out)


def x_tiles(x):
    """x [128, 1024] -> [8 units][4 row blocks][4608]."""
    return np.ascontiguousarray(tiles(x).transpose(1, 0, 2))


# ------------------------------------------------------------------ build / run
def kernel_objects(extra=()):
    p = os.environ.get("PEANO_INSTALL_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(np.__file__))), "llvm-aie")
    inc = os.path.join(os.path.dirname(p), "mlir_aie", "include")
    src = os.path.join(HERE, "omni_core.cc")
    out = {}
    for v in ("body", "tail"):
        tag = "".join(e.lstrip("-D").lower() for e in extra)
        obj = os.path.join(HERE, "build", f"omni_core_{v}{tag}.o")
        if not os.path.exists(obj) or os.path.getmtime(obj) < os.path.getmtime(src):
            cmd = [os.path.join(p, "bin", "clang++"), "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf",
                   "-DNDEBUG", "-Wno-deprecated-declarations", "-Wno-unused-function", f"-I{inc}",
                   # frame sizes, so aiecc can size each core's stack from the call graph
                   "-fstack-size-section",
                   "-D__AIE_API_AIE_ADF_HPP__"] + (["-DOMNI_TAIL"] if v == "tail" else []) + \
                  list(extra) + ["-c", src, "-o", obj]
            subprocess.run(cmd, check=True)
        # IRON caches a compiled design by its MLIR, which names the object but not its
        # contents: put the contents' hash in the name so a changed kernel rebuilds
        digest = hashlib.sha256(open(obj, "rb").read()).hexdigest()[:12]
        named = obj.replace(".o", f"_{digest}.o")
        if not os.path.exists(named):
            with open(obj, "rb") as f, open(named, "wb") as g:
                g.write(f.read())
        out[v] = named
    return out


def build(NL, NG, extra=(), trace=0):
    objs = kernel_objects(extra)
    h = hashlib.sha256()
    for f in (os.path.join(HERE, "omni_step.py"), objs["body"], objs["tail"]):
        h.update(open(f, "rb").read())
    out = os.path.join(HERE, "build", f"step_nl{NL}_g{NG}{f'_tr{trace}' if trace else ''}_{h.hexdigest()[:12]}")
    if not os.path.exists(os.path.join(out, "insts.bin")):
        t = time.time()
        params = dict(NL=NL, NG=NG, obj_body=objs["body"], obj_tail=objs["tail"], trace=trace)
        r = subprocess.run([sys.executable, os.path.join(HERE, "omni_step.py"), "build", json.dumps(params), out],
                           capture_output=True, text=True, cwd=os.path.join(HERE, "build"))
        if r.returncode:
            raise RuntimeError((r.stdout + r.stderr)[-8000:])
        print(f"built {os.path.basename(out)} in {time.time() - t:.0f} s", flush=True)
    return out


class Engine:
    """One build (NL layers, NG row groups of 128: sequences of up to 64 NG tokens) loaded on
    the NPU with its buffers."""

    def __init__(self, model, NL=28, NG=1, weights=None, extra=(), trace=0, build_dir=None):
        import pyxrt

        self.NL, self.NG = NL, NG
        self.L = S.Layout(NL, NG)
        self.dir = build_dir or build(NL, NG, extra, trace)
        self.dev = pyxrt.device(0)
        self.xclbin = pyxrt.xclbin(os.path.join(self.dir, "final.xclbin"))
        self.dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(self.dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())
        words = np.fromfile(os.path.join(self.dir, "insts.bin"), dtype=np.uint32)
        self.ibo = pyxrt.bo(self.dev, words.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1))
        self.ibo.write(words.view(np.uint8), 0)
        self.ibo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.ninsts = words.nbytes
        self.TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

        def bo(n, arg):
            b = pyxrt.bo(self.dev, n, pyxrt.bo.host_only, self.kernel.group_id(3 + arg))
            return b, np.frombuffer(b.map(), dtype=np.uint8, count=n)

        self.act, self.actv = bo(self.L.ntiles * TILE, 0)
        w = weights if weights is not None else pack_weights(model, NL)
        self.w, self.wv = bo(w.size, 1)
        self.wv[:] = np.asarray(w).reshape(-1)
        self.w.sync(self.TO)
        self.log, self.logv = bo(NG * S.HEAD_U * 4 * TILE, 2)
        a = self.actv.reshape(-1, TILE)
        a[:] = 0
        llm = model.llm
        for g in range(NG):
            a[self.L["ROPE", g]:self.L["ROPE", g] + 4] = rope_tiles(llm.rotary_emb.inv_freq.float().numpy(), g)
        for L in range(NL):
            # q/k-norm gamma times sqrt(128) (omni_core.cc emit_qkv)
            g = layer_weights(llm.layers[L])[-1].astype(np.float32) * np.float32(np.sqrt(128.0))
            a[self.L["GAMMA"] + L, :512] = bf16_bits(g).view(np.uint8)
        self.act.sync(self.TO)
        args = [3, self.ibo, self.ninsts, self.act, self.w, self.log]
        if trace:
            self.trace, self.tracev = bo(trace, 3)
            self.tracev[:] = 0
            self.trace.sync(self.TO)
            args.append(self.trace)
        self.run_ = pyxrt.run(self.kernel)
        for i, v in enumerate(args):
            self.run_.set_arg(i, v)

    def trace_events(self):
        """The trace of the last run, in Trace Event Format (parsed with the physical MLIR)."""
        from aie.utils.trace.parse import parse_trace

        self.trace.sync(self.FROM)
        mlir = open(os.path.join(self.dir, "final.prj", "input_with_addresses.mlir")).read()
        return parse_trace(self.tracev.view(np.uint32), mlir)

    def region(self, name, n=None, g=0):
        a = self.actv.reshape(-1, TILE)
        return a[self.L[name, g]:self.L[name, g] + (n or 1)]

    def run(self, xc, xu, timing=None):
        """Cond / uncond embeddings [c_len, 1024], [u_len, 1024] (lengths <= 64 NG) -> their
        logits [c_len, 8256], [u_len, 8256] (fp32)."""
        import pyxrt
        import torch

        t0 = time.perf_counter()
        c_len, u_len = len(xc), len(xu)
        assert max(c_len, u_len) <= 64 * self.NG, (c_len, u_len, self.NG)
        a = self.actv.reshape(-1, TILE)
        a[self.L["PARAMS"], :8] = np.array([c_len, u_len], np.int32).view(np.uint8)
        # position p of sequence s -> row group p // 64, row 64 s + p % 64
        x = torch.zeros((2, self.NG, 64, S.D))
        x[0].view(-1, S.D)[:c_len] = torch.as_tensor(xc)
        x[1].view(-1, S.D)[:u_len] = torch.as_tensor(xu)
        t = tiles_t(x.transpose(0, 1).reshape(self.NG * 128, S.D))          # [4 NG rb, 8 units]
        av = torch.from_numpy(a)
        for g in range(self.NG):
            av[self.L["X", g]:self.L["X", g] + 32] = t[4 * g:4 * g + 4].transpose(0, 1).reshape(32, TILE)
        self.act.sync(self.TO)
        t1 = time.perf_counter()
        self.run_.start()
        st = self.run_.wait()
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"step ended in {st}")
        t2 = time.perf_counter()
        self.log.sync(self.FROM)
        self.act.sync(self.FROM)
        lg = torch.from_numpy(self.logv.view(np.int16)).view(self.NG, S.HEAD_U, 4, TILE // 2)[..., :2048]
        # [group][unit][row block of 32][group of 32 columns][rb][nb][8][8]
        #   -> [sequence][64 NG positions][units * 64 columns]
        lg = (lg.to(torch.int32) << 16).view(torch.float32).view(self.NG, S.HEAD_U, 2, 2, 2, 4, 4, 8, 8)
        out = lg.permute(2, 0, 3, 5, 7, 1, 4, 6, 8).reshape(2, self.NG * 64, S.HEAD_U * 64)
        oc, ou = out[0, :c_len].numpy(), out[1, :u_len].numpy()
        if timing is not None:
            timing.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
        return oc, ou
