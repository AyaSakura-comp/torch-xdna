"""Weight-streaming bfp16 GEMM over the whole NPU array: one PDI for every K and N.

C[M, N] = A[M, K] x W[N, K]^T for M = 4 * MT rows (one row chain of 8 cores per MT rows).
Core (c, r) holds K slice c of row block r's activations (loaded once per GEMM) and gets
K slice c of W one group of NG output columns at a time; the 8 cores of a row add their
slices along the cascade (west to east) and the tail (column 7) writes the group out.

  A   column c through memtile c, split into the 4 row blocks
  W   column c's K slice through memtile c, broadcast to the column's 4 cores
  C   the 4 tails through memtile 7

The cores read their trip counts (chunks of the K slice, output groups) from runtime
parameters, and every buffer size is fixed by (MT, NG): the device configuration does not
depend on K or N, only the instruction stream does, so one hardware context runs every
GEMM of a model back to back with no reconfiguration.

usage (IRON env):  python ws_gemm.py test [--mt 32] [--ng 32] [--shapes 1024x4096 3072x1024 ...]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NCOL, NROW = 8, 4
BLK = 72          # bytes of an 8x8 bfp16 block
KS = 128          # k per chunk
KBMAX = 48        # k blocks of a core's slice at K = 3072


def geometry(MT, NG):
    """Bytes of an A chunk, a core's A slice buffer, a W object; floats of a group's sums."""
    a_chunk = MT // 16 * (KS // 8) * 2 * BLK
    abuf = MT // 16 * KBMAX * 2 * BLK
    b_obj = NG // 16 * (KS // 8) * 2 * BLK
    return a_chunk, abuf, b_obj, MT * NG


def flat_tap(total, offset, tensor_len, unit=2304):
    """A contiguous run of `total` bytes at `offset` as shim BD dimensions (wraps < 1024)."""
    from aie.helpers.taplib import TensorAccessPattern

    assert total % unit == 0, (total, unit)
    n = total // unit
    b = next(d for d in range(min(n, 1023), 0, -1) if n % d == 0)
    a = n // b
    assert a <= 1023, (total, unit)
    return TensorAccessPattern((tensor_len,), offset=offset, sizes=[1, a, b, unit],
                               strides=[0, b * unit, unit, 1])


def design(A, Wt, C, *, MT, NG, K, N, obj):
    import aie.iron as iron
    from aie.iron import Buffer, CascadeFlow, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron import WorkerRuntimeBarrier
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    a_chunk, abuf_b, b_obj, accn = geometry(MT, NG)
    nch, ngr = K // 8 // KS, N // NG
    assert K % (8 * KS) == 0 and nch * (KS // 8) <= KBMAX and N % NG == 0
    u8, f32 = np.dtype[np.uint8], np.dtype[np.float32]
    achunk_ty = np.ndarray[(a_chunk,), u8]
    acol_ty = np.ndarray[(NROW * a_chunk,), u8]
    abuf_ty = np.ndarray[(abuf_b,), u8]
    b_ty = np.ndarray[(b_obj,), u8]
    acc_ty = np.ndarray[(accn,), f32]
    cj_ty = np.ndarray[(NROW * accn,), f32]
    rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]

    k_load = Kernel("ws_load_a", obj, [achunk_ty, abuf_ty, np.int32])
    k_zero = Kernel("ws_zero", obj, [acc_ty])
    k_mac = Kernel("ws_mac", obj, [abuf_ty, b_ty, acc_ty, np.int32])
    k_put = Kernel("ws_put", obj, [acc_ty])
    k_get = Kernel("ws_get", obj, [acc_ty])
    k_out = Kernel("ws_out_f32", obj, [acc_ty, acc_ty])

    a3 = [ObjectFifo(acol_ty, depth=2, name=f"a3_{c}") for c in range(NCOL)]
    a1 = [a3[c].cons().split([r * a_chunk for r in range(NROW)], obj_types=[achunk_ty] * NROW,
                             names=[f"a1_{c}_{r}" for r in range(NROW)], depths=[2] * NROW,
                             tile=Tile(c, 1))
          for c in range(NCOL)]
    b3 = [ObjectFifo(b_ty, depth=2, name=f"b3_{c}") for c in range(NCOL)]
    b1 = [b3[c].cons().forward(obj_type=b_ty, depth=2, name=f"b1_{c}", tile=Tile(c, 1))
          for c in range(NCOL)]
    c3 = ObjectFifo(cj_ty, depth=2, name="c3")
    c1 = c3.prod().join([r * accn for r in range(NROW)], obj_types=[acc_ty] * NROW,
                        names=[f"c1_{r}" for r in range(NROW)], depths=[2] * NROW,
                        tile=Tile(NCOL - 1, 1))

    def start(rtp, bar, a, kl, abuf):
        bar.wait_for_value(1)
        n_ch, n_gr = rtp[0], rtp[1]
        bar.release_with_value(1)
        for ch in range_(n_ch):
            x = a.acquire(1)
            kl(x, abuf, ch)
            a.release(1)
        return n_ch, n_gr

    def groups(n_ch, n_gr, b, km, abuf, acc, first, last):
        for _ in range_(n_gr):
            first(acc)
            for ch in range_(n_ch):
                y = b.acquire(1)
                km(abuf, y, acc, ch)
                b.release(1)
            last(acc)

    def head(a, b, rtp, bar, acc, abuf, kl, kz, km, kp):
        n_ch, n_gr = start(rtp, bar, a, kl, abuf)
        groups(n_ch, n_gr, b, km, abuf, acc, kz, kp)

    def mid(a, b, rtp, bar, acc, abuf, kl, kg, km, kp):
        n_ch, n_gr = start(rtp, bar, a, kl, abuf)
        groups(n_ch, n_gr, b, km, abuf, acc, kg, kp)

    def tail(a, b, cout, rtp, bar, acc, abuf, kl, kg, km, ko):
        n_ch, n_gr = start(rtp, bar, a, kl, abuf)

        def out(acc_):
            o = cout.acquire(1)
            ko(acc_, o)
            cout.release(1)

        groups(n_ch, n_gr, b, km, abuf, acc, kg, out)

    rtps, bars, workers = {}, {}, []
    for r in range(NROW):
        row = []
        for c in range(NCOL):
            rtp = rtps[c, r] = Buffer(rtp_ty, name=f"rtp_{c}_{r}", use_write_rtp=True)
            bar = bars[c, r] = WorkerRuntimeBarrier()
            acc = Buffer(acc_ty, name=f"acc_{c}_{r}")
            abuf = Buffer(abuf_ty, name=f"abuf_{c}_{r}")
            a, b = a1[c][r].cons(), b1[c].cons()
            if c == 0:
                fn, args = head, [a, b, rtp, bar, acc, abuf, k_load, k_zero, k_mac, k_put]
            elif c < NCOL - 1:
                fn, args = mid, [a, b, rtp, bar, acc, abuf, k_load, k_get, k_mac, k_put]
            else:
                fn, args = tail, [a, b, c1[r].prod(), rtp, bar, acc, abuf, k_load, k_get, k_mac, k_out]
            row.append(Worker(fn, args, tile=Tile(c, 2 + r), stack_size=0x400))
        for w0, w1 in zip(row, row[1:]):
            CascadeFlow(w0, w1)
        workers += row

    a_col = nch * NROW * a_chunk
    b_col = ngr * nch * b_obj
    a_len, b_len, c_len = NCOL * a_col, NCOL * b_col, ngr * NROW * accn
    a_taps = [flat_tap(a_col, c * a_col, a_len) for c in range(NCOL)]
    b_taps = [flat_tap(b_col, c * b_col, b_len) for c in range(NCOL)]
    c_tap = flat_tap(c_len * 4 // 4, 0, c_len, unit=512)   # fp32 elements

    def sequence(Ai, Wi, Co, *hd):
        aps, bps, cc = hd[:NCOL], hd[NCOL:2 * NCOL], hd[2 * NCOL]
        for (c, r), rtp in rtps.items():
            rtp[0] = nch
            rtp[1] = ngr
            bars[c, r].set(1)
        for c in range(NCOL):
            aps[c].fill(Ai, a_taps[c])
            bps[c].fill(Wi, b_taps[c])
        cc.drain(Co, c_tap, wait=True)

    hd = [f.prod() for f in a3] + [f.prod() for f in b3] + [c3.cons()]
    rt = Runtime(sequence, [np.ndarray[(a_len,), u8], np.ndarray[(b_len,), u8],
                            np.ndarray[(c_len,), f32]] + hd)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def build(params, out_dir):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    def fn(A: In, Wt: In, C: Out, *, MT: CompileTime[int], NG: CompileTime[int],
           K: CompileTime[int], N: CompileTime[int], obj: CompileTime[str]):
        return design(A, Wt, C, MT=MT, NG=NG, K=K, N=N, obj=obj)

    jit = iron.jit(fn, aiecc_flags=["--reclaim-runtime-bds"])
    os.makedirs(out_dir, exist_ok=True)
    jit.specialize(**params).compile(xclbin_path=os.path.join(out_dir, "final.xclbin"),
                                      inst_path=os.path.join(out_dir, "insts.bin"))


# ------------------------------------------------------------------ host side
def peano():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(np.__file__))), "llvm-aie")


def kernel_object(MT, NG):
    p = os.environ.get("PEANO_INSTALL_DIR") or peano()
    inc = os.path.join(os.path.dirname(p), "mlir_aie", "include")
    obj = os.path.join(HERE, "build", f"ws_gemm_mt{MT}_ng{NG}.o")
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    src = os.path.join(HERE, "ws_gemm.cc")
    if not os.path.exists(obj) or os.path.getmtime(obj) < os.path.getmtime(src):
        subprocess.run([os.path.join(p, "bin", "clang++"), "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf",
                        "-DNDEBUG", "-Wno-deprecated-declarations", f"-I{inc}", "-D__AIE_API_AIE_ADF_HPP__",
                        f"-DMT={MT}", f"-DNG={NG}", "-c", src, "-o", obj], check=True)
    return obj


def bf16_round(x):
    """float32 -> float32 values rounded to bf16 (round to nearest even)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def bfp_encode(x):
    """[..., 8] float32 -> [..., 9] uint8 bfp16ebs8 groups (exponent, 8 mantissas), as the
    cores' conversion: bf16 values, exponent of the largest, mantissa = v * 2^(6 - e)."""
    xb = bf16_round(x)
    e = ((xb.view(np.uint32) >> 23) & 0xFF).astype(np.int32)
    emax = e.max(-1)
    sc = np.ldexp(np.float32(1), 6 - (emax - 127)).astype(np.float32)
    m = np.rint(xb * sc[..., None])
    over = np.abs(m).max(-1) > 127
    emax = np.where(over, emax + 1, emax)
    sc = np.where(over, sc * 0.5, sc)
    m = np.rint(xb * sc[..., None])
    small = emax < 8
    m[small] = 0
    emax[small] = 0
    out = np.empty(x.shape[:-1] + (9,), np.uint8)
    out[..., 0] = emax.astype(np.uint8)
    out[..., 1:] = m.astype(np.int8).view(np.uint8)
    return out


def bfp_decode(g):
    """[..., 9] uint8 -> [..., 8] float32."""
    e = g[..., 0].astype(np.int32)
    m = g[..., 1:].view(np.int8).astype(np.float32)
    return m * np.ldexp(np.float32(1), e - 127 - 6)[..., None].astype(np.float32)


def blocks(x):
    """[R, K] float32 (R, K multiples of 8) -> [R/8, K/8, 72] bfp16 blocks, rows of 8 k."""
    R, K = x.shape
    g = bfp_encode(x.reshape(R // 8, 8, K // 8, 8).transpose(0, 2, 1, 3))   # [R/8, K/8, 8 rows, 9]
    return g.reshape(R // 8, K // 8, BLK)


def unblocks(b):
    """[R/8, K/8, 72] -> [R, K] float32."""
    r8, k8, _ = b.shape
    v = bfp_decode(b.reshape(r8, k8, 8, 9))       # [R/8, K/8, 8, 8]
    return v.transpose(0, 2, 1, 3).reshape(r8 * 8, k8 * 8)


def pack_a(x, MT):
    """A [4*MT, K] float32 -> the DDR stream: [col c][chunk][row block r][MT/16][KS/8][2][72]."""
    M, K = x.shape
    nch = K // 8 // KS
    b = blocks(x)                                            # [M/8, K/8, 72]
    b = b.reshape(NROW, MT // 16, 2, NCOL, nch, KS // 8, BLK)  # r, p, j, c, ch, kb
    return np.ascontiguousarray(b.transpose(3, 4, 0, 1, 5, 2, 6)).reshape(-1)


def pack_w(w, NG):
    """W [N, K] float32 -> [col c][group g][chunk][NG/16][KS/8][2][72]."""
    N, K = w.shape
    nch = K // 8 // KS
    b = blocks(w)                                            # [N/8, K/8, 72]
    b = b.reshape(N // NG, NG // 16, 2, NCOL, nch, KS // 8, BLK)   # g, q, j, c, ch, kb
    return np.ascontiguousarray(b.transpose(3, 0, 4, 1, 5, 2, 6)).reshape(-1)


def unpack_c(c, MT, NG, N):
    """[group][row block][MT/8][NG/8][8][8] fp32 -> C [4*MT, N]."""
    ngr = N // NG
    v = c.reshape(ngr, NROW, MT // 8, NG // 8, 8, 8)
    return v.transpose(1, 2, 4, 0, 3, 5).reshape(NROW * MT, N)


class Program:
    """One xclbin on its own hardware context; instruction streams per shape."""

    def __init__(self, xclbin_path):
        import pyxrt

        self.dev = pyxrt.device(0)
        self.xclbin = pyxrt.xclbin(xclbin_path)
        self.dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(self.dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())
        self.insts = {}

    def add_insts(self, key, path):
        import pyxrt

        words = np.fromfile(path, dtype=np.uint32)
        bo = pyxrt.bo(self.dev, words.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1))
        bo.write(words.view(np.uint8), 0)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.insts[key] = (bo, words.nbytes)

    def bo(self, nbytes, arg):
        import pyxrt

        b = pyxrt.bo(self.dev, nbytes, pyxrt.bo.host_only, self.kernel.group_id(3 + arg))
        return b, np.frombuffer(b.map(), dtype=np.uint8, count=nbytes)

    def run(self, key, *bos):
        import pyxrt

        ib, n = self.insts[key]
        h = self.kernel(3, ib, n, *bos)
        st = h.wait()
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"run ended in {st}")


def build_cached(MT, NG, K, N):
    obj = kernel_object(MT, NG)
    out = os.path.join(HERE, "build", f"ws_mt{MT}_ng{NG}_k{K}_n{N}")
    if not os.path.exists(os.path.join(out, "insts.bin")):
        t = time.time()
        r = subprocess.run([sys.executable, __file__, "build", json.dumps(dict(MT=MT, NG=NG, K=K, N=N, obj=obj)), out],
                           capture_output=True, text=True, cwd=os.path.dirname(obj))
        if r.returncode:
            raise RuntimeError((r.stdout + r.stderr)[-5000:])
        print(f"built {os.path.basename(out)} in {time.time() - t:.0f} s", flush=True)
    return out


def test(a):
    import pyxrt

    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes]
    MT, NG = a.mt, a.ng
    M = NROW * MT
    dirs = {s: build_cached(MT, NG, *s) for s in shapes}
    xcl = {s: open(os.path.join(d, "final.xclbin"), "rb").read() for s, d in dirs.items()}
    same = len({hash(v) for v in xcl.values()}) == 1
    print(f"one xclbin for all shapes: {same}")
    prog = Program(os.path.join(dirs[shapes[0]], "final.xclbin"))
    rng = np.random.default_rng(0)
    for s in shapes:
        prog.add_insts(s, os.path.join(dirs[s], "insts.bin"))
    ok = True
    for rep in range(2):
        for K, N in shapes:
            x = rng.standard_normal((M, K)).astype(np.float32)
            w = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
            ap, wp = pack_a(x, MT), pack_w(w, NG)
            ab, av = prog.bo(ap.nbytes, 0)
            wb, wv = prog.bo(wp.nbytes, 1)
            cb, cv = prog.bo(M * N * 4, 2)
            av[:] = ap
            wv[:] = wp
            ab.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            wb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            prog.run((K, N), ab, wb, cb)
            cb.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            got = unpack_c(cv.view(np.float32), MT, NG, N)
            want = unblocks(blocks(x)).astype(np.float64) @ unblocks(blocks(w)).astype(np.float64).T
            err = np.abs(got - want).max() / np.abs(want).max()
            good = err < 1e-4
            ok &= good
            ts = []
            for _ in range(a.iters):
                t = time.perf_counter()
                prog.run((K, N), ab, wb, cb)
                ts.append(time.perf_counter() - t)
            ms = sorted(ts)[len(ts) // 2] * 1e3
            gb = (wp.nbytes + ap.nbytes + M * N * 4) / ms / 1e6
            print(f"M={M} K={K} N={N}: {'PASS' if good else 'FAIL'} (max err {err:.2e}); "
                  f"{ms:.3f} ms, W {wp.nbytes / ms / 1e6:.1f} GB/s, all traffic {gb:.1f} GB/s, "
                  f"{2 * M * K * N / ms / 1e9:.2f} TFLOPS", flush=True)
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(json.loads(sys.argv[2]), sys.argv[3])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--mt", type=int, default=32)
    ap.add_argument("--ng", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", nargs="+", default=["1024x4096", "2048x1024", "1024x6144", "3072x1024"])
    a = ap.parse_args()
    sys.exit(0 if test(a) else 1)
