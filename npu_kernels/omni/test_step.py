"""Check tl.extra.npu.qwen3_stack on OmniVoice's backbone against a CPU reference with the same
bfp16 rounding points.

usage (IRON env + OmniVoice on the path):
    python test_step.py [--nl 1] [--text short]
Prints, per intermediate of the last layer and for the logits, the relative error (norm of
the difference over the norm of the reference) over the positions that exist (cond
0..c_len-1, uncond 0..u_len-1). The op's build is the one with the fewest row groups that fit
the text; its intermediates are read back from its activation buffer.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/chihmin/src/torch-xdna/examples")
from omni_npu import stack_op  # noqa: E402
from omni_sim import bf16, bfp16  # noqa: E402
from triton.language.extra.npu import qwen3_stack as q3  # noqa: E402


def capture(model, text):
    """The first forward() inputs of generate(text)."""
    calls = []
    orig = model.forward

    def rec(**kw):
        calls.append({k: v.clone() for k, v in kw.items() if torch.is_tensor(v)})
        raise StopIteration

    model.forward = rec
    try:
        model.generate(text=text, num_step=8)
    except StopIteration:
        pass
    model.forward = orig
    return calls[0]


def embeddings(model, kw):
    """cond [c_len, 1024], uncond [u_len, 1024] float32."""
    with torch.no_grad():
        emb = model._prepare_embed_inputs(kw["input_ids"], kw["audio_mask"]).float()
    mask = kw["attention_mask"]
    c_len = int(mask[0, 0, 0].sum())
    u_len = int(mask[1, 0, 0].sum())
    return emb[0, :c_len].numpy(), emb[1, :u_len].numpy()


def reference(model, x, NL):
    """The NPU's math on the CPU for one sequence x [n, 1024]: dict of the last layer's
    intermediates + logits."""
    llm = model.llm
    eps = 1e-6
    n = len(x)
    xr = bfp16(torch.from_numpy(np.ascontiguousarray(x)))
    inv = llm.rotary_emb.inv_freq.float()
    ang = torch.arange(n).float()[:, None] * inv[None, :]
    cos, sin = bfp16(torch.cat([ang.cos(), ang.sin()], 1)).chunk(2, 1)
    out = {}
    for L in range(NL):
        lay = llm.layers[L]
        a, m = lay.self_attn, lay.mlp
        g_in, g_post = lay.input_layernorm.weight.detach().float(), lay.post_attention_layernorm.weight.detach().float()
        wqkv = bfp16(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach().float() * g_in)
        wo, wd = bfp16(a.o_proj.weight.detach().float()), bfp16(m.down_proj.weight.detach().float())
        wg = bfp16(m.gate_proj.weight.detach().float() * g_post)
        wu = bfp16(m.up_proj.weight.detach().float() * g_post)
        gq, gk = a.q_norm.weight.detach().float(), a.k_norm.weight.detach().float()
        r = torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)
        heads = bfp16(bf16((xr @ wqkv.T) * r)).view(n, 32, 128)       # the QKV GEMM's tiles

        def norm_rope(h, g):
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * g
            a, b = h[..., :64], h[..., 64:]
            c, s = cos[:, None, :], sin[:, None, :]
            return bfp16(bf16(torch.cat([a * c - b * s, b * c + a * s], -1)))

        q = norm_rope(heads[:, :16], gq)
        k = norm_rope(heads[:, 16:24], gk)
        v = heads[:, 24:]                                             # bfp16 along d
        vt = bfp16(v.transpose(0, 2)).transpose(0, 2)                 # groups of 8 keys
        o = torch.zeros(n, 16, 128)
        for h in range(16):
            p = torch.softmax((q[:, h] @ k[:, h // 2].T) / math.sqrt(128), -1)
            o[:, h] = p @ vt[:, h // 2]
        o = bfp16(o.reshape(n, 2048))
        x1 = bfp16(bf16(xr + o @ wo.T))
        r = torch.rsqrt(x1.pow(2).mean(-1, keepdim=True) + eps)
        g = bfp16(bf16((x1 @ wg.T) * r))
        u = bfp16(bf16((x1 @ wu.T) * r))
        hh = bfp16(bf16(bf16(torch.nn.functional.silu(g)) * u))
        x2 = bfp16(bf16(x1 + hh @ wd.T))
        out = dict(q=q, k=k, v=v, o=o, x1=x1, gate=g, up=u, x2=x2)
        xr = x2
    with torch.no_grad():
        wh = bfp16((model.audio_heads.weight.float() * llm.norm.weight.float()))
    r = torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)
    out["logits"] = (xr @ wh.T) * r
    return {k: v.numpy() for k, v in out.items()}


def seq_rows(t, NG):
    """Decoded [NG, U, 4 rb, 32, 128] -> [2 sequences, 64 NG positions, U * 128]."""
    U = t.shape[1]
    t = t.reshape(NG, U, 2, 2, 32, 128)                               # [g, u, s, r2, i, col]
    return t.permute(2, 0, 3, 4, 1, 5).reshape(2, NG * 64, U * 128)


def npu_intermediates(prog):
    """The last layer's intermediates a build of the op left in its activation buffer (numpy)."""
    S, TILE, NG = q3._design(), q3.TILE, prog.NG
    reg = lambda name, n: torch.from_numpy(np.stack([prog.region(name, n, g) for g in range(NG)]))  # noqa: E731
    out = {}
    qr = reg("QR", 160).reshape(NG, 8, S.QR_COL, TILE)                 # [g, c, tile]
    q = qr[:, :, S.QR_Q:S.QR_Q + 8].reshape(NG, 16, 4, TILE)           # heads 2c, 2c + 1
    out["q"] = seq_rows(q3.untile(q), NG).reshape(2, -1, 16, 128)
    kv = qr[:, :, S.QR_KV:S.QR_KV + 8].reshape(NG, 8, 2, 2, 2, TILE)     # [g, c, s, K/V, r2]
    k = q3.untile(kv[:, :, :, 0])                                     # [g, c, s, r2, 32, 128]
    out["k"] = k.permute(2, 0, 3, 4, 1, 5).reshape(2, NG * 64, 8, 128)
    v = torch.stack([q3.untile_vt(t) for t in kv[:, :, :, 1].reshape(-1, TILE)]).reshape(NG, 8, 2, 2, 32, 128)
    out["v"] = v.permute(2, 0, 3, 4, 1, 5).reshape(2, NG * 64, 8, 128)
    out["o"] = seq_rows(q3.untile(reg("ATT", 64).reshape(NG, 16, 4, TILE)), NG)
    out["x1"] = seq_rows(q3.untile(reg("X1", 32).reshape(NG, 8, 4, TILE)), NG)
    gu = q3.untile(reg("GU", 192).reshape(NG, 24, 2, 4, TILE))      # [g, i, gate/up, rb, 32, 128]
    out["gate"] = seq_rows(gu[:, :, 0], NG)
    out["up"] = seq_rows(gu[:, :, 1], NG)
    out["x2"] = seq_rows(q3.untile(reg("X", 32).reshape(NG, 8, 4, TILE)), NG)
    return {k_: v_.numpy() for k_, v_ in out.items()}


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", type=int, default=1)
    ap.add_argument("--text", default="short")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    from omnivoice import OmniVoice
    from omnivoice_xdna import TEXTS

    torch.set_num_threads(16)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
    xc, xu = embeddings(model, capture(model, TEXTS[a.text]))
    t = time.time()
    op = stack_op(model, a.nl)
    NG = op.groups_for(len(xc), len(xu))
    print(f"c_len {len(xc)} u_len {len(xu)} NG {NG}; op ready in {time.time() - t:.1f} s", flush=True)
    xc_t, xu_t = torch.from_numpy(xc), torch.from_numpy(xu)
    lc, lu = op(xc_t, xu_t)
    timing = []
    for _ in range(a.reps):
        t = time.perf_counter()
        op(xc_t, xu_t)
        timing.append((time.perf_counter() - t, op.last_npu_s))
    got = npu_intermediates(op.program(NG))
    for s_, (x, lg) in enumerate(((xc, lc), (xu, lu))):
        ref = reference(model, x, a.nl)
        n = len(x)
        errs = [f"{name} {rel(got[name][s_, :n], ref[name]):.2e}" for name in
                ("q", "k", "v", "o", "x1", "gate", "up", "x2")]
        errs.append(f"logits {rel(lg.numpy(), ref['logits']):.2e}")
        print(("cond   " if s_ == 0 else "uncond ") + "  ".join(errs), flush=True)
    tm = np.array(timing)
    print(f"step: {1e3 * np.median(tm[:, 0]):.2f} ms per call, NPU {1e3 * np.median(tm[:, 1]):.2f} ms", flush=True)


if __name__ == "__main__":
    main()
