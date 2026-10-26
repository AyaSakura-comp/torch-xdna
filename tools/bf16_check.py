"""bf16 NPU GEMM: which tilings compile, are they right (vs fp32 math on the same bf16 inputs), how fast.

usage: python tools/bf16_check.py [--time]
"""
import sys
import time

import torch

from torch_xdna import kernels
from torch_xdna.kernels import Tiling

TILINGS = [Tiling(256, 256, 64, 64, 64), Tiling(256, 512, 64, 128, 64), Tiling(128, 512, 32, 128, 64),
           Tiling(512, 512, 64, 128, 64), Tiling(128, 256, 32, 64, 64), Tiling(256, 512, 64, 128, 128)]
SHAPES = [(256, 4096, 1024), (256, 1024, 3072)]

assert kernels.available(), kernels.unavailable_reason()
res = kernels.resident()
torch.manual_seed(0)
for M, N, K in SHAPES:
    a = torch.randn(M, K).bfloat16()
    w = (torch.randn(K, N) * 0.05).bfloat16()
    ref = a.float() @ w.float()
    cpu = torch.empty(0)
    ar, wr = res.to_npu(a), res.to_npu(w, static=True)
    cr = res.empty((M, N), dtype=torch.float32)
    t_cpu = min((lambda t: (torch.nn.functional.linear(a.float(), w.float().t()), time.perf_counter() - t)[1])(
        time.perf_counter()) for _ in range(20)) * 1e3
    for t in TILINGS:
        if M % t.block_m or N % t.block_n or K % t.l2_k:
            continue
        try:
            kernels.mm_bf16(ar, wr, cr, t)
            err = ((cr.cpu() - ref).norm() / ref.norm()).item()
            ms = float("nan")
            if "--time" in sys.argv:
                ts = []
                for _ in range(15):
                    s = time.perf_counter()
                    kernels.mm_bf16(ar, wr, cr, t)
                    ts.append(time.perf_counter() - s)
                ms = sorted(ts)[7] * 1e3
            print(f"{M}x{N}x{K} {t}: rel err {err:.2e}  npu {ms:.3f} ms  (cpu fp32 {t_cpu:.3f} ms)", flush=True)
        except Exception as e:
            lines = [l for l in str(e).splitlines() if "error" in l.lower()]
            print(f"{M}x{N}x{K} {t}: FAIL {(lines[0] if lines else str(e).splitlines()[-1])[:160]}", flush=True)
