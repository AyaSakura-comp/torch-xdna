"""Median launch time of one Triton-XDNA int8 GEMM (no trace), for A/B of compiler settings.

usage: python tools/npu_trace/time_gemm.py M N K block_m block_n l1_m l1_n l2_k
env: PEANO_WRAP (use tools/npu_trace/peano_wrap), SCHED_PATCH, TRITON_CACHE_DIR
"""
import os, sys, time, torch
from torch_xdna import kernels
from torch_xdna.kernels import Tiling
M, N, K, bm, bn, lm, ln, lk = map(int, sys.argv[1:9])
t = Tiling(bm, bn, lm, ln, lk)
assert kernels.available()
if os.environ.get("PEANO_WRAP"):
    import triton.backends.amd_triton_npu.driver as _drv
    _drv.find_peano_root = lambda: os.environ["PEANO_WRAP"]
res = kernels.resident(); Mo = kernels._mod
torch.manual_seed(0)
a = torch.randint(-20, 20, (M, K), dtype=torch.int8); b = torch.randint(-20, 20, (K, N), dtype=torch.int8)
ar, br = res.to_npu(a), res.to_npu(b, static=True); cr = res.empty((M, N), dtype=torch.int32)
sched = kernels.schedule(t)
cfg = Mo["npu_config"]; Mo["triton"].runtime.driver.set_active(Mo["driver"])
cfg._transform_tiling_script = sched; cfg._air_project_path = kernels.project_dir()
k = Mo["kernel"]
launch = lambda: k[(M // t.block_m, N // t.block_n)](ar, br, cr, N, K, BLOCK_M=t.block_m, BLOCK_N=t.block_n, BLOCK_K=t.l2_k)
launch()
ok = torch.equal(cr.cpu(), torch._int_mm(a, b))
ts = []
for _ in range(40):
    s0 = time.perf_counter(); launch(); ts.append(time.perf_counter() - s0)
ms = sorted(ts)[20] * 1e3
print(f"{M}x{N}x{K} {t}: exact={ok} median {ms:.3f} ms  {2 * M * N * K / ms / 1e9:.2f} TOPS")
