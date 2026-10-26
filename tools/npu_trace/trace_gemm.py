"""Hardware event trace of one Triton-XDNA GEMM launch (aircc -trace-size), parsed with aie.utils.trace.

usage: python tools/npu_trace/trace_gemm.py OUTDIR [M N K block_m block_n l1_m l1_n l2_k]
Writes OUTDIR/trace.txt (raw words) and OUTDIR/trace.json (Perfetto), prints per-tile event cycles.
"""
import json, os, subprocess, sys
import torch
from torch_xdna import kernels
from torch_xdna.kernels import Tiling

out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
M, N, K, bm, bn, lm, ln, lk = (list(map(int, sys.argv[2:10])) or [512, 512, 1152, 512, 256, 128, 64, 128])
t = Tiling(bm, bn, lm, ln, lk)
TS = int(os.environ.get("TRACE_BYTES", 4 << 20))  # trace bytes
assert kernels.available()
if os.environ.get("PEANO_WRAP"):       # compile the cores with tools/npu_trace/peano_wrap (logs / extra opt+llc flags)
    import triton.backends.amd_triton_npu.driver as _drv
    _drv.find_peano_root = lambda: os.environ["PEANO_WRAP"]
res = kernels.resident(); Mo = kernels._mod
torch.manual_seed(0)
a = torch.randint(-20, 20, (M, K), dtype=torch.int8); b = torch.randint(-20, 20, (K, N), dtype=torch.int8)
ar, br = res.to_npu(a), res.to_npu(b, static=True)
cflat = res.empty(M * N + TS // 4, dtype=torch.int32)
cr = cflat[: M * N].view(M, N)
cfg = Mo["npu_config"]; Mo["triton"].runtime.driver.set_active(Mo["driver"])
sched = kernels.schedule(t)
patch = os.environ.get("SCHED_PATCH", "")
if patch:
    txt = open(sched).read()
    if "unrollk" in patch:   # fully unroll the per-core K loop of the register block
        nk = t.l2_k // 8
        anchor = "        transform.loop.unroll %vec_loops_to_unroll#1 factor = 2 : !transform.any_op"
        assert anchor in txt
        txt = txt.replace(anchor, anchor + f"\n        transform.loop.unroll %vec_loops#2 factor = {nk} : !transform.any_op")
    if "nohoist" in patch:   # leave the L1/L2 allocs inside the K loop so aircc can ping-pong them
        hs = "        transform.air.hoist_static_alloc %func_hoist : (!transform.any_op) -> ()"
        assert hs in txt
        txt = txt.replace(hs, "")
    if "blk42" in patch:     # 4x2 register block instead of 2x2
        txt = txt.replace("tile_sizes [2, 2, 1, 0, 0, 0]", "tile_sizes [4, 2, 1, 0, 0, 0]")
        txt = txt.replace("transform.loop.unroll %vec_loops_to_unroll#0 factor = 2", "transform.loop.unroll %vec_loops_to_unroll#0 factor = 4")
    sched = os.path.join(out, f"sched_{patch}.mlir"); open(sched, "w").write(txt)
cfg._transform_tiling_script = sched; cfg._air_project_path = os.path.join(out, "air_project")
cfg.aircc_extra_args = (("-trace-size", str(TS), "-trace-offset", str(M * N * 4)) if TS else ()) + tuple(os.environ.get("EXTRA_AIRCC", "").split())
os.chdir(out)                                  # aiecc writes input_with_addresses.mlir to the cwd
k = Mo["kernel"]
launch = lambda: k[(M // t.block_m, N // t.block_n)](ar, br, cr, N, K, BLOCK_M=t.block_m, BLOCK_N=t.block_n, BLOCK_K=t.l2_k)
cflat.zero_() if hasattr(cflat, "zero_") else None
launch()
import time
ts = []
for _ in range(20):
    s0 = time.perf_counter(); launch(); ts.append(time.perf_counter() - s0)
print(f"launch median {sorted(ts)[10] * 1e3:.3f} ms")
ok = torch.equal(cr.cpu(), torch._int_mm(a, b))
words = cflat.cpu()[M * N:].numpy().view("uint32")
nz = int((words != 0).sum())
print(f"result exact={ok}; trace words nonzero {nz} of {len(words)}")
with open("trace.txt", "w") as f:
    for w in words[: max(nz + 64, 1024)]:
        f.write(f"{w:08x}\n")
mlir = [p for p in ("input_with_addresses.mlir", "air_project/input_with_addresses.mlir") if os.path.exists(p)]
print("mlir for parsing:", mlir)
