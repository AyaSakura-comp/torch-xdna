"""Full-array GEMM: int32 output vs the 32-lane SRS epilogue (int8 output), median launch time.

usage: python tools/npu_trace/time_epilogue.py M N K block_m block_n l1_m l1_n l2_k
"""
import os, sys, time, torch
if os.environ.get("PEANO_WRAP"):
    import triton.backends.amd_triton_npu.driver as _drv
    _drv.find_peano_root = lambda: os.environ["PEANO_WRAP"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conv_implicit"))
from ep_isolate import mm_ep, kernels, Tiling, res, Mo, generate_matmul_transform

M, N, K, bm, bn, lm, ln, lk = map(int, sys.argv[1:9])
t = Tiling(bm, bn, lm, ln, lk)
torch.manual_seed(0)
a = torch.randint(-20, 20, (M, K), dtype=torch.int8); b = torch.randint(-20, 20, (K, N), dtype=torch.int8)
ar, br = res.to_npu(a), res.to_npu(b, static=True)
acc = torch._int_mm(a, b)


def med(f, n=30):
    f(); ts = []
    for _ in range(n):
        s0 = time.perf_counter(); f(); ts.append(time.perf_counter() - s0)
    return sorted(ts)[n // 2] * 1e3


cfg = Mo["npu_config"]; Mo["triton"].runtime.driver.set_active(Mo["driver"]); cfg._air_project_path = os.environ.get("KEEP_PROJ") or kernels.project_dir()
c32 = res.empty((M, N), dtype=torch.int32)
cfg._transform_tiling_script = kernels.schedule(t)
f32 = lambda: kernels.mm_i8(ar, br, c32, t)
t32 = med(f32); ok32 = torch.equal(c32.cpu(), acc)
EP = os.environ.get("EP", "srs_only32")
u8 = EP == "srs_relu_u8_32"
c8 = res.empty((M, N), dtype=torch.uint8 if u8 else torch.int8)
sch = kernels.CACHE / f"ep_l32_k{t.l2_k}_{t.l1_m}x{t.l1_n}.mlir"
if not sch.exists():
    txt = generate_matmul_transform(l1_m=t.l1_m, l1_n=t.l1_n, l2_k=t.l2_k, pack_sizes=(8, 8, 8), k_loop=True,
                                    epilogue=True, accum_type="i32")
    sch.write_text(txt.replace("tile_sizes [1, 1, 2, 0]", "tile_sizes [1, 1, 4, 0]"))
def f8():
    cfg._transform_tiling_script = str(sch)
    mm_ep[(M // t.block_m, N // t.block_n)](ar, br, c8, N, K, 0.0, EP, True, BLOCK_M=t.block_m,
                                           BLOCK_N=t.block_n, BLOCK_K=t.l2_k, ACC_I32=True)
t8 = med(f8)
want = {"srs_relu_u8_32": (acc >> 8).clamp(0, 255), "srs_relu32": (acc >> 8).clamp(0, 127),
        "i8_srs32": (acc >> 8).clamp(0, 127), "srs_then_relu32": (acc >> 8).clamp(0, 127)}.get(EP, (acc >> 8).clamp(-128, 127))
dd = (c8.cpu().int() - want).abs()
d = f"{dd.max().item()} ({int((dd > 1).sum())} beyond 1 LSB)"
gop = 2 * M * N * K / 1e9
print(f"{M}x{N}x{K} {t}: int32 out {t32:.3f} ms ({gop / t32:.2f} TOPS, exact={ok32}) | SRS int8 out {t8:.3f} ms "
      f"({gop / t8:.2f} TOPS, max|diff| vs floor-shift {d}) [{EP}]")
