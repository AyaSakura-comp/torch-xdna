"""Which GEMM + epilogue combinations does Triton-XDNA get right? (plain matmul, no conv)"""
import sys, time, torch
import triton, triton.language as tl
from torch_xdna import kernels
from torch_xdna.kernels import Tiling
from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform

assert kernels.available()
res = kernels.resident(); Mo = kernels._mod
torch.manual_seed(0)


@triton.jit
def mm_ep(A, B, C, N: tl.constexpr, K: tl.constexpr, SCALE, EP: tl.constexpr, KLOOP: tl.constexpr,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_I32: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if KLOOP:
        offs_k = tl.arange(0, BLOCK_K)
        if ACC_I32:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K // BLOCK_K):
            a = tl.load(A + offs_m[:, None] * K + k * BLOCK_K + offs_k[None, :])
            b = tl.load(B + (k * BLOCK_K + offs_k)[:, None] * N + offs_n[None, :])
            acc = tl.dot(a, b, acc)
    else:
        offs_k = tl.arange(0, K)
        acc = tl.dot(tl.load(A + offs_m[:, None] * K + offs_k[None, :]), tl.load(B + offs_k[:, None] * N + offs_n[None, :]))
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    if EP == "f32":
        tl.store(c_ptrs, acc.to(tl.float32) * SCALE)
    elif EP == "f32_relu":
        tl.store(c_ptrs, tl.maximum(acc.to(tl.float32) * SCALE, 0.0))
    elif EP == "bf16":
        tl.store(c_ptrs, (acc.to(tl.float32) * SCALE).to(tl.bfloat16))
    elif EP == "i32_relu":
        tl.store(c_ptrs, tl.maximum(acc, 0))
    elif EP == "q_f2i8":
        tl.store(c_ptrs, tl.maximum(acc.to(tl.float32) * 0.01 + 0.5, 0.0).to(tl.int8))
    elif EP == "bf16_relu_c":
        tl.store(c_ptrs, tl.maximum(acc.to(tl.float32) * 0.01, 0.0).to(tl.bfloat16))
    elif EP == "q_bf16min":
        y = tl.maximum(acc.to(tl.float32) * 0.01, 0.0).to(tl.bfloat16)
        tl.store(c_ptrs, tl.minimum(y, 127.0).to(tl.int8))
    elif EP == "srs_relu32":
        tl.store(c_ptrs, tl.maximum((acc >> 8).to(tl.int8), 0))
    elif EP == "srs_only32":
        tl.store(c_ptrs, (acc >> 8).to(tl.int8))
    elif EP == "srs_then_relu32":
        y = tl.minimum(tl.maximum(acc >> 8, -128), 127).to(tl.int8)
        tl.store(c_ptrs, tl.maximum(y, 0))
    elif EP == "srs_relu_u8_32":
        tl.store(c_ptrs, tl.minimum(tl.maximum(acc >> 8, 0), 255).to(tl.uint8))
    elif EP == "srs_relu_i16":
        tl.store(c_ptrs, tl.maximum((acc >> 8).to(tl.int16), 0))
    elif EP == "i8_srs32":
        tl.store(c_ptrs, (tl.minimum(tl.maximum(acc >> 8, 0), 127)).to(tl.int8))
    elif EP == "f32_plain":
        tl.store(c_ptrs, acc.to(tl.float32))
    elif EP == "f32_cscale":
        tl.store(c_ptrs, acc.to(tl.float32) * 0.01)
    elif EP == "i32_mul":
        tl.store(c_ptrs, acc * 3)
    elif EP == "i8_trunc":
        tl.store(c_ptrs, (acc >> 8).to(tl.int8))
    elif EP == "i16_shift":
        tl.store(c_ptrs, (tl.minimum(tl.maximum(acc >> 8, 0), 127)).to(tl.int16))
    elif EP == "i8_shift":
        tl.store(c_ptrs, (tl.minimum(tl.maximum(acc >> 8, 0), 127)).to(tl.int8))


def run(dtype, kloop, ep, M=256, N=256, K=256, t=Tiling(256, 256, 64, 64, 128)):
    i8 = dtype == "i8"
    if i8:
        a = torch.randint(-20, 20, (M, K), dtype=torch.int8); b = torch.randint(-20, 20, (K, N), dtype=torch.int8)
        acc = a.int() @ b.int() if False else torch._int_mm(a, b)
    else:
        a = torch.randn(M, K).bfloat16(); b = (torch.randn(K, N) * 0.1).bfloat16()
        acc = a.float() @ b.float()
    scale = 0.01
    odt = {"srs_relu_u8_32": torch.uint8, "srs_relu32": torch.int8, "srs_only32": torch.int8, "srs_relu_i16": torch.int16, "q_f2i8": torch.int8, "bf16_relu_c": torch.bfloat16, "q_bf16min": torch.int8, "i8_srs32": torch.int8, "f32_plain": torch.float32, "f32_cscale": torch.float32, "i32_mul": torch.int32, "i8_trunc": torch.int8,
           "i16_shift": torch.int16, "f32": torch.float32, "f32_relu": torch.float32, "bf16": torch.bfloat16, "i32_relu": torch.int32,
           "i8_shift": torch.int8}[ep]
    want = {"srs_relu_u8_32": lambda: (acc >> 8).clamp(0, 255), "srs_relu32": lambda: (acc >> 8).clamp(0, 127), "srs_only32": lambda: (acc >> 8).clamp(-128, 127),
            "srs_relu_i16": lambda: (acc >> 8).clamp(min=0), "q_f2i8": lambda: (acc.float() * 0.01 + 0.5).clamp(min=0).floor().clamp(max=127),
            "bf16_relu_c": lambda: (acc.float() * 0.01).relu(),
            "q_bf16min": lambda: (acc.float() * 0.01).relu().bfloat16().float().clamp(max=127).trunc(),
            "i8_srs32": lambda: (acc >> 8).clamp(0, 127), "f32_plain": lambda: acc.float(), "f32_cscale": lambda: acc.float() * 0.01, "i32_mul": lambda: acc * 3,
            "i8_trunc": lambda: (acc >> 8).to(torch.int8), "i16_shift": lambda: (acc >> 8).clamp(0, 127),
            "f32": lambda: acc.float() * scale, "f32_relu": lambda: (acc.float() * scale).relu(),
            "bf16": lambda: (acc.float() * scale), "i32_relu": lambda: acc.relu() if i8 else (acc).relu(),
            "i8_shift": lambda: (acc >> 8).clamp(0, 127)}[ep]()
    ar, br = res.to_npu(a), res.to_npu(b, static=True)
    cr = res.empty((M, N), dtype=odt)
    acc_t = "i32" if i8 else "f32"
    sch = kernels.CACHE / f"iso_{acc_t}_{'k' if kloop else 'nok'}_ep_{t.l1_m}x{t.l1_n}_{t.l2_k}.mlir"
    lanes32 = ep.endswith("32")
    sch = sch.with_name(sch.stem + ("_l32" if lanes32 else "") + ".mlir")
    if not sch.exists():
        txt = generate_matmul_transform(l1_m=t.l1_m, l1_n=t.l1_n, l2_k=t.l2_k, pack_sizes=(8, 8, 8),
                                        k_loop=kloop, epilogue=True, accum_type=acc_t)
        if lanes32:
            assert txt.count("tile_sizes [1, 1, 2, 0]") == 1
            txt = txt.replace("tile_sizes [1, 1, 2, 0]", "tile_sizes [1, 1, 4, 0]")
        sch.write_text(txt)
    cfg = Mo["npu_config"]; Mo["triton"].runtime.driver.set_active(Mo["driver"])
    cfg._transform_tiling_script = str(sch); cfg._air_project_path = kernels.project_dir()
    try:
        mm_ep[(M // t.block_m, N // t.block_n)](ar, br, cr, N, K, scale, ep, kloop, BLOCK_M=t.block_m, BLOCK_N=t.block_n,
                                               BLOCK_K=t.l2_k, ACC_I32=i8)
        got = cr.cpu().float(); w = want.float()
        if odt == torch.int8:
            d = (got - w).abs()
            return f"max|diff| {d.max().item():.0f}, {int((d > 0).sum())} of {d.numel()} differ, nonzero {int((w != 0).sum())}"
        err = ((got - w).norm() / w.norm().clamp(min=1e-9)).item()
        return f"rel_err {err:.2e}" + (" OK" if err < 2e-2 else " WRONG")
    except Exception as e:
        lines = [l for l in str(e).splitlines() if "error" in l.lower() and "Command" not in l] or str(e).splitlines()[-1:]
        return "FAIL " + lines[0][:150]


if __name__ == "__main__":
    for dtype, kloop, ep in [(d, k, e) for d in sys.argv[1].split(",") for k in (True,) for e in sys.argv[2].split(",")
                             if not (d == "bf16" and e.startswith(("i8", "i16", "i32")))]:
        print(f"{dtype:5s} kloop={kloop!s:5s} {ep:9s}: {run(dtype, kloop, ep)}", flush=True)
