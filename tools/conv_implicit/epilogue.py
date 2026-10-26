"""Fuse ReLU + requantize into the implicit-GEMM conv: int8 (or bf16) out instead of int32, timed."""
import sys, time, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
exec(open(__file__.rsplit("/", 1)[0] + "/conv_batch.py").read().split("def med(")[0])   # ConvPlan etc.
from pathlib import Path
from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform


def med(fn, reps=30):
    fn(); fn(); ts = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    return sorted(ts)[reps // 2] * 1e3


def ep_schedule(t, lanes32=False):
    """The epilogue herd vectorizes 16 lanes by default; AIE2P's SRS lowers only for 32/64 lanes (i32->i8)."""
    p = kernels.CACHE / f"transform_i8_ep{'_l32' if lanes32 else ''}_k{t.l2_k}_{t.l1_m}x{t.l1_n}.mlir"
    if not p.exists():
        txt = generate_matmul_transform(l1_m=t.l1_m, l1_n=t.l1_n, l2_k=t.l2_k, pack_sizes=(8, 8, 8),
                                        k_loop=True, epilogue=True, accum_type="i32")
        if lanes32:
            assert txt.count("tile_sizes [1, 1, 2, 0]") == 1
            txt = txt.replace("tile_sizes [1, 1, 2, 0]", "tile_sizes [1, 1, 4, 0]")
        p.write_text(txt)
    return str(p)


def launch_ep(p, out, ep, scale, shift):
    t = p.t
    cfg = M["npu_config"]; M["triton"].runtime.driver.set_active(M["driver"])
    saved = (cfg._transform_tiling_script, cfg._air_project_path)
    cfg._transform_tiling_script = ep_schedule(t, ep == "srs"); cfg._air_project_path = kernels.project_dir()
    try:
        conv_kernel.conv_i8_implicit_ep[(p.Mp // t.block_m, p.Np // t.block_n)](
            p.a, out, p.w_npu, p.Np, p.CI, p.KH, p.KW, p.S, p.WP, p.PH, scale, shift, ep,
            BLOCK_M=t.block_m, BLOCK_N=t.block_n, BLOCK_K=t.l2_k)
    finally:
        cfg._transform_tiling_script, cfg._air_project_path = saved


T = Tiling
LAYERS = [("224 layer1", 64, 64, 56, T(1024, 64, 128, 32, 64)), ("224 layer3", 256, 256, 14, T(256, 256, 64, 128, 128)),
          ("1024 layer1", 64, 64, 256, T(1024, 64, 128, 32, 64)), ("1024 layer3", 256, 256, 64, T(512, 256, 128, 64, 128))]
EPS = [a for a in sys.argv[1:]] or ["srs", "f2i8", "scale_bf16"]
for name, ci, co, H, t in LAYERS:
    conv = torch.nn.Conv2d(ci, co, 3, 1, 1, bias=False).eval()
    x = torch.randn(1, ci, H, H).relu()
    p = ConvPlan(conv, H, t, 1)
    p.layout(x); p.launch()
    acc = p.reference()
    assert torch.equal(p.c.cpu(), acc)
    t32 = med(p.launch)
    amax = acc.abs().max().item()
    shift = max(0, int(amax).bit_length() - 7)          # power-of-two requant scale so outputs span int8
    scale = round(127.0 / amax * 4, 6)                  # calibrated per layer -> compile-time constant; ~2x headroom clipped
    line = f"{name}: int32 out {t32:.3f} ms ({p.Mp * p.Np * 4 / 1e6:.2f} MB)"
    for ep in EPS:
        dt = torch.bfloat16 if ep == "scale_bf16" else torch.int8
        out = res.empty((p.Mp, p.Np), dtype=dt)
        try:
            launch_ep(p, out, ep, scale, shift)
            got = out.cpu()
            if ep == "srs":
                want = torch.div(acc + (1 << (shift - 1)), 1 << shift, rounding_mode="floor").clamp(-128, 127).to(torch.int8)
                d = (got.int() - want.int()).abs(); ok = f"max|diff| vs round-half-up={d.max().item()} ({100 * (d > 0).float().mean().item():.2f}% differ)"
            elif ep == "f2i8":
                want = (acc.float() * scale + 0.5).clamp(0, 127).floor().to(torch.int8)
                d = (got.int() - want.int()).abs(); ok = f"max|diff|={d.max().item()} ({100 * (d > 0).float().mean().item():.1f}% off by 1)"
            elif ep == "shift":
                want = (acc >> shift).clamp(0, 127).to(torch.int8); ok = f"exact={torch.equal(got, want)}"
            elif ep == "scale_i8":
                want = (acc.float() * scale + 0.5).clamp(0, 127).to(torch.int8)
                d = (got.int() - want.int()).abs(); ok = f"max|diff|={d.max().item()} n_diff={int((d > 0).sum())}"
            else:
                want = (acc.float() * scale).relu()
                ok = f"rel_err={((got.float() - want).norm() / want.norm()).item():.2e}"
            tep = med(lambda: launch_ep(p, out, ep, scale, shift))
            line += f" | {ep} {tep:.3f} ms ({p.Mp * p.Np * out.element_size() / 1e6:.2f} MB) {ok}"
        except Exception as e:
            lines = [l for l in str(e).splitlines() if "error" in l.lower()] or str(e).splitlines()[-1:]
            line += f" | {ep} FAIL {lines[0][:160]}"
    print(line, flush=True)
