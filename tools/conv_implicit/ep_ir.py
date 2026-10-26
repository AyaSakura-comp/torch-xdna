"""Build one GEMM with and without the epilogue into kept project dirs; time both."""
import sys, time, torch, os
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ep_isolate import *
base = "/tmp/claude-1000/-home-chihmin/1d0a47b2-e5c7-47f1-9b80-bc022090458a/scratchpad/epir"
M = N = 512; K = 1152; t = Tiling(512, 256, 128, 64, 128)
a = torch.randint(-20, 20, (M, K), dtype=torch.int8); b = torch.randint(-20, 20, (K, N), dtype=torch.int8)
ar, br = res.to_npu(a), res.to_npu(b, static=True)
import sys as _s
SEL = _s.argv[1].split(",") if len(_s.argv) > 1 else None
for ep, odt, sched_ep in [("srs_relu32", torch.int8, True), ("srs_only32", torch.int8, True), ("srs_relu_i16", torch.int16, True), ("none", torch.int32, False), ("i32_relu", torch.int32, True), ("f32_cscale", torch.float32, True),
                          ("bf16_relu_c", torch.bfloat16, True), ("q_f2i8", torch.int8, True)]:
    if SEL and ep not in SEL:
        continue
    cr = res.empty((M, N), dtype=odt)
    l32 = ep.endswith("32")
    sch = kernels.CACHE / f"epir_{'ep' if sched_ep else 'noep'}{'_l32' if l32 else ''}_{t.l1_m}x{t.l1_n}_{t.l2_k}.mlir"
    if not sch.exists():
        txt = generate_matmul_transform(l1_m=t.l1_m, l1_n=t.l1_n, l2_k=t.l2_k, pack_sizes=(8, 8, 8),
                                        k_loop=True, epilogue=sched_ep, accum_type="i32")
        if l32:
            txt = txt.replace("tile_sizes [1, 1, 2, 0]", "tile_sizes [1, 1, 4, 0]")
        sch.write_text(txt)
    cfg = Mo["npu_config"]; Mo["triton"].runtime.driver.set_active(Mo["driver"])
    cfg._transform_tiling_script = str(sch); cfg._air_project_path = f"{base}/{ep}"
    os.makedirs(f"{base}/{ep}", exist_ok=True)
    f = lambda: mm_ep[(M // t.block_m, N // t.block_n)](ar, br, cr, N, K, 0.01, "i32_relu" if ep == "none" else ep, True,
                                                       BLOCK_M=t.block_m, BLOCK_N=t.block_n, BLOCK_K=t.l2_k, ACC_I32=True)
    if ep == "none":
        # the plain kernel: same as i32_relu without the max -> use kernels.mm_i8 instead
        f = lambda: kernels.mm_i8(ar, br, cr, t, project=f"{base}/{ep}")
    try:
        f()
    except Exception as e:
        ls = [l for l in str(e).splitlines() if ("error" in l.lower() or "ERROR" in l) and "Command" not in l] or str(e).splitlines()[-1:]
        print(f"{ep:12s} FAIL {ls[0][:200]}", flush=True); continue
    got = cr.cpu(); acc = torch._int_mm(a, b)
    want = {"srs_relu32": (acc >> 8).clamp(0, 127), "srs_only32": (acc >> 8).clamp(-128, 127),
            "srs_relu_i16": (acc >> 8).clamp(min=0)}.get(ep)
    chk = "" if want is None else f" exact={torch.equal(got.int(), want.int())} max|diff|={(got.int() - want.int()).abs().max().item()}"
    ts = []
    for _ in range(30):
        s = time.perf_counter(); f(); ts.append(time.perf_counter() - s)
    print(f"{ep:12s} {sorted(ts)[15] * 1e3:.3f} ms  out {M * N * cr.element_size() / 1e6:.2f} MB{chk}", flush=True)
