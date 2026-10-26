import triton
import triton.language as tl


@triton.jit
def conv_i8_implicit(X, W, C, N: tl.constexpr, CI: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                     S: tl.constexpr, WP: tl.constexpr, PH: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Implicit-GEMM conv, int8 -> int32. X: S*S phase grids, each PH rows of (WP-wide pixel rows) x CI,
    row-major; W: (KH*KW*CI, N) ordered (ky, kx, ci); C: (M, N) on the WP-wide output grid.
    Tap (ky, kx) reads phase (ky%S, kx%S) shifted by (ky//S)*WP + kx//S rows: a plain block load."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    NCB: tl.constexpr = CI // BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for t in range(0, KH * KW * NCB):
        tap = t // NCB
        cb = t - tap * NCB
        ky = tap // KW
        kx = tap - ky * KW
        row = ((ky % S) * S + (kx % S)) * PH + (ky // S) * WP + (kx // S)
        a = tl.load(X + (offs_m[:, None] + row) * CI + cb * BLOCK_K + offs_k[None, :])
        b = tl.load(W + (t * BLOCK_K + offs_k)[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def conv_i8_implicit_ep(X, C, W, N: tl.constexpr, CI: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                        S: tl.constexpr, WP: tl.constexpr, PH: tl.constexpr, SCALE: tl.constexpr, SHIFT: tl.constexpr,
                        EPILOGUE: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """conv_i8_implicit + the next layer's input made on the core: ReLU and requantize the finished
    int32 sums (K is looped in time on one core, so each accumulator is complete) and store int8/bf16.
    EPILOGUE: "f2i8" -> int8(max(acc*SCALE + .5, 0)), the f32 -> int8 conversion saturates at 127;
    "scale_bf16" -> relu(acc*SCALE) bf16. SCALE must be a compile-time constant (static quantization):
    a runtime scalar reaches the epilogue herd as 0. "shift" / "scale_i8" do not lower on AIE2P
    (aievec.srs i32->i8, arith.minnumf on f32 vectors)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    NCB: tl.constexpr = CI // BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for t in range(0, KH * KW * NCB):
        tap = t // NCB
        cb = t - tap * NCB
        ky = tap // KW
        kx = tap - ky * KW
        row = ((ky % S) * S + (kx % S)) * PH + (ky // S) * WP + (kx // S)
        a = tl.load(X + (offs_m[:, None] + row) * CI + cb * BLOCK_K + offs_k[None, :])
        b = tl.load(W + (t * BLOCK_K + offs_k)[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    if EPILOGUE == "srs":
        tl.store(c_ptrs, (acc >> SHIFT).to(tl.int8))   # one aievec.srs (shift-round-saturate) at 32 lanes
    elif EPILOGUE == "f2i8":
        tl.store(c_ptrs, tl.maximum(acc.to(tl.float32) * SCALE + 0.5, 0.0).to(tl.int8))
    elif EPILOGUE == "shift":
        tl.store(c_ptrs, tl.minimum(tl.maximum(acc >> SHIFT, 0), 127).to(tl.int8))
    elif EPILOGUE == "scale_i8":
        y = acc.to(tl.float32) * SCALE + 0.5
        tl.store(c_ptrs, tl.minimum(tl.maximum(y, 0.0), 127.0).to(tl.int8))
    else:
        tl.store(c_ptrs, tl.maximum(acc.to(tl.float32) * SCALE, 0.0).to(tl.bfloat16))
