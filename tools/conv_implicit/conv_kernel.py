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
