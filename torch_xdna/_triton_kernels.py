"""Triton kernels compiled for the NPU by Triton-XDNA (imported lazily by kernels.py)."""

import triton
import triton.language as tl


@triton.jit
def matmul_i8_k(A, B, C, N: tl.constexpr, K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C[M, N] = A[M, K] @ B[K, N], int8 -> int32; M is the grid's, everything row-major."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def matmul_bf16_k(A, B, C, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C[M, N] = A[M, K] @ B[K, N], bf16 -> f32 accumulate; same layout as matmul_i8_k."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
