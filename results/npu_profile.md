# Profiling the Triton-XDNA GEMM on the NPU (2026-10-26)

Hardware event trace of the AIE cores (tools/npu_trace/), int8 GEMM `matmul_i8_k`, and the
compiler fixes it led to.

## How the trace is taken

* `aircc -trace-size N -trace-offset <bytes of C>` programs every core's trace unit and routes
  the packets to the third kernel argument (C) after the given offset; `trace_gemm.py` allocates C
  with room for them. aiecc writes `input_with_addresses.mlir` to the cwd, which
  `python -m aie.utils.trace.parse` needs.
* aircc hard-codes the core events (37,34,33,1 / 79,75,45,44). `retrace.py` rewrites the two
  event-select words in the cached ELF and in the MLIR, reruns from the cache and parses:
  INSTR_VECTOR, ACTIVE, LOCK_STALL, MEMORY_STALL / STREAM_STALL, INSTR_LOAD, INSTR_STORE, DISABLED.
* A 16- or 32-core herd fails to route the trace packets ("packet flows can deadlock"), so the
  traces are 2x2 herds (4 cores); full-array runs are timed only.
* One VMAC is 512 int8 MACs; INSTR_VECTOR cycles equal (per-core MACs) / 512 in every trace, so the
  vector unit runs at its peak whenever it issues. Everything below is about the other cycles.

## Per-core breakdown (4 cores, 512x512xK, B256x128 or B128x128 blocks)

| build | span (cycles) | VMAC | lock stall | rest (acc load/store, loop) |
|---|---|---|---|---|
| K=1152, L1 128x64 | 508,578 | 29.0% | 35.4% | ~35% |
| K=1152, L1 64x64 | 685,920 | 21.5% | 51.8% | ~26% |
| K=1152, L1 64x64, allocs not hoisted | 511,631 | 28.8% | 35.5% | ~35% |
| **K=1024**, L1 64x64 (L1 + L2 ping-pong) | 341,684 | 38.4% | 13.4% | ~46% |
| K=1024, + inner K loop unrolled in LLVM | 339,293 | 36% | 43.8% | ~17% |

1. **No double buffering when the K loop runs an odd number of steps.** The cores' L1 held one
   A, one B and the C accumulator (56 KB); each K step waited for its A/B DMA, then computed.
   mlir-air's `air-label-scf-for-to-ping-pong` declines loops whose trip count is not provably
   even (unroll-by-2 would leave a third buffer, silently wrong). K=1152 with 128-wide steps is
   9 steps; so is every 3x3 conv with CI = BLOCK_K. With 8 steps L1 and L2 are ping-ponged and the
   lock stall drops from 35% to 13%. Fix on our side: make the step count even (pad K).
   (Triton-XDNA's `hoist_static_alloc` also kept L2 single-buffered; without it the odd case
   improves 686k -> 512k cycles.)
2. **Accumulator traffic not overlapped.** Per K step each 2x2 register block reloads its four
   8x8 int32 accumulators from L1 and stores them back, outside the pipelined inner K loop (II 5
   for 4 VMACs), with spills. Fully unrolling that loop in LLVM (opt `loop-unroll` after aiecc's
   `default<O1>`, via `make_peano_wrap.sh`) lets llc overlap it: the non-stall cycles drop by
   two thirds, and the 4 cores become DMA-bound instead (lock stall 44%).
3. **Full array (8x4 herd, B512x512_L1 64x128) is bound by data movement, not the cores:**
   2048x4096x1024 runs at 6.8-7.0 TOPS (2.47 ms, ~26 GB/s of DDR traffic incl. the int32 C), and
   the LLVM unroll makes it slower (6.2 TOPS). Here the output bytes matter:

| 1024x1024x4096, full array | ms | TOPS | output |
|---|---|---|---|
| int32 C | 1.310 | 6.56 | exact |
| SRS epilogue, int8 (`(acc >> s)` saturated) | 1.154 | 7.45 | within 1 rounding step |
| SRS epilogue, ReLU + uint8 (`clamp(acc >> s, 0, 255)`) | 1.151 | 7.46 | within 1 rounding step |

## Compiler bugs found, and fixes

| where | bug | status |
|---|---|---|
| mlir-aie `AIECoreToStandard.cpp` | AIE2P cores set saturation with AIE2's register index 9 (crPackSize on AIE2P; crSat is 0): integer SRS never saturates, values wrap (125 of 1M off by 255 at K=4096) | fixed on `~/src/mlir-aie-latest` branch `fix/aie2p-crsat` (2cfdf1f), not built; verified by patching the call in the core LLVM IR (`PEANO_OPT_PRE_SED`) |
| llvm-aie (Peano) legalizer | vector smax/smin/umax/umin lower to icmp + select on `<N x s1>`, which cannot be selected at any width ("Unsupported register size" / "cannot select G_SELECT") | fixed on `~/src/llvm-aie-peano` branch `fix/aie2p-vector-minmax` (f7e3a79): 512-bit -> VMAX_LT / VMIN_GE, wider split; built (`build/bin/llc`), used through `PEANO_LLC_REAL` |
| Triton-XDNA epilogue schedule | the epilogue herd vectorizes 16 lanes; AIE2P SRS i32->i8 lowers only at 32/64 lanes | 32-lane tiling (`tile_sizes [1, 1, 4, 0]`) works; generator option still to add |
| mlir-air ping-pong labeler | odd K trip count -> no L1/L2 double buffering | pad K to an even number of steps |
| mlir-aie (wheel e040370) | `clamp(x >> s, -128, 127).to(i8)` then `max(., 0)` compiles to `srs(smax(x, 0), shift 0)`: shift and upper clamp dropped, silently wrong | not fixed (wheel source not public); use the `[0, 255]` unsigned-SRS ReLU, which is correct |
| Peano | no vector shifts (G_ASHR <N x s32>): a shift that is not folded into an SRS fails to compile | open |
| Triton-XDNA epilogue | runtime scalars reach the epilogue herd as 0 | open (constexpr scales work) |
