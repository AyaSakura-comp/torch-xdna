# Why the XDNA NPU loses to the CPU on OmniVoice's GEMMs (2026-10-26)

Box: Ryzen AI MAX+ 395 (npu2 / AIE2P, 8 columns x 4 core rows), XRT + amdxdna 2.23,
Triton-XDNA main (a414baf + resident tensors). W8A8 int8 GEMM, int32 out.

## 1. Tiling sweep (tools/sweep.py, results/sweep.jsonl)

628 (shape x tiling) builds over OmniVoice's 5 fused shapes x M in {128, 256, 512, 1024};
search space block 128-512 x 256-1024, per-core tile 16-64 x 64-256, l2_k 128/256, herds of
16-32 cores. 411 compiled and ran, **all bit-exact**; 217 were rejected by the compiler:

* `aie.dma_bd` length 16384 words > max 16383 (a 64 KB L1 tile, e.g. l2_k=256 x l1_n=256 int8)
* `aie.buffer ... could not be placed` (L1 full)
* `aie-place-tiles` failure (herd taller than 4 core rows)

Best tiling vs the default, and vs the CPU (`torch._int_mm`, min of 60 runs on a busy box):

| M | N | K | default ms | best ms | best tiling | best / CPU |
|---|---|---|---|---|---|---|
| 128 | 4096 | 1024 | 0.612 | 0.583 | B128x1024 L1 16x256 K128 | 3.98x slower |
| 128 | 6144 | 1024 | 1.365 | 0.792 | B128x512 L1 16x128 K256 | 3.96x |
| 128 | 1024 | 2048 | 0.439 | 0.412 | B128x512 L1 32x128 K256 | 5.03x |
| 128 | 1024 | 3072 | 0.517 | 0.517 | B128x512 L1 32x128 K128 | 4.08x |
| 512 | 4096 | 1024 | 1.412 | 1.026 | B256x512 L1 64x128 K128 | 1.81x |
| 1024 | 4096 | 1024 | 1.464 | 1.464 | B512x512 L1 64x128 K128 | 1.46x |
| 1024 | 6144 | 1024 | 2.569 | 2.299 | B256x512 L1 64x128 K128 | 1.02x |

(full table: `tools/sweep.py --summary`; shipped as `torch_xdna/tuned/strix_halo_i8.json`,
used by default via `tuned_tilings=True`). End to end on OmniVoice the tuned table changes
0.90 -> 0.88 s / 1.17 -> 1.16 s / 2.89 -> 2.72 s (7 / 19 / 42 chars): within noise at M=128,
because the launch's fixed cost dominates there.

## 2. Where the fixed cost comes from (tools/xclbin_exp.py)

The same GEMM built as an ELF (Triton-XDNA's default on npu2) and as an xclbin + insts.bin,
both run with raw pyxrt on one persistent hw_context:

| | ELF | xclbin |
|---|---|---|
| qkv 128x4096x1024, back to back | 0.60 ms | **0.31 ms** |
| o 128x1024x2048, back to back | 0.35 ms | **0.16 ms** |
| qkv / o alternating, per run | 0.45 ms | **1.43 ms** |
| ~zero work, 4 / 8 / 16 / 32 cores | 0.21 / 0.22 / 0.28 / 0.39 ms | 0.11 / 0.09 / 0.09 / 0.09 ms |

* The ELF's runtime sequence starts with `aiex.configure`: every run reloads the PDI (core
  ELFs + CDO, ~50 KB), costing 0.12-0.30 ms that grows with the herd's core count.
* An xclbin loads the PDI with the context, so a run is ~0.09 ms + work, but switching between
  two contexts costs ~1.1 ms per switch.
* `aiebu-dump` cannot disassemble these ELFs ("Corrupted transaction binary" on the .pdi
  section); XRT 2.23's dtrace has no documented control-file syntax, so the split above is
  from controlled experiments, not a trace.

## 3. Keeping several contexts resident (tools/coresident_exp.py)

Patched Triton-XDNA with `npu_config.device_columns` / `AMD_TRITON_NPU_DEVICE_COLUMNS` so aircc
targets `npu2_<n>col` (binary claims n columns, relocatable start column; now on
the fork's `main` as 8d31793). Findings:

* npu2_2col with herd (2,4) does not compile: "no ShimNOCTile has 0 input / 1 output DMA
  channel free: all 2 at 2/4 input, 4/4 output used" -- 2 shim tiles cannot feed 6 A/B streams.
  herd (2,2) on npu2_2col and herd (4,4) on npu2_4col compile.
* At run time amdxdna 2.23 still put every context in one 8-column partition (also with the
  FLM server stopped), so 4 narrow contexts in round robin still pay a switch:
  2-col 0.99 ms/run vs 0.63 back to back, 4-col 1.58 vs 0.30. No spatial sharing here.

## 4. One context for every shape (tools/pdi_identity.py, tools/shared_ctx_exp.py)

GEMMs with the same tiling and K compile to **byte-identical PDIs** (128x4096, 128x6144,
128x8704, 128x1024, 256x4096 at K=1024 all md5 f83b0ac8a6); only insts.bin differs. So one
context can run every shape by passing that shape's insts.bin; K=2048/3072 are split into
K=1024 chunks summed on the CPU. All bit-exact. Per OmniVoice layer (4 GEMMs):

| M | shared context (7 runs) | ELF (4 runs) | CPU int8 |
|---|---|---|---|
| 128 | **1.75 ms** | 2.22 ms | 0.49 ms |
| 256 | 3.09 ms | 2.58 ms | 1.13 ms |
| 512 | 4.62 ms | 3.45 ms | 2.10 ms |
| 1024 | 17.97 ms | 6.50 ms | 5.11 ms |

(the shared runs use one 16-core 128-row tiling and copy/sum int32 results in numpy, which
is why they fall behind at large M.)

## Conclusion

Removing the per-launch reload helps only at M=128 and still leaves the NPU ~3.5x behind the
CPU. With the overhead gone, a DDR-fed int8 GEMM on this NPU reaches ~6-7 TOPS (the sweep's
best, and mlir-aie's hand-tuned whole_array GEMM measures 6.9-7.75), against 9.7 TOPS for
the Zen 5 CPU's `_int_mm`, and the NPU has about half the CPU's DRAM bandwidth. For these
shapes the NPU's value is as a second engine running concurrently with the CPU, not as a
faster one.
