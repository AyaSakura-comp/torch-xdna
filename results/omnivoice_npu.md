# OmniVoice on the XDNA2 NPU: the whole backbone step in one dispatch (2026-10-27)

**Result.** Every unmasking step of OmniVoice's backbone (28 Qwen3 layers, the final norm and the
8 x 1025 audio heads, for the cond and uncond sequences together) runs as **one NPU dispatch**,
for any sentence length up to 384 tokens per sequence. `generate()` (8 steps) is **1.65-1.75x
faster than the CPU int8 path** and 5.6-8.4x faster than fp32, with the same speech quality:

| text | tokens (cond / uncond) | NPU | CPU int8 | fp32 | NPU step | CER NPU / int8 / fp32 |
|---|---|---|---|---|---|---|
| short, 1.8 s audio | 58 / 43 | **0.155 s** | 0.260 s | 1.30 s | 12.5 ms | 0.167 / 0.167 / 0.222 |
| mid, 3.9 s | 113 / 92 | **0.260 s** | 0.429 s | 2.00 s | 21.7 ms | 0.024 / 0.000 / 0.000 * |
| long, 8.3 s | 264 / 222 | **0.659 s** | 1.151 s | 3.71 s | 55.3 ms | 0.142 / 0.181 / 0.146 * |

Wall time of `generate(num_step=8)`, median of 5 after a warm-up, 16 CPU threads, the box shared
with other services. "NPU step" is the dispatch alone. CER: Whisper large-v3-turbo transcripts
vs the text, mean over seeds 0-2, or (*) seeds 0-9 for NPU and CPU int8; one mid seed of ten
comes out garbled on the NPU (seed 0), the long text has fewer errors on the NPU than with CPU
int8. Against the fp32
model's first-step logits the NPU is within 1.9-3.0% (cond) / 0.9% (uncond) relative error, CPU
int8 within 1.4-1.5% / 0.8%; the NPU's softmax distributions are the closer ones (mean total
variation 0.14-0.19 vs 0.21-0.25). The earlier per-GEMM NPU path (113 dispatches per step) was
2-3x *slower* than CPU int8 (`npu_investigation.md`).

Code: `npu_kernels/omni/` — `omni_step.py` (IRON design + runtime sequence), `omni_core.cc`
(per-core kernels, Peano), `omni_host.py` (weight packing, builds, `Engine`), `omni_npu.py`
(`NPUBackbone(model).install()`), `test_step.py` (every intermediate vs a CPU reference with the
same rounding points), `e2e_omnivoice.py` (the table above).

## Mapping

Operands are bfp16 (AIE2P's block format: 8 values share an exponent, 7-bit mantissas),
accumulation fp32, epilogue math as bf16 products into fp32 accumulators (AIE2P has no fp32
vector multiply). Data moves as 4608-byte tiles (32 rows x 128 columns of bfp16).

* **Row groups.** Rows come in groups of 128: positions 64g..64g+63 of the cond sequence and of
  the uncond sequence. A build serves NG groups (`NG = ceil(len / 64)`, built on first use and
  cached); padded keys are masked.
* **GEMMs.** A row of 8 cores is a cascade chain over K (core c holds K slice c of its 32 rows,
  loaded once per GEMM); partial sums travel west to east in accumulator registers. Column c's
  weights stream from DDR through memtile c to its 4 cores (one contiguous run per GEMM). The
  tails (column 7) finish each 32-column group in the last chunk's MAC: RMSNorm (sums of squares
  passed along the chain first), residual add, logits, and write bfp16 straight into the output
  object. `silu(gate) * up` is made by the down GEMM's cores while they load their A slice.
  QKV, O and down run once per row group (weights streamed NG times); gate/up and the heads take
  two groups per pass (both A slices in the cores, every weight object used twice).
* **QKV post-pass.** Between the QKV GEMM and the attention every core takes its row block of q
  heads 2c, 2c+1 and of kv head c's K (rows 0, 1) or V (rows 2, 3) back from DDR: q/k-norm,
  gamma, RoPE, V -> V^T. The DDR layout gives each column one run per group
  (RoPE copy | q | K/V^T).
* **Attention.** Column c runs kv head c for both sequences; per round its 4 cores take 2 query
  heads x 2 query blocks, q on A, the column's shared K / V^T tiles on the weights' broadcast.
  Online softmax over 64-key chunks (o and l rescaled by the same bf16 factor, so its rounding
  cancels).
* **Runtime sequence.** ~140 shim DMA tasks per layer (one per column per phase where the
  access pattern allows it, `[(groups, stride), ...]` taps); drains are issued before long
  weight tasks (a full task queue stalls the command processor and can deadlock it).

## How it got here (28-layer step, ms; interleaved A/B where both numbers are given)

| change | NG = 1 | NG = 5 |
|---|---|---|
| first one-dispatch step (4-column attention) | 24.0 | - |
| attention on all 8 columns, K/V on the weight broadcast | 17.7 | - |
| silu(gate)*up made by the down GEMM's loads | 15.75 | - |
| row groups (any length) + no soft-float on the cores (`x*1/K`, `/128` folded into exact bf16 products / gamma) | 14.78 | 73.9 |
| software-pipelined MAC loop (`pipeline_initiation_interval`: 5 bundles per 4 MACs, was 7) | 14.57 -> 13.66 | - |
| epilogues in the MAC's registers, written straight into the output object | 13.66 -> 13.35 | 68.3 |
| QKV post-pass on all cores (the tails were the bottleneck: q/k-norm + RoPE ~3.7k cycles per head) | 13.51 -> 12.73 | 68.3 -> 63.7 |
| attention kernels: QK^T 2x2-blocked and pipelined (3.4k -> 0.55k cycles per key block) | 12.47 -> 12.48 | 63.6 -> 58.7 |
| gate/up and heads two groups per pass | 12.79 -> 12.73 | 58.3 -> 55.5 |
| restrict + unrolled tile copies, one conversion less in the epilogue | - | 55.9 -> 55.0 |

## Where the time goes now

Hardware trace of a layer at NG = 1 (cycles at ~1.66 GHz): QKV 154k and O 81k run at the weight
stream's rate (~1.1k cycles per 4.6 KB object per column); gate/up 229k; down 123k + ~20k while
its 3 x 2 A tiles stream in; post-pass + attention ~100k; DDR round trips between phases ~10 µs
each. Weight-stream floor for the step: ~9 ms (NG = 1).

Limits met on the way:

* **MAC rate.** AIE2P has five 2048-bit accumulator registers (`dm0`-`dm4`), so 2x2 register
  blocking is the most there is; with the cascade transfers, the pipeline's fill/drain and the
  epilogue a 32x32x128 block costs ~800 cycles on the tails. That makes the paired passes MAC-bound (~1.6k cycles per
  object against ~1.1k streamed): pairing saves ~25%, not 50%.
* **Attention grows as NG²** (every round streams all of the sequence's keys): ~6 ms of the long
  text's 55 ms (estimated from the per-call trace); `oa_v` (fp32 o reloaded and stored per 4 MACs; the pipeliner does not take loops
  that store) is 60% of it.
* **Pipelining.** Peano software-pipelines a loop when asked (`#pragma clang loop
  pipeline_initiation_interval(N)`) only if the loop has no stores; copy loops need `__restrict`
  to reorder loads past stores.
* **Shim DMA**: a buffer descriptor has 3 tile-level dimensions besides the 2304-byte halves
  (the innermost wrap is 1023 words) and an outer repeat of at most 63, which decides the DDR
  layouts (and why QKV is not paired: its q / K / V drains would need a fourth dimension).
* **Outside the backbone**: for the long text the DAC codec decode (CPU fp32) is 25% of
  `generate()` and the sampling 6%.

## Running it

```bash
source ~/src/Triton-XDNA/npu-env.sh
export PEANO_INSTALL_DIR=~/src/Triton-XDNA/sandbox/lib/python3.13/site-packages/llvm-aie
cd npu_kernels/omni
~/src/omnivoice-npu/venv/bin/python test_step.py --nl 2 --text long      # intermediates vs reference
~/src/omnivoice-npu/venv/bin/python e2e_omnivoice.py --texts short mid long --modes npu cpu-int8 --asr
```

In code: `from omni_npu import NPUBackbone; NPUBackbone(model).install()`; `model.generate(...)`
then runs every step on the NPU (a step longer than 6 groups falls back to the CPU). The first
use packs the weights (~1 min, cached in `~/.cache/omni_npu/`) and builds the NG in question
(seconds, cached in `npu_kernels/omni/build/`).
