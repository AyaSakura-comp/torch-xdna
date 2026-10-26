# 3x3 convolution at the NPU's peak: cascade chains, Triton op, torch.compile (2026-10-27)

**Result.** Ci = Co = 256, 64-wide feature map, 1536 rows: **2.205 ms of NPU time (IRON harness)
= 52.6 TOPS, 86%
of the 61.0 TOPS int8 peak** (32 AIE2P cores x 512 MAC/cycle x 1.86 GHz), bit-exact against the
CPU. Each core's steady state is **90.2% VMAC**. For comparison, `xrt-smi validate -r gemm` reports
51.0 TOPS, and Triton-XDNA's own GEMM runs at ~7 TOPS on the full array (`npu_profile.md`).

The work, bottom up:

| layer | what | where |
|---|---|---|
| intrinsics | the per-core microkernel: Peano (llvm-aie) intrinsics for VMAC, VSHIFT, cascade, SRS | Triton-XDNA `kernels/conv3x3_cascade.cc` (research copy: `npu_kernels/`) |
| array design | 16 two-core cascade chains, memtile broadcast, stream-major DDR layout (IRON) | `kernels/conv3x3_design.py`, research: `npu_kernels/conv3x3_strips.py` |
| Triton | `tl.extra.npu.conv3x3`: build (cached), run (pyxrt), host layouts | Triton-XDNA `amd_triton_npu/language/npu/conv3x3.py`, branch `feat/npu-conv3x3` (a675854, local) |
| torch | `torch.compile(backend="xdna")` turns 3x3 convs into `xdna::conv3x3` | `torch_xdna/conv.py`, `backend.py` |

## Why a GEMM route can't get there

The Triton-XDNA GEMM is bound by data movement: per K step each core reloads and stores its
int32 accumulators, operands stream from DDR per block, and an odd K-step count (a 3x3 conv with
Ci = BLOCK_K gives 9) disables double buffering. A convolution needs none of that traffic: every
core can keep its weights for the whole image, every input pixel is used 9 times, and the 9·Ci-long
reduction never has to leave the accumulator registers.

## The microkernel (one core)

* **Block**: 16 output pixels x 16 output channels = four 8x8 int32 accumulators (2x2 register
  blocking; AIE2P has five 2048-bit accumulator registers). Per (ky, 8-channel group) it does three
  aligned 512-bit loads of the input line (`v0, v1, v2`); the kx = 1, 2 windows are `shift_bytes`
  (VSHIFT) of them, which issue beside the VMACs. Unaligned loads instead cost a VSHIFT each plus
  spills (57% loop density).
* **Reduction**: 3 ky x CG x 3 kx steps of 4 `mac_8x8_8x8` (512 int8 MACs each). VMAC has a 6-cycle
  result latency but reads its accumulator input at cycle 4 with a bypass, so 4 independent
  accumulators sustain 1 VMAC/cycle.
* **Cascade** (`CascadeFlow` between vertically adjacent cores): the head starts the block with
  `mul_8x8_8x8` and sends the 4 accumulators down as 16 x 512-bit `put_mcd`; the tail seeds its
  accumulators with `get_scd_v64acc32`, reduces its half of the channels, and shift-round-saturates
  (`to_v64int8`, or `to_v64uint8(acc, shift, 0)` = ReLU fused as unsigned saturation) into the
  output line. The fused `VADDMAC ... scd` exists in the ISA but no builtin reaches it; seeding
  from the cascade costs 4 reads per block, hidden in the block hand-over.
* **Weights**: per output-channel pair in consumption order, 36 KB for Ci/2 = 128 channels, plus a
  64-byte trailer whose first int32 is the shift (runtime data: one build serves every layer of a
  geometry). Input: a 4-line sliding window of the core's channel half (20 KB).

Static loop density (VMAC bundles / loop bundles): weight-stationary single-core variant 94.7-95.4%;
cascade roles head 91.1%, mid 87.5%, tail 90.6%. (AMD's mlir-aie `mm` 2x2 kernel: 49/66/79.5% at
K = 64/128/256; its `conv2dk3`: 87.3%.)

Codegen issues on the way, and how the kernel avoids them:

| issue | fix |
|---|---|
| LICM hoists the next output-channel pair's weight loads out of the loop; they don't fit the registers, 8 KB of spills | one flat (pixel block, channel pair) loop; the weight pointer advances with the loop variable |
| `[[gnu::always_inline]]` after a lambda's parameter list is ignored by clang (17.8K bundles) | `__attribute__((always_inline))` |
| only the innermost single-block loop is software-pipelined; `-aie-postpipeliner-maxii` defaults to 60 | keep the per-block body one straight-line region |
| mlir-aie writes AIE2's saturation register index on AIE2P (crSat is 0 there), so SRS never saturates | the tail calls `set_sat(); set_rnd(9)` itself; compiler fix on `~/src/mlir-aie-latest` `fix/aie2p-crsat` |
| `to_v64uint8`'s third argument is the *output* signedness | pass 0 for the unsigned (ReLU) output |

## The array design

```
column c (8 of them), 32 output channels each
   row 5  head  strip 0, channels [0, Ci/2)  ──cascade──►  row 4  tail  strip 0, channels [Ci/2, Ci) ──► y
   row 3  head  strip 1                     ──cascade──►  row 2  tail  strip 1                     ──► y
   row 1  memtile c: column c's weights (head slice, tail slice, each broadcast to both chains),
                     joins its two tails' output rows into one stream
          memtile 2h+s: input half h of strip s, broadcast to that chain position in all 8 columns
   row 0  shim
```

* A 64-pixel band is two 32-pixel strips (input windows 40 wide, halo included); wider images are
  bands stacked vertically with their own zero rows, so one launch covers any W that is a multiple
  of 64.
* DDR layouts are stream-major: every DMA transfer is one contiguous run per row. With the first,
  strided layout (320-byte runs) 1024 rows took 2.72 ms; stream-major 1.65 ms.
* Shim BD limit (16): `aiecc --reclaim-runtime-bds`.

| design | cores | rows (64 wide, Ci = Co = 256) | NPU time | TOPS |
|---|---|---|---|---|
| one column, 4-core chain (Ci quarters) | 4 | 1024 | 1.919 ms | 5.04 |
| 8 columns x 4-core chains, 64-wide lines | 32 | 1024 | 1.789 ms | 43.2 |
| **8 columns x 2 strips x 2-core chains** | 32 | 128 / 384 / 512 / 1536 | 0.232 / 0.588 / 0.756 / **2.205 ms** | 41.7 / 49.3 / 51.2 / **52.6** |

Trace of the strip design (384 rows, `Program.enable_trace`): 87.5% of the run is VMAC; in steady
state a row takes 2554 cycles against 2304 VMAC cycles, **90.2%**. The rest: block hand-over
(~4-6%, static), the head waiting on the cascade (~5.6%, backpressure from the slower tail), the
tail's lock (3.2%) and memory (2.9%) stalls. Short images pay the fill/drain of the pipeline.

## The Triton op: `tl.extra.npu.conv3x3`

```python
from triton.language.extra.npu import conv3x3
op = conv3x3.Conv3x3(w_int8, height=H, width=W, shift=12, relu=False, batch=N)  # w [Co, Ci, 3, 3]
y = op(x_int8)                     # [N, Ci, H, W] -> [N, Co, H, W] int8 (uint8 with relu)
op.set_shift(10)                   # rewrites the weight trailers, no rebuild
conv3x3.share_context(ops)         # a model's ops in one hardware context (see "Whole models")
```

* Not a `@triton.jit` kernel: weights resident across an image and partial sums passed core to
  core have no tile semantics in the language. It sits in `language/extra/npu` next to
  `fused_decode` as an op that is real on this device.
* Builds: the kernel object with Peano (cached by source, defines, toolchain), the IRON design in a
  subprocess (mlir-aie's MLIR bindings and mlir-air's, which the Triton backend has loaded, cannot
  share an interpreter: "Option registered more than once", or a nanobind error with DEEPBIND),
  cached by geometry, as a full ELF (or an xclbin). Runs through pyxrt; ops of one geometry
  share one hardware context.
* Host layouts (`kernels/conv3x3_host.c`, OpenMP over torch's thread count, built with the
  system C compiler on first use; torch permutes otherwise). For the 25 MB input of 1536 x 64:
  pack 0.74-0.85 ms (torch permute 2.8-3.6 ms); unpack 1.5 ms after switching to whole 64-byte
  line stores (1.8-2.1 ms before; 256 x 256: 3.7 -> 1.3 ms), read straight from the output
  buffer's mapping. (Measured with other jobs on the machine, load 7-13; AVX-512 VBMI `vpermb`
  for the 8x8 byte transpose gave nothing measurable, so the portable version stayed.)
* Limits: Ci up to 512, stride 1, padding 1, no bias (first version: Ci a multiple of 16 up to
  256, Co of 32, W of 64; generalized in "Whole models" below).
* `examples/conv3x3_cascade/conv3x3_cascade.py`: 6 shapes covering every split (2- and 4-core
  chains, 1-4 partitions, padded Ci, narrow tiles, batches, ReLU) bit-exact, then all of them
  interleaved in one shared context.

## torch.compile

```python
model = torch.compile(model, backend="xdna", options={"policy": "npu"})
```

* The pass rewrites `conv2d` with a 3x3 parameter weight, stride 1, padding 1 or `"same"`,
  dilation 1, groups 1 and a supported shape into `conv3x3_int8`: per-image activation scale,
  quantize, `xdna::conv3x3`, dequantize (+ bias). The quantize and dequantize are ordinary ops that
  Inductor fuses with their neighbours; the custom op only moves int8. A ReLU that is the only user
  of a bias-free conv is fused into the NPU's saturation.
* Weights int8 per output channel; the shift comes from the layer's first call (its integer sums,
  25% headroom). The CPU route computes the same integers (fp32 conv of the int8 values, exact
  below 2^24, else fp64), so outputs do not depend on routing. Without an NPU, or with
  `policy="cpu"`, the pass leaves convs alone.
* A layer's first NPU launch is compared bit for bit with the CPU's integers; a mismatch, a build or
  launch failure moves that layer to the CPU. `policy="auto"` times both per layer.
* Tests (`tests/test_backend.py`): which convs are rewritten and which ReLU fused; three convs
  (batch 2) on the NPU equal to the CPU's integers, rel err vs fp32 2.9%; padded shapes (Ci 3,
  Co 48, 13 x 37, batch 2) likewise; a corrupted first launch caught and that layer moved to
  the CPU; convs left alone without an NPU.

`examples/conv_xdna.py` (four bias-free 3x3 convs + ReLU, 256 channels, 1 x 64 x 64; median of
10, one run, load ~4):

| | ms | TOPS |
|---|---|---|
| Inductor fp32 on the CPU | 5.72 | 3.4 |
| `xdna`, `policy="npu"`, end to end | 1.68 | 11.5 (rel err vs fp32 3.4%) |
| its NPU launches alone (incl. ~0.1 ms launch overhead each) | 0.80 | 24.2 |

At this size each launch is short (0.2 ms), so the fixed per-launch cost and the host work weigh
more than at 1536 rows; with the machine under heavier load the same comparison was 11.7 vs
2.0 ms. Only the op's integers are checked bit for bit; the 3.4% is int8 quantization over four
layers (per-tensor activations, per-channel weights).

## Whole models: any layer shape, one hardware context (VGG-16 all on the NPU)

To run complete CNNs the op was generalized (Triton-XDNA 0ea3101):

* **Any H x W, batches.** The host places the images on the strips with a plan of pieces:
  32-column tiles, rows spread evenly over the strips, narrow tiles (<= 15 wide) side by side in
  one 32-pixel line, every piece with its own halo lines. A batch is one launch.
* **Ci up to 512** with 4-core chains (Ci/4 per core, the same kernel object with the mid role);
  Ci is padded to the chain's channels (VGG's first conv, Ci = 3, to 16).
* **The whole array for small Co.** R partitions x 2 strips x G channel groups of chains
  (R*G = 8, or 4 with 4-core chains), plus OGP output-channel pairs per chain as local memory
  allows: Co = 64 runs as 4 partitions of 2 groups instead of leaving 6 columns idle. The
  weights move off memtiles 0-3 when those already split the input (6 output DMA channels).
* **All Co slabs in one launch**; the cores reload their weights per slab.

| VGG-16 layer | chain x CG, OGP, G x R | NPU run (batch 1) |
|---|---|---|
| conv1_1 3->64, 224^2 | 2 x 1, 2, 2 x 4 | 0.53 ms |
| conv1_2 64->64, 224^2 | 2 x 4, 2, 2 x 4 | 0.57 ms |
| conv2_1 64->128, 112^2 | 2 x 4, 4, 2 x 4 | 0.52 ms |
| conv2_2 128->128, 112^2 | 2 x 8, 4, 2 x 4 | 0.61 ms |
| conv3_1 128->256, 56^2 | 2 x 8, 4, 4 x 2 | 0.56 ms |
| conv3_2/3 256->256, 56^2 | 2 x 16, 2, 8 x 1 | 0.71 ms each |
| conv4_1 256->512, 28^2 | 2 x 16, 2, 8 x 1 (2 slabs) | 0.69 ms |
| conv4_2/3 512->512, 28^2 | 4 x 16, 2, 4 x 1 (4 slabs) | 0.74 ms each |
| conv5_1-3 512->512, 14^2 | 4 x 16, 2, 4 x 1 (4 slabs, 14-wide tiles in pairs) | 0.68 ms each |

**Hardware contexts.** Two ops built as separate programs alternate at 4.8 ms per pair
(0.20 + 0.09 ms of work): ~2.3 ms per context switch, the same for xclbins and full ELFs. A
model's layers run one after another, so every launch would pay it. `share_context(ops)` merges
the ops' designs (one `aie.device` each) into one full ELF with one kernel per design and the
`load_pdi` of each runtime sequence expanded into explicit writes (`aiecc --expand-load-pdis`;
a plain reloaded PDI left the previous device's routes and timed out on the next run): the
same pair then alternates at 1.13 ms, each launch rewriting its own configuration. That
rewrite is mostly the cores' programs (32 x ~5.4 KB of fully unrolled kernel per layer, of a
~110-220 KB configuration), ~0.4 ms per launch. Control-packet loading (`--load-pdi-to-ctrl-pkt`)
needs shim DMA channels the design already uses. torch-xdna merges after the first pass; ops of
one design share a kernel, and a single design is left in its own context.

**VGG-16** (`examples/vgg16_xdna.py`, ImageNet weights, public-domain sample images): 100% of
the conv and linear MACs on the NPU, top-1 7/8 and logits cosine >= 0.991 against fp32 eager.
Batch 1: 20.0 ms vs 32.2 ms for Inductor fp32 on the CPU; batch 8: 68.5 ms vs 142-162 ms.

| batch 1, 20.2 ms | ms |
|---|---|
| 13 conv launches (configuration ~0.45 ms each, MACs ~0.7 ms in all) | 8.45 |
| their host layouts (pack / unpack) | 2.1 |
| linears (Triton-XDNA GEMM; fc6 streams 103 MB of int8 weights) | 5.7 + 1.5 + 0.7 |
| quantize / dequantize, ReLU, pooling, Python (Inductor, CPU) | 1.85 |

At batch 8 (64.1 ms) the NPU runs take 14.5 ms, the host layouts 12.9 ms, the linears 8.4 ms and
the CPU's fp32 work between the NPU ops 28.4 ms: activations of up to 100 MB in fp32, read and
written several times per layer.

**ResNet-50** gets 36.8% of its conv MACs onto the NPU (the 13 3x3 stride-1 convs; top-1 8/8):
1x1 convs (44% + 7.5% strided), the stride-2 3x3s (8.5%) and the 7x7 stem (2.9%) have no NPU
kernel, and at batch 1 the split run (18.4 ms) is slower than Inductor fp32 alone (10.3 ms).

## What is left

* The CPU between NPU ops: activations go back to fp32 NCHW after every conv, are re-quantized
  and laid out for the next. Keeping them int8 in the NPU's layout from one conv to the next
  (requantize, ReLU, pool and the next layer's halo lines on the NPU) removes both the host
  layouts and the fp32 traffic, the largest costs at batch 8.
* Reconfiguration per launch (~0.45 ms, mostly core programs): a smaller kernel (a rolled channel
  loop) shrinks it; a sequence variant without the reload would serve repeated launches.
* ResNet-style layers: a 1x1 kernel (no ky / kx, more output channels per core), stride 2
  (polyphase or subsampled output), the stem.
* Bias inside the NPU (added to the int32 sums before the shift) would let a conv + bias + ReLU
  fuse completely.
* Ci > 512: longer chains do not fit a column; split the reduction over launches.
