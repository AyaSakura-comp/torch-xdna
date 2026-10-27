# torch-xdna

A `torch.compile` backend that runs a model's GEMMs and 3x3 convolutions on the AMD XDNA NPU
(Ryzen AI, npu2 / AIE2P) through [Triton-XDNA](https://github.com/AyaSakura-comp/Triton-XDNA),
whole Qwen3-0.6B-sized transformer stacks (OmniVoice's backbone) as one NPU launch per call,
and everything else on the CPU.

```python
import torch

model = torch.compile(model, backend="xdna")                         # policy "auto"
model = torch.compile(model, backend="xdna", options={"policy": "npu"})

import torch_xdna
print(torch_xdna.report())   # where the GEMMs ran, NPU launches, autotune decisions, fallbacks
```

The backend is registered through the `torch_dynamo_backends` entry point, so
`backend="xdna"` works without importing `torch_xdna` first.

## How it works

```
torch.compile(model, backend="xdna")
  Dynamo FX graph
   ├─ a Qwen3Model (Qwen3-0.6B shapes) + linear head  ──►  xdna::qwen3_stack: one launch per pair of sequences
   │    (attention under a 4-D mask given to the graph)    (and the whole graph compiled as below as well,
   │                                                         for the calls the NPU does not take)
   ├─ F.linear(x, W, b) / x @ W  with W a parameter   ──►  xdna::linear  (custom op)
   │    linears reading the same x (q/k/v, gate/up)   ──►  xdna::linear_cat (one GEMM) + narrow
   ├─ conv2d 3x3 / stride 1 / pad 1, W a parameter    ──►  quantize ─► xdna::conv3x3 ─► dequantize
   │    (+ ReLU of a bias-free conv, fused)                (int8 op; the rest is ordinary ops)
   └─ everything else (attention, norms, RoPE, sampling, ...)
  Inductor compiles the graph for the CPU; the xdna ops are extern calls in it

xdna::linear at run time (runtime.py)
  weight: quantized once per output channel (int8), cached; padded resident copy for the NPU
  x: quantized per row (int8) ─► route ─► NPU: Triton kernel (K-loop int8 GEMM), rows padded to a bucket
                                       └► CPU: torch._int_mm
  int32 ─► dequantize (+ bias) in fp32 ─► input dtype
```

* **One Triton kernel** (`_triton_kernels.py`, the Triton tutorial GEMM in
  int8 → int32 with a K loop) serves every shape. Triton-XDNA builds one NPU
  program per (row bucket, N, K) on first use; programs are cached on disk.
* **W8A8 on both devices.** The NPU and the CPU path compute the same
  integers, so outputs are bit-identical whichever device a GEMM is routed to.
  `cpu_kernel="fp"` keeps CPU-routed GEMMs in the original precision instead.
* **Any shape.** K is padded to 128, N to 512, rows to the next bucket
  (128/256/512/1024); more rows than the largest bucket are split.
* Patterns offloaded: `F.linear(x, W, b)`, `x @ W` / `torch.mm` / `x.matmul(W)`, and
  `torch.addmm(b, x, W)` (HF `Conv1D`, as in GPT-2), with `W` a 2-D parameter.
  3x3 / stride-1 convolutions run as `xdna::conv3x3` (next section); other convolutions
  stay on the CPU.
* Activation × activation matmuls (attention) are not offloaded: their operands
  change every call, and at these sizes one NPU launch (~0.3 ms) costs more
  than the whole GEMM on the CPU.

## 3x3 convolutions: the cascade op

`xdna::conv3x3` runs Triton-XDNA's `tl.extra.npu.conv3x3` (the fork's `main`): every
core keeps its weights for the whole image, the input channels are split over a chain of 2 or 4
vertically adjacent cores whose partial sums meet over the AIE cascade, and the tail core
requantizes (shift, round, saturate; ReLU as unsigned saturation). The chains fill the array,
split over output channels, image strips and (for small Co) image partitions. How it was built,
measured and traced: [results/conv3x3_cascade.md](results/conv3x3_cascade.md).

| Ci = Co = 256 | NPU time | int8 TOPS |
|---|---|---|
| 1536 x 64, NPU time (IRON harness) | 2.205 ms | **52.6 (86% of 61)** |
| 1536 x 64, one call incl. host layouts | 4.95 ms | 23.4 |
| 256 x 256, one call incl. host layouts | 3.35 ms | 23.1 |

In the graph the conv becomes per-image quantize -> `xdna::conv3x3` (int8 in, int8 out plus a
per-channel scale) -> dequantize (+ bias); the quantize and dequantize are ordinary ops that
Inductor fuses with their neighbours. Weights are int8 per output channel, the shift is fixed on
a layer's first call (25% headroom over its integer sums), and the CPU route computes the same
integers, so the first NPU launch of every layer is checked bit for bit. Taken: 3x3, stride 1,
padding 1 or `"same"`, groups 1, up to 512 input channels; any Co, H and W (the op pads channels
and tiles the image), a batch in one launch. Without an NPU, or with `policy="cpu"`, convs are
left as they are (int8 has no CPU conv kernel to gain from); a layer whose NPU launch fails at
run time computes the same integers on the CPU. After the first pass all NPU convs of different
geometry are merged into one hardware context (`tl.extra.npu.conv3x3.share_context`): two
separate contexts cost ~2.3 ms per switch, a shared one ~0.4 ms of reconfiguration per launch.

`examples/conv_xdna.py`, four bias-free 3x3 convs + ReLU, 256 channels, 1 x 64 x 64, median of
10 (the same run, machine otherwise lightly loaded):

| | ms | TOPS |
|---|---|---|
| Inductor, fp32, CPU | 5.72 | 3.4 |
| `xdna`, `policy="npu"`, end to end | **1.68** | 11.5 (rel err vs fp32 3.4%) |
| the NPU launches alone | 0.80 | 24.2 |

The rest of the end-to-end time is the host: quantize / dequantize (Inductor), the int8 layout
change around each launch, and ~0.1 ms per launch. Under heavier CPU load the fp32 CPU run took
11.7 ms and the NPU run 2.0 ms.

## Whole transformer stacks: one launch per step

`xdna::qwen3_stack` runs Triton-XDNA's `tl.extra.npu.qwen3_stack` (the fork's
`main`): every layer of a Qwen3-0.6B-sized stack (hidden 1024, 16 query / 8 kv
heads of 128, MLP 3072, any depth), the final norm and a linear head of any width, for two
sequences, in one launch. bfp16 operands, fp32 accumulation; the GEMMs are cascade chains with
their norms, residuals and logits fused into the last MAC, RoPE / q-k-norm / online-softmax
attention on all 32 cores. How it was built and measured:
[results/omnivoice_npu.md](results/omnivoice_npu.md).

The backend finds the stack in Dynamo's graph by its modules (a transformers `Qwen3Model` whose
last hidden states feed only a linear layer), checks the shapes, RMSNorm eps 1e-6, RoPE without
scaling and non-causal attention under one 4-D mask that is an input of the graph, and then
compiles the graph twice: with the stack and head replaced by `xdna::qwen3_stack`, and as it
would any other graph. The callable it returns looks at the mask on every call
(`torch_xdna/stack.py`):

| the call | runs |
|---|---|
| every sequence's mask a prefix block of 1-384 positions (its first n positions see exactly those n), later positions padding that sees itself or nothing -- OmniVoice's cond / uncond batch | the fused graph: one NPU launch per pair of sequences; padding rows come back as zeros |
| any other mask (causal, left padding, a 2-D padding mask expanded), longer sequences, no NPU | the plain graph |
| the first fused call of each build (row-group count) | also the original graph eagerly in fp32: the stack's output must be within 10% (OmniVoice: 1.3-2.3%), else `NPUResultMismatch` and the plain graph from then on |

Both `policy="auto"` and `"npu"` take the stack without timing it against the CPU (measured
1.6-1.7x faster end to end at 58, 113 and 264 positions, below); `policy="cpu"` or
`stacks=False` leave it alone.

## Results: whole CNNs

`examples/vgg16_xdna.py`: torchvision VGG-16 (ImageNet weights), `policy="npu"`, `min_rows=1`,
`min_macs=0`, 224x224, public-domain test images (scikit-image / matplotlib sample data):

| VGG-16 | |
|---|---|
| on the NPU | **100% of conv + linear MACs** (13 convs as `xdna::conv3x3`, 3 linears as `xdna::linear`); ReLU, max-pool and the int8 quantize / dequantize stay on the CPU |
| vs fp32 eager, 8 images | top-1 7/8 (the 8th: "bow tie" vs "suit"), top-5 overlap 92%, logits cosine >= 0.991 |
| batch 1 | NPU 19.7-20.0 ms vs Inductor fp32 32.2 ms |
| batch 8 | NPU 68.5-69.7 ms vs Inductor fp32 142-162 ms |

Where batch 1's 20 ms go: 13 conv launches 8.5 ms (each mostly the ~0.45 ms reconfiguration;
the MACs themselves are ~0.7 ms at 45 TOPS), their host layouts 2 ms, the 3 linears 7.8 ms
(the first streams 103 MB of int8 weights per call), the rest 1.9 ms. At batch 8 the CPU's fp32
work between the NPU ops (quantize / dequantize, ReLU, pooling of up to 100 MB activations)
grows to ~28 ms and is the largest part.

ResNet-50 (same images): its 13 3x3 stride-1 convs reach the NPU (36.8% of conv MACs, top-1 8/8);
the 1x1 convs (52% of the MACs), the stride-2 convs and the 7x7 stem have no NPU kernel yet, and
at batch 1 the split run (18.4 ms) is slower than Inductor fp32 alone (10.3 ms).

## Precision, and how wrong answers are kept out

Two things can make an offloaded GEMM's answer wrong, and they are handled separately:

1. **The NPU computing something else than intended** (a miscompiled program, a bad
   cached binary). With `verify="first"` (default) the first launch of every NPU
   program is compared with the CPU on the same inputs: bit for bit in int8, within
   5% relative error in bf16. A program that disagrees raises `NPUResultMismatch`,
   its shape moves to the CPU for the rest of the process, and the call still
   returns the right result. `verify="always"` checks every launch; `"off"` none.
2. **Quantization changing the model's output**, which the NPU computes faithfully:
   * `precision="int8"` (default): W8A8, per-channel weights and per-row activations.
     Fine for ResNet / ViT / embedding models and Qwen2.5, but per-row activation
     scales break models with large activation outliers: GPT-2 goes from perplexity
     23.4 to 67.6 and generates nonsense.
   * `precision="bf16"`: bf16 operands, f32 accumulate. AIE2P's MAC unit runs bf16 as
     **bfp16** (8 values share an exponent, `V64BFP16EBS8`), so a GEMM is ~1% off fp32
     regardless of K, but outliers only cost precision to their 7 block neighbours:
     GPT-2 stays at perplexity 22.1. The CPU route of a bf16 GEMM is the fp32 GEMM.

### A Triton-XDNA bug this found: concurrent builds corrupt binaries

Triton-XDNA builds every kernel in an AIR project directory (default `./air_project`)
and caches what it finds there. Two processes building at once in the same
directory mix each other's intermediates, and the mixed binary is cached and
dispatched without error. Reproduced with two processes building 8 int8 GEMMs each
in a shared directory: 8 of 16 returned garbage (and kept doing so from the cache);
0 of 16 with separate directories. torch_xdna now builds in a per-process directory,
and Triton-XDNA (the fork's `main`, 0e29c35) serializes builds with an flock on
`<air_project>/.build.lock`. (This is what first looked like a K=512 int8 bug in
Whisper and a K=768 bf16 bug in GPT-2's LM head: both programs were first built while
another process was building.)

## Hybrid execution and fallback

| situation | what happens |
|---|---|
| `policy="auto"` (default) | per (bucket, N, K), the first call times NPU and CPU and keeps the faster |
| `policy="npu"` / `"cpu"` | force a device for every offloadable GEMM |
| fewer than `min_rows` rows (decode, GEMV) or < `min_macs` | CPU |
| no NPU (no `/dev/accel/accel0`, no Triton-XDNA, `TORCH_XDNA_DISABLE_NPU=1`) | CPU, same math |
| an NPU compile / context / launch error | that shape moves to the CPU and the call still returns; after 3 failing shapes the NPU is retired for the process |
| `fallback="eager"` | the rest of the graph runs uncompiled instead of through Inductor |
| a Qwen3-0.6B-sized stack + head | one NPU launch per pair of sequences when the mask allows it, else the graph compiled as usual (see "Whole transformer stacks") |

Other accelerators: Triton-XDNA's `triton` package replaces the one ROCm/CUDA
PyTorch depends on, so a GPU cannot be driven from the same process; the
non-GEMM part of the graph targets the CPU.

## Options

`torch.compile(..., options={...})` or `torch_xdna.set_config(...)` for process defaults:

| option | default | |
|---|---|---|
| `policy` | `"auto"` | `auto` / `npu` / `cpu` |
| `fallback` | `"inductor"` | compiler for the rest of the graph: `inductor` / `eager` |
| `precision` | `"int8"` | offloaded GEMM math: `int8` (W8A8) / `bf16` (bfp16 on AIE2P) |
| `verify` | `"first"` | compare NPU programs with the CPU: `first` launch / `always` / `off` |
| `cpu_kernel` | `"int8"` | CPU-routed GEMMs: `int8` (same as NPU) / `fp` |
| `fuse` | `True` | fuse linears sharing an input into one GEMM |
| `stacks` | `True` | run a Qwen3-0.6B-sized stack + head as one NPU launch per call (not with `policy="cpu"`) |
| `tuned_tilings` | `True` | per-shape tiling from the sweep table (`tools/sweep.py`) |
| `row_buckets` | `(128, 256, 512, 1024)` | padded row counts, each one NPU program per (N, K) |
| `min_rows` / `min_macs` | `16` / `2e6` | smaller GEMMs stay on the CPU |
| `verbose` | `False` | log graph rewrites and autotune decisions |

Environment: `TORCH_XDNA_CACHE` (schedules + NPU build dir, default
`~/.cache/torch_xdna`), `TORCH_XDNA_MAX_SESSIONS` (live NPU programs, default
24), `TORCH_XDNA_DISABLE_NPU=1`.

## Setup (this machine)

Triton-XDNA's venv is Python 3.13 (XRT's `pyxrt` is cp313 only);
`~/src/omnivoice-npu/venv` layers OmniVoice's dependencies on top of it.

```bash
source ~/src/Triton-XDNA/npu-env.sh          # XRT + mlir-aie/air paths
~/src/omnivoice-npu/venv/bin/pip install -e ~/src/torch-xdna --no-deps
~/src/omnivoice-npu/venv/bin/python tests/test_backend.py
~/src/omnivoice-npu/venv/bin/python examples/omnivoice_xdna.py --policy npu
```

## Results: OmniVoice TTS

`examples/omnivoice_xdna.py`: OmniVoice (k2-fsa; Qwen3-0.6B backbone, a
diffusion LM run once per unmasking step) with `model.forward` compiled by
this backend, no OmniVoice changes. 8 steps, 16 threads, Ryzen AI MAX+ 395,
`generate()` wall time for a 7 / 19 / 42-character sentence (58 / 113 / 264
positions), median of 10 rounds with the three routes interleaved:

| | 7 chars | 19 chars | 42 chars | |
|---|---|---|---|---|
| `xdna` (defaults: the stack) | **0.163 s** | **0.282 s** | **0.639 s** | every step one NPU launch (1.65 / 1.58 / 1.74x the CPU route) |
| `xdna`, `policy="cpu"` | 0.269 s | 0.445 s | 1.113 s | W8A8 GEMMs on the CPU, Inductor |
| the same NPU op without torch.compile (`npu_kernels/omni/omni_npu.py`) | 0.154 s | 0.271 s | 0.619 s | |
| eager fp32 (another day) | 1.30 s | 2.00 s | 3.71 s | |

The fused route's logits are bit-identical to the direct op's; the 3-6% between them is
Dynamo's per-call work on a 317-input graph (2.5-3.3 ms per step; the fused graph itself is
1-2 ms faster than the direct route, its embeddings fused by Inductor). Speech quality is that
of the op: Whisper CER on par with the CPU int8 route over 10 seeds
([results/omnivoice_npu.md](results/omnivoice_npu.md)).

Before the stack, with one NPU launch per GEMM (5 fused shapes, 904 launches per sentence):

| | 7 chars | 19 chars | 42 chars | GEMMs on |
|---|---|---|---|---|
| eager fp32 (no compile) | 1.39–1.48 s | 2.01–2.54 s | 4.01–4.91 s | — |
| `xdna`, `policy="npu"`, `stacks=False` | 0.92 s | 1.16 s | 2.60 s | NPU (904 launches / sentence) |
| `xdna`, `policy="npu"`, `stacks=False`, `fallback="eager"` | 1.01 s | 1.43 s | 2.89 s | NPU |
| `xdna`, `policy="auto"`, `stacks=False` | 0.40 s | 0.66 s | 1.70 s | CPU (chosen for every shape) |
| `xdna`, `policy="cpu"` | 0.45 s | 0.71 s | 1.79 s | CPU |

(an earlier day, before the int8 quantize / dequantize became graph ops; the eager fp32 ranges
span a quiet run and one with another job on the box.)

Every GEMM of the model can run on the NPU (5 fused shapes: q/k/v, o,
gate/up, down, audio heads), but per GEMM the NPU is 3–5× slower than the
CPU's int8 GEMM at these row counts (`auto` decisions: e.g. 128×4096×1024
NPU 0.75 ms vs CPU 0.14 ms), so `auto` keeps them on the CPU. A launch costs
~0.3 ms fixed (the runtime sequence reconfigures the array per dispatch) and at
128–1024 rows these GEMMs are DRAM-bound, where the NPU has about half the
CPU's bandwidth. W8A8 vs fp32 intelligibility (Whisper CER) is within the
sampling noise of 8-step generation.

## Results: off-the-shelf models (`tests/model_zoo.py`)

`model.forward = torch.compile(model.forward, backend="xdna", options=dict(policy="npu",
min_rows=1, min_macs=0, precision=...))` with no model changes; reference = the same model
in fp32 eager. Every NPU program passed its first-launch check (0 fallbacks), and in int8
every model's output is bit-identical to the CPU int8 run except ResNet-50's: its 3x3 convs
run as int8 convs on the NPU (`xdna::conv3x3`, in both precisions) and stay fp32 under policy
"cpu". The ResNet-50 and DINOv3 rows were re-run on 2026-10-27 on public sample images
(scikit-image, matplotlib); the others are from 2026-10-26, run sequentially on a clean cache.

| model | NPU share of FLOPs | int8 vs fp32 | bf16 vs fp32 |
|---|---|---|---|
| ResNet-50 (torchvision) | 36.8% (13 3x3 stride-1 convs + fc; other convs on CPU) | top-1 8/8, cos 0.9994 | top-1 8/8, cos 0.9995 |
| DINOv3 ViT-L/16 | 99.8% | embedding cos 0.997 | cos 0.9998 |
| GPT-2 (HF, Conv1D) | 100% | **ppl 23.4 -> 67.6, text degenerates** | ppl 22.1, top-1 93%, fluent |
| Qwen2.5-0.5B-Instruct | 100% | ppl 10.09 -> 11.73, fluent, diverges | ppl 9.95, top-1 98%, diverges later |
| multilingual-e5-small | 100% | ranking identical, cos 0.999 | ranking identical, cos 0.9999 |
| Whisper-base (ASR, zh) | 19% (encoder convs, attention on CPU) | 3/3 transcripts identical | 2/3 identical, char sim 0.93 |
| OmniVoice (TTS) | 98.5% | runs; waveform differs (sampling) | runs; waveform differs |

Neither precision is right for every model: int8 keeps Whisper's transcripts exact but breaks
GPT-2; bf16 (bfp16) fixes GPT-2 and Qwen but changes one Whisper transcript. Greedy text and
seeded sampling diverge once any logit moves, so "identical output" is a strict bar; check the
task metric of your own model before choosing.
