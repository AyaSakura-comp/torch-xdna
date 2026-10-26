# torch-xdna

A `torch.compile` backend that runs a model's GEMMs on the AMD XDNA NPU
(Ryzen AI, npu2 / AIE2P) through [Triton-XDNA](https://github.com/AyaSakura-comp/Triton-XDNA),
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
   ├─ F.linear(x, W, b) / x @ W  with W a parameter   ──►  xdna::linear  (custom op)
   │    linears reading the same x (q/k/v, gate/up)   ──►  xdna::linear_cat (one GEMM) + narrow
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
  Convolutions are not offloaded (ResNet-50 puts 0.05% of its FLOPs on the NPU).
* Activation × activation matmuls (attention) are not offloaded: their operands
  change every call, and at these sizes one NPU launch (~0.3 ms) costs more
  than the whole GEMM on the CPU.

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
and the Triton-XDNA branch `feat/device-columns` serializes builds with an flock on
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
median of 3, `generate()` wall time for a 7 / 19 / 42-character sentence:

| | 7 chars | 19 chars | 42 chars | GEMMs on |
|---|---|---|---|---|
| eager fp32 (no compile) | 1.39–1.48 s | 2.01–2.54 s | 4.01–4.91 s | — |
| `xdna`, `policy="npu"` | 0.92 s | 1.16 s | 2.60 s | NPU (904 launches / sentence) |
| `xdna`, `policy="npu"`, `fallback="eager"` | 1.01 s | 1.43 s | 2.89 s | NPU |
| `xdna`, `policy="auto"` | **0.40 s** | **0.66 s** | **1.70 s** | CPU (chosen for every shape) |
| `xdna`, `policy="cpu"` | 0.45 s | 0.71 s | 1.79 s | CPU |

(the eager fp32 ranges span a quiet run and one with another job on the box.)

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
in fp32 eager. Run sequentially on a clean cache: every NPU program passed its first-launch
check (0 fallbacks), and in int8 every model's output is bit-identical to the CPU int8 run.

| model | NPU share of FLOPs | int8 vs fp32 | bf16 vs fp32 |
|---|---|---|---|
| ResNet-50 (torchvision) | 0.05% (fc only; convs stay on CPU) | top-1 8/8, cos 0.99996 | top-1 8/8, cos 0.99999 |
| DINOv3 ViT-L/16 | 99.8% | embedding cos 0.997 | cos 0.9997 |
| GPT-2 (HF, Conv1D) | 100% | **ppl 23.4 -> 67.6, text degenerates** | ppl 22.1, top-1 93%, fluent |
| Qwen2.5-0.5B-Instruct | 100% | ppl 10.09 -> 11.73, fluent, diverges | ppl 9.95, top-1 98%, diverges later |
| multilingual-e5-small | 100% | ranking identical, cos 0.999 | ranking identical, cos 0.9999 |
| Whisper-base (ASR, zh) | 19% (encoder convs, attention on CPU) | 3/3 transcripts identical | 2/3 identical, char sim 0.93 |
| OmniVoice (TTS) | 98.5% | runs; waveform differs (sampling) | runs; waveform differs |

Neither precision is right for every model: int8 keeps Whisper's transcripts exact but breaks
GPT-2; bf16 (bfp16) fixes GPT-2 and Qwen but changes one Whisper transcript. Greedy text and
seeded sampling diverge once any logit moves, so "identical output" is a strict bar; check the
task metric of your own model before choosing.
