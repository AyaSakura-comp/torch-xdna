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
* Activation × activation matmuls (attention) are not offloaded: their operands
  change every call, and at these sizes one NPU launch (~0.3 ms) costs more
  than the whole GEMM on the CPU.

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
| `cpu_kernel` | `"int8"` | CPU-routed GEMMs: `int8` (same as NPU) / `fp` |
| `fuse` | `True` | fuse linears sharing an input into one GEMM |
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
