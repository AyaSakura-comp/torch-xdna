# Convolutions on the NPU: explicit im2col vs implicit GEMM (2026-10-26)

ResNet-50 conv layers, int8 (per-tensor activations, per-channel weights), NPU via Triton-XDNA,
CPU = torch fp32 conv (oneDNN), 16 threads. Scripts: tools/conv_implicit/.

Implicit GEMM (`conv_kernel.py`): K = (ky, kx, ci) is the reduction; weights are reordered
offline to (ky*kx*ci) x Co. The input is padded and laid out NHWC on a width-WP grid, so tap
(ky, kx) is the same block load shifted by ky*WP + kx rows; stride 2 uses a polyphase split
(4 phase grids, 1x the data). Output is computed on the padded grid, extra columns dropped.
Every result was bit-identical to the same integers on the CPU (torch._int_mm).

batch 1, ms (median):

| layer | GEMM M x K x N | cpu fp32 conv | explicit im2col -> NPU | implicit NPU kernel | implicit e2e |
|---|---|---|---|---|---|
| layer1 3x3 | 3136 x 576 x 64 | 0.19 | 4.94 | 0.77 | 1.02 |
| layer2.0 3x3/2 | 784 x 1152 x 128 | 0.28 | - | 0.47 | 0.83 |
| layer2 3x3 | 784 x 1152 x 128 | 0.20 | 1.80 | 0.46 | 0.68 |
| layer3 3x3 | 196 x 2304 x 256 | 0.17 | 1.31 | 0.47 | 0.57 |
| layer4 3x3 | 49 x 4608 x 512 | 0.21 | 1.24 | 0.47 | 0.63 |

The implicit addressing costs nothing against a plain GEMM of the same shape (layer2-4 within
noise; layer1 0.77 vs 0.60). Batch 1 is bound by the ~0.3-0.4 ms per-launch floor.

batch 8 / 32 (NPU kernel / e2e vs CPU):

| layer | B=8 cpu | B=8 npu kernel / e2e | B=32 cpu | B=32 npu kernel / e2e |
|---|---|---|---|---|
| layer1 | 0.91 | 4.40 / 5.06 | 3.77 | 16.5 / 22.9 |
| layer2 | 1.07 | 1.80 / 2.15 | 3.80 | 6.11 / 9.32 |
| layer3 | 1.01 | 0.93 / 1.41 | 3.74 | 2.78 / 3.97 |
| layer4 | 0.98 | 0.83 / 1.18 | 3.90 | 2.34 / 2.91 |

Useful throughput 0.3-3.2 TOPS (NPU int8 peak 51). Whole ResNet-50 fp32 eager on the CPU:
16.7 ms at B=1, so 53 convs x the launch floor alone (~16-21 ms) already matches it: per-layer
launches cannot win at batch 1; layers would have to be fused into one launch with the feature
map kept on chip.
