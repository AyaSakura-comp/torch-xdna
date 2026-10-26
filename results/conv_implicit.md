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

## Batch 1: cut along x*y instead of batch (tools/conv_implicit/tile_xy.py, tile_res.py)

The first tilings spread the herd over N (channels); for layer1 (N=64) that padded N to 128-256.
Putting the cores on M (= the x*y output pixels) instead, batch 1, 224x224, best NPU kernel ms:

| layer | channel-heavy tiling | x*y-heavy tiling | cpu fp32 conv |
|---|---|---|---|
| layer1 | 0.766 (B256x128, 16 cores) | 0.598 (B1024x64_L1128x32, 16 cores) | 0.21 |
| layer2 | 0.455 (B256x256, 16) | 0.334 (B512x128_L1128x64, 8) | 0.19 |
| layer3 | 0.391 (B256x256, 16) | 0.317 (B256x256_L164x128, 8) | 0.16 |
| layer4 | 0.460 (B256x512, 16) | 0.322 (B64x512_L132x128, 8) | 0.20 |

Fewer cores win at these sizes (the per-launch floor grows with the herd: ~0.21 ms at 4 cores,
~0.39 ms at 32). Still ~0.3 ms floor per launch vs 0.15-0.2 ms for the whole CPU conv.
L1 tiles of 128x128 int32 (16384 words) fail: AIE DMA BD max is 16383 words.

At 224x224 the deep layers' x*y is just small (14x14, 7x7). With a larger input, batch 1:

| input | layer | M = x*y | cpu fp32 | npu kernel | npu e2e (incl. Python layout) |
|---|---|---|---|---|---|
| 512 | layer1 | 16384 | 0.83 | 1.52 | 1.81 |
| 512 | layer2 | 4096 | 0.70 | 0.67 | 0.92 |
| 512 | layer3 | 1024 | 0.68 | 0.69 | 1.01 |
| 512 | layer4 | 256 | 0.67 | 0.76 | 1.19 |
| 1024 | layer1 | 65536 | 3.39 | 4.75 | 8.57 |
| 1024 | layer2 | 16384 | 3.08 | 1.84 | 3.10 |
| 1024 | layer3 | 4096 | 2.73 | 1.22 | 1.94 |
| 1024 | layer4 | 1024 | 2.42 | 1.14 | 1.41 |

All bit-exact vs the CPU integers. 1024 layer2-4 run at 2.6-4.3 TOPS; layer1 (64 channels,
K step 64) at 1.0 TOPS. Each tap still re-reads its A block from DDR (9x the input traffic);
loading an x*y tile plus halo once into L2 for all 9 taps is the next compiler step.

## Requantize on the core after the K loop (tools/conv_implicit/epilogue.py, ep_isolate.py, ep_ir.py)

K is looped in time on one core (no split-K), so each accumulator is complete when the loop
ends and an elementwise tail can run on it in L1 before the write-back (Triton-XDNA
`generate_matmul_transform(..., epilogue=True)`). GEMM 512x512x1152 int8, B512x256_L1128x64_K128:

| tail | ms | out | |
|---|---|---|---|
| none (int32) | 0.413-0.431 | 1.05 MB | |
| `max(acc, 0)` int32 | 0.490 | 1.05 MB | exact |
| `acc.to(f32) * c` | 0.962 | 1.05 MB | ok with a constexpr c; a runtime scalar arrives as 0 (silently wrong) |
| `relu(acc * c).to(bf16)` | 0.932 | 0.52 MB | rel err 1% |
| `max(acc * c + .5, 0).to(int8)` | 1.310 | 0.26 MB | f32->i8 wraps instead of saturating in the conv runs |
| `(acc >> s).to(int8)`, 32 lanes | **0.416** | 0.26 MB | one `aievec.srs`; = round-half-up, saturating |

The f32 tails cost 0.5-0.9 ms (f32 elementwise on AIE2P); the SRS tail is free. In the conv:

| layer | int32 out | SRS int8 out |
|---|---|---|
| 224 layer1 | 0.586 ms, 1.05 MB | 0.565 ms, 0.26 MB |
| 224 layer3 | 0.317 ms, 0.26 MB | 0.327 ms, 0.07 MB |
| 1024 layer1 | 4.745 ms, 17.0 MB | 4.310 ms, 4.26 MB |
| 1024 layer3 | 1.137 ms, 4.72 MB | 1.043 ms, 1.18 MB |

Bit-exact against round-half-up. 4x fewer bytes but at most 9% faster: the int32 write-back was
not the bottleneck (17 MB in 4.7 ms = 3.6 GB/s). What it buys is an int8 output the next layer
can read directly.

Toolchain gaps hit:
* The epilogue herd vectorizes 16 lanes; AIE2P SRS (i32->i8) lowers only for 32/64 lanes
  ("failed to legalize aievec.srs"). Tiling the tail 4 rows x 8 (32 lanes) instead works.
* ReLU on the int8 result is hoisted before the truncation into a 32-lane i32 compare, which
  Peano cannot legalize (`G_ICMP <32 x s32>`). At 16 lanes the compare works but SRS does not.
  Needs a Peano legalizer split, or SRS with unsigned saturation (sign=0) as a free ReLU.
* Runtime scalars do not reach the epilogue herd (reads 0); constexpr only.
* `arith.minnumf` on f32 vectors and `aievec.mul_elem` i32 x i32 do not lower.
* examples/matmul_bf16_epilogue does not compile in this sandbox ("requires exactly one
  producer_op handle (got 2)").
