"""3x3 convolutions on the NPU through torch.compile(backend="xdna").

A VGG-style stack of 3x3 convs (no bias, so each ReLU is fused into the NPU's requantization):
accuracy against fp32 eager, and time per forward against the same model compiled by Inductor for
the CPU. The NPU op itself (Triton-XDNA tl.extra.npu.conv3x3) is timed separately, without the host's
quantize / layout work, from its own launches.

    python examples/conv_xdna.py [--channels 256] [--size 64] [--layers 4] [--batch 1]
"""

import argparse
import time

import torch

import torch_xdna
from torch_xdna import runtime


class VGGBlock(torch.nn.Module):
    def __init__(self, c, layers):
        super().__init__()
        self.convs = torch.nn.ModuleList(torch.nn.Conv2d(c, c, 3, padding=1, bias=False) for _ in range(layers))

    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x


def timed(fn, x, reps):
    with torch.no_grad():
        fn(x)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            fn(x)
            ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    torch.manual_seed(0)
    net = VGGBlock(a.channels, a.layers).eval()
    for conv in net.convs:      # He init keeps the activations' scale through the stack
        torch.nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    x = torch.randn(a.batch, a.channels, a.size, a.size)
    with torch.no_grad():
        ref = net(x)
    gmac = a.layers * a.batch * a.channels ** 2 * a.size ** 2 * 9 / 1e9

    cpu = torch.compile(net)
    cpu_ms = timed(cpu, x, a.reps)
    torch._dynamo.reset()
    npu = torch.compile(net, backend="xdna", options={"policy": "npu"})
    t = time.perf_counter()
    with torch.no_grad():
        y = npu(x)
    first_s = time.perf_counter() - t
    runtime.stats.reset()
    npu_ms = timed(npu, x, a.reps)
    err = ((y - ref).norm() / ref.norm()).item()

    # the NPU op alone: the same launches without the host's quantize / pack / unpack
    from torch_xdna import conv as xconv

    op_ms = 0.0
    for _, layer in xconv._layers.values():
        op_ms += timed(lambda _, op=layer.ops[a.batch]: op.launch(), None, a.reps)
    print(f"{a.layers} x conv3x3 {a.channels}->{a.channels} on {a.batch}x{a.size}x{a.size} ({gmac:.2f} GMAC)")
    print(f"  CPU (Inductor, fp32):  {cpu_ms:8.2f} ms  {2 * gmac / cpu_ms:6.2f} TOPS")
    print(f"  NPU (xdna, int8):      {npu_ms:8.2f} ms  {2 * gmac / npu_ms:6.2f} TOPS end to end; rel err vs fp32 "
          f"{err:.4f}; first call {first_s:.1f} s (builds + calibration + verify)")
    print(f"  NPU launches alone:    {op_ms:8.2f} ms  {2 * gmac / op_ms:6.2f} TOPS (incl. ~launch overhead)")
    print(torch_xdna.report())


if __name__ == "__main__":
    main()
