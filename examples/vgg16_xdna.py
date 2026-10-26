"""VGG-16 (torchvision, ImageNet weights) with every conv and linear on the NPU.

torch.compile(backend="xdna", policy "npu", min_rows=1, min_macs=0): the 13 3x3 convs
run as xdna::conv3x3 (tl.extra.npu.conv3x3), the 3 linears as xdna::linear; what is
left on the CPU (ReLU, max-pool, flatten, the int8 quantize / dequantize around each op)
is Inductor's. Compared with fp32 eager on public-domain test images (scikit-image /
matplotlib sample data): top-1 / top-5 agreement and logits cosine; timed against the
same model compiled by Inductor for the CPU.

    python examples/vgg16_xdna.py [--batch 1 8] [--reps 5]
"""

import argparse
import os
import time

import torch
import torchvision
from PIL import Image

import torch_xdna
from torch_xdna import runtime

_SK = os.path.expanduser("~/.local/lib/python3.13/site-packages/skimage/data")
_MPL = os.path.expanduser("~/.local/lib/python3.13/site-packages/matplotlib/mpl-data/sample_data")
IMAGES = [os.path.join(_SK, f) for f in ("astronaut.png", "chelsea.png", "coffee.png", "rocket.jpg",
                                         "motorcycle_left.png", "brick.png", "coins.png")] + \
    [os.path.join(_MPL, "grace_hopper.jpg")]


def layer_macs(model, x):
    """MACs of the convs and linears of one forward pass (hooks on the eager model)."""
    total = {"conv": 0, "linear": 0}

    def hook(m, inp, out):
        if isinstance(m, torch.nn.Conv2d):
            total["conv"] += out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        else:
            total["linear"] += out.numel() * m.in_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total


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
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    model = torchvision.models.vgg16(weights=weights).eval()
    cats = weights.meta["categories"]
    tf = weights.transforms()
    x_all = torch.stack([tf(Image.open(p).convert("RGB")) for p in IMAGES])
    with torch.no_grad():
        ref = model(x_all)
    macs = layer_macs(model, x_all[:1])
    print(f"VGG-16, 224x224: {macs['conv'] / 1e9:.2f} GMAC in convs, {macs['linear'] / 1e9:.3f} GMAC in linears "
          "per image")

    # accuracy: all images through the NPU model at the largest batch (and batch 1 for the first)
    torch._dynamo.reset()
    runtime.stats.reset()
    npu = torch.compile(model, backend="xdna", options=dict(policy="npu", min_rows=1, min_macs=0))
    t = time.perf_counter()
    with torch.no_grad():
        out = torch.cat([npu(x_all[i:i + 1]) for i in range(len(IMAGES))])
    first_s = time.perf_counter() - t
    s = runtime.stats.as_dict()
    npu_share = s["gmacs"].get("npu", 0) / max(1e-9, sum(s["gmacs"].values()))
    eager_gmac = (macs["conv"] + macs["linear"]) * len(IMAGES) / 1e9
    print(f"NPU share of conv+linear MACs: {100 * npu_share:.2f}% ({s['gmacs'].get('npu', 0):.2f} of "
          f"{eager_gmac:.2f} GMAC; calls {s['calls']}, NPU launches {s['npu_launches']}, fallbacks "
          f"{s['fallbacks']}); first pass incl. builds {first_s:.1f} s")
    cos = torch.nn.functional.cosine_similarity(ref, out, dim=1)
    top5_ref, top5_out = ref.topk(5, dim=1).indices, out.topk(5, dim=1).indices
    agree1 = (top5_ref[:, 0] == top5_out[:, 0]).sum().item()
    overlap5 = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top5_ref, top5_out)) / (5 * len(IMAGES))
    print(f"vs fp32 eager on {len(IMAGES)} images: top-1 {agree1}/{len(IMAGES)}, top-5 overlap "
          f"{100 * overlap5:.0f}%, logits cosine min {cos.min():.4f} mean {cos.mean():.4f}")
    for p, r, o in zip(IMAGES, top5_ref[:, 0], top5_out[:, 0]):
        print(f"  {os.path.basename(p):22s} fp32: {cats[r]:28s} npu: {cats[o]}")

    for b in a.batch:
        x = x_all[:b] if b <= len(IMAGES) else x_all.repeat((b + len(IMAGES) - 1) // len(IMAGES), 1, 1, 1)[:b]
        torch._dynamo.reset()
        cpu_ms = timed(torch.compile(model), x, a.reps)
        torch._dynamo.reset()
        npu = torch.compile(model, backend="xdna", options=dict(policy="npu", min_rows=1, min_macs=0))
        with torch.no_grad():
            npu(x)  # builds this batch size's programs
            npu(x)  # puts every conv into one hardware context (see torch_xdna.conv._share)
        runtime.stats.reset()
        npu_ms = timed(npu, x, a.reps)
        s = runtime.stats.as_dict()
        reps = a.reps + 1
        gop = 2 * (macs["conv"] + macs["linear"]) * b / 1e9
        print(f"batch {b}: CPU Inductor fp32 {cpu_ms:.1f} ms ({gop / cpu_ms:.2f} TOPS); NPU {npu_ms:.1f} ms "
              f"({gop / npu_ms:.2f} TOPS), of which inside the NPU ops {1e3 * s['seconds'].get('npu', 0) / reps:.1f} "
              f"ms; {s['npu_launches'] // reps} launches per forward")
    print(torch_xdna.report())


if __name__ == "__main__":
    main()
