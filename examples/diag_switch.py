"""Why is the first generate() after switching sentence ~0.4 s slower on the NPU? Time every launch."""
import time
from collections import defaultdict

import torch

import torch_xdna
from torch_xdna import kernels

torch.set_num_threads(16)
from omnivoice import OmniVoice  # noqa: E402

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
if hasattr(model.llm, "_orig_mod"):
    model.llm = model.llm._orig_mod
model.forward = torch.compile(model.forward, backend="xdna", options={"policy": "npu"})

log = []
orig = kernels.mm_i8


def timed(a, w, c, bm):
    t = time.perf_counter()
    orig(a, w, c, bm)
    log.append(((a.shape[0], w.shape[1], a.shape[1]), time.perf_counter() - t))


kernels.mm_i8 = timed
S, M = "好啊，我來看看。", "今天天氣不錯，我們下午可以去公園走走。"
for text in (S, M, S, M):  # warm both shapes
    model.generate(text=text, num_step=8)
for label, text in (("short", S), ("short", S), ("mid", M), ("mid", M), ("short", S)):
    log.clear()
    t = time.perf_counter()
    model.generate(text=text, num_step=8)
    wall = time.perf_counter() - t
    by = defaultdict(list)
    for shape, dt in log:
        by[shape].append(dt * 1e3)
    slow = [(s, round(d * 1e3, 1)) for s, d in log if d > 0.005]
    print(f"{label}: wall {wall:.3f}s, NPU sum {sum(d for _, d in log):.3f}s, launches {len(log)}, "
          f"slow(>5ms) {len(slow)}: {slow[:8]}")
    print("   per shape median ms:", {f"{k[0]}x{k[1]}x{k[2]}": round(sorted(v)[len(v) // 2], 3) for k, v in by.items()})
