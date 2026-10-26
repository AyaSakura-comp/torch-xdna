"""Re-time torch._int_mm for every (M, N, K) of a sweep, robust to a busy box (min of many runs).

usage: python tools/cpu_baseline.py results/sweep.jsonl results/cpu_baseline.json
"""
import json
import sys
import time

import torch

torch.set_num_threads(16)
shapes = sorted({(r["M"], r["N"], r["K"]) for r in map(json.loads, open(sys.argv[1]))})
out = {}
for M, N, K in shapes:
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-128, 128, (K, N), dtype=torch.int8)
    for _ in range(5):
        torch._int_mm(a, w)
    ts = []
    for _ in range(60):
        t = time.perf_counter()
        torch._int_mm(a, w)
        ts.append(time.perf_counter() - t)
    ts.sort()
    out[f"{M}x{N}x{K}"] = dict(min_ms=round(ts[0] * 1e3, 4), p25_ms=round(ts[15] * 1e3, 4),
                               median_ms=round(ts[30] * 1e3, 4))
    print(M, N, K, out[f"{M}x{N}x{K}"], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
