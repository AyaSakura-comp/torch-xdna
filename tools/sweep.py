"""Sweep NPU tilings of the W8A8 GEMM against the CPU's torch._int_mm, per (rows, N, K).

For every shape and row count, every tiling in the search space is compiled
by Triton-XDNA, checked bit-exact against the CPU and timed through the same
launch path torch_xdna uses (resident tensors, kernel[grid]). Results are
appended to a JSONL file (one line per shape x tiling, failures included with
the compiler's message) so a sweep can be resumed.

usage: python tools/sweep.py [--rows 128 256 512 1024] [--shapes 4096x1024 ...] [--out results/sweep.jsonl]
       python tools/sweep.py --summary results/sweep.jsonl
"""

import argparse
import itertools
import json
import os
import statistics
import time

import torch

from torch_xdna import kernels
from torch_xdna.kernels import Tiling

# OmniVoice's Qwen3-0.6B backbone after fusion: qkv, o, gate_up, down, audio heads (8200 -> 8704)
SHAPES = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072), (8704, 1024)]
MAX_HERD = (8, 4)  # npu2: herd rows along M map to the 8 columns, along N to the 4 core rows


def candidates(M, N, K):
    out = []
    for bm, bn, l1m, l1n, l2k in itertools.product((128, 256, 512), (256, 512, 1024),
                                                   (16, 32, 64), (64, 128, 256), (128, 256)):
        hm, hn = bm // l1m, bn // l1n
        if bm > M or M % bm or N % bn or K % l2k or bm % l1m or bn % l1n:
            continue
        if hm > MAX_HERD[0] or hn > MAX_HERD[1] or hm * hn < 16:
            continue
        out.append(Tiling(bm, bn, l1m, l1n, l2k))
    return out


def med_ms(fn, n=15):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def sweep(rows_list, shapes, out_path):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["M"], r["N"], r["K"], r["tiling"]))
    assert kernels.available(), kernels.unavailable_reason()
    res = kernels.resident()
    torch.manual_seed(0)
    with open(out_path, "a") as f:
        for M in rows_list:
            for N, K in shapes:
                a = torch.randint(-128, 128, (M, K), dtype=torch.int8)
                w = torch.randint(-128, 128, (K, N), dtype=torch.int8)
                ref = a.int() @ w.int()
                cpu_ms = med_ms(lambda: torch._int_mm(a, w))
                ar, wr = res.to_npu(a), res.to_npu(w, static=True)
                cr = res.empty((M, N), dtype=torch.int32)
                for t in candidates(M, N, K):
                    if (M, N, K, str(t)) in done:
                        continue
                    row = dict(M=M, N=N, K=K, tiling=str(t), herd=list(t.herd()), cpu_ms=round(cpu_ms, 4))
                    t0 = time.perf_counter()
                    try:
                        kernels.mm_i8(ar, wr, cr, t)
                        row["compile_s"] = round(time.perf_counter() - t0, 1)
                        row["exact"] = bool(torch.equal(cr.cpu(), ref))
                        row["npu_ms"] = round(med_ms(lambda: kernels.mm_i8(ar, wr, cr, t)), 4)
                        row["tops"] = round(2 * M * N * K / row["npu_ms"] / 1e9, 2)
                        row["npu_over_cpu"] = round(row["npu_ms"] / cpu_ms, 2)
                        row["status"] = "ok" if row["exact"] else "wrong"
                    except Exception as e:  # placement / lowering / runtime failure
                        msg = str(e)
                        errs = [l.strip() for l in msg.splitlines() if "error" in l.lower()]
                        row["status"] = "fail"
                        row["error"] = (errs[0] if errs else msg.splitlines()[-1] if msg else type(e).__name__)[:300]
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row), flush=True)


def summary(path):
    rows = [json.loads(l) for l in open(path)]
    best = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        k = (r["M"], r["N"], r["K"])
        if k not in best or r["npu_ms"] < best[k]["npu_ms"]:
            best[k] = r
    fails = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows)} runs, {len(fails)} failed/wrong")
    print(f"{'M':>5} {'N':>5} {'K':>5}  {'best tiling':26s} {'herd':7s} {'npu ms':>7} {'TOPS':>5} {'cpu ms':>7} {'npu/cpu':>7}")
    for k in sorted(best):
        r = best[k]
        print(f"{k[0]:5d} {k[1]:5d} {k[2]:5d}  {r['tiling']:26s} {str(tuple(r['herd'])):7s} {r['npu_ms']:7.3f} "
              f"{r['tops']:5.2f} {r['cpu_ms']:7.3f} {r['npu_over_cpu']:7.2f}")
    errs = {}
    for r in fails:
        errs.setdefault(r.get("error", r["status"])[:120], []).append(r["tiling"])
    for e, ts in errs.items():
        print(f"  {len(ts)}x {e}  e.g. {ts[:3]}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--shapes", nargs="+", default=None, help="NxK, e.g. 4096x1024")
    ap.add_argument("--out", default="results/sweep.jsonl")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--count", action="store_true", help="only print how many candidates")
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes] if args.shapes else SHAPES
    if args.count:
        for M in args.rows:
            print(M, {f"{n}x{k}": len(candidates(M, n, k)) for n, k in shapes})
    elif args.summary:
        summary(args.summary)
    else:
        sweep(args.rows, shapes, args.out)
