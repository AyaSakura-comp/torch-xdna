"""One xclbin hw_context for every GEMM shape of a layer: same PDI, a different insts.bin per run.

GEMMs that share tiling and K compile to byte-identical PDIs; only the DMA
instruction stream differs with N and M. So one context (PDI loaded once) can
run all of them by passing each shape's insts.bin with the run. K=2048 / 3072
GEMMs are split into K=1024 chunks whose int32 partials are summed on the CPU.

Times one OmniVoice layer (qkv, o, gate_up, down) at M rows three ways:
  shared   one context, 1 + 2 + 1 + 3 = 7 runs
  elf      the current path, one ELF per shape (4 runs, PDI reloaded each)
  cpu      torch._int_mm

usage: python tools/shared_ctx_exp.py [M ...]
"""
import os
import statistics
import sys
import time

import numpy as np
import pyxrt as xrt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xclbin_exp import build  # noqa: E402

from torch_xdna import kernels  # noqa: E402
from torch_xdna.kernels import Tiling  # noqa: E402

S = xrt.xclBOSyncDirection
KC = 1024
LAYER = {"qkv": (4096, 1024), "o": (1024, 2048), "gate_up": (6144, 1024), "down": (1024, 3072)}
T = Tiling(128, 512, 32, 128, 128)


class Shared:
    """One context (from the first build's xclbin) + one instruction BO per (M, N) at K=KC."""

    def __init__(self, dev, M, Ns):
        first = build("xclbin", M, Ns[0], KC, T)
        xb = xrt.xclbin(os.path.join(first, "aie.xclbin"))
        dev.register_xclbin(xb)
        self.ctx = xrt.hw_context(dev, xb.get_uuid())
        self.kern = xrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        self.dev, self.M = dev, M
        self.instr = {}
        for N in Ns:
            ins = np.fromfile(os.path.join(build("xclbin", M, N, KC, T), "insts.bin"), dtype=np.uint32)
            bo = xrt.bo(dev, ins.nbytes, xrt.bo.cacheable, self.kern.group_id(1))
            bo.write(ins.tobytes(), 0)
            bo.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
            self.instr[N] = (bo, len(ins))

    def bo(self, nbytes, arg):
        return xrt.bo(self.dev, nbytes, xrt.bo.host_only, self.kern.group_id(arg))

    def start(self, N, a, w, c):
        bo, n = self.instr[N]
        return self.kern(3, bo, n, a, w, c)


def np_view(bo, dtype, shape):
    return np.frombuffer(bo.map(), dtype=dtype, count=int(np.prod(shape))).reshape(shape)


def main(rows):
    assert kernels.available(), kernels.unavailable_reason()
    for M in rows:  # build everything first: a build runs once in a child, which needs a free hw_context
        for N in {n for n, _ in LAYER.values()}:
            build("xclbin", M, N, KC, T)
    dev = xrt.device(0)
    rng = np.random.default_rng(0)
    for M in rows:
        sh = Shared(dev, M, sorted({n for n, _ in LAYER.values()}))
        # per GEMM: weights split into K chunks, each its own BO; per chunk an A and a C BO
        jobs = {}
        for name, (N, K) in LAYER.items():
            a = rng.integers(-128, 128, (M, K), dtype=np.int8)
            w = rng.integers(-128, 128, (K, N), dtype=np.int8)
            chunks = []
            for k0 in range(0, K, KC):
                ab, wb, cb = sh.bo(M * KC, 3), sh.bo(KC * N, 4), sh.bo(M * N * 4, 5)
                np_view(ab, np.int8, (M, KC))[:] = a[:, k0:k0 + KC]
                np_view(wb, np.int8, (KC, N))[:] = w[k0:k0 + KC]
                for b in (ab, wb):
                    b.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
                chunks.append((ab, wb, cb))
            jobs[name] = (N, a, w, chunks)

        def run_gemm(name):
            N, _, _, chunks = jobs[name]
            for ab, wb, cb in chunks:
                sh.start(N, ab, wb, cb).wait()
            acc = None
            for _, _, cb in chunks:
                cb.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
                part = np_view(cb, np.int32, (M, N))
                acc = part.copy() if acc is None else acc + part
            return acc

        exact = {n: bool(np.array_equal(run_gemm(n), j[1].astype(np.int32) @ j[2].astype(np.int32)))
                 for n, j in jobs.items()}

        def layer():
            for n in LAYER:
                run_gemm(n)

        def med(fn, reps=40):
            for _ in range(5):
                fn()
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t)
            return statistics.median(ts) * 1e3

        shared_ms = med(layer)
        per = {n: round(med(lambda n=n: run_gemm(n)), 3) for n in LAYER}

        # current ELF path (torch_xdna kernels.mm_i8, default tiling for the bucket)
        res = kernels.resident()
        elf_jobs = []
        for name, (N, K) in LAYER.items():
            _, a, w, _ = jobs[name]
            elf_jobs.append((res.to_npu(torch.from_numpy(a)), res.to_npu(torch.from_numpy(w), static=True),
                             res.empty((M, N), dtype=torch.int32)))
        elf_ms = med(lambda: [kernels.mm_i8(a, w, c, M if M <= 512 else 512) for a, w, c in elf_jobs])
        cpu_t = [(torch.from_numpy(jobs[n][1]), torch.from_numpy(jobs[n][2])) for n in LAYER]
        cpu_ms = med(lambda: [torch._int_mm(a, w) for a, w in cpu_t])
        del sh, jobs
        print(f"M={M}: exact {exact}\n  per-layer: shared-context xclbin {shared_ms:.3f} ms (7 runs) | "
              f"ELF {elf_ms:.3f} ms (4 runs) | CPU int8 {cpu_ms:.3f} ms\n  shared per GEMM ms {per}", flush=True)


if __name__ == "__main__":
    main([int(m) for m in sys.argv[1:]] or [128, 256])
