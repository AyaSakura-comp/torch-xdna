"""Is the ~0.29 ms per-launch fixed cost the ELF flow's per-run PDI reload?

The same GEMM is built twice by Triton-XDNA: as an ELF (runtime sequence =
aiex.configure (load PDI) + DMAs) and as an xclbin + insts.bin (PDI loaded
with the hw_context, the instruction stream only programs DMAs). Both are run
with raw pyxrt on one persistent context, so only the device side differs:

  1. each format, the same shape back to back
  2. two shapes alternating (two contexts on the same columns)
  3. fixed cost vs herd size (cores) at negligible work

usage: python tools/xclbin_exp.py [out.json]
"""

import glob
import json
import os

import shutil
import statistics
import sys
import time

import numpy as np
import pyxrt as xrt
import torch

from torch_xdna import kernels
from torch_xdna.kernels import Tiling

OUT = os.path.join(kernels.CACHE, "xclbin_exp")
S = xrt.xclBOSyncDirection


def build(fmt, M, N, K, t, cols=None):
    """Compile (and once run) the GEMM in this output format in a child process; return its artifact dir.

    The format is fixed when Triton-XDNA creates a kernel's launcher, so each
    format needs a process of its own.
    """
    import subprocess

    d = os.path.join(OUT, f"{fmt}_{M}x{N}x{K}_{t}" + (f"_{cols}col" if cols else ""))
    if not (os.path.isdir(d) and any(f.startswith("aie.") for f in os.listdir(d))):
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, AMD_TRITON_NPU_OUTPUT_FORMAT=fmt,
                   TRITON_CACHE_DIR=os.path.join(d, "triton_cache"),  # fresh: a hit leaves no artifact
                   AMD_TRITON_NPU_DEVICE_COLUMNS=str(cols or ""))
        subprocess.run([sys.executable, __file__, "--build", d, str(M), str(N), str(K), *map(str, t)],
                       env=env, check=True, capture_output=True)
    return d


def _build_child(d, M, N, K, t):
    res = kernels.resident()
    a = res.to_npu(torch.randint(-128, 128, (M, K), dtype=torch.int8))
    w = res.to_npu(torch.randint(-128, 128, (K, N), dtype=torch.int8), static=True)
    c = res.empty((M, N), dtype=torch.int32)
    cache = os.environ["TRITON_CACHE_DIR"]
    proj = os.path.join(d, "air_project")
    before = set(glob.glob(os.path.join(cache, "*", "aie.*")) + glob.glob(os.path.join(proj, "aie.*")))
    kernels.mm_i8(a, w, c, t, project=proj)
    new = set(glob.glob(os.path.join(cache, "*", "aie.*")) + glob.glob(os.path.join(proj, "aie.*"))) - before
    # the cache is private to this build dir, so a hit (an earlier attempt compiled but failed later) is ours
    new = new or set(glob.glob(os.path.join(cache, "*", "aie.*")))
    assert new, "no artifact"
    art = os.path.dirname(sorted(new, key=os.path.getmtime)[-1])
    for name in ("aie.elf", "aie.xclbin", "insts.bin", "elf_kernel_name.txt"):
        if os.path.exists(os.path.join(art, name)):
            shutil.copy(os.path.join(art, name), d)


class Runner:
    def __init__(self, dev, d, M, N, K):
        self.M, self.N, self.K = M, N, K
        if os.path.exists(os.path.join(d, "aie.xclbin")):
            self.fmt = "xclbin"
            xb = xrt.xclbin(os.path.join(d, "aie.xclbin"))
            dev.register_xclbin(xb)
            self.ctx = xrt.hw_context(dev, xb.get_uuid())
            kname = xb.get_kernels()[0].get_name()
            self.kern = xrt.kernel(self.ctx, kname)
            instr = np.fromfile(os.path.join(d, "insts.bin"), dtype=np.uint32)
            self.n_instr = len(instr)
            self.bo_i = xrt.bo(dev, instr.nbytes, xrt.bo.cacheable, self.kern.group_id(1))
            self.bo_i.write(instr.tobytes(), 0)
            self.bo_i.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
            gid = [self.kern.group_id(i) for i in (3, 4, 5)]
            self.bos = [xrt.bo(dev, n, xrt.bo.host_only, g) for n, g in zip((M * K, K * N, M * N * 4), gid)]
        else:
            self.fmt = "elf"
            kname = open(os.path.join(d, "elf_kernel_name.txt")).read().strip()
            self.ctx = xrt.hw_context(dev, xrt.elf(os.path.join(d, "aie.elf")))
            self.kern = xrt.ext.kernel(self.ctx, kname)
            self.bos = [xrt.ext.bo(dev, n) for n in (M * K, K * N, M * N * 4)]
        rng = np.random.default_rng(0)
        self.a = rng.integers(-128, 128, (M, K), dtype=np.int8)
        self.w = rng.integers(-128, 128, (K, N), dtype=np.int8)
        for bo, arr in zip(self.bos[:2], (self.a, self.w)):
            bo.write(arr.tobytes(), 0)
            bo.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)

    def run(self):
        if self.fmt == "xclbin":
            r = self.kern(3, self.bo_i, self.n_instr, *self.bos)
            r.wait()
        else:
            r = xrt.run(self.kern)
            for i, b in enumerate(self.bos):
                r.set_arg(i, b)
            r.start()
            r.wait2()

    def exact(self):
        self.run()
        self.bos[2].sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
        c = np.frombuffer(self.bos[2].read(self.M * self.N * 4, 0), dtype=np.int32).reshape(self.M, self.N)
        return bool(np.array_equal(c, self.a.astype(np.int32) @ self.w.astype(np.int32)))


def med(fn, n=50):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return round(statistics.median(ts) * 1e3, 4)


def main():
    assert kernels.available(), kernels.unavailable_reason()
    dev = xrt.device(0)
    out = {}
    t128 = kernels.default_tiling(128)
    shapes = {"qkv": (128, 4096, 1024), "o": (128, 1024, 2048)}
    runners = {}
    for fmt in ("elf", "xclbin"):
        for name, (M, N, K) in shapes.items():
            runners[fmt, name] = Runner(dev, build(fmt, M, N, K, t128), M, N, K)
            r = runners[fmt, name]
            out[f"{fmt} {name} exact"] = r.exact()
            out[f"{fmt} {name} same-shape ms"] = med(r.run)
        q, o = runners[fmt, "qkv"], runners[fmt, "o"]
        out[f"{fmt} alternate qkv/o ms per run"] = round(med(lambda: (q.run(), o.run())) / 2, 4)
        print(json.dumps({k: v for k, v in out.items() if k.startswith(fmt)}), flush=True)

    # fixed cost vs herd size at negligible work: M=128, N=512, K=128
    for t in (Tiling(128, 512, 64, 128, 128), Tiling(128, 512, 32, 128, 128), Tiling(128, 512, 16, 128, 128),
              Tiling(128, 256, 64, 128, 128), Tiling(128, 256, 128, 128, 128)):
        for fmt in ("elf", "xclbin"):
            try:
                r = Runner(dev, build(fmt, 128, 512, 128, t), 128, 512, 128)
                hm, hn = t.herd()
                out[f"{fmt} tiny {t} cores={hm * hn}"] = med(r.run)
            except Exception as e:
                out[f"{fmt} tiny {t}"] = f"fail: {str(e)[:150]}"
            print(fmt, t, out.get(f"{fmt} tiny {t} cores={t.herd()[0] * t.herd()[1]}", out.get(f"{fmt} tiny {t}")),
                  flush=True)
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--build"]:
        assert kernels.available(), kernels.unavailable_reason()
        d, M, N, K, *t = sys.argv[2:]
        _build_child(d, int(M), int(N), int(K), Tiling(*map(int, t)))
    else:
        main()
