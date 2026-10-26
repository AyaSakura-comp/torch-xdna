"""Do GEMM builds that share tiling and K but differ in N / M get byte-identical PDIs (same core
programs + array config), differing only in the DMA instruction stream (insts.bin)?"""
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xclbin_exp import build  # noqa: E402

from torch_xdna.kernels import Tiling  # noqa: E402


def digest(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:10]


T = Tiling(128, 512, 32, 128, 128)
SHAPES = [(128, 4096, 1024), (128, 6144, 1024), (128, 8704, 1024), (128, 1024, 1024), (256, 4096, 1024),
          (128, 1024, 2048)]
for M, N, K in SHAPES:
    d = build("xclbin", M, N, K, T)
    proj = os.path.join(d, "air_project")
    cdo = {os.path.basename(p): digest(p) for p in sorted(glob.glob(os.path.join(proj, "cdo_*", "*.bin")))
           if "empty" not in p and "main_" not in p}
    print(f"{M}x{N}x{K}: insts {digest(os.path.join(d, 'insts.bin'))} "
          f"({os.path.getsize(os.path.join(d, 'insts.bin'))} B)  cdo {cdo}", flush=True)
