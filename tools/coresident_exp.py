"""Can several small (few-column) xclbin GEMM contexts stay resident side by side, so that
round-robin launches across shapes avoid the ~1.1 ms context switch?

For each herd width, builds the 4 OmniVoice layer GEMMs (M=128) as xclbins, opens all
four contexts, and times (a) each shape back to back, (b) round robin qkv->o->gate_up->down.
Also snapshots `xrt-smi examine -r aie-partitions` while the round robin runs.

usage: python tools/coresident_exp.py [out.json]
"""
import json
import subprocess
import sys
import threading

import pyxrt as xrt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from xclbin_exp import Runner, build, med  # noqa: E402

from torch_xdna import kernels  # noqa: E402
from torch_xdna.kernels import Tiling  # noqa: E402

SHAPES = {"qkv": (128, 4096, 1024), "o": (128, 1024, 2048), "gate_up": (128, 6144, 1024), "down": (128, 1024, 3072)}
# name -> (tiling, columns the binary claims; None = whole array)
TILINGS = {"herd(4,4) npu2_4col": (Tiling(128, 512, 32, 128), 4),
           "herd(2,2) npu2_2col": (Tiling(128, 256, 64, 128), 2),
           "herd(4,4) npu2 (full)": (Tiling(128, 512, 32, 128), None)}


def main():
    assert kernels.available(), kernels.unavailable_reason()
    dev = xrt.device(0)
    out = {}
    for tname, (t, cols) in TILINGS.items():
        rs = {n: Runner(dev, build("xclbin", *s, t, cols), *s) for n, s in SHAPES.items()}
        same = {n: med(r.run) for n, r in rs.items()}
        assert all(r.exact() for r in rs.values())
        snap = {}

        def smi():
            snap["txt"] = subprocess.run(["xrt-smi", "examine", "-r", "aie-partitions"],
                                         capture_output=True, text=True).stdout
        th = threading.Timer(1.0, smi)
        th.start()
        rr = med(lambda: [r.run() for r in rs.values()], n=400) / len(rs)
        th.join()
        parts = [l.strip() for l in snap.get("txt", "").splitlines() if "Columns" in l or "|" + str(__import__("os").getpid()) in l]
        out[tname] = dict(same_shape_ms=same, sum_same_ms=round(sum(same.values()), 4),
                          round_robin_ms_per_run=round(rr, 4), partitions=parts)
        print(tname, json.dumps(out[tname]), flush=True)
        del rs
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
