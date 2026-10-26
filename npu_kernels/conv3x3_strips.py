"""3x3 stride-1 int8 convolution on the whole NPU: 16 two-core cascade chains over 32-pixel strips.

Each column runs two chains, one per 32-pixel strip of the image, both computing the column's 32
output channels. A chain's head holds input channels [0, Ci/2), its tail [Ci/2, Ci): a block of
16 pixels x 16 output channels reduces 9 taps x Ci/2 channels (576 VMACs for Ci = 256) per core,
and the head's partial sums go down the cascade to the tail, which writes int8 (SRS).

Per core: weights 36 KB (kept), a 4-line window of its strip and channel half (20 KB), 2 KB of
output: the input stream needs ~2.2 B/cycle, the reduction is long enough that the block
hand-over costs ~5%.

Data movement: memtile 2h+s (h channel half, s strip) broadcasts that input stream to the 8 cores
at chain position h of strip s; memtile c holds column c's two weight slices (each broadcast to
both chains) and forwards its two tails' output rows.

usage (IRON env): python conv3x3_strips.py [--rows 64] [--iters 10] [--relu]
"""

import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CascadeFlow, CompileTime, In, Kernel, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = os.path.dirname(os.path.abspath(__file__))
NC = 8                      # columns
YDEPTH = int(os.environ.get("CONV_YDEPTH", "4"))   # tail output buffers (slack between the two tails)
OWS, WPS = 32, 40           # strip output width, strip input line width (32 + halo, multiple of 8)


def _conv3x3_strips(x_in: In, w_in: In, y_out: Out, *, H: CompileTime[int], CI: CompileTime[int],
                    CO: CompileTime[int], OW: CompileTime[int], WP: CompileTime[int],
                    shift: CompileTime[int], obj: CompileTime[str], relu: CompileTime[bool] = False,
                    trace_size: CompileTime[int] = 0):
    S = OW // OWS                         # strips = chains per column
    assert S == 2 and CO == NC * 32 and CI % 16 == 0
    CG = CI // 2 // 8                     # channel groups per core (one half of the input channels)
    OGP = 2                               # 32 output channels per chain
    line = CG * WPS * 8
    wsl = OGP * 3 * CG * 3 * 2 * 64
    ostrip = 2 * OGP * OWS * 8            # one tail's output per row: 4 channel groups x 32 px
    i8 = np.dtype[np.int8]
    line_ty = np.ndarray[(line,), i8]
    w_ty = np.ndarray[(wsl,), i8]
    wcol_ty = np.ndarray[(2 * wsl,), i8]
    o_ty = np.ndarray[(ostrip,), i8]
    ocol_ty = np.ndarray[(S * ostrip,), i8]

    kargs = [line_ty, line_ty, line_ty, w_ty, line_ty, np.int32]
    k_head = Kernel("conv3x3_head", obj, kargs, stack_size_override=1024)
    k_tail = Kernel("conv3x3_tail_relu" if relu else "conv3x3_tail", obj,
                    [line_ty, line_ty, line_ty, w_ty, o_ty, np.int32], stack_size_override=1024)

    # input stream (h, s) through memtile 2h+s, broadcast to position h of strip s's chains
    x_l3 = {(h, s): ObjectFifo(line_ty, depth=2, name=f"x_l3_{h}{s}") for h in range(2) for s in range(S)}
    x_l1 = {(h, s): x_l3[h, s].cons().forward(obj_type=line_ty, depth=4, name=f"x_l1_{h}{s}",
                                              tile=Tile(2 * h + s, 1))
            for h in range(2) for s in range(S)}
    # weights of column c (head slice, tail slice) through memtile c, each to both chains
    w_l3 = [ObjectFifo(wcol_ty, depth=1, name=f"w_l3_{c}") for c in range(NC)]
    w_l1 = [w_l3[c].cons().split([0, wsl], tile=Tile(c, 1), depths=[1, 1], obj_types=[w_ty, w_ty],
                                 names=[f"w_l1_{c}_0", f"w_l1_{c}_1"]) for c in range(NC)]
    # outputs of column c's two tails, joined in memtile c (strip-major), one stream to the host
    y_l3 = [ObjectFifo(ocol_ty, depth=2, name=f"y_l3_{c}") for c in range(NC)]
    y_l1 = [y_l3[c].prod().join([s * ostrip for s in range(S)], tile=Tile(c, 1), depths=[YDEPTH] * S,
                                obj_types=[o_ty] * S, names=[f"y_l1_{c}{s}" for s in range(S)])
            for c in range(NC)]

    def head_body(x, w, kern):
        wl = w.acquire(1)
        for _ in range_(H):
            ls = x.acquire(3)
            kern(ls[0], ls[1], ls[2], wl, ls[0], shift)
            x.release(1)
        x.release(2)
        w.release(1)

    def tail_body(x, w, y, kern):
        wl = w.acquire(1)
        for _ in range_(H):
            ls = x.acquire(3)
            yo = y.acquire(1)
            kern(ls[0], ls[1], ls[2], wl, yo, shift)
            y.release(1)
            x.release(1)
        x.release(2)
        w.release(1)

    trace = 1 if trace_size else None
    workers = []
    for c in range(NC):
        for s in range(S):
            head = Worker(head_body, [x_l1[0, s].cons(), w_l1[c][0].cons(), k_head],
                          tile=Tile(c, 5 - 2 * s), stack_size=0x800, trace=trace)
            tail = Worker(tail_body, [x_l1[1, s].cons(), w_l1[c][1].cons(), y_l1[c][s].prod(), k_tail],
                          tile=Tile(c, 4 - 2 * s), stack_size=0x800, trace=trace)
            CascadeFlow(head, tail)
            workers += [head, tail]

    rows_in = H + 2
    # DDR layouts are stream-major so every transfer is one contiguous run per row:
    #   x_in  [2 halves][S strips][H+2][CG][WPS][8]   (strip halo columns duplicated on the host)
    #   y_out [H][NC][S strips][4 groups][32][8]      (one 2 KB run per column and row)
    xs_b = rows_in * line
    orow_b = NC * S * ostrip
    x_taps = {(h, s): TensorAccessPattern((2 * S, xs_b), offset=(h * S + s) * xs_b, sizes=[1, 1, 1, xs_b],
                                          strides=[0, 0, 0, 1])
              for h in range(2) for s in range(S)}
    w_taps = [TensorAccessPattern((NC, 2 * wsl), offset=c * 2 * wsl, sizes=[1, 1, 1, 2 * wsl],
                                  strides=[0, 0, 0, 1]) for c in range(NC)]
    y_taps = [TensorAccessPattern((H, orow_b), offset=c * S * ostrip, sizes=[1, 1, H, S * ostrip],
                                  strides=[0, 0, orow_b, 1]) for c in range(NC)]

    xkeys = [(h, s) for h in range(2) for s in range(S)]

    def sequence(xi, wi, yo, *hd):
        xps, wps, ycs = hd[:len(xkeys)], hd[len(xkeys):len(xkeys) + NC], hd[len(xkeys) + NC:]
        for c in range(NC):
            wps[c].fill(wi, w_taps[c])
        for i, k in enumerate(xkeys):
            xps[i].fill(xi, x_taps[k])
        for c in range(NC):
            ycs[c].drain(yo, y_taps[c], wait=True)

    x_ty = np.ndarray[(2 * S * xs_b,), i8]
    wall_ty = np.ndarray[(NC * 2 * wsl,), i8]
    y_ty = np.ndarray[(H * orow_b,), i8]
    hd = ([x_l3[k].prod() for k in xkeys] + [w_l3[c].prod() for c in range(NC)]
          + [y_l3[c].cons() for c in range(NC)])
    rt = Runtime(sequence, [x_ty, wall_ty, y_ty] + hd)
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size:
        from aie.utils.trace.events import CoreEventAIE2P as CE
        prog.enable_trace(trace_size=trace_size, workers=workers[:8],
                          coretile_events=[CE.INSTR_VECTOR, CE.LOCK_STALL, CE.CASCADE_STALL, CE.MEMORY_STALL,
                                           CE.STREAM_STALL, CE.INSTR_CALL, CE.INSTR_RETURN, CE.ACTIVE])
    return prog.resolve_program()


AIECC_FLAGS = ["--reclaim-runtime-bds"]     # long row loops split into more shim BDs than 16
conv3x3_strips = iron.jit(_conv3x3_strips, aiecc_flags=AIECC_FLAGS)


def pack_input(x, wp):
    """x [Ci, H, W] int8 -> [2 halves][2 strips][H+2][Ci/16][WPS][8]: padded (pixel x at column x+1),
    strip s covering padded columns [32s, 32s+40)."""
    ci, h, w = x.shape
    pad = np.zeros((h + 2, ci // 8, wp, 8), np.int8)
    pad[1:h + 1, :, 1:w + 1, :] = x.reshape(ci // 8, 8, h, w).transpose(2, 0, 3, 1)
    cg = ci // 16
    out = np.empty((2, w // OWS, h + 2, cg, WPS, 8), np.int8)
    for hh in range(2):
        for s in range(w // OWS):
            out[hh, s] = pad[:, hh * cg:(hh + 1) * cg, OWS * s:OWS * s + WPS, :]
    return out


def pack_weights(wt):
    """wt [Co, Ci, 3, 3] -> [NC][2 halves][2 og pairs][ky][CG][kx][2][8 ci][8 co]."""
    co, ci = wt.shape[:2]
    cg = ci // 2 // 8
    w = wt.reshape(NC, 2, 2, 8, 2, cg, 8, 3, 3)       # [c, p, j, co8, h, g, ci8, ky, kx]
    w = w.transpose(0, 4, 1, 7, 5, 8, 2, 6, 3)        # [c, h, p, ky, g, kx, j, ci8, co8]
    return np.ascontiguousarray(w).reshape(-1)


def unpack_output(y, h, co, ow):
    """[H][NC][S][4 groups][32][8] -> [Co, H, OW]."""
    s = ow // OWS
    y = y.reshape(h, NC, s, 4, OWS, 8).transpose(1, 3, 5, 0, 2, 4)   # [c, g, 8, h, s, 32]
    return y.reshape(co, h, ow)


def reference(x, wt, shift, relu=False):
    ci, h, w = x.shape
    xp = np.zeros((ci, h + 2, w + 2), np.int64)
    xp[:, 1:h + 1, 1:w + 1] = x
    acc = np.zeros((wt.shape[0], h, w), np.int64)
    for ky in range(3):
        for kx in range(3):
            acc += np.einsum("oc,chw->ohw", wt[:, :, ky, kx].astype(np.int64), xp[:, ky:ky + h, kx:kx + w])
    q = np.floor_divide(acc + (1 << (shift - 1)), 1 << shift)
    if relu:
        return np.clip(q, 0, 255).astype(np.uint8).view(np.int8)
    return np.clip(q, -128, 127).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--ci", type=int, default=256)
    ap.add_argument("--shift", type=int, default=12)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--obj", default=os.path.join(HERE, "build", "conv3x3_strip_cg16.o"))
    a = ap.parse_args()
    co, ow, wp = NC * 32, 64, 72
    rng = np.random.default_rng(0)
    x = rng.integers(-60, 60, (a.ci, a.rows, ow), dtype=np.int8)
    wt = rng.integers(-40, 40, (co, a.ci, 3, 3), dtype=np.int8)
    xt = iron.tensor(pack_input(x, wp).reshape(-1), dtype=np.int8, device="npu")
    wtt = iron.tensor(pack_weights(wt), dtype=np.int8, device="npu")
    yt = iron.zeros(a.rows * co * ow, dtype=np.int8, device="npu")
    kw = dict(H=a.rows, CI=a.ci, CO=co, OW=ow, WP=wp, shift=a.shift, obj=os.path.abspath(a.obj), relu=a.relu)
    if a.trace:
        from aie.utils.trace import TraceConfig
        tc = TraceConfig(a.trace, trace_file=os.path.join(HERE, "build", "trace_strips.txt"))
        iron.jit(_conv3x3_strips, trace_config=tc, aiecc_flags=AIECC_FLAGS)(xt, wtt, yt, **kw,
                                                                              trace_size=a.trace)
        print("trace:", tc.trace_file, "mlir:", tc.physical_mlir_path)
        return
    conv3x3_strips(xt, wtt, yt, **kw)
    got = unpack_output(yt.numpy(), a.rows, co, ow)
    want = reference(x, wt, a.shift, a.relu)
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    print(f"exact={np.array_equal(got, want)} max|diff|={d.max()} mismatches={int((d > 0).sum())} of {d.size}")
    from aie.utils.benchmark import run_iters, other_npu_users
    bench = run_iters(conv3x3_strips, xt, wtt, yt, warmup=2, iters=a.iters, **kw)
    npu_us = bench.npu.median_us if bench.npu is not None else float("nan")
    print(f"npu time {npu_us:.1f} us (dispatch {bench.dispatch.median_us if bench.dispatch else float('nan'):.1f}, "
          f"empty dispatch {bench.dispatch_overhead.median_us if bench.dispatch_overhead else float('nan'):.1f}, "
          f"e2e {bench.e2e.median_us:.1f}); other NPU users: {other_npu_users() or 'none'}")
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        conv3x3_strips(xt, wtt, yt, **kw)
        ts.append(time.perf_counter() - t0)
    ms = sorted(ts)[len(ts) // 2] * 1e3
    macs = a.rows * ow * co * a.ci * 9
    print(f"rows={a.rows} ci={a.ci} co={co} ow={ow}: median {ms:.3f} ms, {2 * macs / ms / 1e9:.2f} TOPS "
          f"on 32 cores (peak at 1.8 GHz: 59.0)")


if __name__ == "__main__":
    main()
