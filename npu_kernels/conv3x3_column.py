"""One NPU column computing a 3x3 int8 convolution with a cascade chain over input channels.

The four cores of the column each hold one quarter of the input channels: their weights stay in
local memory, a three-line window of their channel slice slides down the image, and the partial
sums of every 16-pixel x 16-channel block travel head -> mid -> mid -> tail over the cascade. The
tail shift-round-saturates to int8 and writes the output line.

usage (IRON env):  python conv3x3_column.py [--rows H] [--iters N]
"""

import argparse
import os
import time

import numpy as np

import aie.iron as iron
from aie.iron import CascadeFlow, CompileTime, In, Kernel, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = os.path.dirname(os.path.abspath(__file__))
R = 4                       # chain length = cores per column (rows 2..5)


def geometry(ci, co, ow):
    cg = ci // R // 8            # channel groups per core
    wp = ((ow + 2 + 7) // 8) * 8
    if wp < ow + 8:
        wp += 8
    ogp = co // 16
    return cg, wp, ogp


def _conv3x3_column(x_in: In, w_in: In, y_out: Out, *, H: CompileTime[int], OW: CompileTime[int],
                   WP: CompileTime[int], CG: CompileTime[int], OGP: CompileTime[int],
                   shift: CompileTime[int], obj: CompileTime[str], col: CompileTime[int] = 0,
                   trace_size: CompileTime[int] = 0):
    line = CG * WP * 8                    # one core's channel slice of a padded input row
    wsl = OGP * 3 * CG * 3 * 2 * 64       # one core's weights
    oline = 2 * OGP * OW * 8              # one output row
    i8 = np.dtype[np.int8]
    line_ty = np.ndarray[(line,), i8]
    row_ty = np.ndarray[(R * line,), i8]
    w_ty = np.ndarray[(wsl,), i8]
    wall_ty = np.ndarray[(R * wsl,), i8]
    o_ty = np.ndarray[(oline,), i8]

    kargs = [line_ty, line_ty, line_ty, w_ty, line_ty, np.int32]
    k_head = Kernel("conv3x3_head", obj, kargs, stack_size_override=1024)
    k_mid = Kernel("conv3x3_mid", obj, kargs, stack_size_override=1024)
    k_tail = Kernel("conv3x3_tail", obj, [line_ty, line_ty, line_ty, w_ty, o_ty, np.int32], stack_size_override=1024)

    # input rows: DDR -> memtile of this column -> one channel slice per core
    x_l3 = ObjectFifo(row_ty, depth=2, name="x_l3")
    x_l1 = x_l3.cons().split([r * line for r in range(R)], tile=Tile(col, 1), depths=[4] * R,
                             obj_types=[line_ty] * R, names=[f"x_l1_{r}" for r in range(R)])
    # weights: DDR -> the next column's memtile (DMA channels) -> one slice per core, kept
    w_l3 = ObjectFifo(wall_ty, depth=1, name="w_l3")
    w_l1 = w_l3.cons().split([r * wsl for r in range(R)], tile=Tile(col + 1, 1), depths=[1] * R,
                             obj_types=[w_ty] * R, names=[f"w_l1_{r}" for r in range(R)])
    # output rows: tail -> memtile -> DDR
    y_l1 = ObjectFifo(o_ty, depth=2, name="y_l1")
    y_l3 = y_l1.cons().forward(obj_type=o_ty, depth=2, name="y_l3", tile=Tile(col, 1))

    def chain_body(x, w, kern):
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

    # chain position r holds input-channel slice r; the head sits at the top (row 5), the tail
    # next to the memtile (row 2), cascade flowing downwards as in mlir-aie's cascade matmul
    workers = []
    for r in range(R):
        tile = Tile(col, 5 - r)
        if r == R - 1:
            workers.append(Worker(tail_body, [x_l1[r].cons(), w_l1[r].cons(), y_l1.prod(), k_tail],
                                  tile=tile, stack_size=0x800, trace=1 if trace_size else None))
        else:
            workers.append(Worker(chain_body, [x_l1[r].cons(), w_l1[r].cons(),
                                               k_head if r == 0 else k_mid],
                                  tile=tile, stack_size=0x800, trace=1 if trace_size else None))
    for r in range(R - 1):
        CascadeFlow(workers[r], workers[r + 1])

    def sequence(xi, wi, yo, xp, wp, yc):
        wp.fill(wi)
        xp.fill(xi)
        yc.drain(yo, wait=True)

    x_ty = np.ndarray[((H + 2) * R * line,), i8]
    y_ty = np.ndarray[(H * oline,), i8]
    rt = Runtime(sequence, [x_ty, wall_ty, y_ty, x_l3.prod(), w_l3.prod(), y_l3.cons()])
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size:
        from aie.utils.trace.events import CoreEventAIE2P as CE
        prog.enable_trace(trace_size=trace_size, workers=workers,
                          coretile_events=[CE.INSTR_VECTOR, CE.LOCK_STALL, CE.CASCADE_STALL, CE.MEMORY_STALL,
                                           CE.STREAM_STALL, CE.INSTR_CALL, CE.INSTR_RETURN, CE.ACTIVE])
    return prog.resolve_program()


conv3x3_column = iron.jit(_conv3x3_column)


# ---------------------------------------------------------------- host-side layouts and reference
def pack_input(x, wp):
    """x [Ci, H, W] int8 -> padded rows [H+2][Ci/8][WP][8] (pixel x of a row sits at column x+1)."""
    ci, h, w = x.shape
    out = np.zeros((h + 2, ci // 8, wp, 8), np.int8)
    out[1:h + 1, :, 1:w + 1, :] = x.reshape(ci // 8, 8, h, w).transpose(2, 0, 3, 1)
    return out


def pack_weights(wt, cg, ogp):
    """wt [Co, Ci, 3, 3] int8 -> R slices, each [OGP][ky][CG][kx][2][8 ci][8 co] (consumption order)."""
    co, ci = wt.shape[:2]
    w = wt.reshape(ogp, 2, 8, R, cg, 8, 3, 3)            # [p, j, co8, r, g, ci8, ky, kx]
    w = w.transpose(3, 0, 6, 4, 7, 1, 5, 2)              # [r, p, ky, g, kx, j, ci8, co8]
    return np.ascontiguousarray(w).reshape(-1)


def unpack_output(y, h, co, ow):
    """[H][Co/8][OW][8] -> [Co, H, OW]."""
    return y.reshape(h, co // 8, ow, 8).transpose(1, 3, 0, 2).reshape(co, h, ow)


def reference(x, wt, shift):
    """int32 conv (pad 1), then round half up and saturate to int8, like SRS on the tail."""
    ci, h, w = x.shape
    xp = np.zeros((ci, h + 2, w + 2), np.int64)
    xp[:, 1:h + 1, 1:w + 1] = x
    acc = np.zeros((wt.shape[0], h, w), np.int64)
    for ky in range(3):
        for kx in range(3):
            acc += np.einsum("oc,chw->ohw", wt[:, :, ky, kx].astype(np.int64), xp[:, ky:ky + h, kx:kx + w])
    q = np.floor_divide(acc + (1 << (shift - 1)), 1 << shift)
    return np.clip(q, -128, 127).astype(np.int8), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--ci", type=int, default=256)
    ap.add_argument("--co", type=int, default=32)
    ap.add_argument("--ow", type=int, default=64)
    ap.add_argument("--shift", type=int, default=12)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--obj", default=os.path.join(HERE, "build", "conv3x3_cascade.o"))
    ap.add_argument("--trace", type=int, default=0, help="trace buffer bytes (0: off)")
    a = ap.parse_args()
    cg, wp, ogp = geometry(a.ci, a.co, a.ow)
    rng = np.random.default_rng(0)
    x = rng.integers(-60, 60, (a.ci, a.rows, a.ow), dtype=np.int8)
    wt = rng.integers(-40, 40, (a.co, a.ci, 3, 3), dtype=np.int8)
    xt = iron.tensor(pack_input(x, wp).reshape(-1), dtype=np.int8, device="npu")
    wtt = iron.tensor(pack_weights(wt, cg, ogp), dtype=np.int8, device="npu")
    yt = iron.zeros(a.rows * a.co * a.ow, dtype=np.int8, device="npu")
    kw = dict(H=a.rows, OW=a.ow, WP=wp, CG=cg, OGP=ogp, shift=a.shift, obj=a.obj)
    if a.trace:
        from aie.utils.trace import TraceConfig
        tc = TraceConfig(a.trace, trace_file=os.path.join(HERE, "build", "trace.txt"))
        iron.jit(_conv3x3_column, trace_config=tc)(xt, wtt, yt, **kw, trace_size=a.trace)
        print("trace:", tc.trace_file, "mlir:", tc.physical_mlir_path)
        return
    conv3x3_column(xt, wtt, yt, **kw)
    got = unpack_output(yt.numpy(), a.rows, a.co, a.ow)
    want, acc = reference(x, wt, a.shift)
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    print(f"exact={np.array_equal(got, want)} max|diff|={d.max()} mismatches={int((d > 0).sum())} "
          f"of {d.size}; |acc| max {np.abs(acc).max()}")
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        conv3x3_column(xt, wtt, yt, **kw)
        ts.append(time.perf_counter() - t0)
    ms = sorted(ts)[len(ts) // 2] * 1e3
    macs = a.rows * a.ow * a.co * a.ci * 9
    print(f"rows={a.rows} ci={a.ci} co={a.co} ow={a.ow}: median {ms:.3f} ms, "
          f"{2 * macs / ms / 1e9:.2f} TOPS on 4 cores (peak 4 x 1.84 = 7.37)")


if __name__ == "__main__":
    main()
