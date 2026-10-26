"""3x3 stride-1 int8 convolution on NC columns x 4 rows of the NPU, cascade over input channels.

Column c computes output channels [c*Co/NC, (c+1)*Co/NC); within a column the four cores form a
cascade chain over the four input-channel slices (see conv3x3_cascade.cc). Data movement:
  input   memtile r (r < 4) receives slice r of every padded input row and broadcasts it to the
          core of chain position r in every column (sliding 3-line window in local memory)
  weights memtile c receives column c's four weight slices and splits them, kept for the run
  output  the tail of column c sends its output rows through memtile c to the host
That is 4 + 1 + 1 MM2S channels on memtiles 0-3 and 4 + 1 on the others.

usage (IRON env): python conv3x3_array.py [--cols 8] [--rows 64] [--ci 256] [--co 256]
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
R = 4                       # cores per chain = rows 2..5


def geometry(ci, co, ow, nc, chain=R):
    """Channel groups per core, padded line width, output-channel pairs per chain."""
    cg = ci // chain // 8
    wp = ((ow + 2 + 7) // 8) * 8
    if wp < ow + 8:
        wp += 8
    ogp = co // nc // (R // chain) // 16
    return cg, wp, ogp


def _conv3x3_array(x_in: In, w_in: In, y_out: Out, *, NC: CompileTime[int], H: CompileTime[int],
                   OW: CompileTime[int], WP: CompileTime[int], CG: CompileTime[int], OGP: CompileTime[int],
                   shift: CompileTime[int], obj: CompileTime[str], L: CompileTime[int] = R,
                   relu: CompileTime[bool] = False, trace_size: CompileTime[int] = 0):
    K = R // L                            # chains per column
    line = CG * WP * 8                    # one chain position's channel slice of a padded input row
    wsl = OGP * 3 * CG * 3 * 2 * 64       # one core's weights
    ochain = 2 * OGP * OW * 8             # one chain's part of an output row
    oline = K * ochain                    # one column's part of an output row
    i8 = np.dtype[np.int8]
    line_ty = np.ndarray[(line,), i8]
    w_ty = np.ndarray[(wsl,), i8]
    wcol_ty = np.ndarray[(R * wsl,), i8]
    o_ty = np.ndarray[(ochain,), i8]
    ocol_ty = np.ndarray[(oline,), i8]

    kargs = [line_ty, line_ty, line_ty, w_ty, line_ty, np.int32]
    k_head = Kernel("conv3x3_head", obj, kargs, stack_size_override=1024)
    k_mid = Kernel("conv3x3_mid", obj, kargs, stack_size_override=1024)
    k_tail = Kernel("conv3x3_tail_relu" if relu else "conv3x3_tail", obj,
                    [line_ty, line_ty, line_ty, w_ty, o_ty, np.int32], stack_size_override=1024)

    # input slice h: shim h -> memtile h -> chain position h of every chain (broadcast)
    x_l3 = [ObjectFifo(line_ty, depth=2, name=f"x_l3_{h}") for h in range(L)]
    x_l1 = [x_l3[h].cons().forward(obj_type=line_ty, depth=4, name=f"x_l1_{h}", tile=Tile(h, 1))
            for h in range(L)]
    # weights of column c: shim c -> memtile c -> split over its four cores
    w_l3 = [ObjectFifo(wcol_ty, depth=1, name=f"w_l3_{c}") for c in range(NC)]
    w_l1 = [w_l3[c].cons().split([r * wsl for r in range(R)], tile=Tile(c, 1), depths=[1] * R,
                                 obj_types=[w_ty] * R, names=[f"w_l1_{c}_{r}" for r in range(R)])
            for c in range(NC)]
    # outputs of column c: its K tails -> joined in memtile c -> shim c
    y_l3 = [ObjectFifo(ocol_ty, depth=2, name=f"y_l3_{c}") for c in range(NC)]
    y_l1 = [y_l3[c].prod().join([k * ochain for k in range(K)], tile=Tile(c, 1), depths=[2] * K,
                                obj_types=[o_ty] * K, names=[f"y_l1_{c}_{k}" for k in range(K)])
            for c in range(NC)]

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

    trace = 1 if trace_size else None
    workers, chains = [], []
    for c in range(NC):
        for k in range(K):
            chain = []
            for h in range(L):
                q = k * L + h                       # core of the column, top (row 5) first
                tile = Tile(c, 5 - q)
                if h == L - 1:
                    wk = Worker(tail_body, [x_l1[h].cons(), w_l1[c][q].cons(), y_l1[c][k].prod(), k_tail],
                                tile=tile, stack_size=0x800, trace=trace)
                else:
                    wk = Worker(chain_body, [x_l1[h].cons(), w_l1[c][q].cons(), k_head if h == 0 else k_mid],
                                tile=tile, stack_size=0x800, trace=trace)
                chain.append(wk)
            for h in range(L - 1):
                CascadeFlow(chain[h], chain[h + 1])
            workers += chain
            chains.append(chain)

    rows_in = H + 2
    co8 = 2 * OGP * K * NC                # output channel groups of the whole layer
    # x_in is [H+2][L][line]; slice h of every row. w_in is [NC][R*wsl]. y_out is [H][co8][OW][8],
    # column c owning groups [2*OGP*K*c, 2*OGP*K*(c+1)) of every row.
    x_taps = [TensorAccessPattern((rows_in, L * line), offset=h * line, sizes=[1, 1, rows_in, line],
                                  strides=[0, 0, L * line, 1]) for h in range(L)]
    w_taps = [TensorAccessPattern((NC, R * wsl), offset=c * R * wsl, sizes=[1, 1, 1, R * wsl],
                                  strides=[0, 0, 0, 1]) for c in range(NC)]
    y_taps = [TensorAccessPattern((H, co8 * OW * 8), offset=c * oline, sizes=[1, 1, H, oline],
                                  strides=[0, 0, co8 * OW * 8, 1]) for c in range(NC)]

    def sequence(xi, wi, yo, *handles):
        xps, wps, ycs = handles[:L], handles[L:L + NC], handles[L + NC:]
        for c in range(NC):
            wps[c].fill(wi, w_taps[c])
        for h in range(L):
            xps[h].fill(xi, x_taps[h])
        for c in range(NC):
            ycs[c].drain(yo, y_taps[c], wait=True)

    x_ty = np.ndarray[(rows_in * L * line,), i8]
    w_ty_all = np.ndarray[(NC * R * wsl,), i8]
    y_ty = np.ndarray[(H * co8 * OW * 8,), i8]
    handles = ([x_l3[h].prod() for h in range(L)] + [w_l3[c].prod() for c in range(NC)]
               + [y_l3[c].cons() for c in range(NC)])
    rt = Runtime(sequence, [x_ty, w_ty_all, y_ty] + handles)
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace_size:
        from aie.utils.trace.events import CoreEventAIE2P as CE
        prog.enable_trace(trace_size=trace_size, workers=workers[:2 * R],
                          coretile_events=[CE.INSTR_VECTOR, CE.LOCK_STALL, CE.CASCADE_STALL, CE.MEMORY_STALL,
                                           CE.STREAM_STALL, CE.INSTR_CALL, CE.INSTR_RETURN, CE.ACTIVE])
    return prog.resolve_program()


conv3x3_array = iron.jit(_conv3x3_array)


# ---------------------------------------------------------------- host-side layouts and reference
def pack_input(x, wp):
    """x [Ci, H, W] int8 -> padded rows [H+2][Ci/8][WP][8] (pixel x of a row at column x+1)."""
    ci, h, w = x.shape
    out = np.zeros((h + 2, ci // 8, wp, 8), np.int8)
    out[1:h + 1, :, 1:w + 1, :] = x.reshape(ci // 8, 8, h, w).transpose(2, 0, 3, 1)
    return out


def pack_weights(wt, nc, cg, ogp, chain=R):
    """wt [Co, Ci, 3, 3] -> [NC][K chains][L positions][OGP][ky][CG][kx][2][8 ci][8 co]."""
    k = R // chain
    w = wt.reshape(nc, k, ogp, 2, 8, chain, cg, 8, 3, 3)   # [c, k, p, j, co8, h, g, ci8, ky, kx]
    w = w.transpose(0, 1, 5, 2, 8, 6, 9, 3, 7, 4)          # [c, k, h, p, ky, g, kx, j, ci8, co8]
    return np.ascontiguousarray(w).reshape(-1)


def unpack_output(y, h, co, ow):
    return y.reshape(h, co // 8, ow, 8).transpose(1, 3, 0, 2).reshape(co, h, ow)


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
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--ci", type=int, default=256)
    ap.add_argument("--co", type=int, default=256)
    ap.add_argument("--ow", type=int, default=64)
    ap.add_argument("--shift", type=int, default=12)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--obj", default=os.path.join(HERE, "build", "conv3x3_cascade.o"))
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--chain", type=int, default=R, help="cores per cascade chain (4 or 2)")
    a = ap.parse_args()
    nc = a.cols
    cg, wp, ogp = geometry(a.ci, a.co, a.ow, nc, a.chain)
    rng = np.random.default_rng(0)
    x = rng.integers(-60, 60, (a.ci, a.rows, a.ow), dtype=np.int8)
    wt = rng.integers(-40, 40, (a.co, a.ci, 3, 3), dtype=np.int8)
    xt = iron.tensor(pack_input(x, wp).reshape(-1), dtype=np.int8, device="npu")
    wtt = iron.tensor(pack_weights(wt, nc, cg, ogp, a.chain), dtype=np.int8, device="npu")
    yt = iron.zeros(a.rows * a.co * a.ow, dtype=np.int8, device="npu")
    kw = dict(NC=nc, H=a.rows, OW=a.ow, WP=wp, CG=cg, OGP=ogp, shift=a.shift, obj=a.obj, L=a.chain,
              relu=a.relu)
    if a.trace:
        from aie.utils.trace import TraceConfig
        tc = TraceConfig(a.trace, trace_file=os.path.join(HERE, "build", "trace_array.txt"))
        iron.jit(_conv3x3_array, trace_config=tc)(xt, wtt, yt, **kw, trace_size=a.trace)
        print("trace:", tc.trace_file, "mlir:", tc.physical_mlir_path)
        return
    conv3x3_array(xt, wtt, yt, **kw)
    got = unpack_output(yt.numpy(), a.rows, a.co, a.ow)
    want = reference(x, wt, a.shift, a.relu)
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    print(f"exact={np.array_equal(got, want)} max|diff|={d.max()} mismatches={int((d > 0).sum())} of {d.size}")
    from aie.utils.benchmark import run_iters, other_npu_users
    bench = run_iters(conv3x3_array, xt, wtt, yt, warmup=2, iters=a.iters, **kw)
    npu_us = bench.npu.median_us if bench.npu is not None else float("nan")
    print(f"npu time {npu_us:.1f} us (dispatch {bench.dispatch.median_us if bench.dispatch else float('nan'):.1f}, "
          f"empty dispatch {bench.dispatch_overhead.median_us if bench.dispatch_overhead else float('nan'):.1f}, "
          f"e2e {bench.e2e.median_us:.1f}); other NPU users: {other_npu_users() or 'none'}")
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        conv3x3_array(xt, wtt, yt, **kw)
        ts.append(time.perf_counter() - t0)
    ms = sorted(ts)[len(ts) // 2] * 1e3
    macs = a.rows * a.ow * a.co * a.ci * 9
    ncores = nc * R
    print(f"cols={nc} rows={a.rows} ci={a.ci} co={a.co} ow={a.ow}: median {ms:.3f} ms, "
          f"{2 * macs / ms / 1e9:.2f} TOPS on {ncores} cores (peak at 1.8 GHz: {ncores * 1.8432:.1f})")


if __name__ == "__main__":
    main()
