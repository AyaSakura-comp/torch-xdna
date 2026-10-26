"""Runtime of the xdna::linear op: weight cache, routing NPU <-> CPU, fallback, stats.

Every offloaded GEMM is W8A8: weights per output channel, activations per
row, int8 x int8 -> int32 accumulate, dequantized in fp32. The NPU and the
CPU path (torch._int_mm) compute the same integers, so routing a shape to
either device gives bit-identical outputs.
"""

import logging
import threading
import time
import weakref
from collections import defaultdict

import torch

from . import kernels

log = logging.getLogger("torch_xdna")
_lock = threading.RLock()


# ------------------------------------------------------------------ stats
class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = defaultdict(int)       # device -> GEMM calls
        self.seconds = defaultdict(float)   # device -> wall seconds inside the op
        self.macs = defaultdict(int)        # device -> multiply-accumulates (unpadded)
        self.launches = 0                   # NPU launches (a call may split into several)
        self.fallbacks = []                 # (shape, error) when the NPU failed

    def as_dict(self):
        return dict(calls=dict(self.calls), seconds={k: round(v, 4) for k, v in self.seconds.items()},
                    gmacs={k: round(v / 1e9, 3) for k, v in self.macs.items()},
                    npu_launches=self.launches, fallbacks=list(self.fallbacks),
                    decisions={f"{k[0]}x{k[1]}x{k[2]}": v for k, v in decisions.items()},
                    npu_available=kernels.available(), npu_unavailable_reason=kernels.unavailable_reason())


# Autotune results outlive stats.reset(): (bucket, N, K) -> (device, npu_ms, cpu_ms)
decisions = {}
stats = Stats()


# ------------------------------------------------------------------ weights
def _round_up(x, m):
    return -(-x // m) * m


class QWeight:
    """One weight, quantized once: int8 [K, N] for the CPU, padded resident [Kp, Np] for the NPU."""

    def __init__(self, w: torch.Tensor, transposed: bool):
        wf = w.detach().float()
        if transposed:      # given as [K, N] (x @ W); store per output channel = per column
            wf = wf.t()
        self.N, self.K = wf.shape
        self.sw = wf.abs().amax(dim=1).clamp(min=1e-8) / 127          # [N]
        q = torch.round(wf / self.sw[:, None]).clamp(-127, 127).to(torch.int8)
        self.w_kn = q.t().contiguous()                                  # [K, N] for torch._int_mm
        self.Kp = _round_up(self.K, kernels.BLOCK_K)
        self.Np = _round_up(self.N, kernels.BLOCK_N)
        self._npu = None

    def npu(self):
        """The static resident copy, made on first NPU use."""
        if self._npu is None:
            wt = torch.zeros((self.Kp, self.Np), dtype=torch.int8)
            wt[: self.K, : self.N] = self.w_kn
            self._npu = kernels.resident().to_npu(wt, static=True)
        return self._npu


_weights = {}  # key -> (weakrefs to the source tensors, QWeight)


def _ref(t):
    try:
        return weakref.ref(t)
    except TypeError:
        return lambda: t


def qweight(ws, transposed: bool = False) -> QWeight:
    """The cached QWeight of one weight, or of several concatenated on the output dim (fused linears)."""
    ws = ws if isinstance(ws, (list, tuple)) else [ws]
    key = (tuple((w.data_ptr(), tuple(w.shape), w._version) for w in ws), transposed)
    with _lock:
        hit = _weights.get(key)
        if hit is not None and all(r() is w for r, w in zip(hit[0], ws)):
            return hit[1]
        src = ws[0] if len(ws) == 1 else torch.cat([w.detach() for w in ws])
        qw = QWeight(src, transposed)
        _weights[key] = ([_ref(w) for w in ws], qw)
        return qw


# ------------------------------------------------------------------ activation buffers
class _Staging:
    """Resident int8 activations / int32 results shared by every NPU launch (launches are serialized)."""

    a = c = None

    @classmethod
    def views(cls, M, Kp, Np):
        res = kernels.resident()
        if cls.a is None or cls.a.numel() < M * Kp:
            cls.a = res.empty(max(M * Kp, 1 << 20), dtype=torch.int8)
        if cls.c is None or cls.c.numel() < M * Np:
            cls.c = res.empty(max(M * Np, 1 << 20), dtype=torch.int32)
        return cls.a[: M * Kp].view(M, Kp), cls.c[: M * Np].view(M, Np)


# ------------------------------------------------------------------ the two devices
def _quant_rows(x2d):
    x = x2d.float()
    s = x.abs().amax(dim=1).clamp(min=1e-8) / 127
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s


def _bucket(rows, buckets):
    for b in buckets:
        if b >= rows:
            return b
    return buckets[-1]


def _block_m(bucket):
    return 512 if bucket % 512 == 0 else (256 if bucket % 256 == 0 else 128)


def _cpu_acc(xq, qw):
    return torch._int_mm(xq, qw.w_kn)


def _npu_acc(xq, qw, buckets):
    """int32 [R, N] from the NPU, as launches of at most the largest bucket's rows."""
    R = xq.shape[0]
    out = torch.empty((R, qw.N), dtype=torch.int32)
    w = qw.npu()
    top = buckets[-1]
    for r0 in range(0, R, top):
        r = min(top, R - r0)
        M = _bucket(r, buckets)
        a, c = _Staging.views(M, qw.Kp, qw.Np)
        a[:r, : qw.K].copy_(xq[r0: r0 + r])
        if qw.Kp != qw.K:
            a[:r, qw.K:].zero_()
        kernels.mm_i8(a, w, c, _block_m(M))
        stats.launches += 1
        out[r0: r0 + r].copy_(c[:r, : qw.N])
    return out


def _time(fn, reps=3):
    fn()  # first call compiles / warms
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def _choose(cfg, xq, qw):
    """'npu' or 'cpu' for this call."""
    R = xq.shape[0]
    if cfg.policy == "cpu" or R < cfg.min_rows or R * qw.N * qw.K < cfg.min_macs:
        return "cpu"
    if not kernels.available():
        return "cpu"
    if cfg.policy == "npu":
        return "cpu" if (_bucket(R, cfg.row_buckets), qw.N, qw.K) in _failed else "npu"
    key = (_bucket(R, cfg.row_buckets), qw.N, qw.K)
    if key in _failed:
        return "cpu"
    hit = decisions.get(key)
    if hit is None:
        npu_ms = _time(lambda: _npu_acc(xq, qw, cfg.row_buckets))
        cpu_ms = _time(lambda: _cpu_acc(xq, qw))
        hit = ("npu" if npu_ms < cpu_ms else "cpu", round(npu_ms, 3), round(cpu_ms, 3))
        decisions[key] = hit
        if cfg.verbose:
            log.warning("torch_xdna autotune %s: npu %.3f ms, cpu %.3f ms -> %s", key, npu_ms, cpu_ms, hit[0])
    return hit[0]


_failed = set()          # (bucket, N, K) whose NPU launch raised: CPU from then on
MAX_NPU_FAILURES = 3     # distinct failing shapes before the NPU is retired for the process


def _gemm(cfg, x2d, qw):
    """int32 accumulators [R, N], activation scales [R], and the device that ran it."""
    xq, sx = _quant_rows(x2d)
    with _lock:
        device = _choose(cfg, xq, qw)
        if device == "npu":
            try:
                return _npu_acc(xq, qw, cfg.row_buckets), sx, "npu"
            except Exception as e:  # compile / context / launch failure: this shape moves to the CPU
                shape = (_bucket(xq.shape[0], cfg.row_buckets), qw.N, qw.K)
                _failed.add(shape)
                stats.fallbacks.append((shape, f"{type(e).__name__}: {e}"[:300]))
                log.warning("torch_xdna: NPU GEMM %s failed (%s); it runs on the CPU from now on", shape, e)
                if len(_failed) >= MAX_NPU_FAILURES:
                    kernels.mark_broken(e)
                    log.warning("torch_xdna: %d NPU failures, NPU retired for this process", len(_failed))
        return _cpu_acc(xq, qw), sx, "cpu"


def _account(device, t0, rows, qw):
    stats.calls[device] += 1
    stats.seconds[device] += time.perf_counter() - t0
    stats.macs[device] += rows * qw.N * qw.K


def _fp_only(cfg):
    return cfg.cpu_kernel == "fp" and (cfg.policy == "cpu" or not kernels.available())


def linear(cfg, x, weight, bias, transposed):
    """y = x @ W^T + b (or x @ W when transposed) with W8A8 math on the chosen device."""
    shape = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    N = weight.shape[1] if transposed else weight.shape[0]
    if x2d.shape[0] == 0:
        return x.new_zeros(*shape, N)
    w_fp = weight.t() if transposed else weight
    if _fp_only(cfg):
        return torch.nn.functional.linear(x, w_fp.to(x.dtype), bias)
    t0 = time.perf_counter()
    qw = qweight(weight, transposed)
    acc, sx, device = _gemm(cfg, x2d, qw)
    if device == "cpu" and cfg.cpu_kernel == "fp":
        y = torch.nn.functional.linear(x2d, w_fp.to(x.dtype), bias)
    else:
        y = acc.float() * sx[:, None] * qw.sw[None, :]
        if bias is not None:
            y += bias.float()
        y = y.to(x.dtype)
    _account(device, t0, x2d.shape[0], qw)
    return y.reshape(*shape, N)


def linear_cat(cfg, x, weights, biases):
    """torch.cat([F.linear(x, w, b) for w, b in zip(weights, biases)], -1) as one W8A8 GEMM."""
    shape = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    N = sum(w.shape[0] for w in weights)
    if x2d.shape[0] == 0:
        return x.new_zeros(*shape, N)
    if any(b is not None for b in biases):
        bias = torch.cat([b if b is not None else w.new_zeros(w.shape[0]) for w, b in zip(weights, biases)])
    else:
        bias = None
    if _fp_only(cfg):
        return torch.nn.functional.linear(x, torch.cat(list(weights)).to(x.dtype), bias)
    t0 = time.perf_counter()
    qw = qweight(list(weights))
    acc, sx, device = _gemm(cfg, x2d, qw)
    if device == "cpu" and cfg.cpu_kernel == "fp":
        y = torch.nn.functional.linear(x2d, torch.cat(list(weights)).to(x.dtype), bias)
    else:
        y = acc.float() * sx[:, None] * qw.sw[None, :]
        if bias is not None:
            y += bias.float()
        y = y.to(x.dtype)
    _account(device, t0, x2d.shape[0], qw)
    return y.reshape(*shape, N)
