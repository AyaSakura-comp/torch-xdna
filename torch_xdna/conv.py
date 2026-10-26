"""xdna::conv3x3: 3x3, stride-1, padding-1 convolutions of a parameter weight, int8 on the NPU.

The backend replaces such a conv2d with `conv3x3_int8` below. Its quantize / dequantize steps are
ordinary torch ops, so Inductor fuses them with the ops around the conv; only the integer part is
the custom op, which runs Triton-XDNA's ``tl.extra.npu.conv3x3`` (the whole array; the input
channels split over two cores whose partial sums meet over the AIE cascade). Math:

  weights     int8 per output channel, quantized once
  activations int8 per tensor, per image (sx)
  sums        int32 on the NPU, requantized there by one shift (round half up, saturate to int8, or
              to [0, 255] with a fused ReLU). The shift is fixed on the layer's first call from that
              call's integer sums, with 25% headroom, and travels with the weights (no rebuild).
  output      q * sw[co] * 2^shift * sx (+ bias)

The CPU computes the same integers (an exact fp conv of the int8 values), so a layer's result does
not depend on where it ran. A layer's first NPU launch is compared with them bit for bit; a layer
that disagrees, fails to build or launch, or (policy "auto") is slower stays on the CPU.
"""

import logging
import math
import threading
import time
import weakref

import torch

from . import kernels
from .runtime import NPUResultMismatch, _time, decisions, stats

log = logging.getLogger("torch_xdna")
_lock = threading.RLock()
_layers = {}            # (id(weight), H, W, relu) -> (weakref to weight, ConvLayer)


def supported(ci, co, h, w):
    """Shapes tl.extra.npu.conv3x3 takes: Ci 16..256 in 16s, Co in 32s, W in 64-pixel bands."""
    return ci % 16 == 0 and 16 <= ci <= 256 and co % 32 == 0 and w % 64 == 0 and h >= 1


def conv3x3_int8(x, weight, bias, relu: bool, cfg_id: int):
    """relu?(conv2d(x, weight, bias, padding=1)) through int8: what the backend puts in the graph."""
    sx = x.abs().amax(dim=(1, 2, 3), keepdim=True).float().clamp(min=1e-8) / 127
    xq = torch.round(x.float() / sx).clamp(-127, 127).to(torch.int8)
    q, scale = torch.ops.xdna.conv3x3(xq, weight, relu, cfg_id)
    y = q.float() * (scale.view(1, -1, 1, 1) * sx)
    if bias is not None:
        y = y + bias.float().view(1, -1, 1, 1)
    return y.to(x.dtype)


class ConvLayer:
    """One conv weight at one input size: int8 weights, the requantization shift, the NPU op."""

    def __init__(self, weight, h, w, relu):
        wf = weight.detach().float()
        self.sw = wf.abs().amax(dim=(1, 2, 3)).clamp(min=1e-8) / 127          # [Co]
        self.wq = torch.round(wf / self.sw[:, None, None, None]).clamp(-127, 127).to(torch.int8)
        # |sums| <= 127 * sum |wq| per channel: below 2^24 a float32 conv of the integers is exact
        self.cpu_dtype = (torch.float32 if 127 * float(self.wq.abs().sum(dim=(1, 2, 3)).max()) < 2 ** 24
                          else torch.float64)
        self.h, self.w, self.relu = h, w, relu
        self.shift = self.scale = self.op = None
        self.device = None      # "npu" or "cpu" once the first call has decided
        co, ci = weight.shape[:2]
        self.name = f"conv3x3 {ci}->{co} {h}x{w}" + (" relu" if relu else "")

    def calibrate(self, xq):
        """Fix the shift from this image's integer sums (25% headroom over the largest)."""
        acc = self._acc(xq[None])
        top = float(acc.clamp(min=0).max()) if self.relu else float(acc.abs().max())
        self.shift = min(31, max(1, math.ceil(math.log2(max(top * 1.25, 1.0) / (255 if self.relu else 127)))))
        self.scale = self.sw * 2.0 ** self.shift

    def build(self):
        from triton.language.extra.npu import conv3x3

        self.op = conv3x3.Conv3x3(self.wq, height=self.h, width=self.w, shift=self.shift, relu=self.relu)

    def _acc(self, xq):
        return torch.nn.functional.conv2d(xq.to(self.cpu_dtype), self.wq.to(self.cpu_dtype), padding=1)

    def cpu(self, xq):
        """The NPU's integers on the CPU: exact sums, round half up, saturate."""
        q = (self._acc(xq).to(torch.int32) + (1 << (self.shift - 1))) >> self.shift
        return q.clamp_(0, 255).to(torch.uint8) if self.relu else q.clamp_(-128, 127).to(torch.int8)

    def npu(self, xq, verify=False):
        out = torch.empty((xq.shape[0], self.wq.shape[0], self.h, self.w),
                          dtype=torch.uint8 if self.relu else torch.int8)
        for i in range(xq.shape[0]):
            self.op(xq[i], out=out[i])
            if verify:
                want = self.cpu(xq[i:i + 1])[0]
                if not torch.equal(out[i], want):
                    raise NPUResultMismatch(f"NPU {self.name} disagrees with the CPU: "
                                            f"{int((out[i] != want).sum())} of {want.numel()} values differ")
        stats.launches += xq.shape[0] * len(self.op.slabs)
        return out


def _layer(weight, h, w, relu):
    key = (id(weight), h, w, relu)
    hit = _layers.get(key)
    if hit is not None and hit[0]() is weight:
        return hit[1]
    layer = ConvLayer(weight, h, w, relu)
    _layers[key] = (weakref.ref(weight), layer)
    return layer


def _failed(layer, e):
    layer.device = "cpu"
    stats.fallbacks.append((layer.name, f"{type(e).__name__}: {e}"[:300]))
    log.warning("torch_xdna: NPU %s failed (%s); it runs on the CPU from now on", layer.name, e)


def _first_call(cfg, layer, xq):
    """Build the layer, verify its first launch, and pick its device."""
    if not kernels.available():
        layer.device = "cpu"
        return
    try:
        layer.build()
        layer.npu(xq[:1], verify=cfg.verify != "off")
        layer.device = "npu"
        if cfg.policy == "auto":
            npu_ms, cpu_ms = _time(lambda: layer.npu(xq[:1])), _time(lambda: layer.cpu(xq[:1]))
            layer.device = "npu" if npu_ms < cpu_ms else "cpu"
            decisions[(layer.name,)] = (layer.device, round(npu_ms, 3), round(cpu_ms, 3))
            if cfg.verbose:
                log.warning("torch_xdna autotune %s: npu %.3f ms, cpu %.3f ms -> %s", layer.name, npu_ms,
                            cpu_ms, layer.device)
    except Exception as e:      # build / context / launch failure, or a mismatch
        _failed(layer, e)


def conv3x3(cfg, xq, weight, relu):
    """(q, scale) of xdna::conv3x3 for int8 xq [N, Ci, H, W], on the NPU or the CPU."""
    n, ci, h, w = xq.shape
    co = weight.shape[0]
    t0 = time.perf_counter()
    with _lock:
        layer = _layer(weight, h, w, relu)
        if layer.shift is None:
            layer.calibrate(xq[0])
        if layer.device is None:
            _first_call(cfg, layer, xq)
        q, device = None, "cpu"
        if layer.device == "npu":
            try:
                q, device = layer.npu(xq, verify=cfg.verify == "always"), "npu"
            except Exception as e:
                _failed(layer, e)
        if q is None:
            q = layer.cpu(xq)
    stats.calls[device] += 1
    stats.seconds[device] += time.perf_counter() - t0
    stats.macs[device] += n * co * ci * h * w * 9
    # a fresh tensor: Inductor may reuse an extern op's output buffer once it is dead
    return q, layer.scale.clone()
