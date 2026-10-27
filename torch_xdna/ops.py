"""xdna::linear / linear_cat / conv3x3, the custom ops the backend puts in place of offloadable GEMMs and convs.

To the rest of torch.compile it is an opaque op with a shape function, so
Inductor compiles everything around it and calls it as an extern kernel.
"""

from typing import Optional

import torch

from . import conv, runtime

_configs = []  # cfg_id -> Config; the op's arguments must be plain values


def config_id(cfg) -> int:
    for i, c in enumerate(_configs):
        if c == cfg:
            return i
    _configs.append(cfg)
    return len(_configs) - 1


@torch.library.custom_op("xdna::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           transposed: bool, cfg_id: int) -> torch.Tensor:
    return runtime.linear(_configs[cfg_id], x, weight, bias, transposed)


@linear.register_fake
def _(x, weight, bias, transposed, cfg_id):
    N = weight.shape[1] if transposed else weight.shape[0]
    return x.new_empty(*x.shape[:-1], N)


@torch.library.custom_op("xdna::linear_cat", mutates_args=())
def linear_cat(x: torch.Tensor, weights: list[torch.Tensor], biases: list[Optional[torch.Tensor]],
               cfg_id: int) -> torch.Tensor:
    """Linears that read the same input, as one GEMM: the outputs concatenated on the last dim."""
    return runtime.linear_cat(_configs[cfg_id], x, weights, biases)


@linear_cat.register_fake
def _(x, weights, biases, cfg_id):
    return x.new_empty(*x.shape[:-1], sum(w.shape[0] for w in weights))


@torch.library.custom_op("xdna::linear_i8", mutates_args=())
def linear_i8(xq: torch.Tensor, weights: list[torch.Tensor], transposed: bool,
              cfg_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The integer part of W8A8 linears that read the same input: int8 xq [R, K] ->
    (acc [R, N] int32, sw [N] float32), with x @ W^T ~= acc * sx[:, None] * sw for the row scales sx
    that gave xq. Several weights are one GEMM, their outputs concatenated."""
    return runtime.linear_i8(_configs[cfg_id], xq, weights, transposed)


@linear_i8.register_fake
def _(xq, weights, transposed, cfg_id):
    N = sum(w.shape[1] if transposed else w.shape[0] for w in weights)
    return xq.new_empty((xq.shape[0], N), dtype=torch.int32), xq.new_empty((N,), dtype=torch.float32)


@torch.library.custom_op("xdna::conv3x3", mutates_args=())
def conv3x3(xq: torch.Tensor, weight: torch.Tensor, relu: bool, cfg_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The integer part of a 3x3 / stride-1 / padding-1 conv of a [Co, Ci, 3, 3] parameter weight:
    int8 xq [N, Ci, H, W] -> (q [N, Co, H, W] int8, uint8 with relu; scale [Co] float32), with
    relu?(conv2d(x, weight)) ~= q * scale[co] * sx for the activation scale sx that gave xq."""
    return conv.conv3x3(_configs[cfg_id], xq, weight, relu)


@conv3x3.register_fake
def _(xq, weight, relu, cfg_id):
    q = xq.new_empty((xq.shape[0], weight.shape[0], xq.shape[2], xq.shape[3]),
                     dtype=torch.uint8 if relu else torch.int8)
    return q, weight.new_empty((weight.shape[0],), dtype=torch.float32)
