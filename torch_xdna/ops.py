"""xdna::linear, the custom op the backend puts in place of offloadable GEMMs.

To the rest of torch.compile it is an opaque op with a shape function, so
Inductor compiles everything around it and calls it as an extern kernel.
"""

from typing import Optional

import torch

from . import runtime

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
