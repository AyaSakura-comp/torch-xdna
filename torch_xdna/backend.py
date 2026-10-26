"""torch.compile(model, backend="xdna"): GEMMs and 3x3 convs on the XDNA NPU, everything else on the CPU.

The pass works on Dynamo's FX graph. A GEMM against a parameter --
`F.linear(x, W, b)`, or `x @ W` with a 2-D parameter W -- becomes xdna::linear;
a 3x3 / stride-1 / padding-1 conv2d with a parameter weight becomes quantize ->
xdna::conv3x3 -> dequantize (conv.py); the rest of the graph is left to Inductor
(or run eagerly). What runs where is decided per call at run time (see
runtime.py, conv.py), so a missing or failing NPU only moves GEMMs and convs
back to the CPU.

GEMMs between two activations (attention scores) are not offloaded: their
operands change every call, and at these sizes a launch (~0.3 ms) costs more
than the CPU's whole GEMM.
"""

import logging
import operator

import torch
import torch.nn.functional as F

from . import config as _config
from . import kernels, ops

log = logging.getLogger("torch_xdna")

LINEAR = {torch._C._nn.linear, F.linear}
CONV2D = {torch.conv2d, F.conv2d}
RELU = {torch.relu, F.relu, torch.relu_}
MATMUL = {operator.matmul, torch.matmul}
FLOATS = (torch.float32, torch.bfloat16, torch.float16)


def _static_placeholders(gm, example_inputs):
    """Placeholder nodes that hold parameters (or buffers): the GEMM weights worth quantizing once."""
    out = set()
    nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, ex in zip(nodes, example_inputs):
        if isinstance(ex, torch.nn.Parameter) or "_parameters_" in node.name or "_buffers_" in node.name:
            out.add(node)
    return out


def _weight_ok(node, attrs):
    if node is None:
        return False
    if node.op == "get_attr":
        t = attrs(node.target)
    elif node.op == "placeholder":
        t = node.meta.get("example_value")
    else:
        return False
    return isinstance(t, torch.Tensor) and t.dim() == 2 and t.dtype in FLOATS


def offload_gemms(gm: torch.fx.GraphModule, example_inputs, cfg) -> int:
    """Rewrite offloadable GEMMs to xdna::linear and 3x3 convs to xdna::conv3x3 in place; return how many."""
    static = _static_placeholders(gm, example_inputs)

    def attr(target):
        obj = gm
        for part in target.split("."):
            obj = getattr(obj, part)
        return obj

    def is_weight(n):
        return isinstance(n, torch.fx.Node) and (n in static or n.op == "get_attr") and _weight_ok(n, attr)

    def is_weight_vector(n):
        """A 1-D parameter: a bias that broadcasts over the rows like F.linear's."""
        t = n.meta.get("example_value") if isinstance(n, torch.fx.Node) else None
        return (isinstance(n, torch.fx.Node) and (n in static or n.op == "get_attr")
                and isinstance(t, torch.Tensor) and t.dim() == 1)

    cid = ops.config_id(cfg)
    count = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target in LINEAR:
            x = node.args[0] if node.args else node.kwargs.get("input")
            w = node.args[1] if len(node.args) > 1 else node.kwargs.get("weight")
            b = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
            transposed = False
        elif ((node.op == "call_function" and node.target in MATMUL | {torch.mm})
              or (node.op == "call_method" and node.target in ("matmul", "mm", "__matmul__"))) \
                and len(node.args) == 2 and not node.kwargs:
            x, w = node.args          # x @ W, W [K, N]
            b, transposed = None, True
        elif node.op == "call_function" and node.target is torch.addmm and len(node.args) == 3 \
                and all(node.kwargs.get(k, 1) == 1 for k in ("beta", "alpha")) \
                and set(node.kwargs) <= {"beta", "alpha"}:
            b, x, w = node.args       # bias + x @ W (HF Conv1D: GPT-2 and friends)
            transposed = True
            if not (b is None or is_weight_vector(b)):
                continue
        else:
            continue
        if not is_weight(w):
            continue
        with gm.graph.inserting_after(node):
            new = gm.graph.call_function(torch.ops.xdna.linear.default, (x, w, b, transposed, cid))
        new.meta.update(node.meta)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        count += 1
    if cfg.fuse:
        _fuse_shared_input(gm, cid)
    count += _offload_convs(gm, static, cid, attr, cfg)
    gm.graph.lint()
    gm.recompile()
    return count


def _conv_args(node):
    """(x, weight, bias, stride, padding, dilation, groups) of a conv2d call."""
    names = ("input", "weight", "bias", "stride", "padding", "dilation", "groups")
    defaults = dict(bias=None, stride=1, padding=0, dilation=1, groups=1)
    args = dict(zip(names, node.args), **node.kwargs)
    return [args.get(k, defaults.get(k)) for k in names]


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _offload_convs(gm, static, cid, attr, cfg):
    """3x3 / stride 1 / padding 1 / groups 1 convolutions of a parameter weight -> conv3x3_int8
    (quantize, xdna::conv3x3, dequantize; Inductor fuses the first and last with their neighbours).

    A ReLU that is the conv's only user (and the conv has no bias, so the ReLU applies to the
    requantized sums themselves) is fused: the NPU saturates to [0, 255] instead.
    """
    from .conv import conv3x3_int8, supported

    # int8 convs only pay off on the NPU (the CPU has no int8 conv kernel): without one, or with
    # policy "cpu", the convs stay as they are
    if cfg.policy == "cpu" or not kernels.available():
        return 0
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in CONV2D:
            continue
        x, w, b, stride, padding, dilation, groups = _conv_args(node)
        if (_pair(stride) != (1, 1) or (padding != "same" and _pair(padding) != (1, 1))
                or _pair(dilation) != (1, 1) or groups != 1):
            continue
        if not (isinstance(w, torch.fx.Node) and (w in static or w.op == "get_attr")):
            continue
        wt = w.meta.get("example_value") if w.op == "placeholder" else attr(w.target)
        xt = x.meta.get("example_value") if isinstance(x, torch.fx.Node) else None
        if not (isinstance(wt, torch.Tensor) and wt.dim() == 4 and tuple(wt.shape[2:]) == (3, 3)
                and isinstance(xt, torch.Tensor) and xt.dim() == 4 and xt.dtype in FLOATS):
            continue
        if not supported(int(wt.shape[1]), int(wt.shape[0]), int(xt.shape[2]), int(xt.shape[3])):
            continue
        users = list(node.users)
        relu = None
        if cfg.fuse and b is None and len(users) == 1 and (
                (users[0].op == "call_function" and users[0].target in RELU)
                or (users[0].op == "call_method" and users[0].target in ("relu", "relu_"))):
            relu = users[0]
        with gm.graph.inserting_after(relu or node):
            new = gm.graph.call_function(conv3x3_int8, (x, w, b, relu is not None, cid))
        new.meta.update((relu or node).meta)
        (relu or node).replace_all_uses_with(new)
        if relu is not None:
            gm.graph.erase_node(relu)
        gm.graph.erase_node(node)
        count += 1
    return count


def _fuse_shared_input(gm, cid):
    """xdna::linear nodes reading the same input (q/k/v, gate/up) -> one xdna::linear_cat + narrows.

    One launch instead of several, and one NPU program for the fused shape.
    """
    groups = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.xdna.linear.default and not node.args[3]:
            groups.setdefault(node.args[0], []).append(node)
    for x, members in groups.items():
        if len(members) < 2:
            continue
        weights = [m.args[1] for m in members]
        sizes = [_out_features(w) for w in weights]
        if None in sizes:
            continue
        with gm.graph.inserting_before(members[0]):
            cat = gm.graph.call_function(torch.ops.xdna.linear_cat.default,
                                         (x, weights, [m.args[2] for m in members], cid))
            start = 0
            for m, n in zip(members, sizes):
                part = gm.graph.call_function(torch.narrow, (cat, -1, start, n))
                part.meta.update(m.meta)
                m.replace_all_uses_with(part)
                start += n
        for m in members:
            gm.graph.erase_node(m)


def _out_features(w):
    t = w.meta.get("example_value") if isinstance(w, torch.fx.Node) else None
    return int(t.shape[0]) if isinstance(t, torch.Tensor) else None


def xdna_backend(gm: torch.fx.GraphModule, example_inputs, options=None, mode=None):
    """The torch.compile backend; `options` are torch_xdna.Config fields."""
    cfg = _config.config.merged(options)
    n = offload_gemms(gm, example_inputs, cfg)
    if cfg.verbose:
        log.warning("torch_xdna: %d GEMMs / convs offloadable in a graph of %d nodes (policy=%s, fallback=%s)",
                    n, len(gm.graph.nodes), cfg.policy, cfg.fallback)
    if cfg.fallback == "eager":
        return gm.forward
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(gm, example_inputs)
