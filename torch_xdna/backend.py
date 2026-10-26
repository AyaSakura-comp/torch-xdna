"""torch.compile(model, backend="xdna"): GEMMs on the XDNA NPU, everything else on the CPU.

The pass works on Dynamo's FX graph. A GEMM against a parameter --
`F.linear(x, W, b)`, or `x @ W` with a 2-D parameter W -- becomes xdna::linear;
the rest of the graph is left to Inductor (or run eagerly). What runs where is
decided per call at run time (see runtime.py), so a missing or failing NPU only
moves GEMMs back to the CPU.

GEMMs between two activations (attention scores) are not offloaded: their
operands change every call, and at these sizes a launch (~0.3 ms) costs more
than the CPU's whole GEMM.
"""

import logging
import operator

import torch
import torch.nn.functional as F

from . import config as _config
from . import ops

log = logging.getLogger("torch_xdna")

LINEAR = {torch._C._nn.linear, F.linear}
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
    """Rewrite offloadable GEMMs to xdna::linear in place; return how many."""
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
    gm.graph.lint()
    gm.recompile()
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
        log.warning("torch_xdna: %d GEMMs offloadable in a graph of %d nodes (policy=%s, fallback=%s)",
                    n, len(gm.graph.nodes), cfg.policy, cfg.fallback)
    if cfg.fallback == "eager":
        return gm.forward
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(gm, example_inputs)
