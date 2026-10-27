"""xdna::qwen3_stack: a whole Qwen3-0.6B-sized transformer stack and its linear head as one NPU launch.

The backend looks in Dynamo's graph for a transformers `Qwen3Model` whose last hidden states
feed only a linear layer (OmniVoice's backbone and audio heads, for one), with Qwen3-0.6B's
shapes, any depth, and attention under a 4-D mask that is an input of the graph. Such a graph
is compiled twice:

  fused  the stack and the head replaced by xdna::qwen3_stack, which runs Triton-XDNA's
         tl.extra.npu.qwen3_stack: every layer, the final norm and the head in one launch per
         pair of sequences (bfp16 operands, fp32 accumulation)
  plain  the graph as the backend compiles any other (GEMMs per policy, the rest Inductor)

and the backend returns a dispatcher that picks one per call by looking at the mask: the
fused graph when the NPU is there and every sequence's mask is a prefix block of 1 to 384
positions (each of its first n positions sees exactly those n: the bidirectional layout of a
masked-diffusion model) followed only by padding positions that see themselves or nothing;
the plain graph otherwise (a causal mask, for one). The fused graph's rows for padding
positions are zeros.

The first fused call of each build (number of row groups) is checked against the original
graph run eagerly in fp32: the valid rows of the head's output must be within 10% (relative
norm; 1-3% measured on OmniVoice). A mismatch, or a build or launch failure, sends the stack
to the plain graph for the rest of the process.
"""

import copy
import logging
import operator
import re
import threading
import time
import weakref

import torch
import torch.nn.functional as F

from . import kernels
from .runtime import NPUResultMismatch, stats

log = logging.getLogger("torch_xdna")
_lock = threading.RLock()

LINEAR = {torch._C._nn.linear, F.linear}
SDPA = {torch._C._nn.scaled_dot_product_attention, F.scaled_dot_product_attention}
ADD = {operator.add, torch.add}
MUL = {operator.mul, torch.mul}
ROOTS = ("Qwen3Model",)
D, NH, NKV, DH, INTER, EPS = 1024, 16, 8, 128, 3072, 1e-6
LAYER_KEYS = ("q", "k", "v", "o", "gate", "up", "down", "input_norm", "post_norm", "q_norm", "k_norm")
SHAPES = dict(q=(NH * DH, D), k=(NKV * DH, D), v=(NKV * DH, D), o=(D, NH * DH), gate=(INTER, D), up=(INTER, D),
              down=(D, INTER), input_norm=(D,), post_norm=(D,), q_norm=(DH,), k_norm=(DH,))
VERIFY_TOL = 0.1

_PROJ = re.compile(r"layers\.(\d+)\.(?:self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
_NORMS = {"input_layernorm": "input_norm", "post_attention_layernorm": "post_norm",
          "self_attn.q_norm": "q_norm", "self_attn.k_norm": "k_norm"}
_NORM = re.compile(r"layers\.(\d+)\.(" + "|".join(re.escape(k) for k in _NORMS) + ")")

_op_mod = {}


def _op_module():
    """Triton-XDNA's tl.extra.npu.qwen3_stack, or None if this install has none."""
    if "mod" not in _op_mod:
        try:
            from triton.language.extra.npu import qwen3_stack

            _op_mod["mod"] = qwen3_stack
        except Exception as e:  # an older Triton-XDNA, or none
            log.info("torch_xdna: no tl.extra.npu.qwen3_stack (%s)", e)
            _op_mod["mod"] = None
    return _op_mod["mod"]


# ------------------------------------------------------------------ finding a stack
class _NoStack(Exception):
    pass


class Plan:
    """A stack found in a graph. Nodes are kept by name, so the plan applies to copies."""

    def __init__(self, region, x, mask, mask_index, head, weights, bias, layers, out_features, max_tokens):
        self.region, self.x, self.mask, self.mask_index, self.head = region, x, mask, mask_index, head
        self.weights, self.bias = weights, bias       # role -> tensor; the head's bias or None
        self.layers, self.out_features, self.max_tokens = layers, out_features, max_tokens

    def tensors(self):
        """Every weight, in the order the op takes them: per layer LAYER_KEYS, then norm, head, inv_freq."""
        return [self.weights[(i, k)] for i in range(self.layers) for k in LAYER_KEYS] + \
            [self.weights["norm"], self.weights["head"], self.weights["inv_freq"]]


def _cls_name(c):
    return c.__name__ if isinstance(c, type) else str(c).rsplit(".", 1)[-1].strip("'>")


def _inner(n):
    """(path, class name) of the innermost module a node ran in."""
    path, cls = list(n.meta["nn_module_stack"].values())[-1]
    return path, _cls_name(cls)


def _linear_args(n):
    x = n.args[0] if n.args else n.kwargs.get("input")
    w = n.args[1] if len(n.args) > 1 else n.kwargs.get("weight")
    b = n.args[2] if len(n.args) > 2 else n.kwargs.get("bias")
    return x, w, b


def find(gm, example_inputs):
    """The first stack + head in ``gm`` that tl.extra.npu.qwen3_stack takes, as a Plan, or None."""
    mod = _op_module()
    if mod is None:
        return None
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    value = dict(zip(placeholders, example_inputs))
    roots = {}
    for n in gm.graph.nodes:
        if n.op not in ("call_function", "call_method", "call_module"):
            continue        # inputs carry the module of their first use
        for path, cls in (n.meta.get("nn_module_stack") or {}).values():
            if _cls_name(cls) in ROOTS:
                roots.setdefault(path, []).append(n)
                break
    for root, region in roots.items():
        try:
            return _plan(root, region, value, placeholders, mod.MAX_GROUPS * mod.GROUP_ROWS)
        except _NoStack as e:
            log.info("torch_xdna: %s is not a stack the NPU takes: %s", root, e)
    return None


def _plan(root, region, value, placeholders, max_tokens):
    inside = set(region)

    def static(n):
        return n.op == "placeholder" and (isinstance(value.get(n), torch.nn.Parameter)
                                          or "_parameters_" in n.name or "_buffers_" in n.name)

    weights, sdpa = {}, []

    def put(key, node):
        t = value.get(node)
        if not isinstance(t, torch.Tensor):
            raise _NoStack(f"{key}: no tensor")
        if weights.setdefault(key, t) is not t:
            raise _NoStack(f"two tensors for {key}")

    for n in region:
        path, cls = _inner(n)
        rel = "" if path == root else path[len(root) + 1:]
        others = n.all_input_nodes
        if n.op == "call_function" and n.target in LINEAR:
            _, w, b = _linear_args(n)
            m = _PROJ.fullmatch(rel)
            if m is None or b is not None or not static(w):
                raise _NoStack(f"a linear in {rel or 'the root'}")
            put((int(m[1]), m[2] or m[3]), w)
            others = [a for a in others if a is not w]
        elif n.op == "call_function" and n.target in SDPA:
            sdpa.append(n)
        for a in others:
            if not static(a):
                continue
            m = _NORM.fullmatch(rel)
            if m:
                put((int(m[1]), _NORMS[m[2]]), a)
            elif rel in ("norm", "rotary_emb"):
                put("norm" if rel == "norm" else "inv_freq", a)
            else:
                raise _NoStack(f"parameter {a.name} used in {rel or 'the root'}")
        # the constants the NPU program has built in: RMSNorm's eps, RoPE without scaling
        consts = [a for a in n.args if isinstance(a, float)]
        if n.op == "call_function" and consts:
            if n.target in ADD and cls.endswith("RMSNorm") and any(abs(c - EPS) > 1e-12 for c in consts):
                raise _NoStack(f"RMSNorm eps {consts} (the NPU program has {EPS})")
            if n.target in MUL and rel == "rotary_emb" and any(c != 1.0 for c in consts):
                raise _NoStack(f"RoPE attention scaling {consts}")

    layers = 1 + max((k[0] for k in weights if isinstance(k, tuple)), default=-1)
    if layers < 1:
        raise _NoStack("no layers")
    for i in range(layers):
        for k in LAYER_KEYS:
            t = weights.get((i, k))
            if t is None or tuple(t.shape) != SHAPES[k]:
                raise _NoStack(f"layer {i} {k}: {None if t is None else tuple(t.shape)}, need {SHAPES[k]}")
    if tuple(getattr(weights.get("norm"), "shape", ())) != (D,) or \
            tuple(getattr(weights.get("inv_freq"), "shape", ())) != (DH // 2,):
        raise _NoStack("final norm / RoPE frequencies")

    # attention: bidirectional under one mask, a graph input
    masks = set()
    for n in sdpa:
        kw = dict(zip(("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale"), n.args))
        kw.update(n.kwargs)
        scale = kw.get("scale")
        if kw.get("dropout_p", 0.0) != 0.0 or kw.get("is_causal", False) or \
                (scale is not None and abs(scale - DH ** -0.5) > 1e-6):
            raise _NoStack("attention: dropout, causal or another scale")
        masks.add(kw.get("attn_mask"))
    if len(sdpa) != layers or len(masks) != 1:
        raise _NoStack(f"{len(sdpa)} attentions for {layers} layers, {len(masks)} masks")
    mask = masks.pop()
    mv = value.get(mask)
    if not isinstance(mask, torch.fx.Node) or mask.op != "placeholder" or not isinstance(mv, torch.Tensor) \
            or mv.dim() != 4 or not (mv.dtype == torch.bool or mv.is_floating_point()):
        raise _NoStack("the attention mask is not a 4-D input of the graph")

    # inputs: the hidden states, the mask, sizes
    ext = {}
    for n in region:
        for a in n.all_input_nodes:
            if a not in inside and a is not mask and not static(a):
                ext[a] = None
    rest = [a for a in ext if not (a.op == "placeholder" and isinstance(value.get(a), (int, torch.SymInt)))]
    if len(rest) != 1:
        raise _NoStack(f"inputs {[a.name for a in rest]}")
    x = rest[0]
    xv = x.meta.get("example_value")
    if not (isinstance(xv, torch.Tensor) and xv.dim() == 3 and xv.shape[-1] == D and xv.is_floating_point()):
        raise _NoStack("hidden states are not [B, L, 1024]")

    # the one output: the final norm, into a linear head and nothing else
    outs = [n for n in region if any(u not in inside for u in n.users)]
    if len(outs) != 1 or len(outs[0].users) != 1:
        raise _NoStack("the stack's output has other users than one head")
    head = next(iter(outs[0].users))
    if not (head.op == "call_function" and head.target in LINEAR):
        raise _NoStack(f"the stack's output goes to {head.target}")
    hx, hw, hb = _linear_args(head)
    W = value.get(hw) if static(hw) else None
    B = value.get(hb) if hb is not None and static(hb) else None
    if hx is not outs[0] or not isinstance(W, torch.Tensor) or W.dim() != 2 or W.shape[1] != D or \
            (hb is not None and not (isinstance(B, torch.Tensor) and tuple(B.shape) == (W.shape[0],))):
        raise _NoStack("the head is not a linear of a parameter [N, 1024]")
    weights["head"] = W
    return Plan([n.name for n in region], x.name, mask.name, placeholders.index(mask), head.name, weights, B,
                layers, int(W.shape[0]), max_tokens)


# ------------------------------------------------------------------ the fused graph
def copy_module(gm):
    """A new GraphModule with a copy of ``gm``'s graph (same attributes, same node names)."""
    g = torch.fx.Graph()
    g._codegen = copy.deepcopy(gm.graph._codegen)
    g.output(g.graph_copy(gm.graph, {}))
    new = torch.fx.GraphModule(gm, g)
    new.meta.update(gm.meta)
    return new


def fuse(gm, plan, stack_id, cfg_id):
    """Replace the stack and its head in ``gm`` by xdna::qwen3_stack."""
    by = {n.name: n for n in gm.graph.nodes}
    head = by[plan.head]
    with gm.graph.inserting_before(head):
        new = gm.graph.call_function(torch.ops.xdna.qwen3_stack.default,
                                     (by[plan.x], by[plan.mask], stack_id, cfg_id))
    new.meta.update(head.meta)
    head.replace_all_uses_with(new)
    gm.graph.erase_node(head)
    for name in reversed(plan.region):
        gm.graph.erase_node(by[name])
    gm.graph.lint()
    gm.recompile()


# ------------------------------------------------------------------ run time
class StackUnsupported(RuntimeError):
    pass


class _Stack:
    """One stack found by the backend: its weights (weakly), the NPU op, the checks done."""

    def __init__(self, plan):
        self.refs = [weakref.ref(t) for t in plan.tensors()]
        self.bias = weakref.ref(plan.bias) if plan.bias is not None else None
        self.layers, self.N, self.max_tokens = plan.layers, plan.out_features, plan.max_tokens
        self.name = f"qwen3 stack {self.layers} layers -> {self.N}"
        self.verified = set()   # row-group counts whose first launch matched the model
        self.failed = None      # why the stack runs in the plain graph from now on
        self.capture = False    # keep a copy of the next output (for the check)
        self.captured = None

    def op(self):
        """The Triton-XDNA op over the current weights (packed again when they changed)."""
        ts = [r() for r in self.refs]
        if any(t is None for t in ts):
            raise RuntimeError("the stack's weights are gone")
        key, versions = tuple(id(t) for t in ts), tuple(t._version for t in ts)
        hit = _ops.get(key)
        if hit is not None and hit[1] == versions and all(r() is t for r, t in zip(hit[0], ts)):
            return hit[2]
        n = len(LAYER_KEYS)
        layers = [dict(zip(LAYER_KEYS, ts[i * n:(i + 1) * n])) for i in range(self.layers)]
        norm, head, inv_freq = ts[self.layers * n:]
        op = _op_module().Qwen3Stack(layers, norm=norm, head=head, inv_freq=inv_freq)
        _ops[key] = ([weakref.ref(t) for t in ts], versions, op)
        return op

    def macs(self, lens):
        per_token = self.layers * (D * (NH + 2 * NKV) * DH + NH * DH * D + 3 * D * INTER) + D * self.N
        return per_token * sum(lens) + self.layers * sum(2 * NH * DH * n * n for n in lens)

    def fail(self, e):
        self.failed = f"{type(e).__name__}: {e}"[:300]
        stats.fallbacks.append((self.name, self.failed))
        log.warning("torch_xdna: NPU %s failed (%s); it runs on the CPU from now on", self.name, self.failed)


_stacks = []    # stack id -> _Stack
_ops = {}       # ids of the weight tensors -> (weakrefs, versions, tl.extra.npu.qwen3_stack op)


def register(plan):
    with _lock:
        _stacks.append(_Stack(plan))
        return len(_stacks) - 1


def out_features(stack_id):
    return _stacks[stack_id].N


def lengths(mask):
    """Each sequence's length n when ``mask`` [B, H, L, L] (bool, or float with 0 / large negative)
    lets its first n positions see exactly those n and every later position is padding that sees
    only itself (OmniVoice's) or nothing, else None."""
    if not isinstance(mask, torch.Tensor) or mask.dim() != 4 or mask.shape[-1] != mask.shape[-2]:
        return None
    if mask.dtype == torch.bool:
        allowed = mask
    elif mask.is_floating_point():
        allowed = mask == 0
        if not bool((allowed | (mask <= -1e4)).all()):
            return None
    else:
        return None
    if allowed.shape[1] > 1 and not bool((allowed == allowed[:, :1]).all()):
        return None
    a = allowed[:, 0]
    n = a[:, 0].sum(-1)
    ar = torch.arange(a.shape[-1], device=a.device)
    want = ar[None, None, :] < n[:, None, None]
    rows = ar[None, :, None] < n[:, None, None]
    pad = (a == (ar[:, None] == ar[None, :])) | ~a.any(-1, keepdim=True)
    if not bool(torch.where(rows, a == want, pad).all()):
        return None
    return n.tolist()


def _groups(lens, rows):
    """The row-group counts of the launches (pairs of sequences) of a call."""
    return {-(-max(lens[b:b + 2]) // rows) for b in range(0, len(lens), 2)}


def run(cfg, stack_id, x, mask):
    """xdna::qwen3_stack: hidden states x [B, L, 1024] -> the head's output [B, L, N], on the NPU."""
    st = _stacks[stack_id]
    B, L = x.shape[:2]
    lens = lengths(mask)
    if lens is not None and len(lens) == 1 and B > 1:
        lens = lens * B
    if lens is None or len(lens) != B or mask.shape[-1] != L or not all(1 <= n <= st.max_tokens for n in lens):
        raise StackUnsupported(f"{st.name}: the NPU takes prefix-block masks of 1..{st.max_tokens} positions")
    t0 = time.perf_counter()
    with _lock:
        op = st.op()
        xs = x.detach().float()
        out = torch.zeros((B, L, st.N))
        for b in range(0, B, 2):
            # a pair of sequences per launch; an odd last one gets a 1-position partner
            y0, y1 = op(xs[b, :lens[b]], xs[b + 1, :lens[b + 1]] if b + 1 < B else xs[b, :1])
            out[b, :lens[b]] = y0
            if b + 1 < B:
                out[b + 1, :lens[b + 1]] = y1
            stats.launches += 1
        if st.bias is not None:
            for b, n in enumerate(lens):
                out[b, :n] += st.bias().float()
        if st.capture:
            st.captured = out.clone()
    stats.calls["npu"] += 1
    stats.seconds["npu"] += time.perf_counter() - t0
    stats.macs["npu"] += st.macs(lens)
    return out.to(x.dtype)


class _Stop(Exception):
    pass


class _Until(torch.fx.Interpreter):
    """Runs a graph eagerly up to one node and keeps its value."""

    def __init__(self, gm, name):
        super().__init__(gm, garbage_collect_values=True)
        self.extra_traceback = False
        self.name, self.value = name, None

    def run_node(self, n):
        v = super().run_node(n)
        if n.name == self.name:
            self.value = v
            raise _Stop()
        return v


class Dispatch:
    """What the backend returns for a graph with a stack: the fused graph when the NPU takes the
    call, else the plain one; checks each build's first fused call against the original graph."""

    def __init__(self, plan, stack_id, cfg, fused, plain, orig):
        self.plan, self.stack_id, self.cfg = plan, stack_id, cfg
        self.fused, self.plain, self.orig = fused, plain, orig

    def __call__(self, *args):
        st = _stacks[self.stack_id]
        lens = None
        if st.failed is None and kernels.available():
            lens = lengths(args[self.plan.mask_index])
            if lens is not None and not all(1 <= n <= st.max_tokens for n in lens):
                lens = None
        if lens is not None:
            groups = _groups(lens, _op_module().GROUP_ROWS)
            check = self.cfg.verify == "always" or (self.cfg.verify == "first" and not groups <= st.verified)
            try:
                st.capture = check
                out = self.fused(*args)
                if check:
                    self._check(st, args, lens, groups)
                stats.stacks[(st.name, "npu")] += 1
                return out
            except Exception as e:      # build / launch failure, or a mismatch
                st.fail(e)
            finally:
                st.capture, st.captured = False, None
        stats.stacks[(st.name, "cpu")] += 1
        return self.plain(*args)

    def _check(self, st, args, lens, groups):
        """The fused call's stack output vs the original graph's, run eagerly in fp32."""
        it = _Until(self.orig, self.plan.head)
        with torch.no_grad():
            try:
                it.run(*args)
            except _Stop:
                pass
        if it.value is None:
            raise RuntimeError(f"{self.plan.head} not reached")
        got = torch.cat([st.captured[b, :n] for b, n in enumerate(lens)]).double()
        want = torch.cat([it.value[b, :n] for b, n in enumerate(lens)]).double()
        err = float((got - want).norm() / want.norm().clamp(min=1e-30))
        stats.checks.append((f"{st.name}, row groups {sorted(groups)}", round(err, 4)))
        if not bool(torch.isfinite(got).all()) or err > VERIFY_TOL:
            raise NPUResultMismatch(f"NPU {st.name} disagrees with the model: rel err {err:.3g}")
        st.verified |= groups


def compile_fused(gm, example_inputs, cfg, cfg_id, plan, compile_graph):
    """The Dispatch over the fused and the plain compilation of ``gm`` (which this rewrites)."""
    orig = copy_module(gm)
    fused = copy_module(gm)
    sid = register(plan)
    fuse(fused, plan, sid, cfg_id)
    if cfg.verbose:
        log.warning("torch_xdna: %s (%d nodes) -> one NPU launch per pair of sequences",
                    _stacks[sid].name, len(plan.region) + 1)
    return Dispatch(plan, sid, cfg, compile_graph(fused, example_inputs, cfg), compile_graph(gm, example_inputs, cfg),
                    orig)
