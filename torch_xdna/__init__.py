"""torch_xdna: a torch.compile backend that runs a model's GEMMs on the AMD XDNA NPU.

    import torch, torch_xdna
    model = torch.compile(model, backend="xdna", options={"policy": "auto"})

GEMMs against parameters run W8A8 on the NPU through Triton-XDNA, and so do 3x3
stride-1 convolutions (tl.extra.npu.conv3x3); everything else is compiled by
Inductor for the CPU. Per shape the "auto" policy measures both devices once
and keeps the faster; without an NPU, or when a launch fails, GEMMs and convs
fall back to the CPU with the same int8 math.
See torch_xdna.Config for every option and torch_xdna.report() for where the
work went.
"""

import json

from . import kernels, ops  # noqa: F401  (registers xdna::linear)
from .backend import offload_gemms, xdna_backend
from .config import Config, set_config
from .runtime import stats

__all__ = ["Config", "set_config", "xdna_backend", "offload_gemms", "compile", "stats", "report",
           "npu_available"]
__version__ = "0.1.0"


def _register():
    from torch._dynamo.backends.registry import _BACKENDS, register_backend

    if "xdna" not in _BACKENDS:
        register_backend(xdna_backend, name="xdna")


_register()


def compile(model, **options):
    """torch.compile(model, backend="xdna", options=options)."""
    import torch

    return torch.compile(model, backend="xdna", options=options or None)


def npu_available() -> bool:
    return kernels.available()


def report(as_json=False):
    """Where the GEMMs and convs ran (calls, seconds, GMACs per device), NPU launches, fallbacks, autotune."""
    d = stats.as_dict()
    if as_json:
        return json.dumps(d, indent=1)
    lines = [f"NPU available: {d['npu_available']}" +
             (f" ({d['npu_unavailable_reason']})" if not d["npu_available"] else "")]
    for dev in sorted(d["calls"]):
        lines.append(f"  {dev:4s}: {d['calls'][dev]} calls, {d['gmacs'][dev]} GMAC, {d['seconds'][dev]} s")
    lines.append(f"  NPU launches: {d['npu_launches']}")
    for k, v in d["decisions"].items():
        lines.append(f"  auto {k}: {v[0]} (npu {v[1]} ms, cpu {v[2]} ms)")
    for shape, err in d["fallbacks"]:
        lines.append(f"  fallback {shape}: {err}")
    return "\n".join(lines)
