"""Options of the xdna backend: torch.compile(..., backend="xdna", options={...}) or torch_xdna.config."""

from dataclasses import dataclass, fields, replace

POLICIES = ("auto", "npu", "cpu")
FALLBACKS = ("inductor", "eager")
CPU_KERNELS = ("int8", "fp")


@dataclass(frozen=True)
class Config:
    # Where an offloadable GEMM runs:
    #   auto  measure NPU vs CPU once per (row bucket, N, K), keep the faster
    #   npu   always the NPU (CPU only when it is unavailable or fails)
    #   cpu   never the NPU
    policy: str = "auto"
    # What compiles the rest of the graph (and the GEMMs left on the CPU's side):
    #   inductor  torch.compile's CPU code generator (fuses the pointwise ops)
    #   eager     run the graph as is
    fallback: str = "inductor"
    # A GEMM the NPU does not take runs on the CPU as
    #   int8  the same W8A8 math (torch._int_mm), so results do not depend on routing
    #   fp    the original linear in the input dtype
    cpu_kernel: str = "int8"
    # Fewer rows than this go to the CPU without asking (a GEMV is bandwidth
    # bound, and the CPU has twice the NPU's DRAM bandwidth).
    min_rows: int = 16
    # Row buckets: a launch's rows are padded up to the smallest that holds
    # them; more rows than the largest are split into launches of it. Every
    # (bucket, N, K) is one NPU program and one hw_context.
    row_buckets: tuple = (128, 256, 512, 1024)
    # GEMMs smaller than this many MACs stay on the CPU (a launch costs ~0.3 ms).
    min_macs: int = 2_000_000
    # Linears that read the same input (q/k/v, gate/up) run as one GEMM.
    fuse: bool = True
    verbose: bool = False

    def __post_init__(self):
        assert self.policy in POLICIES, f"policy must be one of {POLICIES}"
        assert self.fallback in FALLBACKS, f"fallback must be one of {FALLBACKS}"
        assert self.cpu_kernel in CPU_KERNELS, f"cpu_kernel must be one of {CPU_KERNELS}"
        assert all(b % 128 == 0 for b in self.row_buckets), "row buckets must be multiples of 128"

    def merged(self, options=None):
        if not options:
            return self
        known = {f.name for f in fields(self)}
        bad = set(options) - known
        assert not bad, f"unknown xdna options {sorted(bad)}; known: {sorted(known)}"
        if "row_buckets" in options:
            options = dict(options, row_buckets=tuple(sorted(options["row_buckets"])))
        return replace(self, **options)


config = Config()


def set_config(**options):
    """Process-wide defaults; per-compile options override them."""
    global config
    config = config.merged(options)
    return config
