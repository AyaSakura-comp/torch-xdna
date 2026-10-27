"""Options of the xdna backend: torch.compile(..., backend="xdna", options={...}) or torch_xdna.config."""

from dataclasses import dataclass, fields, replace

POLICIES = ("auto", "npu", "cpu")
FALLBACKS = ("inductor", "eager")
CPU_KERNELS = ("int8", "fp")
PRECISIONS = ("int8", "bf16")


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
    # Offloaded GEMM math:
    #   int8  W8A8 (per-channel weights, per-row activations), int32 accumulate; NPU == CPU bit for bit
    #   bf16  bf16 operands (bfp16 MACs on AIE2P, 8 values per shared exponent), f32 accumulate;
    #         the CPU route is the fp32 GEMM. More robust to activation outliers.
    precision: str = "int8"
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
    # A whole transformer stack the NPU has a one-launch program for -- a transformers Qwen3Model
    # of Qwen3-0.6B's shapes feeding a linear head, attention under a 4-D mask given to the graph
    # (OmniVoice's backbone) -- runs as that launch (Triton-XDNA's tl.extra.npu.qwen3_stack, bfp16)
    # whenever the mask is a prefix block per sequence of at most 384 positions; other calls run
    # the graph compiled as usual (stack.py). Policies "auto" and "npu" alike.
    stacks: bool = True
    # Pick each NPU launch's tiling from the swept table (tools/sweep.py) instead of the per-row default.
    tuned_tilings: bool = True
    # Check NPU results against the CPU: "first" launch of every NPU program (catches a silently
    # miscompiled program once; it then falls back to the CPU), "always", or "off".
    verify: str = "first"
    verbose: bool = False

    def __post_init__(self):
        assert self.policy in POLICIES, f"policy must be one of {POLICIES}"
        assert self.fallback in FALLBACKS, f"fallback must be one of {FALLBACKS}"
        assert self.cpu_kernel in CPU_KERNELS, f"cpu_kernel must be one of {CPU_KERNELS}"
        assert self.precision in PRECISIONS, f"precision must be one of {PRECISIONS}"
        assert self.verify in ("first", "always", "off"), "verify must be first / always / off"
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
