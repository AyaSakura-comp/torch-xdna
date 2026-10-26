"""The NPU side: one Triton kernel (W8A8 GEMM with a K loop) built by Triton-XDNA.

Nothing here is imported until a GEMM is actually routed to the NPU, so
`import torch_xdna` works on machines without the NPU stack; `available()`
says whether it can be used and, if not, why.
"""

import os
import threading
from pathlib import Path

BLOCK_N = 512
BLOCK_K = 128
# Rows per output block -> per-core tile (l1_m, l1_n). 512 and 256 spread the
# block over a 8x4 / 4x4 herd of 64x128 tiles, 128 over 4x4 of 32x128.
L1 = {512: (64, 128), 256: (64, 128), 128: (32, 128)}
CACHE = Path(os.environ.get("TORCH_XDNA_CACHE", Path.home() / ".cache" / "torch_xdna"))

_lock = threading.RLock()
_project = None


def project_dir() -> str:
    """This process's own AIR build directory (removed at exit).

    Triton-XDNA builds in a project directory and caches what it finds there;
    two processes building in one directory at once silently produce wrong
    binaries (reproduced: 8 of 16 GEMMs wrong), so never share it.
    """
    global _project
    if _project is None:
        import atexit
        import shutil

        _project = str(CACHE / "air_project" / str(os.getpid()))
        os.makedirs(_project, exist_ok=True)
        atexit.register(shutil.rmtree, _project, True)
    return _project
_state = {"ok": None, "error": None}
_mod = {}


def available() -> bool:
    """True if GEMMs can run on the NPU in this process."""
    with _lock:
        if _state["ok"] is None:
            try:
                if os.environ.get("TORCH_XDNA_DISABLE_NPU") == "1":
                    raise RuntimeError("disabled by TORCH_XDNA_DISABLE_NPU=1")
                if not os.path.exists("/dev/accel/accel0"):
                    raise RuntimeError("no /dev/accel/accel0 (NPU off or amdxdna not loaded)")
                _load()
                _state["ok"] = True
            except Exception as e:  # any missing piece means: CPU only
                _state["ok"], _state["error"] = False, f"{type(e).__name__}: {e}"
        return _state["ok"]


def unavailable_reason():
    return _state["error"]


def mark_broken(err: BaseException):
    """A launch failed: stop routing to the NPU for the rest of the process."""
    with _lock:
        _state["ok"], _state["error"] = False, f"{type(err).__name__}: {err}"


def _load():
    import triton
    import triton.backends.amd_triton_npu.driver as drv
    from triton.backends.amd_triton_npu import resident
    from triton.backends.amd_triton_npu.config import npu_config
    from triton.backends.amd_triton_npu.driver import NPUDriver

    # Every (rows, N, K) program holds one hw_context. The driver keeps at most
    # _MAX_LIVE_SESSIONS (default 8, read from the env when `import triton`
    # loads it -- usually before this runs) and rebuilds an evicted one in
    # 50-80 ms, which a model cycling through more shapes pays on every switch.
    # It is read at eviction time, so raise it here; when the device has less
    # room (other NPU users, e.g. an FLM server) the driver evicts on demand.
    want = int(os.environ.get("TORCH_XDNA_MAX_SESSIONS", "24"))
    drv._MAX_LIVE_SESSIONS = max(drv._MAX_LIVE_SESSIONS, want)

    from . import _triton_kernels

    CACHE.mkdir(parents=True, exist_ok=True)
    _mod.update(triton=triton, resident=resident, npu_config=npu_config,
                driver=NPUDriver(), kernel=_triton_kernels.matmul_i8_k, kernel_bf16=_triton_kernels.matmul_bf16_k)


def resident():
    return _mod["resident"]


class Tiling(tuple):
    """(block_m, block_n, l1_m, l1_n, l2_k): one launch program covers block_m x block_n of C,
    spread over a herd of (block_m / l1_m) x (block_n / l1_n) cores; K advances l2_k per step."""

    __slots__ = ()

    def __new__(cls, block_m, block_n, l1_m, l1_n, l2_k=BLOCK_K):
        return super().__new__(cls, (block_m, block_n, l1_m, l1_n, l2_k))

    block_m = property(lambda s: s[0])
    block_n = property(lambda s: s[1])
    l1_m = property(lambda s: s[2])
    l1_n = property(lambda s: s[3])
    l2_k = property(lambda s: s[4])

    def herd(self):
        return self.block_m // self.l1_m, self.block_n // self.l1_n

    def __str__(self):
        return f"B{self.block_m}x{self.block_n}_L1{self.l1_m}x{self.l1_n}_K{self.l2_k}"


def default_tiling(block_m: int) -> Tiling:
    l1_m, l1_n = L1[block_m]
    return Tiling(block_m, BLOCK_N, l1_m, l1_n, BLOCK_K)


_tuned = None


def tuned_table():
    """{(M, N, K): Tiling} from a sweep (tools/sweep.py); TORCH_XDNA_TUNED overrides the shipped one."""
    global _tuned
    if _tuned is None:
        import json

        path = os.environ.get("TORCH_XDNA_TUNED") or str(Path(__file__).parent / "tuned" / "strix_halo_i8.json")
        _tuned = {}
        if os.path.exists(path):
            import re

            for key, v in json.load(open(path)).items():
                m = re.fullmatch(r"B(\d+)x(\d+)_L1(\d+)x(\d+)_K(\d+)", v["tiling"])
                if m:
                    _tuned[tuple(int(x) for x in key.split("x"))] = Tiling(*map(int, m.groups()))
    return _tuned


def tiling_for(M: int, N: int, K: int) -> Tiling:
    """The swept best tiling for this launch when there is one that fits, else the default for its rows."""
    t = tuned_table().get((M, N, K))
    if t is not None and M % t.block_m == 0 and N % t.block_n == 0 and K % t.l2_k == 0:
        return t
    return default_tiling(512 if M % 512 == 0 else (256 if M % 256 == 0 else 128))


def schedule(t, dtype="i8") -> str:
    """The AIR tiling script for a tiling and input type ("i8" or "bf16"), generated once, cached on disk."""
    t = t if isinstance(t, Tiling) else default_tiling(t)
    path = CACHE / f"transform_{dtype}_k{t.l2_k}_{t.l1_m}x{t.l1_n}.mlir"
    if not path.exists():
        from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform

        kw = dict(accum_type="i32") if dtype == "i8" else dict(accum_type="f32")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(generate_matmul_transform(l1_m=t.l1_m, l1_n=t.l1_n, l2_k=t.l2_k,
                                                 pack_sizes=(8, 8, 8), k_loop=True, **kw))
        tmp.replace(path)
    return str(path)


def mm_bf16(a, w, c, tiling, project=None):
    """c[M, N] (f32) = a[M, K] @ w[K, N] (bf16) on the NPU; see mm_i8 for the layout rules."""
    return mm_i8(a, w, c, tiling, project, dtype="bf16")


def mm_i8(a, w, c, tiling, project=None, dtype="i8"):
    """c[M, N] = a[M, K] @ w[K, N] on the NPU; int8 in, int32 out, all resident.

    `tiling` is a Tiling or a row block (then the default tiling for it). M
    must be a multiple of block_m, N of block_n, K of l2_k. The schedule and
    the build directory are process-global settings of Triton-XDNA, read at
    launch, so they are set around this launch and restored after.
    """
    t = tiling if isinstance(tiling, Tiling) else default_tiling(tiling)
    M, K = a.shape
    N = w.shape[1]
    cfg = _mod["npu_config"]
    _mod["triton"].runtime.driver.set_active(_mod["driver"])  # per thread
    saved = (cfg._transform_tiling_script, cfg._air_project_path)
    cfg._transform_tiling_script = schedule(t, dtype)
    cfg._air_project_path = project or project_dir()
    try:
        _mod["kernel" if dtype == "i8" else "kernel_bf16"][(M // t.block_m, N // t.block_n)](
            a, w, c, N, K, BLOCK_M=t.block_m, BLOCK_N=t.block_n, BLOCK_K=t.l2_k)
    finally:
        cfg._transform_tiling_script, cfg._air_project_path = saved
