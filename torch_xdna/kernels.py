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
                driver=NPUDriver(), kernel=_triton_kernels.matmul_i8_k)


def resident():
    return _mod["resident"]


def schedule(block_m: int) -> str:
    """The AIR tiling script for this row block (generated once, cached on disk)."""
    l1_m, l1_n = L1[block_m]
    path = CACHE / f"transform_i8_k{BLOCK_K}_{l1_m}x{l1_n}.mlir"
    if not path.exists():
        from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform

        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(generate_matmul_transform(l1_m=l1_m, l1_n=l1_n, l2_k=BLOCK_K,
                                                 pack_sizes=(8, 8, 8), accum_type="i32", k_loop=True))
        tmp.replace(path)
    return str(path)


def mm_i8(a, w, c, block_m: int):
    """c[M, N] = a[M, K] @ w[K, N] on the NPU; int8 in, int32 out, all resident.

    M must be a multiple of block_m, N of BLOCK_N, K of BLOCK_K. The schedule
    and the build directory are process-global settings of Triton-XDNA, read
    at launch, so they are set around this launch and restored after.
    """
    M, K = a.shape
    N = w.shape[1]
    cfg = _mod["npu_config"]
    _mod["triton"].runtime.driver.set_active(_mod["driver"])  # per thread
    saved = (cfg._transform_tiling_script, cfg._air_project_path)
    cfg._transform_tiling_script = schedule(block_m)
    cfg._air_project_path = str(CACHE / "air_project")
    try:
        _mod["kernel"][(M // block_m, N // BLOCK_N)](
            a, w, c, N, K, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    finally:
        cfg._transform_tiling_script, cfg._air_project_path = saved
