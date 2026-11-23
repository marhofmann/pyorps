"""GPU step lookup table preparation.

Computes intermediate cells and cost factors for all step directions,
formatted for GPU kernel consumption. Pure Python/NumPy (no Numba/Cython).
"""

# ruff: noqa: F401
# pylint: disable=unused-import
import math

import numpy as np

#: True when CuPy could be imported. It says nothing about a device being present: that is only known once a
#: kernel is compiled or launched, and the errors of that step are reported by the code that hits them.
#: ``GPU_IMPORT_ERROR`` keeps the reason for ``False`` so that "GPU not available" messages can name it.
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_IMPORT_ERROR = None
except ImportError as _import_error:
    GPU_AVAILABLE = False
    GPU_IMPORT_ERROR = _import_error


def describe_gpu_error(err: BaseException) -> str:
    """``"<ExceptionType>: <message>"`` for warnings and messages that report a GPU failure."""
    return f"{type(err).__name__}: {err}"


def gpu_probe_errors() -> tuple[type[BaseException], ...]:
    """Exception types that mean "this GPU stack cannot compile or run the kernel" in an availability probe.

    ``ImportError`` (CuPy or an NVIDIA wheel is missing), ``OSError`` (a CUDA library or kernel source file cannot
    be loaded), ``RuntimeError`` (the base of CuPy's ``CUDARuntimeError``, ``CUDADriverError`` and ``NVRTCError``,
    and what CuPy raises when the CUDA toolkit or ``cudadevrt`` is not found) and CuPy's ``CompileException``.
    Everything else (``TypeError``, ``ZeroDivisionError``, ``AttributeError``, ...) is a programming error and must
    propagate. A caught error can still be an out-of-memory condition: check :func:`is_gpu_out_of_memory` first.
    """
    errors: list[type[BaseException]] = [ImportError, OSError, RuntimeError]
    if GPU_AVAILABLE:
        try:
            from cupy.cuda.compiler import CompileException
            errors.append(CompileException)
        except ImportError:  # pragma: no cover - a CuPy build without the NVRTC compiler module
            pass
    return tuple(errors)


def is_gpu_out_of_memory(err: BaseException) -> bool:
    """True when ``err`` reports exhausted device (or host) memory.

    CuPy's pool raises ``cupy.cuda.memory.OutOfMemoryError`` (a ``MemoryError``); a raw CUDA allocation raises
    ``CUDARuntimeError: cudaErrorMemoryAllocation: out of memory`` or ``CUDADriverError: CUDA_ERROR_OUT_OF_MEMORY``.
    """
    if isinstance(err, MemoryError):
        return True
    if not isinstance(err, RuntimeError):
        return False
    text = str(err).lower()
    return "out of memory" in text or "out_of_memory" in text or "memoryallocation" in text


def gpu_out_of_memory_error(err: BaseException, what: str, shape=None, nbytes=None) -> MemoryError:
    """A ``MemoryError`` that says the GPU ran out of memory while doing ``what``; raise it ``from err``.

    ``shape`` and ``nbytes`` describe the array the caller was working on, when it knows them.
    """
    details = []
    if shape is not None:
        details.append(f"array shape {tuple(int(n) for n in shape)}")
    if nbytes is not None:
        details.append(f"{int(nbytes):,} bytes")
    where = f" ({', '.join(details)})" if details else ""
    return MemoryError(f"GPU out of memory while {what}{where}: {describe_gpu_error(err)}")


def gpu_not_available_error(message: str, cause: BaseException | None = None) -> RuntimeError:
    """The ``RuntimeError`` for "no usable GPU", with the reason appended and kept as ``__cause__``.

    ``cause`` defaults to the error of the failed ``import cupy``.
    """
    if cause is None:
        cause = GPU_IMPORT_ERROR
    if cause is None:
        return RuntimeError(message)
    error = RuntimeError(f"{message} (cause: {describe_gpu_error(cause)})")
    error.__cause__ = cause
    return error


def _intermediate_steps_cpu(dr: int, dc: int) -> np.ndarray:
    """Bresenham-like intermediate cells for a (dr, dc) step.

    Returns (N, 2) int8 array of (row_offset, col_offset) pairs.
    Crosses the same CELLS as ``_traversal_numba.intermediate_steps_numba``.
    NOTE: for collinear reducible steps -- (0,2), (2,2), (3,3) -- this lists
    each crossed cell TWICE where the numba version lists it once. Harmless
    for passability (same cell set) and unreachable in practice, since no
    neighbourhood r0-r3 contains a reducible step. It would matter if one
    ever did, because n_inter also scales cost_factor.
    """
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    if abs_dr + abs_dc <= 1:
        return np.zeros((0, 2), dtype=np.int8)

    k = max(abs_dr, abs_dc)
    if k == 1:
        return np.array([[dr, 0], [0, dc]], dtype=np.int8)

    result = np.zeros((2 * (k - 1), 2), dtype=np.int8)
    for i in range(k - 1):
        dr_k = (i + 1) * dr / k
        dc_k = (i + 1) * dc / k
        idx = i * 2
        result[idx, 0] = np.int8(math.floor(dr_k))
        result[idx, 1] = np.int8(math.floor(dc_k))
        result[idx + 1, 0] = np.int8(math.ceil(dr_k))
        result[idx + 1, 1] = np.int8(math.ceil(dc_k))
    return result


def prepare_step_lookup_tables(steps: np.ndarray):
    """Pre-compute intermediate cells and cost factors for all step directions.

    Parameters
    ----------
    steps : (n_dirs, 2) array
        Step directions as (dr, dc) pairs.

    Returns
    -------
    steps_i8 : (n_dirs, 2) int8
    intermediates_lut : (n_dirs, max_inter, 2) int16
    n_intermediates : (n_dirs,) int32
    cost_factors : (n_dirs,) float32
    """
    n_dirs = len(steps)

    # Compute intermediates for each direction
    inter_lists = []
    max_inter = 0
    for i in range(n_dirs):
        dr, dc = int(steps[i, 0]), int(steps[i, 1])
        inter = _intermediate_steps_cpu(dr, dc)
        inter_lists.append(inter)
        max_inter = max(max_inter, len(inter))

    if max_inter == 0:
        max_inter = 1  # avoid zero-sized dimension

    # Pack into fixed-size arrays
    intermediates_lut = np.zeros((n_dirs, max_inter, 2), dtype=np.int16)
    n_intermediates = np.zeros(n_dirs, dtype=np.int32)
    cost_factors = np.zeros(n_dirs, dtype=np.float32)
    steps_i8 = steps.astype(np.int8)

    for i in range(n_dirs):
        inter = inter_lists[i]
        n_inter = len(inter)
        n_intermediates[i] = n_inter
        if n_inter > 0:
            intermediates_lut[i, :n_inter, :] = inter.astype(np.int16)

        dr, dc = float(steps[i, 0]), float(steps[i, 1])
        dist = math.sqrt(dr * dr + dc * dc)
        cost_factors[i] = dist / (2.0 + n_inter)

    return steps_i8, intermediates_lut, n_intermediates, cost_factors
