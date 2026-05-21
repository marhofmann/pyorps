"""ADDS-inspired constrained GPU SSSP V4 — persistent kernel with MTB delegation.

V4 of the GPU constrained SSSP: a single persistent kernel launch with
block 0 as the MTB (Manager Thread Block) and blocks 1+ as WTBs (Worker
Thread Blocks). The MTB reads items from a circular bucket queue and
assigns work to idle WTBs via volatile AssignmentFlag polling. WTBs
relax edges, place towers, and enqueue new items back into the bucket
queue. No cooperative groups required — inter-block coordination uses
volatile flag polling.

Key differences from V3 (host-driven):
- Single kernel launch (no host-driven loop overhead)
- MTB/WTB delegation pattern (no kernel launch per bucket)
- Circular bucket queue with WCC-based segment reads
- ITEMS_PER_BUCKET compile-time constant (no Python-side queue management)

The block-sparse distance storage, tower records, and path reconstruction
are shared with V3.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False


# ============================================================================
# Constants (must match CUDA header defines)
# ============================================================================

_N_BUCKETS = 32
_SEGMENT_SIZE = 32
_ITEMS_PER_BUCKET = 65536
_MTB_STAGING_SIZE = 256
_MAX_TOWER_RECORDS = 2_000_000
_CTL_V4_SIZE = 16

# CTL_V4 indices (must match adds_common.cuh)
_CTL_V4_DONE = 0
_CTL_V4_TOWER_COUNT = 1
_CTL_V4_BLOCK_OVERFLOW = 2
_CTL_V4_POOL_OVERFLOW = 3
_CTL_V4_STALE_ASSIGNMENTS = 4
_CTL_V4_HEAD_LOGICAL = 5
_CTL_V4_EMPTY_SWEEPS = 6
_CTL_V4_ASSIGNMENTS_TOTAL = 7


# ============================================================================
# Kernel source loading (shared with V3 pattern)
# ============================================================================

_KERNEL_DIR = Path(__file__).parent / "kernels"


def _load_kernel_source(main_file: str, block_size: int = 64) -> str:
    """Load CUDA source with #include resolution from kernels/ directory.

    CuPy's RawKernel does not support #include natively, so we resolve
    local #include "xxx.cuh" directives in Python before passing the
    concatenated source to CuPy. System includes (angle-bracket) are
    left intact for nvrtc.

    Parameters:
        main_file: filename in kernels/ directory (e.g. "constrained_adds.cu")
        block_size: injected as #define BLOCK_SIZE before the source
    """
    main_path = _KERNEL_DIR / main_file
    source = main_path.read_text(encoding="utf-8")
    # Inject BLOCK_SIZE before any includes
    source = f"#define BLOCK_SIZE {block_size}\n" + source

    # Track already-included files to respect #pragma once
    included: set = set()

    def resolve_include(match):
        inc_file = match.group(1)
        if inc_file in included:
            return ""  # already included (#pragma once)
        inc_path = _KERNEL_DIR / inc_file
        if inc_path.exists():
            included.add(inc_file)
            content = inc_path.read_text(encoding="utf-8")
            # Remove #pragma once from included content
            content = re.sub(r"#pragma\s+once\s*\n?", "", content)
            # Recursively resolve nested includes
            content = re.sub(
                r'#include\s+"([^"]+)"', resolve_include, content)
            return content
        return match.group(0)  # leave unresolved includes unchanged

    # Only resolve local includes (quoted, not angle-bracket)
    source = re.sub(r'#include\s+"([^"]+)"', resolve_include, source)
    return source


def _inject_defines(source: str, defines: dict) -> str:
    """Prepend #define statements to kernel source."""
    header = "\n".join(f"#define {k} {v}" for k, v in defines.items())
    return header + "\n" + source


# ============================================================================
# Memory budget helpers
# ============================================================================

def _compute_block_size(spc: int, vram_free_bytes: int, n_cells: int,
                        max_visited_fraction: float = 1.0) -> int:
    """Select optimal BLOCK_SIZE (power of 2) from spc and available VRAM.

    Prefers smaller BLOCK_SIZE with more blocks over larger BLOCK_SIZE
    with fewer blocks, because eviction is acceptable (paths within
    0.1% of optimal even with BLOCK_SIZE=32).
    """
    max_blocks = int(n_cells * max_visited_fraction * 1.5)
    # Target: next power of 2 >= spc (no eviction)
    target_bs = 1
    while target_bs < spc:
        target_bs *= 2
    # Reserve 1 GB for input + queues + tower records
    available = max(vram_free_bytes - 1024**3, 256 * 1024**2)
    # Shrink BLOCK_SIZE first (keep max_blocks high) rather than
    # reducing max_blocks (which starves the block pool)
    while target_bs > 32 and target_bs * max_blocks * 10 > available:
        target_bs //= 2
    return max(32, target_bs)


def _compute_constrained_delta(raster: np.ndarray,
                                cost_factors: np.ndarray) -> float:
    """Compute delta for constrained delta-stepping.

    Delta should upper-bound the cost of non-tower terrain edges so
    they are "light" (processed within a single bucket) while tower
    edges (which cost thousands) are "heavy" (deferred to later
    buckets). This gives proper delta-stepping behavior.

    The worst-case non-tower terrain edge cost depends on the
    neighborhood: larger neighborhoods have more intermediate cells
    (e.g. R4 steps like (4,1) visit ~5 intermediates). The cost
    formula is ``(src + intermediates + dst) * cost_factor``, bounded
    by ``(2 + n_inter) * max_raster * cost_factor``. Since
    ``cost_factor = distance / (2 + n_inter)``, the actual edge cost
    is ``distance * max_raster = sqrt(dr^2 + dc^2) * max_raster``.

    The old formula ``2 * max * max_cf * n_dirs`` multiplied by the
    number of directions, which is wrong: n_dirs does not appear in
    the edge cost formula. For R4 (48 dirs) with raster values
    125-835 this produces delta ~38K, far exceeding per-step costs
    (~250-1700), so ALL edges land in bucket 0 and delta-stepping
    degenerates to Bellman-Ford.

    The correct formula: ``max_step_distance * max_raster``, which is
    the tight upper bound on the most expensive single terrain edge.
    We multiply by a small constant (2x) for safety margin to account
    for angle penalties on terrain edges.
    """
    valid = raster.ravel()
    valid = valid[valid < 65535]
    if len(valid) == 0:
        return 100.0
    max_cost = float(max(1, int(valid.max())))
    # max_step_distance = max cost_factor * (2 + n_inter) for any step,
    # but since cf = distance / (2 + n_inter), max_step_distance is just
    # the Euclidean distance of the longest step. However, cost_factors
    # array has cf = dist / (2 + n_inter), so max_edge_cost =
    # max_raster * (2 + n_inter) * cf = max_raster * dist.
    # The longest step distance in the neighborhood is the max of
    # cf * (2 + n_inter) for all directions, but we can approximate
    # by using max_cf * 8 (a generous bound for R1-R4 neighborhoods
    # where the longest step has ~4-5 intermediates, distance ~4.1,
    # thus cost_factor ~0.68, times (2+5)=7 ≈ 4.8 raster cells).
    # Using a fixed multiplier of 8 works for all neighborhood sizes.
    max_cf = float(cost_factors.max())
    return max(1.0, 2.0 * max_cost * max_cf * 8)


def _compute_smem_bytes(n_dirs: int, n_heights: int,
                        max_inter_cols: int) -> int:
    """Compute shared memory bytes for the persistent kernel.

    Must match the layout in constrained_adds.cu's shared memory section:
      s_steps:            n_dirs * 2 bytes (int8)     -> pad to 4-byte boundary
      s_cost_factors:     n_dirs * 4 bytes (float32)
      s_step_distances:   n_dirs * 4 bytes (float32)
      s_angle_valid:      n_dirs^2 bytes (uint8)      -> pad to 4-byte boundary
      s_angle_cost:       n_dirs^2 * 4 bytes (float32)
      s_tower_angle_cost: n_dirs^2 * 4 bytes (float32)
      s_height_premiums:  n_heights * 4 bytes (float32)
      s_intermediates:    n_dirs * max_inter_cols * 2 * 2 bytes (int16) -> pad to 4
      s_n_intermediates:  n_dirs * 4 bytes (int32)
    Plus 256 bytes for MTB shared variables and safety margin.
    """
    n2 = n_dirs * n_dirs
    steps_padded = (n_dirs * 2 + 3) & ~3
    av_padded = (n2 + 3) & ~3
    inter_bytes = n_dirs * max_inter_cols * 2 * 2  # int16 = 2 bytes each
    inter_padded = (inter_bytes + 3) & ~3
    smem = (steps_padded
            + n_dirs * 4        # cost_factors
            + n_dirs * 4        # step_distances
            + av_padded         # angle_valid
            + n2 * 4            # angle_cost
            + n2 * 4            # tower_angle_cost
            + n_heights * 4     # height_premiums
            + inter_padded      # intermediates
            + n_dirs * 4        # n_intermediates
            + 256)              # MTB shared vars + safety margin
    return smem


# ============================================================================
# State encoding helpers (Python-side) — identical to V3
# ============================================================================

def _pack_state(cell, direction, span_bin, height_class,
                spc, n_span_bins, n_heights):
    """Pack (cell, dir, span_bin, height_class) into int64 state."""
    return (cell * spc
            + direction * n_span_bins * n_heights
            + span_bin * n_heights
            + height_class)


def _unpack_state(state, spc, n_span_bins, n_heights):
    """Unpack int64 state into (cell, dir, span_bin, height_class)."""
    cell = state // spc
    rem = state % spc
    direction = rem // (n_span_bins * n_heights)
    rem = rem % (n_span_bins * n_heights)
    span_bin = rem // n_heights
    height_class = rem % n_heights
    return cell, direction, span_bin, height_class


# ============================================================================
# Dynamic block distance proxy for path reconstruction — identical to V3
# ============================================================================

class _DynamicBlockDistProxy:
    """Dynamic block-sparse proxy that mimics array indexing for dist_cpu[state].

    Used for path reconstruction. Blocks are indexed via cell_to_block
    mapping. Uses open-addressing hash (multiplicative hash) with linear
    probing to find entries within a block.
    """
    _BLOCK_EMPTY = 0xFFFF

    def __init__(self, block_entries_cpu, cell_to_block_cpu, block_size,
                 spc, n_span_bins, n_heights):
        self._blocks = block_entries_cpu
        self._cell_to_block = cell_to_block_cpu
        self._block_size = block_size
        self._mask = block_size - 1
        self._spc = spc
        self._n_span_bins = n_span_bins
        self._n_heights = n_heights
        self._sh = n_span_bins * n_heights

    def __getitem__(self, state):
        state = int(state)
        cell = state // self._spc
        block_idx = self._cell_to_block[cell]
        if block_idx < 0:
            return 1e30
        base = block_idx * self._block_size
        rem = state % self._spc
        direction = rem // self._sh
        rem2 = rem % self._sh
        span_bin = rem2 // self._n_heights
        hc = rem2 % self._n_heights
        local_key = direction * self._sh + span_bin * self._n_heights + hc
        h = (local_key * 2654435761) & self._mask
        for probe in range(self._block_size):
            slot = (h + probe) & self._mask
            entry = self._blocks[base + slot]
            k = int(entry['local_key'])
            if k == local_key:
                return float(entry['dist'])
            if k == self._BLOCK_EMPTY:
                return 1e30
        return 1e30


# ============================================================================
# Path reconstruction from tower records — adapted from V3
# ============================================================================

def _direction_walk_backward(target_cell, target_dir, source_cell,
                              cols, steps_np, n_dirs, max_steps=10000):
    """Walk backward from target_cell using inverse of target_dir to reach
    source_cell.

    Returns list of cell indices from source_cell to target_cell.
    """
    path = [target_cell]
    current_cell = target_cell
    current_dir = target_dir

    for _ in range(max_steps):
        if current_cell == source_cell:
            break
        # Step backward: opposite of current direction
        dr = int(steps_np[current_dir, 0])
        dc = int(steps_np[current_dir, 1])
        cur_row = current_cell // cols
        cur_col = current_cell % cols
        prev_row = cur_row - dr
        prev_col = cur_col - dc
        if prev_row < 0 or prev_col < 0 or prev_col >= cols:
            break
        prev_cell = prev_row * cols + prev_col
        path.append(prev_cell)
        current_cell = prev_cell

    path.reverse()
    return path


def _build_tower_record_index_v4(tower_records_cpu, n_tower_records,
                                  spc, n_span_bins, n_heights):
    """Build (cell, dir) -> list of record indices for fast lookup."""
    cell_dir_to_records = {}
    for i in range(n_tower_records):
        rec_state = int(tower_records_cpu['state'][i])
        rc, rd, _, _ = _unpack_state(rec_state, spc, n_span_bins, n_heights)
        key = (rc, rd)
        if key not in cell_dir_to_records:
            cell_dir_to_records[key] = []
        cell_dir_to_records[key].append(i)
    return cell_dir_to_records


def _find_ancestor_record_v4(cell_dir_to_records, tower_records_cpu,
                              walk_cell, walk_dir, max_dist, dist_cpu):
    """Find a tower record at (walk_cell, walk_dir) that is an ancestor."""
    key = (walk_cell, walk_dir)
    if key not in cell_dir_to_records:
        return None
    candidates = cell_dir_to_records[key]
    best_rec = None
    best_d = 1e30
    for idx in candidates:
        s = int(tower_records_cpu['state'][idx])
        if dist_cpu is not None:
            d = float(dist_cpu[s])
            if d < max_dist and d < best_d:
                best_d = d
                best_rec = idx
        else:
            best_rec = idx  # take any
    return best_rec


def _find_prev_tower_v4(current_cell, current_dir, current_dist,
                         source_cell, cols, steps_np, enforce_min_span,
                         min_span, _step_dist_arr, cell_dir_to_records,
                         tower_records_cpu, dist_cpu, spc, n_span_bins,
                         n_heights):
    """Walk backward from current_cell to find the nearest ancestor tower.

    Returns (tower_info_dict, pred_cell, pred_dir, pred_dist) if found,
    or None if no tower was found.
    """
    walk_cell = current_cell
    walk_dist = 0.0  # accumulated physical distance from current_cell
    for step in range(1000):
        dr = int(steps_np[current_dir, 0])
        dc = int(steps_np[current_dir, 1])
        r = walk_cell // cols
        c = walk_cell % cols
        nr, nc = r - dr, c - dc
        if nr < 0 or nc < 0 or nc >= cols:
            break
        walk_cell = nr * cols + nc
        # Accumulate physical distance walked backward
        if _step_dist_arr is not None:
            walk_dist += float(_step_dist_arr[current_dir])

        if walk_cell == source_cell:
            break

        # The tower (pred_cell) is one more step back from walk_cell,
        # so the total distance from current_cell to the tower is
        # walk_dist + step_distance[current_dir].
        if (enforce_min_span and min_span > 0
                and _step_dist_arr is not None):
            tower_dist = walk_dist + float(_step_dist_arr[current_dir])
            if tower_dist < min_span * 0.95:
                # Tower would be too close to the previous tower;
                # this record must be from a different exploration path.
                continue

        rec_idx = _find_ancestor_record_v4(
            cell_dir_to_records, tower_records_cpu,
            walk_cell, current_dir, current_dist, dist_cpu)
        if rec_idx is not None:
            pred_state = int(tower_records_cpu['pred_state'][rec_idx])
            height = float(tower_records_cpu['tower_height'][rec_idx])
            pred_cell, pred_dir, _, _ = _unpack_state(
                pred_state, spc, n_span_bins, n_heights)

            tower_info = {
                'tower_cell': pred_cell,
                'tower_dir': pred_dir,
                'post_cell': walk_cell,
                'post_dir': current_dir,
                'height': height,
            }

            pred_dist = (float(dist_cpu[pred_state])
                         if dist_cpu is not None else current_dist)
            return tower_info, pred_cell, pred_dir, pred_dist

    return None


def _walk_backward_tower_chain_v4(tower_records_cpu, cell_dir_to_records,
                                   target_cell, target_dir, source_cell,
                                   cols, steps_np, spc, n_span_bins,
                                   n_heights, dist_cpu, best_state,
                                   min_span, step_distances):
    """Walk backward from best_state to find the tower chain.

    Enforces min_span between consecutive towers (except from target
    to the first tower found). Returns tower_chain list.
    """
    # Precompute per-direction step distances for min_span enforcement.
    if step_distances is not None:
        _step_dist_arr = np.asarray(step_distances, dtype=np.float64)
    else:
        _step_dist_arr = None

    tower_chain = []
    current_cell = target_cell
    current_dir = target_dir
    current_dist = float(dist_cpu[best_state]) if dist_cpu is not None else 1e30
    # The first backward walk is from the target cell, not from a tower.
    # The last tower before the target can be at any distance (no min_span
    # enforcement). After finding the first tower, subsequent towers must
    # satisfy min_span.
    enforce_min_span = False

    for _ in range(1000):  # safety limit
        if current_cell == source_cell:
            break

        result = _find_prev_tower_v4(
            current_cell, current_dir, current_dist,
            source_cell, cols, steps_np, enforce_min_span,
            min_span, _step_dist_arr, cell_dir_to_records,
            tower_records_cpu, dist_cpu, spc, n_span_bins,
            n_heights)

        if result is None:
            break

        tower_info, current_cell, current_dir, current_dist = result
        tower_chain.append(tower_info)
        # After finding the first tower, enforce min_span for
        # all subsequent tower-to-tower distances.
        enforce_min_span = True

    tower_chain.reverse()
    return tower_chain


def _assemble_path_from_tower_chain(tower_chain, target_cell, target_dir,
                                     source_cell, cols, steps_np, n_dirs):
    """Build full path by direction-walking between waypoints."""
    waypoints = []

    if tower_chain:
        first = tower_chain[0]
        seg = _direction_walk_backward(
            first['tower_cell'], first['tower_dir'], source_cell,
            cols, steps_np, n_dirs)
        waypoints.extend(seg)

        for i, tc in enumerate(tower_chain):
            if i + 1 < len(tower_chain):
                next_tc = tower_chain[i + 1]
                seg = _direction_walk_backward(
                    next_tc['tower_cell'], next_tc['tower_dir'],
                    tc['post_cell'], cols, steps_np, n_dirs)
            else:
                seg = _direction_walk_backward(
                    target_cell, target_dir, tc['post_cell'],
                    cols, steps_np, n_dirs)

            if seg and waypoints and seg[0] == waypoints[-1]:
                seg = seg[1:]
            waypoints.extend(seg)
    else:
        seg = _direction_walk_backward(
            target_cell, target_dir, source_cell, cols, steps_np, n_dirs)
        waypoints.extend(seg)

    tower_cells = [tc['tower_cell'] for tc in tower_chain]
    tower_heights_out = [tc['height'] for tc in tower_chain]

    path_arr = (np.array(waypoints, dtype=np.uint32)
                if waypoints else np.empty(0, dtype=np.uint32))
    tower_arr = (np.array(tower_cells, dtype=np.uint32)
                 if tower_cells else np.empty(0, dtype=np.uint32))
    height_arr = (np.array(tower_heights_out, dtype=np.float32)
                  if tower_heights_out else np.empty(0, dtype=np.float32))
    return path_arr, tower_arr, height_arr


def _reconstruct_from_tower_records(
    tower_records_cpu,
    n_tower_records,
    best_state,
    source_cell,
    spc, n_span_bins, n_heights, n_dirs,
    cols, steps_np,
    dist_cpu=None,
    min_span=0.0,
    step_distances=None,
):
    """Reconstruct path and tower locations from TowerRecord chain.

    Tower records store (state_after_move, pred_state_before_tower).
    Reconstruction walks backward from best_state to source_cell,
    finding tower records along the way.

    Parameters:
        min_span: minimum span between towers in meters. When > 0,
            the backward walk skips tower records that would place
            consecutive towers closer than min_span apart.
        step_distances: float array of physical step distances per
            direction (meters). Required when min_span > 0.

    Returns:
        (path_indices, tower_indices, tower_heights) as numpy arrays
    """
    target_cell, target_dir, _, _ = _unpack_state(
        best_state, spc, n_span_bins, n_heights)

    if n_tower_records == 0:
        if target_cell == source_cell:
            return (np.array([target_cell], dtype=np.uint32),
                    np.empty(0, dtype=np.uint32),
                    np.empty(0, dtype=np.float32))
        path = _direction_walk_backward(
            target_cell, target_dir, source_cell, cols, steps_np, n_dirs)
        return (np.array(path, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))

    # Build (cell, dir) -> list of record indices for fast lookup.
    cell_dir_to_records = _build_tower_record_index_v4(
        tower_records_cpu, n_tower_records, spc, n_span_bins, n_heights)

    # Walk backward from best_state to find the tower chain.
    tower_chain = _walk_backward_tower_chain_v4(
        tower_records_cpu, cell_dir_to_records,
        target_cell, target_dir, source_cell,
        cols, steps_np, spc, n_span_bins, n_heights,
        dist_cpu, best_state, min_span, step_distances)

    # Build full path by direction-walking between waypoints
    return _assemble_path_from_tower_chain(
        tower_chain, target_cell, target_dir,
        source_cell, cols, steps_np, n_dirs)


# ============================================================================
# CUDA path setup (shared with V3)
# ============================================================================

def _ensure_cuda_path():
    """Ensure CuPy can find cudadevrt for linking."""
    import cupy.cuda.compiler as _compiler
    if getattr(_compiler, '_cudadevrt', None) is not None:
        return
    try:
        import nvidia.cuda_runtime as ncr
        for cuda_rt_dir in getattr(ncr, '__path__', []):
            devrt = os.path.join(cuda_rt_dir, "lib", "x64", "cudadevrt.lib")
            if os.path.isfile(devrt):
                os.environ.setdefault("CUDA_PATH", cuda_rt_dir)
                import cupy._environment as _env
                _env._cuda_path = ''
                _compiler._cudadevrt = devrt
                return
    except ImportError:
        pass
    import site
    for sp in site.getsitepackages():
        cuda_rt_dir = os.path.join(sp, "nvidia", "cuda_runtime")
        devrt = os.path.join(cuda_rt_dir, "lib", "x64", "cudadevrt.lib")
        if os.path.isfile(devrt):
            os.environ.setdefault("CUDA_PATH", cuda_rt_dir)
            import cupy._environment as _env
            _env._cuda_path = ''
            _compiler._cudadevrt = devrt
            return


# ============================================================================
# Kernel compilation cache
# ============================================================================

_v4_kernel_cache = {}


def _get_v4_kernel(source: str, name: str):
    """Compile and cache a CuPy RawKernel for V4.

    V4 kernels do NOT use cooperative groups — the persistent kernel
    uses volatile flag polling for inter-block coordination.
    """
    key = (name, hash(source))
    if key not in _v4_kernel_cache:
        _ensure_cuda_path()
        _v4_kernel_cache[key] = cp.RawKernel(
            source, name,
            options=("--std=c++17", "-Xptxas", "-dlcm=cg"),
        )
    return _v4_kernel_cache[key]


# ============================================================================
# V4 availability check
# ============================================================================

_v4_available = None


def _check_v4_available():
    """Check if V4 kernels compile on this GPU."""
    global _v4_available
    if _v4_available is not None:
        return _v4_available
    if not GPU_AVAILABLE:
        _v4_available = False
        return False
    try:
        defines = {
            "N_BUCKETS": _N_BUCKETS,
            "SEGMENT_SIZE": _SEGMENT_SIZE,
            "ITEMS_PER_BUCKET": 1024,
            "MAX_SEGMENTS_PER_BUCKET": 1024 // _SEGMENT_SIZE,
            "MAX_INTER": 1,
            "MAX_WTBS": 4,
            "MTB_STAGING_SIZE": _MTB_STAGING_SIZE,
        }
        source = _load_kernel_source("adds_init.cu", block_size=64)
        source = _inject_defines(source, defines)
        kernel = cp.RawKernel(
            source, "adds_init_pool",
            options=("--std=c++17",))
        _ = kernel.kernel  # force compilation
        _v4_available = True
    except Exception:
        _v4_available = False
    return _v4_available


# ============================================================================
# Sub-functions for constrained_sssp_raster_gpu_v4
# ============================================================================

def _validate_v4_inputs(raster, source_row, source_col, target_row,
                         target_col, height_premiums, n_heights,
                         tower_heights, angle_cost_lut, tower_terrain_costs,
                         tower_angle_costs):
    """GPU check, cell validation, defaults, angle_cost cleanup, cost validation.

    Returns (height_premiums, n_heights, tower_heights, angle_cost_lut).
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU not available. Install cupy: pip install cupy-cuda12x")

    max_cost = int(np.iinfo(np.uint16).max)

    # Validate source and target
    if raster[source_row, source_col] == max_cost:
        raise ValueError(
            f"Source cell ({source_row}, {source_col}) is forbidden (65535)")
    if raster[target_row, target_col] == max_cost:
        raise ValueError(
            f"Target cell ({target_row}, {target_col}) is forbidden (65535)")

    # Defaults
    if height_premiums is None:
        height_premiums = np.zeros(1, dtype=np.float32)
        n_heights = 1
    if tower_heights is None:
        tower_heights = np.zeros(n_heights, dtype=np.float32)

    # Clean angle_cost_lut: replace inf with 0
    angle_cost_lut = angle_cost_lut.copy().astype(np.float32)
    angle_cost_lut[~np.isfinite(angle_cost_lut)] = 0.0

    # Validate non-negative costs
    if np.any(angle_cost_lut < 0):
        raise ValueError("angle_cost_lut must have non-negative values")
    if np.any(tower_terrain_costs < 0):
        raise ValueError("tower_terrain_costs must have non-negative values")
    if np.any(tower_angle_costs < 0):
        raise ValueError("tower_angle_costs must have non-negative values")

    return height_premiums, n_heights, tower_heights, angle_cost_lut


def _compute_v4_block_size(spc, n_cells, max_visited_fraction):
    """Block size + adjustment. Returns (gpu_block_size, max_visited_fraction)."""
    vram_free = int(cp.cuda.Device().mem_info[0])
    gpu_block_size = _compute_block_size(spc, vram_free, n_cells,
                                         max_visited_fraction)

    if gpu_block_size < spc:
        # Eviction is acceptable -- smaller BLOCK_SIZE with more blocks
        # is preferred over larger BLOCK_SIZE with reduced pool.
        # Only attempt to bump BLOCK_SIZE if we can keep a reasonable pool.
        target_bs = 1
        while target_bs < spc:
            target_bs *= 2
        available = max(vram_free - 1024**3, 256 * 1024**2)
        available = max(available - 512 * 1024**2, 128 * 1024**2)
        new_max_blocks = available // (target_bs * 10)
        new_frac = new_max_blocks / (n_cells * 1.5)
        if new_frac >= 0.05:
            gpu_block_size = target_bs
            max_visited_fraction = new_frac
            warnings.warn(
                f"Reduced max_visited_fraction to {new_frac:.3f} "
                f"to fit BLOCK_SIZE={target_bs} >= spc={spc} in VRAM")
        else:
            warnings.warn(
                f"BLOCK_SIZE={gpu_block_size} < spc={spc}: insufficient "
                f"VRAM for lossless exploration. Results may be suboptimal.")

    return gpu_block_size, max_visited_fraction


def _compute_v4_bucket_params(n_cells, n_dirs, vram_free):
    """Bucket queue sizing logic. Returns (items_per_bucket, max_segments, n_buckets, seg_size)."""
    n_buckets = _N_BUCKETS
    seg_size = _SEGMENT_SIZE
    # Items per bucket must scale with raster size. For small rasters
    # (synthetic tests), 65536 is more than enough. For real-world
    # rasters with millions of cells, the frontier in a single bucket
    # can far exceed 65K items, causing pool overflow and dropped work.
    #
    # Heuristic: frontier in a single bucket is roughly
    # sqrt(n_cells) * n_dirs for constrained routing (wavefront width
    # * states per cell). Scale items_per_bucket accordingly, but
    # cap by a VRAM budget (20% of free VRAM after pool reservation).
    frontier_estimate = int(np.sqrt(n_cells) * n_dirs * 2)
    vram_after_pool = max(vram_free - 1024**3, 256 * 1024**2)
    bucket_budget = int(vram_after_pool * 0.20)
    items_from_budget = bucket_budget // (n_buckets * 16)
    # Scale based on frontier estimate, clamped by VRAM budget
    items_per_bucket = max(
        _ITEMS_PER_BUCKET,
        min(frontier_estimate, items_from_budget, 2_000_000)
    )
    # Round down to multiple of segment size
    items_per_bucket = (items_per_bucket // seg_size) * seg_size
    items_per_bucket = max(items_per_bucket, _ITEMS_PER_BUCKET)
    max_segments = items_per_bucket // seg_size
    return items_per_bucket, max_segments, n_buckets, seg_size


def _upload_v4_gpu_data(raster, steps_arr, cost_factors, step_distances,
                         angle_valid_lut, angle_cost_lut, tower_terrain_costs,
                         tower_angle_costs, height_premiums,
                         intermediates_lut, n_intermediates):
    """Upload all data arrays to GPU. Returns dict of GPU arrays."""
    d_raster = cp.asarray(raster.astype(np.uint16))
    d_steps = cp.asarray(steps_arr.reshape(-1).astype(np.int8))
    d_cost_factors = cp.asarray(cost_factors.astype(np.float32))
    d_step_distances = cp.asarray(step_distances.astype(np.float32))
    d_angle_valid = cp.asarray(angle_valid_lut.reshape(-1).astype(np.uint8))
    d_angle_cost = cp.asarray(angle_cost_lut.reshape(-1).astype(np.float32))
    d_tower_terrain = cp.asarray(tower_terrain_costs.astype(np.float32))
    d_tower_angle = cp.asarray(
        tower_angle_costs.reshape(-1).astype(np.float32))
    d_height_premiums = cp.asarray(height_premiums.astype(np.float32))
    d_intermediates = cp.asarray(
        intermediates_lut.reshape(-1).astype(np.int16))
    d_n_intermediates = cp.asarray(n_intermediates.astype(np.int32))
    return {
        'raster': d_raster,
        'steps': d_steps,
        'cost_factors': d_cost_factors,
        'step_distances': d_step_distances,
        'angle_valid': d_angle_valid,
        'angle_cost': d_angle_cost,
        'tower_terrain': d_tower_terrain,
        'tower_angle': d_tower_angle,
        'height_premiums': d_height_premiums,
        'intermediates': d_intermediates,
        'n_intermediates': d_n_intermediates,
    }


def _allocate_v4_buffers(n_cells, gpu_block_size, max_sparse_blocks,
                          n_buckets, items_per_bucket, max_segments,
                          n_wtbs, max_tower_records):
    """Allocate all GPU buffers. Returns dict."""
    # Block-sparse storage
    total_entries = max_sparse_blocks * gpu_block_size
    # BlockEntry = 8 bytes (uint16 local_key, uint16 _pad, float32 dist)
    # Allocate as int32 pairs (2 int32 = 8 bytes per entry)
    d_pool = cp.zeros(total_entries * 2, dtype=cp.int32)
    d_span_pool = cp.zeros(total_entries, dtype=cp.float16)
    d_cell_to_block = cp.full(n_cells, -1, dtype=cp.int32)
    d_block_to_cell = cp.full(max_sparse_blocks, -1, dtype=cp.int32)
    d_n_allocated = cp.zeros(1, dtype=cp.int32)

    # Bucket queue: WorkItem = 16 bytes = 4 int32
    pool_total = n_buckets * items_per_bucket
    d_bucket_pool = cp.zeros(pool_total * 4, dtype=cp.int32)
    d_bucket_resv = cp.zeros(n_buckets, dtype=cp.int32)
    d_bucket_read = cp.zeros(n_buckets, dtype=cp.int32)
    d_bucket_gen = cp.zeros(n_buckets, dtype=cp.int32)
    d_bucket_wcc = cp.zeros(n_buckets * max_segments, dtype=cp.int32)

    # Assignment flags: AssignmentFlag = 16 bytes = 4 int32
    d_af = cp.zeros(n_wtbs * 4, dtype=cp.int32)

    # Tower records: TowerRecordV4 = 24 bytes = 6 int32
    d_tower_records = cp.zeros(max_tower_records * 6, dtype=cp.int32)

    # Control buffer
    d_control = cp.zeros(_CTL_V4_SIZE, dtype=cp.int32)

    # Best target distance = +inf (0x7F800000 as IEEE 754 bit pattern)
    d_best_target = cp.full(1, 0x7F800000, dtype=cp.int32)

    return {
        'pool': d_pool,
        'span_pool': d_span_pool,
        'cell_to_block': d_cell_to_block,
        'block_to_cell': d_block_to_cell,
        'n_allocated': d_n_allocated,
        'bucket_pool': d_bucket_pool,
        'bucket_resv': d_bucket_resv,
        'bucket_read': d_bucket_read,
        'bucket_gen': d_bucket_gen,
        'bucket_wcc': d_bucket_wcc,
        'af': d_af,
        'tower_records': d_tower_records,
        'control': d_control,
        'best_target': d_best_target,
        'total_entries': total_entries,
    }


def _find_v4_best_target(d_cell_to_block, d_pool, target_cell,
                          gpu_block_size, spc, n_span_bins, n_heights):
    """Scan target block for best state.

    Returns (best_dist, best_state) or warns and returns (inf, -1).
    """
    target_block_idx = int(d_cell_to_block[target_cell].item())
    if target_block_idx < 0:
        warnings.warn(
            "GPU constrained SSSP V4 found no path to target "
            "(target cell never visited)")
        return float('inf'), -1

    # Read target cell's block entries
    block_entry_host_dtype = np.dtype([
        ('local_key', np.uint16), ('_pad', np.uint16), ('dist', np.float32)
    ])
    block_base = target_block_idx * gpu_block_size
    target_raw = d_pool[block_base * 2:(block_base + gpu_block_size) * 2].get()
    target_entries_np = np.frombuffer(
        target_raw.tobytes(), dtype=block_entry_host_dtype)

    best_dist = float('inf')
    best_state = -1
    sh_val = n_span_bins * n_heights
    for entry in target_entries_np:
        lk = int(entry['local_key'])
        if lk == 0xFFFF:  # BLOCK_EMPTY
            continue
        d = float(entry['dist'])
        if d < best_dist:
            best_dist = d
            direction = lk // sh_val
            rem = lk % sh_val
            sb = rem // n_heights
            hc = rem % n_heights
            best_state = (target_cell * spc
                          + direction * sh_val
                          + sb * n_heights + hc)

    return best_dist, best_state


# ============================================================================
# Main entry point: persistent kernel V4 constrained SSSP
# ============================================================================

def constrained_sssp_raster_gpu_v4(
    raster: np.ndarray,
    source_row: int, source_col: int,
    target_row: int, target_col: int,
    steps: np.ndarray,
    angle_cost_lut: np.ndarray,
    angle_valid_lut: np.ndarray,
    step_distances: np.ndarray,
    tower_terrain_costs: np.ndarray,
    tower_angle_costs: np.ndarray,
    n_span_bins: int,
    span_bin_size: float,
    min_span: float,
    max_span: float,
    height_premiums: Optional[np.ndarray] = None,
    n_heights: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
    dem: Optional[np.ndarray] = None,
    obstacle_heights: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
    conductor_weight_per_m: float = 0.0,
    conductor_tension: float = 1.0,
    min_clearance: float = 0.0,
    max_gradient_pct: float = 100.0,
    gradient_scale: float = 2.0,
    tower_heights: Optional[np.ndarray] = None,
    area_offsets: Optional[np.ndarray] = None,
    area_offset_starts: Optional[np.ndarray] = None,
    area_offset_counts: Optional[np.ndarray] = None,
    threads_per_block: int = 256,
    margin: float = 1.00001,
    max_tower_records: int = _MAX_TOWER_RECORDS,
    max_visited_fraction: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find constrained shortest path on GPU with tower placement (V4 persistent).

    Uses a single persistent kernel launch with MTB/WTB delegation pattern
    and circular bucket queue for delta-stepping. No cooperative groups.

    Parameters:
        raster: uint16 cost raster (65535 = impassable)
        source_row, source_col: source cell coordinates
        target_row, target_col: target cell coordinates
        steps: int8 (n_dirs, 2) neighborhood steps
        angle_cost_lut: float32 (n_dirs, n_dirs) angle penalty costs
        angle_valid_lut: uint8 (n_dirs, n_dirs) angle validity
        step_distances: float32 (n_dirs,) physical distance per step (meters)
        tower_terrain_costs: float32 (65536,) tower cost by terrain value
        tower_angle_costs: float32 (n_dirs, n_dirs) tower type cost by angle
        n_span_bins: number of span discretization bins
        span_bin_size: physical distance per span bin (meters)
        min_span: minimum span between towers (meters)
        max_span: maximum span between towers (meters)
        height_premiums: float32 (n_heights,) cost premium per height class
        n_heights: number of tower height classes (default 1)
        exclude_mask: optional uint8 mask (1=traversable, 0=blocked)
        dem: optional float32 DEM array for clearance checking and gradient
        obstacle_heights: optional float32 array of obstacle heights (meters)
        cell_size: raster cell size in meters (default 1.0)
        conductor_weight_per_m: conductor weight per meter in N/m (default 0.0)
        conductor_tension: conductor tension in N (default 1.0)
        min_clearance: minimum clearance above ground+obstacles in meters
        max_gradient_pct: maximum allowed gradient percentage (default 100.0)
        gradient_scale: exponential gradient cost scaling factor (default 2.0)
        tower_heights: float32 (n_heights,) actual tower heights in meters
        area_offsets: optional int32 flat array of (dr, dc) pairs
        area_offset_starts: optional int32 (n_dirs*n_dirs,) start index per pair
        area_offset_counts: optional int32 (n_dirs*n_dirs,) count per pair
        threads_per_block: CUDA threads per block (default 256)
        margin: early termination margin (default 1.00001)
        max_tower_records: max tower records to allocate (default 2M)
        max_visited_fraction: expected fraction of cells visited (default 0.4)

    Returns:
        tuple: (path_cell_indices uint32[], tower_cell_indices uint32[],
                tower_heights float32[])

    Raises:
        RuntimeError: if CUDA GPU not available
        ValueError: if source or target is on forbidden cell
    """
    height_premiums, n_heights, tower_heights, angle_cost_lut = \
        _validate_v4_inputs(
            raster, source_row, source_col, target_row, target_col,
            height_premiums, n_heights, tower_heights, angle_cost_lut,
            tower_terrain_costs, tower_angle_costs)

    rows, cols = raster.shape
    n_cells = rows * cols
    n_dirs = len(steps)
    source_cell = source_row * cols + source_col
    target_cell = target_row * cols + target_col
    spc = n_dirs * n_span_bins * n_heights

    # Apply exclude mask
    if exclude_mask is not None:
        raster = raster.copy()
        raster[exclude_mask == 0] = 65535

    if area_offsets is not None:
        warnings.warn(
            "Exact tower area mode not yet supported in V4. "
            "Using uniform mode.")

    # ---- Prepare step lookup tables ----
    from pyorps.utils.traversal_gpu import prepare_step_lookup_tables
    steps_arr, intermediates_lut, n_intermediates, cost_factors = \
        prepare_step_lookup_tables(steps)
    max_inter_cols = intermediates_lut.shape[1]

    # ---- Compute BLOCK_SIZE ----
    gpu_block_size, max_visited_fraction = _compute_v4_block_size(
        spc, n_cells, max_visited_fraction)

    # ---- Compute max_blocks ----
    # Need headroom above n_cells because atomicCAS races in get_block()
    # can waste block indices. For small rasters, guarantee at least 2x
    # n_cells blocks so that every cell can be visited.
    max_sparse_blocks = int(n_cells * max_visited_fraction * 1.5)
    # Floor: at least 2x n_cells for small rasters (atomicCAS headroom),
    # also guard against very low max_visited_fraction
    max_sparse_blocks = max(max_sparse_blocks, n_dirs * 10,
                            min(n_cells * 2, 5000))

    # ---- Compute delta ----
    delta = _compute_constrained_delta(raster, cost_factors)

    # ---- Bucket queue sizing ----
    vram_free = int(cp.cuda.Device().mem_info[0])
    items_per_bucket, max_segments, n_buckets, seg_size = \
        _compute_v4_bucket_params(n_cells, n_dirs, vram_free)

    # ---- Block count: 1 MTB + N WTBs ----
    n_sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    n_blocks = min(n_sms * 2, 28)
    n_blocks = max(n_blocks, 2)  # at least 1 MTB + 1 WTB
    n_wtbs = n_blocks - 1

    # ---- Shared memory ----
    smem_bytes = _compute_smem_bytes(n_dirs, n_heights, max_inter_cols)

    # ---- Compile kernels ----
    defines = {
        "N_BUCKETS": n_buckets,
        "SEGMENT_SIZE": seg_size,
        "ITEMS_PER_BUCKET": items_per_bucket,
        "MAX_SEGMENTS_PER_BUCKET": max_segments,
        "MAX_INTER": max_inter_cols,
        "MAX_WTBS": n_wtbs,
        "MTB_STAGING_SIZE": _MTB_STAGING_SIZE,
    }

    init_source = _load_kernel_source("adds_init.cu",
                                      block_size=gpu_block_size)
    init_source = _inject_defines(init_source, defines)

    main_source = _load_kernel_source("constrained_adds.cu",
                                      block_size=gpu_block_size)
    main_source = _inject_defines(main_source, defines)

    init_pool_kernel = _get_v4_kernel(init_source, "adds_init_pool")
    init_source_kernel = _get_v4_kernel(init_source, "adds_init_source")
    main_kernel = _get_v4_kernel(main_source, "constrained_adds_main")

    # ---- Upload data to GPU ----
    gpu_data = _upload_v4_gpu_data(
        raster, steps_arr, cost_factors, step_distances,
        angle_valid_lut, angle_cost_lut, tower_terrain_costs,
        tower_angle_costs, height_premiums, intermediates_lut,
        n_intermediates)

    # ---- Allocate GPU buffers ----
    bufs = _allocate_v4_buffers(
        n_cells, gpu_block_size, max_sparse_blocks,
        n_buckets, items_per_bucket, max_segments,
        n_wtbs, max_tower_records)

    # ---- Initialize pool (set all entries to empty state) ----
    grid_init = (bufs['total_entries'] + 255) // 256
    init_pool_kernel(
        (grid_init,), (256,),
        (bufs['pool'], bufs['span_pool'], np.int32(bufs['total_entries'])),
    )
    cp.cuda.Stream.null.synchronize()

    # ---- Seed source states into block-sparse pool + bucket 0 ----
    init_source_kernel(
        (1,), (n_dirs,),
        (bufs['pool'], bufs['span_pool'],
         bufs['cell_to_block'], bufs['block_to_cell'], bufs['n_allocated'],
         np.int32(max_sparse_blocks),
         bufs['bucket_pool'], bufs['bucket_resv'], bufs['bucket_wcc'],
         np.int32(source_cell), np.int32(n_dirs),
         np.int32(n_span_bins), np.int32(n_heights)),
    )
    cp.cuda.Stream.null.synchronize()

    # ---- Max assignments per MTB sweep ----
    # Scale with problem size (n_cells * n_dirs) to avoid premature
    # termination on large real-world rasters. The kernel will usually
    # terminate earlier via target margin or double empty sweep.
    max_assignments = max(100_000, n_cells * n_dirs // 10)

    # ---- Launch persistent kernel ----
    main_kernel(
        (n_blocks,), (threads_per_block,),
        (
            # Raster
            gpu_data['raster'], np.int32(rows), np.int32(cols),
            # Block-sparse distance storage
            bufs['pool'], bufs['span_pool'],
            bufs['cell_to_block'], bufs['block_to_cell'], bufs['n_allocated'],
            np.int32(max_sparse_blocks),
            # Bucket queue
            bufs['bucket_pool'], bufs['bucket_resv'], bufs['bucket_read'],
            bufs['bucket_gen'], bufs['bucket_wcc'],
            np.int32(max_sparse_blocks),  # max_pool_blocks (overflow detect)
            # Assignment flags
            bufs['af'], np.int32(n_wtbs),
            # Tower records
            bufs['tower_records'], np.int32(max_tower_records),
            # Control buffer
            bufs['control'],
            # Best target distance + target cell
            bufs['best_target'], np.int32(target_cell),
            # Profile LUTs (loaded to shared memory by each block)
            gpu_data['steps'], gpu_data['cost_factors'],
            gpu_data['step_distances'],
            gpu_data['angle_valid'], gpu_data['angle_cost'],
            gpu_data['tower_terrain'], gpu_data['tower_angle'],
            gpu_data['height_premiums'],
            gpu_data['intermediates'], gpu_data['n_intermediates'],
            # Parameters
            np.int32(n_dirs), np.int32(n_span_bins), np.int32(n_heights),
            np.float32(min_span), np.float32(max_span),
            np.float32(span_bin_size),
            np.float32(delta), np.float32(margin),
            np.int32(max_assignments),
            np.int32(max_inter_cols),
        ),
        shared_mem=smem_bytes,
    )
    cp.cuda.Device().synchronize()

    # ---- Check kernel termination and overflow ----
    control_cpu = bufs['control'].get()

    done_flag = int(control_cpu[_CTL_V4_DONE])
    if done_flag != 1:
        warnings.warn(
            f"V4 kernel did not terminate cleanly (CTL_V4_DONE={done_flag}). "
            f"Results may be incomplete.")

    block_overflow = int(control_cpu[_CTL_V4_BLOCK_OVERFLOW])
    pool_overflow = int(control_cpu[_CTL_V4_POOL_OVERFLOW])
    if block_overflow > 0:
        warnings.warn(
            f"Block-sparse pool exhausted: {block_overflow} states dropped. "
            f"Results may be suboptimal. Increase max_visited_fraction.")
    if pool_overflow > 0:
        warnings.warn(
            f"Bucket pool overflow: {pool_overflow} items dropped.")

    # ---- Find best target state ----
    empty_result = (np.empty(0, dtype=np.uint32),
                    np.empty(0, dtype=np.uint32),
                    np.empty(0, dtype=np.float32))

    best_dist, best_state = _find_v4_best_target(
        bufs['cell_to_block'], bufs['pool'], target_cell,
        gpu_block_size, spc, n_span_bins, n_heights)

    if best_dist >= 1e29 or best_state == -1:
        if best_state == -1 and best_dist != float('inf'):
            # _find_v4_best_target already warned for target_block_idx < 0
            warnings.warn(
                "GPU constrained SSSP V4 found no path to target")
        return empty_result

    # ---- Download tower records ----
    n_tower_records = min(
        int(control_cpu[_CTL_V4_TOWER_COUNT]), max_tower_records)

    tower_record_dtype = np.dtype([
        ('state', np.int64),
        ('pred_state', np.int64),
        ('span_dist', np.float16),
        ('tower_height', np.float16),
        ('tower_cost', np.float32),
    ])
    if n_tower_records > 0:
        raw_bytes = bufs['tower_records'][:n_tower_records * 6].get()
        tower_records_cpu = np.frombuffer(
            raw_bytes.tobytes(), dtype=tower_record_dtype)
    else:
        tower_records_cpu = np.zeros(0, dtype=tower_record_dtype)

    # ---- Download block data for reconstruction ----
    block_entry_host_dtype = np.dtype([
        ('local_key', np.uint16), ('_pad', np.uint16), ('dist', np.float32)
    ])
    n_alloc_final = int(bufs['n_allocated'].item())
    download_size = n_alloc_final * gpu_block_size
    pool_raw = bufs['pool'][:download_size * 2].get()
    pool_np = np.frombuffer(pool_raw.tobytes(), dtype=block_entry_host_dtype)
    c2b_cpu = bufs['cell_to_block'].get()

    dist_cpu = _DynamicBlockDistProxy(
        pool_np, c2b_cpu, gpu_block_size, spc, n_span_bins, n_heights)

    # ---- Reconstruct path ----
    result = _reconstruct_from_tower_records(
        tower_records_cpu, n_tower_records, best_state,
        source_cell, spc, n_span_bins, n_heights, n_dirs,
        cols, steps, dist_cpu=dist_cpu,
        min_span=min_span, step_distances=step_distances)

    return result
