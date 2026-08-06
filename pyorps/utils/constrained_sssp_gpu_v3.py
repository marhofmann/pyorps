"""GPU host-driven constrained pathfinding with dynamic block-sparse storage.

V3 of the GPU constrained SSSP: host-driven delta-stepping with per-launch
kernels (no cooperative groups required). Uses dynamic block allocation where
blocks are only allocated for visited cells, dramatically reducing memory
compared to V2's per-cell fixed block or dense storage.

Algorithm:
    Host-driven delta-stepping loop. Each bucket processes:
    - Light phase: relax edges with cost <= delta (iterate until frontier empty)
    - Heavy phase: relax edges with cost > delta (once per bucket)
    - Bucket advance: classify pending states into near/far queues
    - Full scan fallback: when no pending states, scan all allocated blocks
      to find the next non-empty bucket.

    Tower placement and clearance checking are identical to V2 (warp-
    cooperative protocol with __ballot_sync / __shfl_sync).

    Dynamic block allocation:
    - Blocks are allocated lazily via atomicCAS on cell_to_block[cell].
    - BLOCK_SIZE is a power of 2 chosen at compile time based on spc and VRAM.
    - Within-block storage uses open-addressing hash (multiplicative hash)
      with linear probing.
    - Eviction policy: when block is full, replace worst (highest dist) entry.

    Path reconstruction walks the tower record chain backward (no d_pred),
    identical to V2.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Tuple

try:
    import cupy as cp
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False


# ============================================================================
# State encoding helpers (Python-side) — identical to V2
# ============================================================================

def pack_state(cell, direction, span_bin, height_class,
               spc, n_span_bins, n_heights):
    """Pack (cell, dir, span_bin, height_class) into int64 state."""
    return (cell * spc
            + direction * n_span_bins * n_heights
            + span_bin * n_heights
            + height_class)


def unpack_state(state, spc, n_span_bins, n_heights):
    """Unpack int64 state into (cell, dir, span_bin, height_class)."""
    cell = state // spc
    rem = state % spc
    direction = rem // (n_span_bins * n_heights)
    rem = rem % (n_span_bins * n_heights)
    span_bin = rem // n_heights
    height_class = rem % n_heights
    return cell, direction, span_bin, height_class


# ============================================================================
# V3 Control buffer indices (must match CUDA #defines in dynamic_blocks.cuh)
# ============================================================================

_V3_CTL_OUTPUT = 0
_V3_CTL_SETTLED = 1
_V3_CTL_PENDING = 2
_V3_CTL_NEAR = 3
_V3_CTL_FAR = 4
_V3_CTL_TOWER = 5
_V3_CTL_MIN_DIST = 6
_V3_CTL_OVERFLOW = 7
_V3_CTL_SIZE = 8


# ============================================================================
# Memory budget helpers
# ============================================================================

def compute_memory_budget_v3(n_cells, block_size, max_visited_fraction=1.0):
    """Estimate GPU memory needed in bytes for V3 dynamic block-sparse mode.

    Parameters:
        n_cells: total raster cells (rows * cols)
        block_size: BLOCK_SIZE (power of 2, entries per block)
        max_visited_fraction: fraction of cells expected to be visited

    Returns:
        Estimated bytes of GPU memory needed.
    """
    max_blocks = int(n_cells * max_visited_fraction * 1.5)
    # Pool: BlockEntry (8 bytes) + span (2 bytes) = 10 bytes per slot
    pool_bytes = max_blocks * block_size * 10
    # cell_to_block: int32 per cell
    c2b_bytes = n_cells * 4
    # block_to_cell: int32 per block
    b2c_bytes = max_blocks * 4
    # Queues: 4 queues of int64, buf_size ~ n_cells * 4
    buf_size = max(n_cells * 4, 1 << 20)
    queue_bytes = 4 * buf_size * 8
    # Tower records: 2M * 24 bytes
    tower_bytes = 2_000_000 * 24
    # Control buffer
    control_bytes = _V3_CTL_SIZE * 4

    return pool_bytes + c2b_bytes + b2c_bytes + queue_bytes + tower_bytes + control_bytes


def _compute_block_size(spc, vram_free_bytes, n_cells, max_visited_fraction=1.0):
    """Select optimal BLOCK_SIZE (power of 2) from spc and available VRAM."""
    max_blocks = int(n_cells * max_visited_fraction * 1.5)
    # Target: next power of 2 >= spc (no eviction)
    target_bs = 1
    while target_bs < spc:
        target_bs *= 2
    # Reserve 1 GB for input + queues + tower records
    available = max(vram_free_bytes - 1024**3, 256 * 1024**2)
    # Shrink until it fits: max_blocks * BLOCK_SIZE * 10 bytes per entry
    while target_bs > 32 and target_bs * max_blocks * 10 > available:
        target_bs //= 2
    return max(32, target_bs)


# ============================================================================
# CUDA kernel source loading from kernels/ directory
# ============================================================================

from pathlib import Path
import re

_KERNEL_DIR = Path(__file__).parent / "kernels"


def _load_v3_kernel_source(main_file: str, block_size: int = 64) -> str:
    """Load CUDA source with #include resolution from kernels/ directory.

    CuPy's RawKernel does not support #include natively, so we resolve
    local #include "xxx.cuh" directives in Python before passing the
    concatenated source to CuPy. System includes (angle-bracket) are
    left intact for nvrtc.

    block_size: injected as #define BLOCK_SIZE before the source for
                dynamic block mode. Must be a power of 2.
    """
    main_path = _KERNEL_DIR / main_file
    source = main_path.read_text(encoding='utf-8')
    # Inject BLOCK_SIZE before any includes
    source = f"#define BLOCK_SIZE {block_size}\n" + source

    # Track already-included files to respect #pragma once
    included = set()

    def resolve_include(match):
        inc_file = match.group(1)
        if inc_file in included:
            return ''  # already included (#pragma once)
        inc_path = _KERNEL_DIR / inc_file
        if inc_path.exists():
            included.add(inc_file)
            content = inc_path.read_text(encoding='utf-8')
            # Remove #pragma once from included content
            content = re.sub(r'#pragma\s+once\s*\n?', '', content)
            # Recursively resolve nested includes
            content = re.sub(
                r'#include\s+"([^"]+)"', resolve_include, content)
            return content
        return match.group(0)  # leave unresolved includes unchanged

    # Only resolve local includes (quoted, not angle-bracket)
    source = re.sub(r'#include\s+"([^"]+)"', resolve_include, source)
    return source


# ============================================================================
# Kernel cache + compilation
# ============================================================================

_v3_kernel_cache = {}


def _get_v3_kernel(name, block_size=64):
    """Compile or retrieve cached CuPy RawKernel for V3.

    Loads CUDA source from the kernels/ directory, resolves local
    #include directives, and compiles via CuPy. V3 kernels do NOT
    use cooperative groups — they are launched individually by the host.
    """
    key = (name, block_size)
    if key not in _v3_kernel_cache:
        _ensure_cuda_path()
        # Map kernel name to source file
        if name == "relax_constrained_v3":
            source = _load_v3_kernel_source("relax_constrained_v3.cu",
                                            block_size=block_size)
        elif name in ("init_pool_v3", "init_source_v3"):
            source = _load_v3_kernel_source("init_dynamic.cu",
                                            block_size=block_size)
        elif name == "classify_bucket":
            source = _load_v3_kernel_source("classify_bucket.cu",
                                            block_size=block_size)
        elif name in ("scan_min_dist", "extract_bucket"):
            source = _load_v3_kernel_source("scan_min.cu",
                                            block_size=block_size)
        else:
            raise ValueError(f"Unknown V3 kernel: {name}")
        _v3_kernel_cache[key] = cp.RawKernel(
            source, name, options=("--std=c++17",))
    return _v3_kernel_cache[key]


# ============================================================================
# CUDA path setup (copied from V2)
# ============================================================================

def _ensure_cuda_path():
    """Ensure CuPy can find cudadevrt for cooperative groups linking."""
    import os
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
# Shared memory computation (adapted from V2)
# ============================================================================

def _compute_v3_smem(n_steps, n_heights, max_inter_cols):
    """Compute shared memory bytes for the V3 relax kernel."""
    steps_bytes = n_steps * 2
    steps_padded = (steps_bytes + 3) & ~3
    n_inter_bytes = n_steps * 4        # int32
    cost_factors_bytes = n_steps * 4   # float32
    step_dist_bytes = n_steps * 4      # float32
    n2 = n_steps * n_steps
    angle_cost_bytes = n2 * 4          # float32
    angle_valid_bytes = n2             # uint8
    av_padded = (angle_valid_bytes + 3) & ~3
    tower_angle_bytes = n2 * 4         # float32
    height_bytes = n_heights * 4       # float32 (premiums)
    tower_height_bytes = n_heights * 4 # float32 (actual heights)
    lut_bytes = n_steps * max_inter_cols * 2  # int8
    return (steps_padded + n_inter_bytes + cost_factors_bytes
            + step_dist_bytes + angle_cost_bytes + av_padded
            + tower_angle_bytes + height_bytes + tower_height_bytes
            + lut_bytes)


# ============================================================================
# Delta computation (copied from V2)
# ============================================================================

def _compute_constrained_delta(raster, cost_factors,
                                tower_terrain_costs=None):
    """Compute delta for constrained delta-stepping.

    Picks delta so that non-tower terrain edges are "light" (processed
    within a bucket) while tower-placement edges are "heavy" (deferred
    to future buckets).  This gives proper delta-stepping behavior
    instead of degenerating to Dijkstra (1 step per bucket).

    Formula: 2 * max_raster_val * max_cost_factor * n_dirs
    This upper-bounds the worst-case non-tower edge cost (terrain +
    angle penalty for any direction pair) so all such edges fit in one
    bucket.  Tower costs (~1000+) are always much larger → heavy.

    Delta-stepping is correct for any positive delta; larger values
    reduce bucket count at the cost of some redundant relaxations
    within a bucket (handled by atomicMin).
    """
    valid = raster.ravel()
    valid = valid[valid < 65535]
    if len(valid) == 0:
        return 100.0
    max_cost = float(max(1, int(valid.max())))
    max_cf = float(cost_factors.max())
    n_dirs = len(cost_factors)
    # Upper bound on non-tower edge cost: (src + dst + intermediates)
    # * cost_factor + angle_penalty.  Using 2*max*max_cf*n_dirs is a
    # safe overestimate that keeps ALL terrain edges light.
    return max(1.0, 2.0 * max_cost * max_cf * n_dirs)


# ============================================================================
# Dynamic block distance proxy for path reconstruction
# ============================================================================

class _DynamicBlockDistProxy:
    """Dynamic block-sparse proxy that mimics array indexing for dist_cpu[state].

    Used in V3 dynamic block-sparse mode for path reconstruction. Blocks are
    indexed via cell_to_block mapping (not per-cell fixed offset like V2).
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
# Path reconstruction from tower records (CPU-side) — copied from V2
# ============================================================================

def _build_tower_record_index(tower_records_cpu, n_tower_records,
                              spc, n_span_bins, n_heights):
    """Build (cell, dir) -> list of record indices for fast lookup.

    Returns the cell_dir_to_records dict.
    """
    cell_dir_to_records = {}
    for i in range(n_tower_records):
        rec_state = int(tower_records_cpu['state'][i])
        rc, rd, _, _ = unpack_state(rec_state, spc, n_span_bins, n_heights)
        key = (rc, rd)
        if key not in cell_dir_to_records:
            cell_dir_to_records[key] = []
        cell_dir_to_records[key].append(i)
    return cell_dir_to_records


def _find_ancestor_record(cell_dir_to_records, tower_records_cpu,
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


def _walk_backward_tower_chain(tower_records_cpu, cell_dir_to_records,
                               target_cell, target_dir, source_cell,
                               cols, steps_np,
                               spc, n_span_bins, n_heights,
                               dist_cpu, best_state):
    """Walk backward from best_state building the tower_chain list.

    Returns tower_chain (list of dicts).
    """
    tower_chain = []
    current_cell = target_cell
    current_dir = target_dir
    current_dist = float(dist_cpu[best_state]) if dist_cpu is not None else 1e30

    for _ in range(1000):  # safety limit
        if current_cell == source_cell:
            break

        # Walk backward along current_dir, looking for a tower record
        # that is a proper ancestor (lower distance).
        walk_cell = current_cell
        found_tower = False
        for step in range(1000):
            # Step backward
            dr = int(steps_np[current_dir, 0])
            dc = int(steps_np[current_dir, 1])
            r = walk_cell // cols
            c = walk_cell % cols
            nr, nc = r - dr, c - dc
            if nr < 0 or nc < 0 or nc >= cols:
                break
            walk_cell = nr * cols + nc

            if walk_cell == source_cell:
                break

            # Check for a tower record at this cell
            rec_idx = _find_ancestor_record(
                cell_dir_to_records, tower_records_cpu,
                walk_cell, current_dir, current_dist, dist_cpu)
            if rec_idx is not None:
                pred_state = int(tower_records_cpu['pred_state'][rec_idx])
                height = float(tower_records_cpu['tower_height'][rec_idx])
                pred_cell, pred_dir, _, _ = unpack_state(
                    pred_state, spc, n_span_bins, n_heights)

                tower_chain.append({
                    'tower_cell': pred_cell,
                    'tower_dir': pred_dir,
                    'post_cell': walk_cell,
                    'post_dir': current_dir,
                    'height': height,
                })

                # Jump to pred_state for next segment
                current_cell = pred_cell
                current_dir = pred_dir
                current_dist = (float(dist_cpu[pred_state])
                                if dist_cpu is not None else current_dist)
                found_tower = True
                break

        if not found_tower:
            break  # no more towers found, source must be reachable by walk

    tower_chain.reverse()
    return tower_chain


def _assemble_path_from_tower_chain(tower_chain, target_cell, target_dir,
                                    source_cell, cols, steps_np, n_dirs):
    """Build path from tower chain.

    Returns (path_arr, tower_arr, height_arr).
    """
    # Build full path by direction-walking between waypoints
    waypoints = []

    if tower_chain:
        # Source to first tower
        first = tower_chain[0]
        seg = _direction_walk_backward(
            first['tower_cell'], first['tower_dir'], source_cell,
            cols, steps_np, n_dirs)
        waypoints.extend(seg)

        # Between consecutive towers: walk from post_cell to next tower_cell
        for i, tc in enumerate(tower_chain):
            if i + 1 < len(tower_chain):
                next_tc = tower_chain[i + 1]
                seg = _direction_walk_backward(
                    next_tc['tower_cell'], next_tc['tower_dir'],
                    tc['post_cell'], cols, steps_np, n_dirs)
            else:
                # Last tower to target
                seg = _direction_walk_backward(
                    target_cell, target_dir, tc['post_cell'],
                    cols, steps_np, n_dirs)

            if seg and waypoints and seg[0] == waypoints[-1]:
                seg = seg[1:]
            waypoints.extend(seg)
    else:
        # No towers on path, direct walk
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
):
    """Reconstruct path and tower locations from TowerRecord chain.

    Tower records store (state_after_move, pred_state_before_tower).
    The state_after_move has a reset span_bin (small), while the best target
    state may have a high span_bin from accumulation after the last tower.

    Reconstruction strategy:
    1. Walk backward from best_state along its direction to find the first
       cell that has a tower record with matching (cell, dir) and whose
       state distance is less than best_state distance (confirming it is
       an ancestor).
    2. Use that record's pred_state to jump to the pre-tower state.
    3. From pred_state, repeat: walk backward along pred_dir to find the
       previous tower record.
    4. Continue until reaching source_cell.

    Parameters:
        tower_records_cpu: structured numpy array of TowerRecords
        n_tower_records: number of valid records
        best_state: target state with minimum distance
        source_cell: source cell index
        spc: states per cell
        n_span_bins: number of span bins
        n_heights: number of height classes
        n_dirs: number of directions
        cols: number of columns in raster
        steps_np: (n_dirs, 2) step array
        dist_cpu: optional float32 distance array for chain validation

    Returns:
        (path_indices, tower_indices, tower_heights) as numpy arrays
    """
    target_cell, target_dir, _, _ = unpack_state(
        best_state, spc, n_span_bins, n_heights)

    if n_tower_records == 0:
        # No towers placed -- direct path from source to target
        if target_cell == source_cell:
            return (np.array([target_cell], dtype=np.uint32),
                    np.empty(0, dtype=np.uint32),
                    np.empty(0, dtype=np.float32))
        path = _direction_walk_backward(
            target_cell, target_dir, source_cell, cols, steps_np, n_dirs)
        return (np.array(path, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))

    cell_dir_to_records = _build_tower_record_index(
        tower_records_cpu, n_tower_records, spc, n_span_bins, n_heights)

    tower_chain = _walk_backward_tower_chain(
        tower_records_cpu, cell_dir_to_records,
        target_cell, target_dir, source_cell,
        cols, steps_np,
        spc, n_span_bins, n_heights,
        dist_cpu, best_state)

    return _assemble_path_from_tower_chain(
        tower_chain, target_cell, target_dir,
        source_cell, cols, steps_np, n_dirs)


def _direction_walk_backward(target_cell, target_dir, source_cell,
                              cols, steps_np, n_dirs, max_steps=10000):
    """Walk backward from target_cell using inverse of target_dir to reach source_cell.

    This traces the path cell-by-cell: starting from target_cell, step backward
    (opposite direction) until reaching source_cell or hitting max_steps.

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


# ============================================================================
# V3 availability check
# ============================================================================

_v3_available = None


def _check_v3_available():
    """Check if V3 kernels compile on this GPU."""
    global _v3_available
    if _v3_available is not None:
        return _v3_available
    if not GPU_AVAILABLE:
        _v3_available = False
        return False
    try:
        kernel = _get_v3_kernel("init_pool_v3")
        _ = kernel.kernel  # force compilation
        _v3_available = True
    except Exception:
        _v3_available = False
    return _v3_available


# ============================================================================
# Helper functions for constrained_sssp_raster_gpu_v3
# ============================================================================

def _validate_v3_inputs(raster, source_row, source_col, target_row, target_col,
                        height_premiums, n_heights, tower_heights,
                        angle_cost_lut, tower_terrain_costs, tower_angle_costs):
    """GPU check, cell validation, defaults, angle_cost cleanup, cost validation.

    Returns (source_cell, target_cell, height_premiums, n_heights, tower_heights,
             angle_cost_lut).
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU not available. Install cupy: pip install cupy-cuda12x")

    rows, cols = raster.shape
    max_cost = int(np.iinfo(np.uint16).max)

    # Validate source and target
    source_cell = source_row * cols + source_col
    target_cell = target_row * cols + target_col
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

    # Clean angle_cost_lut: replace inf with 0 (invalid transitions have
    # angle_valid=0 so they are never used; inf would break atomicMin bit-cast)
    angle_cost_lut = angle_cost_lut.copy().astype(np.float32)
    angle_cost_lut[~np.isfinite(angle_cost_lut)] = 0.0

    # Validate non-negative costs
    if np.any(angle_cost_lut < 0):
        raise ValueError("angle_cost_lut must have non-negative values")
    if np.any(tower_terrain_costs < 0):
        raise ValueError("tower_terrain_costs must have non-negative values")
    if np.any(tower_angle_costs < 0):
        raise ValueError("tower_angle_costs must have non-negative values")

    return (source_cell, target_cell, height_premiums, n_heights,
            tower_heights, angle_cost_lut)


def _compute_v3_block_size(spc, n_cells, max_visited_fraction):
    """VRAM check, block size computation, adjustment if gpu_block_size < spc.

    Returns (gpu_block_size, max_visited_fraction).
    """
    vram_free = int(cp.cuda.Device().mem_info[0])
    gpu_block_size = _compute_block_size(spc, vram_free, n_cells,
                                         max_visited_fraction)

    if gpu_block_size < spc:
        # Need BLOCK_SIZE >= spc. Find the right power of 2.
        target_bs = 1
        while target_bs < spc:
            target_bs *= 2
        # Reduce max_visited_fraction to fit target_bs in VRAM
        available = max(vram_free - 1024**3, 256 * 1024**2)
        # Reserve space for queues + towers + input data (~500MB)
        available = max(available - 512 * 1024**2, 128 * 1024**2)
        new_max_blocks = available // (target_bs * 10)
        new_frac = new_max_blocks / (n_cells * 1.5)
        if new_frac >= 0.01:  # at least 1% of cells
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


def _upload_v3_gpu_data(raster, steps_arr, cost_factors,
                        intermediates_lut, n_intermediates,
                        angle_cost_lut, angle_valid_lut, step_distances,
                        tower_terrain_costs, tower_angle_costs,
                        height_premiums, tower_heights,
                        dem, obstacle_heights,
                        area_offsets, area_offset_starts, area_offset_counts):
    """Upload all arrays to GPU, returns a dict of GPU arrays."""
    gpu = {}
    gpu['d_raster'] = cp.asarray(raster.astype(np.uint16))
    gpu['d_steps'] = cp.asarray(steps_arr.astype(np.int8))
    gpu['d_cost_factors'] = cp.asarray(cost_factors.astype(np.float32))
    gpu['d_inter_lut'] = cp.asarray(intermediates_lut.astype(np.int8))
    gpu['d_n_inter'] = cp.asarray(n_intermediates.astype(np.int32))
    gpu['d_angle_cost'] = cp.asarray(
        angle_cost_lut.reshape(-1).astype(np.float32))
    gpu['d_angle_valid'] = cp.asarray(
        angle_valid_lut.reshape(-1).astype(np.uint8))
    gpu['d_step_dist'] = cp.asarray(step_distances.astype(np.float32))
    gpu['d_tower_terrain'] = cp.asarray(
        tower_terrain_costs.astype(np.float32))
    gpu['d_tower_angle'] = cp.asarray(
        tower_angle_costs.reshape(-1).astype(np.float32))
    gpu['d_height_premiums'] = cp.asarray(height_premiums.astype(np.float32))
    gpu['d_tower_heights'] = cp.asarray(tower_heights.astype(np.float32))

    # Upload DEM and obstacle arrays (or create NULL-equivalent empty arrays)
    if dem is not None:
        gpu['d_dem'] = cp.asarray(dem.astype(np.float32).ravel())
    else:
        gpu['d_dem'] = cp.empty(0, dtype=cp.float32)
    if obstacle_heights is not None:
        gpu['d_obstacle'] = cp.asarray(
            obstacle_heights.astype(np.float32).ravel())
    else:
        gpu['d_obstacle'] = cp.empty(0, dtype=cp.float32)

    # Upload area offset arrays (or prepare NULL pointers)
    if area_offsets is not None:
        gpu['d_area_offsets'] = cp.asarray(
            area_offsets.astype(np.int32).ravel())
        gpu['d_area_starts'] = cp.asarray(
            area_offset_starts.astype(np.int32).ravel())
        gpu['d_area_counts'] = cp.asarray(
            area_offset_counts.astype(np.int32).ravel())
    else:
        gpu['d_area_offsets'] = cp.empty(0, dtype=cp.int32)
        gpu['d_area_starts'] = cp.empty(0, dtype=cp.int32)
        gpu['d_area_counts'] = cp.empty(0, dtype=cp.int32)

    return gpu


def _allocate_v3_pools(n_cells, gpu_block_size, max_blocks, buf_size,
                       max_tower_records, spc):
    """Allocate all GPU pool/queue/control arrays. Returns a dict."""
    block_entry_dtype = cp.dtype([
        ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)
    ])

    pools = {}
    pool_size = max_blocks * gpu_block_size

    # Pool arrays
    pools['d_pool'] = cp.empty(pool_size, dtype=block_entry_dtype)
    pools['d_span_pool'] = cp.zeros(pool_size, dtype=cp.float16)

    # Cell-to-block and block-to-cell mappings
    pools['d_cell_to_block'] = cp.full(n_cells, -1, dtype=cp.int32)
    pools['d_block_to_cell'] = cp.full(max_blocks, -1, dtype=cp.int32)
    pools['d_n_allocated'] = cp.zeros(1, dtype=cp.int32)

    # Queues (int64)
    pools['d_queue_a'] = cp.empty(buf_size, dtype=cp.int64)
    pools['d_queue_b'] = cp.empty(buf_size, dtype=cp.int64)
    pools['d_settled'] = cp.empty(buf_size, dtype=cp.int64)
    pools['d_pending'] = cp.empty(buf_size, dtype=cp.int64)

    # Control buffer
    pools['d_control'] = cp.zeros(_V3_CTL_SIZE, dtype=cp.int32)

    # Tower records buffer
    tower_record_bytes = max_tower_records * 24  # sizeof(TowerRecord) = 24
    pools['d_tower_records'] = cp.zeros(tower_record_bytes, dtype=cp.uint8)

    pools['pool_size'] = pool_size
    return pools


def _init_v3_source_states(source_cell, n_dirs, n_heights, height_premiums,
                           spc, n_span_bins, gpu_block_size, max_blocks,
                           d_pool, d_span_pool, d_cell_to_block,
                           d_block_to_cell, d_n_allocated, d_queue_a):
    """Compute and initialize source states. Returns (source_state_ids, n_source)."""
    source_states = []
    for d in range(n_dirs):
        for hc in range(n_heights):
            st = pack_state(source_cell, d, 0, hc, spc, n_span_bins, n_heights)
            source_states.append((st, float(height_premiums[hc])))
    source_state_ids = np.array([s for s, _ in source_states], dtype=np.int64)
    source_init_dists = np.array([p for _, p in source_states], dtype=np.float32)
    n_source = len(source_state_ids)

    # Initialize source states
    d_src_states = cp.asarray(source_state_ids)
    d_src_dists = cp.asarray(source_init_dists)
    init_source_kernel = _get_v3_kernel("init_source_v3",
                                        block_size=gpu_block_size)
    tpb_src = min(256, n_source)
    bpg_src = (n_source + tpb_src - 1) // tpb_src
    init_source_kernel(
        (bpg_src,), (tpb_src,),
        (d_pool, d_span_pool,
         d_cell_to_block, d_block_to_cell, d_n_allocated,
         d_src_states, d_src_dists,
         np.int32(n_source), np.int32(spc), np.int32(n_span_bins),
         np.int32(n_heights), np.int32(max_blocks)))
    cp.cuda.Stream.null.synchronize()

    # Seed initial frontier
    d_queue_a[:n_source] = cp.asarray(source_state_ids)

    return source_state_ids, n_source


def _check_v3_early_termination(d_cell_to_block, d_pool, target_cell,
                                gpu_block_size, best_known, bucket, delta,
                                margin):
    """Check target block for best dist.

    Returns (best_known, should_break).
    """
    target_block = int(d_cell_to_block[target_cell].item())
    if target_block >= 0:
        # Read target block entries, find best dist
        block_base = target_block * gpu_block_size
        target_entries = d_pool[block_base:block_base + gpu_block_size].get()
        block_entry_host_dtype = np.dtype([
            ('local_key', np.uint16), ('_pad', np.uint16),
            ('dist', np.float32)
        ])
        target_entries_np = np.frombuffer(
            target_entries.tobytes(), dtype=block_entry_host_dtype)
        for entry in target_entries_np:
            lk = int(entry['local_key'])
            if lk == 0xFFFF:
                continue
            d = float(entry['dist'])
            if d < best_known:
                best_known = d
        if best_known < 1e29:
            # Check if current bucket is past best_known * margin
            if (bucket + 1) * delta > best_known * margin:
                return best_known, True

    return best_known, False


def _compute_f32_bucket(dist_val, delta):
    """Compute bucket index in float32 to match CUDA kernel arithmetic.

    Python float64 and CUDA float32 disagree at bucket boundaries, so we
    perform the computation in float32 and verify the state actually falls
    within [bucket*delta, (bucket+1)*delta).

    Returns the bucket index.
    """
    delta_f32 = np.float32(delta)
    dist_f32 = np.float32(dist_val)
    bucket = int(dist_f32 / delta_f32)
    # Verify: dist must be in [bucket*delta, (bucket+1)*delta) in f32
    while dist_f32 >= np.float32(bucket + 1) * delta_f32:
        bucket += 1
    while bucket > 0 and dist_f32 < np.float32(bucket) * delta_f32:
        bucket -= 1
    return bucket


def _v3_classify_pending(d_pending, pending_count, bucket, delta, buf_size,
                         d_cell_to_block, d_pool, spc, n_span_bins, n_heights,
                         frontier, d_settled, d_control, tpb,
                         classify_kernel, _DEBUG, iteration):
    """Classify pending states into near/far. Returns (frontier_count, should_continue, new_bucket_offset, preserve_pending)."""
    # Initialize MIN_DIST to +inf for far-state tracking
    inf_bits = int(np.float32(1e30).view(np.int32))
    d_control[_V3_CTL_NEAR] = 0
    d_control[_V3_CTL_FAR] = 0
    d_control[_V3_CTL_MIN_DIST] = inf_bits
    grid = (pending_count + tpb - 1) // tpb
    classify_kernel(
        (grid,), (tpb,),
        (d_pending, np.int32(pending_count), np.int32(bucket),
         np.float32(delta), np.int32(buf_size),
         d_cell_to_block, d_pool,
         np.int64(spc), np.int32(n_span_bins), np.int32(n_heights),
         frontier, d_settled,  # near->frontier, far->settled as temp
         d_control))
    cp.cuda.Stream.null.synchronize()

    frontier_count = int(d_control[_V3_CTL_NEAR].get())
    far_count = int(d_control[_V3_CTL_FAR].get())
    if _DEBUG and iteration <= 200:
        print(f"    CLASSIFY: near={frontier_count} far={far_count}")

    if frontier_count > 0:
        # Copy far back to pending
        if far_count > 0:
            d_pending[:far_count] = d_settled[:far_count]
            d_control[_V3_CTL_PENDING] = far_count
        else:
            d_control[_V3_CTL_PENDING] = 0
        # _reset_pending = False if far_count > 0, True otherwise
        return frontier_count, True, 0, (far_count > 0)

    if far_count > 0:
        d_pending[:far_count] = d_settled[:far_count]
        d_control[_V3_CTL_PENDING] = far_count
        # Skip directly to the bucket containing the closest
        # far state instead of advancing one bucket at a time.
        min_far_bits = int(d_control[_V3_CTL_MIN_DIST].get())
        min_far_dist = float(np.array(
            [min_far_bits], dtype=np.int32).view(np.float32)[0])
        bucket_offset = 0
        if min_far_dist < 1e29:
            new_bucket = _compute_f32_bucket(min_far_dist, delta)
            if _DEBUG and iteration <= 200 and new_bucket > bucket:
                print(f"    SKIP: bucket {bucket} -> {new_bucket} "
                      f"(min_far={min_far_dist:.1f})")
            # Set bucket so the next iteration's bucket += 1 lands
            # on new_bucket (the loop will advance again at top)
            bucket_offset = new_bucket - bucket
        return 0, True, bucket_offset, True

    return 0, False, 0, False


def _v3_scan_min_dist(d_n_allocated, d_pool, d_block_to_cell, d_control,
                      gpu_block_size, bucket, delta, n_cells,
                      scan_kernel, _DEBUG, iteration):
    """Run scan_min_dist kernel and validate results.

    Returns (n_alloc, grid_scan, min_dist, new_bucket) or None if should break.
    """
    n_alloc = int(d_n_allocated.item())
    if _DEBUG and iteration <= 200:
        print(f"  SCAN: n_alloc={n_alloc}")
    if n_alloc == 0:
        if _DEBUG: print("  EXIT: no blocks allocated")
        return None

    # Set MIN_DIST to +inf (as int32 bit pattern)
    inf_bits = int(np.float32(1e30).view(np.int32))
    d_control[_V3_CTL_MIN_DIST] = inf_bits
    grid_scan = min((n_alloc * gpu_block_size + 255) // 256, 1024)
    scan_kernel(
        (grid_scan,), (256,),
        (d_pool, d_block_to_cell, np.int32(n_alloc),
         np.float32(bucket * delta), d_control))
    cp.cuda.Stream.null.synchronize()

    min_dist_bits = int(d_control[_V3_CTL_MIN_DIST].get())
    min_dist = float(np.array([min_dist_bits], dtype=np.int32).view(
        np.float32)[0])
    if _DEBUG and iteration <= 200:
        print(f"  SCAN result: min_dist={min_dist:.1f}, delta={delta:.1f}")
    if min_dist >= 1e29:
        if _DEBUG: print("  EXIT: no unsettled states found")
        return None

    # Sanity check: corrupt distances from pool overflow produce
    # absurdly large values. Bail if min_dist is unreasonable.
    # Maximum plausible cost: n_cells * max_edge_cost * 100
    max_plausible = float(n_cells) * 65535.0 * 100.0
    if min_dist > max_plausible:
        if _DEBUG: print(f"  EXIT: corrupt min_dist={min_dist:.0f}")
        return None

    new_bucket = _compute_f32_bucket(min_dist, delta)
    return n_alloc, grid_scan, min_dist, new_bucket


def _v3_extract_frontier(d_pool, d_block_to_cell, n_alloc, grid_scan,
                         new_bucket, delta, spc, n_span_bins, n_heights,
                         frontier, d_control, buf_size,
                         extract_kernel, _DEBUG, iteration):
    """Extract frontier states for the given bucket.

    Returns frontier_count, or None if should break.
    """
    d_control[_V3_CTL_NEAR] = 0
    extract_kernel(
        (grid_scan,), (256,),
        (d_pool, d_block_to_cell, np.int32(n_alloc),
         np.int32(new_bucket), np.float32(delta),
         np.int64(spc), np.int32(n_span_bins), np.int32(n_heights),
         frontier, d_control, np.int32(buf_size)))
    cp.cuda.Stream.null.synchronize()

    frontier_count = int(d_control[_V3_CTL_NEAR].get())
    if _DEBUG and iteration <= 200:
        print(f"  EXTRACT: bucket={new_bucket} blo={new_bucket*delta:.1f} bhi={(new_bucket+1)*delta:.1f} frontier={frontier_count}")
    if frontier_count == 0:
        if _DEBUG: print("  EXIT: extract found 0 states")
        return None

    return frontier_count


def _v3_full_scan_fallback(d_n_allocated, d_pool, d_block_to_cell, d_control,
                           gpu_block_size, bucket, delta, n_cells,
                           frontier, spc, n_span_bins, n_heights, buf_size,
                           extract_kernel, scan_kernel, _DEBUG, iteration):
    """Full scan + extract. Returns (new_bucket, frontier_count) or None if should break."""
    scan_result = _v3_scan_min_dist(
        d_n_allocated, d_pool, d_block_to_cell, d_control,
        gpu_block_size, bucket, delta, n_cells,
        scan_kernel, _DEBUG, iteration)
    if scan_result is None:
        return None

    n_alloc, grid_scan, min_dist, new_bucket = scan_result

    frontier_count = _v3_extract_frontier(
        d_pool, d_block_to_cell, n_alloc, grid_scan,
        new_bucket, delta, spc, n_span_bins, n_heights,
        frontier, d_control, buf_size,
        extract_kernel, _DEBUG, iteration)
    if frontier_count is None:
        return None

    return new_bucket, frontier_count


def _build_v3_relax_args(frontier_arr, frontier_count, phase, buf_size, bucket,
                         rows, cols, max_cost, n_dirs, max_inter_cols,
                         n_heights, n_span_bins, span_bin_size, min_span_bin,
                         spc, total_states, max_blocks, delta,
                         max_tower_records,
                         cell_size, conductor_weight_per_m,
                         conductor_tension, min_clearance,
                         max_gradient_pct, gradient_scale,
                         gpu, pools,
                         output_arr, dem_ptr, obs_ptr,
                         area_off_ptr, area_starts_ptr, area_counts_ptr):
    """Build the argument tuple for the relax_constrained_v3 kernel launch."""
    return (frontier_arr, np.int32(frontier_count), np.int32(phase),
            np.int32(buf_size), np.int32(bucket),
            gpu['d_raster'], np.int32(rows), np.int32(cols), np.int32(max_cost),
            gpu['d_steps'], gpu['d_cost_factors'], gpu['d_inter_lut'],
            gpu['d_n_inter'],
            np.int32(n_dirs), np.int32(max_inter_cols),
            gpu['d_angle_cost'], gpu['d_angle_valid'], gpu['d_step_dist'],
            gpu['d_tower_terrain'], gpu['d_tower_angle'],
            gpu['d_height_premiums'], gpu['d_tower_heights'],
            np.int32(n_heights),
            np.int32(n_span_bins), np.float32(span_bin_size),
            np.int32(min_span_bin),
            np.int64(spc), np.int64(total_states),
            pools['d_cell_to_block'], pools['d_block_to_cell'],
            pools['d_pool'], pools['d_span_pool'],
            pools['d_n_allocated'], np.int32(max_blocks),
            np.float32(delta),
            output_arr, pools['d_pending'], pools['d_settled'],
            pools['d_control'],
            pools['d_tower_records'], np.int32(max_tower_records),
            dem_ptr, obs_ptr,
            np.float32(cell_size), np.float32(conductor_weight_per_m),
            np.float32(conductor_tension), np.float32(min_clearance),
            np.float32(max_gradient_pct), np.float32(gradient_scale),
            area_off_ptr, area_starts_ptr, area_counts_ptr)


def _run_v3_heavy_phase(settled_count, tpb, smem_bytes,
                        bucket, rows, cols, max_cost, n_dirs, max_inter_cols,
                        n_heights, n_span_bins, span_bin_size, min_span_bin,
                        spc, total_states, max_blocks, delta, buf_size,
                        max_tower_records,
                        cell_size, conductor_weight_per_m,
                        conductor_tension, min_clearance,
                        max_gradient_pct, gradient_scale,
                        gpu, pools, output,
                        dem_ptr, obs_ptr,
                        area_off_ptr, area_starts_ptr, area_counts_ptr,
                        relax_kernel, _DEBUG, iteration):
    """Run the heavy phase. Returns (heavy_output, heavy_pending) or (0, 0) if settled_count is 0."""
    if settled_count == 0:
        return 0, 0

    d_control = pools['d_control']
    d_control[_V3_CTL_OUTPUT] = 0
    grid = (settled_count + tpb - 1) // tpb
    args = _build_v3_relax_args(
        pools['d_settled'], settled_count, 1,  # phase=HEAVY
        buf_size, bucket,
        rows, cols, max_cost, n_dirs, max_inter_cols,
        n_heights, n_span_bins, span_bin_size, min_span_bin,
        spc, total_states, max_blocks, delta,
        max_tower_records,
        cell_size, conductor_weight_per_m,
        conductor_tension, min_clearance,
        max_gradient_pct, gradient_scale,
        gpu, pools, output,
        dem_ptr, obs_ptr,
        area_off_ptr, area_starts_ptr, area_counts_ptr)
    relax_kernel(
        (grid,), (tpb,),
        args,
        shared_mem=smem_bytes)
    cp.cuda.Stream.null.synchronize()

    # Check heavy output -- if within-bucket states produced,
    # loop back to light phase
    heavy_output = int(d_control[_V3_CTL_OUTPUT].get())
    heavy_pending = int(d_control[_V3_CTL_PENDING].get())
    if _DEBUG and iteration <= 200:
        print(f"    HEAVY: output={heavy_output} pending={heavy_pending} "
              f"overflow={int(d_control[_V3_CTL_OVERFLOW].get())}")
    return heavy_output, heavy_pending


def _run_v3_delta_stepping(
    n_cells, n_dirs, n_source, tpb, buf_size, delta, smem_bytes,
    max_inter_cols, rows, cols, max_cost,
    spc, total_states, n_span_bins, n_heights, min_span_bin, span_bin_size,
    cell_size, conductor_weight_per_m, conductor_tension, min_clearance,
    max_gradient_pct, gradient_scale, margin, max_tower_records,
    gpu_block_size, max_blocks,
    dem, obstacle_heights, area_offsets,
    gpu, pools,
    relax_kernel, classify_kernel, scan_kernel, extract_kernel,
    target_cell,
):
    """The main while loop. Returns (best_known, iteration)."""
    import time as _time

    d_cell_to_block = pools['d_cell_to_block']
    d_block_to_cell = pools['d_block_to_cell']
    d_n_allocated = pools['d_n_allocated']
    d_queue_a = pools['d_queue_a']
    d_queue_b = pools['d_queue_b']
    d_pending = pools['d_pending']
    d_control = pools['d_control']
    d_pool = pools['d_pool']

    # Prepare DEM/obstacle pointers: pass actual arrays or NULL (0)
    dem_ptr = gpu['d_dem'] if dem is not None else np.intp(0)
    obs_ptr = gpu['d_obstacle'] if obstacle_heights is not None else np.intp(0)
    area_off_ptr = gpu['d_area_offsets'] if area_offsets is not None else np.intp(0)
    area_starts_ptr = gpu['d_area_starts'] if area_offsets is not None else np.intp(0)
    area_counts_ptr = gpu['d_area_counts'] if area_offsets is not None else np.intp(0)

    bucket = 0
    # Max iterations: generous limit, time limit provides the real safety net.
    max_iters = n_cells * 10
    iteration = 0
    frontier_count = n_source
    frontier = d_queue_a
    output = d_queue_b
    best_known = 1e30
    _t_start = _time.perf_counter()
    _time_limit = 300.0  # 5-minute hard time limit

    _DEBUG = False
    _reset_pending = True  # reset pending at start of new bucket only

    while iteration < max_iters:
        iteration += 1

        # Time limit check (every 50 iterations to avoid overhead)
        if iteration % 50 == 0:
            if _time.perf_counter() - _t_start > _time_limit:
                if _DEBUG:
                    print(f"  EXIT: time limit ({_time_limit:.0f}s)")
                break

        # Reset counters for this bucket
        d_control[_V3_CTL_OUTPUT] = 0
        d_control[_V3_CTL_SETTLED] = 0
        if _reset_pending:
            d_control[_V3_CTL_PENDING] = 0
        _reset_pending = True  # default: reset next time

        # ---- Light phase: iterate until frontier empty ----
        while frontier_count > 0:
            grid = (frontier_count + tpb - 1) // tpb
            args = _build_v3_relax_args(
                frontier, frontier_count, 0,  # phase=LIGHT
                buf_size, bucket,
                rows, cols, max_cost, n_dirs, max_inter_cols,
                n_heights, n_span_bins, span_bin_size, min_span_bin,
                spc, total_states, max_blocks, delta,
                max_tower_records,
                cell_size, conductor_weight_per_m,
                conductor_tension, min_clearance,
                max_gradient_pct, gradient_scale,
                gpu, pools, output,
                dem_ptr, obs_ptr,
                area_off_ptr, area_starts_ptr, area_counts_ptr)
            relax_kernel(
                (grid,), (tpb,),
                args,
                shared_mem=smem_bytes)
            cp.cuda.Stream.null.synchronize()

            frontier_count = int(d_control[_V3_CTL_OUTPUT].get())
            d_control[_V3_CTL_OUTPUT] = 0
            frontier, output = output, frontier

        if _DEBUG and iteration <= 200:
            _s = int(d_control[_V3_CTL_SETTLED].get())
            _p = int(d_control[_V3_CTL_PENDING].get())
            _t = int(d_control[_V3_CTL_TOWER].get())
            _o = int(d_control[_V3_CTL_OVERFLOW].get())
            _a = int(d_n_allocated.get())
            print(f"  iter={iteration} bkt={bucket}: settled={_s} pend={_p} towers={_t} overflow={_o} alloc={_a}")

        # ---- Early termination: check target ----
        best_known, should_break = _check_v3_early_termination(
            d_cell_to_block, d_pool, target_cell,
            gpu_block_size, best_known, bucket, delta, margin)
        if should_break:
            break

        # ---- Heavy phase ----
        settled_count = int(d_control[_V3_CTL_SETTLED].get())
        heavy_output, heavy_pending = _run_v3_heavy_phase(
            settled_count, tpb, smem_bytes,
            bucket, rows, cols, max_cost, n_dirs, max_inter_cols,
            n_heights, n_span_bins, span_bin_size, min_span_bin,
            spc, total_states, max_blocks, delta, buf_size,
            max_tower_records,
            cell_size, conductor_weight_per_m,
            conductor_tension, min_clearance,
            max_gradient_pct, gradient_scale,
            gpu, pools, output,
            dem_ptr, obs_ptr,
            area_off_ptr, area_starts_ptr, area_counts_ptr,
            relax_kernel, _DEBUG, iteration)
        if heavy_output > 0:
            frontier_count = heavy_output
            d_control[_V3_CTL_OUTPUT] = 0
            d_control[_V3_CTL_SETTLED] = 0
            frontier, output = output, frontier
            _reset_pending = False  # preserve pending from this bucket
            continue  # back to light phase

        # ---- Advance bucket ----
        bucket += 1
        pending_count = int(d_control[_V3_CTL_PENDING].get())

        if pending_count > 0:
            (frontier_count, should_continue, bucket_offset,
             preserve_pending) = _v3_classify_pending(
                d_pending, pending_count, bucket, delta, buf_size,
                d_cell_to_block, d_pool, spc, n_span_bins, n_heights,
                frontier, pools['d_settled'], d_control, tpb,
                classify_kernel, _DEBUG, iteration)
            if preserve_pending:
                _reset_pending = False
            if should_continue:
                bucket += bucket_offset
                continue

        # ---- Full scan fallback ----
        scan_result = _v3_full_scan_fallback(
            d_n_allocated, d_pool, d_block_to_cell, d_control,
            gpu_block_size, bucket, delta, n_cells,
            frontier, spc, n_span_bins, n_heights, buf_size,
            extract_kernel, scan_kernel, _DEBUG, iteration)
        if scan_result is None:
            break
        bucket, frontier_count = scan_result

    return best_known, iteration


def _find_v3_best_target(d_cell_to_block, d_pool, target_cell, gpu_block_size,
                         spc, n_span_bins, n_heights):
    """Scan target block entries to find best_state.

    Returns (best_dist, best_state) or warns and returns None.
    """
    target_block_idx = int(d_cell_to_block[target_cell].item())
    if target_block_idx < 0:
        warnings.warn("GPU constrained SSSP V3 found no path to target "
                       "(target cell never visited)")
        return None

    block_base = target_block_idx * gpu_block_size
    target_entries = d_pool[block_base:block_base + gpu_block_size].get()
    block_entry_host_dtype = np.dtype([
        ('local_key', np.uint16), ('_pad', np.uint16), ('dist', np.float32)
    ])
    target_entries_np = np.frombuffer(
        target_entries.tobytes(), dtype=block_entry_host_dtype)

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
            # Reconstruct full state from cell + local_key
            direction = lk // sh_val
            rem = lk % sh_val
            sb = rem // n_heights
            hc = rem % n_heights
            best_state = (target_cell * spc
                          + direction * sh_val
                          + sb * n_heights + hc)

    if best_dist >= 1e29 or best_state == -1:
        warnings.warn("GPU constrained SSSP V3 found no path to target")
        return None

    return best_dist, best_state


def _download_v3_results(d_pool, d_n_allocated, gpu_block_size,
                         d_cell_to_block, d_tower_records, d_control,
                         n_tower_records_max, spc, n_span_bins, n_heights):
    """Download pool and tower records. Returns (dist_cpu, tower_records_cpu, n_tower_records)."""
    control_cpu = d_control.get()
    overflow_count = int(control_cpu[_V3_CTL_OVERFLOW])
    if overflow_count > 0:
        warnings.warn(
            f"GPU constrained SSSP V3 had {overflow_count} overflow events. "
            f"Results may be suboptimal. Consider increasing max_visited_fraction "
            f"or buf_size.")

    n_tower_records = min(int(control_cpu[_V3_CTL_TOWER]), n_tower_records_max)

    block_entry_host_dtype = np.dtype([
        ('local_key', np.uint16), ('_pad', np.uint16), ('dist', np.float32)
    ])

    # Download tower records
    tower_record_dtype = np.dtype([
        ('state', np.int64),
        ('pred_state', np.int64),
        ('span_dist', np.float16),
        ('tower_height', np.float16),
        ('_pad', np.uint8, 4),  # padding to 24 bytes
    ])
    if n_tower_records > 0:
        raw_bytes = d_tower_records[:n_tower_records * 24].get()
        tower_records_cpu = np.frombuffer(raw_bytes, dtype=tower_record_dtype)
    else:
        tower_records_cpu = np.zeros(0, dtype=tower_record_dtype)

    # Download block data for reconstruction
    n_alloc_final = int(d_n_allocated.item())
    pool_cpu = d_pool[:n_alloc_final * gpu_block_size].get()
    pool_np = np.frombuffer(pool_cpu.tobytes(), dtype=block_entry_host_dtype)
    c2b_cpu = d_cell_to_block.get()

    dist_cpu = _DynamicBlockDistProxy(
        pool_np, c2b_cpu, gpu_block_size, spc, n_span_bins, n_heights)

    return dist_cpu, tower_records_cpu, n_tower_records


# ============================================================================
# Main entry point: host-driven V3 constrained SSSP
# ============================================================================

def constrained_sssp_raster_gpu_v3(
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
    max_tower_records: int = 2_000_000,
    max_visited_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find constrained shortest path on GPU with tower placement (V3 host-driven).

    Uses host-driven delta-stepping with per-launch CUDA kernels and dynamic
    block-sparse distance storage. No cooperative groups required.

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
        height_premiums: float32 (n_heights,) cost premium per height class.
        n_heights: number of tower height classes (default 1)
        exclude_mask: optional uint8 mask (1=traversable, 0=blocked)
        dem: optional float32 DEM array for clearance checking and gradient.
        obstacle_heights: optional float32 array of obstacle heights (meters).
        cell_size: raster cell size in meters (default 1.0)
        conductor_weight_per_m: conductor weight per meter in N/m (default 0.0)
        conductor_tension: conductor tension in N (default 1.0)
        min_clearance: minimum clearance above ground+obstacles in meters
        max_gradient_pct: maximum allowed gradient percentage (default 100.0)
        gradient_scale: exponential gradient cost scaling factor (default 2.0)
        tower_heights: float32 (n_heights,) actual tower heights in meters,
            sorted DESCENDING (tallest first). If None, defaults to [0.0].
        area_offsets: optional int32 flat array of (dr, dc) pairs for all
            direction pairs. Used for rotated square tower footprint area cost.
        area_offset_starts: optional int32 (n_dirs*n_dirs,) start index per pair.
        area_offset_counts: optional int32 (n_dirs*n_dirs,) count per pair.
        threads_per_block: CUDA threads per block
        margin: early termination margin (default 1.00001)
        max_tower_records: max tower records to allocate (default 2M)
        max_visited_fraction: expected fraction of cells visited (default 0.15)

    Returns:
        tuple: (path_cell_indices uint32[], tower_cell_indices uint32[],
                tower_heights float32[])

    Raises:
        RuntimeError: if CUDA GPU not available
        ValueError: if source or target is on forbidden cell
    """
    # ---- Validate inputs and compute defaults ----
    (source_cell, target_cell, height_premiums, n_heights,
     tower_heights, angle_cost_lut) = _validate_v3_inputs(
        raster, source_row, source_col, target_row, target_col,
        height_premiums, n_heights, tower_heights,
        angle_cost_lut, tower_terrain_costs, tower_angle_costs)

    rows, cols = raster.shape
    n_cells = rows * cols
    n_dirs = len(steps)
    max_cost = int(np.iinfo(np.uint16).max)
    spc = n_dirs * n_span_bins * n_heights
    total_states = n_cells * spc
    min_span_bin = int(min_span / span_bin_size)

    # Apply exclude mask
    if exclude_mask is not None:
        raster = raster.copy()
        raster[exclude_mask == 0] = 65535

    # ---- Compute BLOCK_SIZE ----
    gpu_block_size, max_visited_fraction = _compute_v3_block_size(
        spc, n_cells, max_visited_fraction)

    # Compute max_blocks
    max_blocks = int(n_cells * max_visited_fraction * 1.5)
    # Floor: small rasters can visit all cells (<=5000 blocks ≈ 3 MB);
    # also guard against atomicCAS race waste in get_block
    max_blocks = max(max_blocks, n_dirs * 10, min(n_cells, 5000))

    # Prepare step lookup tables
    from pyorps.utils.traversal_gpu import prepare_step_lookup_tables
    steps_arr, intermediates_lut, n_intermediates, cost_factors = \
        prepare_step_lookup_tables(steps)
    max_inter_cols = intermediates_lut.shape[1]

    # Compute delta
    delta = _compute_constrained_delta(raster, cost_factors)

    # Buffer size for queues
    buf_size = max(n_cells * 4, 1 << 20)

    # Shared memory
    smem_bytes = _compute_v3_smem(n_dirs, n_heights, max_inter_cols)

    # ---- Upload data to GPU ----
    gpu = _upload_v3_gpu_data(
        raster, steps_arr, cost_factors,
        intermediates_lut, n_intermediates,
        angle_cost_lut, angle_valid_lut, step_distances,
        tower_terrain_costs, tower_angle_costs,
        height_premiums, tower_heights,
        dem, obstacle_heights,
        area_offsets, area_offset_starts, area_offset_counts)

    # ---- Allocate GPU pools ----
    pools = _allocate_v3_pools(n_cells, gpu_block_size, max_blocks, buf_size,
                               max_tower_records, spc)
    pool_size = pools['pool_size']

    # ---- Initialize pool and source states via GPU kernels ----
    init_pool_kernel = _get_v3_kernel("init_pool_v3", block_size=gpu_block_size)
    tpb_init = 256
    bpg_init = (pool_size + tpb_init - 1) // tpb_init
    init_pool_kernel(
        (bpg_init,), (tpb_init,),
        (pools['d_pool'], np.int32(pool_size)))
    cp.cuda.Stream.null.synchronize()

    source_state_ids, n_source = _init_v3_source_states(
        source_cell, n_dirs, n_heights, height_premiums,
        spc, n_span_bins, gpu_block_size, max_blocks,
        pools['d_pool'], pools['d_span_pool'], pools['d_cell_to_block'],
        pools['d_block_to_cell'], pools['d_n_allocated'], pools['d_queue_a'])

    # ---- Compile relax kernel ----
    relax_kernel = _get_v3_kernel("relax_constrained_v3",
                                  block_size=gpu_block_size)
    classify_kernel = _get_v3_kernel("classify_bucket",
                                     block_size=gpu_block_size)
    scan_kernel = _get_v3_kernel("scan_min_dist",
                                 block_size=gpu_block_size)
    extract_kernel = _get_v3_kernel("extract_bucket",
                                    block_size=gpu_block_size)

    tpb = threads_per_block

    # ---- Host-driven delta-stepping driver loop ----
    best_known, iteration = _run_v3_delta_stepping(
        n_cells, n_dirs, n_source, tpb, buf_size, delta, smem_bytes,
        max_inter_cols, rows, cols, max_cost,
        spc, total_states, n_span_bins, n_heights, min_span_bin, span_bin_size,
        cell_size, conductor_weight_per_m, conductor_tension, min_clearance,
        max_gradient_pct, gradient_scale, margin, max_tower_records,
        gpu_block_size, max_blocks,
        dem, obstacle_heights, area_offsets,
        gpu, pools,
        relax_kernel, classify_kernel, scan_kernel, extract_kernel,
        target_cell,
    )

    # ---- Post-run: download results ----
    dist_cpu, tower_records_cpu, n_tower_records = _download_v3_results(
        pools['d_pool'], pools['d_n_allocated'], gpu_block_size,
        pools['d_cell_to_block'], pools['d_tower_records'], pools['d_control'],
        max_tower_records, spc, n_span_bins, n_heights)

    # ---- Find best target state ----
    target_result = _find_v3_best_target(
        pools['d_cell_to_block'], pools['d_pool'], target_cell,
        gpu_block_size, spc, n_span_bins, n_heights)
    if target_result is None:
        return (np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))
    best_dist, best_state = target_result

    # ---- Reconstruct path ----
    result = _reconstruct_from_tower_records(
        tower_records_cpu, n_tower_records, best_state,
        source_cell, spc, n_span_bins, n_heights, n_dirs,
        cols, steps, dist_cpu=dist_cpu)

    return result
