"""GPU persistent-kernel constrained pathfinding with tower-record tracking.

V2 of the GPU constrained SSSP: single-launch persistent cooperative kernel
with full 4D state (cell, direction, span_bin, height_class), inline tower
placement via TowerRecord atomic-append, and float16 span tracking.

Algorithm:
    Persistent cooperative delta-stepping with custom sense-reversing grid
    barrier (no grid.sync() — Blackwell-safe). Each bucket processes:
    - Light phase: edges with cost <= delta (repeats until no new states)
    - Heavy phase: edges with cost > delta (once per bucket)

    Tower placement (hybrid per-thread + warp-cooperative protocol):
    - Same direction: optional tower when span >= min_span
    - Direction change: mandatory tower when span >= min_span
    - Non-tower edges (same-direction span continuation) are processed
      per-thread for maximum parallelism (each thread handles its own
      frontier item independently -- no lane waste).
    - Tower placement uses warp-cooperative round-robin protocol: all 32
      lanes in a warp process their non-tower work independently, then
      cooperate on tower placement via __ballot_sync + __shfl_sync for
      parallel area cost summation and clearance checking.
    - TowerRecord appended atomically for every relaxation that goes via
      a tower placement and improves a state.

    Clearance checking:
    - When DEM is provided, catenary sag is checked along each span at
      tower placement. Parabolic approximation: sag(x) = w*x*(L-x)/(2*T).
    - Variable tower heights: sorted descending, early exit when tallest
      fails clearance (shorter heights will also fail).
    - Gradient penalty on traversal: slope > max_gradient_pct is rejected,
      otherwise cost scaled by exp(gradient_scale * slope / 100).
    - Warp-cooperative clearance: 32 lanes check span cells in parallel,
      then warp-reduce with __ballot_sync to determine if clearance passes.

    Area cost:
    - When area_offsets are provided, tower terrain cost sums all pixels in
      the rotated square footprint (warp-cooperative parallel sum).
    - Forbidden footprint rejection: if ANY pixel in the footprint is 65535,
      the tower placement is rejected (warp ballot check).
    - Area-averaged slope multiplier: exp(gradient_scale * avg_slope / 100)
      applied to the summed terrain cost.
    - When area_offsets is NULL (uniform mode), falls back to single-pixel
      tower_terrain_lut[raster[cell]] with single-pixel slope multiplier.

    Path reconstruction walks the tower record chain backward (no d_pred).
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
# State encoding helpers (Python-side)
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
# Memory budget helpers
# ============================================================================

def compute_memory_budget_gb(rows, cols, n_dirs, n_span_bins, n_heights,
                             sparse=False):
    """Estimate GPU memory needed in GB.

    Parameters:
        sparse: False for dense, True/"sparse" for hash table,
                "block" for block-sparse mode.
    """
    total_cells = rows * cols
    input_bytes = total_cells * (2 + 4 + 4 + 1)  # raster+dem+obs+mask
    lut_bytes = 65536 * 4 + n_dirs * n_dirs * (4 + 1 + 4)
    buf_size = max(total_cells * 4, 1 << 20)
    queue_bytes = 4 * buf_size * 8       # 4 queues x int64
    tower_bytes = 1_000_000 * 24         # 1M tower records

    if sparse == "block":
        # Block-sparse: dynamic BLOCK_SIZE entries per cell
        block_bytes = total_cells * 32 * 8   # BlockEntry: 8 bytes each
        span_bytes = total_cells * 32 * 2    # block_span: float16 each
        state_bytes = block_bytes + span_bytes
    elif sparse:
        total_states = total_cells * n_dirs * n_span_bins * n_heights
        estimated_active = max(total_cells * 8, int(total_states * 0.05))
        estimated_active = min(estimated_active, total_states)
        # Cap by available VRAM if GPU is present
        if GPU_AVAILABLE:
            try:
                vram_free = cp.cuda.Device().mem_info[0]
                max_hash = max(int(vram_free * 0.6), 256 * 1024**2)
                max_ent = max_hash // 16
                estimated_active = min(estimated_active, max_ent // 2)
            except Exception:
                pass
        hash_capacity = 1
        while hash_capacity < estimated_active * 2:
            hash_capacity <<= 1
        state_bytes = hash_capacity * 16  # sizeof(StateEntry) = 16
    else:
        total_states = total_cells * n_dirs * n_span_bins * n_heights
        dist_bytes = total_states * 4        # float32
        span_bytes = total_states * 2        # float16
        state_bytes = dist_bytes + span_bytes

    total = (state_bytes + input_bytes + lut_bytes
             + queue_bytes + tower_bytes)
    return total / (1024 ** 3)


def check_memory_fits(rows, cols, n_dirs, n_span_bins, n_heights,
                      vram_gb=None, sparse=False):
    """Raise MemoryError if state space won't fit in GPU VRAM."""
    if vram_gb is None and GPU_AVAILABLE:
        vram_gb = cp.cuda.Device().mem_info[1] / (1024 ** 3)
    elif vram_gb is None:
        vram_gb = 16.0
    budget = compute_memory_budget_gb(
        rows, cols, n_dirs, n_span_bins, n_heights, sparse=sparse)
    if budget > vram_gb * 0.9:
        if sparse == "block":
            mode_name = "block-sparse"
        elif sparse:
            mode_name = "sparse"
        else:
            mode_name = "dense"
        raise MemoryError(
            f"Constrained GPU SSSP needs {budget:.1f} GB but only "
            f"{vram_gb:.1f} GB available "
            f"({mode_name} mode). "
            f"Reduce neighborhood or span bins.")


# ============================================================================
# Control buffer indices (must match CUDA #defines)
# ============================================================================

_CTL_COUNT_A = 0
_CTL_COUNT_B = 1
_CTL_SETTLED = 2
_CTL_PENDING = 3
_CTL_NEAR = 4
_CTL_FAR = 5
_CTL_BUCKET = 6
_CTL_DONE = 7
_CTL_EARLY_CTR = 8
_CTL_MIN_DIST = 9
_CTL_BARRIER_CNT = 10
_CTL_BARRIER_SENSE = 11
_CTL_TOWER_COUNT = 12
_CTL_QUEUE_OVERFLOW = 13
_CTL_FULL_SCANS = 14
_CTL_SIZE = 16


# ============================================================================
# CUDA kernel source loading from kernels/ directory
# ============================================================================

from pathlib import Path
import re

_KERNEL_DIR = Path(__file__).parent / "kernels"


def _load_kernel_source(main_file: str, block_size: int = 64) -> str:
    """Load CUDA source with #include resolution from kernels/ directory.

    CuPy's RawKernel does not support #include natively, so we resolve
    local #include "xxx.cuh" directives in Python before passing the
    concatenated source to CuPy. System includes (angle-bracket) are
    left intact for nvrtc.

    block_size: injected as #define BLOCK_SIZE before the source for
                block-sparse mode. Must be a power of 2.
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

_v2_kernel_cache = {}


def _get_v2_kernel(name, cooperative=False, block_size=64):
    """Compile or retrieve cached CuPy RawKernel.

    Loads CUDA source from the kernels/ directory, resolves local
    #include directives, and compiles via CuPy.
    """
    key = (name, cooperative, block_size)
    if key not in _v2_kernel_cache:
        # Map kernel name to source file
        if name == "constrained_persistent":
            source = _load_kernel_source("constrained_persistent.cu",
                                         block_size=block_size)
        elif name in ("init_source_states_sparse",
                       "init_block_entries",
                       "init_block_source"):
            source = _load_kernel_source("init_source.cu")
        else:
            raise ValueError(f"Unknown kernel: {name}")
        kwargs = {}
        if cooperative:
            kwargs["enable_cooperative_groups"] = True
            kwargs["options"] = ("--std=c++17", "-Xptxas", "-dlcm=cg")
        else:
            kwargs["options"] = ("--std=c++17",)
        _v2_kernel_cache[key] = cp.RawKernel(source, name, **kwargs)
    return _v2_kernel_cache[key]


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


def _compute_v2_smem(n_steps, n_heights, max_inter_cols):
    """Compute shared memory bytes for the constrained persistent kernel."""
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
# Delta computation
# ============================================================================

def _compute_constrained_delta(raster, cost_factors,
                                tower_terrain_costs=None):
    """Compute delta from terrain + tower costs for fewer buckets.

    The old heuristic (2 * mean_raster * mean_cf) ignored tower costs,
    leading to a very small delta and excessive bucket iterations.
    Including tower cost in the delta estimate ensures that typical
    tower-placement edges (which dominate constrained routing cost) fall
    into the light phase, dramatically reducing the number of buckets.
    """
    valid = raster.ravel()
    valid = valid[valid < 65535]
    if len(valid) == 0:
        return 100.0
    mean_cost = float(valid.mean())
    mean_cf = float(cost_factors.mean())
    mean_edge = mean_cost * mean_cf

    # Include tower terrain cost in delta estimate
    mean_tower = 0.0
    if tower_terrain_costs is not None:
        # Sample tower costs from actual raster values
        sample = valid[:min(10000, len(valid))]
        tower_costs = tower_terrain_costs[sample.astype(np.intp)]
        positive = tower_costs[tower_costs > 0]
        if len(positive) > 0:
            mean_tower = float(positive.mean())

    # Delta covers typical edge cost + half of typical tower cost
    return max(1.0, 2.0 * mean_edge + mean_tower * 0.5)


# ============================================================================
# Sparse distance proxy for path reconstruction
# ============================================================================

class _SparseDistProxy:
    """Dict-backed proxy that mimics array indexing for dist_cpu[state].

    Used in sparse mode to provide the same interface as a dense numpy
    array to _reconstruct_from_tower_records, but backed by a dict of
    {state_index: distance} from the hash table download.
    """

    def __init__(self, dist_dict):
        self._d = dist_dict

    def __getitem__(self, key):
        return self._d.get(int(key), 1e30)


class _BlockDistProxy:
    """Block-sparse proxy that mimics array indexing for dist_cpu[state].

    Used in block-sparse mode.  Internally stores the downloaded
    block_entries structured array and looks up by (cell, local_key).
    """
    _BLOCK_SIZE = 32
    _BLOCK_MASK = 31
    _BLOCK_EMPTY = 0xFFFF

    def __init__(self, block_entries_cpu, spc, n_span_bins, n_heights):
        self._blocks = block_entries_cpu
        self._spc = spc
        self._n_span_bins = n_span_bins
        self._n_heights = n_heights
        self._sh = n_span_bins * n_heights

    def __getitem__(self, key):
        state = int(key)
        cell = state // self._spc
        rem = state % self._spc
        direction = rem // self._sh
        rem2 = rem % self._sh
        span_bin = rem2 // self._n_heights
        hc = rem2 % self._n_heights
        local_key = (direction * self._sh + span_bin * self._n_heights + hc)
        base = cell * self._BLOCK_SIZE
        # Multiplicative hash matching CUDA local_hash
        h = (local_key * 2654435761) & self._BLOCK_MASK
        for probe in range(self._BLOCK_SIZE):
            slot = (h + probe) & self._BLOCK_MASK
            entry = self._blocks[base + slot]
            k = int(entry['local_key'])
            if k == local_key:
                return float(entry['dist'])
            if k == self._BLOCK_EMPTY:
                return 1e30
        return 1e30


# ============================================================================
# Path reconstruction helpers (CPU-side)
# ============================================================================

def _build_tower_record_index(tower_records_cpu, n_tower_records,
                               spc, n_span_bins, n_heights):
    """Build (cell, dir) -> list of record indices for fast lookup."""
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
    """Find a tower record at (walk_cell, walk_dir) that is an ancestor.

    An ancestor record has a state distance strictly less than max_dist,
    confirming it was visited before the current segment endpoint.
    """
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
                                cols, steps_np, spc, n_span_bins, n_heights,
                                dist_cpu, best_state):
    """Walk backward from best_state to find the tower chain.

    Returns list of tower chain dicts (tower_cell, tower_dir, post_cell,
    post_dir, height) in forward order (source to target).
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
    """Assemble full path by direction-walking between waypoints.

    Returns (waypoints, tower_cells, tower_heights_out) lists.
    """
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
    return waypoints, tower_cells, tower_heights_out


# ============================================================================
# Path reconstruction from tower records (CPU-side)
# ============================================================================

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

    # Build (cell, dir) -> list of record indices for fast lookup.
    cell_dir_to_records = _build_tower_record_index(
        tower_records_cpu, n_tower_records, spc, n_span_bins, n_heights)

    # Walk backward from best_state to find the tower chain.
    tower_chain = _walk_backward_tower_chain(
        tower_records_cpu, cell_dir_to_records,
        target_cell, target_dir, source_cell,
        cols, steps_np, spc, n_span_bins, n_heights,
        dist_cpu, best_state)

    # Build full path by direction-walking between waypoints
    waypoints, tower_cells, tower_heights_out = _assemble_path_from_tower_chain(
        tower_chain, target_cell, target_dir, source_cell,
        cols, steps_np, n_dirs)

    path_arr = (np.array(waypoints, dtype=np.uint32)
                if waypoints else np.empty(0, dtype=np.uint32))
    tower_arr = (np.array(tower_cells, dtype=np.uint32)
                 if tower_cells else np.empty(0, dtype=np.uint32))
    height_arr = (np.array(tower_heights_out, dtype=np.float32)
                  if tower_heights_out else np.empty(0, dtype=np.float32))
    return path_arr, tower_arr, height_arr


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
# V2 availability check
# ============================================================================

_v2_available = None


def _check_v2_available():
    """Check if v2 persistent kernel compiles on this GPU."""
    global _v2_available
    if _v2_available is not None:
        return _v2_available
    if not GPU_AVAILABLE:
        _v2_available = False
        return False
    try:
        _ensure_cuda_path()
        kernel = _get_v2_kernel(
            "constrained_persistent",
            cooperative=True)
        _ = kernel.kernel  # force compilation
        _v2_available = True
    except Exception:
        _v2_available = False
    return _v2_available


# ============================================================================
# Main entry point — helpers
# ============================================================================

def _validate_v2_inputs(raster, source_row, source_col, target_row, target_col,
                         height_premiums, n_heights, tower_heights,
                         angle_cost_lut, tower_terrain_costs, tower_angle_costs,
                         exclude_mask):
    """Validate inputs and apply defaults for constrained_sssp_raster_gpu_v2.

    Returns (source_cell, target_cell, height_premiums, n_heights,
             tower_heights, angle_cost_lut, raster) with cleaned values.
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

    # Validate non-negative costs (after inf cleanup)
    if np.any(angle_cost_lut < 0):
        raise ValueError("angle_cost_lut must have non-negative values")
    if np.any(tower_terrain_costs < 0):
        raise ValueError("tower_terrain_costs must have non-negative values")
    if np.any(tower_angle_costs < 0):
        raise ValueError("tower_angle_costs must have non-negative values")

    # Apply exclude mask
    if exclude_mask is not None:
        raster = raster.copy()
        raster[exclude_mask == 0] = 65535

    return (source_cell, target_cell, height_premiums, n_heights,
            tower_heights, angle_cost_lut, raster)


def _resolve_v2_storage_mode(sparse, rows, cols, n_dirs, n_span_bins, n_heights):
    """Determine storage mode from the sparse parameter.

    Returns (use_sparse, use_managed, use_block, storage_mode).
    """
    use_sparse = False
    use_managed = False
    use_block = False
    storage_mode = 0  # STORAGE_DENSE
    if sparse == "block":
        use_block = True
        storage_mode = 2  # STORAGE_BLOCK
    elif sparse == "managed":
        use_managed = True
    elif sparse == "auto":
        dense_gb = compute_memory_budget_gb(
            rows, cols, n_dirs, n_span_bins, n_heights, sparse=False)
        vram_gb = cp.cuda.Device().mem_info[1] / (1024 ** 3)
        if dense_gb <= vram_gb * 0.7:
            pass  # dense fits
        else:
            # Prefer block-sparse when dense doesn't fit
            block_gb = compute_memory_budget_gb(
                rows, cols, n_dirs, n_span_bins, n_heights, sparse="block")
            if block_gb <= vram_gb * 0.8:
                use_block = True
                storage_mode = 2  # STORAGE_BLOCK
            else:
                use_managed = True  # fall back to managed
    elif sparse == "sparse":
        use_sparse = True
        storage_mode = 1  # STORAGE_SPARSE
    else:
        use_sparse = bool(sparse) if sparse not in (False, "dense") else False
        if use_sparse:
            storage_mode = 1  # STORAGE_SPARSE

    if not use_managed and not use_block:
        check_memory_fits(rows, cols, n_dirs, n_span_bins, n_heights,
                          sparse=use_sparse)
    elif use_block:
        check_memory_fits(rows, cols, n_dirs, n_span_bins, n_heights,
                          sparse="block")

    return use_sparse, use_managed, use_block, storage_mode


def _init_v2_block_storage(n_cells, n_dirs, n_span_bins, n_heights,
                            spc, source_state_ids, source_init_dists,
                            n_source, gpu_block_size):
    """Initialize block-sparse storage mode.

    Returns (d_block_entries, d_block_span, d_dist, d_span_dist,
             hash_capacity, hash_mask, d_state_table, gpu_block_size).
    """
    # Block-sparse mode: compute BLOCK_SIZE to fit VRAM
    # spc = states per cell. Ideally BLOCK_SIZE >= spc for no eviction.
    # Cap by VRAM: leave 1 GB for queues + input.
    spc_val = n_dirs * n_span_bins * n_heights
    vram_free = int(cp.cuda.Device().mem_info[0])
    max_block_bytes = max(vram_free - 1024**3, 256 * 1024**2)
    max_bs = max_block_bytes // (n_cells * 10)  # 8 bytes entry + 2 bytes span
    # Round down to power of 2
    gpu_block_size = 1
    while gpu_block_size * 2 <= min(max_bs, spc_val):
        gpu_block_size *= 2
    # Minimum 32
    gpu_block_size = max(32, gpu_block_size)
    block_total = n_cells * gpu_block_size

    block_entry_dtype = cp.dtype([
        ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)
    ])

    # Allocate block entries and span
    d_block_entries = cp.empty(block_total, dtype=block_entry_dtype)
    d_block_span = cp.zeros(block_total, dtype=cp.float16)

    # Initialize block entries via GPU kernel (local_key=BLOCK_EMPTY, dist=1e30)
    _ensure_cuda_path()
    init_be_kernel = _get_v2_kernel("init_block_entries", cooperative=False,
                                    block_size=gpu_block_size)
    tpb_init = 256
    bpg_init = (block_total + tpb_init - 1) // tpb_init
    init_be_kernel(
        (bpg_init,), (tpb_init,),
        (d_block_entries, np.int32(block_total)))
    cp.cuda.Stream.null.synchronize()

    # Initialize source states via GPU kernel
    init_bs_kernel = _get_v2_kernel("init_block_source", cooperative=False,
                                   block_size=gpu_block_size)
    d_src_states = cp.asarray(source_state_ids)
    d_src_dists = cp.asarray(source_init_dists)
    tpb_src = min(256, n_source)
    bpg_src = (n_source + tpb_src - 1) // tpb_src
    init_bs_kernel(
        (bpg_src,), (tpb_src,),
        (d_block_entries, d_block_span,
         d_src_states, d_src_dists,
         np.int32(n_source), np.int32(spc),
         np.int32(n_span_bins), np.int32(n_heights)))
    cp.cuda.Stream.null.synchronize()

    # Dense/sparse arrays not used -- allocate empty placeholders
    d_dist = cp.empty(0, dtype=cp.float32)
    d_span_dist = cp.empty(0, dtype=cp.float16)
    hash_capacity = 0
    hash_mask = 0
    state_entry_dtype = cp.dtype([
        ('key', cp.int64), ('dist', cp.float32),
        ('span_dist', cp.float16), ('_pad', cp.float16)
    ])
    d_state_table = cp.empty(0, dtype=state_entry_dtype)

    return (d_block_entries, d_block_span, d_dist, d_span_dist,
            hash_capacity, hash_mask, d_state_table, gpu_block_size)


def _init_v2_sparse_storage(total_states, n_cells, source_state_ids,
                              source_init_dists, n_source):
    """Initialize sparse hash table storage mode.

    Returns (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
             d_block_entries, d_block_span).
    """
    block_entry_dtype = cp.dtype([
        ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)
    ])

    # Hash table: estimate active states, power-of-2 capacity, load factor 0.5
    # Use 10% of total state space but cap by available VRAM
    n_traversable = int(np.sum(n_cells))  # placeholder, caller passes raster
    estimated_active = max(
        n_traversable * 8,                    # minimum: 8 states per cell
        int(total_states * 0.05),             # 5% of total state space
    )
    # Cap: hash table at 50% of free VRAM (leave room for queues + input + overhead)
    vram_bytes = int(cp.cuda.Device().mem_info[0])  # free VRAM
    max_hash_entries = int(vram_bytes * 0.5) // 16  # 50% VRAM / 16 bytes each
    estimated_active = min(estimated_active, max_hash_entries // 2)  # load factor 0.5
    hash_capacity = 1
    while hash_capacity < estimated_active * 2:  # load factor 0.5
        hash_capacity <<= 1
    # Post-rounding cap: hash table must fit in 50% of free VRAM
    vram_free = int(cp.cuda.Device().mem_info[0])
    max_hash_cap = int(vram_free * 0.5) // 16
    while hash_capacity > max_hash_cap and hash_capacity > (1 << 20):
        hash_capacity >>= 1
    hash_mask = hash_capacity - 1

    # Allocate hash table: 16 bytes per StateEntry
    # memset to 0x7F sets keys to HASH_EMPTY (0x7F7F7F7F7F7F7F7F)
    # and dist to 0x7F7F7F7F (~3.3e38, large positive for atomicMin)
    state_entry_dtype = cp.dtype([
        ('key', cp.int64), ('dist', cp.float32),
        ('span_dist', cp.float16), ('_pad', cp.float16)
    ])
    d_state_table = cp.empty(hash_capacity, dtype=state_entry_dtype)
    cp.cuda.runtime.memset(d_state_table.data.ptr, 0x7F,
                           hash_capacity * 16)

    # Dense arrays not used -- allocate empty placeholders
    d_dist = cp.empty(0, dtype=cp.float32)
    d_span_dist = cp.empty(0, dtype=cp.float16)

    # Initialize source states via GPU kernel
    _ensure_cuda_path()
    init_kernel = _get_v2_kernel(
        "init_source_states_sparse",
        cooperative=False)
    d_src_states = cp.asarray(source_state_ids)
    d_src_dists = cp.asarray(source_init_dists)
    tpb_init = min(256, n_source)
    bpg_init = (n_source + tpb_init - 1) // tpb_init
    init_kernel(
        (bpg_init,), (tpb_init,),
        (d_state_table, np.int32(hash_mask),
         d_src_states, d_src_dists, np.int32(n_source)))
    cp.cuda.Stream.null.synchronize()

    d_block_entries = cp.empty(0, dtype=block_entry_dtype)
    d_block_span = cp.empty(0, dtype=cp.float16)

    return (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
            d_block_entries, d_block_span)


def _init_v2_managed_storage(total_states, source_states):
    """Initialize managed memory storage mode.

    Returns (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
             d_block_entries, d_block_span, dist_ptr, span_ptr).
    """
    block_entry_dtype = cp.dtype([
        ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)
    ])

    # Managed memory mode: dense arrays backed by CUDA managed memory.
    # Pages between VRAM and system RAM automatically — no VRAM limit.
    # Kernel code is identical to dense mode (use_sparse=0).
    dist_bytes = int(total_states * 4)
    span_bytes = int(total_states * 2)

    # Allocate managed memory via CUDA runtime
    dist_ptr = cp.cuda.runtime.mallocManaged(dist_bytes)
    span_ptr = cp.cuda.runtime.mallocManaged(span_bytes)

    # Initialize dist to 1e30 and span to 0
    # Can't use memset for float values, so wrap and fill
    d_dist = cp.ndarray(total_states, dtype=cp.float32,
                        memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(dist_ptr, dist_bytes, None), 0))
    d_span_dist = cp.ndarray(total_states, dtype=cp.float16,
                              memptr=cp.cuda.MemoryPointer(
                                  cp.cuda.UnownedMemory(span_ptr, span_bytes, None), 0))
    d_dist[:] = np.float32(1e30)
    d_span_dist[:] = np.float16(0)
    cp.cuda.Stream.null.synchronize()

    # Initialize source states
    for st, premium in source_states:
        d_dist[int(st)] = np.float32(premium)

    # No hash table (dense mode kernel path)
    hash_capacity = 0
    hash_mask = 0
    state_entry_dtype = cp.dtype([
        ('key', cp.int64), ('dist', cp.float32),
        ('span_dist', cp.float16), ('_pad', cp.float16)
    ])
    d_state_table = cp.empty(0, dtype=state_entry_dtype)

    d_block_entries = cp.empty(0, dtype=block_entry_dtype)
    d_block_span = cp.empty(0, dtype=cp.float16)

    return (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
            d_block_entries, d_block_span, dist_ptr, span_ptr)


def _init_v2_dense_storage(total_states, source_states):
    """Initialize dense storage mode.

    Returns (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
             d_block_entries, d_block_span).
    """
    block_entry_dtype = cp.dtype([
        ('local_key', cp.uint16), ('_pad', cp.uint16), ('dist', cp.float32)
    ])

    # Dense mode: full dist[] and span_dist[] arrays in device memory
    d_dist = cp.full(total_states, np.float32(1e30), dtype=cp.float32)
    d_span_dist = cp.zeros(total_states, dtype=cp.float16)

    # Initialize source states directly
    for st, premium in source_states:
        d_dist[int(st)] = np.float32(premium)

    # No hash table
    hash_capacity = 0
    hash_mask = 0
    state_entry_dtype = cp.dtype([
        ('key', cp.int64), ('dist', cp.float32),
        ('span_dist', cp.float16), ('_pad', cp.float16)
    ])
    d_state_table = cp.empty(0, dtype=state_entry_dtype)

    d_block_entries = cp.empty(0, dtype=block_entry_dtype)
    d_block_span = cp.empty(0, dtype=cp.float16)

    return (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
            d_block_entries, d_block_span)


def _upload_v2_gpu_data(raster, steps_arr, cost_factors, intermediates_lut,
                         n_intermediates, angle_cost_lut, angle_valid_lut,
                         step_distances, tower_terrain_costs, tower_angle_costs,
                         height_premiums, tower_heights, dem, obstacle_heights,
                         area_offsets, area_offset_starts, area_offset_counts):
    """Upload input data to GPU.

    Returns dict of GPU arrays keyed by name.
    """
    d = {}
    d['d_raster'] = cp.asarray(raster.astype(np.uint16))
    d['d_steps'] = cp.asarray(steps_arr.astype(np.int8))
    d['d_cost_factors'] = cp.asarray(cost_factors.astype(np.float32))
    d['d_inter_lut'] = cp.asarray(intermediates_lut.astype(np.int8))
    d['d_n_inter'] = cp.asarray(n_intermediates.astype(np.int32))
    d['d_angle_cost'] = cp.asarray(angle_cost_lut.reshape(-1).astype(np.float32))
    d['d_angle_valid'] = cp.asarray(angle_valid_lut.reshape(-1).astype(np.uint8))
    d['d_step_dist'] = cp.asarray(step_distances.astype(np.float32))
    d['d_tower_terrain'] = cp.asarray(tower_terrain_costs.astype(np.float32))
    d['d_tower_angle'] = cp.asarray(tower_angle_costs.reshape(-1).astype(np.float32))
    d['d_height_premiums'] = cp.asarray(height_premiums.astype(np.float32))
    d['d_tower_heights'] = cp.asarray(tower_heights.astype(np.float32))

    # Upload DEM and obstacle arrays (or create NULL-equivalent empty arrays)
    if dem is not None:
        d['d_dem'] = cp.asarray(dem.astype(np.float32).ravel())
    else:
        d['d_dem'] = cp.empty(0, dtype=cp.float32)
    if obstacle_heights is not None:
        d['d_obstacle'] = cp.asarray(obstacle_heights.astype(np.float32).ravel())
    else:
        d['d_obstacle'] = cp.empty(0, dtype=cp.float32)

    # Upload area offset arrays (or prepare NULL pointers)
    if area_offsets is not None:
        d['d_area_offsets'] = cp.asarray(area_offsets.astype(np.int32).ravel())
        d['d_area_starts'] = cp.asarray(area_offset_starts.astype(np.int32).ravel())
        d['d_area_counts'] = cp.asarray(area_offset_counts.astype(np.int32).ravel())
    else:
        d['d_area_offsets'] = cp.empty(0, dtype=cp.int32)
        d['d_area_starts'] = cp.empty(0, dtype=cp.int32)
        d['d_area_counts'] = cp.empty(0, dtype=cp.int32)

    return d


def _find_best_target_block(d_block_entries, target_cell, spc,
                              gpu_block_size, n_span_bins, n_heights):
    """Find best target state in block-sparse mode.

    Returns (best_dist, best_state) or None if no path found.
    """
    # Block-sparse mode: scan target cell's 32 block slots
    block_base = target_cell * gpu_block_size
    target_block = d_block_entries[block_base:block_base + 32].get()
    block_entry_host_dtype = np.dtype([
        ('local_key', np.uint16), ('_pad', np.uint16), ('dist', np.float32)
    ])
    target_block_np = np.frombuffer(target_block.tobytes(),
                                    dtype=block_entry_host_dtype)
    best_dist = float('inf')
    best_state = -1
    sh_val = n_span_bins * n_heights
    for entry in target_block_np:
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
        return None
    return best_dist, best_state


def _find_best_target_sparse(d_state_table, target_cell, spc):
    """Find best target state in sparse hash table mode.

    Returns (best_dist, best_state, dist_cpu_dict) or None if no path found.
    """
    target_start = target_cell * spc
    # Sparse mode: search hash table for target states
    table_host = d_state_table.get()
    target_end = target_start + spc
    best_dist = float('inf')
    best_state = -1
    for entry in table_host:
        k = int(entry['key'])
        if k == 0x7F7F7F7F7F7F7F7F:  # HASH_EMPTY
            continue
        if target_start <= k < target_end:
            d = float(entry['dist'])
            if d < best_dist:
                best_dist = d
                best_state = k

    if best_dist >= 1e29 or best_state == -1:
        return None

    # Build dist_cpu lookup from hash table for reconstruction
    dist_cpu_dict = {}
    for entry in table_host:
        k = int(entry['key'])
        if k != -1:
            dist_cpu_dict[k] = float(entry['dist'])

    return best_dist, best_state, dist_cpu_dict


def _find_best_target_dense(d_dist, target_cell, spc):
    """Find best target state in dense mode.

    Returns (best_dist, best_state) or None if no path found.
    """
    target_start = target_cell * spc
    # Dense mode: direct array indexing
    target_end = target_start + spc
    target_dists = d_dist[target_start:target_end].get()
    best_offset = int(np.argmin(target_dists))
    best_dist = float(target_dists[best_offset])

    if best_dist >= 1e29:
        return None

    best_state = target_start + best_offset
    return best_dist, best_state


def _find_v2_best_target(use_block, use_sparse, d_block_entries, d_dist,
                           d_state_table, target_cell, spc, gpu_block_size,
                           n_span_bins, n_heights):
    """Find best target state across storage modes.

    Returns (best_dist, best_state, dist_cpu_dict) where dist_cpu_dict is
    only populated in sparse mode (None otherwise). Returns None if no path found.
    """
    if use_block:
        result = _find_best_target_block(
            d_block_entries, target_cell, spc, gpu_block_size,
            n_span_bins, n_heights)
        if result is None:
            return None
        return result[0], result[1], None
    elif use_sparse:
        return _find_best_target_sparse(d_state_table, target_cell, spc)
    else:
        result = _find_best_target_dense(d_dist, target_cell, spc)
        if result is None:
            return None
        return result[0], result[1], None


def _download_v2_for_reconstruction(use_block, use_sparse, d_block_entries,
                                      d_dist, d_state_table, spc,
                                      n_span_bins, n_heights, gpu_block_size,
                                      dist_cpu_dict):
    """Download distance data in the right format for reconstruction.

    Returns dist_cpu (array, _BlockDistProxy, or _SparseDistProxy).
    """
    if use_block:
        # Block-sparse: download block_entries for reconstruction proxy
        blocks_cpu = d_block_entries.get()
        block_entries_host_dtype = np.dtype([
            ('local_key', np.uint16), ('_pad', np.uint16), ('dist', np.float32)
        ])
        blocks_np = np.frombuffer(blocks_cpu.tobytes(),
                                  dtype=block_entries_host_dtype)
        return _BlockDistProxy(blocks_np, spc, n_span_bins, n_heights)
    elif use_sparse:
        # Transfer dist for reconstruction using dict-backed array proxy
        return _SparseDistProxy(dist_cpu_dict)
    else:
        # Transfer dist for reconstruction (helps pick best record)
        return d_dist.get()


def _prepare_v2_kernel_ptrs(gpu_data, dem, obstacle_heights, area_offsets,
                              use_sparse, use_block, d_state_table,
                              d_block_entries, d_block_span):
    """Prepare kernel pointer arguments: actual arrays or NULL (0).

    Returns (dem_ptr, obs_ptr, area_off_ptr, area_starts_ptr,
             area_counts_ptr, state_table_ptr, block_entries_ptr,
             block_span_ptr).
    """
    # Prepare DEM/obstacle pointers: pass actual arrays or NULL (0)
    dem_ptr = gpu_data['d_dem'] if dem is not None else np.intp(0)
    obs_ptr = gpu_data['d_obstacle'] if obstacle_heights is not None else np.intp(0)
    area_off_ptr = gpu_data['d_area_offsets'] if area_offsets is not None else np.intp(0)
    area_starts_ptr = gpu_data['d_area_starts'] if area_offsets is not None else np.intp(0)
    area_counts_ptr = gpu_data['d_area_counts'] if area_offsets is not None else np.intp(0)

    # Sparse hash table pointer: actual array or NULL (0)
    state_table_ptr = d_state_table if use_sparse else np.intp(0)

    # Block-sparse pointers: actual arrays or NULL (0)
    block_entries_ptr = d_block_entries if use_block else np.intp(0)
    block_span_ptr = d_block_span if use_block else np.intp(0)

    return (dem_ptr, obs_ptr, area_off_ptr, area_starts_ptr,
            area_counts_ptr, state_table_ptr, block_entries_ptr,
            block_span_ptr)


def _init_v2_distance_storage(use_block, use_sparse, use_managed,
                                n_cells, n_dirs, n_span_bins, n_heights,
                                spc, total_states, source_state_ids,
                                source_init_dists, n_source, source_states,
                                gpu_block_size):
    """Initialize distance storage based on the selected mode.

    Returns (d_block_entries, d_block_span, d_dist, d_span_dist,
             hash_capacity, hash_mask, d_state_table, gpu_block_size,
             use_sparse, use_managed_ptrs, dist_ptr, span_ptr).
    """
    use_managed_ptrs = False
    dist_ptr = None
    span_ptr = None

    if use_block:
        (d_block_entries, d_block_span, d_dist, d_span_dist,
         hash_capacity, hash_mask, d_state_table,
         gpu_block_size) = _init_v2_block_storage(
            n_cells, n_dirs, n_span_bins, n_heights, spc,
            source_state_ids, source_init_dists, n_source, gpu_block_size)
    elif use_sparse:
        (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
         d_block_entries, d_block_span) = _init_v2_sparse_storage(
            total_states, n_cells, source_state_ids,
            source_init_dists, n_source)
    elif use_managed:
        (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
         d_block_entries, d_block_span,
         dist_ptr, span_ptr) = _init_v2_managed_storage(
            total_states, source_states)
        use_sparse = False  # kernel uses dense code path
        use_managed_ptrs = True
    else:
        (d_dist, d_span_dist, hash_capacity, hash_mask, d_state_table,
         d_block_entries, d_block_span) = _init_v2_dense_storage(
            total_states, source_states)

    return (d_block_entries, d_block_span, d_dist, d_span_dist,
            hash_capacity, hash_mask, d_state_table, gpu_block_size,
            use_sparse, use_managed_ptrs, dist_ptr, span_ptr)


# ============================================================================
# Main entry point
# ============================================================================

def constrained_sssp_raster_gpu_v2(
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
    sparse: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find constrained shortest path on GPU with tower placement (v2 persistent kernel).

    Uses a single-launch persistent cooperative CUDA kernel with custom grid
    barrier (Blackwell-safe). Tracks tower placements via atomic-append
    TowerRecord buffer instead of a full predecessor array.

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
            If None, defaults to single height with 0 premium.
            Must be sorted in same order as tower_heights (descending).
        n_heights: number of tower height classes (default 1)
        exclude_mask: optional uint8 mask (1=traversable, 0=blocked)
        dem: optional float32 DEM array for clearance checking and gradient.
            Shape must match raster (rows, cols). When provided, catenary
            clearance is checked along each span during tower placement and
            gradient penalty is applied to traversal costs.
        obstacle_heights: optional float32 array of obstacle heights (meters).
            Shape must match raster. Used for clearance: conductor must clear
            ground + obstacle by min_clearance.
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
            When None, single-pixel tower cost is used (backward compat).
        area_offset_starts: optional int32 (n_dirs*n_dirs,) start index per
            direction pair in area_offsets.
        area_offset_counts: optional int32 (n_dirs*n_dirs,) count per
            direction pair in area_offsets.
        threads_per_block: CUDA threads per block
        margin: early termination margin (default 1.00001)
        max_tower_records: max tower records to allocate (default 2M)
        sparse: memory mode for distance storage.
            "auto": use block-sparse when dense arrays would exceed
                70% of available VRAM, otherwise use dense arrays.
            "block": force block-sparse mode (32 slots per cell,
                ~10x smaller than dense for typical configs).
            "sparse": force sparse hash table mode (legacy).
            True: force sparse hash table mode (for large rasters).
            False/"dense": force dense array mode (default behavior,
                fastest for small rasters that fit in VRAM).

    Returns:
        tuple: (path_cell_indices uint32[], tower_cell_indices uint32[],
                tower_heights float32[])

    Raises:
        RuntimeError: if CUDA GPU not available
        ValueError: if source or target is on forbidden cell
        MemoryError: if state space exceeds GPU VRAM
    """
    # Validate inputs and apply defaults
    (source_cell, target_cell, height_premiums, n_heights,
     tower_heights, angle_cost_lut, raster) = _validate_v2_inputs(
        raster, source_row, source_col, target_row, target_col,
        height_premiums, n_heights, tower_heights,
        angle_cost_lut, tower_terrain_costs, tower_angle_costs,
        exclude_mask)

    rows, cols = raster.shape
    n_cells = rows * cols
    n_dirs = len(steps)
    max_cost = int(np.iinfo(np.uint16).max)

    spc = n_dirs * n_span_bins * n_heights
    total_states = n_cells * spc
    min_span_bin = int(min_span / span_bin_size)

    # Determine allocation mode: "dense", "managed", "sparse", or "block"
    use_sparse, use_managed, use_block, storage_mode = _resolve_v2_storage_mode(
        sparse, rows, cols, n_dirs, n_span_bins, n_heights)

    # Prepare step lookup tables
    from pyorps.utils.traversal_gpu import prepare_step_lookup_tables
    steps_arr, intermediates_lut, n_intermediates, cost_factors = \
        prepare_step_lookup_tables(steps)
    max_inter_cols = intermediates_lut.shape[1]

    # Compute delta
    delta = _compute_constrained_delta(raster, cost_factors,
                                       tower_terrain_costs)

    # Buffer size for queues
    # Queue buffer: frontier is typically <1% of state space.
    # For managed mode, keep queues small to fit in VRAM.
    buf_size = (max(n_cells * 4, 1 << 20) if use_managed
                else min(total_states, max(n_cells * n_dirs * 2, 1 << 20)))

    # Shared memory
    smem_bytes = _compute_v2_smem(n_dirs, n_heights, max_inter_cols)

    # Compute source states (needed for both dense and sparse)
    source_states = []
    for d in range(n_dirs):
        for hc in range(n_heights):
            st = pack_state(source_cell, d, 0, hc, spc, n_span_bins, n_heights)
            source_states.append((st, float(height_premiums[hc])))
    source_state_ids = np.array([s for s, _ in source_states], dtype=np.int64)
    source_init_dists = np.array([p for _, p in source_states], dtype=np.float32)
    n_source = len(source_state_ids)

    # Initialize distance storage (dense, sparse, or block-sparse)
    gpu_block_size = 64  # default, overridden in block-sparse mode
    (d_block_entries, d_block_span, d_dist, d_span_dist,
     hash_capacity, hash_mask, d_state_table, gpu_block_size,
     use_sparse, use_managed_ptrs, dist_ptr,
     span_ptr) = _init_v2_distance_storage(
        use_block, use_sparse, use_managed,
        n_cells, n_dirs, n_span_bins, n_heights,
        spc, total_states, source_state_ids,
        source_init_dists, n_source, source_states,
        gpu_block_size)

    # Upload data to GPU
    gpu_data = _upload_v2_gpu_data(
        raster, steps_arr, cost_factors, intermediates_lut,
        n_intermediates, angle_cost_lut, angle_valid_lut,
        step_distances, tower_terrain_costs, tower_angle_costs,
        height_premiums, tower_heights, dem, obstacle_heights,
        area_offsets, area_offset_starts, area_offset_counts)

    # Allocate queues (int64)
    d_queue_a = cp.empty(buf_size, dtype=cp.int64)
    d_queue_b = cp.empty(buf_size, dtype=cp.int64)
    d_settled = cp.empty(buf_size, dtype=cp.int64)
    d_pending = cp.empty(buf_size, dtype=cp.int64)

    # Seed initial frontier
    d_queue_a[:n_source] = cp.asarray(source_state_ids)

    # Control buffer
    d_control = cp.zeros(_CTL_SIZE, dtype=cp.int32)
    d_control[_CTL_COUNT_A] = n_source

    # Tower records buffer
    tower_record_bytes = max_tower_records * 24  # sizeof(TowerRecord) = 24
    d_tower_records = cp.zeros(tower_record_bytes, dtype=cp.uint8)

    # Target for early termination
    target_idx_arr = np.array([target_cell], dtype=np.int32)
    d_targets = cp.asarray(target_idx_arr)

    # Compile and launch kernel
    _ensure_cuda_path()
    kernel = _get_v2_kernel(
        "constrained_persistent",
        cooperative=True,
        block_size=gpu_block_size if use_block else 64)

    tpb = threads_per_block
    n_sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    max_blocks = n_sms * 2

    # Prepare kernel pointers
    (dem_ptr, obs_ptr, area_off_ptr, area_starts_ptr,
     area_counts_ptr, state_table_ptr, block_entries_ptr,
     block_span_ptr) = _prepare_v2_kernel_ptrs(
        gpu_data, dem, obstacle_heights, area_offsets,
        use_sparse, use_block, d_state_table,
        d_block_entries, d_block_span)

    kernel(
        (max_blocks,), (tpb,),
        (gpu_data['d_raster'], np.int32(rows), np.int32(cols), np.int32(max_cost),
         gpu_data['d_steps'], gpu_data['d_cost_factors'],
         gpu_data['d_inter_lut'], gpu_data['d_n_inter'],
         np.int32(n_dirs), np.int32(max_inter_cols),
         gpu_data['d_angle_cost'], gpu_data['d_angle_valid'],
         gpu_data['d_step_dist'],
         gpu_data['d_tower_terrain'], gpu_data['d_tower_angle'],
         gpu_data['d_height_premiums'], gpu_data['d_tower_heights'],
         np.int32(n_heights),
         np.int32(n_span_bins), np.float32(span_bin_size),
         np.int32(min_span_bin),
         np.int64(spc), np.int64(total_states),
         d_dist, d_span_dist,
         np.float32(delta), np.int32(100),  # max_light_iters
         d_targets, np.int32(1), np.float32(margin),
         d_control, d_queue_a, d_queue_b, d_settled, d_pending,
         np.int32(buf_size),
         d_tower_records, np.int32(max_tower_records),
         dem_ptr, obs_ptr,
         np.float32(cell_size), np.float32(conductor_weight_per_m),
         np.float32(conductor_tension), np.float32(min_clearance),
         np.float32(max_gradient_pct), np.float32(gradient_scale),
         area_off_ptr, area_starts_ptr, area_counts_ptr,
         state_table_ptr, np.int32(hash_mask),
         np.int32(hash_capacity),
         block_entries_ptr, block_span_ptr,
         np.int32(storage_mode)),
        shared_mem=smem_bytes)

    cp.cuda.Stream.null.synchronize()

    # Check for overflow
    control_cpu = d_control.get()
    overflow_count = int(control_cpu[_CTL_QUEUE_OVERFLOW])
    if overflow_count > 0:
        warnings.warn(
            f"GPU constrained SSSP had {overflow_count} queue overflow events. "
            f"Results may be suboptimal. Consider increasing buf_size.")

    n_tower_records = min(int(control_cpu[_CTL_TOWER_COUNT]), max_tower_records)

    # Find best target state
    target_result = _find_v2_best_target(
        use_block, use_sparse, d_block_entries, d_dist,
        d_state_table, target_cell, spc, gpu_block_size,
        n_span_bins, n_heights)

    if target_result is None:
        warnings.warn("GPU constrained SSSP v2 found no path to target")
        return (np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.float32))

    best_dist, best_state, dist_cpu_dict = target_result

    # Transfer only the used tower records to CPU (not the full buffer)
    tower_record_dtype = np.dtype([
        ('state', np.int64),
        ('pred_state', np.int64),
        ('span_dist', np.float16),
        ('tower_height', np.float16),
        ('_pad', np.uint8, 4),  # padding to 24 bytes
    ])
    if n_tower_records > 0:
        # Only download the used portion (n_tower_records * 24 bytes)
        raw_bytes = d_tower_records[:n_tower_records * 24].get()
        tower_records_cpu = np.frombuffer(
            raw_bytes, dtype=tower_record_dtype)
    else:
        tower_records_cpu = np.zeros(0, dtype=tower_record_dtype)

    # Download distance data for reconstruction
    dist_cpu = _download_v2_for_reconstruction(
        use_block, use_sparse, d_block_entries, d_dist,
        d_state_table, spc, n_span_bins, n_heights,
        gpu_block_size, dist_cpu_dict)

    result = _reconstruct_from_tower_records(
        tower_records_cpu, n_tower_records, best_state,
        source_cell, spc, n_span_bins, n_heights, n_dirs,
        cols, steps, dist_cpu=dist_cpu)

    # Free managed memory if used (not tracked by CuPy pool)
    if use_managed_ptrs:
        try:
            cp.cuda.runtime.free(dist_ptr)
            cp.cuda.runtime.free(span_ptr)
        except Exception:
            pass

    return result
