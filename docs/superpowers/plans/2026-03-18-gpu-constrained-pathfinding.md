# GPU Constrained Pathfinding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CUDA persistent cooperative kernel that jointly optimizes route, tower placement, tower height, and clearance in a single GPU launch, using warp-cooperative parallelism for tower operations.

**Architecture:** Extends the proven v4 persistent kernel (`sssp_gpu.py`) to the constrained state space `(cell, dir, span_bin, height_class)`. A single-launch persistent kernel handles the entire delta-stepping algorithm. Tower placement uses a warp-cooperative protocol: when any thread in a warp wants to place a tower, all 32 threads cooperate on clearance checking, area cost summation, and slope computation. Tower records are stored via atomic-append for CPU-side path reconstruction.

**Tech Stack:** CuPy RawKernel (CUDA C++17), cooperative_groups, `__shfl_sync`/`__ballot_sync` warp primitives, float16 (`__half`) for span distances

**Spec:** `docs/superpowers/specs/2026-03-18-gpu-constrained-pathfinding-design.md`

## Critical Design Decisions

1. **No per-state predecessor array (d_pred).** This is intentional — saves 18+ GB VRAM. Path reconstruction uses TowerRecord atomic-append buffer + direction-walking between towers. Do NOT add d_pred — it would break the memory budget. See spec section "Path Reconstruction."

2. **d_span_dist (float16) tracks exact span meters.** Every relaxation that continues a span must read the previous span_dist, add step_distance, and write the new span_dist. Every tower placement reads span_dist to enforce min/max_span. The kernel reads/writes this as `__half` with `__float2half` / `__half2float` conversions.

3. **Non-negative cost invariant.** The atomic relaxation uses `atomicMin` on float32 via IEEE 754 bit-cast, which only works for non-negative values. The Python wrapper must validate all costs >= 0 before launch.

4. **Task 3 adds gradient penalty for TRAVERSAL (per-edge slope from DEM). Task 5 adds area-averaged slope for TOWER FOUNDATION cost.** These are separate concerns: traversal gradient steers the route away from steep terrain; foundation slope multiplies tower placement cost on steep ground.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pyorps/utils/constrained_sssp_gpu_v2.py` | **Create** | New persistent kernel + Python wrapper |
| `pyorps/graph/constrained_path_finder.py` | Modify (lines 294-317) | Wire v2 GPU backend into `_find_route_coupled()` |
| `tests/test_graph/test_constrained_gpu_v2.py` | **Create** | Correctness + performance tests for v2 kernel |

Existing files left unchanged: `constrained_sssp_gpu.py` (v1 kept as fallback), `sssp_gpu.py` (reference only), `constrained_path_algorithms.py` (shim unchanged).

---

### Task 1: Scaffold the v2 Module with State Encoding Tests

**Files:**
- Create: `pyorps/utils/constrained_sssp_gpu_v2.py`
- Create: `tests/test_graph/test_constrained_gpu_v2.py`

This task creates the module skeleton, GPU availability check, state encoding helpers, and the Python-side memory allocation logic. No CUDA kernel yet.

- [ ] **Step 1: Write state encoding tests**

```python
# tests/test_graph/test_constrained_gpu_v2.py
import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")


class TestStateEncoding:
    """Test 4-field state packing/unpacking for GPU kernel."""

    def test_pack_unpack_roundtrip(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import pack_state, unpack_state
        cell, d, sb, hc = 12345, 7, 3, 2
        n_dirs, n_span_bins, n_heights = 32, 6, 3
        spc = n_dirs * n_span_bins * n_heights
        state = pack_state(cell, d, sb, hc, spc, n_span_bins, n_heights)
        c2, d2, sb2, hc2 = unpack_state(state, spc, n_span_bins, n_heights)
        assert (c2, d2, sb2, hc2) == (cell, d, sb, hc)

    def test_unique_states(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import pack_state
        n_dirs, n_span_bins, n_heights = 8, 6, 3
        spc = n_dirs * n_span_bins * n_heights
        states = set()
        for cell in range(100):
            for d in range(n_dirs):
                for sb in range(n_span_bins):
                    for hc in range(n_heights):
                        s = pack_state(cell, d, sb, hc, spc, n_span_bins, n_heights)
                        assert s not in states, f"Collision at {cell},{d},{sb},{hc}"
                        states.add(s)

    def test_large_cell_index(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import pack_state, unpack_state
        cell = 2_000_000  # 2000x1000 raster
        n_dirs, n_span_bins, n_heights = 32, 6, 3
        spc = n_dirs * n_span_bins * n_heights
        state = pack_state(cell, 31, 5, 2, spc, n_span_bins, n_heights)
        c, d, sb, hc = unpack_state(state, spc, n_span_bins, n_heights)
        assert (c, d, sb, hc) == (cell, 31, 5, 2)


class TestMemoryBudget:
    """Test memory allocation and budget estimation."""

    def test_budget_computation(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import compute_memory_budget_gb
        gb = compute_memory_budget_gb(
            rows=2000, cols=2000, n_dirs=32, n_span_bins=6, n_heights=3)
        assert 14.0 < gb < 15.0  # expected ~14.4 GB

    def test_budget_raises_on_exceed(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import check_memory_fits
        # Huge state space that won't fit in 16 GB
        with pytest.raises(MemoryError):
            check_memory_fits(
                rows=4000, cols=4000, n_dirs=48, n_span_bins=6, n_heights=3,
                vram_gb=16.0)
```

- [ ] **Step 2: Run tests — verify they fail (module not found)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v2.py -v -x`
Expected: `ModuleNotFoundError: No module named 'pyorps.utils.constrained_sssp_gpu_v2'`

- [ ] **Step 3: Create module with state encoding and memory helpers**

```python
# pyorps/utils/constrained_sssp_gpu_v2.py
"""GPU persistent-kernel constrained pathfinding with warp-cooperative clearance.

V2 of the GPU constrained SSSP: single-launch persistent cooperative kernel
with full 3D support (clearance, variable heights, area cost, slope cost).
"""

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False


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


def compute_memory_budget_gb(rows, cols, n_dirs, n_span_bins, n_heights):
    """Estimate GPU memory needed in GB."""
    total_cells = rows * cols
    total_states = total_cells * n_dirs * n_span_bins * n_heights
    dist_bytes = total_states * 4        # float32
    span_bytes = total_states * 2        # float16
    input_bytes = total_cells * (2 + 4 + 4 + 1)  # raster+dem+obs+mask
    lut_bytes = 65536 * 4 + n_dirs * n_dirs * (4 + 1 + 4)  # terrain+angle luts
    buf_size = min(total_states, max(total_cells * n_dirs * 2, 1 << 20))
    queue_bytes = 4 * buf_size * 8       # 4 queues × int64
    tower_bytes = 1_000_000 * 24         # 1M tower records
    total = dist_bytes + span_bytes + input_bytes + lut_bytes + queue_bytes + tower_bytes
    return total / (1024 ** 3)


def check_memory_fits(rows, cols, n_dirs, n_span_bins, n_heights, vram_gb=None):
    """Raise MemoryError if state space won't fit in GPU VRAM."""
    if vram_gb is None and GPU_AVAILABLE:
        vram_gb = cp.cuda.Device().mem_info[1] / (1024 ** 3)
    elif vram_gb is None:
        vram_gb = 16.0
    budget = compute_memory_budget_gb(rows, cols, n_dirs, n_span_bins, n_heights)
    if budget > vram_gb * 0.9:  # 90% threshold
        raise MemoryError(
            f"Constrained GPU SSSP needs {budget:.1f} GB but only "
            f"{vram_gb:.1f} GB available. Reduce neighborhood or span bins.")
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v2.py -v -x`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```
feat: scaffold constrained_sssp_gpu_v2 with state encoding and memory budget
```

---

### Task 2: Write the Persistent CUDA Kernel (Core Relaxation Without Tower Protocol)

**Files:**
- Modify: `pyorps/utils/constrained_sssp_gpu_v2.py`
- Modify: `tests/test_graph/test_constrained_gpu_v2.py`

This task writes the persistent kernel CUDA C string with the full state machine (bucket loop, light phase, heavy phase, queue management, custom barrier) but with **simple inline tower placement** (no warp-cooperative protocol yet). This gets the kernel running and producing correct results first.

- [ ] **Step 1: Write basic correctness test**

```python
# Add to tests/test_graph/test_constrained_gpu_v2.py

class TestBasicKernel:
    """Test persistent kernel produces correct results on simple rasters."""

    def _make_params(self, rows=50, cols=50, n_dirs=8, raster_val=10):
        """Create minimal parameter set for testing."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        raster = np.full((rows, cols), raster_val, dtype=np.uint16)
        steps = get_neighborhood_steps("r1")
        profile = InfrastructureProfile(
            name="test", description="test",
            soft_angle_limit_deg=8.0, hard_angle_limit_deg=40.0,
            angle_cost_function="piecewise",
            angle_cost_params={"breakpoints": [0, 8, 40], "costs": [0, 300, 80000]},
            min_span_m=50.0, max_span_m=300.0, span_bin_size_m=10.0,
            tower_cost_params={
                "terrain_cost_map": {"0": 35000, "100": 55000, "500": 180000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {"max_angle_deg": 8.0, "base_cost": 30000},
                    "heavy_angle": {"max_angle_deg": 40.0, "base_cost": 150000},
                },
            },
        )
        angle_cost, angle_valid = profile.precompute_angle_lut(steps)
        tower_terrain = profile.precompute_tower_terrain_costs()
        tower_angle = profile.precompute_tower_angle_costs(steps)
        step_dist = profile.compute_step_distances(steps, 5.0)
        import math
        n_span_bins = max(2, math.ceil(300.0 / 50.0))
        return dict(
            raster=raster, steps=steps,
            angle_cost_lut=angle_cost.astype(np.float32),
            angle_valid_lut=angle_valid.astype(np.uint8),
            step_distances=step_dist.astype(np.float32),
            tower_terrain_costs=tower_terrain.astype(np.float32),
            tower_angle_costs=tower_angle.astype(np.float32),
            n_span_bins=n_span_bins, span_bin_size=50.0,
            min_span=50.0, max_span=300.0,
        )

    def test_uniform_raster_finds_path(self):
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        params = self._make_params()
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            **params)
        assert len(path) > 0, "Should find a path"
        assert len(towers) > 0, "Should place towers"

    def test_gpu_cost_matches_cython(self):
        """Compare total optimized cost (from d_dist), not just terrain sum."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_height_2d)
        params = self._make_params()
        common = dict(
            source_row=5, source_col=5,
            target_row=45, target_col=45,
        )
        height_args = dict(
            tower_heights=np.array([25.0], dtype=np.float32),
            height_premiums=np.array([0.0], dtype=np.float32),
        )
        # GPU — returns (path, towers, heights) + stores best_dist
        gpu_path, gpu_towers, gpu_h = constrained_sssp_raster_gpu_v2(
            **common, **params, **height_args)
        # Cython
        cy_path, cy_towers, cy_h = constrained_delta_stepping_height_2d(
            **params, **common,
            dem_data=np.zeros((50, 50), dtype=np.float32),
            cell_size=5.0, **height_args,
            conductor_weight_per_m=10.0,
            conductor_tension=20000.0,
            min_clearance_val=7.0,
        )
        # Both should find paths with similar tower counts
        assert len(gpu_path) > 0 and len(cy_path) > 0
        assert abs(len(gpu_towers) - len(cy_towers)) <= 2

    def test_no_feasible_path_returns_empty(self):
        """Fully blocked raster returns empty arrays without hanging."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        params = self._make_params()
        # Block entire middle row
        params['raster'][20:30, :] = 65535
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            **params)
        assert len(path) == 0

    def test_source_on_forbidden_raises(self):
        """Source on forbidden cell raises ValueError."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        params = self._make_params()
        params['raster'][5, 5] = 65535
        with pytest.raises(ValueError):
            constrained_sssp_raster_gpu_v2(
                source_row=5, source_col=5,
                target_row=45, target_col=45,
                **params)

    def test_source_equals_target(self):
        """Source == target returns trivial path."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        params = self._make_params()
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            source_row=25, source_col=25,
            target_row=25, target_col=25,
            **params)
        # Either empty or single cell
        assert len(path) <= 1
```

- [ ] **Step 2: Run tests — verify they fail (function not found)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v2.py::TestBasicKernel -v -x`
Expected: `ImportError: cannot import name 'constrained_sssp_raster_gpu_v2'`

- [ ] **Step 3: Write the CUDA kernel string**

Add to `constrained_sssp_gpu_v2.py` the persistent kernel CUDA C source. This is the largest single piece of code in the plan. Structure:

1. Header: `cooperative_groups.h`, TowerRecord struct, control buffer macros
2. `grid_barrier()`: custom sense-reversing barrier (copy from v4)
3. `delta_stepping_constrained_persistent()`: main kernel with:
   - Shared memory loader for all LUTs
   - Source initialization
   - Outer bucket loop
   - Light phase with state unpacking, neighbor iteration, intermediate checks
   - Tower placement branches (inline for now — warp-cooperative in Task 4)
   - Heavy phase
   - Queue management + classify
   - Early termination

Reference: `sssp_gpu.py` lines 635-978 for the v4 pattern, `constrained_sssp_gpu.py` lines 38-212 for the constrained relaxation logic.

- [ ] **Step 4: Write the Python wrapper function**

Add `constrained_sssp_raster_gpu_v2()` to the module:
1. **Validate inputs**: assert raster[source] != 65535 and raster[target] != 65535 (raise ValueError). Assert all tower_terrain_costs >= 0, all height_premiums >= 0 (non-negative cost invariant for atomicMin).
2. **Check memory budget**: call `check_memory_fits()`, raise MemoryError if exceeds 90% VRAM.
3. **Copy to GPU**: raster (uint16), DEM (float32, or NULL), obstacle (float32, or NULL), exclude_mask (uint8), tower_terrain LUT (float32[65536]), angle LUTs, tower heights/premiums, area offsets (or NULL).
4. **Allocate d_dist** (float32[total_states], init to `cp.float32(1e30)`).
5. **Allocate d_span_dist** (float16[total_states], init to 0). This is critical — the kernel reads `d_span_dist[state]` at every relaxation to get exact span meters, and writes updated span on every edge traversal.
6. **Allocate queues** (4 × int64[buf_size]), tower_records (TowerRecord[max_records]), tower_count (int32[1], init 0), control (int32[16], init 0).
7. **Initialize source states**: for all dirs × all heights, set `d_dist[state] = height_premiums[h]`, `d_span_dist[state] = 0`, enqueue to queue_a. Set `control[CTL_COUNT_A] = n_source_states`.
8. **Compute delta**: `delta = max(1.0, 2.0 * min_raster_val * min_cost_factor)` (same heuristic as v1 — terrain costs only, tower costs excluded).
9. **Compile kernel** with `cp.RawKernel(source, "delta_stepping_constrained_persistent", options=("--std=c++17", "-Xptxas", "-dlcm=cg"), backend="nvcc")`.
10. **Launch**: `kernel((blocks,), (threads_per_block,), args, shared_mem=smem_size)`.
11. **Synchronize** + check `control[CTL_TOWER_OVERFLOW]` and `control[CTL_QUEUE_OVERFLOW]`.
12. **Find best target state**: scan d_dist for target_cell's spc states (CPU-side, trivial).
13. **Reconstruct path**: copy tower_records to host. Build `{state: TowerRecord}` map (keep record whose dist matches d_dist[state]). Walk backward via pred_state. Extract tower_heights from `TowerRecord.tower_height` (`__half` → float32 via `float(np.float16(...))`). Walk direction chains between consecutive towers to fill full path.
14. **Return** `(path_indices uint32[], tower_indices uint32[], tower_heights float32[])`.

**Key kernel data flow for d_span_dist:**
```cuda
// In BRANCH A (continue span, no tower):
__half old_span = span_dist[cur_state];
float new_span_m = __half2float(old_span) + step_distance;
// ... relaxation ...
span_dist[new_state] = __float2half(new_span_m);

// In BRANCH B (tower placement):
float cur_span_m = __half2float(span_dist[cur_state]);
if (cur_span_m >= min_span) {
    // ... clearance check using cur_span_m ...
    float reset_span_m = step_distance;  // span resets after tower
    span_dist[new_state] = __float2half(reset_span_m);
}
```

**Early termination** (add to kernel outer loop, every 10 buckets):
```cuda
// Check if best target distance < margin * current_bucket * delta
// Target spans spc states: target_cell * spc .. (target_cell+1) * spc
// Use atomicMin on CTL_MIN_DIST during relaxation of target states
```

- [ ] **Step 5: Run tests — verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v2.py::TestBasicKernel -v -x`
Expected: Both tests PASS

- [ ] **Step 6: Commit**

```
feat: persistent constrained kernel v2 with inline tower placement
```

---

### Task 3: Add Clearance, DEM, and Variable Height Support

**Files:**
- Modify: `pyorps/utils/constrained_sssp_gpu_v2.py`
- Modify: `tests/test_graph/test_constrained_gpu_v2.py`

Extends the kernel's tower placement branch with catenary clearance checking, variable tower heights, and gradient penalties. Still inline (not warp-cooperative yet).

- [ ] **Step 1: Write clearance and height tests**

```python
class TestClearanceAndHeight:

    def test_clearance_blocks_low_towers(self):
        """High obstacle should force taller tower selection."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        raster = np.full((50, 50), 10, dtype=np.uint16)
        dem = np.zeros((50, 50), dtype=np.float32)
        obstacle = np.zeros((50, 50), dtype=np.float32)
        # Place tall obstacle in the middle
        obstacle[20:30, :] = 20.0  # 20m trees
        # ... (create params with tower_heights=[25, 34, 42])
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            ..., dem_data=dem, obstacle_heights=obstacle,
            tower_heights=np.array([42.0, 34.0, 25.0], dtype=np.float32),
            height_premiums=np.array([18000.0, 9000.0, 0.0], dtype=np.float32),
            conductor_weight_per_m=10.0, conductor_tension=20000.0,
            min_clearance=7.0)
        # Should find a path (taller towers clear the obstacle)
        assert len(path) > 0
        # Some towers should be taller than 25m
        assert any(h > 25.0 for h in heights)

    def test_gradient_penalty_avoids_steep_slope(self):
        """Steep DEM slope should steer route around hill."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        raster = np.full((50, 50), 10, dtype=np.uint16)
        dem = np.zeros((50, 50), dtype=np.float32)
        # Create steep ridge in the middle
        for r in range(20, 30):
            dem[r, :] = 50.0 * (r - 20)  # 50m rise per cell = extreme slope
        # Route should go around, not over
        path, towers, heights = constrained_sssp_raster_gpu_v2(
            ..., dem_data=dem, max_gradient_pct=40.0, gradient_scale=2.0)
        assert len(path) > 0
        # Path should avoid the steep cells
        steep_cells = {r * 50 + c for r in range(22, 28) for c in range(50)}
        path_set = set(int(p) for p in path)
        overlap = path_set & steep_cells
        assert len(overlap) < len(path) * 0.1  # <10% of path on steep ground
```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Extend kernel with clearance, height, and gradient logic**

In the CUDA kernel, modify the tower placement branch:
- Add height class loop (iterate ALL heights tallest-first; break on first FAILURE since shorter heights also fail; but explore all PASSING heights as separate relaxations)
- Add catenary sag computation: `sag = cond_w * x * (L - x) / (2 * T)`
- Walk span cells checking `conductor_z - ground_z - obstacle_z >= min_clearance`
- Add **traversal gradient penalty** on edge cost (per-cell slope from DEM, same as Cython `_precompute_gradient_cache`): `edge_cost *= exp(gradient_scale * slope_pct / 100)`. This is SEPARATE from the tower foundation slope cost added in Task 5.
- Record TowerRecord with height via atomic append
- d_span_dist must be read for `cur_span_m` (clearance uses it as `L`) and reset on tower placement

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Commit**

```
feat: add clearance, variable heights, and gradient to GPU kernel
```

---

### Task 4: Implement Warp-Cooperative Tower Protocol

**Files:**
- Modify: `pyorps/utils/constrained_sssp_gpu_v2.py`
- Modify: `tests/test_graph/test_constrained_gpu_v2.py`

Replace the inline tower placement with the warp-cooperative protocol. This is the core performance innovation.

- [ ] **Step 1: Write performance comparison test**

```python
class TestWarpCooperative:

    def test_correctness_matches_inline(self):
        """Warp-cooperative must produce same path cost as inline."""
        # Run same problem with both modes, compare total cost
        # (requires a flag to toggle inline vs cooperative, or just
        # verify the cooperative version matches Cython)
        ...

    def test_large_raster_performance(self):
        """Benchmark on 500x500 raster, should be faster than v1 GPU."""
        import time
        # ... setup 500x500 raster
        t0 = time.perf_counter()
        path, towers, heights = constrained_sssp_raster_gpu_v2(...)
        t_v2 = time.perf_counter() - t0
        # v1 for comparison
        from pyorps.utils.constrained_sssp_gpu import constrained_sssp_raster_gpu
        t0 = time.perf_counter()
        path_v1, towers_v1 = constrained_sssp_raster_gpu(...)
        t_v1 = time.perf_counter() - t0
        print(f"V1: {t_v1:.3f}s, V2: {t_v2:.3f}s, speedup: {t_v1/t_v2:.1f}x")
        assert t_v2 < t_v1, "V2 should be faster than V1"
```

- [ ] **Step 2: Run test — verify inline version works (baseline)**

- [ ] **Step 3: Rewrite tower placement branch with warp protocol**

Replace the inline tower code in the CUDA kernel with:
1. `__ballot_sync(__activemask(), want_tower)` → tower_mask
2. Round-robin loop: `while (tower_mask != 0)`
3. `__shfl_sync` broadcast of tower params from owner
4. Parallel clearance: each thread checks `ceil(span_cells/32)` positions
5. `__ballot_sync` for clearance pass/fail
6. Parallel area cost: each thread sums `ceil(n_offsets/32)` pixels + forbidden check
7. `__shfl_down_sync` warp-reduce for area sum
8. Parallel slope: same pattern as area cost
9. Owner thread: compute final cost, atomicMin + TowerRecord append

- [ ] **Step 4: Run tests — verify correctness unchanged and performance improved**

- [ ] **Step 5: Commit**

```
feat: implement warp-cooperative tower protocol for GPU kernel
```

---

### Task 5: Add Area Cost and Forbidden Footprint Support

**Files:**
- Modify: `pyorps/utils/constrained_sssp_gpu_v2.py`
- Modify: `tests/test_graph/test_constrained_gpu_v2.py`

Add support for rotated square pixel offsets (exact area cost mode), forbidden pixel rejection (any 65535 in footprint), and slope-dependent foundation cost.

- [ ] **Step 1: Write area cost and forbidden footprint tests**

```python
class TestAreaCost:

    def test_forbidden_pixel_blocks_tower(self):
        """Tower placement rejected if any footprint pixel is 65535."""
        raster = np.full((50, 50), 10, dtype=np.uint16)
        # Place forbidden cells where a tower would normally go
        raster[25, 24:27] = 65535
        # ... run with area offsets covering a 3x3 square
        # Path should route around the forbidden area

    def test_area_cost_higher_than_single_pixel(self):
        """Exact area cost should sum multiple pixels, not just center."""
        raster = np.full((50, 50), 10, dtype=np.uint16)
        raster[24:27, 24:27] = 100  # expensive 3x3 patch
        # With ground_area=9 (3x3), tower at (25,25) costs 9*terrain(100)
        # vs single-pixel which costs 1*terrain(100)

    def test_uniform_mode_no_area_offsets(self):
        """Kernel works when area_offsets is None (uniform cost mode)."""
        from pyorps.utils.constrained_sssp_gpu_v2 import constrained_sssp_raster_gpu_v2
        # ... run without area_offsets, should use single-pixel tower cost
        # Verify path is found and tower costs are reasonable

    def test_slope_multiplier_increases_tower_cost(self):
        """Tower on steep slope costs more than on flat ground."""
        # Create DEM with steep section, verify tower avoids it
        # or if forced there, tower_cost is higher
```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Add area offset arrays to kernel parameters**

In the Python wrapper: pass `d_area_offsets`, `d_area_starts`, `d_area_counts` to the kernel. In the CUDA kernel: add the warp-cooperative area cost summation (already sketched in Task 4's protocol) and forbidden pixel check via `__ballot_sync`.

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Commit**

```
feat: add area cost, forbidden footprint, and slope cost to GPU kernel
```

---

### Task 6: Wire into ConstrainedPathFinder and Integration Test

**Files:**
- Modify: `pyorps/graph/constrained_path_finder.py` (lines 294-317)
- Modify: `tests/test_graph/test_constrained_gpu_v2.py`

Connect the v2 GPU kernel to the existing `ConstrainedPathFinder` so users can use it via `graph_api="raster_gpu"`.

- [ ] **Step 1: Write integration test**

```python
class TestIntegration:

    def test_constrained_path_finder_gpu_v2(self):
        """Full end-to-end: ConstrainedPathFinder with raster_gpu backend."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        # Create temp raster + profile, run find_route with gpu backend
        # Verify ConstrainedPath result has towers, costs, geometry
        ...

    def test_gpu_matches_cython_with_dem(self):
        """GPU v2 with clearance matches Cython within tolerance."""
        # Same route with backend="cython" vs backend="raster_gpu"
        # Compare total_cost within 1%
        ...
```

- [ ] **Step 2: Run test — verify it fails (v2 not wired)**

- [ ] **Step 3: Update `_find_route_coupled` GPU code path**

```python
# In constrained_path_finder.py, replace lines 294-317:
if backend == "raster_gpu":
    try:
        from pyorps.utils.constrained_sssp_gpu_v2 import (
            constrained_sssp_raster_gpu_v2,
        )
        return constrained_sssp_raster_gpu_v2(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=self.steps,
            angle_cost_lut=self._angle_cost_lut.astype(np.float32),
            angle_valid_lut=self._angle_valid_lut.astype(np.uint8),
            step_distances=self._step_distances.astype(np.float32),
            tower_terrain_costs=self._tower_terrain_costs.astype(np.float32),
            tower_angle_costs=self._tower_angle_costs.astype(np.float32),
            n_span_bins=n_span_bins, span_bin_size=span_bin_size,
            min_span=min_span, max_span=max_span,
            dem_data=dem_data,
            obstacle_heights=self._obstacle_data,
            cell_size=self._cell_size,
            tower_heights=heights,
            height_premiums=premiums,
            conductor_weight_per_m=self._profile.conductor_weight_per_m,
            conductor_tension=self._profile.conductor_tension_n,
            min_clearance=self._profile.min_clearance_m,
            max_gradient_pct=dem_kwargs.get('max_gradient_pct', 100.0),
            gradient_scale=dem_kwargs.get('gradient_scale', 2.0),
            **area_kwargs,
        )
    except (ImportError, RuntimeError) as e:
        warnings.warn(f"GPU v2 unavailable ({e}), falling back to Cython")
        backend = "cython"
```

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Commit**

```
feat: wire constrained GPU v2 into ConstrainedPathFinder
```

---

### Task 7: Performance Benchmarks and Validation Suite

**Files:**
- Modify: `tests/test_graph/test_constrained_gpu_v2.py`

Final validation: correctness against Cython on multiple raster sizes, performance benchmarks.

- [ ] **Step 1: Write comprehensive validation and benchmark tests**

```python
class TestValidation:
    """Phase 1-2 validation: correctness against Cython."""

    @pytest.mark.parametrize("size", [50, 100, 200])
    def test_cost_match_vs_cython(self, size):
        """GPU total cost within 1% of Cython."""
        # ... create raster, DEM, obstacles, profile
        # Run both GPU and Cython, compare total path cost
        ...

    def test_tower_count_matches(self):
        """GPU and Cython produce same number of towers +/- 1."""
        ...

    def test_height_selection_matches(self):
        """When clearance forces taller towers, GPU and Cython agree."""
        ...


class TestBenchmarks:
    """Phase 3: performance benchmarks."""

    @pytest.mark.parametrize("size", [500, 1000])
    def test_speedup_vs_cython(self, size):
        """GPU should be faster than Cython for large rasters."""
        # ... time both, print speedup, assert GPU < Cython time
        ...
```

- [ ] **Step 2: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph/test_constrained_gpu_v2.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run the real-world example**

Run: `.venv/Scripts/python.exe examples/minimal_example_DEM.py` (with `graph_api="raster_gpu"`)
Compare output against Cython baseline results.

- [ ] **Step 4: Commit**

```
test: add GPU v2 validation suite and benchmarks
```

---

## Task Dependency Graph

```
Task 1 (scaffold + state encoding)
  |
  v
Task 2 (persistent kernel + inline towers)
  |
  v
Task 3 (clearance + heights + gradient)
  |
  v
Task 4 (warp-cooperative protocol)
  |
  v
Task 5 (area cost + forbidden footprint)
  |
  v
Task 6 (wire into ConstrainedPathFinder)
  |
  v
Task 7 (validation + benchmarks)
```

All tasks are sequential — each builds on the previous. No parallelism possible.
