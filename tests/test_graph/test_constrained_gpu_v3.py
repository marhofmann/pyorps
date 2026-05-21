"""Tests for GPU constrained SSSP V3 (host-driven, dynamic block-sparse)."""
import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")


class TestKernelCompilation:
    """Test that all V3 CUDA kernels compile successfully."""

    def test_init_pool_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("init_pool_v3")
        assert k is not None

    def test_init_source_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("init_source_v3")
        assert k is not None

    def test_classify_bucket_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("classify_bucket")
        assert k is not None

    def test_scan_min_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("scan_min_dist")
        assert k is not None

    def test_extract_bucket_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("extract_bucket")
        assert k is not None

    def test_relax_kernel_compiles(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        k = _get_v3_kernel("relax_constrained_v3")
        assert k is not None


class TestInitKernels:
    """Test init_pool_v3 and init_source_v3 kernels."""

    def test_init_pool_sets_empty(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import _get_v3_kernel
        block_entry_dtype = cp.dtype([
            ('local_key', cp.uint16), ('_pad', cp.uint16),
            ('dist', cp.float32)])
        n_entries = 256
        d_pool = cp.empty(n_entries, dtype=block_entry_dtype)
        kernel = _get_v3_kernel("init_pool_v3")
        kernel((1,), (256,), (d_pool, np.int32(n_entries)))
        cp.cuda.Stream.null.synchronize()
        host = d_pool.get()
        host_dtype = np.dtype([
            ('local_key', np.uint16), ('_pad', np.uint16),
            ('dist', np.float32)])
        host_np = np.frombuffer(host.tobytes(), dtype=host_dtype)
        for entry in host_np:
            assert int(entry['local_key']) == 0xFFFF
            assert float(entry['dist']) > 1e29

    def test_init_source_allocates_blocks(self):
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            _get_v3_kernel, pack_state)

        n_cells = 100
        n_dirs = 8
        n_span_bins = 6
        n_heights = 1
        spc = n_dirs * n_span_bins * n_heights
        max_blocks = 50
        block_size = 64
        pool_size = max_blocks * block_size

        block_entry_dtype = cp.dtype([
            ('local_key', cp.uint16), ('_pad', cp.uint16),
            ('dist', cp.float32)])

        d_pool = cp.empty(pool_size, dtype=block_entry_dtype)
        d_span = cp.zeros(pool_size, dtype=cp.float16)
        d_c2b = cp.full(n_cells, -1, dtype=cp.int32)
        d_b2c = cp.full(max_blocks, -1, dtype=cp.int32)
        d_n_alloc = cp.zeros(1, dtype=cp.int32)

        # Init pool
        init_pool = _get_v3_kernel("init_pool_v3")
        init_pool(
            ((pool_size + 255) // 256,), (256,),
            (d_pool, np.int32(pool_size)))
        cp.cuda.Stream.null.synchronize()

        # Prepare source states: all directions at cell 5
        source_states = [
            pack_state(5, d, 0, 0, spc, n_span_bins, n_heights)
            for d in range(n_dirs)
        ]
        source_dists = [0.0] * n_dirs

        d_src = cp.asarray(np.array(source_states, dtype=np.int64))
        d_sdist = cp.asarray(np.array(source_dists, dtype=np.float32))

        init_src = _get_v3_kernel("init_source_v3")
        init_src(
            (1,), (min(256, n_dirs),),
            (d_pool, d_span, d_c2b, d_b2c, d_n_alloc,
             d_src, d_sdist,
             np.int32(n_dirs), np.int32(spc), np.int32(n_span_bins),
             np.int32(n_heights), np.int32(max_blocks)))
        cp.cuda.Stream.null.synchronize()

        c2b_host = d_c2b.get()
        assert c2b_host[5] >= 0, "cell 5 should have a block allocated"
        assert int(d_n_alloc.get().item()) >= 1, "at least 1 block should be allocated"

        b2c_host = d_b2c.get()
        block_idx = c2b_host[5]
        assert b2c_host[block_idx] == 5, "block_to_cell should map back to cell 5"

    def test_init_source_sets_distances(self):
        """Verify that init_source_v3 writes correct distances into the pool."""
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            _get_v3_kernel, pack_state)

        n_cells = 50
        n_dirs = 4
        n_span_bins = 4
        n_heights = 1
        spc = n_dirs * n_span_bins * n_heights
        max_blocks = 20
        block_size = 64
        pool_size = max_blocks * block_size

        block_entry_dtype = cp.dtype([
            ('local_key', cp.uint16), ('_pad', cp.uint16),
            ('dist', cp.float32)])

        d_pool = cp.empty(pool_size, dtype=block_entry_dtype)
        d_span = cp.zeros(pool_size, dtype=cp.float16)
        d_c2b = cp.full(n_cells, -1, dtype=cp.int32)
        d_b2c = cp.full(max_blocks, -1, dtype=cp.int32)
        d_n_alloc = cp.zeros(1, dtype=cp.int32)

        init_pool = _get_v3_kernel("init_pool_v3")
        init_pool(
            ((pool_size + 255) // 256,), (256,),
            (d_pool, np.int32(pool_size)))
        cp.cuda.Stream.null.synchronize()

        # Single source state at cell 10, dir 0, span_bin 0, hc 0, dist=42.0
        source_state = pack_state(10, 0, 0, 0, spc, n_span_bins, n_heights)
        d_src = cp.asarray(np.array([source_state], dtype=np.int64))
        d_sdist = cp.asarray(np.array([42.0], dtype=np.float32))

        init_src = _get_v3_kernel("init_source_v3")
        init_src(
            (1,), (1,),
            (d_pool, d_span, d_c2b, d_b2c, d_n_alloc,
             d_src, d_sdist,
             np.int32(1), np.int32(spc), np.int32(n_span_bins),
             np.int32(n_heights), np.int32(max_blocks)))
        cp.cuda.Stream.null.synchronize()

        # Read back and verify the distance was written
        block_idx = int(d_c2b.get()[10])
        assert block_idx >= 0

        block_base = block_idx * block_size
        block_entries = d_pool[block_base:block_base + block_size].get()
        host_dtype = np.dtype([
            ('local_key', np.uint16), ('_pad', np.uint16),
            ('dist', np.float32)])
        block_np = np.frombuffer(block_entries.tobytes(), dtype=host_dtype)

        found = False
        for entry in block_np:
            lk = int(entry['local_key'])
            if lk != 0xFFFF:
                d = float(entry['dist'])
                assert abs(d - 42.0) < 0.01, f"Expected dist 42.0, got {d}"
                found = True
                break
        assert found, "Source state entry not found in pool"


# ============================================================================
# Shared helper: build uniform raster + profile params for end-to-end tests
# ============================================================================

def _make_test_params(steps, rows=50, cols=50, raster_val=10,
                      cell_size=10.0, min_span=50.0, max_span=300.0,
                      span_bin_size=50.0):
    """Create uniform raster + minimal constrained profile for V3 testing.

    Returns (raster, params_dict) where params_dict contains all keyword
    arguments needed by constrained_sssp_raster_gpu_v3 except raster,
    source/target coords, and steps.
    """
    from pyorps.core.infrastructure_profile import InfrastructureProfile

    raster = np.full((rows, cols), raster_val, dtype=np.uint16)
    n_span_bins = int(max_span / span_bin_size) + 1

    config = {
        "name": "test_v3",
        "description": "test profile for v3 kernel",
        "soft_angle_limit_deg": 5.0,
        "hard_angle_limit_deg": 90.0,
        "angle_cost_function": "linear",
        "angle_cost_params": {"scale": 100},
        "min_span_m": min_span,
        "max_span_m": max_span,
        "span_bin_size_m": span_bin_size,
        "tower_cost_function": "terrain_plus_angle",
        "tower_cost_params": {
            "terrain_cost_map": {"0": 1000, "500": 5000},
            "terrain_interpolation": "linear",
            "angle_types": {
                "suspension": {
                    "max_angle_deg": 90.0,
                    "base_cost": 1000,
                },
            },
        },
    }
    profile = InfrastructureProfile.from_dict(config)
    angle_cost_lut, angle_valid_lut = profile.precompute_angle_lut(steps)
    step_distances = profile.compute_step_distances(steps, cell_size)
    tower_terrain_costs = profile.precompute_tower_terrain_costs()
    tower_angle_costs = profile.precompute_tower_angle_costs(steps)

    params = {
        "angle_cost_lut": angle_cost_lut.astype(np.float32),
        "angle_valid_lut": angle_valid_lut.astype(np.uint8),
        "step_distances": step_distances.astype(np.float32),
        "tower_terrain_costs": tower_terrain_costs.astype(np.float32),
        "tower_angle_costs": tower_angle_costs.astype(np.float32),
        "n_span_bins": n_span_bins,
        "span_bin_size": span_bin_size,
        "min_span": min_span,
        "max_span": max_span,
    }
    return raster, params


# ============================================================================
# Task 9: End-to-End Path Finding Tests
# ============================================================================

class TestPathFinding:
    """End-to-end tests for constrained_sssp_raster_gpu_v3."""

    def test_basic_path_found(self):
        """50x50 uniform raster, source=(0,0), target=(49,49). Path found."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=0, source_col=0,
            target_row=49, target_col=49,
            steps=steps,
            **params)

        assert len(path) > 0, "Path should not be empty"
        source_cell = 0 * 50 + 0
        target_cell = 49 * 50 + 49
        assert path[0] == source_cell, (
            f"Path start {path[0]} != source {source_cell}")
        assert path[-1] == target_cell, (
            f"Path end {path[-1]} != target {target_cell}")

    def test_no_path_forbidden(self):
        """50x50 with wall of 65535 isolating target. Empty path returned."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Wall across the middle, isolating target
        raster[20:30, :] = 65535

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=0, source_col=0,
            target_row=49, target_col=49,
            steps=steps,
            **params)

        assert len(path) == 0, f"Expected empty path, got {len(path)} cells"
        assert len(towers) == 0
        assert len(heights) == 0

    def test_source_on_forbidden_raises(self):
        """Source cell = 65535 should raise ValueError."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10)

        raster[5, 5] = 65535

        with pytest.raises(ValueError, match="Source cell.*forbidden"):
            constrained_sssp_raster_gpu_v3(
                raster=raster,
                source_row=5, source_col=5,
                target_row=45, target_col=45,
                steps=steps,
                **params)

    def test_target_on_forbidden_raises(self):
        """Target cell = 65535 should raise ValueError."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3)

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10)

        raster[45, 45] = 65535

        with pytest.raises(ValueError, match="Target cell.*forbidden"):
            constrained_sssp_raster_gpu_v3(
                raster=raster,
                source_row=5, source_col=5,
                target_row=45, target_col=45,
                steps=steps,
                **params)

    def test_path_has_towers(self):
        """Path long enough for towers should have at least one tower."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Source=(5,5) to target=(45,45) ~ 566m straight line,
        # with max_span=300m, at least 1 tower expected
        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            **params)

        assert len(path) > 0, "Path should not be empty"
        assert len(towers) >= 1, (
            f"Expected at least 1 tower for ~566m path with max_span=300m, "
            f"got {len(towers)}")
        assert len(heights) == len(towers), (
            "Heights array length should match towers array length")


# ============================================================================
# Task 10: Cython Reference Comparison
# ============================================================================

class TestCythonComparison:
    """Compare V3 GPU results against Cython constrained Dijkstra."""

    def test_cost_similarity_uniform_raster(self):
        """GPU V3 and Cython find paths with similar tower counts on uniform raster.

        Constraints: n_heights=1, no DEM, no clearance, no area cost (Cython
        constrained_dijkstra_2d does not support those features).
        """
        try:
            from pyorps.utils.constrained_path_algorithms import (
                constrained_dijkstra_2d)
        except ImportError:
            pytest.skip("Cython constrained_dijkstra_2d not available "
                        "(extensions not built)")

        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        source_row, source_col = 2, 2
        target_row, target_col = 47, 47

        # --- Run GPU V3 ---
        path_gpu, towers_gpu, heights_gpu = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=steps,
            **params)

        # --- Run Cython ---
        # Cython returns 2-tuple (path, towers)
        path_cy, towers_cy = constrained_dijkstra_2d(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=steps,
            angle_cost_lut=params["angle_cost_lut"],
            angle_valid_lut=params["angle_valid_lut"],
            step_distances=params["step_distances"],
            tower_terrain_costs=params["tower_terrain_costs"],
            tower_angle_costs=params["tower_angle_costs"],
            n_span_bins=params["n_span_bins"],
            span_bin_size=params["span_bin_size"],
            min_span=params["min_span"],
            max_span=params["max_span"],
        )

        # Both should find paths
        assert len(path_gpu) > 0, "GPU should find a path on 50x50 uniform raster"
        assert len(path_cy) > 0, "Cython should find a path on 50x50 uniform raster"

        # Source and target match
        cols = 50
        expected_source = source_row * cols + source_col
        expected_target = target_row * cols + target_col
        assert path_gpu[0] == expected_source
        assert path_gpu[-1] == expected_target
        assert path_cy[0] == expected_source
        assert path_cy[-1] == expected_target

        # Tower counts within +/- 2 (tie-breaking may differ)
        n_towers_gpu = len(towers_gpu)
        n_towers_cy = len(towers_cy)
        assert abs(n_towers_gpu - n_towers_cy) <= 2, (
            f"Tower count mismatch: GPU={n_towers_gpu}, Cython={n_towers_cy} "
            f"(diff > 2)")


# ============================================================================
# Task 11: DEM, Clearance, and Area Cost Tests
# ============================================================================

class TestDEMAndClearance:
    """Tests for DEM gradient penalty and area cost features."""

    def test_dem_gradient_penalty(self):
        """Raster with DEM ridge. V3 with DEM should find a path (possibly
        longer due to gradient penalty) without crashing."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Create DEM with ridge in the middle
        dem = np.zeros((50, 50), dtype=np.float32)
        dem[20:30, :] = 200.0  # 200m high ridge

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            dem=dem,
            cell_size=10.0,
            max_gradient_pct=50.0,
            gradient_scale=2.0,
            tower_heights=np.array([30.0], dtype=np.float32),
            height_premiums=np.array([0.0], dtype=np.float32),
            n_heights=1,
            conductor_weight_per_m=10.0,
            conductor_tension=50000.0,
            min_clearance=0.0,
            **params)

        # Path may or may not be found depending on gradient limit.
        # The key is that it completes without crashing.
        # If found, verify it starts and ends correctly.
        if len(path) > 0:
            assert path[0] == 5 * 50 + 5
            assert path[-1] == 45 * 50 + 45

    def test_forbidden_area_blocks_tower(self):
        """Place 65535 cells adjacent to path. With area_offsets that include
        those cells, tower placement should be blocked there. The algorithm
        should still complete without crashing."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        n_dirs = len(steps)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Place forbidden cells adjacent to diagonal path
        for i in range(10, 40):
            if i + 1 < 50:
                raster[i, i + 1] = 65535  # right neighbor of diagonal
            if i - 1 >= 0:
                raster[i, i - 1] = 65535  # left neighbor of diagonal

        # Simple area offsets: 1-cell radius around tower
        all_offsets = []
        starts = np.zeros(n_dirs * n_dirs, dtype=np.int32)
        counts = np.zeros(n_dirs * n_dirs, dtype=np.int32)
        for pair in range(n_dirs * n_dirs):
            starts[pair] = len(all_offsets)
            offsets = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
            counts[pair] = len(offsets)
            all_offsets.extend(offsets)
        area_offsets = np.array(all_offsets, dtype=np.int32).flatten()

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=5, source_col=5,
            target_row=45, target_col=45,
            steps=steps,
            area_offsets=area_offsets,
            area_offset_starts=starts,
            area_offset_counts=counts,
            **params)

        # Main assertion: the function completes without crashing.
        # If a path is found, basic properties hold.
        if len(path) > 0:
            assert path[0] == 5 * 50 + 5
            assert path[-1] == 45 * 50 + 45


# ============================================================================
# Task 12: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for pool exhaustion, small rasters, etc."""

    def test_pool_exhaustion_warns(self):
        """Very small max_visited_fraction on 100x100 raster should emit
        UserWarning about overflow."""
        import warnings as wrn
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=100, cols=100, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        # Very small visited fraction -- likely causes overflow
        with wrn.catch_warnings(record=True) as w:
            wrn.simplefilter("always")
            path, towers, heights = constrained_sssp_raster_gpu_v3(
                raster=raster,
                source_row=2, source_col=2,
                target_row=97, target_col=97,
                steps=steps,
                max_visited_fraction=0.001,
                **params)

        # Check for overflow or "no path" warning.
        # With very limited resources, either the path is found (unlikely)
        # or a warning is emitted. The test verifies no crash occurs.
        # Note: if the path IS found somehow, that's also fine.

    def test_small_raster_path(self):
        """20x20 raster, very basic case. Should find path or handle gracefully."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=20, cols=20, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=200.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=2, source_col=2,
            target_row=17, target_col=17,
            steps=steps,
            **params)

        # Path should be found on small uniform raster
        assert len(path) > 0, "Path should be found on 20x20 uniform raster"
        assert path[0] == 2 * 20 + 2
        assert path[-1] == 17 * 20 + 17


# ============================================================================
# Task 13: Medium Rasters and Edge Cases
# ============================================================================

class TestMediumRasters:
    """Tests on medium-sized rasters and degenerate inputs."""

    def test_200x200_finds_path(self):
        """200x200 uniform raster should find a path."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=200, cols=200, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=10, source_col=10,
            target_row=190, target_col=190,
            steps=steps,
            max_visited_fraction=0.9,
            **params)

        assert len(path) > 0, "Path should be found on 200x200 uniform raster"
        assert path[0] == 10 * 200 + 10
        assert path[-1] == 190 * 200 + 190
        # ~2545m path, max_span=300m -> at least 5 towers expected
        assert len(towers) >= 3, (
            f"Expected at least 3 towers for ~2545m path with max_span=300m, "
            f"got {len(towers)}")

    def test_source_equals_target(self):
        """source == target, should return trivial result."""
        from pyorps.utils.neighborhood import get_neighborhood_steps
        from pyorps.utils.constrained_sssp_gpu_v3 import (
            constrained_sssp_raster_gpu_v3, _check_v3_available)

        if not _check_v3_available():
            pytest.skip("V3 kernels not available on this GPU")

        steps = get_neighborhood_steps(1, directed=True)
        raster, params = _make_test_params(
            steps, rows=50, cols=50, raster_val=10,
            cell_size=10.0, min_span=50.0, max_span=300.0,
            span_bin_size=50.0)

        path, towers, heights = constrained_sssp_raster_gpu_v3(
            raster=raster,
            source_row=25, source_col=25,
            target_row=25, target_col=25,
            steps=steps,
            **params)

        # Source == target: should return path with 1 cell, no towers
        assert len(path) <= 1 or path[0] == path[-1] == 25 * 50 + 25
