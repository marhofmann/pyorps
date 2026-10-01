"""0-cost-cell hardening tests for the constrained bucket kernels.

The Dial-style bucket kernels (dense ``constrained_dijkstra_2d``,
``constrained_delta_stepping_2d``) use ``delta = max(1.0, 2*min_raster*min_cf)``,
which is exact only while no edge is cheaper than delta. Rasters containing
0-cost cells (free corridors along existing infrastructure) violate that
invariant: before the 2026-08-05 hardening the dijkstra variant livelocked
(batch/bucket swap oscillation) and both variants silently dropped
same-bucket improvements.

These tests pin the fix:

- termination on 0-cost corridors (livelock repro, run in a subprocess with
  a timeout so a regression fails instead of hanging the suite),
- exactness vs the heap-mode reference (``force_sparse=1``), which pops in
  true priority order and is exact by construction.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pyorps.utils._constrained_delta import (
    constrained_delta_stepping_2d,
    constrained_delta_stepping_height_2d,
    constrained_delta_stepping_lazy,
)
from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d

REL_TOL = 1e-6

SPAN_KWARGS = dict(
    n_span_bins=10, span_bin_size=20.0, min_span=40.0, max_span=200.0,
)


def _make_inputs(raster):
    """LUTs and step geometry for an 8-connected constrained search."""
    steps = np.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0],
         [1, 1], [1, -1], [-1, 1], [-1, -1]],
        dtype=np.int8,
    )
    n_dirs = steps.shape[0]
    angle_cost = np.zeros((n_dirs, n_dirs), dtype=np.float32)
    angle_valid = np.ones((n_dirs, n_dirs), dtype=np.uint8)
    step_distances = (
        np.linalg.norm(steps.astype(np.float32), axis=1) * 10.0
    ).astype(np.float32)
    tower_terrain = np.ones(65536, dtype=np.float32) * 5.0
    tower_angle = np.zeros((n_dirs, n_dirs), dtype=np.float32)
    return dict(
        steps=steps,
        angle_cost_lut=angle_cost,
        angle_valid_lut=angle_valid,
        step_distances=step_distances,
        tower_terrain_costs=tower_terrain,
        tower_angle_costs=tower_angle,
    )


def _corridor_raster():
    """10x10 raster, cost 10 everywhere, 0-cost corridor along row 4."""
    raster = np.full((10, 10), 10, dtype=np.uint16)
    raster[4, :] = 0
    return raster


def _run_dijkstra(raster, src, dst, force_sparse=0):
    kw = _make_inputs(raster)
    return constrained_dijkstra_2d(
        raster, src[0], src[1], dst[0], dst[1], kw.pop("steps"),
        **kw, **SPAN_KWARGS, force_sparse=force_sparse, return_dist=1,
    )


def _run_delta(raster, src, dst):
    kw = _make_inputs(raster)
    return constrained_delta_stepping_2d(
        raster, src[0], src[1], dst[0], dst[1], kw.pop("steps"),
        **kw, **SPAN_KWARGS, return_dist=1,
    )


class TestZeroCostTermination:
    """The pre-fix dijkstra variant livelocks on this input (oscillating
    batch/bucket swap as soon as a push lands in the current bucket)."""

    def test_zero_cost_corridor_terminates(self):
        script = textwrap.dedent("""
            import numpy as np
            from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d

            raster = np.full((10, 10), 10, dtype=np.uint16)
            raster[4, :] = 0
            steps = np.array(
                [[0, 1], [1, 0], [0, -1], [-1, 0],
                 [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
            n_dirs = steps.shape[0]
            path, towers = constrained_dijkstra_2d(
                raster, 0, 0, 9, 9, steps,
                angle_cost_lut=np.zeros((n_dirs, n_dirs), dtype=np.float32),
                angle_valid_lut=np.ones((n_dirs, n_dirs), dtype=np.uint8),
                step_distances=(np.linalg.norm(
                    steps.astype(np.float32), axis=1) * 10.0
                ).astype(np.float32),
                tower_terrain_costs=np.ones(65536, dtype=np.float32) * 5.0,
                tower_angle_costs=np.zeros((n_dirs, n_dirs), dtype=np.float32),
                n_span_bins=10, span_bin_size=20.0,
                min_span=40.0, max_span=200.0,
            )
            assert len(path) > 0
            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"kernel failed on 0-cost corridor:\n{result.stderr}"
        )
        assert "OK" in result.stdout

    def test_all_zero_raster_terminates(self):
        raster = np.zeros((8, 8), dtype=np.uint16)
        path, towers, dist = _run_dijkstra(raster, (0, 0), (7, 7))
        assert len(path) > 0
        assert np.isfinite(dist)


class TestZeroCostExactness:
    """Bucket modes must match the heap reference despite delta > min edge."""

    def test_corridor_dense_matches_heap(self):
        raster = _corridor_raster()
        _, _, dist_heap = _run_dijkstra(raster, (0, 0), (9, 9), force_sparse=1)
        _, _, dist_dense = _run_dijkstra(raster, (0, 0), (9, 9))
        assert np.isfinite(dist_heap)
        assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL)

    def test_corridor_delta_matches_heap(self):
        raster = _corridor_raster()
        _, _, dist_heap = _run_dijkstra(raster, (0, 0), (9, 9), force_sparse=1)
        _, _, dist_delta = _run_delta(raster, (0, 0), (9, 9))
        assert dist_delta == pytest.approx(dist_heap, rel=REL_TOL)

    def test_all_zero_raster_matches_heap(self):
        raster = np.zeros((8, 8), dtype=np.uint16)
        _, _, dist_heap = _run_dijkstra(raster, (0, 0), (7, 7), force_sparse=1)
        _, _, dist_dense = _run_dijkstra(raster, (0, 0), (7, 7))
        _, _, dist_delta = _run_delta(raster, (0, 0), (7, 7))
        assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL)
        assert dist_delta == pytest.approx(dist_heap, rel=REL_TOL)

    @pytest.mark.parametrize("seed", range(8))
    def test_randomized_rasters_match_heap(self, seed):
        rng = np.random.default_rng(seed)
        raster = rng.choice(
            np.array([0, 0, 1, 2, 3, 5, 10], dtype=np.uint16), size=(12, 12)
        )
        # Keep endpoints traversable
        src = (0, int(rng.integers(0, 12)))
        dst = (11, int(rng.integers(0, 12)))
        _, _, dist_heap = _run_dijkstra(raster, src, dst, force_sparse=1)
        _, _, dist_dense = _run_dijkstra(raster, src, dst)
        _, _, dist_delta = _run_delta(raster, src, dst)
        if np.isinf(dist_heap):
            assert np.isinf(dist_dense) and np.isinf(dist_delta)
        else:
            assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL), (
                f"dense bucket mode suboptimal at seed {seed}"
            )
            assert dist_delta == pytest.approx(dist_heap, rel=REL_TOL), (
                f"delta-stepping suboptimal at seed {seed}"
            )


# Single height class, zero premium, flat DEM, zero clearance threshold:
# the variable-height problem then reduces exactly to the basic constrained
# problem, so the heap-mode dijkstra serves as reference for the height
# kernels (dense, compact-dense `_height_sparse`, lazy hash-map).
HEIGHT_KWARGS = dict(
    cell_size=10.0,
    tower_heights=np.array([20.0], dtype=np.float32),
    height_premiums=np.array([0.0], dtype=np.float32),
    conductor_weight_per_m=1.5,
    conductor_tension=30000.0,
    min_clearance_val=0.0,
)


def _run_height(raster, src, dst, force_sparse=0):
    kw = _make_inputs(raster)
    dem = np.zeros(raster.shape, dtype=np.float32)
    return constrained_delta_stepping_height_2d(
        raster, src[0], src[1], dst[0], dst[1], kw.pop("steps"),
        **kw, **SPAN_KWARGS, dem_data=dem, **HEIGHT_KWARGS,
        force_sparse=force_sparse, return_dist=1,
    )


def _run_lazy(raster, src, dst):
    kw = _make_inputs(raster)
    dem = np.zeros(raster.shape, dtype=np.float32)
    return constrained_delta_stepping_lazy(
        raster, src[0], src[1], dst[0], dst[1], kw.pop("steps"),
        **kw, **SPAN_KWARGS, dem_data=dem, **HEIGHT_KWARGS,
        return_dist=1,
    )


def _heap_reference_with_dem(raster, src, dst):
    kw = _make_inputs(raster)
    dem = np.zeros(raster.shape, dtype=np.float32)
    return constrained_dijkstra_2d(
        raster, src[0], src[1], dst[0], dst[1], kw.pop("steps"),
        **kw, **SPAN_KWARGS, dem_data=dem, cell_size=10.0,
        force_sparse=1, return_dist=1,
    )


class TestHeightVariantsZeroCost:
    """The compact-dense (`_height_sparse`) and lazy hash-map variants got
    the re-open protocol last: these pin their exactness on 0-cost inputs."""

    def test_corridor_all_height_variants_match_heap(self):
        raster = _corridor_raster()
        _, _, dist_heap = _heap_reference_with_dem(raster, (0, 0), (9, 9))
        _, _, _, dist_dense = _run_height(raster, (0, 0), (9, 9))
        _, _, _, dist_compact = _run_height(raster, (0, 0), (9, 9),
                                            force_sparse=1)
        _, _, _, dist_lazy = _run_lazy(raster, (0, 0), (9, 9))
        assert np.isfinite(dist_heap)
        assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL)
        assert dist_compact == pytest.approx(dist_heap, rel=REL_TOL)
        assert dist_lazy == pytest.approx(dist_heap, rel=REL_TOL)

    # Seeds empirically verified to expose the improvement-after-visit
    # hazard: with the re-open protocol disabled, the compact-dense and
    # lazy kernels return up to 30 % suboptimal costs on these rasters.
    @pytest.mark.parametrize("seed", [260, 264, 267, 272, 273, 290])
    def test_randomized_height_variants_match_heap(self, seed):
        rng = np.random.default_rng(seed)
        n = 16
        raster = rng.choice(
            np.array([0, 0, 0, 1, 1, 2, 3, 5], dtype=np.uint16), size=(n, n)
        )
        src = (0, int(rng.integers(0, n)))
        dst = (n - 1, int(rng.integers(0, n)))
        _, _, dist_heap = _heap_reference_with_dem(raster, src, dst)
        _, _, _, dist_dense = _run_height(raster, src, dst)
        _, _, _, dist_compact = _run_height(raster, src, dst, force_sparse=1)
        _, _, _, dist_lazy = _run_lazy(raster, src, dst)
        if np.isinf(dist_heap):
            assert np.isinf(dist_dense)
            assert np.isinf(dist_compact)
            assert np.isinf(dist_lazy)
        else:
            assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL), (
                f"height dense suboptimal at seed {seed}"
            )
            assert dist_compact == pytest.approx(dist_heap, rel=REL_TOL), (
                f"_height_sparse suboptimal at seed {seed}"
            )
            assert dist_lazy == pytest.approx(dist_heap, rel=REL_TOL), (
                f"lazy variant suboptimal at seed {seed}"
            )


class TestKnownSpanPayloadLimitation:
    """Known model approximation, documented 2026-08-06 (survives the re-open
    hardening): each state (cell, dir, span_bin) stores ONE exact-span float,
    so two same-bin labels with different accumulated spans can have crossing
    utility (max_span headroom vs min_span tower gate). Neither the heap nor
    the bucket kernels are exact under the full (cell, dir, exact-span)
    model; they merely break such ties differently. On this seed the heap's
    ordering finds a degenerate out-and-back route (walking 0-cost cells to
    accumulate traveled span) that all five bucket kernels miss identically.
    Frequency: 1/300 seeds on adversarial dense-0-cost rasters."""

    @pytest.mark.xfail(
        reason="span-payload tie-breaking: quantized state space is not "
               "exact under crossing same-bin span utilities",
        strict=True,
    )
    def test_seed_279_degenerate_out_and_back(self):
        rng = np.random.default_rng(279)
        n = 16
        raster = rng.choice(
            np.array([0, 0, 0, 1, 1, 2, 3, 5], dtype=np.uint16), size=(n, n)
        )
        src = (0, int(rng.integers(0, n)))
        dst = (n - 1, int(rng.integers(0, n)))
        _, _, dist_heap = _run_dijkstra(raster, src, dst, force_sparse=1)
        _, _, dist_dense = _run_dijkstra(raster, src, dst)
        assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL)


class TestSpanConfigValidation:
    """n_span_bins * span_bin_size < max_span lets span bins alias into
    neighboring states (out-of-bounds in dense modes) — must be rejected."""

    def test_dijkstra_rejects_undersized_span_bins(self):
        raster = np.full((10, 10), 10, dtype=np.uint16)
        kw = _make_inputs(raster)
        with pytest.raises(ValueError, match="alias"):
            constrained_dijkstra_2d(
                raster, 0, 0, 9, 9, kw.pop("steps"), **kw,
                n_span_bins=5, span_bin_size=20.0,
                min_span=40.0, max_span=200.0,
            )

    def test_delta_rejects_undersized_span_bins(self):
        raster = np.full((10, 10), 10, dtype=np.uint16)
        kw = _make_inputs(raster)
        with pytest.raises(ValueError, match="alias"):
            constrained_delta_stepping_2d(
                raster, 0, 0, 9, 9, kw.pop("steps"), **kw,
                n_span_bins=5, span_bin_size=20.0,
                min_span=40.0, max_span=200.0,
            )


class TestMinRasterGeOneStillExact:
    """Regression guard: the common case (min raster value >= 1) must keep
    matching the heap reference — the re-open branch should never fire."""

    @pytest.mark.parametrize("seed", range(4))
    def test_randomized_positive_rasters_match_heap(self, seed):
        rng = np.random.default_rng(100 + seed)
        raster = rng.integers(1, 20, size=(12, 12), dtype=np.uint16)
        _, _, dist_heap = _run_dijkstra(raster, (0, 0), (11, 11), force_sparse=1)
        _, _, dist_dense = _run_dijkstra(raster, (0, 0), (11, 11))
        _, _, dist_delta = _run_delta(raster, (0, 0), (11, 11))
        assert dist_dense == pytest.approx(dist_heap, rel=REL_TOL)
        assert dist_delta == pytest.approx(dist_heap, rel=REL_TOL)
