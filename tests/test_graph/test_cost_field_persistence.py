"""Saving a settled field and reopening it must answer identically.

The point of the feature is that a siting run holding ten 240 M-cell fields
(about 19 GB live) can page them off disk instead. That is only worth having
if a reopened field gives the SAME costs and the SAME routes as the live one,
so every test here compares against the live field rather than against a
stored expectation.
"""
import numpy as np
import pytest
import rasterio

import pyorps


CELL = 2.0
SHAPE = (70, 90)


def _raster(tmp_path, name="cost.tif", wall=False):
    rng = np.random.default_rng(11)
    data = rng.integers(80, 400, size=SHAPE).astype(np.uint16)
    if wall:
        # Full height and three cells thick: a one-cell wall is hoppable at
        # r2, whose step set reaches two cells, and a partial one is simply
        # walked around.
        data[:, 44:47] = 65535
    path = tmp_path / name
    transform = rasterio.transform.from_origin(400000.0, 5600000.0, CELL, CELL)
    with rasterio.open(
        path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
        dtype="uint16", crs="EPSG:25832", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _finder(path, origin, target):
    return pyorps.PathFinder(
        dataset_source=str(path),
        source_coords=origin, target_coords=[target],
        search_space_buffer_m=pyorps.full_window_buffer_m(str(path)),
        neighborhood_str="r2", graph_api="cython", ignore_max_cost=True,
    )


def _corners(path):
    with rasterio.open(path) as src:
        b = src.bounds
    return ((b.left + 3 * CELL, b.top - 3 * CELL),
            (b.right - 3 * CELL, b.bottom + 3 * CELL))


def _probe_points(path, n=60, seed=3):
    """Cell centres scattered over the window, so no half-cell ambiguity."""
    rng = np.random.default_rng(seed)
    with rasterio.open(path) as src:
        tr = src.transform
        rows = rng.integers(0, src.height, n)
        cols = rng.integers(0, src.width, n)
    xs, ys = rasterio.transform.xy(tr, rows.tolist(), cols.tolist())
    return list(zip(map(float, xs), map(float, ys)))


@pytest.mark.parametrize("algorithm", ["dijkstra", "delta-stepping"])
class TestRoundTrip:
    def test_costs_match_the_live_field(self, tmp_path, algorithm):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        probes = _probe_points(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm=algorithm) as field:
            live = field.costs_to(probes)
            saved = field.save(tmp_path / f"f_{algorithm}.npz")
        with pyorps.CostField.open(saved) as reopened:
            got = reopened.costs_to(probes)
        np.testing.assert_allclose(got, live, rtol=1e-6)
        assert np.isfinite(live).any()

    def test_routes_match_the_live_field(self, tmp_path, algorithm):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm=algorithm) as field:
            live_path = field.path_to(far, calculate_metrics=True)
            live_cells = [int(c) for c in live_path.path_indices.ravel()] \
                if hasattr(live_path, "path_indices") else None
            live_len = float(live_path.total_length)
            saved = field.save(tmp_path / f"r_{algorithm}.npz")
        with pyorps.CostField.open(saved) as reopened:
            cells = reopened.path_cells(far)
            length = reopened.path_length_m(far)
            coords = reopened.path_coords(far)
        assert cells.size > 1
        assert coords.shape == (cells.size, 2)
        # Length is recomputed from the step geometry, not stored.
        assert length == pytest.approx(live_len, rel=1e-9)
        if live_cells:
            assert [int(c) for c in cells] == live_cells

    def test_metadata_survives(self, tmp_path, algorithm):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm=algorithm) as field:
            live_transform = field.transform
            saved = field.save(tmp_path / f"m_{algorithm}.npz")
        with pyorps.CostField.open(saved) as reopened:
            assert reopened.transform == live_transform
            assert reopened.crs.to_string() == "EPSG:25832"
            assert reopened.shape == SHAPE
            assert reopened.origin == pytest.approx(origin, abs=CELL)
            assert reopened.algorithm == algorithm
            assert reopened.has_paths


class TestUnreachable:
    def test_sealed_off_cells_stay_unreachable(self, tmp_path):
        """inf must survive the round trip, and not become a usable cost."""
        path = _raster(tmp_path, "walled.tif", wall=True)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            live = field.costs_to([far])
            saved = field.save(tmp_path / "walled.npz")
        assert not np.isfinite(live[0]), "fixture should seal the far corner"
        with pyorps.CostField.open(saved) as reopened:
            assert not np.isfinite(reopened.cost_to(far))
            assert reopened.path_cells(far).size == 0
            assert reopened.path_length_m(far) == 0.0


class TestCostsOnly:
    def test_without_paths_is_smaller_and_says_so(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        probes = _probe_points(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            live = field.costs_to(probes)
            with_paths = field.save(tmp_path / "with.npz")
            without = field.save(tmp_path / "without.npz", with_paths=False)
        assert without.stat().st_size < with_paths.stat().st_size
        with pyorps.CostField.open(without) as reopened:
            assert reopened.has_paths is False
            np.testing.assert_allclose(reopened.costs_to(probes), live,
                                       rtol=1e-6)
            with pytest.raises(NotImplementedError, match="with_paths"):
                reopened.path_cells(far)


class TestGuards:
    def test_point_outside_the_window_raises(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin) as field:
            saved = field.save(tmp_path / "g.npz")
        with pyorps.CostField.open(saved) as reopened:
            with pytest.raises(ValueError, match="outside"):
                reopened.cost_to((0.0, 0.0))

    def test_a_truncated_cache_is_refused_with_a_useful_message(self, tmp_path):
        """A run killed mid-write used to poison the next run.

        The partial .npz landed under the real name, and the next run
        opened it and died on numpy's "File is not a zip file".
        """
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            saved = field.save(tmp_path / "good.npz")
        raw = saved.read_bytes()
        saved.write_bytes(raw[: len(raw) // 2])          # simulate the kill
        with pytest.raises(ValueError, match="interrupted mid-write"):
            pyorps.CostField.open(saved)

    def test_save_leaves_no_partial_file_under_the_real_name(self, tmp_path):
        """The write goes to a temp name and is renamed into place."""
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        out = tmp_path / "atomic.npz"
        with finder.cost_field(origin, algorithm="auto") as field:
            field.save(out)
        # Nothing left behind, and what is there opens.
        assert sorted(p.name for p in tmp_path.glob("atomic*")) == [
            "atomic.npz"]
        with pyorps.CostField.open(out) as reopened:
            assert reopened.has_paths

    def test_an_interrupted_save_keeps_the_previous_good_cache(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        out = tmp_path / "keep.npz"
        with finder.cost_field(origin, algorithm="auto") as field:
            field.save(out)
            good = out.read_bytes()
            # A save that dies partway must not touch the existing file.
            import numpy as _np
            real = _np.savez
            try:
                _np.savez = lambda *a, **k: (_ for _ in ()).throw(
                    OSError("disk full"))
                with pytest.raises(OSError):
                    field.save(out)
            finally:
                _np.savez = real
        assert out.read_bytes() == good
        with pyorps.CostField.open(out) as reopened:
            assert reopened.has_paths

    def test_reader_reports_its_footprint(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            saved = field.save(tmp_path / "s.npz")
        with pyorps.CostField.open(saved) as reopened:
            cells = SHAPE[0] * SHAPE[1]
            # 4 B of distance + 1 B of predecessor step per cell.
            assert reopened.memory_bytes == cells * 5

    def test_exported_at_top_level(self):
        assert pyorps.SavedCostField is not None
        assert "SavedCostField" in pyorps.__all__


class TestGpuBackend:
    """The GPU keeps predecessors too, so a saved GPU field carries routes."""

    def test_gpu_round_trip_matches_the_live_field(self, tmp_path):
        pytest.importorskip("cupy")
        path = _raster(tmp_path)
        origin, far = _corners(path)
        probes = _probe_points(path, n=40)
        finder = pyorps.PathFinder(
            dataset_source=str(path),
            source_coords=origin, target_coords=[far],
            search_space_buffer_m=pyorps.full_window_buffer_m(str(path)),
            neighborhood_str="r2", graph_api="raster_gpu",
            ignore_max_cost=True)
        try:
            field = finder.cost_field(origin, algorithm="auto")
        except Exception as exc:                       # noqa: BLE001
            pytest.skip(f"no usable GPU here: {exc}")
        try:
            live = field.costs_to(probes)
            live_cells = [int(c) for c in field._tree.extract_or_resume(
                field._finder.get_node_indices_from_coords(far))]
            saved = field.save(tmp_path / "gpu.npz")
        finally:
            field.close()
            finder.release_device_resources()

        with pyorps.CostField.open(saved) as reopened:
            assert reopened.has_paths, "GPU fields should carry predecessors"
            np.testing.assert_allclose(reopened.costs_to(probes), live,
                                       rtol=1e-5)
            cells = [int(c) for c in reopened.path_cells(far)]
        assert cells == live_cells


class TestMemoryPolicy:
    """Saving is the fallback; the helper is what decides to fall back."""

    def test_a_small_set_fits_and_a_huge_one_does_not(self):
        assert pyorps.fields_fit_in_memory(1, 10_000)
        # Ten 240 M-cell fields is ~19 GB live; no headroom fraction of a
        # normal machine covers that.
        assert not pyorps.fields_fit_in_memory(10, 240_000_000)

    def test_dijkstra_is_costed_higher_than_the_packed_kernels(self):
        """13 B/cell against 8, so the verdict can differ at the margin."""
        cells, n = 40_000_000, 6
        import psutil
        avail = psutil.virtual_memory().available
        # Pick headroom so the two backends straddle the limit.
        head = (n * cells * 10) / avail
        if not 0.0 < head < 1.0:
            pytest.skip("machine memory makes this margin untestable")
        assert pyorps.fields_fit_in_memory(n, cells, algorithm="auto",
                                           headroom=head)
        assert not pyorps.fields_fit_in_memory(n, cells,
                                               algorithm="dijkstra",
                                               headroom=head)
