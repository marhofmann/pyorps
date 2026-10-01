"""The public cost-field surface: exports, ``auto``, and georeferencing.

These pin the seams a downstream project needs so it does not have to reach
into ``finder.raster_handler``. The CIRED 2027 free-siting workflow did
exactly that for the window transform, and hand-rolled the whole-raster
buffer, because neither was reachable any other way.
"""
import math

import numpy as np
import pytest
import rasterio

import pyorps
from pyorps.graph.search_session import (
    _auto_threads,
    _resolve_algorithm,
)


CELL = 2.0


def _raster(tmp_path, value=100, shape=(40, 40)):
    """A uniform cost raster on disk, 2 m cells, EPSG:25832."""
    data = np.full(shape, value, dtype=np.uint16)
    path = tmp_path / "cost.tif"
    transform = rasterio.transform.from_origin(400000.0, 5600000.0, CELL, CELL)
    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="uint16", crs="EPSG:25832", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _finder(path, origin, target):
    return pyorps.PathFinder(
        dataset_source=str(path),
        source_coords=origin, target_coords=[target],
        search_space_buffer_m=pyorps.full_window_buffer_m(str(path)),
        neighborhood_str="r2",
        graph_api="cython",
        ignore_max_cost=True,
    )


def _corners(path):
    with rasterio.open(path) as src:
        b = src.bounds
    return ((b.left + 3 * CELL, b.top - 3 * CELL),
            (b.right - 3 * CELL, b.bottom + 3 * CELL))


class TestPublicExports:
    """``import pyorps`` must be enough to reach the feature."""

    @pytest.mark.parametrize("name", ["CostField", "SearchSession",
                                      "full_window_buffer_m"])
    def test_exported_at_top_level(self, name):
        assert hasattr(pyorps, name)
        assert name in pyorps.__all__

    def test_same_object_as_the_graph_subpackage(self):
        from pyorps.graph import CostField, SearchSession
        assert pyorps.CostField is CostField
        assert pyorps.SearchSession is SearchSession


class TestFullWindowBuffer:
    def test_matches_the_raster_diagonal(self, tmp_path):
        path = _raster(tmp_path)
        with rasterio.open(path) as src:
            b = src.bounds
            want = math.hypot(b.right - b.left, b.top - b.bottom)
        assert pyorps.full_window_buffer_m(str(path)) == pytest.approx(want)

    def test_covers_every_cell_from_any_origin(self, tmp_path):
        """The point of the helper: no candidate falls outside the window."""
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin) as field:
            # Would raise if `far` were outside the window.
            assert np.isfinite(field.cost_to(far))


class TestAutoAlgorithm:
    def test_auto_resolves_to_delta_on_cython(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        assert _resolve_algorithm(finder, "auto") == "delta-stepping"

    def test_explicit_algorithm_is_untouched(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        assert _resolve_algorithm(finder, "dijkstra") == "dijkstra"

    def test_auto_reports_the_resolved_algorithm(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            assert field.algorithm == "delta-stepping"

    def test_auto_agrees_with_dijkstra(self, tmp_path):
        """Same answers, different kernel -- the whole point of `auto`."""
        path = _raster(tmp_path)
        origin, far = _corners(path)
        probes = [far, (far[0] - 6 * CELL, far[1] + 6 * CELL)]

        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            reference = field.costs_to(probes)
        finder.release_device_resources()

        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            got = field.costs_to(probes)
        finder.release_device_resources()

        np.testing.assert_allclose(got, reference, rtol=1e-4)

    def test_auto_never_requests_every_core(self):
        """The spin barrier collapses when nothing is left to run the OS.

        Measured 199x slower at 16 of 16 cores than at 12, so this is the
        one property of the default that really matters.
        """
        import os
        cores = os.cpu_count() or 2
        got = _auto_threads()
        assert got >= 1
        if cores > 1:
            assert got < cores, (
                f"auto asked for {got} of {cores} cores; leaving none free "
                f"stalls every worker at the bucket barrier")

    def test_explicit_thread_count_wins_over_auto(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto",
                               num_threads=1) as field:
            assert field._algo_kwargs["num_threads"] == 1


class TestGeoreferencing:
    """The reach-ins the CIRED workflow had to make, now first class."""

    def test_transform_matches_the_window(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin) as field:
            assert field.transform == finder.raster_handler.window_transform

    def test_crs_is_the_raster_crs(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin) as field:
            assert field.crs.to_string() == "EPSG:25832"

    def test_transform_georeferences_the_field_array(self, tmp_path):
        """Origin cell in array space maps back to the origin coordinate."""
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin) as field:
            arr = field.field_array()
            row, col = np.unravel_index(int(np.argmin(arr)), arr.shape)
            x, y = rasterio.transform.xy(field.transform, row, col)
        assert abs(x - origin[0]) <= CELL
        assert abs(y - origin[1]) <= CELL

    def test_to_geotiff_roundtrips(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        out = tmp_path / "nested" / "field.tif"
        with finder.cost_field(origin, algorithm="auto") as field:
            written = field.to_geotiff(out)
            expected = field.field_array()
            transform = field.transform
        assert written == out
        with rasterio.open(out) as src:
            assert src.crs.to_string() == "EPSG:25832"
            assert src.transform == transform
            assert src.nodata == -1.0
            got = src.read(1, masked=True)
        np.testing.assert_allclose(
            got.filled(np.inf), expected, rtol=1e-4)

    def test_float32_readout_stays_within_one_ulp(self, tmp_path):
        """field_array scales in the OUTPUT dtype, not via float64.

        That saves a whole-field temporary, at the price of doing the
        multiply in float32. The result must still land within one ulp of
        the float64-then-round answer, or the saving is not free.
        """
        path = _raster(tmp_path, value=137, shape=(60, 60))
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            got = field.field_array(dtype=np.float32).ravel()
            exact = (field._tree.field() * field._scale).astype(np.float32)
        ok = np.isfinite(got) & np.isfinite(exact)
        assert ok.any()
        diff = np.abs(got[ok].astype(np.float64)
                      - exact[ok].astype(np.float64))
        assert np.all(diff <= np.spacing(exact[ok]) * 1.000001), (
            f"max deviation {diff.max():.3e} exceeds one float32 ulp")

    def test_float64_readout_is_unchanged(self, tmp_path):
        """The float64 path must be bit-identical to the old formulation."""
        path = _raster(tmp_path, value=211, shape=(50, 50))
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            got = field.field_array(dtype=np.float64).ravel()
            exact = field._tree.field() * field._scale
        np.testing.assert_array_equal(np.nan_to_num(got, posinf=-1.0),
                                      np.nan_to_num(exact, posinf=-1.0))

    def test_field_readout_does_not_corrupt_the_solver(self, tmp_path):
        """Asking for float32 must not hand back a VIEW of the workspace.

        The delta workspace is float32 and live; masking a view of it would
        write inf into the kernel's own labels and poison every later query.
        """
        path = _raster(tmp_path, value=100, shape=(50, 50))
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="auto") as field:
            before = field.cost_to(far)
            field.field_array(dtype=np.float32)
            field.field_array(dtype=np.float32)
            after = field.cost_to(far)
        assert np.isfinite(before)
        assert after == pytest.approx(before, rel=1e-9)

    def test_unreachable_cells_become_nodata(self, tmp_path):
        """An excluded block must read as nodata, not as a cost."""
        shape = (40, 40)
        data = np.full(shape, 100, dtype=np.uint16)
        data[:, 20] = 65535                      # a wall with no gap
        path = tmp_path / "walled.tif"
        transform = rasterio.transform.from_origin(
            400000.0, 5600000.0, CELL, CELL)
        with rasterio.open(
            path, "w", driver="GTiff", height=shape[0], width=shape[1],
            count=1, dtype="uint16", crs="EPSG:25832", transform=transform,
        ) as dst:
            dst.write(data, 1)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        out = tmp_path / "walled_field.tif"
        with finder.cost_field(origin, algorithm="auto") as field:
            field.to_geotiff(out)
        with rasterio.open(out) as src:
            band = src.read(1, masked=True)
        assert band.mask.any(), "the sealed-off half should be nodata"
        assert np.all(band.compressed() >= 0)

    def test_nodata_colliding_with_a_real_cost_is_refused(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        with finder.cost_field(origin) as field:
            arr = field.field_array()
            collide = float(arr[np.isfinite(arr)].max())
            with pytest.raises(ValueError, match="collides"):
                field.to_geotiff(tmp_path / "bad.tif", nodata=collide)

    def test_georeferencing_is_refused_after_close(self, tmp_path):
        path = _raster(tmp_path)
        origin, far = _corners(path)
        finder = _finder(path, origin, far)
        field = finder.cost_field(origin)
        field.close()
        with pytest.raises(RuntimeError):
            _ = field.transform
