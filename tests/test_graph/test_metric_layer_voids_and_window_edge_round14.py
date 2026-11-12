"""Round 14, second pair: two library defects found by the review of the first three.

A. A metric layer that does not cover the whole cost raster was read as 0 outside its coverage, without a word.
   ``reproject_to_grid`` let ``rasterio.warp.reproject`` start its destination at 0 (its docstring said NaN, and the
   warning "outside the source coverage default to 0" could never fire). Now an uncovered cell, a nodata cell of the
   file and a NaN cell are *void*: they still count as 0 for that layer (a coverage gap is not an exclusion zone, so
   routes do not change), but ONE ``PyorpsWarning`` per layer names the layer and the number of cells. A hard bound
   cannot be checked without a value, so a void cell is forbidden when ``hard_max`` or ``hard_min`` is set (before,
   the stored 0 passed ``hard_max`` and failed ``hard_min`` by accident). A layer that covers the grid gives the same
   bands as before. For ``derive: slope_from_dem`` the DEM's rule applies: cells the DEM does not cover are forbidden
   (before they were sea level, a wall of fake slope at the DEM's edge).

B. The search window floors both corners of the buffered hull, so with a small buffer the window could exclude the
   cell of one of the very points it was built from (``PyorpsError: Point (...) lies outside the search window`` on
   the next step). ``RasterHandler.window_from_bounds(..., points=...)`` now grows the window by exactly the missing
   rows and columns; a window that already held every point is returned unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import MultiPoint

from pyorps import PathFinder
from pyorps.core.exceptions import PyorpsError
from pyorps.core.metric_stack import MetricStack, reproject_to_grid
from pyorps.core.warning_types import PyorpsWarning
from pyorps.raster.handler import RasterHandler

N = 40
ORIGIN_X, ORIGIN_Y = 500000.0, 5600000.0
CRS = "EPSG:32632"
GRID = from_origin(ORIGIN_X, ORIGIN_Y, 1.0, 1.0)
HOLE = (slice(1, 11), slice(25, 35))      # 10 x 10 cells around the target of the default route


def _xy(row: int, col: int) -> tuple[float, float]:
    """Centre of a cell of the test grid."""
    return ORIGIN_X + col + 0.5, ORIGIN_Y - row - 0.5


SOURCE = _xy(5, 5)
TARGET = _xy(5, 30)


def _write(path, array, transform=GRID, crs=CRS, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=str(array.dtype), crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(array, 1)
    return str(path)


def _land() -> np.ndarray:
    """A strictly positive layer (no zeros), so that a 0 in a band can only come from a void."""
    rows, cols = np.mgrid[0:N, 0:N]
    return (1.0 + (cols % 7) * 0.5 + rows * 0.01).astype(np.float32)


def _holed(value: float) -> np.ndarray:
    land = _land()
    land[HOLE] = value
    return land


@pytest.fixture
def files(tmp_path):
    land = _land()
    return {
        "cost": _write(tmp_path / "cost.tif", np.full((N, N), 10, np.uint16)),
        "land_array": land,
        "land": _write(tmp_path / "land.tif", land),
        "land_left": _write(tmp_path / "land_left.tif", land[:, :20].copy()),
        "land_coarse": _write(tmp_path / "land_coarse.tif", land[::2, ::2].copy(),
                              from_origin(ORIGIN_X, ORIGIN_Y, 2.0, 2.0)),
        "land_nodata_negative": _write(tmp_path / "land_nodata_negative.tif", _holed(-9999.0), nodata=-9999.0),
        "land_nodata_zero": _write(tmp_path / "land_nodata_zero.tif", _holed(0.0), nodata=0.0),
        "land_nodata_forbidden": _write(tmp_path / "land_nodata_forbidden.tif", _holed(65535.0), nodata=65535.0),
        "land_nan_hole": _write(tmp_path / "land_nan_hole.tif", _holed(np.nan)),
        "dem": _write(tmp_path / "dem.tif",
                      (np.arange(N, dtype=np.float32)[None, :] * 0.05 + np.zeros((N, 1), np.float32))),
        "dem_left": _write(tmp_path / "dem_left.tif",
                           (np.arange(20, dtype=np.float32)[None, :] * 0.05 + np.zeros((N, 1), np.float32))),
    }


def _finder(files, layers, source=SOURCE, target=TARGET, buffer_m=80, **kwargs) -> PathFinder:
    kwargs.setdefault("neighborhood_str", "r1")
    kwargs.setdefault("objective", {"cost": 1, **{name: 1 for name in layers}})
    return PathFinder(files["cost"], source_coords=source, target_coords=target,
                      search_space_buffer_m=buffer_m, metric_layers=layers, **kwargs)


def _build(files, layers, **kwargs):
    """The finder and every warning its construction raised."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        finder = _finder(files, layers, **kwargs)
    return finder, record


def _about(record, name):
    return [w for w in record if f"Metric layer '{name}'" in str(w.message)]


# ===================================================================================== A. voids of a metric layer
def test_a_layer_file_that_covers_the_grid_partly_warns_once_with_name_and_count(files):
    """Before: nothing was said, and the right half of the grid silently read 0."""
    finder, record = _build(files, {"land": files["land_left"]})
    about = _about(record, "land")
    assert len(about) == 1, [str(w.message) for w in record]
    message = str(about[0].message)
    assert f"{N * 20} of {N * N} cell(s)" in message
    assert issubclass(about[0].category, PyorpsWarning) and issubclass(about[0].category, UserWarning)
    assert about[0].filename == __file__, "the warning names the line of the caller"
    land = finder.metric_stack["land"]
    assert np.array_equal(land[:, :20], _land()[:, :20])
    assert (land[:, 20:] == 0).all() and land.dtype == np.float32
    assert not finder.metric_stack.forbidden_mask[:, 20:].any(), "a coverage gap is not an exclusion zone"


def test_the_void_cells_count_as_zero_so_the_route_is_the_one_a_zero_layer_gives(files):
    """The behaviour is unchanged where the warning is new: the same route as with the zero-filled layer."""
    zero_filled = _land()
    zero_filled[:, 20:] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        partial = _finder(files, {"land": files["land_left"]}).find_route()
        explicit = _finder(files, {"land": zero_filled}).find_route()
    assert list(partial.path_indices) == list(explicit.path_indices)
    assert partial.feasibility == pytest.approx(explicit.feasibility)


def test_a_layer_with_nodata_cells_warns_with_their_number_and_does_not_use_the_nodata_value(files):
    """Before: a negative nodata value reached the layer, and ``add_layer`` refused it as a negative intensity."""
    finder, record = _build(files, {"land": files["land_nodata_negative"]})
    about = _about(record, "land")
    assert len(about) == 1 and "100 of 1600 cell(s)" in str(about[0].message)
    land = finder.metric_stack["land"]
    assert (land[HOLE] == 0).all()
    outside = np.ones((N, N), bool)
    outside[HOLE] = False
    assert np.array_equal(land[outside], _land()[outside]), "the cells around the void are untouched"
    assert not finder.metric_stack.forbidden_mask[HOLE].any()


def test_a_zero_nodata_value_is_counted_and_the_band_stays_the_same(files):
    finder, record = _build(files, {"land": files["land_nodata_zero"]})
    about = _about(record, "land")
    assert len(about) == 1 and "100 of 1600 cell(s)" in str(about[0].message)
    assert np.array_equal(finder.metric_stack["land"], _holed(0.0))


def test_nan_cells_of_a_layer_file_are_void_as_well(files):
    finder, record = _build(files, {"land": files["land_nan_hole"]})
    about = _about(record, "land")
    assert len(about) == 1 and "100 of 1600 cell(s)" in str(about[0].message)
    assert np.array_equal(finder.metric_stack["land"], _holed(0.0))


def test_a_nodata_value_of_65535_stays_the_forbidden_marker_and_is_not_a_void(files):
    """65535 is the sentinel of the layers themselves: those cells are forbidden, as they always were."""
    finder, record = _build(files, {"land": files["land_nodata_forbidden"]})
    assert not _about(record, "land")
    stack = finder.metric_stack
    assert stack.forbidden_mask[HOLE].all() and (stack["land"][HOLE] == 0).all()


def test_a_layer_that_covers_the_grid_gives_the_same_band_and_no_warning(files):
    finder, record = _build(files, {"land": files["land"]})
    assert not _about(record, "land")
    assert np.array_equal(finder.metric_stack["land"], _land())
    assert finder.metric_stack["land"].dtype == np.float32


def test_a_layer_of_another_resolution_that_covers_the_grid_is_bit_identical_to_the_old_reprojection(files):
    """The old call (destination 0, no nodata) is the reference: where nothing is uncovered, nothing changes."""
    with rasterio.open(files["land_coarse"]) as src:
        data, transform, crs = src.read(1), src.transform, src.crs
    reference = np.zeros((N, N), np.float32)
    reproject(source=np.ascontiguousarray(data, dtype=np.float32), destination=reference, src_transform=transform,
              src_crs=crs, dst_transform=GRID, dst_crs=CRS, resampling=Resampling.bilinear)
    finder, record = _build(files, {"land": files["land_coarse"]})
    assert not _about(record, "land")
    assert np.array_equal(finder.metric_stack["land"], reference)
    assert np.array_equal(reproject_to_grid(data, transform, crs, (N, N), GRID, CRS), reference)


def test_reproject_to_grid_gives_nan_outside_the_source_as_its_docstring_says(files):
    """The function itself: NaN where uncovered (it used to give 0), the covered values unchanged."""
    left = _land()[:, :20].copy()
    band = reproject_to_grid(left, GRID, CRS, (N, N), GRID, CRS)
    assert np.isnan(band[:, 20:]).all() and np.array_equal(band[:, :20], left)
    holed = reproject_to_grid(_holed(-9999.0), GRID, CRS, (N, N), GRID, CRS, src_nodata=-9999.0)
    assert np.isnan(holed[HOLE]).all() and int(np.isnan(holed).sum()) == 100


def test_an_array_with_its_own_transform_that_stops_inside_the_grid_warns_too(files):
    spec = {"source": _land()[:, :20].copy(), "transform": GRID, "crs": CRS}
    finder, record = _build(files, {"land": spec})
    about = _about(record, "land")
    assert len(about) == 1 and f"{N * 20} of {N * N} cell(s)" in str(about[0].message)
    assert (finder.metric_stack["land"][:, 20:] == 0).all()


def test_a_nan_in_an_array_with_its_own_transform_is_still_forbidden_not_void(files):
    """Only the cells the array does not cover are void; NaN as data is forbidden, as documented in ``add_layer``."""
    array = _land()
    array[HOLE] = np.nan
    spec = {"source": array, "transform": GRID, "crs": CRS}
    finder, record = _build(files, {"land": spec})
    assert not _about(record, "land")
    assert finder.metric_stack.forbidden_mask[HOLE].all()


def test_an_array_of_the_stack_shape_is_taken_as_it_is(files):
    array = _land()
    array[HOLE] = np.nan
    finder, record = _build(files, {"land": array})
    assert not _about(record, "land")
    assert finder.metric_stack.forbidden_mask[HOLE].all()


# ------------------------------------------------------------------------------------------------ hard bounds
@pytest.mark.parametrize("bound", [{"hard_max": 100.0}, {"hard_min": 0.5}, {"hard_max": 100.0, "hard_min": 0.5}])
def test_a_hard_bound_forbids_a_void_cell_instead_of_passing_or_failing_it_as_a_zero(files, bound):
    """Every covered cell satisfies the bound (1 <= land <= 5.1); only the void cells are the question.

    Before, ``hard_max`` let the stored 0 pass and ``hard_min`` forbade it (0 < 0.5), by accident of the fill value.
    Now both give the same, deliberate answer: a bound cannot be checked without a value, so the cell is forbidden.
    """
    finder, record = _build(files, {"land": {"source": files["land_left"], **bound}})
    stack = finder.metric_stack
    assert stack.forbidden_mask[:, 20:].all(), "void cells are forbidden under a bound"
    assert not stack.forbidden_mask[:, :20].any(), "covered cells that satisfy the bound stay allowed"
    message = str(_about(record, "land")[0].message)
    assert "forbidden because hard_max/hard_min is set" in message


def test_a_bound_still_forbids_a_covered_cell_that_violates_it(files):
    finder, _ = _build(files, {"land": {"source": files["land_left"], "hard_max": 3.0}})
    land = _land()[:, :20]
    mask = finder.metric_stack.forbidden_mask
    assert np.array_equal(mask[:, :20], land > 3.0)
    assert mask[:, 20:].all()


def test_a_layer_without_a_bound_forbids_nothing_where_it_is_void(files):
    finder, record = _build(files, {"land": {"source": files["land_left"]}})
    assert not finder.metric_stack.forbidden_mask.any()
    assert "and are not forbidden" in str(_about(record, "land")[0].message)


def test_a_bounded_layer_that_covers_the_grid_forbids_exactly_what_it_did_before(files):
    finder, record = _build(files, {"land": {"source": files["land"], "hard_max": 3.0}})
    assert not _about(record, "land")
    assert np.array_equal(finder.metric_stack.forbidden_mask, _land() > 3.0)


def test_add_layer_checks_the_void_mask_against_the_stack_shape():
    stack = MetricStack.from_single_raster(np.full((4, 6), 10, np.uint16), from_origin(0, 4, 1, 1), CRS)
    with pytest.raises(PyorpsError, match="shape"):
        stack.add_layer("x", np.zeros((4, 6), np.float32), hard_max=1.0, void=np.zeros((3, 3), bool))
    void = np.zeros((4, 6), bool)
    void[0, 0] = True
    stack.add_layer("x", np.zeros((4, 6), np.float32), hard_max=1.0, void=void)
    assert stack.forbidden_mask[0, 0] and int(stack.forbidden_mask.sum()) == 1
    stack.add_layer("y", np.zeros((4, 6), np.float32), void=void)           # no bound: not forbidden by it
    assert int(stack.forbidden_mask.sum()) == 1


# ---------------------------------------------------------------------------------------- slope from a partial DEM
def test_a_slope_layer_from_a_dem_that_covers_the_grid_partly_forbids_the_uncovered_cells(files):
    """Before: the uncovered DEM cells were 0 m, silently, with a wall of fake slope at the DEM's edge."""
    layers = {"steep": {"derive": "slope_from_dem"}}
    finder, record = _build(files, layers, dem=files["dem_left"], target=_xy(5, 15))
    about = [w for w in record if "DEM contains" in str(w.message)]
    assert len(about) >= 1
    assert f"{N * 20} non-finite cell(s) without elevation" in str(about[0].message)
    assert issubclass(about[0].category, PyorpsWarning)
    assert about[0].filename == __file__
    stack = finder.metric_stack
    assert stack.forbidden_mask[:, 20:].all()
    assert not stack.forbidden_mask[:, :20].any()
    full, _ = _build(files, layers, dem=files["dem"], target=_xy(5, 15))
    assert np.array_equal(stack["steep"][:, :18], full.metric_stack["steep"][:, :18]), "away from the gap: the same"


def test_a_slope_layer_from_a_dem_with_nodata_cells_forbids_them(tmp_path, files):
    dem = (np.arange(N, dtype=np.float32)[None, :] * 0.05 + np.zeros((N, 1), np.float32))
    dem[HOLE] = -9999.0
    path = _write(tmp_path / "dem_nodata.tif", dem, nodata=-9999.0)
    finder, record = _build(files, {"steep": {"derive": "slope_from_dem"}}, dem=path)
    assert any("100 non-finite cell(s) without elevation" in str(w.message) for w in record)
    assert finder.metric_stack.forbidden_mask[HOLE].all()


# ======================================================================================== B. the search window
CORNER_POINTS = [(ORIGIN_X + 5.0, ORIGIN_Y - 5.0), (ORIGIN_X + 15.0, ORIGIN_Y - 30.0),
                 (ORIGIN_X + 30.0, ORIGIN_Y - 5.0)]


def test_a_given_point_on_a_cell_corner_stays_inside_the_window_with_a_tiny_buffer():
    """The reproduction: ``[SOURCE, outside, TARGET]`` with ``search_space_buffer_m=1`` on 1 m cells.

    Before: the window was 26 x 26 cells, the second point's cell (row 30) was outside it, and the route stopped
    with ``PyorpsError: Point (...) lies outside the search window``.
    """
    cost = np.full((N, N), 10, np.uint16)
    finder = PathFinder(cost, CORNER_POINTS, [CORNER_POINTS[-1]], search_space_buffer_m=1, neighborhood_str="r1",
                        crs=CRS, transform=GRID)
    handler = finder.raster_handler
    indices = handler.coords_to_indices(CORNER_POINTS, check_bounds=True)        # raised before
    n_rows, n_cols = handler.data.shape[-2:]
    assert ((indices >= 0).all() and (indices[:, 0] < n_rows).all() and (indices[:, 1] < n_cols).all())
    session = finder.search_session(algorithm="dijkstra")
    try:
        assert len(session.route(CORNER_POINTS).path_indices) > 1
    finally:
        session.close()


def test_the_metric_pipeline_with_a_pre_cut_stack_holds_the_corner_points_too(files):
    """The objective path cuts the stack to the corridor before the handler: it must cut the same window."""
    finder = PathFinder(files["cost"], CORNER_POINTS, [CORNER_POINTS[-1]], search_space_buffer_m=1,
                        neighborhood_str="r1", objective={"cost": 1, "land": 1}, metric_layers={"land": files["land"]})
    finder.raster_handler.coords_to_indices(CORNER_POINTS, check_bounds=True)
    session = finder.search_session(algorithm="dijkstra")
    try:
        assert len(session.route(CORNER_POINTS).path_indices) > 1
    finally:
        session.close()


def test_a_session_whose_window_is_rebuilt_for_a_corner_point_keeps_working(files):
    finder = PathFinder(files["cost"], SOURCE, TARGET, search_space_buffer_m=1, neighborhood_str="r1")
    session = finder.search_session(algorithm="dijkstra")
    try:
        session.route([SOURCE, TARGET])
        assert len(session.update(CORNER_POINTS).path_indices) > 1
    finally:
        session.close()


def _hull_bounds(points, buffer_m):
    return MultiPoint(points).convex_hull.buffer(distance=buffer_m, quad_segs=32).bounds


def _bounds_of(window):
    return (int(window.row_off), int(window.col_off), int(window.row_off + window.height),
            int(window.col_off + window.width))


def test_a_window_that_already_held_every_point_is_returned_unchanged_and_others_grow_minimally():
    """Property test over random point sets and buffers (seeded): the old window is the reference."""
    rng = np.random.default_rng(14)
    unchanged = grown = 0
    for _ in range(600):
        count = int(rng.integers(2, 6))
        points = [(ORIGIN_X + float(rng.integers(0, 4 * N)) / 4 + (0.0 if rng.random() < 0.5 else 0.37),
                   ORIGIN_Y - float(rng.integers(0, 4 * N)) / 4 - (0.0 if rng.random() < 0.5 else 0.21))
                  for _ in range(count)]
        buffer_m = float(rng.choice([0.05, 0.4, 1.0, 1.0, 2.0, 3.5, 7.0, 15.0]))
        bounds = _hull_bounds(points, buffer_m)
        old = RasterHandler.window_from_bounds(bounds, GRID, (N, N))
        new = RasterHandler.window_from_bounds(bounds, GRID, (N, N), points=points)
        old_rows0, old_cols0, old_rows1, old_cols1 = _bounds_of(old)
        cells = [(int(np.floor(ORIGIN_Y - y)), int(np.floor(x - ORIGIN_X))) for x, y in points]
        cells = [(r, c) for r, c in cells if 0 <= r < N and 0 <= c < N]
        holds = all(old_rows0 <= r < old_rows1 and old_cols0 <= c < old_cols1 for r, c in cells)
        if holds:
            assert new == old and _bounds_of(new) == _bounds_of(old)
            unchanged += 1
            continue
        grown += 1
        rows0, cols0, rows1, cols1 = _bounds_of(new)
        assert all(rows0 <= r < rows1 and cols0 <= c < cols1 for r, c in cells), (points, buffer_m)
        assert (rows0 <= old_rows0 and cols0 <= old_cols0
                and rows1 >= old_rows1 and cols1 >= old_cols1), "never smaller"
        assert 0 <= rows0 and 0 <= cols0 and rows1 <= N and cols1 <= N, "clipped to the raster"
        # minimal: exactly the union of the old window and the missing cells, nothing more
        assert rows0 == min([old_rows0] + [r for r, _ in cells]) and cols0 == min([old_cols0] + [c for _, c in cells])
        assert rows1 == max([old_rows1] + [r + 1 for r, _ in cells])
        assert cols1 == max([old_cols1] + [c + 1 for _, c in cells])
    assert unchanged > 100 and grown > 50, (unchanged, grown)      # the sample exercises both branches


@pytest.mark.parametrize("buffer_m", [3.0, 5.0, 10.0, 80.0])
def test_the_handler_window_of_a_request_that_held_its_points_is_the_historical_one(buffer_m):
    """Pinned windows: for these buffers every point was already inside, so the window is the old one."""
    cost = np.full((N, N), 10, np.uint16)
    points = [SOURCE, _xy(30, 15), TARGET]
    finder = PathFinder(cost, points, [TARGET], search_space_buffer_m=buffer_m, neighborhood_str="r1", crs=CRS,
                        transform=GRID)
    handler = finder.raster_handler
    old = RasterHandler.window_from_bounds(handler.buffer_geometry.bounds, GRID, (N, N))
    assert handler.window == old
    assert (int(handler.window.col_off), int(handler.window.row_off), int(handler.window.width),
            int(handler.window.height)) == (int(old.col_off), int(old.row_off), int(old.width), int(old.height))


def test_a_point_outside_the_raster_stays_outside_and_is_still_reported():
    outside = (ORIGIN_X - 50.0, ORIGIN_Y - 5.0)
    bounds = _hull_bounds([SOURCE, TARGET], 1.0)
    without = RasterHandler.window_from_bounds(bounds, GRID, (N, N))
    assert RasterHandler.window_from_bounds(bounds, GRID, (N, N), points=[outside, (float("nan"), 1.0)]) == without
    cost = np.full((N, N), 10, np.uint16)
    finder = PathFinder(cost, SOURCE, TARGET, search_space_buffer_m=1, neighborhood_str="r1", crs=CRS, transform=GRID)
    with pytest.raises(PyorpsError, match="outside the search window"):
        finder.raster_handler.coords_to_indices([outside], check_bounds=True)


def test_the_corridor_points_are_every_given_source_and_target():
    cost = np.full((N, N), 10, np.uint16)
    finder = PathFinder(cost, SOURCE, TARGET, search_space_buffer_m=1, neighborhood_str="r1", crs=CRS, transform=GRID)
    assert finder._corridor_points() == [SOURCE, TARGET]
    many = PathFinder(cost, CORNER_POINTS, [CORNER_POINTS[-1]], search_space_buffer_m=1, neighborhood_str="r1",
                      crs=CRS, transform=GRID)
    assert many._corridor_points() == CORNER_POINTS + [CORNER_POINTS[-1]]
    bare = PathFinder(cost, None, None, search_space_buffer_m=1, neighborhood_str="r1", crs=CRS, transform=GRID)
    assert bare._corridor_points() is None
