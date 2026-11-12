"""Round 14: three library defects found by the GUI package.

1. ``SearchSession`` rebuilt its window with a finder that lacked some of the old finder's constructor settings
   (``metric_layers``: ``ObjectiveError: Objective references unknown metric layer(s)``; also ``weight_precision``,
   ``graph_api_options``, ``steps``, the raster read options ...). The clone now takes over every parameter of
   ``PathFinder.__init__`` that is not on an explicit exclusion list, and a test fails when the signature gains a
   parameter that is on neither side.
2. A DEM in another coordinate system, or one that does not touch the search window, stopped ``PathFinder`` with
   rasterio's "Number of columns or rows must be non-negative". It is now an ``InvalidInputError`` that names both
   coordinate systems or both bounds. A DEM that covers the window only partly stays allowed: the cells it does
   not cover are forbidden (they were silently sea level before).
3. A source or target on a cell without DEM elevation started on a forbidden cell on the first route of a finder
   (``NoPathFoundError`` with a message about the cost raster) and was moved by every later one. It is corrected
   at once, like a point on any other forbidden cell, with a warning that names the DEM.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from pyorps import PathFinder
from pyorps.core.exceptions import InvalidInputError, NoPathFoundError, PyorpsError
from pyorps.graph import search_session as session_module
from pyorps.io.geo_dataset import InMemoryRasterDataset, initialize_geo_dataset
from pyorps.utils.neighborhood import get_neighborhood_steps

N = 40
ORIGIN_X, ORIGIN_Y = 500000.0, 5600000.0
CRS = "EPSG:32632"
GRID = from_origin(ORIGIN_X, ORIGIN_Y, 1.0, 1.0)


def _xy(row: int, col: int) -> tuple[float, float]:
    """Centre of a cell of the test grid."""
    return ORIGIN_X + col + 0.5, ORIGIN_Y - row - 0.5


SOURCE = _xy(5, 5)
TARGET = _xy(5, 30)
ABOVE_THE_FIRST_WINDOW = _xy(30, 15)   # far below the 1 m strip around row 5


def _write(path, array, transform=GRID, crs=CRS, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=str(array.dtype), crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(array, 1)
    return str(path)


def _dem_array() -> np.ndarray:
    return (np.arange(N, dtype=np.float32)[None, :] * 0.05 + np.zeros((N, 1), np.float32)).astype(np.float32)


@pytest.fixture
def files(tmp_path):
    """Cost raster, a layer on its grid and DEMs: matching, in another CRS, elsewhere and partial."""
    dem = _dem_array()
    wgs84 = transform_bounds(CRS, "EPSG:4326", ORIGIN_X, ORIGIN_Y - N, ORIGIN_X + N, ORIGIN_Y)
    wgs_grid = from_origin(wgs84[0], wgs84[3], (wgs84[2] - wgs84[0]) / N, (wgs84[3] - wgs84[1]) / N)
    land = np.zeros((N, N), np.float32)
    land[:, 20:] = 3.0
    return {
        "cost": _write(tmp_path / "cost.tif", np.full((N, N), 10, np.uint16)),
        "land": _write(tmp_path / "land.tif", land),
        "land_wgs84": _write(tmp_path / "land_wgs84.tif", land, wgs_grid, "EPSG:4326"),
        "dem": _write(tmp_path / "dem.tif", dem),
        "dem_array": dem,
        "dem_wgs84": _write(tmp_path / "dem_wgs84.tif", dem, wgs_grid, "EPSG:4326"),
        "dem_elsewhere": _write(tmp_path / "dem_elsewhere.tif", dem, from_origin(ORIGIN_X + 10000, ORIGIN_Y, 1, 1)),
        "dem_left_half": _write(tmp_path / "dem_left_half.tif", dem[:, :20].copy()),
        "dem_nodata": _write(tmp_path / "dem_nodata.tif", _with_hole(dem, -9999.0), nodata=-9999.0),
        "dem_nan_hole": _write(tmp_path / "dem_nan_hole.tif", _with_hole(dem, np.nan)),
    }


def _with_hole(dem: np.ndarray, value: float) -> np.ndarray:
    """A DEM with a 10 x 10 void around the cell of TARGET (row 5, column 30)."""
    holed = dem.copy()
    holed[1:11, 25:35] = value
    return holed


def _finder(files, source=SOURCE, target=TARGET, buffer_m=80, **kwargs) -> PathFinder:
    kwargs.setdefault("neighborhood_str", "r1")
    return PathFinder(files["cost"], source_coords=source, target_coords=target, search_space_buffer_m=buffer_m,
                      **kwargs)


# ============================================================================== 1. the clone of a rebuilt window
def test_a_session_that_rebuilds_its_window_keeps_the_metric_layers(files):
    """The reproduction: ``update`` with a point outside the first window raised ObjectiveError before."""
    finder = _finder(files, buffer_m=1, objective={"cost": 1, "land": 1}, metric_layers={"land": files["land"]})
    session = finder.search_session(algorithm="dijkstra")
    try:
        session.route([SOURCE, TARGET])
        assert ABOVE_THE_FIRST_WINDOW[1] < SOURCE[1] - 10        # outside the 1 m strip around row 5
        updated = session.update([SOURCE, ABOVE_THE_FIRST_WINDOW, TARGET])
        assert session.finder is not finder
        assert session.finder.metric_layers == {"land": files["land"]}
        assert len(updated.path_indices) > 1
        assert "land" in session.finder.metric_stack.layer_names
    finally:
        session.close()


def test_a_rebuilt_window_prices_the_route_like_a_finder_built_for_it(files):
    kwargs = {"objective": {"cost": 1, "land": 1}, "metric_layers": {"land": files["land"]}}
    points = [SOURCE, ABOVE_THE_FIRST_WINDOW, TARGET]
    session = _finder(files, buffer_m=1, **kwargs).search_session(algorithm="dijkstra")
    try:
        session.route([SOURCE, TARGET])
        updated = session.update(points)
        fresh = _finder(files, source=points, target=[TARGET], buffer_m=1, **kwargs).search_session(
            algorithm="dijkstra")
        try:
            assert list(updated.path_indices) == list(fresh.route(points).path_indices)
        finally:
            fresh.close()
    finally:
        session.close()


def _signature_parameters() -> tuple[set[str], bool]:
    parameters = inspect.signature(PathFinder.__init__).parameters
    named = {name for name, p in parameters.items()
             if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD}
    return named, any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def test_the_clone_handles_every_constructor_parameter(files):
    """Fails the day ``PathFinder.__init__`` gains a parameter that the clone neither passes nor excludes.

    To fix a failure: pass the new setting in ``search_session._clone_arguments`` (the finder must keep it as an
    attribute), or, if a rebuilt window must not take it over, list it in ``_CLONE_NOT_CARRIED`` with the reason.
    """
    named, has_var_keyword = _signature_parameters()
    assert has_var_keyword, "PathFinder.__init__ lost its **kwargs: the clone passes the raster read options on"
    passed = set(session_module._clone_arguments(_finder(files), [SOURCE, TARGET]))
    excluded = set(session_module._CLONE_NOT_CARRIED)
    assert named - passed - excluded == set(), (
        f"PathFinder.__init__ parameters the window rebuild forgets: {sorted(named - passed - excluded)}")
    assert passed - named == set(), "the clone passes a name PathFinder.__init__ does not take"
    assert passed & excluded == set(), "a parameter is both passed and excluded"
    assert excluded <= named
    assert all(isinstance(reason, str) and len(reason) > 20 for reason in session_module._CLONE_NOT_CARRIED.values())


def test_the_clone_takes_over_every_constructor_setting_of_the_finder(files):
    """Every parameter that can be set on a small raster gets a non-default value; the rebuilt finder has them."""
    r1_steps = get_neighborhood_steps("r1")
    assert len(r1_steps) != len(get_neighborhood_steps("r3"))
    kwargs = {
        "steps": r1_steps,
        "neighborhood_str": "r3",
        "ignore_max_cost": False,
        "graph_api": "networkx",
        "graph_api_options": {"remove_isolated_nodes": False},
        "weight_precision": "float32",
        "objective": {"cost": 1, "land": 0.5},
        "metric_layers": {"land": files["land"]},
        "dem": files["dem"],
        "dem_kwargs": {"gradient_function": "power"},
        "corridor_first": False,
        "use_gpu": True,
        "masked": False,                                   # a raster read option (the ``**kwargs``)
    }
    finder = _finder(files, **kwargs)
    clone = session_module._clone_finder(finder, [SOURCE, ABOVE_THE_FIRST_WINDOW, TARGET])

    assert clone is not finder
    assert np.array_equal(clone.steps, r1_steps)
    assert clone.neighborhood_str == "r3"
    assert clone.ignore_max_cost is False
    assert clone.graph_api_name == "networkx"
    assert clone.graph_api_options == {"remove_isolated_nodes": False}
    assert clone.graph_api_options is not finder.graph_api_options
    assert clone.weight_precision == "float32"
    assert clone.objective is finder.objective
    assert clone.metric_layers == {"land": files["land"]}
    assert clone.dem_dataset is finder.dem_dataset and clone.dem_kwargs == {"gradient_function": "power"}
    assert clone.corridor_first is False
    assert clone.use_gpu is True
    assert clone._raster_arguments["kwargs"] == {"masked": False}
    assert clone._explicit_buffer_m == finder._explicit_buffer_m == 80
    assert clone.dataset.file_source == files["cost"]
    assert clone.source_coords == [SOURCE, ABOVE_THE_FIRST_WINDOW, TARGET]
    assert clone.target_coords == TARGET


def test_the_clone_passes_cost_assumptions_and_dataset_modifications_on(files):
    finder = _finder(files)
    marker_assumptions, marker_modifications = {"category": {"road": 1}}, [{"name": "overlay"}]
    finder._raster_arguments = {"cost_assumptions": marker_assumptions, "datasets_to_modify": marker_modifications,
                                "kwargs": {}}
    arguments = session_module._clone_arguments(finder, [SOURCE, TARGET])
    assert arguments["cost_assumptions"] is marker_assumptions
    assert arguments["datasets_to_modify"] is marker_modifications


def test_a_finder_without_coordinates_hands_its_raster_arguments_to_the_clone(files):
    """The first routing call builds the raster; what it was built with is what the clone gets."""
    finder = PathFinder(files["cost"], None, None, search_space_buffer_m=80, neighborhood_str="r1", masked=False)
    finder.find_route(SOURCE, TARGET)
    assert finder._raster_arguments["kwargs"] == {"masked": False}
    assert session_module._clone_arguments(finder, [SOURCE, TARGET])["masked"] is False


def test_the_clone_of_an_in_memory_raster_still_reads_the_whole_raster():
    cost = np.full((N, N), 10, np.uint16)
    finder = PathFinder(cost, SOURCE, TARGET, search_space_buffer_m=1, neighborhood_str="r1", crs=CRS, transform=GRID)
    session = finder.search_session(algorithm="dijkstra")
    try:
        session.route([SOURCE, TARGET])
        assert len(session.update([SOURCE, ABOVE_THE_FIRST_WINDOW, TARGET]).path_indices) > 1
    finally:
        session.close()


# ============================================================================== 2. a DEM the library cannot use
def _raises_invalid_input(files, dem, match, **kwargs):
    with pytest.raises(InvalidInputError, match=match) as raised:
        _finder(files, dem=dem, **kwargs)
    assert isinstance(raised.value, PyorpsError)
    return str(raised.value)


@pytest.mark.parametrize("objective", [None, {"cost": 1}])
def test_a_dem_file_in_another_crs_names_both_systems_and_says_what_to_do(files, objective):
    message = _raises_invalid_input(files, files["dem_wgs84"], "EPSG:4326", objective=objective)
    assert "EPSG:32632" in message
    assert "reproject" in message and "does not reproject" in message


def test_a_dem_file_that_does_not_overlap_gives_both_bounds(files):
    message = _raises_invalid_input(files, files["dem_elsewhere"], "does not overlap")
    assert "510000.00 to 510040.00" in message          # the DEM
    assert "500000.00 to 500040.00" in message          # the search window
    assert "Clip or extend the DEM" in message


def test_the_check_applies_to_a_dem_given_as_an_open_dataset(files):
    wgs84 = initialize_geo_dataset(files["dem_wgs84"])
    _raises_invalid_input(files, wgs84, "EPSG:4326")
    elsewhere = initialize_geo_dataset(files["dem_elsewhere"])
    _raises_invalid_input(files, elsewhere, "does not overlap")
    in_memory = InMemoryRasterDataset(files["dem_array"], "EPSG:4326", GRID)
    _raises_invalid_input(files, in_memory, "EPSG:4326")
    _finder(files, dem=initialize_geo_dataset(files["dem"])).find_route()      # the same CRS routes


def test_the_check_applies_to_a_dem_given_as_a_tuple_with_its_own_crs_and_transform(files):
    dem = files["dem_array"]
    _raises_invalid_input(files, (dem, "EPSG:4326", GRID), "EPSG:4326")
    _raises_invalid_input(files, (dem, CRS, from_origin(ORIGIN_X + 10000, ORIGIN_Y, 1, 1)), "does not overlap")
    path = _finder(files, dem=(dem, CRS, GRID)).find_route()
    assert len(path.path_indices) > 1


def test_a_dem_array_takes_the_crs_and_transform_of_the_raster_and_routes():
    cost = np.full((N, N), 10, np.uint16)
    dem = _dem_array()
    finder = PathFinder(cost, SOURCE, TARGET, search_space_buffer_m=80, neighborhood_str="r1", crs=CRS,
                        transform=GRID, dem=dem)
    assert len(finder.find_route().path_indices) > 1


def test_a_dem_array_that_is_too_small_for_its_transform_still_overlaps_partly():
    cost = np.full((N, N), 10, np.uint16)
    finder = PathFinder(cost, SOURCE, _xy(5, 15), search_space_buffer_m=80, neighborhood_str="r1", crs=CRS,
                        transform=GRID, dem=_dem_array()[:, :20].copy())
    assert len(finder.find_route().path_indices) > 1


def test_the_error_comes_before_any_route_and_with_a_session_too(files):
    with pytest.raises(InvalidInputError, match="EPSG:4326"):
        _finder(files, dem=files["dem_wgs84"], objective={"cost": 1}).search_session()


def test_a_dem_that_covers_the_window_only_partly_stays_allowed_and_its_gap_is_forbidden(files):
    finder = _finder(files, target=_xy(5, 15), dem=files["dem_left_half"])
    path = finder.find_route()
    assert len(path.path_indices) > 1
    data = finder.raster_handler.data[0]
    assert (data[:, 22:] == 65535).all(), "cells the DEM does not cover are forbidden, not sea level"
    assert not (data[:, :18] == 65535).any()
    cols = {int(c) % N for c in path.path_indices}
    assert max(cols) < 20


def test_nodata_cells_of_a_dem_file_and_nan_voids_are_forbidden_alike(files):
    for key in ("dem_nodata", "dem_nan_hole"):
        finder = _finder(files, dem=files[key])
        finder.find_route()
        hole = finder.raster_handler.data[0][1:11, 25:35]
        assert (hole == 65535).sum() >= 60, key


def test_a_metric_layer_file_in_another_crs_is_reprojected_not_refused(files):
    """The DEM check does not apply to ``metric_layers``: they are reprojected onto the stack grid."""
    finder = _finder(files, objective={"cost": 1, "land": 1}, metric_layers={"land": files["land_wgs84"]})
    path = finder.find_route()
    land = finder.metric_stack["land"]
    assert land.shape == (N, N) and float(np.nanmax(land)) == pytest.approx(3.0)
    assert len(path.path_indices) > 1


# ========================================================================== 3. a point on a cell without DEM elevation
def _route_twice(finder, caught) -> list[list[int]]:
    routes = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            routes.append([int(i) for i in finder.find_route().path_indices])
        caught.append([str(w.message) for w in record if issubclass(w.category, UserWarning)])
    return routes


VARIANTS = [
    pytest.param({}, id="bare-dem-uint16"),
    pytest.param({"objective": {"cost": 1}}, id="objective-uint16"),
    pytest.param({"objective": {"cost": 1}, "weight_precision": "float32", "graph_api": "networkx"},
                 id="objective-float32"),
]


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_target_in_a_dem_void_is_corrected_on_the_first_route_like_on_every_later_one(files, variant):
    """Before: the first route raised NoPathFoundError, the second one corrected the point."""
    finder = _finder(files, dem=files["dem_nan_hole"], **variant)
    caught: list[list[str]] = []
    first, second = _route_twice(finder, caught)
    assert first == second
    target_cell = 5 * N + 30
    assert target_cell not in first, "the route must not end on the cell without elevation"
    corrected = [m for m in caught[0] if "forbidden" in m or "maximum cost" in m]
    assert corrected and "DEM" in corrected[0]
    assert "DEM" in " ".join(caught[1])


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_source_outside_the_dem_is_corrected_too(files, variant):
    finder = _finder(files, source=_xy(5, 30), target=_xy(5, 5), dem=files["dem_nan_hole"], **variant)
    first, second = _route_twice(finder, [])
    assert first == second and 5 * N + 30 not in first


def test_a_target_outside_the_dem_is_moved_to_the_edge_of_the_dem(files):
    finder = _finder(files, target=_xy(5, 30), dem=files["dem_left_half"])
    path = finder.find_route()
    assert max(int(i) % N for i in path.path_indices) < 20


def test_without_a_dem_the_correction_of_a_forbidden_cost_cell_is_unchanged(tmp_path):
    cost = np.full((N, N), 10, np.uint16)
    cost[3:8, 28:33] = 65535
    path = _write(tmp_path / "blocked.tif", cost)
    finder = PathFinder(path, SOURCE, TARGET, search_space_buffer_m=80, neighborhood_str="r1")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        finder.find_route()
    messages = [str(w.message) for w in record]
    corrected = [m for m in messages if "had maximum cost value" in m]
    assert len(corrected) == 1 and "DEM" not in corrected[0]
    assert corrected[0].startswith("1 position(s) had maximum cost value (65535) and were corrected:\n")


def test_a_cost_cell_that_is_forbidden_and_has_a_dem_is_corrected_without_the_dem_note(files, tmp_path):
    cost = np.full((N, N), 10, np.uint16)
    cost[3:8, 28:33] = 65535
    blocked = _write(tmp_path / "blocked.tif", cost)
    finder = PathFinder(blocked, SOURCE, TARGET, search_space_buffer_m=80, neighborhood_str="r1", dem=files["dem"])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        finder.find_route()
    corrected = [str(w.message) for w in record if "had maximum cost value" in str(w.message)]
    assert len(corrected) == 1 and "no elevation" not in corrected[0]


def test_a_point_on_a_dem_void_stays_where_it_is_when_forbidden_cells_are_traversable(files):
    """``ignore_max_cost=False`` never corrects a point; it holds for a DEM void like for a cost cell."""
    finder = _finder(files, dem=files["dem_nan_hole"], ignore_max_cost=False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            finder.find_route()
        except NoPathFoundError:
            pass
    assert not any("corrected" in str(w.message) for w in record)


def test_the_correction_warning_names_the_line_of_the_caller(files):
    finder = _finder(files, dem=files["dem_nan_hole"])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        finder.find_route()
    corrected = [w for w in record if "DEM has no elevation" in str(w.message)]
    assert len(corrected) == 1
    assert corrected[0].filename == __file__
