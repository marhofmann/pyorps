"""Round 16, package R3-C: the length of a route along the terrain model, and sessions for routes with extra metric
layers.

Headless tests. Section A: the terrain length (service, new routes, edited routes, no terrain model, the routes
grid, the notices, the attributes, the export, the project file). Section B: ``objective.force_direct`` for extra
metric layers: a retained session must give the route a direct ``find_route`` gives, for a new route, after a window
rebuild and after a route edit (the proof that lets the GUI use sessions for such routes again).

The closed form of section A: a DEM that rises 1 m per cell column on cells of 10 m (a 10 % ramp) over a uniform cost
raster. The least-cost route along a row is the straight run east, its map length is ``n * 10`` m and its length along
the terrain is ``n * hypot(10, 1)`` m, whatever the backend.
"""
from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest
import rasterio
from conftest import SOURCE, TARGET, TEST_CRS, invoke
from rasterio.transform import from_origin
from shapely.geometry import LineString

from pyorps import PathFinder
from pyorps.gui import ids, layout
from pyorps.gui.callbacks import attrs as attrs_cb
from pyorps.gui.callbacks import edit as edit_cb
from pyorps.gui.callbacks import groups as groups_cb
from pyorps.gui.callbacks import routes as routes_cb
from pyorps.gui.services import cost, project_io, routing
from pyorps.gui.services import objective as obj

RUN_WIRING = (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN)
LIBRARY_BACKENDS = ("networkx", "networkit", "igraph", "rustworkx")
ALL_BACKENDS = ("cython", *LIBRARY_BACKENDS)


def _installed(name):
    return importlib.util.find_spec(name) is not None


def _need(graph_api):
    if graph_api != "cython" and not _installed(graph_api):
        pytest.skip(f"{graph_api} is not installed")


def _write_tif(path, array, transform, crs=TEST_CRS):
    height, width = array.shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype=str(array.dtype),
                       crs=crs, transform=transform) as dst:
        dst.write(array, 1)
    return str(path)


def _close_sessions(state):
    for session in list(state.search_sessions.values()):
        session.close()


def _notices(resp):
    return resp[ids.NOTICES]["data"]


# ============================================================ the ramp (closed form)
RAMP_ROWS, RAMP_COLS, CELL = 30, 200, 10.0
RAMP_X, RAMP_Y = 500000.0, 5600000.0
RUN_ROW = 15
#: one cell column east: 10 m on the map and 1 m up (a 10 % ramp)
STEP_3D = math.hypot(CELL, 1.0)


def _centre(row, col):
    return (RAMP_X + (col + 0.5) * CELL, RAMP_Y - (row + 0.5) * CELL)


RAMP_SOURCE, RAMP_TARGET = _centre(RUN_ROW, 5), _centre(RUN_ROW, 55)


@pytest.fixture(scope="module")
def ramp_files(tmp_path_factory):
    """The uniform cost raster, a DEM on the same grid (1 m per column), a DEM of the same ramp on coarser 25 m
    cells that reaches beyond the raster, and a flat metric layer for the runs without a DEM."""
    folder = tmp_path_factory.mktemp("terrain_ramp")
    transform = from_origin(RAMP_X, RAMP_Y, CELL, CELL)
    columns = np.arange(RAMP_COLS, dtype="float32")
    ramp = np.tile(columns, (RAMP_ROWS, 1)).astype("float32")
    coarse_cols, coarse_rows = 84, 16                    # x from RAMP_X - 50 to RAMP_X + 2050, 25 m cells
    coarse_x = RAMP_X - 50.0 + (np.arange(coarse_cols) + 0.5) * 25.0
    coarse = np.tile((0.1 * (coarse_x - RAMP_X)).astype("float32"), (coarse_rows, 1))
    return {
        "cost": _write_tif(folder / "cost.tif", np.full((RAMP_ROWS, RAMP_COLS), 10, dtype="uint16"), transform),
        "dem": _write_tif(folder / "dem.tif", ramp, transform),
        "dem_coarse": _write_tif(folder / "dem_coarse.tif", coarse,
                                 from_origin(RAMP_X - 50.0, RAMP_Y + 50.0, 25.0, 25.0)),
        "flat": _write_tif(folder / "flat.tif", np.zeros((RAMP_ROWS, RAMP_COLS), dtype="float32"), transform),
    }


def _ramp_kwargs(files, graph_api, dem="dem", **overrides):
    settings = {"weights": {"cost": 1.0}, "dem_layer_id": "dem"}
    kwargs = obj.run_kwargs(settings, layer_path={"dem": files[dem]}, graph_api=graph_api)
    kwargs.update(overrides)
    return kwargs


def _ramp_route(files, graph_api, *, dem="dem", target=RAMP_TARGET, waypoints=(), **overrides):
    return routing.run_routing(
        files["cost"], sources=[RAMP_SOURCE], targets=[target], waypoints=list(waypoints), graph_api=graph_api,
        algorithm="dijkstra", neighborhood="r2", search_buffer_m=100.0,
        **_ramp_kwargs(files, graph_api, dem, **overrides))


def _straight_east(line):
    ys = {round(y, 6) for _x, y in line.coords}
    return len(ys) == 1


@pytest.fixture()
def ramp_layers(state, ramp_files):
    def add(name, path):
        return state.add_layer(name, "raster", crs=TEST_CRS, meta={"source_path": path})

    return {"cost": add("cost", ramp_files["cost"]), "dem": add("DGM", ramp_files["dem"]),
            "flat": add("flat", ramp_files["flat"])}


def _draft(source=RAMP_SOURCE, target=RAMP_TARGET):
    return {"sources": [{"lat": 0, "lng": 0, "x": source[0], "y": source[1]}],
            "targets": [{"lat": 0, "lng": 0, "x": target[0], "y": target[1]}], "waypoints": []}


def _gui_run(app, raster_id, graph_api="cython", *, algorithm="dijkstra", k=2, buffer=100, draft=None):
    args = (1, False, draft or _draft(), raster_id, algorithm, graph_api, k, buffer, False, True, False, False,
            1.0, 100, 0, None, [], [])
    return invoke(app, RUN_WIRING, *args, triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _gui_route(app, state, layers, graph_api, settings):
    state.gui_settings["objective"] = settings
    resp = _gui_run(app, layers["cost"].id, graph_api)
    routes = state.layers_of_kind("route")
    assert routes, _notices(resp)
    return routes[-1], resp


def _dem_settings(layers):
    return {"weights": {"cost": 1.0}, "dem_layer_id": layers["dem"].id}


# ------------------------------------------------------------------ A: new routes
@pytest.mark.parametrize("graph_api", ALL_BACKENDS)
@pytest.mark.parametrize("dem", ["dem", "dem_coarse"])
def test_a_straight_run_up_a_ten_percent_ramp_has_the_closed_form_terrain_length(ramp_files, graph_api, dem):
    """New route, cython (a retained session) and every installed graph library; the coarse DEM is read through the
    library's own alignment to the search window, so a ramp on another grid gives the same length."""
    _need(graph_api)
    finder, built, failed = _ramp_route(ramp_files, graph_api, dem=dem)
    try:
        assert failed == [] and len(built) == 1
        route = built[0]
        assert _straight_east(route.line)
        assert route.cost.total_length == pytest.approx(50 * CELL)                    # the map length stays 2-D
        assert route.cost.terrain_length_m == pytest.approx(50 * STEP_3D, rel=1e-5)
        assert route.cost.terrain_length_m > route.cost.total_length
        assert (route.session is not None) == (graph_api == "cython")
    finally:
        for item in built:
            if item.session is not None:
                item.session.close()


def test_a_direct_cython_route_has_the_same_terrain_length_as_a_session_route(ramp_files):
    _f, session_routes, _ = _ramp_route(ramp_files, "cython", force_direct=False)
    _f, direct_routes, _ = _ramp_route(ramp_files, "cython", force_direct=True)
    assert session_routes[0].session is not None and direct_routes[0].session is None
    session_routes[0].session.close()
    assert direct_routes[0].cost.terrain_length_m == pytest.approx(50 * STEP_3D, rel=1e-5)
    assert direct_routes[0].cost.terrain_length_m == session_routes[0].cost.terrain_length_m


def test_a_route_through_waypoints_measures_every_leg(ramp_files):
    waypoints = [_centre(RUN_ROW, 20), _centre(RUN_ROW, 35)]
    for graph_api in ("cython", "networkx"):
        _need(graph_api)
        _f, built, _ = _ramp_route(ramp_files, graph_api, waypoints=waypoints)
        try:
            assert built[0].cost.total_length == pytest.approx(50 * CELL)
            assert built[0].cost.terrain_length_m == pytest.approx(50 * STEP_3D, rel=1e-5)
        finally:
            if built[0].session is not None:
                built[0].session.close()


# ---------------------------------------------------- A: the library's own number, on a real hill
@pytest.fixture(scope="module")
def hill_files(tmp_path_factory):
    size = 200
    folder = tmp_path_factory.mktemp("terrain_hill")
    rows, cols = np.mgrid[0:size, 0:size]
    x = 500000.0 + cols + 0.5
    y = 5600000.0 - rows - 0.5
    dem = (60.0 * np.exp(-(((x - 500100.0) ** 2 + (y - 5599900.0) ** 2) / (2 * 12.0 ** 2)))).astype("float32")
    stripe = np.zeros((size, size), dtype="float32")
    stripe[:, 90:110] = 50.0
    transform = from_origin(500000.0, 5600000.0, 1, 1)
    return {"dem": _write_tif(folder / "dem.tif", dem, transform),
            "stripe": _write_tif(folder / "stripe.tif", stripe, transform)}


HILL_SETTINGS = {"weights": {"cost": 1.0, "length": 0.2, "gradient": 0.5}, "dem_layer_id": "dem",
                 "gradient": {"multiplier": "exponential"}}


@pytest.mark.parametrize("graph_api,force_direct", [("cython", False), ("cython", True), ("networkx", None)])
def test_the_terrain_length_is_the_length_the_library_reports_for_the_same_route(
        raster_path, hill_files, graph_api, force_direct):
    _need(graph_api)
    kwargs = obj.run_kwargs(HILL_SETTINGS, layer_path=hill_files, graph_api=graph_api)
    if force_direct is not None:
        kwargs["force_direct"] = force_direct
    finder, built, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm="dijkstra", graph_api=graph_api,
        neighborhood="r1", search_buffer_m=80, **kwargs)
    try:
        assert failed == []
        route = built[0]
        reference = finder.find_route(source=SOURCE, target=TARGET, algorithm="dijkstra", calculate_metrics=True)
        assert list(route.line.coords) == [tuple(c) for c in reference.path_coords]
        assert reference.total_length_3d > reference.total_length_2d                  # the hill is on the route
        assert route.cost.terrain_length_m == pytest.approx(reference.total_length_3d, rel=1e-9)
        assert route.cost.total_length == pytest.approx(reference.total_length_2d, rel=1e-9)
        assert routing.terrain_length_m(LineString(reference.path_coords), finder) == pytest.approx(
            reference.total_length_3d, rel=1e-9)
    finally:
        for item in built:
            if item.session is not None:
                item.session.close()


def test_the_terrain_length_of_a_waypoint_route_is_the_sum_of_the_legs_of_the_library(raster_path, hill_files):
    waypoint = (500060.0, 5599940.0)
    kwargs = {**obj.run_kwargs(HILL_SETTINGS, layer_path=hill_files), "force_direct": True}
    finder, built, _ = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], waypoints=[waypoint], algorithm="dijkstra",
        graph_api="cython", neighborhood="r1", search_buffer_m=80, **kwargs)
    legs = [finder.find_route(source=a, target=b, algorithm="dijkstra", calculate_metrics=True)
            for a, b in ((SOURCE, waypoint), (waypoint, TARGET))]
    assert built[0].cost.terrain_length_m == pytest.approx(sum(leg.total_length_3d for leg in legs), rel=1e-9)
    assert built[0].cost.total_length == pytest.approx(sum(leg.total_length_2d for leg in legs), rel=1e-9)


def test_the_elevation_is_the_grid_the_search_read(ramp_files):
    """No second resampling: the DEM the length is read from IS the finder's aligned array, also before the first
    search (the library aligns once and caches it)."""
    finder = PathFinder(ramp_files["cost"], source_coords=[RAMP_SOURCE], target_coords=[RAMP_TARGET],
                        graph_api="cython", neighborhood_str="r2", search_space_buffer_m=100.0,
                        **_ramp_kwargs(ramp_files, "cython", dem="dem_coarse")["objective_kwargs"])
    before = routing.window_dem(finder)
    assert before is not None and before.shape == finder.raster_handler.data[0].shape
    path = finder.find_route(algorithm="dijkstra", calculate_metrics=True)
    assert routing.window_dem(finder) is finder._gradient_dem
    assert routing.window_dem(finder) is before
    assert routing.terrain_length_m(path.path_geometry, finder) == pytest.approx(path.total_length_3d, rel=1e-9)


def test_terrain_length_is_none_without_a_dem_and_for_a_line_outside_the_window(ramp_files):
    plain = PathFinder(ramp_files["cost"], source_coords=[RAMP_SOURCE], target_coords=[RAMP_TARGET],
                       graph_api="cython", neighborhood_str="r2", search_space_buffer_m=100.0)
    assert routing.window_dem(plain) is None
    line = LineString([RAMP_SOURCE, RAMP_TARGET])
    assert routing.terrain_length_m(line, plain) is None
    weighted = PathFinder(ramp_files["cost"], source_coords=[RAMP_SOURCE], target_coords=[RAMP_TARGET],
                          graph_api="cython", neighborhood_str="r2", search_space_buffer_m=100.0,
                          objective={"cost": 1.0, "length": 0.5}, weight_precision="uint16")
    assert routing.terrain_length_m(line, weighted) is None
    finder = PathFinder(ramp_files["cost"], source_coords=[RAMP_SOURCE], target_coords=[RAMP_TARGET],
                        graph_api="cython", neighborhood_str="r2", search_space_buffer_m=100.0,
                        **_ramp_kwargs(ramp_files, "cython")["objective_kwargs"])
    inside = routing.terrain_length_m(LineString([RAMP_SOURCE, _centre(RUN_ROW, 6)]), finder)
    assert inside == pytest.approx(STEP_3D)
    assert routing.terrain_length_m(LineString([RAMP_SOURCE, _centre(RUN_ROW, 150)]), finder) is None


# ------------------------------------------------------------------ A: the GUI run and the edit
@pytest.mark.parametrize("graph_api", ALL_BACKENDS)
def test_the_run_callback_stores_both_lengths_and_says_so(app, state, ramp_layers, graph_api):
    _need(graph_api)
    route, resp = _gui_route(app, state, ramp_layers, graph_api, _dem_settings(ramp_layers))
    try:
        metrics = route.meta["metrics"]
        assert metrics["total_length_m"] == pytest.approx(50 * CELL)
        assert routing.TERRAIN_LENGTH_KEY == "terrain_length_m"
        assert metrics["terrain_length_m"] == pytest.approx(50 * STEP_3D, rel=1e-5)
        assert route.gdf["terrain_length_m"].iloc[0] == pytest.approx(metrics["terrain_length_m"])
        success = [n for n in _notices(resp) if n["severity"] == "success"][-1]
        assert f"Terrain length {50 * STEP_3D:,.0f} m (map length {50 * CELL:,.0f} m)." in success["meaning"]
        # the saved project of an older version did not have it; the stored params stay JSON
        json.dumps(route.meta["params"])
    finally:
        _close_sessions(state)


EDITS = [
    ("moved target", [RAMP_SOURCE, _centre(RUN_ROW, 40)], 35),
    ("added waypoint", [RAMP_SOURCE, _centre(RUN_ROW, 30), RAMP_TARGET], 50),
    ("moved target", [RAMP_SOURCE, _centre(RUN_ROW, 150)], 145),            # leaves the search window: a rebuild
]


@pytest.mark.parametrize("graph_api", ALL_BACKENDS)
@pytest.mark.parametrize("description,points,steps", EDITS, ids=["inside", "waypoint", "outside the window"])
def test_an_edited_route_recomputes_the_terrain_length(app, state, ramp_layers, graph_api, description, points, steps):
    _need(graph_api)
    route, _resp = _gui_route(app, state, ramp_layers, graph_api, _dem_settings(ramp_layers))
    try:
        new_layer, notices = edit_cb.recompute_variant(state, route, [list(p) for p in points], description, [])
        assert new_layer is not None, notices
        metrics = new_layer.meta["metrics"]
        assert metrics["total_length_m"] == pytest.approx(steps * CELL)
        assert metrics["terrain_length_m"] == pytest.approx(steps * STEP_3D, rel=1e-5)
        row = groups_cb._route_row(new_layer)
        assert row["terrain"] == round(steps * STEP_3D) and row["length"] == round(steps * CELL)
        assert f"Terrain length {steps * STEP_3D:,.0f} m (map length {steps * CELL:,.0f} m)." in notices[-1]["meaning"]
    finally:
        _close_sessions(state)


def test_an_edit_of_a_dem_route_that_has_to_rebuild_its_finder_still_gets_a_terrain_length(app, state, ramp_layers):
    route, _resp = _gui_route(app, state, ramp_layers, "networkx" if _installed("networkx") else "cython",
                              _dem_settings(ramp_layers))
    state.finders.clear()                                                    # nothing cached: a new finder is built
    _close_sessions(state)
    state.search_sessions.clear()
    new_layer, notices = edit_cb.recompute_variant(
        state, route, [list(RAMP_SOURCE), list(_centre(RUN_ROW, 40))], "moved target", [])
    assert new_layer is not None, notices
    assert new_layer.meta["metrics"]["terrain_length_m"] == pytest.approx(35 * STEP_3D, rel=1e-5)
    _close_sessions(state)


# ------------------------------------------------------------------ A: no DEM, no terrain length
NO_DEM_CASES = {
    "classic": lambda layers: {},
    "length weight": lambda layers: {"weights": {"cost": 1.0, "length": 0.5}},
    "metric layer": lambda layers: {"weights": {"cost": 1.0},
                                    "layers": [obj.layer_row("landscape", layers["flat"].id, 0.5)]},
}


@pytest.mark.parametrize("case", sorted(NO_DEM_CASES))
@pytest.mark.parametrize("graph_api", ["cython", "networkx"])
def test_a_route_without_a_dem_has_no_terrain_length(app, state, ramp_layers, case, graph_api):
    _need(graph_api)
    route, resp = _gui_route(app, state, ramp_layers, graph_api, NO_DEM_CASES[case](ramp_layers))
    try:
        assert "terrain_length_m" not in route.meta["metrics"]
        assert "terrain_length_m" not in route.gdf.columns
        assert groups_cb._route_row(route)["terrain"] is None
        assert all("Terrain length" not in n["meaning"] for n in _notices(resp) if n["severity"] == "success")
        finder = state.finders[ramp_layers["cost"].id]
        assert routing.terrain_length_m(route.gdf.geometry.iloc[0], finder) is None
        readout = json.dumps(edit_cb.cost_readout(route), default=lambda component: component.to_plotly_json())
        assert "Terrain length" not in readout and "Length (map)" not in readout and "Length" in readout
        new_layer, notices = edit_cb.recompute_variant(
            state, route, [list(RAMP_SOURCE), list(_centre(RUN_ROW, 40))], "moved target", [])
        assert new_layer is not None, notices
        assert "terrain_length_m" not in new_layer.meta["metrics"]
        assert "Terrain length" not in notices[-1]["meaning"]
    finally:
        _close_sessions(state)


def test_evaluate_route_cost_alone_never_sets_a_terrain_length(ramp_files):
    finder = PathFinder(ramp_files["cost"], source_coords=[RAMP_SOURCE], target_coords=[RAMP_TARGET],
                        graph_api="cython", neighborhood_str="r2", search_space_buffer_m=100.0)
    rc = cost.evaluate_route_cost(LineString([RAMP_SOURCE, RAMP_TARGET]), finder.raster_handler)
    assert rc.terrain_length_m is None
    assert cost.RouteCost(1.0, 2.0, 3.0, 1.0).terrain_length_m is None          # the old call signature


# ------------------------------------------------------------------ A: the routes grid, the readout, the attributes
def _routes_grid():
    from pyorps.gui.state import ProjectState

    for component in layout.build_layout(ProjectState())._traverse():
        if getattr(component, "id", None) == ids.ROUTES_GRID:
            return component
    raise AssertionError("the routes grid is not in the layout")


def test_the_routes_grid_has_a_terrain_length_column_next_to_the_map_length():
    grid = _routes_grid()
    fields = [column["field"] for column in grid.columnDefs]
    assert fields.index("terrain") == fields.index("length") + 1
    column = grid.columnDefs[fields.index("terrain")]
    assert column["headerName"] == "Terrain length (m)"
    assert column["editable"] is False and column["type"] == "numericColumn"
    assert column["maxWidth"] >= 60                                         # the fit never goes below 60 px
    assert "hide" not in column
    # the grid rules: the column is fitted by the shared class, not by a columnSize that fails in a closed card
    assert layout.FIT_COLUMNS in grid.className.split()
    assert getattr(grid, "columnSize", None) is None
    assert grid.getRowId == "params.data.id"


def test_a_route_row_has_the_terrain_length_only_when_the_route_has_one(state):
    with_dem = state.add_layer("R1", "route", meta={"metrics": {"total_length_m": 500.4, "terrain_length_m": 502.6},
                                                    "control_points": [[1.0, 2.0], [3.0, 4.0]]})
    without = state.add_layer("R2", "route", meta={"metrics": {"total_length_m": 500.4},
                                                   "control_points": [[1.0, 2.0], [3.0, 4.0]]})
    imported = state.add_layer("R3", "route", meta={"metrics": {}, "control_points": []})
    row = groups_cb._route_row(with_dem)
    assert row["length"] == 500 and row["terrain"] == 503
    assert groups_cb._route_row(without)["terrain"] is None and groups_cb._route_row(without)["length"] == 500
    assert groups_cb._route_row(imported)["terrain"] is None


def test_the_cost_readout_lists_the_terrain_length_under_the_map_length(state):
    layer = state.add_layer("R", "route", meta={"metrics": {
        "total_length_m": 500.0, "terrain_length_m": 502.4938, "total_cost": 1.0, "total_cell_cost": 2.0,
        "geodesic_length_m": 500.0}})
    text = json.dumps(edit_cb.cost_readout(layer), default=lambda component: component.to_plotly_json())
    assert "Length (map)" in text and "Terrain length" in text and "502 m" in text
    assert text.index("Length (map)") < text.index("Terrain length") < text.index("Cost (distance-weighted)")


def test_the_attribute_panel_names_the_terrain_length():
    assert attrs_cb.field_label("terrain_length_m") == "Terrain Length (in m)"
    assert attrs_cb.field_label("total_length_m") == "Total Length (in m)"


def test_the_terrain_length_sentence():
    assert routing.terrain_length_sentence([]) == ""
    assert routing.terrain_length_sentence([(500.0, None)]) == ""
    assert routing.terrain_length_sentence([(500.0, 502.4938)]) == "Terrain length 502 m (map length 500 m)."
    assert routing.terrain_length_sentence([(None, 502.4938)]) == "Terrain length 502 m."
    assert routing.terrain_length_sentence([(500.0, 502.0), (900.0, None), (700.0, 1200.0)]) == (
        "Terrain length of the 2 routes 502 to 1,200 m (the map lengths are 2-D).")


# ------------------------------------------------------------------ A: the export and the project file
def _route_with(state, terrain, name="R"):
    rc = cost.RouteCost(total_length=500.0, total_cost=7.0, total_cell_cost=8.0, geodesic_length_m=500.0,
                        terrain_length_m=terrain)
    return routes_cb.add_route_layer(
        state, line=LineString([RAMP_SOURCE, _centre(RUN_ROW, 25)]), route_cost=rc, params={},
        control_points=[RAMP_SOURCE, _centre(RUN_ROW, 25)], crs=TEST_CRS, name=name)


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".csv", ".shp"])
def test_the_route_export_carries_the_terrain_length_of_a_dem_route_only(state, tmp_path, suffix):
    import geopandas as gpd
    import pandas as pd

    dem_route = _route_with(state, 502.4938, "with dem")
    plain_route = _route_with(state, None, "without dem")
    column = "terrain_le" if suffix == ".shp" else "terrain_length_m"        # a shapefile cuts names to 10 characters
    out = {}
    for label, layer in (("dem", dem_route), ("plain", plain_route)):
        path = project_io.export_route(layer, tmp_path / f"{label}{suffix}")
        out[label] = (pd.read_csv(path, sep=";") if suffix == ".csv" else gpd.read_file(path))
    assert out["dem"][column].iloc[0] == pytest.approx(502.4938)
    assert out["dem"]["total_length_m" if suffix != ".shp" else "total_leng"].iloc[0] == pytest.approx(500.0)
    assert column not in out["plain"].columns                                # a plain route's export is as it was


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg"])
def test_the_export_of_all_routes_leaves_the_terrain_length_empty_for_a_route_without(state, tmp_path, suffix):
    import geopandas as gpd

    _route_with(state, 502.4938, "with dem")
    _route_with(state, None, "without dem")
    frame = gpd.read_file(project_io.export_all_routes(state, tmp_path / f"all{suffix}"))
    by_name = frame.set_index("name")["terrain_length_m"]
    assert by_name["with dem"] == pytest.approx(502.4938)
    assert by_name["without dem"] is None or math.isnan(by_name["without dem"])


def test_the_terrain_length_is_saved_with_the_project_and_an_old_project_opens_without_it(state, tmp_path):
    from pyorps.gui.state import ProjectState

    new_route = _route_with(state, 502.4938, "new")
    old_route = _route_with(state, None, "old")
    path = project_io.save_project(state, tmp_path / "proj" / "project.json")
    manifest = json.loads((tmp_path / "proj" / "project.json").read_text(encoding="utf-8"))
    saved = {entry["name"]: entry["meta"]["metrics"] for entry in manifest["layers"] if entry["kind"] == "route"}
    assert saved["new"]["terrain_length_m"] == pytest.approx(502.4938)
    assert "terrain_length_m" not in saved["old"]                           # how a project of an older version looks
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        by_name = {layer.name: layer for layer in other.layers_of_kind("route")}
        assert groups_cb._route_row(by_name["new"])["terrain"] == 502
        assert by_name["new"].meta["metrics"]["terrain_length_m"] == pytest.approx(502.4938)
        assert groups_cb._route_row(by_name["old"])["terrain"] is None
        assert "terrain_length_m" not in by_name["old"].meta["metrics"]
        readout = json.dumps(edit_cb.cost_readout(by_name["old"]), default=lambda c: c.to_plotly_json())
        assert "Terrain length" not in readout
    finally:
        other.shutdown()
    assert new_route.id and old_route.id


# ============================================================ B: sessions for routes with extra metric layers
HILL_BUFFER = 8
#: left of the search window (its array starts at x = 500012); the window is a band of 8 m around source-target
OUTSIDE = (500005.0, 5599900.0)

#: name -> (metric layers, with a DEM, weights, slope response); "stripe" and "dem" are layer ids of the settings
SESSION_CASES = {
    "file layer": ([("landscape", "stripe", 0.5)], False, {}, None),
    "derived slope with a DEM": ([("steep", obj.SOURCE_DERIVE_SLOPE, 0.1)], True, {}, None),
    "file layer with a DEM": ([("landscape", "stripe", 0.5)], True, {}, None),
    "file layer, derived slope, slope weight": (
        [("landscape", "stripe", 0.5), ("steep", obj.SOURCE_DERIVE_SLOPE, 0.1)], True, {"gradient": 0.5},
        {"multiplier": "exponential"}),
}


def _session_settings(case, ids_by_role):
    layers, with_dem, weights, gradient = SESSION_CASES[case]
    raw = {"weights": {"cost": 1.0, "length": 0.0, "gradient": 0.0, **weights},
           "layers": [obj.layer_row(name, ids_by_role.get(source, source), weight) for name, source, weight in layers]}
    if with_dem:
        raw["dem_layer_id"] = ids_by_role["dem"]
    if gradient:
        raw["gradient"] = gradient
    return raw


def _algorithms(case):
    return ["dijkstra"] if SESSION_CASES[case][1] else ["dijkstra", "delta-stepping"]


def _session_kwargs(case, hill_files):
    settings = _session_settings(case, {"stripe": "stripe", "dem": "dem"})
    return obj.run_kwargs(settings, layer_path=hill_files, graph_api="cython")


def _same_route(first, second):
    """The same cells; an equal-cost tie broken the other way is accepted when the plain cost and the map length
    agree to rounding."""
    (line_a, cost_a), (line_b, cost_b) = first, second
    if list(line_a.coords) != list(line_b.coords):
        assert cost_a.total_cost == pytest.approx(cost_b.total_cost, rel=1e-6)
        assert cost_a.total_length == pytest.approx(cost_b.total_length, rel=1e-6)
    else:
        assert cost_a.total_cost == pytest.approx(cost_b.total_cost, rel=1e-12)
    assert cost_a.terrain_length_m == (pytest.approx(cost_b.terrain_length_m, rel=1e-6)
                                       if cost_b.terrain_length_m is not None else None)


@pytest.mark.parametrize("graph_api,layers,dem,expected", [
    ("cython", True, False, False),                  # extra metric layers: a session equals find_route (below)
    ("cython", True, True, False),                   # ... also with the DEM and the slope derived from it
    ("raster_gpu", True, False, True),               # not run here (no device): the old safeguard stays
    ("raster_fim", True, False, True),
    ("raster_gpu", False, True, True),               # a retained GPU field has no slope terms
    ("networkx", True, False, True),                 # not a session backend either way
])
def test_force_direct_follows_the_proof(graph_api, layers, dem, expected):
    raw = {"weights": {"cost": 1.0}}
    if layers:
        raw["layers"] = [obj.layer_row("landscape", "stripe", 0.5)]
    if dem:
        raw["dem_layer_id"] = "dem"
    assert obj.force_direct(raw, graph_api) is expected


def test_force_direct_keeps_its_other_reasons_for_routes_with_extra_layers():
    raw = {"weights": {"cost": 1.0, "length": 0.2}, "layers": [obj.layer_row("landscape", "stripe", 0.5)],
           "weight_precision": "float32"}
    assert obj.force_direct(raw, "cython") is True                          # 32-bit weights
    assert obj.force_direct({**raw, "weight_precision": "uint16", "dem_layer_id": "dem"}, "raster_gpu") is True
    assert obj.force_direct({**raw, "weight_precision": "uint16"}, "cython") is False
    kwargs = obj.run_kwargs({**raw, "weight_precision": "uint16"}, layer_path={"stripe": "s.tif"})
    assert kwargs["force_direct"] is False and "metric_layers" in kwargs["objective_kwargs"]


@pytest.mark.parametrize("buffer", [HILL_BUFFER, 40.0])           # the search window is a band of this width
@pytest.mark.parametrize("case", list(SESSION_CASES))
def test_a_session_route_equals_the_direct_route_for_a_new_route(raster_path, hill_files, case, buffer):
    kwargs = _session_kwargs(case, hill_files)
    for algorithm in _algorithms(case):
        results = {}
        for forced in (False, True):
            _finder, built, failed = routing.run_routing(
                raster_path, sources=[SOURCE], targets=[TARGET], algorithm=algorithm, graph_api="cython",
                neighborhood="r1", search_buffer_m=buffer, num_threads=1, **{**kwargs, "force_direct": forced})
            assert failed == [] and len(built) == 1
            assert (built[0].session is not None) is (not forced)
            results[forced] = (built[0].line, built[0].cost)
            if built[0].session is not None:
                built[0].session.close()
        _same_route(results[False], results[True])
    assert kwargs["force_direct"] is False                                  # the service decides for sessions


@pytest.mark.parametrize("case", list(SESSION_CASES))
def test_a_session_route_equals_the_direct_route_after_a_window_rebuild(raster_path, hill_files, case):
    kwargs = _session_kwargs(case, hill_files)
    for algorithm in _algorithms(case):
        _finder, built, _ = routing.run_routing(
            raster_path, sources=[SOURCE], targets=[TARGET], algorithm=algorithm, graph_api="cython",
            neighborhood="r1", search_buffer_m=HILL_BUFFER, num_threads=1, **kwargs)
        session = built[0].session
        assert session is not None
        first_finder = session.finder
        points = [SOURCE, OUTSIDE, TARGET]
        via_session = routing.route_through_points(
            first_finder, points, algorithm=algorithm, session=session, cost_handler=routing.plain_cost_handler,
            delta=100, num_threads=1)
        assert session.finder is not first_finder                           # the window really was rebuilt
        reference = PathFinder(raster_path, source_coords=[SOURCE, OUTSIDE], target_coords=[TARGET],
                               graph_api="cython", neighborhood_str="r1", search_space_buffer_m=HILL_BUFFER,
                               **kwargs["objective_kwargs"])
        direct = routing.route_through_points(
            reference, points, algorithm=algorithm, cost_handler=routing.plain_cost_handler,
            **routing.direct_route_kwargs("cython", algorithm, num_threads=1))
        _same_route(via_session, direct)
        # and a second update on the rebuilt window (a retained tree, a moved waypoint) is still the direct route
        moved = [SOURCE, (500006.0, 5599880.0), TARGET]
        again = routing.route_through_points(
            session.finder, moved, algorithm=algorithm, session=session, cost_handler=routing.plain_cost_handler,
            delta=100, num_threads=1)
        direct_again = routing.route_through_points(
            reference, moved, algorithm=algorithm, cost_handler=routing.plain_cost_handler,
            **routing.direct_route_kwargs("cython", algorithm, num_threads=1))
        _same_route(again, direct_again)
        session.close()


@pytest.fixture()
def hill_layers(state, raster_path, hill_files):
    def add(name, path):
        return state.add_layer(name, "raster", crs=TEST_CRS, meta={"source_path": path})

    return {"cost": add("cost", raster_path), "dem": add("DGM", hill_files["dem"]),
            "stripe": add("landscape", hill_files["stripe"])}


def _hill_run(app, raster_id):
    args = (1, False, {"sources": [{"lat": 0, "lng": 0, "x": SOURCE[0], "y": SOURCE[1]}],
                       "targets": [{"lat": 0, "lng": 0, "x": TARGET[0], "y": TARGET[1]}], "waypoints": []},
            raster_id, "dijkstra", "cython", 1, HILL_BUFFER, False, True, False, False, 1.0, 100, 0, None, [], [])
    return invoke(app, RUN_WIRING, *args, triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _direct_reference(raster_path, kwargs, original_window_points, points):
    """The route a direct finder gives for ``points``: on the window of the original run while every point lies
    in it, else on a rebuilt window over the new points (what the GUI's direct edit did)."""
    first = PathFinder(raster_path, source_coords=[original_window_points[0]],
                       target_coords=[original_window_points[1]], graph_api="cython", neighborhood_str="r1",
                       search_space_buffer_m=HILL_BUFFER, **kwargs["objective_kwargs"])
    finder = first if routing.points_in_finder_window(first, points) else PathFinder(
        raster_path, source_coords=list(points), target_coords=[points[-1]], graph_api="cython",
        neighborhood_str="r1", search_space_buffer_m=HILL_BUFFER, **kwargs["objective_kwargs"])
    return routing.route_through_points(finder, points, algorithm="dijkstra",
                                        cost_handler=routing.plain_cost_handler)


EDIT_POINTS = {
    "moved target": [SOURCE, (500175.0, 5599830.0)],
    "added waypoint": [SOURCE, (500060.0, 5599940.0), TARGET],       # on the line source-target, inside the band
    "moved target out of the window": [SOURCE, OUTSIDE],
}


@pytest.mark.parametrize("case", list(SESSION_CASES))
def test_a_gui_route_with_extra_layers_keeps_its_session_and_every_edit_equals_the_direct_route(
        app, state, raster_path, hill_files, hill_layers, case):
    settings = _session_settings(case, {"stripe": hill_layers["stripe"].id, "dem": hill_layers["dem"].id})
    state.gui_settings["objective"] = settings
    resp = _hill_run(app, hill_layers["cost"].id)
    routes = state.layers_of_kind("route")
    assert routes, _notices(resp)
    route = routes[-1]
    paths = {layer.id: layer.meta["source_path"] for layer in hill_layers.values()}
    kwargs = obj.run_kwargs(settings, layer_path=paths, graph_api="cython")
    try:
        assert route.id in state.search_sessions                            # a retained session again
        assert edit_cb._route_plan(route.meta["params"])[3] is False        # the edit does not force a direct route
        for description, points in EDIT_POINTS.items():
            new_layer, notices = edit_cb.recompute_variant(state, route, [list(p) for p in points], description, [])
            assert new_layer is not None, notices
            assert new_layer.id in state.search_sessions                    # edits stay fast
            line = new_layer.gdf.geometry.iloc[0]
            reference_line, reference_cost = _direct_reference(raster_path, kwargs, (SOURCE, TARGET), points)
            metrics = new_layer.meta["metrics"]
            _same_route((line, cost.RouteCost(
                metrics["total_length_m"], metrics["total_cost"], metrics["total_cell_cost"],
                metrics["geodesic_length_m"], terrain_length_m=metrics.get("terrain_length_m"))),
                (reference_line, reference_cost))
            route = new_layer                                             # the next edit starts from this one
    finally:
        _close_sessions(state)
