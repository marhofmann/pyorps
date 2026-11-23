"""Round 17, three confirmed defects of constrained routing in the GUI.

1. A classic constrained run without a feasible route showed "Something went wrong" instead of "No route could be
   found": the service built a tower frame from the library's empty result, which geopandas refuses. These tests run
   the REAL library (a raster with a wall of forbidden cells, the default profile rule, the cython backend) and never
   patch ``run_constrained_path``: the synchronous run, the background job, a single segment and a chain of waypoints
   (the notice names the segment that has no route).
2. In a chain of waypoints every segment reports a terminal tower at both of its ends, so the structure at an interior
   waypoint was counted twice in the total, and the duplicate that was dropped could be the one that carries the
   height premium. One physical tower stands at a waypoint: its cost is counted once, the dearer (at equal cost the
   taller) report is kept, and a note says so when the two disagree. The merged totals equal the sum over the merged
   tower table plus terrain plus turn penalties.
3. Two texts disagreed about what the clearance check needs ("also needs a DSM" against the end heights judged
   against the DEM). The library says: the check is on with a DEM and a profile with the four clearance settings; a
   DSM is optional, is read only together with a DEM and adds trees and buildings; a DSM without a DEM is ignored.
   The tests pin that against the library itself and check that every GUI text agrees.

Headless, tiny synthetic rasters. The constrained kernels are not deterministic with several threads: one thread.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import dash_bootstrap_components as dbc
import numpy as np
import pytest
import yaml
from conftest import invoke, poll_job
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point

from pyorps.gui import ids
from pyorps.gui.callbacks import constrained as run_callbacks
from pyorps.gui.layout import build_layout
from pyorps.gui.services import constrained
from pyorps.gui.services import constrained_ends as ce
from pyorps.gui.services import profile_form as profile_service

CRS = "EPSG:32632"
JOIN_S = 60
FORBIDDEN = 65535
RUN = (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)
TERMINAL_COST = 500.0

BASE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 6, "max_span_m": 20, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 200}, "terrain_interpolation": "linear",
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 100},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300}}}}
CLEARANCE = {"tower_height_m": 30.0, "conductor_weight_per_m": 15.0, "conductor_tension_n": 25000.0,
             "min_clearance_m": 6.0}
WITH_TERMINAL_COST = {**BASE, "terminal_tower_cost": TERMINAL_COST}
POINTS = [(8.0, 8.0), (52.0, 52.0)]

#: a one-row corridor (cost 100 in the row, forbidden elsewhere) for the several-heights runs: the surface model has
#: a 20 m obstacle on columns 42 to 60, so the structure at column 40 needs the tall height class (and pays its premium)
ROWS, COLS, ROW = 5, 80, 2
CORRIDOR_PROFILE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 5, "max_span_m": 12, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 100},
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 50.0},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300.0}}},
    "tower_heights_m": [30.0, 10.0], "tower_height_cost_per_increment": 30.0,
    "conductor_weight_per_m": 0.001, "conductor_tension_n": 1.0e6, "min_clearance_m": 5.0,
    "terminal_tower_cost": TERMINAL_COST}
CORRIDOR_POINTS = [(4.5, 2.5), (40.5, 2.5), (76.5, 2.5)]


# ------------------------------------------------------------------ helpers
@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # the constrained kernels read OMP_NUM_THREADS per call: one thread keeps ties deterministic (and the test
    # independent of the number of cores)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def _write_raster(path, array, dtype="uint16"):
    import rasterio

    rows, cols = array.shape
    with rasterio.open(str(path), "w", driver="GTiff", height=rows, width=cols, count=1, dtype=dtype, crs=CRS,
                       transform=from_bounds(0, 0, cols, rows, cols, rows)) as dst:
        dst.write(array.astype(dtype), 1)
    return str(path)


@pytest.fixture(scope="module")
def uniform_raster(tmp_path_factory):
    """A 60 x 60 m uniform cost raster at 1 m."""
    return _write_raster(tmp_path_factory.mktemp("r17fix") / "uniform.tif", np.full((60, 60), 100, np.uint16))


@pytest.fixture(scope="module")
def flat_dem(tmp_path_factory):
    return _write_raster(tmp_path_factory.mktemp("r17fixdem") / "dem.tif", np.zeros((60, 60), np.float32), "float32")


@pytest.fixture(scope="module")
def wall_raster(tmp_path_factory):
    """The uniform raster with a forbidden wall across column 30: no route crosses it."""
    array = np.full((60, 60), 100, np.uint16)
    array[:, 30] = FORBIDDEN
    return _write_raster(tmp_path_factory.mktemp("r17fixwall") / "wall.tif", array)


@pytest.fixture(scope="module")
def corridor(tmp_path_factory):
    """Cost corridor, flat DEM and a DSM with a 20 m obstacle on columns 42 to 60, as paths."""
    folder = tmp_path_factory.mktemp("r17fixcorridor")
    cost = np.full((ROWS, COLS), FORBIDDEN, np.uint16)
    cost[ROW, :] = 100
    surface = np.zeros((ROWS, COLS), np.float32)
    surface[ROW, 42:61] = 20.0
    return SimpleNamespace(
        raster=_write_raster(folder / "cost.tif", cost),
        dem=_write_raster(folder / "dem.tif", np.zeros((ROWS, COLS), np.float32), "float32"),
        dsm=_write_raster(folder / "dsm.tif", surface, "float32"))


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _path(raster, *, points, profile=BASE, outcome=None, **kwargs):
    return _quiet(constrained.run_constrained_path, raster, points=points, profile=profile, backend="cython",
                  neighborhood="r1", search_buffer_m=100, outcome=outcome, **kwargs)


def _segment(raster, a, b, *, profile, **kwargs):
    """The library's own result of one segment (what the chain adds up)."""
    result, _towers, _crs = _quiet(
        constrained.run_constrained, raster, source=a, target=b, profile=profile, neighborhood="r1",
        search_buffer_m=100, **kwargs)
    return result


def _layer(state, path):
    return state.add_layer("cost", "raster", crs=CRS, meta={"source_path": path})


def _point(x, y):
    return {"lat": 0, "lng": 0, "x": x, "y": y}


def _draft(points=POINTS, waypoints=()):
    return {"sources": [_point(*points[0])], "targets": [_point(*points[1])],
            "waypoints": [_point(*w) for w in waypoints]}


def _run(app, raster_id, *, profile=None, points=POINTS, waypoints=(), dem=None, dsm=None, notices=None):
    text = yaml.safe_dump(BASE if profile is None else profile)
    return invoke(app, RUN, 1, True, _draft(points, waypoints), raster_id, text, "cython", False, 1, 100, dem, dsm,
                  [], list(notices or []), triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _titles(resp) -> list[str]:
    return [n["title"] for n in _notices(resp)]


def _routes(state):
    return state.layers_of_kind("route")


@pytest.fixture()
def heavy(monkeypatch):
    """Every constrained run counts as heavy (the criterion, patched by name): it runs as a background job."""
    monkeypatch.setattr(constrained, "is_heavy", lambda _est: True)


def _components(page):
    return list(page._traverse())  # noqa: SLF001


def _by_id(page, component_id):
    found = [c for c in _components(page) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id}: {len(found)} components"
    return found[0]


# ====================================================================== 1. a run without a route is "no route"
def test_the_service_recognises_the_empty_library_result_before_it_builds_a_frame(wall_raster):
    result, towers, crs = _quiet(constrained.run_constrained, wall_raster, source=(8.0, 8.0), target=(52.0, 52.0),
                                 profile=BASE, neighborhood="r1", search_buffer_m=100)
    assert result.path_geometry is None and result.towers == [] and towers is None and crs is not None


def test_a_single_segment_without_a_route_answers_none_and_names_no_segment_to_say(wall_raster):
    outcome: dict = {}
    assert _path(wall_raster, points=POINTS, outcome=outcome) is None
    assert outcome == {"no_route": {"segment": 1, "of": 1}}
    assert constrained.no_route_segment_text(outcome["no_route"]) == ""
    assert _path(wall_raster, points=POINTS) is None            # without an outcome dict it works as before


@pytest.mark.parametrize("waypoints, segment, ends", [
    ([(20.0, 20.0), (28.0, 28.0)], 3, "from waypoint 2 to the target"),
    ([(20.0, 20.0), (52.0, 20.0)], 2, "from waypoint 1 to waypoint 2"),
    ([(45.0, 20.0), (50.0, 25.0)], 1, "from the source to waypoint 1")])
def test_a_chain_names_the_segment_that_has_no_route(wall_raster, waypoints, segment, ends):
    outcome: dict = {}
    assert _path(wall_raster, points=[POINTS[0], *waypoints, POINTS[1]], outcome=outcome) is None
    assert outcome["no_route"] == {"segment": segment, "of": 3}
    text = constrained.no_route_segment_text(outcome["no_route"])
    assert text == f"Segment {segment} of 3 ({ends}) found no feasible tower-by-tower path."


def test_a_synchronous_run_without_a_route_is_the_no_route_notice_not_the_generic_box(app, state, wall_raster):
    layer = _layer(state, wall_raster)
    resp = _run(app, layer.id)
    assert _titles(resp)[-1] == "No route could be found"
    assert "Something went wrong" not in _titles(resp)
    notice = _notices(resp)[-1]
    assert notice["meaning"] == run_callbacks.no_route_notice()["meaning"]       # one segment: the general sentence
    assert _routes(state) == [] and state.jobs == {}


def test_a_synchronous_chain_without_a_route_says_which_segment(app, state, wall_raster):
    layer = _layer(state, wall_raster)
    resp = _run(app, layer.id, waypoints=[(20.0, 20.0), (52.0, 20.0)])
    notice = _notices(resp)[-1]
    assert notice["title"] == "No route could be found" and "Something went wrong" not in _titles(resp)
    assert notice["meaning"].startswith("Segment 2 of 3 (from waypoint 1 to waypoint 2)")
    assert _routes(state) == []


def test_a_background_run_without_a_route_is_the_no_route_notice_too(app, state, wall_raster, heavy):
    layer = _layer(state, wall_raster)
    start = _run(app, layer.id)
    state.jobs[constrained.JOB_KIND].thread.join(JOIN_S)
    done = poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    notice = done[ids.NOTICES]["data"][-1]
    assert notice["title"] == "No route could be found" and "Something went wrong" not in _titles(done)
    assert notice["meaning"] == run_callbacks.no_route_notice()["meaning"]
    assert "No route was found" in done[ids.OHL_STATUS]["children"] and _routes(state) == []


def test_a_background_chain_without_a_route_says_which_segment(app, state, wall_raster, heavy):
    layer = _layer(state, wall_raster)
    start = _run(app, layer.id, waypoints=[(20.0, 20.0), (28.0, 28.0)])
    state.jobs[constrained.JOB_KIND].thread.join(JOIN_S)
    done = poll_job(app, ids.JOB_ACK_CONSTRAINED, start[ids.NOTICES]["data"])
    notice = done[ids.NOTICES]["data"][-1]
    assert notice["title"] == "No route could be found"
    assert notice["meaning"].startswith("Segment 3 of 3 (from waypoint 2 to the target)")
    assert _routes(state) == []


def test_the_background_worker_names_the_segment_only_when_a_run_has_none(wall_raster, uniform_raster):
    def job(raster, points):
        run = constrained.snapshot_run_inputs(
            raster, points=points, profile=BASE, backend="cython", neighborhood="r1", search_buffer_m=100, dem=None,
            dsm=None)
        return SimpleNamespace(params={"run": run}, cancel=SimpleNamespace(is_set=lambda: False),
                               report=lambda *_a, **_k: None)

    none = _quiet(constrained.background_run, job(wall_raster, [POINTS[0], (20.0, 20.0), POINTS[1]]))
    assert none["result"] is None and none["no_route"] == {"segment": 2, "of": 2}
    found = _quiet(constrained.background_run, job(uniform_raster, POINTS))
    assert found["result"] is not None and "no_route" not in found and set(found) == {"result", "notices"}


# ====================================================================== 2. one tower at an interior waypoint
def _check_identity(out, segments):
    """The merged totals equal the sum over the merged tower table plus terrain plus turn penalties."""
    table = out["towers"]
    breakdown = out["cost_breakdown"]
    summary = out["summary"]
    assert set(breakdown) == {"terrain", "towers", "angle_penalties"}
    assert breakdown["towers"] == pytest.approx(float(table["total_cost"].sum()))
    assert summary["total_tower_cost"] == pytest.approx(breakdown["towers"])
    assert summary["n_towers"] == len(table) == sum(len(s.towers) for s in segments) - (len(segments) - 1)
    assert breakdown["terrain"] == pytest.approx(sum(s.total_terrain_cost for s in segments))
    assert summary["total_angle_penalty_cost"] == pytest.approx(breakdown["angle_penalties"])
    assert out["total_cost"] == pytest.approx(
        float(table["total_cost"].sum()) + breakdown["terrain"] + breakdown["angle_penalties"])


@pytest.mark.parametrize("points", [
    [(8.0, 8.0), (30.0, 30.0), (52.0, 52.0)],
    [(8.0, 8.0), (20.0, 25.0), (38.0, 35.0), (52.0, 52.0)]])
def test_a_chain_counts_every_interior_waypoint_tower_once_with_one_height(uniform_raster, points):
    out = _path(uniform_raster, points=points, profile=WITH_TERMINAL_COST)
    segments = [_segment(uniform_raster, a, b, profile=WITH_TERMINAL_COST) for a, b in zip(points[:-1], points[1:])]
    _check_identity(out, segments)
    # each segment's total holds BOTH of its terminal towers (foundation plus terminal_tower_cost): the tower at an
    # interior waypoint is in two totals, and counted once in the merged one
    twice = [segment.towers[-1].total_cost for segment in segments[:-1]]
    assert len(twice) == len(points) - 2 and all(cost >= TERMINAL_COST for cost in twice)
    assert out["total_cost"] == pytest.approx(sum(s.total_cost for s in segments) - sum(twice))
    assert out["summary"]["total_tower_cost"] == pytest.approx(sum(s.total_tower_cost for s in segments) - sum(twice))
    # every interior waypoint has exactly one tower in the table, and it is a terminal one at that place
    table = out["towers"]
    for x, y in points[1:-1]:
        here = table[table.geometry.centroid.distance(Point(x, y)) < 1.0]
        assert len(here) == 1 and here["tower_type"].iloc[0] == "terminal"
    # one height and no premium: the two reports agree, so there is nothing to say
    assert not any("Waypoint" in text for text in out["summary"].get("search_notes", []))


def test_a_chain_of_one_segment_is_what_the_library_says(uniform_raster):
    out = _path(uniform_raster, points=POINTS, profile=WITH_TERMINAL_COST)
    only = _segment(uniform_raster, *POINTS, profile=WITH_TERMINAL_COST)
    assert out["total_cost"] == pytest.approx(only.total_cost)
    assert out["summary"]["total_tower_cost"] == pytest.approx(only.total_tower_cost)
    assert len(out["towers"]) == only.n_towers
    _check_identity(out, [only])


def test_every_old_summary_key_is_still_there_in_a_chain(uniform_raster):
    out = _path(uniform_raster, points=[POINTS[0], (30.0, 30.0), POINTS[1]], profile=WITH_TERMINAL_COST)
    assert {"line", "towers", "crs", "summary", "total_cost", "total_length", "total_cell_cost"} <= set(out)
    assert {"n_towers", "total_tower_cost", "total_angle_penalty_cost", "max_turn_angle_deg",
            "spans_min_max_avg"} <= set(out["summary"])


def test_several_heights_keep_the_dearer_report_of_the_waypoint_tower_and_say_so(corridor):
    """Segment 2 starts at column 40 under a 20 m obstacle: its source structure takes the tall height class and pays
    the premium, so the two reports of the waypoint tower differ (10 m, 600 against 30 m, 800)."""
    points = CORRIDOR_POINTS
    kwargs = {"dem": corridor.dem, "dsm": corridor.dsm}
    out = _path(corridor.raster, points=points, profile=CORRIDOR_PROFILE, **kwargs)
    segments = [_segment(corridor.raster, a, b, profile=CORRIDOR_PROFILE, **kwargs)
                for a, b in zip(points[:-1], points[1:])]
    first_target, second_source = segments[0].towers[-1], segments[1].towers[0]
    assert (first_target.height_m, first_target.height_premium, first_target.total_cost) == (10.0, 0.0, 600.0)
    assert second_source.height_premium == 200.0 and second_source.height_m == 30.0
    assert second_source.total_cost == 800.0

    _check_identity(out, segments)
    table = out["towers"]
    here = table[table.geometry.centroid.distance(Point(*points[1])) < 0.1]
    assert len(here) == 1 and float(here["height_m"].iloc[0]) == 30.0 and float(here["total_cost"].iloc[0]) == 800.0
    # the dropped report is segment 1's target tower (cost 600): counted once, with the premium
    assert out["total_cost"] == pytest.approx(sum(s.total_cost for s in segments) - first_target.total_cost)
    assert out["summary"]["total_tower_cost"] == pytest.approx(sum(s.total_tower_cost for s in segments) - 600.0)
    assert out["summary"]["height_premiums"] == pytest.approx(float(table["height_premium"].sum()))
    assert out["summary"]["height_premiums"] == pytest.approx(sum(s.cost_breakdown["height_premiums"]
                                                                  for s in segments))
    notes = [t for t in out["summary"]["search_notes"] if t.startswith("Waypoint 1:")]
    assert len(notes) == 1 and "different heights and costs" in notes[0] and "10 m, cost 600" in notes[0]
    assert "30 m, cost 800" in notes[0] and "dearer" in notes[0] and "counted once" in notes[0]


def test_several_heights_without_a_height_class_the_obstacle_needs_agree_and_say_nothing(corridor):
    """Without the DSM the obstacle is invisible: no premium, one height at the waypoint, no note, same identity."""
    points = CORRIDOR_POINTS
    out = _path(corridor.raster, points=points, profile=CORRIDOR_PROFILE, dem=corridor.dem)
    segments = [_segment(corridor.raster, a, b, profile=CORRIDOR_PROFILE, dem=corridor.dem)
                for a, b in zip(points[:-1], points[1:])]
    _check_identity(out, segments)
    assert out["total_cost"] == pytest.approx(sum(s.total_cost for s in segments) - TERMINAL_COST - 100.0)
    assert not any(t.startswith("Waypoint") for t in out["summary"].get("search_notes", []))


# ------------------------------------------------------------------ the rule on hand-made reports
def _tower(x, y, *, cost, height, premium=0.0):
    return SimpleNamespace(location=Point(x, y), total_cost=cost, height_m=height, height_premium=premium)


def _frame(n):
    import geopandas as gpd

    return gpd.GeoDataFrame({"tower_id": range(n)}, geometry=[Point(i, 0) for i in range(n)], crs=CRS)


def _merge(first, second, place=(10.0, 0.0)):
    frames = [_frame(len(first)), _frame(len(second))]
    lines = [LineString([(0.0, 0.0), place]), LineString([place, (20.0, 0.0)])]
    return constrained._merge_waypoint_towers(frames, [first, second], lines)  # noqa: SLF001


def test_the_dearer_report_is_kept_whatever_its_height():
    end = _tower(10, 0, cost=900.0, height=10.0)
    start = _tower(10, 0, cost=800.0, height=30.0, premium=200.0)
    outer = (_tower(0, 0, cost=1.0, height=10.0), _tower(20, 0, cost=1.0, height=10.0))
    drop, dropped, notes = _merge([outer[0], end], [start, outer[1]])
    assert drop == [set(), {0}] and dropped == [start]
    assert len(notes) == 1 and "different heights and costs" in notes[0]
    drop, dropped, notes = _merge([outer[0], _tower(10, 0, cost=800.0, height=10.0)],
                                  [_tower(10, 0, cost=900.0, height=10.0), outer[1]])
    assert drop == [{1}, set()] and len(dropped) == 1 and "different costs for" in notes[0]


def test_at_equal_cost_the_taller_report_is_kept():
    end = _tower(10, 0, cost=600.0, height=10.0)
    start = _tower(10, 0, cost=600.0, height=14.0)
    drop, dropped, notes = _merge([end], [start])
    assert drop == [{0}, set()] and dropped == [end] and "different heights for" in notes[0]
    end, start = _tower(10, 0, cost=600.0, height=14.0), _tower(10, 0, cost=600.0, height=10.0)
    drop, dropped, notes = _merge([end], [start])
    assert drop == [set(), {0}] and dropped == [start] and "different heights for" in notes[0]


def test_equal_reports_drop_the_later_one_and_say_nothing():
    end, start = _tower(10, 0, cost=600.0, height=10.0), _tower(10, 0, cost=600.0, height=10.0)
    drop, dropped, notes = _merge([end], [start])
    assert drop == [set(), {0}] and dropped == [start] and notes == []


def test_a_waypoint_without_a_tower_of_both_segments_at_its_place_is_left_alone():
    near = _tower(10, 0, cost=600.0, height=10.0)
    elsewhere = _tower(13, 0, cost=600.0, height=10.0)
    assert _merge([near], [elsewhere]) == ([set(), set()], [], [])
    assert _merge([], [near]) == ([set(), set()], [], [])
    # a tower table that does not match the towers of its segment is not touched
    lines = [LineString([(0.0, 0.0), (10.0, 0.0)]), LineString([(10.0, 0.0), (20.0, 0.0)])]
    assert constrained._merge_waypoint_towers(  # noqa: SLF001
        [_frame(3), _frame(1)], [[near], [near]], lines) == ([set(), set()], [], [])


# ====================================================================== 3. what the clearance check needs
CLEARANCE_TEXT = ("terrain model (DEM)", "optional")


@pytest.mark.parametrize("profile", [
    {**BASE, **CLEARANCE},
    {"name": "x"},
    {"tower_height_m": 20, "conductor_weight_per_m": 5}])
def test_the_clearance_line_says_the_run_needs_a_dem_and_a_dsm_is_optional(profile):
    text = profile_service.clearance_status(profile)
    assert all(part in text for part in CLEARANCE_TEXT) and "surface model (DSM) is optional" in text
    assert "needs a DSM" not in text and "and a DSM)" not in text and "plus a DSM" not in text
    assert text.startswith("Clearance check: on" if profile_service.clearance_state(profile) == "on"
                           else "Clearance check: off")
    assert profile_service.CLEARANCE_RUN_NEEDS in text


def test_the_inputs_of_the_card_say_what_the_dem_and_the_dsm_do(state):
    page = build_layout(state)
    dem, dsm = _by_id(page, ids.OHL_DEM), _by_id(page, ids.OHL_DSM)
    assert isinstance(dem, dbc.Input) and isinstance(dsm, dbc.Input)

    def help_of(control):               # the hover text sits on the wrapper of the label and the input
        return next(c.title for c in _components(page) if isinstance(getattr(c, "children", None), list)
                    and control in c.children and getattr(c, "title", None))

    dem_help, dsm_help = help_of(dem), help_of(dsm)
    assert "slope and clearance" in dem.placeholder and "clearance check" in dem_help
    assert "off without a DEM" in dem_help and "slopes" in dem_help
    assert "needs the DEM" in dsm.placeholder and "trees and buildings" in dsm.placeholder
    assert "only together with the DEM" in dsm_help and "ignored" in dsm_help and "optional" in dsm_help
    labels = {getattr(c, "html_for", None): c.children for c in _components(page) if isinstance(c, dbc.Label)}
    assert "slope and clearance" in labels[ids.OHL_DEM] and "needs the DEM" in labels[ids.OHL_DSM]
    assert "clearance, optional" not in labels[ids.OHL_DSM] and "slope, optional" not in labels[ids.OHL_DEM]


def test_the_service_roles_agree_with_the_texts_of_the_form():
    assert "terrain model (DEM)" in ce.DEM_ROLE and "off without a DEM" in ce.DEM_ROLE
    assert "optional" in ce.DSM_ROLE and "only together with a DEM" in ce.DSM_ROLE and "ignored" in ce.DSM_ROLE
    assert "terrain model (DEM)" in profile_service.CLEARANCE_RUN_NEEDS


# ------------------------------------------------------------------ the library agrees (the facts behind the texts)
def _towers_of(corridor, **kwargs):
    result = _segment(corridor.raster, CORRIDOR_POINTS[1], CORRIDOR_POINTS[2], profile=CORRIDOR_PROFILE, **kwargs)
    return result, [t.height_m for t in result.towers], [t.height_premium for t in result.towers]


def test_the_library_uses_a_dsm_only_together_with_a_dem(corridor):
    """With DEM and DSM the 20 m obstacle forces the tall class (premiums); a DEM alone sees bare ground only (no
    premium, cheaper); a DSM alone is ignored (the same route as with neither); the clearance data without a DEM do
    not switch the check on."""
    both, heights_both, premiums_both = _towers_of(corridor, dem=corridor.dem, dsm=corridor.dsm)
    dem_only, heights_dem, premiums_dem = _towers_of(corridor, dem=corridor.dem)
    dsm_only, heights_dsm, premiums_dsm = _towers_of(corridor, dsm=corridor.dsm)
    neither, heights_none, premiums_none = _towers_of(corridor)
    assert max(heights_both) == 30.0 and max(premiums_both) == 200.0
    assert set(heights_dem) == {10.0} and set(premiums_dem) == {0.0}
    assert dem_only.total_cost < both.total_cost
    assert heights_dsm == heights_none and premiums_dsm == premiums_none == [0.0] * len(premiums_none)
    assert dsm_only.total_cost == pytest.approx(neither.total_cost)
    assert [t.location.x for t in dsm_only.towers] == [t.location.x for t in neither.towers]


def test_a_connection_height_is_judged_against_the_dem_not_the_dsm():
    heights = {"source_connection_height_m": 20.0}
    with_clearance = {**BASE, **CLEARANCE}
    assert ce.effect(heights, profile=with_clearance, dem="dem.tif").source_acts is True
    assert ce.effect(heights, profile=with_clearance, dem=None).source_acts is False        # a DSM does not help


# ------------------------------------------------------------------ the GUI says what the library ignores
def test_a_dsm_without_a_dem_is_a_note_not_a_silent_ignore():
    with_clearance = {**BASE, **CLEARANCE}
    note, = ce.surface_model_notes(with_clearance, dem=None, dsm="dsm.tif")
    assert "ignored" in note and "only together with a terrain model (DEM)" in note and "DSM minus DEM" in note
    assert ce.surface_model_notes(with_clearance, dem="", dsm="dsm.tif") == [note]
    assert ce.surface_model_notes(with_clearance, dem="   ", dsm="dsm.tif") == [note]


def test_a_dsm_with_a_dem_but_without_clearance_data_is_not_used_and_says_so():
    note, = ce.surface_model_notes(BASE, dem="dem.tif", dsm="dsm.tif")
    assert "not used" in note and "only the clearance check reads it" in note and "tower height" in note


@pytest.mark.parametrize("profile, dem, dsm", [
    ({**BASE, **CLEARANCE}, "dem.tif", "dsm.tif"),      # the DSM is used
    ({**BASE, **CLEARANCE}, "dem.tif", None),           # a DEM alone is a normal run
    ({**BASE, **CLEARANCE}, None, None),
    (BASE, None, None),
    (BASE, None, ""),
    (None, None, None)])
def test_no_note_where_the_library_does_what_the_user_asked(profile, dem, dsm):
    assert ce.surface_model_notes(profile, dem=dem, dsm=dsm) == []


def test_a_run_with_a_dsm_and_no_dem_shows_the_note_and_still_routes(app, state, uniform_raster, flat_dem):
    layer = _layer(state, uniform_raster)
    resp = _run(app, layer.id, profile={**BASE, **CLEARANCE}, dsm=flat_dem)
    notice = next(n for n in _notices(resp) if n["title"] == "Surface model (DSM) has no effect")
    assert notice["severity"] == "warning" and "ignored" in notice["meaning"]
    assert notice["focus_id"] == ids.TAB_ROUTES and notice["focus_control"] == ids.OHL_DSM
    assert len(_routes(state)) == 1


def test_a_run_whose_dsm_is_used_or_that_has_none_shows_no_such_note(app, state, uniform_raster, flat_dem):
    layer = _layer(state, uniform_raster)
    used = _run(app, layer.id, profile={**BASE, **CLEARANCE}, dem=flat_dem, dsm=flat_dem)
    none = _run(app, layer.id, profile={**BASE, **CLEARANCE}, dem=flat_dem)
    assert "Surface model (DSM) has no effect" not in _titles(used) + _titles(none)
