"""Phase 7: constrained overhead-line routing (R13, F9/F10, Section 20)."""
import numpy as np
import pytest

from pyorps.gui import ids
from pyorps.gui.services import constrained

from conftest import invoke

MINI_PROFILE = {
    "soft_angle_limit_deg": 20.0,
    "hard_angle_limit_deg": 60.0,
    "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0},
    "min_span_m": 20,
    "max_span_m": 40,
    # bin <= cell size (1 m) — a coarser bin degenerates the search into
    # a minutes-long run (see validate_span_bin_vs_resolution)
    "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {
        "terrain_cost_map": {"0": 100, "500": 200},
        "terrain_interpolation": "linear",
        "angle_types": {
            "suspension": {"max_angle_deg": 20.0, "base_cost": 100},
            "strain": {"max_angle_deg": 60.0, "base_cost": 300},
        },
    },
}

OHL_SOURCE = (10.0, 10.0)
OHL_TARGET = (90.0, 90.0)


@pytest.fixture(scope="module")
def ohl_raster(tmp_path_factory):
    """A small uniform raster (the reference config that runs in ~0.1 s)."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path_factory.mktemp("ohl") / "uniform.tif"
    data = np.full((100, 100), 100, dtype=np.uint16)
    with rasterio.open(
            str(path), "w", driver="GTiff", height=100, width=100, count=1,
            dtype="uint16", crs="EPSG:32632",
            transform=from_bounds(0, 0, 100, 100, 100, 100)) as dst:
        dst.write(data, 1)
    return str(path)


def _ohl_draft():
    return {"sources": [{"lat": 0, "lng": 0, "x": OHL_SOURCE[0],
                         "y": OHL_SOURCE[1]}],
            "targets": [{"lat": 0, "lng": 0, "x": OHL_TARGET[0],
                         "y": OHL_TARGET[1]}],
            "waypoints": []}


# ----------------------------------------------------- unified UI (task 46)
def test_unconstrained_run_defers_when_constrained_on(app, state):
    """The main Run button is shared: with the constrained toggle ON the
    unconstrained runner steps aside (PreventUpdate)."""
    from dash.exceptions import PreventUpdate

    with pytest.raises(PreventUpdate):
        invoke(app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
               1, True, _ohl_draft(), None, "dijkstra", "cpu", "r1", 80, True,
               False, False, 1.0, 100, 0, False, [], [],
               triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def test_ohl_waypoint_forces_tower(app, state, ohl_raster):
    import yaml
    from shapely.geometry import Point

    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, ohl_raster,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    draft = _ohl_draft()
    draft["waypoints"] = [{"lat": 0, "lng": 0, "x": 50.0, "y": 50.0}]
    invoke(app, (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN),
           1, True, draft, raster_layer.id, yaml.safe_dump(MINI_PROFILE),
           "cython", False, "r1", 100, None, None, [], [],
           triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    # source -> waypoint -> target chained (each waypoint is a segment endpoint)
    assert len(routes[0].meta["control_points"]) == 3
    towers = next(ly for ly in state.layers_of_kind("vector")
                  if ly.meta.get("towers_for") == routes[0].id)
    # a tower is forced at the waypoint (segment junction)
    assert towers.gdf.distance(Point(50.0, 50.0)).min() < 5.0


# ---------------------------------------------------------- profile handling
def test_shipped_profiles_discovered():
    profiles = constrained.shipped_profiles()
    assert "overhead_line_110kv" in profiles
    assert "rural_road" in profiles
    config = constrained.load_profile_dict(
        profiles["overhead_line_110kv"])
    assert config["hard_angle_limit_deg"] >= config["soft_angle_limit_deg"]


@pytest.mark.parametrize("patch,fragment", [
    ({"hard_angle_limit_deg": 3, "soft_angle_limit_deg": 8}, "hard_angle"),
    ({"min_span_m": 500, "max_span_m": 300}, "min_span_m"),
    ({"span_bin_size_m": 0}, "span_bin_size_m"),
    ({"tower_ground_area_m2": 0.5}, "tower_ground_area_m2"),
    ({"tower_area_cost_mode": "banana"}, "tower_area_cost_mode"),
    ({"angle_cost_function": "cubic"}, "angle_cost_function"),
])
def test_validate_profile_rules(patch, fragment):
    config = dict(MINI_PROFILE, **patch)
    notice = constrained.validate_profile(config)
    assert notice is not None and fragment in notice.meaning


def test_validate_profile_ok():
    assert constrained.validate_profile(MINI_PROFILE) is None


def test_span_bin_perf_warning():
    coarse = dict(MINI_PROFILE, span_bin_size_m=10)
    notice = constrained.validate_span_bin_vs_resolution(coarse, 1.0)
    assert notice is not None and "coarser" in notice.title
    assert constrained.validate_span_bin_vs_resolution(
        MINI_PROFILE, 1.0) is None


# ------------------------------------------------------------- F10 gating
def test_backend_gating():
    backend, note = constrained.resolve_backend("cython", False)
    assert backend == "cython" and note is None
    backend, note = constrained.resolve_backend("raster_gpu_v4", False)
    assert backend == "cython" and "reset" in note.title.lower()
    backend, note = constrained.resolve_backend("raster_gpu_v4", True)
    assert backend == "raster_gpu_v4" and "experimental" in \
        note.title.lower()


# --------------------------------------------------------- a real small run
def test_run_constrained_small(ohl_raster):
    result, towers, crs = constrained.run_constrained(
        ohl_raster, source=OHL_SOURCE, target=OHL_TARGET,
        profile=MINI_PROFILE, backend="cython", neighborhood="r1",
        search_buffer_m=100)
    assert result.path_geometry is not None
    assert result.n_towers >= 1
    assert not towers.empty
    assert "tower_type" in towers.columns
    summary = constrained.result_summary(result)
    assert summary["n_towers"] == result.n_towers
    assert "2-D" in summary["caveat"]                     # F9


# --------------------------------------------------------- headless callbacks
def test_ohl_toggle_and_preset(app):
    resp = invoke(app, (f"{ids.OHL_COLLAPSE}.is_open",), True,
                  triggered=[f"{ids.OHL_ENABLE}.value"])
    assert resp[ids.OHL_COLLAPSE]["is_open"] is True
    options = resp[ids.OHL_PROFILE_PRESET]["options"]
    assert any("110" in o["label"] for o in options)

    preset_path = dict((o["label"], o["value"]) for o in options)[
        "overhead_line_110kv"]
    resp = invoke(app, (f"{ids.OHL_PROFILE_TEXT}.value",
                        ids.OHL_PROFILE_PRESET), preset_path,
                  triggered=[f"{ids.OHL_PROFILE_PRESET}.value"])
    assert "soft_angle_limit_deg" in resp[ids.OHL_PROFILE_TEXT]["value"]


def test_ohl_run_via_callback(app, state, ohl_raster):
    import yaml

    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, ohl_raster,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    profile_text = yaml.safe_dump(MINI_PROFILE)

    resp = invoke(
        app, (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN),
        1, True, _ohl_draft(), raster_layer.id, profile_text, "raster_gpu_v4",
        False, "r1", 100, None, None, [], [],
        triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    assert "towers" in resp[ids.OHL_STATUS]["children"]
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    assert routes[0].meta["params"]["constrained"] is True
    # F10: experimental off -> forced back to cython
    assert routes[0].meta["params"]["backend"] == "cython"
    towers_layers = [ly for ly in state.layers_of_kind("vector")
                     if ly.meta.get("towers_for") == routes[0].id]
    assert len(towers_layers) == 1
    # F9 caveat notice present
    titles = [n["title"] for n in resp[ids.NOTICES]["data"]]
    assert "Terrain cost shown is 2-D" in titles


def test_ohl_run_invalid_profile_blocked(app, state, ohl_raster):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, ohl_raster,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    bad = "soft_angle_limit_deg: 30\nhard_angle_limit_deg: 10\n"
    resp = invoke(
        app, ("notices.data", ids.RUN_ROUTING_BTN, ids.OHL_PROFILE_TEXT),
        1, True, _ohl_draft(), raster_layer.id, bad, "cython", False, "r1",
        100, None, None, [], [],
        triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Profile setting invalid"
    assert state.layers_of_kind("route") == []
