"""
Round-10 GUI changes (2026-07-20):

- one notice per distinct warning per user action (guard dedup) + numpy
  "invalid value encountered in cast" silenced as noise
- euclidean leg-distance column in the control-points grid
- QGIS-style graduated raster rendering (class breaks + custom LUT via
  localtileserver register_colormap) with editable per-class colours
- heavy-run gate: estimate → confirm modal → background job + Stop button
- selective project save (raster/vector/cost-table checklist; routes always)
  + dirty tracking for the save-before-close guard
"""
import json
import threading
import warnings
from pathlib import Path

from pyorps.gui import ids

from conftest import SOURCE, TARGET, invoke

GUI_DIR = Path(__file__).resolve().parents[2] / "pyorps" / "gui"


# ========================================================== warning dedup
def test_guard_dedups_identical_warnings_per_action():
    from pyorps.gui.services.errors import guard

    def noisy():
        for _ in range(27):
            warnings.warn("something odd happened", RuntimeWarning)
        warnings.warn("a different problem", RuntimeWarning)
        return 42

    result, notices = guard(noisy, notices=[])
    assert result == 42
    warn_titles = [n["details"] for n in notices
                   if n["severity"] == "warning"]
    assert sorted(warn_titles) == ["a different problem",
                                   "something odd happened"]


def test_guard_silences_numpy_cast_noise():
    from pyorps.gui.services.errors import guard

    def caster():
        warnings.warn("invalid value encountered in cast", RuntimeWarning)
        return "ok"

    result, notices = guard(caster, notices=[])
    assert result == "ok"
    assert notices == []


# ===================================================== leg-distance column
def test_leg_distances_annotates_rows():
    from pyorps.gui.callbacks.interaction import leg_distances

    rows = leg_distances([
        {"kind": "source", "x": 0.0, "y": 0.0, "name": ""},
        {"kind": "waypoint", "x": 3.0, "y": 4.0, "name": "w"},
        {"kind": "target", "x": 3.0, "y": 104.0, "name": ""},
    ])
    assert rows[0]["dist"] is None
    assert rows[1]["dist"] == 5.0
    assert rows[2]["dist"] == 100.0


def test_points_table_has_distance_column(state):
    from pyorps.gui.layout import build_layout

    for component in build_layout(state)._traverse():
        if getattr(component, "id", None) == ids.BUILD_POINTS_TABLE:
            fields = [c.get("field") for c in component.columnDefs]
            assert "dist" in fields
            return
    raise AssertionError("points table not found in layout")


def test_selected_route_rows_carry_leg_distance(state):
    from pyorps.gui.callbacks.edit import points_rows

    layer = state.add_layer("r", "route", meta={
        "control_points": [[0.0, 0.0], [30.0, 40.0], [30.0, 140.0]],
        "waypoint_names": ["w1"]})
    rows = points_rows(layer)
    assert [r["dist"] for r in rows] == [None, 50.0, 100.0]


# ======================================================= graduated renderer
def test_graduated_end_to_end_on_served_raster(state, raster_path):
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.services import graduated

    layer, _ = add_raster_layer(state, raster_path, [])
    config = graduated.build_config(raster_path, "quantile", 4, "viridis")
    assert config["classes"] >= 2
    assert len(config["breaks"]) == config["classes"] + 1
    assert len(config["colors"]) == config["classes"]

    url = graduated.apply_to_layer(layer, config)
    assert "custom" in url                    # custom LUT is being served
    assert layer.meta["graduated"] == config
    assert layer.tile.colormap.startswith("custom:")

    # editing one class colour re-serves under a NEW registered LUT
    config["colors"][0] = "#ff0000"
    url2 = graduated.apply_to_layer(layer, config)
    assert url2 != url


def test_graduated_callbacks_registered(app):
    keys = "\n".join(app.callback_map)
    assert ids.GRAD_STATUS in keys                 # apply_graduated outputs
    blob = json.dumps(
        {k: {"inputs": v.get("inputs"), "state": v.get("state")}
         for k, v in app.callback_map.items()}, default=str)
    assert ids.GRAD_APPLY_BTN in blob
    assert ids.TYPE_GRAD_COLOR in blob             # per-class colour editor


def test_continuous_mode_restores_named_ramp(state, raster_path):
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.services import graduated, tiles

    layer, _ = add_raster_layer(state, raster_path, [])
    graduated.apply_to_layer(
        layer, graduated.build_config(raster_path, "equal", 3, "viridis"))
    assert layer.tile.colormap.startswith("custom:")
    layer.meta.pop("graduated", None)
    tiles.set_colormap(layer.tile, "terrain")
    assert layer.tile.colormap == "terrain"


# ========================================================= routing gate/job
def test_estimate_small_run_needs_no_confirmation(raster_path):
    from pyorps.gui.services import routing

    est = routing.estimate_routing(
        raster_path, sources=[SOURCE], targets=[TARGET],
        neighborhood="r1", search_buffer_m=100.0)
    assert est["n_pairs"] == 1
    assert est["total_cells"] > 0
    assert est["est_seconds"] >= 0
    assert not routing.needs_confirmation(est)


def test_needs_confirmation_thresholds():
    from pyorps.gui.services import routing

    base = {"est_seconds": 1, "est_memory_mb": 1, "n_pairs": 1,
            "total_cells": 1}
    assert not routing.needs_confirmation(dict(base))
    assert routing.needs_confirmation({**base, "est_seconds": 60})
    assert routing.needs_confirmation({**base, "est_memory_mb": 9000})
    assert routing.needs_confirmation({**base, "n_pairs": 40})
    assert routing.needs_confirmation({**base, "total_cells": 60_000_000})


def test_cancel_before_start_yields_no_routes(raster_path):
    from pyorps.gui.services import routing

    cancel = threading.Event()
    cancel.set()
    finder, built, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET],
        neighborhood="r1", search_buffer_m=100.0, cancel=cancel)
    assert built == [] and failed == []


def test_background_job_completes_and_poll_finalizes(app, state, raster_path):
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.services import routing_job

    layer, _ = add_raster_layer(state, raster_path, [])
    params = {
        "raster_path": raster_path,
        "kwargs": dict(sources=[SOURCE], targets=[TARGET], waypoints=[],
                       algorithm="delta-stepping", hardware="cpu",
                       neighborhood="r1", pairwise=False,
                       search_buffer_m=100.0, ignore_max_cost=True,
                       delta=100.0, num_threads=0, use_astar=False),
        "meta": {"raster_layer_id": layer.id, "algorithm": "delta-stepping",
                 "simplify_tol": None, "wp_names": [], "n_sources": 1,
                 "n_targets": 1, "n_waypoints": 0},
    }
    job = routing_job.start_routing_job(state, params)
    job.thread.join(timeout=180)
    assert job.status == "done"
    assert job.result is not None

    invoke(app, (f"{ids.ROUTING_POLL}.disabled", "n_intervals"), 1, [],
           triggered=[f"{ids.ROUTING_POLL}.n_intervals"])
    routes = state.layers_of_kind("route")
    assert len(routes) == 1
    assert state.routing_job is None
    assert routes[0].meta["metrics"]["total_cost"] > 0


def test_confirm_cancel_clears_pending(app, state):
    state.pending_routing = {"raster_path": "x", "kwargs": {}, "meta": {}}
    resp = invoke(app, (f"{ids.ROUTING_CONFIRM_MODAL}.is_open",
                        ids.ROUTING_CONFIRM_CANCEL), 1,
                  triggered=[f"{ids.ROUTING_CONFIRM_CANCEL}.n_clicks"])
    assert resp[ids.ROUTING_CONFIRM_MODAL]["is_open"] is False
    assert state.pending_routing is None


def test_layout_has_confirm_modal_and_hidden_stop(state):
    from pyorps.gui.layout import build_layout

    found = {}
    for component in build_layout(state)._traverse():
        cid = getattr(component, "id", None)
        if cid in (ids.ROUTING_CONFIRM_MODAL, ids.ROUTING_STOP_BTN,
                   ids.ROUTING_POLL):
            found[cid] = component
    assert set(found) == {ids.ROUTING_CONFIRM_MODAL, ids.ROUTING_STOP_BTN,
                          ids.ROUTING_POLL}
    assert found[ids.ROUTING_STOP_BTN].style == {"display": "none"}
    assert found[ids.ROUTING_POLL].disabled is True


# ================================================ selective save + dirty flag
def test_selective_save_filters_kinds_and_clears_dirty(state, tmp_path,
                                                       raster_path):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.services import project_io

    add_raster_layer(state, raster_path, [])
    vec = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)],
                           crs="EPSG:25832")
    state.add_layer("vec", "vector", gdf=vec, crs=vec.crs)
    route = gpd.GeoDataFrame({"name": ["r"]},
                             geometry=[LineString([(0, 0), (1, 1)])],
                             crs="EPSG:25832")
    state.add_layer("r", "route", gdf=route, crs=route.crs,
                    meta={"control_points": [[0, 0], [1, 1]]})
    assert state.dirty is True

    out = project_io.save_project(state, str(tmp_path / "proj"),
                                  cost_table={"rows": [{"cost": 1}]},
                                  include=["vector"])
    manifest = json.loads(Path(out).read_text(encoding="utf-8"))
    kinds = sorted(e["kind"] for e in manifest["layers"])
    assert "vector" in kinds and "route" in kinds     # routes ALWAYS saved
    assert "raster" not in kinds                      # excluded group
    assert manifest["cost_table"] == {}               # not included
    assert state.dirty is False                       # save clears the flag


def test_dirty_flag_lifecycle(state):
    assert state.dirty is False
    layer = state.add_layer("v", "vector", geojson={
        "type": "FeatureCollection", "features": []})
    assert state.dirty is True
    state.remove_layer(layer.id)
    assert state.dirty is True
    state.clear()
    assert state.dirty is False


def test_dirty_store_and_unload_guard_wired(app, state):
    resp = invoke(app, (f"{ids.DIRTY_STORE}.data", ids.LAYERS_VIEW),
                  [], "", triggered=[f"{ids.LAYERS_VIEW}.data"])
    assert resp[ids.DIRTY_STORE]["data"] is False
    state.add_layer("v", "vector", geojson={
        "type": "FeatureCollection", "features": []})
    resp = invoke(app, (f"{ids.DIRTY_STORE}.data", ids.LAYERS_VIEW),
                  [], "", triggered=[f"{ids.LAYERS_VIEW}.data"])
    assert resp[ids.DIRTY_STORE]["data"] is True

    js = (GUI_DIR / "assets" / "unsaved-guard.js").read_text(encoding="utf-8")
    assert "beforeunload" in js and "__pyorpsDirty" in js


def test_save_include_checklist_in_layout(state):
    from pyorps.gui.layout import build_layout

    for component in build_layout(state)._traverse():
        if getattr(component, "id", None) == ids.SAVE_INCLUDE:
            values = {o["value"] for o in component.options}
            assert values == {"raster", "vector", "cost_table"}
            assert set(component.value) == values      # all on by default
            return
    raise AssertionError("save-include checklist not found")
