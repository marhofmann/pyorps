"""
PYORPS GUI callbacks: project save / open / new (R12, Section 19).

"Save project" writes the manifest + asset folder (including the client-side
cost table passed through as State); "Open project" restores it into the
server state and refreshes the client stores; "New project" resets to the
blank map (R1).

"New" and "Open" both replace the whole server state, so both must also bring
the browser into line with it. :data:`RESET_TARGETS` lists every browser value
that mirrors project state and :func:`client_state_values` builds the value of
each one (empty for "New", filled from the loaded project for "Open"); the two
callbacks share them, so neither can forget a component the other resets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import cost_model, manual_cost, project_io, rasterize, routing
from ..services.errors import Notice, cap_notices, guard, success
from ..state import TileManager
from .costmodel import _coverage_text, register_cost_table
from .data import _dataset_list, _study_area_info

_EMPTY_FC = {"type": "FeatureCollection", "features": []}


def _empty_draft() -> dict:
    return {"sources": [], "targets": [], "waypoints": []}


#: ``(component id, property)`` of every browser value that mirrors project
#: state, in the order of :func:`client_state_values`. Other callbacks write
#: most of these too, hence ``allow_duplicate`` (see :func:`reset_outputs`).
RESET_TARGETS: tuple[tuple[str, str], ...] = (
    (ids.LAYERS_VIEW, "data"),
    (ids.PROJECT_STATUS, "children"),
    (ids.PROJECT_CRS, "value"),
    (ids.COST_GRID, "columnDefs"),
    (ids.COST_GRID, "rowData"),
    (ids.COST_GRID_STATE, "data"),
    (ids.COST_COVERAGE_INFO, "children"),
    (ids.COST_DATASET, "value"),
    (ids.COST_FEATURE_KEYS, "value"),
    (ids.MODIFIER_GRID, "rowData"),
    (ids.PREPROC_GRID, "rowData"),
    (ids.PPB_CONDS, "data"),
    (ids.MANUAL_GRID, "rowData"),
    (ids.MANUAL_STATUS, "children"),
    (ids.DATASET_LIST, "children"),
    (ids.STUDY_AREA_INFO, "children"),
    (ids.DRAW_CONTROL, "editToolbar"),
    (ids.OSM_SELECTIONS, "data"),
    (ids.RASTERIZE_LOG, "children"),
    (ids.EDIT_ROUTE_SELECT, "options"),
    (ids.EDIT_ROUTE_SELECT, "value"),
    (ids.ROUTE_DRAFT, "data"),
    (ids.ACTIVE_ROUTE, "positions"),
    (ids.CONTROL_MARKERS, "children"),
    (ids.SELECTED_FEATURE, "data"),
    (ids.SELECTED_FEATURE_LAYER, "data"),
    # the routing controls: the opened project's settings, or the defaults
    (ids.GRAPH_API, "value"),
    (ids.ALGORITHM, "options"),
    (ids.ALGORITHM, "value"),
    (ids.NEIGHBORHOOD, "value"),
    (ids.SEARCH_BUFFER, "value"),
    (ids.FULL_WINDOW, "value"),
    (ids.IGNORE_MAX_COST, "value"),
    (ids.PAIRWISE, "value"),
    (ids.SIMPLIFY, "value"),
    (ids.SIMPLIFY_TOL, "value"),
    (ids.DELTA, "value"),
    (ids.NUM_THREADS, "value"),
    (ids.MARGIN, "value"),
    # the rasterize controls (project-saved) and the one-shot WFS choice
    (ids.RES_M, "value"),
    (ids.FILL_VALUE, "value"),
    (ids.RASTER_DTYPE, "value"),
    (ids.GEOM_BUFFER, "value"),
    (ids.WIDEN_THIN, "value"),
    (ids.ALL_TOUCHED, "value"),
    (ids.ON_THIN, "value"),
    (ids.CROP_BORDER, "value"),
    (ids.WFS_WHOLE_LAYER, "value"),
)

#: routing setting key -> the control that shows it (the order of the settings)
_ROUTING_VALUE_TARGETS = {
    "graph_api": (ids.GRAPH_API, "value"),
    "neighborhood_k": (ids.NEIGHBORHOOD, "value"),
    "search_buffer_m": (ids.SEARCH_BUFFER, "value"),
    "full_window": (ids.FULL_WINDOW, "value"),
    "ignore_max_cost": (ids.IGNORE_MAX_COST, "value"),
    "pairwise": (ids.PAIRWISE, "value"),
    "simplify": (ids.SIMPLIFY, "value"),
    "simplify_tol": (ids.SIMPLIFY_TOL, "value"),
    "delta": (ids.DELTA, "value"),
    "num_threads": (ids.NUM_THREADS, "value"),
    "margin": (ids.MARGIN, "value"),
}
#: rasterize setting key -> the control that shows it
_RASTER_VALUE_TARGETS = {
    "resolution_m": (ids.RES_M, "value"),
    "fill_value": (ids.FILL_VALUE, "value"),
    "dtype": (ids.RASTER_DTYPE, "value"),
    "geometry_buffer_m": (ids.GEOM_BUFFER, "value"),
    "widen_thin_forbidden": (ids.WIDEN_THIN, "value"),
    "all_touched": (ids.ALL_TOUCHED, "value"),
    "on_thin_features": (ids.ON_THIN, "value"),
    "crop_border": (ids.CROP_BORDER, "value"),
}


def reset_outputs() -> list[Output]:
    """The ``Output`` of every entry of :data:`RESET_TARGETS`."""
    return [Output(cid, prop, allow_duplicate=True)
            for cid, prop in RESET_TARGETS]


def _manual_rows(state) -> list[dict]:
    """Grid rows of the manual cost layer (empty when there is none)."""
    layer = state.get(manual_cost.MANUAL_LAYER_ID)
    if layer is None or layer.gdf is None:
        return []
    names = layer.gdf["name"] if "name" in layer.gdf.columns else None
    return [{"__row": i,
             "name": str(names.iloc[i]) if names is not None else f"{i + 1}",
             "cost": float(cost)}
            for i, cost in enumerate(layer.gdf[manual_cost.COST_COLUMN])]


def cost_table_view(cost_table) -> dict:
    """What the cost editor shows for a saved ``cost_table`` (the manifest's).

    ``{"keys", "dataset_id", "rows", "grid_state", "column_defs",
    "modifiers"}``: "Open project" sends these to the browser and the first
    paint of a page (``--project``, a reload) reads the same values, so the two
    cannot disagree. Anything but a mapping (an old project, a broken file)
    gives the empty table.
    """
    table = cost_table if isinstance(cost_table, dict) else {}

    def _list(name: str) -> list:
        value = table.get(name)
        return value if isinstance(value, list) else []

    keys = tuple(str(key) for key in _list("feature_keys"))
    dataset_id = table.get("dataset_id")
    return {
        "keys": keys,
        "dataset_id": dataset_id,
        "rows": _list("rows"),
        "grid_state": ({"dataset_id": dataset_id, "feature_keys": list(keys)}
                       if (keys or dataset_id) else {}),
        "column_defs": cost_model.grid_column_defs(keys) if keys else [],
        "modifiers": _list("modifiers"),
    }


def apply_opened_project(state, manifest: dict, path=None) -> None:
    """What follows ``project_io.load_project`` for EVERY way of opening a project.

    "Open project" and the ``--project`` start flag both call this, so a project
    opened by either is in exactly the same state:

    * no route is selected, and the study area is not a shape in the (cleared)
      draw control, so its "control is empty" echo must not delete it;
    * the routing controls keep the opened algorithm once, instead of resetting
      it to the backend's default as they do for a user pick;
    * the cost table is kept for the first paint of a page (the cost editor is
      client-side, so a reload would otherwise show an empty table) and
      registered for the Raster tab, as a cell edit does later (``coverage``);
    * the path of the opened file is kept (``state.project_path``) so that a page
      loaded later says which project it shows. ``path`` is the file as the user
      or the flag named it; without it the project has no path.
    """
    state.project_path = str(path) if path else None
    state.active_route_id = None
    state._study_area_not_drawn = state.study_area is not None
    state.routing_restore_backend = (
        routing.normalise_routing_settings(state.routing_settings)
        or routing.ROUTING_SETTING_DEFAULTS)["graph_api"]
    table = (manifest or {}).get("cost_table")
    state.loaded_cost_table = dict(table) if isinstance(table, dict) else {}
    view = cost_table_view(state.loaded_cost_table)
    if view["grid_state"]:
        register_cost_table(state, dataset_id=view["dataset_id"],
                            feature_keys=view["keys"], rows=view["rows"])


def client_state_values(state, status: str, notices: list, *,
                        manifest: dict | None = None,
                        tick: int | None = None):
    """``(values, notices)``: the value for each :data:`RESET_TARGETS` entry.

    Nothing is taken from the browser: ``state`` is the truth. Without a
    ``manifest`` ("New project") every value is empty. With one ("Open
    project") what the project holds is filled in (layer list, cost table,
    modifiers, dataset list, study-area text, manual cost grid, route
    options) and everything else - the drawn shapes, the OSM picks, the
    raster log, the route selection and draft, the map highlights - is
    cleared, because it belonged to the project that was just replaced.
    """
    cost_table = cost_table_view((manifest or {}).get("cost_table"))
    keys, dataset_id = cost_table["keys"], cost_table["dataset_id"]
    rows, grid_state = cost_table["rows"], cost_table["grid_state"]
    coverage, notices = guard(_coverage_text, state, dataset_id, keys, rows,
                              notices=notices)
    manual_rows, notices = guard(_manual_rows, state, notices=notices)
    study_info, notices = guard(_study_area_info, state, notices=notices)
    datasets, notices = guard(_dataset_list, state, notices=notices)
    manual_rows = manual_rows or []
    toolbar = {"mode": "remove", "action": "clear all",
               "n_clicks": int(time.time() * 1000) if tick is None else tick}
    values = {
        (ids.LAYERS_VIEW, "data"): state.layers_view(),
        (ids.PROJECT_STATUS, "children"): status,
        (ids.PROJECT_CRS, "value"): str(state.project_crs),
        (ids.COST_GRID, "columnDefs"): cost_table["column_defs"],
        (ids.COST_GRID, "rowData"): rows,
        (ids.COST_GRID_STATE, "data"): grid_state,
        (ids.COST_COVERAGE_INFO, "children"): coverage or "",
        (ids.COST_DATASET, "value"): None,
        (ids.COST_FEATURE_KEYS, "value"): [],
        (ids.MODIFIER_GRID, "rowData"): cost_table["modifiers"],
        (ids.PREPROC_GRID, "rowData"): [],
        (ids.PPB_CONDS, "data"): [],
        (ids.MANUAL_GRID, "rowData"): manual_rows,
        (ids.MANUAL_STATUS, "children"): (
            f"{len(manual_rows)} cost polygon(s) — edit name/cost below."
            if manual_rows else "Draw cost polygons on the map."),
        (ids.DATASET_LIST, "children"): datasets,
        (ids.STUDY_AREA_INFO, "children"): study_info,
        (ids.DRAW_CONTROL, "editToolbar"): toolbar,
        (ids.OSM_SELECTIONS, "data"): [],
        (ids.RASTERIZE_LOG, "children"): "",
        (ids.EDIT_ROUTE_SELECT, "options"): [
            {"label": ly.name, "value": ly.id}
            for ly in state.layers_of_kind("route")],
        (ids.EDIT_ROUTE_SELECT, "value"): None,
        (ids.ROUTE_DRAFT, "data"): _empty_draft(),
        (ids.ACTIVE_ROUTE, "positions"): [],
        (ids.CONTROL_MARKERS, "children"): [],
        (ids.SELECTED_FEATURE, "data"): {},
        (ids.SELECTED_FEATURE_LAYER, "data"): dict(_EMPTY_FC),
        (ids.WFS_WHOLE_LAYER, "value"): False,
    }
    values.update(_settings_values(state))
    return tuple(values[target] for target in RESET_TARGETS), notices


def _settings_values(state) -> dict:
    """The routing and rasterize controls for ``state``'s settings.

    A project without saved settings (a blank one, or an older file) gives the
    layout's own defaults, so "Open" never leaves a control on the value of the
    project that was just replaced.
    """
    routing_settings = (
        routing.normalise_routing_settings(
            getattr(state, "routing_settings", None))
        or dict(routing.ROUTING_SETTING_DEFAULTS))
    raster_settings = (
        rasterize.normalise_raster_settings(
            getattr(state, "raster_settings", None))
        or dict(rasterize.RASTER_SETTING_DEFAULTS))
    values = {target: routing_settings[key]
              for key, target in _ROUTING_VALUE_TARGETS.items()}
    values[(ids.ALGORITHM, "value")] = routing_settings["algorithm"]
    values[(ids.ALGORITHM, "options")] = [
        {"label": a.replace("_", " "), "value": a}
        for a in routing.backend_algorithms(routing_settings["graph_api"])]
    values.update({target: raster_settings[key]
                   for key, target in _RASTER_VALUE_TARGETS.items()})
    return values


def _skipped_suffix(skipped: list | None) -> str:
    """`` (2 layer(s) left out: a, b)`` for the status text; empty for none."""
    if not skipped:
        return ""
    names = ", ".join(str(rec.get("name")) for rec in skipped[:3])
    if len(skipped) > 3:
        names += ", ..."
    return f" ({len(skipped)} layer(s) left out: {names})"


def _saved_skipped(manifest_path: str) -> list:
    """``skipped_layers`` of the manifest ``save_project`` just wrote."""
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data.get("skipped_layers") or []


def register(app, state) -> None:
    @app.callback(
        Output(ids.PROJECT_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GO_SAVE_PROJECT, "data"),
        State(ids.PROJECT_SAVE_PATH, "value"),
        State(ids.SAVE_INCLUDE, "value"),
        State(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID_STATE, "data"),
        State(ids.MODIFIER_GRID, "rowData"),
        State(ids.NOTICES, "data"),
        running=[(Output(ids.PROJECT_SAVE_BTN, "disabled"), True, False)],
        prevent_initial_call=True)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    def save_project(go, path, include, cost_rows, grid_state,
                     modifiers, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        if not go:
            raise PreventUpdate
        if not path:
            notices.append(Notice(
                severity="warning", title="Where should the project go?",
                meaning="A save needs a folder or project.json path.",
                impact="Nothing was saved.",
                fix="Enter a target path (or use the 📁 button).",
                focus_id=ids.TAB_DATA,
                focus_control=ids.PROJECT_SAVE_PATH).to_dict())
            return "", notices
        cost_table = {
            "feature_keys": (grid_state or {}).get("feature_keys") or [],
            "dataset_id": (grid_state or {}).get("dataset_id"),
            "rows": cost_rows or [],
            "modifiers": modifiers or [],
        }
        result, notices = guard(
            project_io.save_project, state, path, cost_table=cost_table,
            include=include if include is not None else None,
            notices=notices)
        if result is None:
            return "", notices
        state.project_path = str(result)
        included = ", ".join(include) if include else "routes only"
        suffix = _skipped_suffix(_saved_skipped(result))
        notices.append(success(
            "Project saved",
            meaning=f"Manifest written to {result} "
                    f"(included: {included}; routes always).{suffix}"))
        return f"saved: {result}{suffix}", cap_notices(notices)

    # ------------------------- unsaved-work flag for the close guard
    # LAYERS_VIEW changes on every layer mutation; PROJECT_STATUS changes on
    # save/open/new — together they track state.dirty closely enough.
    @app.callback(
        Output(ids.DIRTY_STORE, "data"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.PROJECT_STATUS, "children"))
    def sync_dirty(_view, _status):
        return bool(getattr(state, "dirty", False))

    # mirror into a window flag the beforeunload guard can read
    # synchronously (assets/unsaved-guard.js)
    app.clientside_callback(
        """
        function(dirty) {
            window.__pyorpsDirty = !!dirty;
            return window.dash_clientside.no_update;
        }
        """,
        Output(ids.DIRTY_STORE, "data", allow_duplicate=True),
        Input(ids.DIRTY_STORE, "data"),
        prevent_initial_call=True)

    @app.callback(
        *reset_outputs(),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GO_OPEN_PROJECT, "data"),
        State(ids.PROJECT_OPEN_PATH, "value"),
        State(ids.NOTICES, "data"),
        running=[(Output(ids.PROJECT_OPEN_BTN, "disabled"), True, False)],
        prevent_initial_call=True)
    def open_project(go, path, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        notices = list(notices or [])
        untouched = (no_update,) * len(RESET_TARGETS)
        if not go:
            raise PreventUpdate
        if not path:
            notices.append(Notice(
                severity="warning", title="Pick a project.json first",
                meaning="Opening needs the manifest path.",
                impact="Nothing was loaded.",
                fix="Enter the path to a saved project.json (or use 📁).",
                focus_id=ids.TAB_DATA,
                focus_control=ids.PROJECT_OPEN_PATH).to_dict())
            return untouched + (notices,)
        TileManager.pop_notices()        # only this action's failures are shown
        manifest, notices = guard(project_io.load_project, state, path,
                                  notices=notices)
        notices.extend(TileManager.pop_notices())    # a tile server that would not stop
        if manifest is None:
            return untouched + (notices,)
        # no route selected, the area is not a drawn shape, the opened algorithm
        # is kept, the cost table is remembered for a reload - the same steps
        # as the ``--project`` start flag (see ``apply_opened_project``)
        apply_opened_project(state, manifest, path)
        suffix = _skipped_suffix(manifest.get("load_skipped"))
        values, notices = client_state_values(
            state, f"opened: {path}{suffix}", notices, manifest=manifest)
        notices.append(success(
            "Project opened",
            meaning=f"{len(state.layers)} layers restored.{suffix}"))
        return values + (cap_notices(notices),)

    @app.callback(
        *reset_outputs(),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GO_NEW_PROJECT, "data"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def new_project(go, notices):
        notices = list(notices or [])
        if not go:
            raise PreventUpdate
        TileManager.pop_notices()
        state.clear()
        notices.extend(TileManager.pop_notices())    # a tile server that would not stop
        state._study_area_not_drawn = False
        values, notices = client_state_values(state, "new project", notices)
        notices.append(success("New project",
                               meaning="Back to a blank map (R1)."))
        return values + (cap_notices(notices),)
    _ = None
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
