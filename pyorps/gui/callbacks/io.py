"""
PYORPS GUI callbacks: project save / open / new (R12, Section 19).

"Save project" writes the manifest + asset folder (including the client-side
cost table passed through as State); "Open project" restores it into the
server state and refreshes the client stores; "New project" resets to the
blank map (R1).
"""
from __future__ import annotations

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import project_io
from ..services.errors import Notice, guard, success


def register(app, state) -> None:
    @app.callback(
        Output(ids.PROJECT_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.PROJECT_SAVE_BTN, "n_clicks"),
        State(ids.PROJECT_SAVE_PATH, "value"),
        State(ids.SAVE_INCLUDE, "value"),
        State(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID_STATE, "data"),
        State(ids.MODIFIER_GRID, "rowData"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def save_project(n_clicks, path, include, cost_rows, grid_state,
                     modifiers, notices):
        notices = list(notices or [])
        if not n_clicks:
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
        included = ", ".join(include) if include else "routes only"
        notices.append(success(
            "Project saved",
            meaning=f"Manifest written to {result} "
                    f"(included: {included}; routes always)."))
        return f"saved: {result}", notices

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
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.PROJECT_STATUS, "children", allow_duplicate=True),
        Output(ids.PROJECT_CRS, "value"),
        Output(ids.COST_GRID, "rowData", allow_duplicate=True),
        Output(ids.COST_GRID_STATE, "data", allow_duplicate=True),
        Output(ids.MODIFIER_GRID, "rowData", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.PROJECT_OPEN_BTN, "n_clicks"),
        State(ids.PROJECT_OPEN_PATH, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def open_project(n_clicks, path, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        if not path:
            notices.append(Notice(
                severity="warning", title="Pick a project.json first",
                meaning="Opening needs the manifest path.",
                impact="Nothing was loaded.",
                fix="Enter the path to a saved project.json (or use 📁).",
                focus_id=ids.TAB_DATA,
                focus_control=ids.PROJECT_OPEN_PATH).to_dict())
            return (no_update,) * 7 + (notices,)
        manifest, notices = guard(project_io.load_project, state, path,
                                  notices=notices)
        if manifest is None:
            return (no_update,) * 7 + (notices,)
        cost_table = manifest.get("cost_table") or {}
        grid_state = {"dataset_id": cost_table.get("dataset_id"),
                      "feature_keys": cost_table.get("feature_keys") or []}
        route_options = [{"label": ly.name, "value": ly.id}
                         for ly in state.layers_of_kind("route")]
        notices.append(success(
            "Project opened",
            meaning=f"{len(state.layers)} layers restored."))
        return (state.layers_view(), f"opened: {path}",
                manifest.get("project_crs") or no_update,
                cost_table.get("rows") or [], grid_state,
                cost_table.get("modifiers") or [], route_options, notices)

    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.PROJECT_STATUS, "children", allow_duplicate=True),
        Output(ids.ROUTE_DRAFT, "data", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.ACTIVE_ROUTE, "positions", allow_duplicate=True),
        Output(ids.CONTROL_MARKERS, "children", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.PROJECT_NEW_BTN, "n_clicks"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def new_project(n_clicks, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        state.clear()
        notices.append(success("New project",
                               meaning="Back to a blank map (R1)."))
        empty_draft = {"sources": [], "targets": [], "waypoints": []}
        return (state.layers_view(), "new project", empty_draft, [], None,
                [], [], notices)
