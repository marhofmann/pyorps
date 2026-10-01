"""
PYORPS GUI callbacks: 📁 browse buttons — native open/save/folder dialogs
writing into the path inputs (the familiar Windows file-picker windows).
"""
from __future__ import annotations

from dash import Input, Output, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import dialogs

#: (target input id, dialog mode, filetype kind, default extension)
BROWSE_TARGETS = [
    (ids.LOCAL_PATH, "open", "vector", ""),
    (ids.RASTER_LOAD_PATH, "open", "raster", ""),
    (ids.RASTER_SAVE_PATH, "save", "raster", ".tif"),
    (ids.ROUTES_LOAD_PATH, "open", "routes", ""),
    (ids.EXPORT_ROUTE_PATH, "save", "routes", ".geojson"),
    (ids.COST_IMPORT_PATH, "open", "table", ""),
    (ids.COST_EXPORT_PATH, "save", "table", ".csv"),
    (ids.OHL_PROFILE_PATH, "open", "profile", ""),
    (ids.PROJECT_SAVE_PATH, "directory", "project", ""),
    (ids.PROJECT_OPEN_PATH, "open", "project", ""),
]


def browse_button_id(target: str) -> str:
    return f"browse-{target}"


def _make_handler(mode: str, kind: str, defaultextension: str):
    def handler(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        if mode == "open":
            path, error = dialogs.ask_open(kind)
        elif mode == "save":
            path, error = dialogs.ask_save(
                kind, defaultextension=defaultextension)
        else:
            path, error = dialogs.ask_directory()
        if error or not path:
            return no_update            # cancelled (or headless machine)
        return path

    return handler


def register(app, state) -> None:
    for target, mode, kind, defaultextension in BROWSE_TARGETS:
        app.callback(
            Output(target, "value", allow_duplicate=True),
            Input(browse_button_id(target), "n_clicks"),
            prevent_initial_call=True,
        )(_make_handler(mode, kind, defaultextension))
