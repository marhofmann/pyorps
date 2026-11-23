"""
PYORPS GUI callbacks: 📁 browse buttons — native open/save/folder dialogs
writing into the path inputs (the familiar Windows file-picker windows).

The dialog runs in a child process (:mod:`pyorps.gui.services.dialogs`). When
none can be shown (no display, no tkinter) the click answers with a notice that
says so and asks for the path to be typed; cancelling a dialog stays silent.
The notice is APPENDED to the notices store with a ``Patch`` (no ``State`` of
the store needed, so the callbacks keep their single ``n_clicks`` input and the
notice history is never overwritten).
"""
from __future__ import annotations

from dash import Input, Output, Patch, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import dialogs
from ..services.errors import Notice

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


def dialog_unavailable_notice(reason: str) -> Notice:
    """The notice shown when no native file dialog could be opened."""
    return Notice(
        severity="warning",
        title="The file dialog is not available",
        meaning="PYORPS could not open a file dialog on the machine that "
                f"runs the GUI server ({reason}).",
        impact="The path field was not filled in; nothing was loaded or saved.",
        fix="Type or paste the full path into the field next to the 📁 "
            "button instead.",
        details=str(reason))


def _make_handler(mode: str, kind: str, defaultextension: str):
    def handler(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        if mode == "open":
            result = dialogs.ask_open(kind)
        elif mode == "save":
            result = dialogs.ask_save(kind, defaultextension=defaultextension)
        else:
            result = dialogs.ask_directory()
        path, error = result                # DialogResult is (path, error)
        if path:
            return path, no_update
        if error:                           # no dialog could be shown
            append = Patch()
            append.append(dialog_unavailable_notice(error).to_dict())
            return no_update, append
        return no_update, no_update         # cancelled: stay silent

    return handler


def register(app, state) -> None:  # pylint: disable=unused-argument
    for target, mode, kind, defaultextension in BROWSE_TARGETS:
        app.callback(
            Output(target, "value", allow_duplicate=True),
            Output(ids.NOTICES, "data", allow_duplicate=True),
            Input(browse_button_id(target), "n_clicks"),
            prevent_initial_call=True,
        )(_make_handler(mode, kind, defaultextension))
