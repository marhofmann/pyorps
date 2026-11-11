"""
PYORPS GUI callbacks: confirm before destroying (audit T5 #3 and #9).

One in-page modal (title / text / confirm label) and one small store with the
action that waits for the answer. Every destructive button feeds a *gate*
callback below; the callback that really acts (new project, remove layers,
export ...) listens on the action's *GO store* (a counter) and never on the
button itself:

* the gate asks when something would be lost (unsaved work, selected layers,
  routes, a file that already exists) by opening the modal and parking the
  action in :data:`ids.CONFIRM_PENDING`;
* when nothing would be lost (an empty project, a new file name, nothing
  selected) it writes the GO store at once, so the click acts as before;
* "Cancel" closes the modal and drops the pending action: nothing changed;
* the confirm button writes the GO store of the pending action.

No browser dialog is involved (``dcc.ConfirmDialog`` and ``window.confirm``
are blocked by many embedders and cannot be tested headlessly).
"""
from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import project_io

#: values written to the GO stores: always new, so Dash sees a change each time
_SEQUENCE = itertools.count(1)

#: how many names a question lists before it says "and N more"
_MAX_NAMES = 5


@dataclass(frozen=True)
class Question:
    """What the modal says: a title, the text and the confirm button's label."""

    title: str
    body: list
    confirm: str


#: ``ask(state, *values) -> Question | None``: None = nothing would be lost, go
#: ahead; it raises ``PreventUpdate`` when the click has nothing to act on
Asker = Callable[..., "Question | None"]


@dataclass(frozen=True)
class Gate:
    """One destructive action: its button, its GO store and its question."""

    action: str
    button: str
    go: str
    states: tuple[tuple[str, str], ...]
    ask: Asker


# ------------------------------------------------------------------- helpers
def _names(names: list[str]) -> str:
    """``'a', 'b' and 3 more`` for the question text."""
    shown = ", ".join(f"'{name}'" for name in names[:_MAX_NAMES])
    more = len(names) - _MAX_NAMES
    return f"{shown} and {more} more" if more > 0 else shown


def _lines(*paragraphs: str) -> list:
    return [html.P(text, className="small mb-1") for text in paragraphs]


def unsaved_summary(state) -> str:
    """What the open project holds that is not on disk, in plain words."""
    kinds = Counter(layer.kind for layer in state.layers.values()
                    if layer.kind != "study_area")
    parts = [f"{count} {kind}{'' if count == 1 else 's'}"
             for kind, count in sorted(kinds.items())]
    if state.study_area is not None:
        parts.append("the study area")
    return ", ".join(parts) if parts else "changes to the project"


def has_unsaved_work(state) -> bool:
    """Whether replacing the project would lose something: it changed since the
    last save AND something is left in it (an emptied project loses nothing)."""
    return bool(state.dirty and (state.layers or state.study_area is not None))


def _overwrite(path: Any, what: str) -> Question | None:
    """The overwrite question for ``path``; None for a blank or new path."""
    if not path or not str(path).strip():
        return None
    target = Path(str(path).strip())
    if not target.exists():
        return None
    return Question(
        title=f"Replace the existing {what}?",
        body=_lines(f"'{target.name}' already exists in {target.parent}.",
                    "Continuing replaces it; the old content cannot be "
                    "brought back from here."),
        confirm="Replace it")


# -------------------------------------------------------------------- askers
def _ask_new_project(state) -> Question | None:
    if not has_unsaved_work(state):
        return None
    return Question(
        title="Start a new project?",
        body=_lines(f"This project has unsaved work: {unsaved_summary(state)}.",
                    "A new project clears all of it. Use 'Save project' "
                    "first if you want to keep it."),
        confirm="Discard and start new")


def _ask_open_project(state, path) -> Question | None:
    if not path or not has_unsaved_work(state):   # no path: the action says so
        return None
    return Question(
        title="Open another project?",
        body=_lines(f"This project has unsaved work: {unsaved_summary(state)}.",
                    f"Opening '{Path(str(path)).name}' replaces all of it."),
        confirm="Discard and open")


def _ask_save_project(_state, path) -> Question | None:
    if not path or not str(path).strip():
        return None
    return _overwrite(project_io.manifest_path(str(path).strip()),
                      "project file")


def _ask_remove_layers(state, selected) -> Question | None:
    from .layers import _expand_selection

    layers = [state.get(i) for i in _expand_selection(state, selected)]
    layers = [layer for layer in layers if layer is not None]
    if not layers:
        raise PreventUpdate
    count = len(layers)
    body = _lines(f"{count} layer{'' if count == 1 else 's'} will be removed: "
                  f"{_names([layer.name for layer in layers])}.",
                  "There is no undo.")
    if any(layer.kind == "raster" for layer in layers):
        body += _lines("A removed raster is also released from the map "
                       "server; a raster that was never saved is gone.")
    if any(layer.kind == "study_area" for layer in layers):
        body += _lines("Removing the study area also stops clipping to it.")
    return Question(title="Remove the selected layers?", body=body,
                    confirm=f"Remove {count} layer{'' if count == 1 else 's'}")


def _ask_delete_group(state, group) -> Question | None:
    members = state.routes_in_group(group) if group else []
    if not members:                       # the action says "pick a group"
        return None
    count = len(members)
    return Question(
        title="Delete this route group?",
        body=_lines(f"Group '{group}' and its {count} "
                    f"route{'' if count == 1 else 's'} will be deleted: "
                    f"{_names([m.name for m in members])}.",
                    "There is no undo."),
        confirm=f"Delete {count} route{'' if count == 1 else 's'}")


def _ask_delete_variant(state) -> Question | None:
    active = state.get(state.active_route_id) if state.active_route_id else None
    if active is None:
        raise PreventUpdate
    return Question(
        title="Discard this route?",
        body=_lines(f"Route '{active.name}' will be deleted.",
                    "Other variants of the same route stay. There is no undo."),
        confirm="Discard route")


def _ask_export_route(state, path) -> Question | None:
    active = state.get(state.active_route_id) if state.active_route_id else None
    if active is None:                    # the action says "nothing to export"
        return None
    return _overwrite(path, "route file")


def _ask_rasterize(_state, path) -> Question | None:
    return _overwrite(path, "raster file")


def _ask_export_costs(_state, path) -> Question | None:
    return _overwrite(path, "cost table file")


def _ask_save_profile(_state, path) -> Question | None:
    return _overwrite(path, "profile file")


GATES: tuple[Gate, ...] = (
    Gate("new_project", ids.PROJECT_NEW_BTN, ids.GO_NEW_PROJECT, (),
         _ask_new_project),
    Gate("open_project", ids.PROJECT_OPEN_BTN, ids.GO_OPEN_PROJECT,
         ((ids.PROJECT_OPEN_PATH, "value"),), _ask_open_project),
    Gate("save_project", ids.PROJECT_SAVE_BTN, ids.GO_SAVE_PROJECT,
         ((ids.PROJECT_SAVE_PATH, "value"),), _ask_save_project),
    Gate("remove_layers", ids.LAYER_REMOVE_BTN, ids.GO_REMOVE_LAYERS,
         ((ids.LAYERS_GRID, "selectedRows"),), _ask_remove_layers),
    Gate("delete_group", ids.GROUP_DELETE_BTN, ids.GO_DELETE_GROUP,
         ((ids.GROUP_SELECT, "value"),), _ask_delete_group),
    Gate("delete_variant", ids.DELETE_VARIANT_BTN, ids.GO_DELETE_VARIANT, (),
         _ask_delete_variant),
    Gate("export_route", ids.EXPORT_ROUTE_BTN, ids.GO_EXPORT_ROUTE,
         ((ids.EXPORT_ROUTE_PATH, "value"),), _ask_export_route),
    Gate("rasterize", ids.RASTERIZE_BTN, ids.GO_RASTERIZE,
         ((ids.RASTER_SAVE_PATH, "value"),), _ask_rasterize),
    Gate("export_costs", ids.COST_EXPORT_BTN, ids.GO_EXPORT_COSTS,
         ((ids.COST_EXPORT_PATH, "value"),), _ask_export_costs),
    Gate("save_profile", ids.OHL_PROFILE_SAVE_BTN, ids.GO_SAVE_PROFILE,
         ((ids.OHL_PROFILE_PATH, "value"),), _ask_save_profile),
)
GATE_BY_ACTION = {gate.action: gate for gate in GATES}
assert tuple(gate.go for gate in GATES) == ids.GO_STORES


def go_signal() -> int:
    """A fresh value for a GO store (the action callbacks only test it)."""
    return next(_SEQUENCE)


def confirm_outputs() -> list[Output]:
    """The six outputs a gate writes (shared, hence ``allow_duplicate``)."""
    return [Output(ids.CONFIRM_MODAL, "is_open", allow_duplicate=True),
            Output(ids.CONFIRM_TITLE, "children", allow_duplicate=True),
            Output(ids.CONFIRM_BODY, "children", allow_duplicate=True),
            Output(ids.CONFIRM_OK, "children", allow_duplicate=True),
            Output(ids.CONFIRM_PENDING, "data", allow_duplicate=True)]


def register(app, state) -> None:
    def make_gate_callback(gate: Gate) -> None:
        @app.callback(
            Output(gate.go, "data", allow_duplicate=True),
            *confirm_outputs(),
            Input(gate.button, "n_clicks"),
            *[State(component, prop) for component, prop in gate.states],
            prevent_initial_call=True)
        def ask(n_clicks, *values):
            if not n_clicks:
                raise PreventUpdate
            question = gate.ask(state, *values)
            if question is None:          # nothing to lose: act at once
                return (go_signal(),) + (no_update,) * 5
            return (no_update, True, question.title, question.body,
                    question.confirm, {"action": gate.action})

    for gate in GATES:
        make_gate_callback(gate)

    @app.callback(
        Output(ids.CONFIRM_MODAL, "is_open", allow_duplicate=True),
        Output(ids.CONFIRM_PENDING, "data", allow_duplicate=True),
        *[Output(go, "data", allow_duplicate=True) for go in ids.GO_STORES],
        Input(ids.CONFIRM_OK, "n_clicks"),
        State(ids.CONFIRM_PENDING, "data"),
        prevent_initial_call=True)
    def confirm(n_clicks, pending):
        """The user said yes: the pending action's GO store fires."""
        action = (pending or {}).get("action")
        gate = GATE_BY_ACTION.get(action)
        if not n_clicks or gate is None:
            raise PreventUpdate
        signals = [go_signal() if go == gate.go else no_update
                   for go in ids.GO_STORES]
        return (False, None, *signals)

    @app.callback(
        Output(ids.CONFIRM_MODAL, "is_open", allow_duplicate=True),
        Output(ids.CONFIRM_PENDING, "data", allow_duplicate=True),
        Input(ids.CONFIRM_CANCEL, "n_clicks"),
        prevent_initial_call=True)
    def cancel(n_clicks):
        """The user said no: close the modal, forget the action, change nothing."""
        if not n_clicks:
            raise PreventUpdate
        return False, None
