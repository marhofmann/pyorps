"""PYORPS GUI callbacks: objective and terrain card (gaps 1, 2 and 6).

P2 (this file, helpers only): what the run and the route edit need from ``state.gui_settings["objective"]``.
``callbacks/routes.run`` reaches the settings through the state (never through new ``State``s) and calls
:func:`prepare_run`; ``callbacks/edit`` calls :func:`stored_run_kwargs` to rebuild the objective a route was
computed with. Every rule lives in ``services/objective``; this module only turns its problems into notices.

P5 (:func:`register`, the helpers after :func:`stored_layers_notice`): the card's callbacks.

* the options of the terrain model and of the layer source follow the loaded raster layers;
* ``apply_preset_click`` writes the three weights of a preset button (``service.apply_preset``);
* ``add_layer`` / ``delete_layers`` / ``refresh_layer_labels`` keep the layers table; a layer is added only after
  ``check_layer_name`` and the service's layer rules, a bad one gives a notice that names the field;
* ``remember_objective_settings`` mirrors every control into ``state.gui_settings["objective"]`` (always through
  ``normalise_objective_settings``) and the ``OBJ_SETTINGS`` store; ``restore_objective_settings`` puts a project's
  settings back after Open and New;
* ``show_status`` writes the summary, or the first problems, into ``OBJ_STATUS`` (the same facts and wording a run
  refuses with), ``update_states`` disables what cannot apply and says why, ``show_slope_fields`` shows the slope
  fields the chosen shape and multiplier read;
* ``reset_multiplier_scale`` and ``clear_slope_settings`` keep a stale value from surviving a model change or a
  removed terrain model;
* a clientside callback opens the closed card (or any closed card around the control) a "Go fix" button
  focuses.

No loops: the controls are the only inputs of the mirror, which writes the store and the state only; the status and
the disabled states read the store; the two reset callbacks and the presets write controls, but read the store as a
``State`` (the previous settings), and a restore writes the store together with the controls, so a restore never
looks like a user change to them.
"""
from __future__ import annotations

from typing import Any

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..sections import objective as form_section
from ..services import objective as objective_service
from ..services.errors import Notice, cap_notices

#: problem key (``services.objective.PROBLEM_KEYS``) -> id of the control that holds it, the target of the
#: "Go fix" button of the notice. ``layers.<i>.<field>`` keys all point at the grid (see :func:`control_for`).
FIELD_CONTROLS: dict[str, str] = {
    "weights": ids.OBJ_W_COST,
    "weights.cost": ids.OBJ_W_COST,
    "weights.length": ids.OBJ_W_LENGTH,
    "weights.gradient": ids.OBJ_W_GRADIENT,
    "layers": ids.OBJ_LAYERS_GRID,
    "dem_layer_id": ids.OBJ_DEM_LAYER,
    "gradient.max_gradient_pct": ids.OBJ_GRAD_MAX_PCT,
    "gradient.additive": ids.OBJ_GRAD_ADDITIVE,
    "gradient.additive_scale": ids.OBJ_GRAD_ADDITIVE_SCALE,
    "gradient.multiplier": ids.OBJ_GRAD_MULTIPLIER,
    "gradient.mult_scale": ids.OBJ_GRAD_MULT_SCALE,
    "gradient.sig_center_deg": ids.OBJ_GRAD_SIG_CENTER,
    "gradient.sig_steepness": ids.OBJ_GRAD_SIG_STEEPNESS,
    "gradient.sig_max_penalty": ids.OBJ_GRAD_SIG_MAX,
    "gradient.bin_width_pct": ids.OBJ_GRAD_BIN_PCT,
    "gradient.s_max_pct": ids.OBJ_GRAD_SMAX_PCT,
    "weight_precision": ids.OBJ_PRECISION,
    "graph_api": ids.GRAPH_API,
    "algorithm": ids.ALGORITHM,
}

#: what to do about a problem whose fix is not "correct the field"
_FIXES: dict[str, str] = {
    "graph_api": "Choose another backend in the Backend field, or take the terrain model (DEM) out of the "
                 "Objective & terrain card.",
    "algorithm": "Choose Dijkstra in the Algorithm field (delta-stepping has no slope terms), or take the "
                 "terrain model (DEM) out of the Objective & terrain card.",
    "dem_layer_id": "Choose a terrain model (DEM) that is loaded, covers every routing point and has the "
                    "coordinate system of the cost raster, or switch the slope settings off.",
    "weight_precision": "Choose 16-bit, or give the objective something to weigh and use a graph-library "
                        "backend or raster_gpu.",
}


def control_for(key: str) -> str | None:
    """The component id of the control behind a problem key (None for a key without one)."""
    if key in FIELD_CONTROLS:
        return FIELD_CONTROLS[key]
    if key.startswith("layers."):
        return ids.OBJ_LAYERS_GRID
    return None


def problem_notice(problem: objective_service.ObjectiveProblem, *, impact: str = "Nothing was run.") -> dict:
    """The warning box of one problem: it names the setting, the value and the allowed range, and its
    "Go fix" button takes the user to the control (``FIELD_CONTROLS``)."""
    # imported here, not at module level: routes imports this module
    from .routes import field_notice

    return field_notice(problem.label, problem.message, control=control_for(problem.key), tab=ids.TAB_ROUTES,
                        impact=impact, fix=_FIXES.get(problem.key))


def read_settings(state) -> dict[str, Any]:
    """The normalised objective settings of the page (classic settings when none were mirrored yet)."""
    gui_settings = getattr(state, "gui_settings", None) or {}
    return objective_service.normalise_objective_settings(gui_settings.get("objective"))


def raster_layer_ids(state) -> set[str]:
    """Ids of the raster layers that are loaded now."""
    return {layer.id for layer in state.layers_of_kind("raster")}


def layer_names(state) -> dict[str, str]:
    """Raster layer id -> its name (for the summary line and the source column of the layer grid)."""
    return {layer.id: layer.name for layer in state.layers_of_kind("raster")}


def layer_path(state, layer_id: str) -> str | None:
    """The file of a loaded raster layer, None when it is not loaded (or not a raster)."""
    layer = state.get(layer_id) if layer_id else None
    if layer is None or layer.kind != "raster":
        return None
    return (layer.meta or {}).get("source_path")


def _layer_crs(state, layer_id: str | None):
    layer = state.get(layer_id) if layer_id else None
    return getattr(layer, "crs", None) if layer is not None else None


def _dem_notice(problem_text: str, *, impact: str) -> dict:
    return problem_notice(objective_service.ObjectiveProblem(
        "dem_layer_id", "Terrain model (DEM)", problem_text), impact=impact)


def _dem_coverage(state, settings: dict, *, raster_bounds, points, buffer_m: float, notices: list, impact: str):
    """``True`` when the DEM may be used; else a notice was appended. A partial cover only warns."""
    # imported here, not at module level: routes imports this module
    from .routes import raster_file_info

    path = layer_path(state, settings["dem_layer_id"])
    try:
        info = raster_file_info(path)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # a DEM file that cannot be read: say so
        notices.append(_dem_notice(f"cannot be read ({type(exc).__name__}: {exc})", impact=impact))
        return False
    problem, warning = objective_service.dem_coverage(
        info["bounds"], points, raster_bounds=raster_bounds, buffer_m=buffer_m)
    if problem is not None:
        notices.append(problem_notice(problem, impact=impact))
        return False
    if warning:
        notices.append(Notice(
            severity="warning", title="The terrain model does not cover the whole search window",
            meaning=warning, impact="The route is computed, but it stays inside the terrain model.",
            fix="Load a larger terrain model, or lower the search buffer.",
            focus_id=ids.TAB_ROUTES, focus_control=ids.OBJ_DEM_LAYER).to_dict())
    return True


def prepare_run(state, *, raster_layer, graph_api: str, algorithm: str, raster_bounds, points: list,
                buffer_m: float, notices: list, impact: str = "Nothing was run."):
    """Everything the run callback needs from the objective card, checked.

    Returns ``(run_kwargs, meta, notices)``:

    - classic settings: ``({}, {}, notices)``: the run is exactly the plain one;
    - a problem: ``(None, {}, notices)`` with the problem's notice appended: the run stops;
    - otherwise ``run_kwargs`` is ``services.objective.run_kwargs`` (merge it into the ``run_routing``
      keywords) and ``meta`` holds ``objective_summary`` (a line for the success notice) and ``stack_bands``
      (for the heavy-run prognosis).

    ``raster_layer`` is the cost raster layer the run uses, ``points`` are all routing points (sources, targets,
    waypoints) in its CRS, ``buffer_m`` the search buffer the run will use and ``raster_bounds`` the cost
    raster's ``(left, bottom, right, top)``.
    """
    settings = read_settings(state)
    if objective_service.is_classic(settings):
        return {}, {}, notices

    def path_of(layer_id: str) -> str | None:
        return layer_path(state, layer_id)

    found = objective_service.problems(
        settings, graph_api=graph_api, algorithm=algorithm, raster_ids=raster_layer_ids(state),
        raster_crs=getattr(raster_layer, "crs", None),
        dem_crs=_layer_crs(state, settings["dem_layer_id"]))
    found = found or objective_service.missing_files(settings, path_of)
    if found:
        notices.append(problem_notice(found[0], impact=impact))
        return None, {}, notices
    if settings["dem_layer_id"] and not _dem_coverage(
            state, settings, raster_bounds=raster_bounds, points=points, buffer_m=buffer_m, notices=notices,
            impact=impact):
        return None, {}, notices
    kwargs = objective_service.run_kwargs(settings, layer_path=path_of, graph_api=graph_api)
    meta = {"objective_summary": objective_service.summary(settings, graph_api, algorithm,
                                                           layer_names=layer_names(state)),
            "stack_bands": objective_service.stack_bands(settings)}
    return kwargs, meta, notices


def stored_run_kwargs(state, params: dict, graph_api: str | None = None):
    """``(run_kwargs, problems)`` to rebuild the objective a route was computed with.

    ``params`` are the route's stored parameters (``objective_spec``). A route without a spec is a plain one:
    ``({}, [])``. When a layer the spec names is no longer loaded (or its file is gone) the result is
    ``(None, problems)``: the edit cannot rebuild the finder (see :func:`stored_layers_notice`).
    """
    spec = (params or {}).get("objective_spec")
    if not spec or objective_service.is_classic(spec):
        return {}, []
    found = objective_service.layer_problems(spec, raster_layer_ids(state))
    found += objective_service.missing_files(spec, lambda layer_id: layer_path(state, layer_id))
    if found:
        return None, found
    kwargs = objective_service.run_kwargs(spec, layer_path=lambda layer_id: layer_path(state, layer_id),
                                          graph_api=graph_api or params.get("graph_api"))
    return kwargs, []


def stored_layers_notice(found: list[objective_service.ObjectiveProblem]) -> dict:
    """The box for an edit that cannot rebuild its objective: the terrain or metric layer is gone."""
    first = found[0]
    return Notice(
        severity="warning", title="The terrain or metric layer this route used is not loaded",
        meaning="; ".join(f"{p.label} {p.message}" for p in found) + ".",
        impact="The edit was not applied: it needs the same layers the route was computed with.",
        fix="Load that layer again, then compute the route again with Run routing.",
        focus_id=ids.TAB_ROUTES, focus_control=control_for(first.key)).to_dict()


# --------------------------------------------------------------------------- what the card shows
#: how many problems the status line lists (the first one is the one a run stops at)
STATUS_PROBLEM_LIMIT = 3
#: label of a dropdown entry for a layer id that is not loaded any more
NOT_LOADED_LABEL = "(layer no longer loaded)"

_N_WEIGHTS = len(form_section.WEIGHT_IDS)
_N_GRADIENT = len(form_section.GRADIENT_IDS)

#: what the clientside "Go fix" helper runs: open every closed ``<details>`` around the control a notice
#: focuses (this card, "Algorithm & search" for the Backend and Algorithm fields, ...) so that the scroll and
#: the flash of the existing handler land on a visible control
OPEN_CARD_JS = """
function(data) {
    if (data && data.control) {
        var node = document.getElementById(data.control);
        while (node) {
            if (node.tagName === 'DETAILS' && !node.open) {
                node.open = true;
            }
            node = node.parentElement;
        }
    }
    return window.dash_clientside.no_update;
}
"""


def raster_options(state) -> list[dict]:
    """The loaded raster layers as dropdown options."""
    return [{"label": layer.name, "value": layer.id} for layer in state.layers_of_kind("raster")]


def dem_options(state, current: str | None = None) -> list[dict]:
    """The terrain model options: the loaded rasters, plus the chosen layer when it is gone (so the dropdown
    still shows what the settings hold and the status line can name the problem)."""
    options = raster_options(state)
    if current and current not in {option["value"] for option in options}:
        options.append({"label": NOT_LOADED_LABEL, "value": current})
    return options


def source_options(state, dem_layer_id: str | None) -> list[dict]:
    """The layer source options: the loaded rasters, and the slope from the DEM once a terrain model is chosen."""
    options = raster_options(state)
    if dem_layer_id:
        options.append({"label": objective_service.DERIVE_SLOPE_LABEL,
                        "value": objective_service.SOURCE_DERIVE_SLOPE})
    return options


def current_problems(state, settings: Any, graph_api: str | None, algorithm: str | None,
                     raster_layer_id: str | None = None) -> list[objective_service.ObjectiveProblem]:
    """What would stop a run with these settings now: the checks ``prepare_run`` makes (the coverage of the
    routing points by the DEM needs the points and stays a run-time check)."""
    s = objective_service.normalise_objective_settings(settings)
    found = objective_service.problems(
        s, graph_api=graph_api, algorithm=algorithm, raster_ids=raster_layer_ids(state),
        raster_crs=_layer_crs(state, raster_layer_id), dem_crs=_layer_crs(state, s["dem_layer_id"]))
    return found or objective_service.missing_files(s, lambda layer_id: layer_path(state, layer_id))


def status_for(state, settings: Any, graph_api: str | None, algorithm: str | None,
               raster_layer_id: str | None = None) -> tuple[str, str]:
    """``(text, className)`` of ``OBJ_STATUS``: the first problems, else the summary of the settings."""
    s = objective_service.normalise_objective_settings(settings)
    found = current_problems(state, s, graph_api, algorithm, raster_layer_id)
    if found:
        lines = [f"{problem.label} {problem.message}." for problem in found[:STATUS_PROBLEM_LIMIT]]
        if len(found) > STATUS_PROBLEM_LIMIT:
            lines.append(f"... and {len(found) - STATUS_PROBLEM_LIMIT} more.")
        return "\n".join(lines), form_section.STATUS_CLASS_PROBLEM
    return (objective_service.summary(s, graph_api, algorithm, layer_names=layer_names(state)),
            form_section.STATUS_CLASS_OK)


def settings_from_form(weights, gradient, dem_layer_id, precision, rows) -> dict[str, Any]:
    """The control values as the normalised settings dict (the only way the state is written)."""
    return objective_service.normalise_objective_settings({
        "weights": dict(zip(objective_service.WEIGHT_KEYS, weights)),
        "layers": form_section.clean_rows(rows),
        "dem_layer_id": dem_layer_id,
        "gradient": dict(zip(form_section.GRADIENT_IDS, gradient)),
        "weight_precision": precision,
    })


def new_layer_problem(state, name: Any, source: Any, weight: Any, dem_layer_id: str | None,
                      existing: list[dict]) -> tuple[str, str, str] | None:
    """``(field, message, control)`` for the first reason a new layer cannot join ``existing``, else None.

    The name is checked first (``check_layer_name``), then the source and the weight by the service's own layer
    rules, applied to the new row only (a stale row that is already in the table does not block a new one).
    """
    error = objective_service.check_layer_name(name, [str(row.get("name") or "") for row in existing])
    if error is not None:
        return "Layer name", error, ids.OBJ_LAYER_NAME
    row = objective_service.layer_row(name, source)
    row["weight"] = weight
    candidate = {"weights": {"cost": 1.0}, "layers": [*existing, row], "dem_layer_id": dem_layer_id}
    found = objective_service.problems(candidate, raster_ids=raster_layer_ids(state))
    prefix = f"layers.{len(existing)}."
    for problem in found:
        if not problem.key.startswith(prefix):
            continue
        field = problem.key[len(prefix):]
        if field == "source" and problem.value == objective_service.SOURCE_DERIVE_SLOPE:
            return "Layer source", problem.message, ids.OBJ_DEM_LAYER
        control = ids.OBJ_LAYER_SOURCE if field == "source" else ids.OBJ_LAYER_WEIGHT
        label = {"source": "Layer source", "weight": objective_service.FIELD_SPECS["layers.weight"].label}.get(
            field, problem.label)
        return label, problem.message, control
    return None


_LAYER_FIXES = {
    ids.OBJ_LAYER_NAME: "Choose a name that starts with a letter, uses only letters, digits and underscores, and "
                        "is not taken by another layer.",
    ids.OBJ_LAYER_SOURCE: "Pick a raster layer in the 'From raster' field (the slope from the terrain model once "
                          "one is chosen).",
    ids.OBJ_LAYER_WEIGHT: "Enter a weight inside the allowed range.",
    ids.OBJ_DEM_LAYER: "Choose a terrain model (DEM) first, then add the slope layer.",
}


def _layer_notice(field: str, message: str, control: str) -> dict:
    # imported here, not at module level: routes imports this module
    from .routes import field_notice

    return field_notice(field, message, control=control, tab=ids.TAB_ROUTES, impact="No layer was added.",
                        fix=_LAYER_FIXES.get(control))


def _triggered_id():
    """The id of the component that fired the running callback (None outside a Dash request)."""
    from dash import ctx

    try:
        return ctx.triggered_id
    except Exception:  # pylint: disable=broad-exception-caught  # no Dash context
        return None


def _cell_edit_triggered() -> bool:
    """Did a cell edit of the layers table fire the running callback?"""
    from dash import ctx

    try:
        return f"{ids.OBJ_LAYERS_GRID}.cellValueChanged" in ctx.triggered_prop_ids
    except Exception:  # pylint: disable=broad-exception-caught  # no Dash context
        return False


# --------------------------------------------------------------------------- registration
def register(app, state) -> None:  # pylint: disable=too-many-statements
    """Register the objective-card callbacks."""
    grid = ids.OBJ_LAYERS_GRID
    copy_targets = [Output(cid, prop, allow_duplicate=True) for cid, prop in form_section.FORM_TARGETS]

    # ------------------------------------------------ options follow the loaded raster layers
    @app.callback(
        Output(ids.OBJ_DEM_LAYER, "options"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.OBJ_DEM_LAYER, "value"))
    def sync_dem_options(_view, current):
        """The loaded rasters; a chosen layer that is gone stays listed (named as not loaded)."""
        return dem_options(state, current)

    @app.callback(
        Output(ids.OBJ_LAYER_SOURCE, "options"),
        Output(ids.OBJ_LAYER_SOURCE, "value"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.OBJ_DEM_LAYER, "value"),
        State(ids.OBJ_LAYER_SOURCE, "value"))
    def sync_source_options(_view, dem_layer_id, current):
        """The loaded rasters, plus the slope from the DEM once one is chosen; a pick that is no longer
        offered is cleared."""
        options = source_options(state, dem_layer_id)
        offered = {option["value"] for option in options}
        return options, (None if current and current not in offered else no_update)

    # ------------------------------------------------ presets
    @app.callback(
        Output(ids.OBJ_W_COST, "value", allow_duplicate=True),
        Output(ids.OBJ_W_LENGTH, "value", allow_duplicate=True),
        Output(ids.OBJ_W_GRADIENT, "value", allow_duplicate=True),
        Input(ids.OBJ_PRESET_CLASSIC_BTN, "n_clicks"),
        Input(ids.OBJ_PRESET_SHORTEST_BTN, "n_clicks"),
        Input(ids.OBJ_PRESET_GENTLEST_BTN, "n_clicks"),
        State(ids.OBJ_SETTINGS, "data"),
        prevent_initial_call=True)
    def apply_preset_click(classic, shortest, gentlest, settings):
        """A preset button writes the three weights; layers, terrain model and slope response stay."""
        name = form_section.PRESET_BUTTONS.get(_triggered_id())
        if name is None or not {"classic": classic, "shortest": shortest, "gentlest": gentlest}[name]:
            raise PreventUpdate
        weights = objective_service.apply_preset(settings, name)["weights"]
        return tuple(weights[key] for key in objective_service.WEIGHT_KEYS)

    # ------------------------------------------------ the layers table
    @app.callback(
        Output(grid, "rowData", allow_duplicate=True),
        Output(ids.OBJ_LAYER_NAME, "value", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.OBJ_LAYER_ADD_BTN, "n_clicks"),
        State(ids.OBJ_LAYER_NAME, "value"),
        State(ids.OBJ_LAYER_SOURCE, "value"),
        State(ids.OBJ_LAYER_WEIGHT, "value"),
        State(ids.OBJ_DEM_LAYER, "value"),
        State(grid, "rowData"),
        State(grid, "virtualRowData"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def add_layer(n_clicks, name, source, weight, dem_layer_id, rows, virtual, notices):
        """Add the layer of the form to the table, or say in a notice which field is not usable."""
        if not n_clicks:
            raise PreventUpdate
        live = form_section.live_rows(rows, virtual)
        problem = new_layer_problem(state, name, source, weight, dem_layer_id, live)
        if problem is not None:
            return no_update, no_update, cap_notices([*(notices or []), _layer_notice(*problem)])
        layer = objective_service.layer_row(name, source, weight)
        row = form_section.layer_rows([layer], layer_names(state))[0]
        return [*live, row], "", no_update

    @app.callback(
        Output(grid, "rowData", allow_duplicate=True),
        Input(ids.OBJ_LAYER_DEL_BTN, "n_clicks"),
        State(grid, "rowData"),
        State(grid, "virtualRowData"),
        State(grid, "selectedRows"),
        prevent_initial_call=True)
    def delete_layers(n_clicks, rows, virtual, selected):
        """Remove the selected rows (the rows are read as the grid holds them: a cell edit is kept)."""
        dropped = {row.get("id") for row in selected or [] if isinstance(row, dict)}
        if not n_clicks or not dropped:
            raise PreventUpdate
        return [row for row in form_section.live_rows(rows, virtual) if row.get("id") not in dropped]

    @app.callback(
        Output(grid, "rowData", allow_duplicate=True),
        Input(ids.LAYERS_VIEW, "data"),
        State(grid, "rowData"),
        State(grid, "virtualRowData"),
        prevent_initial_call=True)
    def refresh_layer_labels(_view, rows, virtual):
        """A layer that was renamed or removed changes the source label of the rows that use it."""
        live = form_section.live_rows(rows, virtual)
        names = layer_names(state)
        updated = [{**row, "source_label": objective_service.source_label(row.get("source"), names)}
                   for row in live]
        if updated == live:
            raise PreventUpdate
        return updated

    # ------------------------------------------------ the settings, kept on the server for saving
    @app.callback(
        Output(ids.OBJ_SETTINGS, "data"),
        *[Input(cid, "value") for cid in form_section.WEIGHT_IDS.values()],
        *[Input(cid, "value") for cid in form_section.GRADIENT_IDS.values()],
        Input(ids.OBJ_DEM_LAYER, "value"),
        Input(ids.OBJ_PRECISION, "value"),
        Input(grid, "rowData"),
        Input(grid, "cellValueChanged"),
        State(grid, "virtualRowData"))
    def remember_objective_settings(*args):
        """Mirror every control into ``state.gui_settings["objective"]``: the project file saves it from there
        and a run reads it from there. A blank value takes its default and an unusable one is kept as it is and
        named by the status line; nothing is clamped here."""
        weights = args[:_N_WEIGHTS]
        gradient = args[_N_WEIGHTS:_N_WEIGHTS + _N_GRADIENT]
        dem_layer_id, precision, row_data, _edit, virtual = args[_N_WEIGHTS + _N_GRADIENT:]
        rows = virtual if _cell_edit_triggered() and virtual is not None else row_data
        settings = settings_from_form(weights, gradient, dem_layer_id, precision, rows)
        state.gui_settings["objective"] = settings
        return settings

    # ------------------------------------------------ Open / New project
    @app.callback(
        *copy_targets,
        Output(ids.OBJ_LAYER_NAME, "value", allow_duplicate=True),
        Output(ids.OBJ_LAYER_SOURCE, "value", allow_duplicate=True),
        Output(ids.OBJ_LAYER_WEIGHT, "value", allow_duplicate=True),
        Input(ids.PROJECT_STATUS, "children"),
        prevent_initial_call=True)
    def restore_objective_settings(project_status):
        """The saved settings come back after Open; New (and a project without them) puts the defaults back."""
        if str(project_status or "").lower().startswith("saved"):
            raise PreventUpdate                  # a save changes nothing the page does not hold
        return (*form_section.form_props(read_settings(state), layer_names(state)), "", None,
                objective_service.FIELD_SPECS["layers.weight"].default)

    # ------------------------------------------------ status line, disabled states, shown fields
    @app.callback(
        Output(ids.OBJ_STATUS, "children"),
        Output(ids.OBJ_STATUS, "className"),
        Input(ids.OBJ_SETTINGS, "data"),
        Input(ids.GRAPH_API, "value"),
        Input(ids.ALGORITHM, "value"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.ROUTE_RASTER, "value"))
    def show_status(settings, graph_api, algorithm, _view, raster_layer_id):
        """The summary of the settings, or the first problems: a DEM with delta-stepping or with a backend
        without slope terms is named here, before the run, in the words the run uses."""
        return status_for(state, settings, graph_api, algorithm, raster_layer_id)

    @app.callback(
        Output(ids.OBJ_W_GRADIENT_WRAP, "disabled"),
        Output(ids.OBJ_W_GRADIENT_WRAP, "title"),
        Output(ids.OBJ_SLOPE_WRAP, "disabled"),
        Output(ids.OBJ_SLOPE_WRAP, "title"),
        Output(form_section.DEM_WRAP, "title"),
        Output(ids.OBJ_PRECISION_WRAP, "title"),
        Output(ids.OBJ_PRECISION, "options"),
        Input(ids.OBJ_SETTINGS, "data"),
        Input(ids.GRAPH_API, "value"),
        Input(ids.ALGORITHM, "value"))
    def update_states(settings, graph_api, algorithm):
        """Disable what cannot apply and say why: the slope weight and response need a terrain model, the 32-bit
        choice needs an objective and a backend with float weights."""
        states = form_section.control_states(settings, graph_api, algorithm)
        return (states["weight_disabled"], states["weight_title"], states["slope_disabled"],
                states["slope_title"], states["dem_title"], states["precision_title"],
                states["precision_options"])

    @app.callback(
        Output(form_section.ADDITIVE_SCALE_WRAP, "style"),
        Output(form_section.MULT_SCALE_WRAP, "style"),
        Output(form_section.SIGMOID_WRAP, "style"),
        Output(ids.OBJ_GRAD_MULT_SCALE, "placeholder"),
        Input(ids.OBJ_GRAD_ADDITIVE, "value"),
        Input(ids.OBJ_GRAD_MULTIPLIER, "value"))
    def show_slope_fields(additive, multiplier):
        """Show only the slope fields the chosen shape and multiplier read (``applicable_gradient_keys``)."""
        scale = objective_service.mult_scale_default(None if multiplier == "none" else multiplier)
        placeholder = "model default" if scale is None else f"model default: {objective_service.format_number(scale)}"
        return (*form_section.shown_blocks(additive, multiplier), placeholder)

    # ------------------------------------------------ a model change must not leave a stale value behind
    @app.callback(
        Output(ids.OBJ_GRAD_MULT_SCALE, "value", allow_duplicate=True),
        Input(ids.OBJ_GRAD_MULTIPLIER, "value"),
        State(ids.OBJ_SETTINGS, "data"),
        prevent_initial_call=True)
    def reset_multiplier_scale(multiplier, settings):
        """Another multiplier model gets its own default scale (the scale of the old model means nothing to it).
        A restore writes the store with the controls, so the model it brings back is not a change here."""
        model = None if multiplier in (None, "none") else multiplier
        if model == objective_service.normalise_objective_settings(settings)["gradient"]["multiplier"]:
            raise PreventUpdate
        return objective_service.mult_scale_default(model)

    @app.callback(
        Output(ids.OBJ_W_GRADIENT, "value", allow_duplicate=True),
        Output(ids.OBJ_GRAD_MAX_PCT, "value", allow_duplicate=True),
        Output(ids.OBJ_GRAD_ADDITIVE, "value", allow_duplicate=True),
        Output(ids.OBJ_GRAD_MULTIPLIER, "value", allow_duplicate=True),
        Input(ids.OBJ_DEM_LAYER, "value"),
        State(ids.OBJ_SETTINGS, "data"),
        prevent_initial_call=True)
    def clear_slope_settings(dem_layer_id, settings):
        """Taking the terrain model away switches the slope weight, the slope limit and the two slope lists
        off: their controls are disabled without a DEM, so a value left in them could never be changed back."""
        held = objective_service.normalise_objective_settings(settings)["dem_layer_id"]
        if dem_layer_id or not held:
            raise PreventUpdate
        return 0.0, None, "none", "none"

    # ------------------------------------------------ a "Go fix" button must find its control
    # The output is the card's own ``open`` flag (never written: the script opens the elements itself and
    # returns no_update). It must not be ``FOCUS_FLASH.data``: the scroll-and-flash handler of the notices
    # already owns that (output, input) pair, and two callbacks with the same pair are a duplicate output.
    app.clientside_callback(
        OPEN_CARD_JS,
        Output(ids.OBJ_CARD, "open", allow_duplicate=True),
        Input(ids.FOCUS_FLASH, "data"),
        prevent_initial_call=True)
