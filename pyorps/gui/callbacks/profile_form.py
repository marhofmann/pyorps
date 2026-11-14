"""PYORPS GUI callbacks: the constrained-profile form (gap 5, package P6).

The form and the YAML text stay in step through the profile service (``services/profile_form``):

* ``fill_form``: a shipped profile, a loaded profile file or the "YAML -> form" button fills the form
  (fields, tables, base store, status lines).
* ``form_to_yaml``: every form edit is merged into the YAML text (``form_to_text``), so the run, which
  reads that text, always uses what the form shows. Keys the form does not show survive.
* ``restore_form``: "Open project" and "New project" put the saved profile back (or the defaults).
* ``add_row`` / ``delete_rows`` per table, ``show_turn_blocks`` (turn cost factor for linear and quadratic,
  turn table for piecewise) and ``remember_constrained_settings`` (the page's constrained settings are kept
  in ``state.gui_settings["constrained"]``, which the project file saves).
* The "Objective weights" block (round 16): ``remember_constrained_objective`` mirrors the two weights into
  ``state.gui_settings["constrained"]["objective"]`` (weights at their defaults leave no key) and the
  ``OHLO_SETTINGS`` store; ``remember_constrained_settings`` carries that key over when it rewrites the dict.
  ``show_objective_status`` writes the summary or the first problems and the disabled state (backend, clearance
  check, DEM and profile are its inputs), ``reset_constrained_objective`` is the "Reset weights" button and
  ``restore_constrained_objective`` puts the saved weights back after Open and New. The run reads the weights
  from the state (``callbacks/constrained.objective_settings``), never from a new ``State``.
* The "Line ends" group (round 17), the same shape: ``remember_constrained_ends`` mirrors the two connection
  heights into ``state.gui_settings["constrained"]["ends"]`` (no height set leaves no key) and the ``OHLE_SETTINGS``
  store, ``remember_constrained_settings`` carries the key over, ``show_ends_status`` writes the summary or the first
  problems and the disabled state of the heights (backend, terrain model, profile and heights are its inputs),
  ``show_rule_status`` the line under the end-span rule select and its disabled state, ``clear_constrained_ends``
  is the "Clear heights" button and ``restore_constrained_ends`` puts the saved heights back after Open and New.
  The rule itself is a profile field (the 24th form key) and travels with the profile. The run reads the heights
  from the state (``callbacks/constrained.ends_settings``), never from a new ``State``.

No loops: the YAML text is only a ``State`` of the form callbacks (never an ``Input``), and a form edit
that changes nothing the base store does not already show writes nothing, so a fill does not write the
text back.
"""
from __future__ import annotations

from pathlib import Path

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from pyorps.core.exceptions import InvalidInputError, MissingDependencyError

from .. import ids
from ..sections import profile_form as form_section
from ..services import constrained, project_io
from ..services import constrained_ends as ends_service
from ..services import constrained_objective as objective_service
from ..services import profile_form as service
from ..services.errors import Notice, cap_notices, translate_exception
from .routes import _triggered_id

DEFAULT_BACKEND = "cython"

_N_FIELDS = len(service.FORM_KEYS)
_COLUMN_KEYS = tuple(form_section.TABLE_COLUMN_KEYS.values())
_HIDDEN = {"display": "none"}
_WEIGHT_KEYS = objective_service.WEIGHT_KEYS
_HEIGHT_KEYS = ends_service.HEIGHT_KEYS


# --------------------------------------------------------------------------- notices
def _yaml_notice(exc: Exception) -> dict:
    """The box for text the form cannot read or write: no PyYAML, or text that is no YAML mapping."""
    if isinstance(exc, MissingDependencyError):
        return Notice(
            severity="warning", title="PyYAML is not installed", meaning=str(exc),
            impact="The form cannot read or write the YAML profile text. A run reads that text, "
                   "so form edits do not reach it.",
            fix="Install PyYAML (pip install \"pyorps[yaml]\") and restart the GUI.",
            focus_id=ids.TAB_ROUTES, focus_control=ids.OHLF_YAML_DETAILS).to_dict()
    return Notice(
        severity="warning", title="The profile text could not be read", meaning=str(exc),
        impact="The form was not changed.",
        fix="Correct the text in Raw YAML (advanced), then press \"YAML → form\" again.",
        focus_id=ids.TAB_ROUTES, focus_control=ids.OHLF_YAML_DETAILS).to_dict()


def _read(read, notices: list) -> tuple[dict | None, list]:
    """``read()`` as ``(profile, notices)``; a failure becomes a notice and the profile is None."""
    try:
        return read(), notices
    except (InvalidInputError, MissingDependencyError) as exc:
        notices.append(_yaml_notice(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught  # every failure ends in a notice box, never a traceback
        notices.append(translate_exception(exc).to_dict())
    return None, cap_notices(notices)


def _load_file(path) -> dict | None:
    """A profile file as a dict, or None: the card's own "Load" callback explains a failure."""
    try:
        return constrained.load_profile_dict(path)
    except Exception:  # pylint: disable=broad-exception-caught  # load_profile_file reports it
        return None


# --------------------------------------------------------------------------- settings
def _saved_settings(state) -> dict:
    settings = (getattr(state, "gui_settings", None) or {}).get("constrained")
    return settings if isinstance(settings, dict) else {}


def _store_objective(state, weights: dict) -> None:
    """Write the objective weights into ``state.gui_settings["constrained"]``; defaults remove the key.

    The other keys of the constrained settings stay as they are (``remember_constrained_settings`` owns
    them, and carries the weights over when it rewrites the dict).
    """
    held = state.gui_settings.get("constrained")
    if objective_service.is_default(weights):
        if isinstance(held, dict):
            held.pop("objective", None)
        return
    saved = project_io.normalise_gui_settings({"constrained": {"objective": weights}})["constrained"]["objective"]
    if isinstance(held, dict):
        held["objective"] = saved
    else:
        state.gui_settings["constrained"] = {"objective": saved}


def _store_ends(state, heights: dict) -> None:
    """Write the connection heights into ``state.gui_settings["constrained"]``; no height set removes the key.

    The other keys of the constrained settings stay as they are (``remember_constrained_settings`` owns them, and
    carries the heights over when it rewrites the dict).
    """
    held = state.gui_settings.get("constrained")
    if ends_service.is_default(heights):
        if isinstance(held, dict):
            held.pop("ends", None)
        return
    saved = project_io.normalise_gui_settings({"constrained": {"ends": heights}})["constrained"]["ends"]
    if isinstance(held, dict):
        held["ends"] = saved
    else:
        state.gui_settings["constrained"] = {"ends": saved}


def _profile_to_remember(text, previous):
    """The profile the settings keep: the text when it parses; blank text means none.

    Text that does not parse (the user is typing) or cannot be read at all (no PyYAML) keeps the
    last saved profile.
    """
    try:
        return service.parse_profile_text(text)
    except MissingDependencyError:
        return previous
    except InvalidInputError:
        return None if not str(text or "").strip() else previous


def _edited_grids() -> set[str]:
    """The grids whose cell edit triggered the running callback (none outside a Dash request)."""
    from dash import ctx

    try:
        return {prop_id.rpartition(".")[0] for prop_id in ctx.triggered_prop_ids
                if prop_id.endswith(".cellValueChanged")}
    except Exception:  # pylint: disable=broad-exception-caught  # no Dash context
        return set()


def _text_of(profile) -> str:
    return service.profile_to_text(profile) if profile else ""


def _same_profile(text, profile) -> bool:
    """Does the YAML text already hold ``profile`` (blank text and no profile count as equal)?"""
    if not str(text or "").strip():
        return not profile
    try:
        parsed = service.parse_profile_text(text)
    except (InvalidInputError, MissingDependencyError):
        return False
    # the saved profile went through JSON (a project file does that): compare like with like
    return project_io.normalise_gui_settings({"profile": parsed}).get("profile") == (profile or {})


# --------------------------------------------------------------------------- registration
def register(app, state) -> None:  # pylint: disable=too-many-statements
    """Register the profile-form callbacks."""
    fill_targets = [Output(cid, prop) for cid, prop in form_section.FORM_TARGETS]
    copy_targets = [Output(cid, prop, allow_duplicate=True) for cid, prop in form_section.FORM_TARGETS]

    # ------------------------------------------------ fill the form from a profile
    @app.callback(
        *fill_targets,
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.OHL_PROFILE_PRESET, "value"),
        Input(ids.OHL_PROFILE_LOAD_BTN, "n_clicks"),
        Input(ids.OHLF_FROM_YAML_BTN, "n_clicks"),
        State(ids.OHL_PROFILE_PATH, "value"),
        State(ids.OHL_PROFILE_TEXT, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def fill_form(preset, load_clicks, yaml_clicks, path, text, notices):
        """A shipped profile, a profile file or the YAML text fills the form (the text itself is
        written by the existing preset and load callbacks, or is the source)."""
        notices = list(notices or [])
        trigger = _triggered_id()
        if trigger == ids.OHL_PROFILE_PRESET and preset:
            profile, notices = _read(
                lambda: service.parse_profile_text(Path(preset).read_text(encoding="utf-8")), notices)
        elif trigger == ids.OHL_PROFILE_LOAD_BTN and load_clicks:
            profile = _load_file(path)
            if profile is None:
                raise PreventUpdate
        elif trigger == ids.OHLF_FROM_YAML_BTN and yaml_clicks:
            profile, notices = _read(lambda: service.parse_profile_text(text), notices)
        else:
            raise PreventUpdate
        if profile is None:
            return (*([no_update] * len(form_section.FORM_TARGETS)), notices)
        problems = service.profile_problems(profile)
        if trigger == ids.OHLF_FROM_YAML_BTN:       # the user asked for this: say what is wrong
            notice = service.problem_notice(
                problems, controls=form_section.CONTROLS, tab=ids.TAB_ROUTES,
                impact="The form shows the text as it is; fix it before the next constrained run.")
            if notice is not None:
                notices.append(notice)
        return (*form_section.form_props(profile, problems), cap_notices(notices))

    # ------------------------------------------------ form edit -> YAML text
    @app.callback(
        Output(ids.OHL_PROFILE_TEXT, "value", allow_duplicate=True),
        Output(ids.OHLF_STATUS, "children", allow_duplicate=True),
        Output(ids.OHLF_STATUS, "className", allow_duplicate=True),
        Output(ids.OHLF_CLEARANCE_INFO, "children", allow_duplicate=True),
        Output(ids.OHLF_BASE, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        *[Input(ids.OHLF_IDS[key], "value") for key in service.FORM_KEYS],
        *[Input(grid, "rowData") for grid in form_section.GRID_IDS],
        *[Input(grid, "cellValueChanged") for grid in form_section.GRID_IDS],
        State(ids.OHL_PROFILE_TEXT, "value"),
        State(ids.OHLF_BASE, "data"),
        State(ids.OHLF_STATUS, "children"),
        State(ids.NOTICES, "data"),
        *[State(grid, "virtualRowData") for grid in form_section.GRID_IDS],
        prevent_initial_call=True)
    def form_to_yaml(*args):
        """Merge the form into the YAML text; show the problems; keep the base store current."""
        values = dict(zip(service.FORM_KEYS, args[:_N_FIELDS]))
        row_data = args[_N_FIELDS:_N_FIELDS + 3]
        text, base, status, notices = args[_N_FIELDS + 6:_N_FIELDS + 10]
        live_rows = args[_N_FIELDS + 10:_N_FIELDS + 13]
        edited = _edited_grids()
        # a cell edit: the grid's own rows are current at once; its ``rowData`` follows later
        angle, terrain, types = (
            form_section.clean_rows(live if grid in edited and live is not None else rows, columns)
            for grid, rows, live, columns in zip(form_section.GRID_IDS, row_data, live_rows, _COLUMN_KEYS))
        base = dict(base) if isinstance(base, dict) else {}
        notices = list(notices or [])

        shown_values, *shown_rows = service.form_values(base)
        shown_rows = [form_section.clean_rows(rows, columns)
                      for rows, columns in zip(shown_rows, _COLUMN_KEYS)]
        if values == shown_values and [angle, terrain, types] == shown_rows:
            raise PreventUpdate                  # nothing the user changed: a fill, a restore

        new_text = no_update
        before = len(notices)
        try:
            merge_base = service.base_profile(text, base)
            problems = service.form_problems(values, angle, terrain, types, merge_base)
            yaml_text, profile, _ = service.form_to_text(
                values, angle, terrain, types, base_text=text, fallback=base)
            if yaml_text != text:
                new_text = yaml_text
        except MissingDependencyError as exc:    # the form still checks its values, the text stays
            profile = service.profile_from_form(values, angle, terrain, types, base)
            problems = service.form_problems(values, angle, terrain, types, base)
            notice = _yaml_notice(exc)
            if notice["title"] not in [n.get("title") for n in notices[-5:]]:   # once, not on every edit
                notices.append(notice)

        # a row that is half filled is shown in the status line only: the user is still typing it
        row_problems = service.table_row_problems(angle, terrain, types)
        field_problems = problems[:len(problems) - len(row_problems)]
        previous_first = (str(status or "").splitlines() or [""])[0]
        if field_problems and field_problems[0][1] != previous_first:
            notice = service.problem_notice(
                field_problems, controls=form_section.CONTROLS, tab=ids.TAB_ROUTES,
                impact="The profile text was updated with this value; fix it before the next "
                       "constrained run.")
            if notice is not None:
                notices.append(notice)

        status_text, css = form_section.status_for(problems, profile)
        return (new_text, status_text, css, service.clearance_status(profile), profile,
                cap_notices(notices) if len(notices) > before else no_update)

    # ------------------------------------------------ Open / New project
    @app.callback(
        *copy_targets,
        Output(ids.OHL_PROFILE_TEXT, "value", allow_duplicate=True),
        Output(ids.OHL_BACKEND, "value"),
        Output(ids.OHL_EXPERIMENTAL, "value"),
        Output(ids.OHL_DEM, "value"),
        Output(ids.OHL_DSM, "value"),
        Input(ids.PROJECT_STATUS, "children"),
        State(ids.OHL_PROFILE_TEXT, "value"),
        State(ids.OHL_BACKEND, "value"),
        State(ids.OHL_EXPERIMENTAL, "value"),
        State(ids.OHL_DEM, "value"),
        State(ids.OHL_DSM, "value"),
        prevent_initial_call=True)
    def restore_form(project_status, text, backend, experimental, dem, dsm):
        """The saved profile and settings come back after Open; New puts the defaults back."""
        if str(project_status or "").lower().startswith("saved"):
            raise PreventUpdate                  # a save changes nothing the page does not hold
        settings = _saved_settings(state)
        profile = settings.get("profile") if isinstance(settings.get("profile"), dict) else {}
        saved = (settings.get("backend") or DEFAULT_BACKEND, bool(settings.get("experimental")),
                 settings.get("dem") or "", settings.get("dsm") or "")
        if _same_profile(text, profile) and saved == (
                backend or DEFAULT_BACKEND, bool(experimental), dem or "", dsm or ""):
            raise PreventUpdate                  # already what the project holds (keeps YAML comments)
        try:
            yaml_text = _text_of(profile)
        except MissingDependencyError:
            yaml_text = no_update                # without PyYAML the text is left alone
        return (*form_section.form_props(profile), yaml_text, saved[0], saved[1], saved[2], saved[3])

    # ------------------------------------------------ table rows
    for table_key, (grid_id, add_id, delete_id) in form_section.TABLE_CONTROLS.items():
        _register_table_buttons(app, grid_id, add_id, delete_id,
                                form_section.TABLE_COLUMN_KEYS[table_key])

    # ------------------------------------------------ turn cost factor or turn table
    @app.callback(
        Output(form_section.ANGLE_SCALE_WRAP, "style"),
        Output(form_section.ANGLE_TABLE_WRAP, "style"),
        Input(ids.OHLF_IDS["angle_cost_function"], "value"))
    def show_turn_blocks(function):
        """Piecewise turn cost uses the table; linear and quadratic use the factor."""
        piecewise = function not in ("linear", "quadratic")
        return (_HIDDEN, {}) if piecewise else ({}, _HIDDEN)

    # ------------------------------------------------ the page's constrained settings, kept on the server
    @app.callback(
        Output(ids.OHL_SETTINGS, "data"),
        Input(ids.OHL_BACKEND, "value"),
        Input(ids.OHL_EXPERIMENTAL, "value"),
        Input(ids.OHL_DEM, "value"),
        Input(ids.OHL_DSM, "value"),
        Input(ids.OHL_PROFILE_TEXT, "value"))
    def remember_constrained_settings(backend, experimental, dem, dsm, text):
        """Mirror backend, experimental flag, DEM, DSM and the profile into
        ``state.gui_settings["constrained"]``: the project file saves them from there."""
        previous = _saved_settings(state).get("profile")
        objective = _saved_settings(state).get("objective")       # kept by remember_constrained_objective
        ends = _saved_settings(state).get("ends")                 # kept by remember_constrained_ends
        settings = {"backend": backend or DEFAULT_BACKEND, "experimental": bool(experimental),
                    "dem": dem or None, "dsm": dsm or None}
        profile = _profile_to_remember(text, previous)
        if profile:
            settings["profile"] = profile
        if isinstance(objective, dict) and objective:
            settings["objective"] = objective
        if isinstance(ends, dict) and ends:
            settings["ends"] = ends
        settings = project_io.normalise_gui_settings({"constrained": settings}).get("constrained", {})
        state.gui_settings["constrained"] = settings
        return settings

    # ------------------------------------------------ the objective weights (round 16)
    @app.callback(
        Output(ids.OHLO_SETTINGS, "data"),
        *[Input(ids.OHLO_IDS[key], "value") for key in _WEIGHT_KEYS])
    def remember_constrained_objective(*values):
        """Mirror the weights into ``state.gui_settings["constrained"]["objective"]`` (the project file saves
        them from there) and into the ``OHLO_SETTINGS`` store. Weights at their defaults leave no key behind, so
        a project that never used them is saved exactly as before."""
        settings = objective_service.normalise(dict(zip(_WEIGHT_KEYS, values)))
        _store_objective(state, settings)
        return settings

    @app.callback(
        Output(ids.OHLO_STATUS, "children"),
        Output(ids.OHLO_STATUS, "className"),
        Output(ids.OHLO_FIELDS, "disabled"),
        Output(ids.OHLO_FIELDS, "title"),
        Input(ids.OHLO_SETTINGS, "data"),
        Input(ids.OHL_BACKEND, "value"),
        Input(ids.OHL_EXPERIMENTAL, "value"),
        Input(ids.OHL_DEM, "value"),
        Input(ids.OHLF_BASE, "data"))
    def show_objective_status(weights, backend, experimental, dem, base):
        """The summary of the weights, or the first problems (the words a run refuses with); the number inputs
        are disabled, with the reason, while the backend or the clearance check of the profile cannot take them."""
        view = form_section.objective_view(weights, backend=backend, experimental=experimental, dem=dem,
                                           profile=base)
        text, css = form_section.objective_status_props(view)
        return text, css, not view.enabled, view.reason or None

    @app.callback(
        *[Output(ids.OHLO_IDS[key], "value") for key in _WEIGHT_KEYS],
        Input(ids.OHLO_RESET_BTN, "n_clicks"),
        prevent_initial_call=True)
    def reset_constrained_objective(n_clicks):
        """"Reset weights": every weight back to its default (also the way out while the inputs are disabled)."""
        if not n_clicks:
            raise PreventUpdate
        return tuple(objective_service.DEFAULTS[key] for key in _WEIGHT_KEYS)

    @app.callback(
        *[Output(ids.OHLO_IDS[key], "value", allow_duplicate=True) for key in _WEIGHT_KEYS],
        Output(ids.OHLO_BLOCK, "open"),
        Input(ids.PROJECT_STATUS, "children"),
        *[State(ids.OHLO_IDS[key], "value") for key in _WEIGHT_KEYS],
        prevent_initial_call=True)
    def restore_constrained_objective(project_status, *current):
        """The saved weights come back after Open; New puts the defaults back. The mirror then follows the
        controls, so the store is not written here."""
        if str(project_status or "").lower().startswith("saved"):
            raise PreventUpdate                  # a save changes nothing the page does not hold
        saved = objective_service.normalise(_saved_settings(state).get("objective"))
        if objective_service.normalise(dict(zip(_WEIGHT_KEYS, current))) == saved:
            raise PreventUpdate                  # already what the project holds
        shown = objective_service.input_values(saved)
        return (*[shown[key] for key in _WEIGHT_KEYS], not objective_service.is_default(saved))

    # ------------------------------------------------ the line ends (round 17)
    @app.callback(
        Output(ids.OHLE_SETTINGS, "data"),
        *[Input(ids.OHLE_IDS[key], "value") for key in _HEIGHT_KEYS])
    def remember_constrained_ends(*values):
        """Mirror the connection heights into ``state.gui_settings["constrained"]["ends"]`` (the project file saves
        them from there) and into the ``OHLE_SETTINGS`` store. No height set leaves no key behind, so a project that
        never used them is saved exactly as before."""
        settings = ends_service.normalise(dict(zip(_HEIGHT_KEYS, values)))
        _store_ends(state, settings)
        return settings

    @app.callback(
        Output(ids.OHLE_STATUS, "children"),
        Output(ids.OHLE_STATUS, "className"),
        Output(ids.OHLE_FIELDS, "disabled"),
        Output(ids.OHLE_FIELDS, "title"),
        Input(ids.OHLE_SETTINGS, "data"),
        Input(ids.OHL_BACKEND, "value"),
        Input(ids.OHL_EXPERIMENTAL, "value"),
        Input(ids.OHL_DEM, "value"),
        Input(ids.OHLF_BASE, "data"))
    def show_ends_status(heights, backend, experimental, dem, base):
        """The summary of the heights, or the first problems (the words a run refuses with), then the notes (a
        height that cannot act is "reported only"); the number inputs are disabled, with the reason, while the
        profile has no span limits or the backend is a GPU one."""
        view = form_section.ends_view(heights, backend=backend, experimental=experimental, dem=dem, profile=base)
        text, css = form_section.ends_status_props(view)
        return text, css, not view.enabled, view.reason or None

    @app.callback(
        Output(ids.OHLE_RULE_STATUS, "children"),
        Output(ids.OHLE_RULE_STATUS, "className"),
        Output(ids.OHLF_IDS["terminal_span_rule"], "disabled"),
        Input(ids.OHLF_IDS["terminal_span_rule"], "value"),
        Input(ids.OHL_BACKEND, "value"),
        Input(ids.OHL_EXPERIMENTAL, "value"),
        Input(ids.OHLF_BASE, "data"))
    def show_rule_status(rule, backend, experimental, base):
        """The line under the end-span rule: what the chosen rule does, why the select is disabled (the profile
        has no span limits), or that a GPU backend refuses "Never place one"."""
        view = form_section.rule_view(rule, backend=backend, experimental=experimental, profile=base)
        return view.text, view.css, view.disabled

    @app.callback(
        *[Output(ids.OHLE_IDS[key], "value") for key in _HEIGHT_KEYS],
        Input(ids.OHLE_CLEAR_BTN, "n_clicks"),
        prevent_initial_call=True)
    def clear_constrained_ends(n_clicks):
        """"Clear heights": both heights back to blank (also the way out while the inputs are disabled)."""
        if not n_clicks:
            raise PreventUpdate
        return tuple(ends_service.DEFAULTS[key] for key in _HEIGHT_KEYS)

    @app.callback(
        *[Output(ids.OHLE_IDS[key], "value", allow_duplicate=True) for key in _HEIGHT_KEYS],
        Output(ids.OHLE_BLOCK, "open"),
        Input(ids.PROJECT_STATUS, "children"),
        *[State(ids.OHLE_IDS[key], "value") for key in _HEIGHT_KEYS],
        prevent_initial_call=True)
    def restore_constrained_ends(project_status, *current):
        """The saved heights come back after Open; New puts them back to blank. The mirror then follows the
        controls, so the store is not written here. The group opens when a height is saved or the saved profile
        names an end-span rule other than the default (the rule select itself is restored with the profile)."""
        if str(project_status or "").lower().startswith("saved"):
            raise PreventUpdate                  # a save changes nothing the page does not hold
        saved = ends_service.normalise(_saved_settings(state).get("ends"))
        if ends_service.normalise(dict(zip(_HEIGHT_KEYS, current))) == saved:
            raise PreventUpdate                  # already what the project holds
        shown = ends_service.input_values(saved)
        profile = _saved_settings(state).get("profile")
        rule = profile.get("terminal_span_rule") if isinstance(profile, dict) else None
        rule_set = isinstance(rule, str) and rule != ends_service.DEFAULT_RULE
        return (*[shown[key] for key in _HEIGHT_KEYS], (not ends_service.is_default(saved)) or rule_set)


def _register_table_buttons(app, grid_id: str, add_id: str, delete_id: str,
                            columns: tuple[str, ...]) -> None:
    """"+ row" and "- selected" of one table."""

    @app.callback(
        Output(grid_id, "rowData", allow_duplicate=True),
        Input(add_id, "n_clicks"),
        State(grid_id, "rowData"),
        prevent_initial_call=True)
    def add_row(n_clicks, rows):
        if not n_clicks:
            raise PreventUpdate
        return [*(rows or []), form_section.blank_row(columns)]

    @app.callback(
        Output(grid_id, "rowData", allow_duplicate=True),
        Input(delete_id, "n_clicks"),
        State(grid_id, "rowData"),
        State(grid_id, "selectedRows"),
        prevent_initial_call=True)
    def delete_rows(n_clicks, rows, selected):
        dropped = {row.get("id") for row in selected or [] if isinstance(row, dict)}
        if not n_clicks or not dropped:
            raise PreventUpdate
        return [row for row in rows or [] if row.get("id") not in dropped]
