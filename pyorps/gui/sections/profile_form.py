"""PYORPS GUI section: constrained-profile form (gap 5, package P6).

The constrained-routing profile as a form in the "Constrained routing" options of the Routes tab. Every
label, unit, limit, option list, help text and table column comes from the specs of the profile service
(``services/profile_form``: ``FIELD_SPECS``, ``TABLE_SPECS``); nothing is typed twice here.

Layout, top to bottom: the "YAML -> form" button and the status line, then six groups in the order of
``ids.OHLF_KEYS`` (the scalar fields) with the three tables placed next to the fields they belong to,
and the information-only group. The raw YAML text (``ids.OHL_PROFILE_TEXT``) stays the single run input;
``yaml_block()`` puts it into a collapsed ``<details>`` "Raw YAML (advanced)" below the form.

Besides ``section()`` and ``prefill()`` the module holds the shared pieces the callbacks use: the ids of
the two wrapper divs this module adds, the grid and button ids per table, ``FORM_TARGETS`` (every
component property a profile fills) and ``form_props`` (the values of those properties for a profile).

Round 16 (package R3-B): a collapsed group "Objective weights (advanced)" sits in the form, after the tower
types and before the information-only group. It holds the two objective weights of constrained routing (turn
penalty and tower cost; ``services/constrained_objective`` owns their specs and rules), a status line, a
"Reset weights" button and the mirror store. ``objective_block`` builds it, ``objective_view`` says what its
status line and disabled state are for the page as it is (``profile_facts`` reads the profile), and
``prefill_objective`` puts saved weights back at first paint. ``OBJECTIVE_CONTROLS`` maps a problem key to the
control the "Go fix" button of its notice takes the user to.

Round 17 (package P5): a collapsed group "Line ends" closes the form. It holds the end-span rule, which is a
profile field (the 24th form key, a select whose options come from ``constrained_ends.rule_options()``), a status
line under it (the help of the chosen rule, why the select is disabled, what a GPU backend refuses), and the two
connection heights of the run (``services/constrained_ends`` owns their specs and rules): number inputs in a
fieldset, a status line, a "Clear heights" button and the mirror store. ``line_ends_block`` builds it;
``ends_view`` and ``rule_view`` say what the status lines and disabled states are for the page as it is;
``prefill_ends`` puts saved heights back at first paint; ``ends_controls`` maps a problem key to the control the
"Go fix" button of its notice takes the user to.
"""
from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import dash_bootstrap_components as dbc
from dash import dcc, html

from pyorps.core.exceptions import PyorpsError

from .. import ids
from ..services import constrained
from ..services import constrained_ends as ends_service
from ..services import constrained_objective as objective_service
from ..services import profile_form as service

#: wrapper of the "Turn cost factor" field (shown for linear and quadratic turn cost only)
ANGLE_SCALE_WRAP = ids.OHLF_ANGLE_SCALE_WRAP
#: wrapper of the turn cost table (shown for piecewise turn cost only)
ANGLE_TABLE_WRAP = ids.OHLF_ANGLE_TABLE_WRAP

#: table key -> (grid id, "+ row" button id, "- selected" button id)
TABLE_CONTROLS: dict[str, tuple[str, str, str]] = {
    service.TABLE_ANGLE: (ids.OHLF_ANGLE_GRID, ids.OHLF_ANGLE_ADD_BTN, ids.OHLF_ANGLE_DEL_BTN),
    service.TABLE_TERRAIN: (ids.OHLF_TERRAIN_GRID, ids.OHLF_TERRAIN_ADD_BTN, ids.OHLF_TERRAIN_DEL_BTN),
    service.TABLE_TYPES: (ids.OHLF_TYPES_GRID, ids.OHLF_TYPES_ADD_BTN, ids.OHLF_TYPES_DEL_BTN),
}

#: table key -> the column keys of its rows
TABLE_COLUMN_KEYS: dict[str, tuple[str, ...]] = {
    service.TABLE_ANGLE: service.ANGLE_COLUMNS,
    service.TABLE_TERRAIN: service.TERRAIN_COLUMNS,
    service.TABLE_TYPES: service.TYPE_COLUMNS,
}

#: the three grids in the order the callbacks pass their rows (turn, terrain, types)
GRID_IDS: tuple[str, ...] = tuple(TABLE_CONTROLS[key][0] for key in TABLE_CONTROLS)

#: form key (and table key) -> the component the "Go fix" button of a notice takes the user to
CONTROLS: dict[str, str] = {
    **ids.OHLF_IDS,
    **{key: controls[0] for key, controls in TABLE_CONTROLS.items()},
    "profile": ids.OHLF_STATUS,
}

#: every component property a profile fills, in the order of ``form_props``'s values
FORM_TARGETS: tuple[tuple[str, str], ...] = (
    *((ids.OHLF_IDS[key], "value") for key in service.FORM_KEYS),
    *((grid, "rowData") for grid in GRID_IDS),
    (ids.OHLF_BASE, "data"),
    (ids.OHLF_STATUS, "children"),
    (ids.OHLF_STATUS, "className"),
    (ids.OHLF_CLEARANCE_INFO, "children"),
)

STATUS_CLASS_OK = "small mt-1 text-muted"
STATUS_CLASS_PROBLEM = "small mt-1 text-danger"

# --------------------------------------------------------------------------- objective weights (round 16)
assert tuple(ids.OHLO_KEYS) == objective_service.WEIGHT_KEYS
#: weight key -> the number input of the weight (the order of ``objective_service.WEIGHT_KEYS``)
WEIGHT_IDS: dict[str, str] = dict(ids.OHLO_IDS)
#: problem key of the objective weights -> the component the "Go fix" button of its notice takes the user to
OBJECTIVE_CONTROLS: dict[str, str] = {
    **WEIGHT_IDS, "backend": ids.OHL_BACKEND, "clearance": ids.OHLF_IDS["min_clearance_m"]}

# --------------------------------------------------------------------------- line ends (round 17)
assert tuple(ids.OHLE_KEYS) == ends_service.HEIGHT_KEYS
#: height key -> the number input of the height (the order of ``ends_service.HEIGHT_KEYS``)
HEIGHT_IDS: dict[str, str] = dict(ids.OHLE_IDS)
#: the select of the end-span rule, a field of the profile form
RULE_ID = ids.OHLF_IDS["terminal_span_rule"]
#: the select shows the library's four values with the labels of ``constrained_ends`` (one table for both)
assert [o["value"] for o in ends_service.rule_options()] == list(service.FIELD_SPECS["terminal_span_rule"].options)
assert service.FIELD_SPECS["terminal_span_rule"].default == ends_service.DEFAULT_RULE
#: form key -> the options (label and value) of a select whose labels are not the plain values
OPTION_LABELS: dict[str, list[dict[str, str]]] = {"terminal_span_rule": ends_service.rule_options()}


# --------------------------------------------------------------------------- rows of the tables
def with_row_ids(rows: list | None) -> list[dict]:
    """Copies of ``rows`` that each carry an ``id`` (the grids use it as the row id)."""
    out = []
    for row in rows or ():
        row = dict(row) if isinstance(row, Mapping) else {}
        row["id"] = row.get("id") or uuid.uuid4().hex[:8]
        out.append(row)
    return out


def clean_rows(rows: list | None, columns: tuple[str, ...]) -> list[dict]:
    """The rows as the service reads them: only the column keys (no ``id``), a missing cell is None."""
    return [{column: row.get(column) for column in columns}
            for row in rows or () if isinstance(row, Mapping)]


def blank_row(columns: tuple[str, ...]) -> dict:
    """A new empty table row."""
    return with_row_ids([{column: None for column in columns}])[0]


# --------------------------------------------------------------------------- status and values
def status_for(problems: list[tuple[str, str]], profile: Mapping) -> tuple[str, str]:
    """``(text, className)`` of ``OHLF_STATUS``: the first problems, else what the form cannot show."""
    if not problems and not profile:
        return "", STATUS_CLASS_OK                 # nothing loaded yet: nothing to report
    lines = []
    if problems:
        lines.append(service.problems_text(problems))
    unused = service.unused_keys(profile)
    if unused:
        lines.append("Not read by the router (kept in the YAML): " + ", ".join(unused) + ".")
    unknown = service.unknown_keys(profile)
    if unknown:
        lines.append("Not profile settings (ignored when a profile is built, kept in the YAML): "
                     + ", ".join(unknown) + ".")
    if not lines:
        lines.append("No problems found.")
    return "\n".join(lines), STATUS_CLASS_PROBLEM if problems else STATUS_CLASS_OK


def form_props(profile: Mapping, problems: list[tuple[str, str]] | None = None) -> tuple:
    """The values of :data:`FORM_TARGETS` that show ``profile``.

    The 24 fields and the three tables come from ``form_values``, the base store keeps the whole
    profile (also the keys the form does not show), the status line lists the problems of the profile
    (``problems=None``: found here) and the clearance line says whether the clearance check is on.
    """
    profile = dict(profile or {})
    values, angle_rows, terrain_rows, type_rows = service.form_values(profile)
    if problems is None:
        problems = service.profile_problems(profile) if profile else []
    text, css = status_for(problems, profile)
    return (*(values[key] for key in service.FORM_KEYS),
            with_row_ids(angle_rows), with_row_ids(terrain_rows), with_row_ids(type_rows),
            profile, text, css, service.clearance_status(profile))


# --------------------------------------------------------------------------- the form
def _caption(spec: service.FieldSpec) -> str:
    """The label with its unit in brackets."""
    return f"{spec.label} ({spec.unit})" if spec.unit else spec.label


def _tooltip(spec: service.FieldSpec) -> str:
    return f"{spec.help} Allowed: {service.describe_range(spec)}."


def _step(spec: service.FieldSpec):
    """The ``step`` of a number input: whole numbers step by 1, everything else takes any decimal."""
    return 1 if spec.whole else "any"


def _control(spec: service.FieldSpec, value):
    """The input component of one field, with its limits taken from ``spec``."""
    component_id = ids.OHLF_IDS[spec.key]
    if spec.kind == "select":
        return dbc.Select(id=component_id, size="sm", value=value,
                          options=OPTION_LABELS.get(spec.key)
                          or [{"label": option, "value": option} for option in spec.options])
    if spec.kind == "text":
        return dbc.Input(id=component_id, type="text", size="sm", value=value,
                         maxLength=spec.max_length, debounce=True)
    if spec.kind == "list":
        return dbc.Input(id=component_id, type="text", size="sm", value=value,
                         placeholder="blank", debounce=True)
    return dbc.Input(id=component_id, type="number", size="sm", value=value,
                     min=spec.minimum, max=spec.maximum, step=_step(spec),
                     placeholder="blank", debounce=True)


def _field(key: str, values: Mapping):
    """One labelled field: label with unit, input, the help text as hover text."""
    from ..layout import _help, _lbl

    spec = service.FIELD_SPECS[key]
    return _help([_lbl(_caption(spec), ids.OHLF_IDS[key]), _control(spec, values[key])],
                 _tooltip(spec), className="mb-1")


def _pair(keys: tuple[str, ...], values: Mapping):
    """Fields side by side (two per row)."""
    return dbc.Row([dbc.Col(_field(key, values), width=6) for key in keys], className="g-1")


def _column_defs(table: service.TableSpec) -> list[dict]:
    defs = []
    for column in table.columns:
        defs.append({
            "field": column.key, "headerName": _caption(column), "editable": True, "minWidth": 70,
            "cellDataType": "text" if column.kind == "text" else "number",
            "headerTooltip": f"{column.label}. Allowed: {service.describe_range(column)}."})
    return defs


def _table(table_key: str):
    """One table: caption, help, the grid and its "+ row" / "- selected" buttons."""
    from ..layout import section_grid

    table = service.TABLE_SPECS[table_key]
    grid_id, add_id, delete_id = TABLE_CONTROLS[table_key]
    grid = section_grid(grid_id, _column_defs(table), multi=True, height="150px", responsive=True)
    grid.getRowId = "params.data.id"
    return html.Div([
        html.Div(table.label, className="small fw-semibold"),
        html.Small(table.help, className="text-muted d-block"),
        grid,
        dbc.ButtonGroup([
            dbc.Button("+ row", id=add_id, size="sm", color="secondary", outline=True,
                       title=f"Add an empty row to the {table.label.lower()}"),
            dbc.Button("− selected", id=delete_id, size="sm", color="secondary", outline=True,
                       title=f"Delete the selected rows of the {table.label.lower()}"),
        ], className="mt-1"),
    ], className="mb-1")


def _group(title: str, children: list, *, open: bool = True, **props):  # pylint: disable=redefined-builtin
    return html.Details([
        html.Summary(title, className="small fw-semibold", style={"cursor": "pointer"}),
        *children,
    ], open=open, className="mb-2 border-bottom pb-1", **props)


# --------------------------------------------------------------------------- the objective weights block
def profile_facts(profile: Mapping | None) -> dict[str, bool | None]:
    """What the weights block needs to know about a profile: is the clearance check on, does it set spans.

    ``spans_on`` is None while no profile is loaded (nothing to say about spans yet).
    """
    profile = dict(profile or {})
    if not profile:
        return {"clearance_on": False, "spans_on": None}
    return {"clearance_on": service.clearance_state(profile) == "on",
            "spans_on": profile.get("min_span_m") is not None and profile.get("max_span_m") is not None}


def objective_view(weights: Mapping | None, *, backend: str | None, experimental: bool, dem: str | None,
                   profile: Mapping | None) -> objective_service.View:
    """The status line and the disabled state of the block for the page as it is now.

    The backend is the one a run would use: an experimental backend without the experimental flag is reset
    to cython by the run (``constrained.resolve_backend``), so the block treats it as cython too.
    """
    resolved, _ = constrained.resolve_backend(backend or "cython", bool(experimental))
    return objective_service.view(weights, backend=resolved, has_dem=bool(str(dem or "").strip()),
                                  **profile_facts(profile))


def objective_status_props(view: objective_service.View) -> tuple[str, str]:
    """``(text, className)`` of ``OHLO_STATUS``."""
    return view.text, STATUS_CLASS_PROBLEM if view.problem else STATUS_CLASS_OK


def _weight_field(key: str, value):
    """One labelled weight: label with unit, number input (limits from the spec), the help as hover text."""
    from ..layout import _help, _lbl

    spec = objective_service.FIELD_SPECS[key]
    component_id = WEIGHT_IDS[key]
    return _help(
        [_lbl(objective_service.label_with_unit(spec), component_id),
         dbc.Input(id=component_id, type="number", size="sm", value=value, min=spec.minimum, max=spec.maximum,
                   step="any", placeholder=objective_service.format_number(spec.default), debounce=True)],
        f"{spec.help} Allowed: {objective_service.describe_range(spec)}.", className="mb-1")


def objective_block(weights: Mapping | None = None, view: objective_service.View | None = None):
    """The collapsed "Objective weights" block of the constrained card.

    ``weights`` fills the number inputs (``None``: the defaults); ``view`` is the first status line and disabled
    state (``None``: the weights as they are, on the cython backend without a clearance check). The block is
    open when a saved weight is not at its default, so a project that uses weights shows them.
    """
    from ..layout import _help

    values = objective_service.input_values(weights)
    view = view or objective_service.view(weights)
    text, css = objective_status_props(view)
    return _group("Objective weights (advanced)", [
        html.Small("Multiply the turn costs and the tower costs of the profile, for example to make bends "
                   "cheaper or towers dearer. Both weights at 1 route exactly as the profile says.",
                   className="text-muted d-block"),
        html.Div(text, id=ids.OHLO_STATUS, className=css, style={"whiteSpace": "pre-line"}),
        html.Fieldset([_weight_field(key, values[key]) for key in objective_service.WEIGHT_KEYS],
                      id=ids.OHLO_FIELDS, disabled=not view.enabled, title=view.reason or None),
        _help(dbc.Button("Reset weights", id=ids.OHLO_RESET_BTN, size="sm", color="secondary", outline=True,
                         className="mt-1"),
              "Set both weights back to 1: the turn and tower costs of the profile apply as written."),
        dcc.Store(id=ids.OHLO_SETTINGS, data=objective_service.normalise(weights)),
    ], open=not objective_service.is_default(weights), id=ids.OHLO_BLOCK)


# --------------------------------------------------------------------------- the line ends block
def ends_view(heights: Mapping | None, *, backend: str | None, experimental: bool, dem: str | None,
              profile: Mapping | None) -> objective_service.View:
    """The status line and the disabled state of the two heights for the page as it is now.

    The backend is the one a run would use (an experimental backend without the experimental flag is reset to
    cython by the run); the terrain model (DEM) is the one of the constrained card, a blank one counts as none;
    ``profile`` is the profile dict (the base store of the form).
    """
    resolved, _ = constrained.resolve_backend(backend or "cython", bool(experimental))
    return ends_service.view(heights, backend=resolved, profile=dict(profile or {}),
                             dem=str(dem or "").strip() or None)


def ends_status_props(view: objective_service.View) -> tuple[str, str]:
    """``(text, className)`` of ``OHLE_STATUS``."""
    return view.text, STATUS_CLASS_PROBLEM if view.problem else STATUS_CLASS_OK


class RuleView(NamedTuple):
    """What the end-span rule shows under its select: the text, its class and whether the select is disabled."""

    text: str
    css: str
    disabled: bool


def rule_view(rule: Any, *, backend: str | None, experimental: bool, profile: Mapping | None) -> RuleView:
    """The line under the rule select for the page as it is now.

    Without span limits the profile places no towers: the select is disabled and the line says why
    (``rule_support``). Otherwise the line is the help of the chosen rule; on a GPU backend it also says that
    "Never place one" is refused there (in the colour of an error while that rule is chosen).
    """
    resolved, _ = constrained.resolve_backend(backend or "cython", bool(experimental))
    support = ends_service.rule_support(dict(profile or {}), resolved)
    if not support.enabled:
        return RuleView(support.reason, STATUS_CLASS_OK, True)
    value = rule if isinstance(rule, str) and rule else ends_service.DEFAULT_RULE
    entry = ends_service.RULE_LABELS.get(value)
    lines = [entry.help] if entry else []
    if support.refused:
        names = "; ".join(ends_service.rule_label(refused) for refused in support.refused)
        lines.append(f"Refused on this backend: {names}. {support.refused_reason}")
    return RuleView("\n".join(lines), STATUS_CLASS_PROBLEM if value in support.refused else STATUS_CLASS_OK, False)


def ends_controls(heights: Mapping | None, profile: Mapping | None, dem: str | None) -> dict[str, str]:
    """Problem key of the line ends -> the component the "Go fix" button of its notice takes the user to.

    The size problem (``"bins"``) has two possible causes: it goes to the rule select when the rule "Never place
    one" acts, otherwise to the first connection height the search uses.
    """
    effect = ends_service.effect(heights, profile=dict(profile or {}), dem=str(dem or "").strip() or None)
    bins = RULE_ID
    if not effect.enforce:
        acting = [key for key, acts in zip(ends_service.HEIGHT_KEYS, (effect.source_acts, effect.target_acts)) if acts]
        bins = HEIGHT_IDS[acting[0]] if acting else RULE_ID
    return {**HEIGHT_IDS, "rule": RULE_ID, "backend": ids.OHL_BACKEND, "bins": bins}


def _height_field(key: str, value):
    """One labelled height: label with unit, number input (limits from the spec), the help as hover text."""
    from ..layout import _help, _lbl

    spec = ends_service.FIELD_SPECS[key]
    component_id = HEIGHT_IDS[key]
    return _help(
        [_lbl(_caption(spec), component_id),
         dbc.Input(id=component_id, type="number", size="sm", value=value, min=spec.minimum, max=spec.maximum,
                   step="any", placeholder="not set", debounce=True)],
        f"{spec.help} Allowed: {objective_service.describe_range(spec)}, or leave it blank.", className="mb-1")


def _rule_field(value, disabled: bool):
    """The labelled end-span rule select (a profile field: the form callbacks read and fill it like the others)."""
    from ..layout import _help, _lbl

    spec = service.FIELD_SPECS["terminal_span_rule"]
    control = _control(spec, value)
    control.disabled = disabled
    return _help([_lbl(_caption(spec), RULE_ID), control], _tooltip(spec), className="mb-1")


def line_ends_block(heights: Mapping | None = None, view: objective_service.View | None = None,
                    rule: RuleView | None = None, *, open: bool | None = None):  # pylint: disable=redefined-builtin
    """The collapsed "Line ends" group at the end of the form: the end-span rule and the two connection heights.

    ``heights`` fill the number inputs (``None``: not set); ``view`` and ``rule`` are the first status lines and
    disabled states (``None``: the page without a profile on the cython backend). The group is open when a height
    is set (or when ``open`` says so), so a project that uses them shows them. The rule select is a profile field:
    its value is filled by the profile form like every other field; only its disabled state and the line under it
    belong to this block.
    """
    from ..layout import _help

    values = ends_service.input_values(heights)
    view = view or ends_view(heights, backend=None, experimental=False, dem=None, profile=None)
    rule = rule or rule_view(None, backend=None, experimental=False, profile=None)
    text, css = ends_status_props(view)
    rule_default = service.form_values({})[0]["terminal_span_rule"]
    return _group("Line ends", [
        html.Small("What happens at the two ends of the line: the end-span rule (a setting of the profile) and the "
                   "heights at which the conductor leaves the structure at the source and at the target (a setting "
                   "of this run).", className="text-muted d-block"),
        _rule_field(rule_default, rule.disabled),
        html.Div(rule.text, id=ids.OHLE_RULE_STATUS, className=rule.css, style={"whiteSpace": "pre-line"}),
        html.Div("Connection heights", className="small fw-semibold mt-1"),
        html.Div(text, id=ids.OHLE_STATUS, className=css, style={"whiteSpace": "pre-line"}),
        html.Fieldset([_height_field(key, values[key]) for key in ends_service.HEIGHT_KEYS],
                      id=ids.OHLE_FIELDS, disabled=not view.enabled, title=view.reason or None),
        _help(dbc.Button("Clear heights", id=ids.OHLE_CLEAR_BTN, size="sm", color="secondary", outline=True,
                         className="mt-1"),
              "Leave both connection heights blank: the line ends behave as before."),
        dcc.Store(id=ids.OHLE_SETTINGS, data=ends_service.normalise(heights)),
    ], open=(not ends_service.is_default(heights)) if open is None else open, id=ids.OHLE_BLOCK)


def section() -> list:
    """The profile form for the constrained options (spliced after the "Profile file" row)."""
    from ..layout import _help

    values = service.form_values({})[0]
    groups = [
        _group("Profile", [_field("name", values), _field("description", values)]),
        _group("Turns", [
            _pair(("soft_angle_limit_deg", "hard_angle_limit_deg"), values),
            _field("angle_cost_function", values),
            html.Div(_field("angle_cost_scale", values), id=ANGLE_SCALE_WRAP,
                     style={"display": "none"}),
            html.Div(_table(service.TABLE_ANGLE), id=ANGLE_TABLE_WRAP),
        ]),
        _group("Spans", [
            _pair(("min_span_m", "max_span_m"), values),
            _field("span_bin_size_m", values),
            html.Small("The end-span rule and the connection heights at the line ends are in the group Line ends "
                       "below.", className="text-muted d-block"),
        ]),
        _group("Terrain and slope", [
            _field("terrain_interpolation", values),
            _table(service.TABLE_TERRAIN),
            _pair(("max_gradient_percent", "gradient_scale"), values),
        ]),
        _group("Tower heights and clearance", [
            _pair(("tower_height_m", "tower_heights_m"), values),
            _pair(("tower_height_cost_per_increment", "terminal_tower_height_m"), values),
            _pair(("terminal_tower_cost", "conductor_weight_per_m"), values),
            _pair(("conductor_tension_n", "min_clearance_m"), values),
            html.Div(id=ids.OHLF_CLEARANCE_INFO, className="small text-muted mb-1",
                     children=service.clearance_status({})),
        ]),
        _group("Foundations and tower types", [
            _pair(("tower_ground_area_m2", "tower_area_cost_mode"), values),
            _table(service.TABLE_TYPES),
        ]),
        objective_block(),
        _group("Information only (the router does not read it)", [
            _field("recommended_buffer_m", values),
        ]),
        line_ends_block(),      # last: it holds the 24th form field (the page shows the fields in key order)
    ]
    return [html.Div([
        html.Small("Every change here updates the YAML profile that a run uses. Keys the form does not "
                   "show stay in the YAML.", className="text-muted d-block"),
        _help(dbc.Button("YAML → form", id=ids.OHLF_FROM_YAML_BTN, size="sm", color="secondary",
                         outline=True, className="my-1"),
              "Fill the form from the text in Raw YAML (advanced). Use it after you edited that "
              "text by hand."),
        html.Div(id=ids.OHLF_STATUS, className=STATUS_CLASS_OK, style={"whiteSpace": "pre-line"},
                 children=""),
        *groups,
    ], className="ohlf-form my-1")]


def yaml_block() -> list:
    """The raw YAML profile editor in a collapsed "Raw YAML (advanced)" block."""
    from ..layout import _lbl  # lazily: layout imports this module

    return [
        html.Details([
            html.Summary("Raw YAML (advanced)"),
            html.Small("The form above edits this text. Edit it by hand only for settings the form does "
                       "not show, then press \"YAML → form\".", className="text-muted d-block"),
            _lbl("Infrastructure profile (YAML)", ids.OHL_PROFILE_TEXT),
            dbc.Textarea(id=ids.OHL_PROFILE_TEXT, size="sm",
                         style={"height": "160px", "fontFamily": "monospace",
                                "fontSize": "11px"}),
        ], id=ids.OHLF_YAML_DETAILS, open=False),
    ]


# --------------------------------------------------------------------------- first paint
def prefill(state, put: Callable[[str, str, Any], None]) -> None:
    """First-paint values from ``state.gui_settings["constrained"]``.

    A blank project (no saved settings) puts nothing, so the layout's own defaults stay. The saved
    profile fills the form, the tables, the base store, the status lines and the YAML text; backend,
    experimental flag, DEM and DSM go into their controls.
    """
    settings = (getattr(state, "gui_settings", None) or {}).get("constrained")
    if not isinstance(settings, Mapping) or not settings:
        return
    profile = settings.get("profile")
    if isinstance(profile, Mapping) and profile:
        for (component_id, prop), value in zip(FORM_TARGETS, form_props(profile)):
            put(component_id, prop, value)
        with contextlib.suppress(PyorpsError):   # without PyYAML the text stays empty; the form still shows
            put(ids.OHL_PROFILE_TEXT, "value", service.profile_to_text(profile))
    if settings.get("backend"):
        put(ids.OHL_BACKEND, "value", settings["backend"])
    if "experimental" in settings:
        put(ids.OHL_EXPERIMENTAL, "value", bool(settings["experimental"]))
    for key, component_id in (("dem", ids.OHL_DEM), ("dsm", ids.OHL_DSM)):
        if settings.get(key):
            put(component_id, "value", settings[key])
    prefill_objective(settings, put)
    prefill_ends(settings, put)
    put(ids.OHL_SETTINGS, "data", dict(settings))


def prefill_objective(settings: Mapping, put: Callable[[str, str, Any], None]) -> None:
    """First-paint values of the "Objective weights" block from ``settings["objective"]``.

    A project that never changed a weight has no ``objective`` key and puts nothing: the block keeps its
    defaults. Otherwise the weights, the mirror store, the status line, the disabled state and the open flag
    of the block follow the saved weights and the saved backend, DEM and profile.
    """
    weights = settings.get("objective")
    if not isinstance(weights, Mapping) or not weights:
        return
    view = objective_view(weights, backend=settings.get("backend"), experimental=bool(settings.get("experimental")),
                          dem=settings.get("dem"), profile=settings.get("profile"))
    for key, value in objective_service.input_values(weights).items():
        put(WEIGHT_IDS[key], "value", value)
    put(ids.OHLO_SETTINGS, "data", objective_service.normalise(weights))
    text, css = objective_status_props(view)
    put(ids.OHLO_STATUS, "children", text)
    put(ids.OHLO_STATUS, "className", css)
    put(ids.OHLO_FIELDS, "disabled", not view.enabled)
    put(ids.OHLO_FIELDS, "title", view.reason or None)
    put(ids.OHLO_BLOCK, "open", not objective_service.is_default(weights))


def prefill_ends(settings: Mapping, put: Callable[[str, str, Any], None]) -> None:
    """First-paint values of the "Line ends" group from ``settings["ends"]`` and the saved profile.

    A project with neither saved heights nor a saved profile puts nothing: the group keeps its defaults. Otherwise
    the two status lines and the disabled states follow the saved heights, backend, DEM and profile (the rule
    select itself is filled with the profile by :func:`prefill`); saved heights also fill the inputs and the mirror
    store. The group is open when a height is set or the saved profile names a rule other than the default.
    """
    heights = settings.get("ends")
    profile = settings.get("profile")
    has_heights = isinstance(heights, Mapping) and bool(heights)
    profile = dict(profile) if isinstance(profile, Mapping) else {}
    if not has_heights and not profile:
        return
    backend, experimental, dem = settings.get("backend"), bool(settings.get("experimental")), settings.get("dem")
    view = ends_view(heights if has_heights else None, backend=backend, experimental=experimental, dem=dem,
                     profile=profile)
    rule = profile.get("terminal_span_rule")
    rview = rule_view(rule, backend=backend, experimental=experimental, profile=profile)
    if has_heights:
        for key, value in ends_service.input_values(heights).items():
            put(HEIGHT_IDS[key], "value", value)
        put(ids.OHLE_SETTINGS, "data", ends_service.normalise(heights))
    text, css = ends_status_props(view)
    put(ids.OHLE_STATUS, "children", text)
    put(ids.OHLE_STATUS, "className", css)
    put(ids.OHLE_FIELDS, "disabled", not view.enabled)
    put(ids.OHLE_FIELDS, "title", view.reason or None)
    put(ids.OHLE_RULE_STATUS, "children", rview.text)
    put(ids.OHLE_RULE_STATUS, "className", rview.css)
    put(RULE_ID, "disabled", rview.disabled)
    rule_changed = isinstance(rule, str) and rule != ends_service.DEFAULT_RULE
    put(ids.OHLE_BLOCK, "open", has_heights and not ends_service.is_default(heights) or rule_changed)
