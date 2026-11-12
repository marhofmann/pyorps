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
"""
from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from pyorps.core.exceptions import PyorpsError

from .. import ids
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

    The 23 fields and the three tables come from ``form_values``, the base store keeps the whole
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
                          options=[{"label": option, "value": option} for option in spec.options])
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


def _group(title: str, children: list, *, open: bool = True):  # pylint: disable=redefined-builtin
    return html.Details([
        html.Summary(title, className="small fw-semibold", style={"cursor": "pointer"}),
        *children,
    ], open=open, className="mb-2 border-bottom pb-1")


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
        _group("Information only (the router does not read it)", [
            _field("recommended_buffer_m", values),
        ]),
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
    put(ids.OHL_SETTINGS, "data", dict(settings))
