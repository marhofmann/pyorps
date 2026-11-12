"""PYORPS GUI section: objective and terrain card (gaps 1, 2 and 6, package P5).

A collapsible card "Objective & terrain" in the Routes tab, right after "Algorithm & search", closed by
default. It sets what a plain route minimises: the objective weights, extra metric layers, a terrain model
(DEM) with the slope response, and the weight precision. Every label, unit, limit, default and help text of a
number comes from the specs of the objective service (``services/objective``: ``FIELD_SPECS``,
``PRECISION_LABELS``, ``SLOPE_NEEDS_DEM`` ...); the wording the specs do not hold (the two select lists, the
card help) is defined here once.

Top to bottom: the status line (the summary of the settings, or the first problems), the three preset
buttons, the terrain model, the three weights, the extra layers (a form to add one, a table with the layers),
the slope response (a closed accordion) and the weight precision. The slope weight and the slope response
are disabled until a terrain model is chosen; the 32-bit choice is disabled until the objective is not plain
cost routing and the backend takes float weights. The reason sits on the wrapper div as hover text (a
disabled control shows none of its own).

Besides ``section()`` and ``prefill()`` the module holds the pieces the callbacks share: the ids of the wrapper
divs it adds, ``FORM_TARGETS`` (every component property a settings dict fills) with ``form_props``, the grid
row helpers and ``control_states`` (the disabled flags, hover texts and options that follow the settings,
the backend and the algorithm).
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from .. import ids
from ..services import objective as service
from ..services import routing

#: wrapper of the terrain model dropdown (its hover text names why a DEM would be refused)
DEM_WRAP = ids.OBJ_DEM_WRAP
#: wrapper of the "Exponential shape scale" field (shown for the exponential slope shape only)
ADDITIVE_SCALE_WRAP = ids.OBJ_ADDITIVE_SCALE_WRAP
#: wrapper of the "Multiplier scale" field (shown for the exponential, energy and squared multipliers)
MULT_SCALE_WRAP = ids.OBJ_MULT_SCALE_WRAP
#: wrapper of the three sigmoid fields (shown for the sigmoid multiplier only)
SIGMOID_WRAP = ids.OBJ_SIGMOID_WRAP

#: preset button id -> the preset it writes (``service.PRESETS``)
PRESET_BUTTONS: dict[str, str] = {
    ids.OBJ_PRESET_CLASSIC_BTN: "classic",
    ids.OBJ_PRESET_SHORTEST_BTN: "shortest",
    ids.OBJ_PRESET_GENTLEST_BTN: "gentlest",
}
assert tuple(PRESET_BUTTONS.values()) == tuple(service.PRESETS)
PRESET_LABELS: dict[str, str] = {"classic": "Cost only", "shortest": "Shortest", "gentlest": "Gentlest (needs DEM)"}
PRESET_HELP: dict[str, str] = {
    "classic": "Cost weight 1, the other weights 0: the classic routing. Layers, terrain model and slope "
               "response stay as they are.",
    "shortest": "Only the length counts: the geometrically shortest route (3-D length with a terrain model).",
    "gentlest": "Slope weight 1 and a small length weight (0.01): the route with the least slope penalty. "
                "Needs a terrain model (DEM).",
}

#: weight key -> the number input of the weight
WEIGHT_IDS: dict[str, str] = {"cost": ids.OBJ_W_COST, "length": ids.OBJ_W_LENGTH, "gradient": ids.OBJ_W_GRADIENT}
assert tuple(WEIGHT_IDS) == service.WEIGHT_KEYS

#: slope response key -> its control (the order of ``service.OBJECTIVE_DEFAULTS["gradient"]``)
GRADIENT_IDS: dict[str, str] = {
    "max_gradient_pct": ids.OBJ_GRAD_MAX_PCT,
    "additive": ids.OBJ_GRAD_ADDITIVE,
    "additive_scale": ids.OBJ_GRAD_ADDITIVE_SCALE,
    "multiplier": ids.OBJ_GRAD_MULTIPLIER,
    "mult_scale": ids.OBJ_GRAD_MULT_SCALE,
    "sig_center_deg": ids.OBJ_GRAD_SIG_CENTER,
    "sig_steepness": ids.OBJ_GRAD_SIG_STEEPNESS,
    "sig_max_penalty": ids.OBJ_GRAD_SIG_MAX,
    "bin_width_pct": ids.OBJ_GRAD_BIN_PCT,
    "s_max_pct": ids.OBJ_GRAD_SMAX_PCT,
}
assert tuple(GRADIENT_IDS) == tuple(service.OBJECTIVE_DEFAULTS["gradient"])
#: the two slope response keys that are a list choice (shown as "none" when blank)
CHOICE_KEYS = ("additive", "multiplier")

#: number spec key -> the number input that edits it (``FIELD_SPECS`` keys without the grid-only layer bounds)
NUMBER_IDS: dict[str, str] = {
    **{f"weights.{key}": WEIGHT_IDS[key] for key in WEIGHT_IDS},
    "layers.weight": ids.OBJ_LAYER_WEIGHT,
    **{f"gradient.{key}": GRADIENT_IDS[key] for key in GRADIENT_IDS if key not in CHOICE_KEYS},
}

#: the new-layer form and the table of the layers
LAYER_FORM_IDS: tuple[str, ...] = (ids.OBJ_LAYER_NAME, ids.OBJ_LAYER_SOURCE, ids.OBJ_LAYER_WEIGHT)
#: the columns of a layer row the service reads (a grid row also carries ``id`` and ``source_label``)
LAYER_COLUMNS: tuple[str, ...] = ("name", "source", "weight", "hard_max", "hard_min")

#: every component property a settings dict fills, in the order of ``form_props``'s values
FORM_TARGETS: tuple[tuple[str, str], ...] = (
    *((WEIGHT_IDS[key], "value") for key in WEIGHT_IDS),
    *((GRADIENT_IDS[key], "value") for key in GRADIENT_IDS),
    (ids.OBJ_DEM_LAYER, "value"),
    (ids.OBJ_PRECISION, "value"),
    (ids.OBJ_LAYERS_GRID, "rowData"),
    (ids.OBJ_SETTINGS, "data"),
)

STATUS_CLASS_OK = "small mt-1 text-muted"
STATUS_CLASS_PROBLEM = "small mt-1 text-danger"

SHAPE_LABEL = "Slope shape"
MULTIPLIER_LABEL = "Slope penalty multiplier"
#: the options of the two slope response lists: value -> text ("none" is how a blank list value is shown)
SHAPE_OPTIONS: dict[str, str] = {
    "none": "none (a slope weight acts linearly)",
    **{shape: shape for shape in service.ADDITIVE_SHAPES},
}
MULTIPLIER_OPTIONS: dict[str, str] = {"none": "none", **{model: model for model in service.MULTIPLIER_MODELS}}
SHAPE_HELP = ("How the slope weight grows with the slope: linear, quadratic or exponential. Blank (none): linear "
              "as soon as the slope weight is above 0.")
MULTIPLIER_HELP = "Multiplies the whole per-metre cost on slopes (a steeper cell costs a multiple of a flat one)."

INTRO = ("What a plain route minimises: the cost raster, a length term, a slope term from a terrain model and "
         "extra raster layers. Cost only is the classic routing and changes nothing. Constrained routing and the "
         "corridor graph do not use these settings.")
DEM_HELP = ("Elevation in metres, taken from a loaded raster layer. It must use the coordinate system of the cost "
            "raster. With a terrain model every route length is the 3-D length, the cost weight stays in force "
            "(a terrain model always switches the objective on), and on cython only the Dijkstra algorithm runs.")
LAYERS_HELP = ("Extra raster layers that add to the objective: weight x layer value x metres. Name a layer, pick "
               "its raster (or the slope from the terrain model), then press + Add layer. Cells above 'Forbid "
               "above' or below 'Forbid below' become forbidden.")
SLOPE_HELP = ("The shape and the limit of the slope term. Open it to see every field; each one is explained on "
              "hover.")
PRECISION_LABEL = "Weight precision"
PRECISION_HELP = "Use 32-bit when a warning says the combined surface lost resolution."


# --------------------------------------------------------------------------- rows of the layer table
def with_row_ids(rows: list | None) -> list[dict]:
    """Copies of ``rows`` that each carry an ``id`` (the grid uses it as the row id)."""
    out = []
    for row in rows or ():
        row = dict(row) if isinstance(row, Mapping) else {}
        row["id"] = row.get("id") or uuid.uuid4().hex[:8]
        out.append(row)
    return out


def clean_rows(rows: list | None) -> list[dict]:
    """The rows as the service reads them: only the layer columns (no ``id``, no source label)."""
    return [{column: row.get(column) for column in LAYER_COLUMNS}
            for row in rows or () if isinstance(row, Mapping)]


def layer_rows(layers: list | None, layer_names: Mapping[str, str] | None = None) -> list[dict]:
    """Grid rows for ``layers`` (service layer dicts): each with an ``id`` and the source label."""
    rows = []
    for layer in layers or ():
        row = {column: layer.get(column) for column in LAYER_COLUMNS}
        row["source_label"] = service.source_label(row["source"], layer_names)
        rows.append(row)
    return with_row_ids(rows)


def live_rows(row_data: list | None, virtual: list | None) -> list[dict]:
    """The rows the grid holds now: ``virtualRowData`` follows every cell edit, ``rowData`` only a callback."""
    rows = virtual if virtual is not None else row_data
    return [dict(row) for row in rows or () if isinstance(row, Mapping)]


# --------------------------------------------------------------------------- values of the controls
def _number_or_none(value: Any) -> float | None:
    """A control can only show a number: anything else (a text left in a saved file) shows as blank."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def form_props(settings: Any, layer_names: Mapping[str, str] | None = None) -> tuple:
    """The values of :data:`FORM_TARGETS` that show ``settings`` (normalised here).

    The last value is the normalised settings themselves: they go to the ``OBJ_SETTINGS`` store, so a restore
    and the mirror callback agree on what the page holds.
    """
    s = service.normalise_objective_settings(settings)
    gradient = s["gradient"]
    values: list[Any] = [_number_or_none(s["weights"][key]) for key in WEIGHT_IDS]
    for key in GRADIENT_IDS:
        value = gradient[key]
        values.append((value or "none") if key in CHOICE_KEYS else _number_or_none(value))
    precision = s["weight_precision"]
    values += [s["dem_layer_id"], precision if precision in service.PRECISIONS else service.PRECISIONS[0],
               layer_rows(s["layers"], layer_names), s]
    return tuple(values)


def control_states(settings: Any, graph_api: str | None, algorithm: str | None) -> dict[str, Any]:
    """What follows the settings, the Backend and the Algorithm: disabled flags, hover texts, options.

    ``weight_disabled`` / ``weight_title``: the slope weight; ``slope_disabled`` / ``slope_title``: the slope
    response block (both off until a terrain model is chosen, the title then is ``SLOPE_NEEDS_DEM``);
    ``dem_title``: the terrain model (never disabled, so a conflict can be seen and cleared; its title says why
    a DEM would be refused); ``precision_title`` and ``precision_options``: the 32-bit choice is the one that
    can be disabled, the 16-bit one never is, so a stale 32-bit choice can always be left.
    """
    s = service.normalise_objective_settings(settings)
    slope = service.slope_support(s)
    terrain = service.terrain_support(graph_api, algorithm)
    precision = service.precision_support(s, graph_api)
    weight_spec = service.FIELD_SPECS["weights.gradient"]
    return {
        "weight_disabled": not slope.enabled,
        "weight_title": _tooltip(weight_spec) if slope.enabled else slope.reason,
        "slope_disabled": not slope.enabled,
        "slope_title": SLOPE_HELP if slope.enabled else slope.reason,
        "dem_title": DEM_HELP if terrain.enabled else
        f"{DEM_HELP} With the current Backend and Algorithm a terrain model would be refused: {terrain.reason}",
        "precision_title": PRECISION_HELP if precision.enabled else precision.reason,
        "precision_options": precision_options(precision.enabled),
    }


def precision_options(float32_enabled: bool) -> list[dict]:
    """The radio options: 16-bit is always selectable, 32-bit only where it applies."""
    return [{"label": f" {service.PRECISION_LABELS[name]}", "value": name,
             "disabled": name == "float32" and not float32_enabled} for name in service.PRECISIONS]


def shown_blocks(additive: str | None, multiplier: str | None) -> tuple[dict, dict, dict]:
    """The ``style`` of the three conditional wrappers (exponential scale, multiplier scale, sigmoid fields)."""
    keys = service.applicable_gradient_keys({"additive": additive if additive != "none" else None,
                                             "multiplier": multiplier if multiplier != "none" else None})
    hidden = {"display": "none"}
    return (({} if "additive_scale" in keys else hidden),
            ({} if "mult_scale" in keys else hidden),
            ({} if "sig_center_deg" in keys else hidden))


# --------------------------------------------------------------------------- the controls
def _caption(spec: service.FieldSpec) -> str:
    """The label with its unit in brackets (no unit shown for ``-``)."""
    return service.label_with_unit(spec)


def _tooltip(spec: service.FieldSpec) -> str:
    return f"{spec.help} Allowed: {service.describe_range(spec)}."


def _minimum(spec_key: str) -> float | None:
    """The ``min`` of a number input: the spec's minimum; the slope range is at least one table step wide."""
    spec = service.FIELD_SPECS[spec_key]
    if spec.minimum is None and spec_key == "gradient.s_max_pct":
        return service.FIELD_SPECS["gradient.bin_width_pct"].minimum
    return spec.minimum


#: placeholders of the fields whose blank means something
_PLACEHOLDERS: dict[str, str] = {"gradient.max_gradient_pct": "no limit", "gradient.mult_scale": "model default"}


def _number(spec_key: str, value: Any, *, className: str = "mb-1", hover: bool = True):
    """One labelled number input with its limits, unit and help text taken from its spec.

    ``hover=False`` leaves the help text off the field: a wrapper around it (the slope weight's fieldset)
    carries the hover text instead, so that its "needs a terrain model" reason is not shadowed.
    """
    from ..layout import _help, _lbl

    spec = service.FIELD_SPECS[spec_key]
    component_id = NUMBER_IDS[spec_key]
    control = dbc.Input(id=component_id, type="number", size="sm", value=value, min=_minimum(spec_key),
                        max=spec.maximum, step="any", debounce=True, placeholder=_PLACEHOLDERS.get(spec_key))
    children = [_lbl(_caption(spec), component_id), control]
    if not hover:
        return html.Div(children, className=className)
    return _help(children, _tooltip(spec), className=className)


def _select(component_id: str, label: str, options: Mapping[str, str], value: Any, tooltip: str):
    from ..layout import _help, _lbl

    return _help([_lbl(label, component_id),
                  dbc.Select(id=component_id, size="sm", value=value,
                             options=[{"label": text, "value": option} for option, text in options.items()])],
                 tooltip, className="mb-1")


def _presets():
    buttons = [dbc.Button(PRESET_LABELS[name], id=button_id, size="sm", color="secondary", outline=True,
                          title=PRESET_HELP[name])
               for button_id, name in PRESET_BUTTONS.items()]
    return html.Div([html.Div("Presets (set the three weights)", className="small"),
                     html.Div(buttons, className="d-flex flex-wrap gap-1 mb-1")])


def _terrain_model(value, states: Mapping[str, Any]):
    from ..layout import _lbl

    return html.Div([
        _lbl("Terrain model (DEM)", ids.OBJ_DEM_LAYER),
        dcc.Dropdown(id=ids.OBJ_DEM_LAYER, options=[], value=value, clearable=True, className="small",
                     placeholder="None (flat ground)"),
    ], id=DEM_WRAP, title=states["dem_title"], className="mb-1")


def _weights(values: Mapping[str, Any], states: Mapping[str, Any]):
    """The three weights; ``values`` maps the weight keys to their numbers."""
    return html.Div([
        html.Div("Weights: cost x raster value + length x metres + slope x slope response", className="small"),
        dbc.Row([dbc.Col(_number("weights.cost", values["cost"]), width=6),
                 dbc.Col(_number("weights.length", values["length"]), width=6)], className="g-1"),
        html.Fieldset(_number("weights.gradient", values["gradient"], hover=False), id=ids.OBJ_W_GRADIENT_WRAP,
                      disabled=states["weight_disabled"], title=states["weight_title"]),
    ])


def _column_defs() -> list[dict]:
    """The columns of the layers table: name and source are shown, weight and the two bounds are edited."""
    defs: list[dict] = [
        {"field": "name", "headerName": "Layer", "editable": False, "minWidth": 70, "cellDataType": "text"},
        {"field": "source_label", "headerName": "Source", "editable": False, "minWidth": 90, "cellDataType": "text",
         "headerTooltip": "The raster layer, or the slope derived from the terrain model."},
    ]
    for column, key, header in (("weight", "layers.weight", "Weight"), ("hard_max", "layers.hard_max", None),
                                ("hard_min", "layers.hard_min", None)):
        spec = service.FIELD_SPECS[key]
        defs.append({"field": column, "headerName": header or spec.label, "editable": True, "minWidth": 70,
                     "cellDataType": "number",
                     "headerTooltip": f"{_caption(spec)}. {spec.help} Allowed: {service.describe_range(spec)}."})
    return defs


def _layers_block():
    from ..layout import _lbl, section_grid

    grid = section_grid(ids.OBJ_LAYERS_GRID, _column_defs(), multi=True, height="140px", responsive=True)
    grid.getRowId = "params.data.id"
    return html.Div([
        html.Div("Extra metric layers", className="small fw-semibold mt-2"),
        html.Small(LAYERS_HELP, className="text-muted d-block"),
        dbc.Row([
            dbc.Col([_lbl("Layer name", ids.OBJ_LAYER_NAME),
                     dbc.Input(id=ids.OBJ_LAYER_NAME, type="text", size="sm", value="", maxLength=32,
                               placeholder="e.g. landscape")], width=5),
            dbc.Col([_lbl("From raster", ids.OBJ_LAYER_SOURCE),
                     dcc.Dropdown(id=ids.OBJ_LAYER_SOURCE, options=[], value=None, clearable=True,
                                  className="small", placeholder="Pick a raster layer")], width=7),
        ], className="g-1 mb-1"),
        dbc.Row([
            dbc.Col(_number("layers.weight", service.FIELD_SPECS["layers.weight"].default), width=6),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("+ Add layer", id=ids.OBJ_LAYER_ADD_BTN, size="sm", color="secondary", outline=True,
                           title="Add the layer above to the table"),
                dbc.Button("− Remove selected", id=ids.OBJ_LAYER_DEL_BTN, size="sm", color="secondary",
                           outline=True, title="Remove the selected rows of the table"),
            ]), width=6, className="d-flex align-items-end mb-1"),
        ], className="g-1"),
        grid,
    ])


def _slope_block(gradient: Mapping[str, Any], states: Mapping[str, Any]):
    """The slope response in a closed accordion; one wrapper fieldset disables every control inside it.

    ``gradient`` holds the shown values (a blank list value is already ``"none"``).
    """
    additive_scale, mult_scale, sigmoid = shown_blocks(gradient["additive"], gradient["multiplier"])
    fields = [
        _number("gradient.max_gradient_pct", gradient["max_gradient_pct"]),
        _select(ids.OBJ_GRAD_ADDITIVE, SHAPE_LABEL, SHAPE_OPTIONS, gradient["additive"], SHAPE_HELP),
        html.Div(_number("gradient.additive_scale", gradient["additive_scale"]), id=ADDITIVE_SCALE_WRAP,
                 style=additive_scale),
        _select(ids.OBJ_GRAD_MULTIPLIER, MULTIPLIER_LABEL, MULTIPLIER_OPTIONS, gradient["multiplier"],
                MULTIPLIER_HELP),
        html.Div(_number("gradient.mult_scale", gradient["mult_scale"]), id=MULT_SCALE_WRAP, style=mult_scale),
        html.Div([_number("gradient.sig_center_deg", gradient["sig_center_deg"]),
                  _number("gradient.sig_steepness", gradient["sig_steepness"]),
                  _number("gradient.sig_max_penalty", gradient["sig_max_penalty"])],
                 id=SIGMOID_WRAP, style=sigmoid),
        html.Details([
            html.Summary("Look-up table (advanced)", className="small", style={"cursor": "pointer"}),
            _number("gradient.bin_width_pct", gradient["bin_width_pct"]),
            _number("gradient.s_max_pct", gradient["s_max_pct"]),
        ], className="mb-1"),
    ]
    return html.Fieldset(
        dbc.Accordion([dbc.AccordionItem(fields, title="Slope response")], start_collapsed=True, className="mt-1"),
        id=ids.OBJ_SLOPE_WRAP, disabled=states["slope_disabled"], title=states["slope_title"])


def _precision_block(value, states: Mapping[str, Any]):
    from ..layout import _group

    radios = dbc.RadioItems(id=ids.OBJ_PRECISION, value=value, className="small",
                            options=states["precision_options"])
    return html.Div(_group(PRECISION_LABEL, ids.OBJ_PRECISION, radios), id=ids.OBJ_PRECISION_WRAP,
                    title=states["precision_title"], className="mt-2")


def section() -> list:
    """The "Objective & terrain" card for the Routes tab: closed by default."""
    from ..layout import _collapsible_card

    defaults = service.default_settings()
    # a blank list value is "none" on the page
    gradient = {key: (value or "none") if key in CHOICE_KEYS else value
                for key, value in defaults["gradient"].items()}
    states = control_states(defaults, routing.DEFAULT_BACKEND, routing.default_algorithm(routing.DEFAULT_BACKEND))
    children = [
        html.Div(service.summary(defaults), id=ids.OBJ_STATUS, className=STATUS_CLASS_OK, role="status",
                 style={"whiteSpace": "pre-line"}, **{"aria-live": "polite"}),
        _presets(),
        _terrain_model(None, states),
        _weights(defaults["weights"], states),
        _layers_block(),
        _slope_block(gradient, states),
        _precision_block(defaults["weight_precision"], states),
    ]
    card = _collapsible_card("Objective & terrain", children, intro=INTRO, open=False)
    card.id = ids.OBJ_CARD
    return [card]


# --------------------------------------------------------------------------- first paint
def prefill(state, put: Callable[[str, str, Any], None]) -> None:
    """First-paint values from ``state.gui_settings["objective"]``.

    A blank project (no saved settings) puts nothing, so the layout's own defaults stay.
    """
    from ..callbacks.objective import layer_names

    saved = (getattr(state, "gui_settings", None) or {}).get("objective")
    if not isinstance(saved, Mapping) or not saved:
        return
    for (component_id, prop), value in zip(FORM_TARGETS, form_props(saved, layer_names(state))):
        put(component_id, prop, value)
