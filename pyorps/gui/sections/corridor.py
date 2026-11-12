"""PYORPS GUI section: shared-trench corridor-graph card (gap 4, package P12).

A collapsible card "Corridor graph" in the Routes tab, closed by default, after the constrained-routing and
cost-field cards. Every label, unit, limit, option and help text comes from the specs of the corridor service
(``services/corridor.FIELD_SPECS``); the one thing the specs do not hold is the wording of the two options of
"Points to connect" (they carry the values ``all`` and ``sources_targets`` only), which :data:`TERMINAL_LABELS`
names.

Top to bottom: the help text, the five settings, the "Build corridor graph" button (disabled, with the reason as
hover text, until the Backend field is cython and the route draft has two points), the status line, and the
"Save as GeoPackage" row. ``ids.COR_POLL`` and ``ids.COR_SETTINGS`` are already in the page.

Besides ``section()`` and ``prefill()`` the module holds the pieces the callbacks share: ``SETTING_IDS`` (setting
key -> control id), ``CONTROLS`` (problem key -> the component a "Go fix" button scrolls to), ``FORM_TARGETS``,
``form_values`` and ``readiness``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from pyorps.core.exceptions import InvalidInputError

from .. import ids
from ..services import corridor as service

#: wrapper of the Build button: a disabled button shows no hover text of its own, so the reason sits on this div
RUN_WRAP = ids.COR_RUN_WRAP

#: setting key -> the control that edits it (the order of ``service.SETTING_KEYS``)
SETTING_IDS: dict[str, str] = {
    "terminals": ids.COR_TERMINALS,
    "k_per_pair": ids.COR_K,
    "min_shared_length_m": ids.COR_MIN_SHARED,
    "max_pair_cost": ids.COR_MAX_PAIR_COST,
    "validate_pair_costs": ids.COR_VALIDATE,
}
assert tuple(SETTING_IDS) == service.SETTING_KEYS

#: problem key -> the component the "Go fix" button of a notice takes the user to. Four of the problem keys of
#: the service belong to other cards of the Routes tab: the draft's points table (points), the raster dropdown
#: (raster), the Backend field (graph_api) and the neighbourhood field (neighborhood_k).
CONTROLS: dict[str, str] = {
    **SETTING_IDS,
    "settings": ids.COR_CARD,
    "points": ids.BUILD_POINTS_TABLE,
    "raster": ids.ROUTE_RASTER,
    "graph_api": ids.GRAPH_API,
    "neighborhood_k": ids.NEIGHBORHOOD,
    "graph": ids.COR_SAVE_BTN,
    "save_path": ids.COR_SAVE_PATH,
}

#: every component property the five settings fill, in the order of ``service.SETTING_KEYS``
FORM_TARGETS: tuple[tuple[str, str], ...] = tuple((SETTING_IDS[key], "value") for key in service.SETTING_KEYS)

#: the words for the two values of "Points to connect"
TERMINAL_LABELS: dict[str, str] = {
    service.TERMINALS_ALL: "All control points",
    service.TERMINALS_ENDS: "Sources and targets only",
}
assert tuple(TERMINAL_LABELS) == service.TERMINAL_CHOICES

INTRO = ("Joins the points of the route draft by their cheapest routes and charges every trench section that "
         "several routes share only once.")

HELP = ("It is built on the plain cost raster chosen under 'Route on raster', on the cython backend only: no DEM, "
        "no objective weights and no constrained profile are used. It reads the neighbourhood order, the search "
        "buffer, the whole-raster switch and the forbidden-cells switch of 'Algorithm & search'. The run cannot be "
        "stopped, but the page stays usable while it runs.")

HELP_SAVE = ("The graph itself is not stored in the project file. After a project is reloaded (opened again, or the "
             "GUI started with --project) its map layers are back, but build the graph again before you save it as "
             "a GeoPackage.")


# --------------------------------------------------------------------------- values and readiness
def form_values(settings: Any) -> tuple:
    """The values of :data:`FORM_TARGETS` for stored settings (``{}`` or ``None``: the spec defaults)."""
    cfg = {**service.CORRIDOR_DEFAULTS, **service.normalise_corridor_settings(settings)}
    return tuple(cfg[key] for key in service.SETTING_KEYS)


def readiness(graph_api: Any, draft: Any, terminals: Any) -> tuple[bool, str]:
    """``(ready, text)``: may the Build button be clicked, and what its hover text says.

    Ready means the Backend field is cython and the route draft gives two points for the chosen "Points to
    connect". Everything else (the raster, the cell positions, the settings) is checked by the click itself, which
    names what is wrong.
    """
    problem = service.backend_problem(graph_api)
    if problem is not None:
        return False, problem[1]
    which = terminals if terminals in service.TERMINAL_CHOICES else service.TERMINALS_ALL
    try:
        count = len(service.terminals_from_draft(draft, which))
    except InvalidInputError:
        count = 0
    if count < 2:
        return False, (f"A corridor graph connects at least 2 points, but the route draft gives {count}. Add more "
                       f"points on the map (sources, targets and, with '{TERMINAL_LABELS[service.TERMINALS_ALL]}', "
                       f"waypoints).")
    return True, f"Build the shared-trench corridor graph of the {count} points of the route draft."


# --------------------------------------------------------------------------- the controls
def _caption(spec: service.SettingSpec) -> str:
    """The label with its unit in brackets."""
    return f"{spec.label} ({spec.unit})" if spec.unit else spec.label


def _tooltip(spec: service.SettingSpec) -> str:
    if spec.kind == "number":
        return f"{spec.help} Allowed: {service.describe_range(spec)}."
    return spec.help


def _step(spec: service.SettingSpec):
    """The ``step`` of a number input: whole numbers step by 1, everything else takes any decimal."""
    return 1 if spec.whole else "any"


def _number(key: str, value):
    from ..layout import _help, _lbl

    spec = service.FIELD_SPECS[key]
    component_id = SETTING_IDS[key]
    control = dbc.Input(
        id=component_id, type="number", size="sm", value=value, min=spec.minimum, max=spec.maximum,
        step=_step(spec), debounce=True,
        placeholder=f"blank: {spec.blank_means}" if not spec.required and spec.blank_means else None)
    return _help([_lbl(_caption(spec), component_id), control], _tooltip(spec), className="mb-1")


def _terminals(value):
    from ..layout import _group, _help

    spec = service.FIELD_SPECS["terminals"]
    radios = dbc.RadioItems(
        id=ids.COR_TERMINALS, value=value, className="small",
        options=[{"label": f" {TERMINAL_LABELS[option]}", "value": option} for option in spec.options])
    return _help(_group(spec.label, ids.COR_TERMINALS, radios), _tooltip(spec), className="mb-1")


def _validate(value):
    from ..layout import _help

    spec = service.FIELD_SPECS["validate_pair_costs"]
    return _help(dbc.Switch(id=ids.COR_VALIDATE, value=bool(value), label=spec.label, className="small"),
                 _tooltip(spec), className="mb-1")


def section() -> list:
    """The "Corridor graph" card for the Routes tab: closed by default."""
    from ..layout import _browse, _collapsible_card, _lbl

    values = dict(zip(service.SETTING_KEYS, form_values({})))
    ready, hint = readiness(None, None, values["terminals"])
    children = [
        html.Small(HELP, className="text-muted d-block mb-1"),
        _terminals(values["terminals"]),
        _number("k_per_pair", values["k_per_pair"]),
        _number("min_shared_length_m", values["min_shared_length_m"]),
        _number("max_pair_cost", values["max_pair_cost"]),
        _validate(values["validate_pair_costs"]),
        html.Div(
            dbc.Button("Build corridor graph", id=ids.COR_RUN_BTN, size="sm", color="primary", className="gui-cta",
                       disabled=not ready),
            id=RUN_WRAP, title=hint, className="mt-1"),
        html.Div(id=ids.COR_STATUS, className="small mt-1", role="status", style={"whiteSpace": "pre-line"},
                 **{"aria-live": "polite"}),
        html.Small(HELP_SAVE, className="text-muted d-block mt-2"),
        _lbl("Save as GeoPackage (.gpkg)", ids.COR_SAVE_PATH),
        dbc.InputGroup([
            dbc.Input(id=ids.COR_SAVE_PATH, size="sm", placeholder="Save the latest corridor graph as .gpkg"),
            _browse(ids.COR_SAVE_PATH, "the corridor graph file"),
            dbc.Button("Save graph", id=ids.COR_SAVE_BTN, size="sm", color="secondary", outline=True,
                       title="Write the latest corridor graph as a GeoPackage with the layers "
                             f"{service.SEGMENTS_LAYER} and {service.NODES_LAYER}"),
        ], size="sm"),
    ]
    card = _collapsible_card("Corridor graph", children, intro=INTRO, open=False)
    card.id = ids.COR_CARD
    return [card]


# --------------------------------------------------------------------------- first paint
def prefill(state, put: Callable[[str, str, Any], None]) -> None:
    """First-paint values from ``state.gui_settings["corridor"]``.

    A blank project (no saved settings) puts nothing, so the layout's own defaults stay.
    """
    settings = (getattr(state, "gui_settings", None) or {}).get("corridor")
    saved = service.normalise_corridor_settings(settings if isinstance(settings, Mapping) else None)
    if not saved:
        return
    for (component_id, prop), value in zip(FORM_TARGETS, form_values(saved)):
        put(component_id, prop, value)
    put(ids.COR_SETTINGS, "data", dict(saved))
