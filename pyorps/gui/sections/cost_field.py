"""PYORPS GUI section: cost-field (siting) card (gap 3, package P11).

A collapsible card "Cost field" in the Routes tab, closed by default, after the constrained-routing card. It
shows what it costs to reach every cell of the search window from the fixed points (the sources, or every
control point of the route draft), as one map layer. Every label, option list, unit and limit comes from the
cost-field service (``services/cost_field``: ``ORIGIN_OPTIONS``, ``COMBINE_OPTIONS``, ``ALGORITHM_OPTIONS``,
``FIELD_LABELS``, ``UNIT``, ``UNIT_NOTE``, ``max_threads()``); nothing is typed twice here.

Top to bottom: the help text, the four settings (fixed points, combine rule, kernel, threads), the "Compute cost
field" button (disabled, with the reason as hover text, until the Backend field names a backend that can compute a
field and a cost raster is picked) next to the hidden Stop button, the status line and the best-cell readout.
``ids.CF_POLL`` and ``ids.CF_SETTINGS`` are already in the page.

Besides ``section()`` and ``prefill()`` the module holds the pieces the callbacks share: ``SETTING_IDS`` (setting
key -> control id), ``CONTROLS`` (problem key -> the component a "Go fix" button scrolls to), ``FORM_TARGETS``,
``form_values`` and ``readiness``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import dash_bootstrap_components as dbc
from dash import html

from .. import ids
from ..services import cost_field as service
from ..services import routing

#: wrapper of the Compute button: a disabled button shows no hover text of its own, so the reason sits on this div
RUN_WRAP = ids.CF_RUN_WRAP

#: setting key -> the control that edits it (the order of ``service.COST_FIELD_KEYS``)
SETTING_IDS: dict[str, str] = {
    "origins": ids.CF_ORIGINS,
    "combine": ids.CF_COMBINE,
    "algorithm": ids.CF_ALGORITHM,
    "threads": ids.CF_THREADS,
}
assert tuple(SETTING_IDS) == service.COST_FIELD_KEYS

#: problem key -> the component the "Go fix" button of a notice takes the user to. Four of the problem keys of the
#: service belong to other cards of the Routes tab: the Backend field, the neighbourhood field, the search buffer
#: (also the advice of a memory problem) and the raster dropdown.
CONTROLS: dict[str, str] = {
    **SETTING_IDS,
    "graph_api": ids.GRAPH_API,
    "neighborhood": ids.NEIGHBORHOOD,
    "search_buffer_m": ids.SEARCH_BUFFER,
    "memory": ids.SEARCH_BUFFER,
    "raster": ids.ROUTE_RASTER,
}

#: every component property the four settings fill, in the order of ``service.COST_FIELD_KEYS``
FORM_TARGETS: tuple[tuple[str, str], ...] = tuple((SETTING_IDS[key], "value") for key in service.COST_FIELD_KEYS)

INTRO = ("Shows what it costs to reach every cell around the fixed points, as a map layer. "
         "Use it to find a cheap place for one station that connects several points.")

HELP = (
    "The fixed points are the sources of the route draft, or all of its control points. For every cell the "
    "field gives the cost of the cheapest way to reach it from a fixed point. 'Sum over the points' adds that "
    "cost over all fixed points: the cheapest cell is where one station that connects all of them would cost "
    "least. 'Lowest of the points' keeps the cost from the nearest fixed point instead: a catchment map. "
    f"The unit is {service.UNIT}: {service.UNIT_NOTE}. "
    "Cells that cannot be reached, and forbidden cells, are shown as no data (transparent).")

HELP_SCOPE = (
    "The field is computed on the plain cost raster picked under 'Route on raster'. The objective and terrain "
    "card is not applied. It reads the backend, neighbourhood order, search buffer, whole-raster switch and "
    "forbidden-cells switch of 'Algorithm & search', and needs a backend that can compute a field "
    f"({', '.join(service.FIELD_BACKENDS)}). It runs in the background and the page stays usable. Stop acts "
    "between fixed points: the fixed point being solved always finishes, and a stopped run adds no layer.")

#: the hover texts of the controls
TIP_ORIGINS = ("Which points of the route draft the field starts from: the sources only, or every control point "
               f"(sources, targets and waypoints). Allowed: 1 to {service.MAX_ORIGINS} points, because every "
               "point needs one full search of the window.")
TIP_COMBINE = ("Sum over the points: the cost from every fixed point is added, so the cheapest cell is the best "
               "place for one station that connects them all (siting). Lowest of the points: the cost from the "
               "nearest fixed point, a catchment map.")
TIP_ALGORITHM = ("auto picks the fastest full-field kernel of the backend. dijkstra is the serial solver and "
                 "the only one that applies a terrain model. The GPU backends choose their kernel themselves.")


def tip_threads() -> str:
    return ("Worker threads of the parallel CPU kernel (cython backend). 0 = automatic: the library takes three "
            f"quarters of the cores. Allowed: whole numbers from 0 to {service.max_threads()} threads; every core "
            "is not offered, because the parallel kernel gets very slow when it uses all of them. "
            "Not used by dijkstra or the GPU backends.")


# --------------------------------------------------------------------------- values and readiness
def form_values(settings: Any) -> tuple:
    """The values of :data:`FORM_TARGETS` for stored settings (``{}`` or ``None``: the service defaults)."""
    cfg = {**service.COST_FIELD_DEFAULTS, **service.normalise_cost_field_settings(settings)}
    return tuple(cfg[key] for key in service.COST_FIELD_KEYS)


def readiness(graph_api: Any, raster_picked: bool) -> tuple[bool, str]:
    """``(ready, text)``: may the Compute button be clicked, and what its hover text says.

    Ready means the Backend field names a backend that can compute a field (and is installed) and a cost
    raster is picked. Everything else (the fixed points, the settings, the raster file) is checked by the
    click itself, which names what is wrong.
    """
    problems = service.check_backend(graph_api or routing.DEFAULT_BACKEND)
    if problems:
        return False, problems[0].message
    if not raster_picked:
        return False, ("Pick a cost raster under 'Route on raster' first: the cost field is computed on it "
                       "(build or load one in the Raster tab).")
    return True, "Compute the cost field of the fixed points on the picked cost raster."


# --------------------------------------------------------------------------- the controls
def _radio(key: str, options: list[dict], value, tip: str):
    from ..layout import _group, _help

    control = dbc.RadioItems(id=SETTING_IDS[key], value=value, options=[dict(o) for o in options],
                             className="small")
    return _help(_group(service.label_for(key), SETTING_IDS[key], control), tip, className="mb-1")


def _kernel(value):
    from ..layout import _help, _lbl

    return _help([_lbl(service.label_for("algorithm"), ids.CF_ALGORITHM),
                  dbc.Select(id=ids.CF_ALGORITHM, size="sm", value=value,
                             options=[dict(o) for o in service.ALGORITHM_OPTIONS])],
                 TIP_ALGORITHM, className="mb-1")


def _threads(value):
    from ..layout import _help, _lbl

    return _help([_lbl(f"{service.label_for('threads')} (0 = automatic)", ids.CF_THREADS),
                  dbc.Input(id=ids.CF_THREADS, type="number", size="sm", value=value, min=0,
                            max=service.max_threads(), step=1, debounce=True)],
                 tip_threads(), className="mb-1")


def section() -> list:
    """The "Cost field" card for the Routes tab: closed by default."""
    from ..layout import _collapsible_card

    values = dict(zip(service.COST_FIELD_KEYS, form_values({})))
    ready, hint = readiness(routing.DEFAULT_BACKEND, False)
    children = [
        html.Small(HELP, className="text-muted d-block mb-1"),
        html.Small(HELP_SCOPE, className="text-muted d-block mb-1"),
        _radio("origins", service.ORIGIN_OPTIONS, values["origins"], TIP_ORIGINS),
        _radio("combine", service.COMBINE_OPTIONS, values["combine"], TIP_COMBINE),
        _kernel(values["algorithm"]),
        _threads(values["threads"]),
        html.Div([
            html.Div(dbc.Button("Compute cost field", id=ids.CF_RUN_BTN, size="sm", color="primary",
                                className="gui-cta", disabled=not ready),
                     id=RUN_WRAP, title=hint),
            dbc.Button("⏹ Stop", id=ids.CF_STOP_BTN, size="sm", color="danger", style={"display": "none"},
                       title="Stop the cost field after the fixed point being solved; a stopped run adds "
                             "no layer"),
        ], className="d-flex align-items-center gap-2 mt-1"),
        html.Div(id=ids.CF_STATUS, className="small mt-1", role="status", style={"whiteSpace": "pre-line"},
                 **{"aria-live": "polite"}),
        html.Div(id=ids.CF_BEST_INFO, className="small fw-semibold mt-1", role="status",
                 **{"aria-live": "polite"}),
    ]
    card = _collapsible_card("Cost field", children, intro=INTRO, open=False)
    card.id = ids.CF_CARD
    return [card]


# --------------------------------------------------------------------------- first paint
def prefill(state, put: Callable[[str, str, Any], None]) -> None:
    """First-paint values from ``state.gui_settings["cost_field"]``.

    A blank project (no saved settings) puts nothing, so the layout's own defaults stay.
    """
    settings = (getattr(state, "gui_settings", None) or {}).get("cost_field")
    saved = service.normalise_cost_field_settings(settings if isinstance(settings, Mapping) else None)
    if not saved:
        return
    for (component_id, prop), value in zip(FORM_TARGETS, form_values(saved)):
        put(component_id, prop, value)
    put(ids.CF_SETTINGS, "data", dict(saved))
