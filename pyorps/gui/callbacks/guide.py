"""
PYORPS GUI callbacks: the workflow guide + step gating (task 57).

Some steps only make sense once earlier ones are done: a cost raster needs a
dataset with cost assumptions; routing needs a cost raster. This module (1)
renders an ordered, ticking checklist so the user always knows the order and
what is next, and (2) greys out the buttons whose prerequisites are not yet met
so a dead-end click is impossible.
"""
from __future__ import annotations

from dash import Input, Output, html

from .. import ids
from ..layout import WORKFLOW_STEPS


#: compact step labels for the progress rail (full titles in the tooltip)
_SHORT_LABELS = ["Area", "Data", "Costs", "Raster", "Routes"]


def _guide_children(done: list[bool]) -> html.Div:
    """The workflow rendered as a horizontal stepper / progress rail."""
    steps = []
    next_marked = False
    for i, (title, hint) in enumerate(WORKFLOW_STEPS):
        if done[i]:
            cls, dot = "wf-done", "✓"
        elif not next_marked:
            cls, dot = "wf-current", str(i + 1)
            next_marked = True
        else:
            cls, dot = "wf-todo", str(i + 1)
        label = _SHORT_LABELS[i] if i < len(_SHORT_LABELS) else title
        steps.append(html.Div([
            html.Div(dot, className="wf-dot"),
            html.Div(label, className="wf-label"),
        ], className=f"wf-step {cls}", title=f"{i + 1}. {title} — {hint}"))
    return html.Div(steps, className="wf-stepper",
                    title="Workflow — do these in order")


def register(app, state) -> None:
    @app.callback(
        Output(ids.WORKFLOW_GUIDE, "children"),
        Output(ids.COST_SEED_BTN, "disabled"),
        Output(ids.RASTERIZE_BTN, "disabled"),
        Output(ids.RUN_ROUTING_BTN, "disabled"),
        Output(ids.RASTER_COMBINE_BTN, "disabled"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.COST_GRID_STATE, "data"),
        Input(ids.COST_DATASET, "value"),
        Input(ids.COST_FEATURE_KEYS, "value"),
        Input(ids.ROUTE_RASTER, "value"))
    def sync_workflow_and_gating(_view, grid_state, cost_dataset,
                                 feature_keys, route_raster):  # pylint: disable=unused-argument  # Dash passes every Input
        vectors = state.layers_of_kind("vector")
        rasters = state.layers_of_kind("raster")
        routes = state.layers_of_kind("route")

        has_area = state.study_area is not None
        has_data = len(vectors) > 0
        cost_seeded = bool(grid_state and grid_state.get("dataset_id")
                           and grid_state.get("feature_keys"))
        has_raster = len(rasters) > 0
        has_routes = len(routes) > 0
        done = [has_area, has_data, cost_seeded, has_raster, has_routes]

        seed_disabled = not (cost_dataset and feature_keys)
        rasterize_disabled = not cost_seeded
        run_disabled = not has_raster
        combine_disabled = len(rasters) < 2
        return (_guide_children(done), seed_disabled, rasterize_disabled,
                run_disabled, combine_disabled)
