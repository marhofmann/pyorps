"""
PYORPS GUI callbacks: attribute inspection (C5).

Clicking any feature of a rendered GeoJSON layer shows its properties in the
Attributes tab. Feature clicks and map clicks can both fire; this callback
only reads the feature payload, so no disambiguation is needed here (the map
dispatcher handles click *modes* separately).
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids


#: technical column name -> human-readable label (routes carry the metric
#: columns; anything not listed keeps its own name)
FIELD_LABELS = {
    "name": "Name",
    "total_length_m": "Total Length (in m)",
    "geodesic_length_m": "Geodesic Length (in m)",
    "total_cost": "Total Cost",
    "total_cell_cost": "Raw Cell Cost",
    "n_forbidden_cells": "Forbidden Cells Crossed",
    "crosses_forbidden": "Crosses Forbidden Area",
    "osm_id": "OSM ID",
    "osm_type": "OSM Element Type",
    "nutzart": "Land Use (nutzart)",
    "bez": "Sub-Type (bez)",
}


def field_label(key: str) -> str:
    return FIELD_LABELS.get(str(key), str(key))


def format_value(value) -> str:
    """Round numbers to 2 digits (thousands-separated), tidy the rest."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    # strings stay verbatim — "35390" may be a postal code, not a number
    return str(value).strip()


def properties_table(properties: dict, title: str = "") -> list:
    rows = [html.Tr([html.Td(html.B(field_label(k)), className="pe-2"),
                     html.Td(format_value(v))])
            for k, v in sorted((properties or {}).items())]
    header = [html.Div(title, className="fw-bold small mb-1")] if title else []
    if not rows:
        return header + [html.Small("No attributes on this feature.",
                                    className="text-muted")]
    return header + [dbc.Table(html.Tbody(rows), size="sm", striped=True,
                               className="small mb-0")]


def register(app, state) -> None:
    @app.callback(
        Output(ids.ATTR_PANEL, "children"),
        Output(ids.SELECTED_FEATURE_LAYER, "data"),
        Output(ids.SELECTED_FEATURE, "data"),
        Input({"type": ids.TYPE_LAYER_GEOJSON, "id": ALL}, "clickData"),
        State(ids.LAYERS_GRID, "selectedRows"),
        prevent_initial_call=True)
    def inspect_feature(click_datas, layer_selection):  # pylint: disable=unused-argument  # Dash callback input
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        trigger = ctx.triggered_id
        if not trigger:
            raise PreventUpdate
        # find the value that fired (triggered inputs carry the new value)
        feature = None
        for item in ctx.triggered:
            if item.get("value"):
                feature = item["value"]
                break
        if not feature:
            raise PreventUpdate
        clicked_id = trigger.get("id")
        layer = state.get(clicked_id)
        title = layer.name if layer else ""
        properties = (feature.get("properties")
                      if isinstance(feature, dict) else None) or {}
        attrs = properties_table(properties, title)

        # Feature 3: if this feature belongs to the layer selected in the
        # Layers grid, select + highlight it; otherwise just show attributes.
        selected_layer_id = (layer_selection[0].get("id")
                             if layer_selection else None)
        if (selected_layer_id and clicked_id == selected_layer_id
                and isinstance(feature, dict)):
            highlight = {"type": "FeatureCollection", "features": [feature]}
            return (attrs, highlight,
                    {"layer_id": clicked_id, "props": properties})
        return attrs, no_update, no_update
