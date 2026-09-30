"""
PYORPS GUI callbacks: the cost-model editor (R3, R5 — the centerpiece).

Feature-key selection (single column or ordered combination), the AG-Grid
cost table with a first-class Forbidden flag (F11) and coverage validation
(F4), the ordered modifier list (F12), and table import/export.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import cost_model, validate
from ..services.errors import Notice, guard, success


def _vector_options(state) -> list[dict]:
    return [{"label": ly.name, "value": ly.id}
            for ly in state.layers_of_kind("vector")]


def register_cost_table(state, *, dataset_id, feature_keys, rows,
                        name: str | None = None) -> str:
    """Register/refresh a cost table so the Raster tab can pair it with any
    COMPATIBLE dataset (feature columns must exist there). Seeding and
    importing write here; cell edits keep the entry fresh via `coverage`."""
    keys = tuple(feature_keys or ())
    layer = state.get(dataset_id) if dataset_id else None
    table_id = f"tbl-{dataset_id or 'imported'}-{'-'.join(keys)}"
    existing = state.cost_tables.get(table_id) or {}
    label = name or existing.get("name") or (
        f"{layer.name}: {' × '.join(keys)}" if layer
        else f"Imported: {' × '.join(keys)}")
    state.cost_tables[table_id] = {
        "name": label, "dataset_id": dataset_id,
        "feature_keys": list(keys), "rows": list(rows or [])}
    return table_id


def _coverage_text(state, dataset_id, feature_keys, rows) -> str:
    layer = state.get(dataset_id) if dataset_id else None
    if layer is None or not feature_keys:
        return ""
    assumptions = cost_model.assumptions_from_grid_rows(
        rows or [], tuple(feature_keys))
    missing = validate.uncovered_categories(
        layer.gdf, tuple(feature_keys), assumptions)
    if not missing:
        return "✓ every category is covered (or has a \"\" catch-all)."
    shown = ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
    return (f"⚠ {len(missing)} categories without cost → forbidden (F4): "
            f"{shown}")


def register(app, state) -> None:
    # -------------------------------------------- dataset options follow layers
    @app.callback(Output(ids.COST_DATASET, "options"),
                  Output(ids.ROUTE_RASTER, "options"),
                  Output(ids.RASTER_COMBINE_SELECT, "options"),
                  Output(ids.MERGE_SELECT, "options"),
                  Output(ids.PPB_DATASET, "options"),
                  Input(ids.LAYERS_VIEW, "data"))
    def sync_options(_view):
        raster_options = [{"label": ly.name, "value": ly.id}
                          for ly in state.layers_of_kind("raster")]
        vector_options = _vector_options(state)
        return (vector_options, raster_options, raster_options,
                vector_options, vector_options)

    def _vector_columns():
        return sorted({
            c for ly in state.layers_of_kind("vector") if ly.gdf is not None
            for c in ly.gdf.columns if c != ly.gdf.geometry.name})

    @app.callback(Output(ids.MODIFIER_GRID, "columnDefs"),
                  Input(ids.LAYERS_VIEW, "data"),
                  State(ids.MODIFIER_GRID, "columnDefs"))
    def sync_modifier_editor(_view, column_defs):
        names = [ly.name for ly in state.layers_of_kind("vector")]
        # a leading "" lets a condition be left blank (no column selected)
        columns = [""] + _vector_columns()
        column_defs = column_defs or []
        for col in column_defs:
            field = col.get("field")
            if field == "dataset":
                col["cellEditorParams"] = {"values": names}
            elif field == "column":
                col["cellEditorParams"] = {"values": columns}
        return column_defs

    # ---------------------------------------- preprocessing steps (Feature 4)
    @app.callback(Output(ids.PREPROC_GRID, "columnDefs"),
                  Input(ids.LAYERS_VIEW, "data"),
                  State(ids.PREPROC_GRID, "columnDefs"))
    def sync_preproc_editor(_view, column_defs):
        names = [ly.name for ly in state.layers_of_kind("vector")]
        columns = [""] + _vector_columns()
        for col in column_defs or []:
            field = col.get("field")
            if field == "dataset":
                col["cellEditorParams"] = {"values": names}
            elif field in ("column", "target"):
                col["cellEditorParams"] = {"values": columns}
        return column_defs or []

    @app.callback(
        Output(ids.PREPROC_GRID, "rowData", allow_duplicate=True),
        Input(ids.PREPROC_ADD_BTN, "n_clicks"),
        State(ids.PREPROC_GRID, "rowData"),
        prevent_initial_call=True)
    def add_preproc_step(n_clicks, rows):
        if not n_clicks:
            raise PreventUpdate
        return (rows or []) + [{"dataset": "", "op": "buffer", "column": "",
                                "operator": "all", "value": "",
                                "target": "", "arg": "0", "condition": ""}]

    @app.callback(
        Output(ids.PREPROC_GRID, "rowData", allow_duplicate=True),
        Input(ids.PREPROC_DEL_BTN, "n_clicks"),
        State(ids.PREPROC_GRID, "rowData"),
        State(ids.PREPROC_GRID, "selectedRows"),
        prevent_initial_call=True)
    def delete_preproc_step(n_clicks, rows, selected):
        import json

        if not n_clicks or not selected:
            raise PreventUpdate
        # json keys, not tuple(sorted(items())): builder rows carry a LIST of
        # conditions, and lists are unhashable
        drop = {json.dumps(r, sort_keys=True, default=str) for r in selected}
        return [r for r in (rows or [])
                if json.dumps(r, sort_keys=True, default=str) not in drop]

    # ---------------- condition-group step builder (round 11): N conditions
    # combined by & / |, python-like preview, appended as ONE step
    @app.callback(Output(ids.PPB_COLUMN, "options"),
                  Output(ids.PPB_COLUMN, "value"),
                  Input(ids.PPB_DATASET, "value"))
    def ppb_columns(dataset_id):
        layer = state.get(dataset_id) if dataset_id else None
        if layer is None or layer.gdf is None:
            return [], None
        cols = [c for c in layer.gdf.columns if c != layer.gdf.geometry.name]
        return [{"label": c, "value": c} for c in cols], None

    @app.callback(Output(ids.PPB_VALUE, "options"),
                  Output(ids.PPB_VALUE, "value"),
                  Input(ids.PPB_COLUMN, "value"),
                  State(ids.PPB_DATASET, "value"))
    def ppb_values(column, dataset_id):
        """The values PRESENT in the picked column — a searchable dropdown
        (dcc.Dropdown filters as you type, so even hundreds of distinct
        values stay findable)."""
        layer = state.get(dataset_id) if dataset_id else None
        if layer is None or layer.gdf is None or not column \
                or column not in layer.gdf.columns:
            return [], None
        values = layer.gdf[column].dropna().astype(str).str.strip()
        uniques = sorted(v for v in values.unique() if v)[:2000]
        return [{"label": v, "value": v} for v in uniques], None

    @app.callback(
        Output(ids.PPB_CONDS, "data"),
        Input(ids.PPB_ADD_COND_BTN, "n_clicks"),
        State(ids.PPB_COLUMN, "value"),
        State(ids.PPB_OPERATOR, "value"),
        State(ids.PPB_VALUE, "value"),
        State(ids.PPB_CONDS, "data"),
        prevent_initial_call=True)
    def ppb_add_condition(n_clicks, column, operator, value, conds):
        if not n_clicks or not column:
            raise PreventUpdate
        conds = list(conds or [])
        conds.append({"column": column, "operator": operator or "==",
                      "value": value})
        return conds

    @app.callback(
        Output(ids.PPB_CONDS, "data", allow_duplicate=True),
        Input({"type": ids.TYPE_PPB_COND_REMOVE, "index": ALL}, "n_clicks"),
        State(ids.PPB_CONDS, "data"),
        prevent_initial_call=True)
    def ppb_remove_condition(clicks, conds):
        trigger = ctx.triggered_id
        if not trigger or not any(clicks or []):
            raise PreventUpdate
        conds = list(conds or [])
        index = int(trigger.get("index"))
        if not 0 <= index < len(conds):
            raise PreventUpdate
        conds.pop(index)
        return conds

    @app.callback(
        Output(ids.PPB_COND_LIST, "children"),
        Output(ids.PPB_PREVIEW, "children"),
        Input(ids.PPB_CONDS, "data"),
        Input(ids.PPB_COMBINE, "value"),
        Input(ids.PPB_OP, "value"),
        Input(ids.PPB_TARGET, "value"),
        Input(ids.PPB_ARG, "value"))
    def ppb_preview(conds, combine, op, target, arg):
        from ..presets import step_label

        conds = list(conds or [])
        chips = []
        for i, cond in enumerate(conds):
            text = cost_model.condition_label(
                cond.get("column"), cond.get("operator"), cond.get("value"))
            chips.append(html.Div([
                html.Span(text, className="me-1"),
                dbc.Button("×", id={"type": ids.TYPE_PPB_COND_REMOVE,
                                    "index": i},
                           size="sm", color="link",
                           className="p-0 text-danger",
                           title="Remove this condition"),
            ], className="d-inline-block border rounded px-1 me-1 mb-1"))
        if not chips:
            chips = [html.Small("No conditions yet — the step would apply "
                                "to ALL features.", className="text-muted")]
        step = {"conditions": conds, "combine": combine or "&",
                "op": op or "buffer", "target": target, "arg": arg}
        return chips, step_label(step)

    @app.callback(
        Output(ids.PREPROC_GRID, "rowData", allow_duplicate=True),
        Output(ids.PPB_CONDS, "data", allow_duplicate=True),
        Input(ids.PPB_ADD_STEP_BTN, "n_clicks"),
        State(ids.PPB_CONDS, "data"),
        State(ids.PPB_COMBINE, "value"),
        State(ids.PPB_OP, "value"),
        State(ids.PPB_TARGET, "value"),
        State(ids.PPB_ARG, "value"),
        State(ids.PPB_DATASET, "value"),
        State(ids.PREPROC_GRID, "rowData"),
        prevent_initial_call=True)
    def ppb_add_step(n_clicks, conds, combine, op, target, arg, dataset_id,
                     rows):
        from ..presets import step_label

        if not n_clicks:
            raise PreventUpdate
        layer = state.get(dataset_id) if dataset_id else None
        step = {
            "dataset": layer.name if layer else "",
            "op": op or "buffer",
            "conditions": list(conds or []),
            "combine": combine or "&",
            "target": target or "", "arg": arg,
            # the legacy triple stays blank — the group drives the mask
            "column": "", "operator": "all", "value": "",
        }
        step["condition"] = step_label(step)
        return (rows or []) + [step], []

    # ---------------- custom cost polygons: snapshot + per-polygon edit (33)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MANUAL_GRID, "rowData", allow_duplicate=True),
        Output(ids.MANUAL_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.MANUAL_CREATE_BTN, "n_clicks"),
        State(ids.DRAW_CONTROL, "geojson"),
        State(ids.MANUAL_NAME, "value"),
        State(ids.MANUAL_COST, "value"),
        State(ids.MANUAL_MODE, "value"),
        State(ids.MANUAL_GRID, "rowData"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def create_manual_layer(n_clicks, drawn, name, cost, mode, m_rows, notices):
        from ..services import manual_cost

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        features = (drawn or {}).get("features") or []
        display = (name or "Manual costs").strip() or "Manual costs"
        result, notices = guard(
            manual_cost.sync_manual_layer, state, features,
            project_crs=state.project_crs,
            default_cost=float(cost if cost is not None else 65535),
            mode=mode or "override", name=display, prev_rows=m_rows,
            notices=notices)
        if result is None:
            return no_update, no_update, no_update, notices
        layer, rows = result
        if layer is None:
            notices.append(Notice(
                severity="warning", title="No polygons drawn",
                meaning="Set drawing to 'cost polygons' and draw at least one.",
                impact="No cost layer was created.",
                fix="Draw a rectangle/polygon, then Snapshot.",
                focus_id=ids.TAB_COST).to_dict())
            return no_update, [], "Draw cost polygons on the map.", notices
        notices.append(success(
            f"Custom cost layer '{display}' — {len(rows)} polygon(s)",
            meaning="Edit names/costs in the table; use it as a Cost dataset "
                    "or a per-feature override modifier (Column = cost)."))
        status = (f"Created '{display}' with {len(rows)} polygon(s). "
                  "Edit names/costs in the table.")
        return state.layers_view(), rows, status, notices

    # remove selected polygon(s) from the manual cost layer via the list (62)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MANUAL_GRID, "rowData", allow_duplicate=True),
        Output(ids.MANUAL_STATUS, "children", allow_duplicate=True),
        Input(ids.MANUAL_DEL_BTN, "n_clicks"),
        State(ids.DRAW_CONTROL, "geojson"),
        State(ids.MANUAL_GRID, "rowData"),
        State(ids.MANUAL_GRID, "selectedRows"),
        State(ids.MANUAL_NAME, "value"),
        State(ids.MANUAL_COST, "value"),
        State(ids.MANUAL_MODE, "value"),
        prevent_initial_call=True)
    def remove_manual_selected(n_clicks, drawn, rows, selected, name, cost,
                               mode):
        from ..services import manual_cost

        if not n_clicks or not selected:
            raise PreventUpdate
        layer = state.get(manual_cost.MANUAL_LAYER_ID)
        wkts = (layer.meta or {}).get("wgs84_wkts") if layer else None
        if layer is None or not wkts:
            raise PreventUpdate
        removed = {int(r["__row"]) for r in selected if r.get("__row") is not None}
        for idx in removed:
            if 0 <= idx < len(wkts):
                state.manual_excluded_geoms.add(wkts[idx])
        keep_rows = [r for r in (rows or []) if int(r["__row"]) not in removed]
        features = (drawn or {}).get("features") or []
        _layer, new_rows = manual_cost.sync_manual_layer(
            state, features, project_crs=state.project_crs,
            default_cost=float(cost if cost is not None else 65535),
            mode=mode or "override",
            name=(name or "Manual costs").strip() or "Manual costs",
            prev_rows=keep_rows)
        status = (f"{len(new_rows)} cost polygon(s) — removed {len(removed)}."
                  if new_rows else "All cost polygons removed.")
        return state.layers_view(), new_rows, status

    # per-polygon name/cost edits write back to the manual layer (task 33)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Input(ids.MANUAL_GRID, "cellValueChanged"),
        prevent_initial_call=True)
    def edit_manual_grid(events):
        from ..services import geo, manual_cost

        layer = state.get(manual_cost.MANUAL_LAYER_ID)
        if layer is None or layer.gdf is None or not events:
            raise PreventUpdate
        changed = False
        for event in events if isinstance(events, list) else [events]:
            data = event.get("data") or {}
            row = data.get("__row")
            column = (event.get("colId")
                      or event.get("column", {}).get("colId"))
            if row is None or column not in ("name", manual_cost.COST_COLUMN):
                continue
            if not (0 <= int(row) < len(layer.gdf)):
                continue
            value = data.get(column)
            if column == manual_cost.COST_COLUMN:
                value = cost_model.coerce_factor(value)
            layer.gdf.iat[int(row),
                          layer.gdf.columns.get_loc(column)] = value
            changed = True
        if not changed:
            raise PreventUpdate
        layer.geojson = geo.gdf_to_wgs84_geojson(layer.gdf)   # names on the map
        return state.layers_view()

    # ------------------------------------------------- propose feature columns
    @app.callback(Output(ids.COST_FEATURE_KEYS, "options"),
                  Output(ids.COST_FEATURE_KEYS, "value"),
                  Input(ids.COST_DATASET, "value"),
                  prevent_initial_call=True)
    def propose(dataset_id):
        layer = state.get(dataset_id) if dataset_id else None
        if layer is None or layer.gdf is None:
            raise PreventUpdate
        proposed, candidates = cost_model.propose_features(layer.gdf)
        counts, _ = cost_model.feature_analysis(layer.gdf, candidates)
        n_by_col = dict(counts)
        options = [
            {"label": f"{c} ({n_by_col.get(c, '?')} categories)",
             "value": c}
            for c in candidates]
        return options, list(proposed)

    # -------------------------------------- category counts / combination size
    @app.callback(Output(ids.COST_FEATURE_INFO, "children"),
                  Input(ids.COST_FEATURE_KEYS, "value"),
                  Input(ids.COST_DATASET, "value"))
    def feature_info(feature_keys, dataset_id):
        layer = state.get(dataset_id) if dataset_id else None
        if layer is None or layer.gdf is None or not feature_keys:
            return ""
        per_column, n_combo = cost_model.feature_analysis(
            layer.gdf, list(feature_keys))
        parts = [f"{col}: {n} categories" for col, n in per_column]
        if n_combo is not None:
            main, side = feature_keys[0], feature_keys[1]
            parts.append(f"combination {main} x {side}: {n_combo} distinct "
                         "pairs in the data (= cost-table rows before "
                         "catch-alls)")
        elif len(feature_keys or []) == 1:
            parts.append("single feature: one cost row per category")
        if len(feature_keys or []) > 2:
            parts.append("note: only the first two columns are used "
                         "(main + one side feature)")
        return html.Ul([html.Li(p) for p in parts],
                       className="small text-info mb-1")

    # ------------------------------------------------------------ seed the grid
    @app.callback(
        Output(ids.COST_GRID, "columnDefs"),
        Output(ids.COST_GRID, "rowData"),
        Output(ids.COST_GRID_STATE, "data"),
        Output(ids.COST_COVERAGE_INFO, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.COST_SEED_BTN, "n_clicks"),
        State(ids.COST_DATASET, "value"),
        State(ids.COST_FEATURE_KEYS, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def seed(n_clicks, dataset_id, feature_keys, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        layer = state.get(dataset_id) if dataset_id else None
        if layer is None or not feature_keys:
            notices.append(Notice(
                severity="warning", title="Pick a dataset and feature "
                "column(s) first",
                meaning="The cost table is seeded from a loaded vector "
                        "layer and its selected feature columns.",
                impact="No cost table was created.",
                fix="Select a base dataset and at least one feature column.",
                focus_id=ids.TAB_COST,
                focus_control=ids.COST_DATASET).to_dict())
            return no_update, no_update, no_update, no_update, notices
        keys = tuple(feature_keys)[:2]
        assumptions, notices = guard(cost_model.seed_assumptions, layer.gdf,
                                     keys, notices=notices)
        if assumptions is None:
            return no_update, no_update, no_update, no_update, notices
        rows = cost_model.grid_rows_from_assumptions(assumptions, keys)
        grid_state = {"dataset_id": dataset_id, "feature_keys": list(keys)}
        register_cost_table(state, dataset_id=dataset_id, feature_keys=keys,
                            rows=rows)
        seeded_from = ("notebook defaults"
                       if keys == cost_model.NOTEBOOK_KEYS
                       else "zero-cost template")
        notices.append(success(f"Cost table seeded ({seeded_from})",
                               meaning=f"{len(rows)} rows for "
                                       f"{' x '.join(keys)}."))
        coverage = _coverage_text(state, dataset_id, keys, rows)
        return (cost_model.grid_column_defs(keys), rows, grid_state,
                coverage, notices)

    # ----------------------------------------------------- add / delete rows
    @app.callback(
        Output(ids.COST_GRID, "rowData", allow_duplicate=True),
        Input(ids.COST_ADD_ROW_BTN, "n_clicks"),
        State(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID_STATE, "data"),
        prevent_initial_call=True)
    def add_row(n_clicks, rows, grid_state):
        if not n_clicks or not grid_state:
            raise PreventUpdate
        keys = tuple(grid_state.get("feature_keys") or [])
        row = {key: "" for key in keys}
        row.update({"cost": 0, "forbidden": False})
        return (rows or []) + [row]

    @app.callback(
        Output(ids.COST_GRID, "rowData", allow_duplicate=True),
        Input(ids.COST_DEL_ROW_BTN, "n_clicks"),
        State(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID, "selectedRows"),
        prevent_initial_call=True)
    def delete_rows(n_clicks, rows, selected):
        if not n_clicks or not selected:
            raise PreventUpdate
        drop = {tuple(sorted(r.items())) for r in selected}
        return [r for r in (rows or [])
                if tuple(sorted(r.items())) not in drop]

    # ------------------------------------------------ live coverage validation
    @app.callback(
        Output(ids.COST_COVERAGE_INFO, "children", allow_duplicate=True),
        Input(ids.COST_GRID, "cellValueChanged"),
        Input(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID_STATE, "data"),
        prevent_initial_call=True)
    def coverage(_event, rows, grid_state):
        if not grid_state:
            raise PreventUpdate
        # keep the registered table in sync with the live grid so the Raster
        # tab always rasterizes the edited values, never a stale snapshot
        register_cost_table(
            state, dataset_id=grid_state.get("dataset_id"),
            feature_keys=tuple(grid_state.get("feature_keys") or []),
            rows=rows)
        return _coverage_text(state, grid_state.get("dataset_id"),
                              tuple(grid_state.get("feature_keys") or []),
                              rows)

    # -------------------------------------------------------------- import
    @app.callback(
        Output(ids.COST_GRID, "columnDefs", allow_duplicate=True),
        Output(ids.COST_GRID, "rowData", allow_duplicate=True),
        Output(ids.COST_GRID_STATE, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.COST_IMPORT_BTN, "n_clicks"),
        State(ids.COST_IMPORT_PATH, "value"),
        State(ids.COST_DATASET, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def import_table(n_clicks, path, dataset_id, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        result, notices = guard(cost_model.import_table, path,
                                notices=notices)
        if result is None:
            return no_update, no_update, no_update, notices
        assumptions, keys = result
        rows = cost_model.grid_rows_from_assumptions(assumptions, keys)
        grid_state = {"dataset_id": dataset_id, "feature_keys": list(keys)}
        import os
        register_cost_table(state, dataset_id=dataset_id, feature_keys=keys,
                            rows=rows,
                            name=f"Imported {os.path.basename(str(path))}: "
                                 f"{' × '.join(keys)}")
        notices.append(success("Cost table imported",
                               meaning=f"{len(rows)} rows, feature key "
                                       f"{' x '.join(keys)}."))
        return (cost_model.grid_column_defs(keys), rows, grid_state, notices)

    # -------------------------------------------------------------- export
    @app.callback(
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.COST_EXPORT_BTN, "n_clicks"),
        State(ids.COST_EXPORT_PATH, "value"),
        State(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID_STATE, "data"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def export_table(n_clicks, path, rows, grid_state, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        if not path or not grid_state:
            notices.append(Notice(
                severity="warning", title="Nothing to export",
                meaning="Seed/edit a cost table and enter an export path "
                        "(.csv/.json/.xlsx) first.",
                impact="No file was written.",
                fix="Enter a target path, e.g. costs.csv.",
                focus_id=ids.TAB_COST).to_dict())
            return notices
        keys = tuple(grid_state.get("feature_keys") or [])
        assumptions = cost_model.assumptions_from_grid_rows(rows or [], keys)
        result, notices = guard(cost_model.export_table, assumptions, keys,
                                path, notices=notices)
        if result is not None:
            notices.append(success("Cost table exported",
                                   meaning=f"Written to {result}."))
        return notices

    # ---------------------------------------------------- modifier list (F12)
    @app.callback(
        Output(ids.MODIFIER_GRID, "rowData", allow_duplicate=True),
        Input(ids.MODIFIER_ADD_BTN, "n_clicks"),
        State(ids.MODIFIER_GRID, "rowData"),
        prevent_initial_call=True)
    def add_modifier(n_clicks, rows):
        if not n_clicks:
            raise PreventUpdate
        return (rows or []) + [{"dataset": "", "column": "",
                                "operator": "all", "value": "",
                                "mode": "multiply", "factor": "1.0",
                                "buffer_m": 0}]

    @app.callback(
        Output(ids.MODIFIER_GRID, "rowData", allow_duplicate=True),
        Input(ids.MODIFIER_DEL_BTN, "n_clicks"),
        State(ids.MODIFIER_GRID, "rowData"),
        State(ids.MODIFIER_GRID, "selectedRows"),
        prevent_initial_call=True)
    def delete_modifier(n_clicks, rows, selected):
        if not n_clicks or not selected:
            raise PreventUpdate
        drop = {tuple(sorted(r.items())) for r in selected}
        return [r for r in (rows or [])
                if tuple(sorted(r.items())) not in drop]
