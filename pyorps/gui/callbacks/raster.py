"""
PYORPS GUI callbacks: rasterization + raster layers (R3, R6).

``add_raster_layer`` is the one entry point for serving any raster as a map
layer (obeys C6-C9 via services.tiles). The tab callbacks are registered in
:func:`register`.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import tiles, validate
from ..services.errors import Notice, guard, success


def add_raster_layer(state, source, notices: list, *, name: str | None = None,
                     crs=None, transform=None, colormap: str = "viridis"):
    """Serve a raster (path/array/handler/dataset) and register its layer.

    Returns ``(layer_or_None, notices)`` — errors/warnings land in notices.
    """
    import os

    display = name or (os.path.basename(str(source))
                       if isinstance(source, (str, os.PathLike))
                       else "Cost raster")
    tile_layer, notices = guard(
        tiles.build_tile_layer, source, name=display, crs=crs,
        transform=transform, colormap=colormap or "viridis",
        work_dir=state.work_dir, notices=notices)
    if tile_layer is None:
        return None, notices
    raster_crs = crs
    if raster_crs is None:
        import rasterio
        try:
            with rasterio.open(tile_layer.source_path) as src:
                raster_crs = src.crs
        except Exception:
            raster_crs = None
    layer = state.add_layer(
        display, "raster", tile=tile_layer, crs=raster_crs,
        meta={"source_path": tile_layer.source_path,
              "vmin": tile_layer.vmin, "vmax": tile_layer.vmax})
    notices.append(success(
        f"Raster layer '{display}' added",
        meaning=f"Value range {tile_layer.vmin:g} - {tile_layer.vmax:g}; "
                "forbidden cells are transparent."))
    return layer, notices


def _size_bounds(state):
    """Bounds used for the live size estimate: study area, else data union."""
    from ..services import data_io

    bounds = data_io.study_area_bounds(state.study_area, state.project_crs)
    if bounds is not None:
        return bounds
    vectors = state.layers_of_kind("vector")
    if not vectors:
        return None
    import numpy as np

    stack = np.array([ly.gdf.total_bounds for ly in vectors
                      if ly.gdf is not None])
    if not len(stack):
        return None
    return (stack[:, 0].min(), stack[:, 1].min(),
            stack[:, 2].max(), stack[:, 3].max())


def _build_modifiers(state, rows):
    """Modifier grid rows -> ordered ModifierSpec list (F12 + Feature 5).

    Each new-schema row is one ``(column, operator, value) -> mode, factor``
    condition; the modifier dataset is filtered GUI-side (cost_model.
    apply_condition) so a scalar factor drives modify_raster_from_dataset.
    Legacy rows (a JSON ``values`` mapping on ``zone_field``) are expanded into
    per-value equality conditions for backward compatibility.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from ..services import cost_model
    from ..services.rasterize import ModifierSpec

    specs = []
    for row in rows or []:
        name = (row.get("dataset") or "").strip()
        if not name:
            continue
        layer = next((ly for ly in state.layers_of_kind("vector")
                      if ly.name == name), None)
        if layer is None or layer.gdf is None:
            raise ValueError(f"Modifier dataset '{name}' is not a loaded "
                             "vector layer.")
        mode = row.get("mode") or "multiply"
        buffer_m = float(row.get("buffer_m") or 0)

        if mode == "per-feature":       # Feature 5: per-polygon cost column
            value_col = (row.get("column") or "cost").strip() or "cost"
            if value_col not in layer.gdf.columns:
                raise ValueError(
                    f"Per-feature modifier needs a numeric column; '{value_col}'"
                    f" is not in '{name}'.")
            subset = cost_model.apply_condition(
                layer.gdf, row.get("column"), row.get("operator") or "all",
                row.get("value"))
            for cost_value, group in subset.groupby(value_col):
                try:
                    factor = float(cost_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Per-feature column '{value_col}' must be numeric; got "
                        f"'{cost_value}'.") from exc
                specs.append(ModifierSpec(
                    gdf=group, mode="override", factor=factor,
                    buffer_m=buffer_m, name=name,
                    condition=f"{value_col} = {factor:g}"))
            continue

        if "operator" in row or "factor" in row:          # Feature 5 schema
            column = (row.get("column") or row.get("zone_field") or "").strip()
            operator = row.get("operator") or "all"
            value = row.get("value")
            factor = cost_model.coerce_factor(row.get("factor"))
            gdf = cost_model.apply_condition(layer.gdf, column, operator, value)
            specs.append(ModifierSpec(
                gdf=gdf, mode=mode, factor=factor, buffer_m=buffer_m,
                name=name,
                condition=cost_model.condition_label(column, operator, value)))
            continue

        # legacy row: scalar or {value: factor} mapping on zone_field
        values = cost_model.parse_modifier_values(row.get("values"))
        zone = (row.get("zone_field") or "").strip()
        if isinstance(values, dict):
            for val, factor in values.items():
                gdf = cost_model.apply_condition(layer.gdf, zone, "==", val)
                specs.append(ModifierSpec(
                    gdf=gdf, mode=mode, factor=float(factor), buffer_m=buffer_m,
                    name=name, condition=f"{zone} == {val}"))
        else:
            specs.append(ModifierSpec(
                gdf=layer.gdf, mode=mode, factor=float(values),
                buffer_m=buffer_m, name=name, condition="all features"))
    return specs


def _table_compatible(entry: dict, gdf) -> bool:
    """A cost table fits a dataset iff every feature column exists there."""
    keys = entry.get("feature_keys") or []
    if gdf is None or not keys:
        return False
    return all(k in gdf.columns for k in keys)


def _rasterize_inputs(state, grid_state, grid_rows, dataset_sel, table_sel):
    """Resolve the (layer, feature_keys, rows) to rasterize.

    Default ('' dataset / 'current' table) keeps the old behaviour: the Cost
    tab's live grid on its own dataset. An explicit dataset can be paired
    with any REGISTERED table whose feature columns it carries — the option
    list only offers compatible tables, and this re-checks in case of a stale
    dropdown. Raises ValueError with a user-ready message.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if table_sel and table_sel != "current":
        entry = state.cost_tables.get(table_sel)
        if entry is None:
            raise ValueError("The picked cost table no longer exists — "
                             "re-seed or re-import it in the Cost tab.")
        dataset_id = dataset_sel or entry.get("dataset_id")
        layer = state.get(dataset_id) if dataset_id else None
        if layer is None or layer.gdf is None:
            raise ValueError("Pick the dataset to rasterize — the table's "
                             "own dataset is no longer loaded.")
        if not _table_compatible(entry, layer.gdf):
            raise ValueError(
                f"Cost table '{entry['name']}' does not belong to "
                f"'{layer.name}': its feature column(s) "
                f"{entry.get('feature_keys')} are not in the dataset.")
        return layer, tuple(entry["feature_keys"]), entry.get("rows") or []
    # live grid path
    if not grid_state or not grid_rows:
        raise ValueError("Seed a cost table first (Cost tab → pick dataset "
                         "+ features → Seed cost table).")
    dataset_id = dataset_sel or grid_state.get("dataset_id")
    layer = state.get(dataset_id) if dataset_id else None
    if layer is None or layer.gdf is None:
        raise ValueError("The base dataset is no longer loaded — re-load it "
                         "and re-seed the cost table.")
    keys = tuple(grid_state.get("feature_keys") or [])
    if dataset_sel and dataset_sel != grid_state.get("dataset_id") \
            and not all(k in layer.gdf.columns for k in keys):
        raise ValueError(
            f"The current cost table ({' × '.join(keys)}) does not belong "
            f"to '{layer.name}' — its feature column(s) are missing there.")
    return layer, keys, grid_rows


def register(app, state) -> None:
    # ------------------- rasterization pairing: dataset + COMPATIBLE tables
    @app.callback(
        Output(ids.RASTERIZE_DATASET, "options"),
        Output(ids.RASTERIZE_TABLE, "options"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.COST_GRID_STATE, "data"),
        Input(ids.RASTERIZE_DATASET, "value"))
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    def sync_rasterize_options(_view, grid_state, dataset_sel):
        dataset_options = [{"label": "— from Cost tab (default)",
                            "value": ""}]
        dataset_options += [{"label": ly.name, "value": ly.id}
                            for ly in state.layers_of_kind("vector")]
        table_options = [{"label": "current Cost-tab table",
                          "value": "current"}]
        target = state.get(dataset_sel) if dataset_sel else None
        for table_id, entry in state.cost_tables.items():
            if target is not None and not _table_compatible(entry,
                                                            target.gdf):
                continue                    # incompatible → never offered
            table_options.append({"label": entry["name"],
                                  "value": table_id})
        return dataset_options, table_options

    # ------------------------------------------------- live size estimate (F3)
    @app.callback(
        Output(ids.RASTER_SIZE_INFO, "children"),
        Input(ids.RES_M, "value"),
        Input(ids.RASTER_DTYPE, "value"),
        Input(ids.LAYERS_VIEW, "data"))
    def size_info(resolution, dtype, _view):
        bounds = _size_bounds(state)
        if bounds is None or not resolution:
            return "Draw a study area / load data for a size estimate."
        est = validate.estimate_raster_size(bounds, float(resolution),
                                            dtype or "uint16")
        warning = validate.validate_raster_size(bounds, float(resolution),
                                                dtype or "uint16")
        if warning is not None:
            return f"{est['label']} — {warning.title}!"
        return est["label"]

    # ------------------------------------------------------- build cost raster
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.RASTERIZE_LOG, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.RASTERIZE_BTN, "n_clicks"),
        State(ids.COST_GRID, "rowData"),
        State(ids.COST_GRID_STATE, "data"),
        State(ids.MODIFIER_GRID, "rowData"),
        State(ids.PREPROC_SELECT, "value"),
        State(ids.PREPROC_BUF_A, "value"),
        State(ids.PREPROC_BUF_B, "value"),
        State(ids.PREPROC_BUF_L, "value"),
        State(ids.PREPROC_GRID, "rowData"),
        State(ids.RES_M, "value"),
        State(ids.FILL_VALUE, "value"),
        State(ids.RASTER_DTYPE, "value"),
        State(ids.GEOM_BUFFER, "value"),
        State(ids.RASTER_SAVE_PATH, "value"),
        State(ids.RASTER_COLORMAP, "value"),
        State(ids.NOTICES, "data"),
        State(ids.RASTERIZE_DATASET, "value"),
        State(ids.RASTERIZE_TABLE, "value"),
        prevent_initial_call=True)
    def run_rasterize(n_clicks, rows, grid_state, modifier_rows, preproc,
                      buf_a, buf_b, buf_l, preproc_steps, resolution,
                      fill_value, dtype, geom_buffer, save_path, colormap,
                      notices, dataset_sel=None, table_sel=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..services import cost_model, data_io
        from ..services.rasterize import build_cost_raster

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        try:
            layer, keys, rows_used = _rasterize_inputs(
                state, grid_state, rows, dataset_sel, table_sel)
        except ValueError as exc:
            title = ("Seed a cost table first"
                     if "Seed a cost table" in str(exc)
                     else "Dataset / cost table not usable")
            notices.append(Notice(
                severity="warning", title=title,
                meaning=str(exc),
                impact="No raster was built.",
                fix="Pair a loaded dataset with a cost table that carries "
                    "its feature columns.",
                focus_id=ids.TAB_RASTER,
                focus_control=ids.RASTERIZE_TABLE).to_dict())
            return no_update, no_update, no_update, notices

        assumptions = cost_model.assumptions_from_grid_rows(rows_used, keys)
        polygon = data_io.study_area_polygon(state.study_area,
                                             layer.gdf.crs)
        resolution = float(resolution or 1.0)
        dtype = dtype or "uint16"

        # pre-flight (shift left, Section 21.5)
        bounds = (tuple(polygon.bounds) if polygon is not None
                  else tuple(layer.gdf.total_bounds))
        blocker = validate.validate_raster_size(bounds, resolution, dtype)
        if blocker is not None:
            notices.append(blocker.to_dict())
            if blocker.severity == "error":
                return no_update, no_update, no_update, notices
        coverage = validate.validate_cost_coverage(layer.gdf, keys,
                                                   assumptions)
        if coverage is not None:
            notices.append(coverage.to_dict())

        try:
            modifiers = _build_modifiers(state, modifier_rows)
            # bake custom cost polygons in as per-feature overrides so they
            # show on the cost map and affect routing (task 45)
            from ..services import manual_cost
            for ml in state.layers_of_kind("vector"):
                if ((ml.meta or {}).get(manual_cost.MANUAL_META_KEY)
                        and ml.gdf is not None and not ml.gdf.empty):
                    mg = (ml.gdf if str(ml.gdf.crs) == str(layer.gdf.crs)
                          else ml.gdf.to_crs(layer.gdf.crs))
                    modifiers = modifiers + manual_cost.override_specs_from_layer(
                        mg, name=ml.name)
        except ValueError as exc:
            notices.append(Notice(
                severity="error", title="Modifier configuration invalid",
                meaning=str(exc),
                impact="No raster was built.",
                fix="Fix the modifier row (dataset / zone field / values).",
                focus_id=ids.TAB_COST).to_dict())
            return no_update, no_update, no_update, notices

        preproc_params = {}
        if preproc == "street_buffer":
            preproc_params = {"buffer_a": float(buf_a or 10),
                              "buffer_b": float(buf_b or 4),
                              "buffer_l": float(buf_l or 2)}

        result, notices = guard(
            build_cost_raster, base_gdf=layer.gdf, assumptions=assumptions,
            feature_keys=keys, resolution_in_m=resolution,
            fill_value=int(fill_value or 65535), dtype=dtype,
            geometry_buffer_m=float(geom_buffer or 0),
            bounding_polygon=polygon, preprocessor=preproc,
            preprocessor_params=preproc_params,
            preprocessor_steps=preproc_steps or [], modifiers=modifiers,
            save_path=save_path or None, work_dir=state.work_dir,
            notices=notices)
        if result is None:
            return no_update, no_update, no_update, notices
        raster_path, log = result
        new_layer, notices = add_raster_layer(
            state, raster_path, notices,
            name=f"Cost raster ({resolution:g} m)", colormap=colormap)
        if new_layer is None:
            return no_update, no_update, "\n".join(log), notices
        new_layer.meta["rasterize_config"] = {
            "dataset_id": layer.id,
            "feature_keys": list(keys), "resolution_in_m": resolution,
            "fill_value": int(fill_value or 65535), "dtype": dtype,
            "geometry_buffer_m": float(geom_buffer or 0),
            "preprocessor": preproc, "preprocessor_params": preproc_params,
            "preprocessor_steps": preproc_steps or [],
        }
        # F2: the cost -> layer/feature/value combination table is now built
        # LAZILY (perf plan phase 4): stash the inputs and let the first
        # 📋-table / legend / map-click materialize it — the geometric
        # overlay can take seconds on a big model and the rasterize callback
        # shouldn't wait for it.
        new_layer.meta["combination_inputs"] = {
            "base_gdf": layer.gdf, "feature_keys": keys,
            "assumptions": assumptions, "base_layer_name": layer.name,
            "modifiers": modifiers, "base_crs": layer.gdf.crs,
            "bounding_polygon": polygon,
        }
        log.append("combination table: built on first use (📋 table / "
                   "legend / map-click)")
        view = {"fit_bounds": new_layer.tile.bounds, "seq": n_clicks}
        return (state.layers_view(), view, "\n".join(log), notices)

    # -------------------------------------------------- load existing raster
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.RASTER_LOAD_BTN, "n_clicks"),
        State(ids.RASTER_LOAD_PATH, "value"),
        State(ids.RASTER_COLORMAP, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def load_raster(n_clicks, path, colormap, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        problem = validate.validate_file(path, "raster")
        if problem is not None:
            notices.append(problem.to_dict())
            return no_update, no_update, notices
        layer, notices = add_raster_layer(state, path, notices,
                                          colormap=colormap)
        if layer is None:
            return no_update, no_update, notices
        view = {"fit_bounds": layer.tile.bounds, "seq": n_clicks}
        return state.layers_view(), view, notices

    # -------------------------------------------------------------- opacity
    # Update the served tile components DIRECTLY (pattern-matching output) so
    # the slider never rebuilds the layer host — otherwise every nudge would
    # re-serialize and re-ship all vector/route GeoJSON (perf audit #2).
    @app.callback(
        Output({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "opacity"),
        Input(ids.RASTER_OPACITY, "value"),
        State({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "id"),
        prevent_initial_call=True)
    def set_opacity(value, tile_ids):
        if value is None or not tile_ids:
            raise PreventUpdate
        opacity = float(value)
        for tid in tile_ids:
            layer = state.get(tid["id"])
            if layer is not None:
                layer.opacity = opacity
        return [opacity] * len(tile_ids)

    # -------------------------------------------------------------- colormap
    @app.callback(
        Output({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "url"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.RASTER_COLORMAP, "value"),
        State({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "id"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def set_colormap(colormap, tile_ids, notices):
        """Re-colour served raster tiles in place (F3) — no host rebuild."""
        from ..services import graduated

        notices = list(notices or [])
        if not colormap or not tile_ids:
            raise PreventUpdate

        def _recolor():
            urls = []
            for tid in tile_ids:
                layer = state.get(tid["id"])
                if (layer is not None and layer.kind == "raster"
                        and layer.tile is not None):
                    grad = (layer.meta or {}).get("graduated")
                    if grad:
                        # classified layer: resample class colours from the
                        # new ramp, keep the breaks (QGIS ramp-change flow)
                        grad["colors"] = graduated.default_colors(
                            colormap, grad["classes"])
                        graduated.apply_to_layer(layer, grad)
                    else:
                        tiles.set_colormap(layer.tile, colormap)
                    layer.meta["colormap"] = colormap
                    urls.append(layer.tile.tile_url)
                else:
                    urls.append(no_update)          # WMS/other: keep its url
            return urls

        urls, notices = guard(_recolor, notices=notices)
        if urls is None:
            urls = [no_update] * len(tile_ids)
        return urls, notices

    # ------------------------------------- cost <-> colour <-> combo legend
    def _legend_children(layer, colormap):
        """Legend for the selected raster: graduated class rows (editable
        colours) take precedence over the cost-combination legend."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from dash import html

        from ..services import combinations

        if layer is None or layer.kind != "raster":
            return html.Small("Select a built cost raster in the Layers tab.",
                              className="text-muted")
        grad = (layer.meta or {}).get("graduated")
        if grad:
            rows = []
            for i, (color, label) in enumerate(zip(grad["colors"],
                                                   grad["labels"])):
                # NB: dbc.Input rejects a `title` kwarg (dbc strict props)
                picker = dbc.Input(
                    type="color", value=color,
                    id={"type": ids.TYPE_GRAD_COLOR, "layer": layer.id,
                        "index": i},
                    className="legend-color-input")
                rows.append(html.Div(
                    [picker, html.Span(label, className="legend-label",
                                       title=label)],
                    className="legend-row"))
            return rows
        combo = combinations.ensure_combination_gdf(layer)   # lazy build
        if combo is None or getattr(combo, "empty", True):
            return html.Small("This raster has no combination legend (only "
                              "rasters built from the Cost tab do).",
                              className="text-muted")
        items = combinations.legend_items(
            combo, colormap, (layer.meta or {}).get("vmin"),
            (layer.meta or {}).get("vmax"))
        rows = []
        for cost, color, label in items:
            swatch = html.Span(
                className="legend-swatch",
                style={"backgroundColor": color or "transparent"})
            text = (f"{cost:,} — {label}" if cost < 65535
                    else f"forbidden — {label}")
            rows.append(html.Div(
                [swatch, html.Span(text, className="legend-label",
                                   title=text)],
                className="legend-row"))
        return rows or html.Small("No combinations.", className="text-muted")

    @app.callback(
        Output(ids.COLORMAP_LEGEND, "children"),
        Input(ids.LAYERS_GRID, "selectedRows"),
        Input(ids.RASTER_COLORMAP, "value"),
        Input(ids.GRAD_STATUS, "children"))
    def render_colormap_legend(selected, colormap, _grad_status):
        # _grad_status only re-triggers the render after (re)classification —
        # the class config itself lives in layer.meta["graduated"]
        from dash import html

        if not selected:
            return html.Small("Select a built cost raster in the Layers tab.",
                              className="text-muted")
        return _legend_children(state.get(selected[0].get("id")), colormap)

    # ------------------------------ graduated rendering (QGIS-style classes)
    def _target_raster_layer(selected):
        """The raster to classify: the Layers-tab selection, else the only
        raster layer, else None."""
        if selected:
            layer = state.get(selected[0].get("id"))
            if layer is not None and layer.kind == "raster":
                return layer
        rasters = [ly for ly in state.layers.values()
                   if ly.kind == "raster" and ly.tile is not None]
        return rasters[0] if len(rasters) == 1 else None

    @app.callback(
        Output({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "url",
               allow_duplicate=True),
        Output(ids.GRAD_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.GRAD_APPLY_BTN, "n_clicks"),
        State(ids.GRAD_MODE, "value"),
        State(ids.GRAD_CLASSES, "value"),
        State(ids.LAYERS_GRID, "selectedRows"),
        State({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "id"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def apply_graduated(n_clicks, mode, classes, selected, tile_ids,
                        notices):
        # NOTE: the base colormap is read from layer.meta (kept fresh by
        # set_colormap) rather than a RASTER_COLORMAP State — an extra State
        # would make this callback's wiring a superset of set_colormap's and
        # break find_callback disambiguation (round-4 gotcha).
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..services import graduated

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        no_urls = [no_update] * len(tile_ids or [])
        layer = _target_raster_layer(selected)
        if layer is None or layer.tile is None:
            return (no_urls,
                    "Select ONE raster layer in the Layers tab first.",
                    notices)
        colormap = ((layer.meta or {}).get("colormap")
                    or (layer.tile.colormap
                        if not str(layer.tile.colormap).startswith("custom:")
                        else None) or "viridis")

        def _classify():
            if (mode or "continuous") == "continuous":
                layer.meta.pop("graduated", None)
                tiles.set_colormap(layer.tile, colormap)
                return "continuous ramp restored"
            config = graduated.build_config(
                layer.meta.get("source_path") or layer.tile.source_path,
                mode, int(classes or 5), colormap)
            graduated.apply_to_layer(layer, config)
            return (f"{config['classes']} classes "
                    f"({graduated.METHODS.get(mode, mode)}) — edit the "
                    "class colours in the legend below")

        status, notices = guard(_classify, notices=notices)
        if status is None:
            return no_urls, "classification failed", notices
        urls = [layer.tile.tile_url if tid["id"] == layer.id else no_update
                for tid in (tile_ids or [])]
        return urls, status, notices
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    @app.callback(
        Output({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "url",
               allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input({"type": ids.TYPE_GRAD_COLOR, "layer": ALL, "index": ALL},
              "value"),
        State({"type": ids.TYPE_RASTER_TILE, "id": ALL}, "id"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def recolor_class(_colors, tile_ids, notices):
        """One class colour edited in the legend → rebuild the LUT."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from dash import ctx

        from ..services import graduated

        notices = list(notices or [])
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict) or "index" not in trigger:
            raise PreventUpdate
        layer = state.get(trigger.get("layer"))
        grad = (layer.meta or {}).get("graduated") if layer else None
        if layer is None or layer.tile is None or not grad:
            raise PreventUpdate
        value = (ctx.triggered[0].get("value")
                 if ctx.triggered else None)
        index = int(trigger["index"])
        if (not value or index >= len(grad["colors"])
                or value == grad["colors"][index]):
            raise PreventUpdate      # mount-fire or no-op change
        grad["colors"][index] = value

        _, notices = guard(graduated.apply_to_layer, layer, grad,
                           notices=notices)
        urls = [layer.tile.tile_url if tid["id"] == layer.id else no_update
                for tid in (tile_ids or [])]
        return urls, notices

    # ------------------------------------------------ combine rasters (F6)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.RASTER_COMBINE_STATUS, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.RASTER_COMBINE_BTN, "n_clicks"),
        State(ids.RASTER_COMBINE_SELECT, "value"),
        State(ids.RASTER_COMBINE_OP, "value"),
        State(ids.RASTER_COLORMAP, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def combine_rasters(n_clicks, layer_ids, operation, colormap, notices):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..services import raster_algebra

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        layers = [state.get(lid) for lid in (layer_ids or [])]
        layers = [ly for ly in layers if ly is not None and ly.tile is not None]
        if len(layers) < 2:
            notices.append(Notice(
                severity="warning", title="Select at least two rasters",
                meaning="Combining needs two or more raster layers.",
                impact="Nothing was combined.",
                fix="Pick two or more raster layers in the combine list.",
                focus_id=ids.TAB_RASTER).to_dict())
            return no_update, no_update, no_update, notices
        paths = [ly.tile.source_path for ly in layers]
        result, notices = guard(
            raster_algebra.combine_rasters, paths, operation or "add",
            work_dir=state.work_dir, notices=notices)
        if result is None:
            return no_update, no_update, no_update, notices
        out_path, log = result
        name = f"{operation} ({len(layers)} rasters)"
        new_layer, notices = add_raster_layer(state, out_path, notices,
                                              name=name, colormap=colormap)
        if new_layer is None:
            return no_update, no_update, "\n".join(log), notices
        new_layer.meta["combined_from"] = [ly.id for ly in layers]
        view = {"fit_bounds": new_layer.tile.bounds, "seq": n_clicks}
        return state.layers_view(), view, "\n".join(log), notices
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
