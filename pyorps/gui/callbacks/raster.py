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
from ..services import jobs, tiles, validate
from ..services import rasterize as rasterize_service
from ..services.errors import Notice, cap_notices, guard, success
from . import job_poll
from .routes import field_notice, number_or_default, numeric_field_problem


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
        except Exception:  # pylint: disable=broad-exception-caught  # CRS probe is best-effort
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
    if not stack.size:
        return None
    return (stack[:, 0].min(), stack[:, 1].min(),
            stack[:, 2].max(), stack[:, 3].max())


def _whole_number(value, label: str, name: str, *, dtype=None) -> int:
    """A modifier cell that must be a whole number from 0 up (and fit the raster
    type when it is known); a ValueError names the row and the column."""
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Modifier '{name}': {label} must be a number "
                         f"(got '{value}').") from None
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError(f"Modifier '{name}': {label} must be a whole number "
                         f"from 0 up (got {value}).")
    if dtype:
        import numpy as np

        try:
            info = np.iinfo(dtype)
        except ValueError:                       # a float raster: any number
            return int(number)
        if number > info.max:
            raise ValueError(
                f"Modifier '{name}': {label} {int(number)} does not fit the "
                f"raster type {dtype} (allowed {info.min} to {info.max}).")
    return int(number)


def _zone_value(text, column, gdf, name: str):
    """The zone cell as the value the library compares the column with: a
    number for a numeric column, else the stripped text."""
    import pandas as pd

    if pd.api.types.is_numeric_dtype(gdf[column]):
        try:
            zone = float(str(text).strip().replace(",", "."))
        except ValueError:
            raise ValueError(
                f"Modifier '{name}': the zone value '{text}' is not a number "
                f"but column '{column}' is numeric.") from None
    else:
        zone = str(text).strip()
    if not zone:      # the library skips the zone rule for a falsy value
        raise ValueError(
            f"Modifier '{name}': a zone value of 0 or empty is not supported "
            "(the library would skip the zone rule); pick another value or "
            "use a condition on the row instead.")
    return zone


def _modifier_options(row: dict, gdf, name: str, dtype=None) -> dict:
    """The ``ignore_value`` / ``forbidden_value`` / zone arguments of a row.

    Old rows have none of the four columns: they mean "keep forbidden cells,
    forbid as 65535, no zone rule" - exactly what every row did before.
    """
    from .. import presets

    keep = row.get("keep_forbidden")
    keep = True if keep is None or keep == "" else bool(keep)
    forbidden = number_or_default(row.get("forbidden_value"),
                                  presets.FORBIDDEN)
    options = {
        "ignore_value": presets.FORBIDDEN if keep else None,
        "forbidden_value": _whole_number(forbidden, "'Forbid as'", name,
                                         dtype=dtype),
        "zone_field": None, "forbidden_zone": None,
    }
    column = str(row.get("zone_column") or "").strip()
    zone = row.get("forbidden_zone")
    has_zone = zone is not None and str(zone).strip() != ""
    if not column and not has_zone:
        return options
    if not column or not has_zone:
        raise ValueError(
            f"Modifier '{name}': the zone rule needs both a zone column and "
            "a zone value.")
    if column not in gdf.columns:
        raise ValueError(f"Modifier '{name}': the zone column '{column}' is "
                         "not an attribute of the dataset.")
    options["zone_field"] = column
    options["forbidden_zone"] = _zone_value(zone, column, gdf, name)
    return options


def _build_modifiers(state, rows, dtype=None):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
        buffer_m = float(number_or_default(row.get("buffer_m"), 0))
        if buffer_m < 0:
            raise ValueError(f"Modifier '{name}': the buffer must not be "
                             f"negative (got {buffer_m:g} m).")
        options = _modifier_options(row, layer.gdf, name, dtype)

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
                    condition=f"{value_col} = {factor:g}", **options))
            continue

        if "operator" in row or "factor" in row:          # Feature 5 schema
            column = (row.get("column") or row.get("zone_field") or "").strip()
            operator = row.get("operator") or "all"
            value = row.get("value")
            factor = cost_model.coerce_factor(row.get("factor"))
            gdf = cost_model.apply_condition(layer.gdf, column, operator, value)
            specs.append(ModifierSpec(
                gdf=gdf, mode=mode, factor=factor, buffer_m=buffer_m,
                name=name, **options,
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
                    name=name, condition=f"{zone} == {val}", **options))
        else:
            specs.append(ModifierSpec(
                gdf=layer.gdf, mode=mode, factor=float(values),
                buffer_m=buffer_m, name=name, condition="all features",
                **options))
    return specs


def _table_compatible(entry: dict, gdf) -> bool:
    """A cost table fits a dataset iff every feature column exists there."""
    keys = entry.get("feature_keys") or []
    if gdf is None or not keys:
        return False
    return all(k in gdf.columns for k in keys)


def _rasterize_inputs(state, grid_state, grid_rows, dataset_sel, table_sel):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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


def _fill_value_problem(fill_value, dtype: str) -> dict | None:
    """A notice when ``fill_value`` cannot be stored in a raster of ``dtype``."""
    import numpy as np

    kwargs = {"control": ids.FILL_VALUE, "tab": ids.TAB_RASTER,
              "impact": "No raster was built."}
    problem = numeric_field_problem(fill_value, "Fill value", whole=True,
                                    **kwargs)
    if problem is not None or fill_value is None or fill_value == "":
        return problem
    number = float(fill_value)
    try:
        info = np.iinfo(dtype)
    except ValueError:                       # a float dtype: any whole number
        return None
    if not info.min <= number <= info.max:
        return field_notice(
            "Fill value", f"{number:g} does not fit the raster type "
            f"{dtype} (allowed {info.min} to {info.max})", **kwargs)
    return None


def fill_value_limits(dtype) -> tuple[float | None, float | None]:
    """``(min, max)`` the raster type ``dtype`` can store (``None``: no limit)."""
    import numpy as np

    try:
        info = np.iinfo(dtype or "uint16")
    except (TypeError, ValueError):          # float32: any whole number
        return None, None
    return info.min, info.max


def _choice_problem(value, field: str, allowed: tuple[str, ...], *,
                    control: str) -> dict | None:
    """A notice when a dropdown holds something it never offers (a stale page,
    a restored setting); a blank value means "the default" and is fine."""
    if value is None or value == "" or value in allowed:
        return None
    return field_notice(
        field, f"must be one of {', '.join(allowed)} (got '{value}')",
        control=control, tab=ids.TAB_RASTER, impact="No raster was built.")


def _rasterize_field_problem(resolution, fill_value, dtype, geom_buffer,
                             preproc, buf_a, buf_b, buf_l,
                             on_thin=None) -> dict | None:
    """The first unusable rasterize form field as a notice, or None."""
    tab = ids.TAB_RASTER
    kwargs = {"tab": tab, "impact": "No raster was built."}
    checks = [
        numeric_field_problem(resolution, "Resolution (m)", control=ids.RES_M,
                              minimum=0, allow_equal=False, **kwargs),
        _choice_problem(dtype, "Data type", rasterize_service.RASTER_DTYPES,
                        control=ids.RASTER_DTYPE),
        _fill_value_problem(fill_value, dtype or "uint16"),
        numeric_field_problem(geom_buffer, "Geometry buffer (m)",
                              control=ids.GEOM_BUFFER, minimum=0, **kwargs),
        _choice_problem(on_thin, "On thin features",
                        rasterize_service.THIN_POLICIES, control=ids.ON_THIN),
    ]
    if preproc == "street_buffer":
        checks += [
            numeric_field_problem(value, label, control=control, minimum=0,
                                  **kwargs)
            for value, label, control in (
                (buf_a, "Street buffer A (m)", ids.PREPROC_BUF_A),
                (buf_b, "Street buffer B (m)", ids.PREPROC_BUF_B),
                (buf_l, "Street buffer L (m)", ids.PREPROC_BUF_L))]
    return next((check for check in checks if check is not None), None)


#: what the two background-job outputs of ``run_rasterize`` (poll, Stop button) hold when a run does not
#: touch the job. ``RASTERIZE_BUSY`` stays a ``running=`` target of that callback alone: the Build button is
#: gated only while the callback itself runs, so a click during a background run reaches the job
#: service's refusal notice.
_NO_JOB = (no_update, no_update)
#: the Stop button of the rasterize job is hidden unless a job runs
STOP_HIDDEN = job_poll.STOP_HIDDEN
#: where the state keeps the rasterize job
_RASTERIZE_SLOT = jobs.kind_slot(rasterize_service.JOB_KIND)
#: what a stop is promised to do, for the status line and the Stop answer
STOP_NOTE = ("The raster is built in stages (base layer, each modifier, crop, save): the run stops when the "
             "stage in flight is finished, and then builds nothing.")


def rasterize_status_text(job) -> str:
    """The status line of a running rasterize job: its progress, and that a Stop is waiting."""
    text = job.progress_text()
    if job.cancel.is_set() and job.status == jobs.RUNNING:
        text += " (stop requested, the run ends at the next stage boundary)"
    return text


def _combination_inputs(base_layer_name: str, build_kwargs: dict) -> dict:
    """The lazy combination-table inputs a built raster layer keeps (perf plan phase 4)."""
    return {
        "base_gdf": build_kwargs["base_gdf"], "feature_keys": build_kwargs["feature_keys"],
        "assumptions": build_kwargs["assumptions"], "base_layer_name": base_layer_name,
        "modifiers": build_kwargs["modifiers"], "base_crs": build_kwargs["base_gdf"].crs,
        "bounding_polygon": build_kwargs["bounding_polygon"],
    }


def finalize_rasterize(state, raster_path: str, log: list[str], notices: list[dict], *, resolution: float,
                       colormap, config: dict, combination: dict, seq):
    """What happens after a cost raster is built: layer, tiles, coverage info, log.

    The one path for the synchronous run (inside ``run_rasterize``) and a background run (the job
    poll, ``callbacks/raster_job.py``, through :func:`rasterize_outcome`). ``config`` is the layer's
    ``rasterize_config``, ``combination`` its lazy ``combination_inputs``, ``seq`` the map-view counter
    that makes the map fit the new layer. Returns the four values of ``LAYERS_VIEW``, ``MAP_VIEW``,
    ``RASTERIZE_LOG`` and ``NOTICES``.
    """
    new_layer, notices = add_raster_layer(
        state, raster_path, notices,
        name=f"Cost raster ({resolution:g} m)", colormap=colormap)
    if new_layer is None:
        return no_update, no_update, "\n".join(log), notices
    new_layer.meta["rasterize_config"] = config
    # F2: the cost -> layer/feature/value combination table is now built
    # LAZILY (perf plan phase 4): stash the inputs and let the first
    # 📋-table / legend / map-click materialize it — the geometric
    # overlay can take seconds on a big model and the rasterize callback
    # shouldn't wait for it.
    new_layer.meta["combination_inputs"] = combination
    log.append("combination table: built on first use (📋 table / "
               "legend / map-click)")
    view = {"fit_bounds": new_layer.tile.bounds, "seq": seq}
    return state.layers_view(), view, "\n".join(log), notices


def rasterize_outcome(state, job):
    """The ended rasterize job as ``(results, notices)`` for the job poll (``callbacks/job_poll.py``).

    ``results`` are the values of ``LAYERS_VIEW``, ``MAP_VIEW`` and ``RASTERIZE_LOG``, ``notices`` the notice
    dicts to add. Done: the shared :func:`finalize_rasterize` adds the layer. Stopped: a notice, nothing
    changes in the project. Failed: the translated error as a notice. Runs once per job, in a poll tick.
    """
    notices: list[dict] = []
    if job.status == jobs.FAILED:
        failed = job.error_notice()
        if failed is not None:
            notices.append(failed)
        return (no_update, no_update, job.progress_text()), notices
    if job.status == jobs.CANCELLED:
        notices.append(Notice(
            severity="info", title="Rasterizing stopped",
            meaning="Stop was pressed; the run ended at a stage boundary.",
            impact="No raster was built and the project is unchanged.",
            fix="Press Build cost raster again to start over.").to_dict())
        return (no_update, no_update, f"{job.progress_text()} No raster was built."), notices
    result = job.result or {}
    notices.extend(result.get("notices") or [])
    view, map_view, log, notices = finalize_rasterize(
        state, result["path"], list(result["log"]), notices, **job.params["finalize"])
    return (view, map_view, log), notices


def _start_background_run(state, build_kwargs: dict, finalize: dict, base_layer_name: str,
                          notices: list[dict], n_cells: int):
    """Start the heavy build as a background job; the six outputs of ``run_rasterize``.

    The worker gets a snapshot of the build inputs (``build_kwargs``), the poll callback the ``finalize``
    keywords of :func:`finalize_rasterize` plus the lazy combination inputs made from that same snapshot
    (they must describe the data the raster was built from). A refused start (another job runs) answers
    with the job service's notice and changes nothing.
    """
    refusal = jobs.refusal_notice(state, rasterize_service.JOB_KIND)
    if refusal is not None:
        notices.append(refusal)
        return (no_update, no_update, no_update, cap_notices(notices), *_NO_JOB)
    snapshot = rasterize_service.snapshot_build_inputs(build_kwargs)
    finalize = {**finalize, "combination": _combination_inputs(base_layer_name, snapshot)}
    job = jobs.start_job(
        state, rasterize_service.JOB_KIND, rasterize_service.background_build,
        label=rasterize_service.JOB_LABEL, params={"build": snapshot, "finalize": finalize})
    if job is None:             # another start won the race between the check and here
        notices.append(jobs.refusal_notice(state, rasterize_service.JOB_KIND)
                       or jobs.busy_notice(jobs.busy_label(state) or rasterize_service.JOB_LABEL))
        return (no_update, no_update, no_update, cap_notices(notices), *_NO_JOB)
    notices.append(Notice(
        severity="info", title="Rasterizing runs in the background",
        meaning=f"About {n_cells:,} cells is a large raster, so it is built in a background job and the page "
                "stays usable.",
        impact="The layer is added when the job has finished. " + STOP_NOTE,
        fix="Watch the status line below the Build button; press Stop to end the run.").to_dict())
    return (no_update, no_update, rasterize_status_text(job), cap_notices(notices),
            False, {})


def _held_job_outputs(state, notices: list[dict]):
    """A Build click while the state holds a rasterize job: the six outputs of ``run_rasterize``.

    While the job runs the click is refused with the job service's notice. While its outcome is being
    handed to the page the click is refused too and the poll stays on. A job that has ended but was never
    polled (e.g. the page was reloaded) switches the poll on, which hands its outcome over, instead of
    refusing every later click. None when the state holds no rasterize job.
    """
    phase = jobs.held_phase(state, _RASTERIZE_SLOT)
    if phase is None:
        return None
    if phase == jobs.HELD_RUNNING:
        notices.append(jobs.refusal_notice(state, rasterize_service.JOB_KIND)
                       or jobs.busy_notice(rasterize_service.JOB_LABEL))
        return (no_update, no_update, no_update, cap_notices(notices), *_NO_JOB)
    if phase == jobs.HELD_HANDOVER:
        notices.append(job_poll.handover_notice(rasterize_service.JOB_LABEL))
    else:
        notices.append(job_poll.ended_notice("rasterize", "Build cost raster"))
    return no_update, no_update, no_update, cap_notices(notices), False, STOP_HIDDEN


def _cached_raster(build_kwargs: dict):
    """The cached file a build with these inputs would reuse, else None (also when the probe fails)."""
    try:
        return rasterize_service.cached_raster_path(**build_kwargs)
    except Exception:  # pylint: disable=broad-exception-caught  # the build itself reports a real problem
        return None


def register(app, state) -> None:
    # ------------------- rasterization pairing: dataset + COMPATIBLE tables
    @app.callback(
        Output(ids.RASTERIZE_DATASET, "options"),
        Output(ids.RASTERIZE_TABLE, "options"),
        Input(ids.LAYERS_VIEW, "data"),
        Input(ids.COST_GRID_STATE, "data"),
        Input(ids.RASTERIZE_DATASET, "value"))
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    def sync_rasterize_options(_view, grid_state, dataset_sel):  # pylint: disable=unused-argument
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

    # --------------- the rasterize controls, kept on the server for saving
    @app.callback(
        Output(ids.RASTER_SETTINGS, "data"),
        *[Input(control, "value") for control in
          (ids.RES_M, ids.FILL_VALUE, ids.RASTER_DTYPE, ids.GEOM_BUFFER,
           ids.WIDEN_THIN, ids.ALL_TOUCHED, ids.ON_THIN, ids.CROP_BORDER)])
    def remember_raster_settings(*values):
        """Mirror the rasterize controls into ``state.raster_settings``: the
        project file saves them from there, and a rebuilt page shows them.
        Empty or unusable fields fall back to their defaults."""
        settings = rasterize_service.normalise_raster_settings(
            dict(zip(rasterize_service.RASTER_SETTING_KEYS, values)))
        state.raster_settings = settings
        return settings

    # ------------------------------ the fill value's range follows the type
    @app.callback(
        Output(ids.FILL_VALUE, "min"),
        Output(ids.FILL_VALUE, "max"),
        Input(ids.RASTER_DTYPE, "value"))
    def fill_value_range(dtype):
        """The browser's own limits of the fill value: what the chosen raster
        type can store (the run still names a value outside it)."""
        return fill_value_limits(dtype)

    # ------------------------------------------------------- build cost raster
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.RASTERIZE_LOG, "children"),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        # a heavy run is handed to a background job (callbacks/raster_job.py):
        # these two switch its poll and its Stop button on
        Output(ids.RASTERIZE_POLL, "disabled", allow_duplicate=True),
        Output(ids.RASTERIZE_STOP_BTN, "style", allow_duplicate=True),
        Input(ids.GO_RASTERIZE, "data"),
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
        State(ids.WIDEN_THIN, "value"),
        State(ids.ALL_TOUCHED, "value"),
        State(ids.ON_THIN, "value"),
        State(ids.CROP_BORDER, "value"),
        running=[(Output(ids.RASTERIZE_BUSY, "data"), True, False)],
        prevent_initial_call=True)
    def run_rasterize(go, rows, grid_state, modifier_rows, preproc,
                      buf_a, buf_b, buf_l, preproc_steps, resolution,
                      fill_value, dtype, geom_buffer, save_path, colormap,
                      notices, dataset_sel=None, table_sel=None,
                      widen_thin=False, all_touched=False, on_thin="warn",
                      crop_border=False):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from ..services import cost_model, data_io
        from ..services.rasterize import build_cost_raster

        notices = list(notices or [])
        if not go:
            raise PreventUpdate
        held = _held_job_outputs(state, notices)
        if held is not None:           # one run at a time; an ended one nobody polled is collected now
            return held
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
            return no_update, no_update, no_update, notices, *_NO_JOB

        # form fields: a value that cannot be valid is named, never replaced
        problem = _rasterize_field_problem(
            resolution, fill_value, dtype, geom_buffer, preproc, buf_a,
            buf_b, buf_l, on_thin)
        if problem is not None:
            notices.append(problem)
            return no_update, no_update, no_update, notices, *_NO_JOB

        # a non-numeric cost cell is named in a notice (#16) instead of
        # raising a raw error; the guard below catches anything else
        cost_problem = cost_model.cost_problem_notice(rows_used, keys)
        if cost_problem is not None:
            notices.append(cost_problem)
            return no_update, no_update, no_update, notices, *_NO_JOB
        assumptions, notices = guard(cost_model.assumptions_from_grid_rows,
                                     rows_used, keys, notices=notices)
        if assumptions is None:
            return no_update, no_update, no_update, notices, *_NO_JOB
        polygon = data_io.study_area_polygon(state.study_area,
                                             layer.gdf.crs)
        resolution = float(number_or_default(resolution, 1.0))
        dtype = dtype or "uint16"
        fill = int(number_or_default(fill_value, 65535))
        geometry_buffer = float(number_or_default(geom_buffer, 0))
        # the thin-feature burn and the crop: a blank policy is the library's
        # own default ("warn"); the two switches default to off
        on_thin = number_or_default(on_thin, "warn")
        widen_thin, all_touched, crop_border = (
            bool(widen_thin), bool(all_touched), bool(crop_border))

        # pre-flight (shift left, Section 21.5)
        bounds = (tuple(polygon.bounds) if polygon is not None
                  else tuple(layer.gdf.total_bounds))
        blocker = validate.validate_raster_size(bounds, resolution, dtype)
        if blocker is not None:
            notices.append(blocker.to_dict())
            if blocker.severity == "error":
                return no_update, no_update, no_update, notices, *_NO_JOB
        coverage = validate.validate_cost_coverage(layer.gdf, keys,
                                                   assumptions)
        if coverage is not None:
            notices.append(coverage.to_dict())

        try:
            modifiers = _build_modifiers(state, modifier_rows, dtype)
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
            return no_update, no_update, no_update, notices, *_NO_JOB

        preproc_params = {}
        if preproc == "street_buffer":
            preproc_params = {
                "buffer_a": float(number_or_default(buf_a, 10)),
                "buffer_b": float(number_or_default(buf_b, 4)),
                "buffer_l": float(number_or_default(buf_l, 2))}

        build_kwargs = {
            "base_gdf": layer.gdf, "assumptions": assumptions,
            "feature_keys": keys, "resolution_in_m": resolution,
            "fill_value": fill, "dtype": dtype,
            "geometry_buffer_m": geometry_buffer,
            "bounding_polygon": polygon, "preprocessor": preproc,
            "preprocessor_params": preproc_params,
            "preprocessor_steps": preproc_steps or [], "modifiers": modifiers,
            "save_path": save_path or None, "work_dir": state.work_dir,
            "widen_thin_forbidden": widen_thin, "all_touched": all_touched,
            "on_thin_features": on_thin, "crop_border": crop_border}
        finalize = {
            "resolution": resolution, "colormap": colormap, "seq": go,
            "config": {
                "dataset_id": layer.id,
                "feature_keys": list(keys), "resolution_in_m": resolution,
                "fill_value": fill, "dtype": dtype,
                "geometry_buffer_m": geometry_buffer,
                "preprocessor": preproc, "preprocessor_params": preproc_params,
                "preprocessor_steps": preproc_steps or [],
                "widen_thin_forbidden": widen_thin, "all_touched": all_touched,
                "on_thin_features": on_thin, "crop_border": crop_border,
            },
            "combination": _combination_inputs(layer.name, build_kwargs)}

        # a big build (and not one the cache already holds) runs in the
        # background; everything below this block is the synchronous run
        n_cells = validate.estimate_raster_size(bounds, resolution,
                                                dtype)["n_cells"]
        if (rasterize_service.is_heavy(n_cells)
                and _cached_raster(build_kwargs) is None):
            return _start_background_run(state, build_kwargs, finalize,
                                         layer.name, notices, n_cells)

        result, notices = guard(build_cost_raster, **build_kwargs,
                                notices=notices)
        if result is None:
            return no_update, no_update, no_update, notices, *_NO_JOB
        raster_path, log = result
        return (*finalize_rasterize(state, raster_path, log, notices,
                                    **finalize), *_NO_JOB)

    # -------------------------------------------------- load existing raster
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.MAP_VIEW, "data", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.RASTER_LOAD_BTN, "n_clicks"),
        State(ids.RASTER_LOAD_PATH, "value"),
        State(ids.RASTER_COLORMAP, "value"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True,
        running=[(Output(ids.RASTER_LOAD_BTN, "disabled"), True, False)])
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
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
        combo, _ = guard(combinations.ensure_combination_gdf, layer)
        if combo is None and (layer.meta or {}).get("combination_error"):
            # a failed lazy build is shown here (the legend renders on every
            # selection, so a notice per render would be noise); it is logged
            # and the next selection retries
            return html.Small(
                "The cost-combination legend could not be built: "
                f"{layer.meta['combination_error']} — see the log; selecting "
                "the raster again retries.", className="text-danger")
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
        rasters = [ly for ly in state.ordered_layers()
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
        prevent_initial_call=True,
        running=[(Output(ids.GRAD_APPLY_BTN, "disabled"), True, False)])
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
        if (mode or "continuous") not in ("continuous", "unique"):
            problem = numeric_field_problem(
                classes, "Classes", control=ids.GRAD_CLASSES,
                tab=ids.TAB_RASTER, minimum=2, whole=True,
                impact="The raster was not re-classified.")
            if problem is None and classes is not None and float(classes) > 32:
                problem = field_notice(
                    "Classes", "must be at most 32", control=ids.GRAD_CLASSES,
                    tab=ids.TAB_RASTER,
                    impact="The raster was not re-classified.")
            if problem is not None:
                notices.append(problem)
                return no_urls, "check the Classes field", notices
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
                mode, int(number_or_default(classes, 5)), colormap)
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
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
        running=[(Output(ids.RASTER_COMBINE_BUSY, "data"), True, False)],
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
    _ = None
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
