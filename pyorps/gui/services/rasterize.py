"""
PYORPS GUI: cost-raster building (R3) — GeoRasterizer + ordered modifiers +
preprocessing, with config-hash caching.

Mirrors the notebook workflow exactly (Section 10.4): rasterize the base
dataset with the edited cost assumptions, then apply each modifier in order
(multiply = factor, override = replace — F12), then save to an explicit path
(F2: never None) written tiled+overviewed for serving (C8 happens in tiles.py
when the layer is served; the routing raster itself is written by pyorps).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import presets
from .cost_model import wrap_assumptions


@dataclass
class ModifierSpec:
    """One ordered modifier condition (F12 + Feature 5).

    ``gdf`` is the modifier dataset already filtered by its
    ``(column, operator, value)`` condition (callbacks/raster._build_modifiers
    does the filtering via cost_model.apply_condition), so a plain scalar
    ``factor`` is applied uniformly to the matching geometries — pyorps'
    ``modify_raster_from_dataset`` only needs the equality-free scalar path.
    """

    gdf: Any                          # condition-filtered GeoDataFrame
    mode: str = "multiply"            # "multiply" | "override"
    factor: float = 1.0               # scalar multiply factor / override value
    buffer_m: float = 0.0
    #: cells holding this value are left alone (None: every cell may change)
    ignore_value: float | None = presets.FORBIDDEN
    #: the library's zone options: features whose ``zone_field`` equals
    #: ``forbidden_zone`` are burnt as ``forbidden_value`` instead of ``factor``
    forbidden_value: int = presets.FORBIDDEN
    zone_field: str | None = None
    forbidden_zone: Any = None
    name: str = ""
    condition: str = "all features"   # human label of the (column, op, value)

    def cost_assumptions(self):
        """The cost_assumptions argument for modify_raster_from_dataset."""
        return float(self.factor)

    @property
    def is_empty(self) -> bool:
        return self.gdf is None or len(self.gdf) == 0


#: rows hashed per chunk, so the WKB copy of a big layer never exists whole
_DIGEST_CHUNK = 50_000


def _referenced_strings(value: Any) -> set[str]:
    """Every string anywhere inside ``value`` (dicts / lists / tuples)."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        out: set[str] = set()
        for key, item in value.items():
            out |= _referenced_strings(key) | _referenced_strings(item)
        return out
    if isinstance(value, (list, tuple, set)):
        out = set()
        for item in value:
            out |= _referenced_strings(item)
        return out
    return set()


def gdf_digest(gdf, columns=None) -> str:
    """Content fingerprint of a GeoDataFrame: geometry + the named columns.

    Hashes the WKB bytes of the geometries (in chunks, so a big layer is never
    copied whole) and the values of ``columns`` only (``None`` = geometry
    only; ``"all"`` = every attribute column). Never serialises to GeoJSON.
    The feature COUNT alone is not a fingerprint: moving one vertex of a
    polygon, or reloading a layer that has the same number of features, must
    not reuse an old raster.
    """
    if gdf is None:
        return "none"
    import numpy as np
    import pandas as pd
    import shapely

    hasher = hashlib.sha256()
    hasher.update(str(gdf.crs).encode("utf-8"))
    hasher.update(len(gdf).to_bytes(8, "little"))
    geometry = gdf.geometry.values
    for start in range(0, len(geometry), _DIGEST_CHUNK):
        for wkb in shapely.to_wkb(np.asarray(geometry[start:start + _DIGEST_CHUNK])):
            hasher.update(wkb if wkb is not None else b"")
    if isinstance(columns, str) and columns == "all":
        columns = [c for c in gdf.columns if c != gdf.geometry.name]
    for column in sorted(c for c in (columns or []) if c in gdf.columns):
        hasher.update(str(column).encode("utf-8"))
        values = gdf[column]
        try:
            hashed = pd.util.hash_pandas_object(values, index=False)
        except TypeError:                       # unhashable cells (dict / list)
            hashed = pd.util.hash_pandas_object(values.astype(str), index=False)
        hasher.update(hashed.to_numpy().tobytes())
    return hasher.hexdigest()[:16]


def _base_digest(base_gdf, feature_keys, preprocessor, preprocessor_params,
                 preprocessor_steps) -> str:
    """Digest of the base layer: geometry + the attribute columns the build
    can read (the cost keys and any column named in a preprocessing step or
    parameter; a NAMED preprocessor reads columns by its own names, so every
    attribute counts then)."""
    if preprocessor and preprocessor != "none":
        columns: Any = "all"
    else:
        named = (_referenced_strings(list(feature_keys))
                 | _referenced_strings(preprocessor_params)
                 | _referenced_strings(preprocessor_steps))
        columns = [c for c in base_gdf.columns if c in named]
    return gdf_digest(base_gdf, columns)


#: the thin-feature policies of ``GeoRasterizer.rasterize(on_thin_features=)``
THIN_POLICIES = ("warn", "raise", "ignore")

#: the rasterize settings a project keeps, in the order of the controls:
#: key -> value of a blank page. They are the library's own defaults, except
#: the geometry buffer: the GUI starts at 1 m (the library at 0 m) because it
#: clears most sub-cell parcel defects of cadastral data.
RASTER_SETTING_DEFAULTS: dict[str, Any] = {
    "resolution_m": 1.0,
    "fill_value": presets.FORBIDDEN,
    "dtype": "uint16",
    "geometry_buffer_m": 1,
    "widen_thin_forbidden": False,
    "all_touched": False,
    "on_thin_features": "warn",
    "crop_border": False,
}
RASTER_SETTING_KEYS = tuple(RASTER_SETTING_DEFAULTS)
RASTER_DTYPES = ("uint16", "uint32", "float32")


def _finite_number(value: Any, default: Any) -> Any:
    """``value`` when it is a plain finite number, else ``default``."""
    import math

    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        return value
    return default


def normalise_raster_settings(raw: Any) -> dict[str, Any]:
    """The rasterize settings of a project file, as the controls take them.

    An older project has none: ``{}``, so the page keeps its own defaults.
    Otherwise every key is present, and a missing or unusable value falls back
    to its default (a typed number is kept as typed: the run checks it and
    names the field).
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    out = dict(RASTER_SETTING_DEFAULTS)
    for key in ("resolution_m", "fill_value", "geometry_buffer_m"):
        out[key] = _finite_number(raw.get(key), RASTER_SETTING_DEFAULTS[key])
    if raw.get("dtype") in RASTER_DTYPES:
        out["dtype"] = raw["dtype"]
    if raw.get("on_thin_features") in THIN_POLICIES:
        out["on_thin_features"] = raw["on_thin_features"]
    for key in ("widen_thin_forbidden", "all_touched", "crop_border"):
        value = raw.get(key)
        out[key] = value if isinstance(value, bool) else RASTER_SETTING_DEFAULTS[key]
    return out


def config_hash(*, feature_keys, assumptions, resolution_in_m, fill_value,
                dtype, geometry_buffer_m, bounds, preprocessor,
                preprocessor_params, modifier_meta, n_features,
                preprocessor_steps=None, data_digest: str = "",
                widen_thin_forbidden: bool = False, all_touched: bool = False,
                on_thin_features: str = "warn",
                crop_border: bool = False) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Stable hash of everything that changes the raster (cache key).

    ``data_digest`` is the content fingerprint of the base layer
    (:func:`gdf_digest`); every modifier carries its own as ``"digest"`` in
    ``modifier_meta``. The feature counts stay in the key as a cheap extra.
    The thin-feature switches and the border crop change the burnt cells (or
    the extent), and the policy decides whether a defect warns or stops the
    build: a cached raster would hide that, so all four are part of the key.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    payload = json.dumps({
        "keys": list(feature_keys), "assumptions": assumptions,
        "res": resolution_in_m, "fill": fill_value, "dtype": dtype,
        "buf": geometry_buffer_m, "bounds": bounds,
        "pre": [preprocessor, preprocessor_params],
        "steps": preprocessor_steps or [],
        "mods": modifier_meta, "n": n_features, "data": data_digest,
        "thin": [bool(widen_thin_forbidden), bool(all_touched),
                 on_thin_features, bool(crop_border)],
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_cost_raster(*, base_gdf, assumptions: dict,
                      feature_keys: tuple[str, ...],
                      resolution_in_m: float = 1.0,
                      fill_value: int = presets.FORBIDDEN,
                      dtype: str = "uint16",
                      geometry_buffer_m: float = 0.0,
                      bounding_polygon=None,
                      preprocessor: str | None = None,
                      preprocessor_params: dict | None = None,
                      preprocessor_steps: list[dict] | None = None,
                      modifiers: list[ModifierSpec] | None = None,
                      save_path: str | Path | None = None,
                      work_dir: str | Path = ".",
                      use_cache: bool = True,
                      widen_thin_forbidden: bool = False,
                      all_touched: bool = False,
                      on_thin_features: str = "warn",
                      crop_border: bool = False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Base rasterize -> ordered modifiers -> optional crop -> save.
    Returns (path, log).

    ``save_path=None`` resolves to a config-hashed file in ``work_dir`` (F2:
    pyorps itself is never called with a None save path / CWD default).
    ``widen_thin_forbidden`` / ``all_touched`` / ``on_thin_features`` go to
    ``GeoRasterizer.rasterize`` as they are (the library's defaults are off /
    off / warn); ``crop_border`` cuts the outer rows and columns that hold
    only forbidden cells (``GeoRasterizer.shrink_raster``) before the save.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps import CostAssumptions, GeoRasterizer, initialize_geo_dataset

    modifiers = list(modifiers or [])
    preprocessor_params = dict(preprocessor_params or {})
    preprocessor_steps = list(preprocessor_steps or [])
    log: list[str] = []

    modifier_meta = [{
        "mode": m.mode, "factor": m.factor, "condition": m.condition,
        "buf": m.buffer_m, "name": m.name,
        "n": 0 if m.gdf is None else int(len(m.gdf)),
        # the modifier's factor is a scalar, so only its geometry is used
        "digest": gdf_digest(m.gdf),
        # how it meets cells that are already forbidden, and its zone rule
        "ignore": m.ignore_value, "forbid": m.forbidden_value,
        "zone": [m.zone_field, m.forbidden_zone],
    } for m in modifiers]
    digest = config_hash(
        feature_keys=feature_keys, assumptions=assumptions,
        resolution_in_m=resolution_in_m, fill_value=fill_value, dtype=dtype,
        geometry_buffer_m=geometry_buffer_m,
        bounds=(tuple(bounding_polygon.bounds)
                if bounding_polygon is not None
                else tuple(base_gdf.total_bounds)),
        preprocessor=preprocessor, preprocessor_params=preprocessor_params,
        preprocessor_steps=preprocessor_steps,
        modifier_meta=modifier_meta, n_features=int(len(base_gdf)),
        data_digest=_base_digest(base_gdf, feature_keys, preprocessor,
                                 preprocessor_params, preprocessor_steps),
        widen_thin_forbidden=widen_thin_forbidden, all_touched=all_touched,
        on_thin_features=on_thin_features, crop_border=crop_border)

    if save_path:
        final_path = Path(save_path)
    else:
        final_path = Path(work_dir) / f"cost_{digest}.tif"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and final_path.exists() and not save_path:
        log.append(f"cache hit — reusing {final_path.name} "
                   f"(config {digest})")
        return str(final_path), log

    named_fn: Callable | None = None
    if preprocessor and preprocessor != "none":
        spec = presets.PREPROCESSORS.get(preprocessor)
        if spec is None or spec["factory"] is None:
            raise ValueError(f"Unknown preprocessor '{preprocessor}'.")
        named_fn = spec["factory"](**preprocessor_params)
        log.append(f"preprocessing: {preprocessor} {preprocessor_params}")

    steps_fn: Callable | None = None
    active_steps = [s for s in preprocessor_steps
                    if (s.get("op") or "").strip().lower()
                    in presets.PREPROC_STEP_OPS]
    if active_steps:
        steps_fn = presets.make_steps_preprocessor(active_steps)
        log.append(f"preprocessing: {len(active_steps)} custom step(s)")

    # compose: the named preprocessor first, then the ordered custom steps
    if named_fn and steps_fn:
        def preprocessing_function(gdf):
            return steps_fn(named_fn(gdf))
    else:
        preprocessing_function = named_fn or steps_fn

    # work on a copy — preprocessors mutate geometry/columns in place
    dataset = initialize_geo_dataset(base_gdf.copy())
    rasterizer = GeoRasterizer(
        dataset, CostAssumptions(wrap_assumptions(assumptions, feature_keys)))
    rasterizer.rasterize(
        resolution_in_m=resolution_in_m, fill_value=fill_value, dtype=dtype,
        geometry_buffer_m=geometry_buffer_m,
        bounding_box=bounding_polygon,
        preprocessing_function=preprocessing_function,
        widen_thin_forbidden=widen_thin_forbidden, all_touched=all_touched,
        on_thin_features=on_thin_features)
    log.append(f"base raster {rasterizer.raster.shape} at "
               f"{resolution_in_m} m ({dtype}, fill={fill_value})")
    if widen_thin_forbidden or all_touched or on_thin_features != "warn":
        log.append(f"thin features: widen={widen_thin_forbidden}, "
                   f"all_touched={all_touched}, on_thin_features="
                   f"{on_thin_features}")

    for spec in modifiers:
        if spec.is_empty:
            log.append(f"modifier '{spec.name}' [{spec.condition}]: no "
                       "matching features — skipped")
            continue
        rasterizer.modify_raster_from_dataset(
            input_data=spec.gdf.copy(),
            cost_assumptions=spec.cost_assumptions(),
            multiply=spec.mode == "multiply",
            geometry_buffer_m=spec.buffer_m,
            ignore_value=spec.ignore_value,
            zone_field=spec.zone_field,
            forbidden_zone=spec.forbidden_zone,
            forbidden_value=spec.forbidden_value)
        log.append(f"modifier '{spec.name}' [{spec.condition}] "
                   f"{spec.mode} {spec.factor:g} on {len(spec.gdf)} feature(s)")

    if crop_border:
        before = rasterizer.raster.shape
        rasterizer.shrink_raster(presets.FORBIDDEN)
        after = rasterizer.raster.shape
        log.append(f"cropped the forbidden border: {before} -> {after}"
                   if after != before
                   else "crop: no outer row or column is fully forbidden")

    rasterizer.save_raster(str(final_path))          # F2: explicit path
    log.append(f"saved {final_path}")
    return str(final_path), log
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
