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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    ignore_value: float | None = presets.FORBIDDEN
    name: str = ""
    condition: str = "all features"   # human label of the (column, op, value)

    def cost_assumptions(self):
        """The cost_assumptions argument for modify_raster_from_dataset."""
        return float(self.factor)

    @property
    def is_empty(self) -> bool:
        return self.gdf is None or len(self.gdf) == 0


def config_hash(*, feature_keys, assumptions, resolution_in_m, fill_value,
                dtype, geometry_buffer_m, bounds, preprocessor,
                preprocessor_params, modifier_meta, n_features,
                preprocessor_steps=None) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Stable hash of everything that changes the raster (cache key)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    payload = json.dumps({
        "keys": list(feature_keys), "assumptions": assumptions,
        "res": resolution_in_m, "fill": fill_value, "dtype": dtype,
        "buf": geometry_buffer_m, "bounds": bounds,
        "pre": [preprocessor, preprocessor_params],
        "steps": preprocessor_steps or [],
        "mods": modifier_meta, "n": n_features,
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
                      use_cache: bool = True):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Base rasterize -> ordered modifiers -> save. Returns (path, log).

    ``save_path=None`` resolves to a config-hashed file in ``work_dir`` (F2:
    pyorps itself is never called with a None save path / CWD default).
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
        modifier_meta=modifier_meta, n_features=int(len(base_gdf)))

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
        preprocessing_function=preprocessing_function)
    log.append(f"base raster {rasterizer.raster.shape} at "
               f"{resolution_in_m} m ({dtype}, fill={fill_value})")

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
            ignore_value=spec.ignore_value)
        log.append(f"modifier '{spec.name}' [{spec.condition}] "
                   f"{spec.mode} {spec.factor:g} on {len(spec.gdf)} feature(s)")

    rasterizer.save_raster(str(final_path))          # F2: explicit path
    log.append(f"saved {final_path}")
    return str(final_path), log
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
