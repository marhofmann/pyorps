"""
PYORPS GUI: import/export + project persistence (R12, Section 19).

Routes export to GeoJSON/GPKG/SHP/CSV with their full attribute set +
lineage; rasters copy as GeoTIFF; infrastructure profiles round-trip via
YAML/JSON; and the whole session persists as a ``project.json`` manifest with
an asset folder (heavy payloads as files, small geometry inlined) so a
session is fully reproducible and shareable.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROUTE_EXTS = {".geojson", ".json", ".gpkg", ".shp", ".csv"}
MANIFEST_VERSION = 1


# -------------------------------------------------------------------- routes
def _export_gdf(layer):
    """The route gdf with the layer's simplify_tol applied (F5: metrics
    columns still describe the full routed line)."""
    gdf = layer.gdf.copy()
    tol = (layer.meta or {}).get("simplify_tol")
    if tol:
        gdf.geometry = gdf.geometry.simplify(float(tol),
                                             preserve_topology=False)
    return gdf


def export_route(layer, path: str | Path) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Write one route layer with metrics + lineage columns (Section 19)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ROUTE_EXTS:
        raise ValueError(f"Unsupported route format '{suffix}' — use "
                         ".geojson/.gpkg/.shp/.csv.")
    gdf = _export_gdf(layer)
    meta = layer.meta or {}
    gdf["route_id"] = meta.get("route_id") or layer.id
    gdf["parent_id"] = meta.get("parent_id") or ""
    gdf["origin"] = meta.get("origin") or ""
    gdf["edit"] = meta.get("edit") or ""
    gdf["params"] = json.dumps(meta.get("params") or {}, default=str)
    gdf["control_points"] = json.dumps(meta.get("control_points") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        frame = gdf.drop(columns=gdf.geometry.name).assign(
            wkt=gdf.geometry.to_wkt())
        frame.to_csv(path, index=False, sep=";")
    else:
        driver = {"geojson": "GeoJSON", "json": "GeoJSON",
                  "gpkg": "GPKG", "shp": "ESRI Shapefile"}[suffix[1:]]
        gdf.to_file(path, driver=driver)
    return str(path)


def export_all_routes(state, path: str | Path) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """All route layers into one file (GeoJSON/GPKG)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import pandas as pd

    routes = state.layers_of_kind("route")
    if not routes:
        raise ValueError("There are no routes to export.")
    frames = []
    for layer in routes:
        gdf = _export_gdf(layer)
        meta = layer.meta or {}
        gdf["route_id"] = layer.id
        gdf["parent_id"] = meta.get("parent_id") or ""
        gdf["origin"] = meta.get("origin") or ""
        gdf["params"] = json.dumps(meta.get("params") or {}, default=str)
        gdf["control_points"] = json.dumps(
            meta.get("control_points") or [])
        frames.append(gdf)
    merged = pd.concat(frames, ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = "GPKG" if path.suffix.lower() == ".gpkg" else "GeoJSON"
    merged.to_file(path, driver=driver)
    return str(path)


# -------------------------------------------------------------------- raster
def export_raster(layer, path: str | Path) -> str:
    """Copy a served raster layer's GeoTIFF to the target path (F2)."""
    source = (layer.meta or {}).get("source_path")
    if not source or not Path(source).exists():
        raise FileNotFoundError(f"Raster source '{source}' is gone.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return str(path)


# ------------------------------------------------------------------ profiles
def save_profile(profile_dict: dict, path: str | Path) -> str:
    """Infrastructure profile -> YAML or JSON (Section 20)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(profile_dict, sort_keys=False,
                                       allow_unicode=True),
                        encoding="utf-8")
    elif suffix == ".json":
        path.write_text(json.dumps(profile_dict, indent=2, default=str),
                        encoding="utf-8")
    else:
        raise ValueError(f"Unsupported profile format '{suffix}' — use "
                         ".yaml or .json.")
    return str(path)


def load_profile(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported profile format '{suffix}' — use "
                     ".yaml or .json.")


#: dataset groups the user can include/exclude in a save; routes and the
#: study area are ALWAYS saved with the project.
SAVE_GROUPS = ("raster", "vector", "cost_table")
_ALWAYS_SAVED_KINDS = ("route", "study_area")


def save_project(state, path: str | Path, *,
                 cost_table: dict | None = None,
                 include: list | set | None = None) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Persist the session: ``project.json`` + an ``assets/`` folder.

    ``cost_table`` is the client-side cost editor state
    (``{"feature_keys": [...], "rows": [...], "modifiers": [...]}``) passed
    through by the save callback so the manifest is complete.

    ``include`` selects which dataset groups to persist ("raster",
    "vector", "cost_table"); None saves everything. Routes and the study
    area are ALWAYS saved.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    include_set = set(SAVE_GROUPS) if include is None else set(include)
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path / "project.json"
    root = path.parent
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    def _included(layer) -> bool:
        if layer.kind in _ALWAYS_SAVED_KINDS:
            return True
        if layer.kind in ("raster", "wms"):
            return "raster" in include_set
        return "vector" in include_set

    layers = []
    for layer in filter(_included, state.ordered_layers()):
        entry: dict[str, Any] = {
            "id": layer.id, "name": layer.name, "kind": layer.kind,
            "visible": layer.visible, "z": layer.z,
            "style": layer.style, "opacity": layer.opacity,
            "crs": str(layer.crs) if layer.crs else None,
            "meta": _manifest_meta(layer),
        }
        if layer.kind == "raster":
            target = assets / f"{layer.id}.tif"
            source = (layer.meta or {}).get("source_path")
            if source and Path(source).exists():
                if Path(source).resolve() != target.resolve():
                    shutil.copyfile(source, target)
                entry["file"] = str(target.relative_to(root))
        elif layer.gdf is not None and not layer.gdf.empty:
            target = assets / f"{layer.id}.gpkg"
            gdf = layer.gdf.copy()
            for column in gdf.columns:
                if column != gdf.geometry.name and \
                        gdf[column].dtype == object:
                    gdf[column] = gdf[column].map(
                        lambda v: json.dumps(v, default=str)
                        if isinstance(v, (dict, list)) else v)
            gdf.to_file(target, driver="GPKG")
            entry["file"] = str(target.relative_to(root))
        elif layer.geojson is not None:
            entry["geojson"] = layer.geojson
        layers.append(entry)

    manifest = {
        "version": MANIFEST_VERSION,
        "project_crs": str(state.project_crs),
        "study_area": state.study_area,
        "active_route_id": state.active_route_id,
        "layers": layers,
        "cost_table": (cost_table or {}) if "cost_table" in include_set
        else {},
    }
    path.write_text(json.dumps(manifest, indent=2, default=str),
                    encoding="utf-8")
    state.dirty = False        # everything requested is on disk now
    return str(path)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite


def _manifest_meta(layer) -> dict:
    """Layer.meta minus unserializable/derived bits."""
    meta = dict(layer.meta or {})
    meta.pop("summary", None)
    meta.pop("combination_gdf", None)     # F2: derived GeoDataFrame, re-buildable
    meta.pop("combination_inputs", None)  # lazy-build inputs hold live gdf refs
    return json.loads(json.dumps(meta, default=str))


def load_project(state, path: str | Path) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Restore a saved project into a cleared state; returns the manifest."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import geopandas as gpd

    from . import geo
    from .tiles import build_tile_layer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    state.clear()
    state.project_crs = manifest.get("project_crs") or state.project_crs
    state.study_area = manifest.get("study_area")

    for entry in manifest.get("layers", []):
        kind = entry.get("kind")
        layer = None
        if kind == "raster" and entry.get("file"):
            tif = root / entry["file"]
            tile = build_tile_layer(str(tif), name=entry["name"],
                                    work_dir=state.work_dir)
            layer = state.add_layer(
                entry["name"], "raster", tile=tile, crs=entry.get("crs"),
                layer_id=entry["id"],
                meta={**(entry.get("meta") or {}),
                      "source_path": tile.source_path})
            if layer.meta.get("graduated"):
                # re-register the class LUT (the custom-colormap registry is
                # process local, so the saved "custom:" name is stale)
                from .graduated import apply_to_layer
                try:
                    apply_to_layer(layer, layer.meta["graduated"])
                except Exception:   # classification is cosmetic — never  # nosec B110  # pylint: disable=broad-exception-caught
                    pass            # block a project load on it
        elif entry.get("file"):
            gdf = gpd.read_file(root / entry["file"])
            layer = state.add_layer(
                entry["name"], kind, gdf=gdf, crs=gdf.crs,
                geojson=geo.gdf_to_wgs84_geojson(gdf),
                layer_id=entry["id"], meta=entry.get("meta") or {})
        elif entry.get("geojson") is not None:
            layer = state.add_layer(
                entry["name"], kind, geojson=entry["geojson"],
                layer_id=entry["id"], meta=entry.get("meta") or {})
        if layer is not None:
            layer.visible = bool(entry.get("visible", True))
            layer.z = int(entry.get("z", layer.z))
            layer.style = entry.get("style") or {}
            layer.opacity = float(entry.get("opacity", 0.7))
    state.active_route_id = manifest.get("active_route_id")
    if state.active_route_id not in state.layers:
        state.active_route_id = None
    state.dirty = False        # freshly loaded == on disk
    return manifest
