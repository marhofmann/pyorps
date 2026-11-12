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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..state import LAYER_KINDS
from . import rasterize, routing
from .errors import ProjectLayersSkippedWarning, ProjectLoadError

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


def normalise_gui_settings(raw: Any) -> dict[str, dict]:
    """The per-card settings of the page (``state.gui_settings``) as plain JSON.

    ``{"objective": {...}, "cost_field": {...}, "corridor": {...},
    "constrained": {...}}``: each card owns one key and normalises its own
    values when it reads them; this only makes the whole safe to write to and
    read from ``project.json``. Anything that is not a mapping (a project
    saved before round 14 has no key at all) gives ``{}``; an entry that is
    not a mapping, or does not survive JSON, is left out.
    """
    if not isinstance(raw, dict):
        return {}
    settings: dict[str, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            settings[str(key)] = json.loads(json.dumps(value, default=str))
        except (TypeError, ValueError):       # tuple keys, circular refs ...
            continue
    return settings


def manifest_path(path: str | Path) -> Path:
    """The ``project.json`` a save to ``path`` writes (a folder gets one)."""
    path = Path(path)
    return path if path.suffix.lower() == ".json" else path / "project.json"


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
    skipped: list[dict] = []
    path = manifest_path(path)
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
            else:
                # nothing to write: report it instead of saving an entry that
                # load_project could only skip silently
                skipped.append(_skipped(layer, "its raster file is gone "
                                        f"({source or 'no source path'})"))
                continue
        elif layer.kind == "wms":
            # a live tile overlay: the descriptor in meta["wms"] IS the data
            if not (layer.meta or {}).get("wms"):
                skipped.append(_skipped(layer, "it has no WMS descriptor"))
                continue
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
        else:
            skipped.append(_skipped(layer, "it holds no data"))
            continue
        layers.append(entry)

    manifest = {
        "version": MANIFEST_VERSION,
        "project_crs": str(state.project_crs),
        "study_area": state.study_area,
        "active_route_id": state.active_route_id,
        "routing_settings": routing.normalise_routing_settings(
            getattr(state, "routing_settings", None)),
        "raster_settings": rasterize.normalise_raster_settings(
            getattr(state, "raster_settings", None)),
        "gui_settings": normalise_gui_settings(
            getattr(state, "gui_settings", None)),
        "layers": layers,
        "skipped_layers": skipped,
        "cost_table": (cost_table or {}) if "cost_table" in include_set
        else {},
    }
    path.write_text(json.dumps(manifest, indent=2, default=str),
                    encoding="utf-8")
    # everything requested is on disk now - unless layers had to be left out:
    # those still exist only in this session, so the close guard keeps warning
    state.dirty = bool(skipped)
    _warn_skipped("Not saved", skipped)
    return str(path)
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite


def _skipped(layer_or_entry, reason: str) -> dict:
    """One ``skipped_layers`` record for a layer object or a manifest entry."""
    get = (layer_or_entry.get if isinstance(layer_or_entry, dict)
           else lambda key: getattr(layer_or_entry, key, None))
    return {"id": get("id"), "name": get("name"), "kind": get("kind"),
            "reason": reason}


def _warn_skipped(action: str, skipped: list[dict]) -> None:
    """Hand the skipped layers back as ONE warning (``guard`` turns it into a
    notification box; the service return values stay as they were)."""
    if not skipped:
        return
    lines = "; ".join(f"'{rec['name']}' ({rec['kind']}): {rec['reason']}"
                      for rec in skipped)
    warnings.warn(f"{action}: {len(skipped)} layer(s) - {lines}.",
                  ProjectLayersSkippedWarning, stacklevel=3)


def _manifest_meta(layer) -> dict:
    """Layer.meta minus unserializable/derived bits."""
    meta = dict(layer.meta or {})
    meta.pop("combination_error", None)   # runtime note, not project data
    meta.pop("summary", None)
    meta.pop("combination_gdf", None)     # F2: derived GeoDataFrame, re-buildable
    meta.pop("combination_inputs", None)  # lazy-build inputs hold live gdf refs
    meta.pop("staged_from", None)         # an edit staged by the page: not project data
    return json.loads(json.dumps(meta, default=str))


@dataclass
class _StagedLayer:
    """A layer fully read and built, waiting for the swap into the state."""

    entry: dict                       # the manifest entry (view props live here)
    add_args: dict[str, Any]          # keyword arguments for ``state.add_layer``
    tile: Any = None                  # a served raster (its tile server runs!)


def _asset_error(entry: dict, asset: Path, problem: str,
                 cause: BaseException | None = None) -> ProjectLoadError:
    """The error that names the layer and the asset file that failed."""
    error = ProjectLoadError(
        f"Layer '{entry.get('name')}' ({entry.get('kind')}) could not be "
        f"loaded: its asset file '{asset}' {problem}.",
        asset=str(asset), layer=str(entry.get("name")))
    error.__cause__ = cause
    return error


def _stage_layer(entry: dict, root: Path, work_dir: Path):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Read one manifest entry into a :class:`_StagedLayer` WITHOUT touching
    the session. Returns ``(staged, None)``, or ``(None, reason)`` for an entry
    that has nothing to restore; raises :class:`ProjectLoadError` when its asset
    file is missing or unreadable."""
    import geopandas as gpd

    from . import geo
    from .tiles import build_tile_layer

    kind = entry.get("kind")
    meta = entry.get("meta") or {}
    name = entry.get("name") or "layer"
    if kind not in LAYER_KINDS:
        return None, f"unknown layer kind {kind!r}"
    if kind == "wms":
        if not meta.get("wms"):
            return None, "its WMS descriptor is missing"
        return _StagedLayer(entry, {
            "name": name, "kind": "wms", "crs": entry.get("crs"),
            "layer_id": entry.get("id"), "meta": meta}), None
    if kind == "raster":
        if not entry.get("file"):
            return None, "its raster file was not saved"
        tif = root / entry["file"]
        if not tif.exists():
            raise _asset_error(entry, tif, "is missing")
        try:
            tile = build_tile_layer(str(tif), name=name, work_dir=work_dir)
        except Exception as exc:
            raise _asset_error(
                entry, tif, f"could not be read ({type(exc).__name__}: {exc})",
                exc) from exc
        layer_meta = {**meta, "source_path": tile.source_path}
        if layer_meta.get("graduated"):
            # re-register the class LUT (the custom-colormap registry is
            # process local, so the saved "custom:" name is stale)
            from types import SimpleNamespace

            from .graduated import apply_to_layer
            try:
                apply_to_layer(SimpleNamespace(tile=tile, meta=layer_meta),
                               layer_meta["graduated"])
            except Exception:  # nosec B110  # pylint: disable=broad-exception-caught
                # classification is cosmetic — never block a project load on it
                pass
        return _StagedLayer(entry, {
            "name": name, "kind": "raster", "tile": tile,
            "crs": entry.get("crs"), "layer_id": entry.get("id"),
            "meta": layer_meta}, tile=tile), None
    if entry.get("file"):
        gpkg = root / entry["file"]
        if not gpkg.exists():
            raise _asset_error(entry, gpkg, "is missing")
        try:
            gdf = gpd.read_file(gpkg)
            geojson = geo.gdf_to_wgs84_geojson(gdf)
        except Exception as exc:
            raise _asset_error(
                entry, gpkg,
                f"could not be read ({type(exc).__name__}: {exc})", exc) from exc
        return _StagedLayer(entry, {
            "name": name, "kind": kind, "gdf": gdf, "crs": gdf.crs,
            "geojson": geojson, "layer_id": entry.get("id"),
            "meta": meta}), None
    if entry.get("geojson") is not None:
        return _StagedLayer(entry, {
            "name": name, "kind": kind, "geojson": entry["geojson"],
            "layer_id": entry.get("id"), "meta": meta}), None
    return None, "it holds no data"


def _release_staged(staged: list[_StagedLayer]) -> None:
    """Stop the tile servers of layers that were built but never adopted."""
    from ..state import TileManager

    for item in staged:
        if item.tile is not None:
            TileManager.shutdown_client(getattr(item.tile, "tile_client", None))


def load_project(state, path: str | Path) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Restore a saved project into ``state``, all or nothing; returns the manifest.

    Loading is transactional: every layer is read and built FIRST, into a
    staging list that never touches ``state`` (so the tile servers started for
    raster layers are the only side effect). Only when ALL layers loaded is the
    current session replaced, in one step under ``state.locked()``. If any asset
    is missing or corrupt, those tile servers are stopped, ``state`` is exactly
    as it was, and a :class:`~pyorps.gui.services.errors.ProjectLoadError`
    names the layer and the asset file.

    Entries with nothing to restore (an old save without the raster file, an
    unknown kind) are left out and reported: the returned manifest gains
    ``"load_skipped"`` (``[{"id", "name", "kind", "reason"}]``) and the same list
    is raised as one ``ProjectLayersSkippedWarning`` (``guard`` shows it as a
    warning box).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProjectLoadError(
            f"The project file '{path}' could not be read "
            f"({type(exc).__name__}: {exc}).", asset=str(path)) from exc
    root = path.parent

    staged: list[_StagedLayer] = []
    skipped: list[dict] = []
    try:
        for entry in manifest.get("layers", []):
            item, reason = _stage_layer(entry, root, state.work_dir)
            if item is None:
                skipped.append(_skipped(entry, reason))
            else:
                staged.append(item)
    except BaseException:
        _release_staged(staged)
        raise

    with state.locked():                 # the swap: in memory, no I/O
        state.clear()
        state.project_crs = manifest.get("project_crs") or state.project_crs
        state.study_area = manifest.get("study_area")
        for item in staged:
            layer = state.add_layer(**item.add_args)
            entry = item.entry
            layer.visible = bool(entry.get("visible", True))
            layer.z = int(entry.get("z", layer.z))
            layer.style = entry.get("style") or {}
            layer.opacity = float(entry.get("opacity", 0.7))
        # an older project has no routing settings: the page keeps its defaults
        state.routing_settings = routing.normalise_routing_settings(
            manifest.get("routing_settings"))
        # likewise for the rasterize controls
        state.raster_settings = rasterize.normalise_raster_settings(
            manifest.get("raster_settings"))
        # and for the cards of round 14 (objective, cost field, corridor,
        # constrained profile): an older project gives {}
        state.gui_settings = normalise_gui_settings(
            manifest.get("gui_settings"))
        state.active_route_id = manifest.get("active_route_id")
        if state.active_route_id not in state.layers:
            state.active_route_id = None
        state.dirty = False        # freshly loaded == on disk
    manifest["load_skipped"] = skipped
    _warn_skipped("Not restored", skipped)
    return manifest
