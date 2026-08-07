"""
Cold-start end-to-end pipeline benchmark (performance plan item 0.4b).

Measures the FULL chain for a vector-input cold start with per-phase
attribution:

    vector load (WFS replay / local file / synthetic)
      -> cost-assumption application
      -> rasterize()            (legacy uint16 flow)
         | rasterize_metrics()  (objective / MetricStack flow)
      -> MetricStack.combine()
      -> RasterHandler windowing + buffer mask
      -> graph / backend construction
      -> solve
      -> path reconstruction
      -> path metrics
      -> export (GeoDataFrame + file write)

The plan claims the data path is 85-97 % of end-to-end wall-clock at
> 5 km. Nothing in this repo measures it, so this harness gates the whole
Phase 2 rasterization workstream and has to be trustworthy:

* Timings are taken by this harness around the PUBLIC API, never read out
  of ``PathFinder.runtimes`` (which is misattributed today: graph
  construction is timed inside "shortest_path" and the cython "total"
  omits components). ``runtimes`` IS recorded alongside, so the two can be
  cross-checked once the accounting fix lands.
* Phases that repeat work already contained in a later phase (the explicit
  cost-assumption application, the standalone combine, the explicit
  endpoint snapping) are flagged ``duplicate_of`` and excluded from the
  totals, so no share is inflated.
* Every phase records RSS at its boundaries — a transient peak *inside* a
  phase is not observed, so ``peak_rss_bytes`` is a lower bound.
* Cell accounting (data extent vs RasterHandler window vs the true
  buffered corridor) sits next to the timings, because the plan's claim is
  that the data path burns 2-13x more cells than the search needs.
* rep 0 is the cold process (it carries the ``import_pyorps`` phase and
  every first-call numba/NVRTC compile); reps >= 1 are warm.

WFS is never hit live during a benchmark run. ``--fetch`` records every
HTTP response of the loader to disk once (a deliberate, user-initiated
network run); the default replays those bytes through the unmodified
loader, so chunk scheduling, GML parsing and dedup are all still measured
- only the network latency is removed (its recorded value is reported
from the cache manifest).

Usage:
    # smoke-test the plumbing on a tiny synthetic fixture (no network)
    .venv/Scripts/python.exe benchmarks/benchmark_coldstart.py --small

    # one deliberate network run that fills the cache
    .venv/Scripts/python.exe benchmarks/benchmark_coldstart.py --fetch

    # repeatable benchmark (default: replay from the cache)
    .venv/Scripts/python.exe benchmarks/benchmark_coldstart.py \
        --extent-km 8 --warm-reps 2

    ... [--pipelines legacy|objective|both] [--graph-api cython]
        [--source wfs|local|synthetic] [--vector FILE]
        [--overlays overlays.json]   # modify_raster_from_dataset kwargs,
                                     # legacy flow only; must be recorded
                                     # by the same --fetch run

Memory note: at the 8 km / 1 m default the objective flow retains ~29 B
per cell (K = 4 metric bands + category + forbidden + combined surface),
i.e. ~1.9 GB at 64 M cells. Use --extent-km 4 on a constrained machine.
"""

import argparse
import gzip
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

# Interpreter-start anchor: everything above this line is stdlib-only, so
# the pyorps/geopandas/rasterio import cost lands inside a measured phase.
_T0 = time.perf_counter()

HERE = Path(__file__).parent
DATA = HERE / "coldstart_data"
RESULTS = HERE / "results"

#: Hessen ALKIS "vereinfachte Nutzung" — the base layer of
#: examples/prepare_data_for_distribution_grid_planning.ipynb.
DEFAULT_WFS = {
    "url": "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
    "layer": "ave_Nutzung",
}

#: Committed cost assumptions of the same notebook (nutzart > bez -> cost).
DEFAULT_COST_CSV = (HERE.parent / "examples" / "data" / "cost_assumptions"
                    / "base_landuse_cost_assumptions.csv")

#: A > 5 km pair inside the notebook's demo area (EPSG:25832).
DEFAULT_SOURCE = (477500.0, 5598500.0)
DEFAULT_TARGET = (482500.0, 5594000.0)

#: Objective for the K = 4 metric-stack flow.
DEFAULT_OBJECTIVE = {"cost": 1.0, "landscape": 800.0, "permit": 50.0,
                     "soil": 10.0}

IMPASSABLE = 65535

#: Phases whose cost is contained in a later phase as well. Measured and
#: reported, but never summed into a total or a share.
DUPLICATE_PHASES = {
    "cost_assumptions_apply": "rasterize / rasterize_metrics",
    "metric_stack_combine": "combine_and_handler",
    "endpoint_snapping": "solve_and_reconstruct",
}

#: Stages that belong to the data path (the 85-97 % claim).
DATA_PATH_PHASES = {
    "import_pyorps", "vector_load", "cost_assumptions_build",
    "rasterize", "rasterize_metrics", "path_finder_shell",
    "combine_and_handler", "raster_handler",
}


# ----------------------------------------------------------------------
# Phase recorder
# ----------------------------------------------------------------------

class PhaseRecorder:
    """Ordered wall-clock + RSS log of the pipeline stages."""

    def __init__(self, psutil_module):
        self._proc = psutil_module.Process(os.getpid())
        self.entries: list[dict] = []

    def rss(self) -> int:
        return int(self._proc.memory_info().rss)

    @contextmanager
    def timed(self, name: str):
        rss_before = self.rss()
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            entry = {
                "phase": name,
                "seconds": seconds,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": self.rss(),
                "data_path": name in DATA_PATH_PHASES
                or name.startswith("overlay_"),
            }
            if name in DUPLICATE_PHASES:
                entry["duplicate_of"] = DUPLICATE_PHASES[name]
                entry["data_path"] = True
            self.entries.append(entry)

    def add_derived(self, name: str, seconds: float, note: str) -> None:
        self.entries.append({"phase": name, "seconds": seconds,
                             "derived": True, "note": note})

    def summarize(self, wall_seconds: float) -> dict:
        counted = [e for e in self.entries
                   if not e.get("derived") and "duplicate_of" not in e]
        duplicated = [e for e in self.entries if "duplicate_of" in e]
        total = sum(e["seconds"] for e in counted)
        data_path = sum(e["seconds"] for e in counted if e["data_path"])
        measured = total + sum(e["seconds"] for e in duplicated)
        return {
            "wall_s": wall_seconds,
            "attributed_s": total,
            "duplicate_s": sum(e["seconds"] for e in duplicated),
            "unattributed_s": wall_seconds - measured,
            "data_path_s": data_path,
            "data_path_share_pct": 100.0 * data_path / total if total else 0.0,
            # Phase-boundary sampler: a transient peak inside a phase (e.g.
            # rasterio's temporaries during a burn) is not observed.
            "peak_rss_bytes": max(
                (max(e["rss_before_bytes"], e["rss_after_bytes"])
                 for e in self.entries if "rss_after_bytes" in e), default=0),
            "peak_rss_sampling": "phase boundaries only",
        }


# ----------------------------------------------------------------------
# WFS response cache (record once, replay forever)
# ----------------------------------------------------------------------

class WFSCache:
    """Disk cache of raw WFS HTTP responses, keyed by url + parameters.

    The cache sits at the ``requests.get`` seam inside
    ``pyorps.io.vector_loader``: chunk scheduling, the WFS version
    fallback loop, GML/GeoJSON parsing and geometry dedup all still run
    on replay. Failed requests are recorded too, so the loader's
    version-fallback path replays identically.
    """

    MANIFEST = "manifest.json"

    def __init__(self, directory: Path):
        self.dir = directory
        self.manifest_path = directory / self.MANIFEST
        self.entries: dict[str, dict] = {}
        self.meta: dict = {}
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text())
            self.entries = payload.get("entries", {})
            self.meta = payload.get("meta", {})

    @property
    def exists(self) -> bool:
        return bool(self.entries)

    @staticmethod
    def key(url: str, params: dict | None) -> str:
        items = sorted((str(k), str(v)) for k, v in (params or {}).items())
        raw = f"{url}?{urlencode(items)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def payload_path(self, key: str) -> Path:
        return self.dir / f"resp_{key}.bin.gz"

    def write(self, key: str, entry: dict, content: bytes | None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if content is not None:
            with gzip.open(self.payload_path(key), "wb") as handle:
                handle.write(content)
        self.entries[key] = entry

    def read(self, key: str) -> bytes:
        with gzip.open(self.payload_path(key), "rb") as handle:
            return handle.read()

    def save(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.manifest_path.write_text(json.dumps(
            {"meta": meta, "entries": self.entries}, indent=1))

    def stats(self) -> dict:
        stored = [e for e in self.entries.values() if e.get("status") == "ok"]
        return {
            "requests": len(self.entries),
            "stored_responses": len(stored),
            "failed_requests": len(self.entries) - len(stored),
            "payload_bytes": sum(e.get("bytes", 0) for e in stored),
            "recorded_network_sum_s": sum(e.get("network_s", 0.0)
                                          for e in self.entries.values()),
            "meta": self.meta,
        }


class _CachedResponse:
    """Minimal stand-in for the parts of requests.Response the loader uses."""

    def __init__(self, content: bytes, headers: dict, status_code: int,
                 http_error):
        self.content = content
        self.headers = headers
        self.status_code = status_code
        self._http_error = http_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise self._http_error(f"cached status {self.status_code}")

    def json(self):
        return json.loads(self.content)


class _RequestsShim:
    """Replaces the ``requests`` module object bound in vector_loader.

    Only ``get`` and ``RequestException`` are used there; keeping the shim
    module-local means the real ``requests`` stays untouched for every
    other consumer in the process.
    """

    def __init__(self, requests_module, cache: WFSCache, record: bool):
        import threading

        self._requests = requests_module
        self._cache = cache
        self._record = record
        self._lock = threading.Lock()
        self.RequestException = requests_module.RequestException
        self.calls = 0
        self.misses: list[str] = []

    def get(self, url, params=None, timeout=None, **kwargs):
        key = self._cache.key(url, params)
        with self._lock:
            self.calls += 1
        if self._record:
            return self._record_get(key, url, params, timeout, **kwargs)
        return self._replay_get(key, url, params)

    def _record_get(self, key, url, params, timeout, **kwargs):
        start = time.perf_counter()
        try:
            response = self._requests.get(url, params=params,
                                          timeout=timeout, **kwargs)
        except self._requests.RequestException as exc:
            with self._lock:
                self._cache.write(key, {
                    "status": "exception",
                    "exception": type(exc).__name__,
                    "message": str(exc)[:400],
                    "url": url,
                    "params": {str(k): str(v) for k, v in (params or {}).items()},
                    "network_s": time.perf_counter() - start,
                }, None)
            raise
        network_s = time.perf_counter() - start
        content = response.content
        with self._lock:
            self._cache.write(key, {
                "status": "ok",
                "status_code": int(response.status_code),
                "content_type": response.headers.get("Content-Type", ""),
                "bytes": len(content),
                "url": url,
                "params": {str(k): str(v) for k, v in (params or {}).items()},
                "network_s": network_s,
            }, content)
        return response

    def _replay_get(self, key, url, params):
        entry = self._cache.entries.get(key)
        if entry is None:
            with self._lock:
                self.misses.append(f"{url} {sorted((params or {}).items())}")
            raise RuntimeError(
                "WFS cache miss — the replay would have to hit the network. "
                "Re-record with --fetch (the request set changes when the "
                "extent, layer or URL changes).")
        if entry["status"] != "ok":
            raise self.RequestException(entry.get("message", "cached failure"))
        return _CachedResponse(
            self._cache.read(key),
            {"Content-Type": entry.get("content_type", "")},
            entry.get("status_code", 200),
            self._requests.HTTPError,
        )


@contextmanager
def wfs_cache_active(cache: WFSCache, record: bool):
    """Route vector_loader's HTTP calls through the cache, then restore."""
    import pyorps.io.vector_loader as vector_loader

    original = vector_loader.requests
    shim = _RequestsShim(original, cache, record)
    vector_loader.requests = shim
    # The capabilities lru_cache would make later reps skip work the first
    # rep paid for; clearing it keeps every rep comparable.
    vector_loader._fetch_capabilities_xml.cache_clear()
    try:
        yield shim
    finally:
        vector_loader.requests = original


# ----------------------------------------------------------------------
# Imports (kept out of module scope so the cost is measurable)
# ----------------------------------------------------------------------

def import_pyorps():
    import psutil
    from rasterio.features import rasterize as rio_rasterize

    from pyorps import CostAssumptions, GeoRasterizer, Objective, PathFinder
    from pyorps.io.geo_dataset import initialize_geo_dataset

    return SimpleNamespace(
        psutil=psutil, rio_rasterize=rio_rasterize,
        CostAssumptions=CostAssumptions, GeoRasterizer=GeoRasterizer,
        Objective=Objective, PathFinder=PathFinder,
        initialize_geo_dataset=initialize_geo_dataset,
    )


# ----------------------------------------------------------------------
# Cost assumptions
# ----------------------------------------------------------------------

def to_multi_metric(assumptions):
    """Derive K = 4 metric leaves from a legacy scalar cost structure.

    ``cost`` is preserved exactly; ``landscape``/``permit``/``soil`` are
    deterministic functions of it, so the metric flow rasterizes and
    combines the same feature set as the legacy flow with K + 2 geometry
    passes instead of one. Forbidden leaves stay scalar 65535 (a metric
    dict mixing 65535 with finite values warns and is redundant).
    """
    if isinstance(assumptions, dict):
        return {key: to_multi_metric(value)
                for key, value in assumptions.items()}
    value = float(assumptions)
    if value >= IMPASSABLE:
        return IMPASSABLE
    return {
        "cost": value,
        "landscape": round(value / 1000.0, 4),
        "permit": round((value % 97) / 100.0, 4),
        "soil": round((value % 13) / 10.0, 4),
    }


def build_cost_assumptions(api, cfg, multi_metric: bool):
    """Return (CostAssumptions, metric_names) for the requested flow."""
    if cfg["source_kind"] == "synthetic":
        base = dict(SYNTHETIC_COSTS)
        nested = {"nutzart": base}
    else:
        loaded = api.CostAssumptions(str(cfg["cost_csv"]))
        keys = [loaded.main_feature, *loaded.side_features]
        nested = {tuple(keys) if len(keys) > 1 else keys[0]:
                  loaded.cost_assumptions}
    if multi_metric:
        key, table = next(iter(nested.items()))
        nested = {key: to_multi_metric(table)}
    assumptions = api.CostAssumptions(nested)
    return assumptions, assumptions.metric_names


# ----------------------------------------------------------------------
# Synthetic fixture (--small / --source synthetic)
# ----------------------------------------------------------------------

SYNTHETIC_COSTS = {
    "acker": 437,
    "wald": 405,
    "wasser": 332,
    "strasse": 350,
    "sperr": IMPASSABLE,
}

SYNTHETIC_ORIGIN = (500000.0, 5600000.0)


def build_synthetic_vector(side_m: float, parcel_m: float, seed: int):
    """A deterministic land-use parcel grid — no network, no fixtures."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    rng = np.random.default_rng(seed)
    classes = list(SYNTHETIC_COSTS)
    weights = np.array([0.45, 0.25, 0.08, 0.19, 0.03])
    n = int(side_m // parcel_m)
    minx, miny = SYNTHETIC_ORIGIN

    names, geoms = [], []
    for row in range(n):
        for col in range(n):
            x0 = minx + col * parcel_m
            y0 = miny + row * parcel_m
            names.append(classes[int(rng.choice(len(classes), p=weights))])
            geoms.append(box(x0, y0, x0 + parcel_m, y0 + parcel_m))
    gdf = gpd.GeoDataFrame({"nutzart": names, "geometry": geoms},
                           crs="EPSG:25832")
    # Endpoints must sit on passable parcels for the run to be meaningful;
    # intersects (not contains) also catches endpoints on a parcel edge.
    for point in synthetic_pair(side_m):
        hit = gdf.geometry.intersects(_point(point))
        gdf.loc[hit, "nutzart"] = "acker"
    return gdf


def _point(xy):
    from shapely.geometry import Point
    return Point(*xy)


def synthetic_pair(side_m: float):
    minx, miny = SYNTHETIC_ORIGIN
    return ((minx + 0.03 * side_m, miny + 0.5 * side_m),
            (minx + 0.97 * side_m, miny + 0.5 * side_m))


# ----------------------------------------------------------------------
# Dataset construction per source kind
# ----------------------------------------------------------------------

def make_dataset_source(api, cfg):
    """Return (dataset_source, load_kwargs, bbox) for initialize_geo_dataset."""
    kind = cfg["source_kind"]
    if kind == "wfs":
        return dict(cfg["wfs"]), {"auto_match": True}, tuple(cfg["bbox"])
    if kind == "local":
        return cfg["vector_path"], {}, tuple(cfg["bbox"])
    if kind == "synthetic":
        return build_synthetic_vector(cfg["synthetic_side_m"],
                                      cfg["synthetic_parcel_m"],
                                      cfg["seed"]), {}, None
    raise ValueError(f"Unknown source kind: {kind}")


# ----------------------------------------------------------------------
# The two pipelines
# ----------------------------------------------------------------------

def _load_vector(api, cfg, phases):
    source, load_kwargs, bbox = make_dataset_source(api, cfg)
    with phases.timed("vector_load"):
        dataset = api.initialize_geo_dataset(source, bbox=bbox)
        dataset.load_data(**load_kwargs)
    if dataset.data is None or len(dataset.data) == 0:
        raise RuntimeError(
            "The vector source returned no features — check the bbox, the "
            "layer name or the cache.")
    return dataset


def _solve_and_report(api, pf, cfg, phases, source_xy, target_xy):
    """Shared tail: graph -> solve -> metrics -> export."""
    with phases.timed("graph_construction"):
        pf.create_graph()

    with phases.timed("endpoint_snapping"):
        pf.get_node_indices_from_coords(source_xy)
        pf.get_node_indices_from_coords(target_xy)

    solve_kwargs = {}
    if cfg["algorithm"] != "dijkstra":
        solve_kwargs["num_threads"] = cfg["num_threads"]
        if cfg["delta"] is not None:
            solve_kwargs["delta"] = cfg["delta"]

    with phases.timed("solve_and_reconstruct"):
        path = pf.find_route(algorithm=cfg["algorithm"],
                             calculate_metrics=False, **solve_kwargs)
    combined = phases.entries[-1]["seconds"]
    kernel_s = float(pf.runtimes.get("shortest_path", 0.0))
    phases.add_derived(
        "solve_kernel", kernel_s,
        "PathFinder.runtimes['shortest_path'] with the graph pre-built; "
        "contained in solve_and_reconstruct")
    phases.add_derived(
        "path_reconstruction", max(combined - kernel_s, 0.0),
        "solve_and_reconstruct minus solve_kernel (coords, LineString, "
        "endpoint snapping repeat)")

    with phases.timed("path_metrics"):
        pf.calculate_path_metrics(path.path_indices, path)

    with phases.timed("export_geodataframe"):
        pf.create_path_geodataframe()

    out_dir = Path(tempfile.mkdtemp(prefix="pyorps_coldstart_"))
    with phases.timed("export_write"):
        pf.save_paths(str(out_dir / "route.geojson"))
    export_bytes = sum(f.stat().st_size for f in out_dir.iterdir())
    for f in out_dir.iterdir():
        f.unlink()
    out_dir.rmdir()
    return path, export_bytes


def run_legacy(api, cfg, phases, source_xy, target_xy):
    """Legacy uint16 flow: rasterize() -> RasterHandler -> solve."""
    dataset = _load_vector(api, cfg, phases)

    with phases.timed("cost_assumptions_build"):
        assumptions, metric_names = build_cost_assumptions(api, cfg, False)

    rasterizer = api.GeoRasterizer(dataset, assumptions)
    with phases.timed("cost_assumptions_apply"):
        rasterizer.cost_manager.apply_to_geodataframe(rasterizer.base_data)

    with phases.timed("rasterize"):
        rasterizer.rasterize(resolution_in_m=cfg["resolution"])

    for index, params in enumerate(cfg["overlays"]):
        label = params.get("label", str(index))
        with phases.timed(f"overlay_{index}_{label}"):
            rasterizer.modify_raster_from_dataset(
                **{k: v for k, v in params.items() if k != "label"})

    with phases.timed("raster_handler"):
        pf = api.PathFinder(
            dataset_source=rasterizer.raster,
            transform=rasterizer.transform,
            crs=rasterizer.crs,
            source_coords=source_xy,
            target_coords=target_xy,
            search_space_buffer_m=cfg["buffer_m"],
            neighborhood_str=cfg["neighborhood"],
            graph_api=cfg["graph_api"],
        )
    pf.geo_rasterizer = rasterizer

    path, export_bytes = _solve_and_report(api, pf, cfg, phases,
                                           source_xy, target_xy)
    retained = {
        "cost_raster_bytes": int(rasterizer.raster.nbytes),
        "metric_stack_bytes": 0,
        "combined_weights_bytes": 0,
        "feasibility_bytes": 0,
    }
    return SimpleNamespace(
        pf=pf, path=path, dataset=dataset,
        extent_shape=tuple(rasterizer.raster.shape),
        n_features=int(len(dataset.data)), metric_names=metric_names,
        retained=retained, export_bytes=export_bytes)


def run_objective(api, cfg, phases, source_xy, target_xy):
    """Objective flow: rasterize_metrics() -> combine() -> handler -> solve."""
    dataset = _load_vector(api, cfg, phases)

    with phases.timed("cost_assumptions_build"):
        assumptions, metric_names = build_cost_assumptions(api, cfg, True)

    rasterizer = api.GeoRasterizer(dataset, assumptions)
    with phases.timed("cost_assumptions_apply"):
        rasterizer.cost_manager.apply_to_geodataframe(rasterizer.base_data)

    with phases.timed("rasterize_metrics"):
        stack = rasterizer.rasterize_metrics(
            resolution_in_m=cfg["resolution"], include_category=True)

    objective_weights = {name: weight
                         for name, weight in cfg["objective"].items()
                         if name in metric_names or name == "length"}
    combine_standalone = None
    if cfg["standalone_combine"]:
        with phases.timed("metric_stack_combine"):
            preview = stack.combine(api.Objective(objective_weights),
                                    quantize=True)
        # Released immediately: keeping it would double-count ~6 B/cell of
        # weights + feasibility against the peak-RSS measurement.
        combine_standalone = {"scale": float(preview.scale),
                              "resolution": float(preview.resolution)}
        del preview

    with phases.timed("path_finder_shell"):
        pf = api.PathFinder(
            dataset_source=dataset.data,
            crs=dataset.data.crs,
            source_coords=source_xy,
            target_coords=None,  # defers the data path to set_objective
            objective=objective_weights,
            search_space_buffer_m=cfg["buffer_m"],
            neighborhood_str=cfg["neighborhood"],
            graph_api=cfg["graph_api"],
        )
    pf.metric_stack = stack
    pf.target_coords = target_xy

    with phases.timed("combine_and_handler"):
        pf.set_objective(objective_weights)

    if cfg["standalone_combine"]:
        combine_s = next(e["seconds"] for e in phases.entries
                         if e["phase"] == "metric_stack_combine")
        both_s = next(e["seconds"] for e in phases.entries
                      if e["phase"] == "combine_and_handler")
        phases.add_derived(
            "raster_handler", max(both_s - combine_s, 0.0),
            "combine_and_handler minus the standalone combine measurement")

    path, export_bytes = _solve_and_report(api, pf, cfg, phases,
                                           source_xy, target_xy)

    result = getattr(pf, "_combine_result", None)
    retained = {
        "cost_raster_bytes": 0,
        "metric_stack_bytes": _stack_bytes(stack),
        "combined_weights_bytes": int(result.weights.nbytes)
        if result is not None else 0,
        "feasibility_bytes": int(result.feasibility.nbytes)
        if result is not None and result.feasibility is not None else 0,
    }
    quantization = None
    if result is not None:
        quantization = {"scale": float(result.scale),
                        "resolution": float(result.resolution),
                        "legacy_passthrough": bool(result.legacy_passthrough)}
    return SimpleNamespace(
        pf=pf, path=path, dataset=dataset, extent_shape=tuple(stack.shape),
        n_features=int(len(dataset.data)), metric_names=metric_names,
        retained=retained, export_bytes=export_bytes,
        quantization=quantization,
        standalone_combine=combine_standalone)


def _stack_bytes(stack) -> int:
    total = sum(stack[name].nbytes for name in stack.layer_names)
    for band in (stack.forbidden_mask, stack.category, stack.dem):
        if band is not None:
            total += int(band.nbytes)
    return int(total)


# ----------------------------------------------------------------------
# Cell accounting: extent vs window vs corridor
# ----------------------------------------------------------------------

def cell_accounting(api, outcome):
    """extent / window / corridor cell counts — the core Phase 2 claim."""
    handler = outcome.pf.raster_handler
    window = handler.window
    extent_rows, extent_cols = outcome.extent_shape
    window_cells = int(window.height) * int(window.width)

    inside = api.rio_rasterize(
        [(handler.buffer_geometry, 1)],
        out_shape=(int(window.height), int(window.width)),
        transform=handler.window_transform,
        fill=0,
        dtype="uint8",
    )
    corridor_cells = int(inside.sum())
    extent_cells = int(extent_rows) * int(extent_cols)
    path_cells = int(len(outcome.path.path_indices))

    return {
        "extent_shape": [int(extent_rows), int(extent_cols)],
        "extent_cells": extent_cells,
        "window_shape": [int(window.height), int(window.width)],
        "window_cells": window_cells,
        "corridor_cells": corridor_cells,
        "path_cells": path_cells,
        "search_space_buffer_m": float(handler.search_space_buffer_m),
        "extent_over_window": extent_cells / window_cells if window_cells else 0,
        "window_over_corridor": (window_cells / corridor_cells
                                 if corridor_cells else 0),
        "extent_over_corridor": (extent_cells / corridor_cells
                                 if corridor_cells else 0),
        "corridor_over_path": (corridor_cells / path_cells
                               if path_cells else 0),
    }


# ----------------------------------------------------------------------
# One repetition
# ----------------------------------------------------------------------

def run_once(api, cfg, cache, pipeline: str, phases,
             extra_wall: float = 0.0) -> dict:
    """One full cold-start pipeline; ``extra_wall`` carries pre-recorded
    phases (the interpreter import) into the wall-clock accounting."""
    source_xy = tuple(cfg["source_xy"])
    target_xy = tuple(cfg["target_xy"])
    http_calls = None

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # The cache wraps the WHOLE run, so no stage — base layer or
        # overlay — can reach the network during a benchmark.
        if cfg["source_kind"] == "wfs":
            with wfs_cache_active(cache, record=False) as shim:
                outcome = _dispatch(api, cfg, phases, pipeline,
                                    source_xy, target_xy)
                http_calls = shim.calls
        else:
            outcome = _dispatch(api, cfg, phases, pipeline,
                                source_xy, target_xy)
    wall = time.perf_counter() - start + extra_wall

    path = outcome.path
    summary = phases.summarize(wall)
    cells = cell_accounting(api, outcome)
    retained_total = sum(outcome.retained.values())

    unique_warnings = []
    seen = set()
    for item in caught:
        text = f"{item.category.__name__}: {str(item.message)[:200]}"
        if text not in seen:
            seen.add(text)
            unique_warnings.append(text)
        if len(unique_warnings) >= 20:
            break

    return {
        "pipeline": pipeline,
        "summary": summary,
        "phases": phases.entries,
        "cells": cells,
        "features": outcome.n_features,
        "metric_names": list(outcome.metric_names),
        "retained_bytes": {**outcome.retained, "total": retained_total},
        "retained_bytes_per_extent_cell": (
            retained_total / cells["extent_cells"]
            if cells["extent_cells"] else 0.0),
        "export_bytes": outcome.export_bytes,
        "quantization": getattr(outcome, "quantization", None),
        "http_calls": http_calls,
        "overlays_applied": (len(cfg["overlays"]) if pipeline == "legacy"
                             else 0),
        "overlays_skipped_reason": (
            "the metric pipeline rejects datasets_to_modify "
            "(path_finder.py:_create_metric_raster_handler)"
            if cfg["overlays"] and pipeline == "objective" else None),
        "route": {
            "total_length_m": _maybe_float(path.total_length),
            "total_cost": _maybe_float(path.total_cost),
            "euclidean_m": _maybe_float(path.euclidean_distance),
            "detour_factor": (path.total_length / path.euclidean_distance
                              if path.total_length and path.euclidean_distance
                              else None),
            "metrics": path.metrics,
            "feasibility": _maybe_float(path.feasibility),
        },
        "pathfinder_runtimes": dict(outcome.pf.runtimes),
        "warnings": unique_warnings,
    }


def _dispatch(api, cfg, phases, pipeline, source_xy, target_xy):
    if pipeline == "legacy":
        return run_legacy(api, cfg, phases, source_xy, target_xy)
    return run_objective(api, cfg, phases, source_xy, target_xy)


def _maybe_float(value):
    return float(value) if value is not None else None


def _json_default(value):
    """Keep numbers numeric in the result file; stringify the rest."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_run(record: dict) -> None:
    summary = record["summary"]
    print(f"\n--- {record['pipeline']} pipeline, rep {record['rep']} "
          f"({record['process']}) ---")
    print(f"  {'phase':<30}{'seconds':>10}{'% attr':>8}   rss_after")
    total = summary["attributed_s"] or 1.0
    for entry in record["phases"]:
        if entry.get("derived"):
            print(f"  {entry['phase']:<30}{entry['seconds']:>10.3f}"
                  f"{'':>8}   (derived)")
            continue
        tag = " [dup]" if "duplicate_of" in entry else ""
        share = "" if "duplicate_of" in entry else \
            f"{100.0 * entry['seconds'] / total:>7.1f}%"
        print(f"  {entry['phase'] + tag:<30}{entry['seconds']:>10.3f}"
              f"{share:>8}   {entry['rss_after_bytes'] / 2**20:>8.0f} MiB")
    print(f"  {'TOTAL (attributed)':<30}{summary['attributed_s']:>10.3f}")
    print(f"  {'wall clock':<30}{summary['wall_s']:>10.3f}"
          f"   (unattributed {summary['unattributed_s']:+.3f} s)")
    print(f"  DATA PATH SHARE: {summary['data_path_share_pct']:.1f} %  "
          f"({summary['data_path_s']:.2f} s of {summary['attributed_s']:.2f} s)")
    print(f"  peak RSS {summary['peak_rss_bytes'] / 2**20:.0f} MiB, "
          f"retained {record['retained_bytes']['total'] / 2**20:.0f} MiB "
          f"({record['retained_bytes_per_extent_cell']:.1f} B/cell)")

    cells = record["cells"]
    print(f"  cells: extent {cells['extent_cells']:,} "
          f"({cells['extent_shape'][0]}x{cells['extent_shape'][1]}) | "
          f"window {cells['window_cells']:,} | "
          f"corridor {cells['corridor_cells']:,} | "
          f"path {cells['path_cells']:,}")
    print(f"         extent/window x{cells['extent_over_window']:.2f}  "
          f"window/corridor x{cells['window_over_corridor']:.2f}  "
          f"extent/corridor x{cells['extent_over_corridor']:.2f}")
    route = record["route"]
    if route["total_length_m"]:
        print(f"  route: {route['total_length_m']:.0f} m "
              f"(detour x{route['detour_factor']:.2f})")
    for text in record["warnings"][:5]:
        print(f"  warning: {text}")


def aggregate(records: list[dict]) -> dict:
    """Cold (rep 0) vs warm (rep >= 1) split, per pipeline."""
    out = {}
    for pipeline in sorted({r["pipeline"] for r in records}):
        rows = [r for r in records if r["pipeline"] == pipeline]
        cold = [r for r in rows if r["rep"] == 0]
        warm = [r for r in rows if r["rep"] > 0]
        entry = {
            "cold_wall_s": [r["summary"]["wall_s"] for r in cold],
            "warm_wall_s": [r["summary"]["wall_s"] for r in warm],
            "cold_data_path_share_pct": [r["summary"]["data_path_share_pct"]
                                         for r in cold],
            "warm_data_path_share_pct": [r["summary"]["data_path_share_pct"]
                                         for r in warm],
        }
        if warm:
            entry["warm_wall_median_s"] = statistics.median(
                entry["warm_wall_s"])
            entry["cold_minus_warm_s"] = (entry["cold_wall_s"][0]
                                          - entry["warm_wall_median_s"])
        out[pipeline] = entry
    return out


# ----------------------------------------------------------------------
# Configuration / CLI
# ----------------------------------------------------------------------

def record_wfs_cache(api, cfg, cache: WFSCache) -> None:
    """The one deliberate network pass: fill the cache, then get out.

    Kept separate from the benchmark reps so that no measured run can ever
    issue an HTTP request, and so that a multi-rep benchmark does not hit
    the state endpoints once per repetition.
    """
    source, load_kwargs, bbox = make_dataset_source(api, cfg)
    start = time.perf_counter()
    with wfs_cache_active(cache, record=True) as shim:
        dataset = api.initialize_geo_dataset(source, bbox=bbox)
        dataset.load_data(**load_kwargs)
        n_features = 0 if dataset.data is None else int(len(dataset.data))
        if cfg["overlays"]:
            # Overlay requests are issued by modify_raster_from_dataset with
            # a bbox derived from the rasterized extent — the only way to
            # record exactly the request set the replay will ask for is to
            # run the burn once here.
            assumptions, _ = build_cost_assumptions(api, cfg, False)
            rasterizer = api.GeoRasterizer(dataset, assumptions)
            rasterizer.rasterize(resolution_in_m=cfg["resolution"])
            for index, params in enumerate(cfg["overlays"]):
                rasterizer.modify_raster_from_dataset(
                    **{k: v for k, v in params.items() if k != "label"})
                print(f"  overlay {index} recorded")
        calls = shim.calls
    wall = time.perf_counter() - start

    cache.save({
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": cfg["wfs"]["url"],
        "layer": cfg["wfs"]["layer"],
        "bbox": cfg["bbox"],
        "network_wall_s": wall,
        "http_calls": calls,
        "features": n_features,
    })
    stats = cache.stats()
    print(f"WFS cache recorded: {n_features:,} features, {calls} requests, "
          f"{stats['payload_bytes'] / 2**20:.1f} MiB, {wall:.1f} s of "
          f"network wall-clock -> {cache.dir}")


def build_config(args) -> dict:
    source_xy = tuple(args.source_xy)
    target_xy = tuple(args.target_xy)

    if args.small:
        side = args.synthetic_side_m
        source_xy, target_xy = synthetic_pair(side)

    half = args.extent_km * 500.0
    midx = 0.5 * (source_xy[0] + target_xy[0])
    midy = 0.5 * (source_xy[1] + target_xy[1])
    bbox = (midx - half, midy - half, midx + half, midy + half)

    return {
        "source_kind": args.source,
        "wfs": {"url": args.url, "layer": args.layer},
        "vector_path": args.vector,
        "cost_csv": str(args.cost_csv),
        "bbox": list(bbox),
        "extent_km": args.extent_km,
        "resolution": args.resolution,
        "buffer_m": args.buffer,
        "neighborhood": args.neighborhood,
        "graph_api": args.graph_api,
        "algorithm": args.algorithm,
        "num_threads": args.num_threads,
        "delta": args.delta,
        "objective": DEFAULT_OBJECTIVE,
        "source_xy": list(source_xy),
        "target_xy": list(target_xy),
        "fetch": args.fetch,
        "overlays": (json.loads(args.overlays.read_text())
                     if args.overlays else []),
        "standalone_combine": not args.no_standalone_combine,
        "synthetic_side_m": args.synthetic_side_m,
        "synthetic_parcel_m": args.synthetic_parcel_m,
        "seed": args.seed,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cold-start end-to-end pipeline benchmark (plan 0.4b)")
    parser.add_argument("--source", choices=("wfs", "local", "synthetic"),
                        default="wfs")
    parser.add_argument("--fetch", action="store_true",
                        help="deliberate network run that records the WFS "
                             "responses into the cache (never automatic)")
    parser.add_argument("--cache-dir", type=Path,
                        default=DATA / "wfs_cache")
    parser.add_argument("--url", default=DEFAULT_WFS["url"])
    parser.add_argument("--layer", default=DEFAULT_WFS["layer"])
    parser.add_argument("--vector", default=None,
                        help="vector file for --source local")
    parser.add_argument("--cost-csv", type=Path, default=DEFAULT_COST_CSV)
    parser.add_argument("--extent-km", type=float, default=8.0,
                        help="square data extent around the route midpoint")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--buffer", type=float, default=700.0,
                        help="search_space_buffer_m (omit value -1 for auto)")
    parser.add_argument("--neighborhood", default="r2")
    parser.add_argument("--graph-api", default="cython")
    parser.add_argument("--algorithm", default="dijkstra")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--pipelines", choices=("legacy", "objective", "both"),
                        default="both")
    parser.add_argument("--overlays", type=Path, default=None,
                        help="JSON list of modify_raster_from_dataset kwargs "
                             "(legacy flow only; the metric pipeline rejects "
                             "datasets_to_modify). Each entry may carry a "
                             "'label' used as the phase name.")
    parser.add_argument("--warm-reps", type=int, default=1,
                        help="repetitions AFTER the cold one (rep 0)")
    parser.add_argument("--no-standalone-combine", action="store_true",
                        help="skip the extra MetricStack.combine measurement")
    parser.add_argument("--source-xy", type=float, nargs=2,
                        default=list(DEFAULT_SOURCE))
    parser.add_argument("--target-xy", type=float, nargs=2,
                        default=list(DEFAULT_TARGET))
    parser.add_argument("--synthetic-side-m", type=float, default=640.0)
    parser.add_argument("--synthetic-parcel-m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--small", "--dry-run", action="store_true",
                        dest="small",
                        help="tiny synthetic fixture, no network — smoke "
                             "test for the plumbing")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.small:
        args.source = "synthetic"
        args.warm_reps = min(args.warm_reps, 1)
        args.buffer = min(args.buffer, 100.0)
    if args.buffer is not None and args.buffer < 0:
        args.buffer = None
    if args.source == "local" and not args.vector:
        parser.error("--source local requires --vector FILE")
    if args.fetch and args.source != "wfs":
        parser.error("--fetch only applies to --source wfs")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    cache = WFSCache(args.cache_dir)
    if cfg["source_kind"] == "wfs" and not args.fetch and not cache.exists:
        print(f"No WFS cache at {args.cache_dir}.\n"
              f"Run once with --fetch to record it (that run DOES hit the "
              f"network), or use --source synthetic / --small.",
              file=sys.stderr)
        return 2

    startup_s = time.perf_counter() - _T0
    import_start = time.perf_counter()
    api = import_pyorps()
    import_s = time.perf_counter() - import_start

    if args.fetch and cfg["source_kind"] == "wfs":
        record_wfs_cache(api, cfg, cache)
        print("NOTE: the recording pass already warmed this interpreter — "
              "rep 0 of a --fetch run is not a cold measurement. Re-run "
              "without --fetch for the cold numbers.")

    pipelines = (["legacy", "objective"] if args.pipelines == "both"
                 else [args.pipelines])

    records = []
    for rep in range(args.warm_reps + 1):
        for pipeline in pipelines:
            phases = PhaseRecorder(api.psutil)
            extra_wall = 0.0
            if rep == 0 and pipeline == pipelines[0]:
                # The import is paid once per process and belongs to the
                # cold accounting; later runs inherit a warm interpreter.
                phases.entries.append({
                    "phase": "import_pyorps", "seconds": import_s,
                    "rss_before_bytes": 0, "rss_after_bytes": phases.rss(),
                    "data_path": True,
                })
                extra_wall = import_s
            record = run_once(api, cfg, cache, pipeline, phases,
                              extra_wall=extra_wall)
            record["rep"] = rep
            record["process"] = "cold" if rep == 0 else "warm"
            records.append(record)
            print_run(record)

    payload = {
        "config": cfg,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "interpreter_startup_s": startup_s,
            "import_pyorps_s": import_s,
            "cold_measurement_valid": not args.fetch,
        },
        "wfs_cache": cache.stats() if cfg["source_kind"] == "wfs" else None,
        "runs": records,
        "aggregate": aggregate(records),
    }

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or RESULTS / f"coldstart_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=_json_default))

    print("\n" + "=" * 66)
    for pipeline, entry in payload["aggregate"].items():
        cold = entry["cold_wall_s"][0]
        warm = entry.get("warm_wall_median_s")
        share = entry["cold_data_path_share_pct"][0]
        warm_text = f"{warm:.2f} s" if warm is not None else "n/a"
        print(f"{pipeline:<10} cold {cold:7.2f} s | warm {warm_text:>9} | "
              f"data path {share:5.1f} %")
    print(f"results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
