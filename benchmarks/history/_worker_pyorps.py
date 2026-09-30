"""Run ONE benchmark job against ONE pyorps checkout, in its own process.

Invoked as ``python _worker_pyorps.py <job.json>``; writes ``job["out_json"]``.

Runs in a *separate process* for three reasons: two pyorps versions cannot
coexist in one interpreter, a crash in one version must not take the whole run
down, and peak memory is only meaningful per process.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

JOB = json.loads(open(sys.argv[1], encoding="utf-8").read())


def peak_rss_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or getattr(mi, "rss", None)
        return round(peak / 2 ** 30, 3) if peak else None
    except Exception:
        return None


REC = {k: JOB[k] for k in ("job_id", "engine", "version", "kind", "neighborhood",
                           "graph_api", "algorithm", "use_gpu", "repeat",
                           "exec_mode") if k in JOB}
REC["pid"] = os.getpid()
REC["status"] = "started"


def dump(status, **kw):
    REC.update(kw)
    REC["status"] = status
    REC["peak_rss_gb"] = peak_rss_gb()
    tmp = JOB["out_json"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(REC, fh, indent=1, default=str)
    os.replace(tmp, JOB["out_json"])
    brief = {k: v for k, v in REC.items() if k != "paths"}
    print(json.dumps(brief, default=str)[:1500], flush=True)


# ---------------------------------------------------------------- import setup
CHECKOUT = os.path.abspath(JOB["checkout"])

# The venv carries an editable-install finder pointing at the MAIN working
# tree. It is a sys.meta_path finder and would shadow the checkout under test,
# silently benchmarking HEAD while the record claims v0.1.0.
sys.meta_path = [
    f for f in sys.meta_path
    if "editable" not in getattr(type(f), "__module__", "").lower()
    and "editable" not in getattr(f, "__name__", "").lower()
]
sys.path.insert(0, CHECKOUT)

try:
    import numpy as np
    import geopandas as gpd
    import pyorps
except Exception:
    dump("import_error", error=traceback.format_exc())
    sys.exit(1)

resolved = os.path.abspath(pyorps.__file__)
if not resolved.lower().startswith(CHECKOUT.lower()):
    dump("wrong_checkout", pyorps_file=resolved,
         error="imported pyorps from " + resolved + ", expected under " + CHECKOUT)
    sys.exit(1)

REC["pyorps_file"] = resolved
REC["pyorps_version"] = getattr(pyorps, "__version__", "?")
REC["python"] = sys.version.split()[0]

# networkit >= 11 segfaults when addEdges() is handed uint32 index arrays.
# pyorps only started casting to uint64 later, so every pre-fix version dies
# with SIGSEGV on today's networkit. The shim is recorded in the result, so a
# measurement taken with it can never be mistaken for unpatched behaviour.
try:
    import pyorps.graph.api.networkit_api as _nk

    if "uint64" not in open(_nk.__file__, encoding="utf-8").read():
        _orig_create_graph = _nk.NetworkitAPI.create_graph

        def _patched_create_graph(self, from_nodes, to_nodes, cost=None, **kw):
            return _orig_create_graph(self,
                                      np.asarray(from_nodes, dtype=np.uint64),
                                      np.asarray(to_nodes, dtype=np.uint64),
                                      cost, **kw)

        _nk.NetworkitAPI.create_graph = _patched_create_graph
        REC["nk_uint64_shim"] = True
except Exception as exc:  # pragma: no cover
    REC["nk_shim_error"] = repr(exc)

# HEAD refuses graph-library backends above MAX_LIBRARY_BACKEND_CELLS
# (5,000,000) because they preallocate an explicit edge list -- our 38 Mcell
# window is far above it. That guard did not exist before, so without this
# override HEAD would be the ONLY version missing from the controlled
# networkit comparison, and the gap would look like a measurement failure
# rather than the deliberate safety change it is. The guard's own message
# names this as the research-use escape hatch. Recorded either way.
if JOB.get("allow_large_raster"):
    try:
        import pyorps.graph.api.graph_library_api as _gla
        if hasattr(_gla, "MAX_LIBRARY_BACKEND_CELLS"):
            REC["max_library_backend_cells"] = _gla.MAX_LIBRARY_BACKEND_CELLS
            _gla.MAX_LIBRARY_BACKEND_CELLS = None
            REC["large_raster_guard_overridden"] = True
        else:
            REC["large_raster_guard_overridden"] = False  # version predates it
    except Exception as exc:
        REC["large_raster_guard_error"] = repr(exc)

IN = JOB["inputs"]

# ------------------------------------------------------------------ rasterise
if JOB["kind"] == "rasterize":
    try:
        bbox = gpd.read_file(IN["bbox"])
        out_tif = JOB["raster_out"]
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)

        t0 = time.perf_counter()
        ds = pyorps.initialize_geo_dataset(IN["alkis"], bbox=bbox)
        ds.load_data()
        t1 = time.perf_counter()
        rasterizer = pyorps.GeoRasterizer(ds, JOB["cost_assumptions"], bbox)
        t2 = time.perf_counter()
        rasterizer.rasterize(save_path=out_tif)
        t3 = time.perf_counter()

        import rasterio
        with rasterio.open(out_tif) as src:
            arr = src.read(1)
        vals, counts = np.unique(arr, return_counts=True)

        dump("ok", vector_load_s=t1 - t0, rasterizer_init_s=t2 - t1,
             rasterize_s=t3 - t2, total_s=t3 - t0,
             raster_shape=list(arr.shape),
             raster_checksum=int(arr.astype(np.int64).sum()),
             raster_histogram={int(v): int(c) for v, c in zip(vals, counts)})

        if not JOB.get("keep_raster"):
            del arr
            try:
                os.remove(out_tif)
            except OSError:
                pass
    except Exception:
        dump("error", error=traceback.format_exc())
        sys.exit(1)
    sys.exit(0)

# -------------------------------------------------------------------- routing
try:
    source = gpd.read_file(IN["source"])
    targets = gpd.read_file(IN["targets"])
    bus_to_row = {int(b): i for i, b in enumerate(targets["bus"])}

    nb = JOB["neighborhood"]
    if nb == "r3":
        groups = [[bus_to_row[b] for b in cluster] for cluster in JOB["r3_clusters"]]
    else:
        groups = [[bus_to_row[b] for b in JOB["bus_order"]]]

    init_kw = {}
    if JOB.get("graph_api"):
        init_kw["graph_api"] = JOB["graph_api"]
    if JOB.get("use_gpu"):
        init_kw["use_gpu"] = True
    if JOB.get("search_space_buffer_m") is not None:
        init_kw["search_space_buffer_m"] = JOB["search_space_buffer_m"]

    route_kw = {}
    algo = JOB.get("algorithm") or "dijkstra"
    if algo != "dijkstra":
        route_kw["algorithm"] = algo

    paths_out, group_stats = [], []
    t_all0 = time.perf_counter()

    for gi, rows in enumerate(groups):
        ts = targets.iloc[rows]
        t0 = time.perf_counter()
        pf = pyorps.PathFinder(IN["ref_raster"], source_coords=source,
                               target_coords=ts, neighborhood_str=nb,
                               ignore_max_cost=JOB["ignore_max_cost"], **init_kw)
        t1 = time.perf_counter()
        res = pf.find_route(**route_kw)
        t2 = time.perf_counter()

        plist = list(res) if hasattr(res, "__iter__") else [res]
        if hasattr(res, "all") and not isinstance(res, list):
            try:
                plist = list(res.all)
            except Exception:
                pass

        for p, row in zip(plist, rows):
            geom = getattr(p, "path_geometry", None)
            coords = getattr(p, "path_coords", None)
            paths_out.append(dict(
                group=gi,
                target_row=int(row),
                bus=int(targets.iloc[row]["bus"]),
                graph_api=getattr(p, "graph_api", None),
                algorithm=getattr(p, "algorithm", None),
                total_cost=getattr(p, "total_cost", None),
                total_length=getattr(p, "total_length", None),
                geom_length_m=(geom.length if geom is not None else None),
                n_vertices=(len(coords) if coords is not None else None),
                euclidean_distance=getattr(p, "euclidean_distance", None),
                search_space_buffer_m=getattr(p, "search_space_buffer_m", None),
                runtimes=getattr(p, "runtimes", None),
                wkt=(geom.wkt if geom is not None else None),
            ))

        group_stats.append(dict(group=gi, n_targets=len(rows),
                                init_s=t1 - t0, find_route_s=t2 - t1,
                                n_paths=len(plist)))
        del pf, res

    t_all1 = time.perf_counter()
    dump("ok", routing_total_s=t_all1 - t_all0, groups=group_stats,
         n_paths=len(paths_out), paths=paths_out)
except Exception:
    dump("error", error=traceback.format_exc())
    sys.exit(1)
