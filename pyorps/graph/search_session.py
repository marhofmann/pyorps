"""Retained search fields: incremental route edits and rooted cost fields.

A :class:`SearchSession` keeps the settled SSSP tree of each leg after
the first route so later edits extract or resume instead of starting
Dijkstra from scratch. With ``reuse_reverse=True`` it additionally keeps
a tree rooted at each leg END, which makes moving the SOURCE as cheap as
moving the target.

A :class:`CostField` keeps ONE field rooted at a terminal that does not
move, and prices an unbounded number of moving endpoints against it in
O(1) each -- substation siting, service-area maps, k-candidate ranking.

``update(points)`` is exactly as optimal as chaining fresh
``PathFinder.find_route`` calls -- reuse is only a speedup, and with the
default ``reuse_reverse=False`` the returned cell sequence is
bit-identical too. Under ``reuse_reverse=True`` the COST is still
identical but a source-moved leg may come back as a different tied
optimum.

Default ``find_route`` stays one-shot. Retention is opt-in via
``PathFinder.search_session`` / ``PathFinder.cost_field``.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from pyorps.core.exceptions import NoPathFoundError
from pyorps.core.path import Path
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.io.geo_dataset import InMemoryRasterDataset, LocalRasterDataset
from pyorps.utils._raster_context import NO_EXCLUSION_VALUE

if TYPE_CHECKING:
    from pyorps.graph.path_finder import PathFinder

Coordinate = tuple[float, float]

_FULL_FIELD_MARGIN = 1e30
_FINITE_DIST = 1e20


def _delta_settled(dist: float, cutoff: float) -> bool:
    """Meyer–Sanders settle test: finite is not enough.

    A cell is settled only after its bucket has been processed, i.e.
    ``dist[v] < current_logical_bucket * delta``. After a full-field
    solve the cutoff is ``+inf`` and every finite label is final.
    """
    return dist < cutoff


def _as_point(point) -> Coordinate:
    return (float(point[0]), float(point[1]))


def _as_points(points: Sequence) -> list[Coordinate]:
    return [_as_point(p) for p in points]


def _uses_gradients(finder: PathFinder) -> bool:
    """True when the finder's search applies per-edge slope terms (a DEM
    with an objective); see ``PathFinder._uses_gradient_kernels``."""
    check = getattr(finder, "_uses_gradient_kernels", None)
    return bool(check()) if callable(check) else False


#: Precision the labels of each tree kind are ACCUMULATED in. Only the
#: float64 Dijkstra labels are the graph's least cost to float64 rounding;
#: the delta-stepping and GPU workspaces add in float32 (1.3e-5 relative
#: on 41.8 M cells) and, above one thread, may even settle a label a
#: little high. The eikonal field is a continuous value, not a graph
#: distance at all. A saved field is a certified bound only on "float64".
_LABEL_PRECISION = {"dijkstra": "float64", "delta": "float32",
                    "gpu": "float32", "fim": "float32"}


def _require_gradient_support(finder: PathFinder, kind: str) -> None:
    """Refuse a full-field kernel that would drop the DEM's slope terms."""
    if kind in ("delta", "gpu") and _uses_gradients(finder):
        raise NotImplementedError(
            f"the {kind} full-field kernel does not apply the DEM's "
            f"per-edge slope terms, so its field would price flat terrain; "
            f"use algorithm='dijkstra' (what 'auto' picks here on cython)")


def _backend_kind(finder: PathFinder, algorithm: str) -> str:
    api = (finder.graph_api_name or "cython").lower()
    algo = (algorithm or "dijkstra").lower()
    if api == "raster_fim" or algo in ("fim", "eikonal", "raster_fim"):
        return "fim"
    if api == "raster_gpu":
        return "gpu"
    if algo in ("delta-stepping", "delta-stepping-circular"):
        return "delta"
    return "dijkstra"


#: Fraction of the cores ``algorithm="auto"`` asks for. The delta-stepping
#: kernel synchronises every bucket on a spin barrier, so a worker that
#: cannot get a core does not merely fail to help -- it holds every other
#: worker spinning at the barrier. Requesting ALL of them is therefore not
#: the fastest setting but by far the slowest: measured on 41.8 M cells at
#: r2 on a 16-core machine, with under 8 % foreign load,
#:
#:     4 threads  7.03 s     12 threads  4.61 s  <- best
#:     8 threads  5.09 s     13 threads  4.69 s
#:                           14 threads  6.78 s
#:                           16 threads  917 s   <- 199x slower
#:
#: The collapse is at the core count, where nothing is left to run the OS
#: and the main thread. Three quarters lands on the measured optimum here,
#: never equals the core count on any machine, and leaves a quarter of the
#: box for whatever else is running. It is a DEFAULT, not a limit: pass
#: ``num_threads=`` to override it.
_AUTO_THREAD_FRACTION = 0.75


def _auto_threads() -> int:
    """Worker count for ``algorithm="auto"``; see :data:`_AUTO_THREAD_FRACTION`."""
    cores = os.cpu_count() or 2
    return max(1, int(cores * _AUTO_THREAD_FRACTION))


def _resolve_algorithm(finder: PathFinder, algorithm: str) -> str:
    """Map ``"auto"`` onto the fastest FULL-FIELD algorithm available.

    ``CostField`` settles the whole window for any query that is not a
    handful of points, and on that workload delta-stepping beats the
    serial Dijkstra by about 8x on the CPU and 15x on the GPU (41.8 M
    cells at r2 on an otherwise idle 16-core machine: 38.1 s -> 4.7 s ->
    2.6 s, agreeing to 1.3e-05 relative). Dijkstra keeps the default
    because it is the only backend that can stop early, which is what a
    two-or-three-candidate query wants.

    ``raster_gpu`` and ``raster_fim`` pick their kernel from the graph
    API rather than from this string (see :func:`_backend_kind`), so on
    those backends the choice is already made and ``"auto"`` only needs
    to name something the backend will accept.
    """
    if (algorithm or "").lower() != "auto":
        return algorithm
    api = (finder.graph_api_name or "cython").lower()
    if api in ("raster_gpu", "raster_fim"):
        return "delta-stepping"
    if api != "cython" or _uses_gradients(finder):
        # A library backend (networkx, networkit, ...) has no delta
        # kernel of ours to reach; the Dijkstra path is the only one
        # CostField knows how to root there. And with a DEM only the
        # Dijkstra solver applies the per-edge slope terms -- the
        # delta-stepping kernel would silently price flat terrain.
        return "dijkstra"
    try:
        import pyorps.utils._delta_stepping  # noqa: F401  # pylint: disable=unused-import
    except ImportError:
        return "dijkstra"
    return "delta-stepping"


def full_window_buffer_m(dataset_source: Any) -> float:
    """Buffer that makes the search window the WHOLE raster.

    ``search_space_buffer_m=None`` does not mean "no window": it runs a
    heuristic estimator sized for one source/target pair, which is the
    wrong shape for a :class:`CostField` whose candidates are scattered
    over the entire raster. Passing this value instead is the explicit
    way to say "settle everywhere"::

        finder = PathFinder(
            dataset_source=raster,
            source_coords=xy, target_coords=[xy],
            search_space_buffer_m=full_window_buffer_m(raster),
        )

    Parameters:
        dataset_source: Anything ``initialize_geo_dataset`` accepts, or
            an already-built raster dataset, or a path.

    Returns:
        The raster's diagonal in CRS units, which no buffer needs to
        exceed to cover every cell from any origin inside it.
    """
    bounds = getattr(dataset_source, "bounds", None)
    if bounds is None:
        data = getattr(dataset_source, "data", None)
        bounds = getattr(data, "bounds", None)
    if bounds is None:
        import rasterio
        with rasterio.open(str(dataset_source)) as src:
            bounds = src.bounds
    left, bottom, right, top = (bounds[0], bounds[1], bounds[2], bounds[3])
    return float(math.hypot(right - left, top - bottom))


#: Predecessor byte meaning "no parent": an unreached cell, or the root.
_NO_STEP = 255

#: Bump when the on-disk layout changes in a way an old reader misreads.
_FIELD_FORMAT_VERSION = 1


#: Cells per block in the predecessor encoder. The encoder needs several
#: index arrays per cell, so doing a 240 M-cell field in one pass built
#: about 20 GB of temporaries -- more than the machine, for a field whose
#: own labels are under 2 GB. Eight million keeps the working set in the
#: low hundreds of MB whatever the raster size.
_ENCODE_BLOCK = 1 << 23


def _encode_pred_steps(pred: np.ndarray, steps: np.ndarray,
                       cols: int) -> np.ndarray:
    """Absolute predecessor ids -> one byte per cell, a block at a time.

    A predecessor is always exactly one step away, so storing WHICH step
    costs a byte instead of the four an absolute cell id needs. On the
    240 M-cell windows the siting study uses that is 240 MB against
    960 MB, and the saved field is then smaller in memory than the live
    kernel workspace it came from.
    """
    if len(steps) > _NO_STEP:
        raise ValueError(
            f"{len(steps)} steps exceed the {_NO_STEP} a one-byte "
            f"predecessor can name; save with with_paths=False")
    out = np.full(pred.size, _NO_STEP, dtype=np.uint8)
    reach = int(np.abs(steps).max())
    span = 2 * reach + 1
    lut = np.full(span * span, _NO_STEP, dtype=np.uint8)
    for i, (dr, dc) in enumerate(steps):
        lut[(int(dr) + reach) * span + (int(dc) + reach)] = i

    cols = np.int64(cols)
    for start in range(0, pred.size, _ENCODE_BLOCK):
        block = pred[start:start + _ENCODE_BLOCK]
        reached = block >= 0
        if not reached.any():
            continue
        here = np.flatnonzero(reached).astype(np.int64)
        parent = block[here].astype(np.int64)
        row, col = np.divmod(here + np.int64(start), cols)
        prow, pcol = np.divmod(parent, cols)
        dr = row - prow
        dc = col - pcol
        if np.any(np.abs(dr) > reach) or np.any(np.abs(dc) > reach):
            raise ValueError(
                "a predecessor lies further than one step away; the field "
                "cannot be step-encoded")
        enc = lut[(dr + reach) * span + (dc + reach)]
        if np.any(enc == _NO_STEP):
            raise ValueError(
                "a predecessor offset is not in the step table; the field "
                "cannot be step-encoded")
        out[start:start + _ENCODE_BLOCK][reached] = enc
    return out


#: How :meth:`CostField.save` can store the labels of a raw field.
#:
#: ``"float32"``       round to nearest, 4 B/cell (the default);
#: ``"float32-down"``  round toward minus infinity, 4 B/cell: every stored
#:                     value is a lower bound, as plan section 3.7 wants
#:                     for pruning layers;
#: ``"float64"``       exact, 8 B/cell, for fields a certificate reads.
FIELD_STORAGE = ("float32", "float32-down", "float64")

#: A raw field written before storage modes existed stored
#: ``float32(float32(label) * float32(cell size))``: three round-to-nearest
#: steps, up to 2.0 ulp of the result (measured: label 2096785.9374999874,
#: cell 1.0001707673072695 stores 0.75 below the float64 product). Nudging
#: four ulps each way brackets the true value with margin.
_LEGACY_NUDGE_ULPS = 4
_LEGACY_ROUNDING_ULPS = 2.0


def _store_labels(values: np.ndarray, scale: float,
                  storage: str) -> np.ndarray:
    """Scale float64 labels by ``scale`` and store them as ``storage``.

    One float64 multiply, then one rounding to the storage type, block by
    block so the temporaries stay small. ``float32-down`` corrects every
    value that rounded up by one ulp toward minus infinity.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if storage == "float64":
        out = np.empty(flat.size, dtype=np.float64)
    else:
        out = np.empty(flat.size, dtype=np.float32)
    neg = np.float32(-np.inf)
    top = np.finfo(np.float32).max
    for start in range(0, flat.size, _ENCODE_BLOCK):
        x = flat[start:start + _ENCODE_BLOCK] * float(scale)
        if storage == "float64":
            out[start:start + _ENCODE_BLOCK] = x
            continue
        with np.errstate(over="ignore"):
            y = x.astype(np.float32)
        # A finite label beyond float32's range rounds to inf; the largest
        # float32 keeps it finite, and the reader's nudge brackets it.
        big = np.isinf(y) & np.isfinite(x)
        if big.any():
            y[big] = top
        if storage == "float32-down":
            over = y > x
            if over.any():
                y[over] = np.nextafter(y[over], neg)
        out[start:start + _ENCODE_BLOCK] = y
    return out


def _storage_meta(storage: str) -> dict[str, Any]:
    """The ``storage`` meta block written beside the labels."""
    if storage == "float64":
        return {"dtype": "float64", "rounding": "exact"}
    if storage == "float32-down":
        return {"dtype": "float32", "rounding": "down"}
    return {"dtype": "float32", "rounding": "nearest"}


def _nudge_ulps(storage: dict | None, dtype) -> tuple[int, int, float]:
    """``(down, up, rounding)``: the reader's outward nudge on each side and
    how far the stored value may sit from the label, all in ulps."""
    if storage is None:
        if np.dtype(dtype) == np.float32:           # pre-A1 raw field
            return (_LEGACY_NUDGE_ULPS, _LEGACY_NUDGE_ULPS,
                    _LEGACY_ROUNDING_ULPS)
        return 0, 0, 0.0
    rounding = storage.get("rounding")
    if rounding == "nearest":
        return 1, 1, 0.5
    if rounding == "down":
        return 0, 1, 1.0
    return 0, 0, 0.0


def _raw_error_bound(stored: np.ndarray, nudge_down: int,
                     rounding: float) -> float:
    """Largest understatement of a nudged-down stored field, in its units.

    ``(rounding + nudge_down)`` ulps of the largest finite stored value; 0
    for an exact float64 field. Relative to the kernel's own labels.
    """
    if stored.dtype == np.float64 or (nudge_down == 0 and rounding == 0.0):
        return 0.0
    top = -np.inf
    flat = stored.ravel()
    for start in range(0, flat.size, _ENCODE_BLOCK):
        block = flat[start:start + _ENCODE_BLOCK]
        fin = block[np.isfinite(block)]
        if fin.size:
            top = max(top, float(fin.max()))
    if not np.isfinite(top):
        return 0.0
    if top >= float(np.finfo(stored.dtype).max):
        return math.inf                     # a clamped overflow: unbounded
    ulp = float(np.spacing(np.asarray(top, dtype=stored.dtype)))
    return (float(rounding) + int(nudge_down)) * ulp


def _nudge(values: np.ndarray, ulps: int, toward: float) -> np.ndarray:
    """Move every finite value ``ulps`` ulps toward ``toward``, in its dtype.

    Costs are non-negative, so a downward nudge is clamped at zero.
    """
    out = np.array(values, copy=True)
    if ulps <= 0:
        return out
    fin = np.isfinite(out)
    if not fin.any():
        return out
    sub = out[fin]
    target = np.asarray(toward, dtype=out.dtype)
    for _ in range(int(ulps)):
        sub = np.nextafter(sub, target)
    if toward < 0:
        np.maximum(sub, 0, out=sub)
    out[fin] = sub
    return out


def _source_sha256(dataset) -> str | None:
    """SHA-256 of the raster a finder was built on: its file, or its array."""
    from pyorps.io.provenance import sha256_array, sha256_file

    if isinstance(dataset, LocalRasterDataset):
        src = dataset.file_source
        if isinstance(src, (str, os.PathLike)) and os.path.exists(src):
            return sha256_file(src)
        return None
    if isinstance(dataset, InMemoryRasterDataset):
        return sha256_array(np.asarray(dataset.data))
    return None


def _luts_hash(luts) -> str | None:
    """Parameter hash of a gradient LUT set (a dataclass or a mapping)."""
    if luts is None:
        return None
    import dataclasses

    from pyorps.io.provenance import parameter_hash

    if dataclasses.is_dataclass(luts):
        items = {f.name: getattr(luts, f.name)
                 for f in dataclasses.fields(luts)}
    elif isinstance(luts, dict):
        items = dict(luts)
    else:
        raise TypeError(f"cannot hash gradient LUTs of type "
                        f"{type(luts).__name__}")
    return parameter_hash({k: (np.asarray(v).tolist()
                               if isinstance(v, np.ndarray) else v)
                           for k, v in items.items()})


def _crs_string(crs) -> str | None:
    """One spelling per CRS, so ``"epsg:25832"`` and a rasterio CRS match."""
    if crs is None:
        return None
    try:
        from rasterio.crs import CRS
        return CRS.from_user_input(crs).to_string()
    except Exception:                               # noqa: BLE001  # pylint: disable=broad-exception-caught
        return str(crs)


#: The compiled module each tree kind runs; its file hash is the default
#: ``code`` section, so a field settled by an older build is refused
#: after the kernel is rebuilt.
_KERNEL_MODULES = {"dijkstra": "pyorps.utils._dijkstra",
                   "delta": "pyorps.utils._delta_stepping",
                   "gpu": "pyorps.utils.sssp_gpu",
                   "fim": "pyorps.utils.eikonal_gpu"}


def _code_section(kind: str, code: dict | None) -> dict:
    """The kernel module and its file hash, plus the caller's ``code``."""
    import importlib

    from pyorps.io.provenance import sha256_file

    name = _KERNEL_MODULES.get(kind)
    out: dict[str, Any] = {"kernel": name, "kernel_sha256": None}
    if name is not None:
        try:
            # nosemgrep - name comes from the fixed _KERNEL_MODULES table
            path = getattr(importlib.import_module(name), "__file__", None)
        except ImportError:
            path = None
        if path and os.path.exists(path):
            out["kernel_sha256"] = sha256_file(path)
    out.update(code or {})
    return out


def _algorithm_section(algorithm: str, kind: str, algo_kwargs: dict) -> dict:
    """The algorithm with its EFFECTIVE parameters, as ``_make_tree`` uses
    them: a default Delta or thread count is recorded, not left out."""
    algo: dict[str, Any] = {"name": str(algorithm), "kind": str(kind),
                            "label_precision": _LABEL_PRECISION.get(
                                kind, "float32")}
    if kind == "delta":
        algo["delta"] = algo_kwargs.get("delta", 100)
        threads = int(algo_kwargs.get("num_threads", 0) or 0)
        algo["num_threads"] = threads if threads > 0 else (os.cpu_count()
                                                           or 1)
    elif kind == "gpu":
        algo["delta"] = algo_kwargs.get("delta", "auto")
    return algo


def _field_key(finder: PathFinder, origin_idx: int, algorithm: str,
               kind: str, algo_kwargs: dict, *,
               source_sha256: str | None | object = ...) -> dict:
    """The sections of a field's record that its LABELS depend on.

    ``raster``, ``graph``, ``algorithm``, ``seeds`` and ``weights``, read
    from the finder as it is now. With a DEM the graph build is forced
    first, exactly as the Dijkstra tree forces it: that build writes the
    DEM's voids into the raster and makes the slope tables, so a writer
    and a reader hash the same window. ``source_sha256`` reuses an earlier
    hash of the source raster (it does not change under a live finder).
    """
    from pyorps.io import provenance as prov

    dem = luts = None
    if _uses_gradients(finder):
        api = finder.graph_api
        dem = getattr(api, "dem_data", None)
        luts = getattr(api, "gradient_luts", None)
    handler = finder.raster_handler
    win = handler.window
    objective = getattr(finder, "objective", None)
    crs = handler.raster_dataset.crs if handler.raster_dataset else None
    if source_sha256 is ...:
        source_sha256 = _source_sha256(finder.dataset)
    raster = {
        "source_sha256": source_sha256,
        "window": [int(win.row_off), int(win.col_off),
                   int(win.height), int(win.width)],
        "window_sha256": prov.sha256_array(_raster_of(finder)),
        "window_transform": [float(v) for v in
                             tuple(handler.window_transform)[:6]],
        "crs": _crs_string(crs),
        "cell_size_m": _cell_size(finder),
    }
    graph = {
        "steps": np.asarray(finder.steps)[:, :2].astype(np.int64).tolist(),
        "neighborhood": str(finder.neighborhood_str),
        "ignore_max_cost": bool(finder.ignore_max_cost),
        "max_value": int(_max_value(finder)),
        "graph_api": str(finder.graph_api_name),
        "dem_sha256": prov.sha256_array(dem) if dem is not None else None,
        "gradient_luts": _luts_hash(luts),
        "objective": (objective.fingerprint()
                      if hasattr(objective, "fingerprint") else None),
    }
    seeds = {"seed_hash": prov.seed_hash([origin_idx], [0.0]),
             "n_seeds": 1, "cells": [int(origin_idx)]}
    weights = {"length_rate": 0.0, "weight_mult": 1.0,
               "no_transit_hash": prov.cells_hash([])}
    return json.loads(prov.canonical_json({
        "raster": raster, "graph": graph,
        "algorithm": _algorithm_section(algorithm, kind, algo_kwargs),
        "seeds": seeds, "weights": weights}))


def _storage_section(storage: str, codec: str = "raw",
                     error_bound: float | None = None) -> dict:
    """The ``storage`` section for a raw storage mode or a lossy codec."""
    from pyorps.io.field_codec import FIELD_CODECS, is_lossy

    if codec not in FIELD_CODECS:
        raise ValueError(f"unknown codec {codec!r}; have {FIELD_CODECS}")
    if codec == "raw":
        if storage not in FIELD_STORAGE:
            raise ValueError(f"unknown storage {storage!r}; "
                             f"have {FIELD_STORAGE}")
        return _storage_meta(storage)
    if is_lossy(codec) and error_bound is None:
        raise ValueError(
            f"codec={codec!r} needs an explicit error_bound -- the "
            f"whole point of a fixed quantum is that the error is a "
            f"stated parameter rather than a measured surprise")
    return {"dtype": "codec", "codec": codec, "rounding": "down",
            "error_bound": float(error_bound)}


def _require_bound_storage(kind: str, storage: str, codec: str) -> None:
    """Refuse storage that promises a bound the kernel's labels lack."""
    if _LABEL_PRECISION.get(kind) == "float64":
        return
    if codec == "raw" and storage in ("float64", "float32-down"):
        raise ValueError(
            f"storage={storage!r} promises a bound on the least cost, but "
            f"a {kind} field's labels are float32-accumulated (and above "
            f"one thread may settle a little high); settle the field with "
            f"algorithm='dijkstra' for a certified bound, or save it with "
            f"storage='float32'")


def _field_provenance(finder: PathFinder, origin_idx: int, algorithm: str,
                      kind: str, algo_kwargs: dict, storage: str | dict, *,
                      cost_model=None, code=None, dw=None, extra=None,
                      key: dict | None = None) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """The plan-A1 provenance record of a field rooted at ``origin_idx``.

    ``storage`` is a :data:`FIELD_STORAGE` name, or an already built
    storage section (the lossy codecs). ``key`` is a :func:`_field_key`
    taken earlier -- when the field was settled; without it the key is
    read from the finder now.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps.io import provenance as prov

    if isinstance(storage, str):
        storage = _storage_meta(storage)
    if key is None:
        key = _field_key(finder, origin_idx, algorithm, kind, algo_kwargs)
    return prov.record(**key, storage=storage, cost_model=cost_model,
                       dw=dw, code=_code_section(kind, code), extra=extra)


def cost_field_provenance(finder: PathFinder, origin, *,
                          algorithm: str = "auto", storage: str = "float32",
                          codec: str = "raw",
                          error_bound: float | None = None,
                          cost_model=None, code=None, dw=None,
                          **algo_kwargs: Any) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """The provenance record a :class:`CostField` would be saved with.

    Build this from the reader's own inputs and pass it as
    ``CostField.open(path, expect=record)``: the reopened field is then
    refused unless it came from the same raster window, step table,
    algorithm, kernel build, seeds, weights, storage and cost model (plan
    Phase A1).

    Resolves ``algorithm="auto"`` and its default ``num_threads`` exactly
    as :class:`CostField` does, so the two records compare equal. With a
    DEM this builds the finder's graph, as settling the field would.
    ``codec``/``error_bound`` describe a lossy save, as in
    :meth:`CostField.save`.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if finder.raster_handler is None:
        finder.create_raster_handler()
    section = _storage_section(storage, codec, error_bound)
    requested = algorithm
    algorithm = _resolve_algorithm(finder, algorithm)
    kwargs = dict(algo_kwargs)
    if (requested or "").lower() == "auto" and algorithm.startswith(
            "delta-stepping"):
        kwargs.setdefault("num_threads", _auto_threads())
    kind = _backend_kind(finder, algorithm)
    _require_gradient_support(finder, kind)
    _require_bound_storage(kind, storage, codec)
    idx = _index(finder, _as_point(origin))
    return _field_provenance(finder, idx, algorithm, kind, kwargs, section,
                             cost_model=cost_model, code=code, dw=dw)


def fields_fit_in_memory(n_fields: int, cells: int, *,
                         algorithm: str = "auto",
                         headroom: float = 0.5) -> bool:
    """Can ``n_fields`` settled fields be held at once, or must they spill?

    Keeping fields live is always the better option -- an open
    :class:`CostField` needs no decode and answers everything a saved one
    does. :meth:`CostField.save` exists for the case where they simply do
    not fit. This is the test for that case::

        if fields_fit_in_memory(len(terminals), rows * cols):
            fields = [finder.cost_field(t, algorithm="auto")
                      for t in terminals]          # hold them all
        else:
            for t in terminals:                    # page them through
                with finder.cost_field(t, algorithm="auto") as f:
                    f.save(cache / f"field_{t}.npz")

    Parameters:
        n_fields: How many roots are wanted at the same time.
        cells: Cells in the search window, i.e. ``rows * cols``.
        algorithm: Which kernel will settle them; the delta and GPU
            workspaces pack a label and a predecessor into 8 B per cell,
            the Dijkstra solver keeps 13 B.
        headroom: Fraction of currently AVAILABLE memory the fields may
            occupy. The default leaves half for everything else,
            including the rasters and whatever else the machine is doing.

    Returns:
        True when they fit under that headroom. False -- or when
        available memory cannot be determined -- means spill.
    """
    per_cell = 13 if (algorithm or "").lower() == "dijkstra" else 8
    want = int(n_fields) * int(cells) * per_cell
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # no psutil, no guess
        return False
    return want <= available * float(headroom)


def _max_value(finder: PathFinder) -> int:
    return (IMPASSABLE_CELL_COST if finder.ignore_max_cost
            else NO_EXCLUSION_VALUE)


def _raster_of(finder: PathFinder) -> np.ndarray:
    data = finder.raster_handler.data
    if data.ndim == 3:
        data = data[0]
    return np.ascontiguousarray(data)


def _points_in_window(finder: PathFinder, points: Sequence[Coordinate]) -> bool:
    handler = finder.raster_handler
    if handler is None:
        return False
    indices = handler.coords_to_indices(list(points))
    rows, cols = handler.data.shape[-2:]
    for row, col in indices:
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return False
    return True


def _clone_finder(finder: PathFinder, points: Sequence[Coordinate]) -> PathFinder:
    """New PathFinder whose window covers ``points``. Drops retained trees."""
    from pyorps.graph.path_finder import PathFinder

    dataset = finder.dataset
    if isinstance(dataset, InMemoryRasterDataset):
        data = np.asarray(dataset.data)
        if data.ndim == 3:
            data = data[0]
        source: Any = data
        transform = dataset.transform
        crs = dataset.crs
    elif isinstance(dataset, LocalRasterDataset):
        source = dataset.file_source
        transform = None
        crs = dataset.crs
    else:
        data = _raster_of(finder)
        source = data
        transform = finder.raster_handler.window_transform
        crs = dataset.crs

    buffer_m = finder._explicit_buffer_m  # pylint: disable=protected-access  # same-package collaborator
    if buffer_m is None:
        buffer_m = finder.search_space_buffer_m
    return PathFinder(
        dataset_source=source,
        source_coords=list(points),
        target_coords=[points[-1]],
        search_space_buffer_m=buffer_m,
        neighborhood_str=finder.neighborhood_str,
        ignore_max_cost=finder.ignore_max_cost,
        graph_api=finder.graph_api_name,
        crs=crs,
        transform=transform,
        use_gpu=finder.use_gpu,
        objective=finder.objective,
        corridor_first=finder.corridor_first,
    )


def _index(finder: PathFinder, point: Coordinate) -> int:
    return int(finder.get_node_indices_from_coords(point))


def _indices(finder: PathFinder, points: Sequence[Coordinate]) -> np.ndarray:
    """Vectorised :func:`_index`: one snapping pass for the whole batch.

    ``get_node_indices_from_coords`` corrects max-cost endpoints in a
    single Numba call over all of them (``path_finder.py:1797``), so a
    256k-candidate query pays that scan once instead of 256k times.
    """
    raw = finder.get_node_indices_from_coords(list(points))
    return np.atleast_1d(np.asarray(raw)).astype(np.uint32, copy=False)


def _cell_size(finder: PathFinder) -> float:
    """CRS units per cell -- the factor between labels and ``Path.total_cost``.

    Kernels accumulate in CELL units (``cost_factor = sqrt(dr^2+dc^2)/
    (2+n_intermediates)``); reporting is metric. Same square-cell
    assumption as everywhere else in pyorps.
    """
    return float(abs(finder.raster_handler.window_transform.a))


def _steps_closed_under_negation(steps) -> bool:
    """True when ``-s`` is in the step set for every step ``s``.

    Duplicated from ``RasterGPUAPI`` rather than imported: that module
    reaches for cupy at import time and this one must stay CPU-only.
    """
    try:
        pairs = {(int(a), int(b))
                 for a, b in np.asarray(steps).reshape(-1, 2)}
    except (TypeError, ValueError):
        return False
    return bool(pairs) and all((-a, -b) in pairs for a, b in pairs)


def _reversible(finder: PathFinder) -> tuple[bool, str]:
    """Whether ``d(u -> v) == d(v -> u)`` holds for every pair here.

    Returns ``(ok, reason)``; ``reason`` is empty when ok. The
    preconditions, each verified against the kernels:

    * the step set is closed under negation -- ``directed=True``, which
      ``path_finder.py:381`` uses for cython/raster_gpu/cugraph. A
      hand-passed or ``directed=False`` half set makes the cython kernel
      a DAG and there is no symmetry to exploit;
    * the edge weight is symmetric in its endpoints,
      ``(raster[u] + sum(intermediates) + raster[v]) * cost_factor``,
      with a direction-free intermediate multiset
      (``_raster_context.pyx:33-96``) and
      ``cost_factor[s] == cost_factor[-s]``;
    * the gradient term bins ``fabs(dem[v] - dem[u])`` and applies the
      same LUT pair both ways (``_dijkstra.pyx:350-354``,
      ``raster_gpu_api.py:140-146``), so a DEM does NOT break this. A
      SIGNED slope response would, and the backend that lands it must
      set ``symmetric_edge_weights = False``;
    * an ``objective`` only rescales cell VALUES, so it is unaffected;
    * a FIM lazy grade limit routes through the per-pair masking loop
      (``raster_fim_api.py:1589``) which one field cannot reproduce.
    """
    is_fim = (finder.graph_api_name or "").lower() == "raster_fim"
    if not is_fim and not _steps_closed_under_negation(finder.steps):
        # FIM solves a continuous eikonal PDE on an 8-neighbor stencil
        # baked into the kernel, not the pyorps neighborhood step list
        # (path_finder.py:381 excludes raster_fim from directed=True for
        # exactly this reason -- ``finder.steps`` is not even consulted,
        # see raster_fim_api.py). The step-closure precondition is about
        # the discrete Dijkstra/delta kernels, so it does not apply here.
        return False, ("the step set is not closed under negation -- build "
                       "the neighborhood with directed=True")
    api = finder._graph_api          # never build one just to ask  # pylint: disable=protected-access
    if api is not None:
        if not getattr(api, "symmetric_edge_weights", True):
            return False, (f"{type(api).__name__} declares asymmetric edge "
                           f"weights")
        if (getattr(api, "_grade_limit", None) is not None
                and getattr(api, "_grade_limit_mode", None) == "lazy"):
            return False, ("a lazy grade limit is enforced per pair, not "
                           "per field")
    return True, ""


def _delta_dummy_target(raster: np.ndarray, root_idx: int,
                        max_value: int) -> int:
    """A target index that cannot truncate a full-field delta solve.

    ``delta_stepping_2d_persistent`` has two traps for a caller who only
    wants a field:

    * it returns BEFORE allocating labels when either endpoint is
      impassable (``_delta_stepping.pyx:1963-1965`` vs ``:1982``), so an
      arbitrary dummy on a blocked cell -- window corner (0, 0) is the
      usual one -- silently produces an empty workspace;
    * it stops one bucket after the target settles,
      ``cutoff = dist[target] * margin`` (``:2149-2153``), so a dummy
      whose label is exactly ``0.0`` caps the field at ``delta``. That
      rules out ``target == source``.

    A PASSABLE cell with a NON-ZERO cost satisfies both: every edge into
    it weighs at least ``raster[v] * cost_factor > 0``, so its label is
    positive and ``margin = 1e30`` puts the cutoff out of reach. When no
    such cell exists every passable cell costs 0, every reachable label
    is 0, and bucket 0 settles the whole component before the break --
    the field is complete either way.
    """
    flat = np.asarray(raster).reshape(-1)
    if 0 <= max_value <= 65535:
        candidates = (flat != max_value) & (flat > 0)
    else:                                   # exclusion disabled entirely
        candidates = flat > 0
    candidates = np.asarray(candidates).copy()
    if root_idx < candidates.size:
        candidates[root_idx] = False
    hit = int(np.argmax(candidates))
    if candidates.size and candidates[hit]:
        return hit
    return 0 if root_idx != 0 else (1 if flat.size > 1 else 0)


def _make_tree(finder: PathFinder, kind: str, root_idx: int,
               algo_kwargs: dict):
    """Build the backend tree adapter rooted at ``root_idx``."""
    if kind == "dijkstra":
        from pyorps.utils._dijkstra import make_dijkstra_solver
        api = finder.graph_api
        solver = make_dijkstra_solver(
            _raster_of(finder), finder.steps,
            max_value=_max_value(finder),
            dem=getattr(api, "dem_data", None),
            gradient_luts=getattr(api, "gradient_luts", None),
        )
        return _DijkstraTree(solver, root_idx)
    if kind == "delta":
        return _DeltaTree(
            _raster_of(finder), finder.steps, _max_value(finder), root_idx,
            delta=algo_kwargs.get("delta", 100),
            num_threads=algo_kwargs.get("num_threads", 0),
        )
    if kind == "gpu":
        return _GpuTree(
            _raster_of(finder), finder.steps, finder.ignore_max_cost,
            root_idx, delta=algo_kwargs.get("delta", "auto"),
        )
    return _FimTree(finder.graph_api, root_idx)


def _check_session_key(finder: PathFinder, neighborhood, ignore_max,
                       api_name, what: str) -> None:
    """Shared invalidation guard for the retained-field classes."""
    if (finder.neighborhood_str != neighborhood
            or finder.ignore_max_cost != ignore_max
            or finder.graph_api_name != api_name):
        raise ValueError(
            f"{what} invalidated: neighborhood, ignore-max, or "
            f"algorithm/backend changed -- open a new one")


#: costs_to(settle="auto") switches to one settle_all() at this many
#: points. Below it, per-point resume is strictly less work; above it
#: the union disk is the window anyway and 256k Python-level
#: search_until calls cost more than the expansions they save.
_SETTLE_ALL_POINTS = 1024


def _stitch(leg_indices: list[np.ndarray]) -> list[int]:
    out: list[int] = []
    for idxs in leg_indices:
        cells = [int(i) for i in idxs]
        if out and cells and out[-1] == cells[0]:
            cells = cells[1:]
        out.extend(cells)
    return out


class _DijkstraTree:
    def __init__(self, solver, root_idx: int):
        self.solver = solver
        self.root_idx = int(root_idx)
        solver.reset_root(self.root_idx)
        self._expansions = 0

    def is_settled(self, idx: int) -> bool:
        return bool(self.solver.is_settled(int(idx)))

    def extract_or_resume(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if self.solver.is_settled(idx):
            self._expansions = 0
            return np.asarray(self.solver.extract_path(idx), dtype=np.uint32)
        ok = self.solver.search_until(idx)
        self._expansions = int(self.solver.expansion_count)
        if not ok:
            return np.empty(0, dtype=np.uint32)
        return np.asarray(self.solver.extract_path(idx), dtype=np.uint32)

    def peek_dist(self, idx: int) -> float:
        return float(self.solver.peek_dist(int(idx)))

    def peek_dists(self, idxs) -> np.ndarray:
        return np.asarray(self.solver.peek_dists(idxs), dtype=np.float64)

    def settle(self, idx: int) -> bool:
        idx = int(idx)
        if self.solver.is_settled(idx):
            self._expansions = 0
            return True
        ok = bool(self.solver.search_until(idx))
        self._expansions = int(self.solver.expansion_count)
        return ok

    def settle_all(self) -> None:
        self.solver.settle_all()
        self._expansions = int(self.solver.expansion_count)

    def field(self, dtype=np.float64) -> np.ndarray:
        return np.asarray(self.solver.dist_array(), dtype=dtype)

    def pred(self) -> np.ndarray | None:
        """Predecessor per cell as int32, ``-1`` where unreached.

        int32, not int64: the solver already stores it that way, and on a
        240 M-cell field the upcast alone would cost 1.9 GB for nothing.
        """
        return np.asarray(self.solver.prev_array(), dtype=np.int32)

    @property
    def expansion_count(self) -> int:
        return self._expansions

    def memory_bytes(self) -> int:
        return int(self.solver.memory_bytes())

    def close(self) -> None:
        self.solver.release()


class _DeltaTree:
    def __init__(self, raster, steps, max_value, root_idx, delta, num_threads):
        from pyorps.utils._delta_stepping import (
            DeltaWorkspace,
            delta_stepping_2d_persistent,
        )
        self._ws = DeltaWorkspace(raster, max_value)
        self.root_idx = int(root_idx)
        dummy = _delta_dummy_target(raster, self.root_idx, max_value)
        delta_stepping_2d_persistent(
            raster, steps, np.uint64(self.root_idx), np.uint64(dummy),
            delta=float(delta), max_value=max_value,
            num_threads=int(num_threads), margin=_FULL_FIELD_MARGIN,
            workspace=self._ws)
        self._expansions = 1
        self._cutoff = float("inf")

    def _label(self, idx: int) -> float:
        """Settled label or ``inf``.

        ``INF_F32`` is 1e38, a large FINITE float32 (``_heap.pyx:22``),
        so a bare ``dist < cutoff`` test calls an unreachable cell
        settled -- threshold on ``_FINITE_DIST`` as well.
        """
        dist = float(self._ws.peek_dist(int(idx)))
        if dist >= _FINITE_DIST or not _delta_settled(dist, self._cutoff):
            return math.inf
        return dist

    def is_settled(self, idx: int) -> bool:
        return self._label(idx) < math.inf

    def extract_or_resume(self, idx: int) -> np.ndarray:
        self._expansions = 0
        idx = int(idx)
        if idx == self.root_idx:
            return np.array([self.root_idx], dtype=np.uint32)
        if not self.is_settled(idx):
            return np.empty(0, dtype=np.uint32)
        return np.asarray(
            self._ws.extract_path(self.root_idx, idx), dtype=np.uint32)

    def peek_dist(self, idx: int) -> float:
        return self._label(idx)

    def peek_dists(self, idxs) -> np.ndarray:
        # Full field by construction, so one whole-array unpack beats a
        # per-cell call once the batch is more than a handful.
        arr = np.asarray(idxs, dtype=np.int64).ravel()
        full = self.field()
        out = np.full(arr.size, math.inf, dtype=np.float64)
        ok = (arr >= 0) & (arr < full.size)
        if ok.any():
            out[ok] = full[arr[ok]]
        return out

    def settle(self, idx: int) -> bool:
        self._expansions = 0
        return self.is_settled(idx)

    def settle_all(self) -> None:
        self._expansions = 0          # full field by construction

    def field(self, dtype=np.float64) -> np.ndarray:
        # np.array(..., copy=True), NOT np.asarray: the workspace buffer is
        # float32 and live, so asking for float32 would hand back a VIEW and
        # the inf-masking below would corrupt the solver's own labels.
        out = np.array(self._ws.dist_array(), dtype=dtype, copy=True)
        out[out >= _FINITE_DIST] = math.inf
        return out

    def pred(self) -> np.ndarray | None:
        """Predecessor per cell as int32, ``-1`` where unreached.

        int32 rather than int64: a cell id fits, and the wider copy would
        cost 1.9 GB on a 240 M-cell field for no extra information.
        """
        raw = np.asarray(self._ws.pred_array())
        out = raw.astype(np.int32)
        out[raw == np.uint32(0xFFFFFFFF)] = -1
        return out

    @property
    def expansion_count(self) -> int:
        return self._expansions

    def memory_bytes(self) -> int:
        return int(self._ws.memory_bytes())

    def close(self) -> None:
        self._ws.release()


class _GpuTree:
    def __init__(self, raster, steps, ignore_max, root_idx, delta):
        from pyorps.utils.sssp_gpu import GpuSsspSession
        self._session = GpuSsspSession(
            raster, steps, ignore_max=ignore_max, delta=delta)
        self.root_idx = int(root_idx)
        self._session.solve(
            self.root_idx, target_indices=None, download=False,
            return_predecessor=True)
        self._expansions = 1

    def is_settled(self, idx: int) -> bool:
        dist = float(self._session._d_dist[int(idx)].get())  # pylint: disable=protected-access
        return _delta_settled(dist, float("inf")) and dist < 1e29

    def extract_or_resume(self, idx: int) -> np.ndarray:
        self._expansions = 0
        paths, _costs = self._session.extract_paths(
            self.root_idx, [int(idx)], repair=True)
        path = paths[0] if paths else None
        if path is None:
            return np.empty(0, dtype=np.uint32)
        return np.asarray(path, dtype=np.uint32)

    def peek_dist(self, idx: int) -> float:
        return float(self.peek_dists(np.array([idx]))[0])

    def peek_dists(self, idxs) -> np.ndarray:
        # One device gather, not n. CuPy fancy indexing does NOT bounds
        # check, so out-of-range entries are answered on the host -- same
        # guard as sssp_gpu.py:2855.
        import cupy as cp
        arr = np.asarray(idxs, dtype=np.int64).ravel()
        out = np.full(arr.size, math.inf, dtype=np.float64)
        ok = (arr >= 0) & (arr < self._session.n_pixels)
        if ok.any():
            got = self._session._d_dist[  # pylint: disable=protected-access  # same-package collaborator
                cp.asarray(np.ascontiguousarray(arr[ok]))].get()
            got = got.astype(np.float64)
            got[got >= 1e29] = math.inf
            out[ok] = got
        return out

    def settle(self, idx: int) -> bool:
        self._expansions = 0
        return self.is_settled(idx)

    def settle_all(self) -> None:
        self._expansions = 0

    def field(self, dtype=np.float64) -> np.ndarray:
        # .get() already returns a fresh host array, so asking for its own
        # float32 dtype costs no second copy and masking it is safe.
        out = self._session._d_dist.get().astype(dtype, copy=False)  # pylint: disable=protected-access,no-member
        out[out >= 1e29] = math.inf
        return out

    def pred(self) -> np.ndarray | None:
        """Predecessor per cell as int32, ``-1`` where unreached.

        The device array is downloaded once. ``solve`` already ran
        ``v5_repair_pred`` over the whole raster -- the async kernel can
        leave a raced link pointing at a stale parent, and that post-pass
        is what makes every entry, not just the ones a walk touches,
        agree with the final labels.
        """
        d_pred = getattr(self._session, "_d_pred", None)
        if d_pred is None:
            return None
        raw = np.asarray(d_pred.get(), dtype=np.int32)
        raw[raw < 0] = -1
        return raw

    @property
    def expansion_count(self) -> int:
        return self._expansions

    def memory_bytes(self) -> int:
        return int(self._session.device_bytes)

    def close(self) -> None:
        self._session.close()


class _FimTree:
    def __init__(self, api, root_idx: int):
        self._api = api
        self.root_idx = int(root_idx)
        self._field = api._solve(np.array([self.root_idx]), target_index=None)
        self._trace = (api._last_trace_host, api._last_trace_device)
        self._expansions = 1

    def is_settled(self, idx: int) -> bool:
        return self._api._field_cost(self._field, int(idx)) < _FINITE_DIST  # pylint: disable=protected-access

    def extract_or_resume(self, idx: int) -> np.ndarray:
        self._expansions = 0
        paths = self._api._paths_from_field(  # pylint: disable=protected-access  # same-package collaborator
            self._field, np.array([self.root_idx]),
            np.array([int(idx)], dtype=np.int64), reverse=True,
            trace=self._trace)
        path = paths[0] if paths else None
        if path is None:
            return np.empty(0, dtype=np.uint32)
        return np.asarray(path, dtype=np.uint32)

    def peek_dist(self, idx: int) -> float:
        return float(self._api._field_cost(self._field, int(idx)))  # pylint: disable=protected-access

    def peek_dists(self, idxs) -> np.ndarray:
        arr = np.asarray(idxs, dtype=np.int64).ravel()
        out = np.full(arr.size, math.inf, dtype=np.float64)
        if self._field is None:
            return out
        flat = np.asarray(self._field).ravel()
        ok = (arr >= 0) & (arr < flat.size)
        if ok.any():
            got = flat[arr[ok]].astype(np.float64)
            got[got >= 1e29] = math.inf      # FINITE_LIMIT, eikonal_gpu.py:97
            out[ok] = got
        return out

    def settle(self, idx: int) -> bool:
        self._expansions = 0
        return self.is_settled(idx)

    def settle_all(self) -> None:
        self._expansions = 0

    def field(self, dtype=np.float64) -> np.ndarray:
        if self._field is None:
            return np.empty(0, dtype=dtype)
        out = np.asarray(self._field, dtype=dtype).ravel().copy()
        out[out >= 1e29] = math.inf
        return out

    def pred(self) -> np.ndarray | None:
        """The eikonal field is traced by descent, not by predecessors."""
        return None

    @property
    def expansion_count(self) -> int:
        return self._expansions

    def memory_bytes(self) -> int:
        n = 0
        if self._field is not None:
            n += int(np.asarray(self._field).nbytes)
        host = self._trace[0]
        if host is not None:
            n += int(np.asarray(host).nbytes)
        return n

    def close(self) -> None:
        self._field = None
        self._trace = (None, None)


@dataclass
class _Leg:
    start: Coordinate
    end: Coordinate
    indices: list[int] = field(default_factory=list)


class SearchSession:
    """Retain per-leg search trees across source/target/waypoint edits.

    Invariants
    ----------
    * Same cost raster, neighborhood, ignore-max, and algorithm. Changing
      any of those invalidates the session (``update`` / ``route`` raise).
    * ``update(points)`` returns the same cell sequence as chaining fresh
      ``find_route`` on each consecutive pair.
    * Control points outside the current search window rebuild the
      windowed raster and drop the trees, then solve exactly on the new
      window.
    * One session is per route, not per raster.
    * With ``reuse_reverse=True`` a Dijkstra tree is 13 B/cell -- 0.95 GB
      on the 73 M-cell windows the substation case study uses -- and up
      to two trees can be resident per interior waypoint. ``max_trees``
      LRU-evicts down to a cap (default 4); anyone at that scale should
      pass ``max_trees=1`` or ``2``.
    """

    def __init__(self, finder: PathFinder, algorithm: str = "dijkstra", *,
                 reuse_reverse: bool = False, max_trees: int = 4,
                 **algo_kwargs: Any):
        if finder.raster_handler is None:
            finder.create_raster_handler()
        self._finder = finder
        self._algorithm = algorithm
        self._algo_kwargs = dict(algo_kwargs)
        self._kind = _backend_kind(finder, algorithm)
        self._neighborhood = finder.neighborhood_str
        self._ignore_max = finder.ignore_max_cost
        self._api_name = finder.graph_api_name
        self._points: list[Coordinate] = []
        self._legs: list[_Leg] = []
        self._forward: dict[Coordinate, Any] = {}
        #: Trees rooted at a leg END, keyed by end coordinate. Only
        #: populated with reuse_reverse=True; makes a SOURCE move as
        #: cheap as a target move, at the cost of a possibly different
        #: tied optimum on that leg.
        self._backward: dict[Coordinate, Any] = {}
        #: (kind, root) least-recently-used first; see _trim_trees.
        self._lru: list[tuple[str, Coordinate]] = []
        self._closed = False
        self.last_dirty_legs: tuple[tuple[Coordinate, Coordinate], ...] = ()
        self._last_expansions = 0
        self._last_leg_expansions = 0
        self._reuse_reverse = bool(reuse_reverse)
        self._max_trees = max(1, int(max_trees))
        ok, why = _reversible(finder)
        self._can_reverse = self._reuse_reverse and ok
        self._reverse_blocked = why if (self._reuse_reverse and not ok) else ""

    @property
    def finder(self) -> PathFinder:
        """The :class:`PathFinder` this session searches on."""
        return self._finder

    @property
    def has_route(self) -> bool:
        """True once :meth:`route` has run and :meth:`close` has not."""
        return bool(self._points) and not self._closed

    def uses_same_raster(self, finder: PathFinder) -> bool:
        """True when ``finder`` still sees the raster this session was built on."""
        mine, other = self._finder.dataset, finder.dataset
        if isinstance(mine, InMemoryRasterDataset) and isinstance(
                other, InMemoryRasterDataset):
            return mine.file_source is other.file_source
        if isinstance(mine, LocalRasterDataset) and isinstance(
                other, LocalRasterDataset):
            return mine.file_source == other.file_source
        return mine is other

    @property
    def expansion_count(self) -> int:
        """Nodes newly settled on the last ``route`` / ``update`` search."""
        return self._last_expansions

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the resident trees; 0 once closed."""
        if self._closed:
            return 0
        return sum(tree.memory_bytes()
                   for tree in (*self._forward.values(),
                                *self._backward.values()))

    def route(self, points: Sequence) -> Path:
        """First solve (or full rebuild). Retains per-leg trees."""
        self._check_open()
        self._check_key()
        pts = _as_points(points)
        if len(pts) < 2:
            raise ValueError("SearchSession.route needs at least two points")
        if not _points_in_window(self._finder, pts):
            self._rebuild_window(pts)
        self._drop_trees()
        self._points = pts
        self._legs = []
        dirty: list[tuple[Coordinate, Coordinate]] = []
        expansions = 0
        for start, end in zip(pts[:-1], pts[1:]):
            indices = self._solve_leg(start, end)
            self._legs.append(_Leg(start, end, indices))
            dirty.append((start, end))
            expansions += self._last_leg_expansions
        self.last_dirty_legs = tuple(dirty)
        self._last_expansions = expansions
        return self._path_from_legs(pts)

    def update(self, points: Sequence) -> Path:
        """Exact reroute; skips unchanged legs; resume/extract on dirty legs."""
        self._check_open()
        self._check_key()
        pts = _as_points(points)
        if len(pts) < 2:
            raise ValueError("SearchSession.update needs at least two points")
        if not self._points:
            return self.route(pts)
        if not _points_in_window(self._finder, pts):
            self._rebuild_window(pts)
            return self.route(pts)
        old = {(leg.start, leg.end): leg for leg in self._legs}
        new_legs: list[_Leg] = []
        dirty: list[tuple[Coordinate, Coordinate]] = []
        expansions = 0
        for start, end in zip(pts[:-1], pts[1:]):
            reused = old.get((start, end))
            if reused is not None:
                new_legs.append(reused)
                continue
            indices = self._resolve_dirty(start, end)
            new_legs.append(_Leg(start, end, indices))
            dirty.append((start, end))
            expansions += self._last_leg_expansions
        self._gc_trees(pts)
        self._points = pts
        self._legs = new_legs
        self.last_dirty_legs = tuple(dirty)
        self._last_expansions = expansions
        return self._path_from_legs(pts)

    def close(self) -> None:
        """Drop every retained tree. The session cannot be reused after."""
        self._drop_trees()
        self._legs = []
        self._points = []
        self._closed = True
        self._last_expansions = 0
        self.last_dirty_legs = ()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("SearchSession is closed")

    def _check_key(self) -> None:
        _check_session_key(self._finder, self._neighborhood, self._ignore_max,
                           self._api_name, "SearchSession")

    def _rebuild_window(self, points: Sequence[Coordinate]) -> None:
        self._drop_trees()
        self._finder = _clone_finder(self._finder, points)
        if self._finder.raster_handler is None:
            self._finder.create_raster_handler()

    def _drop_trees(self) -> None:
        for store in (self._forward, self._backward):
            for tree in store.values():
                tree.close()
            store.clear()
        self._lru.clear()

    def _gc_trees(self, points: Sequence[Coordinate]) -> None:
        starts = set(points[:-1])
        ends = set(points[1:])
        for root in list(self._forward):
            if root not in starts:
                self._evict("fwd", root)
        for root in list(self._backward):
            if root not in ends:
                self._evict("bwd", root)

    def _evict(self, kind: str, root: Coordinate) -> None:
        store = self._forward if kind == "fwd" else self._backward
        tree = store.pop(root, None)
        if tree is not None:
            tree.close()
        key = (kind, root)
        if key in self._lru:
            self._lru.remove(key)

    def _touch(self, kind: str, root: Coordinate) -> None:
        key = (kind, root)
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def _trim_trees(self) -> None:
        """Evict least-recently-used trees down to ``max_trees``.

        A Dijkstra tree is 13 B/cell -- 0.95 GB on the 73 M-cell windows
        the substation case study uses -- and reuse_reverse=True can hold
        two per interior waypoint. An unbounded cache is not a cache. The
        most recently touched entry is never evicted: it is the one the
        caller is about to read.
        """
        while len(self._lru) > self._max_trees:
            kind, root = self._lru[0]
            self._evict(kind, root)

    def _tree_for(self, cache: dict, kind: str, root: Coordinate,
                  root_idx: int):
        tree = cache.get(root)
        if tree is None:
            tree = _make_tree(self._finder, self._kind, root_idx,
                              self._algo_kwargs)
            cache[root] = tree
        self._touch(kind, root)
        self._trim_trees()
        return tree

    def _solve_leg(self, start: Coordinate, end: Coordinate) -> list[int]:
        start_idx = _index(self._finder, start)
        end_idx = _index(self._finder, end)
        tree = self._tree_for(self._forward, "fwd", start, start_idx)
        path = tree.extract_or_resume(end_idx)
        self._last_leg_expansions = int(tree.expansion_count)
        if path.size == 0:
            raise NoPathFoundError(start_idx, end_idx)
        return [int(i) for i in path]

    def _solve_leg_reverse(self, start: Coordinate,
                           end: Coordinate) -> list[int]:
        """The same leg, walked out of a tree rooted at ``end``.

        Sound only under :func:`_reversible`; checked once at
        construction. The walk is reversed into start -> end order.
        """
        start_idx = _index(self._finder, start)
        end_idx = _index(self._finder, end)
        tree = self._tree_for(self._backward, "bwd", end, end_idx)
        path = tree.extract_or_resume(start_idx)
        self._last_leg_expansions = int(tree.expansion_count)
        if path.size == 0:
            raise NoPathFoundError(start_idx, end_idx)
        return [int(i) for i in path[::-1]]

    def _resolve_dirty(self, start: Coordinate, end: Coordinate) -> list[int]:
        # A warm forward tree always wins: that walk is bit-identical to
        # a fresh find_route, which is the default contract. A tree
        # rooted at ``end`` is an equally optimal walk on an undirected
        # raster, but neighbour order makes the predecessor chain differ
        # whenever several shortest paths tie -- so it is only reached
        # with reuse_reverse=True, and only where it replaces a COLD
        # forward search rather than a warm one. That makes dragging the
        # source against a pinned end cost one search in total, not one
        # per drag.
        if start in self._forward or not self._can_reverse:
            return self._solve_leg(start, end)
        return self._solve_leg_reverse(start, end)

    def _path_from_legs(self, points: Sequence[Coordinate]) -> Path:
        indices = _stitch([np.asarray(leg.indices, dtype=np.uint32)
                           for leg in self._legs])
        if not indices:
            raise NoPathFoundError(
                _index(self._finder, points[0]),
                _index(self._finder, points[-1]))
        return self._finder._create_path_result(  # pylint: disable=protected-access  # same-package collaborator
            np.asarray(indices, dtype=np.uint32),
            points[0], points[-1], self._algorithm, False)


class CostField:
    """A reusable least-cost field rooted at ONE fixed terminal.

    :class:`SearchSession` keeps a field per moving LEG. This keeps one
    field for a terminal that does not move and prices an unbounded
    number of moving endpoints against it::

        with finder.cost_field(substation) as field:
            costs = field.costs_to(candidate_sites)     # (n,), inf where
            best = candidate_sites[int(np.argmin(costs))]  # unreachable
            route = field.path_to(best, calculate_metrics=True)

    The default ``algorithm="auto"`` picks the fastest FULL-FIELD kernel
    the backend offers -- delta-stepping on the CPU, the GPU kernel under
    ``graph_api="raster_gpu"``, measured 8.1x and 14.7x faster than
    Dijkstra on 41.8 M cells at r2. That is the right default because the
    whole point of a field is to settle once and look up many times::

        with finder.cost_field(pcc) as field:
            field.to_geotiff("reach_from_pcc.tif")

    Pass ``algorithm="dijkstra"`` for the opposite case: it alone can
    STOP EARLY, so pricing two or three nearby candidates never settles
    the rest of the window::

        with finder.cost_field(pcc, algorithm="dijkstra") as field:
            costs = field.costs_to(two_or_three_points)

    Direction does not matter. The raster graph is undirected with
    symmetric edge weights, so ``d(origin -> p) == d(p -> origin)`` and
    the field rooted at the FIXED terminal answers every query -- see
    :func:`_reversible` for the precondition list the constructor
    enforces. Among several tied optimal walks the one :meth:`path_to`
    returns may differ from ``find_route(p, origin)``; the cost never
    does.

    Units
    -----
    :meth:`costs_to` is in the units of :attr:`Path.total_cost` (cell
    value x metres): the cell-space label scaled by
    ``abs(window_transform.a)``. Equal to ``Path.total_cost`` to
    floating-point rounding on the cython backend, to float32 precision
    on the GPU. On ``raster_fim`` it is the CONTINUOUS eikonal value the
    search minimized, which the discrete recompute behind
    ``Path.total_cost`` prices UPWARD -- see
    ``PathFinder._total_cost_basis``.

    Memory
    ------
    A Dijkstra tree is 13 B/cell: 0.95 GB on a 73 M-cell window. Hold ONE
    at a time, bulk-gather, then :meth:`close`. Reopening later is a new
    ``finder.cost_field(origin)`` call.
    """

    def __init__(self, finder: PathFinder, origin, algorithm: str = "auto",
                 **algo_kwargs: Any):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if finder.raster_handler is None:
            finder.create_raster_handler()
        requested = algorithm
        algorithm = _resolve_algorithm(finder, algorithm)
        if (requested or "").lower() == "auto" and algorithm.startswith(
                "delta-stepping"):
            # Leave the machine some cores; an explicit value still wins.
            algo_kwargs.setdefault("num_threads", _auto_threads())
        origin = _as_point(origin)
        if not _points_in_window(finder, [origin]):
            raise ValueError(
                f"CostField origin {origin} is outside the search window; "
                f"widen search_space_buffer_m or move the origin")
        ok, why = _reversible(finder)
        if not ok:
            raise ValueError(
                f"CostField needs symmetric edge weights: {why}. The field "
                f"is rooted at the fixed terminal and read in the other "
                f"direction, which is only sound when d(u->v) == d(v->u).")
        self._finder = finder
        self._origin = origin
        self._algorithm = algorithm
        self._algo_kwargs = dict(algo_kwargs)
        self._kind = _backend_kind(finder, algorithm)
        _require_gradient_support(finder, self._kind)
        self._neighborhood = finder.neighborhood_str
        self._ignore_max = finder.ignore_max_cost
        self._api_name = finder.graph_api_name
        if self._kind in ("dijkstra", "delta"):
            raster = _raster_of(finder)
            if raster.dtype != np.uint16:
                raise ValueError(
                    f"CostField on the {self._kind} backend needs a uint16 "
                    f"weight raster, got {raster.dtype}")
        self._scale = _cell_size(finder)
        self._origin_idx = _index(finder, origin)
        if (finder.ignore_max_cost
                and _raster_of(finder).reshape(-1)[self._origin_idx]
                == IMPASSABLE_CELL_COST):
            raise ValueError(
                f"origin {origin} snapped to an impassable cell; a root the "
                f"exclude mask forbids relaxes outward but cannot be reached "
                f"back, which breaks the symmetry the field relies on")
        self._tree = _make_tree(finder, self._kind, self._origin_idx,
                                self._algo_kwargs)
        # What the labels were settled FROM, taken now: save() refuses to
        # stamp a record read off a finder that has changed since.
        self._key = _field_key(finder, self._origin_idx, algorithm,
                               self._kind, self._algo_kwargs)
        self._expansions = int(self._tree.expansion_count)
        self._settled_all = self._kind != "dijkstra"   # others are full-field
        self._field_cache: np.ndarray | None = None
        self._closed = False

    # ---------------------------------------------------------- properties

    @property
    def finder(self) -> PathFinder:
        """The :class:`PathFinder` this field was settled on."""
        return self._finder

    @property
    def origin(self) -> Coordinate:
        """The FIXED terminal the field is rooted at, as ``(x, y)``."""
        return self._origin

    @property
    def algorithm(self) -> str:
        """The algorithm actually in use, with ``"auto"`` resolved."""
        return self._algorithm

    @property
    def transform(self):
        """Affine transform of the field, in the raster's CRS.

        Georeferences :meth:`field_array` cell (row, col) to map (x, y).
        The field covers the SEARCH WINDOW, not the whole source raster,
        so this is the window's transform and not the file's.
        """
        self._check_open()
        return self._finder.raster_handler.window_transform

    @property
    def crs(self):
        """CRS of :attr:`transform`, i.e. the raster's own CRS."""
        self._check_open()
        return self._finder.raster_handler.raster_dataset.crs

    @property
    def expansion_count(self) -> int:
        """Cells settled since construction, cumulative.

        Bounded by one full search over the window no matter how many
        queries ran: a cell is popped once for the lifetime of the root.
        """
        return self._expansions

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the search tree and any cached field; 0 once closed."""
        if self._closed:
            return 0
        n = int(self._tree.memory_bytes())
        if self._field_cache is not None:
            n += int(self._field_cache.nbytes)
        return n

    # ------------------------------------------------------------- queries

    def is_settled(self, point) -> bool:
        """True when :meth:`costs_to` would return a final answer here."""
        self._check_open()
        return bool(self._tree.is_settled(
            _index(self._finder, _as_point(point))))

    def costs_to(self, points: Sequence, *, settle: str = "auto") -> np.ndarray:
        """Least-cost distance from :attr:`origin` to each point.

        Parameters:
            points: Sequence of (x, y) in the raster CRS.
            settle: ``"auto"`` (default) resumes the search per point
                below ``_SETTLE_ALL_POINTS`` and runs one
                :meth:`settle_all` at or above it; ``"each"`` and
                ``"all"`` force those; ``"none"`` reads the field as it
                stands, so an unsettled cell reads ``inf``. Only the
                cython Dijkstra backend has anything to settle -- the
                others are full-field on construction.

        Returns:
            float64 array, same length as ``points``, in the units of
            ``Path.total_cost``. ``inf`` where no route exists.

        Raises:
            ValueError: a point falls outside the search window. That is
                not the same answer as "unreachable", so it is not
                silently reported as ``inf``.

        Notes:
            Proving a point UNREACHABLE drains the heap, so that one
            query costs a full solve -- and every later one is then free.
        """
        self._check_open()
        self._check_key()
        pts = _as_points(points)
        if not pts:
            return np.empty(0, dtype=np.float64)
        if not _points_in_window(self._finder, pts):
            raise ValueError(
                "some points fall outside the CostField search window; "
                "widen search_space_buffer_m so it covers every candidate")
        idxs = _indices(self._finder, pts)
        mode = settle
        if mode == "auto":
            mode = "all" if idxs.size >= _SETTLE_ALL_POINTS else "each"
        if mode == "all":
            self.settle_all()
        elif mode == "each":
            self._settle_each(idxs)
        elif mode != "none":
            raise ValueError(f"unknown settle mode {settle!r}")
        return self._gather(idxs) * self._scale

    def cost_to(self, point) -> float:
        """``costs_to`` for one point."""
        return float(self.costs_to([point])[0])

    def settle_all(self) -> None:
        """Force the complete field. Idempotent."""
        self._check_open()
        if self._settled_all:
            return
        self._tree.settle_all()
        self._expansions += int(self._tree.expansion_count)
        self._settled_all = True
        self._field_cache = None

    def path_to(self, point, *, calculate_metrics: bool = False,
                reverse: bool = False) -> Path:
        """The route between :attr:`origin` and ``point`` as a ``Path``.

        Parameters:
            point: The moving endpoint.
            calculate_metrics: Fill ``total_length`` / ``total_cost`` /
                ``length_by_category``. Off by default: the cost is
                already available from :meth:`costs_to` for free, and
                the metric kernel is O(path) per call.
            reverse: Return the walk as ``point -> origin`` instead of
                ``origin -> point``. Equally optimal (the graph is
                undirected); only the orientation differs.

        Raises:
            NoPathFoundError: no route exists.

        Notes:
            Registers the Path on ``finder.paths`` like every other
            route, so do NOT call this in a ranking loop over every
            candidate -- rank with ``costs_to``, extract the shortlist.
        """
        self._check_open()
        self._check_key()
        pt = _as_point(point)
        if not _points_in_window(self._finder, [pt]):
            raise ValueError(f"point {pt} is outside the CostField window")
        idx = _index(self._finder, pt)
        self._settle_each(np.array([idx], dtype=np.uint32))
        cells = self._tree.extract_or_resume(idx)
        if cells.size == 0:
            raise NoPathFoundError(self._origin_idx, idx)
        if reverse:
            cells = cells[::-1]
            source, target = pt, self._origin
        else:
            source, target = self._origin, pt
        return self._finder._create_path_result(  # pylint: disable=protected-access  # same-package collaborator
            np.ascontiguousarray(cells, dtype=np.uint32),
            source, target, self._algorithm, calculate_metrics)

    def field_array(self, dtype=np.float32) -> np.ndarray:
        """The whole field as a window-shaped 2-D array, for map output.

        Settles first -- a partial Dijkstra disk written to a GeoTIFF
        would be a map of "how far the search happened to have got",
        which is not a quantity. ``inf`` marks unreachable cells; write
        it out with a nodata value rather than as-is.

        Values are in the units of :meth:`costs_to`. float32 by default:
        a 73 M-cell window is 292 MB at float32 and 584 MB at float64,
        and no downstream raster consumer needs the extra digits.
        """
        self._check_open()
        self.settle_all()
        rows, cols = self._finder.raster_handler.data.shape[-2:]
        # Take the field straight from the backend in the OUTPUT dtype when
        # the float64 gather cache is not already built, and scale in place.
        # The obvious `(cache * scale).astype(dtype)` costs two extra
        # whole-field temporaries -- on the 240 M-cell windows the siting
        # study uses that is 1.9 GB + 1.0 GB on top of a 1.9 GB cache, which
        # is what made a field "about 4 GB" and ruled out the full raster.
        flat = (self._field_cache if self._field_cache is not None
                else self._tree.field(dtype=dtype))
        if flat.size != rows * cols:
            raise RuntimeError(
                f"backend field has {flat.size} cells, window has "
                f"{rows * cols}")
        out = np.empty(flat.size, dtype=dtype)
        np.multiply(flat, self._scale, out=out, casting="unsafe")
        return out.reshape(rows, cols)

    def to_geotiff(self, path, *, dtype=np.float32, nodata: float = -1.0,
                   compress: str = "deflate"):
        """Write the settled field to a georeferenced GeoTIFF.

        Settles first, exactly as :meth:`field_array` does.

        Parameters:
            path: Destination file. Parent directories are created.
            dtype: Band dtype. float32 by default -- a 73 M-cell window
                is 292 MB at float32 and 584 MB at float64, and no map
                consumer needs the extra digits.
            nodata: Value written where the field is ``inf``, i.e. where
                no route exists. Written into the band description too,
                so QGIS and rasterio both mask it without being told.
                Must not collide with a reachable cost, hence the
                negative default -- costs are non-negative.
            compress: rasterio creation option; ``None`` to disable.

        Returns:
            The path written, as a ``pathlib.Path``.
        """
        from pathlib import Path as _Path

        import rasterio

        self._check_open()
        # field_array hands back an array nobody else holds, so the nodata
        # fill goes in place. Boolean-indexing it and np.where-ing it each
        # allocate another whole field.  A FINITE nodata can only equal a
        # finite cell, so the collision test needs no isfinite mask of its own.
        arr = self.field_array(dtype=dtype)
        if np.isfinite(nodata) and bool(np.any(arr == nodata)):
            raise ValueError(
                f"nodata={nodata} collides with a reachable cost in this "
                f"field; pick a value no cell can take")
        np.copyto(arr, nodata, where=~np.isfinite(arr))
        out = arr
        dest = _Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff", "height": out.shape[0], "width": out.shape[1],
            "count": 1, "dtype": out.dtype.name, "crs": self.crs,
            "transform": self.transform, "nodata": nodata,
        }
        if compress:
            profile["compress"] = compress
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(out, 1)
            dst.set_band_description(
                1, f"least-cost distance from {self._origin}")
        return dest

    def provenance(self, *, storage: str = "float32", codec: str = "raw",
                   error_bound: float | None = None, cost_model=None,
                   code=None, dw=None, extra=None) -> dict:
        """The plan-A1 provenance record of this field.

        The same record :meth:`save` writes with these arguments; see
        :func:`cost_field_provenance` for building the one a reader
        expects without settling a field. The input sections are the ones
        taken when the field was settled; raises if the finder has changed
        since.
        """
        self._check_open()
        section = _storage_section(storage, codec, error_bound)
        _require_bound_storage(self._kind, storage, codec)
        return _field_provenance(
            self._finder, self._origin_idx, self._algorithm, self._kind,
            self._algo_kwargs, section, cost_model=cost_model, code=code,
            dw=dw, extra=extra, key=self._settled_key())

    def save(self, path, *, with_paths: bool = True,
             compress: bool = False, codec: str = "raw",
             error_bound: float | None = None, storage: str = "float32",
             provenance: dict | None = None,
             extra_meta: dict | None = None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Write the settled field so it can be reopened without re-searching.

        Prefer keeping fields in memory -- an open :class:`CostField`
        answers everything this does and needs no decode. Reach for this
        when the fields do not all FIT: ten 240 M-cell fields are about
        19 GB live, and saving lets a run hold one at a time and page the
        rest off disk. :func:`fields_fit_in_memory` is the test for that.

        Measured on 41.8 M cells, against a 4.67 s settle of the same
        field:

            compress=False   save 2.59 s   open 0.19 s   5.00 B/cell
            compress=True    save 12.1 s   open 0.94 s   3.07 B/cell

        Uncompressed is the default because it is the only one that is
        cheaper than simply searching again: writing it costs about half
        a settle and reading it back a twenty-fifth, where deflating
        costs more than two settles to save 39% of the disk.

        Parameters:
            path: Destination ``.npz``. Parent directories are created.
            with_paths: Also store the predecessor, so the reopened field
                can return routes and not only costs. Adds one byte per
                cell. Works on ``dijkstra``, ``delta-stepping`` and the
                GPU kernel; the eikonal backend traces by descent rather
                than by predecessors, so there it raises and
                ``with_paths=False`` saves costs alone.
            compress: Deflate the arrays -- smaller, but see above.
            codec: ``"raw"`` (the default, and the only exact one) or
                ``"fixed-quantum"``, which floors every label onto a
                quantum of ``error_bound`` EUR and stores the residuals.
                A lossy codec turns the reopened field into an INTERVAL:
                :meth:`SavedCostField.costs_to` then returns the lower
                end unless asked for the upper, and the choice belongs
                to whichever inequality the caller is about to write.
                See :mod:`pyorps.io.field_codec`.
            error_bound: Guaranteed one-sided understatement in EUR,
                required by ``codec="fixed-quantum"``.
            storage: How a ``"raw"`` field stores its labels, one of
                :data:`FIELD_STORAGE`. ``"float32"`` (the default) rounds
                to nearest, so a stored value may sit half an ulp either
                side of the label; :meth:`SavedCostField.costs_to` nudges
                it outward so ``bound="lower"`` and ``"upper"`` stay sound.
                ``"float32-down"`` rounds toward minus infinity, so the
                stored value itself is a lower bound. ``"float64"`` is
                exact at twice the size: use it for any field a
                certificate reads (plan section 3.7).
            provenance: Extra sections for the plan-A1 provenance record
                (``cost_model``, ``code``, ``dw``, ``extra``); merged into
                the record :meth:`provenance` builds from the finder.
            extra_meta: Free-form JSON-able metadata stored under
                ``meta["extra_meta"]``. Never compared by a reader.

        Returns:
            The path written, as a ``pathlib.Path``.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from pathlib import Path as _Path

        from pyorps.io import provenance as _prov
        from pyorps.io.field_codec import encode_field

        # Argument validation first: a bad codec name is the caller's
        # mistake whether or not this field happens to still be open, and
        # none of it should wait for a settle.
        storage_section = _storage_section(storage, codec, error_bound)
        sections = dict(provenance or {})
        unknown = set(sections) - {"cost_model", "code", "dw", "extra"}
        if unknown:
            raise ValueError(
                f"provenance may add only cost_model, code, dw and extra "
                f"sections; the rest is taken from the finder, got "
                f"{sorted(unknown)}")
        self._check_open()
        _require_bound_storage(self._kind, storage, codec)
        record = _field_provenance(
            self._finder, self._origin_idx, self._algorithm, self._kind,
            self._algo_kwargs, storage_section,
            cost_model=sections.get("cost_model"), code=sections.get("code"),
            dw=sections.get("dw"), extra=sections.get("extra"),
            key=self._settled_key())
        self.settle_all()
        rows, cols = self._finder.raster_handler.data.shape[-2:]
        steps = np.asarray(self._finder.steps)[:, :2].astype(np.int64)

        payload: dict[str, Any] = {}
        codec_meta: dict[str, Any] = {"codec": "raw"}
        if codec == "raw":
            # Stored already scaled, i.e. in the units costs_to returns,
            # so a reader needs no cell size to price a candidate. One
            # float64 multiply, then ONE rounding to the storage type.
            payload["dist"] = _store_labels(
                self._tree.field(dtype=np.float64), self._scale, storage)
            codec_meta["storage"] = _storage_meta(storage)
            down, _up, rounding = _nudge_ulps(codec_meta["storage"],
                                              payload["dist"].dtype)
            # Written, not only computed on reopen: a tool reading the
            # meta directly must not see 0 on a float32 field.
            codec_meta["error_bound"] = _raw_error_bound(
                payload["dist"], down, rounding)
            codec_meta["label_precision"] = _LABEL_PRECISION.get(
                self._kind, "float32")
        else:
            arrays, codec_meta = encode_field(
                self.field_array(dtype=np.float64), codec=codec,
                error_bound=float(error_bound))
            codec_meta = dict(codec_meta, label_precision=_LABEL_PRECISION
                              .get(self._kind, "float32"))
            payload.update(arrays)
        if with_paths:
            pred = self._tree.pred()
            if pred is None:
                raise NotImplementedError(
                    f"the {self._kind} backend keeps no predecessor array, "
                    f"so a saved field cannot return routes; save with "
                    f"with_paths=False to store costs only")
            payload["pred_step"] = _encode_pred_steps(pred, steps, cols)

        meta = {
            "format": _FIELD_FORMAT_VERSION,
            "rows": int(rows), "cols": int(cols),
            "transform": [float(v) for v in tuple(self.transform)[:6]],
            "crs": str(self.crs) if self.crs is not None else None,
            "cell_size_m": float(self._scale),
            "origin_xy": [float(self._origin[0]), float(self._origin[1])],
            "origin_cell": int(self._origin_idx),
            "algorithm": str(self._algorithm),
            "neighborhood": str(self._neighborhood),
            "ignore_max_cost": bool(self._ignore_max),
            "graph_api": str(self._api_name),
            "steps": steps.tolist(),
            "units": "cost as Path.total_cost (cell value x metres)",
            "has_paths": bool(with_paths),
        }
        # Dispatch on the codec key, NOT on _FIELD_FORMAT_VERSION: that
        # one is a strict-equality check, so bumping it to introduce a
        # codec would reject every field already written to disk.
        meta.update({k: v for k, v in codec_meta.items()
                     if k not in ("rows", "cols")})
        meta["provenance"] = record
        if extra_meta:
            meta["extra_meta"] = json.loads(_prov.canonical_json(extra_meta))
        payload["meta"] = np.array(json.dumps(meta))

        dest = _Path(path)
        if dest.suffix != ".npz":
            # numpy appends .npz when the name lacks it; decide the final
            # name here so the atomic rename below lands where the caller
            # will look for it.
            dest = dest.with_suffix(".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Write beside the target and rename. A field takes tens of seconds
        # to write, and a run killed inside that window used to leave a
        # partial .npz under the real name -- which the NEXT run opened and
        # died on with "File is not a zip file", having trusted a cache that
        # was never finished. os.replace is atomic, so an interrupted save
        # leaves a stray temp file and the previous good cache intact.
        # The temp name must itself end in .npz, or numpy appends one and
        # the rename below chases a path that does not exist.
        tmp = dest.with_name(f"{dest.stem}.partial-{os.getpid()}.npz")
        writer = np.savez_compressed if compress else np.savez
        try:
            writer(tmp, **payload)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return dest

    @staticmethod
    def open(path, *, expect: dict | None = None,
             ignore: Sequence[str] = ("extra",)) -> SavedCostField:
        """Reopen a field written by :meth:`save`. See :class:`SavedCostField`.

        ``expect`` is the provenance record the caller requires, usually
        from :func:`cost_field_provenance`; a field whose stored record
        differs in any key -- or that has none -- raises
        :class:`pyorps.io.provenance.ProvenanceMismatch`.
        """
        return SavedCostField(path, expect=expect, ignore=ignore)

    # ------------------------------------------------------------ lifetime

    def close(self) -> None:
        """Release the field. Reopen with a new ``finder.cost_field``."""
        if self._closed:
            return
        self._tree.close()
        self._field_cache = None
        self._closed = True
        sessions = getattr(self._finder, "_search_sessions", None)
        if sessions is not None and self in sessions:
            sessions.remove(self)

    def __enter__(self) -> CostField:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------ internals

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("CostField is closed")

    def _check_key(self) -> None:
        _check_session_key(self._finder, self._neighborhood, self._ignore_max,
                           self._api_name, "CostField")

    def _settled_key(self) -> dict:
        """The key taken when the field was settled, after checking that
        the finder still matches it (window data, objective, DEM, ...)."""
        from pyorps.io.provenance import diff

        self._check_key()
        now = _field_key(self._finder, self._origin_idx, self._algorithm,
                         self._kind, self._algo_kwargs,
                         source_sha256=self._key["raster"]["source_sha256"])
        changed = diff(self._key, now, ignore=())
        if changed:
            raise ValueError(
                "the finder changed since this field was settled, so its "
                "labels no longer describe it (" + "; ".join(changed[:4])
                + (" ..." if len(changed) > 4 else "")
                + ") -- settle a new field")
        return self._key

    def _settle_each(self, idxs: np.ndarray) -> None:
        if self._settled_all:
            return
        for idx in idxs.tolist():
            self._tree.settle(idx)
            self._expansions += int(self._tree.expansion_count)
        self._field_cache = None

    def _cached_field(self) -> np.ndarray:
        if self._field_cache is None:
            self._field_cache = self._tree.field()
        return self._field_cache

    def _gather(self, idxs: np.ndarray) -> np.ndarray:
        """Label lookup, one whole-field pass at most.

        Above the threshold the field array is materialised once and
        reused, so repeated 256k-candidate queries do not re-unpack a
        73 M-cell array each time.
        """
        if self._settled_all and idxs.size >= _SETTLE_ALL_POINTS:
            full = self._cached_field()
            out = np.full(idxs.size, math.inf, dtype=np.float64)
            arr = idxs.astype(np.int64, copy=False)
            ok = arr < full.size
            if ok.any():
                out[ok] = full[arr[ok]]
            return out
        return self._tree.peek_dists(idxs)


class SavedCostField:
    """A settled field read back from disk, without a solver behind it.

    Opened by :meth:`CostField.open`. It answers the two questions a
    siting run asks of a field -- what does it cost to reach this
    candidate, and by which route -- from arrays alone, so ten fields can
    be paged through one at a time instead of held live::

        with CostField.open("field_PCC0.npz") as field:
            costs = field.costs_to(candidate_sites)   # (n,), inf = no route
            cells = field.path_cells(best)            # route as cell ids
            xy = field.path_coords(best)              # ... and as coordinates

    This is the FALLBACK. A live :class:`CostField` answers the same
    questions plus everything else, so open one of those unless the set
    does not fit in memory.

    What it cannot do: a live field nudges a coordinate that lands on an
    impassable cell onto a passable neighbour, because it still has the
    raster. This does not have the raster, so such a point reads as
    unreachable rather than being quietly moved.
    """

    def __init__(self, path, *, expect: dict | None = None,
                 ignore: Sequence[str] = ("extra",)):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from pathlib import Path as _Path

        self._path = _Path(path)
        try:
            handle = np.load(self._path, allow_pickle=False)
        except Exception as exc:                         # noqa: BLE001
            raise ValueError(
                f"{self._path} is not a readable saved field ({exc}). A "
                f"field written by a run that was interrupted mid-write is "
                f"the usual cause; delete it and let it be settled again."
            ) from exc
        from pyorps.io.field_codec import codec_of, decode_field

        with handle as data:
            meta = json.loads(str(data["meta"]))
            if int(meta.get("format", 0)) != _FIELD_FORMAT_VERSION:
                raise ValueError(
                    f"{self._path.name} is format {meta.get('format')}, "
                    f"this build reads {_FIELD_FORMAT_VERSION}")
            if expect is not None:
                # Before the arrays: a refused field costs no read.
                from pyorps.io.provenance import require_match
                require_match(expect, meta.get("provenance"),
                              what=f"saved field {self._path.name}",
                              ignore=ignore)
            codec = codec_of(meta)
            if codec == "raw":
                self._dist = np.asarray(data["dist"])
                self._decoded = None
            else:
                self._decoded = decode_field(
                    {k: np.asarray(data[k]) for k in data.files
                     if k not in ("meta", "pred_step")}, meta)
                self._dist = self._decoded.lower.ravel()
            self._pred_step = (np.asarray(data["pred_step"])
                               if "pred_step" in data.files else None)
        self._meta = meta
        self._codec = codec
        self._rows = int(meta["rows"])
        self._cols = int(meta["cols"])
        self._steps = np.asarray(meta["steps"], dtype=np.int64)
        self._origin_idx = int(meta["origin_cell"])
        self._cell_size = float(meta["cell_size_m"])
        expected = self._rows * self._cols
        if self._dist.size != expected:
            raise ValueError(
                f"{self._path.name} holds {self._dist.size} labels for a "
                f"{self._rows}x{self._cols} window")
        # How far a stored raw label may sit from the kernel's label, in
        # ulps of its own dtype, on the low and the high side. A field
        # written before storage modes existed carries no "storage" block.
        self._nudge_down, self._nudge_up, self._rounding = 0, 0, 0.0
        if self._decoded is None:
            (self._nudge_down, self._nudge_up,
             self._rounding) = _nudge_ulps(meta.get("storage"),
                                           self._dist.dtype)
        self._error_bound: float | None = None

    # ---------------------------------------------------------- properties

    @property
    def origin(self) -> Coordinate:
        """The FIXED terminal the saved field was rooted at, as ``(x, y)``."""
        x, y = self._meta["origin_xy"]
        return (float(x), float(y))

    @property
    def algorithm(self) -> str:
        """The algorithm that settled this field, as recorded at save time."""
        return str(self._meta["algorithm"])

    @property
    def has_paths(self) -> bool:
        """True when :meth:`path_cells` works, i.e. saved with predecessors."""
        return self._pred_step is not None

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the window the field covers."""
        return (self._rows, self._cols)

    @property
    def transform(self):
        """Affine transform of the field, in the raster's CRS."""
        from affine import Affine
        return Affine(*self._meta["transform"])

    @property
    def crs(self):
        """CRS of :attr:`transform`; ``None`` when the raster had none."""
        from rasterio.crs import CRS
        raw = self._meta.get("crs")
        return CRS.from_string(raw) if raw else None

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the loaded distance and predecessor arrays."""
        n = int(self._dist.nbytes)
        if self._pred_step is not None:
            n += int(self._pred_step.nbytes)
        return n

    # ------------------------------------------------------------- queries

    def _index(self, point) -> int:
        """Cell holding ``point``, by the same rule the live field uses."""
        from rasterio.transform import rowcol
        x, y = _as_point(point)
        row, col = rowcol(self.transform, x, y)
        row = int(np.asarray(row).ravel()[0])
        col = int(np.asarray(col).ravel()[0])
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(
                f"point {(x, y)} is outside the saved window "
                f"({self._rows}x{self._cols})")
        return row * self._cols + col

    def _indices(self, points) -> np.ndarray:
        """Cells holding ``points``, vectorised.

        Siting prices candidates by the hundred thousand, so this goes
        through rasterio's array form of ``rowcol`` once rather than
        calling :meth:`_index` per point.
        """
        from rasterio.transform import rowcol
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be an (n, 2) array of (x, y)")
        rows, cols = rowcol(self.transform, arr[:, 0], arr[:, 1])
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        outside = ((rows < 0) | (rows >= self._rows)
                   | (cols < 0) | (cols >= self._cols))
        if outside.any():
            first = int(np.flatnonzero(outside)[0])
            raise ValueError(
                f"{int(outside.sum())} point(s) fall outside the saved "
                f"window ({self._rows}x{self._cols}), first is "
                f"{tuple(arr[first])}")
        return rows * self._cols + cols

    @property
    def codec(self) -> str:
        """Which codec wrote this field; ``"raw"`` for an exact one."""
        return getattr(self, "_codec", "raw")

    @property
    def error_bound(self) -> float:
        """Guaranteed one-sided understatement of ``bound="lower"``, in EUR,
        relative to the labels the kernel settled.

        ``0.0`` only for a field stored exactly (``storage="float64"``). A
        float32 field understates by at most the rounding plus the nudge,
        a few ulps of its largest finite label. Whether those labels are
        themselves the least cost is :attr:`certifiable`.
        """
        if self._error_bound is None:
            if self._decoded is not None:
                self._error_bound = float(self._decoded.width())
            else:
                self._error_bound = _raw_error_bound(
                    self._dist, self._nudge_down, self._rounding)
        return self._error_bound

    @property
    def label_precision(self) -> str:
        """``"float64"`` when the kernel accumulated labels in float64 (the
        Dijkstra solver), else ``"float32"``. Read from the record, or
        from the algorithm of a field written before records existed."""
        rec = self._meta.get("provenance") or {}
        prec = (rec.get("algorithm") or {}).get("label_precision")
        if prec is None:
            prec = self._meta.get("label_precision")
        if prec is None:
            prec = ("float64" if str(self._meta.get("algorithm", "")).lower()
                    == "dijkstra" else "float32")
        return str(prec)

    @property
    def certifiable(self) -> bool:
        """True when ``bound="lower"``/``"upper"`` bracket the graph's least
        cost itself, not only the kernel's labels: the labels were
        accumulated in float64 by the Dijkstra solver. A delta-stepping,
        GPU or eikonal field is a good estimate but never a certificate
        (plan section 3.7)."""
        return self.label_precision == "float64" and math.isfinite(
            self.error_bound)

    @property
    def provenance(self) -> dict | None:
        """The plan-A1 provenance record stored with the field, if any."""
        return self._meta.get("provenance")

    def costs_to(self, points: Sequence, *,
                 bound: str = "lower") -> np.ndarray:
        """Least-cost distance from :attr:`origin` to each point.

        Same units and the same ``inf``-where-unreachable convention as
        :meth:`CostField.costs_to`.

        Parameters:
            bound: Which end of the interval the value sits at.
                ``"lower"`` (the default) never overstates, so it is the
                end a screening bound or a ``<=`` prune wants; ``"upper"``
                never understates, which is what a reported objective, an
                incumbent, or any inequality that SUBTRACTS this value
                requires. Only a field stored exactly
                (``storage="float64"``) has both ends equal; a float32
                field is nudged outward by its stored rounding.
                ``"label"`` returns the stored value as it is -- the
                nearest estimate, what a live field would report, and no
                bound at all. The ends bracket the least cost only on a
                :attr:`certifiable` field; otherwise they bracket the
                kernel's labels.
        """
        if bound not in ("lower", "upper", "label"):
            raise ValueError(f"bound must be 'lower', 'upper' or 'label', "
                             f"got {bound!r}")
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return np.empty(0, dtype=np.float64)
        idx = self._indices(arr.reshape(-1, 2))
        if self._decoded is not None:
            if bound in ("lower", "label"):
                return self._dist[idx].astype(np.float64)
            return self._decoded.choose(bound).ravel()[idx]
        raw = self._dist[idx]
        if bound == "label":
            return raw.astype(np.float64)
        if bound == "lower":
            return _nudge(raw, self._nudge_down, -np.inf).astype(np.float64)
        return _nudge(raw, self._nudge_up, np.inf).astype(np.float64)

    def cost_to(self, point, *, bound: str = "lower") -> float:
        """:meth:`costs_to` for one point."""
        return float(self.costs_to([point], bound=bound)[0])

    def path_cells(self, point) -> np.ndarray:
        """The route origin -> point as flat cell indices.

        Walks the stored predecessor steps. Empty when no route exists.
        """
        if self._pred_step is None:
            raise NotImplementedError(
                f"{self._path.name} was saved with with_paths=False, so it "
                f"prices candidates but cannot return routes")
        idx = self._index(point)
        if idx == self._origin_idx:
            return np.array([idx], dtype=np.int64)
        if not np.isfinite(self._dist[idx]):
            return np.empty(0, dtype=np.int64)
        walk = [idx]
        cur = idx
        limit = self._rows * self._cols + 1
        while len(walk) < limit:
            step = int(self._pred_step[cur])  # pylint: disable=unsubscriptable-object  # attribute is an array when set
            if step == _NO_STEP:
                return np.empty(0, dtype=np.int64)
            dr, dc = self._steps[step]
            cur = cur - int(dr) * self._cols - int(dc)
            walk.append(cur)
            if cur == self._origin_idx:
                walk.reverse()
                return np.array(walk, dtype=np.int64)
        return np.empty(0, dtype=np.int64)

    def path_coords(self, point) -> np.ndarray:
        """:meth:`path_cells` as an ``(n, 2)`` array of map coordinates."""
        from rasterio.transform import xy
        cells = self.path_cells(point)
        if cells.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        rows, cols = np.divmod(cells, self._cols)
        xs, ys = xy(self.transform, rows.tolist(), cols.tolist())
        return np.column_stack([np.asarray(xs, dtype=np.float64),
                                np.asarray(ys, dtype=np.float64)])

    def path_length_m(self, point) -> float:
        """Route length in metres, summed over the steps it is made of.

        Consecutive cells of :meth:`path_cells` are exactly one step
        apart, and PYORPS measures a step as
        ``sqrt(dr^2 + dc^2) * cell_size``, so this reproduces
        ``Path.total_length`` without the metric kernel.
        """
        cells = self.path_cells(point)
        if cells.size < 2:
            return 0.0
        rows, cols = np.divmod(cells, self._cols)
        dr = np.diff(rows.astype(np.float64))
        dc = np.diff(cols.astype(np.float64))
        return float(np.hypot(dr, dc).sum() * self._cell_size)

    def field_array(self, dtype=np.float32) -> np.ndarray:
        """The whole field as a window-shaped 2-D array."""
        return self._dist.astype(dtype, copy=False).reshape(
            self._rows, self._cols)

    # ------------------------------------------------------------ lifetime

    def close(self) -> None:
        """Release the loaded arrays. Reopen with :meth:`CostField.open`."""
        self._dist = np.empty(0, dtype=np.float32)
        self._pred_step = None
        self._decoded = None

    def __enter__(self) -> SavedCostField:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"SavedCostField({self._path.name!r}, "
                f"{self._rows}x{self._cols}, paths={self.has_paths})")


class Leg(NamedTuple):
    """One route out of a field, live or reopened.

    The two kinds of field carry the same answers behind different names,
    so :meth:`CostFieldSet.path_to` returns this instead and the caller
    never has to know which it holds.

    One caveat on :attr:`cost`, because the set picks resident-vs-spilled
    by itself: a RESIDENT field re-prices the route in float64 and reports
    ``Path.total_cost``; a SPILLED one has no raster behind it and reports
    the stored float32 label instead. The two agree to float32 precision
    (order 1e-7 relative), but they are not the same computation, so do not
    difference them against each other and expect zero. For a number whose
    provenance does not depend on how much memory the machine had, read
    :meth:`CostFieldSet.costs_to`, which is the label on both paths.
    """
    cells: np.ndarray          #: flat cell indices, origin first
    coords: np.ndarray         #: (n, 2) map coordinates, origin first
    length_m: float            #: route length in metres
    cost: float                #: route cost; see the note above on provenance


class CostFieldSet:
    """One settled field per FIXED terminal; every moving point is a lookup.

    This is the entry point for siting: a substation may stand anywhere,
    but the turbines and grid connection points do not move, so the search
    is rooted at THEM and turned around. One sweep per terminal replaces
    one sweep per candidate::

        with finder.cost_fields(turbines + pccs, labels=names) as fields:
            costs = fields.costs_to(candidate_sites)   # (n_terminals, n)
            best = candidate_sites[int(costs.sum(axis=0).argmin())]
            leg = fields.path_to("WT0", best)          # cells, coords, m, EUR

    Ten terminals and forty million candidates is ten searches, not forty
    million. The inversion is exact, not an approximation: the raster graph
    is undirected, so ``d(terminal -> candidate) == d(candidate ->
    terminal)`` and the field answers a question posed the other way round.
    Measured against independently computed pairwise routes on the CIRED
    case study: agreement 5.5e-09 relative, symmetry exactly 0.0.

    Fields are settled with delta-stepping by default -- verified exact,
    deterministic and thread-invariant against the float64 Dijkstra over
    35.6 M reachable cells -- which is about 8x the serial Dijkstra on the
    CPU and 15x on ``graph_api="raster_gpu"``.

    Memory
    ------
    Fields are held live when they fit and paged through disk when they do
    not; :func:`fields_fit_in_memory` makes that call, and ``spill`` forces
    either way. Ten 240 M-cell fields are about 19 GB live, so on a big
    window the set will spill -- each field is then settled once, written,
    and reopened on demand at 5 B/cell.
    """

    def __init__(self, finder: PathFinder, origins: Sequence, *,
                 algorithm: str = "auto", labels: Sequence[str] | None = None,
                 cache_dir=None, spill: bool | None = None,
                 codec: str = "raw", error_bound: float | None = None,
                 **algo_kwargs: Any):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        import tempfile
        from pathlib import Path as _Path

        from pyorps.io.field_codec import is_lossy

        if is_lossy(codec):
            # Whether a field is held live or written through the codec
            # is decided by psutil at run time, so a lossy spill would
            # make the numbers this set returns depend on how much
            # memory the machine happened to have. Save the fields
            # explicitly if a lossy cache is what is wanted.
            raise ValueError(
                f"CostFieldSet refuses codec={codec!r}: it chooses "
                f"resident-vs-spilled from available memory, so a lossy "
                f"spill would make precision machine-dependent. Settle "
                f"one CostField at a time and call .save(codec=...) if a "
                f"quantised cache is what you want.")

        if finder.raster_handler is None:
            finder.create_raster_handler()
        pts = _as_points(origins)
        if not pts:
            raise ValueError("CostFieldSet needs at least one origin")
        if labels is None:
            labels = [f"t{i}" for i in range(len(pts))]
        labels = [str(x) for x in labels]
        if len(labels) != len(pts):
            raise ValueError(
                f"{len(labels)} labels for {len(pts)} origins")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")

        self._finder = finder
        self._labels = tuple(labels)
        self._origins = tuple(pts)
        self._index_of = {lab: i for i, lab in enumerate(self._labels)}
        self._algorithm = algorithm
        self._algo_kwargs = dict(algo_kwargs)
        self._codec = codec
        self._error_bound = error_bound
        rows, cols = finder.raster_handler.data.shape[-2:]
        self._cells = int(rows) * int(cols)

        resolved = _resolve_algorithm(finder, algorithm)
        if spill is None:
            spill = not fields_fit_in_memory(len(pts), self._cells,
                                             algorithm=resolved)
        self._spilled = bool(spill)
        self._tmp = None
        if self._spilled:
            if cache_dir is None:
                self._tmp = tempfile.mkdtemp(prefix="pyorps_fields_")
                cache_dir = self._tmp
            self._cache = _Path(cache_dir)
            self._cache.mkdir(parents=True, exist_ok=True)
        else:
            self._cache = None

        self._live: list[CostField | None] = [None] * len(pts)
        self._paths: list[Any] = [None] * len(pts)
        self._closed = False
        self._settle_all_fields()

    # ---------------------------------------------------------- properties

    @property
    def labels(self) -> tuple[str, ...]:
        """Name of each terminal, in the row order :meth:`costs_to` returns."""
        return self._labels

    @property
    def origins(self) -> tuple[Coordinate, ...]:
        """The FIXED terminals, as ``(x, y)``, aligned with :attr:`labels`."""
        return self._origins

    @property
    def spilled(self) -> bool:
        """True when fields live on disk rather than in memory."""
        return self._spilled

    @property
    def algorithm(self) -> str:
        """The algorithm actually used, with ``"auto"`` resolved."""
        return _resolve_algorithm(self._finder, self._algorithm)

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the resident fields. A spilled set holds at most one."""
        return sum(f.memory_bytes for f in self._live if f is not None)

    # ------------------------------------------------------------ settling

    def _settle_all_fields(self) -> None:
        for i, origin in enumerate(self._origins):
            cfield = self._finder.cost_field(
                origin, algorithm=self._algorithm, **self._algo_kwargs)
            if self._spilled:
                # Settle, write, release: only one field is resident while
                # the set is being built, whatever its size.
                self._paths[i] = cfield.save(
                    self._cache / f"field_{self._labels[i]}.npz")
                cfield.close()
            else:
                self._live[i] = cfield

    def _open(self, i: int):
        """The field for terminal ``i``, live or reopened from disk."""
        if self._live[i] is not None:
            return self._live[i], False
        return SavedCostField(self._paths[i]), True

    # ------------------------------------------------------------- queries

    def _resolve(self, label) -> int:
        if isinstance(label, (int, np.integer)) and label not in self._index_of:
            return int(label)
        try:
            return self._index_of[str(label)]
        except KeyError:
            raise KeyError(
                f"no terminal {label!r}; have {list(self._labels)}") from None

    def costs_to(self, points: Sequence, labels: Sequence | None = None
                 ) -> np.ndarray:
        """Cost from every terminal to every point.

        Returns:
            ``(n_terminals, n_points)`` float64, row order matching
            :attr:`labels` (or ``labels`` when given). ``inf`` where no
            route exists.
        """
        self._check_open()
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        idxs = ([self._resolve(x) for x in labels] if labels is not None
                else list(range(len(self._labels))))
        out = np.empty((len(idxs), arr.shape[0]), dtype=np.float64)
        for row, i in enumerate(idxs):
            cfield, temporary = self._open(i)
            try:
                # The stored label, not a bound: resident and spilled
                # fields then report the same number to float32 precision.
                out[row] = (cfield.costs_to(arr, bound="label") if temporary
                            else cfield.costs_to(arr))
            finally:
                if temporary:
                    cfield.close()
        return out

    def cost_to(self, label, point) -> float:
        """Cost from one terminal to one point."""
        return float(self.costs_to([point], labels=[label])[0, 0])

    def nearest(self, points: Sequence, labels: Sequence | None = None):
        """Cheapest terminal for each point.

        Returns:
            ``(costs, label_index)`` -- the minimum over the chosen
            terminals and which one achieved it. Points reachable from no
            terminal come back as ``inf`` with index ``-1``.
        """
        costs = self.costs_to(points, labels=labels)
        best = np.argmin(costs, axis=0)
        value = costs[best, np.arange(costs.shape[1])]
        best = np.where(np.isfinite(value), best, -1)
        return value, best

    def path_to(self, label, point) -> Leg:
        """The route between one terminal and a point.

        Raises:
            NoPathFoundError: no route exists.
        """
        self._check_open()
        i = self._resolve(label)
        cfield, temporary = self._open(i)
        try:
            if isinstance(cfield, SavedCostField):
                cells = cfield.path_cells(point)
                if cells.size == 0:
                    raise NoPathFoundError(i, -1)
                return Leg(cells, cfield.path_coords(point),
                           float(cfield.path_length_m(point)),
                           float(cfield.costs_to([point], bound="label")[0]))
            path = cfield.path_to(point, calculate_metrics=True)
            coords = np.asarray([[float(x), float(y)]
                                 for x, y in path.path_coords])
            cells = np.asarray(getattr(path, "path_indices", []),
                               dtype=np.int64).ravel()
            return Leg(cells, coords, float(path.total_length),
                       float(path.total_cost))
        finally:
            if temporary:
                cfield.close()

    # ------------------------------------------------------------ lifetime

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("CostFieldSet is closed")

    def close(self) -> None:
        """Release every field, and any temporary cache this set created."""
        if self._closed:
            return
        for f in self._live:
            if f is not None:
                f.close()
        self._live = [None] * len(self._labels)
        if self._tmp is not None:
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
        self._closed = True

    def __enter__(self) -> CostFieldSet:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        where = "on disk" if self._spilled else "in memory"
        return (f"CostFieldSet({len(self._labels)} terminals, "
                f"{self.algorithm}, {where})")
