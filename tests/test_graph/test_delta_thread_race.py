"""Multi-threaded delta-stepping stays optimal on tie-heavy rasters.

Regression test for a kernel race: the persistent delta-stepping kernels were
nondeterministically suboptimal at two or more threads (0.08 - 2.8 % too
expensive on some runs; fixed 2026-08-05). A race only shows on some runs, so
each instance is solved ``REPS`` times at ``THREADS`` threads, and every
returned path is re-priced in float64 and compared with an independent float64
Dijkstra written here with ``heapq``.

The rasters provoke ties: almost every cell costs the same, some cells cost 0
and a few are forbidden (``65535``, the ``ignore_max_cost=True`` sentinel), so
many equal-cost labels compete for the same cell inside one bucket.

Covered entry points: everything the public API reaches. The ``cython``
backend (``pyorps/graph/api/cython_api.py``) calls the four persistent kernels
(``delta_stepping_2d_persistent``, ``..._single_source_multiple_targets_
persistent``, ``..._some_pairs_shortest_paths_persistent``, ``..._multiple_
sources_multiple_targets_persistent``) with ``delta=100`` and the exact
termination margin; ``pyorps.utils`` exports the same four under the plain
names. ``SearchSession`` (``pyorps/graph/search_session.py``) reaches
``delta_stepping_2d_persistent`` a second way, through ``_DeltaTree``: a
full-field solve into a ``DeltaWorkspace`` and a walk of the retained labels.

TOLERANCE (derived from the kernel's arithmetic, ``_delta_stepping.pyx``
``relax_vertex_edges_inline`` and ``_raster_context.pyx``
``_get_cost_factor_cython_f32``; the extensions build with ``/fp:precise`` or
``-fno-fast-math``, ``setup.py``):

* The cell sum ``raster[u] + sum(raster[intermediates]) + raster[v]`` adds
  uint16 integers in float32. It stays below ``2**24`` for every
  neighbourhood used here, so it is EXACT.
* The float32 cost factor ``sqrtf(dr*dr + dc*dc) / (2 + n)`` costs two
  roundings and the product with the exact sum one more: every edge weight is
  ``w * (1 + theta_3)``, ``|theta_k| <= gamma_k = k*u / (1 - k*u)``,
  ``u = 2**-24``.
* A label adds one edge weight per hop in float32 (one rounding per hop), so
  for any path ``P`` with ``H`` hops the kernel's value ``d32(P)`` lies within
  ``gamma_(H+3) * c(P)`` of its exact cost ``c(P)`` (all weights >= 0).
* Rounding is monotone, so an exact (race-free) kernel returns a path ``P``
  with ``d32(P) <= d32(P*)`` for the float64 optimum ``P*`` (``H*`` hops):

      c(P) <= c(P*) * (1 + gamma_(H*+3)) / (1 - gamma_(H+3))

  i.e. a relative excess of at most about ``(H + H* + 6) * u``; at the 9 to 42
  hops of these instances that is at most ~5e-6, while the recorded race
  overpaid by 8e-4 to 2.8e-2, and ``test_judge_rejects_a_route_one_detour_
  longer`` pins that a detour of one cell is rejected.
* The reference and the re-pricing accumulate in float64; ``(H + H* + 8)``
  units of ``2**-53`` cover both.

The same tolerance bounds the excess in the other direction: a path cheaper
than the reference by more than it means the reference is wrong.
"""
from __future__ import annotations

import functools
import heapq
import math

import numpy as np
import pytest
from exactness_referee import EdgeModel, reprice_path

from pyorps.graph.api.cython_api import EXACT_TERMINATION_MARGIN

THREADS = 4
REPS = 6
DELTA = 100.0          # the default the cython backend and SearchSession pass
MAX_VALUE = 65535      # forbidden cells; ignore_max_cost=True semantics
U32 = 2.0 ** -24
U64 = 2.0 ** -53

SOURCES = ((1, 1), (38, 2), (20, 21))
TARGETS = ((38, 38), (1, 37), (26, 5))


# ------------------------------------------------------------------------------------------ rasters


def _pin_endpoints(raster: np.ndarray) -> np.ndarray:
    for r, c in SOURCES + TARGETS:
        raster[r, c] = 1
    return raster


def _flat_ties() -> np.ndarray:
    """Every cell costs 1, except 12 % zero-cost and 4 % forbidden cells."""
    rng = np.random.default_rng(20260805)
    raster = np.ones((40, 40), dtype=np.uint16)
    draw = rng.random(raster.shape)
    raster[draw < 0.12] = 0
    raster[draw > 0.96] = MAX_VALUE
    return _pin_endpoints(raster)


def _two_level() -> np.ndarray:
    """Costs 1 or 2 at random, 8 % zero-cost and 4 % forbidden cells."""
    rng = np.random.default_rng(811)
    raster = rng.integers(1, 3, size=(40, 40)).astype(np.uint16)
    draw = rng.random(raster.shape)
    raster[draw < 0.08] = 0
    raster[draw > 0.96] = MAX_VALUE
    return _pin_endpoints(raster)


def _zero_lake_wall() -> np.ndarray:
    """Cost 3 with a zero-cost block and a forbidden wall with two gaps."""
    raster = np.full((40, 40), 3, dtype=np.uint16)
    raster[12:24, 14:30] = 0
    raster[30, 3:37] = MAX_VALUE
    raster[30, 10] = 3
    raster[30, 31] = 0
    return _pin_endpoints(raster)


RASTERS = {
    "flat_ties": _flat_ties,
    "two_level": _two_level,
    "zero_lake_wall": _zero_lake_wall,
}


@functools.cache
def _steps() -> np.ndarray:
    from pyorps.utils.neighborhood import get_neighborhood_steps

    return np.ascontiguousarray(get_neighborhood_steps("r2"), dtype=np.int8)


@functools.cache
def _model() -> EdgeModel:
    return EdgeModel(_steps(), max_value=MAX_VALUE, cost_factor_precision="float64")


@functools.cache
def _raster(name: str) -> np.ndarray:
    # Writable on purpose: the kernels take a typed (non-const) buffer.
    return RASTERS[name]()


def _flat(cell) -> int:
    return cell[0] * 40 + cell[1]


# ------------------------------------------------------------------------------------------ reference


def _edges(raster: np.ndarray):
    """Per cell: (neighbour, float64 weight) under the kernels' edge model."""
    rows, cols = raster.shape
    model = _model()
    table = [model.edge(int(dr), int(dc)) for dr, dc in _steps()]
    steps = [(int(dr), int(dc)) for dr, dc in _steps()]
    out = []
    for r in range(rows):
        for c in range(cols):
            adj = []
            if raster[r, c] != MAX_VALUE:
                for (dr, dc), (_, inter, factor) in zip(steps, table):
                    vr, vc = r + dr, c + dc
                    if not (0 <= vr < rows and 0 <= vc < cols):
                        continue
                    if raster[vr, vc] == MAX_VALUE:
                        continue
                    total = float(raster[r, c]) + float(raster[vr, vc])
                    legal = True
                    for ddr, ddc in inter:
                        ir, ic = r + ddr, c + ddc
                        if (not (0 <= ir < rows and 0 <= ic < cols)
                                or raster[ir, ic] == MAX_VALUE):
                            legal = False
                            break
                        total += float(raster[ir, ic])
                    if legal:
                        adj.append((vr * cols + vc, total * factor))
            out.append(adj)
    return out


@functools.cache
def _reference(name: str, source: int) -> tuple[np.ndarray, np.ndarray]:
    """Float64 Dijkstra from ``source``: (cost, hops of the path found) per cell."""
    adj = _edges(_raster(name))
    n = len(adj)
    dist = np.full(n, math.inf)
    hops = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                hops[v] = hops[u] + 1
                heapq.heappush(heap, (nd, v))
    return dist, hops


def _tolerance(hops_ref: int, hops_cand: int) -> float:
    def gamma(k):
        return k * U32 / (1.0 - k * U32)

    g_ref, g_cand = gamma(hops_ref + 3), gamma(hops_cand + 3)
    return (g_ref + g_cand) / (1.0 - g_cand) + (hops_ref + hops_cand + 8) * U64


def _judge(name: str, source: int, target: int, path, tolerance=_tolerance) -> str:
    """'' when ``path`` is a legal route at the reference cost, else the reason."""
    cells = [int(x) for x in path]
    if not cells:
        return "no path"
    if cells[0] != source or cells[-1] != target:
        return f"runs {cells[0]} -> {cells[-1]}"
    priced = reprice_path(cells, _raster(name), _model())
    if not priced.ok:
        return f"illegal route: {priced.violations[:3]}"
    dist, hops = _reference(name, source)
    want = float(dist[target])
    tol = tolerance(int(hops[target]), priced.hops)
    if abs(priced.cost - want) > tol * want:
        return (f"cost {priced.cost!r}, reference {want!r} "
                f"(excess {(priced.cost - want) / want:+.3e}, tolerance {tol:.3e})")
    return ""


def _pairs():
    return [(_flat(s), _flat(t)) for s in SOURCES for t in TARGETS]


# ------------------------------------------------------------------------------------------ entry points


def _run_2d(raster, steps):
    from pyorps.utils._delta_stepping import delta_stepping_2d_persistent

    return {(s, t): delta_stepping_2d_persistent(
        raster, steps, np.uint64(s), np.uint64(t), delta=DELTA,
        max_value=MAX_VALUE, num_threads=THREADS, margin=EXACT_TERMINATION_MARGIN)
        for s, t in _pairs()}


def _run_sssmt(raster, steps):
    from pyorps.utils._delta_stepping import (
        delta_stepping_single_source_multiple_targets_persistent as kernel,
    )

    targets = np.array([_flat(t) for t in TARGETS], dtype=np.uint64)
    out = {}
    for s in (_flat(c) for c in SOURCES):
        paths = kernel(raster, steps, np.uint64(s), targets, delta=DELTA,
                       max_value=MAX_VALUE, num_threads=THREADS)
        out.update({(s, int(t)): p for t, p in zip(targets, paths)})
    return out


def _run_some_pairs(raster, steps):
    from pyorps.utils._delta_stepping import (
        delta_stepping_some_pairs_shortest_paths_persistent as kernel,
    )

    pairs = _pairs()
    paths = kernel(raster, steps, np.array([s for s, _ in pairs], dtype=np.uint64),
                   np.array([t for _, t in pairs], dtype=np.uint64), delta=DELTA,
                   max_value=MAX_VALUE, return_paths=True, num_threads=THREADS,
                   margin=EXACT_TERMINATION_MARGIN)
    return dict(zip(pairs, paths))


def _run_all_pairs(raster, steps):
    from pyorps.utils._delta_stepping import (
        delta_stepping_multiple_sources_multiple_targets_persistent as kernel,
    )

    sources = [_flat(c) for c in SOURCES]
    targets = [_flat(c) for c in TARGETS]
    paths = kernel(raster, steps, np.array(sources, dtype=np.uint64),
                   np.array(targets, dtype=np.uint64), delta=DELTA,
                   max_value=MAX_VALUE, return_paths=True, num_threads=THREADS)
    return {(s, t): paths[i][j] for i, s in enumerate(sources)
            for j, t in enumerate(targets)}


def _run_session_tree(raster, steps):
    from pyorps.graph.search_session import _DeltaTree

    out = {}
    for s in (_flat(c) for c in SOURCES):
        tree = _DeltaTree(raster, steps, MAX_VALUE, s, delta=DELTA, num_threads=THREADS)
        try:
            out.update({(s, _flat(t)): tree.extract_or_resume(_flat(t)) for t in TARGETS})
        finally:
            tree.close()
    return out


ENTRY_POINTS = {
    "2d_persistent": _run_2d,
    "sssmt_persistent": _run_sssmt,
    "some_pairs_persistent": _run_some_pairs,
    "all_pairs_persistent": _run_all_pairs,
    "session_tree": _run_session_tree,
}


# ------------------------------------------------------------------------------------------ tests


@pytest.mark.parametrize("name", RASTERS)
def test_reference_agrees_with_the_cython_dijkstra(name):
    """Guard the guard: the heapq reference prices the kernels' edge model.

    The Cython Dijkstra accumulates in float64 from a float32 cost factor (one
    rounding of the factor), so the two optima agree to ``2u`` plus float64
    slack.
    """
    from pyorps.utils._dijkstra import make_dijkstra_solver

    def tol(hops_ref, hops_cand):
        return 2 * U32 / (1 - U32) + (hops_ref + hops_cand + 8) * U64

    raster = np.ascontiguousarray(_raster(name))
    solver = make_dijkstra_solver(raster, _steps(), max_value=MAX_VALUE)
    for s, t in _pairs():
        assert math.isfinite(_reference(name, s)[0][t]), f"{s} -> {t} unreachable"
        reason = _judge(name, s, t, solver.single_pair(s, t), tolerance=tol)
        assert not reason, f"{s} -> {t}: {reason}"


def test_judge_rejects_a_route_one_detour_longer():
    """The tolerance is tight enough to see a small suboptimality.

    The route source -> m -> target through a cell ``m`` next to the optimal
    route is legal and only slightly more expensive; the judge must reject it.
    """
    from pyorps.utils._dijkstra import make_dijkstra_solver

    name = "two_level"
    raster = np.ascontiguousarray(_raster(name))
    solver = make_dijkstra_solver(raster, _steps(), max_value=MAX_VALUE)
    s, t = _pairs()[0]
    best = [int(x) for x in solver.single_pair(s, t)]
    assert _judge(name, s, t, best) == ""
    dist, _ = _reference(name, s)
    on_route = set(best)
    worse = []
    for m in best[len(best) // 2:]:
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            r, c = divmod(m, 40)
            cand = (r + dr) * 40 + (c + dc)
            if not (0 <= r + dr < 40 and 0 <= c + dc < 40) or cand in on_route:
                continue
            if raster[r + dr, c + dc] == MAX_VALUE:
                continue
            first = [int(x) for x in solver.single_pair(s, cand)]
            second = [int(x) for x in solver.single_pair(cand, t)]
            if first and second and not set(first[:-1]) & set(second):
                worse.append(first + second[1:])
    # Many detours tie with the optimum on this raster; take the cheapest
    # one that does not.
    excess = {i: reprice_path(p, raster, _model()).cost / dist[t] - 1.0
              for i, p in enumerate(worse)}
    strictly_worse = {i: e for i, e in excess.items() if e > 1e-9}
    assert strictly_worse, "no strictly longer detour on this raster"
    i = min(strictly_worse, key=strictly_worse.get)
    assert strictly_worse[i] < 0.05
    assert _judge(name, s, t, worse[i]).startswith("cost")


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("name", RASTERS)
def test_delta_stepping_is_optimal_at_four_threads_on_every_run(name, entry):
    raster = np.ascontiguousarray(_raster(name))
    failures = []
    for rep in range(REPS):
        paths = ENTRY_POINTS[entry](raster, _steps())
        assert set(paths) == set(_pairs()), f"run {rep}: wrong pairs {sorted(paths)}"
        for (s, t), path in sorted(paths.items()):
            reason = _judge(name, s, t, path)
            if reason:
                failures.append(f"run {rep}, {s} -> {t}: {reason}")
    assert not failures, (f"{len(failures)} of {REPS * len(_pairs())} routes off "
                          f"the optimum:\n" + "\n".join(failures[:20]))
