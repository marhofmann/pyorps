"""Delta-stepping on rasters with neighbouring cost-0 cells.

Before the fix, the CPU delta-stepping kernels compared the packed 64-bit
``(float32 distance << 32) | predecessor`` label as a whole, so a relaxation
over a zero-weight edge that only TIED the distance was still accepted when
its predecessor id was smaller. Two neighbouring cost-0 cells at the same
distance then became each other's predecessor (cells 0 and 1 of the gradient
test raster), and every path walk that ran into the 2-cycle only stopped at
``max_path_length``, building a list of about a billion entries
(``SearchSession`` / the GUI blocked for over a minute per route).

These tests pin:

* every public delta-stepping variant RETURNS on zero-cost rasters at 1 and 4
  threads (each raster/thread combination runs in a subprocess with a
  timeout, so a regression fails instead of hanging the suite or eating RAM);
* the returned path costs what the Cython Dijkstra path costs;
* the retained predecessor fields contain no cycle;
* ``_DeltaTree`` builds on the gradient raster the GUI test uses;
* every predecessor walk is bounded and raises ``PyorpsRuntimeError`` on a
  hand-made cyclic field.

The subprocess timeout can be lowered with ``PYORPS_TEST_ZERO_COST_TIMEOUT``
(seconds) when proving a failure on a broken build: a hanging walk grows its
list by several hundred MB per second.
"""
from __future__ import annotations

import functools
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

WORKER_TIMEOUT = float(os.environ.get("PYORPS_TEST_ZERO_COST_TIMEOUT", "90"))
THREADS = (1, 4)


# ------------------------------------------------------------------------------------------ rasters


def _gradient_corner():
    """The GUI test pattern (gradient over 200 cells) cropped to 40 x 40.

    ``(x + y) / 400 * 100`` truncated: the ten cells with ``x + y < 4`` cost
    0, so the corner holds several neighbouring zero cells at equal distance.
    """
    y, x = np.mgrid[0:40, 0:40]
    raster = ((x + y) / 400 * 100).astype(np.uint16)
    return raster, [(5, 0), (20, 20)], [(0, 0), (39, 39)]


def _zero_stripe():
    raster = np.full((40, 40), 10, dtype=np.uint16)
    raster[20, :] = 0
    return raster, [(0, 0), (39, 5)], [(20, 39), (39, 39)]


def _all_zero():
    raster = np.zeros((30, 30), dtype=np.uint16)
    return raster, [(0, 0), (15, 15)], [(29, 29), (0, 29)]


def _zero_plateau():
    raster = np.full((40, 40), 50, dtype=np.uint16)
    raster[15:25, 15:25] = 0
    return raster, [(2, 2), (37, 2)], [(20, 20), (37, 37)]


RASTERS = {
    "gradient_corner": _gradient_corner,
    "zero_stripe": _zero_stripe,
    "all_zero": _all_zero,
    "zero_plateau": _zero_plateau,
}

#: Path-returning variants reachable through PathFinder / SearchSession.
VARIANTS = ("delta-stepping", "delta-stepping-circular", "multi_target",
            "pairwise", "all_pairs", "session")

#: Retained predecessor fields checked for cycles.
FIELDS = ("kernel_2d", "kernel_2d_persistent", "kernel_sssmt",
          "kernel_sssmt_persistent", "session_tree")


# ------------------------------------------------------------------------------------------ cycle check


def _cycle_cells(pred) -> int:
    """Cells whose predecessor chain never ends (pointer doubling, O(n log n)).

    ``pred`` holds cell ids; anything outside ``[0, n)`` (``-1`` or
    ``0xFFFFFFFF``) ends a chain. A source with a predecessor counts too: the
    root of a tree must not have one.
    """
    pred = np.asarray(pred).astype(np.int64)
    n = pred.size
    nxt = np.where((pred < 0) | (pred >= n), n, pred)
    nxt = np.append(nxt, n)
    for _ in range(int(math.ceil(math.log2(n + 1))) + 1):
        nxt = nxt[nxt]
    return int(np.count_nonzero(nxt[:n] != n))


# ------------------------------------------------------------------------------------------ worker


def _emit(case, **payload):
    print(json.dumps({"case": case, **payload}), flush=True)


def _worker(name: str, threads: int) -> None:  # pragma: no cover - runs in a subprocess
    from rasterio.transform import from_origin

    from pyorps.graph.path_finder import PathFinder
    from pyorps.graph.search_session import SearchSession, _raster_of
    from pyorps.utils import _delta_stepping as ds

    raster, srcs, tgts = RASTERS[name]()
    rows, cols = raster.shape

    def xy(cell):
        return (cell[1] + 0.5, rows - cell[0] - 0.5)

    def flat(cell):
        return cell[0] * cols + cell[1]

    def label(coord, cells):
        return min(range(len(cells)), key=lambda k: math.dist(coord, xy(cells[k])))

    finder = PathFinder(
        dataset_source=raster, crs="EPSG:32632",
        transform=from_origin(0.0, float(rows), 1.0, 1.0),
        source_coords=[xy(srcs[0])], target_coords=[xy(tgts[0])],
        search_space_buffer_m=10_000.0, graph_api="cython",
        neighborhood_str="r2", ignore_max_cost=True)

    def run(case, fn):
        try:
            out = fn()
        except Exception as exc:  # reported, judged by the parent
            _emit(case, error=f"{type(exc).__name__}: {exc}")
            return None
        return out

    def route(source, target, algorithm, **kw):
        return finder.find_route(source=source, target=target, algorithm=algorithm,
                                 calculate_metrics=False, **kw)

    # Dijkstra references (never affected by the bug).
    for i, s in enumerate(srcs):
        for j, t in enumerate(tgts):
            p = run(f"ref|{i}|{j}", lambda s=s, t=t: route(xy(s), xy(t), "dijkstra"))
            if p is not None:
                _emit(f"ref|{i}|{j}", path=[int(c) for c in p.path_indices])
    window = _raster_of(finder)
    _emit("window", raster=np.asarray(window).tolist(),
          steps=np.asarray(finder.steps).tolist())

    for algo in ("delta-stepping", "delta-stepping-circular"):
        for j, t in enumerate(tgts):
            case = f"{algo}|0|{j}"
            p = run(case, lambda t=t, algo=algo: route(
                xy(srcs[0]), xy(t), algo, num_threads=threads))
            if p is not None:
                _emit(case, path=[int(c) for c in p.path_indices])

    coll = run("multi_target|0|0", lambda: route(
        xy(srcs[0]), [xy(t) for t in tgts], "delta-stepping", num_threads=threads))
    if coll is not None:
        for p in coll:
            _emit(f"multi_target|0|{label(p.target, tgts)}",
                  path=[int(c) for c in p.path_indices])

    coll = run("pairwise|0|0", lambda: route(
        [xy(s) for s in srcs], [xy(t) for t in tgts], "delta-stepping",
        pairwise=True))
    if coll is not None:
        for p in coll:
            i, j = label(p.source, srcs), label(p.target, tgts)
            _emit(f"pairwise|{i}|{j}", path=[int(c) for c in p.path_indices])

    coll = run("all_pairs|0|0", lambda: route(
        [xy(s) for s in srcs], [xy(t) for t in tgts], "delta-stepping",
        num_threads=threads))
    if coll is not None:
        for p in coll:
            i, j = label(p.source, srcs), label(p.target, tgts)
            _emit(f"all_pairs|{i}|{j}", path=[int(c) for c in p.path_indices])

    # Direct kernels on the full field: margin 1e30 settles every cell.
    r = np.ascontiguousarray(raster)
    steps = np.ascontiguousarray(finder.steps, dtype=np.int8)
    s0, far = np.uint64(flat(srcs[0])), np.uint64(flat(tgts[1]))
    targets = np.array([flat(t) for t in tgts], dtype=np.uint64)
    kernels = {
        "kernel_2d": lambda ws: ds.delta_stepping_2d(
            r, steps, s0, far, delta=100.0, max_value=65535,
            num_threads=threads, margin=1e30, workspace=ws),
        "kernel_2d_persistent": lambda ws: ds.delta_stepping_2d_persistent(
            r, steps, s0, far, delta=100.0, max_value=65535,
            num_threads=threads, margin=1e30, workspace=ws),
        "kernel_sssmt": lambda ws: ds.delta_stepping_single_source_multiple_targets(
            r, steps, s0, targets, delta=100.0, max_value=65535,
            num_threads=threads, workspace=ws),
        "kernel_sssmt_persistent":
            lambda ws: ds.delta_stepping_single_source_multiple_targets_persistent(
                r, steps, s0, targets, delta=100.0, max_value=65535,
                num_threads=threads, workspace=ws),
    }
    for case, fn in kernels.items():
        ws = ds.DeltaWorkspace(r, 65535)
        if run(case, lambda fn=fn, ws=ws: fn(ws)) is not None:
            _emit(case, cycles=_cycle_cells(ws.pred_array()))

    for i, j in ((0, 0), (1, 1)):
        session = SearchSession(finder, "delta-stepping", num_threads=threads)
        case = f"session|{i}|{j}"
        p = run(case, lambda session=session, i=i, j=j: session.route(
            [xy(srcs[i]), xy(tgts[j])]))
        if p is not None:
            _emit(case, path=[int(c) for c in p.path_indices])
            if i == 0:
                trees = list(session._forward.values())  # pylint: disable=protected-access
                _emit("session_tree",
                      cycles=sum(_cycle_cells(tree.pred()) for tree in trees),
                      trees=len(trees))
        session.close()
    _emit("done")


# ------------------------------------------------------------------------------------------ parent side


def _run_subprocess(args, timeout):
    """Run this file as a worker; return (stdout lines, failure text or '')."""
    env = dict(os.environ)
    env.setdefault("NUMBA_NUM_THREADS", "4")
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), *args],
            capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return out.splitlines(), f"did not return within {timeout:.0f} s (killed)"
    failure = "" if proc.returncode == 0 else (
        f"worker exited with {proc.returncode}:\n{proc.stderr[-3000:]}")
    return proc.stdout.splitlines(), failure


@functools.cache
def _results(name: str, threads: int) -> tuple[dict, str]:
    lines, failure = _run_subprocess(["worker", name, str(threads)], WORKER_TIMEOUT)
    records: dict[str, dict] = {}
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "case" in rec:
            records[rec["case"]] = rec
    return records, failure


def _variant_records(name, threads, variant):
    records, failure = _results(name, threads)
    hits = {k: v for k, v in records.items() if k.split("|")[0] == variant}
    if not hits:
        pytest.fail(f"{variant} on {name} at {threads} thread(s) produced no "
                    f"result: {failure or 'worker gave no output for it'}")
    return hits, records


PARAMS = [pytest.param(name, threads, id=f"{name}-t{threads}")
          for name in RASTERS for threads in THREADS]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name,threads", PARAMS)
def test_delta_variant_returns(name, threads, variant):
    hits, _ = _variant_records(name, threads, variant)
    errors = {k: v["error"] for k, v in hits.items() if "error" in v}
    assert not errors, errors


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name,threads", PARAMS)
def test_delta_variant_costs_what_dijkstra_costs(name, threads, variant):
    from exactness_referee import EdgeModel, optimality_tolerance, reprice_path

    hits, records = _variant_records(name, threads, variant)
    window = records.get("window")
    assert window is not None, "the worker did not report its raster window"
    raster = np.asarray(window["raster"], dtype=np.uint16)
    model = EdgeModel(np.asarray(window["steps"], dtype=np.int8))
    for case, rec in sorted(hits.items()):
        assert "error" not in rec, f"{case}: {rec['error']}"
        _, i, j = case.split("|")
        ref = records.get(f"ref|{i}|{j}")
        assert ref is not None and ref.get("path"), f"no Dijkstra reference for {case}"
        path = rec["path"]
        assert path, f"{case}: no path, Dijkstra found one"
        assert path[0] == ref["path"][0] and path[-1] == ref["path"][-1], (
            f"{case}: runs {path[0]} -> {path[-1]}, Dijkstra "
            f"{ref['path'][0]} -> {ref['path'][-1]}")
        got = reprice_path(path, raster, model)
        want = reprice_path(ref["path"], raster, model)
        assert got.ok, f"{case}: illegal path {got.violations}"
        tol = optimality_tolerance(want.hops, got.hops)
        assert abs(got.cost - want.cost) <= tol * want.cost + 1e-9, (
            f"{case}: cost {got.cost!r}, Dijkstra {want.cost!r}")


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("name,threads", PARAMS)
def test_predecessor_field_has_no_cycle(name, threads, field):
    records, failure = _results(name, threads)
    rec = records.get(field)
    if rec is None:
        pytest.fail(f"{field} on {name} at {threads} thread(s) produced no "
                    f"result: {failure or 'worker gave no output for it'}")
    assert "error" not in rec, rec["error"]
    assert rec["cycles"] == 0, f"{rec['cycles']} cells lie on or lead into a predecessor cycle"


# ------------------------------------------------------------------------------------------ _DeltaTree


def _tree_worker(threads: int) -> None:  # pragma: no cover - runs in a subprocess
    from pyorps.graph.search_session import _DeltaTree
    from pyorps.utils.neighborhood import get_neighborhood_steps

    n = 200
    y, x = np.mgrid[0:n, 0:n]
    raster = ((x + y) / (2 * n) * 100).astype(np.uint16)
    steps = np.asarray(get_neighborhood_steps("r2"), dtype=np.int8)
    tree = _DeltaTree(raster, steps, 65535, 1000, delta=100.0, num_threads=threads)
    field = tree.field()
    path = tree.extract_or_resume(39000)
    _emit("tree", finite=int(np.isfinite(field).sum()), cells=int(field.size),
          cycles=_cycle_cells(tree.pred()), path_ends=[int(path[0]), int(path[-1])])
    tree.close()


@pytest.mark.parametrize("threads", THREADS, ids=lambda t: f"t{t}")
def test_delta_tree_builds_on_gradient_raster(threads):
    """The full-field solve behind SearchSession / the GUI routes must not block."""
    lines, failure = _run_subprocess(["tree", str(threads)], WORKER_TIMEOUT)
    recs = [json.loads(line) for line in lines if line.startswith("{")]
    assert recs and not failure, failure
    rec = recs[-1]
    assert rec["finite"] == rec["cells"]
    assert rec["cycles"] == 0
    assert rec["path_ends"] == [1000, 39000]


# ------------------------------------------------------------------------------------------ walk bound


def _packed(dist, pred):
    """Packed kernel labels: float32 distance in the high word, predecessor low."""
    hi = np.asarray(dist, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (hi << np.uint64(32)) | np.asarray(pred, dtype=np.uint64)


def test_kernel_walk_raises_on_a_cyclic_predecessor_array():
    from pyorps.utils._delta_stepping import _walk_packed_path

    from pyorps.core.exceptions import PyorpsError, PyorpsRuntimeError

    none = 0xFFFFFFFF
    # 4 -> 3 -> 2 -> 1 -> 0 -> 1 -> ...: the shape the gradient raster produced.
    dist_pred = _packed([1.5, 1.5, 1.5, 1.5, 2.0, 0.0],
                        [1, 0, 1, 2, 3, none])
    with pytest.raises(PyorpsRuntimeError, match="cycle") as info:
        _walk_packed_path(dist_pred, 5, 4)
    assert isinstance(info.value, PyorpsError)


def test_kernel_walk_still_returns_paths_and_reports_no_path():
    from pyorps.utils._delta_stepping import _walk_packed_path

    none = 0xFFFFFFFF
    dist_pred = _packed([0.0, 1.0, 2.0, 9.0], [none, 0, 1, none])
    assert _walk_packed_path(dist_pred, 0, 2).tolist() == [0, 1, 2]
    assert _walk_packed_path(dist_pred, 0, 0).tolist() == [0]
    assert _walk_packed_path(dist_pred, 0, 3).size == 0      # unreached
    # A full-length simple path (every cell on it) is not a cycle.
    chain = _packed(np.arange(6, dtype=np.float32), [none, 0, 1, 2, 3, 4])
    assert _walk_packed_path(chain, 0, 5).tolist() == [0, 1, 2, 3, 4, 5]


def test_saved_field_walk_raises_on_a_cyclic_predecessor_field():
    from pyorps.core.exceptions import PyorpsRuntimeError
    from pyorps.graph.search_session import SavedCostField

    steps = np.array([[0, 1], [0, -1]], dtype=np.int8)
    field = object.__new__(SavedCostField)
    # cells 0 and 1 point at each other (step 1 = (0,-1): parent is cell+1;
    # step 0 = (0,1): parent is cell-1); origin 3 is never reached.
    field._pred_step = np.array([1, 0, 0, 255], dtype=np.uint8)
    field._steps = steps
    field._rows, field._cols = 1, 4
    field._dist = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    field._origin_idx = 3
    field._index = lambda point: int(point)
    with pytest.raises(PyorpsRuntimeError, match="cycle"):
        field.path_cells(2)


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    if sys.argv[1] == "worker":
        _worker(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "tree":
        _tree_worker(int(sys.argv[1 + 1]))
