"""No backend may cut the corner of a diagonal barrier.

THE INVARIANT
-------------
A step from ``(r, c)`` to ``(r+dr, c+dc)`` is admitted only if every
*intermediate* cell of that step is passable. For a single diagonal step the
intermediate set is exactly the two flanking cells ``(r+dr, c)`` and
``(r, c+dc)`` — ``_calculate_intermediate_steps_cython`` in
``pyorps/utils/_raster_context.pyx`` (the ``max(|dr|, |dc|) == 1`` branch,
line 60) pushes ``(dr, 0)`` and ``(0, dc)`` and nothing else. Every relaxation
and every edge builder in the project then rejects the step as soon as ONE of
those cells is impassable.

So pyorps does not merely refuse to squeeze through a corner where BOTH
flanking cells are blocked — it refuses the strictly larger set where EITHER
is. The corner rule is therefore not a feature to be switched on; it is an
emergent consequence of how intermediates are enumerated, and it has been in
force since the intermediate decomposition existed. What is missing is not the
rule but its pin: nothing outside this file stops a future edge-construction
change (a "fast path" for diagonals, a new kernel, a rewritten LUT) from
silently dropping it, and the failure would be silent by construction — the
route stays optimal for the graph that was built, the graph is just wrong.

WHY IT MATTERS
--------------
A barrier thinner than one cell burns as a one-cell-per-row staircase whose
cells touch only at their corners (see
``tests/test_raster/test_barrier_connectivity.py``). ``naive_eight_connected``
below — a deliberately naive flood fill that looks only at the two endpoints —
walks straight across such a staircase. pyorps does not. That difference is
the whole guarantee, and it is what these tests measure.

WHAT IS COVERED
---------------
* the three independent transcriptions of the intermediate LUT
  (Cython / CUDA-host / numba) agree, direction for direction;
* every edge builder — ``construct_edges``, ``construct_edges_3d``,
  ``get_outgoing_edges``, ``is_valid_node``, and the float32
  ``metric_edges.construct_edges_weights`` — creates no corner-cut edge, and
  more strongly creates no edge at all whose intermediates are blocked;
* the solvers — Cython Dijkstra and delta stepping, the fused delta-stepping
  kernel, networkx, networkit, the GPU raster kernels (V5/V4 and the default
  dispatch), raster_fim, and the constrained CPU planners — all fail to reach
  across a sealed diagonal barrier;
* a static guard over the CUDA sources that cannot be executed here (the
  constrained GPU planners wedge the card, see ``test_constrained_gpu_v4``).

Every no-path assertion is paired with a POSITIVE CONTROL that punches a
single cell out of the barrier and requires the same call to succeed, so none
of them can pass because the harness was broken.

``ignore_max=True`` is pinned explicitly everywhere: with ``ignore_max=False``
the exclude mask is all-ones (``_raster_context.pyx:357``,
``_traversal.pyx:561``) and there is no impassability for the rule to act on,
so these tests would pass vacuously.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from pyorps.core.exceptions import NoPathFoundError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.utils import _traversal
from pyorps.utils.metric_edges import (
    _intermediate_offsets,
    construct_edges_weights,
)
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.utils.traversal_gpu import _intermediate_steps_cpu

try:
    import cupy  # noqa: F401
    GPU_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the machine
    GPU_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Small enough that r3 edge construction and three GPU launches stay in the
#: millisecond range, large enough that the barrier poses the question many
#: times over (see test_the_fixture_poses_the_question).
N = 24
FREE = 10

NEIGHBORHOODS = ("r1", "r2", "r3")

#: Opposite corners of the grid, one on each side of the main diagonal.
SRC_RC = (0, N - 1)
DST_RC = (N - 1, 0)
SRC = SRC_RC[0] * N + SRC_RC[1]
DST = DST_RC[0] * N + DST_RC[1]

#: The one cell removed from the barrier by the positive controls. Its own two
#: flanking cells are free, so a route may legitimately pass through it.
GAP_RC = (N // 2, N // 2)


def steps_for(neighborhood):
    return get_neighborhood_steps(neighborhood).astype(np.int8)


def barrier_raster(gap=False):
    """A grid split in two by an impassable MAIN DIAGONAL.

    The barrier is one cell wide, so the two triangles touch each other only
    at cell corners: for every ``i``, the free cells ``(i, i+1)`` and
    ``(i+1, i)`` are diagonal neighbours whose two flanking cells ``(i, i)``
    and ``(i+1, i+1)`` are both impassable. That is the corner cut, posed in
    its purest form and repeated the length of the grid.

    With ``gap=True`` one barrier cell is freed — the positive control.
    """
    raster = np.full((N, N), FREE, dtype=np.uint16)
    for i in range(N):
        raster[i, i] = IMPASSABLE_CELL_COST
    if gap:
        raster[GAP_RC] = FREE
    return raster


def corner_cut_pairs(raster):
    """Every ``((r, c), (tr, tc))`` a corner-cutting backend could exploit.

    Both cells free, diagonal neighbours, and BOTH flanking cells impassable —
    i.e. the pairs that only a router ignoring intermediates can connect.
    """
    free = raster != IMPASSABLE_CELL_COST
    pairs = []
    for r in range(N):
        for c in range(N):
            if not free[r, c]:
                continue
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                tr, tc = r + dr, c + dc
                if not (0 <= tr < N and 0 <= tc < N) or not free[tr, tc]:
                    continue
                if not free[tr, c] and not free[r, tc]:
                    pairs.append(((r, c), (tr, tc)))
    return pairs


def naive_eight_connected(raster, start, goal):
    """Can a NAIVE 8-connected walker get from *start* to *goal*?

    Deliberately naive: a diagonal is admitted whenever both ENDPOINTS are
    free, with no look at the flanking cells. Kept as the reference that makes
    the guarantee visible — it is what pyorps would do if the intermediate
    check were ever dropped. Do not repoint it at anything in pyorps.
    """
    free = raster != IMPASSABLE_CELL_COST
    seen = np.zeros_like(free)
    seen[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < N and 0 <= nc < N and free[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return False


def edge_set(from_nodes, to_nodes):
    return set(zip(from_nodes.tolist(), to_nodes.tolist()))


def assert_no_corner_cut(edges, raster, what):
    """No edge joins a corner-cut pair, and no edge has a blocked intermediate.

    The second assertion is the stronger one and the one that actually pins
    the mechanism: pyorps rejects a step when EITHER flanking cell is blocked.
    """
    offending = [p for p in corner_cut_pairs(raster)
                 if (p[0][0] * N + p[0][1], p[1][0] * N + p[1][1]) in edges]
    assert not offending, (
        f"{what} created {len(offending)} corner-cutting edges, "
        f"e.g. {offending[0]}")

    free = raster != IMPASSABLE_CELL_COST
    for a, b in edges:
        ar, ac = divmod(int(a), N)
        br, bc = divmod(int(b), N)
        for ir, ic in _traversal.intermediate_steps_numba(np.int8(br - ar),
                                                          np.int8(bc - ac)):
            assert free[ar + ir, ac + ic], (
                f"{what} created edge {(ar, ac)}->{(br, bc)} whose "
                f"intermediate {(ar + ir, ac + ic)} is impassable")


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------

def test_the_fixture_poses_the_question():
    """Guard: the barrier really does offer corner cuts to take."""
    raster = barrier_raster()
    pairs = corner_cut_pairs(raster)
    # 2*(N-1) ordered pairs across the diagonal, minus none - both directions
    # of each of the N-1 touching corners.
    assert len(pairs) == 2 * (N - 1) == 46
    assert naive_eight_connected(raster, SRC_RC, DST_RC), (
        "the naive walker must cross - it is the contrast that gives the "
        "pyorps assertions below their meaning")
    assert raster[SRC_RC] != IMPASSABLE_CELL_COST
    assert raster[DST_RC] != IMPASSABLE_CELL_COST


def test_positive_control_fixture_is_actually_connected():
    """Guard: with one cell freed even the strict rule can get across."""
    assert naive_eight_connected(barrier_raster(gap=True), SRC_RC, DST_RC)


# ---------------------------------------------------------------------------
# The three transcriptions of the intermediate LUT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_intermediate_luts_agree(neighborhood):
    """Cython, CUDA-host and numba must enumerate the same intermediates.

    They agree today only because three separate transcriptions were kept in
    step by hand; a divergence would split the CPU and GPU backends without
    any test noticing.
    """
    for dr, dc in steps_for(neighborhood):
        cython = np.asarray(
            _traversal.intermediate_steps_numba(np.int8(dr), np.int8(dc)),
            dtype=np.int8)
        gpu_host = np.asarray(_intermediate_steps_cpu(int(dr), int(dc)),
                              dtype=np.int8)
        numba = np.asarray(_intermediate_offsets(int(dr), int(dc)),
                           dtype=np.int8)
        assert np.array_equal(cython, gpu_host), (dr, dc, cython, gpu_host)
        assert np.array_equal(cython, numba), (dr, dc, cython, numba)


@pytest.mark.parametrize("dr,dc", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_a_diagonal_decomposes_into_exactly_its_two_flanking_cells(dr, dc):
    """THE mechanism, stated as an assertion.

    This is the whole of option B: because a diagonal's intermediates are the
    two flanking cells, the generic "all intermediates must be passable" test
    that every backend already runs *is* a corner rule.
    """
    got = np.asarray(
        _traversal.intermediate_steps_numba(np.int8(dr), np.int8(dc)),
        dtype=np.int8)
    assert {tuple(p) for p in got} == {(dr, 0), (0, dc)}
    assert len(got) == 2


@pytest.mark.parametrize("dr,dc", [(0, 1), (1, 0), (0, -1), (-1, 0)])
def test_a_cardinal_step_has_no_intermediates(dr, dc):
    """Cardinal steps cost nothing to validate - there is no corner to cut."""
    got = _traversal.intermediate_steps_numba(np.int8(dr), np.int8(dc))
    assert len(got) == 0


# ---------------------------------------------------------------------------
# Edge builders (the library backends can only route on what these emit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_construct_edges_creates_no_corner_cut(neighborhood):
    raster = barrier_raster()
    steps = steps_for(neighborhood)
    from_nodes, to_nodes = _traversal.construct_edges(
        raster, steps, ignore_max=True)[:2]
    edges = edge_set(from_nodes, to_nodes)
    assert edges, "positive control: the builder must emit SOME edges"
    assert_no_corner_cut(edges, raster, "construct_edges")


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_construct_edges_3d_creates_no_corner_cut(neighborhood):
    """The DEM builder inlines the rule instead of calling _is_valid_node.

    A second, independent copy (``_traversal.pyx:_find_valid_nodes_3d``), so a
    DEM run could in principle diverge from a 2D run. A flat DEM isolates the
    admission rule from the gradient penalty.
    """
    raster = barrier_raster()
    dem = np.zeros(raster.shape, dtype=np.float32)
    from_nodes, to_nodes = _traversal.construct_edges_3d(
        raster, dem, steps_for(neighborhood), True)[:2]
    edges = edge_set(from_nodes, to_nodes)
    assert edges
    assert_no_corner_cut(edges, raster, "construct_edges_3d")


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_metric_edges_float_builder_creates_no_corner_cut(neighborhood):
    """The float32 builder calls forbidden ``non-finite``, not 65535.

    A fourth transcription of the rule (numba, ``metric_edges.py``) over a
    different representation of impassability - the most likely to drift.
    """
    raster = barrier_raster()
    weights = raster.astype(np.float32)
    weights[raster == IMPASSABLE_CELL_COST] = np.inf
    from_nodes, to_nodes = construct_edges_weights(
        weights, steps_for(neighborhood), True)[:2]
    edges = edge_set(from_nodes, to_nodes)
    assert edges
    assert_no_corner_cut(edges, raster, "construct_edges_weights")


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_get_outgoing_edges_creates_no_corner_cut(neighborhood):
    """Public API, used by no solver - but it must not disagree with them."""
    raster = barrier_raster()
    steps = steps_for(neighborhood)
    for (r, c), (tr, tc) in corner_cut_pairs(raster):
        to_nodes, _ = _traversal.get_outgoing_edges(
            r * N + c, raster, steps, N, N)
        assert (tr * N + tc) not in set(to_nodes.tolist()), (
            f"get_outgoing_edges offered the corner cut "
            f"{(r, c)}->{(tr, tc)}")


def test_is_valid_node_rejects_the_corner_cut():
    """The rule at its smallest callable unit, both flanks and one flank."""
    raster = barrier_raster()
    exclude = (raster != IMPASSABLE_CELL_COST).astype(np.uint8)
    out_cost = np.zeros(1, dtype=np.float64)

    # Both flanks blocked - the corner cut proper.
    inter = _traversal.intermediate_steps_numba(np.int8(1), np.int8(-1))
    assert not _traversal.is_valid_node(3, 4, 4, 3, exclude, inter, raster,
                                        N, N, out_cost)

    # ONE flank blocked: (3,4)->(4,5) grazes the barrier cell (4,4) while its
    # other flank (3,5) is free. pyorps is stricter than the corner rule asks.
    inter = _traversal.intermediate_steps_numba(np.int8(1), np.int8(1))
    assert not _traversal.is_valid_node(3, 4, 4, 5, exclude, inter, raster,
                                        N, N, out_cost), (
        "a diagonal grazing a single impassable flank must also be rejected")

    # Positive control: a diagonal clear of the barrier is admitted.
    assert _traversal.is_valid_node(2, 6, 3, 7, exclude, inter, raster,
                                    N, N, out_cost)


def test_ignore_max_false_removes_the_impassability_the_rule_acts_on():
    """Pins WHY every test here passes ignore_max=True.

    With ignore_max=False the exclude mask is all-ones and 65535 is just an
    expensive cell, so a corner-cut assertion would hold vacuously. Asserting
    the opposite outcome here keeps that from being mistaken for the rule.
    """
    raster = barrier_raster()
    steps = steps_for("r1")
    from_nodes, to_nodes = _traversal.construct_edges(
        raster, steps, ignore_max=False)[:2]
    edges = edge_set(from_nodes, to_nodes)
    crossings = [p for p in corner_cut_pairs(raster)
                 if (p[0][0] * N + p[0][1], p[1][0] * N + p[1][1]) in edges]
    assert crossings, (
        "with ignore_max=False nothing is impassable, so the barrier cells "
        "are traversable and the diagonal edges exist - if this ever stops "
        "being true the ignore_max semantics changed, not the corner rule")


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_two_diagonally_touching_cells_are_not_connected(neighborhood):
    """The atomic case: an 8x8 grid whose ONLY passable cells are diagonal.

    Nothing else can explain a missing edge here - no bounds, no cost, no
    search order. If the graph is empty, the corner rule is doing it.
    """
    raster = np.full((8, 8), IMPASSABLE_CELL_COST, dtype=np.uint16)
    raster[3, 3] = FREE
    raster[4, 4] = FREE
    from_nodes = _traversal.construct_edges(
        raster, steps_for(neighborhood), ignore_max=True)[0]
    assert len(from_nodes) == 0


# ---------------------------------------------------------------------------
# Solvers: nothing may route across the sealed barrier
# ---------------------------------------------------------------------------

def _cython_path(raster, neighborhood, algorithm):
    from pyorps.graph.api.cython_api import CythonAPI
    api = CythonAPI(raster, steps_for(neighborhood), ignore_max=True)
    return api.shortest_path(SRC, DST, algorithm=algorithm)


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
@pytest.mark.parametrize("algorithm", ["dijkstra", "delta-stepping"])
def test_cython_backend_cannot_cross(neighborhood, algorithm):
    assert len(_cython_path(barrier_raster(), neighborhood, algorithm)) == 0
    # Positive control - the same call succeeds once one cell is freed.
    assert len(_cython_path(barrier_raster(gap=True), neighborhood,
                            algorithm)) > 0


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_fused_delta_stepping_cannot_cross(neighborhood):
    """Not wired into path_algorithms - pinned so it cannot drift.

    ``_delta_stepping_fused`` is reachable only from its own tests today, but
    it is the exact parallel alternative earmarked to replace the production
    kernel; it must already agree.
    """
    from pyorps.utils._delta_stepping_fused import delta_stepping_2d_fused
    steps = steps_for(neighborhood)
    path = delta_stepping_2d_fused(barrier_raster(), steps, SRC, DST,
                                   100.0, IMPASSABLE_CELL_COST)
    assert len(path) == 0
    path = delta_stepping_2d_fused(barrier_raster(gap=True), steps, SRC, DST,
                                   100.0, IMPASSABLE_CELL_COST)
    assert len(path) > 0


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
@pytest.mark.parametrize("library", ["networkx", "networkit"])
def test_graph_library_backends_cannot_cross(neighborhood, library):
    """The library backends never see the raster - only the edge list.

    Whatever algorithm the library runs, it cannot cut a corner because no
    such edge was ever handed to it. The rule is enforced at construction.
    """
    pytest.importorskip(library)
    module = pytest.importorskip(f"pyorps.graph.api.{library}_api")
    api_cls = getattr(module, "NetworkxAPI" if library == "networkx"
                      else "NetworkitAPI")
    steps = steps_for(neighborhood)
    with pytest.raises(NoPathFoundError):
        api_cls(barrier_raster(), steps, ignore_max=True).shortest_path(
            SRC, DST)
    assert api_cls(barrier_raster(gap=True), steps,
                   ignore_max=True).shortest_path(SRC, DST)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_raster_gpu_backend_cannot_cross(neighborhood):
    """The GPU has no exclude mask - it compares the raster against max_cost.

    A different representation of impassability reaching the same verdict is
    the point of testing it separately from the Cython path.
    """
    from pyorps.graph.api.raster_gpu_api import RasterGPUAPI
    steps = steps_for(neighborhood)
    with pytest.raises(NoPathFoundError):
        RasterGPUAPI(barrier_raster(), steps,
                     ignore_max=True).shortest_path(SRC, DST)
    assert RasterGPUAPI(barrier_raster(gap=True), steps,
                        ignore_max=True).shortest_path(SRC, DST)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
@pytest.mark.parametrize("kernel", ["dispatch", "v4", "v5"])
def test_gpu_kernel_variants_leave_the_far_side_unreached(kernel):
    """Every GPU kernel generation, at the distance-array level.

    ``sssp_raster_gpu`` dispatches V5 -> V4 -> V3, so calling the two current
    generations explicitly is what stops a regression in one of them from
    being hidden by the fallback.
    """
    from pyorps.utils import sssp_gpu
    fn = {"dispatch": sssp_gpu.sssp_raster_gpu,
          "v4": sssp_gpu.sssp_raster_gpu_v4,
          "v5": sssp_gpu.sssp_raster_gpu_v5}[kernel]
    steps = steps_for("r1")

    result = fn(barrier_raster(), steps, SRC, ignore_max=True)
    dist = result[0] if isinstance(result, tuple) else result
    assert not np.isfinite(dist[DST]), (
        f"{kernel} reached the far side of a sealed diagonal barrier")

    result = fn(barrier_raster(gap=True), steps, SRC, ignore_max=True)
    dist = result[0] if isinstance(result, tuple) else result
    assert np.isfinite(dist[DST]), "positive control failed"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_raster_fim_cannot_cross():
    """The eikonal backend has no diagonal step at all - and needs none.

    ``eikonal_gpu`` solves a 4-point Godunov update, so there is no edge for a
    corner rule to attach to; a corner-touching staircase 4-separates the free
    cells and the wavefront simply never arrives. It is STRICTER than the
    corner rule by construction, which is why it belongs here even though the
    rule is not expressible in it.

    NOTE for anyone extending this file: raster_fim's *output* cell list may
    contain a diagonal pair with blocked flanks, because
    ``polyline_to_cells`` drops samples that land on forbidden cells (its own
    docstring says so). Assert reachability here, never the shape of the
    returned cells.
    """
    from pyorps.graph.api.raster_fim_api import RasterFIMAPI
    steps = steps_for("r1")
    with pytest.raises(NoPathFoundError):
        RasterFIMAPI(barrier_raster(), steps,
                     ignore_max=True).shortest_path(SRC, DST)
    assert RasterFIMAPI(barrier_raster(gap=True), steps,
                        ignore_max=True).shortest_path(SRC, DST)


# ---------------------------------------------------------------------------
# Constrained planner (extended state: cell x direction x span bin)
# ---------------------------------------------------------------------------

def _constrained_kwargs(steps):
    """Neutral turn/tower model: only the terrain and the corner rule matter."""
    n_dirs = len(steps)
    return dict(
        angle_cost_lut=np.zeros((n_dirs, n_dirs), dtype=np.float32),
        angle_valid_lut=np.ones((n_dirs, n_dirs), dtype=np.uint8),
        step_distances=(np.hypot(steps[:, 0].astype(np.float32),
                                 steps[:, 1].astype(np.float32))
                        * 10.0).astype(np.float32),
        tower_terrain_costs=np.full(65536, 5.0, dtype=np.float32),
        tower_angle_costs=np.zeros((n_dirs, n_dirs), dtype=np.float32),
        n_span_bins=10, span_bin_size=20.0, min_span=10.0, max_span=200.0,
    )


@pytest.mark.parametrize("planner", ["dijkstra", "delta"])
def test_constrained_cpu_planners_cannot_cross(planner):
    """The constrained kernels precompute the rule into ``icache_status``.

    They evaluate it once per (cell, incoming direction) rather than per
    relaxation (``_constrained_context.pyx:_precompute_intermediate_cache``),
    so this is a genuinely different code path to the same invariant - and
    the one with five duplicated relaxation sites behind it.
    """
    if planner == "dijkstra":
        from pyorps.utils._constrained_dijkstra import (
            constrained_dijkstra_2d as solve,
        )
    else:
        from pyorps.utils._constrained_delta import (
            constrained_delta_stepping_2d as solve,
        )
    steps = steps_for("r1")
    kwargs = _constrained_kwargs(steps)

    path, _ = solve(barrier_raster(), SRC_RC[0], SRC_RC[1], DST_RC[0],
                    DST_RC[1], steps, **kwargs)
    assert len(path) == 0

    path, _ = solve(barrier_raster(gap=True), SRC_RC[0], SRC_RC[1],
                    DST_RC[0], DST_RC[1], steps, **kwargs)
    assert len(path) > 0, "positive control failed"


# ---------------------------------------------------------------------------
# The CUDA sources that cannot be executed here
# ---------------------------------------------------------------------------

#: ``for (int k = 0; k < ni; k++)`` / ``< n_inter`` - the intermediate
#: rejection loop, in the only two spellings the kernels use.
_INTERMEDIATE_LOOP = re.compile(
    r"for\s*\(\s*int\s+k\s*=\s*0;\s*k\s*<\s*(?:ni|n_inter)\b")

#: Audited 2026-08-11. Each count is the number of neighbour-relaxation sites
#: in that file, every one of which rejects a step whose intermediates are
#: blocked. The constrained GPU planners cannot be run here (they pin the card
#: and the CUDA context can outlive a killed process - see
#: test_constrained_gpu_v4), so this static count is the only guard they have.
#: If you ADD a kernel, verify by hand that it walks the intermediate LUT and
#: then bump the number; if a count DROPS, a corner rule was deleted.
#: relax_constrained_v3.cu (2 sites) was removed with the constrained GPU v3
#: backend on 2026-08-11 — it returned an empty path on an obstacle-free
#: raster and v4 does not build on it. Dropping the entry is correct here; a
#: count that drops for any OTHER reason is still a deleted corner rule.
_CUDA_INTERMEDIATE_SITES = {
    "pyorps/utils/sssp_gpu.py": 9,
    "pyorps/utils/constrained_sssp_gpu.py": 1,
    "pyorps/utils/kernels/constrained_persistent.cu": 4,
    "pyorps/utils/kernels/adds_wtb.cuh": 1,
    "pyorps/utils/kernels/adds_tower.cuh": 1,
}


@pytest.mark.parametrize("relpath,expected",
                         sorted(_CUDA_INTERMEDIATE_SITES.items()))
def test_cuda_kernels_still_walk_the_intermediate_lut(relpath, expected):
    if not (PROJECT_ROOT / relpath).exists():
        pytest.skip("source tree not available (installed-wheel test run)")
    source = (PROJECT_ROOT / relpath).read_text(encoding="utf-8")
    found = len(_INTERMEDIATE_LOOP.findall(source))
    assert found == expected, (
        f"{relpath} has {found} intermediate-rejection loops, expected "
        f"{expected}. A drop means a kernel now admits diagonal steps "
        f"without checking their flanking cells; a rise means a new kernel "
        f"needs the same hand check before this number is raised.")
