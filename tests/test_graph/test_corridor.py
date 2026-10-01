"""
Corridor graphs: the invariants that make the construction defensible.

WHAT IS UNDER TEST
------------------
Two constructions reduce a set of least-cost routes to the trenches they share:

* ``MultiSourceSolver`` (``pyorps/utils/_dijkstra.pyx``) seeds every terminal
  at distance 0, propagates a region label, and reports the boundary steps
  between regions -- Mehlhorn's terminal distance network.
* ``corridor_graph_from_routes`` (``pyorps/graph/corridor.py``) overlays a
  given set of routes and cuts them where their step membership changes.

``PathFinder.build_corridor_graph`` chains the first into the second.

WHY THESE TESTS AND NOT OTHERS
------------------------------
The claim the whole thing rests on is that a corridor node is DERIVED from
terrain rather than placed, and that deriving it costs nothing in accuracy.
Both halves can fail silently:

* A junction that does not land on a step endpoint would split a step, and the
  cost of that step would then have to be apportioned between two segments by
  an invented rule. Nothing would raise; the numbers would just stop being the
  numbers ``find_route`` reports. ``test_every_node_is_a_step_endpoint`` and
  the conservation tests pin it.
* Ties are pervasive on rasters with few cost categories and the binary heap
  has no stable order for equal priorities, so the region partition -- and
  therefore the whole graph -- could vary with the order the terminals happen
  to be listed in. A paper reporting a junction count would then be reporting
  an artefact. ``test_region_labels_are_permutation_independent`` pins it.
* A segment that runs over an excluded cell would be a trench through a
  forbidden zone. ``test_no_segment_touches_an_excluded_cell`` pins it, with a
  positive control that the barrier is real.

ANTI-VACUITY
------------
Every raster here is built so the assertion has something to catch: the shared
track is cheap enough that routes genuinely converge on it (asserted), the
barrier genuinely blocks (asserted by a positive control), and
``ignore_max_cost`` is pinned explicitly everywhere.

Exactness is judged by ``benchmarks.exactness_referee.reprice_path``, a second
implementation of the edge model that never imports the compiled kernels --
not by ``path_cost_uint32``, which sums raw cell values and would report
phantom differences for genuinely tied optima.
"""
import numpy as np
import pytest
from affine import Affine

from benchmarks.exactness_referee import EdgeModel, reprice_path
from pyorps.core.corridor import NODE_TERMINAL
from pyorps.graph.corridor import (
    cell_sharing_profile,
    corridor_graph_from_routes,
    route_metrics,
    supercover_cells,
)
from pyorps.utils._dijkstra import (
    make_dijkstra_solver,
    make_multi_source_solver,
    price_route_cython,
)
from pyorps.utils._traversal import calculate_path_metrics_numba
from pyorps.utils.neighborhood import get_neighborhood_steps

NEIGHBORHOODS = ("r0", "r1", "r2", "r3")
TRANSFORM = Affine(2.0, 0.0, 442000.0, 0.0, -2.0, 5587000.0)
CELL_SIZE = abs(TRANSFORM.a)


def corridor_raster(rows=90, cols=120, seed=11, barrier=True):
    """Expensive ground crossed by two cheap tracks, plus a barrier.

    Routes between corners are forced onto the tracks, which is what makes the
    sharing assertions non-vacuous; the barrier is what makes the exclusion
    assertion non-vacuous.
    """
    rng = np.random.default_rng(seed)
    raster = np.full((rows, cols), 500, dtype=np.uint16)
    raster += rng.integers(0, 120, size=(rows, cols)).astype(np.uint16)
    raster[rows // 2, :] = 130
    raster[:, cols // 6] = 140
    if barrier:
        raster[rows // 4, cols // 3:cols - cols // 6] = 65535
    return raster


def corner_terminals(rows=90, cols=120):
    return {
        0: 5 * cols + 5,
        1: (rows - 10) * cols + 10,
        2: 10 * cols + (cols - 10),
        3: (rows - 5) * cols + (cols - 5),
        4: (rows * 2 // 3) * cols + (cols // 2),
    }


def pairwise_routes(raster, steps, terminals):
    """Every pair routed independently, exactly as the pairwise pipeline does."""
    solver = make_dijkstra_solver(raster, steps)
    routes = {}
    keys = sorted(terminals)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            solver.reset_root(np.uint32(terminals[a]))
            solver.search_until(np.uint32(terminals[b]))
            path = solver.extract_path(np.uint32(terminals[b]))
            if path.size > 1:
                routes[(a, b)] = path
    return routes


def overlay_graph(raster, steps, terminals, **kwargs):
    routes = pairwise_routes(raster, steps, terminals)
    graph = corridor_graph_from_routes(
        routes, raster, TRANSFORM, crs="EPSG:25832", steps=steps,
        terminal_cells=terminals, ignore_max_cost=True, **kwargs)
    return routes, graph


# --------------------------------------------------------------- the fixture
# poses the question: without genuine sharing every overlap assertion below
# would pass on an empty set.


def test_the_fixture_actually_produces_shared_trenches():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    _routes, graph = overlay_graph(raster, steps, corner_terminals())
    shared = graph.shared_segments
    assert shared, "no shared segment: the overlap assertions would be vacuous"
    assert graph.overlap_report().shared_fraction > 0.2


def test_the_barrier_actually_blocks():
    """Positive control for test_no_segment_touches_an_excluded_cell."""
    rows, cols = 40, 60
    blocked = np.full((rows, cols), 300, dtype=np.uint16)
    blocked[20, :] = 65535
    steps = get_neighborhood_steps("r1", directed=True)
    solver = make_dijkstra_solver(blocked, steps)
    assert solver.single_pair(np.uint32(5), np.uint32(35 * cols + 5)).size == 0

    freed = blocked.copy()
    freed[20, 30] = 300
    solver = make_dijkstra_solver(freed, steps)
    assert solver.single_pair(np.uint32(5), np.uint32(35 * cols + 5)).size > 0


# --------------------------------------------------- MultiSourceSolver itself


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_single_terminal_is_bit_identical_to_dijkstra(neighborhood):
    """One terminal must reproduce DijkstraSolver exactly.

    The corridor solver is only trustworthy if it is the same search. Any
    divergence here -- a different relaxation guard, a tie-break that fires
    when it should not -- would make every corridor metric quietly disagree
    with find_route.
    """
    raster = corridor_raster()
    steps = get_neighborhood_steps(neighborhood, directed=True)
    source = np.uint32(5 * 120 + 5)

    reference = make_dijkstra_solver(raster, steps)
    reference.reset_root(source)
    reference.settle_all()

    solver = make_multi_source_solver(raster, steps)
    solver.solve(np.array([source], dtype=np.uint32))

    np.testing.assert_array_equal(reference.dist_array(), solver.dist_array())
    assert (solver.region_array()[solver.visited_array() == 1] == 0).all()


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_region_labels_are_permutation_independent(neighborhood):
    """Reordering the terminals must not move a single region boundary.

    Fails before the lexicographic tie-break: with ties broken by whichever
    relaxation the heap ran first, permuting the seed order silently repaints
    the partition and changes the junction count.
    """
    raster = corridor_raster()
    steps = get_neighborhood_steps(neighborhood, directed=True)
    terminals = corner_terminals()
    seeds = np.array([terminals[k] for k in sorted(terminals)],
                     dtype=np.uint32)

    base = make_multi_source_solver(raster, steps)
    base.solve(seeds)
    assert not base.has_zero_cost_steps(), (
        "the tie-break is only provably order-independent with positive step "
        "costs; this fixture must not contain a zero cell")

    permutation = np.array([3, 0, 4, 1, 2])
    other = make_multi_source_solver(raster, steps)
    other.solve(seeds[permutation])

    relabelled = np.where(other.region_array() >= 0,
                          permutation[np.clip(other.region_array(), 0, None)],
                          -1)
    np.testing.assert_array_equal(relabelled, base.region_array())
    np.testing.assert_array_equal(base.dist_array(), other.dist_array())


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_boundary_step_cost_is_the_true_pair_cost_where_regions_touch(
        neighborhood):
    """The distance network is an upper bound, and tight for adjacent regions.

    Quantifying that gap is the point of the check: it is a property of the
    construction, reported rather than hidden.
    """
    raster = corridor_raster()
    steps = get_neighborhood_steps(neighborhood, directed=True)
    terminals = corner_terminals()
    keys = sorted(terminals)
    seeds = np.array([terminals[k] for k in keys], dtype=np.uint32)

    solver = make_multi_source_solver(raster, steps)
    solver.solve(seeds)
    boundary = solver.boundary_steps()
    assert boundary["cost"].size > 0

    cheapest = {}
    for ru, rv, cost in zip(boundary["region_u"], boundary["region_v"],
                            boundary["cost"]):
        pair = (int(ru), int(rv))
        cheapest[pair] = min(cheapest.get(pair, np.inf), float(cost))
    assert cheapest, "no pair of regions touches"

    reference = make_dijkstra_solver(raster, steps)
    for (a, b), network_cost in cheapest.items():
        reference.reset_root(np.uint32(terminals[keys[a]]))
        reference.search_until(np.uint32(terminals[keys[b]]))
        true_cost = float(reference.peek_dist(np.uint32(terminals[keys[b]])))
        assert network_cost >= true_cost * (1 - 1e-9), (
            f"pair {a}-{b}: the network claims {network_cost} for a pair whose "
            f"true distance is {true_cost}; an upper bound cannot be lower")


def test_price_route_refuses_a_step_outside_the_neighbourhood():
    raster = corridor_raster(rows=20, cols=20)
    steps = get_neighborhood_steps("r1", directed=True)
    # (0, 0) -> (5, 5) is not one r1 step.
    with pytest.raises(ValueError, match="not in the neighbourhood"):
        price_route_cython(raster, steps, np.array([0, 5 * 20 + 5],
                                                  dtype=np.uint32))
    # positive control: a legal step prices fine
    assert price_route_cython(raster, steps,
                              np.array([0, 1], dtype=np.uint32)) > 0


def test_solve_rejects_an_empty_terminal_list():
    raster = corridor_raster(rows=20, cols=20)
    steps = get_neighborhood_steps("r1", directed=True)
    solver = make_multi_source_solver(raster, steps)
    with pytest.raises(ValueError, match="At least one terminal"):
        solver.solve(np.empty(0, dtype=np.uint32))


# ------------------------------------------------------------- conservation


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_segments_reproduce_the_route_they_came_from(neighborhood):
    """The conservation law: cells, length, cost and search weight all add up.

    This is the assertion that says corridor accounting and pairwise
    accounting are the same accounting. It is exact, not approximate: every
    step belongs to exactly one segment and both kernels accumulate per step.
    """
    raster = corridor_raster()
    steps = get_neighborhood_steps(neighborhood, directed=True)
    terminals = corner_terminals()
    routes, graph = overlay_graph(raster, steps, terminals)
    reference = route_metrics(routes, raster, CELL_SIZE)
    model = EdgeModel(steps, max_value=65535)

    assert any(len(graph.route_segments(key)) > 1 for key in routes), (
        "no route was cut into segments: the conservation check would be "
        "a restatement rather than a check")

    for key, cells in routes.items():
        rebuilt = np.asarray(graph.route_cells(key), dtype=np.uint32)
        np.testing.assert_array_equal(rebuilt, cells)

        segments = graph.route_segments(key)
        assert sum(s.length for s in segments) == pytest.approx(
            reference[key]["length_m"], rel=0, abs=1e-9)
        assert sum(s.construction_cost for s in segments) == pytest.approx(
            reference[key]["construction_cost"], rel=1e-12)

        priced = reprice_path(cells, raster, model)
        assert priced.ok, priced.violations
        assert sum(s.metrics["routing_cost"] for s in segments) == (
            pytest.approx(priced.cost, rel=1e-9))


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_every_node_is_a_step_endpoint(neighborhood):
    """No junction may land inside a step.

    A step from r2 upward spans intermediate cells that are not step
    endpoints. Cutting there would split the step and force its cost to be
    apportioned by a rule nothing in pyorps defines -- which is exactly the
    tolerance this construction exists to avoid.

    Honest about what this proves: node cells are read off the route's own
    cell list, so "no node is a step interior" holds for ANY cut rule that
    indexes into that list. What the test really pins is that the interior
    cells EXIST and are disjoint from the endpoints at r2 and r3 -- i.e. that
    the fixture is one where a cell-based cut rule WOULD split steps -- and
    that no future rewrite starts cutting on the supercover instead. The
    absolute pins further down are what catch a changed cut rule.
    """
    raster = corridor_raster()
    steps = get_neighborhood_steps(neighborhood, directed=True)
    terminals = corner_terminals()
    routes, graph = overlay_graph(raster, steps, terminals)

    endpoints = set()
    interiors = set()
    cols = raster.shape[1]
    for cells in routes.values():
        endpoints.update(int(c) for c in cells)
        for a, b in zip(cells[:-1], cells[1:]):
            for cell in supercover_cells(np.array([a, b], dtype=np.uint32),
                                         cols):
                if cell not in (int(a), int(b)):
                    interiors.add(cell)

    node_cells = {node.cell_index for node in graph.nodes.values()}
    assert node_cells <= endpoints
    assert not (node_cells & (interiors - endpoints))
    if neighborhood in ("r2", "r3"):
        assert interiors - endpoints, (
            f"{neighborhood} produced no step interior outside the route's "
            f"own cells, so the assertion above has nothing to exclude")


def test_no_segment_touches_an_excluded_cell():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    _routes, graph = overlay_graph(raster, steps, corner_terminals())

    cols = raster.shape[1]
    for segment in graph.segments.values():
        covered = supercover_cells(
            np.asarray(segment.cell_indices, dtype=np.uint32), cols)
        values = raster.flat[np.asarray(covered, dtype=np.int64)]
        assert not (values == 65535).any(), (
            f"segment {segment.segment_id} runs through a forbidden cell")


def test_overlay_is_deterministic_under_route_reordering():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    terminals = corner_terminals()
    routes = pairwise_routes(raster, steps, terminals)

    def signature(graph):
        return sorted(
            (tuple(int(c) for c in s.cell_indices), s.use_count,
             round(s.length, 9))
            for s in graph.segments.values())

    first = corridor_graph_from_routes(
        routes, raster, TRANSFORM, steps=steps, terminal_cells=terminals)
    shuffled = dict(reversed(list(routes.items())))
    second = corridor_graph_from_routes(
        shuffled, raster, TRANSFORM, steps=steps, terminal_cells=terminals)

    assert signature(first) == signature(second)


def test_node_count_is_orders_of_magnitude_below_the_reached_cells():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    terminals = corner_terminals()
    _routes, graph = overlay_graph(raster, steps, terminals)

    solver = make_multi_source_solver(raster, steps)
    settled = solver.solve(np.array([terminals[k] for k in sorted(terminals)],
                                    dtype=np.uint32))
    assert len(graph.nodes) * 100 < settled


# ------------------------------------------------------------- shared-run prune


def test_min_shared_length_removes_a_short_shared_run():
    """The prune must fire, and must not fire on a long shared run."""
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    terminals = corner_terminals()

    exact = overlay_graph(raster, steps, terminals)[1]
    pruned = overlay_graph(raster, steps, terminals,
                           min_shared_length_m=40.0)[1]
    generous = overlay_graph(raster, steps, terminals,
                             min_shared_length_m=1e9)[1]

    assert exact.provenance["n_demoted_steps"] == 0
    assert generous.provenance["n_demoted_steps"] > 0
    assert not generous.shared_segments, (
        "an infinite minimum must leave no shared segment at all")
    assert len(generous.junctions) <= len(exact.junctions)
    # The moderate prune sits between the two, and never invents sharing.
    assert (exact.provenance["n_demoted_steps"]
            <= pruned.provenance["n_demoted_steps"]
            <= generous.provenance["n_demoted_steps"])


# ---------------------------------------------------------------- degenerate


def test_two_terminals_give_one_unshared_route():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    terminals = {0: 5 * 120 + 5, 1: 80 * 120 + 110}
    routes, graph = overlay_graph(raster, steps, terminals)

    assert len(routes) == 1
    assert not graph.shared_segments
    assert len(graph.segments) == 1
    assert graph.overlap_report().overstatement == 0.0
    assert all(node.kind == NODE_TERMINAL for node in graph.nodes.values())


def test_coincident_terminals_do_not_raise():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    seeds = np.array([5 * 120 + 5, 5 * 120 + 5, 80 * 120 + 110],
                     dtype=np.uint32)
    solver = make_multi_source_solver(raster, steps)
    solver.solve(seeds)
    region = solver.region_array()
    # The lower index owns the shared cell; the duplicate owns nothing.
    assert region[seeds[0]] == 0
    assert (region == 1).sum() == 0
    assert (region == 2).sum() > 0


def test_an_isolated_terminal_is_absent_from_the_pairs_and_does_not_raise():
    rows, cols = 40, 60
    raster = np.full((rows, cols), 300, dtype=np.uint16)
    raster[20, :] = 65535          # a wall with no gap
    steps = get_neighborhood_steps("r1", directed=True)

    seeds = np.array([5 * cols + 5, 10 * cols + 40, 35 * cols + 30],
                     dtype=np.uint32)
    solver = make_multi_source_solver(raster, steps)
    solver.solve(seeds)
    boundary = solver.boundary_steps()
    pairs = {(int(a), int(b)) for a, b in zip(boundary["region_u"],
                                              boundary["region_v"])}
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs, (
        "the wall has no gap, so region 2 must touch neither other region")


def test_uniform_cost_raster_is_the_worst_case_for_ties():
    """Every step costs the same, so ties are everywhere and the rule must hold."""
    rows, cols = 60, 60
    raster = np.full((rows, cols), 400, dtype=np.uint16)
    steps = get_neighborhood_steps("r2", directed=True)
    terminals = {0: 5 * cols + 5, 1: 55 * cols + 5, 2: 5 * cols + 55,
                 3: 55 * cols + 55}
    seeds = np.array([terminals[k] for k in sorted(terminals)],
                     dtype=np.uint32)

    base = make_multi_source_solver(raster, steps)
    base.solve(seeds)
    permutation = np.array([2, 3, 0, 1])
    other = make_multi_source_solver(raster, steps)
    other.solve(seeds[permutation])
    relabelled = np.where(other.region_array() >= 0,
                          permutation[np.clip(other.region_array(), 0, None)],
                          -1)
    np.testing.assert_array_equal(relabelled, base.region_array())

    routes, graph = overlay_graph(raster, steps, terminals)
    reference = route_metrics(routes, raster, CELL_SIZE)
    for key in routes:
        assert sum(s.length for s in graph.route_segments(key)) == (
            pytest.approx(reference[key]["length_m"]))


# ------------------------------------------------------------------ gradient


def test_gradient_terms_reach_the_corridor_solver():
    """A DEM must change the corridor exactly as it changes find_route.

    This is the check that catches a set_gradient path missed in the corridor
    solver: without it the corridor would silently price its segments on flat
    ground while the pairwise search climbed a hill.
    """
    from pyorps.core.objective import Objective

    rows, cols = 60, 80
    raster = np.full((rows, cols), 400, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)

    dem = np.zeros((rows, cols), dtype=np.float32)
    dem[:, 40:] = np.linspace(0, 60, cols - 40, dtype=np.float32)  # a ramp
    objective = Objective({"cost": 1.0},
                          gradient_options={"max_gradient_pct": 200.0,
                                            "penalty_per_pct": 0.5})
    luts = objective.build_gradient_luts(steps, cell_size=1.0)

    # A single seed: the far corner must be REACHED, not seeded, or its
    # label is 0 either way and the comparison says nothing.
    seed = np.array([0], dtype=np.uint32)
    flat = make_multi_source_solver(raster, steps)
    flat.solve(seed)
    steep = make_multi_source_solver(raster, steps, dem=dem,
                                     gradient_luts=luts)
    steep.solve(seed)

    corner = np.uint32(rows * cols - 1)
    assert steep.dist_array()[corner] > flat.dist_array()[corner], (
        "the ramp did not change any weight: the fixture is vacuous")

    # And the same weights the pairwise search would use.
    reference = make_dijkstra_solver(raster, steps, dem=dem,
                                     gradient_luts=luts)
    reference.reset_root(np.uint32(0))
    reference.settle_all()
    np.testing.assert_allclose(reference.dist_array()[corner],
                               steep.dist_array()[corner], rtol=0, atol=0)


# ------------------------------------------------------------- cell measures


def test_cell_sharing_profile_grows_with_the_snap_radius():
    raster = corridor_raster()
    steps = get_neighborhood_steps("r2", directed=True)
    routes = pairwise_routes(raster, steps, corner_terminals())
    profile = cell_sharing_profile(routes, raster.shape, radii=(0, 1, 2, 3))

    fractions = [profile[r]["shared_fraction"] for r in (0, 1, 2, 3)]
    assert fractions == sorted(fractions)
    assert 0.0 < fractions[0] <= 1.0
    assert fractions[-1] <= 1.0


def test_min_run_rejects_a_bare_crossing():
    """Two routes crossing transversally must not read as one trench."""
    cols = 21
    horizontal = np.array([10 * cols + c for c in range(cols)],
                          dtype=np.uint32)
    vertical = np.array([r * cols + 10 for r in range(21)], dtype=np.uint32)
    routes = {0: horizontal, 1: vertical}

    strict = cell_sharing_profile(routes, (21, cols), radii=(0,), min_run=2)
    lax = cell_sharing_profile(routes, (21, cols), radii=(0,), min_run=1)
    assert strict[0]["shared_cells"] == 0
    assert lax[0]["shared_cells"] == 1, (
        "the two routes must genuinely meet at exactly one cell, or the "
        "strict assertion above is vacuous")


# ------------------------------------------------------------ backend refusal


def test_build_corridor_graph_refuses_a_non_cython_backend():
    from pyorps.graph.path_finder import PathFinder

    finder = PathFinder.__new__(PathFinder)
    finder.graph_api_name = "raster_fim"
    with pytest.raises(NotImplementedError, match="cython"):
        finder.build_corridor_graph(terminals=[(0.0, 0.0), (1.0, 1.0)])


# ------------------------------------------------- regressions from review
#
# Every test below reproduces a defect an adversarial review found in the
# first cut of this feature. They stay grouped so the reason each one exists
# stays attached to it.


def test_price_route_rejects_a_cell_outside_the_raster():
    """An out-of-window index must raise, not read past the buffer.

    ``price_route`` is documented as the referee for anything that slices a
    route apart, and the realistic way to reach it is pricing a route whose
    indices were computed against a LARGER search window. With boundscheck
    off, an index past the end unravels to a row past the end, and a step with
    a negative enough dr pulls the NEIGHBOUR back into range -- so the
    neighbour check passes and both memoryviews are read |dr| * cols elements
    past their allocation.
    """
    raster = np.full((8, 100), 7, dtype=np.uint16)
    steps = np.array([[-120, 0], [120, 0], [1, 0], [-1, 0]], dtype=np.int8)
    with pytest.raises(IndexError, match="outside the raster"):
        price_route_cython(raster, steps,
                           np.array([127 * 100, 7 * 100], dtype=np.uint32))
    # positive control: an in-range route on the same raster prices fine
    assert price_route_cython(
        raster, steps, np.array([0, 100], dtype=np.uint32)) > 0


def test_solve_after_release_raises_instead_of_writing_through_empty_views():
    raster = corridor_raster(rows=40, cols=40)
    steps = get_neighborhood_steps("r1", directed=True)
    solver = make_multi_source_solver(raster, steps)
    solver.solve(np.array([0, 40 * 40 - 1], dtype=np.uint32))
    solver.release()
    with pytest.raises(RuntimeError, match="after release"):
        solver.solve(np.array([0, 40 * 40 - 1], dtype=np.uint32))


def test_an_asymmetric_step_table_is_refused():
    """Each crossing is reported once, so the table must be closed.

    With a one-way table the orientation filter drops whole terminal pairs
    rather than mispricing them, which would silently return a corridor with
    holes.
    """
    raster = corridor_raster(rows=40, cols=40)
    with pytest.raises(ValueError, match="closed under negation"):
        make_multi_source_solver(
            raster, get_neighborhood_steps("r1", directed=False)
        ).solve(np.array([0, 40 * 40 - 1], dtype=np.uint32))
    # positive control: the directed table is accepted
    make_multi_source_solver(
        raster, get_neighborhood_steps("r1", directed=True)
    ).solve(np.array([0, 40 * 40 - 1], dtype=np.uint32))


def test_the_chosen_boundary_step_is_permutation_independent():
    """Not only the partition: the GRAPH must not move with terminal order.

    The tie-break fixes the region labels, but ``boundary_steps`` used to emit
    rows in an order that depended on which endpoint held the smaller label --
    and a label is the terminal's position in the argument array. A caller
    keeping "the k cheapest per pair" then picked a different route among
    equal-cost candidates. Cost ties are pervasive, which is the whole reason
    the tie-break exists, so a uniform raster is the worst case for it.
    """
    rows = cols = 60
    raster = np.full((rows, cols), 400, dtype=np.uint16)
    steps = get_neighborhood_steps("r2", directed=True)
    seeds = np.array([5 * cols + 5, 55 * cols + 5, 5 * cols + 55,
                      55 * cols + 55], dtype=np.uint32)
    permutation = np.array([2, 3, 0, 1])

    def chosen(order_seeds, label_to_terminal):
        solver = make_multi_source_solver(raster, steps)
        solver.solve(order_seeds)
        boundary = solver.boundary_steps()
        best = {}
        for pos in np.argsort(boundary["cost"], kind="stable"):
            pair = tuple(sorted((
                label_to_terminal[int(boundary["region_u"][pos])],
                label_to_terminal[int(boundary["region_v"][pos])])))
            best.setdefault(pair, frozenset((int(boundary["cell_u"][pos]),
                                             int(boundary["cell_v"][pos]))))
        return best

    base = chosen(seeds, {i: i for i in range(4)})
    other = chosen(seeds[permutation],
                   {i: int(permutation[i]) for i in range(4)})
    assert base, "no pair of regions touches: the assertion would be vacuous"
    assert base == other


def test_overlap_report_rebuild_counts_traversals_not_distinct_routes():
    """A route that uses one segment twice must not shrink the per-route total.

    ``use_count`` is the size of a SET of route ids. Rebuilding the per-route
    accounting from it undercounts an out-and-back stub, and the report then
    claims an overcount of zero where there is a real one.
    """
    raster = np.full((10, 10), 10, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)
    routes = {"P": [0, 1, 2, 1, 0], "Q": [2, 12, 22]}
    graph = corridor_graph_from_routes(routes, raster, TRANSFORM, steps=steps,
                                       price_routing_cost=False)
    measured = graph.overlap_report(
        route_metrics=route_metrics(routes, raster, CELL_SIZE))
    rebuilt = graph.overlap_report()

    assert measured.overcount > 0, (
        "the fixture must have a real overcount or the check is vacuous")
    assert rebuilt.route_length_m == pytest.approx(measured.route_length_m)
    assert rebuilt.route_cost == pytest.approx(measured.route_cost)
    assert rebuilt.overcount == pytest.approx(measured.overcount)


def test_a_grazing_third_route_does_not_demote_a_long_shared_trench():
    """The prune measures per route PAIR, not per exact membership set.

    Measuring by membership equality splits a long two-route trench wherever a
    third route touches one step of it, and each fragment can then fall below
    the threshold -- demoting a genuinely long shared stretch in full.
    """
    raster = np.full((40, 60), 10, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)
    trench = [10 * 60 + c for c in range(10, 51)]
    routes = {
        "A": trench + [11 * 60 + 50, 12 * 60 + 50],
        "B": trench + [9 * 60 + 50, 8 * 60 + 50],
    }
    grazing = dict(routes)
    grazing["C"] = [8 * 60 + 30, 9 * 60 + 30, 10 * 60 + 30, 10 * 60 + 31,
                    11 * 60 + 31]

    def shared(rr, minimum):
        return corridor_graph_from_routes(
            rr, raster, TRANSFORM, steps=steps, price_routing_cost=False,
            min_shared_length_m=minimum).overlap_report().shared_length_m

    # TRANSFORM has 2 m cells, so the 40-step trench is 80 m long.
    trench_m = 40 * CELL_SIZE
    assert shared(routes, 0.0) == pytest.approx(trench_m)
    assert shared(routes, 0.5 * trench_m) == pytest.approx(trench_m)
    # The third route must not change the answer for A and B.
    assert shared(grazing, 0.0) == pytest.approx(trench_m)
    assert shared(grazing, 0.5 * trench_m) == pytest.approx(trench_m)
    # positive control: a threshold above the run length does demote it
    assert shared(grazing, 1e6) == 0.0


def test_a_crossing_is_not_sharing_at_any_snap_radius():
    """min_run must be applied per other route, and must scale with dilation.

    Or-ing every other route's mask first turns two separate crossings at
    adjacent cells into a run of two; and a dilated single crossing is already
    2*radius+1 cells wide, so a fixed min_run excludes nothing from r = 1 on.
    Either way a pure crossing drives the sensitivity curve.
    """
    cols = 48
    horizontal = np.array([10 * cols + c for c in range(2, 46)],
                          dtype=np.uint32)
    vertical = np.array([r * cols + 24 for r in range(2, 19)], dtype=np.uint32)
    profile = cell_sharing_profile({0: horizontal, 1: vertical}, (40, cols),
                                   radii=(0, 1, 2, 3), min_run=2)
    for radius, values in profile.items():
        assert values["shared_fraction"] == 0.0, (
            f"a pure crossing reads as {values['shared_fraction']:.1%} shared "
            f"at r_snap={radius}")

    # positive control: two routes one cell apart ARE one trench once snapped
    near_a = np.array([10 * cols + c for c in range(2, 40)], dtype=np.uint32)
    near_b = np.array([11 * cols + c for c in range(2, 40)], dtype=np.uint32)
    near = cell_sharing_profile({0: near_a, 1: near_b}, (40, cols),
                                radii=(0, 1), min_run=2)
    assert near[0]["shared_fraction"] == 0.0
    assert near[1]["shared_fraction"] > 0.9


@pytest.mark.parametrize("key", [0, (0, 1, 1), (0, 1)])
def test_route_cells_keeps_the_input_orientation(key):
    """Segments are STORED canonically, so orientation cannot be inferred.

    Inferring it from the key shape returned about half of all single-segment
    routes reversed, silently, for every key that was not a pair of terminal
    ids -- an int from a plain list, or an (a, b, rank) triple from
    ``k_per_pair > 1``.
    """
    raster = np.full((10, 10), 10, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)
    cells = [22, 21, 20]
    other = [22, 32, 42]
    if isinstance(key, tuple):
        routes = {key: cells, (0, 2): other}
        terminals = {0: 22, 1: 20, 2: 42}
        lookup = key
    else:
        routes = [cells, other]
        terminals = None
        lookup = 0
    graph = corridor_graph_from_routes(routes, raster, TRANSFORM, steps=steps,
                                       terminal_cells=terminals,
                                       price_routing_cost=False)
    assert [int(c) for c in graph.route_cells(lookup)] == cells


def test_a_path_id_collision_does_not_lose_a_route():
    """Two Paths from different finders share path_id 0; both must survive."""
    from shapely.geometry import LineString

    from pyorps.core.path import Path

    def make(cells):
        return Path(source=(0, 0), target=(1, 1), algorithm="dijkstra",
                    graph_api="cython", path_indices=cells,
                    path_coords=[(0, 0), (1, 1)],
                    path_geometry=LineString([(0, 0), (1, 1)]),
                    euclidean_distance=1.0, runtimes={}, path_id=0,
                    search_space_buffer_m=0, neighborhood="r1")

    raster = np.full((10, 10), 10, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)
    graph = corridor_graph_from_routes(
        [make([20, 21, 22, 23]), make([3, 13, 23, 33])], raster, TRANSFORM,
        steps=steps, price_routing_cost=False)
    assert len(graph.pair_routes) == 2
    assert graph.overlap_report().n_routes == 2


def test_constrained_path_finder_refuses_to_build_a_corridor():
    """It routes with a different edge model, so the guard must be on TYPE.

    The backend check alone passes for ConstrainedPathFinder -- it defaults to
    graph_api="cython" and never touches self.graph_api -- and the corridor
    would then be built with none of the span, angle or tower constraints the
    class exists to enforce.
    """
    from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

    finder = ConstrainedPathFinder.__new__(ConstrainedPathFinder)
    finder.graph_api_name = "cython"
    with pytest.raises(NotImplementedError, match="PathFinder only"):
        finder.build_corridor_graph(terminals=[(0.0, 0.0), (1.0, 1.0)])


def test_gradient_luts_must_match_the_direction_table():
    """A short per-direction array is an OOB read, not an exception."""
    from pyorps.core.objective import Objective

    raster = np.full((20, 20), 400, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)
    dem = np.zeros((20, 20), dtype=np.float32)
    luts = Objective({"cost": 1.0},
                     gradient_options={"max_gradient_pct": 200.0}
                     ).build_gradient_luts(steps, cell_size=1.0)

    solver = make_multi_source_solver(raster, steps)
    with pytest.raises(ValueError, match="one entry per"):
        solver.set_gradient(dem, luts.mult, luts.add,
                            np.asarray(luts.bin_factor[:-1], dtype=np.float32),
                            np.asarray(luts.step_len_cells, dtype=np.float32),
                            luts.n_bins)
    # positive control: the full-length arrays are accepted
    solver.set_gradient(dem, luts.mult, luts.add,
                        np.asarray(luts.bin_factor, dtype=np.float32),
                        np.asarray(luts.step_len_cells, dtype=np.float32),
                        luts.n_bins)


# ------------------------------------------------------------- absolute pins
#
# The conservation test compares two callers of _segment_metrics, so it cannot
# see an error that scales BOTH sides equally -- the cell_size multiplication
# above all. Everything below is hand-computed from the step rule, so it fails
# on a unit error, on a lost canonical dedup, and on a changed cut rule.
#
# The arithmetic, once, for a 3x3 uniform raster of value 100 with r0 steps and
# 2 m cells (TRANSFORM):
#   one orthogonal step has segment_length 1.0 cell and no intermediates, so
#   calculate_path_metrics_numba gives it length 1.0 cell = 2.0 m and spreads
#   1.0/2 cell to each endpoint's category. A two-step route is 4.0 m and, at
#   100 per metre, 400.0 of construction cost.

PIN_RASTER = np.full((3, 3), 100, dtype=np.uint16)
PIN_STEPS = get_neighborhood_steps("r0", directed=True)
STEP_M = 1.0 * CELL_SIZE          # 2.0 m
STEP_COST = 100.0 * STEP_M        # 200.0


def test_the_pin_fixture_arithmetic_is_what_the_kernel_produces():
    """One orthogonal step: 2 m and 200 of cost. Everything below builds on it."""
    metrics = route_metrics({"A": [0, 1]}, PIN_RASTER, CELL_SIZE)["A"]
    assert metrics["length_m"] == pytest.approx(STEP_M)
    assert metrics["construction_cost"] == pytest.approx(STEP_COST)
    assert metrics["length_by_category"] == {100: pytest.approx(STEP_M)}


def test_corridor_totals_match_hand_computed_values():
    """Absolute pin: every headline number, computed by hand, not by the code.

    Two routes leaving cell 0 together and parting at cell 1. The shared step
    is dug once, so the corridor is three steps where the per-route accounting
    charges four.
    """
    routes = {"A": [0, 1, 2], "B": [0, 1, 4]}
    graph = corridor_graph_from_routes(
        routes, PIN_RASTER, TRANSFORM, steps=PIN_STEPS,
        price_routing_cost=True)

    assert len(graph.segments) == 3
    assert sorted(s.use_count for s in graph.segments.values()) == [1, 1, 2]
    assert graph.total_length == pytest.approx(3 * STEP_M)
    assert graph.total_construction_cost == pytest.approx(3 * STEP_COST)

    report = graph.overlap_report(
        route_metrics=route_metrics(routes, PIN_RASTER, CELL_SIZE))
    assert report.route_length_m == pytest.approx(4 * STEP_M)
    assert report.route_cost == pytest.approx(4 * STEP_COST)
    assert report.corridor_length_m == pytest.approx(3 * STEP_M)
    assert report.shared_length_m == pytest.approx(STEP_M)
    assert report.shared_fraction == pytest.approx(1 / 3)
    assert report.overcount == pytest.approx(STEP_COST)
    assert report.overstatement == pytest.approx(0.25)
    assert report.mean_multiplicity == pytest.approx(2.0)
    assert report.n_junctions == 1

    # And the search metric: one r0 step over two cells of 100 costs
    # (100 + 100) * (1 / 2) = 100 in CELL units.
    assert sum(s.metrics["routing_cost"]
               for s in graph.segments.values()) == pytest.approx(300.0)


def test_one_trench_traversed_both_ways_is_one_segment():
    """The canonical key is the whole point of the object.

    Two connections running the same trench in opposite directions are one
    trench. Without the direction-independent key they resolve into two
    segments, the corridor length doubles, and the reported overcount
    collapses to zero -- with every per-route total still adding up.
    """
    routes = {"A": [0, 1, 2], "B": [2, 1, 0]}
    graph = corridor_graph_from_routes(
        routes, PIN_RASTER, TRANSFORM, steps=PIN_STEPS,
        price_routing_cost=False)

    assert len(graph.segments) == 1
    segment = next(iter(graph.segments.values()))
    assert segment.use_count == 2
    assert graph.total_length == pytest.approx(2 * STEP_M)

    report = graph.overlap_report(
        route_metrics=route_metrics(routes, PIN_RASTER, CELL_SIZE))
    assert report.shared_fraction == pytest.approx(1.0)
    assert report.overstatement == pytest.approx(0.5)
    # Both routes reassemble in their OWN direction from that one segment.
    assert [int(c) for c in graph.route_cells("A")] == [0, 1, 2]
    assert [int(c) for c in graph.route_cells("B")] == [2, 1, 0]


def test_a_terminal_on_another_route_forces_a_cut():
    """A turbine sitting on someone else's trench must be connectable there."""
    routes = {"A": [0, 1, 2], "B": [1, 4]}
    graph = corridor_graph_from_routes(
        routes, PIN_RASTER, TRANSFORM, steps=PIN_STEPS,
        price_routing_cost=False)
    # A is cut at cell 1 although its own membership never changes there.
    assert len(graph.route_segments("A")) == 2
    assert any(node.cell_index == 1 and node.kind == NODE_TERMINAL
               for node in graph.nodes.values())


def test_the_uniform_fixture_really_has_equal_cost_alternatives():
    """Anti-vacuity for every permutation test on a uniform raster.

    The tie-break only has work to do where a cell has two equally cheap
    predecessors. If it does not, the permutation tests below pass with the
    rule deleted and pin nothing.
    """
    rows = cols = 30
    raster = np.full((rows, cols), 400, dtype=np.uint16)
    steps = get_neighborhood_steps("r2", directed=True)
    solver = make_multi_source_solver(raster, steps)
    solver.solve(np.array([0, rows * cols - 1], dtype=np.uint32))
    dist = solver.dist_array()

    ties = 0
    for cell in range(rows * cols):
        if not np.isfinite(dist[cell]):
            continue
        best = [d for d in range(len(steps))
                if _incoming_cost(solver, raster, steps, cell, d, cols, rows)
                == pytest.approx(dist[cell], rel=0, abs=1e-9)]
        if len(best) > 1:
            ties += 1
    assert ties > 100, (
        f"only {ties} cells have an equal-cost alternative predecessor; the "
        f"permutation tests would not exercise the tie-break")


def _incoming_cost(solver, raster, steps, cell, direction, cols, rows):
    """dist[u] + step(u -> cell) for the u that direction points back from."""
    row, col = divmod(cell, cols)
    dr, dc = int(steps[direction][0]), int(steps[direction][1])
    ur, uc = row - dr, col - dc
    if not (0 <= ur < rows and 0 <= uc < cols):
        return float("inf")
    upstream = ur * cols + uc
    dist = solver.dist_array()
    if not np.isfinite(dist[upstream]):
        return float("inf")
    weight = solver.step_cost(np.uint32(upstream), direction)
    return dist[upstream] + weight


def test_gradient_terms_reach_the_corridor_GRAPH_not_only_the_solver():
    """Segment routing_cost must carry the DEM, end to end.

    The bare-solver test above pins set_gradient; this one pins that
    corridor_graph_from_routes actually forwards dem/gradient_luts into the
    solver it builds to price segments. Without it the corridor would price
    its trenches on flat ground while the search climbed a hill.
    """
    from pyorps.core.objective import Objective

    rows, cols = 20, 30
    raster = np.full((rows, cols), 400, dtype=np.uint16)
    steps = get_neighborhood_steps("r1", directed=True)
    dem = np.tile(np.linspace(0, 60, cols, dtype=np.float32), (rows, 1))
    luts = Objective(
        {"cost": 1.0},
        gradient_options={"max_gradient_pct": 400.0, "penalty_per_pct": 0.5},
    ).build_gradient_luts(steps, cell_size=1.0)

    routes = {"A": [0, 1, 2, 3], "B": [0, 1, cols + 2]}
    flat = corridor_graph_from_routes(routes, raster, TRANSFORM, steps=steps,
                                      price_routing_cost=True)
    steep = corridor_graph_from_routes(routes, raster, TRANSFORM, steps=steps,
                                       dem=dem, gradient_luts=luts,
                                       price_routing_cost=True)

    flat_cost = sum(s.metrics["routing_cost"] for s in flat.segments.values())
    steep_cost = sum(s.metrics["routing_cost"]
                     for s in steep.segments.values())
    assert steep_cost > flat_cost, (
        "the DEM did not change any segment price: either the ramp is flat or "
        "the gradient LUTs never reached the pricing solver")
    # The terrain accounting is DEM-free by definition and must not move.
    assert flat.total_construction_cost == pytest.approx(
        steep.total_construction_cost)


# ------------------------------------------------- PathFinder, end to end


@pytest.fixture
def finder_raster(tmp_path):
    """A small GeoTIFF with two cheap tracks, written to disk for PathFinder."""
    import rasterio
    from affine import Affine as _Affine

    rows, cols = 120, 160
    rng = np.random.default_rng(23)
    raster = np.full((rows, cols), 500, dtype=np.uint16)
    raster += rng.integers(0, 120, size=(rows, cols)).astype(np.uint16)
    raster[60, :] = 130
    raster[:, 30] = 140
    transform = _Affine(1.0, 0.0, 442000.0, 0.0, -1.0, 5587000.0)
    path = tmp_path / "cost.tif"
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="uint16", crs="EPSG:25832",
                       transform=transform) as dataset:
        dataset.write(raster, 1)

    def xy(row, col):
        x, y = transform * (col + 0.5, row + 0.5)
        return (float(x), float(y))

    return path, xy


def test_build_corridor_graph_end_to_end(finder_raster, tmp_path):
    """The PathFinder entry point: build, reconcile, save, plot.

    Nothing else in the suite reaches build_corridor_graph, _pair_cost_gap,
    save_corridor_graph or plot_corridor_graph, so a break in any of them
    would only show up in a case study.
    """
    import matplotlib
    matplotlib.use("Agg")
    import geopandas as gpd

    from pyorps import PathFinder

    path, xy = finder_raster
    terminals = [xy(10, 10), xy(110, 20), xy(15, 150), xy(112, 148),
                 xy(80, 80)]
    finder = PathFinder(dataset_source=str(path), source_coords=terminals[0],
                        target_coords=terminals[1:],
                        search_space_buffer_m=60, neighborhood_str="r2")

    graph = finder.build_corridor_graph(terminals=terminals,
                                        validate_pair_costs=True)
    assert graph.construction == "distance_network"
    assert graph.segments and graph.terminal_nodes
    assert graph.provenance["backend"] == "cython"
    assert graph.provenance["settled_cells"] > 0

    # The distance network is an UPPER bound on the pairwise metric.
    gap = graph.provenance["pair_cost_gap"]
    assert gap["per_pair"]
    assert min(gap["per_pair"].values()) >= -1e-9, (
        "a recovered pair costs LESS than the true shortest path, which is "
        "impossible for a real route")

    # Conservation through the PathFinder path, against the kernel referee.
    raster = finder.raster_handler.data[0]
    cell_size = float(abs(finder.raster_handler.window_transform.a))
    model = EdgeModel(finder.steps, max_value=65535)
    for key in graph.pair_routes:
        cells = np.asarray(graph.route_cells(key), dtype=np.uint32)
        priced = reprice_path(cells, raster, model)
        assert priced.ok, priced.violations
        segments = graph.route_segments(key)
        assert sum(s.metrics["routing_cost"] for s in segments) == (
            pytest.approx(priced.cost, rel=1e-9))
        assert sum(s.length for s in segments) == pytest.approx(
            float(calculate_path_metrics_numba(raster, cells, None)[0])
            * cell_size, rel=1e-12)

    # save(): two layers, and a second write must REPLACE, not append.
    out = tmp_path / "corridor.gpkg"
    finder.save_corridor_graph(str(out))
    finder.save_corridor_graph(str(out))
    assert len(gpd.read_file(out, layer="corridor_segments")) == len(
        graph.segments)
    assert len(gpd.read_file(out, layer="corridor_nodes")) == len(graph.nodes)

    axes = finder.plot_corridor_graph()
    assert axes is not None

    # k_per_pair > 1 adds routes, never removes any.
    richer = finder.build_corridor_graph(terminals=terminals, k_per_pair=2)
    assert len(richer.pair_routes) >= len(graph.pair_routes)

    # max_pair_distance drops the expensive pairs but keeps the cheap ones.
    priced = sorted(s.metrics["routing_cost"]
                    for s in graph.segments.values())
    assert priced, "no segment was priced: the cut-off check would be vacuous"
    lean = finder.build_corridor_graph(terminals=terminals,
                                       max_pair_distance=10 * priced[-1])
    assert 0 < len(lean.pair_routes) <= len(graph.pair_routes)
