"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland

Backward-compatibility shim — imports from Cython _traversal module.
Falls back to Numba implementations if Cython extension is not available.

THE INTERMEDIATE-CELL INVARIANT (read before changing any edge builder)
----------------------------------------------------------------------
A step from ``(r, c)`` to ``(r+dr, c+dc)`` is admitted only if EVERY
intermediate cell of that step is passable. ``intermediate_steps_numba``
(``_raster_context.pyx:_calculate_intermediate_steps_cython``) enumerates them:
a cardinal step has none, a single diagonal decomposes into exactly its two
flanking cells ``(dr, 0)`` and ``(0, dc)``, and a longer r2/r3 step samples
both the floor and the ceil of every fractional position, so no cell the
segment could graze is missed.

The routing consequence is not stated anywhere in the algorithm literature
this project cites, so it is stated here: **pyorps cannot cut the corner of a
diagonal barrier.** Two free cells that touch only at a corner are never
connected, because the two cells flanking that corner are the step's
intermediates. The rule is in fact stricter than "no corner cutting" — a
diagonal is rejected when EITHER flank is impassable, not only when both are.

This is an emergent property of the intermediate enumeration, not an explicit
check, and it is relied on by every backend (Cython Dijkstra and delta
stepping, the graph libraries via the edge list built here, the CUDA raster
kernels via the same LUT uploaded by ``traversal_gpu``, and the constrained
planners via their precomputed ``icache_status``). A "fast path" for diagonal
steps that skips the intermediate loop would silently reintroduce corner
cutting: routes stay optimal for the graph that was built, and the graph is
what is wrong. ``tests/test_graph/test_corner_cutting_invariant.py`` pins it.

It has no effect where nothing is impassable — with ``ignore_max=False`` the
exclude mask is all-ones and every cell, 65535 included, is traversable.
"""

try:
    from pyorps.utils._traversal import (  # noqa: F401
        # Gradient
        calculate_gradient_penalty,
        # Core path functions
        calculate_path_metrics_numba,
        calculate_region_bounds,
        calculate_segment_length,
        check_max_values,
        # Graph construction
        construct_edges,
        construct_edges_3d,
        # Distance calculations
        euclidean_distances_numba,
        # Position correction
        find_nearest_valid_positions_numba,
        find_valid_nodes,
        find_valid_nodes_3d,
        get_cost_factor_numba,
        get_max_number_of_edges,
        # Path analysis
        get_outgoing_edges,
        intermediate_steps_numba,
        # Node validation
        is_valid_node,
    )
    from pyorps.utils._traversal import (  # noqa: F401
        # Index manipulation
        py_ravel_index as ravel_index,
    )

    _CYTHON_TRAVERSAL = True

except ImportError:
    # Fallback: use the Numba implementations
    _CYTHON_TRAVERSAL = False

    from pyorps.utils._traversal_numba import (  # noqa: F401
        calculate_path_metrics_numba,
        intermediate_steps_numba,
        construct_edges,
        construct_edges_3d,
        get_max_number_of_edges,
        euclidean_distances_numba,
        get_cost_factor_numba,
        ravel_index,
        calculate_region_bounds,
        is_valid_node,
        find_valid_nodes,
        find_valid_nodes_3d,
        get_outgoing_edges,
        calculate_segment_length,
        find_nearest_valid_positions_numba,
        check_max_values,
        calculate_gradient_penalty,
    )
