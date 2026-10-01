"""DEPRECATED shim: the route builder moved to
:mod:`pyorps.gui.services.routing` (Section 15).

The v1 API is preserved: ``run_routing`` here returns ``(finder, results)``
(failed pairs silently skipped) — the new service additionally returns the
failed pairs; use it directly for error reporting.
"""
from __future__ import annotations

from pyorps.gui.services.routing import (  # noqa: F401  # pylint: disable=unused-import
    BuiltRoute,
    make_pairs,
    resolve_backend,
)
from pyorps.gui.services.routing import run_routing as _run_routing

# v1 UI algorithm mapping (kept for backwards compatibility)
ALGORITHMS = {
    "dijkstra": "dijkstra",
    "bidirectional_dijkstra": "bidirectional_dijkstra",
    "delta-stepping": "delta-stepping",
}


def run_routing(raster_path, *, sources, targets, waypoints=None,
                algorithm="delta-stepping", hardware="cpu",
                neighborhood="r2", pairwise=False, search_buffer_m=None):
    """v1-compatible wrapper: returns (finder, [BuiltRoute])."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    finder, results, _failed = _run_routing(
        raster_path, sources=sources, targets=targets, waypoints=waypoints,
        algorithm=algorithm, hardware=hardware, neighborhood=neighborhood,
        pairwise=pairwise, search_buffer_m=search_buffer_m)
    return finder, results
