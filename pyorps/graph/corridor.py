"""
Corridor construction: reduce a set of raster routes to their shared trenches.

The overlay construction of :mod:`pyorps.core.corridor`. Given routes computed
over one raster window, it finds every cell where routes merge or diverge,
promotes those cells to nodes, and returns the stretches between them as
segments carrying their construction cost once.

Why steps, not cells, are the unit of coincidence
-------------------------------------------------
PYORPS prices a route step by step::

    step_cost = (raster[u] + sum(raster[intermediates]) + raster[v]) * factor
    factor    = sqrt(dr^2 + dc^2) / (2 + n_intermediates)

and ``calculate_path_metrics_numba`` attributes each step's length to the cells
it touches by the same rule. From r2 upward a step spans more than two cells,
so "these two routes use the same cell" does not imply "they pay the same
step". Cutting on cells would therefore split a step and force its cost to be
apportioned between two segments by an invented rule.

Cutting on STEP identity avoids that entirely: a junction can only land on a
step endpoint, every step belongs to exactly one segment, and the segment
metrics sum back to the route metrics exactly rather than approximately. That
exactness is the whole reason no tolerance parameter appears in this module.

The price of step identity is that two routes covering the same ground with
DIFFERENT step decompositions register as unshared. :func:`cell_sharing_profile`
measures how much that costs on real data, and is reported next to the exact
number rather than replacing it.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from shapely.geometry import LineString

from pyorps.core.corridor import (
    NODE_JUNCTION,
    NODE_TERMINAL,
    CorridorGraph,
    CorridorNode,
    CorridorSegment,
)
from pyorps.core.path import Path, PathCollection
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.utils._dijkstra import make_multi_source_solver
from pyorps.utils._raster_context import NO_EXCLUSION_VALUE
from pyorps.utils._traversal import (
    calculate_path_metrics_numba,
    unique_categories,
)

__all__ = [
    "corridor_graph_from_routes",
    "route_metrics",
    "cell_sharing_profile",
    "supercover_cells",
]


# --------------------------------------------------------------------- input


def _normalize_routes(routes) -> list[tuple[Any, np.ndarray]]:
    """Accept a mapping, a Path/PathCollection or bare cell sequences.

    Returns a list of ``(route_key, cells_uint32)``. The key is what the
    caller identified the route by: a ``(terminal_a, terminal_b)`` tuple from a
    mapping, the ``path_id`` of a Path, or the position in a plain list.
    """
    if isinstance(routes, PathCollection):
        routes = list(routes)

    normalized: list[tuple[Any, np.ndarray]] = []

    if isinstance(routes, Mapping):
        items: Iterable = routes.items()
    elif isinstance(routes, Sequence) and not isinstance(routes, (str, bytes)):
        items = enumerate(routes)
    else:
        raise TypeError(
            "routes must be a mapping of key -> cell sequence, a sequence of "
            f"cell sequences, or a PathCollection; got {type(routes)!r}")

    seen_keys: set = set()
    for key, value in items:
        # The caller's key is the identity, always. Re-keying a Path on its
        # path_id looks tidier and is wrong: path_id is a per-PathFinder
        # counter, so a list assembled from two finders collides, one route's
        # segment list overwrites the other's in pair_routes, and
        # route_segments then silently returns the wrong geometry while every
        # total still adds up.
        cells = np.asarray(
            value.path_indices if isinstance(value, Path) else value,
            dtype=np.uint32)
        if cells.ndim != 1:
            raise ValueError(
                f"route {key!r} is not a 1D cell sequence (shape {cells.shape})")
        if cells.size < 2:
            # A degenerate route contributes no step and therefore no trench;
            # dropping it is not the same as failing, and a coincident
            # source/target pair is a legitimate input.
            continue
        if key in seen_keys:
            raise ValueError(
                f"route key {key!r} appears twice; keys identify routes in "
                f"pair_routes, so a duplicate would silently drop one of them")
        seen_keys.add(key)
        normalized.append((key, cells))

    if not normalized:
        raise ValueError("no route with at least two cells was supplied")
    return normalized


def _terminal_table(
        normalized: list[tuple[Any, np.ndarray]],
        terminal_cells: Mapping[Any, int] | None,
) -> tuple[dict[int, Any], dict[Any, int]]:
    """Build ``cell -> terminal_id`` and ``terminal_id -> cell``.

    Without an explicit table the terminals are the route endpoints, numbered
    in order of first appearance so the numbering is a function of the route
    list alone.
    """
    if terminal_cells is not None:
        by_id = {tid: int(cell) for tid, cell in terminal_cells.items()}
        by_cell = {cell: tid for tid, cell in by_id.items()}
        if len(by_cell) != len(by_id):
            raise ValueError(
                "terminal_cells maps two terminals onto the same cell; "
                "deduplicate them before building a corridor graph")
        return by_cell, by_id

    by_cell: dict[int, Any] = {}
    by_id: dict[Any, int] = {}
    for _key, cells in normalized:
        for cell in (int(cells[0]), int(cells[-1])):
            if cell not in by_cell:
                tid = len(by_cell)
                by_cell[cell] = tid
                by_id[tid] = cell
    return by_cell, by_id


# ------------------------------------------------------------------- metrics


def _segment_metrics(
        cells: np.ndarray,
        raster: np.ndarray,
        cell_size: float,
        categories: np.ndarray,
) -> dict[str, Any]:
    """Length and per-category length of one cell run, in CRS units.

    Uses ``calculate_path_metrics_numba`` -- the same kernel PathFinder reports
    a Path with -- so a segment is measured by the code the route was measured
    by, not by a second implementation of the same rule.
    """
    total_length_cells, cats, lengths_cells = calculate_path_metrics_numba(
        raster, np.ascontiguousarray(cells, dtype=np.uint32), categories)
    lengths_m = np.asarray(lengths_cells) * cell_size
    by_category = {int(c): float(m)
                   for c, m in zip(cats, lengths_m) if m > 0.0}
    return {
        "length_m": float(total_length_cells) * cell_size,
        "construction_cost": float(
            sum(value * metres for value, metres in by_category.items())),
        "length_by_category": by_category,
    }


def route_metrics(
        routes,
        raster: np.ndarray,
        cell_size: float,
) -> dict[Any, dict[str, Any]]:
    """Per-route length and construction cost, measured like a Path.

    The independent witness for the conservation law: compare this against the
    sum over the segments a route resolves into.
    """
    normalized = _normalize_routes(routes)
    categories = unique_categories(raster)
    return {key: _segment_metrics(cells, raster, cell_size, categories)
            for key, cells in normalized}


# ---------------------------------------------------------------- the graph


def corridor_graph_from_routes(
        routes,
        raster: np.ndarray,
        transform,
        crs=None,
        *,
        steps: np.ndarray | None = None,
        max_value: int = IMPASSABLE_CELL_COST,
        ignore_max_cost: bool = True,
        dem: np.ndarray | None = None,
        gradient_luts=None,
        terminal_cells: Mapping[Any, int] | None = None,
        min_shared_length_m: float = 0.0,
        price_routing_cost: bool = True,
        pricer=None,
        provenance: dict[str, Any] | None = None,
) -> CorridorGraph:
    """Segment a set of raster routes at the cells where they merge or diverge.

    Parameters:
        routes: Mapping ``key -> cell sequence``, a sequence of cell
            sequences, a list of Path objects or a PathCollection. Cells are
            linear indices into ``raster``.
        raster: The uint16 cost raster the routes were computed on (the search
            WINDOW, not the full file - the cell indices must index it).
        transform: Affine transform of that same window.
        crs: CRS to stamp on the exported geometries.
        steps: Neighbourhood step table. Required for ``price_routing_cost``;
            also validates that every route step is one the kernel could take.
        max_value / ignore_max_cost: Obstacle semantics, as in PathFinder.
        dem, gradient_luts: Optional per-edge gradient terms, so a corridor
            built over a DEM is priced by the same weights the search used.
        terminal_cells: Optional ``terminal_id -> cell`` table. When omitted the
            terminals are the route endpoints, numbered by first appearance.
        min_shared_length_m: Discard a shared run shorter than this and revert
            those steps to per-route membership. 0 (the default) keeps the
            construction exact; a positive value is the CIRED-plan degenerate-
            junction prune and is reported in ``provenance``.
        price_routing_cost: Also price every segment in the SEARCH metric
            (cell units) with the kernel's own arithmetic. Needs ``steps``.
        pricer: An existing MultiSourceSolver over the SAME raster and step
            table, reused instead of building a second one. A solver owns 17
            bytes per cell, so on a real search window that is hundreds of
            megabytes saved and one exclude-mask scan avoided.

    Returns:
        A :class:`~pyorps.core.corridor.CorridorGraph`.
    """
    normalized = _normalize_routes(routes)
    raster = np.ascontiguousarray(raster)
    if raster.ndim != 2:
        raise ValueError(f"raster must be 2D, got shape {raster.shape}")
    rows, cols = raster.shape
    cell_size = float(abs(transform.a))

    for key, cells in normalized:
        if cells.max() >= rows * cols:
            raise ValueError(
                f"route {key!r} references cell {int(cells.max())}, outside "
                f"the {rows}x{cols} raster window it was priced against")

    by_cell, by_id = _terminal_table(normalized, terminal_cells)
    # Derived once: the table depends only on the raster, and both the
    # prune and every segment measurement need it.
    categories = unique_categories(raster)

    # ---- 1. step membership -------------------------------------------
    # An undirected key: two routes running the same trench in opposite
    # directions are one trench.
    membership: dict[tuple[int, int], set[int]] = defaultdict(set)
    route_steps: list[list[tuple[int, int]]] = []
    for route_idx, (_key, cells) in enumerate(normalized):
        keys = []
        for a, b in zip(cells[:-1], cells[1:]):
            a, b = int(a), int(b)
            step_key = (a, b) if a < b else (b, a)
            membership[step_key].add(route_idx)
            keys.append(step_key)
        route_steps.append(keys)

    # ---- 2. optional prune of degenerate shared runs -------------------
    demoted: set[tuple[int, int]] = set()
    if min_shared_length_m > 0.0:
        demoted = _short_shared_runs(
            normalized, route_steps, membership, raster, cell_size,
            categories, min_shared_length_m)

    def members_of(step_key: tuple[int, int], route_idx: int) -> frozenset[int]:
        if step_key in demoted:
            return frozenset((route_idx,))
        return frozenset(membership[step_key])

    # ---- 3. cut every route where its membership changes ---------------
    segments: dict[tuple[int, ...], int] = {}
    seg_records: dict[int, dict[str, Any]] = {}
    pair_routes: dict[Any, list[int]] = {}
    route_starts: dict[Any, int] = {}
    node_cells: dict[int, dict[str, Any]] = {}

    for route_idx, (key, cells) in enumerate(normalized):
        keys = route_steps[route_idx]
        cut_at = [0, len(cells) - 1]
        for i in range(1, len(cells) - 1):
            cell = int(cells[i])
            if cell in by_cell:
                # A terminal sitting on this route is a connection point: the
                # route must be splittable there or nothing can join it.
                cut_at.append(i)
                continue
            if members_of(keys[i - 1], route_idx) != members_of(keys[i],
                                                               route_idx):
                cut_at.append(i)
        cut_at = sorted(set(cut_at))

        ordered_ids: list[int] = []
        for start, stop in zip(cut_at[:-1], cut_at[1:]):
            run = cells[start:stop + 1]
            canonical = _canonical(run)
            seg_id = segments.get(canonical)
            if seg_id is None:
                seg_id = len(segments)
                segments[canonical] = seg_id
                seg_records[seg_id] = {
                    "cells": np.asarray(canonical, dtype=np.uint32),
                    "members": set(),
                }
            seg_records[seg_id]["members"].add(route_idx)
            ordered_ids.append(seg_id)

            for endpoint in (int(run[0]), int(run[-1])):
                record = node_cells.setdefault(
                    endpoint, {"segments": set(), "routes": set()})
                record["segments"].add(seg_id)
                record["routes"].add(route_idx)

        pair_routes[key] = ordered_ids
        route_starts[key] = int(cells[0])

    # ---- 4. build the objects ------------------------------------------
    solver = None
    if price_routing_cost:
        solver = pricer
        if solver is None:
            if steps is None:
                raise ValueError(
                    "price_routing_cost=True needs the neighbourhood step "
                    "table; pass steps=, pricer=, or set "
                    "price_routing_cost=False")
            effective_max = (int(max_value) if ignore_max_cost
                             else NO_EXCLUSION_VALUE)
            solver = make_multi_source_solver(
                raster, np.ascontiguousarray(steps, dtype=np.int8),
                max_value=effective_max, dem=dem,
                gradient_luts=gradient_luts)

    node_ids: dict[int, int] = {}
    graph_nodes: dict[int, CorridorNode] = {}
    for cell in sorted(node_cells):
        node_id = len(node_ids)
        node_ids[cell] = node_id
        terminal_id = by_cell.get(cell)
        # Every non-terminal node exists because a route's membership changed
        # there, which is a merge or a divergence by definition; a transversal
        # crossing shares no STEP and therefore never reaches this table.
        kind = NODE_TERMINAL if terminal_id is not None else NODE_JUNCTION
        graph_nodes[node_id] = CorridorNode(
            node_id=node_id,
            cell_index=cell,
            coords=_cell_to_xy(cell, cols, transform),
            kind=kind,
            terminal_id=terminal_id,
            segment_ids=tuple(sorted(node_cells[cell]["segments"])),
        )

    graph_segments: dict[int, CorridorSegment] = {}
    for seg_id, record in seg_records.items():
        cells = record["cells"]
        metrics = _segment_metrics(cells, raster, cell_size, categories)
        if solver is not None:
            metrics["routing_cost"] = float(solver.price_route(cells))
        coords = [_cell_to_xy(int(c), cols, transform) for c in cells]
        graph_segments[seg_id] = CorridorSegment(
            segment_id=seg_id,
            node_a=node_ids[int(cells[0])],
            node_b=node_ids[int(cells[-1])],
            cell_indices=cells,
            coords=coords,
            geometry=LineString(coords) if len(coords) > 1
            else LineString([coords[0], coords[0]]),
            length=metrics["length_m"],
            metrics=metrics,
            use_count=len(record["members"]),
            members=tuple(sorted(record["members"])),
            metrics_basis=(
                "sum(cell value x 2D length) over the search raster"
                if solver is None else
                "sum(cell value x 2D length) over the search raster; "
                "routing_cost is the kernel step weight in cell units"),
        )

    prov = {
        "min_shared_length_m": float(min_shared_length_m),
        "n_demoted_steps": len(demoted),
        "n_routes": len(normalized),
        "cell_size_m": cell_size,
        "raster_shape": (rows, cols),
    }
    if provenance:
        prov.update(provenance)

    return CorridorGraph(
        segments=graph_segments,
        nodes=graph_nodes,
        terminal_nodes={tid: node_ids[cell] for tid, cell in by_id.items()
                        if cell in node_ids},
        pair_routes=pair_routes,
        route_starts=route_starts,
        crs=crs,
        cell_size=cell_size,
        construction="overlay",
        provenance=prov,
    )


def _canonical(run: np.ndarray) -> tuple[int, ...]:
    """Direction-independent key for a cell run.

    Two routes traversing one trench in opposite directions must resolve to
    ONE segment, so the key is the lexicographically smaller of the run and
    its reverse.
    """
    forward = tuple(int(c) for c in run)
    backward = forward[::-1]
    return forward if forward <= backward else backward


def _cell_to_xy(cell: int, cols: int, transform) -> tuple[float, float]:
    """Cell centre in CRS units."""
    row, col = divmod(int(cell), cols)
    x, y = transform * (col + 0.5, row + 0.5)
    return (float(x), float(y))


def _short_shared_runs(
        normalized, route_steps, membership, raster, cell_size, categories,
        min_shared_length_m: float,
) -> set[tuple[int, int]]:
    """Steps whose shared run is too short to be a real common trench.

    Two routes that graze for a couple of cells are two trenches, not one; the
    CIRED plan prunes those before they generate junctions.

    A run is measured per unordered route PAIR, not per exact membership set.
    Measuring by membership equality looks equivalent and is not: a third route
    that touches even one step of a long two-route trench changes that step's
    membership from ``{A, B}`` to ``{A, B, C}``, which splits the run into
    pieces that can each fall below the threshold -- and a genuinely long
    shared trench is then demoted in full. Walking the steps that a given pair
    both carry has no such failure mode, and it is what "these two routes share
    this stretch" means in the first place.

    A step survives the prune if ANY pair carrying it shares a long enough
    stretch there, so the prune can never turn a long shared stretch into an
    unshared one.
    """
    keep_shared: set[tuple[int, int]] = set()
    candidates: set[tuple[int, int]] = set()

    carried: dict[int, list[tuple[int, int]]] = {
        idx: route_steps[idx] for idx in range(len(normalized))}

    for route_idx, (_key, cells) in enumerate(normalized):
        keys = carried[route_idx]
        partners = {other for step in keys for other in membership[step]
                    if other != route_idx}
        for other in partners:
            shared_flags = [other in membership[step] for step in keys]
            for start, stop in _runs(shared_flags):
                run_cells = cells[start:stop + 1]
                length_m = float(calculate_path_metrics_numba(
                    raster, np.ascontiguousarray(run_cells, dtype=np.uint32),
                    categories)[0]) * cell_size
                run_keys = set(keys[start:stop])
                if length_m >= min_shared_length_m:
                    keep_shared |= run_keys
                else:
                    candidates |= run_keys

    return candidates - keep_shared


# ------------------------------------------------------- cell-level measures


def supercover_cells(cells: np.ndarray, cols: int) -> list[int]:
    """Every cell a route physically covers, in order, duplicates removed.

    A step from r2 upward crosses intermediate cells that never appear in the
    route's own cell list, yet a trench dug along it touches them. This is the
    ground the route occupies, which is what a cell-level sharing measure has
    to compare.
    """
    from pyorps.utils._traversal import intermediate_steps_numba

    covered: list[int] = []
    seen: set[int] = set()

    def push(idx: int) -> None:
        if idx not in seen:
            seen.add(idx)
            covered.append(idx)

    for a, b in zip(cells[:-1], cells[1:]):
        ra, ca = divmod(int(a), cols)
        rb, cb = divmod(int(b), cols)
        push(int(a))
        for dr, dc in intermediate_steps_numba(rb - ra, cb - ca):
            push((ra + int(dr)) * cols + (ca + int(dc)))
    push(int(cells[-1]))
    return covered


def cell_sharing_profile(
        routes,
        shape: tuple[int, int],
        radii: Sequence[int] = (0,),
        min_run: int = 2,
) -> dict[int, dict[str, float]]:
    """How much ground two or more routes cover, allowing a snap radius.

    The step-exact measure in :func:`corridor_graph_from_routes` misses two
    routes that cover the same ground with different step decompositions, and
    misses two routes that run one cell apart -- one trench in reality,
    disjoint cell sets on the grid. This is the sensitivity a reviewer asks
    for: repeat the coincidence test with the other route's footprint dilated
    by ``r`` cells and report how the shared fraction moves.

    ``min_run`` implements the CIRED plan's rule that a transversal crossing is
    not sharing: a coincidence must persist over at least that many consecutive
    cells of a route to count.

    Parameters:
        routes: as in :func:`corridor_graph_from_routes`.
        shape: ``(rows, cols)`` of the raster window.
        radii: snap radii in cells. 0 is the exact, tolerance-free test.
        min_run: minimum run of consecutive shared cells along a route.

    Returns:
        ``radius -> {'shared_cells', 'total_cells', 'shared_fraction'}``.
    """
    from scipy.ndimage import binary_dilation

    normalized = _normalize_routes(routes)
    rows, cols = shape
    footprints = [supercover_cells(cells, cols)
                  for _key, cells in normalized]

    union: set[int] = set()
    for fp in footprints:
        union |= set(fp)

    indices = [np.asarray(fp, dtype=np.int64) for fp in footprints]
    membership = [set(fp) for fp in footprints]

    profile: dict[int, dict[str, float]] = {}
    for radius in radii:
        # The run test is applied per OTHER ROUTE, not to the OR over all of
        # them. Or-ing first asks "is this cell shared with someone for
        # min_run cells", so two different routes crossing at adjacent cells
        # manufacture a run of 2 and read as sharing. The threshold also has
        # to grow with the radius: a dilated single crossing is already
        # 2*radius+1 cells wide, which would clear any fixed min_run from
        # r = 1 on and let pure crossings drive the whole sensitivity curve.
        effective_run = min_run + 2 * int(radius)
        shared: set[int] = set()

        if radius <= 0:
            # No dilation, so no raster at all: a set membership test answers
            # it, and a full-window boolean mask per route would be tens of
            # megabytes each on a real search window.
            for i, idx in enumerate(indices):
                for j, other in enumerate(membership):
                    if i == j:
                        continue
                    hit = np.fromiter((int(v) in other for v in idx),
                                      dtype=bool, count=idx.size)
                    for start, stop in _runs(hit):
                        if stop - start >= effective_run:
                            shared.update(int(v) for v in idx[start:stop])
        else:
            size = 2 * int(radius) + 1
            structure = np.ones((size, size), dtype=bool)
            # ONE dilated mask alive at a time. Holding all of them is
            # rows*cols bytes per route, which is ~900 MB for 28 routes on a
            # 5000x6600 window.
            for j, own in enumerate(indices):
                mask = np.zeros((rows, cols), dtype=bool)
                mask.flat[own] = True
                mask = binary_dilation(mask, structure)
                for i, idx in enumerate(indices):
                    if i == j:
                        continue
                    hit = mask.flat[idx]
                    for start, stop in _runs(hit):
                        if stop - start >= effective_run:
                            shared.update(int(v) for v in idx[start:stop])
                del mask

        profile[int(radius)] = {
            "shared_cells": float(len(shared)),
            "total_cells": float(len(union)),
            "shared_fraction": (len(shared) / len(union)) if union else 0.0,
        }
    return profile


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of the True runs in a boolean array."""
    out: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(flags):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out
