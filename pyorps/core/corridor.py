"""
Corridor graphs: routes reduced to shared trenches and the nodes between them.

A least-cost path is computed for one source-target pair at a time, and a
planning model that charges construction cost per route therefore excavates
every stretch two routes have in common more than once. A *corridor graph*
removes that double count: its nodes are the terminals plus the cells where
routes merge or diverge, and its edges are the stretches between them, each
carrying its construction cost exactly once.

Two constructions produce this object, and both are tolerance-free because a
raster route is an ordered list of cells over one grid, so "these two routes
coincide here" is an identity, not a proximity test:

* **Overlay** (:func:`pyorps.graph.corridor.corridor_graph_from_routes`) -
  segment a GIVEN set of routes at the cells where their membership changes.
  This is the construction that measures how much of an existing layout is
  shared.
* **Distance network** (:meth:`pyorps.PathFinder.build_corridor_graph`) -
  seed every terminal in one multi-source sweep, cut at the Voronoi boundary
  and contract the predecessor forest. This is the construction that GENERATES
  a candidate graph, in the sense of Mehlhorn's terminal distance network [1].

Metric naming follows the dual-metric plan's vocabulary
(``docs/superpowers/plans/2026-06-09-dual-metric-cost-model.md``): a segment
carries a metric VECTOR, never a single ``cost`` float, and the two accountings
are named rather than conflated:

``length_m``
    2D length in CRS units, the same quantity ``Path.total_length`` reports.
``construction_cost``
    ``sum(cell value x 2D length in metres)`` - the same recompute
    ``Path.total_cost`` performs, in EUR when the raster holds EUR/m.
``routing_cost``
    The value the SEARCH minimized, in CELL units. It is not the construction
    cost whenever a DEM, a gradient response or an objective quantization is in
    play; :attr:`CorridorSegment.metrics_basis` says which.
``length_by_category``
    Metres per raster cost value, so a caller can re-price a segment under a
    different cost table without re-routing.

Segment metrics are ADDITIVE by construction: every step of a route belongs to
exactly one segment, and both length and category length are accumulated per
step, so concatenating the segments of a route reproduces that route's own
metrics exactly (not approximately). ``tests/test_graph/test_corridor.py`` pins
that as a conservation law.

References:
    [1] Mehlhorn, K.: 'A faster approximation algorithm for the Steiner problem
        in graphs', Inf. Process. Lett., 1988, 27, (3), pp. 125-128
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

from shapely.geometry import LineString, Point

from pyorps.core.types import CoordinateList, CoordinateTuple, NodeList

#: Node kinds. A ``junction`` is the whole point of the object: routes merge or
#: diverge there, so it is a Steiner node DERIVED from terrain rather than
#: placed by geometric construction. A ``terminal`` is an input point. A
#: ``crossing`` is where routes touch without sharing a step - two trenches
#: that cross are not one trench, and nothing branches there electrically.
NODE_TERMINAL = "terminal"
NODE_JUNCTION = "junction"
NODE_CROSSING = "crossing"


@dataclass
class CorridorNode:
    """A cell promoted to a node of the corridor graph."""

    node_id: int
    cell_index: int
    coords: CoordinateTuple
    kind: str = NODE_JUNCTION
    #: Terminal index when ``kind == NODE_TERMINAL``, else None.
    terminal_id: int | None = None
    #: Ids of the segments incident to this node.
    segment_ids: tuple[int, ...] = ()

    @property
    def degree(self) -> int:
        return len(self.segment_ids)

    @property
    def geometry(self) -> Point:
        return Point(self.coords)


@dataclass
class CorridorSegment:
    """One stretch of trench between two corridor nodes.

    ``cell_indices`` is the ordered cell run from ``node_a`` to ``node_b``,
    endpoints included; consecutive entries differ by one entry of the
    neighbourhood step table, so a segment is a valid PYORPS route in its own
    right and can be re-priced by the same kernels.
    """

    segment_id: int
    node_a: int
    node_b: int
    cell_indices: NodeList
    coords: CoordinateList
    geometry: LineString
    length: float
    metrics: dict[str, Any]
    #: How many distinct routes use this segment. DIAGNOSTIC ONLY - see the
    #: warning on :attr:`CorridorGraph.pair_routes`.
    use_count: int = 1
    #: Which routes use it, by route id. Diagnostic only, same warning.
    members: tuple[int, ...] = ()
    #: What ``metrics['routing_cost']`` reflects, mirroring
    #: ``Path.total_cost_basis``.
    metrics_basis: str = ""

    @property
    def construction_cost(self) -> float:
        """Cost of digging this trench once, in raster cost units x metres."""
        return float(self.metrics.get("construction_cost", 0.0))

    @property
    def shared(self) -> bool:
        return self.use_count > 1

    def __str__(self) -> str:
        return (f"CorridorSegment(id={self.segment_id}, "
                f"{self.node_a}->{self.node_b}, "
                f"length={self.length:,.1f} m, "
                f"cost={self.construction_cost:,.0f}, "
                f"used_by={self.use_count})")


@dataclass
class CorridorOverlapReport:
    """How much of a set of routes is one trench rather than several.

    ``overcount`` is the quantity a per-route cost model overstates: the sum of
    the routes' own construction costs minus the sum over the segments they
    resolve into, where a shared segment is counted once. ``overstatement`` is
    that as a fraction of the per-route total, i.e. the share of the reported
    construction CAPEX that is an artefact of pricing routes independently.
    """

    n_routes: int
    n_segments: int
    n_nodes: int
    n_junctions: int
    #: Sum over routes of their own length / cost (the per-route accounting).
    route_length_m: float
    route_cost: float
    #: Sum over DISTINCT segments (the corridor accounting).
    corridor_length_m: float
    corridor_cost: float
    #: Length and cost carried by segments used by two or more routes.
    shared_length_m: float
    shared_cost: float
    #: shared_length_m / corridor_length_m.
    shared_fraction: float
    #: route_cost - corridor_cost.
    overcount: float
    #: overcount / route_cost.
    overstatement: float
    #: Length-weighted mean number of routes per shared segment.
    mean_multiplicity: float = 0.0
    #: Fraction of the covered CELLS (not of length) that two or more routes
    #: cover, from cell_sharing_profile. Wider than shared_fraction: it also
    #: counts ground two routes cover with different step decompositions,
    #: which the step-exact measure misses. Different denominator - do not
    #: read the two percentages as comparable.
    cell_shared_fraction: float | None = None
    notes: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = [
            "Corridor overlap report",
            "=" * 52,
            f"  routes                {self.n_routes}",
            f"  segments              {self.n_segments}",
            f"  nodes                 {self.n_nodes} "
            f"({self.n_junctions} derived junctions)",
            "",
            f"  per-route length      {self.route_length_m:>14,.1f} m",
            f"  corridor length       {self.corridor_length_m:>14,.1f} m",
            f"  shared length         {self.shared_length_m:>14,.1f} m"
            f"  ({self.shared_fraction * 100:.1f} % of corridor)",
            "",
            f"  per-route cost        {self.route_cost:>14,.0f}",
            f"  corridor cost         {self.corridor_cost:>14,.0f}",
            f"  overcount             {self.overcount:>14,.0f}"
            f"  ({self.overstatement * 100:.1f} % of per-route cost)",
        ]
        if self.cell_shared_fraction is not None:
            lines.append(
                f"  cell-level sharing    "
                f"{self.cell_shared_fraction * 100:>13.1f} % of covered cells")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_routes": self.n_routes,
            "n_segments": self.n_segments,
            "n_nodes": self.n_nodes,
            "n_junctions": self.n_junctions,
            "route_length_m": self.route_length_m,
            "route_cost": self.route_cost,
            "corridor_length_m": self.corridor_length_m,
            "corridor_cost": self.corridor_cost,
            "shared_length_m": self.shared_length_m,
            "shared_cost": self.shared_cost,
            "shared_fraction": self.shared_fraction,
            "overcount": self.overcount,
            "overstatement": self.overstatement,
            "mean_multiplicity": self.mean_multiplicity,
            "cell_shared_fraction": self.cell_shared_fraction,
            "notes": list(self.notes),
        }


@dataclass
class CorridorGraph:
    """Terminals plus derived junctions, joined by corridor segments.

    Warning:
        ``pair_routes``, ``CorridorSegment.use_count`` and
        ``CorridorSegment.members`` are DIAGNOSTIC. Handing them to an
        optimiser reintroduces the pairwise view the graph exists to remove:
        the whole point is that a segment is trenched once regardless of how
        many connections happen to run over it, so the count must not enter a
        cost term. They are kept for provenance, plotting and the conservation
        tests.
    """

    segments: dict[int, CorridorSegment] = field(default_factory=dict)
    nodes: dict[int, CorridorNode] = field(default_factory=dict)
    #: terminal index -> node id.
    terminal_nodes: dict[int, int] = field(default_factory=dict)
    #: (terminal_a, terminal_b) -> ordered segment ids. Diagnostic.
    pair_routes: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    #: route key -> the cell the route STARTS at. Segments are stored in a
    #: canonical orientation so that a trench two routes traverse in opposite
    #: directions is one object; without this, reassembling a route has
    #: nothing to orient the first segment against and returns it backwards
    #: about half the time.
    route_starts: dict[Any, int] = field(default_factory=dict)
    runtimes: dict[str, float] = field(default_factory=dict)
    crs: Any = None
    cell_size: float = 1.0
    #: How the graph was built: 'overlay' or 'distance_network'.
    construction: str = "overlay"
    #: Free-form provenance (neighbourhood, backend, parameters, warnings).
    provenance: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- summaries

    @property
    def total_length(self) -> float:
        """Trench length once, in CRS units."""
        return float(sum(s.length for s in self.segments.values()))

    @property
    def total_construction_cost(self) -> float:
        """Construction cost charged once per segment."""
        return float(sum(s.construction_cost for s in self.segments.values()))

    @property
    def shared_segments(self) -> list[CorridorSegment]:
        return [s for s in self.segments.values() if s.use_count > 1]

    @property
    def junctions(self) -> list[CorridorNode]:
        return [n for n in self.nodes.values() if n.kind == NODE_JUNCTION]

    def _resolve_key(self, key):
        """Find a route key, tolerating a swapped or unordered pair."""
        if key in self.pair_routes:
            return key
        if isinstance(key, tuple) and len(key) == 2:
            for candidate in ((key[1], key[0]),
                              (min(key), max(key)),
                              (max(key), min(key))):
                if candidate in self.pair_routes:
                    return candidate
        raise KeyError(f"no route {key!r} in this corridor graph")

    def route_segments(self, key) -> list[CorridorSegment]:
        """The segments one route resolves into, in route order."""
        return [self.segments[i]
                for i in self.pair_routes[self._resolve_key(key)]]

    def segments_of(self, terminal_a, terminal_b) -> list[CorridorSegment]:
        """The segments a pair's route resolves into, in order."""
        return self.route_segments((terminal_a, terminal_b))

    def route_cells(self, *key) -> list[int]:
        """Concatenated cell sequence of one route.

        Segments are stored in a direction-independent canonical orientation
        so that a trench two routes traverse in opposite directions is ONE
        object; reassembling a route therefore has to re-orient each segment
        against the running end rather than trusting its stored order. The
        result is exactly the cell list the pairwise search produced, which is
        what makes it a check rather than a restatement.
        """
        route_key = key[0] if len(key) == 1 else tuple(key)
        segments = self.route_segments(route_key)
        if not segments:
            return []

        first = list(segments[0].cell_indices)
        resolved = self._resolve_key(route_key)
        start_cell = self.route_starts.get(resolved)

        if len(segments) == 1:
            # A single-segment route has no neighbour to orient against, so
            # the orientation has to come from the route itself. Inferring it
            # from the key shape is what went wrong before: segments are
            # STORED in a canonical orientation, so about half of them are
            # reversed, and any key that is not a pair of terminal ids -- an
            # int from a plain list, a (a, b, rank) triple from k_per_pair > 1
            # -- silently returned the route backwards.
            if start_cell is None and isinstance(resolved, tuple) and len(
                    resolved) >= 2:
                node = self.terminal_nodes.get(resolved[0])
                if node is not None and node in self.nodes:
                    start_cell = self.nodes[node].cell_index
            if start_cell is not None and first and first[-1] == start_cell \
                    and first[0] != start_cell:
                first = first[::-1]
            return first

        if start_cell is not None and first and first[0] != start_cell \
                and first[-1] == start_cell:
            return self._chain(first[::-1], segments[1:])

        second = set(segments[1].cell_indices)
        if first[0] in second and first[-1] not in second:
            first = first[::-1]
        return self._chain(first, segments[1:])

    @staticmethod
    def _chain(head: list[int], rest) -> list[int]:
        """Append each segment to the running end, re-orienting as needed."""
        cells = list(head)
        for seg in rest:
            run = list(seg.cell_indices)
            if run[0] == cells[-1]:
                cells.extend(run[1:])
            elif run[-1] == cells[-1]:
                cells.extend(run[-2::-1])
            else:
                raise ValueError(
                    f"segment {seg.segment_id} does not attach to the route "
                    f"built so far: it runs {run[0]}..{run[-1]} and the route "
                    f"ends at {cells[-1]}")
        return cells

    # ---------------------------------------------------------------- export

    def to_geodataframe(self):
        """Segments as a GeoDataFrame, one row per segment."""
        from geopandas import GeoDataFrame

        records, geoms = [], []
        for seg in self.segments.values():
            record = {
                "segment_id": seg.segment_id,
                "node_a": seg.node_a,
                "node_b": seg.node_b,
                "length_m": seg.length,
                "construction_cost": seg.construction_cost,
                "routing_cost": seg.metrics.get("routing_cost"),
                "n_cells": len(seg.cell_indices),
                "use_count": seg.use_count,
                "members": ",".join(str(m) for m in seg.members),
                "shared": int(seg.use_count > 1),
            }
            for value, metres in sorted(
                    seg.metrics.get("length_by_category", {}).items()):
                record[f"length_cost_{int(value)}"] = float(metres)
            records.append(record)
            geoms.append(seg.geometry)
        return GeoDataFrame(records, geometry=geoms, crs=self.crs)

    def nodes_to_geodataframe(self):
        """Nodes as a GeoDataFrame, one row per node."""
        from geopandas import GeoDataFrame

        records, geoms = [], []
        for node in self.nodes.values():
            records.append({
                "node_id": node.node_id,
                "kind": node.kind,
                "terminal_id": node.terminal_id,
                "cell_index": int(node.cell_index),
                "degree": node.degree,
            })
            geoms.append(node.geometry)
        return GeoDataFrame(records, geometry=geoms, crs=self.crs)

    def save(self, path: str) -> None:
        """Write segments and nodes as two layers of one GeoPackage.

        An existing file is REPLACED, not added to. The nodes layer has to be
        written with ``mode="a"`` so it lands beside the segments instead of
        replacing them, and appending into a stale layer silently doubles every
        node on a re-run -- which then reads as twice as many terminals and
        junctions as the graph actually has.
        """
        from pathlib import Path as _Path

        target = _Path(path)
        if target.exists():
            target.unlink()
        self.to_geodataframe().to_file(target, layer="corridor_segments",
                                       driver="GPKG")
        self.nodes_to_geodataframe().to_file(target, layer="corridor_nodes",
                                             driver="GPKG", mode="a")

    # ---------------------------------------------------------------- report

    def overlap_report(
            self,
            route_metrics: dict[tuple[int, int], dict[str, float]] | None = None,
            cell_shared_fraction: float | None = None,
            notes: tuple[str, ...] = (),
    ) -> CorridorOverlapReport:
        """Compare the per-route accounting against the corridor accounting.

        Parameters:
            route_metrics: Optional ``(a, b) -> {'length_m':, 'construction_cost':}``
                measured on the ORIGINAL routes. When omitted the per-route
                totals are rebuilt from the segments and how often each is
                traversed, which does not independently witness the
                conservation law - pass the originals when you want the
                comparison to be a check rather than a restatement.
            cell_shared_fraction: Optional wider, cell-level sharing measure.
            notes: Caveats to carry into the report.
        """
        corridor_length = self.total_length
        corridor_cost = self.total_construction_cost
        shared_length = float(sum(s.length for s in self.shared_segments))
        shared_cost = float(
            sum(s.construction_cost for s in self.shared_segments))

        # APPEARANCES, not distinct routes. use_count is the size of a SET of
        # route ids, so it undercounts a route that traverses one segment more
        # than once (an out-and-back stub past a terminal), and the rebuilt
        # per-route total is then smaller than the truth while nothing raises.
        multiplicity = Counter(chain.from_iterable(self.pair_routes.values()))

        if route_metrics is None:
            route_length = float(sum(
                self.segments[i].length * n for i, n in multiplicity.items()))
            route_cost = float(sum(
                self.segments[i].construction_cost * n
                for i, n in multiplicity.items()))
            n_routes = len(self.pair_routes)
        else:
            route_length = float(sum(
                m["length_m"] for m in route_metrics.values()))
            route_cost = float(sum(
                m["construction_cost"] for m in route_metrics.values()))
            n_routes = len(route_metrics)

        weighted = sum(s.length * multiplicity.get(s.segment_id, s.use_count)
                       for s in self.shared_segments)
        mean_multiplicity = (weighted / shared_length) if shared_length else 0.0

        return CorridorOverlapReport(
            n_routes=n_routes,
            n_segments=len(self.segments),
            n_nodes=len(self.nodes),
            n_junctions=len(self.junctions),
            route_length_m=route_length,
            route_cost=route_cost,
            corridor_length_m=corridor_length,
            corridor_cost=corridor_cost,
            shared_length_m=shared_length,
            shared_cost=shared_cost,
            shared_fraction=(shared_length / corridor_length
                             if corridor_length else 0.0),
            overcount=route_cost - corridor_cost,
            overstatement=((route_cost - corridor_cost) / route_cost
                           if route_cost else 0.0),
            mean_multiplicity=mean_multiplicity,
            cell_shared_fraction=cell_shared_fraction,
            notes=tuple(notes),
        )

    def __str__(self) -> str:
        return (f"CorridorGraph(construction={self.construction!r}, "
                f"segments={len(self.segments)}, nodes={len(self.nodes)}, "
                f"junctions={len(self.junctions)}, "
                f"length={self.total_length:,.1f} m)")

    def __repr__(self) -> str:
        return str(self)
