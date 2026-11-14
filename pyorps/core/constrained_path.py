"""Constrained path and tower data structures."""

import math
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from pyorps.core.path import Path


@dataclass
class Tower:
    """Represents an intermediate structure along the route."""

    location: Point
    cell_index: int
    tower_type: str
    turn_angle_deg: float
    terrain_cost: float
    angle_cost: float
    total_cost: float
    span_to_previous_m: float
    span_to_next_m: float | None
    tower_id: int
    height_m: float | None = None
    bisector_angle_rad: float | None = None
    ground_area_m2: float | None = None
    #: One sentence on a flagged tower (a short end span that was kept, a tower the end-span rule removed), else None.
    note: str | None = None
    #: The height premium the search charged for this tower's height class (several tower heights only), in the
    #: unit of ``total_cost``; 0.0 for a tower of the base height, with one tower height, and for a terminal tower
    #: (except the source, see ``ConstrainedPath.cost_breakdown``). It is part of ``total_cost``.
    height_premium: float = 0.0


@dataclass
class ConstrainedPath(Path):
    """Path with infrastructure constraint results.

    All constrained-specific fields have defaults to satisfy Python dataclass
    inheritance rules (parent Path has optional fields with defaults).
    """

    profile_name: str = ""
    towers: list[Tower] = field(default_factory=list)
    n_towers: int = 0
    total_terrain_cost: float = 0.0
    total_tower_cost: float = 0.0
    total_angle_penalty_cost: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    spans: list[float] = field(default_factory=list)
    min_span_actual_m: float = 0.0
    max_span_actual_m: float = 0.0
    avg_span_m: float = 0.0
    turn_angles: list[float] = field(default_factory=list)
    max_turn_angle_deg: float = 0.0
    tower_type_counts: dict[str, int] = field(default_factory=dict)
    tower_type_costs: dict[str, float] = field(default_factory=dict)
    #: The profile's ``terminal_span_rule`` this route was reported under.
    terminal_span_rule: str = "remove"
    #: Search towers the rule removed from ``towers`` (next to an end point), with the costs the search charged.
    #: Their ``tower_id`` is -1: they are not part of the line. Not counted in ``n_towers`` or ``total_tower_cost``;
    #: their summed cost is ``cost_breakdown["removed_towers"]`` (present only when a tower was removed).
    removed_towers: list[Tower] = field(default_factory=list)
    #: One line per flagged or broken span (a short end span that was kept, a merged span above the longest
    #: span, an end span that fails the clearance check). Empty when every span is in order.
    span_notes: list[str] = field(default_factory=list)
    #: The connection-point heights this route was searched and reported with (metres above the ground at the end
    #: cell); None when not set. A set height is the reported height of that end's terminal tower; the search used
    #: it for the end span when the clearance check was on (``cost_breakdown`` and ``search_notes`` say whether).
    source_connection_height_m: float | None = None
    target_connection_height_m: float | None = None
    #: The span-bin size (metres) the search used. The bins only merge search states: coarse bins (today's default,
    #: ``min_span_m``) can lose the optimum, so a run with ``terminal_span_rule="enforce"`` or with a connection
    #: height the search used takes bins no longer than the shortest step (exact). None when the route did not come
    #: from the finder, and for a profile without span limits (no span bins are used then).
    span_bin_size_m: float | None = None
    #: What the search did that the profile alone does not say (the exact span bins of an ``"enforce"`` or
    #: connection-height run, a height premium charged for the source). Empty for a default run.
    search_notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Fill the totals the constrained builders do not pass in."""
        if self.total_cost is None:
            self.total_cost = (self.total_terrain_cost +
                               self.total_tower_cost +
                               self.total_angle_penalty_cost)
        if self.total_length is None:
            # The constrained kernels never run PathFinder's reporting pass,
            # so nothing else populates this and every consumer summing it
            # silently accumulated 0.0. path_coords are already CRS
            # coordinates (the builder applies the transform per cell), so
            # the polyline length is metric directly — no cell-size factor,
            # unlike the cell-space kernel PathFinder has to scale.
            self.total_length = self._polyline_length(self.path_coords)

    @staticmethod
    def _polyline_length(coords) -> float:
        """Length of a coordinate polyline in CRS units; 0.0 if degenerate."""
        if coords is None:
            return 0.0
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            return 0.0
        steps = np.diff(points[:, :2], axis=0)
        return float(np.hypot(steps[:, 0], steps[:, 1]).sum())

    @staticmethod
    def _rotated_square(cx, cy, side, angle_rad):
        """Create a square polygon centered at (cx, cy), rotated by angle_rad."""
        half = side / 2.0
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotated = [
            (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            for dx, dy in corners
        ]
        rotated.append(rotated[0])
        return Polygon(rotated)

    def towers_to_geodataframe(self, crs=None) -> gpd.GeoDataFrame:
        """Export towers as polygon GeoDataFrame for GIS.

        Each tower geometry is a rotated square representing the foundation
        footprint. The square is rotated to bisect the angle between incoming
        and outgoing line segments. Falls back to point geometry when no
        ground area is set.

        A result without towers (an empty search result) gives an empty frame
        that still has a geometry column, the CRS and the columns every tower
        row has.
        """
        records = []
        with_notes = any(t.note for t in self.towers)
        with_premium = any(t.height_premium for t in self.towers)
        for t in self.towers:
            if (t.ground_area_m2 is not None and t.ground_area_m2 > 1.0
                    and t.bisector_angle_rad is not None):
                side = math.sqrt(t.ground_area_m2)
                geom = self._rotated_square(
                    t.location.x, t.location.y, side, t.bisector_angle_rad)
            else:
                geom = t.location

            rec = {
                "tower_id": t.tower_id,
                "geometry": geom,
                "tower_type": t.tower_type,
                "turn_angle_deg": t.turn_angle_deg,
                "terrain_cost": t.terrain_cost,
                "angle_cost": t.angle_cost,
                "total_cost": t.total_cost,
                "span_to_previous_m": t.span_to_previous_m,
                "span_to_next_m": t.span_to_next_m,
            }
            if t.height_m is not None:
                rec["height_m"] = t.height_m
            if t.ground_area_m2 is not None:
                rec["ground_area_m2"] = t.ground_area_m2
            if with_premium:  # a column only when a premium was charged: older exports keep their columns
                rec["height_premium"] = t.height_premium
            if with_notes:  # a column only when a tower says something: older exports keep their columns
                rec["note"] = t.note
            records.append(rec)
        if not records:
            # geopandas cannot assign a CRS to a frame without a geometry column
            empty = {"tower_id": [], "geometry": gpd.GeoSeries([], crs=crs), "tower_type": [],
                     "turn_angle_deg": [], "terrain_cost": [], "angle_cost": [], "total_cost": [],
                     "span_to_previous_m": [], "span_to_next_m": []}
            return gpd.GeoDataFrame(empty, geometry="geometry")
        return gpd.GeoDataFrame(records, crs=crs)

    def to_geodataframe_dict(self) -> dict:
        """Extended dict for GeoDataFrame export (includes tower summary)."""
        base = super().to_geodataframe_dict()
        base.update(
            {
                "profile_name": self.profile_name,
                "n_towers": self.n_towers,
                "total_tower_cost": self.total_tower_cost,
                "total_terrain_cost": self.total_terrain_cost,
                "total_angle_penalty_cost": self.total_angle_penalty_cost,
                "min_span_m": self.min_span_actual_m,
                "max_span_m": self.max_span_actual_m,
                "avg_span_m": self.avg_span_m,
                "max_turn_angle_deg": self.max_turn_angle_deg,
            }
        )
        for ttype, count in self.tower_type_counts.items():
            base[f"n_{ttype}_towers"] = count
            base[f"cost_{ttype}_towers"] = self.tower_type_costs[ttype]
        if self.removed_towers:  # only when the end-span rule removed a tower: older exports keep their keys
            base["n_removed_towers"] = len(self.removed_towers)
        for name in ("source_connection_height_m", "target_connection_height_m"):
            if getattr(self, name) is not None:  # only when set: older exports keep their keys
                base[name] = getattr(self, name)
        return base
