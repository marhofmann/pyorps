"""Constrained path and tower data structures."""

import math
from dataclasses import dataclass, field
from typing import Optional

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
    span_to_next_m: Optional[float]
    tower_id: int
    height_m: Optional[float] = None
    bisector_angle_rad: Optional[float] = None
    ground_area_m2: Optional[float] = None


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

    def __post_init__(self):
        """Compute total_cost from constrained cost components."""
        if self.total_cost is None:
            self.total_cost = (self.total_terrain_cost +
                               self.total_tower_cost +
                               self.total_angle_penalty_cost)

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
        """
        records = []
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
            records.append(rec)
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
        return base
