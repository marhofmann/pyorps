"""Tests for ConstrainedPath.__post_init__ total_cost computation."""

import unittest
import numpy as np
from shapely.geometry import Point, LineString

from pyorps.core.constrained_path import Tower, ConstrainedPath


def _make_path(**overrides):
    """Create a minimal ConstrainedPath for testing."""
    defaults = dict(
        source=(0, 0),
        target=(100, 0),
        algorithm="test",
        graph_api="cython",
        path_indices=np.array([0, 1, 2]),
        path_coords=np.array([[0, 0], [1, 0], [2, 0]]),
        path_geometry=LineString([(0, 0), (1, 0), (2, 0)]),
        euclidean_distance=100.0,
        runtimes={"pathfinding": 0.1},
        path_id=0,
        search_space_buffer_m=500.0,
        neighborhood="r1",
        profile_name="test",
        towers=[],
        n_towers=0,
        total_terrain_cost=1000.0,
        total_tower_cost=5000.0,
        total_angle_penalty_cost=200.0,
    )
    defaults.update(overrides)
    return ConstrainedPath(**defaults)


class TestPostInit(unittest.TestCase):

    def test_total_cost_auto_computed_when_none(self):
        """When total_cost is not passed, __post_init__ computes it from components."""
        path = _make_path(
            total_terrain_cost=1000.0,
            total_tower_cost=5000.0,
            total_angle_penalty_cost=200.0,
        )
        # total_cost not passed -> defaults to None -> __post_init__ computes
        self.assertAlmostEqual(path.total_cost, 6200.0)

    def test_total_cost_preserved_when_explicit(self):
        """When total_cost is explicitly passed, __post_init__ preserves it."""
        path = _make_path(
            total_terrain_cost=1000.0,
            total_tower_cost=5000.0,
            total_angle_penalty_cost=200.0,
            total_cost=999.0,
        )
        self.assertAlmostEqual(path.total_cost, 999.0)

    def test_total_cost_zero_components(self):
        path = _make_path(
            total_terrain_cost=0.0,
            total_tower_cost=0.0,
            total_angle_penalty_cost=0.0,
        )
        self.assertAlmostEqual(path.total_cost, 0.0)

    def test_towers_with_height(self):
        """Towers with height_m set appear in geodataframe."""
        t = Tower(
            location=Point(0, 0), cell_index=0, tower_type="suspension",
            turn_angle_deg=0.0, terrain_cost=1000.0, angle_cost=500.0,
            total_cost=1500.0, span_to_previous_m=100.0, span_to_next_m=100.0,
            tower_id=0, height_m=60.0,
        )
        path = _make_path(towers=[t], n_towers=1)
        gdf = path.towers_to_geodataframe()
        self.assertIn("height_m", gdf.columns)
        self.assertAlmostEqual(gdf.iloc[0]["height_m"], 60.0)


if __name__ == "__main__":
    unittest.main()
