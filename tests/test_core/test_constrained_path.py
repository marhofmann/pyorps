import unittest
import numpy as np
from shapely.geometry import Point, LineString

from pyorps.core.constrained_path import Tower, ConstrainedPath


class TestTower(unittest.TestCase):

    def test_tower_creation(self):
        tower = Tower(
            location=Point(472000, 5593400),
            cell_index=1234,
            tower_type="suspension",
            turn_angle_deg=3.5,
            terrain_cost=80000.0,
            angle_cost=60000.0,
            total_cost=140000.0,
            span_to_previous_m=250.0,
            span_to_next_m=300.0,
            tower_id=0,
        )
        self.assertEqual(tower.tower_type, "suspension")
        self.assertAlmostEqual(tower.total_cost, 140000.0)


class TestConstrainedPath(unittest.TestCase):

    def setUp(self):
        self.tower1 = Tower(
            location=Point(472000, 5593400),
            cell_index=100,
            tower_type="suspension",
            turn_angle_deg=2.0,
            terrain_cost=80000.0,
            angle_cost=60000.0,
            total_cost=140000.0,
            span_to_previous_m=250.0,
            span_to_next_m=300.0,
            tower_id=0,
        )
        self.tower2 = Tower(
            location=Point(472300, 5593400),
            cell_index=200,
            tower_type="heavy_angle",
            turn_angle_deg=18.0,
            terrain_cost=180000.0,
            angle_cost=300000.0,
            total_cost=480000.0,
            span_to_previous_m=300.0,
            span_to_next_m=None,
            tower_id=1,
        )
        self.path = ConstrainedPath(
            source=(472000.0, 5593400.0),
            target=(472500.0, 5593400.0),
            algorithm="constrained-dijkstra",
            graph_api="cython",
            path_indices=np.array([1, 50, 100, 150, 200, 2]),
            path_coords=np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
            path_geometry=LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]),
            euclidean_distance=500.0,
            runtimes={"pathfinding": 0.5},
            path_id=0,
            search_space_buffer_m=1000.0,
            neighborhood="r2",
            profile_name="overhead_line_380kv",
            towers=[self.tower1, self.tower2],
            n_towers=2,
            total_terrain_cost=50000.0,
            total_tower_cost=620000.0,
            total_angle_penalty_cost=1500.0,
            cost_breakdown={"terrain": 50000.0, "towers": 620000.0, "angle_penalties": 1500.0},
            spans=[250.0, 300.0],
            min_span_actual_m=250.0,
            max_span_actual_m=300.0,
            avg_span_m=275.0,
            turn_angles=[2.0, 18.0],
            max_turn_angle_deg=18.0,
            tower_type_counts={"suspension": 1, "heavy_angle": 1},
            tower_type_costs={"suspension": 140000.0, "heavy_angle": 480000.0},
        )

    def test_constrained_path_inherits_from_path(self):
        from pyorps.core.path import Path
        self.assertIsInstance(self.path, Path)

    def test_tower_count(self):
        self.assertEqual(self.path.n_towers, 2)

    def test_cost_breakdown(self):
        self.assertAlmostEqual(self.path.total_tower_cost, 620000.0)
        self.assertAlmostEqual(self.path.total_terrain_cost, 50000.0)

    def test_towers_to_geodataframe(self):
        gdf = self.path.towers_to_geodataframe()
        self.assertEqual(len(gdf), 2)
        self.assertIn("tower_type", gdf.columns)
        self.assertIn("turn_angle_deg", gdf.columns)
        self.assertIn("total_cost", gdf.columns)
        self.assertEqual(gdf.iloc[0]["tower_type"], "suspension")
        self.assertEqual(gdf.iloc[1]["tower_type"], "heavy_angle")

    def test_to_geodataframe_dict_includes_tower_summary(self):
        result = self.path.to_geodataframe_dict()
        self.assertIn("n_towers", result)
        self.assertIn("total_tower_cost", result)
        self.assertIn("profile_name", result)
        self.assertEqual(result["n_towers"], 2)
        self.assertEqual(result["profile_name"], "overhead_line_380kv")


if __name__ == "__main__":
    unittest.main()
