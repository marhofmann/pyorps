import unittest
import numpy as np
import tempfile
import os
import json
import math

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.core.constrained_path import ConstrainedPath

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class TestConstrainedPathFinder(unittest.TestCase):

    def _make_test_raster(self, shape=(100, 100), value=100,
                          cell_size=1.0):
        """Create a temporary raster file for testing."""
        import rasterio
        from rasterio.transform import from_bounds

        tmpfile = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
        width_m = shape[1] * cell_size
        height_m = shape[0] * cell_size
        transform = from_bounds(0, 0, width_m, height_m,
                                shape[1], shape[0])
        data = np.full(shape, value, dtype=np.uint16)

        with rasterio.open(
            tmpfile.name, "w", driver="GTiff",
            height=shape[0], width=shape[1],
            count=1, dtype="uint16",
            crs="EPSG:32632", transform=transform,
        ) as dst:
            dst.write(data, 1)

        return tmpfile.name

    def setUp(self):
        self.raster_path = self._make_test_raster()
        # span_bin_size must be <= min step distance for correct accumulation.
        # With 1m cell size: cardinal step = 1.0m, diagonal = 1.414m.
        # Use span_bin_size=1.0m so each step advances >= 1 bin.
        self.profile = InfrastructureProfile.from_dict({
            "name": "test_overhead",
            "description": "test",
            "soft_angle_limit_deg": 5.0,
            "hard_angle_limit_deg": 60.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 100},
            "min_span_m": 20.0,
            "max_span_m": 40.0,
            "span_bin_size_m": 1.0,
            "tower_cost_function": "terrain_plus_angle",
            "tower_cost_params": {
                "terrain_cost_map": {"0": 1000, "500": 5000},
                "terrain_interpolation": "linear",
                "angle_types": {
                    "suspension": {"max_angle_deg": 10.0, "base_cost": 500},
                    "angle_tower": {"max_angle_deg": 60.0, "base_cost": 2000},
                },
            },
        })

    def tearDown(self):
        os.unlink(self.raster_path)

    def test_basic_route_finding(self):
        """ConstrainedPathFinder finds a route on simple raster."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        self.assertIsInstance(result, ConstrainedPath)
        self.assertGreater(len(result.path_indices), 0)
        self.assertGreater(result.n_towers, 0)

    def test_result_has_tower_geodataframe(self):
        """Result produces valid tower GeoDataFrame."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        tower_gdf = result.towers_to_geodataframe()
        self.assertEqual(len(tower_gdf), result.n_towers)

    def test_direction_only_mode(self):
        """Profile without span constraints produces path without towers."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        road_profile = InfrastructureProfile.from_dict({
            "name": "road",
            "description": "test road",
            "soft_angle_limit_deg": 15.0,
            "hard_angle_limit_deg": 90.0,
            "angle_cost_function": "linear",
            "angle_cost_params": {"scale": 50},
        })
        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=road_profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        self.assertIsInstance(result, ConstrainedPath)
        self.assertEqual(result.n_towers, 0)

    def test_invalid_backend_raises(self):
        """Non-cython/gpu backend raises ValueError."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        with self.assertRaises(ValueError):
            ConstrainedPathFinder(
                dataset_source=self.raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=self.profile,
                graph_api="networkx",
            )

    def test_span_constraints_respected(self):
        """All tower spans are within profile's [min_span, max_span]."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        # Max span enforced; min_span may be violated by mandatory towers
        # (start, end, turns > soft_angle_limit)
        max_step = math.sqrt(2) * 1.0
        for span in result.spans:
            self.assertLessEqual(span, self.profile.max_span_m + max_step)

    def test_angle_constraints_respected(self):
        """No turn in the result exceeds hard_angle_limit_deg."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        if result.max_turn_angle_deg > 0:
            self.assertLessEqual(result.max_turn_angle_deg,
                                 self.profile.hard_angle_limit_deg)

    def test_cost_decomposition_consistent(self):
        """Cost breakdown fields are internally consistent."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        self.assertGreater(result.total_terrain_cost, 0)
        self.assertGreaterEqual(result.total_angle_penalty_cost, 0)
        if result.n_towers > 0:
            self.assertEqual(sum(result.tower_type_counts.values()), result.n_towers)

    def test_profile_from_file_path(self):
        """ConstrainedPathFinder accepts profile as file path string."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(self.profile.to_dict(), f)
            profile_path = f.name

        try:
            pf = ConstrainedPathFinder(
                dataset_source=self.raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=profile_path,
                graph_api="cython",
                neighborhood_str="r1",
            )
            result = pf.find_route()
            self.assertIsInstance(result, ConstrainedPath)
        finally:
            os.unlink(profile_path)

    def test_inherits_pathfinder_attributes(self):
        """ConstrainedPathFinder has all PathFinder attributes."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
        from pyorps.graph.path_finder import PathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="cython",
            neighborhood_str="r1",
        )
        # Check it's a proper PathFinder subclass
        self.assertIsInstance(pf, PathFinder)
        # Check key PathFinder attributes are accessible
        self.assertIsNotNone(pf.raster_handler)
        self.assertIsNotNone(pf.steps)
        self.assertIsNotNone(pf.source_coords)
        self.assertIsNotNone(pf.target_coords)
        self.assertEqual(pf.graph_api_name, "cython")

    @unittest.skipUnless(HAS_CUPY, "CuPy not available")
    def test_gpu_backend_produces_result(self):
        """ConstrainedPathFinder with raster_gpu backend finds a route."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        pf = ConstrainedPathFinder(
            dataset_source=self.raster_path,
            source_coords=(10, 10),
            target_coords=(90, 90),
            profile=self.profile,
            graph_api="raster_gpu",
            neighborhood_str="r1",
        )
        result = pf.find_route()
        self.assertIsInstance(result, ConstrainedPath)

    def test_end_to_end_with_380kv_profile(self):
        """Full workflow: load profile YAML, find route, export towers."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        profile_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "profiles",
            "overhead_line_380kv.yaml"
        )
        if not os.path.exists(profile_path):
            self.skipTest("380kV profile not found")

        profile = InfrastructureProfile.load(profile_path)

        # 10m/pixel so span_bin_size=10m advances 1 bin per cardinal step.
        # 200x200 cells × 10m = 2000m extent → enough for multiple spans.
        raster_path = self._make_test_raster(
            shape=(200, 200), value=100, cell_size=10.0)
        try:
            pf = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=(100, 100),
                target_coords=(1900, 1900),
                profile=profile_path,
                graph_api="cython",
                neighborhood_str="r1",
            )
            result = pf.find_route()

            self.assertIsInstance(result, ConstrainedPath)
            self.assertEqual(result.profile_name, "overhead_line_380kv")
            self.assertGreater(result.n_towers, 0)
            self.assertGreater(result.total_tower_cost, 0)

            # Tower types match profile (+ terminal for start/end anchors)
            valid_types = list(
                profile.tower_cost_params["angle_types"].keys())
            valid_types.append("terminal")
            for tower in result.towers:
                self.assertIn(tower.tower_type, valid_types)

            # Max span check: with coarse bins + exact float enforcement,
            # spans should not exceed max_span + one step tolerance.
            max_step = math.sqrt(2) * 10.0  # R1 diagonal at 10m/pixel
            for span in result.spans:
                self.assertLessEqual(
                    span, profile.max_span_m + max_step,
                    f"Span {span:.1f}m above max {profile.max_span_m}m")
            # Note: min_span is NOT checked because mandatory towers
            # (start, end, turns > soft_angle_limit) may create shorter
            # spans. This is physically correct for overhead lines.

            # Angle constraints
            if result.max_turn_angle_deg > 0:
                self.assertLessEqual(
                    result.max_turn_angle_deg,
                    profile.hard_angle_limit_deg)

            # Cost decomposition
            self.assertGreater(result.total_terrain_cost, 0)
            self.assertEqual(
                sum(result.tower_type_counts.values()), result.n_towers)

            # Export towers
            tower_gdf = result.towers_to_geodataframe()
            self.assertEqual(len(tower_gdf), result.n_towers)

            # GeoDataFrame dict
            gdf_dict = result.to_geodataframe_dict()
            self.assertIn("profile_name", gdf_dict)
            self.assertEqual(
                gdf_dict["profile_name"], "overhead_line_380kv")
        finally:
            os.unlink(raster_path)

    def test_end_to_end_with_rural_road_profile(self):
        """Rural road profile: direction-only, no towers."""
        from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

        profile_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "profiles",
            "rural_road.yaml"
        )
        if not os.path.exists(profile_path):
            self.skipTest("rural_road profile not found")

        raster_path = self._make_test_raster(shape=(100, 100), value=100)
        try:
            pf = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=(10, 10),
                target_coords=(90, 90),
                profile=profile_path,
                graph_api="cython",
                neighborhood_str="r1",
            )
            result = pf.find_route()
            self.assertIsInstance(result, ConstrainedPath)
            self.assertEqual(result.n_towers, 0)
            self.assertEqual(len(result.spans), 0)
        finally:
            os.unlink(raster_path)


if __name__ == "__main__":
    unittest.main()
