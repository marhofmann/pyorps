"""Tests for MetricStack.combine: scalarization, pure-scaling quantization,
the zero-copy legacy fast path, and the resolution diagnostics.

Phase 3 of docs/superpowers/plans/2026-08-04-feasibility-multi-objective-routing.md
"""
import unittest
import warnings

import numpy as np
from rasterio.transform import from_origin

from pyorps.core.exceptions import MetricStackError
from pyorps.core.metric_stack import FORBIDDEN_VALUE, MetricStack
from pyorps.core.objective import Objective

TRANSFORM = from_origin(0.0, 100.0, 1.0, 1.0)
CRS = "EPSG:25832"


def stack_with_layers(**layers):
    stack = MetricStack(TRANSFORM, CRS)
    for name, values in layers.items():
        stack.add_layer(name, np.asarray(values, dtype=np.float32))
    return stack


class TestLegacyFastPath(unittest.TestCase):
    def setUp(self):
        self.raster = np.array([[10, 20], [65535, 30]], dtype=np.uint16)
        self.stack = MetricStack.from_single_raster(
            self.raster, TRANSFORM, CRS)

    def test_pure_cost_is_zero_copy(self):
        result = self.stack.combine(Objective.cheapest())
        self.assertTrue(result.legacy_passthrough)
        self.assertIs(result.weights, self.raster)   # THE original object
        self.assertEqual(result.scale, 1.0)
        self.assertIsNone(result.feasibility)
        self.assertTrue(self.stack.is_legacy_alias)  # still unmaterialized

    def test_scaled_cost_stays_zero_copy(self):
        """Pure scaling preserves the argmin — any {'cost': w} passes."""
        result = self.stack.combine(Objective({"cost": 4.0}))
        self.assertTrue(result.legacy_passthrough)
        self.assertIs(result.weights, self.raster)
        self.assertEqual(result.scale, 0.25)         # W = F * (1/w)

    def test_non_cost_objective_materializes(self):
        result = self.stack.combine(Objective.shortest())
        self.assertFalse(result.legacy_passthrough)
        self.assertFalse(self.stack.is_legacy_alias)


class TestCombine(unittest.TestCase):
    def test_weighted_sum_and_scaling(self):
        stack = stack_with_layers(
            cost=[[10.0, 20.0], [40.0, 0.0]],
            landscape=[[1.0, 0.0], [0.5, 0.25]],
        )
        objective = Objective({"cost": 2.0, "landscape": 100.0})
        result = stack.combine(objective)

        f_expected = 2.0 * stack["cost"] + 100.0 * stack["landscape"]
        np.testing.assert_allclose(result.feasibility, f_expected, rtol=1e-6)
        f_max = f_expected.max()
        self.assertAlmostEqual(result.scale, 65534.0 / f_max, places=6)
        w_expected = np.clip(np.rint(f_expected * result.scale),
                             1, 65534).astype(np.uint16)
        np.testing.assert_array_equal(result.weights, w_expected)
        self.assertAlmostEqual(result.resolution, f_max / 65534.0)

    def test_ratio_preservation(self):
        """Pure scaling: quantized weights preserve feasibility ratios."""
        stack = stack_with_layers(cost=[[100.0, 200.0], [400.0, 800.0]])
        result = stack.combine(Objective.cheapest())
        w = result.weights.astype(np.float64)
        self.assertAlmostEqual(w[0, 1] / w[0, 0], 2.0, places=3)
        self.assertAlmostEqual(w[1, 1] / w[0, 0], 8.0, places=3)
        self.assertEqual(result.weights[1, 1], 65534)  # max uses full range

    def test_zero_weight_layer_skipped(self):
        stack = stack_with_layers(
            cost=[[10.0, 20.0]],
            landscape=[[999.0, 0.0]],
        )
        result = stack.combine(Objective({"cost": 1.0, "landscape": 0.0}))
        np.testing.assert_allclose(result.feasibility, stack["cost"])

    def test_length_objective_is_uniform(self):
        stack = stack_with_layers(cost=[[10.0, 500.0], [3.0, 70.0]])
        result = stack.combine(Objective.shortest())
        self.assertTrue(np.all(result.weights == 65534))

    def test_length_term_adds_constant(self):
        stack = stack_with_layers(cost=[[10.0, 20.0]])
        result = stack.combine(Objective({"cost": 1.0, "length": 5.0}))
        np.testing.assert_allclose(result.feasibility,
                                   stack["cost"] + 5.0)

    def test_forbidden_and_floor(self):
        cost = np.array([[FORBIDDEN_VALUE, 1e6], [0.001, 1e6]],
                        dtype=np.float32)
        stack = stack_with_layers(cost=cost)
        result = stack.combine(Objective.cheapest())
        self.assertEqual(result.weights[0, 0], 65535)     # forbidden
        self.assertEqual(result.weights[0, 1], 65534)     # max
        self.assertEqual(result.weights[1, 0], 1)         # floor clip, not 0

    def test_unknown_layer_rejected(self):
        stack = stack_with_layers(cost=[[1.0, 2.0]])
        from pyorps.core.exceptions import ObjectiveError
        with self.assertRaises(ObjectiveError):
            stack.combine(Objective({"cost": 1.0, "landscape": 3.0}))

    def test_all_forbidden_rejected(self):
        cost = np.full((2, 2), FORBIDDEN_VALUE, dtype=np.float32)
        stack = stack_with_layers(cost=cost)
        with self.assertRaises(MetricStackError):
            stack.combine(Objective.cheapest())

    def test_zero_surface_warns_and_uniform(self):
        stack = stack_with_layers(cost=[[0.0, 0.0], [0.0, 0.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stack.combine(Objective.cheapest())
        self.assertTrue(any("uniform" in str(w.message) for w in caught))
        self.assertTrue(np.all(result.weights == 1))


class TestDiagnostics(unittest.TestCase):
    def test_class_collapse_warning(self):
        """A dominating constant crushes small class differences."""
        base = np.full((20, 20), 1_000_000.0, dtype=np.float32)
        base[:, :10] += 1.0    # two classes, 1.0 apart in float
        stack = stack_with_layers(cost=base)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stack.combine(Objective.cheapest())
        self.assertTrue(any("collapsed" in str(w.message) for w in caught))
        # the two classes are indeed indistinguishable after quantization
        self.assertEqual(len(np.unique(result.weights)), 1)

    def test_median_dominance_warning(self):
        """A single huge outlier pushes the median below 8 levels."""
        cost = np.full((20, 20), 0.001, dtype=np.float32)
        cost[0, 0] = 100.0
        stack = stack_with_layers(cost=cost)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stack.combine(Objective.cheapest())
        self.assertTrue(any("levels" in str(w.message) for w in caught))

    def test_healthy_surface_no_warnings(self):
        rng = np.random.default_rng(11)
        cost = (rng.random((20, 20)).astype(np.float32) * 400.0) + 100.0
        stack = stack_with_layers(cost=cost)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stack.combine(Objective.cheapest())
        self.assertEqual([str(w.message) for w in caught], [])


if __name__ == "__main__":
    unittest.main()
