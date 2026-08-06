"""Tests for the feasibility Objective and the gradient-response LUT builder.

Phase 1 of docs/superpowers/plans/2026-08-04-feasibility-multi-objective-routing.md
"""
import math
import unittest

import numpy as np

from pyorps.core.exceptions import ObjectiveError
from pyorps.core.objective import (
    ADDITIVE_MODELS,
    MULTIPLIER_MODELS,
    GradientOptions,
    Objective,
)

R1_STEPS = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.int8)


def legacy_penalty(name: str, s_pct: float) -> float:
    """Scalar reference replicas of _traversal_numba.calculate_gradient_penalty.

    The legacy code takes slope as a rise/run ratio; our domain is percent.
    """
    slope = s_pct / 100.0
    if name == "exponential":
        return min(math.exp(slope * slope * 2.0), 10000.0)
    if name == "power":
        n = abs(math.atan(slope)) / (math.pi / 2.0)
        return min(1.0 + (100.0 * n) ** (1.0 + 4.0 * n), 10000.0)
    if name == "energy":
        stretch = math.sqrt(1.0 + slope * slope)
        g = math.sin(math.atan(slope))
        return min(stretch * math.exp(3.0 * g * g), 10000.0)
    if name == "sigmoid":
        deg = math.degrees(math.atan(slope))
        x = (abs(deg) - 15.0) * 0.2
        return 1.0 + 999.0 / (1.0 + math.exp(-x))
    # "squared" / default
    n = abs(math.atan(slope)) / (math.pi / 2.0)
    return 1.0 + 400.0 * n * n


class TestObjectiveBasics(unittest.TestCase):
    def test_default_is_cheapest(self):
        obj = Objective()
        self.assertEqual(obj.weights, {"cost": 1.0})
        self.assertEqual(obj, Objective.cheapest())
        self.assertFalse(obj.has_gradient_terms)
        self.assertEqual(obj.required_layers, ["cost"])

    def test_presets(self):
        self.assertEqual(Objective.shortest().weights, {"length": 1.0})
        gentle = Objective.gentlest()
        self.assertEqual(gentle.weights, {"gradient": 1.0, "length": 0.01})
        # gradient weighted => additive response defaults to linear
        self.assertEqual(gentle.gradient_options.additive, "linear")
        # reserved layers need no raster layer behind them
        self.assertEqual(gentle.required_layers, [])
        with self.assertRaises(ObjectiveError):
            Objective.gentlest(length_eps=0.0)

    def test_weight_validation(self):
        with self.assertRaises(ObjectiveError):
            Objective({})
        with self.assertRaises(ObjectiveError):
            Objective({"cost": -1.0})
        with self.assertRaises(ObjectiveError):
            Objective({"cost": float("nan")})
        with self.assertRaises(ObjectiveError):
            Objective({"cost": float("inf")})
        with self.assertRaises(ObjectiveError):
            Objective({"cost": "expensive"})
        with self.assertRaises(ObjectiveError):
            Objective({"cost": 0.0, "landscape": 0.0})  # all-zero
        with self.assertRaises(ObjectiveError):
            Objective({"": 1.0})
        # zero weights are allowed alongside a positive one
        obj = Objective({"cost": 1.0, "landscape": 0.0})
        self.assertEqual(obj["landscape"], 0.0)
        self.assertEqual(obj.required_layers, ["cost"])

    def test_validate_layers(self):
        obj = Objective({"cost": 1.0, "landscape": 2.0, "length": 0.1})
        obj.validate_layers({"cost", "landscape"})  # ok, length reserved
        with self.assertRaises(ObjectiveError) as ctx:
            obj.validate_layers({"cost"})
        self.assertIn("landscape", str(ctx.exception))

    def test_getitem_missing_is_zero(self):
        obj = Objective({"cost": 1.0})
        self.assertEqual(obj["gradient"], 0.0)

    def test_from_priorities(self):
        obj = Objective.from_priorities(
            {"cost": 5.0, "landscape": 3.0, "length": 2.0},
            scales={"cost": 100.0, "landscape": 0.5},
        )
        self.assertAlmostEqual(obj["cost"], 0.05)
        self.assertAlmostEqual(obj["landscape"], 6.0)
        self.assertAlmostEqual(obj["length"], 2.0)  # reserved: scale 1
        with self.assertRaises(ObjectiveError):
            Objective.from_priorities({"permit": 1.0}, scales={})
        with self.assertRaises(ObjectiveError):
            Objective.from_priorities({"cost": 1.0}, scales={"cost": 0.0})

    def test_serialization_roundtrip(self):
        obj = Objective(
            {"cost": 1.0, "gradient": 40.0},
            gradient_options={"additive": "quadratic",
                              "multiplier": "exponential",
                              "multiplier_params": {"scale": 1.5},
                              "max_gradient_pct": 35.0},
        )
        restored = Objective.from_dict(obj.to_dict())
        self.assertEqual(obj, restored)
        self.assertEqual(obj.fingerprint(), restored.fingerprint())
        # different weights => different fingerprint
        other = Objective({"cost": 2.0, "gradient": 40.0},
                          gradient_options=obj.gradient_options)
        self.assertNotEqual(obj.fingerprint(), other.fingerprint())
        with self.assertRaises(ObjectiveError):
            Objective.from_dict({"gradient_options": {}})

    def test_callable_response_not_restorable(self):
        obj = Objective({"gradient": 1.0, "length": 0.01},
                        gradient_options={"additive": lambda s: s * 2.0})
        payload = obj.to_dict()
        self.assertTrue(
            str(payload["gradient_options"]["additive"]).startswith(
                "callable:"))
        with self.assertRaises(ObjectiveError):
            Objective.from_dict(payload)

    def test_gradient_options_validation(self):
        with self.assertRaises(ObjectiveError):
            GradientOptions(bin_width_pct=0.0)
        with self.assertRaises(ObjectiveError):
            GradientOptions(max_gradient_pct=-5.0)
        with self.assertRaises(ObjectiveError):
            GradientOptions(additive="cubic")
        with self.assertRaises(ObjectiveError):
            GradientOptions(multiplier="parabolic")
        # hard limit above the domain extends the domain
        opts = GradientOptions(max_gradient_pct=250.0)
        self.assertGreater(opts.s_max_pct, 250.0)

    def test_repr_mentions_gradient_config(self):
        obj = Objective({"cost": 1.0},
                        gradient_options={"multiplier": "exponential",
                                          "max_gradient_pct": 30.0})
        text = repr(obj)
        self.assertIn("exponential", text)
        self.assertIn("30", text)


class TestGradientLUTs(unittest.TestCase):
    def test_stretch_only_geometry(self):
        """No responses configured: Γ_mult is the pure 3D stretch, Γ_add 0."""
        obj = Objective({"cost": 1.0})
        luts = obj.build_gradient_luts(R1_STEPS, cell_size=1.0)
        self.assertEqual(luts.n_bins,
                         int(math.ceil(200.0 / 0.25)))
        for b in (0, 10, 400, luts.n_bins - 1):
            s_center = (b + 0.5) * luts.bin_width_pct
            expected = math.sqrt(1.0 + (s_center / 100.0) ** 2)
            self.assertAlmostEqual(float(luts.mult[b]), expected, places=5)
        self.assertTrue(np.all(luts.add == 0.0))
        self.assertGreaterEqual(luts.max_finite_mult, 1.0)
        self.assertEqual(luts.max_add, 0.0)

    def test_legacy_multiplier_parity(self):
        """Each legacy model LUT = stretch * legacy penalty at bin centers."""
        for name in MULTIPLIER_MODELS:
            obj = Objective({"cost": 1.0},
                            gradient_options={"multiplier": name})
            luts = obj.build_gradient_luts(R1_STEPS, cell_size=1.0)
            for s_probe in (0.125, 5.125, 15.125, 30.125, 60.125):
                b = int(s_probe * luts.bin_inv)
                s_center = (b + 0.5) * luts.bin_width_pct
                stretch = math.sqrt(1.0 + (s_center / 100.0) ** 2)
                expected = stretch * legacy_penalty(name, s_center)
                self.assertAlmostEqual(
                    float(luts.mult[b]) / expected, 1.0, places=4,
                    msg=f"model={name}, s={s_center}")

    def test_multiplier_params_override(self):
        base = Objective({"cost": 1.0},
                         gradient_options={"multiplier": "exponential"})
        strong = Objective(
            {"cost": 1.0},
            gradient_options={"multiplier": "exponential",
                              "multiplier_params": {"scale": 6.0}})
        b = 200  # ~50% slope
        lut_base = base.build_gradient_luts(R1_STEPS, 1.0)
        lut_strong = strong.build_gradient_luts(R1_STEPS, 1.0)
        self.assertGreater(float(lut_strong.mult[b]),
                           float(lut_base.mult[b]))

    def test_additive_term_scaling(self):
        """add = w_gradient * g(s) * stretch * quant_scale."""
        w_gradient, quant_scale = 2.0, 0.5
        obj = Objective({"gradient": w_gradient, "length": 0.01},
                        gradient_options={"additive": "linear"})
        luts = obj.build_gradient_luts(R1_STEPS, cell_size=1.0,
                                       quant_scale=quant_scale)
        for b in (0, 40, 400):
            s_center = (b + 0.5) * luts.bin_width_pct
            stretch = math.sqrt(1.0 + (s_center / 100.0) ** 2)
            expected = w_gradient * s_center * stretch * quant_scale
            self.assertAlmostEqual(float(luts.add[b]), expected, places=4)
        # multiplicative table stays pure stretch
        self.assertAlmostEqual(
            float(luts.mult[0]),
            math.sqrt(1.0 + (0.125 / 100.0) ** 2), places=6)

    def test_additive_shapes_monotone(self):
        for name in ADDITIVE_MODELS:
            obj = Objective({"gradient": 1.0, "length": 0.01},
                            gradient_options={"additive": name})
            luts = obj.build_gradient_luts(R1_STEPS, 1.0)
            self.assertTrue(np.all(np.diff(luts.add) >= 0.0),
                            msg=f"additive model {name} not monotone")

    def test_hard_gradient_limit(self):
        obj = Objective({"cost": 1.0},
                        gradient_options={"max_gradient_pct": 30.0})
        luts = obj.build_gradient_luts(R1_STEPS, cell_size=1.0)
        below = int(29.0 * luts.bin_inv)
        above = int(31.0 * luts.bin_inv)
        self.assertTrue(np.isfinite(luts.mult[below]))
        self.assertTrue(np.isinf(luts.mult[above]))
        self.assertEqual(float(luts.add[above]), 0.0)
        # last bin (clamp target for extreme slopes) must be forbidden
        self.assertTrue(np.isinf(luts.mult[-1]))
        # max_finite_mult ignores the forbidden bins
        self.assertTrue(np.isfinite(luts.max_finite_mult))

    def test_inv_horiz_m(self):
        steps = np.array([[1, 0], [1, 1], [2, 1]], dtype=np.int8)
        obj = Objective({"cost": 1.0})
        luts = obj.build_gradient_luts(steps, cell_size=5.0)
        expected = [1.0 / (1.0 * 5.0),
                    1.0 / (math.sqrt(2.0) * 5.0),
                    1.0 / (math.sqrt(5.0) * 5.0)]
        np.testing.assert_allclose(luts.inv_horiz_m, expected, rtol=1e-6)

    def test_packed_layout(self):
        obj = Objective({"gradient": 1.0, "length": 0.01})
        luts = obj.build_gradient_luts(R1_STEPS, 1.0)
        self.assertEqual(luts.packed.shape, (luts.n_bins, 2))
        self.assertTrue(luts.packed.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(luts.packed[:, 0], luts.mult)
        np.testing.assert_array_equal(luts.packed[:, 1], luts.add)

    def test_custom_callable_responses(self):
        obj = Objective(
            {"gradient": 1.0, "length": 0.01},
            gradient_options={"additive": lambda s: s ** 1.5,
                              "multiplier": lambda s: 1.0 + s / 100.0})
        luts = obj.build_gradient_luts(R1_STEPS, 1.0)
        b = 100
        s_center = (b + 0.5) * luts.bin_width_pct
        stretch = math.sqrt(1.0 + (s_center / 100.0) ** 2)
        self.assertAlmostEqual(float(luts.mult[b]),
                               stretch * (1.0 + s_center / 100.0), places=4)
        self.assertAlmostEqual(float(luts.add[b]),
                               s_center ** 1.5 * stretch, places=3)

    def test_invalid_responses_rejected(self):
        negative_mult = Objective(
            {"cost": 1.0},
            gradient_options={"multiplier": lambda s: s - 100.0})
        with self.assertRaises(ObjectiveError):
            negative_mult.build_gradient_luts(R1_STEPS, 1.0)
        zero_mult = Objective(
            {"cost": 1.0},
            gradient_options={"multiplier": lambda s: np.zeros_like(s)})
        with self.assertRaises(ObjectiveError):
            zero_mult.build_gradient_luts(R1_STEPS, 1.0)
        negative_add = Objective(
            {"gradient": 1.0, "length": 0.01},
            gradient_options={"additive": lambda s: -s})
        with self.assertRaises(ObjectiveError):
            negative_add.build_gradient_luts(R1_STEPS, 1.0)

    def test_input_validation(self):
        obj = Objective({"cost": 1.0})
        with self.assertRaises(ObjectiveError):
            obj.build_gradient_luts(R1_STEPS, cell_size=0.0)
        with self.assertRaises(ObjectiveError):
            obj.build_gradient_luts(R1_STEPS, cell_size=1.0, quant_scale=0.0)
        with self.assertRaises(ObjectiveError):
            obj.build_gradient_luts(np.zeros((0, 2)), cell_size=1.0)
        with self.assertRaises(ObjectiveError):
            obj.build_gradient_luts(np.array([[0, 0]]), cell_size=1.0)

    def test_has_gradient_terms(self):
        self.assertFalse(Objective({"cost": 1.0}).has_gradient_terms)
        self.assertTrue(Objective({"gradient": 1.0, "length": 0.01})
                        .has_gradient_terms)
        self.assertTrue(
            Objective({"cost": 1.0},
                      gradient_options={"multiplier": "exponential"})
            .has_gradient_terms)
        self.assertTrue(
            Objective({"cost": 1.0},
                      gradient_options={"max_gradient_pct": 25.0})
            .has_gradient_terms)


if __name__ == "__main__":
    unittest.main()
