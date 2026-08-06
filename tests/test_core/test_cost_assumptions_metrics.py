"""Tests for multi-metric CostAssumptions leaves (feasibility model, Phase 1).

Covers dict leaves ({"cost": ..., "landscape": ...}), the weight/factor
resolution rule, forbidden propagation, per-metric GeoDataFrame columns,
file round trips, and — critically — pinning of every legacy behavior.
"""
import os
import tempfile
import unittest
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.core.exceptions import FormatError


def make_gdf(landuse, types=None):
    geometries = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])
        for i in range(len(landuse))
    ]
    data = {'landuse': landuse, 'geometry': geometries}
    if types is not None:
        data['type'] = types
    return gpd.GeoDataFrame(data, geometry='geometry')


MULTI = {
    "landuse": {
        "Forest": {"cost": 365, "landscape": 0.9, "permit": 0.6},
        "Agriculture": {"cost": 107, "landscape": 0.2},
        "Residential": 65535,
        "Grassland": 130,
    }
}


class TestMultiMetricParsing(unittest.TestCase):
    def test_flat_multi_metric(self):
        ca = CostAssumptions(MULTI)
        self.assertTrue(ca.is_multi_metric)
        self.assertEqual(ca.metric_names, ["cost", "landscape", "permit"])
        # cost view is scalar and public API compatible
        self.assertEqual(ca.cost_assumptions["Forest"], 365.0)
        self.assertEqual(ca.cost_assumptions["Grassland"], 130.0)
        # unspecified metrics default to 0
        self.assertEqual(ca.metric_assumptions["permit"]["Agriculture"], 0.0)
        self.assertEqual(ca.metric_assumptions["landscape"]["Grassland"], 0.0)

    def test_scalar_leaf_forbidden_propagates(self):
        ca = CostAssumptions(MULTI)
        for metric in ca.metric_names:
            self.assertEqual(
                ca.metric_assumptions[metric]["Residential"], 65535.0)

    def test_dict_leaf_forbidden_propagates_with_warning(self):
        source = {"landuse": {
            "Forest": {"cost": 365, "landscape": 0.9},
            "Water": {"cost": 65535, "landscape": 0.4},
        }}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ca = CostAssumptions(source)
        self.assertTrue(any("forbidden" in str(w.message).lower()
                            for w in caught))
        for metric in ca.metric_names:
            self.assertEqual(ca.metric_assumptions[metric]["Water"], 65535.0)

    def test_weight_resolution_rule(self):
        source = {"landuse": {
            "Forest": {"cost": 365},                       # weight = cost
            "Agriculture": {"cost": 107, "factor": 3.0},   # weight = 321
            "Heath": {"cost": 220, "weight": 9000},        # explicit wins
            "Moor": {"cost": 50, "factor": 2.0, "weight": 400.0},
        }}
        ca = CostAssumptions(source)
        self.assertIn("weight", ca.metric_names)
        weight = ca.metric_assumptions["weight"]
        self.assertEqual(weight["Forest"], 365.0)
        self.assertEqual(weight["Agriculture"], 321.0)
        self.assertEqual(weight["Heath"], 9000.0)
        self.assertEqual(weight["Moor"], 400.0)  # explicit beats factor
        # 'factor' is a modifier, never a metric
        self.assertNotIn("factor", ca.metric_names)
        # weight metric exists only when factor/weight appear somewhere
        plain = CostAssumptions({"landuse": {"a": {"cost": 1.0}}})
        self.assertNotIn("weight", plain.metric_names)

    def test_scalar_leaves_feed_weight_metric(self):
        source = {"landuse": {
            "Forest": {"cost": 100, "factor": 2.0},
            "Grass": 30,
        }}
        ca = CostAssumptions(source)
        self.assertEqual(ca.metric_assumptions["weight"]["Grass"], 30.0)
        self.assertEqual(ca.metric_assumptions["weight"]["Forest"], 200.0)

    def test_validation_errors(self):
        with self.assertRaises(FormatError):  # key outside declared metrics
            CostAssumptions({"landuse": {"a": {"cost": 1, "landscpe": 2}}},
                            metrics=["landscape"])
        # without declaration, unknown names auto-discover as new metrics
        typo = CostAssumptions({"landuse": {"a": {"cost": 1, "landscpe": 2}}})
        self.assertIn("landscpe", typo.metric_names)
        with self.assertRaises(FormatError):  # negative metric
            CostAssumptions({"landuse": {"a": {"cost": -5}}})
        with self.assertRaises(FormatError):  # factor without cost
            CostAssumptions({"landuse": {"a": {"factor": 2.0}}})
        with self.assertRaises(FormatError):  # NaN
            CostAssumptions(
                {"landuse": {"a": {"cost": float("nan")}}})
        with self.assertRaises(FormatError):  # mixed metric/nested leaves
            CostAssumptions({"landuse": {
                "a": {"cost": 1.0},
                "b": {"dense": 2.0},
            }})

    def test_declared_metrics_disambiguate(self):
        # No reserved key in the leaf: only declared metrics make it a leaf
        source = {"landuse": {"Forest": {"landscape": 0.9}}}
        legacy = CostAssumptions(source)  # looks like nested side features
        self.assertFalse(legacy.is_multi_metric)
        declared = CostAssumptions(
            {"landuse": {"Forest": {"landscape": 0.9}}},
            metrics=["landscape"])
        self.assertTrue(declared.is_multi_metric)
        self.assertEqual(
            declared.metric_assumptions["landscape"]["Forest"], 0.9)
        self.assertEqual(declared.cost_assumptions["Forest"], 0.0)

    def test_tuple_key_multi_metric(self):
        source = {("landuse", "type"): {
            ("forest", "dense"): {"cost": 1.0, "landscape": 0.8},
            ("forest", ""): {"cost": 0.5},
            ("water", "river"): 5.0,
        }}
        ca = CostAssumptions(source)
        self.assertTrue(ca.is_multi_metric)
        self.assertEqual(ca.main_feature, "landuse")
        self.assertEqual(ca.side_features, ["type"])
        self.assertEqual(
            ca.metric_assumptions["landscape"][("forest", "dense")], 0.8)
        self.assertEqual(
            ca.metric_assumptions["landscape"][("water", "river")], 0.0)

    def test_nested_multi_metric(self):
        source = {"landuse": {
            "forest": {"dense": {"cost": 1.0, "landscape": 0.8},
                       "sparse": {"cost": 0.5}},
            "water": {"river": {"cost": 5.0, "landscape": 0.1}},
        }}
        ca = CostAssumptions(source)
        self.assertTrue(ca.is_multi_metric)
        self.assertEqual(
            ca.cost_assumptions["forest"]["dense"], 1.0)
        self.assertEqual(
            ca.metric_assumptions["landscape"]["forest"]["sparse"], 0.0)


class TestMultiMetricApply(unittest.TestCase):
    def test_apply_writes_one_column_per_metric(self):
        ca = CostAssumptions(MULTI)
        gdf = make_gdf(["Forest", "Agriculture", "Residential", "Grassland"])
        result = ca.apply_to_geodataframe(gdf)
        self.assertIn('cost', result.columns)
        self.assertIn('landscape', result.columns)
        self.assertIn('permit', result.columns)
        self.assertEqual(result.loc[0, 'cost'], 365.0)
        self.assertEqual(result.loc[0, 'landscape'], 0.9)
        self.assertEqual(result.loc[1, 'permit'], 0.0)
        self.assertEqual(result.loc[2, 'cost'], 65535.0)
        self.assertEqual(result.loc[2, 'landscape'], 65535.0)
        self.assertEqual(result.loc[3, 'landscape'], 0.0)

    def test_apply_nested_multi_metric(self):
        source = {"landuse": {
            "forest": {"dense": {"cost": 1.0, "landscape": 0.8},
                       "sparse": {"cost": 0.5, "landscape": 0.3}},
            "water": {"river": {"cost": 5.0, "landscape": 0.1}},
        }}
        ca = CostAssumptions(source)
        gdf = make_gdf(["forest", "forest", "water"],
                       types=["dense", "sparse", "river"])
        result = ca.apply_to_geodataframe(gdf, side_features="type")
        self.assertEqual(result.loc[0, 'cost'], 1.0)
        self.assertEqual(result.loc[0, 'landscape'], 0.8)
        self.assertEqual(result.loc[1, 'landscape'], 0.3)
        self.assertEqual(result.loc[2, 'cost'], 5.0)

    def test_metric_name_collision_with_feature_column(self):
        ca = CostAssumptions(
            {"landuse": {"Forest": {"cost": 1.0, "type": 2.0}}},
            metrics=["type"])
        gdf = make_gdf(["Forest"], types=["dense"])
        with self.assertRaises(FormatError):
            ca.apply_to_geodataframe(gdf, side_features="type")


class TestLegacyPinning(unittest.TestCase):
    """Legacy inputs must pass through the normalizer completely untouched."""

    def test_simple_dict_untouched(self):
        source = {"landuse": {"forest": 1.0, "water": 5.0}}
        ca = CostAssumptions(source)
        self.assertFalse(ca.is_multi_metric)
        self.assertEqual(ca.metric_names, ["cost"])
        # the cost view IS the original dict object — zero-copy alias
        self.assertIs(ca.metric_assumptions["cost"], ca.cost_assumptions)
        self.assertIs(ca.cost_assumptions, source["landuse"])

    def test_nested_dict_untouched(self):
        source = {"landuse": {
            "forest": {"dense": 1.0, "sparse": 0.5},
            "water": {"river": 5.0},
        }}
        ca = CostAssumptions(source)
        self.assertFalse(ca.is_multi_metric)
        self.assertIs(ca.cost_assumptions, source["landuse"])
        gdf = make_gdf(["forest", "forest", "water"],
                       types=["dense", "sparse", "river"])
        result = ca.apply_to_geodataframe(gdf, side_features="type")
        self.assertEqual(result.loc[0, 'cost'], 1.0)
        self.assertEqual(result.loc[1, 'cost'], 0.5)
        self.assertEqual(result.loc[2, 'cost'], 5.0)

    def test_tuple_dict_untouched(self):
        source = {("landuse", "type"): {
            ("forest", "dense"): 1.0,
            ("forest", ""): 0.5,
        }}
        ca = CostAssumptions(source)
        self.assertFalse(ca.is_multi_metric)
        self.assertIs(ca.cost_assumptions, source[("landuse", "type")])

    def test_int_keyed_modifier_dict_untouched(self):
        # Raster-modifier style: {field: {zone_id: factor}}
        source = {"zone": {1: 100, 2: 2, 3: 1.5}}
        ca = CostAssumptions(source)
        self.assertFalse(ca.is_multi_metric)
        self.assertEqual(ca.cost_assumptions, {1: 100, 2: 2, 3: 1.5})

    def test_legacy_csv_multiple_numerics_without_cost_header(self):
        """Numeric extras without a 'cost' header keep legacy hierarchy
        semantics: first numeric column is the cost, the rest join the
        feature hierarchy."""
        ca = CostAssumptions()
        df = pd.DataFrame({
            'landuse': ['forest', 'water'],
            'zone': [7, 8],            # numeric, first => cost column
            'value': [1.5, 2.5],       # numeric, second => hierarchy
        })
        result = ca.convert_df_to_cost_dict(df)
        self.assertEqual(result, {('forest', 1.5): 7, ('water', 2.5): 8})


class TestMultiMetricFileRoundTrips(unittest.TestCase):
    def setUp(self):
        self.ca = CostAssumptions(MULTI)

    def _roundtrip(self, suffix, save, load_kwargs=None):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            path = tmp.name
        try:
            save(path)
            loaded = CostAssumptions(path, **(load_kwargs or {}))
            self.assertEqual(loaded.metric_names, self.ca.metric_names)
            for metric in self.ca.metric_names:
                self.assertEqual(
                    dict(loaded.metric_assumptions[metric]),
                    dict(self.ca.metric_assumptions[metric]),
                    msg=f"metric '{metric}' did not survive {suffix}")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_csv_roundtrip(self):
        self._roundtrip('.csv', self.ca.to_csv)

    def test_json_roundtrip(self):
        self._roundtrip('.json', self.ca.to_json)

    def test_excel_roundtrip(self):
        self._roundtrip('.xlsx', self.ca.to_excel)

    def test_csv_has_metric_columns(self):
        df = self.ca.cost_dict_to_df(self.ca.cost_assumptions)
        for column in ("landuse", "cost", "landscape", "permit"):
            self.assertIn(column, df.columns)

    def test_json_metadata_lists_metrics(self):
        import json
        with tempfile.NamedTemporaryFile(suffix='.json',
                                         delete=False) as tmp:
            path = tmp.name
        try:
            self.ca.to_json(path)
            with open(path, encoding='ISO-8859-15') as f:
                data = json.load(f)
            self.assertEqual(data['metadata']['metrics'],
                             ["cost", "landscape", "permit"])
            self.assertEqual(
                data['cost_assumptions']['Forest']['landscape'], 0.9)
        finally:
            os.unlink(path)

    def test_csv_weight_column_roundtrip(self):
        ca = CostAssumptions({"landuse": {
            "Forest": {"cost": 100, "factor": 2.0},
            "Grass": {"cost": 30},
        }})
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            path = tmp.name
        try:
            ca.to_csv(path)
            loaded = CostAssumptions(path)
            self.assertEqual(
                loaded.metric_assumptions["weight"]["Forest"], 200.0)
            self.assertEqual(
                loaded.metric_assumptions["weight"]["Grass"], 30.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
