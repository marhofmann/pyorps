"""Phase 3: cost-model editor + rasterization (R3, R5, F4/F11/F12)."""
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon

from pyorps.gui import ids
from pyorps.gui.services import cost_model, rasterize

from conftest import invoke

CRS = "EPSG:25832"


def _land_use_gdf():
    """A small 2-column ALKIS-like dataset with 4 x 100m2 squares."""
    def square(x, y, size=50):
        return Polygon([(x, y), (x + size, y), (x + size, y + size),
                        (x, y + size)])

    return gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Wald", "Weg", "Sumpf"],
         "bez": ["Nadelholz", "Laubholz", "", ""],
         "zone": [1, 2, 2, 3]},
        geometry=[square(0, 0), square(50, 0), square(0, 50),
                  square(50, 50)],
        crs=CRS)


# ----------------------------------------------------------- feature proposal
def test_propose_features_prefers_notebook_keys():
    proposed, candidates = cost_model.propose_features(_land_use_gdf())
    assert proposed == ("nutzart", "bez")
    assert "nutzart" in candidates and "bez" in candidates


# ---------------------------------------------------------------- seeding
def test_seed_notebook_defaults():
    seeded = cost_model.seed_assumptions(_land_use_gdf(),
                                         ("nutzart", "bez"))
    assert seeded["Wald"]["Nadelholz"] == 405
    assert seeded["Sumpf"][""] == 65535


def test_seed_zero_template_single_and_double():
    gdf = _land_use_gdf()
    single = cost_model.seed_assumptions(gdf, ("nutzart",))
    assert single["Wald"] == 0 and single[""] == 0     # catch-all (F4)
    double = cost_model.seed_assumptions(gdf, ("nutzart", "zone"))
    assert double["Wald"][""] == 0


# ------------------------------------------------------------- grid mapping
def test_grid_roundtrip_combination():
    assumptions = {"Wald": {"Nadelholz": 405, "": 405},
                   "Sumpf": {"": 65535}}
    rows = cost_model.grid_rows_from_assumptions(assumptions,
                                                 ("nutzart", "bez"))
    assert {r["nutzart"] for r in rows} == {"Wald", "Sumpf"}
    forbidden_row = next(r for r in rows if r["nutzart"] == "Sumpf")
    assert forbidden_row["forbidden"] is True          # F11 first-class
    back = cost_model.assumptions_from_grid_rows(rows, ("nutzart", "bez"))
    assert back == assumptions


def test_grid_roundtrip_single_and_forbidden_flag_wins():
    rows = [{"use": "forest", "cost": 10, "forbidden": False},
            {"use": "swamp", "cost": 5, "forbidden": True}]
    back = cost_model.assumptions_from_grid_rows(rows, ("use",))
    assert back == {"forest": 10, "swamp": 65535}


def test_grid_column_defs():
    defs = cost_model.grid_column_defs(("nutzart", "bez"))
    fields = [d["field"] for d in defs]
    assert fields == ["nutzart", "bez", "cost", "forbidden"]


# ------------------------------------------------------------ import/export
@pytest.mark.parametrize("ext", [".csv", ".json", ".xlsx"])
def test_table_roundtrip(tmp_path, ext):
    if ext == ".xlsx":
        pytest.importorskip("openpyxl")
    assumptions = {"Wald": {"Nadelholz": 405, "": 405},
                   "Weg": {"": 300}}
    path = tmp_path / f"costs{ext}"
    cost_model.export_table(assumptions, ("nutzart", "bez"), path)
    back, keys = cost_model.import_table(path)
    assert tuple(keys) == ("nutzart", "bez")
    assert back["Wald"]["Nadelholz"] == 405
    assert back["Weg"][""] == 300


def test_export_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        cost_model.export_table({"a": 1}, ("use",), tmp_path / "x.txt")


# ------------------------------------------------------------ modifier values
def test_parse_modifier_values():
    assert cost_model.parse_modifier_values("1.25") == 1.25
    assert cost_model.parse_modifier_values(2) == 2.0
    assert cost_model.parse_modifier_values('{"1": 100, "2": 2}') == \
        {"1": 100.0, "2": 2.0}
    with pytest.raises(ValueError, match="neither"):
        cost_model.parse_modifier_values("banana")
    with pytest.raises(ValueError, match="empty"):
        cost_model.parse_modifier_values("")


# ------------------------------------------------------------- rasterization
def test_build_cost_raster_base_and_modifiers(tmp_path):
    gdf = _land_use_gdf()
    assumptions = {"Wald": {"Nadelholz": 100, "Laubholz": 200, "": 100},
                   "Weg": {"": 300},
                   "Sumpf": {"": 65535}}
    # multiply modifier: zone 2 doubles the cost (Feature 5: condition-filtered)
    modifier = rasterize.ModifierSpec(
        gdf=cost_model.apply_condition(gdf, "zone", "==", "2"),
        mode="multiply", factor=2.0, name="zones", condition="zone == 2")
    path, log = rasterize.build_cost_raster(
        base_gdf=gdf, assumptions=assumptions,
        feature_keys=("nutzart", "bez"), resolution_in_m=1.0,
        modifiers=[modifier], work_dir=tmp_path)
    with rasterio.open(path) as src:
        data = src.read(1)
    values = set(np.unique(data))
    # Wald/Nadelholz 100 stays (zone 1); Laubholz 200 -> 400 (zone 2);
    # Weg 300 -> 600 (zone 2); Sumpf stays forbidden (ignore_value)
    assert {100, 400, 600, 65535} <= values
    assert any("modifier" in line for line in log)

    # cache: same config -> reuse without rebuilding
    path2, log2 = rasterize.build_cost_raster(
        base_gdf=gdf, assumptions=assumptions,
        feature_keys=("nutzart", "bez"), resolution_in_m=1.0,
        modifiers=[modifier], work_dir=tmp_path)
    assert path2 == path
    assert any("cache hit" in line for line in log2)


def test_build_cost_raster_override(tmp_path):
    gdf = _land_use_gdf()
    assumptions = {"Wald": 100, "Weg": 300, "Sumpf": 400}
    override = rasterize.ModifierSpec(
        gdf=gdf[gdf.nutzart == "Weg"], mode="override", factor=42.0,
        name="flat", condition="nutzart == Weg")
    path, _ = rasterize.build_cost_raster(
        base_gdf=gdf, assumptions=assumptions, feature_keys=("nutzart",),
        modifiers=[override], work_dir=tmp_path, use_cache=False)
    with rasterio.open(path) as src:
        data = src.read(1)
    assert 42 in np.unique(data)


def test_modifier_condition_operators():
    """Feature 5: (column, operator, value) filtering of a modifier dataset."""
    gdf = _land_use_gdf()   # zone = [1, 2, 2, 3], nutzart = [Wald,Wald,Weg,Sumpf]
    assert len(cost_model.apply_condition(gdf, "zone", "==", "2")) == 2
    assert len(cost_model.apply_condition(gdf, "zone", "!=", "2")) == 2
    assert len(cost_model.apply_condition(gdf, "zone", "<", "3")) == 3
    assert len(cost_model.apply_condition(gdf, "zone", ">=", 3)) == 1
    assert len(cost_model.apply_condition(gdf, "nutzart", "in",
                                          "Wald, Weg")) == 3
    assert len(cost_model.apply_condition(gdf, "bez", "is-empty", None)) == 2
    assert len(cost_model.apply_condition(gdf, "zone", "all", None)) == 4
    with pytest.raises(ValueError, match="not an attribute"):
        cost_model.apply_condition(gdf, "nope", "==", "x")
    with pytest.raises(ValueError, match="numeric"):
        cost_model.apply_condition(gdf, "nutzart", "<", "Wald")


def test_modifier_factor_and_label():
    assert cost_model.coerce_factor("1,25") == 1.25
    assert cost_model.coerce_factor(65535) == 65535.0
    with pytest.raises(ValueError, match="not a number"):
        cost_model.coerce_factor("banana")
    assert cost_model.condition_label("zone", "all", None) == "all features"
    assert cost_model.condition_label("zone", "==", "2") == "zone == 2"


def test_spec_cost_assumptions_is_scalar():
    gdf = _land_use_gdf()
    spec = rasterize.ModifierSpec(gdf=gdf, factor=3.0)
    assert spec.cost_assumptions() == 3.0
    assert rasterize.ModifierSpec(gdf=gdf[gdf.zone == 99]).is_empty


# --------------------------------------------------------- headless callbacks
@pytest.fixture()
def loaded_state(app, state):
    from pyorps.gui.services import geo

    gdf = _land_use_gdf()
    layer = state.add_layer("landuse", "vector", gdf=gdf, crs=gdf.crs,
                            geojson=geo.gdf_to_wgs84_geojson(gdf))
    return layer


def test_options_follow_layers(app, state, loaded_state):
    resp = invoke(app, (f"{ids.COST_DATASET}.options",),
                  state.layers_view())
    options = resp[ids.COST_DATASET]["options"]
    assert options == [{"label": "landuse", "value": loaded_state.id}]


def test_propose_callback(app, state, loaded_state):
    resp = invoke(app, (f"{ids.COST_FEATURE_KEYS}.options",),
                  loaded_state.id,
                  triggered=[f"{ids.COST_DATASET}.value"])
    assert resp[ids.COST_FEATURE_KEYS]["value"] == ["nutzart", "bez"]


def test_seed_callback_and_coverage(app, state, loaded_state):
    resp = invoke(app, (f"{ids.COST_GRID}.columnDefs", ids.COST_SEED_BTN),
                  1, loaded_state.id, ["nutzart", "bez"], [],
                  triggered=[f"{ids.COST_SEED_BTN}.n_clicks"])
    rows = resp[ids.COST_GRID]["rowData"]
    assert any(r["nutzart"] == "Wald" and r["cost"] == 405 for r in rows)
    grid_state = resp[ids.COST_GRID_STATE]["data"]
    assert grid_state["feature_keys"] == ["nutzart", "bez"]
    # notebook seed covers everything in this little dataset
    assert "✓" in resp[ids.COST_COVERAGE_INFO]["children"]


def test_rasterize_callback_end_to_end(app, state, loaded_state):
    keys = ("nutzart", "bez")
    assumptions = cost_model.seed_assumptions(loaded_state.gdf, keys)
    rows = cost_model.grid_rows_from_assumptions(assumptions, keys)
    grid_state = {"dataset_id": loaded_state.id,
                  "feature_keys": list(keys)}
    resp = invoke(
        app, (f"{ids.RASTERIZE_LOG}.children", ids.RASTERIZE_BTN),
        1, rows, grid_state, [], "none", 10, 4, 2, [],
        1.0, 65535, "uint16", 0, None, "viridis", [], "", "current",
        triggered=[f"{ids.RASTERIZE_BTN}.n_clicks"])
    log = resp[ids.RASTERIZE_LOG]["children"]
    assert "base raster" in log and "saved" in log
    rasters = state.layers_of_kind("raster")
    assert len(rasters) == 1
    assert rasters[0].meta["rasterize_config"]["resolution_in_m"] == 1.0


def test_rasterize_without_table_warns(app, state):
    resp = invoke(
        app, (f"{ids.RASTERIZE_LOG}.children", ids.RASTERIZE_BTN),
        1, [], {}, [], "none", 10, 4, 2, [],
        1.0, 65535, "uint16", 0, None, "viridis", [], "", "current",
        triggered=[f"{ids.RASTERIZE_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Seed a cost table first"
