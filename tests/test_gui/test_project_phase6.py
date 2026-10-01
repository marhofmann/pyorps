"""Phase 6: import/export round-trips + project persistence (R12)."""
import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from pyorps.gui import ids
from pyorps.gui.services import project_io

from conftest import SOURCE, TARGET, invoke
from test_routes_phase4 import _draft


@pytest.fixture()
def rich_state(app, state, raster_path):
    """A session with a vector layer, raster, and one built route."""
    from pyorps.gui.services import geo

    gdf = gpd.GeoDataFrame(
        {"use": ["forest"]},
        geometry=[Polygon([(500000, 5599800), (500200, 5599800),
                           (500200, 5600000)])], crs="EPSG:32632")
    state.add_layer("landuse", "vector", gdf=gdf, crs=gdf.crs,
                    geojson=geo.gdf_to_wgs84_geojson(gdf))
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    invoke(app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
           1, False, _draft({"source": [SOURCE], "target": [TARGET]}),
           raster_layer.id, "dijkstra", "cpu", "r1", 80, True, False,
           False, 1.0, 100, 0, False, [], [],
           triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    state.study_area = {"type": "Feature", "properties": {},
                        "geometry": {"type": "Polygon", "coordinates": [[
                            [9.0, 50.5], [9.01, 50.5], [9.01, 50.51],
                            [9.0, 50.5]]]}}
    return state


# ------------------------------------------------------------ route formats
@pytest.mark.parametrize("ext", [".geojson", ".gpkg", ".csv"])
def test_export_route_formats(rich_state, tmp_path, ext):
    route = rich_state.layers_of_kind("route")[0]
    path = project_io.export_route(route, tmp_path / f"route{ext}")
    if ext == ".csv":
        import pandas as pd

        frame = pd.read_csv(path, sep=";")
        assert "wkt" in frame.columns
        assert frame.iloc[0]["origin"] == "built"
    else:
        back = gpd.read_file(path)
        assert back.iloc[0]["origin"] == "built"
        assert json.loads(back.iloc[0]["control_points"])[0] == \
            pytest.approx(list(SOURCE))


def test_export_route_bad_format(rich_state, tmp_path):
    route = rich_state.layers_of_kind("route")[0]
    with pytest.raises(ValueError, match="Unsupported route format"):
        project_io.export_route(route, tmp_path / "route.docx")


def test_export_all_routes(rich_state, tmp_path):
    path = project_io.export_all_routes(rich_state, tmp_path / "all.gpkg")
    back = gpd.read_file(path)
    assert len(back) == 1


def test_export_raster(rich_state, tmp_path):
    raster = rich_state.layers_of_kind("raster")[0]
    out = project_io.export_raster(raster, tmp_path / "out.tif")
    import rasterio

    with rasterio.open(out) as src:
        assert src.width > 0


# ---------------------------------------------------------------- profiles
@pytest.mark.parametrize("ext", [".yaml", ".json"])
def test_profile_roundtrip(tmp_path, ext):
    profile = {"soft_angle_limit_deg": 6, "hard_angle_limit_deg": 35,
               "min_span_m": 50, "max_span_m": 300,
               "angle_types": {"suspension": {"max_angle_deg": 5,
                                              "base_cost": 40000}}}
    path = project_io.save_profile(profile, tmp_path / f"p{ext}")
    assert project_io.load_profile(path) == profile


# ------------------------------------------------------------ project files
def test_project_roundtrip(rich_state, tmp_path):
    cost_table = {"feature_keys": ["use"], "dataset_id": "x",
                  "rows": [{"use": "forest", "cost": 10,
                            "forbidden": False}],
                  "modifiers": []}
    manifest_path = project_io.save_project(rich_state, tmp_path / "proj",
                                            cost_table=cost_table)
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    assert manifest["version"] == 1
    assert len(manifest["layers"]) == 3
    assert manifest["cost_table"]["rows"][0]["use"] == "forest"

    n_layers = len(rich_state.layers)
    route_meta = rich_state.layers_of_kind("route")[0].meta

    # restore into the same (cleared) state
    back = project_io.load_project(rich_state, manifest_path)
    assert len(rich_state.layers) == n_layers
    assert rich_state.study_area is not None
    restored_route = rich_state.layers_of_kind("route")[0]
    assert restored_route.meta["origin"] == "built"
    assert restored_route.meta["control_points"] == \
        route_meta["control_points"]
    raster = rich_state.layers_of_kind("raster")[0]
    assert raster.tile is not None            # re-served
    vector = rich_state.layers_of_kind("vector")[0]
    assert vector.geojson["type"] == "FeatureCollection"


# --------------------------------------------------------- headless callbacks
def test_save_open_new_project_callbacks(app, rich_state, tmp_path):
    target = tmp_path / "session"
    resp = invoke(app, (f"{ids.PROJECT_STATUS}.children",
                        ids.PROJECT_SAVE_BTN),
                  1, str(target), ["raster", "vector", "cost_table"],
                  [{"use": "forest", "cost": 10, "forbidden": False}],
                  {"feature_keys": ["use"], "dataset_id": None}, [], [],
                  triggered=[f"{ids.PROJECT_SAVE_BTN}.n_clicks"])
    assert "saved" in resp[ids.PROJECT_STATUS]["children"]
    manifest_path = target / "project.json"
    assert manifest_path.exists()

    # new project clears everything
    resp = invoke(app, ("layers-view.data", ids.PROJECT_NEW_BTN), 1, [],
                  triggered=[f"{ids.PROJECT_NEW_BTN}.n_clicks"])
    assert rich_state.layers == {}

    # open restores layers + the cost table into the client stores
    resp = invoke(app, ("layers-view.data", ids.PROJECT_OPEN_BTN),
                  1, str(manifest_path), [],
                  triggered=[f"{ids.PROJECT_OPEN_BTN}.n_clicks"])
    assert len(rich_state.layers) == 3
    assert resp[ids.COST_GRID]["rowData"][0]["use"] == "forest"
    assert resp[ids.COST_GRID_STATE]["data"]["feature_keys"] == ["use"]
