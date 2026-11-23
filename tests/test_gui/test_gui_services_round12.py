"""
Round 12: GUI service-layer defects from the T5 robustness audit.

#2  open project is transactional (a missing asset keeps the session)
#5  rasterize cache key carries a content digest (a moved vertex is a miss)
#6  manual cost polygons keep name/cost by geometry after a middle delete
#14 loading / classifying a raster reads a decimated single band
#15 the dataset's own nodata drives the value range and the tile URL
#20 a failed combination-table build can be retried
#24 the notice history is capped, compact and diffed by id
#25 ProjectState survives concurrent mutation and reading
#26 save / open report skipped layers; WMS overlays round-trip
"""
from __future__ import annotations

import json
import sys
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from conftest import invoke
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from pyorps.gui import ids
from pyorps.gui.services import (
    combinations,
    errors,
    graduated,
    logbook,
    manual_cost,
    project_io,
    rasterize,
    tiles,
)

EMPTY_FC = {"type": "FeatureCollection", "features": []}
UTM = "EPSG:32632"


def _write_raster(path, data, *, nodata=None, crs=UTM):
    """A small GeoTIFF (1 m cells) with an optional declared nodata value."""
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(
            path, "w", driver="GTiff", height=array.shape[1],
            width=array.shape[2], count=array.shape[0], dtype=array.dtype,
            crs=crs, transform=from_origin(500000, 5600000, 1, 1),
            nodata=nodata) as dst:
        dst.write(array)
    return str(path)


def _saved_project(state, raster_path, tmp_path) -> Path:
    """vector + raster + route layers saved to ``tmp_path/proj``."""
    vector = gpd.GeoDataFrame({"use": ["forest"]}, geometry=[box(0, 0, 10, 10)],
                              crs=UTM)
    state.add_layer("landuse", "vector", gdf=vector, crs=vector.crs,
                    geojson=EMPTY_FC)
    tile = tiles.build_tile_layer(raster_path, name="cost",
                                  work_dir=state.work_dir)
    state.add_layer("cost", "raster", tile=tile, crs=UTM,
                    meta={"source_path": tile.source_path})
    route = gpd.GeoDataFrame({"name": ["r"]},
                             geometry=[LineString([(0, 0), (5, 5)])], crs=UTM)
    state.add_layer("route", "route", gdf=route, crs=route.crs,
                    geojson=EMPTY_FC, meta={"control_points": [[0, 0], [5, 5]]})
    return Path(project_io.save_project(state, tmp_path / "proj"))


# ======================================================== #2 transactional open
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_failed_open_leaves_the_session_untouched(state, raster_path, tmp_path,
                                                  monkeypatch, damage):
    manifest_path = _saved_project(state, raster_path, tmp_path)
    route = state.layers_of_kind("route")[0]
    asset = manifest_path.parent / "assets" / f"{route.id}.gpkg"
    if damage == "missing":
        asset.unlink()
    else:
        asset.write_bytes(b"this is not a geopackage")

    # unsaved work that opening the project must not destroy
    unsaved = state.add_layer("unsaved", "vector", geojson=EMPTY_FC)
    state.study_area = {"type": "Feature", "geometry": None}
    before = dict(state.layers)
    clients_before = dict(state.tiles._clients)

    # the raster is staged (its tile server started) BEFORE the route fails
    stopped: list[bool] = []
    real_build = tiles.build_tile_layer

    def spy(*args, **kwargs):
        tile = real_build(*args, **kwargs)
        real_stop = tile.tile_client.shutdown
        tile.tile_client.shutdown = lambda: (stopped.append(True), real_stop())
        return tile

    monkeypatch.setattr(tiles, "build_tile_layer", spy)

    with pytest.raises(errors.ProjectLoadError) as info:
        project_io.load_project(state, manifest_path)

    assert f"{route.id}.gpkg" in str(info.value)      # names the asset
    assert info.value.layer == "route"
    assert set(state.layers) == set(before)           # same layers ...
    assert all(state.layers[k] is v for k, v in before.items())   # same objects
    assert state.layers[unsaved.id] is unsaved
    assert state.study_area == {"type": "Feature", "geometry": None}
    assert state.dirty is True
    assert state.tiles._clients == clients_before     # live servers kept
    assert stopped == [True]                          # the staged one stopped


def test_open_failure_is_a_notice_that_names_the_asset(app, state, raster_path,
                                                       tmp_path):
    manifest_path = _saved_project(state, raster_path, tmp_path)
    vector = state.layers_of_kind("vector")[0]
    (manifest_path.parent / "assets" / f"{vector.id}.gpkg").unlink()
    ids_before = set(state.layers)

    resp = invoke(app, ("layers-view.data", ids.GO_OPEN_PROJECT),
                  1, str(manifest_path), [],
                  triggered=[f"{ids.GO_OPEN_PROJECT}.data"])

    notices = resp[ids.NOTICES]["data"]
    assert notices[-1]["title"] == "Project could not be opened"
    assert f"{vector.id}.gpkg" in notices[-1]["meaning"]
    assert "exactly as it was" in notices[-1]["impact"]
    assert set(state.layers) == ids_before


def test_a_good_open_still_replaces_the_session(state, raster_path, tmp_path):
    manifest_path = _saved_project(state, raster_path, tmp_path)
    state.add_layer("scratch", "vector", geojson=EMPTY_FC)
    manifest = project_io.load_project(state, manifest_path)
    assert sorted(ly.name for ly in state.layers.values()) == [
        "cost", "landuse", "route"]
    assert manifest["load_skipped"] == []
    assert state.layers_of_kind("raster")[0].tile is not None
    assert state.dirty is False


# ============================================= #26 skipped layers and WMS
def test_save_reports_a_raster_whose_file_is_gone(state, tmp_path):
    state.add_layer("Lost DEM", "raster",
                    meta={"source_path": str(tmp_path / "vanished.tif")})
    state.add_layer("kept", "vector", geojson=EMPTY_FC)

    with pytest.warns(errors.ProjectLayersSkippedWarning, match="Lost DEM"):
        path = project_io.save_project(state, tmp_path / "proj")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert [e["name"] for e in manifest["layers"]] == ["kept"]
    assert manifest["skipped_layers"][0]["name"] == "Lost DEM"
    assert "vanished.tif" in manifest["skipped_layers"][0]["reason"]
    assert state.dirty is True      # the lost layer exists only in this session

    # guard turns the warning into a box the user sees, naming the layer
    result, notices = errors.guard(project_io.save_project, state,
                                   tmp_path / "proj2")
    assert result is not None
    box_ = next(n for n in notices
                if n["title"] == "Some layers were left out of the project")
    assert box_["severity"] == "warning" and "Lost DEM" in box_["meaning"]


def test_wms_overlay_round_trips(state, tmp_path):
    wms = {"base_url": "https://example.test/wms", "layers": "dop20",
           "format": "image/png", "transparent": True, "version": "1.3.0",
           "bounds": [[50.5, 9.0], [50.6, 9.1]]}
    layer = state.add_layer("Aerial", "wms", crs="EPSG:3857",
                            meta={"wms": wms, "attribution": "(c) test"})
    layer.opacity = 0.4
    path = project_io.save_project(state, tmp_path / "proj")
    state.clear()

    with warnings.catch_warnings():
        warnings.simplefilter("error", errors.ProjectLayersSkippedWarning)
        manifest = project_io.load_project(state, path)

    restored = state.layers_of_kind("wms")
    assert len(restored) == 1 and restored[0].id == layer.id
    assert restored[0].meta["wms"] == wms
    assert restored[0].meta["attribution"] == "(c) test"
    assert restored[0].opacity == pytest.approx(0.4)
    assert manifest["load_skipped"] == []


def test_open_reports_entries_it_cannot_restore(state, tmp_path):
    manifest = {"version": 1, "project_crs": UTM, "layers": [
        {"id": "a", "name": "old raster", "kind": "raster", "meta": {}},
        {"id": "b", "name": "odd", "kind": "hologram", "meta": {}},
        {"id": "c", "name": "fine", "kind": "vector",
         "geojson": EMPTY_FC, "meta": {}}]}
    path = tmp_path / "project.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.warns(errors.ProjectLayersSkippedWarning, match="old raster"):
        result = project_io.load_project(state, path)

    assert [rec["name"] for rec in result["load_skipped"]] == [
        "old raster", "odd"]
    assert list(state.layers) == ["c"]


# ======================================================= #5 rasterize cache
def _zone(x_corner: float) -> gpd.GeoDataFrame:
    """One cost polygon; ``x_corner`` is where its top-right vertex sits."""
    return gpd.GeoDataFrame(
        {"cost": [900.0]}, crs="EPSG:25832",
        geometry=[Polygon([(5, 5), (25, 5), (x_corner, 25), (5, 25)])])


def _build(tmp_path, base, zone):
    spec = rasterize.ModifierSpec(gdf=zone, mode="override", factor=900.0,
                                  name="manual", condition="cost = 900")
    return rasterize.build_cost_raster(
        base_gdf=base, assumptions={"Wald": 100, "Weg": 300},
        feature_keys=("nutzart",), resolution_in_m=1.0, modifiers=[spec],
        work_dir=tmp_path)


def _base(use="Wald", junk="a"):
    return gpd.GeoDataFrame({"nutzart": [use], "junk": [junk]},
                            geometry=[box(0, 0, 40, 40)], crs="EPSG:25832")


def _is_hit(log):
    return any("cache hit" in line for line in log)


def test_rasterize_cache_misses_when_a_modifier_vertex_moves(tmp_path):
    path1, log1 = _build(tmp_path, _base(), _zone(25))
    path2, log2 = _build(tmp_path, _base(), _zone(25))
    assert not _is_hit(log1)
    assert _is_hit(log2) and path2 == path1           # same inputs: reused

    # one vertex moved: same polygon count, same factor, same total bounds
    path3, log3 = _build(tmp_path, _base(), _zone(15))
    assert not _is_hit(log3)
    assert path3 != path1
    with rasterio.open(path1) as a, rasterio.open(path3) as b:
        assert not np.array_equal(a.read(1), b.read(1))


def test_rasterize_cache_misses_when_a_used_attribute_changes(tmp_path):
    _, log1 = _build(tmp_path, _base("Wald"), _zone(25))
    _, log2 = _build(tmp_path, _base("Weg"), _zone(25))     # cost key differs
    assert not _is_hit(log1) and not _is_hit(log2)
    # an attribute column nothing reads does not invalidate the cache
    _, log3 = _build(tmp_path, _base("Weg", junk="zzz"), _zone(25))
    assert _is_hit(log3)


def test_gdf_digest_follows_geometry_and_named_columns_only():
    base = _base()
    assert rasterize.gdf_digest(base) == rasterize.gdf_digest(base.copy())
    moved = base.copy()
    moved.geometry = gpd.GeoSeries([box(0, 0, 40, 41)], crs=base.crs)
    assert rasterize.gdf_digest(moved) != rasterize.gdf_digest(base)
    other = base.copy()
    other["junk"] = "b"
    assert rasterize.gdf_digest(other) == rasterize.gdf_digest(base)
    assert rasterize.gdf_digest(other, ["junk"]) != \
        rasterize.gdf_digest(base, ["junk"])


# ================================================ #6 manual cost by geometry
def _feature(i: int) -> dict:
    x0 = 9.0 + 0.1 * i
    return {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[
            [x0, 50.5], [x0 + 0.05, 50.5], [x0 + 0.05, 50.55], [x0, 50.55],
            [x0, 50.5]]]}}


def _sync(state, features, prev):
    return manual_cost.sync_manual_layer(
        state, features, project_crs="EPSG:25832", default_cost=42,
        mode="override", name="Manual costs", prev_rows=prev)


def _four_edited_rows(state, features):
    _layer, rows = _sync(state, features, [])
    return [dict(row, name=f"P{i + 1}", cost=100.0 * (i + 1))
            for i, row in enumerate(rows)]       # the user edited the grid


def test_deleting_a_middle_polygon_keeps_costs_on_their_polygons(state):
    feats = [_feature(i) for i in range(4)]
    rows = _four_edited_rows(state, feats)

    # polygon 2 is deleted with the map's delete tool; the grid still has 4 rows
    layer, new_rows = _sync(state, [feats[0], feats[2], feats[3]], rows)

    assert [(r["name"], r["cost"]) for r in new_rows] == [
        ("P1", 100.0), ("P3", 300.0), ("P4", 400.0)]
    assert layer.gdf["cost"].tolist() == [100.0, 300.0, 400.0]


def test_removing_via_the_list_keeps_costs_on_their_polygons(state):
    feats = [_feature(i) for i in range(4)]
    rows = _four_edited_rows(state, feats)
    # what remove_manual_selected passes: the kept rows keep their old __row
    kept = [rows[0], rows[2], rows[3]]
    _layer, new_rows = _sync(state, [feats[0], feats[2], feats[3]], kept)
    assert [r["cost"] for r in new_rows] == [100.0, 300.0, 400.0]
    assert [r["__row"] for r in new_rows] == [0, 1, 2]


def test_an_edited_polygon_keeps_its_row_and_a_new_one_gets_the_default(state):
    feats = [_feature(i) for i in range(3)]
    rows = _four_edited_rows(state, feats)[:3]
    # a vertex of polygon 2 is moved (no geometric match -> index fallback),
    # and a 4th polygon is drawn (nothing to inherit -> default cost)
    moved = json.loads(json.dumps(feats[1]))
    moved["geometry"]["coordinates"][0][2] = [9.16, 50.56]
    _layer, new_rows = _sync(state, [feats[0], moved, feats[2], _feature(7)],
                             rows)
    assert [r["cost"] for r in new_rows] == [100.0, 200.0, 300.0, 42.0]
    assert new_rows[1]["name"] == "P2"


# ========================================= #14 decimated reads / #15 nodata
class _FakeBigRaster:
    """A dataset that refuses full-resolution reads (and records the calls)."""

    height, width, count, nodata = 40_000, 30_000, 5, None

    def __init__(self):
        self.calls = []

    def read(self, indexes=None, **kwargs):
        self.calls.append((indexes, kwargs))
        shape = kwargs.get("out_shape")
        assert shape is not None, "full-resolution read of a huge raster"
        return np.zeros(shape, dtype="uint16")


def test_a_huge_raster_is_read_decimated_and_one_band_only():
    src = _FakeBigRaster()
    band = tiles.read_band_decimated(src)
    (indexes, kwargs), = src.calls
    assert indexes == 1                         # band 1, not all five bands
    rows, cols = kwargs["out_shape"]
    assert rows * cols <= tiles.MAX_STATS_CELLS == 2_000_000
    assert band.shape == (rows, cols)
    assert rows / cols == pytest.approx(40_000 / 30_000, rel=0.01)


def test_loading_and_classifying_never_read_the_full_file(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    path = _write_raster(tmp_path / "multi.tif",
                         rng.integers(0, 1000, (3, 300, 300)).astype("uint16"))
    monkeypatch.setattr(tiles, "MAX_STATS_CELLS", 5_000)
    reads = []
    original = rasterio.io.DatasetReader.read

    def spy(self, indexes=None, *args, **kwargs):
        reads.append((indexes, kwargs.get("out_shape")))
        return original(self, indexes, *args, **kwargs)

    monkeypatch.setattr(rasterio.io.DatasetReader, "read", spy)

    _, band, _, _ = tiles._resolve_source(path, None, None, tmp_path)
    values = graduated.sample_values(path)

    assert band.size <= 5_000 and values.size <= 5_000
    assert len(reads) == 2
    for indexes, out_shape in reads:
        assert indexes == 1                      # never the 3-band default
        assert out_shape is not None and out_shape[0] * out_shape[1] <= 5_000
    # tile range: min/max come from this decimated read (documented in tiles.py)
    lo, hi = tiles._valid_range(band, float(np.iinfo("uint16").max))
    assert 0 <= lo <= hi < 1000


@pytest.mark.parametrize("dtype, nodata", [
    ("float32", -9999.0), ("int16", -32768), ("float32", 3.4028234663852886e38)])
def test_declared_nodata_is_left_out_of_the_value_range(tmp_path, dtype, nodata):
    data = np.linspace(10, 20, 100 * 100).reshape(100, 100).astype(dtype)
    data[:10, :] = nodata
    path = _write_raster(tmp_path / "dem.tif", data, nodata=nodata)

    declared = tiles.dataset_nodata(path, np.dtype(dtype))
    assert declared == pytest.approx(float(nodata))
    _, band, _, _ = tiles._resolve_source(path, None, None, tmp_path)
    lo, hi = tiles._valid_range(band, declared)
    assert 10 <= lo < hi <= 20                  # the nodata crushed it before

    values = graduated.sample_values(path)
    assert values.min() >= 10 and values.max() <= 20


def test_tile_layer_carries_the_datasets_nodata(tmp_path):
    data = np.linspace(10, 20, 100 * 100).reshape(100, 100).astype("float32")
    data[:10, :] = -9999.0
    path = _write_raster(tmp_path / "dem.tif", data, nodata=-9999.0)
    layer = tiles.build_tile_layer(path, name="dem", work_dir=tmp_path)
    try:
        assert layer.nodata == -9999.0
        assert 10 <= layer.vmin < layer.vmax <= 20
        assert "nodata=-9999" in layer.tile_url
        assert layer.tile_url.startswith("http://localhost:")   # host unchanged
    finally:
        layer.tile_client.shutdown()


def test_undeclared_nodata_keeps_the_dtype_default(tmp_path):
    ints = _write_raster(tmp_path / "cost.tif",
                         np.full((20, 20), 65535, dtype="uint16"))
    assert tiles.dataset_nodata(ints, np.dtype("uint16")) == 65535.0
    floats = _write_raster(tmp_path / "f.tif", np.ones((20, 20), "float32"))
    assert np.isnan(tiles.dataset_nodata(floats, np.dtype("float32")))


# ============================================ #20 combination build retry
def test_a_failed_combination_build_keeps_its_inputs_and_can_retry(monkeypatch):
    table = gpd.GeoDataFrame(
        {"base": ["b"], "cost": [100], "modifiers": [""], "__row": [0]},
        geometry=[box(0, 0, 1, 1)], crs=UTM)
    attempts = []

    def flaky(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise MemoryError("combination-build-boom-20")
        return table

    monkeypatch.setattr(combinations, "build_combination_table", flaky)
    layer = SimpleNamespace(name="cost raster", meta={"combination_inputs": {
        "base_gdf": None, "feature_keys": ("k",), "assumptions": {}}})

    assert combinations.ensure_combination_gdf(layer) is None
    assert "combination_inputs" in layer.meta                 # kept
    assert "MemoryError" in layer.meta["combination_error"]
    log = logbook.read_log()
    assert "combination-build-boom-20" in log and "could not be built" in log

    assert combinations.ensure_combination_gdf(layer) is table   # retry works
    assert "combination_inputs" not in layer.meta
    assert "combination_error" not in layer.meta
    assert combinations.ensure_combination_gdf(layer) is table   # and is cached
    assert len(attempts) == 2


# ============================================================ #24 notices
def _boom(label):
    raise RuntimeError(f"boom-{label}")


def test_notice_history_is_capped_and_holds_no_traceback():
    notices: list[dict] = []
    for i in range(30):
        _, notices = errors.guard(_boom, i, notices=notices)

    assert len(notices) <= 20 == errors.MAX_STORED_NOTICES
    assert notices[-1]["details"] == "RuntimeError: boom-29"   # newest kept
    assert all("Traceback" not in (n["details"] or "") for n in notices)
    assert all(len(json.dumps(n)) < 1500 for n in notices)
    # the whole traceback stays server-side, for the log file
    assert "Traceback (most recent call last)" in \
        errors.full_details(notices[-1]["id"])


def test_short_details_survive_and_long_ones_are_cut():
    short = errors.Notice(severity="info", title="t", details="d").to_dict()
    assert short["details"] == "d"
    assert errors.full_details(short["id"]) is None
    long_text = "x" * 5_000
    cut = errors.Notice(severity="warning", title="t",
                        details=long_text).to_dict()
    assert len(cut["details"]) <= errors.MAX_DETAILS_CHARS
    assert errors.full_details(cut["id"]) == long_text


def _open_flags(resp):
    return {t["props"]["id"]["index"]: t["props"]["is_open"]
            for t in resp[ids.NOTICE_STACK]["children"]}


def test_render_notices_opens_only_new_ones_and_logs_the_full_traceback(
        app, monkeypatch):
    from pyorps.gui.callbacks import notifications

    clock = [1_000.0]
    monkeypatch.setattr(notifications, "_clock", lambda: clock[0])

    _, first = errors.guard(_boom, "log-me-12345")
    resp = invoke(app, ("notice-stack.children",), first)
    assert list(_open_flags(resp).values()) == [True]
    log = logbook.read_log()
    assert "log-me-12345" in log
    assert "Traceback (most recent call last)" in log     # full text in the log

    # 2 s later a second notice arrives: the first toast is still showing
    # (error timer 10 s), so it stays open and only the new one opens
    clock[0] += 2
    _, both = errors.guard(_boom, "second", notices=first)
    resp = invoke(app, ("notice-stack.children",), both)
    assert _open_flags(resp) == {first[0]["id"]: True, both[-1]["id"]: True}

    # 20 s later the first one has timed out client-side: a third notice must
    # NOT bring it (or the second) back
    clock[0] += 20
    _, three = errors.guard(_boom, "third", notices=both)
    resp = invoke(app, ("notice-stack.children",), three)
    assert _open_flags(resp) == {first[0]["id"]: False,
                                 both[-1]["id"]: False,
                                 three[-1]["id"]: True}


def test_render_notices_never_renders_more_than_the_cap(app):
    notices = [{"id": f"n{i}", "severity": "info", "title": f"t{i}"}
               for i in range(35)]
    resp = invoke(app, ("notice-stack.children",), notices)
    assert len(resp[ids.NOTICE_STACK]["children"]) == 20


# ============================================================ #25 threading
def test_concurrent_mutation_and_reading_do_not_race(state):
    errors_seen: list[BaseException] = []
    stop = threading.Event()

    def guarded(fn):
        def run():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - the test's whole point
                errors_seen.append(exc)
        return run

    @guarded
    def adder():
        for i in range(150):
            state.add_layer(f"v{i}", "vector", geojson=EMPTY_FC)

    @guarded
    def remover():
        for _ in range(150):
            current = list(state.layers)
            if current:
                state.remove_layer(current[0])

    @guarded
    def reader():
        while not stop.is_set():
            state.layers_view()
            state.layers_of_kind("vector")

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)               # make thread switches frequent
    try:
        workers = [threading.Thread(target=fn) for fn in (adder, adder, remover)]
        watcher = threading.Thread(target=reader)
        watcher.start()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        stop.set()
        watcher.join(timeout=60)
    finally:
        sys.setswitchinterval(previous)

    assert errors_seen == []
    z_values = [ly.z for ly in state.layers.values()]
    assert len(z_values) == len(set(z_values))        # no duplicate paint order


def test_slow_cleanup_runs_after_the_lock_is_released(state):
    outcome: dict = {}

    class Client:
        def shutdown(self):
            def probe():
                got = state.lock.acquire(timeout=2)
                outcome["lock_was_free"] = got
                if got:
                    state.lock.release()
            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()

    layer = state.add_layer("r", "raster",
                            tile=SimpleNamespace(tile_client=Client()))
    state.remove_layer(layer.id)
    assert outcome == {"lock_was_free": True}


def test_locked_block_defers_cleanup_until_it_ends(state):
    stopped = []

    class Client:
        def shutdown(self):
            stopped.append(True)

    layer = state.add_layer("r", "raster",
                            tile=SimpleNamespace(tile_client=Client()))
    with state.locked():
        state.remove_layer(layer.id)
        assert layer.id not in state.layers and stopped == []   # queued
    assert stopped == [True]
