"""The data manifest: a synthetic round trip, and the refusals (plan Phase B)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")

from pyorps.io.provenance import ProvenanceMismatch
from pyorps.siting.data_manifest import (
    KINDS,
    DataManifest,
    LayerEntry,
    ManifestError,
)

CASE = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting"
REAL = CASE / "data" / "data_manifest.yaml"
# the wheel-test job runs the tests without the repository around them
needs_case = pytest.mark.skipif(not (REAL.exists() and (CASE / "scripts" / "download_log.py").exists()),
                                reason="case-study files not present")


def _write_raster(path, arr, origin=(1000.0, 5000.0), res=10.0, crs="EPSG:25832"):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype=str(arr.dtype),
                       crs=crs, transform=from_origin(origin[0], origin[1],
                                                      res, res)) as dst:
        dst.write(arr, 1)


def _write_vector(path, crs="EPSG:25832"):
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"name": ["a", "b", "c"]},
                           geometry=[Point(1010, 4990), Point(1090, 4910),
                                     Point(9000, 9000)], crs=crs)
    gdf.to_file(path, driver="GPKG")


@pytest.fixture()
def manifest(tmp_path):
    arr = np.arange(100 * 100, dtype=np.uint32).reshape(100, 100)
    (tmp_path / "data").mkdir()
    _write_raster(tmp_path / "data" / "dem.tif", arr)
    _write_vector(tmp_path / "data" / "lines.gpkg")
    f = tmp_path / "manifest.yaml"
    f.write_text("""schema: 1
layers:
  dem:
    kind: dem
    format: raster
    status: pending
    source: "synthetic"
    licence: "test"
    crs: EPSG:25832
  lines:
    kind: lines
    format: vector
    status: pending
    source: "synthetic"
    licence: "ODbL"
    crs: EPSG:25832
  roads:
    kind: roads
    format: vector
    status: pending
    source: "synthetic"
    licence: "to be confirmed at download"
""", encoding="utf-8")
    m = DataManifest.load(f)
    m.register("dem", tmp_path / "data" / "dem.tif", captured="2026-10-05")
    m.register("lines", tmp_path / "data" / "lines.gpkg", captured="2026-10-05")
    m.save()
    return DataManifest.load(f), arr, tmp_path


@needs_case
def test_the_shipped_manifest_is_valid_and_its_present_layers_verify():
    m = DataManifest.load(REAL)
    assert {e.kind for e in m.layers.values()} <= set(KINDS)
    assert all(e.licence and e.source for e in m.layers.values())
    present = [k for k, v in m.layers.items() if v.status == "present"]
    assert set(present) | set(m.pending()) == set(m.layers)
    on_disk = [k for k in present if m.file_of(k).is_file()]
    if on_disk:                       # a clone without the data skips the hash check
        assert set(m.verify(on_disk)) == set(on_disk)
    # a pending layer (there may be none once every layer is downloaded) has
    # no file: it can neither be located, verified nor read
    pending = m.pending()
    assert all(m.layer(k).status == "pending" for k in pending)
    for k in pending:
        with pytest.raises(ManifestError, match="pending"):
            m.file_of(k)
        with pytest.raises(ManifestError, match="pending"):
            m.verify([k])
    for k in (k for k in pending if m.layer(k).format == "raster"):
        with pytest.raises(ManifestError, match="pending"):
            m.read_window(k)


def test_round_trip_keeps_hashes_and_dates(manifest):
    m, _arr, _tmp = manifest
    assert m.pending() == ["roads"]
    e = m.layer("dem")
    assert e.status == "present" and len(e.sha256) == 64
    assert e.captured == "2026-10-05" and e.path == "data/dem.tif"
    assert m.verify() == {"dem": e.sha256, "lines": m.layer("lines").sha256}
    prov = m.provenance(["dem"])
    assert prov["dem"]["sha256"] == e.sha256 and prov["dem"]["licence"] == "test"


def test_a_changed_file_is_refused(manifest):
    m, arr, tmp = manifest
    _write_raster(tmp / "data" / "dem.tif", arr + 1)        # replaced content
    with pytest.raises(ProvenanceMismatch, match="dem: sha256"):
        m.verify(["dem"])
    with pytest.raises(ProvenanceMismatch):
        m.read_window("dem")


def test_a_missing_file_is_refused(manifest):
    m, _arr, tmp = manifest
    (tmp / "data" / "lines.gpkg").unlink()
    with pytest.raises(ProvenanceMismatch, match="file missing"):
        m.verify(["lines"])


def test_a_pending_layer_cannot_be_read_or_verified(manifest):
    m, _arr, _tmp = manifest
    with pytest.raises(ManifestError, match="pending"):
        m.verify(["roads"])
    with pytest.raises(ManifestError, match="no layer"):
        m.layer("nope")


def test_the_window_is_the_slice_with_its_transform(manifest):
    m, arr, _tmp = manifest
    full = m.read_window("dem")
    np.testing.assert_array_equal(full.array, arr)
    assert full.sha256 == m.layer("dem").sha256 and full.crs == "EPSG:25832"
    # bounds 1020..1060 east, 4950..4980 north = columns 2..5, rows 2..4
    w = m.read_window("dem", bounds=(1020.0, 4950.0, 1060.0, 4980.0))
    np.testing.assert_array_equal(w.array, arr[2:5, 2:6])
    assert (w.row_off, w.col_off) == (2, 2)
    assert (w.transform.c, w.transform.f) == (1020.0, 4980.0)
    # off-grid bounds grow outward to whole pixels
    w2 = m.read_window("dem", bounds=(1021.0, 4951.0, 1059.0, 4979.0))
    np.testing.assert_array_equal(w2.array, arr[2:5, 2:6])


def test_vector_bbox_clips_by_extent(manifest):
    m, _arr, _tmp = manifest
    gdf = m.read_vector("lines", bbox=(1000, 4900, 1100, 5000))
    assert sorted(gdf["name"]) == ["a", "b"]
    assert len(m.read_vector("lines")) == 3
    with pytest.raises(ManifestError, match="not a raster"):
        m.read_window("lines")
    with pytest.raises(ManifestError, match="not a vector"):
        m.read_vector("dem")


def test_a_wrong_crs_is_refused(manifest):
    m, arr, tmp = manifest
    other = tmp / "data" / "dem_wgs.tif"
    _write_raster(other, arr, crs="EPSG:32632")
    m.register("dem", other, captured="2026-10-05")
    with pytest.raises(ManifestError, match="CRS"):
        m.read_window("dem")


@pytest.mark.parametrize("patch,match", [
    ({"kind": "banana"}, "kind"),
    ({"format": "csv"}, "format"),
    ({"status": "done"}, "status"),
    ({"source": ""}, "source and licence"),
    ({"status": "present", "path": "x", "sha256": "abc",
      "captured": "2026-10-05"}, "sha256"),
    ({"status": "present", "path": "x", "sha256": "0" * 64,
      "captured": "yesterday"}, "ISO date"),
    ({"status": "present", "sha256": "0" * 64, "captured": "2026-10-05"},
     "path"),
])
def test_bad_entries_are_refused(patch, match):
    base = dict(id="x", kind="dem", format="raster", path=None,
                status="pending", source="s", licence="l")
    base.update(patch)
    with pytest.raises(ManifestError, match=match):
        LayerEntry(**base)


def test_unknown_fields_and_schema_are_refused(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("schema: 2\nlayers: {a: {kind: dem}}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="schema"):
        DataManifest.load(f)
    f.write_text("schema: 1\nlayers:\n  a: {kind: dem, format: raster, "
                 "source: s, licence: l, colour: red}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="unknown field"):
        DataManifest.load(f)


def test_the_manifest_can_move_with_its_files(manifest, tmp_path_factory):
    m, _arr, tmp = manifest
    new = tmp_path_factory.mktemp("moved")
    shutil.copytree(tmp / "data", new / "data")
    shutil.copy2(tmp / "manifest.yaml", new / "manifest.yaml")
    moved = DataManifest.load(new / "manifest.yaml")
    assert moved.verify()                    # relative paths: still valid


# ---------------------------------------------- provenance fields (time, method, use)


def test_captured_at_used_for_and_retrieved_with_round_trip(manifest):
    m, _arr, tmp = manifest
    m.annotate("dem", captured_at="2026-10-05T14:41:52+02:00",
               retrieved_with="a script, python 3.13",
               used_for=["done: checked", "planned: slope limit; not yet used"])
    m.save()
    again = DataManifest.load(tmp / "manifest.yaml")
    e = again.layer("dem")
    assert e.captured_at == "2026-10-05T14:41:52+02:00"
    assert e.retrieved_with == "a script, python 3.13"
    assert e.used_for == ("done: checked", "planned: slope limit; not yet used")
    assert again.layer("lines").used_for == ()             # untouched layers stay plain


def test_captured_at_needs_an_offset_and_the_captured_date():
    base = dict(id="x", kind="dem", format="raster", path="p", status="present",
                source="s", licence="l", sha256="0" * 64, captured="2026-10-05")
    LayerEntry(**base, captured_at="2026-10-05T14:41:52+02:00")
    with pytest.raises(ManifestError, match="UTC offset"):
        LayerEntry(**base, captured_at="2026-10-05T14:41:52")
    with pytest.raises(ManifestError, match="captured date"):
        LayerEntry(**base, captured_at="2026-10-06T00:00:01+02:00")
    with pytest.raises(ManifestError, match="ISO 8601"):
        LayerEntry(**base, captured_at="after lunch")
    with pytest.raises(ManifestError, match="list of strings"):
        LayerEntry(**base, used_for="one string")


def test_registering_a_file_again_clears_the_old_exact_time(manifest):
    m, arr, tmp = manifest
    m.annotate("dem", captured_at="2026-10-05T14:41:52+02:00")
    _write_raster(tmp / "data" / "dem.tif", arr + 1)
    e = m.register("dem", tmp / "data" / "dem.tif", captured="2026-10-06")
    assert e.captured_at is None and e.captured == "2026-10-06"


@needs_case
def test_the_download_log_appends_and_never_rewrites(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "download_log_under_test", CASE / "scripts" / "download_log.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    monkeypatch.setattr(dl, "LOG", tmp_path / "log.jsonl")
    assert dl.read_events() == []
    ev = dl.append_event(layer_id="a", service="u", what="w", file="f", sha256="0" * 64,
                         **dl.now_stamps(), reconstructed=False)
    dl.append_event(layer_id="b", service="u", what="w", file="f", sha256="1" * 64,
                    reconstructed=True, reconstructed_because="test")
    got = dl.read_events()
    assert [e["layer_id"] for e in got] == ["a", "b"] and got[0] == ev
    assert got[0]["time_local"][-6:] in ("+01:00", "+02:00") or got[0]["time_local"]
    assert got[0]["time_utc"].endswith("+00:00") and got[0]["tool_versions"]["python"]
    with pytest.raises(ValueError, match="needs"):
        dl.append_event(layer_id="c")
    assert len((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()) == 2


@needs_case
def test_the_shipped_log_covers_every_present_layer_with_a_matching_hash():
    import importlib.util

    m = DataManifest.load(REAL)
    spec = importlib.util.spec_from_file_location(
        "download_log_real", CASE / "scripts" / "download_log.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    events = dl.read_events()
    for lid, e in m.layers.items():
        if e.status != "present" or e.captured != "2026-10-05":
            continue
        hits = [ev for ev in events if ev["layer_id"] == lid and ev["sha256"] == e.sha256]
        assert hits, f"no download event for {lid} with the manifest's hash"
        assert e.captured_at and e.used_for and e.retrieved_with
