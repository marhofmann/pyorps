"""Saved fields carry a plan-A1 provenance record and sound bounds.

Two properties matter for the siting certificate:

* a reader that states what it expects gets the field only if every key
  matches -- raster window, steps, algorithm, seeds, weights, storage and
  cost model -- and a field written before records existed is refused;
* ``costs_to(bound="lower")`` never exceeds the float64 label and
  ``bound="upper"`` never falls below it, whatever the storage. Before
  this, a raw float32 field recorded ``error_bound=0.0`` although it was
  rounded to nearest (plan finding M3-13).
"""
import json

import numpy as np
import pytest
import rasterio

import pyorps
from pyorps.graph.search_session import cost_field_provenance
from pyorps.io.provenance import ProvenanceMismatch

CELL = 1.0000017516        # the case-study raster's pixel width
SHAPE = (60, 80)


def _raster(tmp_path, name="cost.tif", seed=11):
    rng = np.random.default_rng(seed)
    data = rng.integers(125, 4000, size=SHAPE).astype(np.uint16)
    path = tmp_path / name
    transform = rasterio.transform.from_origin(400000.0, 5600000.0, CELL, CELL)
    with rasterio.open(
        path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
        dtype="uint16", crs="EPSG:25832", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _finder(path, origin, target):
    return pyorps.PathFinder(
        dataset_source=str(path),
        source_coords=origin, target_coords=[target],
        search_space_buffer_m=pyorps.full_window_buffer_m(str(path)),
        neighborhood_str="r2", graph_api="cython", ignore_max_cost=True,
    )


def _points(path, n=80, seed=5):
    rng = np.random.default_rng(seed)
    with rasterio.open(path) as src:
        tr = src.transform
        rows = rng.integers(0, src.height, n)
        cols = rng.integers(0, src.width, n)
    xs, ys = rasterio.transform.xy(tr, rows.tolist(), cols.tolist())
    return list(zip(map(float, xs), map(float, ys)))


@pytest.fixture
def setup(tmp_path):
    path = _raster(tmp_path)
    with rasterio.open(path) as src:
        b = src.bounds
    origin = (b.left + 3.5 * CELL, b.top - 3.5 * CELL)
    far = (b.right - 3.5 * CELL, b.bottom + 3.5 * CELL)
    return path, origin, far, _points(path)


class TestRecordOnDisk:
    def test_saved_record_equals_the_expected_record(self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        cm = {"hash": "abc", "parameters": {"trench_eur_per_m": 125.0}}
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            live_record = field.provenance(storage="float64", cost_model=cm)
            saved = field.save(tmp_path / "f.npz", storage="float64",
                               provenance={"cost_model": cm},
                               extra_meta={"note": "test"})
        expected = cost_field_provenance(finder, origin, algorithm="dijkstra",
                                         storage="float64", cost_model=cm)
        assert expected == live_record
        with pyorps.CostField.open(saved, expect=expected) as reopened:
            assert reopened.provenance == expected
        meta = json.loads(str(np.load(saved)["meta"]))
        assert meta["extra_meta"] == {"note": "test"}
        assert meta["storage"] == {"dtype": "float64", "rounding": "exact"}

    def test_record_identifies_the_raster_and_the_window(self, tmp_path,
                                                         setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        rec = cost_field_provenance(finder, origin, algorithm="dijkstra")
        from pyorps.io.provenance import sha256_file
        assert rec["raster"]["source_sha256"] == sha256_file(path)
        assert rec["raster"]["window"] == [0, 0, SHAPE[0], SHAPE[1]]
        assert rec["graph"]["neighborhood"] == "r2"
        assert rec["algorithm"] == {"name": "dijkstra", "kind": "dijkstra",
                                    "label_precision": "float64"}
        assert rec["code"]["kernel"] == "pyorps.utils._dijkstra"
        assert rec["seeds"]["n_seeds"] == 1

    def test_a_different_raster_is_refused(self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz")
        other = _raster(tmp_path, name="other.tif", seed=12)
        expected = cost_field_provenance(_finder(other, origin, far), origin,
                                         algorithm="dijkstra")
        with pytest.raises(ProvenanceMismatch, match="source_sha256"):
            pyorps.CostField.open(saved, expect=expected)

    def test_a_different_algorithm_or_storage_is_refused(self, tmp_path,
                                                         setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz", storage="float32")
        want = cost_field_provenance(finder, origin, algorithm="dijkstra",
                                     storage="float64")
        with pytest.raises(ProvenanceMismatch, match="storage"):
            pyorps.CostField.open(saved, expect=want)
        want = cost_field_provenance(finder, origin,
                                     algorithm="delta-stepping",
                                     storage="float32", num_threads=1)
        with pytest.raises(ProvenanceMismatch, match="algorithm"):
            pyorps.CostField.open(saved, expect=want)

    def test_a_different_origin_is_refused(self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz")
        moved = (origin[0] + 5 * CELL, origin[1])
        want = cost_field_provenance(finder, moved, algorithm="dijkstra")
        with pytest.raises(ProvenanceMismatch, match="seeds"):
            pyorps.CostField.open(saved, expect=want)

    def test_a_legacy_field_without_record_is_refused(self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz")
        legacy = _strip_to_legacy(saved, tmp_path / "legacy.npz")
        want = cost_field_provenance(finder, origin, algorithm="dijkstra")
        with pytest.raises(ProvenanceMismatch, match="before plan"):
            pyorps.CostField.open(legacy, expect=want)
        # Without an expectation it still opens: provenance is opt-in.
        with pyorps.CostField.open(legacy) as reopened:
            assert reopened.provenance is None

    def test_unknown_provenance_sections_are_refused(self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            with pytest.raises(ValueError, match="only cost_model"):
                field.save(tmp_path / "f.npz",
                           provenance={"raster": {"window": [0]}})
            with pytest.raises(ValueError, match="storage"):
                field.save(tmp_path / "f.npz", storage="float16")


def _strip_to_legacy(src, dest):
    """Rewrite a saved field as the pre-A1 writer did: float32 of a float32
    product, and no storage or provenance keys."""
    with np.load(src) as data:
        payload = {k: np.asarray(data[k]) for k in data.files}
    meta = json.loads(str(payload["meta"]))
    meta.pop("provenance", None)
    meta.pop("storage", None)
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(dest, **payload)
    return dest


class TestSoundBounds:
    @pytest.mark.parametrize("storage", ["float32", "float32-down",
                                         "float64"])
    def test_lower_and_upper_bracket_the_float64_label(self, tmp_path, setup,
                                                       storage):
        path, origin, far, probes = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            exact = field.costs_to(probes)          # float64 labels
            saved = field.save(tmp_path / f"f_{storage}.npz", storage=storage)
        with pyorps.CostField.open(saved) as reopened:
            lo = reopened.costs_to(probes, bound="lower")
            hi = reopened.costs_to(probes, bound="upper")
            eb = reopened.error_bound
        fin = np.isfinite(exact)
        assert fin.all()
        assert np.all(lo <= exact) and np.all(exact <= hi)
        assert np.all(exact - lo <= eb + 1e-12)
        if storage == "float64":
            assert eb == 0.0
            np.testing.assert_array_equal(lo, exact)
            np.testing.assert_array_equal(hi, exact)
        else:
            assert eb > 0.0
            assert np.all(hi - lo <= 4 * eb)

    def test_float32_down_stores_lower_bounds(self, tmp_path, setup):
        path, origin, far, probes = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            full = field.field_array(dtype=np.float64).ravel()
            saved = field.save(tmp_path / "rd.npz", storage="float32-down")
        with np.load(saved) as data:
            stored = np.asarray(data["dist"])
        assert stored.dtype == np.float32
        fin = np.isfinite(full)
        assert np.all(stored[fin].astype(np.float64) <= full[fin])

    def test_legacy_float32_field_is_nudged_outward(self, tmp_path, setup):
        path, origin, far, probes = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            exact = field.costs_to(probes)
            # The pre-A1 writer's arithmetic: float32 label * float32 cell.
            f32 = field._tree.field(dtype=np.float32)
            old = np.empty(f32.size, dtype=np.float32)
            np.multiply(f32, field._scale, out=old, casting="unsafe")
            saved = field.save(tmp_path / "f.npz", storage="float32")
        with np.load(saved) as data:
            payload = {k: np.asarray(data[k]) for k in data.files}
        payload["dist"] = old
        meta = json.loads(str(payload["meta"]))
        meta.pop("storage")
        meta.pop("provenance")
        payload["meta"] = np.array(json.dumps(meta))
        legacy = tmp_path / "legacy.npz"
        np.savez(legacy, **payload)
        with pyorps.CostField.open(legacy) as reopened:
            lo = reopened.costs_to(probes, bound="lower")
            hi = reopened.costs_to(probes, bound="upper")
        assert np.all(lo <= exact) and np.all(exact <= hi)

    def test_costs_match_the_live_field_closely(self, tmp_path, setup):
        path, origin, far, probes = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            live = field.costs_to(probes)
            saved = field.save(tmp_path / "f.npz")
        with pyorps.CostField.open(saved) as reopened:
            np.testing.assert_allclose(reopened.costs_to(probes), live,
                                       rtol=3e-7)

    def test_bad_bound_name_raises(self, tmp_path, setup):
        path, origin, far, probes = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz")
        with pyorps.CostField.open(saved) as reopened:
            with pytest.raises(ValueError, match="bound"):
                reopened.costs_to(probes[:1], bound="middle")


# ---------------------------------------------------------------------------
# Review of A1 (2026-09-24): one test per finding.

N_DEM = 41


def _dem_finder(dem):
    from rasterio.transform import from_origin
    return pyorps.PathFinder(
        dataset_source=np.full((N_DEM, N_DEM), 100, dtype=np.uint16),
        crs="EPSG:25832", transform=from_origin(0.0, float(N_DEM), 1.0, 1.0),
        source_coords=(2.5, N_DEM - 0.5 - 20),
        target_coords=(N_DEM - 2.5, N_DEM - 0.5 - 20),
        search_space_buffer_m=200, graph_api="cython",
        dem=dem, objective={"cost": 1.0})


def _tilted(slope_pct=40.0):
    return (np.arange(N_DEM, dtype=np.float32)[:, None] * slope_pct / 100.0
            * np.ones((1, N_DEM), dtype=np.float32))


class TestFloat32Kernels:
    """C1: a delta-stepping field's labels are float32-accumulated, so it
    may not be saved as if it were a bound on the least cost."""

    def test_bound_storage_is_refused_for_a_delta_field(self, tmp_path,
                                                        setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="delta-stepping",
                               num_threads=1) as field:
            for storage in ("float64", "float32-down"):
                with pytest.raises(ValueError, match="float32-accumulated"):
                    field.save(tmp_path / "d.npz", storage=storage)
                with pytest.raises(ValueError, match="float32-accumulated"):
                    field.provenance(storage=storage)
            saved = field.save(tmp_path / "d.npz", storage="float32")
        with pyorps.CostField.open(saved) as reopened:
            assert reopened.label_precision == "float32"
            assert not reopened.certifiable
        with pytest.raises(ValueError, match="float32-accumulated"):
            cost_field_provenance(finder, origin, algorithm="delta-stepping",
                                  storage="float64", num_threads=1)

    def test_a_dijkstra_field_is_certifiable(self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz", storage="float32-down")
        with pyorps.CostField.open(saved) as reopened:
            assert reopened.label_precision == "float64"
            assert reopened.certifiable

    def test_default_delta_and_threads_are_recorded(self, setup):
        """m1: the defaults _make_tree uses are recorded, so an explicit
        delta=100 and the default give the same record."""
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        a = cost_field_provenance(finder, origin, algorithm="delta-stepping",
                                  num_threads=1)
        b = cost_field_provenance(finder, origin, algorithm="delta-stepping",
                                  num_threads=1, delta=100)
        assert a == b
        assert a["algorithm"]["delta"] == 100
        assert a["algorithm"]["num_threads"] == 1
        c = cost_field_provenance(finder, origin, algorithm="delta-stepping")
        assert c["algorithm"]["num_threads"] >= 1


class TestSlopeTerms:
    """C2: the delta and GPU full-field kernels ignore the DEM."""

    def test_auto_picks_dijkstra_with_a_dem(self):
        f = _dem_finder(_tilted())
        with f.cost_field((2.5, N_DEM - 0.5 - 20), algorithm="auto") as field:
            assert field.algorithm == "dijkstra"

    def test_explicit_delta_with_a_dem_is_refused(self):
        f = _dem_finder(_tilted())
        with pytest.raises(NotImplementedError, match="slope"):
            f.cost_field((2.5, N_DEM - 0.5 - 20), algorithm="delta-stepping",
                         num_threads=1)
        with pytest.raises(NotImplementedError, match="slope"):
            cost_field_provenance(f, (2.5, N_DEM - 0.5 - 20),
                                  algorithm="delta-stepping", num_threads=1)

    def test_a_dem_field_with_voids_is_accepted_by_a_fresh_reader(
            self, tmp_path):
        """M3: the reader builds the graph as the writer did, so the DEM
        hash, the slope tables and the void-stamped window all match."""
        dem = _tilted()
        dem[5, 30] = np.nan
        origin = (2.5, N_DEM - 0.5 - 20)
        writer = _dem_finder(dem)
        with writer.cost_field(origin, algorithm="auto") as field:
            saved = field.save(tmp_path / "dem.npz", storage="float64")
        want = cost_field_provenance(_dem_finder(dem.copy()), origin,
                                     algorithm="auto", storage="float64")
        assert want["graph"]["dem_sha256"] is not None
        assert want["graph"]["gradient_luts"] is not None
        with pyorps.CostField.open(saved, expect=want) as reopened:
            assert reopened.certifiable
        flat = cost_field_provenance(_dem_finder(_tilted(10.0)), origin,
                                     algorithm="auto", storage="float64")
        with pytest.raises(ProvenanceMismatch, match="dem_sha256"):
            pyorps.CostField.open(saved, expect=flat)


class TestSettleTimeRecord:
    """M1: the record describes the inputs the labels came from."""

    def test_a_raster_changed_after_settling_refuses_to_save(self, tmp_path,
                                                             setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            data = finder.raster_handler.data
            (data[0] if data.ndim == 3 else data)[10, 10] += 7
            with pytest.raises(ValueError, match="window_sha256"):
                field.save(tmp_path / "f.npz")
            with pytest.raises(ValueError, match="changed since"):
                field.provenance()

    def test_ignore_max_cost_changed_after_settling_refuses_to_save(
            self, tmp_path, setup):
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            finder.ignore_max_cost = False
            with pytest.raises(ValueError, match="invalidated"):
                field.save(tmp_path / "f.npz")


class TestMetaAndEdgeCases:
    @pytest.mark.parametrize("storage", ["float32", "float32-down",
                                         "float64"])
    def test_meta_error_bound_is_the_real_bound(self, tmp_path, setup,
                                                storage):
        """M2: the bound is in the file, not only computed on reopen."""
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz", storage=storage)
        with np.load(saved) as data:
            meta = json.loads(str(data["meta"]))
        with pyorps.CostField.open(saved) as reopened:
            assert meta["error_bound"] == reopened.error_bound
        assert (meta["error_bound"] == 0.0) == (storage == "float64")
        assert meta["label_precision"] == "float64"

    def test_legacy_rounding_is_two_ulps(self):
        """m2: the reviewer's counter-example to the old 1.5 ulp."""
        from pyorps.graph.search_session import (_nudge, _nudge_ulps,
                                                 _raw_error_bound)
        label, cell = 2096785.9374999874, 1.0001707673072695
        stored = np.array([np.float32(label) * np.float32(cell)],
                          dtype=np.float32)
        down, _up, rounding = _nudge_ulps(None, np.float32)
        lower = float(_nudge(stored, down, -np.inf)[0])
        true = label * cell
        assert lower <= true
        assert true - lower <= _raw_error_bound(stored, down, rounding)

    def test_float32_overflow_stays_finite(self):
        """m3: a finite label past float32's range is clamped, and the
        bound then says it is unbounded rather than lying."""
        from pyorps.graph.search_session import (_raw_error_bound,
                                                 _store_labels)
        out = _store_labels(np.array([1.0, 1e39, np.inf]), 1.0, "float32")
        assert np.isfinite(out[1]) and out[1] == np.finfo(np.float32).max
        assert np.isinf(out[2])
        assert _raw_error_bound(out, 1, 0.5) == float("inf")

    def test_crs_spellings_normalise(self):
        from pyorps.graph.search_session import _crs_string
        assert _crs_string("epsg:25832") == _crs_string("EPSG:25832")
        assert _crs_string(None) is None

    def test_codec_fields_can_be_expected(self, tmp_path, setup):
        """m5: a reader can build the record of a fixed-quantum save."""
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "q.npz", codec="fixed-quantum",
                               error_bound=1.0, with_paths=False)
        want = cost_field_provenance(finder, origin, algorithm="dijkstra",
                                     codec="fixed-quantum", error_bound=1.0)
        with pyorps.CostField.open(saved, expect=want):
            pass

    def test_expectation_is_checked_before_the_arrays(self, tmp_path, setup):
        """m5: a refused field is refused on its meta alone."""
        path, origin, far, _ = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz")
        with np.load(saved) as data:
            payload = {k: np.asarray(data[k]) for k in data.files}
        payload["dist"] = payload["dist"][:10]            # a broken body
        broken = tmp_path / "broken.npz"
        np.savez(broken, **payload)
        with pytest.raises(ValueError, match="labels for a"):
            pyorps.CostField.open(broken)
        other = cost_field_provenance(finder, origin, algorithm="dijkstra",
                                      storage="float64")
        with pytest.raises(ProvenanceMismatch):
            pyorps.CostField.open(broken, expect=other)

    def test_label_bound_is_the_stored_value(self, tmp_path, setup):
        """m4: ``bound="label"`` is what CostFieldSet reads from a spilled
        field, as the Leg docstring says, not the nudged lower bound."""
        path, origin, far, probes = setup
        finder = _finder(path, origin, far)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            saved = field.save(tmp_path / "f.npz")
        with np.load(saved) as data:
            stored = np.asarray(data["dist"])
        with pyorps.CostField.open(saved) as reopened:
            label = reopened.costs_to(probes, bound="label")
            lower = reopened.costs_to(probes)
            idx = reopened._indices(np.asarray(probes))
        assert np.all(lower <= label)
        np.testing.assert_array_equal(label, stored[idx].astype(np.float64))


def test_file_hash_cache_sees_a_replacement_with_the_same_mtime(tmp_path):
    """m5: a same-size file renamed over the old one, mtime preserved."""
    import os

    from pyorps.io.provenance import sha256_file
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 64)
    first = sha256_file(a)
    st = a.stat()
    b = tmp_path / "b.bin"
    b.write_bytes(b"y" * 64)
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(b, a)
    assert a.stat().st_mtime_ns == st.st_mtime_ns
    assert sha256_file(a) != first
