"""The plan-A1 provenance record: hashing, assembly and strict comparison.

A reused field is only valid if it came from the same inputs as the run
that reads it, so the comparison must catch every difference -- a key on
one side only, a one-ulp change in a seed label, ``True`` against ``1`` --
and must not be fooled by the order the seeds were listed in.
"""
import json

import numpy as np
import pytest

from pyorps.io import provenance as prov


class TestHashing:
    def test_file_hash_matches_hashlib(self, tmp_path):
        import hashlib
        p = tmp_path / "a.bin"
        data = bytes(range(256)) * 5000
        p.write_bytes(data)
        assert prov.sha256_file(p) == hashlib.sha256(data).hexdigest()

    def test_file_hash_cache_sees_a_rewrite(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"one")
        first = prov.sha256_file(p)
        p.write_bytes(b"two, longer")
        assert prov.sha256_file(p) != first

    def test_array_hash_depends_on_dtype_shape_and_values(self):
        a = np.arange(12, dtype=np.uint16).reshape(3, 4)
        h = prov.sha256_array(a)
        assert prov.sha256_array(a.copy()) == h
        assert prov.sha256_array(a.astype(np.int32)) != h
        assert prov.sha256_array(a.reshape(4, 3)) != h
        b = a.copy()
        b[2, 3] += 1
        assert prov.sha256_array(b) != h

    def test_array_hash_of_a_view_equals_its_copy(self):
        base = np.arange(400, dtype=np.float64).reshape(20, 20)
        view = base[3:17:2, 1:19]
        assert prov.sha256_array(view) == prov.sha256_array(view.copy())

    def test_seed_hash_is_order_free_and_ulp_sensitive(self):
        cells = [5, 1, 9]
        labels = [2.0, 0.5, 7.25]
        h = prov.seed_hash(cells, labels)
        assert prov.seed_hash([9, 5, 1], [7.25, 2.0, 0.5]) == h
        nudged = [2.0, 0.5, float(np.nextafter(7.25, 8.0))]
        assert prov.seed_hash(cells, nudged) != h
        assert prov.seed_hash([1], None) == prov.seed_hash([1], [0.0])

    def test_seed_hash_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="labels"):
            prov.seed_hash([1, 2], [0.0])

    def test_cells_hash_ignores_order_and_duplicates(self):
        assert prov.cells_hash([3, 1, 3, 2]) == prov.cells_hash([1, 2, 3])
        assert prov.cells_hash([1, 2]) != prov.cells_hash([1, 2, 3])

    def test_parameter_hash_is_key_order_free(self):
        assert (prov.parameter_hash({"a": 1, "b": [1.5, 2]})
                == prov.parameter_hash({"b": [1.5, 2], "a": 1}))
        assert (prov.parameter_hash({"a": 1.0})
                != prov.parameter_hash({"a": 1.0000000000000002}))

    def test_source_file_hashes_are_repo_relative(self):
        import pyorps.io.provenance as mod
        out = prov.source_file_hashes([mod])
        assert list(out) == ["pyorps/io/provenance.py"]
        assert len(out["pyorps/io/provenance.py"]) == 64


class TestRecord:
    def test_record_normalises_numpy_and_round_trips(self):
        rec = prov.record(raster={"window": np.array([1, 2, 3, 4]),
                                  "cell": np.float32(1.5)},
                          storage={"dtype": "float64"})
        assert rec["schema"] == prov.PROVENANCE_SCHEMA
        assert rec["raster"]["window"] == [1, 2, 3, 4]
        assert json.loads(json.dumps(rec)) == rec

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="unknown provenance section"):
            prov.record(rastr={"a": 1})

    def test_non_finite_values_survive_as_strings(self):
        rec = prov.record(extra={"x": float("inf")})
        assert rec["extra"]["x"] == "inf"


class TestComparison:
    def _base(self):
        return prov.record(
            raster={"window": [0, 0, 10, 12], "source_sha256": "ab"},
            algorithm={"name": "dijkstra"},
            seeds={"seed_hash": prov.seed_hash([7], [0.0])},
            extra={"seconds": 1.2})

    def test_identical_records_match(self):
        prov.require_match(self._base(), self._base())

    def test_extra_is_never_compared(self):
        a, b = self._base(), self._base()
        b["extra"]["seconds"] = 99.0
        prov.require_match(a, b)

    def test_every_difference_is_listed(self):
        a, b = self._base(), self._base()
        b["raster"]["window"][2] = 11
        b["algorithm"]["name"] = "delta-stepping"
        b["algorithm"]["num_threads"] = 4
        with pytest.raises(prov.ProvenanceMismatch) as info:
            prov.require_match(a, b)
        text = "\n".join(info.value.differences)
        assert "raster.window[2]" in text
        assert "algorithm.name" in text
        assert "algorithm.num_threads" in text
        assert len(info.value.differences) == 3

    def test_a_section_on_one_side_only_is_a_mismatch(self):
        a = self._base()
        b = prov.record(**{k: v for k, v in self._base().items()
                           if k not in ("schema", "seeds")})
        with pytest.raises(prov.ProvenanceMismatch, match="seeds"):
            prov.require_match(a, b)

    def test_bool_is_not_int_but_int_is_float(self):
        a = prov.record(graph={"flag": True, "n": 1})
        b = prov.record(graph={"flag": 1, "n": 1.0})
        diffs = prov.diff(a, b)
        assert diffs == ["graph.flag: expected true, stored 1"]

    def test_missing_record_is_refused(self):
        with pytest.raises(prov.ProvenanceMismatch, match="before plan"):
            prov.require_match(self._base(), None)

    def test_ignore_can_skip_a_section(self):
        a, b = self._base(), self._base()
        b["algorithm"]["name"] = "other"
        prov.require_match(a, b, ignore=("extra", "algorithm"))


class TestSidecar:
    def test_round_trip(self, tmp_path):
        target = tmp_path / "out.tif"
        rec = prov.record(storage={"dtype": "float32", "rounding": "down"})
        written = prov.write_sidecar(target, rec)
        assert written.name == "out.tif.prov.json"
        assert prov.read_sidecar(target) == rec

    def test_absent_sidecar_reads_none(self, tmp_path):
        assert prov.read_sidecar(tmp_path / "nothing.tif") is None
