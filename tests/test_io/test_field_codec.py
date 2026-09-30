"""Fixed-quantum field storage: the bound has to be a guarantee.

Section 2 / 3.5 of
``docs/superpowers/plans/2026-09-20-generalized-free-siting-facility-chains.md``.

The point of fixing the QUANTUM rather than the bit budget is that the
error stops being a measurement and becomes a parameter. So the tests
that matter are the ones that would catch it not being one: the
understatement bound at several quanta and on several value ranges, the
reachability mask (without it an unreachable cell decodes to a finite
filler and every bound over it is a lie), and the interval surviving
the decode's own float arithmetic.
"""

import numpy as np
import pytest

from pyorps.io.field_codec import (
    DecodedField,
    FIELD_CODECS,
    codec_of,
    decode_field,
    encode_field,
    is_lossy,
)


def _field(rows=160, cols=220, seed=1, holes=True):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    f = (np.hypot(yy - rows / 2, xx - cols / 2) * 1200.0
         + 4e4 * np.sin(xx / 37.0) + 3e4 * np.cos(yy / 29.0)
         + rng.random((rows, cols)) * 5e3)
    f = f - f.min()
    if holes:
        f[20:40, 30:60] = np.inf
        f[0, :] = np.inf
    return f


class TestRegistry:
    def test_names(self):
        assert set(FIELD_CODECS) == {"raw", "fixed-quantum"}
        assert is_lossy("fixed-quantum")
        assert not is_lossy("raw")

    def test_codec_key_defaults_to_raw(self):
        """Dispatch is on the codec key, never on the format version.

        The version is compared for strict equality, so bumping it to
        introduce a codec would reject every field already on disk.
        """
        assert codec_of({}) == "raw"
        assert codec_of({"format": 1}) == "raw"
        assert codec_of({"codec": "fixed-quantum"}) == "fixed-quantum"

    def test_unknown_codec_refused(self):
        with pytest.raises(ValueError, match="unknown codec"):
            encode_field(np.ones((8, 8)), codec="brotli-9000")


class TestErrorBound:
    @pytest.mark.parametrize("eb", [100.0, 10.0, 1.0, 0.1])
    def test_understatement_never_exceeds_the_quantum(self, eb):
        f = _field()
        payload, meta = encode_field(f, error_bound=eb)
        d = decode_field(payload, meta)
        ok = np.isfinite(f)
        under = f[ok] - d.lower[ok]
        assert np.all(under >= 0.0), "a decoded value must never exceed"
        assert under.max() <= eb * (1 + 1e-9)
        assert np.all(d.upper[ok] >= f[ok])
        assert d.width() == eb

    @pytest.mark.parametrize("scale", [1.0, 1e3, 1e7])
    def test_bound_is_independent_of_dynamic_range(self, scale):
        """The whole claim of a fixed quantum, in one assertion.

        A range-based codec's error grows with the values; this one's
        does not, which is why shrinking tiles and promoting outliers
        never moved it.
        """
        f = _field(rows=80, cols=100, holes=False) * scale
        payload, meta = encode_field(f, error_bound=1.0)
        d = decode_field(payload, meta)
        assert (f - d.lower).max() <= 1.0 * (1 + 1e-9)

    def test_quantum_must_be_positive_and_finite(self):
        for bad in (0.0, -1.0, np.inf, np.nan):
            with pytest.raises(ValueError, match="finite positive quantum"):
                encode_field(np.ones((8, 8)), error_bound=bad)

    def test_negative_values_refused(self):
        f = np.ones((8, 8))
        f[2, 2] = -5.0
        with pytest.raises(ValueError, match="negative field values"):
            encode_field(f, error_bound=1.0)


class TestReachability:
    def test_mask_round_trips_exactly(self):
        f = _field()
        payload, meta = encode_field(f, error_bound=5.0)
        d = decode_field(payload, meta)
        assert np.array_equal(d.reachable, np.isfinite(f))
        assert np.all(np.isinf(d.lower[~d.reachable]))
        assert np.all(np.isinf(d.upper[~d.reachable]))

    def test_an_all_unreachable_field_survives(self):
        f = np.full((16, 16), np.inf)
        d = decode_field(*reversed(tuple(reversed(
            encode_field(f, error_bound=1.0)))))
        assert not d.reachable.any()
        assert np.all(np.isinf(d.lower))


class TestIntervalDiscipline:
    def test_choose_names_the_side(self):
        f = _field(rows=40, cols=40)
        d = decode_field(*(lambda t: (t[0], t[1]))(
            encode_field(f, error_bound=10.0)))
        assert np.array_equal(d.choose("lower"), d.lower)
        assert np.array_equal(d.choose("upper"), d.upper)
        with pytest.raises(ValueError, match="no third option"):
            d.choose("value")

    def test_exact_codec_has_a_degenerate_interval(self):
        f = _field(rows=40, cols=40)
        payload, meta = encode_field(f, codec="raw")
        d = decode_field(payload, meta)
        assert d.exact and d.width() == 0.0
        assert np.array_equal(d.upper, d.lower)
        ok = np.isfinite(f)
        assert np.allclose(d.lower[ok], f[ok], rtol=1e-6)

    def test_upper_is_strictly_above_a_true_value_at_a_bin_edge(self):
        """The decode's own rounding must not break the inequality.

        ``k * quantum`` is a rounded product, so both ends are nudged
        one ulp outward -- Neumaier & Shcherbina's safe-bound rule.
        """
        q = 0.1
        f = (np.arange(1, 1025, dtype=np.float64) * q).reshape(32, 32)
        payload, meta = encode_field(f, error_bound=q)
        d = decode_field(payload, meta)
        assert np.all(d.lower <= f)
        assert np.all(d.upper >= f)


class TestLayout:
    def test_shape_and_tiling_round_trip(self):
        for shape, tile in [((160, 220), 256), ((160, 220), 64),
                            ((7, 5), 4), ((256, 256), 128)]:
            f = _field(*shape, holes=False)
            payload, meta = encode_field(f, error_bound=1.0, tile=tile)
            d = decode_field(payload, meta)
            assert d.lower.shape == shape
            assert (f - d.lower).max() <= 1.0 * (1 + 1e-9)

    def test_frames_are_independently_sized(self):
        """Per-tile-row framing is what makes a partial read possible."""
        f = _field(rows=256, cols=256, holes=False)
        payload, meta = encode_field(f, error_bound=1.0, tile=64)
        offsets = payload["q_frames"]
        assert meta["frames"] == 256 // 64
        assert offsets.size == meta["frames"] + 1
        assert np.all(np.diff(offsets) > 0)
        assert offsets[-1] == payload["q_body"].size

    def test_smaller_than_a_deflated_float32_baseline(self):
        """Quoted against the COMPRESSED baseline, not the raw one.

        The plan's revision 3 found the 9.1x headline conflated the
        codec with simply deflating: merely compressing is 1.65x free,
        so that is the number to beat.
        """
        import zlib
        f = _field(rows=300, cols=400)
        baseline = len(zlib.compress(
            np.where(np.isfinite(f), f, np.inf).astype(np.float32).tobytes(),
            6))
        payload, _ = encode_field(f, error_bound=10.0)
        size = sum(v.nbytes for v in payload.values())
        assert size < baseline, (size, baseline)


class TestSavedCostFieldIntegration:
    def test_round_trip_through_a_saved_field(self, tmp_path):
        pytest.importorskip("rasterio")
        import json

        from pyorps.graph.search_session import SavedCostField, \
            _FIELD_FORMAT_VERSION

        rows, cols = 48, 64
        f = _field(rows, cols)
        payload, codec_meta = encode_field(f, error_bound=2.0)
        meta = {"format": _FIELD_FORMAT_VERSION, "rows": rows, "cols": cols,
                "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 0.0],
                "crs": None, "cell_size_m": 10.0, "origin_xy": [5.0, -5.0],
                "origin_cell": 0, "algorithm": "dijkstra",
                "neighborhood": "r2", "ignore_max_cost": True,
                "graph_api": "cython", "steps": [[0, 1], [1, 0]],
                "units": "test", "has_paths": False}
        meta.update({k: v for k, v in codec_meta.items()
                     if k not in ("rows", "cols")})
        dest = tmp_path / "field.npz"
        np.savez(dest, meta=np.array(json.dumps(meta)), **payload)

        saved = SavedCostField(dest)
        assert saved.codec == "fixed-quantum"
        assert saved.error_bound == 2.0
        pts = [(15.0, -15.0), (35.0, -25.0)]
        lo = saved.costs_to(pts, bound="lower")
        hi = saved.costs_to(pts, bound="upper")
        assert np.all(hi >= lo)
        assert np.all(hi - lo <= 2.0 * (1 + 1e-6))
        truth = np.array([f[1, 1], f[2, 3]])
        assert np.all(lo <= truth) and np.all(hi >= truth)

    def test_cost_field_set_refuses_a_lossy_codec(self):
        """A psutil-driven spill must not make precision machine-dependent."""
        from pyorps.graph.search_session import CostFieldSet
        with pytest.raises(ValueError, match="machine-dependent"):
            CostFieldSet(object(), [(0.0, 0.0)], codec="fixed-quantum")

    def test_save_demands_an_explicit_error_bound(self):
        from pyorps.graph.search_session import CostField

        class _Stub:
            _closed = False
        # The check runs before anything touches the solver, so a bare
        # object is enough to reach it.
        with pytest.raises(ValueError, match="explicit error_bound"):
            CostField.save(_Stub(), "x.npz", codec="fixed-quantum")


def test_decoded_field_is_frozen():
    d = DecodedField(lower=np.zeros((2, 2)), error_bound=1.0,
                     reachable=np.ones((2, 2), bool), codec="fixed-quantum")
    with pytest.raises(Exception):
        d.error_bound = 2.0
