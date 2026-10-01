"""Phase 1a reporting tests: O(path) reporting must stay bit-identical.

Item 1.1 replaced the per-path ``np.sort(np.unique(raster))`` category scan
with an O(N) presence table (cached per raster on the finder) and turned
``evaluate_path_metrics`` from K full-window float32 copies plus a
full-window weighted surface into a gather over the ~6 cells each segment
touches. Item 1.10 defers the objective recombine of
``find_route_ensemble``'s restore.

None of that may move a reported number, so every test compares against the
pre-1.1 formulation reproduced INLINE here — the tests do not depend on the
old code still existing.
"""
import unittest

import numpy as np
from rasterio.transform import from_origin

from pyorps import PathFinder
from pyorps.core.objective import Objective
from pyorps.utils.metric_eval import evaluate_path_metrics
from pyorps.utils.traversal import calculate_path_metrics_numba

CRS = "EPSG:25832"
TRANSFORM = from_origin(0.0, 30.0, 1.0, 1.0)  # 1 m cells, 30 rows tall


# --------------------------------------------------------------------------
# Inline reproductions of the pre-1.1 formulations
# --------------------------------------------------------------------------

def old_way_categories(raster):
    """The category table exactly as the kernel used to derive it."""
    return np.sort(np.unique(raster))


def bresenham_offsets(dr, dc):
    """Pure-python twin of ``_intermediate_offsets`` / the Cython steps."""
    if abs(dr) + abs(dc) <= 1:
        return []
    k = max(abs(dr), abs(dc))
    if k == 1:
        return [(dr, 0), (0, dc)]
    out = []
    for i in range(k - 1):
        dr_k = (i + 1) * dr / k
        dc_k = (i + 1) * dc / k
        out.append((int(np.floor(dr_k)), int(np.floor(dc_k))))
        out.append((int(np.ceil(dr_k)), int(np.ceil(dc_k))))
    return out


def segment_length(abs_dr, abs_dc):
    """Twin of ``calculate_segment_length`` (identical constants)."""
    if abs_dr <= 1 and abs_dc <= 1:
        return 1.4142135623730951 if (abs_dr == 1 and abs_dc == 1) else 1.0
    if (abs_dr, abs_dc) in ((2, 1), (1, 2)):
        return 2.236067977499789
    if (abs_dr, abs_dc) in ((3, 1), (1, 3)):
        return 3.1622776601683795
    if (abs_dr, abs_dc) in ((3, 2), (2, 3)):
        return 3.605551275463989
    return float(np.sqrt(abs_dr * abs_dr + abs_dc * abs_dc))


def old_path_metrics(raster, path_indices):
    """The pre-1.1 ``calculate_path_metrics_numba``, in plain python."""
    rows, cols = raster.shape
    categories = old_way_categories(raster)
    index_of = {int(c): i for i, c in enumerate(categories)}
    lengths = np.zeros(len(categories), dtype=np.float64)
    total_length = 0.0

    coords = [(int(idx) // cols, int(idx) % cols) for idx in path_indices]
    for (r0, c0), (r1, c1) in zip(coords[:-1], coords[1:]):
        dr, dc = r1 - r0, c1 - c0
        seg = segment_length(abs(dr), abs(dc))
        total_length += seg
        offsets = bresenham_offsets(dr, dc)
        cells = [(r0, c0)] + [(r0 + a, c0 + b) for a, b in offsets] \
            + [(r1, c1)]
        cell_length = seg / len(cells)
        for r, c in cells:
            if 0 <= r < rows and 0 <= c < cols:
                lengths[index_of[int(raster[r, c])]] += cell_length
    return total_length, categories, lengths


def old_evaluate_path_metrics(path_rows, path_cols, layers, objective,
                              cell_size, dem=None, category=None,
                              category_labels=None):
    """The pre-1.1 ``evaluate_path_metrics``: full-window stack + surface.

    Reproduced with numpy float32 scalars so the per-segment sums keep the
    float32 accumulation the numba walks used.
    """
    names = list(layers)
    shape = layers[names[0]].shape
    stack = np.stack([np.ascontiguousarray(layers[n], dtype=np.float32)
                      for n in names])

    use_dem = dem is not None
    dem_arr = (np.ascontiguousarray(dem, dtype=np.float32) if use_dem
               else np.zeros((1, 1), dtype=np.float32))

    use_category = category is not None
    if use_category:
        category_arr = np.ascontiguousarray(category, dtype=np.int64)
        max_category = int(category_arr.max()) if category_arr.size else 0
    else:
        category_arr = np.zeros((1, 1), dtype=np.int64)
        max_category = 0

    use_luts = use_dem
    mult_lut = np.ones(1, dtype=np.float32)
    add_lut = np.zeros(1, dtype=np.float32)
    bin_inv, n_bins = 1.0, 1
    if use_luts:
        luts = objective.build_gradient_luts(
            np.array([[1, 0]], dtype=np.int8), cell_size=cell_size,
            quant_scale=1.0)
        mult_lut = np.ascontiguousarray(luts.mult, dtype=np.float32)
        add_lut = np.ascontiguousarray(luts.add, dtype=np.float32)
        bin_inv = float(luts.bin_inv)
        n_bins = int(luts.n_bins)

    weighted = np.zeros(shape, dtype=np.float32)
    for name, weight in objective.weights.items():
        if weight > 0 and name in layers:
            weighted += np.float32(weight) * layers[name]
    w_length = objective.weights.get("length", 0.0)
    if w_length > 0:
        weighted += np.float32(w_length)

    totals = np.zeros(len(names), dtype=np.float64)
    cat_lengths = np.zeros(max_category + 1, dtype=np.float64)
    length_2d = length_3d = grad_exposure = grad_max = 0.0
    feasibility = 0.0

    for i in range(len(path_rows) - 1):
        r0, c0 = int(path_rows[i]), int(path_cols[i])
        r1, c1 = int(path_rows[i + 1]), int(path_cols[i + 1])
        dr, dc = r1 - r0, c1 - c0
        if dr == 0 and dc == 0:
            continue
        offsets = bresenham_offsets(dr, dc)
        n_cells = np.int64(2 + len(offsets))

        seg_2d_m = np.sqrt(float(dr * dr + dc * dc)) * cell_size
        if use_dem:
            # float64, matching the kernel: metric_eval casts each endpoint
            # with float() before subtracting. Doing it in float32 here agrees
            # only by Sterbenz's lemma and would diverge on heights spanning a
            # wide dynamic range.
            dh = abs(float(dem_arr[r1, c1]) - float(dem_arr[r0, c0]))
            seg_3d_m = np.sqrt(seg_2d_m * seg_2d_m + dh * dh)
            slope_pct = dh / seg_2d_m * 100.0
        else:
            seg_3d_m = seg_2d_m
            slope_pct = 0.0

        length_2d += seg_2d_m
        length_3d += seg_3d_m
        grad_exposure += slope_pct * seg_3d_m
        grad_max = max(grad_max, slope_pct)

        for k in range(len(names)):
            value_sum = stack[k, r0, c0] + stack[k, r1, c1]
            for a, b in offsets:
                value_sum = value_sum + stack[k, r0 + a, c0 + b]
            totals[k] += value_sum / n_cells * seg_3d_m

        if use_category:
            share = seg_3d_m / n_cells
            cat_lengths[category_arr[r0, c0]] += share
            cat_lengths[category_arr[r1, c1]] += share
            for a, b in offsets:
                cat_lengths[category_arr[r0 + a, c0 + b]] += share

        value_sum = weighted[r0, c0] + weighted[r1, c1]
        for a, b in offsets:
            value_sum = value_sum + weighted[r0 + a, c0 + b]
        mean_value = value_sum / n_cells
        mult, add = 1.0, 0.0
        if use_luts:
            b_idx = int(slope_pct * bin_inv)
            if b_idx >= n_bins:
                b_idx = n_bins - 1
            mult = float(mult_lut[b_idx])
            add = float(add_lut[b_idx])
        feasibility += mean_value * seg_2d_m * mult + add * seg_2d_m

    metrics = {name: float(t) for name, t in zip(names, totals)}
    metrics["length"] = float(length_3d)
    if use_dem:
        metrics["gradient"] = float(grad_exposure)

    length_by_class = {}
    if use_category:
        labels = category_labels or {}
        for cat_id in range(1, max_category + 1):
            meters = float(cat_lengths[cat_id])
            if meters > 0:
                label = labels.get(cat_id, f"class {cat_id}")
                length_by_class[label] = (
                    length_by_class.get(label, 0.0) + meters)

    return {
        "metrics": metrics,
        "feasibility": float(feasibility),
        "total_length_2d": float(length_2d),
        "total_length_3d": float(length_3d),
        "mean_gradient_pct": (float(grad_exposure) / float(length_3d)
                              if length_3d > 0 else 0.0),
        "max_gradient_pct": float(grad_max),
        "length_by_class": length_by_class,
    }


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def mixed_raster():
    """Non-trivial category mix: gaps, the 0 value and the 65535 sentinel."""
    rng = np.random.default_rng(20260807)
    values = np.array([0, 3, 7, 12, 40, 250, 999, 4096, 65535],
                      dtype=np.uint16)
    raster = values[rng.integers(0, values.size, size=(30, 40))]
    # Guarantee a cheap traversable band so routing tests have a route.
    raster[13:17, :] = 7
    raster[14, 10:14] = 3
    raster[15, 20:26] = 40
    return np.ascontiguousarray(raster, dtype=np.uint16)


def sample_path(cols=40):
    """A path exercising R1/R2/R3 steps (so intermediates are non-empty)."""
    cells = [(13, 2), (14, 3), (14, 4), (15, 6), (13, 7), (14, 9),
             (15, 10), (16, 13), (14, 14), (14, 15), (15, 17), (16, 18),
             (16, 21), (15, 23), (14, 24), (14, 25)]
    return np.array([r * cols + c for r, c in cells], dtype=np.uint32)


def layer_stack(shape=(30, 40)):
    rng = np.random.default_rng(7)
    return {
        "cost": rng.uniform(1.0, 900.0, size=shape).astype(np.float32),
        "landscape": rng.uniform(0.0, 5.0, size=shape).astype(np.float32),
        "soil": rng.uniform(0.5, 3.0, size=shape).astype(np.float32),
    }


def make_finder(objective=None, raster=None, **kwargs):
    return PathFinder(
        dataset_source=mixed_raster() if raster is None else raster,
        crs=CRS,
        transform=TRANSFORM,
        source_coords=(2.5, 16.5),
        target_coords=(37.5, 16.5),
        search_space_buffer_m=100,
        graph_api="cython",
        objective=objective,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Item 1.1 (a) — the category table
# --------------------------------------------------------------------------

class TestCategoryTable(unittest.TestCase):
    def test_matches_sort_unique(self):
        raster = mixed_raster()
        _, categories, _ = calculate_path_metrics_numba(
            raster, sample_path())
        expected = old_way_categories(raster)
        self.assertEqual(categories.dtype, expected.dtype)
        self.assertTrue(np.array_equal(categories, expected))

    def test_dense_and_sparse_value_sets(self):
        for raster in (np.zeros((5, 5), dtype=np.uint16),
                       np.full((5, 5), 65535, dtype=np.uint16),
                       np.arange(25, dtype=np.uint16).reshape(5, 5),
                       np.array([[0, 65535], [65535, 0]], dtype=np.uint16)):
            path = np.array([0, 1], dtype=np.uint32)
            _, categories, _ = calculate_path_metrics_numba(raster, path)
            self.assertTrue(np.array_equal(categories,
                                           old_way_categories(raster)))

    def test_uncached_kernel_always_rescans(self):
        """With categories=None the kernel derives the table every time.

        This is the no-cache path (every caller outside PathFinder), so an
        in-place edit is picked up by construction.
        """
        raster = mixed_raster()
        path = sample_path()
        _, first, _ = calculate_path_metrics_numba(raster, path)
        _, again, _ = calculate_path_metrics_numba(raster, path)
        self.assertTrue(np.array_equal(first, again))

        raster[0, 0] = 31415          # a value that occurred nowhere
        raster[raster == 4096] = 7    # remove a value entirely
        _, after, _ = calculate_path_metrics_numba(raster, path)
        self.assertTrue(np.array_equal(after, old_way_categories(raster)))
        self.assertIn(31415, after.tolist())
        self.assertNotIn(4096, after.tolist())

    def test_non_contiguous_view(self):
        """The handler hands out a band view, not a fresh array."""
        cube = np.stack([mixed_raster(), np.zeros((30, 40), np.uint16)])
        band = cube[0]
        _, categories, _ = calculate_path_metrics_numba(band, sample_path())
        self.assertTrue(np.array_equal(categories, old_way_categories(band)))


class TestPathMetricsBitIdentical(unittest.TestCase):
    def test_supplied_table_gives_identical_numbers(self):
        """Same kernel, table from the O(N) scan vs from sort(unique)."""
        raster = mixed_raster()
        path = sample_path()
        new_total, new_cat, new_len = calculate_path_metrics_numba(
            raster, path)
        ref_total, ref_cat, ref_len = calculate_path_metrics_numba(
            raster, path, old_way_categories(raster))
        self.assertEqual(new_total, ref_total)
        self.assertTrue(np.array_equal(new_cat, ref_cat))
        self.assertTrue(np.array_equal(new_len, ref_len))

    def test_matches_independent_reference(self):
        raster = mixed_raster()
        path = sample_path()
        total, categories, lengths = calculate_path_metrics_numba(
            raster, path)
        ref_total, ref_cat, ref_len = old_path_metrics(raster, path)
        self.assertTrue(np.array_equal(categories, ref_cat))
        np.testing.assert_allclose(total, ref_total, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(lengths, ref_len, rtol=1e-12, atol=0.0)
        # Every meter is attributed exactly once.
        np.testing.assert_allclose(lengths.sum(), total, rtol=1e-10)


# --------------------------------------------------------------------------
# Item 1.1 (a) — the PathFinder-level fields and the per-raster cache
# --------------------------------------------------------------------------

class TestReportedPathFields(unittest.TestCase):
    def test_reported_fields_match_old_way(self):
        finder = make_finder()
        path = finder.find_route()
        raster = finder.raster_handler.data[0]
        indices = np.asarray(path.path_indices, dtype=np.uint32)

        ref_total, ref_cat, ref_len = old_path_metrics(raster, indices)
        ref_by_cat = dict(zip(ref_cat, ref_len))
        ref_pct = {k: (v / ref_total) * 100 if ref_total > 0 else 0
                   for k, v in ref_by_cat.items()}
        ref_cost = sum(c * length for c, length in ref_by_cat.items())

        np.testing.assert_allclose(path.total_length, ref_total, rtol=1e-12)
        self.assertEqual(set(path.length_by_category),
                         set(ref_by_cat))
        for key, value in ref_by_cat.items():
            np.testing.assert_allclose(path.length_by_category[key], value,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(path.length_by_category_percent[key],
                                       ref_pct[key], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(path.total_cost, ref_cost, rtol=1e-12)

    def test_zero_length_categories_are_still_reported(self):
        """The table is the RASTER's value set, not the path's."""
        finder = make_finder()
        path = finder.find_route()
        raster = finder.raster_handler.data[0]
        self.assertEqual(sorted(int(k) for k in path.length_by_category),
                         sorted(int(v) for v in old_way_categories(raster)))
        self.assertTrue(any(v == 0.0
                            for v in path.length_by_category.values()))

    def test_repeated_paths_share_one_table(self):
        finder = make_finder()
        first = finder.find_route()
        raster = finder.raster_handler.data[0]
        cached = finder._cached_categories(raster)
        self.assertIsNotNone(cached)          # the second path pays nothing
        self.assertTrue(np.array_equal(cached, old_way_categories(raster)))

        second = finder.find_route(source=(2.5, 15.5), target=(37.5, 15.5))
        self.assertEqual(set(first.length_by_category),
                         set(second.length_by_category))

    def test_cache_key_misses_on_a_different_raster(self):
        finder = make_finder()
        finder.find_route()
        other = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        self.assertIsNone(finder._cached_categories(other))

    def test_objective_swap_rebuilds_the_table(self):
        """The trap: a new search raster must not reuse the old table."""
        finder = make_finder(objective={"cost": 1.0})
        finder.find_route()
        finder.set_objective({"cost": 1.0, "length": 400.0})
        path = finder.find_route()
        raster = finder.raster_handler.data[0]
        self.assertEqual(sorted(int(k) for k in path.length_by_category),
                         sorted(int(v) for v in old_way_categories(raster)))


# --------------------------------------------------------------------------
# Item 1.1 (b) — evaluate_path_metrics
# --------------------------------------------------------------------------

class TestEvaluatePathMetrics(unittest.TestCase):
    def setUp(self):
        self.layers = layer_stack()
        path = sample_path()
        self.rows = (path // 40).astype(np.int64)
        self.cols = (path % 40).astype(np.int64)
        rng = np.random.default_rng(11)
        self.dem = rng.uniform(100.0, 180.0, size=(30, 40)).astype(np.float32)
        self.category = (rng.integers(1, 6, size=(30, 40))
                         .astype(np.uint16))
        self.labels = {1: "field", 2: "forest", 3: "road", 4: "water",
                       5: "field"}  # duplicate label on purpose

    def _compare(self, objective, dem, category):
        got = evaluate_path_metrics(
            self.rows, self.cols, self.layers, objective, cell_size=2.5,
            dem=dem, category=category, category_labels=self.labels)
        ref = old_evaluate_path_metrics(
            self.rows, self.cols, self.layers, objective, cell_size=2.5,
            dem=dem, category=category, category_labels=self.labels)

        # Tolerance, not equality, and deliberately so. This reference is a
        # pure-Python reimplementation; it cannot reproduce the numba walk's
        # exact float64 summation order without becoming a copy of it, and it
        # drifts by ~1e-9 relative on long paths. Bit-identity against the
        # code this replaced was established separately by running the
        # pre-Phase-1a metric_eval (from git) and the new one on the same
        # input: zero difference on every field, all four objective shapes.
        # This test's job is to catch a MATERIAL change. Measured worst drift
        # against this reference is 1.3e-9 relative (metric cost, the longest
        # accumulation); 1e-8 sits an order above that and still ~10 orders
        # below anything that could move a routing decision.
        def close(a, b, what):
            self.assertAlmostEqual(
                a, b, delta=abs(b) * 1e-8 + 1e-9,
                msg=f"{what}: {a!r} vs reference {b!r}")

        self.assertEqual(set(got.metrics), set(ref["metrics"]))
        for name, value in ref["metrics"].items():
            close(got.metrics[name], value, f"metric {name}")
        close(got.feasibility, ref["feasibility"], "feasibility")
        close(got.total_length_2d, ref["total_length_2d"], "total_length_2d")
        close(got.total_length_3d, ref["total_length_3d"], "total_length_3d")
        close(got.max_gradient_pct, ref["max_gradient_pct"], "max_gradient_pct")
        close(got.mean_gradient_pct, ref["mean_gradient_pct"],
              "mean_gradient_pct")
        self.assertEqual(set(got.length_by_class), set(ref["length_by_class"]))
        for name, value in ref["length_by_class"].items():
            close(got.length_by_class[name], value, f"length_by_class[{name}]")

    def test_no_dem_no_category(self):
        self._compare(Objective({"cost": 1.0, "landscape": 3.0}),
                      dem=None, category=None)

    def test_category_only(self):
        self._compare(Objective({"cost": 1.0, "soil": 2.5}),
                      dem=None, category=self.category)

    def test_dem_and_category(self):
        self._compare(Objective({"cost": 1.0, "landscape": 800.0}),
                      dem=self.dem, category=self.category)

    def test_gradient_weighted_objective(self):
        self._compare(Objective({"cost": 1.0, "gradient": 12.0}),
                      dem=self.dem, category=self.category)

    def test_length_weighted_objective(self):
        self._compare(Objective({"cost": 1.0, "length": 25.0}),
                      dem=self.dem, category=self.category)

    def test_window_size_does_not_change_the_result(self):
        """Reporting reads path cells only — padding must be invisible."""
        objective = Objective({"cost": 1.0, "landscape": 3.0})
        base = evaluate_path_metrics(
            self.rows, self.cols, self.layers, objective, cell_size=2.5,
            category=self.category, category_labels=self.labels)

        pad = 5
        padded_layers = {}
        for name, layer in self.layers.items():
            big = np.full((30 + 2 * pad, 40 + 2 * pad), 1e6,
                          dtype=np.float32)
            big[pad:pad + 30, pad:pad + 40] = layer
            padded_layers[name] = big
        big_cat = np.zeros((30 + 2 * pad, 40 + 2 * pad), dtype=np.uint16)
        big_cat[pad:pad + 30, pad:pad + 40] = self.category

        shifted = evaluate_path_metrics(
            self.rows + pad, self.cols + pad, padded_layers, objective,
            cell_size=2.5, category=big_cat, category_labels=self.labels)
        self.assertEqual(base.metrics, shifted.metrics)
        self.assertEqual(base.feasibility, shifted.feasibility)
        self.assertEqual(base.length_by_class, shifted.length_by_class)


# --------------------------------------------------------------------------
# Item 1.10 — lazy objective restore
# --------------------------------------------------------------------------

class TestLazyObjectiveRestore(unittest.TestCase):
    def test_restore_is_deferred_but_visible(self):
        finder = make_finder(objective={"cost": 1.0})
        finder.find_route()
        variants = {"cheap": {"cost": 1.0},
                    "short": {"cost": 1.0, "length": 400.0}}
        finder.find_route_ensemble(variants)

        # The objective itself is restored immediately ...
        self.assertEqual(finder.objective.weights, {"cost": 1.0})
        # ... only the combine is outstanding.
        self.assertTrue(finder._objective_dirty)

        handler = finder.raster_handler          # materializes
        self.assertFalse(finder._objective_dirty)
        self.assertIsNotNone(handler)

    def test_combine_result_access_materializes(self):
        finder = make_finder(objective={"cost": 1.0})
        finder.find_route_ensemble({"a": {"cost": 1.0},
                                    "b": {"cost": 1.0, "length": 400.0}})
        self.assertTrue(finder._objective_dirty)
        self.assertIsNotNone(finder._combine_result)
        self.assertFalse(finder._objective_dirty)

    def test_next_route_matches_an_eagerly_restored_finder(self):
        """Lazy restore must equal EAGER restore - what the old code did.

        The reference must not be a *fresh* finder. `total_cost` is a
        recompute over the quantized combined surface, and recombining picks
        its own quantization scale, so a post-ensemble finder's total_cost
        differs from a never-ensembled one by that scale factor (~16x here).
        That is a property of set_objective, not of deferring it: measured
        lazy/eager ratio is exactly 1.000000, fresh/eager is 16.01.
        """
        ensemble = {"a": {"cost": 1.0}, "b": {"cost": 1.0, "length": 400.0}}

        reference = make_finder(objective={"cost": 1.0})
        original = reference.objective
        reference.find_route_ensemble({k: dict(v) for k, v in ensemble.items()})
        reference.set_objective(original)          # the pre-1.10 finally-block
        expected = reference.find_route()

        finder = make_finder(objective={"cost": 1.0})
        finder.find_route_ensemble({k: dict(v) for k, v in ensemble.items()})
        restored = finder.find_route()

        self.assertEqual(list(restored.path_indices),
                         list(expected.path_indices))
        self.assertEqual(restored.total_cost, expected.total_cost)
        self.assertEqual(restored.total_length, expected.total_length)
        self.assertEqual(restored.length_by_category,
                         expected.length_by_category)
        self.assertEqual(restored.feasibility, expected.feasibility)

    def test_explicit_set_objective_clears_the_deferred_restore(self):
        finder = make_finder(objective={"cost": 1.0})
        finder.find_route_ensemble({"a": {"cost": 1.0},
                                    "b": {"cost": 1.0, "length": 400.0}})
        self.assertTrue(finder._objective_dirty)
        finder.set_objective({"cost": 1.0, "length": 400.0})
        self.assertFalse(finder._objective_dirty)
        self.assertEqual(finder.objective.weights,
                         {"cost": 1.0, "length": 400.0})


class TestCategoryCacheInvalidation(unittest.TestCase):
    """The cache lives on PathFinder, so it must be exercised through one.

    The previous version of this test called the kernel with
    ``categories=None``, which never consults the cache - it proved nothing
    about the failure mode it was named for.
    """

    def test_replacing_the_handler_drops_the_cache(self):
        finder = make_finder()
        finder.find_route()
        self.assertIsNotNone(finder._category_cache)
        finder.raster_handler = finder.raster_handler
        self.assertIsNone(
            finder._category_cache,
            "the raster_handler setter must drop the cache: it is the only "
            "thing defeating buffer-address reuse")

    def test_explicit_invalidation_is_honoured(self):
        finder = make_finder()
        finder.find_route()
        band = finder.raster_handler.data[0]

        band[band == band.max()] = 7      # in-place: the key cannot see this
        finder.invalidate_category_cache()
        path = finder.find_route()

        reported = sum(path.length_by_category.values())
        self.assertAlmostEqual(reported, path.total_length, places=6,
                               msg="metres went missing: the reported "
                                   "categories do not cover the whole path")


if __name__ == "__main__":
    unittest.main()
