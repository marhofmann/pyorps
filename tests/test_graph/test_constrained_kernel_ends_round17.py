"""Line ends in the constrained CPU kernels (round 17, package P2).

Covers the three line-end arguments of the Cython kernels and the height
reconstruction fix:

- ``target_min_span`` (all six kernel paths: Dijkstra dense and sparse,
  delta-stepping 2-D, variable-height dense, sparse and lazy): a state at the
  target is accepted only when its last span is at least that long, unless the
  line has no line tower at all;
- ``source_attach_height`` / ``target_attach_height`` (the three height
  kernels): the connection-point heights of the two ends enter the clearance
  check of the first and the last span, so the search optimises the first and
  the last tower's position and height for them; these end spans are sampled
  in full (review fix: the first cell after the start, the source cell, a
  one-cell last span and the target cell itself), the spans between two line
  towers keep the kernel's sampling (cells a+2 .. b);
- the reported tower heights are the towers' own (before the fix, tower k
  reported the height class of tower k-1).

The optimality reference is an exhaustive dynamic programme over every tower
chain of a one-row corridor, written here in plain Python (independent of the
kernels and of the library's profile classes). The corridor admits exactly one
path (east, cell by cell), so a route is fully described by its tower
positions and heights, and with span bins of one cell every span length has
its own search state: the kernels are then exact (fact A6.3 of the plan).
"""

import math

import numpy as np
import pytest

from pyorps.utils import _constrained_delta as DL
from pyorps.utils import _constrained_dijkstra as DJ
from pyorps.utils.neighborhood import get_neighborhood_steps

FORBIDDEN = 65535
STEPS = get_neighborhood_steps("r1")
EAST = next(i for i, (dr, dc) in enumerate(STEPS) if dr == 0 and dc == 1)
REL = 1e-9
# The compact-dense sparse height kernel keeps its distances in float32.
REL_F32 = 1e-6


def rel(kernel):
    return REL_F32 if kernel == "hsparse" else REL

KERNELS_2D = ("dij_dense", "dij_sparse", "delta2d")
KERNELS_H = ("hdense", "hsparse", "lazy")


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # The delta-stepping kernels read OMP_NUM_THREADS per call; one thread
    # keeps tie-breaking deterministic and the test independent of the machine.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------- inputs


def tables(tower_map=((0, 40.0), (200, 300.0)), straight_cost=50.0):
    n = len(STEPS)
    ang = [math.atan2(float(dc), float(dr)) for dr, dc in STEPS]
    acl = np.zeros((n, n), np.float32)
    val = np.ones((n, n), np.uint8)
    for i in range(n):
        for j in range(n):
            d = ang[j] - ang[i]
            turn = abs(math.atan2(math.sin(d), math.cos(d)))
            if turn > math.radians(90.0):
                val[i, j] = 0
                acl[i, j] = np.inf
            else:
                acl[i, j] = math.degrees(turn)
    xs = np.array([p[0] for p in tower_map], float)
    ys = np.array([p[1] for p in tower_map], float)
    return dict(steps=np.ascontiguousarray(STEPS, dtype=np.int8), angle_cost_lut=acl,
                angle_valid_lut=val,
                step_distances=np.array([math.hypot(dr, dc) for dr, dc in STEPS], np.float32),
                tower_terrain_costs=np.interp(np.arange(65536), xs, ys).astype(np.float32),
                tower_angle_costs=np.full((n, n), straight_cost, np.float32))


class Corridor:
    """One open row; source at its west end, target at its east end."""

    def __init__(self, values, *, min_span, max_span, rows=3, row=1, dem=None, obstacles=None,
                 heights=(12.0,), premiums=(0.0,), w=8.0, tension=100.0, min_clearance=5.0,
                 bin_size=1.0, n_bins=None, tower_map=((0, 40.0), (200, 300.0))):
        n = len(values)
        self.n, self.row, self.cols = n, row, n
        self.raster = np.full((rows, n), FORBIDDEN, np.uint16)
        self.raster[row, :] = np.asarray(values, np.uint16)
        self.dem = np.zeros((rows, n), np.float32)
        if dem is not None:
            self.dem[row, :] = np.asarray(dem, np.float32)
        self.obs = np.zeros((rows, n), np.float32)
        for col, h in (obstacles or {}).items():
            self.obs[row, col] = h
        self.heights = np.asarray(heights, np.float32)
        self.premiums = np.asarray(premiums, np.float32)
        self.min_span, self.max_span = float(min_span), float(max_span)
        self.w, self.tension, self.mc = float(w), float(tension), float(min_clearance)
        self.tb = tables(tower_map)
        self.bins = dict(n_span_bins=n_bins or int(math.ceil(max_span / bin_size)) + 1,
                         span_bin_size=float(bin_size), min_span=self.min_span,
                         max_span=self.max_span)
        self.min_abs_margin = math.inf

    # ---- kernel calls

    def base(self):
        return dict(raster=self.raster, source_row=self.row, source_col=0,
                    target_row=self.row, target_col=self.n - 1, **self.tb, **self.bins)

    def hkw(self):
        return dict(dem_data=self.dem, cell_size=1.0, tower_heights=self.heights,
                    height_premiums=self.premiums, conductor_weight_per_m=self.w,
                    conductor_tension=self.tension, min_clearance_val=self.mc,
                    obstacle_heights=self.obs, max_gradient_pct=1.0e6, gradient_scale=0.0)

    def run(self, kernel, **opts):
        if kernel == "dij_dense":
            res = DJ.constrained_dijkstra_2d(**self.base(), return_dist=1, **opts)
        elif kernel == "dij_sparse":
            res = DJ.constrained_dijkstra_2d(**self.base(), force_sparse=1, return_dist=1, **opts)
        elif kernel == "delta2d":
            res = DL.constrained_delta_stepping_2d(**self.base(), return_dist=1, **opts)
        elif kernel == "hdense":
            res = DL.constrained_delta_stepping_height_2d(**self.base(), **self.hkw(), return_dist=1, **opts)
        elif kernel == "hsparse":
            res = DL.constrained_delta_stepping_height_2d(**self.base(), **self.hkw(), force_sparse=1,
                                                          return_dist=1, **opts)
        elif kernel == "lazy":
            res = DL.constrained_delta_stepping_lazy(**self.base(), **self.hkw(), return_dist=1, **opts)
        else:  # pragma: no cover
            raise AssertionError(kernel)
        path = [int(c) for c in res[0]]
        towers = [int(c) % self.cols for c in res[1]]
        heights = [float(h) for h in res[2]] if len(res) == 4 else None
        return path, towers, heights, float(res[-1])

    # ---- independent model

    def steps_cost(self):
        r = self.raster[self.row].astype(float)
        return float(sum((r[c] + r[c + 1]) * 0.5 for c in range(self.n - 1)))

    def tower_cost(self, col):
        val = int(self.raster[self.row, col])
        return float(self.tb["tower_terrain_costs"][val]) + float(self.tb["tower_angle_costs"][EAST, EAST])

    def margin(self, a, b, ha, hb, first=2):
        """Smallest clearance margin of the span a -> b (cells of the row).

        Mirrors the kernel's sampling, chord between DEM + height at both
        ends, parabolic sag w x (L - x) / (2 T):

        - ``first=2``, a span between two line towers: cells a+2 .. b (the
          first cell after tower A is not tested; a span of one cell is not
          checked at all);
        - ``first=1``, the last span to a target connection height: every
          cell a+1 .. b, also when the span is one cell long;
        - ``first=0``, a span from a source connection structure: cells
          a .. b (the source cell itself included).
        """
        n = b - a
        if first >= 2 and n <= 1:
            return math.inf
        dem, obs = self.dem[self.row].astype(float), self.obs[self.row].astype(float)
        za, zb, L = dem[a] + ha, dem[b] + hb, float(n)
        m = math.inf
        for step in range(min(first, 2), n + 1):
            x, c = float(step), a + step
            cz = za + (zb - za) * x / L - self.w * x * (L - x) / (2.0 * self.tension)
            m = min(m, cz - dem[c] - obs[c] - self.mc)
        self.min_abs_margin = min(self.min_abs_margin, abs(m))
        return m

    def clears(self, a, b, ha, hb, first=2):
        return self.margin(a, b, ha, hb, first) >= 0.0

    def source_options(self, opts, with_heights):
        """(height, premium) of the structure at the source."""
        if not with_heights:
            return [(None, 0.0)]
        if opts.get("source_attach_height") is not None:
            return [(float(opts["source_attach_height"]), 0.0)]
        return [(float(h), float(p)) for h, p in zip(self.heights, self.premiums)]

    @staticmethod
    def first_cell(q, opts):
        """Sampling of a span that starts at q: from the source cell (0) when it
        starts at a source connection structure, else the given default."""
        return 0 if (q == 0 and opts.get("source_attach_height") is not None) else None

    def span_ok(self, a, b, ha, hb, with_heights, opts=None):
        s = b - a
        if not (self.min_span <= s < self.max_span):
            return False
        first = self.first_cell(a, opts or {})
        return not with_heights or self.clears(a, b, ha, hb, 2 if first is None else first)

    def end_ok(self, q, hq, opts, with_heights):
        s = self.n - 1 - q
        if s >= self.max_span:
            return False
        tms = float(opts.get("target_min_span", 0.0))
        if tms > 0.0 and q != 0 and s < tms:
            return False
        tgt = opts.get("target_attach_height")
        if with_heights and tgt is not None:
            last = self.n - 1
            if float(tgt) - float(self.obs[self.row, last]) < self.mc:  # the target cell itself
                return False
            first = self.first_cell(q, opts)
            return self.clears(q, last, hq, float(tgt), 1 if first is None else first)
        return True

    def oracle(self, opts, with_heights):
        """Exhaustive DP over every tower chain: the optimal J (inf if none)."""
        last = self.n - 1
        classes = ([(float(h), float(p)) for h, p in zip(self.heights, self.premiums)]
                   if with_heights else [(None, 0.0)])
        best = {0: {h: p for h, p in self.source_options(opts, with_heights)}}
        for p in range(1, last):
            here = {}
            for hk, pk in classes:
                cand = math.inf
                for q, prev in best.items():
                    for hq, cq in prev.items():
                        if cq < cand and self.span_ok(q, p, hq, hk, with_heights, opts):
                            cand = min(cand, cq + self.tower_cost(p) + pk)
                if cand < math.inf:
                    here[hk] = min(here.get(hk, math.inf), cand)
            if here:
                best[p] = here
        total = math.inf
        for q, prev in best.items():
            for hq, cq in prev.items():
                if self.end_ok(q, hq, opts, with_heights):
                    total = min(total, cq)
        return total + self.steps_cost() if total < math.inf else math.inf

    def price(self, towers, heights, opts, with_heights):
        """Check a kernel result against every rule and price it (inf if invalid)."""
        if towers != sorted(set(towers)) or any(not 0 < t < self.n - 1 for t in towers):
            return math.inf
        if with_heights:
            if heights is None or len(heights) != len(towers):
                return math.inf
            if any(float(h) not in {float(x) for x in self.heights} for h in heights):
                return math.inf
        hs = heights if with_heights else [None] * len(towers)
        prem = {float(h): float(p) for h, p in zip(self.heights, self.premiums)}
        body = sum(self.tower_cost(t) + (prem[float(h)] if with_heights else 0.0)
                   for t, h in zip(towers, hs))
        for i in range(1, len(towers)):
            if not self.span_ok(towers[i - 1], towers[i], hs[i - 1], hs[i], with_heights):
                return math.inf
        best_src = math.inf
        for h0, p0 in self.source_options(opts, with_heights):
            if towers:
                ok = self.span_ok(0, towers[0], h0, hs[0], with_heights, opts)
                ok = ok and self.end_ok(towers[-1], hs[-1], opts, with_heights)
            else:
                ok = self.end_ok(0, h0, opts, with_heights)
            if ok:
                best_src = min(best_src, p0)
        if best_src == math.inf:
            return math.inf
        return self.steps_cost() + body + best_src


def assert_optimal(case, kernel, opts, with_heights):
    path, towers, heights, dist = case.run(kernel, **opts)
    ref = case.oracle(opts, with_heights)
    if ref == math.inf:
        assert dist == math.inf and path == [] and towers == []
        return towers, heights
    assert dist == pytest.approx(ref, rel=rel(kernel)), (kernel, opts, towers, heights)
    assert path == [case.row * case.cols + c for c in range(case.n)]
    priced = case.price(towers, heights, opts, with_heights)
    assert priced == pytest.approx(ref, rel=REL), (kernel, opts, towers, heights, priced, ref)
    return towers, heights


def spans_of(towers, n):
    chain = [0] + list(towers) + [n - 1]
    return [b - a for a, b in zip(chain[:-1], chain[1:])]


# ---------------------------------------------------------------- neutral defaults


def _grid_case():
    rng = np.random.default_rng(5)
    r = rng.integers(1, 200, size=(14, 16)).astype(np.uint16)
    r[rng.random(r.shape) < 0.1] = FORBIDDEN
    r[1, 1] = r[12, 13] = 30
    dem = (rng.normal(0, 0.5, size=r.shape)).astype(np.float32)
    obs = np.where(rng.random(r.shape) < 0.1, 10.0, 0.0).astype(np.float32)
    base = dict(raster=r, source_row=1, source_col=1, target_row=12, target_col=13, **tables(),
                n_span_bins=4, span_bin_size=2.0, min_span=2.0, max_span=6.0)
    hkw = dict(dem_data=dem, cell_size=1.0, tower_heights=np.array([20.0, 11.0], np.float32),
               height_premiums=np.array([60.0, 0.0], np.float32), conductor_weight_per_m=1.0,
               conductor_tension=500.0, min_clearance_val=5.0, obstacle_heights=obs,
               max_gradient_pct=500.0, gradient_scale=0.3)
    return base, hkw


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and np.array_equal(x, y)
        else:
            assert x == y


@pytest.mark.parametrize("force_sparse", [0, 1])
def test_neutral_defaults_equal_omitted_arguments(force_sparse):
    base, hkw = _grid_case()
    dj = DJ.constrained_dijkstra_2d(**base, force_sparse=force_sparse, return_dist=1)
    _same(dj, DJ.constrained_dijkstra_2d(**base, force_sparse=force_sparse, return_dist=1,
                                         target_min_span=0.0))
    d2 = DL.constrained_delta_stepping_2d(**base, return_dist=1)
    _same(d2, DL.constrained_delta_stepping_2d(**base, return_dist=1, target_min_span=0.0))
    for neutral in (dict(target_min_span=0.0, source_attach_height=None, target_attach_height=None),
                    dict(source_attach_height=float("nan"), target_attach_height=float("nan"))):
        h = DL.constrained_delta_stepping_height_2d(**base, **hkw, force_sparse=force_sparse, return_dist=1)
        _same(h, DL.constrained_delta_stepping_height_2d(**base, **hkw, force_sparse=force_sparse,
                                                         return_dist=1, **neutral))
        lz = DL.constrained_delta_stepping_lazy(**base, **hkw, return_dist=1)
        _same(lz, DL.constrained_delta_stepping_lazy(**base, **hkw, return_dist=1, **neutral))
    assert len(dj[0]) > 0 and len(h[0]) > 0


# ---------------------------------------------------------------- argument checks


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_target_min_span_rejects_invalid_values(bad):
    case = Corridor([100] * 10, min_span=3, max_span=8)
    for kernel in KERNELS_2D + KERNELS_H:
        with pytest.raises(ValueError, match="target_min_span"):
            case.run(kernel, target_min_span=bad)


@pytest.mark.parametrize("name", ["source_attach_height", "target_attach_height"])
@pytest.mark.parametrize("bad", [0.0, -2.0, float("inf"), 1.0e40])
def test_attach_heights_reject_invalid_values(name, bad):
    case = Corridor([100] * 10, min_span=3, max_span=8)
    for kernel in KERNELS_H:
        with pytest.raises(ValueError, match=name):
            case.run(kernel, **{name: bad})


@pytest.mark.parametrize("kernel", KERNELS_2D)
@pytest.mark.parametrize("name", ["source_attach_height", "target_attach_height"])
def test_kernels_without_heights_refuse_attach_heights(kernel, name):
    # The 2-D kernels have no height model: an end height must not be ignored.
    case = Corridor([100] * 10, min_span=3, max_span=8)
    with pytest.raises(TypeError):
        case.run(kernel, **{name: 10.0})


# ---------------------------------------------------------------- target_min_span


@pytest.mark.parametrize("kernel", KERNELS_2D + KERNELS_H)
def test_target_min_span_on_the_1x24_corridor(kernel):
    # The planner's probe (plan A2, bins of min_span as in the finder) ends
    # with a short last span: [7, 7, 7, 2].
    probe = Corridor([100] * 24, min_span=3, max_span=8, bin_size=3.0, n_bins=3,
                     tower_map=((0, 100.0), (500, 100.0)))
    _, towers, _, dist = probe.run(kernel)
    assert spans_of(towers, 24) == [7, 7, 7, 2]
    # The rule needs exact span states: with bins of min_span the merge of
    # span states (plan A6.3) can hide every valid line end on this tie-rich
    # corridor, so the rule is checked with bins of one cell.
    case = Corridor([100] * 24, min_span=3, max_span=8, tower_map=((0, 100.0), (500, 100.0)))
    _, towers_e, _, dist_e = case.run(kernel, target_min_span=3.0)
    spans = spans_of(towers_e, 24)
    assert all(3 <= s < 8 for s in spans), spans
    assert dist_e == pytest.approx(dist, rel=rel(kernel))  # a tie on the uniform corridor
    assert dist_e == pytest.approx(case.oracle({"target_min_span": 3.0}, kernel in KERNELS_H), rel=rel(kernel))


@pytest.mark.parametrize("kernel", KERNELS_2D + KERNELS_H)
def test_direct_span_from_the_source_is_exempt(kernel):
    # 6 cells: a single span of 5 m from the source, no line tower; the rule
    # asks for 7 m, but a line without any line tower is exempt.
    case = Corridor([100] * 6, min_span=3, max_span=8)
    path0, towers0, _, dist0 = case.run(kernel)
    path, towers, _, dist = case.run(kernel, target_min_span=7.0)
    assert towers == towers0 == []
    assert path == path0 and dist == dist0 < math.inf


@pytest.mark.parametrize("kernel", KERNELS_2D + KERNELS_H)
def test_no_valid_line_end_gives_an_empty_result(kernel):
    # L-shaped corridor whose last leg (2 m) is shorter than the rule (3 m):
    # the turn needs a tower, so every line ends with a 2 m span.
    r = np.full((8, 10), FORBIDDEN, np.uint16)
    r[1, 1:8] = 50
    r[1:4, 7] = 50
    case = Corridor([50] * 3, min_span=2, max_span=6)
    case.raster, case.cols = r, 10
    case.dem = np.zeros(r.shape, np.float32)
    case.obs = np.zeros(r.shape, np.float32)
    base = dict(raster=r, source_row=1, source_col=1, target_row=3, target_col=7, **case.tb, **case.bins)
    kw = {} if kernel in KERNELS_2D else case.hkw()
    fn = {"dij_dense": DJ.constrained_dijkstra_2d, "dij_sparse": DJ.constrained_dijkstra_2d,
          "delta2d": DL.constrained_delta_stepping_2d, "hdense": DL.constrained_delta_stepping_height_2d,
          "hsparse": DL.constrained_delta_stepping_height_2d, "lazy": DL.constrained_delta_stepping_lazy}[kernel]
    if kernel in ("dij_sparse", "hsparse"):
        kw["force_sparse"] = 1
    free = fn(**base, **kw, return_dist=1)
    assert len(free[0]) > 0 and free[-1] < math.inf
    res = fn(**base, **kw, return_dist=1, target_min_span=3.0)
    assert len(res[0]) == 0 and len(res[1]) == 0 and res[-1] == math.inf


# ---------------------------------------------------------------- probe corridors (heights)


def probe_case(obstacle_cols, *, obstacle_height=20.0, exact=True):
    """The planner's probe corridor: 5 x 61, heights 30 / 10 m (premium 200), min 5, max 12.

    ``exact=False`` is the probe itself (its bins of 5 m and its almost
    sag-free conductor); ``exact=True`` uses bins of 1 m (exact search) and a
    conductor with a visible sag, so that no clearance decision sits on a
    rounding edge between the float32 kernel and the float64 model."""
    kw = dict(bin_size=1.0, w=8.0, tension=100.0) if exact else dict(bin_size=5.0, n_bins=3, w=0.001,
                                                                     tension=1.0e6)
    return Corridor([100] * 61, rows=5, row=2, min_span=5, max_span=12, heights=(30.0, 10.0),
                    premiums=(200.0, 0.0), obstacles={c: obstacle_height for c in obstacle_cols},
                    min_clearance=5.0, tower_map=((0, 100.0), (500, 100.0)), **kw)


@pytest.mark.parametrize("kernel", KERNELS_H)
def test_reported_heights_are_the_towers_own(kernel):
    case = probe_case(range(29, 32), exact=False)  # plan A4, first probe
    _, towers, heights, dist = case.run(kernel)
    assert dist == 6950.0
    assert towers == [11, 22, 31, 42, 53]
    assert heights == [10.0, 10.0, 30.0, 10.0, 10.0]  # the tower in the obstacle is the tall one

    def all_spans_clear(hs):
        chain = [(0, 10.0)] + list(zip(towers, hs))
        return all(case.clears(a, b, ha, hb) for (a, ha), (b, hb) in zip(chain[:-1], chain[1:]))

    assert all_spans_clear(heights)
    shifted = [10.0] + heights[:-1]  # what the kernels reported before the fix
    assert shifted == [10.0, 10.0, 10.0, 30.0, 10.0]
    assert not all_spans_clear(shifted)


@pytest.mark.parametrize("kernel", KERNELS_H)
def test_reported_heights_pass_clearance_on_random_corridors(kernel):
    # assert_optimal prices the reported towers and heights with the
    # independent model (every span's clearance included); the list the
    # kernels reported before the fix (shifted by one tower, the source
    # class first) fails that check or prices differently.
    old_style_wrong = 0
    for seed in range(6):
        case = random_corridor(seed, heights=(24.0, 12.0), premiums=(150.0, 0.0))
        towers, heights = assert_optimal(case, kernel, {}, True)
        if towers:
            shifted = [12.0] + heights[:-1]
            ref = case.oracle({}, True)
            if case.price(towers, shifted, {}, True) != pytest.approx(ref, rel=REL):
                old_style_wrong += 1
    assert old_style_wrong >= 2


@pytest.mark.parametrize("kernel", KERNELS_H)
@pytest.mark.parametrize("end_height", [10.0, 30.0])
def test_target_attach_height_checks_and_optimises_the_last_span(kernel, end_height):
    # Plan A4, second probe: without an end height the last span is never
    # checked, and the returned one fails with either end height.
    today = probe_case(range(55, 58), exact=False)
    _, towers, heights, _ = today.run(kernel)
    assert towers[-1] == 54 and heights[-1] == 10.0
    assert not today.clears(towers[-1], 60, heights[-1], end_height)
    case = probe_case(range(55, 58))
    towers_e, heights_e = assert_optimal(case, kernel, {"target_attach_height": end_height}, True)
    assert towers_e and case.clears(towers_e[-1], 60, heights_e[-1], end_height, first=1)
    assert case.min_abs_margin > 1e-3


@pytest.mark.parametrize("kernel", KERNELS_H)
@pytest.mark.parametrize("end", ["source", "target"])
def test_low_and_high_connection_heights(kernel, end):
    # Obstacles of 12 m right after the source (or right before the target).
    cols = range(3, 6) if end == "source" else range(55, 58)
    case = probe_case(cols, obstacle_height=12.0)
    key = f"{end}_attach_height"
    t_low, h_low = assert_optimal(case, kernel, {key: 8.0}, True)
    t_high, h_high = assert_optimal(case, kernel, {key: 40.0}, True)
    d_low, d_high = case.oracle({key: 8.0}, True), case.oracle({key: 40.0}, True)
    assert d_high < d_low < math.inf  # a high gantry is cheaper than a low one here
    if end == "source":
        assert h_low[0] == 30.0 and h_high[0] == 10.0  # low gantry: tall first tower
        assert case.clears(0, t_low[0], 8.0, h_low[0], first=0)
        assert case.clears(0, t_high[0], 40.0, h_high[0], first=0)
    else:
        assert case.clears(t_low[-1], 60, h_low[-1], 8.0, first=1)
        assert case.clears(t_high[-1], 60, h_high[-1], 40.0, first=1)
    assert case.min_abs_margin > 1e-3


@pytest.mark.parametrize("kernel", KERNELS_H)
def test_source_premium_is_not_charged_with_a_connection_height(kernel):
    # Every class carries a premium here (the lowest 50): the default search
    # seeds the source with a class and charges it; a connection structure
    # at the source is not a line tower and pays nothing.
    case = Corridor([100] * 30, min_span=3, max_span=8, heights=(24.0, 12.0), premiums=(200.0, 50.0),
                    w=0.001, tension=1.0e6)
    *_, d_default = case.run(kernel)
    *_, d_src = case.run(kernel, source_attach_height=12.0)
    assert d_default - d_src == pytest.approx(50.0, abs=1e-6 * d_default)
    assert d_src == pytest.approx(case.oracle({"source_attach_height": 12.0}, True), rel=rel(kernel))


# ---------------------------------------------------------------- brute-force oracle


def random_corridor(seed, *, heights, premiums, n=28):
    rng = np.random.default_rng(1000 + seed)
    values = rng.integers(1, 200, size=n)
    cols = rng.choice(np.arange(2, n - 1), size=4, replace=False)
    obstacles = {int(c): round(float(rng.uniform(6.0, 18.0)), 1) + 0.03 for c in cols}
    dem = 1.5 * np.sin(np.arange(n) / 3.0 + seed) + 0.05 * np.arange(n)
    return Corridor(values, min_span=3, max_span=8, dem=dem, obstacles=obstacles, heights=heights,
                    premiums=premiums, w=8.0, tension=100.0, min_clearance=5.0)


OPTION_SETS = {
    "none": {},
    "tms": {"target_min_span": 3.0},
    "src": {"source_attach_height": 9.5},
    "tgt": {"target_attach_height": 7.0},
    "both": {"source_attach_height": 30.0, "target_attach_height": 9.5},
    "all": {"target_min_span": 3.0, "source_attach_height": 16.5, "target_attach_height": 30.0},
}
HEIGHT_SETS = {"fixed": ((20.0,), (0.0,)), "variable": ((24.0, 12.0), (150.0, 0.0))}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("opt", ["none", "tms"])
@pytest.mark.parametrize("kernel", KERNELS_2D)
def test_oracle_2d_kernels(kernel, opt, seed):
    case = random_corridor(seed, heights=(12.0,), premiums=(0.0,))
    assert_optimal(case, kernel, OPTION_SETS[opt], False)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("hset", sorted(HEIGHT_SETS))
@pytest.mark.parametrize("opt", sorted(OPTION_SETS))
@pytest.mark.parametrize("kernel", KERNELS_H)
def test_oracle_height_kernels(kernel, opt, hset, seed):
    heights, premiums = HEIGHT_SETS[hset]
    case = random_corridor(seed, heights=heights, premiums=premiums)
    assert_optimal(case, kernel, OPTION_SETS[opt], True)
    # no clearance decision of this case sits on a rounding edge (float32 kernel vs float64 oracle)
    assert case.min_abs_margin > 1e-3


@pytest.mark.parametrize("kernel", ("delta2d",) + KERNELS_H)
def test_end_options_with_several_threads(kernel, monkeypatch):
    # The per-end data are read-only in the parallel phase; the kernels cap
    # the team at the OpenMP limit, so this runs on any machine. Ties may
    # pick other towers, so only the optimum and the validity are compared.
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    with_heights = kernel in KERNELS_H
    opts = OPTION_SETS["all"] if with_heights else OPTION_SETS["tms"]
    for seed in range(3):
        case = random_corridor(seed, heights=(24.0, 12.0), premiums=(150.0, 0.0))
        assert_optimal(case, kernel, opts, with_heights)


def test_oracle_cases_exercise_the_end_options():
    """The option sets change the optimum on these corridors (the tests are not vacuous)."""
    changed = set()
    for seed in range(3):
        for heights, premiums in HEIGHT_SETS.values():
            case = random_corridor(seed, heights=heights, premiums=premiums)
            base = case.oracle({}, True)
            for name, opts in OPTION_SETS.items():
                if name != "none" and case.oracle(opts, True) != base:
                    changed.add(name)
    assert changed >= {"tms", "src", "tgt", "both", "all"}
