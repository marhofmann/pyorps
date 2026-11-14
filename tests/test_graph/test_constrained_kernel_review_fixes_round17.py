"""Review fixes in the constrained CPU kernels (round 17, after package P2).

Four findings of an independent review, each fixed in the Cython kernels and
checked here on every kernel entry (Dijkstra dense and sparse, delta-stepping
2-D, variable-height dense, sparse and lazy):

1. A route that crosses itself can stand a second tower on a cell it passed
   before; the reconstruction dropped that tower (the distance paid for it).
   Every tower is now reported, a cell may repeat in the tower list, and the
   kernel's distance equals the cost recomputed from the route and its towers.
2. The spans at a connection point are sampled in full: a span from a source
   connection height from the source cell on (the source cell and the first
   cell after it included); the last span to a target connection height from
   the first cell after the last tower on, also when it is one step long, and
   the target cell itself must clear (target height - obstacle >= minimum
   clearance). Spans between two line towers keep their sampling.
3. With a line-end option on, span bins in which two span lengths of one
   direction (accumulated in float32 as the search does) share a bin raise
   ValueError instead of returning no path or a wrong route.
4. The three line-end arguments reject bool and str (TypeError), heights and
   lengths above 1e6 m, and a positive target_min_span that is 0 in float32.

The reference is an independent plain-Python search over the exact state
graph (cell, direction, steps in the current span, height class, "no tower
yet"): no span bins and no walk back to find a span start. Spans and the
clearance test use float32 exactly as the kernels do, so decisions at exact
thresholds agree; cases whose clearance margin comes within 1e-3 m of zero
are skipped (counted) rather than compared.
"""

import heapq
import math

import numpy as np
import pytest

from pyorps.utils import _constrained_context as CC
from pyorps.utils import _constrained_delta as DL
from pyorps.utils import _constrained_dijkstra as DJ
from pyorps.utils.neighborhood import get_neighborhood_steps

F = 65535
f32 = np.float32
KERNELS_2D = ("dij_dense", "dij_sparse", "delta2d")
KERNELS_H = ("hdense", "hsparse", "lazy")
ALL_KERNELS = KERNELS_2D + KERNELS_H
HSETS = [((30.0,), (0.0,)), ((30.0, 10.0), (200.0, 0.0)), ((35.0, 20.0, 10.0), (300.0, 100.0, 0.0))]


def rel(kernel):
    # The compact-dense sparse height kernel keeps its distances in float32.
    return 1e-6 if kernel == "hsparse" else 1e-9


def rel_identity(kernel):
    return 1e-5 if kernel == "hsparse" else 1e-9


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # One thread: deterministic tie-breaking, independent of the machine.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------- inputs


def steps_of(name):
    return np.ascontiguousarray(get_neighborhood_steps(name), dtype=np.int8)


def tables(steps, *, cell=1.0, scale=1.0, hard=90.0, tower_map=((0, 40.0), (200, 300.0)),
           angle_types=((10.0, 20.0), (45.0, 60.0), (90.0, 150.0))):
    n = len(steps)
    ang = [math.atan2(float(dc), float(dr)) for dr, dc in steps]
    acl = np.zeros((n, n), np.float64)
    val = np.ones((n, n), bool)
    tac = np.zeros((n, n), np.float64)
    types = sorted(angle_types)
    for i in range(n):
        for j in range(n):
            diff = ang[j] - ang[i]
            turn = abs(math.atan2(math.sin(diff), math.cos(diff)))
            if turn > math.radians(hard):
                val[i, j] = False
                acl[i, j] = np.inf
            else:
                acl[i, j] = scale * math.degrees(turn)
            deg = abs(math.degrees(math.atan2(math.sin(diff), math.cos(diff))))
            for mx, cost in types:
                if deg <= mx:
                    tac[i, j] = cost
                    break
            else:
                tac[i, j] = types[-1][1]
    xs = np.array([p[0] for p in tower_map], float)
    ys = np.array([p[1] for p in tower_map], float)
    sd = np.array([math.hypot(dr, dc) * cell for dr, dc in steps])
    return dict(steps=np.ascontiguousarray(steps, dtype=np.int8),
                angle_cost_lut=acl.astype(np.float32), angle_valid_lut=val.astype(np.uint8),
                step_distances=sd.astype(np.float32),
                tower_terrain_costs=np.interp(np.arange(65536), xs, ys).astype(np.float32),
                tower_angle_costs=tac.astype(np.float32))


def bins_exact(steps, cell, min_span, max_span):
    """The finder's exact bins: a bin below the shortest step, with margin."""
    shortest = float(np.float32(np.min([math.hypot(dr, dc) * cell for dr, dc in steps])))
    k = max(1, math.ceil(max_span / shortest))
    return dict(n_span_bins=k + 1, span_bin_size=float(np.float32(shortest * k / (k + 0.25))),
                min_span=float(min_span), max_span=float(max_span))


def bins_coarse(min_span, max_span):
    """Bins of min_span (the finder's default runs)."""
    return dict(n_span_bins=max(2, math.ceil(max_span / min_span) + 1), span_bin_size=float(min_span),
                min_span=float(min_span), max_span=float(max_span))


def intermediate_offsets(dr, dc):
    """Intermediate cells of one step (transcription of the kernels' rule)."""
    adr, adc = abs(dr), abs(dc)
    if adr + adc <= 1:
        return []
    if max(adr, adc) == 1:
        return [(dr, 0), (0, dc)]
    k = max(adr, adc)
    out = []
    for p in range(1, k):
        rk, ck = p * dr / k, p * dc / k
        fr, fc = int(math.floor(rk)), int(math.floor(ck))
        out.append((fr, fc))
        cr, cc = int(math.ceil(rk)), int(math.ceil(ck))
        if (fr, fc) != (cr, cc):
            out.append((cr, cc))
    return out


class Grid:
    """A raster problem: kernels and oracle read the same inputs."""

    def __init__(self, raster, src, tgt, steps, *, min_span, max_span, cell=1.0, heights=(30.0,),
                 premiums=(0.0,), obs=None, w=50.0, tension=500.0, mc=3.0, bins=None, tb=None):
        self.raster = np.ascontiguousarray(raster, dtype=np.uint16)
        self.rows, self.cols = self.raster.shape
        self.src, self.tgt = tuple(src), tuple(tgt)
        self.steps = np.ascontiguousarray(steps, dtype=np.int8)
        self.min_span, self.max_span, self.cell = float(min_span), float(max_span), float(cell)
        self.heights = np.asarray(heights, np.float32)
        self.premiums = np.asarray(premiums, np.float32)
        self.obs = np.zeros(self.raster.shape, np.float32) if obs is None else np.asarray(obs, np.float32)
        self.w, self.tension, self.mc = float(w), float(tension), float(mc)
        self.tb = tb if tb is not None else tables(self.steps, cell=cell)
        self.bins = bins if bins is not None else bins_exact(self.steps, cell, min_span, max_span)

    def base(self):
        return dict(raster=self.raster, source_row=self.src[0], source_col=self.src[1],
                    target_row=self.tgt[0], target_col=self.tgt[1], **self.tb, **self.bins,
                    cell_size=self.cell)

    def hkw(self):
        return dict(dem_data=np.zeros(self.raster.shape, np.float32), tower_heights=self.heights,
                    height_premiums=self.premiums, conductor_weight_per_m=self.w,
                    conductor_tension=self.tension, min_clearance_val=self.mc, obstacle_heights=self.obs,
                    max_gradient_pct=100.0, gradient_scale=2.0)

    def run(self, kernel, **opts):
        """(path cells, tower cells, tower heights or None, dist) of one kernel entry."""
        base = self.base()
        if kernel == "dij_dense":
            res = DJ.constrained_dijkstra_2d(**base, return_dist=1, **opts)
        elif kernel == "dij_sparse":
            res = DJ.constrained_dijkstra_2d(**base, force_sparse=1, return_dist=1, **opts)
        elif kernel == "delta2d":
            res = DL.constrained_delta_stepping_2d(**base, return_dist=1, **opts)
        elif kernel == "hdense":
            res = DL.constrained_delta_stepping_height_2d(**base, **self.hkw(), return_dist=1, **opts)
        elif kernel == "hsparse":
            res = DL.constrained_delta_stepping_height_2d(**base, **self.hkw(), force_sparse=1,
                                                          return_dist=1, **opts)
        elif kernel == "lazy":
            res = DL.constrained_delta_stepping_lazy(**base, **self.hkw(), return_dist=1, **opts)
        else:  # pragma: no cover
            raise AssertionError(kernel)
        path = [(int(p) // self.cols, int(p) % self.cols) for p in res[0]]
        towers = [(int(t) // self.cols, int(t) % self.cols) for t in res[1]]
        heights = [float(h) for h in res[2]] if len(res) == 4 else None
        return path, towers, heights, float(res[-1])


# ---------------------------------------------------------------- independent oracle


class Oracle:
    """Plain-Python Dijkstra over the exact constrained state graph (flat DEM)."""

    def __init__(self, g, *, with_heights, end_sampling="full"):
        self.g = g
        self.steps = [(int(a), int(b)) for a, b in g.steps]
        self.nd = len(self.steps)
        self.inter = [intermediate_offsets(dr, dc) for dr, dc in self.steps]
        self.cf = [float(f32(math.sqrt(dr * dr + dc * dc) / (2.0 + len(self.inter[k]))))
                   for k, (dr, dc) in enumerate(self.steps)]
        self.sd32 = [f32(x) for x in g.tb["step_distances"]]
        self.acl = g.tb["angle_cost_lut"]
        self.valid = g.tb["angle_valid_lut"]
        self.tac = g.tb["tower_angle_costs"]
        self.ttc = g.tb["tower_terrain_costs"]
        self.min32, self.max32 = f32(g.min_span), f32(g.max_span)
        self.cell = float(f32(g.cell))
        self.heights = [f32(h) for h in g.heights] if with_heights else None
        self.prem = [float(f32(p)) for p in g.premiums]
        self.w, self.T, self.mc = f32(g.w), f32(g.tension), f32(g.mc)
        self.full_ends = end_sampling == "full"
        self.sensitive = 0
        # span lengths per direction, accumulated as the kernels do
        self.span = []
        for d in range(self.nd):
            arr = [f32(0.0)]
            while True:
                new = float(arr[-1]) + float(self.sd32[d])
                if not new < float(self.max32):
                    break
                arr.append(f32(new))
            self.span.append(arr)
        self._clear_cache = {}
        self._step_cache = {}

    # -- clearance (float32 as in the kernels)

    def clear(self, br, bc, n, d, ha, hb, first):
        key = (br, bc, n, d, float(ha), float(hb), first)
        if key not in self._clear_cache:
            self._clear_cache[key] = self._clear(br, bc, n, d, f32(ha), f32(hb), first)
        return self._clear_cache[key]

    def _sample(self, x, r, c, attach_a, attach_b, span_len):
        chord = f32(attach_a + f32(f32(f32(attach_b - attach_a) * x) / span_len))
        sag = f32(float(f32(f32(self.w * x) * f32(span_len - x))) / (2.0 * float(self.T)))
        cond = f32(chord - sag)
        clr = f32(f32(cond - f32(0.0)) - f32(self.g.obs[r, c]))
        if abs(float(clr) - float(self.mc)) < 1e-3:
            self.sensitive += 1
        return not clr < self.mc

    def _clear(self, br, bc, n, d, ha, hb, first):
        """first: 2 = spans between line towers, 1 = full from the first cell, 0 = also the start cell."""
        dr, dc = self.steps[d]
        ar, ac = br - n * dr, bc - n * dc
        if not (0 <= ar < self.g.rows and 0 <= ac < self.g.cols):
            return False
        s = self.sd32[d]
        attach_a, attach_b = ha, hb
        if first >= 2:
            if n <= 1:
                return True
            span_len = f32(f32(n) * s)
            max_sag = f32(float(f32(f32(self.w * span_len) * span_len)) / (8.0 * float(self.T)))
            min_h = ha if ha < hb else hb
            if float(f32(f32(min_h - max_sag) - self.mc)) > 50.0:
                return True
        else:
            if n <= 0:
                ob = f32(self.g.obs[br, bc])
                if first == 0 and f32(f32(attach_a - f32(0.0)) - ob) < self.mc:
                    return False
                return not f32(f32(attach_b - f32(0.0)) - ob) < self.mc
            span_len = f32(f32(n) * s)
        ok = True
        if first == 0:
            ok &= self._sample(f32(0.0), ar, ac, attach_a, attach_b, span_len)
        ni = len(self.inter[d])
        for step in range(n):
            wr, wc = ar + step * dr, ac + step * dc
            for i, (ir, ic) in enumerate(self.inter[d]):
                rr, cc = wr + ir, wc + ic
                if not (0 <= rr < self.g.rows and 0 <= cc < self.g.cols):
                    return False
                x = f32(f32(f32(step) * s) + f32(s * f32(f32(i + 1) / f32(ni + 1))))
                ok &= self._sample(x, rr, cc, attach_a, attach_b, span_len)
            if step + 1 >= first:
                ok &= self._sample(f32(f32(step + 1) * s), wr + dr, wc + dc, attach_a, attach_b, span_len)
        return ok

    # -- graph

    def step(self, r, c, d):
        """(next cell, terrain cost) of one step or None when blocked."""
        key = (r, c, d)
        if key not in self._step_cache:
            self._step_cache[key] = self._step(r, c, d)
        return self._step_cache[key]

    def _step(self, r, c, d):
        g = self.g
        dr, dc = self.steps[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < g.rows and 0 <= nc < g.cols) or g.raster[nr, nc] == F:
            return None
        icost = f32(0.0)
        for ir, ic in self.inter[d]:
            rr, cc = r + ir, c + ic
            if not (0 <= rr < g.rows and 0 <= cc < g.cols) or g.raster[rr, cc] == F:
                return None
            icost = f32(icost + f32(g.raster[rr, cc]))
        terrain = (float(g.raster[r, c]) + float(icost) + float(g.raster[nr, nc])) * self.cf[d] * self.cell
        return (nr, nc), terrain

    def end_ok(self, r, c, d, n, h, notower, tmin, src_h, tgt_h):
        span = self.span[d][n]
        if tmin > 0.0 and not notower and float(span) < float(f32(tmin)):
            return False
        if self.heights is not None and tgt_h is not None:
            from_src = src_h is not None and notower
            ha = f32(src_h) if from_src else self.heights[h]
            if self.full_ends:
                if f32(f32(tgt_h) - f32(self.g.obs[r, c])) < self.mc:
                    return False
                first = 0 if from_src else 1
            else:
                first = 2
            return self.clear(r, c, n, d, ha, f32(tgt_h), first)
        return True

    def solve(self, tmin=0.0, src_h=None, tgt_h=None):
        """Optimal cost under the options (inf when there is no line)."""
        g = self.g
        use_h = self.heights is not None
        if not use_h:
            src_h = tgt_h = None
        nh = len(self.heights) if use_h else 1
        no_src_tower = tmin > 0.0 or src_h is not None
        heap, best, done, cnt = [], {}, set(), 0
        for d in range(self.nd):
            if use_h and src_h is not None:
                seeds = [(nh - 1, 0.0)]
            elif use_h:
                seeds = [(h, self.prem[h]) for h in range(nh)]
            else:
                seeds = [(0, 0.0)]
            for h, c0 in seeds:
                st = (g.src[0], g.src[1], d, 0, h, True)
                best[st] = c0
                heapq.heappush(heap, (c0, cnt, st))
                cnt += 1
        while heap:
            cost, _, st = heapq.heappop(heap)
            if st in done or cost > best[st]:
                continue
            done.add(st)
            r, c, d, n, h, notower = st
            if (r, c) == g.tgt:
                if self.end_ok(r, c, d, n, h, notower, tmin, src_h, tgt_h):
                    return cost
                continue
            span = self.span[d][n]
            can_tower = float(span) >= float(self.min32) and not (no_src_tower and (r, c) == g.src)
            from_src = src_h is not None and notower
            for d2 in range(self.nd):
                if not self.valid[d][d2]:
                    continue
                nxt = self.step(r, c, d2)
                if nxt is None:
                    continue
                (nr, nc), terrain = nxt
                edge = terrain + float(self.acl[d][d2])
                if d2 == d and n + 1 < len(self.span[d]):
                    cnt = self._push(heap, best, cost + edge, (nr, nc, d, n + 1, h, notower), cnt)
                if not can_tower:
                    continue
                tt = float(self.ttc[int(g.raster[r, c])]) + float(self.tac[d][d2])
                if not use_h:
                    cnt = self._push(heap, best, cost + edge + tt, (nr, nc, d2, 1, 0, False), cnt)
                    continue
                ha = f32(src_h) if from_src else self.heights[h]
                first = 0 if (from_src and self.full_ends) else 2
                for hb in range(nh):
                    if self.clear(r, c, n, d, ha, self.heights[hb], first):
                        cnt = self._push(heap, best, cost + edge + (tt + self.prem[hb]),
                                         (nr, nc, d2, 1, hb, False), cnt)
        return math.inf

    @staticmethod
    def _push(heap, best, cost, st, cnt):
        if cost < best.get(st, math.inf):
            best[st] = cost
            heapq.heappush(heap, (cost, cnt, st))
        return cnt + 1


# ---------------------------------------------------------------- route check (duplicates allowed)


def tower_assignments(cells, dirs, towers, limit=400):
    """Every increasing assignment of the reported towers to path positions that puts a
    tower at each direction change (a cell can be passed several times)."""
    changes = {i for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1]}
    out = []

    def rec(k, start, acc):
        if len(out) >= limit:
            return
        if k == len(towers):
            if changes <= set(acc):
                out.append(list(acc))
            return
        for i in range(start, len(cells) - 1):
            if i in changes and cells[i] != towers[k]:
                return  # a direction change before the next tower stays uncovered
            if cells[i] == towers[k]:
                acc.append(i)
                rec(k + 1, i + 1, acc)
                acc.pop()
                if i in changes:
                    return

    rec(0, 1, [])
    return out


def check_route(o, path, towers, heights, dist, kernel, tmin=0.0, src_h=None, tgt_h=None):
    """Validate a kernel route independently; returns the recomputed cost (asserts on problems)."""
    g = o.g
    use_h = o.heights is not None
    assert path[0] == g.src and path[-1] == g.tgt
    dirs = [o.steps.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path[:-1], path[1:])]
    for i, (a, d) in enumerate(zip(path[:-1], dirs)):
        assert o.step(a[0], a[1], d) is not None
        if i > 0:
            assert o.valid[dirs[i - 1]][d]
    classes = None
    if use_h:
        assert heights is not None and len(heights) == len(towers)
        hv = [float(h) for h in o.heights]
        classes = [hv.index(float(f32(h))) for h in heights]
    problems = []
    for idx in tower_assignments(path, dirs, towers):
        src_classes = [None] if (not use_h or src_h is not None) else range(len(o.heights))
        for sc in src_classes:
            probs, cost = _route_problems(o, path, dirs, idx, classes, sc, tmin, src_h, tgt_h)
            if not probs and cost == pytest.approx(dist, rel=rel_identity(kernel)):
                return cost
            problems.append((idx, sc, probs, cost))
    raise AssertionError(f"{kernel}: no valid reading of the route (dist {dist}): {problems[:3]}")


def _route_problems(o, cells, dirs, idx, classes, src_class, tmin, src_h, tgt_h):
    g = o.g
    use_h = o.heights is not None
    no_src_tower = tmin > 0.0 or src_h is not None
    bounds = [0] + idx + [len(cells) - 1]
    probs = []
    last = len(bounds) - 2
    for k in range(last + 1):
        a, b = bounds[k], bounds[k + 1]
        n = b - a
        if len(set(dirs[a:b])) != 1:
            return [f"span {k} is not straight"], None
        d = dirs[a]
        if n >= len(o.span[d]):
            return [f"span {k} reaches max_span"], None
        span = o.span[d][n]
        if k < last:
            if float(span) < float(o.min32):
                probs.append(f"span {k} below min_span")
            if no_src_tower and cells[b] == g.src:
                probs.append("tower on the source cell")
        elif k > 0 and tmin > 0.0 and float(span) < float(f32(tmin)):
            probs.append("last span below target_min_span")
        if use_h:
            from_src = k == 0 and src_h is not None
            ha = f32(src_h) if from_src else o.heights[src_class if k == 0 else classes[k - 1]]
            if k < last:
                if not o.clear(cells[b][0], cells[b][1], n, d, ha, o.heights[classes[k]], 0 if from_src else 2):
                    probs.append(f"span {k} fails clearance")
            elif tgt_h is not None:
                if f32(f32(tgt_h) - f32(g.obs[g.tgt])) < o.mc:
                    probs.append("target cell fails clearance")
                if not o.clear(g.tgt[0], g.tgt[1], n, d, ha, f32(tgt_h), 0 if from_src else 1):
                    probs.append("last span fails clearance")
    cost = 0.0 if (not use_h or src_h is not None) else o.prem[src_class]
    tower_at = {i: k for k, i in enumerate(idx)}
    for i, d in enumerate(dirs):
        d_in = dirs[i - 1] if i > 0 else d
        _, terrain = o.step(cells[i][0], cells[i][1], d)
        cost += terrain + float(o.acl[d_in][d])
        if i in tower_at:
            cost += float(o.ttc[int(g.raster[cells[i]])]) + float(o.tac[d_in][d])
            if use_h:
                cost += o.prem[classes[tower_at[i]]]
    return probs, cost


# ---------------------------------------------------------------- the reviewer's random cases


def reviewer_case(i, seed0=1000):
    """The review's opts_probe case generator (same draws, same order)."""
    steps = steps_of("r1")
    rng = np.random.default_rng(seed0 + i)
    rows, cols = int(rng.integers(4, 8)), int(rng.integers(8, 15))
    raster = rng.integers(10, 60, size=(rows, cols)).astype(np.uint16)
    raster[rng.random((rows, cols)) < 0.1] = F
    cells = [(r, c) for r in range(rows) for c in range(cols) if raster[r, c] != F]
    while True:
        src = cells[int(rng.integers(0, len(cells)))]
        tgt = cells[int(rng.integers(0, len(cells)))]
        if abs(src[1] - tgt[1]) + abs(src[0] - tgt[0]) >= 3:
            break
    min_span = float(rng.choice([2.5, 3.0]))
    max_span = float(rng.choice([5.5, 6.5, 7.5]))
    tmin = float(rng.choice([0.0, 0.0, min_span, 3.5, 4.5, 5.2]))
    hs, pr = HSETS[int(rng.integers(0, len(HSETS)))]
    obs = np.zeros((rows, cols), np.float32)
    obs[rng.random((rows, cols)) < 0.15] = float(rng.choice([6.0, 12.0, 18.0, 25.0]))
    src_h = [None, None, 5.0, 12.0, 30.0][int(rng.integers(0, 5))]
    tgt_h = [None, None, 5.0, 12.0, 30.0][int(rng.integers(0, 5))]
    g = Grid(raster, src, tgt, steps, min_span=min_span, max_span=max_span, heights=hs, premiums=pr,
             obs=obs)
    return g, tmin, src_h, tgt_h


def options(kernel, tmin, src_h, tgt_h):
    opts = {"target_min_span": tmin} if tmin > 0 else {}
    if kernel in KERNELS_H:
        if src_h is not None:
            opts["source_attach_height"] = src_h
        if tgt_h is not None:
            opts["target_attach_height"] = tgt_h
    return opts


# ---------------------------------------------------------------- fix 1: towers at a self-crossing


def test_self_crossing_route_reports_every_tower():
    # The review's case: 7 x 12 grid, source (2, 10), target (4, 3), target_min_span 5.2;
    # the route turns 8 times and the kernels reported 7 towers (the distance paid 8).
    g, tmin, src_h, tgt_h = reviewer_case(18)
    assert g.raster.shape == (7, 12) and g.src == (2, 10) and g.tgt == (4, 3) and tmin == 5.2
    for kernel in ALL_KERNELS:
        with_h = kernel in KERNELS_H
        o = Oracle(g, with_heights=with_h)
        path, towers, heights, dist = g.run(kernel, target_min_span=tmin)
        assert dist < math.inf
        assert dist == pytest.approx(o.solve(tmin), rel=rel(kernel))
        cost = check_route(o, path, towers, heights, dist, kernel, tmin)
        assert cost == pytest.approx(dist, rel=rel_identity(kernel))  # the cost identity
        dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])]
        turns = sum(1 for x, y in zip(dirs[:-1], dirs[1:]) if x != y)
        assert turns == 8 and len(towers) == 8 and len(set(towers)) == 7, (kernel, towers)
        if heights is not None:
            assert len(heights) == len(towers)
        if with_h:
            # with the case's connection heights (12 m, 5 m) there is no line at all
            assert g.run(kernel, **options(kernel, tmin, src_h, tgt_h))[-1] == o.solve(tmin, src_h, tgt_h)


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_duplicate_tower_cells_keep_heights_aligned(kernel):
    # Every self-crossing case of the review's first 60 seeds: the tower list, the height
    # list and the distance agree (one tower per state transition, each with its own height).
    seen = repeated = 0
    with_h = kernel in KERNELS_H
    for i in range(60):
        g, tmin, src_h, tgt_h = reviewer_case(i)
        variants = [(src_h, tgt_h), (None, None)] if with_h else [(None, None)]
        for sh, th in variants:
            path, towers, heights, dist = g.run(kernel, **options(kernel, tmin, sh, th))
            if dist == math.inf or len(set(path)) == len(path):
                continue
            o = Oracle(g, with_heights=with_h)
            assert dist == pytest.approx(o.solve(tmin, sh, th), rel=rel(kernel)), i
            check_route(o, path, towers, heights, dist, kernel, tmin, sh, th)
            seen += 1
            repeated += len(set(towers)) < len(towers)
    assert seen >= 3 and repeated >= 1, (seen, repeated)


# ---------------------------------------------------------------- fix 2: clearance at the line ends


def corridor(rows, cols, row, value=30):
    r = np.full((rows, cols), F, np.uint16)
    r[row, :] = value
    return r


def corridor_grid(obstacles, **kw):
    obs = np.zeros((3, 30), np.float32)
    for (r, c), h in obstacles.items():
        obs[r, c] = h
    return Grid(corridor(3, 30, 1), (1, 0), (1, 29), steps_of("r1"), min_span=5.0, max_span=12.0,
                obs=obs, **kw)


@pytest.mark.parametrize("col", [1, 2, 3])
def test_low_source_height_next_to_an_obstacle(col):
    # Source connection 3 m, towers 30 m, clearance 3 m, an 8 m obstacle right after the
    # source: at column 1 no first span clears (the kernels used to skip that cell).
    g = corridor_grid({(1, col): 8.0})
    o = Oracle(g, with_heights=True)
    want = o.solve(src_h=3.0)  # the source cell clears by exactly 0 m (3 m - 0 m = 3 m): kept
    if col == 1:
        assert want == math.inf
        assert Oracle(g, with_heights=True, end_sampling="legacy").solve(src_h=3.0) < math.inf
    for kernel in KERNELS_H:
        path, towers, heights, dist = g.run(kernel, source_attach_height=3.0)
        assert dist == pytest.approx(want, rel=rel(kernel))
        if want == math.inf:
            assert path == [] and towers == []
        else:
            check_route(o, path, towers, heights, dist, kernel, src_h=3.0)


@pytest.mark.parametrize("kernel", KERNELS_H)
def test_obstacle_on_the_end_cells_is_checked_at_both_ends(kernel):
    # A 25 m obstacle on the end cell with a 5 m connection height: no line, at the
    # source end as at the target end (the source cell used to be skipped).
    for where, opts in (((1, 0), {"source_attach_height": 5.0}),
                        ((1, 29), {"target_attach_height": 5.0, "target_min_span": 3.0}),
                        ((1, 29), {"target_attach_height": 5.0})):
        g = corridor_grid({where: 25.0})
        o = Oracle(g, with_heights=True)
        assert o.solve(opts.get("target_min_span", 0.0), opts.get("source_attach_height"),
                       opts.get("target_attach_height")) == math.inf
        path, towers, heights, dist = g.run(kernel, **opts)
        assert dist == math.inf and path == [] and towers == []


@pytest.mark.parametrize("kernel", KERNELS_H)
def test_one_step_last_span_is_checked(kernel):
    # No end rule, target connection 5 m, 25 m obstacle on the target cell: the kernels
    # used to accept a last span of one step without any check.
    g = corridor_grid({(1, 29): 25.0})
    assert Oracle(g, with_heights=True, end_sampling="legacy").solve(tgt_h=5.0) < math.inf
    *_, dist = g.run(kernel, target_attach_height=5.0)
    assert dist == math.inf


@pytest.mark.parametrize("kernel", KERNELS_H)
def test_first_cell_before_the_target_is_checked(kernel):
    # A 9 m obstacle one cell before the target, target connection 5 m: the last span may
    # not pass over it low (the first cell after the last tower was never sampled; a
    # tower right before it now has to be tall enough or stand elsewhere).
    g = corridor_grid({(1, 28): 9.0}, heights=(30.0, 10.0), premiums=(200.0, 0.0))
    for opts in ({"target_attach_height": 5.0}, {"target_attach_height": 5.0, "target_min_span": 3.0}):
        o = Oracle(g, with_heights=True)
        want = o.solve(opts.get("target_min_span", 0.0), None, 5.0)
        assert o.sensitive == 0
        path, towers, heights, dist = g.run(kernel, **opts)
        assert dist == pytest.approx(want, rel=rel(kernel))
        if want < math.inf:
            check_route(o, path, towers, heights, dist, kernel, opts.get("target_min_span", 0.0), None, 5.0)


# ---------------------------------------------------------------- fix 3: span bins


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_coarse_bins_with_target_min_span_raise(kernel):
    # 1 x 24 corridor, min 3, max 8, bins of min_span: the spans of 1 and 2 steps share
    # bin 0; with the end rule on this used to give no path silently.
    coarse = Grid(corridor(3, 24, 1, 100), (1, 0), (1, 23), steps_of("r1"), min_span=3.0, max_span=8.0,
                  heights=(10.0,), mc=0.001, w=1.0, tension=1e6, bins=bins_coarse(3.0, 8.0))
    with pytest.raises(ValueError, match=r"span_bin_size 3 m .*\(step [0-9.]+ m\) the spans of 1 and 2 steps"):
        coarse.run(kernel, target_min_span=3.0)
    *_, dist = coarse.run(kernel)  # options off: nothing is checked, default runs are unchanged
    assert dist < math.inf
    exact = Grid(coarse.raster, coarse.src, coarse.tgt, coarse.steps, min_span=3.0, max_span=8.0,
                 heights=(10.0,), mc=0.001, w=1.0, tension=1e6)
    path, towers, heights, dist = exact.run(kernel, target_min_span=3.0)
    o = Oracle(exact, with_heights=kernel in KERNELS_H)
    assert dist < math.inf
    assert dist == pytest.approx(o.solve(3.0), rel=rel(kernel))


@pytest.mark.parametrize("kernel", KERNELS_H)
@pytest.mark.parametrize("opt", [{"source_attach_height": 12.0}, {"target_attach_height": 12.0}])
def test_coarse_bins_with_a_connection_height_raise(kernel, opt):
    g = Grid(corridor(3, 24, 1, 100), (1, 0), (1, 23), steps_of("r1"), min_span=3.0, max_span=8.0,
             heights=(10.0,), bins=bins_coarse(3.0, 8.0))
    with pytest.raises(ValueError, match="too coarse for the line-end options"):
        g.run(kernel, **opt)


@pytest.mark.parametrize("tmin", [1.5, 2.0, 3.1])
def test_float32_accumulation_collision_raises(tmin):
    # Cell 0.3 m, bin 0.3 m: float32 accumulation puts some spans one bin low, so two
    # spans share a bin (no path for target_min_span 2.0 and 3.1 before the check).
    steps = steps_of("r1")
    cell = 0.3
    s32 = float(np.float32(cell))
    bins = dict(n_span_bins=math.ceil(4.0 / s32) + 2, span_bin_size=s32, min_span=1.5, max_span=4.0)
    g = Grid(corridor(3, 70, 1), (1, 0), (1, 69), steps, min_span=1.5, max_span=4.0, cell=cell, bins=bins)
    for kernel in ALL_KERNELS:
        with pytest.raises(ValueError, match=r"span_bin_size 0\.300000012 m .* share span bin"):
            g.run(kernel, target_min_span=tmin)
    exact = Grid(g.raster, g.src, g.tgt, steps, min_span=1.5, max_span=4.0, cell=cell)
    want = Oracle(exact, with_heights=False).solve(tmin)
    for kernel in KERNELS_2D:
        *_, dist = exact.run(kernel, target_min_span=tmin)
        assert dist < math.inf
        assert dist == pytest.approx(want, rel=rel(kernel))


@pytest.mark.parametrize("nbh", ["r1", "r2", "r3"])
@pytest.mark.parametrize("cell", [1.0, 0.3, 2.5, 0.1])
@pytest.mark.parametrize("max_span", [8.0, 50.0, 420.0])
def test_finder_exact_bins_pass_the_check(nbh, cell, max_span):
    steps = steps_of(nbh)
    sd = tables(steps, cell=cell)["step_distances"]
    b = bins_exact(steps, cell, 1.0, max_span)
    CC.check_exact_span_bins(sd, b["span_bin_size"], max_span, b["n_span_bins"])
    with pytest.raises(ValueError, match="share span bin"):  # one bin for every span below max_span
        CC.check_exact_span_bins(sd, max_span, max_span, 2)


def test_bins_that_do_not_cover_the_spans_raise():
    sd = tables(steps_of("r1"))["step_distances"]
    with pytest.raises(ValueError, match="n_span_bins"):
        CC.check_exact_span_bins(sd, 0.9, 8.0, 5)


# ---------------------------------------------------------------- fix 4: argument checks


def small_grid():
    return Grid(corridor(3, 30, 1), (1, 0), (1, 29), steps_of("r1"), min_span=5.0, max_span=12.0)


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), "3", b"3", None, [3.0]])
def test_target_min_span_rejects_non_numbers(bad):
    g = small_grid()
    for kernel in ALL_KERNELS:
        with pytest.raises(TypeError):
            g.run(kernel, target_min_span=bad)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan"), 1.0e7, 3.0e38, 1.0e39, 1.0e-50])
def test_target_min_span_rejects_bad_values(bad):
    g = small_grid()
    for kernel in ALL_KERNELS:
        with pytest.raises(ValueError, match="target_min_span"):
            g.run(kernel, target_min_span=bad)


@pytest.mark.parametrize("name", ["source_attach_height", "target_attach_height"])
@pytest.mark.parametrize("bad", [True, np.bool_(False), "7", b"7"])
def test_attach_heights_reject_non_numbers(name, bad):
    g = small_grid()
    for kernel in KERNELS_H:
        with pytest.raises(TypeError, match=name):
            g.run(kernel, **{name: bad})


@pytest.mark.parametrize("name", ["source_attach_height", "target_attach_height"])
@pytest.mark.parametrize("bad", [0.0, -2.0, float("inf"), 1.0000001e6, 3.0e38, 1.0e39])
def test_attach_heights_reject_bad_values(name, bad):
    g = small_grid()
    for kernel in KERNELS_H:
        with pytest.raises(ValueError, match=name):
            g.run(kernel, **{name: bad})


def test_numeric_spellings_are_accepted_alike():
    g = small_grid()
    for kernel in ALL_KERNELS:
        ref = g.run(kernel, target_min_span=7.0)
        for v in (7, np.float32(7.0), np.float64(7.0), np.array(7.0), np.int64(7)):
            assert g.run(kernel, target_min_span=v) == ref
    for kernel in KERNELS_H:
        ref = g.run(kernel, source_attach_height=12.0, target_attach_height=1.0e6)
        assert ref[-1] < math.inf
        assert g.run(kernel, source_attach_height=np.float32(12.0), target_attach_height=1.0e6) == ref
        assert g.run(kernel, source_attach_height=None, target_attach_height=float("nan")) == g.run(kernel)


# ---------------------------------------------------------------- random grids vs the oracle


def random_case(seed, nbh):
    rng = np.random.default_rng(seed)
    steps = steps_of(nbh)
    if nbh == "r1":
        rows, cols = int(rng.integers(4, 8)), int(rng.integers(8, 14))
    else:
        rows, cols = int(rng.integers(4, 7)), int(rng.integers(7, 11))
    raster = rng.integers(10, 60, size=(rows, cols)).astype(np.uint16)
    raster[rng.random((rows, cols)) < 0.1] = F
    cells = [(r, c) for r in range(rows) for c in range(cols) if raster[r, c] != F]
    while True:
        src = cells[int(rng.integers(0, len(cells)))]
        tgt = cells[int(rng.integers(0, len(cells)))]
        if abs(src[1] - tgt[1]) + abs(src[0] - tgt[0]) >= 3:
            break
    min_span = float(rng.choice([2.5, 3.0]))
    max_span = float(rng.choice([5.5, 6.5, 7.5]))
    tmin = float(rng.choice([0.0, min_span, 3.5, 4.5, 5.2]))
    hs, pr = HSETS[int(rng.integers(0, len(HSETS)))]
    obs = np.zeros((rows, cols), np.float32)
    obs[rng.random((rows, cols)) < 0.15] = float(rng.choice([6.0, 9.0, 12.0, 18.0, 25.0]))
    src_h = [None, 4.0, 7.0, 12.0, 30.0][int(rng.integers(0, 5))]
    tgt_h = [None, 4.0, 7.0, 12.0, 30.0][int(rng.integers(0, 5))]
    g = Grid(raster, src, tgt, steps, min_span=min_span, max_span=max_span, heights=hs, premiums=pr, obs=obs)
    return g, tmin, src_h, tgt_h


N_RANDOM = {"r1": 40, "r2": 16}


@pytest.mark.parametrize("nbh", sorted(N_RANDOM))
def test_random_grids_against_the_oracle(nbh):
    """Turns, diagonals (r1) and knight steps (r2), loops, every option, every kernel."""
    stats = dict(compared=0, sensitive=0, feasible=0, infeasible=0, crossing=0, repeated_tower=0,
                 turns=0, diagonal=0, ends_matter=0, options_matter=0)
    for i in range(N_RANDOM[nbh]):
        g, tmin, src_h, tgt_h = random_case(7000 + 97 * i, nbh)
        for with_h in (False, True):
            o = Oracle(g, with_heights=with_h)
            sh, th = (src_h, tgt_h) if with_h else (None, None)
            want = o.solve(tmin, sh, th)
            if o.sensitive:
                stats["sensitive"] += 1
                continue
            stats["compared"] += 1
            stats["feasible" if want < math.inf else "infeasible"] += 1
            if with_h and (sh is not None or th is not None):
                legacy = Oracle(g, with_heights=True, end_sampling="legacy")
                stats["ends_matter"] += legacy.solve(tmin, sh, th) != want
            if tmin > 0 or sh is not None or th is not None:
                stats["options_matter"] += Oracle(g, with_heights=with_h).solve() != want
            for kernel in (KERNELS_H if with_h else KERNELS_2D):
                path, towers, heights, dist = g.run(kernel, **options(kernel, tmin, src_h, tgt_h))
                assert dist == pytest.approx(want, rel=rel(kernel)), (i, kernel, tmin, sh, th)
                if want == math.inf:
                    assert path == [] and towers == []
                    continue
                check_route(o, path, towers, heights, dist, kernel, tmin, sh, th)
                steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])]
                stats["crossing"] += len(set(path)) < len(path)
                stats["repeated_tower"] += len(set(towers)) < len(towers)
                stats["turns"] += any(x != y for x, y in zip(steps[:-1], steps[1:]))
                stats["diagonal"] += any(abs(dr) + abs(dc) > 1 for dr, dc in steps)
    # the comparisons are not vacuous
    assert stats["compared"] >= 0.8 * 2 * N_RANDOM[nbh], stats
    assert stats["feasible"] >= 4 and stats["turns"] >= 4 and stats["diagonal"] >= 4, stats
    assert stats["options_matter"] >= 2 and stats["infeasible"] >= 1, stats
    if nbh == "r1":
        # the full end-span sampling decides some of these cases (fix 2); routes that
        # cross themselves are covered by the review's cases above
        assert stats["ends_matter"] >= 1, stats
