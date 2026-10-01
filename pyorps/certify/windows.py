"""Window certificates (plan rev. 5, section 3.5, Phase D2).

Fields are computed on WINDOWS, never on the whole 1 G-cell raster, and a
windowed field is only a lower bound until a certificate says where it is
exact. With ``W`` a cell set, ``G_W`` the steps whose endpoints and
intermediate cells all lie in ``W``, and ``dW`` the cells of ``W`` from which
some step leaves ``G_W``:

**Path, single source** ``a``. Any walk ``a -> s`` that uses a step outside
``G_W`` first reaches ``dW`` (cost ``>= beta_a = min_{x in dW} d_W(a, x)``)
and last re-enters through ``dW`` (cost ``>= b(s) = d_W(dW, s)``). So

    ``LB(s) = min(d_W(a, s), beta_a + b(s))``

is a global lower bound, and ``d_W(a, s)`` is EXACT wherever
``d_W(a, s) <= beta_a + b(s)``. For ``s`` outside ``W``,
``d(a, s) >= beta_a + c_min * dist(s, W)``.

**Path, seeded** (seeds ``Q`` with labels ``V``). With
``beta_out = min_{q in Q \\ W} V(q) + c_min * dist(q, W)`` and
``beta_in = min_{x in dW}`` of the in-window field, ``d_W`` is exact where
``d_W(s) <= min(beta_in, beta_out) + b(s)``, and
``min(d_W, min(beta_in, beta_out) + b)`` bounds it everywhere.

``c_min`` is the drain's smallest cost per cell of length:
``weight_mult * min cell value + length_rate`` -- a PYORPS step costs the
length-weighted mean of the cells it touches, so never less than the
smallest of them per unit length.

**Tree** (the collector DW). With ``w_min = (1 + omega) c + mu_min l``
(``mu_min`` the cheapest rate of any label) and no turbine blocking, every
edge of a collector costs at least its ``w_min`` weight. An edge outside
``G_W`` lies on some turbine's path to the root, and that path costs at
least ``beta_j`` inside ``W`` before it leaves and ``b(g)`` after it last
re-enters. So a collector using any non-``G_W`` step costs at least
``min_j beta_j + b(g)``, and the in-window value is exact wherever it is
``<=`` that (needs every turbine in ``W``).

**Ellipse window.** With ``B(s)`` the budget left for the collector at
root ``s``, a competitive design only uses cells ``x`` with
``min_i d(t_i, x) + min_s [d(x, s) + B_max - B(s)] <= B_max`` -- two
drains (plan section 3.5).

All values here are in CELL units, like the Dijkstra kernels; multiply by
the cell size for EUR. The certificate is about the graph, so it holds
for any non-negative raster.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TreeCertificate",
    "WindowCertificate",
    "boundary_cells",
    "certify_path_field",
    "certify_tree_field",
    "drain",
    "ellipse_window",
    "step_table",
]

EXCLUDED = 65535


def _factor32(dr: int, dc: int, n: int) -> float:
    d = np.sqrt(np.float32(dr * dr + dc * dc), dtype=np.float32)
    return float(d / np.float32(2.0 + n))


def step_table(steps) -> list[tuple[int, int, list[tuple[int, int]], float,
                                    float]]:
    """``(dr, dc, intermediates, float32 cost factor, length)`` per step,
    exactly as the Cython kernels hold them (plan D8)."""
    from pyorps.graph.tower_field import intermediate_offsets
    out = []
    for dr, dc in np.asarray(steps)[:, :2]:
        dr, dc = int(dr), int(dc)
        inter = intermediate_offsets(dr, dc)
        out.append((dr, dc, inter, _factor32(dr, dc, len(inter)),
                    math.hypot(dr, dc)))
    return out


def boundary_cells(inside: np.ndarray, steps) -> np.ndarray:
    """``dW``: cells of ``W`` with a step whose end or an intermediate cell
    lies outside ``W`` (or off the raster)."""
    inside = np.asarray(inside, dtype=bool)
    rows, cols = inside.shape
    out = np.zeros_like(inside)
    padded = np.zeros((rows + 8, cols + 8), dtype=bool)
    padded[4:4 + rows, 4:4 + cols] = inside
    for dr, dc, inter, _f, _l in step_table(steps):
        for a, b in [(dr, dc)] + list(inter):
            if abs(a) > 4 or abs(b) > 4:
                raise ValueError("steps longer than 4 cells are not supported")
            shifted = padded[4 + a:4 + a + rows, 4 + b:4 + b + cols]
            out |= inside & ~shifted
    return out


def drain(values: np.ndarray, steps, seeds, labels, *, length_rate=0.0,
          weight_mult=1.0, excluded: np.ndarray | None = None,
          no_transit=None, engine: str = "auto", return_prev: bool = False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """``min_k labels[k] + d_w(seeds[k], v)`` for every cell, in cell units.

    ``excluded`` cells (and ``values == 65535``) are neither entered nor
    crossed. Uses ``MultiSourceSolver.solve_stream`` when the built
    extension has it (``engine="kernel"``), otherwise a pure-Python heap
    Dijkstra with the kernel's exact arithmetic (``engine="python"``).

    With ``return_prev`` it returns ``(dist, prev)``: ``prev`` holds, per
    flat cell, the cell the last step came from, ``-1`` where the seed's
    own label won (a seed wins ties) and where nothing arrived -- the
    trace code of the drain (plan section 3.2, item 7).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    vals = np.asarray(values)
    ras = np.asarray(vals, dtype=np.uint16).copy()
    if excluded is not None:
        ras[np.asarray(excluded, dtype=bool)] = EXCLUDED
    seeds = np.asarray(seeds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    order = np.argsort(labels, kind="stable")
    seeds, labels = seeds[order], labels[order]
    use_kernel = engine == "kernel"
    if engine == "auto":
        try:
            from pyorps.utils import _dijkstra as dj
            use_kernel = hasattr(dj.MultiSourceSolver, "solve_stream")
        except ImportError:
            use_kernel = False
    if use_kernel:
        from pyorps.utils import _dijkstra as dj
        st = np.asarray(steps, dtype=np.int8)
        solver = dj.make_multi_source_solver(ras, st, lean=True)
        solver.solve_stream(seeds, labels, length_rate=float(length_rate),
                            weight_mult=float(weight_mult),
                            no_transit=no_transit, keep_prev=return_prev)
        dist = solver.dist_array().reshape(ras.shape)
        if return_prev:
            prev = solver.prev_array().astype(np.int64)
            prev[~np.isfinite(dist.ravel())] = -1
            return dist, prev
        return dist
    return _python_drain(ras, steps, seeds, labels, float(length_rate),
                         float(weight_mult), no_transit, return_prev)


def _python_drain(ras, steps, seeds, labels, length_rate, weight_mult,
                  no_transit, return_prev=False):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    rows, cols = ras.shape
    blocked = ras == EXCLUDED
    nt = np.zeros_like(blocked)
    if no_transit is not None:
        nt.ravel()[np.asarray(no_transit, dtype=np.int64)] = True
    table = step_table(steps)
    dist = np.full(rows * cols, math.inf)
    prev = np.full(rows * cols, -1, dtype=np.int64)
    done = np.zeros(rows * cols, dtype=bool)
    heap = []
    for c, lab in zip(seeds.tolist(), labels.tolist()):
        if blocked.ravel()[c]:
            continue
        if lab < dist[c]:
            dist[c] = lab
            heapq.heappush(heap, (lab, c))
    while heap:
        d, u = heapq.heappop(heap)
        if done[u] or d > dist[u]:
            continue
        done[u] = True
        ur, uc = divmod(u, cols)
        for dr, dc, inter, fac, ln in table:
            vr, vc = ur + dr, uc + dc
            if not (0 <= vr < rows and 0 <= vc < cols):
                continue
            if blocked[vr, vc] or nt[vr, vc]:
                continue
            ok = True
            s = float(ras[ur, uc]) + float(ras[vr, vc])
            for a, b in inter:
                ir, ic = ur + a, uc + b
                if (not (0 <= ir < rows and 0 <= ic < cols)
                        or blocked[ir, ic] or nt[ir, ic]):
                    ok = False
                    break
                s += float(ras[ir, ic])
            if not ok:
                continue
            v = vr * cols + vc
            nd = d + weight_mult * (s * fac) + length_rate * ln
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    dist[~done] = math.inf
    if return_prev:
        prev[~done] = -1
        return dist.reshape(rows, cols), prev
    return dist.reshape(rows, cols)


@dataclass
class WindowCertificate:
    """A windowed field and where it is exact.

    Attributes:
        field: ``d_W`` in cell units (``inf`` outside ``W``).
        lower: the global lower bound ``min(d_W, beta + b)`` inside ``W``.
        exact: bool mask, where ``field`` equals the unwindowed field.
        beta: the boundary term (``min(beta_in, beta_out)``).
        b: ``d_W(dW, s)``.
    """
    field: np.ndarray
    lower: np.ndarray
    exact: np.ndarray
    beta: float
    b: np.ndarray

    @property
    def exact_fraction(self) -> float:
        inside = np.isfinite(self.field)
        return float(self.exact[inside].mean()) if inside.any() else 0.0


def certify_path_field(values: np.ndarray, steps, inside: np.ndarray, seeds,
                       labels, *, length_rate: float = 0.0,
                       weight_mult: float = 1.0, excluded=None,
                       no_transit=None, engine: str = "auto"
                       ) -> WindowCertificate:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Seeded path certificate of plan section 3.5 on one window.

    Parameters:
        values: The raster (cell values), full extent.
        steps: The neighbourhood step table.
        inside: bool mask of the window ``W``.
        seeds, labels: Seed cells (flat indices into ``values``) and their
            labels in cell units; seeds outside ``W`` enter through
            ``beta_out``.
        excluded: Cells no walk may enter anywhere.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    vals = np.asarray(values)
    inside = np.asarray(inside, dtype=bool)
    rows, cols = inside.shape  # pylint: disable=unused-variable
    exc = np.zeros_like(inside) if excluded is None else \
        np.asarray(excluded, dtype=bool)
    exc = exc | (vals == EXCLUDED)
    seeds = np.asarray(seeds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    in_w = inside.ravel()[seeds]
    win_exc = exc | ~inside
    dW = boundary_cells(inside, steps) & ~exc
    # in-window field from the seeds inside W
    if in_w.any():
        field = drain(vals, steps, seeds[in_w], labels[in_w],
                      length_rate=length_rate, weight_mult=weight_mult,
                      excluded=win_exc, no_transit=no_transit, engine=engine)
    else:
        field = np.full(inside.shape, math.inf)
    # b(s) = d_W(dW, s): one zero-seeded drain from the boundary cells
    bcells = np.flatnonzero(dW.ravel())
    if bcells.size:
        b = drain(vals, steps, bcells, np.zeros(bcells.size),
                  length_rate=length_rate, weight_mult=weight_mult,
                  excluded=win_exc, no_transit=no_transit, engine=engine)
    else:
        b = np.full(inside.shape, math.inf)
    beta_in = float(field.ravel()[bcells].min()) if bcells.size else math.inf
    beta_out = math.inf
    if (~in_w).any():
        finite = vals[~exc & (vals != EXCLUDED)]
        c_min = weight_mult * float(finite.min() if finite.size else 0.0) \
            + length_rate
        ins = np.argwhere(inside)
        for q, lab in zip(seeds[~in_w], labels[~in_w]):
            qr, qc = divmod(int(q), cols)
            gap = np.sqrt(((ins - (qr, qc)) ** 2).sum(1)).min()
            beta_out = min(beta_out, float(lab) + c_min * float(gap))
    beta = min(beta_in, beta_out)
    bound = beta + b
    lower = np.minimum(field, bound)
    exact = inside & (field <= bound)
    return WindowCertificate(field=field, lower=lower, exact=exact,
                             beta=beta, b=b)


@dataclass
class TreeCertificate:
    """Where an in-window collector field equals the unwindowed one.

    Attributes (EUR):
        lower: ``min(mv_window, bound)`` inside ``W``, ``inf`` outside.
        exact: ``mv_window <= bound`` inside ``W``.
        bound: ``(min_j beta_j + b(g)) * cell_m``.
        beta: ``min_j beta_j`` in EUR.
    """
    lower: np.ndarray
    exact: np.ndarray
    bound: np.ndarray
    beta: float

    @property
    def exact_fraction(self) -> float:
        fin = np.isfinite(self.lower)
        return float(self.exact[fin].mean()) if fin.any() else 0.0


def certify_tree_field(values: np.ndarray, steps, inside: np.ndarray,
                       turbines, mv_window: np.ndarray, *, mu_min: float,
                       trench_mult: float = 1.0, cell_m: float = 1.0,
                       engine: str = "auto") -> TreeCertificate:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """The tree certificate of plan section 3.5 for a collector field.

    Parameters:
        values: The raster, full extent.
        inside: bool mask of the window ``W``; every turbine must be in it.
        turbines: Flat cell indices of the turbines.
        mv_window: The collector field (EUR) computed with every cell
            outside ``W`` excluded.
        mu_min: The cheapest per-metre rate of any label
            (``min(mu_star(model)[1:])``).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    vals = np.asarray(values)
    inside = np.asarray(inside, dtype=bool)
    turb = np.asarray(turbines, dtype=np.int64).ravel()
    if not inside.ravel()[turb].all():
        raise ValueError("the tree certificate needs every turbine in W")
    if mu_min < 0:
        raise ValueError("mu_min must be >= 0")
    exc = ~inside | (vals == EXCLUDED)
    dW = np.flatnonzero((boundary_cells(inside, steps) & ~exc).ravel())
    mv = np.asarray(mv_window, dtype=np.float64).reshape(vals.shape)
    if dW.size == 0:
        # nothing leaves W: the window is closed, the field is exact
        bound = np.full(vals.shape, math.inf)
        beta = math.inf
    else:
        from_turbines = drain(vals, steps, turb, np.zeros(turb.size),
                              length_rate=mu_min, weight_mult=trench_mult,
                              excluded=exc, engine=engine).ravel()
        beta = float(from_turbines[dW].min())
        b = drain(vals, steps, dW, np.zeros(dW.size), length_rate=mu_min,
                  weight_mult=trench_mult, excluded=exc, engine=engine)
        bound = (beta + b) * float(cell_m)
        beta *= float(cell_m)
    lower = np.where(inside, np.minimum(mv, bound), math.inf)
    exact = inside & np.isfinite(mv) & (mv <= bound)
    return TreeCertificate(lower=lower, exact=exact, bound=bound, beta=beta)


def ellipse_window(values: np.ndarray, steps, turbines, root_budget, *,
                   mu_min: float, trench_mult: float = 1.0,
                   cell_m: float = 1.0, engine: str = "auto") -> np.ndarray:
    """``W_ell``: every cell a competitive collector can use (plan 3.5).

    Parameters:
        root_budget: EUR per cell, ``B(s)`` -- what is left for the
            collector if the UW stands at ``s`` (``Z_UB - LB_nonMV(s) +
            eps``); ``nan``, ``inf`` or negative where ``s`` is no root.
        mu_min: As in :func:`certify_tree_field`.

    Returns:
        bool mask. A design rooted at ``s`` whose collector costs at most
        ``B(s)`` never leaves it.
    """
    vals = np.asarray(values)
    turb = np.asarray(turbines, dtype=np.int64).ravel()
    B = np.asarray(root_budget, dtype=np.float64).ravel()
    roots = np.flatnonzero(np.isfinite(B) & (B >= 0))
    if roots.size == 0:
        return np.zeros(vals.shape, dtype=bool)
    cell = float(cell_m)
    b_max = float(B[roots].max())
    d_t = drain(vals, steps, turb, np.zeros(turb.size), length_rate=mu_min,
                weight_mult=trench_mult, engine=engine) * cell
    h = drain(vals, steps, roots, (b_max - B[roots]) / cell,
              length_rate=mu_min, weight_mult=trench_mult,
              engine=engine) * cell
    return (d_t + h) <= b_max * (1 + 1e-12)
