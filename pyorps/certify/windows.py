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

import math
from dataclasses import dataclass

import numpy as np

from pyorps.core.exceptions import InvalidInputError
from pyorps.graph.drain import EXCLUDED, boundary_cells, drain, step_table

__all__ = [
    "TreeCertificate",
    "WindowCertificate",
    "WindowGrowth",
    "boundary_cells",
    "certify_path_field",
    "certify_tree_field",
    "drain",
    "ellipse_window",
    "grow_window",
    "step_table",
]


@dataclass
class WindowCertificate:
    """A windowed field and where it is exact.

    Attributes:
        field: ``d_W`` in cell units (``inf`` outside ``W``).
        lower: the global lower bound ``min(d_W, beta + b)`` inside ``W``.
        exact: bool mask, where ``field`` equals the unwindowed field.
        beta: the boundary term (``min(beta_in, beta_out)``).
        b: ``d_W(dW, s)``.
        inside: bool mask of the window ``W`` (``None`` when not recorded).
    """
    field: np.ndarray
    lower: np.ndarray
    exact: np.ndarray
    beta: float
    b: np.ndarray
    inside: np.ndarray | None = None

    @property
    def exact_fraction(self) -> float:
        inside = np.isfinite(self.field)
        return float(self.exact[inside].mean()) if inside.any() else 0.0

    @property
    def uncertified(self) -> np.ndarray:
        """Cells of ``W`` where the field is NOT shown to be exact. At the
        growth cap these keep ``lower`` as their bound and are reported as
        such (review finding CODE-10)."""
        return self._in_window() & ~self.exact

    def _in_window(self) -> np.ndarray:
        if self.inside is not None:
            return self.inside
        return np.isfinite(self.field)

    def bracket(self, row: int, col: int) -> tuple[float, float]:
        """``(lower, upper)`` for the true field value at one cell of ``W``:
        equal where exact, otherwise ``lower`` (a global lower bound) and the
        in-window value (a walk that stays in ``W``, so an upper bound). A
        caller must report the bracket, never pick an end, for a cell that is
        not certified (plan 3.5, CODE-10)."""
        if not self._in_window()[row, col]:
            raise InvalidInputError("the cell lies outside the window")
        if self.exact[row, col]:
            v = float(self.field[row, col])          # may be inf: unreachable
            return v, v
        return float(self.lower[row, col]), float(self.field[row, col])


@dataclass
class WindowGrowth:
    """The outcome of one :func:`grow_window` request.

    Attributes:
        window: The window to use next (the grown one, or the old one when
            growth was refused).
        grown: True when the window got bigger.
        capped: True when growth was refused (cell cap or retry limit); the
            caller keeps ``min(d_W, beta + b)`` as the bound and reports the
            uncertified cells.
        reason: Why growth was refused, ``""`` when it happened.
        cells: Number of cells in ``window``.
        regrowths: Regrowths used after this call.
    """
    window: np.ndarray
    grown: bool
    capped: bool
    reason: str
    cells: int
    regrowths: int


def grow_window(inside: np.ndarray, *, margin_cells: int, cap_cells: int,
                regrowths: int = 0, max_regrowths: int = 1) -> WindowGrowth:
    """One capped regrowth of a window (plan 3.5, D2, review CODE-10).

    A failed border self-check used to mean "grow and re-run" with neither a
    cap nor an exit, and one attempt of the collector DW takes hours. Growth is
    now refused once ``max_regrowths`` have been used (the plan: at most one)
    or when the grown window would exceed ``cap_cells`` (the plan: 145 M cells
    for the DW, 240 M for single-source fields). On refusal the caller keeps
    the lower bound everywhere and reports a bracket where the certificate does
    not hold (:meth:`WindowCertificate.bracket`).

    Parameters:
        inside: bool mask of the current window.
        margin_cells: Cells added on every side (a square dilation, clipped to
            the raster).
        cap_cells: Largest allowed window, in cells.
        regrowths: Regrowths already used for this window.
        max_regrowths: How many are allowed in all.
    """
    cur = np.asarray(inside, dtype=bool)
    n = int(cur.sum())
    if margin_cells < 1:
        raise InvalidInputError("margin_cells must be >= 1")
    if regrowths >= max_regrowths:
        return WindowGrowth(cur, False, True,
                            f"{regrowths} regrowth(s) used, at most "
                            f"{max_regrowths} allowed", n, regrowths)
    grown = cur.copy()
    rows, cols = cur.shape
    # square dilation, separable: rows then columns
    k = int(margin_cells)
    tmp = cur.copy()
    for d in range(1, k + 1):
        tmp[d:, :] |= cur[:-d, :]
        tmp[:-d, :] |= cur[d:, :]
    grown = tmp.copy()
    for d in range(1, k + 1):
        grown[:, d:] |= tmp[:, :-d]
        grown[:, :-d] |= tmp[:, d:]
    m = int(grown.sum())
    if m > cap_cells:
        return WindowGrowth(cur, False, True,
                            f"the grown window has {m} cells, the cap is "
                            f"{cap_cells}", n, regrowths)
    if m == n:
        return WindowGrowth(cur, False, True,
                            "the window already covers every reachable cell "
                            "of the raster", n, regrowths)
    return WindowGrowth(grown, True, False, "", m, regrowths + 1)


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
                             beta=beta, b=b, inside=inside.copy())


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
        raise InvalidInputError("the tree certificate needs every turbine in W")
    if mu_min < 0:
        raise InvalidInputError("mu_min must be >= 0")
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
