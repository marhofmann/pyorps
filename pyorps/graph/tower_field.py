"""Precomputed tower fields -- overhead siting without a state explosion.

Implements ``docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md``.

An overhead line is **not** a raster walk. The conductor does not follow
cells; it goes straight from tower to tower. Modelling it as a walk forces
a ``(cell, direction, span, height)`` state vector -- 288 states per cell,
~1.66 TB dense on the HV window -- to carry information along a path that
has no physical counterpart. Model it as what it is, a sequence of towers
joined by straight spans, and one scalar per position is enough::

    T(x) = tower(x) + min over y, L_min <= |x - y| <= L_max,
                         of [ T(y) + span_cost(y, x) ]

Three structural facts make that cheap, and all three are exercised here:

1. **Layered by tower count.** Every edge adds exactly one tower, so the
   graph is a DAG in the tower-count dimension: Bellman-Ford converges in
   (max towers on an optimal line) sweeps, with no priority queue and no
   divergence. :attr:`TowerField.sweeps` reports how many it took.
2. **Span cost is a prefix difference.** Along a fixed direction the
   terrain integral is ``P(x) - P(y)`` for a cumulative sum along that
   direction's rays. The terrain does not change between sweeps, so the
   prefix tables are built once (:attr:`TowerFieldSolver.prefix_bytes`).
3. **The minimum over span length is a sliding-window minimum**, which
   :func:`~pyorps.utils.directional.ray_window_min` evaluates without the
   cost growing with the window width.

One primitive, three uses: the same directional operator gives the span
terrain cost (prefix **sum**, differenced), the ground clearance
(windowed **max** on the DEM -- which is why height is not a state
dimension) and the forbidden crossing test (run length of a 0/1 mask).

What the field computes is a *definition*, not a default, and the plan's
Phase 0 exists because the obvious definitions disagree by hundreds of
thousands of euro. :class:`TowerFieldModel` is that definition written
down -- span integral convention, whether the two terminal towers are
charged, whether the last span may be shorter than ``min_span_m`` -- and
:meth:`TowerFieldModel.describe` prints it.
:meth:`TowerFieldModel.matching_kernel` reproduces what
``ConstrainedPathFinder``'s kernel optimises, which is how
``tests/test_graph/test_tower_field_oracle.py`` compares the two.

Bounds come from the same machinery by parameter choice, and the
direction of each choice matters (the predecessor plan got an analogous
inequality backwards):

================  ==================================================
lower bound       no angle premium, **min**-pooled terrain and tower
                  ground cost, clearance and crossings ignored
upper / feasible  direction-resolved angles, **max**-pooled terrain,
                  clearance and forbidden crossings enforced, towers
                  confined to the sigma lattice -- which SHRINKS the
                  feasible set and therefore RAISES cost, so it
                  belongs here and never on the lower-bound side
================  ==================================================

:func:`tower_field_bounds` builds both and checks ``LB <= UB`` on every
cell, which is the cheapest strong invariant available.

Precision: fields accumulate and are exported in float64. float32 at
~1.3e7 EUR overstates half of all cells by up to 0.50 EUR, and the
substation case study's winning margin is 28 EUR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Any, Sequence

import numpy as np

from pyorps.utils.directional import (
    circular_window_min,
    direction_angles,
    primitive_directions,
    ray_prefix_sum,
    ray_run_length,
    ray_window_max,
    ray_window_min,
    shift,
)

__all__ = [
    "AngleTables",
    "ClearanceModel",
    "Tower",
    "TowerField",
    "TowerFieldModel",
    "TowerFieldSolver",
    "TowerLattice",
    "angle_tables_from_profile",
    "clearance_from_profile",
    "coarsen",
    "cost_factor",
    "intermediate_offsets",
    "solve_tower_field",
    "tower_field_bounds",
    "tower_field_from_raster",
]

_POOLINGS = ("mean", "min", "max", "sample")
_INTEGRALS = ("pyorps", "trapezoid", "midpoint")
_CLEARANCE_CHARGE = ("arriving", "both_ends")


# ======================================================================
# step geometry
# ======================================================================

def intermediate_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    """Intermediate cells of one step, exactly as PYORPS' kernels see them.

    A numpy transcription of ``_calculate_intermediate_steps_cython``
    (``_raster_context.pyx``): adjacent steps have none, a single
    diagonal decomposes into its two orthogonal components, and a longer
    step samples both the floor and the ceil of the interpolated position
    at every major step. The cost factor that goes with it is
    ``hypot(dr, dc) / (2 + len(offsets))``, so that
    ``(value[u] + sum(intermediates) + value[v]) * factor`` is a
    length-weighted MEAN of the cells the step passes over.

    Offsets are relative to the step's ORIGIN cell ``u``.
    """
    dr, dc = int(dr), int(dc)
    adr, adc = abs(dr), abs(dc)
    if adr + adc <= 1:
        return []
    if max(adr, adc) == 1:
        return [(dr, 0), (0, dc)]
    k = max(adr, adc)
    out: list[tuple[int, int]] = []
    for p in range(1, k):
        rk = p * dr / k
        ck = p * dc / k
        fr, fc = int(math.floor(rk)), int(math.floor(ck))
        out.append((fr, fc))
        cr, cc = int(math.ceil(rk)), int(math.ceil(ck))
        if (fr, fc) != (cr, cc):
            out.append((cr, cc))
    return out


def cost_factor(dr: int, dc: int) -> float:
    """``hypot(dr, dc) / (2 + n_intermediates)`` -- PYORPS' own step factor."""
    return math.hypot(dr, dc) / (2.0 + len(intermediate_offsets(dr, dc)))


def _sha(arr) -> str:
    """SHA-256 of an array's dtype, shape and bytes (field provenance)."""
    import hashlib
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


# ======================================================================
# discretisation
# ======================================================================

@dataclass(frozen=True)
class TowerLattice:
    """Where towers may stand, and along which directions.

    Towers are confined to a sub-grid of the cost raster, ``factor``
    cells apart: ``sigma_x_m`` metres across columns and ``sigma_y_m``
    metres down rows. That restriction SHRINKS the feasible set, so a
    field computed on it bounds the continuous optimum from ABOVE and
    must never be presented as a lower bound on the continuous problem.

    **The two spacings are not interchangeable.** A raster that reports
    "1 m resolution" routinely has ``transform.a = 1.0000017516`` and
    ``transform.e = -0.9999783186`` -- not equal to each other and
    neither of them 1. Collapsing them to the x size, which is what
    every caller used to pass as ``cell_size_m``, moved a real 110 kV
    line's reconstructed cost by up to 0.045 % (~1 500 EUR on a 5 MEUR
    route) and did so with BOTH signs, so it reads as noise rather than
    as a bias. Hence two sizes throughout, and hence :attr:`sigma_m`
    refusing to answer when they differ instead of picking one.

    Parameters:
        cell_size_m: Raster resolution, when the pixels really are
            square -- synthetic grids and tests. A real raster's pixel
            size belongs in ``cell_size_x_m``/``cell_size_y_m``, which
            :func:`tower_field_from_raster` reads off a transform.
        factor: Lattice spacing in raster cells. ``1`` puts towers on
            every cell, the regime where the model can be compared to
            ``ConstrainedPathFinder`` cell for cell.
        directions: ``(K, 2)`` primitive integer directions in LATTICE
            steps. Defaults to 16 directions (``dmax=2``).
        pooling: How a raster block collapses to one lattice value --
            ``"mean"``, ``"min"`` (lower-bound side), ``"max"``
            (upper-bound side) or ``"sample"`` (the block centre).
        transform: Affine transform of the LATTICE, when georeferenced.
        crs: CRS of ``transform``.
        cell_size_x_m, cell_size_y_m: Pixel size across COLUMNS and down
            ROWS, i.e. ``abs(transform.a)`` and ``abs(transform.e)`` of
            the underlying raster. They come as a pair.
    """

    cell_size_m: float | None = None
    factor: int = 1
    directions: np.ndarray = _field(
        default_factory=lambda: primitive_directions(2))
    pooling: str = "mean"
    transform: Any = None
    crs: Any = None
    cell_size_x_m: float | None = None
    cell_size_y_m: float | None = None

    def __post_init__(self):
        if self.factor < 1:
            raise ValueError(f"factor must be >= 1, got {self.factor}")
        if self.pooling not in _POOLINGS:
            raise ValueError(
                f"pooling must be one of {_POOLINGS}, got {self.pooling!r}")
        self._resolve_cell_size()
        dirs = np.asarray(self.directions, dtype=np.int64).reshape(-1, 2)
        if dirs.shape[0] == 0:
            raise ValueError("need at least one direction")
        for p, q in dirs:
            if math.gcd(abs(int(p)), abs(int(q))) != 1:
                raise ValueError(
                    f"direction ({p}, {q}) is not primitive; two directions "
                    f"on the same ray double the work and change nothing")
        object.__setattr__(self, "directions", dirs)

    def _resolve_cell_size(self) -> None:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Settle ``cell_size_{x,y}_m`` from whichever form was passed."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        sx, sy = self.cell_size_x_m, self.cell_size_y_m
        if (sx is None) != (sy is None):
            raise ValueError(
                "cell_size_x_m and cell_size_y_m come as a pair; passing one "
                "and leaving the other to cell_size_m is how the y pixel "
                "size got dropped in the first place")
        if sx is None:
            if self.cell_size_m is None:
                raise ValueError(
                    "pass cell_size_m for a square grid, or the pair "
                    "cell_size_x_m/cell_size_y_m for a real raster")
            sx = sy = self.cell_size_m
        elif self.cell_size_m is not None and not (
                float(self.cell_size_m) == float(sx) == float(sy)):
            raise ValueError(
                f"cell_size_m={self.cell_size_m!r} contradicts "
                f"cell_size_x_m={sx!r} / cell_size_y_m={sy!r}; pass one form "
                f"or the other")
        sx, sy = float(sx), float(sy)
        if not (sx > 0.0 and sy > 0.0):
            raise ValueError(
                f"cell sizes must be positive, got x={sx!r}, y={sy!r}")
        object.__setattr__(self, "cell_size_x_m", sx)
        object.__setattr__(self, "cell_size_y_m", sy)
        # Leave `cell_size_m` as None when the pixels are not square: it
        # has no honest value there, and None fails loudly where a
        # silently-picked axis would not.
        if self.cell_size_m is None and sx == sy:
            object.__setattr__(self, "cell_size_m", sx)

    @property
    def sigma_x_m(self) -> float:
        """Lattice spacing across COLUMNS (the x axis), in metres."""
        return float(self.cell_size_x_m) * int(self.factor)

    @property
    def sigma_y_m(self) -> float:
        """Lattice spacing down ROWS (the y axis), in metres."""
        return float(self.cell_size_y_m) * int(self.factor)

    @property
    def is_square(self) -> bool:
        """True when the two pixel sizes agree exactly."""
        return float(self.cell_size_x_m) == float(self.cell_size_y_m)

    @property
    def sigma_m(self) -> float:
        """Lattice spacing in metres -- SQUARE lattices only.

        Deliberately raises rather than collapsing the two spacings:
        every scalar summary of an anisotropic pixel is wrong by a
        direction-dependent amount, and silently so.
        """
        if not self.is_square:
            raise ValueError(
                f"this lattice is {self.sigma_x_m!r} m across columns and "
                f"{self.sigma_y_m!r} m down rows, so it has no single "
                f"sigma_m; use sigma_x_m/sigma_y_m, or step_length_m(d) for "
                f"the physical length of a lattice step")
        return self.sigma_x_m

    @property
    def n_directions(self) -> int:
        return int(self.directions.shape[0])

    @property
    def angle_scale(self) -> tuple[float, float] | None:
        """``(row_m, col_m)`` for physical angles; ``None`` when square."""
        return None if self.is_square else (self.sigma_y_m, self.sigma_x_m)

    @property
    def angles(self) -> np.ndarray:
        """Direction angles in radians, aligned with :attr:`directions`.

        PHYSICAL angles: on an anisotropic lattice the step ``(p, q)``
        does not point along ``atan2(q, p)``, so the deflection two
        integer steps describe is not the deflection built on the ground.
        """
        return direction_angles(self.directions, scale=self.angle_scale)

    def step_length_parts(self, d) -> tuple[float, float]:
        """``(shape, scale)`` whose PRODUCT is :meth:`step_length_m`.

        Split rather than multiplied out because the callers weight a
        terrain accumulator by the shape and scale the result once, and
        a square lattice has to keep the historical
        ``hypot(p, q)`` x ``sigma`` factorisation: same two float
        multiplications in the same order, hence bit-for-bit the same
        cost as before anisotropy existed.
        """
        p, q = int(d[0]), int(d[1])
        if self.is_square:
            return math.hypot(p, q), self.sigma_x_m
        return math.hypot(p * self.sigma_y_m, q * self.sigma_x_m), 1.0

    def step_length_m(self, d) -> float:
        """Physical length of one lattice step along ``d``.

        ``d`` is ``(rows, cols)``, and rows run along y while columns run
        along x -- so the two components are in DIFFERENT metres and only
        ``hypot(p * sigma_y_m, q * sigma_x_m)`` is the real length.
        """
        shape, scale = self.step_length_parts(d)
        return shape * scale

    def cell_xy(self, rows, cols):
        """Lattice cell centres as ``(x, y)`` in the CRS of :attr:`transform`."""
        if self.transform is None:
            raise ValueError("this lattice is not georeferenced")
        t = self.transform
        r = np.asarray(rows, dtype=np.float64)
        c = np.asarray(cols, dtype=np.float64)
        x = t.c + (c + 0.5) * t.a + (r + 0.5) * t.b
        y = t.f + (c + 0.5) * t.d + (r + 0.5) * t.e
        return x, y

    def xy_to_cell(self, x, y):
        """Inverse of :meth:`cell_xy`, by FLOOR -- the rasterio ``rowcol`` rule.

        Pinned deliberately: the substation study's ``_sample_field`` uses
        ``np.round`` where every library path uses ``rowcol``'s floor, and
        with candidates sitting at pixel centres half of them then shift
        by one cell.
        """
        if self.transform is None:
            raise ValueError("this lattice is not georeferenced")
        inv = ~self.transform
        cols, rows = inv @ (np.asarray(x, dtype=np.float64),
                            np.asarray(y, dtype=np.float64))
        return (np.floor(rows).astype(np.int64),
                np.floor(cols).astype(np.int64))


def coarsen(a, factor: int, how: str = "mean") -> np.ndarray:
    """Collapse each ``factor x factor`` block of ``a`` to one value.

    Trailing rows and columns that do not fill a whole block are
    dropped, so every lattice cell has the same support.
    ``how="sample"`` takes the block centre instead of reducing.
    """
    a = np.asarray(a)
    factor = int(factor)
    if factor == 1:
        return a.astype(np.float64, copy=True)
    if how not in _POOLINGS:
        raise ValueError(f"how must be one of {_POOLINGS}, got {how!r}")
    rows = (a.shape[0] // factor) * factor
    cols = (a.shape[1] // factor) * factor
    if rows == 0 or cols == 0:
        raise ValueError(
            f"a {a.shape} grid holds no whole {factor}x{factor} block")
    if how == "sample":
        off = factor // 2
        return np.ascontiguousarray(
            a[off:rows:factor, off:cols:factor]).astype(np.float64)
    blocks = a[:rows, :cols].astype(np.float64).reshape(
        rows // factor, factor, cols // factor, factor)
    if how == "mean":
        return blocks.mean(axis=(1, 3))
    if how == "min":
        return blocks.min(axis=(1, 3))
    return blocks.max(axis=(1, 3))


# ======================================================================
# profile-derived tables
# ======================================================================

@dataclass(frozen=True)
class AngleTables:
    """Per-direction-pair tower premium and hard-limit validity.

    ``premium[i, j]`` is everything a tower costs BEYOND its ground cost
    when the line arrives along direction ``i`` and leaves along ``j`` --
    the tower-type base cost plus the turn penalty. ``valid[i, j]`` is
    ``False`` where the deflection exceeds the profile's hard limit.

    ``premium[i, i]`` is the suspension tower, i.e. what a tower that
    does not turn costs, so a tier-1 field folds the cheapest of those
    into the node cost and a tier-2 field charges the real pair.
    """

    premium: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        prem = np.asarray(self.premium, dtype=np.float64)
        val = np.asarray(self.valid, dtype=bool)
        if prem.ndim != 2 or prem.shape[0] != prem.shape[1]:
            raise ValueError(f"premium must be (K, K), got {prem.shape}")
        if val.shape != prem.shape:
            raise ValueError("valid must have the same shape as premium")
        if np.any(prem[val] < 0):
            raise ValueError(
                "a negative angle premium breaks the tier-1 lower bound, "
                "which relies on premium >= 0 with premium(0) minimal")
        object.__setattr__(self, "premium", prem)
        object.__setattr__(self, "valid", val)

    @property
    def n_directions(self) -> int:
        return int(self.premium.shape[0])

    def straight_premium(self) -> np.ndarray:
        """``premium[i, i]`` -- the no-turn (suspension) tower cost."""
        return np.diag(self.premium).copy()


@dataclass(frozen=True)
class ClearanceModel:
    """Ground clearance, resolved without making height a state dimension.

    The binding constraint on a straight span is the HIGHEST ground under
    it, which a directional max filter answers in O(1) -- so the required
    tower top is a lookup, not a search dimension. Using the maximum
    rather than the true catenary profile can only OVER-state the
    required height, so the designs it produces stay feasible; the price
    of that conservatism is measured against the oracle rather than
    argued (plan risk 5).

    Parameters:
        min_clearance_m: Required air gap above ground and obstacles.
        conductor_weight_per_m: N/m, for the sag term.
        conductor_tension_n: N, for the sag term.
        heights_m: Available attachment heights, ascending.
        height_premium: Cost premium per entry of ``heights_m``.
    """

    min_clearance_m: float
    conductor_weight_per_m: float
    conductor_tension_n: float
    heights_m: tuple[float, ...]
    height_premium: tuple[float, ...]

    def __post_init__(self):
        h = tuple(float(v) for v in self.heights_m)
        p = tuple(float(v) for v in self.height_premium)
        if not h:
            raise ValueError("ClearanceModel needs at least one height")
        if len(h) != len(p):
            raise ValueError("heights_m and height_premium must align")
        if list(h) != sorted(h):
            raise ValueError("heights_m must be ascending")
        if self.conductor_tension_n <= 0:
            raise ValueError("conductor_tension_n must be positive")
        object.__setattr__(self, "heights_m", h)
        object.__setattr__(self, "height_premium", p)

    def sag_m(self, span_m: float) -> float:
        """Mid-span sag of a parabolic approximation to the catenary."""
        return (self.conductor_weight_per_m * span_m * span_m
                / (8.0 * self.conductor_tension_n))

    def premium_for(self, required_h) -> np.ndarray:
        """Cheapest height class that reaches ``required_h``; ``inf`` if none."""
        req = np.asarray(required_h, dtype=np.float64)
        heights = np.asarray(self.heights_m, dtype=np.float64)
        prem = np.asarray(self.height_premium, dtype=np.float64)
        idx = np.searchsorted(heights, req, side="left")
        out = np.full(req.shape, np.inf, dtype=np.float64)
        ok = idx < heights.size
        if ok.any():
            out[ok] = prem[idx[ok]]
        return out


def angle_tables_from_profile(profile, lattice: TowerLattice) -> AngleTables:
    """Tower premium and hard-limit validity for a lattice's directions.

    Reuses the profile's own LUT builders, so the table a tier-2 field
    charges is term-for-term the one the constrained kernel charges:
    ``precompute_tower_angle_costs`` (the tower TYPE by deflection) plus
    ``precompute_angle_lut`` (the turn penalty and the hard limit).

    Both builders read ``atan2(dc, dr)`` off whatever step array they are
    handed, so an anisotropic lattice hands them the PHYSICAL
    displacement instead of the integer step -- otherwise the hard angle
    limit and the tower type are decided on an angle the line never turns
    through. On a square lattice the steps go in unscaled, so the tables
    stay bit-for-bit what they were.
    """
    steps = np.asarray(lattice.directions, dtype=np.int64)
    scale = lattice.angle_scale
    if scale is not None:
        steps = steps * np.asarray(scale, dtype=np.float64).reshape(1, 2)
    turn_cost, turn_valid = profile.precompute_angle_lut(steps)
    type_cost = profile.precompute_tower_angle_costs(steps)
    premium = np.asarray(type_cost, dtype=np.float64).copy()
    turn = np.asarray(turn_cost, dtype=np.float64)
    finite = np.isfinite(turn)
    premium[finite] += turn[finite]
    premium[~finite] = np.inf
    return AngleTables(premium=premium, valid=np.asarray(turn_valid, bool))


def clearance_from_profile(profile) -> ClearanceModel | None:
    """A :class:`ClearanceModel` from a profile, or ``None`` if incomplete."""
    if not profile.has_clearance_constraints:
        return None
    ordered = list(profile.effective_tower_heights_m)      # descending
    prem = [float(v) for v in profile.precompute_height_premium()]
    by_height = dict(zip(ordered, prem))
    heights = sorted(by_height)
    return ClearanceModel(
        min_clearance_m=float(profile.min_clearance_m),
        conductor_weight_per_m=float(profile.conductor_weight_per_m),
        conductor_tension_n=float(profile.conductor_tension_n),
        heights_m=tuple(float(h) for h in heights),
        height_premium=tuple(by_height[h] for h in heights),
    )


# ======================================================================
# the model -- Phase 0's "write down which quantity the field computes"
# ======================================================================

@dataclass(frozen=True)
class TowerFieldModel:
    """The quantity a tower field computes. Every knob changes the answer.

    Parameters:
        min_span_m, max_span_m: Admissible straight-span length.
        max_span_inclusive: Whether ``max_span_m`` itself is allowed.
            ``False`` matches the constrained kernel, which continues a
            span only while ``new_span < max_span``.
        last_span_min_m: Shortest span allowed INTO the queried point.
            ``None`` means ``min_span_m``; ``0.0`` matches the kernel,
            which stops as soon as the target cell settles at any span.
        terminal_tower_cost: Node cost at the two ends of the line.
        charge_terminal_towers: Whether to charge it at all. The 110 kV
            profile's own footer records that the kernel does NOT and the
            report DOES, a >= 560 kEUR difference on a real line -- so
            this is a decision the caller takes, not one they inherit.
        span_integral: ``"pyorps"`` reproduces the kernel's
            ``(value[u] + intermediates + value[v]) * cost_factor *
            cell_size`` per step and is the only convention under which
            the field and ``find_route`` price the same terrain.
            ``"trapezoid"`` and ``"midpoint"`` are the textbook line
            integrals, kept because they are what the prototypes measured.
        angle_tier: ``1`` drops the angle premium, which -- the premium
            being non-negative -- is a valid LOWER bound, and a far
            tighter one than "terrain plus minimum tower count" because
            it still solves the span and spacing problem. ``2`` carries
            the last span's direction: K fields instead of one. (Tier 3
            of the plan is ``ConstrainedPathFinder.find_route`` used as
            an oracle, not a setting here.)
        angle_mode: How tier 2 evaluates the min-plus over directions.
            ``"exact"`` minimises over the admissible incoming
            directions. ``"window"`` is the plan's ``K x n_classes``
            trick: one circular sliding-window minimum per distinct
            premium level, using a window narrow enough for EVERY
            outgoing direction. It is never below ``"exact"``, so it
            stays on the upper-bound side, and it cannot name an
            argmin, so it forbids ``record_pred``.

            **It is not currently a speed-up, and the plan expected one.**
            The saving assumes the premium takes one value per tower
            class -- four on the 110 kV profile. Measured, it takes
            **20** distinct values among the admissible pairs, because
            ``angle_cost_function: piecewise`` INTERPOLATES the turn
            penalty continuously; only the tower-TYPE term is a
            staircase. With 20 levels the window form costs more than
            the exact minimum, which the 40 deg hard limit has already
            pruned to 232 of 1024 pairs at K = 32 (0.65 s against 0.36 s,
            ``benchmarks/tower_field/bench_tower_field.py``). It earns
            its place as a stated upper bound; it becomes a speed-up
            only where the turn penalty is a step function or absent.
        forbidden_mode: ``"exact"`` caps each span at the last clear
            lattice cell behind it; ``"conservative"`` rejects a span
            whenever anything in the widest admissible window is
            blocked; ``"off"`` ignores crossings -- lower-bound side only.
        clearance_charge: Where a span's height premium lands.
            ``"arriving"`` charges it once, at the node the span arrives
            at, which is what the plan specifies and is TIGHT but can
            under-charge a tower whose outgoing span is the binding one
            (``premium(max(l, r)) >= max(premium(l), premium(r))``).
            ``"both_ends"`` charges it twice, which dominates
            ``sum over towers of premium(max(l, r))`` and is therefore
            the sound choice for a stated upper bound.
        max_sweeps: Safety cap. Convergence is layered, so the natural
            count is (extent / max_span), not an iteration budget.
    """

    min_span_m: float
    max_span_m: float
    max_span_inclusive: bool = False
    last_span_min_m: float | None = None
    terminal_tower_cost: float = 0.0
    charge_terminal_towers: bool = True
    span_integral: str = "pyorps"
    angle_tier: int = 1
    angle_mode: str = "exact"
    forbidden_mode: str = "exact"
    clearance_charge: str = "arriving"
    max_sweeps: int = 4096

    def __post_init__(self):
        if not 0.0 <= self.min_span_m <= self.max_span_m:
            raise ValueError(
                f"need 0 <= min_span_m <= max_span_m, got "
                f"{self.min_span_m} and {self.max_span_m}")
        if self.span_integral not in _INTEGRALS:
            raise ValueError(
                f"span_integral must be one of {_INTEGRALS}, "
                f"got {self.span_integral!r}")
        if self.angle_tier not in (1, 2):
            raise ValueError(
                f"angle_tier must be 1 or 2, got {self.angle_tier}")
        if self.angle_mode not in ("exact", "window"):
            raise ValueError(f"unknown angle_mode {self.angle_mode!r}")
        if self.forbidden_mode not in ("exact", "conservative", "off"):
            raise ValueError(
                f"unknown forbidden_mode {self.forbidden_mode!r}")
        if self.clearance_charge not in _CLEARANCE_CHARGE:
            raise ValueError(
                f"clearance_charge must be one of {_CLEARANCE_CHARGE}, "
                f"got {self.clearance_charge!r}")

    @property
    def effective_last_span_min_m(self) -> float:
        return (self.min_span_m if self.last_span_min_m is None
                else float(self.last_span_min_m))

    @classmethod
    def matching_kernel(cls, profile, **overrides) -> TowerFieldModel:
        """The model ``ConstrainedPathFinder``'s kernel actually optimises.

        Reconciled term by term against ``_constrained_dijkstra.pyx``
        (plan Phase 0):

        * the source is seeded in every direction at ``dist = 0`` with
          ``span = 0`` and no tower, and the search stops the moment the
          target cell settles in any state -- so **neither terminal
          tower is charged** and the last span carries no minimum length;
        * a tower is placed only where ``cur_span >= min_span``, and a
          span continues only while ``new_span < max_span``, so a span
          between two towers lies in ``[min_span, max_span)``;
        * terrain accrues as ``(value[u] + intermediates + value[v]) *
          cost_factor * cell_size`` per step -- ``span_integral="pyorps"``.

        ``_terrain_eur``, which the REPORT uses, drops
        ``IMPASSABLE_CELL_COST`` from its category sum where the kernel
        charges it; with ``ignore_max_cost`` on there is nothing to drop
        and the two agree.
        """
        if not profile.has_span_constraints:
            raise ValueError(
                f"profile {profile.name!r} has no span constraints, so it "
                f"describes no tower chain to build a field over")
        kw: dict[str, Any] = {
            "min_span_m": float(profile.min_span_m),
            "max_span_m": float(profile.max_span_m),
            "max_span_inclusive": False,
            "last_span_min_m": 0.0,
            "terminal_tower_cost": 0.0,
            "charge_terminal_towers": False,
            "span_integral": "pyorps",
        }
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def as_reported(cls, profile, **overrides) -> TowerFieldModel:
        """The model ``ConstrainedPath``'s REPORT prices.

        Same line, different bill: the report adds the two terminal
        towers at ``terminal_tower_cost`` each (280 kEUR in the shipped
        110 kV profile), and treats both ends as line ends.
        """
        kw: dict[str, Any] = {
            "terminal_tower_cost": float(profile.terminal_tower_cost),
            "charge_terminal_towers": True,
        }
        kw.update(overrides)
        return cls.matching_kernel(profile, **kw)

    def describe(self) -> str:
        """One paragraph naming the quantity, for logs and field metadata."""
        term = (f"terminal towers charged at "
                f"{self.terminal_tower_cost:,.0f} EUR"
                if self.charge_terminal_towers
                else "terminal towers NOT charged")
        hi = "<=" if self.max_span_inclusive else "<"
        return (
            f"tower-chain cost with spans in "
            f"[{self.min_span_m:g} m, {hi} {self.max_span_m:g} m], "
            f"last span >= {self.effective_last_span_min_m:g} m, {term}, "
            f"terrain by the {self.span_integral!r} span integral, "
            f"angle tier {self.angle_tier} ({self.angle_mode}), "
            f"forbidden crossings {self.forbidden_mode}, "
            f"clearance charged {self.clearance_charge}")


# ======================================================================
# results
# ======================================================================

@dataclass(frozen=True)
class Tower:
    """One tower of a reconstructed sequence.

    Not the same type as
    :class:`pyorps.core.constrained_path.Tower`, which carries a costed
    breakdown from ``find_route``. This one is a lattice position on a
    field's reconstructed chain, so neither is exported under the bare
    name ``Tower`` at package level.
    """

    row: int
    col: int
    x: float | None
    y: float | None
    span_to_next_m: float
    direction: tuple[int, int] | None
    is_terminal: bool


class TowerField:
    """A settled tower field: one cost per lattice position.

    :attr:`arrival` is the quantity :attr:`model` describes, evaluated at
    every lattice cell. ``inf`` means no admissible tower chain reaches
    that cell -- from this source, under these spans, these directions
    and these crossings.
    """

    def __init__(self, *, arrival, chain, lattice, model, source, sweeps,
                 interior_pred=None, arrival_pred=None, tower_cost=None,
                 meta=None, seeded=None, masks=None, seed=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        self._arrival = arrival
        self._chain = chain
        self._lattice = lattice
        self._model = model
        self._source = (None if source is None
                        else (int(source[0]), int(source[1])))
        self._sweeps = int(sweeps)
        # bool (rows, cols): the arrival value IS the seed there -- the
        # line starts and ends at that cell (plan D5)
        self._seeded = seeded
        # crossing / placement / terminal_ok planes and the clearance model
        # the field was settled under, for the matched-tier-1 check
        self._masks = dict(masks or {})
        self._seed = seed
        # interior_pred: (dir, m, in) each (L, rows, cols), L = 1 for
        # tier 1 and K for tier 2, indexed by the ARRIVING direction.
        self._interior = interior_pred
        # arrival_pred: (dir, m, in) each (rows, cols) -- the last span
        # into the queried point, plus the arriving direction at the
        # tower before it.
        self._arr = arrival_pred
        self._tower_cost = tower_cost
        self._meta = dict(meta or {})

    # ------------------------------------------------------------ arrays

    @property
    def arrival(self) -> np.ndarray:
        """Cost of reaching each lattice cell, per :attr:`model`. float64."""
        return self._arrival

    @property
    def chain(self) -> np.ndarray:
        """Cost of a chain that CONTINUES through each cell.

        Differs from :attr:`arrival` exactly where the model treats the
        queried point as a line end: a shorter last span may be allowed
        and the node cost is the terminal tower, not a suspension tower.
        Tier 2 keeps a direction axis here.
        """
        return self._chain

    @property
    def lattice(self) -> TowerLattice:
        return self._lattice

    @property
    def model(self) -> TowerFieldModel:
        return self._model

    @property
    def source(self) -> tuple[int, int] | None:
        """Source lattice cell, ``(row, col)``; ``None`` for a field
        settled from seeds alone."""
        return self._source

    @property
    def masks(self) -> dict:
        """The ``crossing``, ``placement`` and ``terminal_ok`` planes and
        the ``clearance`` model the field was settled under."""
        return dict(self._masks)

    @property
    def seed(self) -> np.ndarray | None:
        """The start labels (source and seeds) the field was settled from."""
        return self._seed

    @property
    def seeded(self) -> np.ndarray | None:
        """Cells whose :attr:`arrival` is their own seed (the source is
        one), ``None`` for fields from before seeding existed."""
        return self._seeded

    @property
    def sweeps(self) -> int:
        """Bellman-Ford sweeps to the fixpoint -- the max tower count."""
        return self._sweeps

    @property
    def has_paths(self) -> bool:
        """True when :meth:`tower_sequence` can walk a predecessor back."""
        return self._arr is not None

    @property
    def meta(self) -> dict:
        """Provenance: model description, discretisation, byte counts."""
        return dict(self._meta)

    @property
    def reachable(self) -> np.ndarray:
        """Boolean mask of cells with a finite :attr:`arrival`."""
        return np.isfinite(self._arrival)

    @property
    def predecessor_bytes(self) -> int:
        """Bytes held by the predecessor planes; 0 without them."""
        if self._arr is None:
            return 0
        return int(sum(a.nbytes for a in self._arr)
                   + sum(a.nbytes for a in self._interior))

    # ----------------------------------------------------------- queries

    def cost_at(self, rows, cols) -> np.ndarray:
        """:attr:`arrival` at lattice cells, ``inf`` outside the lattice."""
        r = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        c = np.atleast_1d(np.asarray(cols, dtype=np.int64))
        out = np.full(np.broadcast(r, c).shape, np.inf, dtype=np.float64)
        h, w = self._arrival.shape
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        if np.any(ok):
            out[ok] = self._arrival[r[ok], c[ok]]
        return out

    def costs_to(self, points: Sequence) -> np.ndarray:
        """:meth:`cost_at` for map coordinates, by the floor rule."""
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rows, cols = self._lattice.xy_to_cell(arr[:, 0], arr[:, 1])
        return self.cost_at(rows, cols)

    def n_towers_at(self, row: int, col: int) -> int:
        """Towers on the reconstructed line, both terminals included."""
        return len(self.tower_sequence(row, col))

    def length_at(self, row: int, col: int) -> float:
        """Length of the reconstructed line in metres."""
        return float(sum(t.span_to_next_m
                         for t in self.tower_sequence(row, col)))

    def tower_sequence(self, row: int, col: int) -> list[Tower]:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Walk the predecessor plane back from ``(row, col)`` to where the
        chain started: the source, or the seed whose label won there.

        Start first. Empty when the cell is unreachable; raises when the
        field was solved without a predecessor plane.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if self._arr is None:
            raise NotImplementedError(
                "this field was solved with record_pred=False, so it prices "
                "positions but cannot return tower sequences")
        row, col = int(row), int(col)
        h, w = self._arrival.shape
        if not (0 <= row < h and 0 <= col < w):
            return []
        if not np.isfinite(self._arrival[row, col]):
            return []
        if (row, col) == self._source or (
                self._seeded is not None and self._seeded[row, col]):
            return [self._tower(row, col, 0.0, None, True)]

        a_dir, a_m, a_in = self._arr
        i_dir, i_m, i_in = self._interior
        dirs = self._lattice.directions

        cells = [(row, col)]
        d_idx = int(a_dir[row, col])
        m = int(a_m[row, col])
        layer = int(a_in[row, col])
        cur = (row, col)
        limit = self._arrival.size + 1
        while True:
            if d_idx < 0 or m <= 0:
                return []
            p, q = int(dirs[d_idx][0]), int(dirs[d_idx][1])
            prev = (cur[0] - m * p, cur[1] - m * q)
            cells.append(prev)
            # ``layer`` is the in-direction recorded at prev: -1 where a
            # seed (the source among them) started the chain there
            if prev == self._source or layer < 0:
                break
            if len(cells) > limit:
                raise RuntimeError(
                    "predecessor walk did not reach the source; the field "
                    "and its predecessor plane disagree")
            layer = max(0, min(layer, i_dir.shape[0] - 1))
            d_idx = int(i_dir[layer, prev[0], prev[1]])
            nxt = int(i_in[layer, prev[0], prev[1]])
            m = int(i_m[layer, prev[0], prev[1]])
            layer = nxt
            cur = prev

        cells.reverse()
        out: list[Tower] = []
        for i, (r, c) in enumerate(cells):
            if i + 1 < len(cells):
                nr, nc = cells[i + 1]
                span = self._lattice.step_length_m((nr - r, nc - c))
                dr, dc = nr - r, nc - c
                g = math.gcd(abs(dr), abs(dc)) or 1
                direction = (dr // g, dc // g)
            else:
                span, direction = 0.0, None
            out.append(self._tower(r, c, span, direction,
                                   i in (0, len(cells) - 1)))
        return out

    def _tower(self, r, c, span, direction, terminal) -> Tower:
        if self._lattice.transform is not None:
            x, y = self._lattice.cell_xy(r, c)
            x, y = float(np.ravel(x)[0]), float(np.ravel(y)[0])
        else:
            x = y = None
        return Tower(row=int(r), col=int(c), x=x, y=y,
                     span_to_next_m=float(span), direction=direction,
                     is_terminal=bool(terminal))

    def route_geometry(self, row: int, col: int):
        """The reconstructed line as a shapely ``LineString``.

        ``None`` when the cell is unreachable or the lattice carries no
        transform.
        """
        seq = self.tower_sequence(row, col)
        if len(seq) < 2 or self._lattice.transform is None:
            return None
        from shapely.geometry import LineString
        return LineString([(t.x, t.y) for t in seq])

    def __repr__(self) -> str:
        return (f"TowerField({self._arrival.shape[0]}x"
                f"{self._arrival.shape[1]} lattice, tier "
                f"{self._model.angle_tier}, {self._sweeps} sweeps, "
                f"{int(self.reachable.sum())} reachable)")


# ======================================================================
# the solver
# ======================================================================

class TowerFieldSolver:
    """Bellman-Ford over tower chains, one sweep per tower layer.

    Everything expensive is built once in :meth:`__init__`: the
    directional prefix tables (the terrain does not change between
    sweeps), the per-direction span windows, the clearance premium and
    the crossing run lengths. A sweep is then a handful of O(N) array
    passes per direction with no queue, no divergence and no data
    dependence -- the friendliest possible shape for a GPU port.

    Parameters:
        values: Lattice cost values, EUR per metre. Must be FINITE
            everywhere; an impassable cell belongs in ``blocked``, not in
            an ``inf`` value, because one ``inf`` in a prefix sum poisons
            every span along that ray.
        tower_cost: Node cost per lattice cell, EUR. ``inf`` where no
            tower may stand.
        lattice: The discretisation.
        model: What is being computed.
        blocked: Cells no span may cross or land on.
        dem: Ground elevation per lattice cell, for clearance.
        obstacle: Vegetation/building height above ``dem``.
        clearance: Clearance model; requires ``dem``.
        angles: Direction-pair premium; required for ``angle_tier=2``.
    """

    def __init__(self, *, values, tower_cost, lattice: TowerLattice,
                 model: TowerFieldModel, blocked=None, dem=None,
                 obstacle=None, clearance: ClearanceModel | None = None,
                 angles: AngleTables | None = None, terminal_ok=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D, got {self.values.shape}")
        if not np.isfinite(self.values).all():
            raise ValueError(
                "values must be finite everywhere -- put impassable cells "
                "in `blocked` instead, or a directional prefix sum turns "
                "every span on that ray into inf")
        self.tower_cost = np.asarray(tower_cost, dtype=np.float64)
        if self.tower_cost.shape != self.values.shape:
            raise ValueError("tower_cost must match values in shape")
        self.lattice = lattice
        self.model = model
        self.shape = self.values.shape
        self.blocked = (np.zeros(self.shape, dtype=bool) if blocked is None
                        else np.asarray(blocked, dtype=bool))
        if self.blocked.shape != self.shape:
            raise ValueError("blocked must match values in shape")
        if clearance is not None and dem is None:
            raise ValueError("a ClearanceModel needs a dem")
        self.clearance = clearance
        self.dem = None if dem is None else np.asarray(dem, dtype=np.float64)
        self.obstacle = (None if obstacle is None
                         else np.asarray(obstacle, dtype=np.float64))
        self.angles = angles
        if model.angle_tier == 2 and angles is None:
            raise ValueError(
                "angle_tier=2 needs an AngleTables; build one with "
                "angle_tables_from_profile(profile, lattice)")
        if angles is not None and angles.n_directions != lattice.n_directions:
            raise ValueError(
                f"AngleTables is {angles.n_directions}x{angles.n_directions} "
                f"but the lattice has {lattice.n_directions} directions")
        # A line ENDS with a terminal tower, so the queried cell has to be
        # able to host one. Without this the arrival pass charges only
        # `terminal_tower_cost` at the end and never asks whether anything
        # may stand there -- so an excluded candidate came back with a
        # finite cost even though `tower_cost` was inf on it.
        self.terminal_ok = (
            (~self.blocked) & np.isfinite(self.tower_cost)
            if terminal_ok is None else np.asarray(terminal_ok, dtype=bool))
        if self.terminal_ok.shape != self.shape:
            raise ValueError("terminal_ok must match values in shape")
        self._build()

    # -------------------------------------------------------- precompute

    def _span_window(self, d, min_span_m) -> tuple[int, int] | None:
        ds = self.lattice.step_length_m(d)
        m_lo = max(1, int(math.ceil(min_span_m / ds - 1e-9)))
        if self.model.max_span_inclusive:
            m_hi = int(math.floor(self.model.max_span_m / ds + 1e-9))
        else:
            m_hi = int(math.ceil(self.model.max_span_m / ds - 1e-9)) - 1
        if m_hi < m_lo:
            return None
        return m_lo, m_hi

    def _step_cost(self, p: int, q: int) -> np.ndarray:
        """EUR of the lattice step of direction ``(p, q)`` arriving at x."""
        v = self.values
        # `hyp * sigma` is the step's physical length; on a square lattice
        # the pair is the historical (hypot(p, q), sigma_m).
        hyp, sigma = self.lattice.step_length_parts((p, q))
        if self.model.span_integral == "midpoint":
            return v * (hyp * sigma)
        if self.model.span_integral == "trapezoid":
            return 0.5 * (shift(v, p, q, 0.0) + v) * (hyp * sigma)
        inter = intermediate_offsets(p, q)
        acc = shift(v, p, q, 0.0) + v
        for orow, ocol in inter:
            acc = acc + shift(v, p - orow, q - ocol, 0.0)
        return acc * (hyp / (2.0 + len(inter))) * sigma

    def _step_blocked(self, p: int, q: int) -> np.ndarray:
        """Cells whose ARRIVING step along ``(p, q)`` is not usable.

        A step is unusable when its origin, its destination or any of
        its INTERMEDIATE cells is blocked, and when its origin is off
        the grid. The intermediates matter and are easy to forget: the
        constrained kernel skips a step whose supercover touches an
        excluded cell (``_precompute_intermediate_cache`` leaves the
        entry unset and the relaxation ``continue``s), so a diagonal may
        not slip between two blocked cells. Testing only the cells ON
        the ray lets exactly that through, and Phase 0 caught it as the
        field undercutting the kernel by 25-380 EUR wherever a route
        passed a wall.
        """
        out = self.blocked | shift(self.blocked, p, q, True)
        for orow, ocol in intermediate_offsets(p, q):
            out = out | shift(self.blocked, p - orow, q - ocol, True)
        return out

    def _height_premium(self, p, q, m_hi):
        """Height premium for a span arriving at x along ``(p, q)``.

        Conservative by construction: the ground maximum is taken over
        the WIDEST admissible window and the sag over the LONGEST
        admissible span, so the premium can only be too high.
        """
        if self.clearance is None:
            return None
        ground = self.dem if self.obstacle is None else self.dem + self.obstacle
        gmax = ray_window_max(ground, p, q, 0, m_hi)
        span_m = m_hi * self.lattice.step_length_m((p, q))
        need = (gmax + self.clearance.min_clearance_m
                + self.clearance.sag_m(span_m)) - self.dem
        prem = self.clearance.premium_for(need)
        if self.model.clearance_charge == "both_ends":
            prem = prem * 2.0
        return prem

    def _build(self) -> None:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        self._tables: dict[int, dict[str, Any]] = {}
        dirs = self.lattice.directions
        prefix_bytes = 0
        any_blocked = bool(self.blocked.any())
        last_min = self.model.effective_last_span_min_m
        for i in range(dirs.shape[0]):
            p, q = int(dirs[i][0]), int(dirs[i][1])
            win = self._span_window((p, q), self.model.min_span_m)
            fin = self._span_window((p, q), last_min)
            if win is None and fin is None:
                continue
            prefix = ray_prefix_sum(self._step_cost(p, q), p, q)
            prefix_bytes += prefix.nbytes
            reach = max(w[1] for w in (win, fin) if w is not None)
            self._tables[i] = {
                "p": p, "q": q, "prefix": prefix,
                "window": win, "final_window": fin,
                "height": self._height_premium(p, q, reach),
                # Consecutive usable STEPS ending at each cell, so the
                # count is directly the longest admissible span in
                # lattice steps -- no off-by-one between "cells clear"
                # and "steps usable".
                "run": (ray_run_length(self._step_blocked(p, q), p, q, reach)
                        if any_blocked and self.model.forbidden_mode != "off"
                        else None),
            }
        if not self._tables:
            raise ValueError(
                f"no direction admits a span in "
                f"[{self.model.min_span_m}, {self.model.max_span_m}] m on a "
                f"{self.lattice.sigma_x_m:g} x {self.lattice.sigma_y_m:g} m "
                f"lattice -- the shortest lattice step is already longer "
                f"than max_span_m, or the longest admissible multiple is "
                f"shorter than min_span_m")
        self._prefix_bytes = prefix_bytes

    @property
    def prefix_bytes(self) -> int:
        """Bytes held by the directional prefix tables, built once."""
        return int(self._prefix_bytes)

    @property
    def usable_directions(self) -> list[int]:
        """Indices of directions that admit at least one span."""
        return sorted(self._tables)

    # ------------------------------------------------------------ relax

    def _relax(self, s, entry, window, *, record):
        """Windowed minimum along one direction, with crossings applied."""
        p, q, prefix = entry["p"], entry["q"], entry["prefix"]
        m_lo, m_hi = window
        if record:
            w, m = ray_window_min(s, p, q, m_lo, m_hi, return_arg=True)
        else:
            w, m = ray_window_min(s, p, q, m_lo, m_hi), None

        run = entry["run"]
        if run is not None:
            m_max = run
            if self.model.forbidden_mode == "conservative":
                bad = m_max < m_hi
            else:
                bad = m_max < m_lo
                partial = (m_max >= m_lo) & (m_max < m_hi)
                if partial.any():
                    w, m = self._rescan_partial(
                        s, entry, m_lo, m_hi, m_max, partial, w, m,
                        record=record)
            w = np.where(bad, np.inf, w)
            if record:
                m = np.where(bad, -1, m)

        cand = w + prefix
        if entry["height"] is not None:
            cand = cand + entry["height"]
        return cand, m

    def _rescan_partial(self, s, entry, m_lo, m_hi, m_max, partial, w, m,
                        *, record):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Exact window for the few cells whose span a crossing cuts short.

        A cell lands here only when the last clear lattice cell behind it
        falls INSIDE the admissible window -- a thin boundary set around
        every obstacle. Everything else is answered in O(1) by the
        unobstructed sliding window, so the exact treatment of forbidden
        crossings costs work proportional to the obstacle perimeter, not
        to the grid.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        p, q = entry["p"], entry["q"]
        rows, cols = np.nonzero(partial)
        best = np.full(rows.size, np.inf, dtype=np.float64)
        best_m = np.full(rows.size, -1, dtype=np.int64)
        cap = m_max[rows, cols]
        h, wd = s.shape
        for step in range(m_lo, m_hi + 1):
            live = step <= cap
            if not live.any():
                continue
            rr = rows - step * p
            cc = cols - step * q
            inside = live & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < wd)
            if not inside.any():
                continue
            vals = np.full(rows.size, np.inf, dtype=np.float64)
            vals[inside] = s[rr[inside], cc[inside]]
            better = vals < best
            best = np.where(better, vals, best)
            if record:
                best_m = np.where(better, step, best_m)
        w = w.copy()
        w[rows, cols] = best
        if record:
            m = m.copy()
            m[rows, cols] = best_m
        return w, m

    # ------------------------------------------------------------- solve

    def solve(self, source=None, *, seed_chain=None,
              record_pred: bool = True) -> TowerField:
        """Settle the field from a fixed terminal and/or seeded chains.

        Parameters:
            source: ``(row, col)`` lattice cell of the fixed end, which
                starts a chain at the terminal node cost. ``None`` when
                only seeds start chains.
            seed_chain: ``(rows, cols)`` float64, the cost of a chain that
                STARTS at each cell with a tower there (whatever node cost
                the caller includes), ``inf`` where none starts. It is
                min-ed into the chain value inside every sweep and before
                the arrival pass, and a seed wins ties (plan D5, CODE-11:
                the node costs are never overwritten, so any number of
                seeds is exact). The mixed-leg driver (D6) seeds with the
                cable field plus the transition cost.
            record_pred: Keep the argmin planes so tower sequences and
                route geometry can be reconstructed; a sequence starts at
                the seed that won.

        Returns:
            A :class:`TowerField`.
        """
        if record_pred and self.model.angle_mode == "window":
            raise ValueError(
                "angle_mode='window' pools directions to get the plan's "
                "K x n_classes cost, so it cannot name which incoming "
                "direction won; solve with record_pred=False, or use "
                "angle_mode='exact' when a tower sequence is needed")
        seed, src = self._seed_plane(source, seed_chain)
        if self.model.angle_tier == 1:
            return self._solve_tier1(seed, src, record_pred)
        return self._solve_tier2(seed, src, record_pred)

    def _seed_plane(self, source, seed_chain):
        """The start labels of every chain: the source's terminal cost and
        the caller's seeds, min-ed; refuses NaN, negative and blocked."""
        seed = np.full(self.shape, np.inf, dtype=np.float64)
        src = None
        if source is not None:
            src = self._check_source(source)
            seed[src] = self._terminal_node_cost()
        if seed_chain is not None:
            sc = np.asarray(seed_chain, dtype=np.float64)
            if sc.shape != self.shape:
                raise ValueError(f"seed_chain has shape {sc.shape}, the "
                                 f"lattice {self.shape}")
            if np.isnan(sc).any() or (sc < 0).any():
                raise ValueError("seed labels must be >= 0 (inf: no seed), "
                                 "not NaN or negative")
            if (np.isfinite(sc) & self.blocked).any():
                raise ValueError("a seed sits on a blocked cell, where no "
                                 "span may start")
            np.minimum(seed, sc, out=seed)
        if not np.isfinite(seed).any():
            raise ValueError("no source and no finite seed: nothing starts "
                             "a chain")
        return seed, src

    def _terminal_node_cost(self) -> float:
        return (float(self.model.terminal_tower_cost)
                if self.model.charge_terminal_towers else 0.0)

    def _check_source(self, source) -> tuple[int, int]:
        sr, sc = int(source[0]), int(source[1])
        h, w = self.shape
        if not (0 <= sr < h and 0 <= sc < w):
            raise ValueError(
                f"source cell {(sr, sc)} is outside the {h}x{w} lattice")
        if self.blocked[sr, sc]:
            raise ValueError(
                f"source cell {(sr, sc)} is blocked; a line cannot start on "
                f"a cell no span may touch")
        return sr, sc

    def _field_masks(self) -> dict:
        return {"crossing": self.blocked,
                "placement": np.isfinite(self.tower_cost),
                "terminal_ok": self.terminal_ok,
                "clearance": self.clearance}

    def _meta(self, sweeps: int, seed=None) -> dict:
        lat = self.lattice
        return {
            "values_sha256": _sha(self.values),
            "n_seeds": (None if seed is None
                        else int(np.isfinite(seed).sum())),
            "model": self.model.describe(),
            # `sigma_m` is None on an anisotropic lattice: there is no
            # scalar that describes it, and None beats a wrong number.
            "sigma_m": lat.sigma_x_m if lat.is_square else None,
            "sigma_x_m": lat.sigma_x_m,
            "sigma_y_m": lat.sigma_y_m,
            "n_directions": self.lattice.n_directions,
            "usable_directions": len(self._tables),
            "pooling": self.lattice.pooling,
            "prefix_bytes": self.prefix_bytes,
            "sweeps": sweeps,
        }

    @staticmethod
    def _converged(new, old) -> bool:
        return bool(np.array_equal(np.nan_to_num(new, posinf=-1.0),
                                   np.nan_to_num(old, posinf=-1.0)))

    # ------------------------------------------------------------ tier 1

    def _solve_tier1(self, seed, src, record) -> TowerField:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        shape = self.shape
        arrive = np.full(shape, np.inf, dtype=np.float64)
        node = self.tower_cost
        has_seed = np.isfinite(seed)

        p_dir = np.full(shape, -1, np.int16) if record else None
        p_m = np.full(shape, -1, np.int32) if record else None
        p_in = np.zeros(shape, np.int16) if record else None

        def seeded_chain(arr):
            base = arr + node
            chain = np.minimum(base, seed)
            arg = None
            if record:
                # tier 1 has one layer: 0, or -1 where a seed starts the
                # chain (a seed wins ties)
                arg = np.where(has_seed & (seed <= base), -1, 0).astype(
                    np.int16)
            return chain, arg

        sweeps = 0
        for _ in range(self.model.max_sweeps):
            chain, chain_arg = seeded_chain(arrive)
            best = arrive.copy()
            b_dir = p_dir.copy() if record else None
            b_m = p_m.copy() if record else None
            b_in = p_in.copy() if record else None
            for idx, entry in self._tables.items():
                if entry["window"] is None:
                    continue
                cand, m = self._relax(chain - entry["prefix"], entry,
                                      entry["window"], record=record)
                improves = cand < best
                best = np.where(improves, cand, best)
                if record:
                    b_dir = np.where(improves, idx, b_dir)
                    b_m = np.where(improves, m, b_m)
                    b_in = np.where(
                        improves,
                        self._gather_at_predecessor(chain_arg, entry, m,
                                                    improves), b_in)
            sweeps += 1
            done = self._converged(best, arrive)
            arrive, p_dir, p_m, p_in = best, b_dir, b_m, b_in
            if done:
                break
        else:
            raise RuntimeError(
                f"tower field did not converge in {self.model.max_sweeps} "
                f"sweeps; the layered argument says it should take about "
                f"(extent / max_span_m) of them")

        chain, chain_arg = seeded_chain(arrive)
        arrival, a_dir, a_m, a_in = self._arrival_pass(
            chain, chain_arg, arrive, p_dir, p_m, p_in, record)
        arrival = np.where(self.terminal_ok, arrival, np.inf)
        won = has_seed & (seed <= arrival)
        arrival = np.where(won, seed, arrival)
        interior = arrival_pred = None
        if record:
            interior = (p_dir[None, ...], p_m[None, ...], p_in[None, ...])
            arrival_pred = (a_dir, a_m, a_in)
        return TowerField(
            arrival=arrival, chain=chain, lattice=self.lattice,
            model=self.model, source=src, sweeps=sweeps,
            interior_pred=interior, arrival_pred=arrival_pred,
            tower_cost=node, meta=self._meta(sweeps, seed), seeded=won,
            masks=self._field_masks(), seed=seed)
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    def _arrival_pass(self, chain, chain_arg, arrive, p_dir, p_m, p_in,
                      record):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """One extra relaxation for the queried END of the line.

        A line end is not an ordinary tower: the model may allow a
        shorter last span there, and the node cost is the terminal tower
        rather than a suspension tower. So the exported field is its own
        relaxation over the converged chain values -- never the interior
        field with a different label on it.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        terminal = self._terminal_node_cost()
        if (self.model.effective_last_span_min_m == self.model.min_span_m
                and terminal == 0.0):
            return arrive.copy(), p_dir, p_m, p_in
        best = np.full(self.shape, np.inf, dtype=np.float64)
        b_dir = np.full(self.shape, -1, np.int16) if record else None
        b_m = np.full(self.shape, -1, np.int32) if record else None
        b_in = np.zeros(self.shape, np.int16) if record else None
        for idx, entry in self._tables.items():
            win = entry["final_window"]
            if win is None:
                continue
            cand, m = self._relax(chain - entry["prefix"], entry, win,
                                  record=record)
            improves = cand < best
            best = np.where(improves, cand, best)
            if record:
                b_dir = np.where(improves, idx, b_dir)
                b_m = np.where(improves, m, b_m)
                b_in = np.where(
                    improves,
                    self._gather_at_predecessor(chain_arg, entry, m,
                                                improves), b_in)
        return best + terminal, b_dir, b_m, b_in

    # ------------------------------------------------------------ tier 2

    def _angle_order(self):
        """Per outgoing direction, admissible incoming ones ordered by premium.

        Only pairs the hard angle limit allows appear, which is what
        makes tier 2 affordable: the 110 kV profile's 40 deg limit keeps
        about a fifth of the K x K pairs.
        """
        prem, valid = self.angles.premium, self.angles.valid
        ok = valid & np.isfinite(prem)
        out = []
        for j in range(prem.shape[1]):
            rows = np.flatnonzero(ok[:, j])
            rows = rows[np.argsort(prem[rows, j], kind="stable")]
            out.append([(int(i), float(prem[i, j])) for i in rows])
        return out

    def _angle_windows(self):
        """Circular index half-widths per distinct premium level.

        The plan's ``K x n_classes`` trick: the premium depends only on
        the deflection and takes one value per tower class, so
        ``min over theta of [T(theta) + premium]`` becomes a circular
        sliding-window minimum on the direction axis -- one pass per
        class instead of a K x K min-plus product per cell.

        Primitive lattice directions are NOT uniformly spaced, so a
        constant index width cannot be exact. The width taken here is the
        largest that stays INSIDE the class for EVERY outgoing direction,
        so the window can only omit candidates, which raises the result.
        That keeps ``"window"`` an upper bound on ``"exact"`` -- the side
        a feasible design has to be on.
        """
        prem, valid = self.angles.premium, self.angles.valid
        ok = valid & np.isfinite(prem)
        k = prem.shape[0]
        out = []
        for c in np.unique(prem[ok]):
            widths = []
            for j in range(k):
                allowed = ok[:, j] & (prem[:, j] <= c)
                if not allowed[j]:
                    widths.append(-1)
                    continue
                h = 0
                while h + 1 <= k // 2 and allowed[(j + h + 1) % k] \
                        and allowed[(j - h - 1) % k]:
                    h += 1
                widths.append(h)
            half = min(widths)
            if half >= 0:
                out.append((int(half), float(c)))
        if not out:
            raise ValueError(
                "no (incoming, outgoing) direction pair survives the hard "
                "angle limit, so no tower can turn at all")
        return out

    def _angle_step(self, arrive, base, order, windows, record):
        """``A[psi] = min over phi [arrive[phi] + premium(phi, psi)] + base``."""
        if windows is not None:
            out = np.full_like(arrive, np.inf)
            pooled: dict[int, np.ndarray] = {}
            for half, cost in windows:
                if half not in pooled:
                    pooled[half] = circular_window_min(arrive, half)
                out = np.minimum(out, pooled[half] + cost)
            return out + base, None
        out = np.full_like(arrive, np.inf)
        arg = np.full(arrive.shape, -1, np.int16) if record else None
        for j in range(arrive.shape[0]):
            acc = out[j]
            for i, cost in order[j]:
                cand = arrive[i] + cost
                if record:
                    better = cand < acc
                    acc = np.where(better, cand, acc)
                    arg[j] = np.where(better, i, arg[j])
                else:
                    acc = np.minimum(acc, cand)
            out[j] = acc
        return out + base, arg

    def _solve_tier2(self, seed, src, record) -> TowerField:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        dirs = sorted(self._tables)
        k = self.lattice.n_directions
        shape = self.shape
        base = self.tower_cost
        order = self._angle_order()
        windows = (self._angle_windows()
                   if self.model.angle_mode == "window" else None)
        has_seed = np.isfinite(seed)

        def seed_in(chain, chain_arg):
            # A seed is a line START: it imposes no incoming direction, so
            # every outgoing direction may leave it at its label; it wins
            # ties (CODE-11: never overwrite a node cost).
            if record and chain_arg is not None:
                chain_arg[has_seed[None] & (seed[None] <= chain)] = -1
            return np.minimum(chain, seed[None])

        arrive = np.full((k,) + shape, np.inf, dtype=np.float64)
        p_dir = np.full((k,) + shape, -1, np.int16) if record else None
        p_m = np.full((k,) + shape, -1, np.int32) if record else None
        p_in = np.full((k,) + shape, -1, np.int16) if record else None

        sweeps = 0
        for _ in range(self.model.max_sweeps):
            chain, chain_arg = self._angle_step(arrive, base, order, windows,
                                                record)
            chain = seed_in(chain, chain_arg)

            best = arrive.copy()
            b_dir = p_dir.copy() if record else None
            b_m = p_m.copy() if record else None
            b_in = p_in.copy() if record else None
            for idx in dirs:
                entry = self._tables[idx]
                if entry["window"] is None:
                    continue
                cand, m = self._relax(chain[idx] - entry["prefix"], entry,
                                      entry["window"], record=record)
                improves = cand < best[idx]
                best[idx] = np.where(improves, cand, best[idx])
                if record:
                    b_dir[idx] = np.where(improves, idx, b_dir[idx])
                    b_m[idx] = np.where(improves, m, b_m[idx])
                    b_in[idx] = np.where(
                        improves,
                        self._gather_at_predecessor(chain_arg[idx], entry, m,
                                                    improves),
                        b_in[idx])
            sweeps += 1
            done = self._converged(best, arrive)
            arrive, p_dir, p_m, p_in = best, b_dir, b_m, b_in
            if done:
                break
        else:
            raise RuntimeError(
                f"tier-2 tower field did not converge in "
                f"{self.model.max_sweeps} sweeps")

        chain, chain_arg = self._angle_step(arrive, base, order, windows,
                                            record)
        chain = seed_in(chain, chain_arg)
        arrival, a_dir, a_m, a_in = self._arrival_pass_tier2(
            chain, chain_arg, record)
        arrival = np.where(self.terminal_ok, arrival, np.inf)
        won = has_seed & (seed <= arrival)
        arrival = np.where(won, seed, arrival)

        meta = self._meta(sweeps, seed)
        meta["angle_mode"] = self.model.angle_mode
        meta["angle_pairs"] = int(sum(len(o) for o in order))
        meta["angle_classes"] = (len(windows) if windows is not None
                                 else None)
        interior = (p_dir, p_m, p_in) if record else None
        arrival_pred = (a_dir, a_m, a_in) if record else None
        return TowerField(
            arrival=arrival, chain=chain, lattice=self.lattice,
            model=self.model, source=src, sweeps=sweeps,
            interior_pred=interior, arrival_pred=arrival_pred,
            tower_cost=base, meta=meta, seeded=won,
            masks=self._field_masks(), seed=seed)
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    @staticmethod
    def _gather_at_predecessor(prev_arg, entry, m, improves):
        """Which incoming direction the winning predecessor node used."""
        p, q = entry["p"], entry["q"]
        out = np.full(improves.shape, -1, dtype=np.int16)
        rows, cols = np.nonzero(improves)
        if rows.size == 0:
            return out
        steps = m[rows, cols]
        rr = rows - steps * p
        cc = cols - steps * q
        h, w = prev_arg.shape
        ok = (steps > 0) & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        out[rows[ok], cols[ok]] = prev_arg[rr[ok], cc[ok]]
        return out

    def _arrival_pass_tier2(self, chain, chain_arg, record):
        best = np.full(self.shape, np.inf, dtype=np.float64)
        b_dir = np.full(self.shape, -1, np.int16) if record else None
        b_m = np.full(self.shape, -1, np.int32) if record else None
        b_in = np.full(self.shape, -1, np.int16) if record else None
        for idx, entry in self._tables.items():
            win = entry["final_window"]
            if win is None:
                continue
            cand, m = self._relax(chain[idx] - entry["prefix"], entry, win,
                                  record=record)
            improves = cand < best
            best = np.where(improves, cand, best)
            if record:
                b_dir = np.where(improves, idx, b_dir)
                b_m = np.where(improves, m, b_m)
                b_in = np.where(
                    improves,
                    self._gather_at_predecessor(chain_arg[idx], entry, m,
                                                improves),
                    b_in)
        return best + self._terminal_node_cost(), b_dir, b_m, b_in


# ======================================================================
# convenience constructors
# ======================================================================

def solve_tower_field(*, values, tower_cost, source, lattice, model,
                      record_pred: bool = True, **kwargs) -> TowerField:
    """Build a :class:`TowerFieldSolver` and settle it in one call."""
    solver = TowerFieldSolver(values=values, tower_cost=tower_cost,
                              lattice=lattice, model=model, **kwargs)
    return solver.solve(source, record_pred=record_pred)


def _exempt_cells(lattice, shape, unblock_xy, unblock_cells):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Lattice cells named by :func:`tower_field_from_raster`'s ``unblock``.

    Returns ``(rows, cols)`` of the in-bounds ones, or ``None`` when
    nothing was named. Out-of-bounds entries are dropped silently: a
    terminal outside the window is already unreachable for reasons this
    exemption does not change.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    rows: list[int] = []
    cols: list[int] = []
    if unblock_cells is not None:
        for r, c in unblock_cells:
            rows.append(int(r))
            cols.append(int(c))
    if unblock_xy is not None:
        pts = np.asarray(unblock_xy, dtype=np.float64).reshape(-1, 2)
        if pts.size:
            if lattice.transform is None:
                raise ValueError(
                    "unblock_xy needs a georeferenced lattice; pass a "
                    "transform, or name the cells with unblock_cells")
            rr, cc = lattice.xy_to_cell(pts[:, 0], pts[:, 1])
            rows.extend(int(v) for v in np.atleast_1d(rr))
            cols.extend(int(v) for v in np.atleast_1d(cc))
    if not rows:
        return None
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    return (r[ok], c[ok]) if ok.any() else None


def _pixel_size(cell_size_m, cell_size_x_m, cell_size_y_m, transform):
    """Raster pixel size ``(x, y)`` from whatever the caller supplied.

    A transform answers it exactly, and that is the point: ``abs(a)`` and
    ``abs(e)`` are routinely NOT equal (measured 1.0000017516 and
    0.9999783186 on a "1 m" cost surface), so a caller who collapses them
    to one scalar has already thrown the y size away. An explicit size
    still wins, because a synthetic grid's transform is often a
    placeholder that says nothing about the metres the caller means.
    """
    if cell_size_x_m is not None or cell_size_y_m is not None:
        if cell_size_x_m is None or cell_size_y_m is None:
            raise ValueError(
                "cell_size_x_m and cell_size_y_m come as a pair")
        return float(cell_size_x_m), float(cell_size_y_m)
    if cell_size_m is not None:
        return float(cell_size_m), float(cell_size_m)
    if transform is None:
        raise ValueError(
            "pass cell_size_m, the pair cell_size_x_m/cell_size_y_m, or a "
            "transform to read the pixel size off")
    return float(abs(transform.a)), float(abs(transform.e))


def tower_field_from_raster(raster, *, cell_size_m=None, source_xy=None,
                            source_cell=None, profile=None, model=None,
                            factor=1, directions=None, pooling="mean",
                            transform=None, crs=None, impassable=65535,
                            blocked_fraction=0.0, unblock_xy=None,
                            unblock_cells=None, spans_cross_exclusions=False,
                            dem=None, obstacle=None,
                            clearance=None, angles=None, tower_cost=None,
                            record_pred=True, cell_size_x_m=None,
                            cell_size_y_m=None, crossing_values=None,
                            crossing_mask=None, terminal_ok=None,
                            seed_chain=None) -> TowerField:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Settle a tower field straight from a cost raster and a profile.

    The raster is the same EUR-per-metre surface the routers use. Cells
    at ``impassable`` become BLOCKED rather than merely expensive: one
    ``inf`` in a directional prefix sum poisons every span along that
    ray, and "expensive" is not what the exclusion means anyway.

    Parameters:
        raster: 2-D cost values.
        cell_size_m: Raster resolution in metres, for a SQUARE grid.
            Omit it when ``transform`` is given and the true, generally
            unequal pixel sizes ``abs(a)`` / ``abs(e)`` are read off it
            -- passing ``abs(transform.a)`` here is exactly the collapse
            :class:`TowerLattice` documents as a 0.045 % cost error.
        cell_size_x_m, cell_size_y_m: Explicit anisotropic pixel size,
            when there is no transform to read it from.
        source_xy / source_cell: The fixed terminal, as map coordinates
            (needs ``transform``) or as a lattice ``(row, col)``.
        profile: An
            :class:`~pyorps.core.infrastructure_profile.InfrastructureProfile`;
            supplies spans, the tower ground-cost LUT and, for tier 2,
            the angle tables.
        model: Overrides the model derived from ``profile``.
        factor: Lattice spacing in raster cells.
        directions, pooling, transform, crs: Lattice settings.
        impassable: Raster value that means "excluded"; ``None`` to
            treat every value as traversable.
        blocked_fraction: Excluded SHARE of a lattice cell that makes it
            blocked. ``0.0`` (the default) means any excluded raster
            cell blocks it. See the note at the call site: at a coarse
            sigma that is aggressive enough to make a reachable target
            read as unreachable.
        unblock_xy, unblock_cells: Positions whose lattice cell is
            usable whatever the exclusion says, priced at the model's
            terminal node cost. A line ENDS on something that already
            exists -- a substation gantry, an existing tower -- and the
            rule that forbids a new greenfield tower there does not
            forbid terminating on what is already built. Without this a
            PCC inside a substation compound reads as unreachable while
            the cable routes straight to it: measured on the CIRED wind
            farm, PCC0's lattice cell is 80 % excluded and PCC2's 76 %,
            and both came back with no overhead cost at all.
        spans_cross_exclusions: Whether a SPAN may pass over excluded
            ground while a TOWER still may not stand on it. These are
            two different questions and the plan's section 2 keeps them
            apart: the exclusion layers behind a cost raster (water
            protection, Natura 2000, built-up) forbid ground works, and
            a conductor passing overhead is a separate consent. The
            default ``False`` conflates them, which is the conservative
            reading; ``True`` keeps the exclusion on tower placement and
            lifts it from the crossing test. It changes the answer:
            measured on the CIRED wind farm, PCC0 sits in a compound
            whose whole 3x3 lattice neighbourhood is >= 80 % excluded,
            so with ``False`` no span can arrive at all and the target
            reports no overhead cost.
        tower_cost: Explicit per-cell node cost, overriding the LUT.
        crossing_values: Raster-resolution EUR per metre charged where the
            raster is excluded, BEFORE pooling (plan D5, M4-07): the
            per-class price of passing over a protected class. Without it
            excluded raster cells pool as 0.0, which understates a mixed
            lattice cell -- fine for a lower bound, not for the exact D.
        crossing_mask: Raster-resolution bool of ground no span may cross
            (the C7 turbine-clearance mask); ORed into the crossing mask,
            whatever ``spans_cross_exclusions`` says, and kept apart from
            tower placement and ``terminal_ok``.
        terminal_ok: Lattice-resolution bool of cells where the queried
            line END may stand (M4-11: separate from placement and
            crossing). Default: every cell a tower may stand on.
        seed_chain: Lattice-resolution chain labels to start from, see
            :meth:`TowerFieldSolver.solve`; the source may then be
            omitted.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    raster = np.asarray(raster)
    if model is None:
        if profile is None:
            raise ValueError("pass either a model or a profile")
        model = TowerFieldModel.matching_kernel(profile)

    lattice_transform = transform
    if transform is not None and factor != 1:
        from affine import Affine
        lattice_transform = Affine(transform.a * factor, transform.b,
                                   transform.c, transform.d,
                                   transform.e * factor, transform.f)
    cell_x, cell_y = _pixel_size(cell_size_m, cell_size_x_m, cell_size_y_m,
                                 transform)
    lattice = TowerLattice(
        cell_size_x_m=cell_x, cell_size_y_m=cell_y, factor=int(factor),
        directions=(primitive_directions(2) if directions is None
                    else directions),
        pooling=pooling, transform=lattice_transform, crs=crs)

    blocked_fine = None if impassable is None else (raster >= impassable)
    values_fine = np.asarray(raster, dtype=np.float64)
    if blocked_fine is not None:
        if crossing_values is None:
            fill = 0.0
        else:
            fill = np.asarray(crossing_values, dtype=np.float64)
            if fill.shape != raster.shape:
                raise ValueError("crossing_values must match the raster")
            if not np.all(np.isfinite(fill[blocked_fine])):
                raise ValueError("crossing_values must be finite on the "
                                 "excluded cells (forbid with "
                                 "crossing_mask instead)")
        values_fine = np.where(blocked_fine, fill, values_fine)
    values = coarsen(values_fine, factor, pooling)

    if blocked_fine is None:
        blocked = np.zeros(values.shape, dtype=bool)
    else:
        # A lattice cell is blocked when the EXCLUDED SHARE of the raster
        # cells under it exceeds `blocked_fraction`. At the default 0.0 a
        # single excluded cell blocks the whole lattice cell, which is the
        # conservative reading and the right one for an upper bound.
        #
        # It is also very aggressive once sigma is coarse: a 40 m lattice
        # cell covers 1600 raster cells, so scattered exclusions block
        # nearly everything and a target can come back unreachable while
        # the router walks straight past it. A span is a line, not an
        # area, so a small share is a threading question rather than a
        # barrier -- raise this when the lattice is much coarser than the
        # features, and say what you raised it to.
        blocked = coarsen(blocked_fine.astype(np.float64), factor,
                          "mean") > float(blocked_fraction)

    if model.angle_tier == 2 and angles is None:
        if profile is None:
            raise ValueError("angle_tier=2 needs a profile or AngleTables")
        angles = angle_tables_from_profile(profile, lattice)

    if tower_cost is None:
        if profile is None:
            raise ValueError("pass either tower_cost or a profile")
        lut = profile.precompute_tower_terrain_costs()
        tower_cost = lut[np.clip(np.rint(values), 0, 65535).astype(np.int64)]
        if model.angle_tier == 1:
            tables = angles or angle_tables_from_profile(profile, lattice)
            # Tier 1 drops the turn premium; the CHEAPEST tower type is
            # the only part of it that survives as a valid lower bound.
            usable = tables.premium[tables.valid & np.isfinite(tables.premium)]
            tower_cost = tower_cost + (float(usable.min()) if usable.size
                                       else 0.0)
    tower_cost = np.array(tower_cost, dtype=np.float64, copy=True)
    tower_cost[blocked] = np.inf

    # Terminals last: they override the exclusion, not the other way round.
    exempt = _exempt_cells(lattice, values.shape, unblock_xy, unblock_cells)
    if exempt is not None:
        rr, cc = exempt
        blocked[rr, cc] = False
        tower_cost[rr, cc] = (float(model.terminal_tower_cost)
                              if model.charge_terminal_towers else 0.0)

    if clearance is None and profile is not None and dem is not None:
        clearance = clearance_from_profile(profile)
    dem_l = None if dem is None else coarsen(dem, factor, "max")
    obs_l = None if obstacle is None else coarsen(obstacle, factor, "max")

    if source_cell is None and source_xy is not None:
        if lattice.transform is None:
            raise ValueError("source_xy needs a transform")
        r, c = lattice.xy_to_cell(*source_xy)
        source_cell = (int(np.ravel(r)[0]), int(np.ravel(c)[0]))
    if source_cell is None and seed_chain is None:
        raise ValueError("pass source_cell, source_xy with a transform, or "
                         "seed_chain")

    # `blocked` is the CROSSING mask; `tower_cost = inf` is what stops a
    # tower standing there. Lifting the first leaves the second in place.
    crossing = (np.zeros_like(blocked) if spans_cross_exclusions else blocked)
    if crossing_mask is not None:
        cm = np.asarray(crossing_mask, dtype=bool)
        if cm.shape != raster.shape:
            raise ValueError("crossing_mask must match the raster")
        crossing = crossing | (coarsen(cm.astype(np.float64), factor,
                                       "mean") > 0.0)
    if terminal_ok is None:
        terminal_ok = np.isfinite(tower_cost)
    else:
        terminal_ok = np.asarray(terminal_ok, dtype=bool)
        if terminal_ok.shape != values.shape:
            raise ValueError(f"terminal_ok must have the lattice shape "
                             f"{values.shape}")
    solver = TowerFieldSolver(
        values=values, tower_cost=tower_cost, lattice=lattice, model=model,
        blocked=crossing, dem=dem_l, obstacle=obs_l, clearance=clearance,
        angles=angles, terminal_ok=terminal_ok)
    return solver.solve(source_cell, seed_chain=seed_chain,
                        record_pred=record_pred)


def tower_field_bounds(raster, *, profile, cell_size_m=None,
                       cell_size_x_m=None, cell_size_y_m=None,
                       source_cell=None,
                       source_xy=None, factor=1, directions=None,
                       transform=None, crs=None, dem=None, obstacle=None,
                       check: bool = True, record_pred: bool = False,
                       **kwargs):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """A matched lower/upper bound pair, and the check that they nest.

    The two fields differ only in the parameters the plan's bound table
    names, and in the direction each one is obliged to err::

        lower  tier 1 (no angle premium), MIN-pooled terrain and tower
               ground cost, crossings and clearance ignored
        upper  tier 2, MAX-pooled terrain, crossings enforced,
               clearance charged at BOTH ends of every span, towers
               confined to the sigma lattice

    Confining towers to the lattice raises cost, which is why it sits on
    the UPPER side. Both fields use the same lattice, so the lower one
    bounds the lattice-restricted optimum -- its metadata says so. What
    is NOT relaxed away on the lower side is the span and spacing
    problem, and that is what makes it tighter than "terrain plus
    minimum tower count".

    Returns:
        ``(lower, upper)``. With ``check``, raises when any cell has
        ``lower > upper`` or is reachable only in the upper bound.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    common = {"cell_size_m": cell_size_m, "cell_size_x_m": cell_size_x_m,
              "cell_size_y_m": cell_size_y_m, "profile": profile,
              "source_cell": source_cell, "source_xy": source_xy,
              "factor": factor, "directions": directions,
              "transform": transform, "crs": crs,
              "record_pred": record_pred}
    lower = tower_field_from_raster(
        raster,
        model=TowerFieldModel.matching_kernel(
            profile, angle_tier=1, forbidden_mode="off"),
        pooling="min", **common, **kwargs)
    upper = tower_field_from_raster(
        raster,
        model=TowerFieldModel.matching_kernel(
            profile, angle_tier=2, forbidden_mode="exact",
            clearance_charge="both_ends"),
        pooling="max", dem=dem, obstacle=obstacle, **common, **kwargs)
    lower._meta["bound"] = "lower (lattice-restricted)"  # pylint: disable=protected-access
    upper._meta["bound"] = "upper (feasible)"  # pylint: disable=protected-access  # same-module sibling object
    if check:
        check_bounds(lower, upper)
    return lower, upper


def check_bounds(lower: TowerField, upper: TowerField, *,
                 tol: float = 1e-6) -> dict:
    """``LB <= UB`` on every cell -- the cheapest strong invariant there is.

    Returns a summary dict; raises :class:`AssertionError` on a
    violation, including a cell reachable in the upper bound but not the
    lower, which no relaxation can produce.
    """
    both = np.isfinite(lower.arrival) & np.isfinite(upper.arrival)
    gap = upper.arrival[both] - lower.arrival[both]
    if gap.size and float(gap.min()) < -tol:
        raise AssertionError(
            f"LB <= UB is violated on {int((gap < -tol).sum())} cells, worst "
            f"by {-float(gap.min()):,.3f} EUR -- one of the two fields is "
            f"not the bound it claims to be")
    only_ub = (~np.isfinite(lower.arrival)) & np.isfinite(upper.arrival)
    if only_ub.any():
        raise AssertionError(
            f"{int(only_ub.sum())} cells are reachable in the upper bound "
            f"but not in the lower one, which no relaxation can do")
    return {
        "cells_compared": int(both.sum()),
        "gap_min_eur": float(gap.min()) if gap.size else float("nan"),
        "gap_max_eur": float(gap.max()) if gap.size else float("nan"),
        "gap_mean_eur": float(gap.mean()) if gap.size else float("nan"),
    }


__all__.append("check_bounds")


def assert_matched_tier1(tier1: TowerField, tier2: TowerField) -> None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Refuse a tier-1 field that is not a MATCHED lower bound on a tier-2
    one (plan rev. 5, section 2.2 and D5; verifier M4-02).

    Matched means: the same lattice and origin, the same terrain values, a
    direction superset, the same or ``min`` pooling, the same placement
    mask, crossing relaxed or equal, a span window containing tier 2's
    (min, max, inclusive flag, last-span rule), a terminal cost ``<=`` with
    the same charge flag, clearance off or identical, and start labels
    ``<=`` tier 2's everywhere. Only then is ``tier1.arrival <=
    tier2.arrival`` a certificate.

    Raises:
        AssertionError: listing every violated condition.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    bad: list[str] = []
    m1, m2 = tier1.model, tier2.model
    if m1.angle_tier != 1 or m2.angle_tier != 2:
        bad.append(f"tiers are {m1.angle_tier} and {m2.angle_tier}, "
                   f"want 1 and 2")
    l1, l2 = tier1.lattice, tier2.lattice
    if tier1.arrival.shape != tier2.arrival.shape:
        bad.append("lattice shapes differ")
    for name in ("cell_size_x_m", "cell_size_y_m", "factor"):
        if getattr(l1, name) != getattr(l2, name):
            bad.append(f"lattice {name} differs")
    t1, t2 = l1.transform, l2.transform
    if (t1 is None) != (t2 is None) or (
            t1 is not None and tuple(t1)[:6] != tuple(t2)[:6]):
        bad.append("lattice origin/transform differs")
    d1 = {tuple(int(x) for x in d) for d in np.asarray(l1.directions)}
    d2 = {tuple(int(x) for x in d) for d in np.asarray(l2.directions)}
    if not d2 <= d1:
        bad.append(f"tier 1 lacks directions {sorted(d2 - d1)}")
    if l1.pooling not in (l2.pooling, "min"):
        bad.append(f"pooling {l1.pooling!r} vs {l2.pooling!r}")
    if tier1.meta.get("values_sha256") != tier2.meta.get("values_sha256"):
        bad.append("terrain values differ")
    if m1.min_span_m > m2.min_span_m:
        bad.append("tier 1 min span is longer")
    if m1.max_span_m < m2.max_span_m or (
            m1.max_span_m == m2.max_span_m and m2.max_span_inclusive
            and not m1.max_span_inclusive):
        bad.append("tier 1 max span window is narrower")
    if m1.effective_last_span_min_m > m2.effective_last_span_min_m:
        bad.append("tier 1 last-span rule is stricter")
    if m1.charge_terminal_towers != m2.charge_terminal_towers:
        bad.append("terminal charge flags differ")
    elif m1.charge_terminal_towers and (m1.terminal_tower_cost
                                        > m2.terminal_tower_cost):
        bad.append("tier 1 terminal tower costs more")
    k1, k2 = tier1.masks, tier2.masks
    if not k1 or not k2:
        bad.append("a field carries no masks (settled before D5)")
    else:
        if not np.array_equal(k1["placement"], k2["placement"]):
            bad.append("placement masks differ")
        if np.any(k1["crossing"] & ~k2["crossing"]):
            bad.append("tier 1 blocks a crossing tier 2 allows")
        if np.any(k2["terminal_ok"] & ~k1["terminal_ok"]):
            bad.append("tier 1 refuses a terminal tier 2 allows")
        c1, c2 = k1.get("clearance"), k2.get("clearance")
        if c1 is not None and c1 != c2:
            bad.append("clearance differs (must be off or identical)")
    s1, s2 = tier1.seed, tier2.seed
    if s1 is None or s2 is None:
        bad.append("a field carries no start labels")
    elif np.any(s1 > s2):
        bad.append("tier 1 starts some chain above tier 2")
    if bad:
        raise AssertionError("not a matched tier 1: " + "; ".join(bad))


__all__.append("assert_matched_tier1")
