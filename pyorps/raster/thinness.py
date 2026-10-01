"""Sub-cell forbidden features: measure them, detect them, widen them.

GDAL burns a cell iff its CENTRE falls inside the polygon. For a forbidden
feature narrower than one cell that is a coin toss decided by sub-pixel
alignment, and it fails silently: a 0.8 m barrier on a 1 m grid burns 60 cells
when its centre sits at x=30.1 and ZERO cells at x=30.0, x=30.9 or x=31.0.
The returned route is genuinely optimal for the raster that was burned — the
raster just does not contain the barrier. Shifting the study-area origin by
10 cm flips it.

THE CONSTANT IS sqrt(2) * resolution, NOT resolution / 2
---------------------------------------------------------
A strip of width ``w`` at angle ``t`` is guaranteed to cover a pixel centre at
every offset only once ``w >= resolution * (|cos t| + |sin t|)``, which peaks
at ``sqrt(2) * resolution`` for a 45-degree strip. A measured sweep (23 angles
x 24 sub-pixel offsets, width 0.30..2.00 by 0.01) reproduces that curve
exactly: 1.00 res at 0/90 degrees, 1.37 at 30, 1.42 at 45.

The reason is geometric, not empirical. A ``res x res`` axis-aligned square
placed anywhere always contains at least one pixel centre, and as it slides
along a curve the lattice points it holds form a contiguous block that never
empties — so the union of the covered cells is non-empty AND 4-connected. The
smallest disk containing that square has radius ``res / sqrt(2)``. A feature
whose minimum width is at least ``sqrt(2) * res`` therefore inscribes such a
disk at every one of its points, and its burn cannot vanish or fragment.

``res / 2`` is NOT enough: the sweep flags 30-degree and 45-degree slivers as
still-vanishing after a ``+res/2`` buffer at every sub-pixel offset.

WHY 4-CONNECTIVITY IS THE ACCEPTANCE CRITERION
----------------------------------------------
A barrier burned as a diagonal staircase is 8-connected but not 4-connected:
its cells touch only at corners. pyorps' own routers do not slip through such
a gap (a diagonal step is rejected if EITHER flanking cell is impassable — see
``tests/test_raster/test_barrier_connectivity.py`` and
``pyorps/utils/_raster_context.pyx``), but a 4-connected burn is the property
that makes a barrier impermeable to ANY 8-connected walker, and it coincides
exactly with "does not vanish" at every angle measured. So one predicate
covers both failure modes.

4-connectivity alone is not sufficient as a per-feature acceptance test,
though: an L-shaped feature whose fat body burns as one component stays
4-connected while its 0.30 m arm silently disappears. That is why
:func:`is_thin` is a per-POINT test (an opening residual) rather than a
whole-feature one, and why detection asks "is the burned footprint a faithful
representation of this geometry" (:func:`thin_parts` +
``partially_vanished``) rather than the much weaker "did this FEATURE burn
any cells at all", which such an L passes while carrying a hole.

WHY NEITHER REPAIR IS THE DEFAULT
---------------------------------
At a fixed resolution you cannot simultaneously guarantee that a sub-cell
BARRIER is represented and that a sub-cell GAP is preserved. They are the same
geometry seen from opposite sides, so any repair that fattens barriers closes
gaps: the information simply is not in the raster.

HOW MUCH each repair closes is a property of the fixture, not a constant of the
repair, so it is quoted here as a range over four parameterizations of one
sweep rather than as a single percentage. Every row is reproducible from
``tests/test_raster/sweep_forbidden_repair_tradeoff.py``: a 64 x 64 m extent at
1 m cells, a forbidden wall of the stated thickness spanning the extent at
x = 30 m + offset and carrying one central gate, 8 gate widths x 8 sub-pixel
offsets (0.000 .. 0.875 m in steps of 0.125 m), counting only the
configurations in which the PLAIN rule leaves the left edge 4-connected to the
right edge (identical counts under 8-connectivity):

    gate widths        wall 0.4 m (thin, C widens it)  wall 2.0 m (fat, C skips)
    0.4..1.8 / 0.2     plain 0/52 | C 52/52 | D 52/52  plain 0/32 | C 0/32 | D 32/32
    0.6..2.7 / 0.3     plain 0/58 | C 34/58 | D 34/58  plain 0/48 | C 0/48 | D 24/48
    0.5..4.0 / 0.5     plain 0/58 | C 22/58 | D 18/58  plain 0/48 | C 0/48 | D  8/48
    0.25..2.0 / 0.25   plain 0/52 | C 48/52 | D 44/52  plain 0/32 | C 0/32 | D 24/32

Only what reproduces across ALL of them may be relied on:

* the plain rule and the default seal NOTHING — 0 of N in every row, by
  construction, because the default does not move a cell;
* on a FAT wall C seals NOTHING (0 of N everywhere) while D seals a large
  fraction (17-100 % here). That is the sharp, robust difference between them;
* on a THIN wall BOTH seal a substantial fraction (C 38-100 %, D 31-100 %) and
  the two are within a few configurations of each other, in either order;
* therefore neither repair is safe as a default.

The bare percentages are NOT: the same option C on the same 0.4 m wall measures
38 % or 100 % depending only on which gate widths the sweep happens to contain,
and an earlier sweep whose gate widths were never recorded reported 52 %. A
default may not silently turn a solvable routing problem into
``NoPathFoundError``, so BOTH repairs are opt-in and the default only DETECTS
(:func:`report_forbidden_burn_defects`, which does not change a single cell).
The only fix that is correct rather than a trade is a finer cell size, and
:func:`suggest_resolution` computes it.

Because a repair is a trade, detection must be able to see the harm the repair
CAUSES, not only the harm it removes: :func:`detect_repair_seals` compares the
4-connectivity of FREE space before and after the repair and reports every free
region the repair split in two.
"""
import math
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine

#: Minimum width, in cells, at which a forbidden feature's burn is guaranteed
#: to be non-empty and 4-connected at EVERY sub-pixel alignment. See the module
#: docstring for the derivation and the measured sweep that confirms it.
MIN_FORBIDDEN_WIDTH_CELLS: float = math.sqrt(2.0)

#: Opening residual, in cells, below which a feature counts as fat. The
#: threshold is not delicate: with mitre joins a square, a 256-point circle, a
#: fat L and an annulus all score EXACTLY 0.0, while a 0.9 x 0.9 speck scores
#: 0.81 — any tolerance in (0, 0.8] behaves identically.
DEFAULT_TOLERANCE_CELLS: float = 0.5

#: Mitre limit for the opening. This is the real policy knob: it decides how
#: sharp a CORNER counts as thin, bevelling once ``1/sin(apex/2) > limit``.
#: 2.0 flags corners sharper than 60 degrees, 5.0 (shapely's default) sharper
#: than 23.1 degrees, 20.0 sharper than 5.7 degrees. 5.0 is kept because a
#: wedge sharper than 23 degrees genuinely IS sub-cell near its tip. Round
#: joins are NOT usable here: their sharp-corner artefact scores 0.22 cells for
#: any 90-degree corner, which collides with the tolerance.
DEFAULT_MITRE_LIMIT: float = 5.0

#: Rows per block in the burned-id scans. Blocking keeps peak memory at one
#: block rather than one full boolean copy of the raster.
_SCAN_BLOCK_ROWS: int = 2048

#: 4-connectivity structuring element for :func:`scipy.ndimage.label`.
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

#: Extra widening margin, in cells, on top of the exact safe width. A strip of
#: width EXACTLY ``sqrt(2)*res`` erodes to a degenerate line, so :func:`is_thin`
#: reads its own repaired output as still-thin: measured, 125 of 400 widened
#: features at 1 m and 343 of 400 at 5 m. That is a cry-wolf signal on data the
#: default has already fixed, and it costs a needless per-feature re-burn in
#: detection. 1e-4 cells (0.1 mm at a 1 m grid) clears it for 400/400 at every
#: resolution tested, while being some four orders of magnitude below the
#: smallest distance a cell centre can resolve.
_WIDEN_MARGIN_CELLS: float = 1e-4

#: Neutral element for :func:`shapely.difference` when an opening died.
_EMPTY_POLYGON = shapely.from_wkt("POLYGON EMPTY")

#: Smallest thin PART, as a fraction of a cell's area, that is still worth
#: checking. Mitre joins leave numerical dust along concave corners of an
#: otherwise fat feature; a genuine sub-cell arm is orders of magnitude
#: larger (a 0.4 x 2 m arm is 0.8 of a 1 m cell). Anything below this cannot
#: hole a barrier even if it is missing entirely.
_MIN_THIN_PART_CELL_AREA: float = 0.01

#: Cells per chunk when measuring how much of a thin part lies in free cells.
#: The measurement only has to reach a threshold, so it is chunked and stops
#: early; a long fence's footprint can otherwise run to thousands of cells.
_GAP_CELL_CHUNK: int = 256


class ThinForbiddenFeatureWarning(UserWarning):
    """A forbidden feature did not survive the burn intact.

    Its own category so callers can promote it to an error with
    ``warnings.simplefilter('error', ThinForbiddenFeatureWarning)`` without
    catching every other UserWarning pyorps emits.
    """


class ForbiddenBurnRoutingAssessment(UserWarning):
    """Routing suitability verdict after a forbidden-feature burn audit.

    Emitted when the burn is intact (no hard defects). Distinct from
    :class:`ThinForbiddenFeatureWarning` so callers can treat a clean audit as
    informational without conflating it with a broken barrier.
    """


class ThinForbiddenFeatureError(ValueError):
    """Raised instead of warning when ``on_thin_features='raise'``."""


class SealedOpeningWarning(UserWarning):
    """A forbidden-feature repair disconnected free space that was connected.

    Distinct from :class:`ThinForbiddenFeatureWarning` on purpose: that one
    says the raster is MISSING a barrier the data has, this one says the raster
    has INVENTED a barrier the data does not have. They call for opposite
    reactions, so they must be filterable apart.
    """


class SealedOpeningError(ValueError):
    """Raised instead of warning when ``on_thin_features='raise'``."""


# ----------------------------------------------------------------------
# The thin predicate
# ----------------------------------------------------------------------

def safe_forbidden_width_m(resolution_in_m: float) -> float:
    """Width at or above which a forbidden feature always burns intact."""
    return MIN_FORBIDDEN_WIDTH_CELLS * float(resolution_in_m)


def recommended_geometry_buffer_m(resolution_in_m: float) -> float:
    """Outward buffer (m) that clears most cadastral burn defects at this cell size.

    Measured on Hessen ALKIS windows at 1 m cells: ``geometry_buffer_m`` of
    one cell (``resolution_in_m``) removes vanished and holed forbidden
    features; half a cell (``resolution_in_m / 2``) is enough for the
    ``small_raster.tiff`` extent. The batch example workflow uses 1 m for the
    same reason — sub-cell parcel boundaries need a small outward buffer before
    the burn, not a finer grid.
    """
    return float(resolution_in_m)


def _as_geometry_array(geometries) -> np.ndarray:
    """Normalise any geometry container to a 1-D object ndarray."""
    if hasattr(geometries, "to_numpy"):  # GeoSeries / Series
        geometries = geometries.to_numpy()
    array = np.asarray(geometries, dtype=object)
    return np.atleast_1d(array)


def _opening(geometries: np.ndarray, radius: float, mitre_limit: float):
    """Morphological opening (erode then dilate) and where it emptied out.

    The opening removes exactly the parts of the shape that no disk of that
    radius fits inside, which is what makes every measure derived from it a
    per-POINT one: an L keeps its fat body and loses only the thin arm,
    whereas every global measure (negative buffer emptiness, 2*area/perimeter,
    oriented-envelope short side) calls that same L fat. Measured on 15
    shapes, the global measures scored 10-11/15; the two-tier test built on
    this scores 15/15.
    """
    eroded = shapely.buffer(geometries, -radius, join_style="mitre",
                            mitre_limit=mitre_limit)
    vanished = shapely.is_missing(eroded) | shapely.is_empty(eroded)
    opened = shapely.buffer(eroded, radius, join_style="mitre",
                            mitre_limit=mitre_limit)
    return vanished, opened


def _opening_residual(geometries: np.ndarray, radius: float,
                      mitre_limit: float):
    """Area lost to a morphological opening, and where the erosion vanished."""
    vanished, opened = _opening(geometries, radius, mitre_limit)
    residual = shapely.area(geometries) - np.nan_to_num(shapely.area(opened))
    return vanished, residual


def thin_parts(
        geometries,
        resolution_in_m: float,
        *,
        mitre_limit: float = DEFAULT_MITRE_LIMIT,
) -> np.ndarray:
    """The sub-cell PARTS of each feature: geometry minus its opening.

    ``is_thin`` answers "is this feature thin somewhere"; this answers
    "WHERE". The distinction is the whole point of the partial-vanish check:
    a forbidden feature made of a 2 m base and a 0.4 m arm burns cells (so it
    is not vanished) and burns them as ONE 4-connected block (so it is not
    fragmented), yet the arm is missing and the barrier has a hole exactly
    the length of that arm. The arm is what this function returns.

    Where the opening is empty the whole feature is thin, so the difference
    degenerates to the feature itself - which is correct, not a special case.

    Parameters:
        geometries: Sequence, GeoSeries or object array of shapely geometries.
        resolution_in_m: Cell size the features will be burned at.
        mitre_limit: How sharp a corner counts as thin; see the constant.

    Returns:
        An object ndarray of the same length; entries are empty geometries
        for features that are fat everywhere.
    """
    geometries = _as_geometry_array(geometries)
    if geometries.size == 0:
        return geometries
    _, opened = _opening(geometries,
                         float(resolution_in_m) / MIN_FORBIDDEN_WIDTH_CELLS,
                         mitre_limit)
    # shapely.difference propagates None; an empty polygon is the neutral
    # element that makes "the erosion died" mean "all of it is thin".
    opened = np.where(shapely.is_missing(opened), _EMPTY_POLYGON, opened)
    return np.asarray(shapely.difference(geometries, opened), dtype=object)


def is_thin(
        geometries,
        resolution_in_m: float,
        *,
        tolerance_cells: float = DEFAULT_TOLERANCE_CELLS,
        mitre_limit: float = DEFAULT_MITRE_LIMIT,
) -> np.ndarray:
    """Which features are narrower than ``sqrt(2) * resolution`` SOMEWHERE.

    Two tiers: the erosion vanished entirely (the feature is thin everywhere),
    or the opening lost more than ``tolerance_cells`` of a cell's area (the
    feature is thin somewhere). Recall against a burned 6000x6000 raster with
    5000 forbidden features was 1003/1003 defective features at this erosion
    radius — 100 %, which is a theorem, not a fluke (module docstring). It
    over-flags 165 features that happened to burn cleanly at the alignment
    tested; those are precisely the features a 10 cm origin shift would break.

    "Narrow" means INSCRIBED width, which is the property the burn depends on:
    a 0.4 m tab flush against a 30 m block is covered by a disk centred inside
    the block, burns intact, and is correctly NOT flagged. Only a part that no
    disk of radius ``res/sqrt(2)`` can reach counts.

    Cost: ~10 us per simple feature (52 ms for 5000). Affordable on a
    FORBIDDEN SUBSET, NOT as a pass over a 300k-feature land-use layer, where
    the same test measures 11.6 s — the cost scales with vertex count.

    Parameters:
        geometries: Sequence, GeoSeries or object array of shapely geometries.
        resolution_in_m: Cell size the features will be burned at.
        tolerance_cells: Opening residual, in cell areas, still counted as fat.
        mitre_limit: How sharp a corner counts as thin; see the constant.

    Returns:
        Boolean array, one entry per geometry.
    """
    geometries = _as_geometry_array(geometries)
    if geometries.size == 0:
        return np.zeros(0, dtype=bool)
    resolution_in_m = float(resolution_in_m)
    vanished, residual = _opening_residual(
        geometries, resolution_in_m / MIN_FORBIDDEN_WIDTH_CELLS, mitre_limit)
    return np.asarray(
        vanished | (residual > tolerance_cells * resolution_in_m ** 2))


def min_feature_width(
        geometry,
        resolution_in_m: float,
        *,
        iterations: int = 16,
        tolerance_cells: float = DEFAULT_TOLERANCE_CELLS,
        mitre_limit: float = DEFAULT_MITRE_LIMIT,
) -> float | None:
    """Narrowest width of one feature, or None if it is at least the safe width.

    This is a GATE, not a caliper: the bisection only searches below
    ``sqrt(2) * resolution_in_m``, because above that width the burn is
    guaranteed intact and the exact number would be meaningless to the caller.
    ``None`` therefore means "at least ``safe_forbidden_width_m()``", never
    "unknown".

    Cost: ~0.3 ms per feature (16 opening evaluations).

    Parameters:
        geometry: A single shapely geometry.
        resolution_in_m: Cell size the feature will be burned at.
        iterations: Bisection steps; 16 resolves the width to ~1/65000 of the
            search interval, far below anything a planner would act on.
        tolerance_cells: Opening residual, in cell areas, still counted as fat.
        mitre_limit: How sharp a corner counts as thin; see the constant.

    Returns:
        The narrowest width in metres, or ``None`` when the feature is already
        at least ``safe_forbidden_width_m(resolution_in_m)`` wide.
    """
    cap = safe_forbidden_width_m(resolution_in_m)
    geometry = np.asarray([geometry], dtype=object)

    def thin_at(radius: float) -> bool:
        vanished, residual = _opening_residual(geometry, radius, mitre_limit)
        return bool(vanished[0]
                    or residual[0] > tolerance_cells * resolution_in_m ** 2)

    # hi is always a radius at which the shape reads thin, lo one at which it
    # does not; radius 0 is an identity opening, so lo = 0 starts valid.
    if not thin_at(cap / 2.0):
        return None
    lo, hi = 0.0, cap / 2.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if thin_at(mid):
            hi = mid
        else:
            lo = mid
    return 2.0 * hi


def widen_thin_features(
        geometries,
        resolution_in_m: float,
        *,
        thin: np.ndarray | None = None,
        tolerance_cells: float = DEFAULT_TOLERANCE_CELLS,
        mitre_limit: float = DEFAULT_MITRE_LIMIT,
) -> np.ndarray:
    """Buffer every thin feature up to the safe width (option C).

    Each flagged feature is buffered by ``(sqrt(2)*res - w) / 2`` with ``w`` its
    measured minimum width, so the result is exactly ``sqrt(2)*res`` wide where
    it was thinnest and no wider. That tight rule and the blunt
    ``+res/sqrt(2)`` fallback both burned non-empty and 4-connected for 14
    shapes x 11 sub-pixel origin shifts x 3 resolutions; ``+res/2`` and
    ``+0.6*res`` did not (0/11 for 30- and 45-degree slivers).

    OPT-IN, NOT THE DEFAULT. This is the better of the two repairs — it only
    ever touches a feature that is ACTUALLY thinner than a cell, where the
    blunt all-touched overlay (option D) fattens fat forbidden features too —
    but "better" is not "safe". What the sweep in the module docstring shows is
    that "better" means EXACTLY ONE thing, and only on FAT features: on a 2.0 m
    wall this option sealed 0 of the gates the plain rule leaves passable under
    every parameterization tried, where the all-touched overlay sealed 17-100 %
    of them. On a THIN (0.4 m) wall - the only case where this option acts at
    all - it sealed 38-100 % of them, which is the same band as the overlay's
    31-100 %; the two are not distinguishable there. A widened barrier and a
    closed gate are the same event seen from the two sides of the same sub-cell
    geometry; see the module docstring for the exact gate widths and offsets.

    CAUTION — this changes geometry, not just the raster. Two forbidden
    features less than ``sqrt(2)*res`` apart MERGE, swallowing any corridor
    between them, and a widened feature grows by up to ``res/sqrt(2)`` on each
    side. The difference to the overlay is that only features that are already
    sub-cell can do this, and a sub-cell forbidden feature is exactly the case
    where the raster cannot represent the gap either way.

    Non-thin features are returned unchanged (identity, not a re-buffer).

    Parameters:
        geometries: Sequence, GeoSeries or object array of shapely geometries.
            Pass the FORBIDDEN subset only — widening an ordinary land-use
            class would move its boundary for no reason.
        resolution_in_m: Cell size the features will be burned at.
        thin: Precomputed :func:`is_thin` mask, to avoid measuring twice.
        tolerance_cells: Opening residual, in cell areas, still counted as fat.
        mitre_limit: How sharp a corner counts as thin; see the constant.

    Returns:
        An object ndarray of the same length, with the thin entries buffered.
    """
    geometries = _as_geometry_array(geometries)
    if geometries.size == 0:
        return geometries
    if thin is None:
        thin = is_thin(geometries, resolution_in_m,
                       tolerance_cells=tolerance_cells,
                       mitre_limit=mitre_limit)
    target = safe_forbidden_width_m(resolution_in_m)
    # See _WIDEN_MARGIN_CELLS: exactly the safe width still reads thin.
    margin = _WIDEN_MARGIN_CELLS * float(resolution_in_m)
    widened = geometries.copy()
    for index in np.flatnonzero(np.asarray(thin, dtype=bool)):
        width = min_feature_width(geometries[index], resolution_in_m,
                                  tolerance_cells=tolerance_cells,
                                  mitre_limit=mitre_limit)
        # width is None only for a caller-supplied `thin` that disagrees with
        # the predicate; the blunt fallback is safe at every angle.
        distance = target / 2.0 if width is None else max(
            0.0, (target - width) / 2.0)
        distance += margin
        if distance > 0.0:
            widened[index] = shapely.buffer(geometries[index], distance)
    return widened


# ----------------------------------------------------------------------
# Option E — resolution guidance
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ResolutionAdvice:
    """What cell size would let the forbidden features survive the burn.

    ``safe_resolution_in_m`` and ``cells_at_safe_resolution`` must always be
    read together. Real data makes the point: a layer whose narrowest forbidden
    feature is 5 cm needs a 3.6 cm cell, i.e. 2.8e10 cells over a 6 km extent.
    Resolution alone usually cannot fix this — widening (option C) can.
    """
    resolution_in_m: float
    n_forbidden: int
    n_thinner_than_safe_width: int
    n_thinner_than_a_cell: int
    narrowest_width_m: float | None = None
    narrowest_index: int | None = None
    safe_resolution_in_m: float | None = None
    cells_at_safe_resolution: float | None = None

    def summary(self) -> str:
        """One actionable sentence, or '' when nothing needs saying."""
        if self.narrowest_width_m is None:
            return ""
        parts = [
            f"the narrowest forbidden feature is {self.narrowest_width_m:.3g} m " +
            f"wide (index {self.narrowest_index}), but a feature must be at " +
            f"least {safe_forbidden_width_m(self.resolution_in_m):.3g} m " +
            f"(sqrt(2) cells) wide to survive a " +
            f"{self.resolution_in_m:.3g} m burn at every alignment"
        ]
        if self.safe_resolution_in_m is not None:
            cells = ""
            if self.cells_at_safe_resolution is not None:
                cells = (f", which is {self.cells_at_safe_resolution:.3g} "
                         f"cells over the same extent")
            parts.append(
                f"resolving it needs a cell size of "
                f"{self.safe_resolution_in_m:.3g} m{cells}")
        return "; ".join(parts)


def suggest_resolution(
        forbidden_geometries,
        resolution_in_m: float,
        bounds: tuple[float, float, float, float] | None = None,
        *,
        thin: np.ndarray | None = None,
        tolerance_cells: float = DEFAULT_TOLERANCE_CELLS,
        mitre_limit: float = DEFAULT_MITRE_LIMIT,
) -> ResolutionAdvice:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Option E: report the narrowest forbidden feature and the cell size for it.

    Only the flagged (thin) features are measured, at ~0.3 ms each; fat ones
    are known to be at least ``sqrt(2) * resolution_in_m`` wide and cannot be
    the narrowest.

    Parameters:
        forbidden_geometries: Geometries whose burned value is impassable.
        resolution_in_m: The cell size in use.
        bounds: ``(minx, miny, maxx, maxy)`` of the raster, used only to turn
            the suggested resolution into a cell count.
        thin: Precomputed :func:`is_thin` mask, to avoid measuring twice —
            option A already needs it to find partially vanished features.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    geometries = _as_geometry_array(forbidden_geometries)
    n_forbidden = int(geometries.size)
    if n_forbidden == 0:
        return ResolutionAdvice(float(resolution_in_m), 0, 0, 0)

    if thin is None:
        thin = is_thin(geometries, resolution_in_m,
                       tolerance_cells=tolerance_cells,
                       mitre_limit=mitre_limit)
    flagged = np.flatnonzero(np.asarray(thin, dtype=bool))
    if flagged.size == 0:
        return ResolutionAdvice(float(resolution_in_m), n_forbidden, 0, 0)

    widths = {}
    for index in flagged:
        width = min_feature_width(geometries[index], resolution_in_m,
                                  tolerance_cells=tolerance_cells,
                                  mitre_limit=mitre_limit)
        if width is not None:
            widths[int(index)] = width
    if not widths:
        return ResolutionAdvice(float(resolution_in_m), n_forbidden,
                                int(flagged.size), 0)

    narrowest_index = min(widths, key=widths.get)
    narrowest = widths[narrowest_index]
    safe_resolution = narrowest / MIN_FORBIDDEN_WIDTH_CELLS
    cells = None
    if bounds is not None and safe_resolution > 0.0:
        minx, miny, maxx, maxy = bounds
        cells = ((maxx - minx) / safe_resolution) * \
                ((maxy - miny) / safe_resolution)
    return ResolutionAdvice(
        resolution_in_m=float(resolution_in_m),
        n_forbidden=n_forbidden,
        n_thinner_than_safe_width=int(flagged.size),
        n_thinner_than_a_cell=int(sum(1 for w in widths.values()
                                      if w < resolution_in_m)),
        narrowest_width_m=narrowest,
        narrowest_index=narrowest_index,
        safe_resolution_in_m=safe_resolution,
        cells_at_safe_resolution=cells,
    )


# ----------------------------------------------------------------------
# Option A — detection after the burn
# ----------------------------------------------------------------------

class ForbiddenBurnSeverity(str, Enum):
    """How badly the forbidden burn affects route-planning reliability."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ForbiddenBurnAssessment:
    """Severity verdict and practical guidance for route planning."""

    severity: ForbiddenBurnSeverity
    routing_suitable: bool
    headline: str
    takeaways: tuple[str, ...]


@dataclass(frozen=True)
class ForbiddenBurnReport:
    """What the burn did to the forbidden features.

    Every index array here indexes into the forbidden SEQUENCE that was handed
    to :func:`detect_forbidden_burn_defects`, not into the full layer.

    Three of the four are DEFECTS and clear ``ok``: the feature burned no
    cells (``vanished``), a sub-cell PART of it burned no cells while the rest
    did (``partially_vanished``), or its cells touch only at corners
    (``fragmented``). ``at_risk`` is not a defect: those features burned
    acceptably at THIS sub-pixel alignment but are narrower than
    ``sqrt(2)`` cells somewhere, so a 10 cm shift of the study-area origin
    would break them. Keeping it out of ``ok`` is deliberate — a detector that
    fires on 165 features that burned fine gets switched off, and then it
    protects nobody.
    """
    n_forbidden: int
    resolution_in_m: float
    vanished: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    fragmented: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    partially_vanished: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    at_risk: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    fragmentation_checked: bool = True
    advice: ResolutionAdvice | None = None

    @property
    def ok(self) -> bool:
        """True when every forbidden feature burned intact."""
        return (self.vanished.size == 0 and self.fragmented.size == 0
                and self.partially_vanished.size == 0)

    def assess(self) -> ForbiddenBurnAssessment:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Evaluate routing suitability from defect counts."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        n_v = int(self.vanished.size)
        n_pv = int(self.partially_vanished.size)
        n_f = int(self.fragmented.size)
        n_r = int(self.at_risk.size)
        safe_w = safe_forbidden_width_m(self.resolution_in_m)

        if n_v or n_pv:
            takeaways = []
            if n_v:
                takeaways.append(
                    f"{n_v} forbidden feature(s) burned zero cells — routes can "
                    f"cross land your vector data marks as excluded.")
            if n_pv:
                takeaways.append(
                    f"{n_pv} forbidden feature(s) have sub-cell holes — part of "
                    f"the geometry is missing from the cost surface.")
            takeaways.append(
                "Do not rely on this raster for compliance-critical exclusion "
                "routing until you use a finer cell size or repair the geometry.")
            buf = recommended_geometry_buffer_m(self.resolution_in_m)
            takeaways.append(
                f"Cadastral land-use polygons often need a small outward buffer "
                f"before rasterization: try geometry_buffer_m={buf:.3g} m "
                f"(one cell) or {buf / 2:.3g} m (half a cell) on ALKIS-style "
                f"data before raising the cell size.")
            if self.advice is not None and self.advice.safe_resolution_in_m:
                takeaways.append(
                    f"Suggested cell size from the narrowest feature: "
                    f"{self.advice.safe_resolution_in_m:.3g} m.")
            return ForbiddenBurnAssessment(
                severity=ForbiddenBurnSeverity.CRITICAL,
                routing_suitable=False,
                headline="NOT RELIABLE for route planning at this cell size",
                takeaways=tuple(takeaways),
            )

        if n_f:
            takeaways = (
                f"{n_f} forbidden feature(s) burned as corner-touching cell "
                f"chains (8-connected but not 4-connected).",
                "pyorps routers reject diagonal steps when either flanking "
                "cell is impassable, so routes are unlikely to slip through.",
                "The raster still misrepresents continuous barriers — treat "
                "route costs near these features as approximate.",
            )
            if not self.fragmentation_checked:
                takeaways = takeaways + (
                    "Corner-touching barriers were not checked (scipy missing) — "
                    "some fragmentation may be undetected.",
                )
            return ForbiddenBurnAssessment(
                severity=ForbiddenBurnSeverity.MODERATE,
                routing_suitable=True,
                headline="USABLE WITH CAUTION for route planning",
                takeaways=takeaways,
            )

        if n_r:
            takeaways = (
                f"{n_r} forbidden feature(s) are narrower than {safe_w:.3g} m "
                f"somewhere but burned intact at this sub-pixel alignment.",
                "Shifting the study-area origin by roughly 10 cm can flip "
                "whether those features appear on the cost surface.",
                "Fine for exploratory routing; use a finer cell size or lock "
                "the raster origin if the project boundary is not final.",
            )
            if self.advice is not None and self.advice.safe_resolution_in_m:
                takeaways = takeaways + (
                    f"Suggested cell size from the narrowest feature: "
                    f"{self.advice.safe_resolution_in_m:.3g} m.",
                )
            return ForbiddenBurnAssessment(
                severity=ForbiddenBurnSeverity.LOW,
                routing_suitable=True,
                headline="SUITABLE for route planning (alignment-sensitive)",
                takeaways=takeaways,
            )

        takeaways = (
            f"All {self.n_forbidden} forbidden feature(s) burned intact at "
            f"{self.resolution_in_m:.3g} m cells.",
            "The cost surface faithfully represents your exclusion zones for "
            "least-cost routing.",
        )
        if not self.fragmentation_checked:
            takeaways = takeaways + (
                "Corner-touching fragmentation was not checked (scipy missing).",
            )
        return ForbiddenBurnAssessment(
            severity=ForbiddenBurnSeverity.NONE,
            routing_suitable=True,
            headline="SUITABLE for route planning",
            takeaways=takeaways,
        )

    def assessment_message(self) -> str:
        """Multi-line severity verdict and practical takeaways for the user."""
        verdict = self.assess()
        lines = [
            "Forbidden-feature burn audit " +
            f"({self.resolution_in_m:.3g} m cells, " +
            f"{self.n_forbidden} forbidden feature(s))",
            f"Severity: {verdict.severity.value.upper()} | " +
            f"Route planning: {verdict.headline}",
            "",
            "Practical takeaways:",
        ]
        lines.extend(f"  • {item}" for item in verdict.takeaways)
        return "\n".join(lines)

    def print_routing_assessment(self, *, file=None) -> None:
        """Print :meth:`assessment_message` (defaults to stderr)."""
        if file is None:
            file = sys.stderr
        print(self.assessment_message(), file=file)

    def summary(self) -> str:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Aggregate counts plus one example — never one line per feature.

        1168 warnings on a 5000-feature layer would train the reader to ignore
        the mechanism, so the message names counts and a single index.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if self.ok:
            return ""
        parts = []
        if self.vanished.size:
            parts.append(
                f"{self.vanished.size} of {self.n_forbidden} forbidden "
                f"features burned NO cells at all (e.g. index "
                f"{int(self.vanished[0])}) - they are absent from the cost "
                f"surface and cannot block a route")
        if self.partially_vanished.size:
            parts.append(
                f"{self.partially_vanished.size} of {self.n_forbidden} "
                f"forbidden features burned only in part (e.g. index "
                f"{int(self.partially_vanished[0])}): a sub-cell portion of "
                f"the geometry burned no cells at all, so the barrier has a "
                f"hole exactly where that portion runs")
        if self.fragmented.size:
            parts.append(
                f"{self.fragmented.size} of {self.n_forbidden} forbidden "
                f"features burned into cells that touch only at their corners "
                f"(e.g. index {int(self.fragmented[0])}) - such a barrier is "
                f"8- but not 4-connected")
        if self.at_risk.size:
            parts.append(
                f"{self.at_risk.size} of {self.n_forbidden} forbidden "
                f"features are narrower than "
                f"{safe_forbidden_width_m(self.resolution_in_m):.3g} m "
                f"somewhere and would break under a different sub-pixel "
                f"alignment")
        if not self.fragmentation_checked:
            parts.append("corner-touching barriers were NOT checked (scipy is "
                         "not installed)")
        message = "; ".join(parts)
        if self.advice is not None and self.advice.summary():
            message += ". " + self.advice.summary()
        return message


def _burn_forbidden_id_band(geometries, out_shape, transform,
                            all_touched: bool = False) -> np.ndarray:
    """Burn the forbidden subset ALONE as 1-based ids, background 0.

    Detection must NOT run against the cost raster: ``fill_value`` defaults to
    ``IMPASSABLE_CELL_COST`` too, so ``raster == IMPASSABLE_CELL_COST`` is
    "forbidden feature OR outside every feature" and a scan of it is dominated
    by the background boundary (measured: 48478 corner-touching cells, 3070 of
    5000 features flagged, essentially all spurious).
    """
    n = len(geometries)
    dtype = "uint16" if n <= np.iinfo(np.uint16).max else "uint32"
    if n > np.iinfo(np.uint32).max:
        raise ValueError(f"{n} forbidden features exceed the id band capacity")
    return rasterize(
        ((geom, index) for index, geom in enumerate(geometries, start=1)),
        out_shape=out_shape,
        fill=0,
        dtype=dtype,
        transform=transform,
        all_touched=all_touched,
    )


def _burn_coverage(geometries, out_shape: tuple[int, int], transform: Affine,
                   all_touched: bool = False) -> np.ndarray:
    """Boolean mask of the cells ``geometries`` cover under the burn rule.

    Values are irrelevant here — only WHICH cells a set of features reaches —
    so this burns a 1-byte hit band rather than the cost band.
    """
    geometries = _as_geometry_array(geometries)
    if geometries.size == 0:
        return np.zeros(out_shape, dtype=bool)
    return rasterize(
        ((geom, 1) for geom in geometries),
        out_shape=out_shape,
        fill=0,
        dtype="uint8",
        transform=transform,
        all_touched=all_touched,
    ).astype(bool)


def _feature_window(
        geometry, transform: Affine, out_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Clipped ``(row0, row1, col0, col1)`` a geometry's bbox can touch.

    Per-feature re-burns below must not cost a full-raster pass each, so they
    burn into a window around the feature's own bounds. The window transform
    keeps the SAME cell centres, so GDAL's scanline decision is bit-identical
    to what the full-extent burn made; only the cells that cannot possibly be
    touched are left out. One cell of margin absorbs the all-touched rim.

    ``None`` means the feature lies wholly outside the raster. That is not a
    thinness defect and must not be reported as one: nothing about the cell
    size would change it, and under a caller-supplied ``bounding_box`` it is
    routine.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    minx, miny, maxx, maxy = (float(v) for v in shapely.bounds(geometry))
    if not all(map(math.isfinite, (minx, miny, maxx, maxy))):
        return None
    inverse = ~transform
    corners = [inverse * (x, y) for x in (minx, maxx) for y in (miny, maxy)]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    row0 = max(0, int(math.floor(min(rows))) - 1)
    row1 = min(out_shape[0], int(math.ceil(max(rows))) + 1)
    col0 = max(0, int(math.floor(min(cols))) - 1)
    col1 = min(out_shape[1], int(math.ceil(max(cols))) + 1)
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, row1, col0, col1


def _burn_alone(geometry, window: tuple[int, int, int, int],
                transform: Affine, all_touched: bool) -> np.ndarray:
    """Burn ONE geometry into ``window``, as a boolean hit mask."""
    row0, row1, col0, col1 = window
    return rasterize(
        [(geometry, 1)],
        out_shape=(row1 - row0, col1 - col0),
        fill=0,
        dtype="uint8",
        transform=transform * Affine.translation(col0, row0),
        all_touched=all_touched,
    ).astype(bool)


def _confirm_vanished(candidates: np.ndarray, geometries: np.ndarray,
                      out_shape: tuple[int, int],
                      transform: Affine) -> np.ndarray:
    """Keep only the candidates that burn nothing when burned ALONE.

    The id band is a ``MergeAlg.replace`` burn, so a forbidden feature painted
    over by a LATER forbidden feature is absent from it although every cell it
    covers is still impassable. Reporting that as "vanished" is a false
    positive on ordinary overlapping data, and a detector that cries wolf gets
    switched off. Re-burning each candidate on its own — in its own window, so
    the cost is per-feature and not per-raster — separates "covers no cell
    centre" from "merely lost the overlap".
    """
    confirmed = []
    for index in candidates:
        window = _feature_window(geometries[index], transform, out_shape)
        if window is None:
            continue
        if not _burn_alone(geometries[index], window, transform,
                           all_touched=False).any():
            confirmed.append(int(index))
    return np.asarray(confirmed, dtype=np.int64)


def _uncovered_part_area(part, gap: np.ndarray,
                         window: tuple[int, int, int, int],
                         transform: Affine, min_area: float) -> float:
    """How much of ``part`` lies in cells that no forbidden feature burned.

    The gap cells come from the part's ALL_TOUCHED footprint, which over-reaches
    by up to a whole cell on each side, so their count is not a measure of
    anything: a 0.4 m strip that sits 5 mm inside a neighbouring forbidden
    feature still touches the free cell next door. Only the part's own AREA
    inside those cells says whether a barrier really has a hole there.

    Stops as soon as ``min_area`` is reached — the caller only needs the
    predicate, and a long fence can touch thousands of cells.

    The cell polygon is the bounding box of the four corners, which is the cell
    itself for the north-up transforms pyorps burns into and a conservative
    over-approximation for a rotated one.
    """
    row0, _, col0, _ = window
    rows, cols = np.nonzero(gap)
    rows = rows.astype(np.int64) + row0
    cols = cols.astype(np.int64) + col0
    total = 0.0
    for start in range(0, rows.size, _GAP_CELL_CHUNK):
        r = rows[start:start + _GAP_CELL_CHUNK]
        c = cols[start:start + _GAP_CELL_CHUNK]
        x0 = transform.c + transform.a * c + transform.b * r
        y0 = transform.f + transform.d * c + transform.e * r
        x1 = transform.c + transform.a * (c + 1) + transform.b * (r + 1)
        y1 = transform.f + transform.d * (c + 1) + transform.e * (r + 1)
        cells = shapely.box(np.minimum(x0, x1), np.minimum(y0, y1),
                            np.maximum(x0, x1), np.maximum(y0, y1))
        total += float(np.sum(shapely.area(shapely.intersection(part, cells))))
        if total >= min_area:
            break
    return total


def _partially_vanished_ids(geometries: np.ndarray, thin: np.ndarray,
                            band: np.ndarray, transform: Affine,
                            resolution_in_m: float,
                            mitre_limit: float,
                            passable: np.ndarray | None = None) -> np.ndarray:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Features with a sub-cell PART that burned nothing at all.

    "Did this FEATURE burn any cells" is the wrong question. A forbidden
    polygon made of a 2 m base and a 0.4 m arm burns cells (not vanished) as
    ONE 4-connected block (not fragmented), yet the arm burns zero cells and
    the barrier ends up with a hole the length of that arm. The useful
    question is whether the burned footprint faithfully represents the
    GEOMETRY, so the two signals are combined: :func:`is_thin` predicts which
    features are at risk, and the burn confirms it part by part.

    Each connected thin part is tested on its own — testing them jointly would
    let a fat feature's own corner slivers, which sit next to burned cells,
    vouch for an arm ten cells away.

    A part is judged on whether ITS OWN footprint is represented, in two steps,
    and never on incidental overlap with a different part of the same feature.
    Accepting a part because ANY cell of its ALL_TOUCHED footprint carries a
    forbidden burn made recall a coin flip on sub-pixel alignment: measured, a
    23.4 m long, 0.4 m wide arm off a fat base burns ZERO cells of its own, yet
    ONE cell of its 24-cell footprint overlaps a cell the FAT BASE burned — and
    that single cell excused the whole arm.

    1. Does the part cover a cell CENTRE? Then it burned that cell itself (the
       part is a subset of its feature, which burned every centre it covers),
       and it is represented. This is the only "present" verdict.
    2. Otherwise it burned nothing of its own. That is a hole only where the
       raster is actually free, so the part's own area inside the cells of its
       footprint that NO forbidden feature burned must reach ``min_area``. This
       keeps the forbidden-over-forbidden case out: a feature painted over by a
       later forbidden feature lies wholly inside burned cells, so its
       uncovered area is zero and the cell is still impassable either way.

    "No forbidden feature burned it" is NOT the same as "the route can walk
    there", and the difference is the same nodata/forbidden conflation that
    used to make :func:`detect_repair_seals` cry wolf: with the usual
    ``fill_value=IMPASSABLE_CELL_COST`` a cell that no feature covers at all is
    impassable too, so a thin arm running along the edge of the data or past a
    hole in it would be reported as holing a barrier that nothing can walk
    through anyway. Pass ``passable`` — the finished raster's ``!= impassable``
    mask — and those cells are excluded. Without it the step keeps its older,
    over-reporting meaning, which is why every in-tree caller supplies it.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    flagged = np.flatnonzero(np.asarray(thin, dtype=bool))
    if flagged.size == 0:
        return np.zeros(0, dtype=np.int64)

    burned = band != 0
    out_shape = band.shape
    min_area = _MIN_THIN_PART_CELL_AREA * resolution_in_m ** 2
    parts_of = thin_parts(geometries[flagged], resolution_in_m,
                          mitre_limit=mitre_limit)

    holed = []
    for index, residual in zip(flagged, parts_of):
        for part in shapely.get_parts(residual):
            if shapely.area(part) < min_area:
                continue
            window = _feature_window(part, transform, out_shape)
            if window is None:
                continue
            if _burn_alone(part, window, transform, all_touched=False).any():
                continue  # step 1: the part covers a cell centre of its own
            row0, row1, col0, col1 = window
            touched = _burn_alone(part, window, transform, all_touched=True)
            gap = touched & ~burned[row0:row1, col0:col1]
            if passable is not None:
                gap &= passable[row0:row1, col0:col1]
            if not gap.any():
                continue  # every cell it reaches is impassable anyway
            if _uncovered_part_area(part, gap, window, transform,
                                    min_area) >= min_area:
                holed.append(int(index))
                break
    return np.asarray(holed, dtype=np.int64)


def _vanished_ids(band: np.ndarray, n: int) -> np.ndarray:
    """Ids that appear nowhere in the band, scanned in row blocks.

    A boolean scatter beats ``np.bincount`` here (515 ms vs 699 ms at 144 M
    cells) and the blocking keeps the extra allocation at ``n`` bytes instead
    of a full-raster copy.
    """
    seen = np.zeros(n + 1, dtype=bool)
    for row in range(0, band.shape[0], _SCAN_BLOCK_ROWS):
        seen[band[row:row + _SCAN_BLOCK_ROWS].ravel()] = True
    return np.flatnonzero(~seen[1:]).astype(np.int64)


def _fragmented_ids(band: np.ndarray, geometries) -> np.ndarray:
    """Features whose burned cells split into more 4-components than parts.

    A per-FEATURE test, not a global one. The raw count of 2x2 diagonal
    pinches is useless as a warning (33895 of them on a raster whose features
    were nearly all fat — every 45-degree boundary of a perfectly fat polygon
    produces the pattern) and a global 4-vs-8 component count is true but
    unattributable. The numpy-only substitute for this test was measured at
    39.5 % recall and is deliberately not offered as a fallback.

    A 2-part MultiPolygon legitimately burns 2 components, hence the
    comparison against ``shapely.get_num_geometries`` rather than against 1.
    """
    from scipy import ndimage  # optional dependency, see the caller

    components, n_components = ndimage.label(band != 0,
                                             structure=_FOUR_CONNECTED)
    burned = band != 0
    ids = band[burned].astype(np.int64)
    labels = components[burned].astype(np.int64)
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)

    stride = n_components + 1
    n = len(geometries)
    if n * stride < 2 ** 62:
        pairs = np.unique(ids * stride + labels)
        feature_of_pair = pairs // stride
    else:  # pathological component count; two-column unique instead
        pairs = np.unique(np.stack([ids, labels], axis=1), axis=0)
        feature_of_pair = pairs[:, 0]
    counts = np.bincount(feature_of_pair, minlength=n + 1)[1:]
    parts = shapely.get_num_geometries(_as_geometry_array(geometries))
    return np.flatnonzero(counts > parts).astype(np.int64)


def detect_forbidden_burn_defects(
        forbidden_geometries,
        out_shape: tuple[int, int],
        transform: Affine,
        *,
        resolution_in_m: float = 1.0,
        check_fragmentation: bool = True,
        all_touched: bool = False,
        tolerance_cells: float = DEFAULT_TOLERANCE_CELLS,
        mitre_limit: float = DEFAULT_MITRE_LIMIT,
        passable: np.ndarray | None = None,
) -> ForbiddenBurnReport:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Option A: what the pixel-centre rule did to the forbidden features.

    Reports forbidden features that (a) burned zero cells, (b) burned only in
    part — a sub-cell arm of an otherwise fat feature that burned nothing,
    which is a hole in the barrier and is invisible to both other checks —
    and (c) burned into cells that are 8- but not 4-connected. It also lists
    the features that are merely AT RISK: thin geometry that happened to burn
    acceptably at this alignment. (a) and (c) are read off ONE extra scan
    conversion of the forbidden subset alone — measured at 12.1 % of a 3.3 s
    main burn (173 ms burn + 83 ms vanished scan + 144 ms connectivity scan on
    36 M cells with 5000 forbidden features).

    (b) and the at-risk list come from :func:`is_thin` (~10 us per feature on
    the forbidden SUBSET) plus one small windowed re-burn per thin part; both
    run unconditionally, because a check that only runs once another check has
    already failed is gated behind the very defect it exists to find.

    (c) needs :mod:`scipy.ndimage`. When it is missing the check is skipped
    and the report says so rather than silently substituting a 39.5 %-recall
    approximation.

    ``all_touched`` selects WHICH burn is being inspected: False (the default)
    is GDAL's pixel-centre rule, i.e. what the raster looks like without the
    overlay; True inspects the overlay's own footprint and is how a test or a
    UI can confirm that option D really did repair the barrier.

    Parameters:
        forbidden_geometries: The geometries whose burned value is
            IMPASSABLE_CELL_COST, in any order.
        out_shape: ``(rows, cols)`` of the raster they were burned into.
        transform: The affine transform of that raster.
        resolution_in_m: Cell size, carried into the report for its message.
        check_fragmentation: Run the corner-touching check (needs scipy).
        all_touched: Which burn to inspect; see above.
        tolerance_cells: Opening residual, in cell areas, still counted as fat.
        mitre_limit: How sharp a corner counts as thin; see the constant.
        passable: Optional boolean mask, ``raster != IMPASSABLE_CELL_COST`` of
            the raster these features were burned into. Used by (b) alone, to
            keep a sub-cell arm that runs through NODATA out of the report: with
            the default ``fill_value=IMPASSABLE_CELL_COST`` an uncovered cell is
            impassable, so a "hole" there is not a hole. Omitting it keeps the
            older behaviour, which over-reports exactly those cells.

    Returns:
        A :class:`ForbiddenBurnReport`; ``report.ok`` is True when every
        forbidden feature burned intact.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    geometries = _as_geometry_array(forbidden_geometries)
    n = int(geometries.size)
    if n == 0:
        return ForbiddenBurnReport(0, float(resolution_in_m))

    band = _burn_forbidden_id_band(geometries, out_shape, transform,
                                   all_touched=all_touched)
    vanished = _confirm_vanished(_vanished_ids(band, n), geometries,
                                 out_shape, transform)

    thin = is_thin(geometries, resolution_in_m,
                   tolerance_cells=tolerance_cells, mitre_limit=mitre_limit)
    # A feature that burned NOTHING is already reported in full; asking which
    # of its parts is missing would only name the same feature twice.
    partial_candidates = thin.copy()
    partial_candidates[vanished] = False
    partially_vanished = _partially_vanished_ids(
        geometries, partial_candidates, band, transform, resolution_in_m,
        mitre_limit, passable=passable)

    fragmented = np.zeros(0, dtype=np.int64)
    checked = False
    if check_fragmentation:
        try:
            fragmented = _fragmented_ids(band, geometries)
            checked = True
        except ImportError:
            checked = False
    return ForbiddenBurnReport(
        n_forbidden=n,
        resolution_in_m=float(resolution_in_m),
        vanished=vanished,
        fragmented=fragmented,
        partially_vanished=partially_vanished,
        at_risk=np.flatnonzero(thin).astype(np.int64),
        fragmentation_checked=checked or not check_fragmentation,
    )


def _enrich_report_with_advice(
        report: ForbiddenBurnReport,
        forbidden_geometries,
        bounds: tuple[float, float, float, float] | None,
) -> ForbiddenBurnReport:
    """Attach resolution advice when thin or defective features were found."""
    if report.at_risk.size == 0 and report.ok:
        return report
    thin = np.zeros(report.n_forbidden, dtype=bool)
    thin[report.at_risk] = True
    advice = suggest_resolution(
        forbidden_geometries, report.resolution_in_m, bounds, thin=thin)
    return ForbiddenBurnReport(
        n_forbidden=report.n_forbidden,
        resolution_in_m=report.resolution_in_m,
        vanished=report.vanished,
        fragmented=report.fragmented,
        partially_vanished=report.partially_vanished,
        at_risk=report.at_risk,
        fragmentation_checked=report.fragmentation_checked,
        advice=advice,
    )


def _defect_detail_message(report: ForbiddenBurnReport,
                           resolution_in_m: float) -> str:
    """Technical defect counts plus repair opt-in guidance."""
    return (
        f"Forbidden features did not survive rasterization at "
        f"{resolution_in_m:.3g} m: {report.summary()}. The only fix that is "
        f"correct rather than a trade is a finer cell size. Both repairs are "
        f"OPT-IN because both close legitimate sub-cell OPENINGS - a gate, a "
        f"culvert, a gap between parcels - while they seal barriers: swept over "
        f"8 gate widths x 8 sub-pixel offsets under four different gate-width "
        f"sets (see raster.thinness), and counting only the configurations the "
        f"plain rule leaves passable, widen_thin_forbidden=True sealed 38-100 % "
        f"of them in a 0.4 m wall and 0 % in a 2 m wall, all_touched=True "
        f"31-100 % and 17-100 %.")


def report_forbidden_burn_defects(
        forbidden_geometries,
        out_shape: tuple[int, int],
        transform: Affine,
        *,
        resolution_in_m: float = 1.0,
        bounds: tuple[float, float, float, float] | None = None,
        on_thin_features: str = "warn",
        stacklevel: int = 2,
        passable: np.ndarray | None = None,
        routing_assessment: bool = True,
) -> ForbiddenBurnReport | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Run option A and act on it: warn, raise, or stay silent.

    Returns the report (None when detection was switched off), so callers can
    keep it for a UI without re-running the scan.

    Parameters:
        forbidden_geometries: The geometries whose burned value is
            IMPASSABLE_CELL_COST.
        out_shape: ``(rows, cols)`` of the raster they were burned into.
        transform: The affine transform of that raster.
        resolution_in_m: Cell size, used for the message and for option E.
        bounds: ``(minx, miny, maxx, maxy)``, so the advice can state what the
            suggested cell size would cost in cells.
        on_thin_features: 'warn' (default), 'raise' or 'ignore'.
        stacklevel: Passed to :func:`warnings.warn` so the warning points at
            the caller's rasterize() call rather than at this module.
        passable: The finished raster's ``!= IMPASSABLE_CELL_COST`` mask; see
            :func:`detect_forbidden_burn_defects`.
        routing_assessment: When True (default), evaluate severity and emit a
            user-facing routing suitability message via :func:`warnings.warn`.

    Raises:
        ThinForbiddenFeatureError: When a defect is found and
            ``on_thin_features='raise'``.
        ValueError: On an unknown ``on_thin_features`` value.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if on_thin_features not in ("warn", "raise", "ignore"):
        raise ValueError(
            "on_thin_features must be 'warn', 'raise' or 'ignore', got "
            f"{on_thin_features!r}")
    if on_thin_features == "ignore":
        return None

    report = detect_forbidden_burn_defects(
        forbidden_geometries, out_shape, transform,
        resolution_in_m=resolution_in_m, passable=passable)
    report = _enrich_report_with_advice(report, forbidden_geometries, bounds)

    if routing_assessment:
        assessment = report.assessment_message()
        if report.ok:
            warnings.warn(
                assessment, ForbiddenBurnRoutingAssessment,
                stacklevel=stacklevel)
            return report

        message = f"{assessment}\n\n{_defect_detail_message(report, resolution_in_m)}"
        if on_thin_features == "raise":
            raise ThinForbiddenFeatureError(message)
        warnings.warn(message, ThinForbiddenFeatureWarning,
                      stacklevel=stacklevel)
        return report

    if report.ok:
        return report

    message = _defect_detail_message(report, resolution_in_m)
    if on_thin_features == "raise":
        raise ThinForbiddenFeatureError(message)
    warnings.warn(message, ThinForbiddenFeatureWarning, stacklevel=stacklevel)
    return report


# ----------------------------------------------------------------------
# The harm a repair CAUSES — free-space connectivity before vs after
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RepairSealReport:
    """What a forbidden-feature repair did to FREE space.

    Option A is structurally blind to this: it inspects the REPAIRED burn,
    which is defect-free by construction, so it detects only the defect the
    repair removes and never the one it introduces. Measured, it emitted zero
    warnings on a fixture where widening sealed a 1.20 m gate and produced
    ``NoPathFoundError``.

    ``n_split_regions`` is the signal. It counts free regions of the PLAIN burn
    that the repair broke into two or more pieces — a region that merely got
    SMALLER is expected and does not count, and one the repair swallowed whole
    does not count either.
    """
    n_free_regions_before: int
    n_free_regions_after: int
    n_split_regions: int
    cells_lost: int
    example_cell: tuple[int, int] | None = None
    checked: bool = True

    @property
    def ok(self) -> bool:
        """True when the repair did not disconnect free space."""
        return self.n_split_regions == 0

    def summary(self) -> str:
        """One sentence naming the count and one example cell."""
        if self.ok:
            return ""
        where = ("" if self.example_cell is None
                 else f" (e.g. at row {self.example_cell[0]}, column "
                      f"{self.example_cell[1]})")
        return (
            f"it SEALED free space: {self.n_split_regions} region(s) that the "
            f"plain pixel-centre burn left connected are cut into two or more "
            f"pieces{where}, and {self.cells_lost} free cell(s) were lost in "
            f"total")


def _split_free_regions(passable_before: np.ndarray,
                        passable_after: np.ndarray):
    """Free regions of ``before`` that ``after`` breaks into several pieces.

    THE TEST, and why this one. The cheap candidates all have a false case:

    * "fewer free cells" fires on every repair, since every repair removes
      free cells; shrinking a region is expected and is not a seal.
    * "more components after than before" misses a repair that splits one
      region while swallowing another whole (+1 and -1 cancel), and it also
      fires when a repair merely detaches a rim of cells into a stranded
      speck, which strands nothing that was reachable through it.

    So each free region of the repaired burn is mapped back to the region of
    the plain burn it came from — the repair only ever REMOVES free cells, so
    that map is well defined — and a before-region claimed by two or more
    after-regions is exactly a region the repair cut in two. Removing the last
    link between two areas is the same event: it is one before-region becoming
    two. A region that vanished entirely claims none and is silent.

    Connectivity is 4-, not 8-, because that is what pyorps' own routers can
    walk: a diagonal step is admissible only when BOTH flanking cells are
    passable, so any diagonal move is also a 4-path through either flank.

    Returns ``(n_before, n_after, split_before_labels, label_band_before)``.
    """
    from scipy import ndimage  # optional dependency, see the caller

    labels_before, n_before = ndimage.label(passable_before,
                                            structure=_FOUR_CONNECTED)
    labels_after, n_after = ndimage.label(passable_after,
                                          structure=_FOUR_CONNECTED)
    both = passable_before & passable_after
    before_ids = labels_before[both].astype(np.int64)
    after_ids = labels_after[both].astype(np.int64)
    if before_ids.size == 0:
        return n_before, n_after, np.zeros(0, dtype=np.int64), labels_before

    stride = n_after + 1
    if (n_before + 1) * stride < 2 ** 62:
        pairs = np.unique(before_ids * stride + after_ids)
        before_of_pair = pairs // stride
    else:  # pathological component count; two-column unique instead
        pairs = np.unique(np.stack([before_ids, after_ids], axis=1), axis=0)
        before_of_pair = pairs[:, 0]
    counts = np.bincount(before_of_pair, minlength=n_before + 1)
    return n_before, n_after, np.flatnonzero(counts > 1), labels_before


def detect_repair_seals(
        raster: np.ndarray,
        plain_geometries,
        repaired_geometries,
        out_shape: tuple[int, int],
        transform: Affine,
        *,
        all_touched: bool = False,
        impassable=None,
        plain_passable: np.ndarray | None = None,
        other_geometries=None,
) -> RepairSealReport | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Did widening or the all-touched overlay disconnect free space?

    ``raster`` is the FINISHED raster, i.e. the repair has already been applied
    to it. The plain burn's passable mask CANNOT be reconstructed from it, and
    this function used to try:

        other = impassable_after & ~forbidden_after
        passable_before = ~(other | forbidden_before)

    ``rasterize()`` fills cells that no feature covers with
    ``IMPASSABLE_CELL_COST`` — the SAME value a forbidden feature burns — so a
    NODATA cell and a forbidden cell are byte-identical in ``raster``. A nodata
    cell that the REPAIRED forbidden footprint happens to cover is therefore
    dropped from ``other`` and comes back out of that expression as
    passable-before. Those phantom cells bridge free regions the plain burn
    never connected, the bridge is gone in ``after``, and the check reports a
    split that never existed. A detector that cries wolf gets switched off, so
    the shortcut is dropped: the caller states the plain burn's passable mask
    and it is compared directly.

    Give exactly one of:

    * ``plain_passable`` — ``plain_raster != impassable`` for the raster the
      PLAIN rule would have produced. Free for the all-touched overlay (it is
      the raster as it stood before the overlay pass) and one re-burn for
      widening. This is what :class:`~pyorps.raster.rasterizer.GeoRasterizer`
      passes.
    * ``other_geometries`` — every NON-forbidden geometry that was burned. The
      plain passable mask is then burned here as "covered by a non-forbidden
      feature AND not covered by the plain forbidden footprint", which is exact
      because forbidden features carry the highest value and therefore win
      every overlap. Assumes the raster's fill is ``impassable`` (the default).

    Neither is optional: without one of them the plain burn is genuinely not
    recoverable, and the check raises rather than guessing.

    Doing the burn is CHEAPER than the reconstruction was, because it replaces
    two forbidden id-band burns with one: measured on 4401 / 8801 features at
    3000x3000 and 6000x6000 cells, 40.8 ms against 58.0 ms and 155.4 ms against
    193.3 ms. What this check actually costs is the two
    ``scipy.ndimage.label`` passes below — 302 ms and 1157 ms on the same two
    rasters — and that is unchanged. It only ever runs when a repair was
    explicitly enabled.

    Parameters:
        raster: The finished cost band, with the repair applied.
        plain_geometries: The forbidden geometries as DIGITIZED.
        repaired_geometries: The forbidden geometries as BURNED (widened, or
            the same objects when only the overlay is in play).
        out_shape: ``(rows, cols)`` of the raster.
        transform: Its affine transform.
        all_touched: True when the overlay (option D) was applied. Names the
            repair in the message; no burn depends on it any more.
        impassable: The value that marks a forbidden cell; defaults to
            ``IMPASSABLE_CELL_COST``.
        plain_passable: The plain burn's passable mask; see above.
        other_geometries: The non-forbidden geometries; see above.

    Returns:
        A :class:`RepairSealReport`, or ``None`` when :mod:`scipy.ndimage` is
        missing and the check could not run.

    Raises:
        ValueError: When neither ``plain_passable`` nor ``other_geometries``
            is given, or the mask does not match ``out_shape``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps.core.types import IMPASSABLE_CELL_COST

    if impassable is None:
        impassable = IMPASSABLE_CELL_COST
    plain = _as_geometry_array(plain_geometries)
    repaired = _as_geometry_array(repaired_geometries)
    if plain.size == 0 and repaired.size == 0:
        return RepairSealReport(0, 0, 0, 0)
    if plain_passable is None and other_geometries is None:
        raise ValueError(
            "detect_repair_seals needs the PLAIN burn: pass plain_passable "
            "(plain_raster != impassable) or other_geometries (every "
            "non-forbidden geometry). It cannot be read off the finished "
            "raster, because the fill value and a forbidden burn are the same "
            "number there.")
    try:
        import scipy.ndimage  # noqa: F401  (checked before paying for a burn)  # pylint: disable=unused-import
    except ImportError:
        return None

    impassable_after = raster == impassable
    passable_after = ~impassable_after
    if plain_passable is not None:
        passable_before = np.asarray(plain_passable, dtype=bool)
        if passable_before.shape != tuple(out_shape):
            raise ValueError(
                f"plain_passable has shape {passable_before.shape}, expected "
                f"{tuple(out_shape)}")
    else:
        forbidden_before = _burn_forbidden_id_band(plain, out_shape,
                                                   transform) != 0
        passable_before = (_burn_coverage(other_geometries, out_shape,
                                          transform) & ~forbidden_before)
    n_before, n_after, split, labels_before = _split_free_regions(
        passable_before, passable_after)

    lost = passable_before & ~passable_after
    example = None
    if split.size:
        is_split = np.zeros(n_before + 1, dtype=bool)
        is_split[split] = True
        seal_cells = np.argwhere(lost & is_split[labels_before])
        if seal_cells.size:
            example = (int(seal_cells[0][0]), int(seal_cells[0][1]))
    return RepairSealReport(
        n_free_regions_before=int(n_before),
        n_free_regions_after=int(n_after),
        n_split_regions=int(split.size),
        cells_lost=int(lost.sum()),
        example_cell=example,
    )


def report_repair_seals(
        raster: np.ndarray,
        plain_geometries,
        repaired_geometries,
        out_shape: tuple[int, int],
        transform: Affine,
        *,
        all_touched: bool = False,
        impassable=None,
        resolution_in_m: float = 1.0,
        on_thin_features: str = "warn",
        stacklevel: int = 2,
        plain_passable: np.ndarray | None = None,
        other_geometries=None,
) -> RepairSealReport | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Run the seal check and act on it: warn, raise, or stay silent.

    Shares the ``on_thin_features`` knob with option A on purpose — both
    answer "what should happen when the burn does not faithfully represent the
    vector data", and a caller who silenced one is not served by the other
    still firing.

    ``plain_passable`` / ``other_geometries`` are the plain-burn evidence
    :func:`detect_repair_seals` requires; one of them must be given.

    Raises:
        SealedOpeningError: When the repair sealed something and
            ``on_thin_features='raise'``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if on_thin_features not in ("warn", "raise", "ignore"):
        raise ValueError(
            "on_thin_features must be 'warn', 'raise' or 'ignore', got "
            f"{on_thin_features!r}")
    if on_thin_features == "ignore":
        return None

    report = detect_repair_seals(raster, plain_geometries, repaired_geometries,
                                 out_shape, transform,
                                 all_touched=all_touched, impassable=impassable,
                                 plain_passable=plain_passable,
                                 other_geometries=other_geometries)
    if report is None or report.ok:
        return report

    repair = ("all_touched=True" if all_touched else "widen_thin_forbidden=True")
    message = (
        f"The forbidden-feature repair {repair} changed the raster at "
        f"{resolution_in_m:.3g} m so that {report.summary()}. That may be the "
        f"barrier you wanted to close - and it may be a gate, a culvert or a "
        f"parcel gap that the data leaves open and a route needs. The raster "
        f"cannot tell them apart: a sub-cell BARRIER and a sub-cell GAP are "
        f"the same geometry seen from opposite sides, so no repair can "
        f"preserve both at this cell size. Check that the connections you "
        f"expect still exist, or drop the repair (the default) and raise the "
        f"resolution instead - suggest_resolution() computes the cell size "
        f"that resolves both.")
    if on_thin_features == "raise":
        raise SealedOpeningError(message)
    warnings.warn(message, SealedOpeningWarning, stacklevel=stacklevel)
    return report
