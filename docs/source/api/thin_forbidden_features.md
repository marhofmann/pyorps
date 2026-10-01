---
title: "Thin Barriers and Forbidden Zones"
summary: "Detect, widen and repair forbidden features that are thinner than a cell."
status: experimental
since: "0.4.0"
available_in: pypi
module: "pyorps.raster.thinness"
api:
  - pyorps.MIN_FORBIDDEN_WIDTH_CELLS
  - pyorps.safe_forbidden_width_m
  - pyorps.recommended_geometry_buffer_m
  - pyorps.is_thin
  - pyorps.min_feature_width
  - pyorps.thin_parts
  - pyorps.widen_thin_features
  - pyorps.detect_forbidden_burn_defects
  - pyorps.ForbiddenBurnReport
  - pyorps.ForbiddenBurnAssessment
  - pyorps.ForbiddenBurnSeverity
  - pyorps.ForbiddenBurnRoutingAssessment
  - pyorps.suggest_resolution
  - pyorps.ResolutionAdvice
  - pyorps.ThinForbiddenFeatureWarning
  - pyorps.ThinForbiddenFeatureError
  - pyorps.detect_repair_seals
  - pyorps.RepairSealReport
  - pyorps.SealedOpeningWarning
  - pyorps.SealedOpeningError
---
# 🚧 Thin Barriers & Forbidden Zones

A forbidden zone can only stop a route if it exists in the raster. This page
explains a failure mode that makes **narrow** forbidden features — a fence
line, a 0.8 m retaining wall, a 0.3 m easement strip, a hedgerow digitized as a
thin polygon — silently disappear when the vector data is converted to a cost
raster, why the fix is **a finer cell size** rather than a switch, and what the
two available repairs cost you if you turn one on anyway.

:::{important}
**The default changes nothing.** `rasterize()` burns exactly what GDAL's
pixel-centre rule burns — bit for bit, the same array PYORPS produced before
any of this existed — and *warns* when a forbidden feature did not survive.
Both repairs are opt-in, because both of them close legitimate sub-cell
**openings** while they seal barriers. See
[Why neither repair is the default](#why-neither-repair-is-the-default).
:::

Everything on this page concerns **forbidden** features only: features whose
cost value is `65535` (`IMPASSABLE_CELL_COST`). Ordinary land-use classes are
never affected.

---

(thin-forbidden-features-how-a-polygon-becomes-cells)=
## How a polygon becomes cells

Rasterization ("burning") walks the output grid and asks, for every cell, a
single question:

> **Is the centre point of this cell inside the polygon?**

If yes, the cell takes that polygon's cost. If no, it does not. That is GDAL's
default rule and PYORPS inherits it. It is fast, unambiguous, and it makes the
raster a fair sample of the vector data — for features that are a few cells
wide.

For features **narrower than one cell** it breaks down, because a shape can
slip *between* the centre points of the cells it crosses:

```
  cell centres:   ·     ·     ·     ·     ·        ·     ·     ·     ·     ·
                              ║                             ║
  0.8 m barrier:              ║                          ║
                              ║                             ║
                  ·     ·     ·     ·     ·        ·     ·  ║  ·     ·     ·
                     centred at x = 30.1                centred at x = 31.0
                     -> 60 cells burned                 -> 0 cells burned
```

The barrier is identical in both cases. Only its sub-pixel position differs.

### Failure 1 — the barrier vanishes

Measured on a 1 m grid: a **0.8 m wide vertical barrier** spanning the whole
study area burned **60 cells with its centre at x = 30.1 m, and ZERO cells at
x = 30.0, x = 30.9 and x = 31.0**. Over the seven sub-pixel alignments tested,
**3 of 7 burned nothing at all**.

Nothing in the input data distinguishes the good case from the bad one.
Shifting the study-area origin by 10 cm flips the outcome. On a synthetic test
layer with 28 forbidden features, detection reports *"3 of 28 forbidden
features burned NO cells at all"*.

A vanished barrier is not a weak barrier. It is **absent**: the router has
never heard of it, and the route will run straight through it.

### Failure 2 — the fence with holes

A thin **diagonal** feature (measured: a 0.3 m wide diagonal on a 1 m grid)
does not vanish. It burns one cell per row, in a staircase whose cells touch
only at their **corners** — nowhere along it do two burned cells share an edge.

Whether that is a hole depends on the router. A naive 8-connected walker steps
diagonally from one free cell to the next, through the corner, and crosses the
barrier. **PYORPS does not** — see
[Corner cutting is an existing guarantee](#corner-cutting-is-an-existing-guarantee)
below; this was measured across every backend and none of them crosses. But the
raster still no longer represents a continuous barrier, and anything else that
consumes it — a GIS export, another tool, your own connectivity check — will
see a fence with holes.

:::{warning}
Both failures are **completely silent**. No exception, no warning in the
result, no visible artefact in the route. The route PYORPS returns is genuinely
the cheapest route *for the raster that was built*. The raster is what is
wrong.
:::

### How wide is wide enough?

A forbidden feature is guaranteed to survive the burn at **every** sub-pixel
alignment once it is at least

```
safe width = sqrt(2) x resolution        (about 1.41 cells)
```

wide. The reason is geometric: an axis-aligned
`res × res` square placed anywhere always contains at least one cell centre,
and the smallest disk containing that square has radius `res / √2`. A feature
that inscribes such a disk everywhere always covers cell centres, and the cells
it covers form an edge-connected (not merely corner-touching) chain.

A measured sweep (23 angles × 24 sub-pixel offsets, widths 0.30 m to 2.00 m in
1 cm steps) reproduces the curve exactly: **1.00 cells at 0° and 90°, 1.37 at
30°, 1.42 at 45°**.

:::{caution}
Half a cell is **not** enough. Slivers at 30° and 45° still vanish after being
widened by `res / 2`, at every sub-pixel offset tested. Use
`safe_forbidden_width_m(resolution)` rather than a rule of thumb.
:::

```python
from pyorps import safe_forbidden_width_m

safe_forbidden_width_m(1.0)   # 1.4142... m at 1 m resolution
safe_forbidden_width_m(5.0)   # 7.0710... m at 5 m resolution
```

---

(thin-forbidden-features-the-five-options)=
## The five options

| | What | Effect on the RASTER | Effect on the ROUTES | Fixes |
|---|---|---|---|---|
| **A** | Detect after burning: forbidden features that vanished, that burned only *in part*, and barriers that came out corner-connected only — **the default**, plus a check on what a repair SEALED | none | none | nothing, but **reports** |
| **B** | Forbid corner-cutting in the router | none | none — **already guaranteed** by PYORPS | the fence with holes |
| **C** | Widen forbidden features narrower than a cell **before** burning — opt-in | only near those features | near thin features | barriers, **at the cost of gates** |
| **D** | `all_touched` for **forbidden features only** — opt-in | forbidden zones grow by ≤ 1 cell | routes hugging a forbidden edge shift by ≤ 1 cell | barriers, **at a higher cost in gates** |
| **E** | Resolution guidance: report the narrowest forbidden feature so you can pick a cell size — **the only fix that is not a trade** | none | none | informational |

Verified for **both C and D**: vanishing went from 3 of 7 alignments to
**0 of 7**, the 0.3 m diagonal barrier became edge-connected and blocks, and
every cell that comes out **passable** stayed **bit-identical** to the plain
burn.

And verified for both: they close openings the data leaves open — see
[Why neither repair is the default](#why-neither-repair-is-the-default).

### A — detection

Burns the forbidden features a second time, on their own, and asks three
questions per feature:

1. did it burn **zero cells**?
2. did a sub-cell **part** of it burn zero cells while the rest burned — an
   arm, a spur, a tail, i.e. a hole in an otherwise present barrier?
3. did its cells split into more edge-connected pieces than the geometry has
   parts (a corner-touching staircase)?

It also lists the features that are merely **at risk**: thin geometry that
happened to burn acceptably at *this* sub-pixel alignment and would break under
a different one. At-risk features are reported but do **not** clear
`report.ok` — a detector that fails a clean burn on geometry alone fires on 165
of 1168 features in the measured 6000 × 6000 sweep, gets switched off, and then
protects nobody.

:::{note}
Question 2 exists because question 1 is the wrong question on its own. One
forbidden polygon consisting of a 2 m base plus a 0.4 m arm burns cells (so it
has not vanished) and burns them as one edge-connected block (so it is not
fragmented) — while the arm burns **nothing**, leaving the barrier column
"40 of 64 impassable, 24 passable". The useful question is not *"did this
feature burn any cells"* but *"is the burned footprint a faithful
representation of this geometry"*: geometry thinness predicts the risk, and the
burn confirms it part by part.

A thin part is judged on **its own** footprint. It counts as present only when
it covers a cell centre itself; an all-touched footprint that merely overlaps a
cell some *other* part of the same feature burned does not count, because that
made recall a coin flip on sub-pixel alignment — measured, a 23.4 m long,
0.4 m wide arm burned zero cells of its own while one cell of its 24-cell
footprint touched a cell the fat base had burned, and that single cell excused
the whole arm. A part that burned nothing is then a defect only where the
raster is actually free: a feature painted over by a *later* forbidden feature
lies wholly inside impassable cells and is correctly not reported.
:::

Detection changes nothing. It only tells you what the pixel-centre rule did.

It runs **under the default**, where it is the only thing standing between you
and a silently absent barrier, and whenever the `all_touched` overlay is off:

```python
rasterizer.rasterize(resolution_in_m=1.0)   # detection only, raster unchanged
```

emits a `ThinForbiddenFeatureWarning` of the form

```text
Forbidden features did not survive rasterization at <resolution> m:
<N> of <M> forbidden features burned NO cells at all (e.g. index <i>) - they
are absent from the cost surface and cannot block a route. ... The only fix
that is correct rather than a trade is a finer cell size. Both repairs are
OPT-IN because both close legitimate sub-cell OPENINGS - a gate, a culvert, a
gap between parcels - while they seal barriers: swept over 8 gate widths x 8
sub-pixel offsets under four different gate-width sets (see raster.thinness),
and counting only the configurations the plain rule leaves passable,
widen_thin_forbidden=True sealed 38-100 % of them in a 0.4 m wall and 0 % in a
2 m wall, all_touched=True 31-100 % and 17-100 %.
```

The message reports **counts plus one example index**, never one line per
feature — one measured 5000-feature layer would otherwise have produced 1168
warnings, which only trains you to ignore them. If a defect is found, option E's advice is
appended to the same message.

Set `on_thin_features="raise"` to turn that warning into a hard
`ThinForbiddenFeatureError`, or `"ignore"` to skip the check entirely. You can
also promote the warning selectively:

```python
import warnings
from pyorps import ThinForbiddenFeatureWarning

warnings.simplefilter("error", ThinForbiddenFeatureWarning)
```

To inspect a burn without going through `rasterize()`, call the checker
directly. Its `all_touched=` argument selects **which** burn is inspected:
`False` (default) is the plain pixel-centre burn, `True` inspects the overlay's
own footprint — useful to confirm that option D really did repair a barrier.

```python
from pyorps import detect_forbidden_burn_defects

report = detect_forbidden_burn_defects(
    forbidden_geometries, out_shape=(6000, 6000), transform=transform,
    resolution_in_m=1.0,
)
report.ok                  # True when every forbidden feature burned intact
report.vanished            # indices into forbidden_geometries
report.partially_vanished  # indices with a sub-cell part that burned nothing
report.fragmented          # indices whose cells touch only at corners
report.at_risk             # thin geometry that burned fine at THIS alignment
print(report.summary())
```

:::{note}
`vanished` really means "covers no cell centre". A forbidden feature painted
over by a *later* forbidden feature is **not** reported: it disappears from the
detector's own id band (which is a `MergeAlg.replace` burn) but every cell it
covers is still impassable, so each candidate is re-burned on its own — in a
window around its own bounds, so the cost is per-feature, not per-raster —
before it is reported. Features that lie entirely outside the raster extent are
not reported either; nothing about the cell size would change them.
:::

:::{note}
The corner-touching check needs `scipy.ndimage`, which is a declared PYORPS
dependency. If SciPy is somehow absent that one check is **skipped**, and the
report says so (`report.fragmentation_checked is False`) rather than quietly
substituting a 39.5 %-recall approximation. The other checks always run.
:::

### B — no corner cutting (already guaranteed)

See [the dedicated section](#corner-cutting-is-an-existing-guarantee) below.
There is nothing to switch on.

### C — widen thin features before burning (opt-in)

Buffers every forbidden feature narrower than `√2 · resolution` up to exactly
that width — by `(√2·res − w) / 2` on each side, where `w` is its measured
narrowest width — and burns the widened geometry.

```python
rasterizer.rasterize(resolution_in_m=1.0, widen_thin_forbidden=True)
rasterizer.rasterize(resolution_in_m=1.0)                          # C is OFF
```

**Effect on the raster:** only near the thin features. **Effect on the routes:**
near thin features. Fat features are returned unchanged, not re-buffered — and
that is exactly why this is the better of the two repairs.

:::{danger}
**It is still not safe.** A wall that is itself sub-cell gets widened, and both
lips of any gate in it advance by up to `res / √2`. Swept over 8 gate widths ×
8 sub-pixel offsets under four different gate-width sets, counting only the
configurations the plain rule leaves passable, this option **sealed 38–100 % of
the gates in a 0.4 m wall** — the same band as the all-touched overlay's
31–100 %, so on a thin wall it is *not* the better repair. Its advantage is
confined to fat features: **0 %** in a 2 m wall under every set, which it never
touches, against 17–100 % for the overlay. That is why it is opt-in, and why
PYORPS warns with a `SealedOpeningWarning` when it happens — see
[When a repair seals something](#when-a-repair-seals-something).
:::

:::{caution}
This changes geometry, not just the raster. Two forbidden features less than
`√2 · resolution` apart will **merge**, swallowing any corridor between them,
and a widened feature grows by up to `res / √2` on each side. The difference to
the overlay is that only features that are *already sub-cell* can do this — and
a sub-cell forbidden feature is precisely the case the raster could not
represent either way.
:::

A burn that actually widens something disables the class-band cache for that
call: which features are forbidden depends on the cost table, so the burned
geometry would depend on it too. A layer that merely *has* forbidden features,
none of them thin, keeps its cache.

The same operation is available standalone:

```python
from pyorps import is_thin, min_feature_width, widen_thin_features

flags = is_thin(geometries, resolution_in_m=1.0)        # bool per feature
width = min_feature_width(geometries[3], 1.0)           # m, or None if safe
fixed = widen_thin_features(geometries, 1.0)            # widened copy
```

`is_thin` measures **inscribed** width — the property the burn actually depends
on. A 0.4 m tab flush against a 30 m block is *not* thin: a disk centred in the
block covers it and it burns intact. A genuine 0.4 × 10 m spike off a straight
edge *is* flagged. `min_feature_width` is a **gate, not a caliper**: it only
searches below `√2 · resolution` and returns `None` for anything at or above
that width, which means "safe", never "unknown".

### D — `all_touched` for forbidden features (opt-in)

After the ordinary burn, PYORPS can run a **second scan conversion of the
forbidden features alone**, in `ALL_TOUCHED` mode: a cell is painted if the
feature touches it *at all*, not only when it covers the centre. That result is
painted over the finished raster.

```python
rasterizer.rasterize(resolution_in_m=1.0, all_touched=True)
```

**Effect on the raster:** a forbidden zone grows by **at most one cell**, and
only there. Every other cell is untouched — a boolean hit-mask is burned
separately and used as an index, so no cell outside a forbidden feature is ever
written. Ordinary land-use classes keep GDAL's pixel-centre rule
**bit-for-bit**.

**Effect on the routes:** a route that was hugging the edge of a forbidden zone
shifts by up to one cell. A route that was passing through a vanished barrier
now goes around it (that is the point). **And a route through an opening
narrower than about two cells stops existing** — see below.

:::{danger}
**`all_touched=True` seals legitimate sub-cell openings.** It grows *every*
forbidden feature, not only the thin ones, so the two sides of a gate, a
culvert or a gap between parcels each advance by up to one cell and meet in the
middle. Measured end to end through `PathFinder`, a 2 m forbidden wall carrying
a 1.20 m slit:

| | open cells in the wall columns | route |
|---|---|---|
| plain pixel-centre rule (**the default**) | 4 | found, cost 500 |
| **C** `widen_thin_forbidden=True` | **4** | **found, cost 500** |
| **D** `all_touched=True` | **0** | **`NoPathFoundError`** |

(The wall here is 2 m — fat, so C never touches it. Make the wall itself
sub-cell and C closes the gate too; see the sweep below.)

A solvable problem becomes an unsolvable one, silently, because the raster no
longer contains an opening the data clearly has.
:::

**Reach for it** only when you want maximum conservatism and every opening
narrower than a cell may be treated as closed — a worst-case exclusion study,
say. For the ordinary case option C does the same job to fewer gates, and a
finer cell size does it to none.

:::{note}
Painting the forbidden features last is exactly the existing overlap contract:
features are burned in ascending cost order, so the most expensive feature
already wins any overlap, and `65535` is the most expensive value there is.
:::

### Why neither repair is the default

Both repair a vanishing barrier. Both also close openings. On a **fat** wall
only D does, because C touches only features that are actually thinner than a
cell:

| fixture | plain rule | **C** widen_thin | **D** all_touched |
|---|---|---|---|
| 1.20 m slit in a **2 m** wall stays open | 4 cells | **4 cells** | 0 — **SEALED** |
| 0.8 m barrier vanishes (of 7 alignments) | 3 | **0** | 0 |

But make the wall itself sub-cell and C closes the gate too. Sweeping a wall
with one gate over **8 gate widths × 8 sub-pixel offsets**, and counting only
the configurations in which the plain rule leaves the left edge 4-connected to
the right edge (the counts are the same under 8-connectivity):

The sweep is reproduced by
`tests/test_raster/sweep_forbidden_repair_tradeoff.py` — a 64 × 64 m extent at
1 m cells, a forbidden wall of the stated thickness spanning the extent at
`x = 30 m + offset` with one central gate, offsets `0.000 … 0.875 m` in steps
of 0.125 m. **The gate widths are a parameter of the sweep, not of the repair**,
so all four sets that were run are shown:

| gate widths | wall | passable configurations | plain | **default** | **C** | **D** |
|---|---|---|---|---|---|---|
| 0.4 … 1.8 step 0.2 | 0.4 m (thin — C widens it) | 52 | 0 | **0** | **52 (100 %)** | 52 (100 %) |
| 0.6 … 2.7 step 0.3 | 0.4 m | 58 | 0 | **0** | **34 (58.6 %)** | 34 (58.6 %) |
| 0.5 … 4.0 step 0.5 | 0.4 m | 58 | 0 | **0** | **22 (37.9 %)** | 18 (31.0 %) |
| 0.25 … 2.0 step 0.25 | 0.4 m | 52 | 0 | **0** | **48 (92.3 %)** | 44 (84.6 %) |
| 0.4 … 1.8 step 0.2 | 2.0 m (fat — C skips it) | 32 | 0 | **0** | **0** | 32 (100 %) |
| 0.6 … 2.7 step 0.3 | 2.0 m | 48 | 0 | **0** | **0** | 24 (50.0 %) |
| 0.5 … 4.0 step 0.5 | 2.0 m | 48 | 0 | **0** | **0** | 8 (16.7 %) |
| 0.25 … 2.0 step 0.25 | 2.0 m | 32 | 0 | **0** | **0** | 24 (75.0 %) |

Read the **structure**, not the percentages — a single number from one sweep is
a property of the gate widths chosen, and an earlier sweep whose widths were
never recorded reported 51.6 % for C where these report anything from 37.9 % to
100 %. What holds in every row:

* the plain rule and the default seal **nothing** (0 of N, by construction);
* on a **fat** wall C seals **nothing** and D seals a large fraction
  (17–100 %) — the sharp, robust difference between the two repairs;
* on a **thin** wall **both** seal a substantial fraction (C 38–100 %,
  D 31–100 %) and the two are within a few configurations of each other, in
  either order;
* therefore **neither repair is safe as a default**.

C is the better repair only where a fat feature is involved. On a thin wall it
destroys as many gates as the blunt overlay does.

:::{important}
This is not a bug in either option, and no third option fixes it. **At a fixed
cell size you cannot simultaneously guarantee that a sub-cell BARRIER is
represented and that a sub-cell GAP is preserved.** They are the same geometry
seen from opposite sides; any repair that fattens barriers closes gaps. The
information is not in the raster.

So the default is **detection only** — the raster stays bit-identical to a
plain GDAL burn — and the fix that is *correct* rather than a trade is a finer
cell size, which option E computes for you.
:::

The two knobs are independent: switching D on does not switch C on.

### When a repair seals something

Option A cannot see this by itself. It inspects the burn it is given, and once
a repair has run that burn is defect-free *by construction* — measured, it
emitted **zero** warnings on a fixture where widening sealed a 1.20 m gate and
the subsequent route raised `NoPathFoundError`.

So whenever a repair is active, PYORPS additionally compares the **connectivity
of free space** before and after it:

```python
from pyorps import detect_repair_seals

report = detect_repair_seals(raster, digitized_geoms, burned_geoms,
                             out_shape, transform,
                             other_geometries=non_forbidden_geoms)
report.n_split_regions      # free regions the repair cut in two
report.n_free_regions_before, report.n_free_regions_after
report.cells_lost
report.example_cell         # (row, col) of one lost cell on a split region
```

:::{important}
**The "before" side has to be stated, not inferred.** `rasterize()` fills cells
that no feature covers with `IMPASSABLE_CELL_COST` — the same value a forbidden
feature burns — so in the finished raster a NODATA cell and a forbidden cell are
byte-identical. An earlier version of this check reconstructed the plain burn as
`impassable_after & ~forbidden_after`, and every nodata cell that the repaired
footprint happened to cover came back out of that expression as *passable
before*. Those phantom cells bridged free regions the plain burn had never
connected, the bridge was gone afterwards, and the check reported a split that
never existed.

So the plain burn is really re-done. Pass **either** `other_geometries` (every
non-forbidden geometry — the mask is then burned as "covered by a non-forbidden
feature and not by the plain forbidden footprint", exact because forbidden
features win every overlap) **or** `plain_passable` (`plain_raster != impassable`,
which is what `GeoRasterizer` hands over). With neither, the function raises
rather than guessing. Inside `rasterize()` the extra cost is **zero** for
`all_touched=True` — the raster before the overlay pass *is* the plain burn —
and one 1-byte re-burn for `widen_thin_forbidden=True`.
:::

Free space is labelled with **4-connectivity**, because that is what a PYORPS
router can actually walk (a diagonal step needs *both* flanking cells passable,
so every move it can make is also a 4-path). Each free region of the repaired
burn is mapped back to the region of the plain burn it came from, and a
plain-burn region claimed by two or more repaired regions is exactly one the
repair cut in half. That test is chosen so that:

* a region that merely got **smaller** does not fire — every repair shrinks
  free space, and shrinking strands nobody;
* a region the repair **swallowed whole** does not fire — it claims no
  successor and there was nothing left to reach;
* a repair that splits one region while swallowing another still fires, which a
  bare "more components than before" count would miss (+1 and −1 cancel).

A hit raises `SealedOpeningWarning` (or `SealedOpeningError` under
`on_thin_features="raise"`). The message is deliberately two-sided: the
disconnection **may be the barrier you wanted to close**, and it may be a gate
you needed. The raster cannot tell them apart — that is the whole point — so
what the warning asks you to do is check that the connections you expect still
exist, or drop the repair and raise the resolution.

### E — resolution guidance

Reports the narrowest forbidden feature in the layer, the cell size that would
resolve it, and — always together with it — how many cells that would cost over
the same extent.

```python
from pyorps import suggest_resolution

advice = suggest_resolution(forbidden_geometries, resolution_in_m=1.0,
                            bounds=(minx, miny, maxx, maxy))
print(advice.summary())
advice.narrowest_width_m         # m
advice.safe_resolution_in_m      # cell size that would resolve it
advice.cells_at_safe_resolution  # what that costs over `bounds`
```

**Effect on the raster and on the routes:** none. It is informational, and it
runs automatically as part of option A's warning message once a defect exists.

**Reach for it** while choosing `resolution_in_m` for a new study area — and
read the two numbers together. Real data makes the point: a layer whose
narrowest forbidden feature is 5 cm needs a 3.6 cm cell, i.e. **2.8 × 10¹⁰
cells** over a 6 km extent. Resolution alone usually cannot fix *that* layer —
and the honest reading is that at any affordable cell size a 5 cm feature is
simply not representable, so what a repair gives you is a barrier that blocks
plus an unknown number of gates that no longer open. Turn one on knowing that,
and read the `SealedOpeningWarning` it produces.

---

(thin-forbidden-features-defaults-and-why)=
## Defaults, and why

`widen_thin_forbidden=False` **and** `all_touched=False` — **the raster is a
plain GDAL burn**
: Under the default the burned array is bit-identical to what the pixel-centre
  rule produces on its own, and that identity is pinned by a full-array
  regression test over a mixed-class fixture at EPSG:25832 magnitudes. A
  default may warn; it may not change a cell. Both repairs seal openings the
  data leaves open (38–100 % and 31–100 % of the gates in the measured sweeps,
  on a thin wall), and a default
  may not silently turn a solvable routing problem into an unsolvable one.

`on_thin_features="warn"`
: The one knob for "the burn does not faithfully represent the vector data", in
  both directions. It covers option A — a forbidden feature dropped, holed or
  fragmented, checked whenever `all_touched` is off, i.e. under the default —
  and the seal check, which runs whenever a repair is on. With the overlay on,
  option A has nothing left to find and is skipped; the seal check is what
  replaces it.

No corner-cutting
: Not a new option — see below.

---

(thin-forbidden-features-corner-cutting-is-an-existing-guarantee)=
## Corner cutting is an existing guarantee

PYORPS routers have never cut corners, and option B is therefore **not** a new
feature. It is a property that was already true, and is now pinned by a
regression test and documented.

The mechanism is not a corner test — there is no such test anywhere in the
tree. It falls out of how diagonal moves are priced. Every step in a
neighborhood carries a list of **intermediate cells** it passes over (see
{doc}`../concepts/neighborhoods`), and for a single diagonal step that list is exactly its
**two flanking cells**. Every backend then applies the same generic rule:

> A step is only admissible if **all** of its intermediate cells are passable.

So a diagonal step is rejected whenever either flanking cell is impassable —
which is strictly stricter than "do not cut corners", since the usual
formulation only rejects when *both* flanks are blocked.

This was measured, not assumed. On a 24 × 24 grid with an impassable main
diagonal there are 46 opportunities to cut a corner. A naive 8-connected flood
fill crosses. Nothing in PYORPS does: all four edge builders produce **zero**
corner-cutting edges at r1, r2 and r3, and every solver — Cython Dijkstra and
delta-stepping, NetworkX, NetworKit, the GPU raster backends, the eikonal/FIM
backend and the constrained CPU planners — reports **no path** across a sealed
thin diagonal barrier. Each of those assertions is paired with a positive
control that frees one barrier cell and requires the same call to succeed.

:::{note}
One reporting caveat: the `raster_fim` backend's returned *cell list* can
contain a diagonal pair whose flanks are blocked, because samples that land on
forbidden cells are dropped when the polyline is converted back to cells. The
field it propagates never crosses such a corner — no diagonal edge exists in
its stencil. This affects the shape of the returned cell list, not the route's
admissibility.
:::

---

(thin-forbidden-features-the-one-thing-that-is-impossible)=
## The one thing that is impossible

**A barrier that is absent from the raster cannot be routed around by any
router rule.**

Options A, B and E do not change a single cell. That is a feature — but it also
means that none of them can fix a *vanished* barrier. No corner rule, no
neighborhood, no algorithm and no backend can avoid an obstacle that is not
represented in the data it searches. The mirror image is just as true: no
router can walk through an opening that a fattened burn has closed, which is
what option D does to every gap narrower than about two cells.

It follows that **fixing the vanishing case always changes the raster.** The
only two options that fix it are D (grow the forbidden zone to every cell it
touches) and C (grow the geometry before burning), and both, by construction,
produce a different cost surface than the plain burn — and, because a sub-cell
barrier and a sub-cell gap are the same geometry seen from opposite sides, a
cost surface in which some sub-cell openings are gone. If you need bit-identical
rasters *and* sub-cell barriers that block, you need a finer cell size — see
option E, and read the cell count before committing to it.

---

(thin-forbidden-features-what-it-costs)=
## What it costs

Measured on a 2000 × 2000 raster with 2201 features, 200 of them forbidden:

| Step | Time | Note |
|---|---|---|
| plain burn | 38 ms | baseline |
| **D** `all_touched=True` | 38 ms | within noise — the overlay burns only the forbidden subset |
| **A** detection | +41 ms | 14 ms burn + 13 ms vanished scan + 14 ms connectivity scan |
| **A + E** advice | +46 ms | the message, not the checks — only paid once a defect exists |
| **C** widening | +31 ms | |
| seal check | one plain re-burn (C only) + two `ndimage.label` passes | only when C or D is on; skipped entirely under the default |

**The seal check does re-burn the plain raster** — and that made it *cheaper*,
not dearer. It used to reconstruct the plain burn from the finished one, which
is unsound (see [When a repair seals something](#when-a-repair-seals-something))
and cost two forbidden-subset burns. Now:

* `all_touched=True` — **no burn at all.** The raster as it stands before the
  overlay pass *is* the plain burn, so the mask is one array comparison.
* `widen_thin_forbidden=True` — **one** re-burn of the un-widened geometry
  sequence, into a 1-byte passable/forbidden band rather than the cost band.

Measured on a synthetic layer of 4401 / 8801 features (10 % forbidden fences,
half of them sub-cell), best of 3:

| raster | plain re-burn (new) | the 2 id-band burns it replaces | whole plain cost burn | the 2 `ndimage.label` passes |
|---|---|---|---|---|
| 3000 × 3000 (9 M cells) | **40.8 ms** | 58.0 ms | 60.3 ms | 302 ms |
| 6000 × 6000 (36 M cells) | **155.4 ms** | 193.3 ms | 156.6 ms | 1157 ms |

The burn is not what this check costs; the connected-component labelling is.

The vanish confirmation and the partial-vanish check add one small windowed
re-burn per candidate — sized to the feature's own bounds, not to the raster —
plus the `is_thin` pass over the forbidden subset (~10 µs per feature).

The first call into `scipy.ndimage` costs a one-off 235 ms import; steady-state
connectivity scanning is 14 ms over 4 M cells.

The thinness predicate itself costs about 10 µs per simple feature (52 ms for
5000). That is affordable on a **forbidden subset**; it is not a pass to run
over a 300 000-feature land-use layer, where the same test measures 11.6 s —
the cost scales with vertex count. `min_feature_width` costs about 0.3 ms per
feature and is only applied to features already flagged as thin.

On a 6000 × 6000 raster with 5000 forbidden features, option A caught
**1003 of 1003** defective features. It also flags 165 features that happened
to burn cleanly at the alignment tested — those are precisely the features a
10 cm origin shift would break.

---

(thin-forbidden-features-where-the-options-apply)=
## Where the options apply

| Entry point | A | C | D | E |
|---|---|---|---|---|
| `GeoRasterizer.rasterize()` | ✓ (default) | opt-in | opt-in | ✓ (inside A's message) |
| `GeoRasterizer.rasterize_metrics()` | ✓ (default) | opt-in | opt-in (cost band only) | ✓ |
| `PathFinder(...)` with vector input | ✓ (default) | opt-in | opt-in | ✓ |
| `GeoRasterizer.modify_raster_from_dataset()` | — | — | — | — |

`PathFinder` forwards these keywords to the rasterizer:

```python
pf = PathFinder(
    dataset_source="landuse.gpkg",
    source_coords=source, target_coords=target,
    cost_assumptions="costs.csv",
    resolution_in_m=1.0,
    widen_thin_forbidden=False,  # the default; True sealed 38-100 % of measured gates
    all_touched=False,           # the default; True sealed 31-100 %
    on_thin_features="warn",     # the default
)
```

:::{warning}
**Overlays are not covered.** Forbidden zones added *after* the base burn with
{doc}`modify_raster_from_dataset <geo_rasterizer>` — the usual way to stamp in
nature reserves at `65535` — still use the pixel-centre rule, and a sub-cell
feature added that way can still vanish. Until that path is covered, either
include such layers in the base burn, or buffer them yourself with
`geometry_buffer_m` (a buffer of at least `safe_forbidden_width_m(res) / 2`
guarantees the burn survives).
:::

In `rasterize_metrics()` only the **cost** band is overlaid. The other metric
bands are deliberately left alone: forbidden-ness rides the cost band, and
repainting e.g. a gradient band on the extra rim would invent values that no
feature ever supplied.

---

(thin-forbidden-features-api-summary)=
## API summary

All of these are importable from `pyorps` directly, or from `pyorps.raster`.

| Name | What it is |
|---|---|
| `MIN_FORBIDDEN_WIDTH_CELLS` | `√2` — the safe width in cells |
| `safe_forbidden_width_m(res)` | the safe width in metres |
| `is_thin(geometries, res)` | boolean array: which features are narrower than the safe width *somewhere* |
| `min_feature_width(geom, res)` | narrowest width in metres, or `None` if the feature is already safe |
| `thin_parts(geometries, res)` | the sub-cell PARTS of each feature (geometry minus its opening) |
| `widen_thin_features(geometries, res)` | option C, standalone |
| `detect_forbidden_burn_defects(...)` | option A, returns a `ForbiddenBurnReport` |
| `ForbiddenBurnReport` | `.ok`, `.vanished`, `.partially_vanished`, `.fragmented`, `.at_risk`, `.fragmentation_checked`, `.summary()` |
| `suggest_resolution(...)` | option E, returns a `ResolutionAdvice` |
| `ResolutionAdvice` | `.narrowest_width_m`, `.safe_resolution_in_m`, `.cells_at_safe_resolution`, `.summary()` |
| `ThinForbiddenFeatureWarning` | warning category emitted by option A |
| `ThinForbiddenFeatureError` | raised when `on_thin_features="raise"` |
| `detect_repair_seals(...)` | did a repair disconnect free space? returns a `RepairSealReport` |
| `RepairSealReport` | `.ok`, `.n_split_regions`, `.n_free_regions_before/after`, `.cells_lost`, `.example_cell`, `.summary()` |
| `SealedOpeningWarning` | warning category emitted by the seal check |
| `SealedOpeningError` | raised when `on_thin_features="raise"` |
