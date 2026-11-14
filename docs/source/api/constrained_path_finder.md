---
title: "Constrained Path Finder"
summary: "Coupled route and tower search for overhead lines with turn-angle and span limits."
status: experimental
since: "0.4.0"
available_in: pypi
module: "pyorps.graph.constrained_path_finder"
api:
  - pyorps.graph.constrained_path_finder.ConstrainedPathFinder
  - pyorps.core.infrastructure_profile.InfrastructureProfile
---
# 🧪 Constrained Path Finding

Constrained path finding couples route optimization with tower placement in a single algorithm. Instead of first finding a route and then placing towers along it, PYORPS solves both problems simultaneously using an extended-state Dijkstra algorithm.

(constrained-path-finder-overview)=
## Overview

The algorithm operates on an extended state space where each state is a tuple `(cell, direction, span_length)`. This allows the algorithm to track not only position but also the current heading and the distance since the last tower, enforcing turn angle limits and span constraints during the search itself.

(constrained-path-finder-key-concepts)=
## Key Concepts

Turn angles
: Each tower type has angular limits. Suspension towers handle near-straight runs (small deflection angles). Angle towers accommodate larger turns. The algorithm penalizes or forbids turns that exceed the limits for a given tower type.

Span constraints
: Towers must be spaced within a minimum and maximum span range. The algorithm tracks distance since the last tower and forces tower placement when the maximum span is reached.

Tower types
: **Suspension** -- straight runs with minimal deflection. **Light angle** -- moderate turns. **Heavy angle** -- large turns. **Terminal** -- the structure at the route's start and end point (see {ref}`constrained-path-finder-line-ends`).

(constrained-path-finder-infrastructureprofile)=
## InfrastructureProfile

Configure infrastructure parameters using `InfrastructureProfile`:

```{code-block} python
from pyorps.core.infrastructure_profile import InfrastructureProfile

profile = InfrastructureProfile(
    name="example_line",
    description="Example overhead line",
    soft_angle_limit_deg=3.0,
    hard_angle_limit_deg=30.0,
    angle_cost_function="linear",   # "linear", "quadratic" or "piecewise"
    min_span_m=200,
    max_span_m=400,
    span_bin_size_m=10,             # required when both span limits are set
    conductor_weight_per_m=15.0,    # N/m
)
```

`name`, `description`, `soft_angle_limit_deg`, `hard_angle_limit_deg` and `angle_cost_function` are required; all other fields have defaults. Tower costs are configured through `tower_cost_function` and `tower_cost_params` (see the shipped `profiles/*.yaml` files), and a complete profile can be loaded from such a file with `InfrastructureProfile.load(path)`.

The profile defines the physical and cost parameters of the infrastructure. `soft_angle_limit_deg` is the threshold below which suspension towers suffice; `hard_angle_limit_deg` is the absolute maximum allowed deflection angle.

`terminal_span_rule` (default `"remove"`) says what happens to a tower that stands closer than `min_span_m` to an end point; see {ref}`constrained-path-finder-line-ends`. `terminal_tower_height_m` and `terminal_tower_cost` describe the structures at the two end points and are used by the report only. The search does not read `span_bin_size_m` (it is required when both span limits are set, but the search chooses its own span bins; see {ref}`constrained-path-finder-exact-bins`).

(constrained-path-finder-constrainedpathfinder)=
## ConstrainedPathFinder

`ConstrainedPathFinder` is a separate class from `PathFinder`, purpose-built for constrained routing:

```{code-block} python
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

cpf = ConstrainedPathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    profile=profile,
)
result = cpf.find_route()
```

(constrained-path-finder-line-ends)=
## Line ends

An overhead line starts and ends at a structure that the search does not place: a substation gantry, an existing tower, or just the point you gave. The report lists a **terminal tower** at both end points. The search places the **line towers** between them. It keeps every span between two line towers at least `min_span_m` and below `max_span_m`, and it keeps the first span (source to first line tower) at least `min_span_m`. It also keeps the last span (last line tower to target) below `max_span_m`, but it accepts the target at any span length. A line tower can therefore stand close to the target: when that is the cheapest line, when two lines cost the same, or when the line has to turn shortly before the target (a turn always needs a tower). The last span is then shorter than `min_span_m`.

Two settings decide what happens at the ends. The end-span rule is a field of the profile. The two connection heights belong to a run.

(constrained-path-finder-end-span-rule)=
### End-span rule

`InfrastructureProfile.terminal_span_rule` takes one of the four values of `pyorps.core.infrastructure_profile.TERMINAL_SPAN_RULES`. It has no effect for a profile without span limits. A profile without the key loads as `"remove"`; any other value raises `InvalidInputError`. Because it is a profile field, it is saved in the profile file and in the GUI project, and the GUI profile form shows it.

- `"remove"` (default, the result of earlier versions): a first or last line tower that stands closer than `min_span_m` to its end point is removed from the result after the search, and the end span becomes longer. Three things can go wrong, and each is reported: the merged span can reach `max_span_m`, the removed tower can leave a bend without a tower, or the merged span can fail the clearance check. Each case is one entry of `ConstrainedPath.span_notes` and is listed in one `SpanRuleWarning` per route. The removed tower is in `ConstrainedPath.removed_towers` (its `tower_id` is -1, and its `Tower.note` says why it was removed). Its cost is `cost_breakdown["removed_towers"]`. It is not in `towers`, `n_towers`, `total_tower_cost` or `total_cost`.
- `"remove_if_valid"`: the tower is removed only when the line stays valid without it: the tower stands on a straight stretch, the merged span stays below `max_span_m`, and (with the clearance check on) the merged span clears the ground. Otherwise it is kept, like under `"keep"`, and `Tower.note` and `span_notes` say why. No warning is given for a kept tower.
- `"keep"`: every tower the search placed is kept. A short end span is flagged in `Tower.note` and in `span_notes`; no warning is given, because you asked for it. The search keeps the first span at least `min_span_m` (up to rounding of the distance), so `"keep"` can only produce a short span at the target end. Swapping source and target can therefore change the result.
- `"enforce"`: the search itself keeps every end span at least `min_span_m` long. A line without any line tower (one span from source to target) is exempt from the minimum. Nothing has to be removed or flagged afterwards, and every span of the result lies in `[min_span_m, max_span_m)`. This rule needs exact span bins (see {ref}`constrained-path-finder-exact-bins`), so it can return a cheaper line than `"keep"` on the same problem.

`"remove"`, `"remove_if_valid"` and `"keep"` change the result after the search and work on every backend. `"enforce"` changes the search and runs on the CPU backends only.

(constrained-path-finder-connection-heights)=
### Connection heights

`source_connection_height_m` and `target_connection_height_m` give the height, in metres above the ground at the end cell, at which the conductor leaves the structure at that end (for example a gantry at the source and an existing tower at the target). They are arguments of `ConstrainedPathFinder` (also read/write properties) and keywords of `find_route`:

```{code-block} python
cpf = ConstrainedPathFinder(
    dataset_source="cost_raster.tiff", source_coords=source, target_coords=target,
    profile=profile, dem="terrain.tiff",
    source_connection_height_m=18.0, target_connection_height_m=12.0,
)
result = cpf.find_route()                                  # both heights
result = cpf.find_route(target_connection_height_m=None)   # source height only
```

The default is `None` (not set), which is the behaviour of earlier versions. In `find_route`, leaving a keyword out means the finder's setting, and passing `None` means "not set for this call". A height must be a number above 0: a value that is not a number (a `bool` is not) raises `InvalidTypeError`, and a value that is not finite or not above 0 raises `InvalidInputError`. For a chain of waypoints, run one `find_route` per segment, pass the source height only on the first segment and the target height only on the last, and `None` for the ends at waypoints.

When the heights act. A height reaches the search only when the clearance check is on: a terrain model (`dem=`), a profile with `tower_height_m` or `tower_heights_m`, `conductor_weight_per_m`, `conductor_tension_n` and `min_clearance_m`, and both span limits. Then the search checks the end spans from the heights:

- First span: the conductor leaves the source at the source height. The search chooses the position and the height of the first line tower so that this span clears the ground, and the source pays no height premium.
- Last span: the search accepts the target only when the last span clears the ground down to the target height. It chooses the position and the height of the last line tower for it.
- For these two spans the search samples every cell, the first cell after the start and the end cell included (the target cell for the last span), and it tests a span of one step too. For the first span it also samples the source cell, where the conductor leaves the structure.

Without the clearance check (no DEM, or a profile without the clearance data or span limits), a height cannot change the route. It is only reported as the height of the end tower, and one `PyorpsWarning` per finder says that it has no effect. With the check on, a height below the profile's `min_clearance_m` raises `InvalidInputError`: the conductor cannot be higher above the ground at its own end cell than the structure it hangs from, so no span from there could clear.

`terminal_tower_height_m` of the profile stays a value for the report. It never reaches the search. An end tower reports the connection height when one is set, else `terminal_tower_height_m`, else the line height (the lowest one with several tower heights; for the source see {ref}`constrained-path-finder-results`).

(constrained-path-finder-line-ends-errors)=
### When the settings leave no route

If `"enforce"` or a connection height the search used leaves no route, `find_route` raises `NoPathFoundError`. Its message names the settings and the exact span bins, and says what to try: move the end point, allow shorter end spans (`"keep"` or `"remove_if_valid"`), or change the connection height. A run with none of these settings keeps its earlier behaviour: an empty search returns an empty `ConstrainedPath`.

The GPU backends `raster_gpu` and `raster_gpu_v4` raise `AlgorithmNotImplementedError` for `"enforce"` (with span limits) and for a connection height that the search would use. The error comes before any GPU code runs. Use `cython` or `cython_parallel` for these settings. A connection height that has no effect is not refused on a GPU backend; the finder only warns about it.

(constrained-path-finder-exact-bins)=
### The exact search

The search merges the states of one cell and direction that share a span bin. A default run uses bins `min_span_m` wide (`ConstrainedPath.span_bin_size_m` says which size a run used). Such coarse bins keep the search small, but they can hide the cheapest line (see {ref}`constrained-path-finder-objective-limits`). The end rules would make this worse: a coarse bin can merge away the state that a minimum last span or an end-span clearance needs, and then the search reports no route although one exists.

A run with `"enforce"`, or with a connection height that the search uses, therefore takes exact bins: the bin is just under the shortest step of the neighbourhood, so that every span length has its own bin. With `K = ceil(max_span_m / shortest step)` there are `K + 1` bins. `span_bin_size_m` and `search_notes` of the result say so. The finder never falls back to coarse bins silently.

The search state space is the number of cells in the search window times the number of directions, the number of bins and the number of height classes (more than one only with the clearance check and several tower heights). The finder refuses the run with an `InvalidInputError` when a bound is exceeded. Four class attributes of `ConstrainedPathFinder` hold the bounds:

- `EXACT_BINS_MAX_STATES` (2,000,000,000): the number of states.
- `EXACT_BINS_MAX_BYTES` (4 GiB): the memory that the kernel of the run allocates. `ConstrainedPathFinder.exact_bins_memory(cells, n_dirs, n_span_bins, n_heights, backend=..., clearance=...)` returns the estimate (an `ExactBinsMemory`). For the kernels that keep their states in hash maps it takes every state as reached, so it is an upper bound.
- `EXACT_BINS_MAX_STEPS` (2048): the longest span in steps of the shortest step. Above it, single-precision span sums could put two neighbouring spans into one bin again.
- `EXACT_BINS_MAX_COUNT` (65535): the number of bins (a span bin is held in 16 bits).

The error message gives the numbers of the run and the bound. To get under it, use a coarser raster, a smaller search window (`search_space_buffer_m`), a shorter `max_span_m`, or another rule: `"keep"` or `"remove_if_valid"` without connection heights keeps the coarse bins. On a machine with more memory, raise `EXACT_BINS_MAX_STATES` and `EXACT_BINS_MAX_BYTES` together.

The bounds are tight at fine resolution. With the default bounds, the `r2` neighbourhood (16 directions), one tower height and a longest span of 400 m, a raster with 1 m cells gives 401 bins, and the memory bound allows a square search window of about 166 cells (166 m) on a side. With 5 m cells it is about 370 cells (1.8 km). The memory bound is the one that is reached first. At 1 m cells with long spans, only small windows are accepted.

(constrained-path-finder-results)=
## Results

The result is a `ConstrainedPath` object containing:

- **Route geometry** -- the optimized path as a coordinate sequence
- **Tower locations** -- positions where towers are placed along the route
- **Tower types** -- classification of each tower (suspension, light angle, heavy angle, terminal)
- **Tower costs** -- individual cost for each tower: the foundation cost (`Tower.terrain_cost`, from the land use under the tower) plus the tower-type cost (`Tower.angle_cost`, from the turn angle)
- **Turn penalties** -- `total_angle_penalty_cost`, the turn penalties of the profile the search charged on this route
- **Total cost** -- `total_cost` = terrain cost + tower costs + turn penalties, also split in `cost_breakdown`

The fields that describe the line ends and the search are:

- `Tower.height_m` -- the reported height of the tower. An end tower reports its connection height when one is set (see {ref}`constrained-path-finder-connection-heights`).
- `Tower.height_premium` -- the premium that the search charged for the tower's height class (several tower heights only). It is part of `Tower.total_cost`. It is 0.0 for a tower of the base height, with one tower height, and for a terminal tower other than the source.
- `Tower.note` -- one sentence on a flagged tower: a short end span that was kept, or a tower that the end-span rule removed.
- `ConstrainedPath.terminal_span_rule` -- the end-span rule the route was reported under.
- `ConstrainedPath.removed_towers` -- the towers that the rule removed next to an end point, with the costs the search charged (`tower_id` is -1). They are not part of `towers`, `n_towers` or `total_tower_cost`.
- `ConstrainedPath.span_notes` -- one line per flagged or broken span: a short end span that was kept, a merged span at or above `max_span_m`, or a span that fails the clearance check. It is empty when every span is in order. The `SpanRuleWarning` of the route lists the first six cases; `span_notes` has all of them.
- `ConstrainedPath.source_connection_height_m` and `.target_connection_height_m` -- the heights the route was reported with (`None` when not set).
- `ConstrainedPath.span_bin_size_m` -- the span-bin size of the search in metres (`None` without span limits).
- `ConstrainedPath.search_notes` -- what the search did that the profile alone does not say: that it used exact span bins, that it charged a height premium for the source, or that it took the lowest-height shortcut (see {ref}`constrained-path-finder-objective-limits`). It is empty for a plain default run.
- `cost_breakdown["height_premiums"]` -- the sum of the height premiums (several tower heights with the clearance check). It is already inside `"towers"`, not an extra term.
- `cost_breakdown["removed_towers"]` -- the summed cost of the removed towers. The key is present only when a tower was removed, and the sum is not part of `total_cost`.

Terminal towers are listed with their costs although the search does not charge them (they are the same for every route between the two points). Apart from the limits listed under {ref}`constrained-path-finder-objective-limits`, every other reported cost is what the search minimised. The identity is

`total_cost` - (`terrain_cost` + `angle_cost` of the two terminal towers) + `cost_breakdown["removed_towers"]` = the optimum of the search

where the optimum is divided by the scale when an objective is used, and `cost_breakdown["removed_towers"]` is 0 when no tower was removed. Height premiums are inside `total_cost` and inside the optimum, so they do not appear in the identity. A turn sharper than every angle type of the profile is priced with the type of the largest `max_angle_deg`, in the search and in the report alike.

The source terminal tower is reported with the height class that the search charged. With several tower heights and no source connection height, the search lets the source take every height class and charges the cheapest class that lets the first span clear. The report gives the source tower that height and the premium, which is part of the source tower's `total_cost`, and `search_notes` says so when the premium is above 0. A source connection height makes the source pay no premium.

A route that crosses itself can carry two towers on one cell. The report places towers by their position along the route, so such a route lists two `Tower` objects with the same `cell_index`.

With the clearance check on, the report checks every span of the returned line in full: every cell of the span, the first one after a tower too, and spans of one step. A span that fails is listed in `span_notes` and in the route's `SpanRuleWarning`. The route and its numbers do not change.

The terrain is priced per metre for every cell size on the CPU backends, with or without a DEM: a step over a raster with cells of size `c` costs the cell values times the step length in metres, and a tower or a turn costs the same whether the cells are 0.5 m or 5 m wide. The reported terrain cost is priced per metre in the same way, so the identity above holds at any cell size. The GPU solvers are the exception (see {ref}`constrained-path-finder-objective-limits`).

If a step of the returned route is not a step of the neighbourhood, the turn penalties cannot be recomputed: `total_angle_penalty_cost` and `total_cost` are then NaN and one `PyorpsWarning` says so. The route and its towers are still returned.

(constrained-path-finder-objective)=
## Objective Weights

`ConstrainedPathFinder(..., objective={...})` (backend `"cython"`, not with the clearance check) minimises

`J = sum_k w_k * M_k + w_tower * T + w_smooth * A`

where `M_k` are the cell layers of the objective (with `"length"`, see {ref}`metric-objective`), `T` the tower costs of the profile and `A` its turn penalties. The reserved weights `"tower"` and `"smoothness"` default to 1 (the profile costs apply as written); every other missing weight is 0. Only the ratios of the weights matter: `{"cost": 2}` gives the route of `{"cost": 1, "tower": 0.5, "smoothness": 0.5}`.

The search adds the terms in one unit: the search raster is the objective times the quantisation scale of the metric stack, and the turn and tower tables are multiplied by the same scale. The reported costs are in the user's units (the weighted terms, the search value divided by the scale): with `"tower": 5` every tower cost shows 5 times the profile cost, with `"tower": 0` it shows 0. With an objective, `cost_breakdown` also holds the profile's own costs of the route, `"profile_towers"` and `"profile_angle_penalties"`; `Path.metrics` and `Path.feasibility` give the exact, unquantised cell terms. Without an objective, or with `{"cost": 1}` and both reserved weights at 1, the scale is 1 and nothing is rescaled. The weighted tables always follow the current objective, also after `set_objective()`, which refuses the same objectives as the constructor.

An objective with a `"gradient"` weight is refused (`FeatureNotImplementedError`; without a DEM the metric stack refuses it first, with an `InvalidInputError` that asks for `dem=`): constrained routing keeps the gradient model of the profile (`gradient_cost_params`, `max_gradient_percent`).

An objective that evaluates to 0 on every traversable cell prices the ground at 0: no weighted cell layer is above 0 there and there is no `"length"` weight (for example `{"smoothness": 2}`). The route then minimises only the tower and turn costs, and a `PyorpsWarning` says so. A constant positive layer is priced: the condition is the value of the objective, not the number of layers. Cell weights so small next to `"tower"` and `"smoothness"` that the scaled tables leave the float32 range of the kernel raise a `NumericalPrecisionError`.

(constrained-path-finder-objective-quantisation)=
### Quantisation of the terrain term

Off the cost-only fast path (an objective with a `"length"` weight or with a second cell layer) the search raster has 16 bit, so the terrain term is rounded to a grid of `max(F) / 65534` in the user's units, where `max(F)` is the largest value of the objective on a traversable cell of the search window. The reported terrain cost is the sum on that rounded raster. Its error per route is at most about `0.5 * n * max(F) / 65534`, with `n` the number of cells on the route. The bound is a worst case (the roundings of the cells have either sign), and it matters only for very long routes on a raster with a large maximum value, next to the total cost of the route.

(constrained-path-finder-objective-limits)=
### Known limitations

These hold today. The first three are open: a change would alter existing results, so they are not changed yet. (A tower next to an end point is dropped only under `terminal_span_rule="remove"`, and then it is reported and warned about; see {ref}`constrained-path-finder-end-span-rule`.)

- Open: a change would alter existing results. A default run uses coarse span bins (the bin is `min_span_m` wide). Coarse bins merge states with different spans, so the search can miss the cheapest line, and with obstacles it can report no route although one exists. `"enforce"` and the connection heights use exact bins, so `"enforce"` can return a cheaper line than `"keep"` on the same problem.
- Open: a change would alter existing results. The search's clearance test does not sample the first cell after a tower, does not test a span of one step between two line towers, and does not sample the cell a line starts on when there is no source connection height. It also treats a span as clear without a walk along it when the lower tower is more than 50 m taller than the largest sag plus `min_clearance_m` (ground that rises more than that under such a span is not seen). The report checks every span in full and warns with `SpanRuleWarning`; it does not change the route. A route can therefore come out of the search with a span that is below the clearance.
- Open: a change would alter existing results. With several tower heights, a DEM and clearance data, a default run first searches with the lowest height only and returns that line if one exists. It does not compare that line with lines that use one tall tower at its premium, which can be cheaper. `search_notes` says when this happened. A run with a line-end setting (`"enforce"` or a connection height the search uses) does the full search over all heights.
- With a DEM the search applies the slope factors of the profile to the foundation costs, and with `tower_area_cost_mode="exact"` it prices the whole footprint of a tower. The report prices the foundation at the centre cell and does not recompute the slope factors, so the identity stated under Results does not hold for these profiles.
- A raster whose cells are not square is measured with the x size of the cell (the `a` of the transform) on both axes, with or without a DEM. Steps and spans along the y axis then have the wrong length. One `PyorpsWarning` per finder says so. Resample the raster to square cells.
- The GPU constrained solvers (`raster_gpu`, `raster_gpu_v4`) price the terrain per cell, not per metre (a `PyorpsWarning` says so when the cell size is not 1), and they refuse the line-end settings (`"enforce"` and the connection heights the search would use).

(constrained-path-finder-catenary-model)=
## Catenary Model

PYORPS includes a catenary sag model for conductor clearance validation. Given the span length, conductor weight, and tension, the model computes the maximum sag and verifies that ground clearance requirements are met. This is evaluated during the search, not as a post-processing step. The report then checks every span of the returned line again, in full, and warns about a span that fails (see {ref}`constrained-path-finder-results` and {ref}`constrained-path-finder-objective-limits`).

```{image} ../_static/generated/constrained_towers.png
:alt: Constrained routing result showing tower placement along the optimized path
:width: 100%
:align: center
```
