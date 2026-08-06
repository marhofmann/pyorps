# Cost Semantics & the Feasibility Objective

PYORPS routes by minimizing a **feasibility objective**

$$F(\text{path}) = \sum_k w_k \cdot M_k(\text{path})$$

a user-weighted combination of named metrics. The default,
`objective={"cost": 1.0}`, reproduces classic cost-optimal routing exactly.

## Two questions, two kinds of numbers

1. **Where should the route go?** — decided by the *objective* during the
   search. Weights are ordinal steering: only their ratios matter.
2. **What does the route entail?** — answered by the *metrics* on every
   result: honest, response-free physical totals (EUR, meters, grade
   exposure), identical in meaning across all runs.

Nothing about the route choice is ever post-processed — every criterion
acts inside the search kernels. The evaluator only *reports*.

## Cell metrics vs. edge metrics

| Kind | Examples | How it enters the search |
|---|---|---|
| **Cell** metric | `cost` (EUR/m), `landscape`, `permit`, `terrain_slope` | combined into the weight raster (exact — the edge weight is linear in cell values) |
| **Edge** metric | `gradient` (along-route grade) | computed per relaxed edge from the DEM, through a slope-response lookup table |
| Implicit | `length` | a constant per-meter term (no raster behind it) |

The along-route gradient **cannot** be a cell layer: on a uniform hillside
every cell has the same steepness, but a step along the contour is level
while a step up the fall line is steep — only the kernels can tell the
difference.

## Defining metrics

Cost-assumption leaves carry named values (scalars stay valid and mean
`{"cost": value}`):

```python
cost_assumptions = {"landuse": {
    "Forest":      {"cost": 365, "landscape": 0.9, "permit": 0.6},
    "Agriculture": {"cost": 107, "factor": 3.0},   # weight = cost * factor
    "Residential": 65535,                          # forbidden in ALL metrics
    "Grassland":   130,                            # legacy scalar
}}
```

Every metric layer is a **per-meter intensity**: crossing 100 m of
`landscape = 0.9` contributes `90 index·m`. Forbidden (exactly 65535, inf
or NaN in *any* layer) is absolute regardless of weights. Additional
layers come from `metric_layers=` (prebuilt rasters, arrays, or
`{"derive": "slope_from_dem"}` with optional `hard_max`/`hard_min`).

## Slope: geometry vs. response

With a DEM, **slope always affects the path length** — the 3D stretch
$\sqrt{1 + s^2}$ is unconditional geometry, and all per-meter metrics
accrue over 3D meters. *How* slope additionally contributes to the
feasibility is configurable:

```python
PathFinder(..., dem="dgm1.tif",
    objective={"cost": 1.0, "gradient": 40.0},
    gradient_options=dict(
        additive="quadratic",          # g(s) for the gradient weight
        multiplier="exponential",      # optional penalty on the whole feasibility
        max_gradient_pct=30.0,         # hard grade limit (edge forbidden)
    ))
```

Supported on the `cython`, graph-library and `raster_gpu` backends, all
consuming the same discretized response tables (provable parity).

## Reading results

```python
path = finder.find_route()
path.feasibility        # the minimized objective (compare only under identical weights)
path.metrics            # {"cost": EUR, "landscape": index*m, "length": m, "gradient": %*m}
path.total_length_3d    # true 3D meters (total_length_2d retained)
path.max_gradient_pct   # steepest step on the route
path.length_by_class    # {"Forest": 412.0, ...} meters per feature class
path.objective_spec     # weights + quantization scale (provenance)
path.analyze()          # formatted report with all of the above
```

Two rules for interpretation:

- **`feasibility` is not money.** It is the achieved value of *your*
  weighted objective; with responses configured it intentionally differs
  from `sum(w_k * metrics[k])`.
- **The route is optimal w.r.t. the weights, not w.r.t. EUR.** If any
  non-cost weight is positive, a cheaper route may exist *by design* —
  compare `metrics["cost"]` across objective variants to price your
  policy.

> Design rationale and implementation phases:
> `docs/superpowers/plans/2026-08-04-feasibility-multi-objective-routing.md`
