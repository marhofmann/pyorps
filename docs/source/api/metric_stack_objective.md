---
title: "Metric Stack and Objective"
summary: "Keep several metrics apart, weight them, compare route variants and route over a DEM."
status: experimental
since: "0.4.0"
available_in: pypi
module: "pyorps.core.metric_stack"
api:
  - pyorps.Objective
  - pyorps.GradientOptions
  - pyorps.MetricStack
  - pyorps.RouteEnsemble
---
# Metric Stack and Objective

Cost-only routing collapses everything into one number. `MetricStack`, `Objective` and `RouteEnsemble` keep several metrics apart (cost, length, slope, your own layers), let you weight them, and compare the resulting route variants. The meaning of weights and metrics is explained in {doc}`../concepts/cost_semantics`.

(metric-stack)=
## MetricStack

Aligned float32 metric layers that share one grid and one forbidden mask.

- `MetricStack.from_single_raster(raster, transform, ...)` wraps an ordinary cost raster as the layer `cost`.
- `add_layer(name, values, ...)` adds a layer; `attach_dem(dem, ...)` attaches elevation, from which slope is derived; `window(window)` cuts a sub-window; `combine(objective, ...)` produces the raster the search runs on; `save(path)` and `MetricStack.load(path)` persist it.

(metric-objective)=
## Objective and GradientOptions

```python
from pyorps import Objective, GradientOptions

objective = Objective(
    weights={"cost": 1.0, "gradient": 0.5},
    gradient_options=GradientOptions(max_gradient_pct=20.0),
)
```

`weights` maps layer names to non-negative weights and defaults to `{"cost": 1.0}`, which reproduces classic cost-optimal routing. The reserved names `"length"` and `"gradient"` need no raster layer. `GradientOptions` controls only the slope response (additive term, multiplier, hard grade limit and the slope discretisation); the 3D length stretch is applied whenever a DEM is present.

Pass an objective to `PathFinder(..., objective=...)`. With a DEM, only the Cython Dijkstra applies slope terms; `total_length` then becomes the 3D length.

(metric-ensemble)=
## RouteEnsemble

`PathFinder.find_route_ensemble(objectives)` routes the same pair under several objectives, one after another, and returns a `RouteEnsemble`. `to_dataframe()` gives the comparison table and `pareto_front([...])` the non-dominated subset.

(metric-limits)=
## Limits

Delta-stepping does not support gradient terms. Constrained GPU look-up-table parity is not complete.

(metric-dem-3d)=
## DEM-Aware Routing (3D)

DEM-aware routing incorporates elevation data into the cost model, penalizing steep terrain. This is essential for elevation-sensitive infrastructure such as overhead power lines, underground cables, and pipelines where gradient limits apply.

### Setup

Pass a Digital Elevation Model (DEM) file to PathFinder via the `dem` parameter:

```{code-block} python
from pyorps import PathFinder

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    dem="elevation_model.tiff",
)
result = pf.find_route()
```

The DEM must cover the same geographic extent as the cost raster and use the same CRS. PYORPS will resample the DEM to match the cost raster resolution if they differ.

### How It Works

When a DEM is provided, PYORPS computes elevation gradients between adjacent cells. These gradients are combined with the base cost raster to produce elevation-aware edge costs. Steep uphill and downhill segments receive higher costs, discouraging routes over mountainous terrain when flatter alternatives exist.

### Slope Penalty Model

The gradient cost is controlled by two parameters in the infrastructure profile:

**`max_gradient_percent`**
: The maximum allowed terrain gradient in percent. Cells with a steeper gradient are treated as impassable (cost = 65535). For example, `max_gradient_percent: 40.0` blocks any edge steeper than 40%.

**`gradient_cost_function`** and **`gradient_cost_params`**
: Defines how gradients below the maximum are penalized. PYORPS supports:

  - **Exponential** (default): cost scales as `base_cost * scale^(gradient_pct)`. Gentle slopes add little cost, but cost rises sharply near the limit.
    ```yaml
    gradient_cost_function: exponential
    gradient_cost_params:
      scale: 2.0   # doubling factor per percent gradient
    ```

  - **Linear**: cost increases proportionally with gradient.
    ```yaml
    gradient_cost_function: linear
    gradient_cost_params:
      cost_per_percent: 100.0
    ```

| Gradient | Linear (100/%) | Exponential (scale=2.0) |
|----------|---------------|------------------------|
| 0% | 0 | 0 |
| 5% | 500 | 32 |
| 10% | 1,000 | 1,024 |
| 20% | 2,000 | 1,048,576 |
| 30% | 3,000 | ~1 billion |

The exponential model strongly discourages steep terrain while barely penalizing gentle slopes, making it ideal for most infrastructure routing.

### Underground Cable Routing with DEM

For underground cables, 3D routing avoids steep terrain that would increase trenching costs and erosion risk. The route shifts to follow contour lines and gentler slopes:

```{code-block} python
pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    dem="elevation_model.tiff",
    neighborhood_str="r2",
)
result = pf.find_route()
```

```{image} ../_static/generated/underground_3d.png
:width: 100%
:alt: Underground cable routing comparison — 2D vs 3D (DEM-aware)
```

The left panel shows the standard 2D route (cost raster only), while the right panel shows how the DEM-aware route adjusts to avoid steep slopes, overlaid on the elevation model.

### DEM + DSM for Overhead Lines

For overhead infrastructure, obstacle clearance matters. Use a Digital Surface Model (DSM) alongside the DEM to account for buildings, trees, and other above-ground features:

```{code-block} python
pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    dem="elevation_model.tiff",
    dem_kwargs={"dsm": "surface_model.tiff"},
)
result = pf.find_route()
```

The DSM provides surface heights (including vegetation and structures), while the DEM provides bare-earth elevation. The difference (DSM - DEM) gives obstacle heights, which PYORPS uses to evaluate clearance constraints for conductor sag between towers.

### Constrained 3D Routing (Overhead Lines)

When combined with `ConstrainedPathFinder`, DEM-aware routing jointly optimizes route geometry, tower placement, and terrain adaptation:

```{image} ../_static/generated/3d_routing.png
:width: 100%
:alt: 2D vs 3D constrained overhead line routing with tower placement
```

The left panel shows a 2D constrained route (cost raster only), while the right panel shows the 3D route with DEM+DSM awareness. The 3D route adapts to terrain, potentially choosing different tower positions and requiring more towers to navigate elevation changes.

### Use Cases

- **Power line routing** over mountainous terrain, where steep slopes increase construction cost and require taller towers
- **Underground cable routing** in hilly areas, avoiding excessive trenching on steep terrain
- **Pipeline routing** subject to maximum gradient constraints
- **Road routing** where grade limits must be respected
