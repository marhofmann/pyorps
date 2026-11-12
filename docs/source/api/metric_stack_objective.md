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

(metric-layers-coverage)=
### Metric layers that do not cover the grid

`PathFinder(..., metric_layers={"name": <tif or spec>})` reprojects every file onto the grid of the cost raster. A cell the file does not cover, a nodata cell of the file and a NaN cell have no value for that layer (a *void* cell). A void cell counts as 0 for that layer, so a route over a gap is priced as if the layer were 0 there, and it is not forbidden: a coverage gap is not an exclusion zone. `PathFinder` says so, once per layer, with a `PyorpsWarning` that names the layer and the number of cells (before, the gap was read as 0 without a word). `hard_max` and `hard_min` cannot be checked where there is no value, so with either bound set the void cells are forbidden. A file whose nodata value is 65535 keeps its meaning: those cells are forbidden, as a 65535 in any layer is. A layer that covers the grid completely gives exactly the same values as before and no warning, and an array of the grid's own shape is taken as it is (NaN in it is forbidden). For `{"derive": "slope_from_dem"}` the DEM's rule applies: cells the DEM does not cover, or has as nodata, are forbidden (before they were sea level) and one warning counts them.

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

Pass an objective to `PathFinder(..., objective=...)`. With a DEM, every backend applies the slope terms and `total_length` becomes the 3D length (`total_length_2d` keeps the 2D length); the backends that cannot evaluate them themselves hand the request to one that can, see {ref}`metric-dem-backends`.

(metric-ensemble)=
## RouteEnsemble

`PathFinder.find_route_ensemble(objectives)` routes the same pair under several objectives, one after another, and returns a `RouteEnsemble`. `to_dataframe()` gives the comparison table and `pareto_front([...])` the non-dominated subset.

(metric-limits)=
## Limits

Delta-stepping has no slope terms. With a DEM, `find_route(algorithm="delta-stepping")` and a `SearchSession` run as Dijkstra instead and say so once per finder or session with an `AlgorithmFallbackWarning`; `Path.algorithm` names Dijkstra. `CostField` and `CostFieldSet` settle a field with slope terms by the Dijkstra tree: `algorithm="auto"` and `algorithm="dijkstra"` do so on every backend (on `graph_api="raster_gpu"` with one `AlgorithmFallbackWarning` per finder, because the GPU kernel has no slope terms), while an explicit delta-stepping or GPU algorithm with a DEM raises `FeatureNotImplementedError`. Constrained GPU look-up-table parity is not complete.

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

The DEM must be a raster in the CRS of the cost raster: PYORPS does not reproject a DEM. A DEM in another CRS, or one whose extent does not overlap the search window, raises an `InvalidInputError` that names both coordinate systems (or both bounds) and asks you to reproject or clip the DEM first; it used to stop with rasterio's "Number of columns or rows must be non-negative". A DEM given as a tuple `(array, crs, transform)` keeps its own CRS and transform; a bare array takes those of the cost raster. PYORPS resamples the DEM (bilinear) onto the search window of the cost raster, whatever its resolution. A DEM that covers the window only partly is allowed. Cells the DEM does not cover, voids in it (NaN, or the file's nodata value) become forbidden cells: hard barriers with `ignore_max_cost=True` (the default), traversable at the maximum cost with `ignore_max_cost=False` (on the uint16 search raster; on the float32 weight raster they are `+inf` and stay barriers). They are forbidden before the first route, so a source or target on one is moved to the nearest cell that has elevation, with a warning that names the DEM, exactly like a point on any other forbidden cell (with `ignore_max_cost=False` no point is moved). A vector `dem` raises an `InvalidInputError`: it used to be dropped without a word.

(metric-dem-bare)=
### A DEM Without an Objective

A DEM without an `objective` means the same on every backend: the edge length is the 3D length in metres, and every edge cost is multiplied by the exponential slope penalty `exp(2 s^2)` (`s` is rise over horizontal run, both in metres). The slope tables use bins of 0.25 % up to `s_max_pct` = 200 %; a steeper edge uses the last bin, so the largest penalty is `exp(8)`, about 2981 (the 3D length stretch comes on top), not a cap of 10 000. That is exactly `objective={"cost": 1}` with `GradientOptions(multiplier="exponential")`. `dem_kwargs={"gradient_function": ...}` selects one of the other models (`"power"`, `"energy"`, `"sigmoid"`; any other name is the `"squared"` model, the quadratic default of the old edge builder); it is a slope option, not a raster read option. `Path.total_length` is the 3D length.

This fixes the old graph-library behaviour at cell sizes other than 1 m, and it does not reproduce it. The old edge builder of the graph-library backends measured the horizontal run in cells instead of metres, so a 10 % slope on 10 m cells was priced as a 100 % slope, and it capped the penalty at 10 000 and each edge at 65 535. At any cell size other than 1 m, routes of these backends over a bare DEM change; the new, metric formula is the physically correct one. At 1 m run in cells and in metres agree and only the caps differ.

For the 3D length without a slope penalty, ask for it: `objective={"cost": 1}` and the DEM. The finder's `objective` stays `None` for a DEM alone, so cost assumptions and dataset overlays work as before.

### How It Works

With a DEM, PYORPS prices every edge of the search by its 3D length, `sqrt(run^2 + rise^2)`, where the rise is the height difference of the two cells. With the slope `s = rise / run`, the length stretch is `sqrt(1 + s^2)` and is always applied. On top of it the objective sets the slope response: a slope multiplier and an additive slope exposure (the `"gradient"` weight), both configured with `GradientOptions`, and an optional hard grade limit. The response is tabulated once (bins of `bin_width_pct`) and every backend reads the same tables, so they return routes of the same cost; routes of equal cost can differ in their cells.

(metric-dem-backends)=
### Backends

| Backend | DEM and slope terms |
|---|---|
| `cython` | Computed natively by the Dijkstra solver. `"delta-stepping"` and `"delta-stepping-circular"` run as Dijkstra (`AlgorithmFallbackWarning`). |
| `networkx`, `networkit`, `igraph`, `rustworkx` | Computed natively, also in a retained `SearchSession`. |
| `raster_gpu` | Computed natively by the GPU delta-stepping kernel in `find_route`. A retained `SearchSession` has no GPU slope terms and runs the cython Dijkstra (`AlgorithmFallbackWarning`). |
| `raster_fim` | Does not route a DEM: its eikonal solution is continuous, not the discrete least-cost route, and it has no slope multiplier or additive term (on a steep cone its route costs tens of percent more than the optimum). The finder warns (`AlgorithmFallbackWarning`) and runs the cython Dijkstra; `graph_api` becomes `"cython"`. Its DEM mode stays available through `RasterFIMAPI`. |
| `cugraph` | Not implemented. |

A search never silently drops the terrain: every request with a DEM either computes the slope terms or runs on a backend that does, with a warning.

### Slope Response

`GradientOptions` sets how slope adds to the cost, on top of the length stretch:

- `additive`: `"linear"`, `"quadratic"` or `"exponential"`, weighted by `objective["gradient"]`;
- `multiplier`: `"exponential"`, `"power"`, `"energy"`, `"sigmoid"` or `"squared"`, applied to the whole per-metre cost;
- `max_gradient_pct`: edges steeper than this are forbidden.

The `"exponential"` multiplier is `exp(2 s^2)`:

| Slope s | 5 % | 10 % | 20 % | 50 % | 100 % | 200 % |
|---|---|---|---|---|---|---|
| Multiplier | 1.005 | 1.020 | 1.083 | 1.65 | 7.39 | 2981 |

A slope above `s_max_pct` (default 200 %) uses the value of the last bin.

### Slope Penalty of the Constrained Finder

`ConstrainedPathFinder` keeps the slope model of its infrastructure profile, not the objective's. The gradient cost is controlled by two parameters in the profile:

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
