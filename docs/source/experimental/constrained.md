# 🧪 Constrained Path Finding

:::{admonition} Experimental Feature
:class: warning

This feature is under active development. The API may change between releases.
Use in production with caution.
:::

Constrained path finding couples route optimization with tower placement in a single algorithm. Instead of first finding a route and then placing towers along it, PYORPS solves both problems simultaneously using an extended-state Dijkstra algorithm.

## Overview

The algorithm operates on an extended state space where each state is a tuple `(cell, direction, span_length)`. This allows the algorithm to track not only position but also the current heading and the distance since the last tower, enforcing turn angle limits and span constraints during the search itself.

## Key Concepts

Turn angles
: Each tower type has angular limits. Suspension towers handle near-straight runs (small deflection angles). Angle towers accommodate larger turns. The algorithm penalizes or forbids turns that exceed the limits for a given tower type.

Span constraints
: Towers must be spaced within a minimum and maximum span range. The algorithm tracks distance since the last tower and forces tower placement when the maximum span is reached.

Tower types
: **Suspension** -- straight runs with minimal deflection. **Light angle** -- moderate turns. **Heavy angle** -- large turns. **Dead-end** -- placed at route start and end points.

## InfrastructureProfile

Configure infrastructure parameters using `InfrastructureProfile`:

```{code-block} python
from pyorps.core.infrastructure_profile import InfrastructureProfile

profile = InfrastructureProfile(
    max_span_m=400,
    min_span_m=200,
    soft_angle_limit_deg=3.0,
    hard_angle_limit_deg=30.0,
    tower_base_cost=50000,
    conductor_weight_per_m=15.0,
)
```

The profile defines the physical and cost parameters of the infrastructure. `soft_angle_limit_deg` is the threshold below which suspension towers suffice; `hard_angle_limit_deg` is the absolute maximum allowed deflection angle.

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

## Results

The result is a `ConstrainedPath` object containing:

- **Route geometry** -- the optimized path as a coordinate sequence
- **Tower locations** -- positions where towers are placed along the route
- **Tower types** -- classification of each tower (suspension, light angle, heavy angle, dead-end)
- **Tower costs** -- individual cost for each tower, based on terrain and turn angle
- **Total cost** -- combined routing cost, tower costs, and conductor costs

## Catenary Model

PYORPS includes a catenary sag model for conductor clearance validation. Given the span length, conductor weight, and tension, the model computes the maximum sag and verifies that ground clearance requirements are met. This is evaluated during the search, not as a post-processing step.

```{image} ../_static/generated/constrained_towers.png
:alt: Constrained routing result showing tower placement along the optimized path
:width: 100%
:align: center
```
