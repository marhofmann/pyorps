---
title: "PathFinder"
summary: "The main entry point: routing modes, find_route parameters, re-routing and route simplification."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.graph.path_finder"
api:
  - pyorps.PathFinder
---
# ⚡ Path Finding

PYORPS supports five routing modes that cover all combinations of single and multiple source/target coordinates. The routing mode is determined automatically based on the number of sources and targets provided to the `PathFinder`.

(path-finder-routing-modes)=
## Routing Modes

| Mode | Sources | Targets | Returns | Description |
|------|---------|---------|---------|-------------|
| Single-to-Single | 1 | 1 | `Path` | One optimal route |
| Single-to-Multi | 1 | N | `PathCollection` | One source to each target |
| Multi-to-Single | N | 1 | `PathCollection` | Each source to one target |
| Multi-to-Multi | N | M | `PathCollection` | All combinations (N x M paths) |
| Pairwise | N | N | `PathCollection` | Matched pairs only (N paths) |

(path-finder-basic-usage)=
## Basic Usage

### Single to Single

The simplest case: one source, one target, one path.

```{code-block} python
path = path_finder.find_route()
```

```{image} ../_static/generated/route_s2s.png
:width: 100%
:alt: Single to Single routing result
```

### Single to Multi

One source, multiple targets. Returns a `PathCollection` with one path per target.

```{code-block} python
pf = PathFinder(source_coords=src, target_coords=[t1, t2, t3], ...)
paths = pf.find_route()
```

```{image} ../_static/generated/route_s2m.png
:width: 100%
:alt: Single to Multi routing result
```

### Multi to Multi

Multiple sources, multiple targets. By default, computes all combinations.

```{code-block} python
pf = PathFinder(source_coords=[s1, s2], target_coords=[t1, t2], ...)
paths = pf.find_route(pairwise=False)  # Returns 4 paths: s1->t1, s1->t2, s2->t1, s2->t2
```

```{image} ../_static/generated/route_m2m.png
:width: 100%
:alt: Multi to Multi routing result (all combinations)
```

### Pairwise

Matched pairs only. Requires equal numbers of sources and targets.

```{code-block} python
pf = PathFinder(source_coords=[s1, s2], target_coords=[t1, t2], ...)
paths = pf.find_route(pairwise=True)  # Returns 2 paths: s1->t1, s2->t2
```

```{image} ../_static/generated/route_pairwise.png
:width: 100%
:alt: Multi to Multi pairwise routing result
```

(path-finder-findroute-parameters)=
## `find_route()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | CoordinateInput | None | Override source (uses init value if None) |
| `target` | CoordinateInput | None | Override target (uses init value if None) |
| `algorithm` | str | "dijkstra" | Algorithm: "dijkstra", "delta-stepping", etc. |
| `calculate_metrics` | bool | True | Calculate path length, cost, categories |
| `pairwise` | bool | False | Pairwise mode (requires equal source/target count) |

(path-finder-re-routing)=
## Re-Routing

You can override the source and/or target coordinates without recreating the `PathFinder`. This reuses the already-loaded raster and graph configuration, which is significantly faster than creating a new `PathFinder` instance.

```{code-block} python
# Initial route
path_finder.find_route()

# Re-route with new endpoints (no raster reload needed)
new_path = path_finder.find_route(
    source=(473000, 5593000),
    target=(473500, 5594500),
)
```

:::{tip}
Re-routing is especially useful when exploring alternative endpoints on the same cost raster, or when integrating PYORPS into an optimization loop.
:::

(path-finder-simplify)=
## Simplifying the Route Geometry

```{pyorps-status} unreleased main
```

By default a route follows the cell centres and has one vertex per cell. `find_route(..., simplify={...})` replaces this geometry with a simplified line that has far fewer vertices, for display and export. The route cost and length stay those of the unsimplified route, because a simplified line can cut across forbidden cells, and a warning says so. The dictionary names the method and its tolerance. The route start and end never move.

```python
path = path_finder.find_route(
    simplify={"method": "douglas_peucker", "tolerance": 5.0},
)
```

| Method | Tolerance |
|---|---|
| `douglas_peucker` | Distance in CRS units. |
| `douglas_peucker_topology` | Distance in CRS units; avoids self-intersections. |
| `visvalingam` | Area in squared CRS units (needs the optional `simplify` extra). |
| `grid` | Grid size; vertices are snapped to it. |

The result keeps `simplification_method`, `simplification_tolerance` and the unsimplified `original_path_geometry`.

(path-finder-error-handling)=
## Error Handling

When no valid path exists between source and target (for example, when all routes are blocked by impassable cells with value 65535), `find_route()` raises a `NoPathFoundError`:

```{code-block} python
from pyorps.core.exceptions import NoPathFoundError

try:
    path = path_finder.find_route()
except NoPathFoundError:
    print("No valid path exists between source and target.")
```

:::{note}
This typically happens when the source or target is surrounded by impassable cells, or when a continuous barrier of impassable cells separates them entirely.
:::
