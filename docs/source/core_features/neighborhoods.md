# 🔗 Neighborhoods

Neighborhood connectivity defines which adjacent cells can be reached from any given cell during routing. A larger neighborhood produces smoother, more realistic paths at the cost of additional computation time.

## Overview

In a raster grid, each cell has a set of reachable neighbors. The simplest case (R0) allows movement only in the four cardinal directions, producing staircase-like paths. Adding diagonal connections (R1) improves this, and extending to knight's-move-like steps (R2 and beyond) produces progressively smoother results.

```{image} ../_static/images/intermediate_steps.PNG
:alt: Intermediate step patterns for different neighborhood sizes
:width: 100%
:align: center
```

## Predefined Neighborhoods

PYORPS provides eight predefined neighborhood configurations, named R0 through R7:

| Neighborhood | Connections | Description |
|-------------|-------------|-------------|
| R0 | 4 | Cardinal only (N, S, E, W) |
| R1 | 8 | Cardinal + diagonal |
| R2 (default) | 16 | Adds knight's-move-like steps |
| R3 | 32 | Extended knight's-move |
| R4 | 48+ | Further extended |
| R5 | 64+ | High-precision |
| R6 | 80+ | Very high precision |
| R7 | 96+ | Maximum precision, slowest |

```{image} ../_static/images/R3-complete.PNG
:alt: Complete R3 neighborhood showing all 32 connection directions
:width: 100%
:align: center
```

## Usage

### Using a Predefined Neighborhood

Pass the neighborhood name as a string when creating the `PathFinder`:

```{code-block} python
pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    neighborhood_str="r2",  # default
)
```

### Custom Neighborhood

For full control, provide a custom steps array. Each row defines a relative (row, column) offset to a reachable neighbor:

```{code-block} python
import numpy as np

# Define a custom 6-connected neighborhood
custom_steps = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1],  # cardinal
    [1, 1], [-1, -1],                    # two diagonals
])
pf = PathFinder(..., steps=custom_steps)
```

## Choosing a Neighborhood

The choice of neighborhood involves a trade-off between path quality and computation time:

- **R0** -- Fastest computation, but paths are restricted to axis-aligned movement (staircase effect). Suitable only for rough feasibility checks.
- **R1** -- Adds diagonal movement. Good for quick estimates where path smoothness is not critical.
- **R2** -- Best balance of quality and speed. The default for a reason: paths are noticeably smoother than R1 with modest overhead.
- **R3** -- Higher precision with smoother curves. Recommended when path quality matters and computation time is acceptable.
- **R4 to R7** -- Progressively higher precision. Use these when maximum path smoothness is required and computation time is not a constraint.

:::{tip}
Start with R2 (the default). If the resulting paths show visible angular artifacts, try R3. Going beyond R3 rarely provides noticeable improvement for most practical applications.
:::

## Performance vs. Quality

Larger neighborhoods increase the number of edges in the graph, which directly affects memory usage and computation time. The relationship is roughly linear: R3 processes approximately twice as many edges as R2.

```{image} ../_static/generated/neighborhood_comparison.png
:alt: Comparison of paths produced by different neighborhoods
:width: 100%
:align: center
```

```{image} ../_static/generated/neighborhood_sweep.gif
:alt: Animated sweep through neighborhoods R0 to R7
:width: 100%
:align: center
```

:::{note}
The computation time increase from a larger neighborhood is often less significant than the time spent loading and preparing the raster data. For small to medium rasters, the difference between R2 and R3 may be negligible.
:::
