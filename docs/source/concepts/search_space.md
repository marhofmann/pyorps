---
title: "Search Space Control"
summary: "How the search window around source and target is chosen and sized."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.raster.handler"
api: []
---
# 🔍 Search Space Control

Routing performance scales directly with the number of raster cells the algorithm must explore. Limiting the search space to a relevant corridor around the source and target dramatically reduces computation time without sacrificing result quality.

---

(search-space-why-limit-the-search-space)=
## Why Limit the Search Space

A 10 km route on a 1 m resolution raster could span millions of cells if the full raster is used. In practice, the optimal path rarely deviates far from the direct line between source and target, so most of those cells are never relevant. Restricting the search area to a buffer around the source-target axis is the single most effective performance optimization.

---

(search-space-buffer-parameter)=
## Buffer Parameter

The `search_space_buffer_m` parameter defines a corridor (in meters) around the convex hull of source and target coordinates. Only raster cells within this corridor are loaded and processed.

```python
from pyorps import PathFinder

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
    search_space_buffer_m=600,  # 600m corridor around source-target line
)
```

```{image} ../_static/images/buffer_600.png
:alt: Search space with 600m buffer
:width: 100%
:align: center
```

:::{warning}
Neither `search_space_buffer_m=0` nor `search_space_buffer_m=None` gives you the **full raster**, despite what earlier versions of this page said.

- `0` is a zero-width buffer around the convex hull of the points — the **smallest** search space, not the largest. With a single source/target pair the hull is a line with no area, so `0` raises a `ValueError`.
- `None` (the default) calls `estimate_buffer_width`, which samples the raster and returns a value between **200 m and 4000 m**, sized for one source/target pair.

To search the whole raster — which is what a `CostField` over scattered candidates needs — pass `full_window_buffer_m`:

```python
from pyorps import PathFinder, full_window_buffer_m

finder = PathFinder(raster, source, target,
                    search_space_buffer_m=full_window_buffer_m(raster))
```

This can cause excessive memory usage and long computation times for large rasters, so use it deliberately.
:::

---

(search-space-choosing-the-right-buffer)=
## Choosing the Right Buffer

The buffer must be large enough to include the optimal path, which may deviate from a straight line to avoid high-cost or impassable areas. General guidance:

| Buffer | Cells (approx, 1m res) | Runtime | When to use |
|--------|------------------------|---------|-------------|
| 200 m | ~40k | Fast | Short routes, uniform cost landscape |
| 600 m | ~360k | Moderate | Typical urban/rural routing |
| 1200 m | ~1.4M | Slower | Complex landscapes with large obstacles |
| `None` | 200–4000 m, estimated | Varies | The default; sized for one source/target pair |
| `full_window_buffer_m(raster)` | Full raster | Slowest | `CostField` over scattered candidates |

:::{tip}
Start with a moderate buffer (e.g., 600 m) and increase it if the algorithm reports that the path touches the buffer boundary. A path running along the edge of the search space may indicate that a wider corridor is needed.
:::

---

(search-space-bounding-box)=
## Bounding Box

The `bbox` parameter provides an alternative way to restrict the search area to a rectangular region. It accepts a tuple `(xmin, ymin, xmax, ymax)`, a Shapely Polygon, or a GeoDataFrame.

```python
pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    bbox=(465000, 5585000, 480000, 5600000),
)
```

The bounding box is applied during data loading, so only the relevant portion of the file is read into memory.

---

(search-space-mask)=
## Mask

The `mask` parameter restricts the search area to an arbitrary polygon (or set of polygons). This is useful for non-rectangular study areas.

```python
from shapely.geometry import Polygon

study_area = Polygon([
    (465000, 5585000), (475000, 5585000),
    (478000, 5595000), (462000, 5598000),
])

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    mask=study_area,
)
```

A GeoDataFrame can also be used as a mask:

```python
import geopandas as gpd

mask_gdf = gpd.read_file("study_area.gpkg")

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    mask=mask_gdf,
)
```

---

(search-space-resource-warning)=
## Resource Warning

When the raster exceeds `MAX_SAFE_CELLS` (2^32 - 1 = 4,294,967,295 cells) without any search space restriction, PYORPS raises a `ResourceWarning`. This threshold exists because the Cython shortest-path kernels use `uint32` cell indices, and exceeding this limit would cause silent index overflow.

For very large rasters, always set `search_space_buffer_m`, `bbox`, or `mask` to keep the effective cell count within safe bounds.

---

(search-space-combining-parameters)=
## Combining Parameters

The `search_space_buffer_m`, `bbox`, and `mask` parameters can be combined. The buffer is applied within the region defined by the bounding box or mask:

```python
pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    bbox=(460000, 5580000, 490000, 5610000),
    search_space_buffer_m=800,
)
```

In this example, the raster is first clipped to the bounding box during loading, and then the search space is further narrowed to an 800 m corridor around the source-target axis.

```{image} ../_static/generated/search_space_buffers.png
:alt: Combined search space restrictions
:width: 100%
:align: center
```
