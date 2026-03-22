# 🚀 Quick Start

This tutorial walks through a complete routing example in six steps, from importing PYORPS to exporting the result as a GeoJSON file.

:::{note}
This guide assumes you have PYORPS installed and its Cython extensions compiled. See {doc}`installation` if you have not set up the package yet.
:::

## The Minimal Example

### Step 1: Import PathFinder

`PathFinder` is the main interface for all routing operations in PYORPS.

```{code-block} python
from pyorps import PathFinder
```

### Step 2: Define Source and Target Coordinates

Coordinates must be specified in the same coordinate reference system (CRS) as your raster dataset. For example, if your raster uses EPSG:25832 (UTM zone 32N), provide coordinates in meters:

```{code-block} python
source = (472000, 5593400)
target = (472800, 5594000)
```

### Step 3: Create the PathFinder

Pass a raster file path along with the source and target coordinates. PYORPS ships with a sample raster for testing:

```{code-block} python
path_finder = PathFinder(
    dataset_source="path/to/sample_raster.tiff",
    source_coords=source,
    target_coords=target,
)
```

:::{tip}
Set `search_space_buffer_m` to limit the routing area around your source and target. This significantly reduces memory usage and computation time for large rasters:

```python
path_finder = PathFinder(
    dataset_source="path/to/sample_raster.tiff",
    source_coords=source,
    target_coords=target,
    search_space_buffer_m=5000,  # 5 km buffer
)
```
:::

### Step 4: Find the Optimal Route

Call `find_route()` to compute the least-cost path:

```{code-block} python
result = path_finder.find_route()
print(result)
```

The result is a `Path` object (or `PathCollection` for multi-source/target routing) containing the route geometry, total cost, and length:

```{code-block} text
Path(id=0, source=(472000, 5593400), target=(472800, 5594000), length_m=1045.32, cost=12847.50, runtime_total=0.23)
```

### Step 5: Visualize the Result

Plot the route overlaid on the cost raster:

```{code-block} python
path_finder.plot_paths()
```

```{image} ../_static/generated/quickstart_path.png
:alt: Quickstart routing result showing the optimal path on a cost raster
:width: 100%
:align: center
```

```{image} ../_static/generated/dijkstra_wavefront.gif
:width: 100%
:alt: Dijkstra wavefront expansion animation
```

### Step 6: Export as GeoJSON

Save the route to a GeoJSON file for use in GIS software (QGIS, ArcGIS, etc.):

```{code-block} python
path_finder.save_paths("route.geojson")
```

## Complete Script

Here is the full example as a single script:

```{code-block} python
:linenos:
from pyorps import PathFinder

# Define coordinates (must match raster CRS)
source = (472000, 5593400)
target = (472800, 5594000)

# Create PathFinder
path_finder = PathFinder(
    dataset_source="path/to/sample_raster.tiff",
    source_coords=source,
    target_coords=target,
)

# Find the optimal route
result = path_finder.find_route()
print(result)

# Visualize
path_finder.plot_paths()

# Export as GeoJSON
path_finder.save_paths("route.geojson")
```

## About Coordinate Reference Systems

:::{important}
Your source and target coordinates **must** use the same CRS as the input raster. PYORPS does not reproject coordinates automatically.

To check a raster's CRS, you can use rasterio:

```python
import rasterio
with rasterio.open("path/to/sample_raster.tiff") as src:
    print(src.crs)   # e.g., EPSG:25832
    print(src.bounds) # coordinate extent of the raster
```

Make sure your source and target coordinates fall within the raster's bounds.
:::

## About Cost Values

Each cell in the cost raster represents the cost of traversing that location. PYORPS uses `uint16` values internally:

- **0** -- zero cost (free to traverse)
- **1 to 65534** -- increasing cost
- **65535** -- impassable / forbidden area

The optimal route minimizes the cumulative cost from source to target.

## Next Steps

Now that you have found your first route, explore the full capabilities of PYORPS:

- {doc}`../core_features/data_input` -- Load rasters, vector data, WFS services, or in-memory arrays
- {doc}`../core_features/cost_assumptions` -- Customize land-use cost mappings and assumptions
- {doc}`../core_features/neighborhoods` -- Control path smoothness with different neighborhood sizes (r1, r2, r3, ...)
- {doc}`../core_features/path_finding` -- Multi-source, multi-target, and pairwise routing modes
