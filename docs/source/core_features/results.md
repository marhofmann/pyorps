# 📊 Results & Export

After computing a route, PYORPS returns structured result objects that provide access to path geometry, cost metrics, and runtime information. Results can be exported to standard geospatial formats for use in GIS software.

## Path Object

A single `Path` object is returned by `find_route()` for single source-target routing. It contains all information about the computed route.

```{code-block} python
result = path_finder.find_route()

# Endpoints
print(result.source)              # (472000, 5593400)
print(result.target)              # (472800, 5594000)

# Metrics
print(result.total_length)        # Length in CRS units (meters for projected CRS)
print(result.total_cost)          # Accumulated cost along the path
print(result.total_cell_cost)     # Sum of raw cell cost values
print(result.euclidean_distance)  # Straight-line distance between source and target

# Algorithm info
print(result.algorithm)           # "dijkstra"
print(result.graph_api)           # "cython"
print(result.neighborhood)        # "r2"

# Geometry
print(result.path_geometry)       # Shapely LineString

# Timing
print(result.runtimes)            # {"graph_creation": 0.5, "shortest_path": 1.2, ...}
```

## Path Metrics

Each `Path` provides a breakdown of the route length by cost category. This is useful for understanding how much of the route passes through different land-use types or cost zones.

```{code-block} python
# Length per cost value (in CRS units)
result.length_by_category
# {100: 523.4, 200: 112.1, 500: 89.7, ...}

# Percentage of total length per cost value
result.length_by_category_percent
# {100: 0.45, 200: 0.12, 500: 0.08, ...}
```

:::{tip}
Use `length_by_category` to assess how much of a route traverses expensive areas (e.g., nature reserves, urban zones) and whether alternative routes might reduce exposure to high-cost regions.
:::

## PathCollection

For multi-source or multi-target routing, `find_route()` returns a `PathCollection` that holds multiple `Path` objects.

```{code-block} python
paths = path_finder.find_route()  # Multi-target routing

# Basic operations
print(len(paths))                 # Number of paths

# Iterate over all paths
for path in paths:
    print(path.total_length)

# Access a specific path by ID
single = paths.get(path_id=0)
```

## GeoDataFrame Creation

Convert results to a GeoPandas `GeoDataFrame` for further analysis or integration with other geospatial workflows:

```{code-block} python
gdf = path_finder.create_path_geodataframe()
print(gdf.columns)
# Index(['source', 'target', 'length', 'cost', 'geometry', ...])
```

The GeoDataFrame includes one row per path with columns for source/target coordinates, length, cost, and the route geometry as a Shapely `LineString`.

## Export Formats

Save results directly to standard geospatial file formats:

```{code-block} python
# GeoJSON (widely supported, human-readable)
path_finder.save_paths("routes.geojson")

# Shapefile (legacy format, broad GIS compatibility)
path_finder.save_paths("routes.shp")

# GeoPackage (modern, recommended for complex datasets)
path_finder.save_paths("routes.gpkg")
```

:::{note}
The export format is determined automatically from the file extension. All formats supported by GeoPandas/Fiona are available.
:::

## Working with Results in GIS

Exported files can be opened directly in GIS applications such as QGIS or ArcGIS for further analysis and visualization:

- **QGIS** -- Drag and drop the exported file into the Layers panel, or use Layer > Add Layer > Add Vector Layer.
- **ArcGIS** -- Use the Add Data button or drag the file into the map view.

The exported geometries retain the coordinate reference system (CRS) of the input raster, so they will align correctly with other geospatial layers in the same CRS.
