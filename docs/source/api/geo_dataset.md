---
title: "GeoDataset and Data Input"
summary: "Load raster, vector, WFS and in-memory data through the GeoDataset hierarchy, including windowed reads."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.io.geo_dataset"
api:
  - pyorps.GeoDataset
  - pyorps.VectorDataset
  - pyorps.RasterDataset
  - pyorps.InMemoryVectorDataset
  - pyorps.LocalVectorDataset
  - pyorps.WFSVectorDataset
  - pyorps.LocalRasterDataset
  - pyorps.InMemoryRasterDataset
  - pyorps.initialize_geo_dataset
---
# 📁 Data Input

PYORPS accepts geospatial data in multiple formats: raster files, vector files, remote Web Feature Services, and in-memory objects. The `PathFinder` class auto-detects the input type and creates the appropriate internal dataset representation.

---

(geo-dataset-raster-input)=
## Raster Input

GeoTIFF files (or other rasterio-supported formats) can be passed directly. The raster values are interpreted as cell costs.

```python
from pyorps import PathFinder

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
)
```

Numpy arrays also work, but you must supply a coordinate reference system and an affine transform so PYORPS knows the spatial extent of the data.

```python
from rasterio.transform import from_bounds
from pyorps import PathFinder

transform = from_bounds(west, south, east, north, width, height)

pf = PathFinder(
    dataset_source=my_array,
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
    crs="EPSG:25832",
    transform=transform,
)
```

(geo-dataset-vector-input)=
## Vector Input

Shapefiles, GeoJSON, and GeoPackage files require `cost_assumptions` to map feature attributes to numeric cost values.

```python
pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="costs.csv",
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
)
```

Supported vector formats: `.shp`, `.geojson`, `.json`, `.gpkg`, `.gml`, `.kml`.

(geo-dataset-wfs-remote-input)=
## WFS Remote Input

Data from a Web Feature Service can be loaded by passing a dictionary with `url` and `layer` keys.

```python
wfs_source = {
    "url": "https://example.com/wfs",
    "layer": "landuse_layer",
}

pf = PathFinder(
    dataset_source=wfs_source,
    cost_assumptions="costs.csv",
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
)
```

(geo-dataset-in-memory-input)=
## In-Memory Input

GeoDataFrames and numpy arrays can be passed directly without writing to disk.

**GeoDataFrame** (vector data -- requires cost assumptions):

```python
import geopandas as gpd

pf = PathFinder(
    dataset_source=my_geodataframe,
    cost_assumptions="costs.csv",
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
)
```

**Numpy array** (raster data -- requires CRS and transform):

```python
from rasterio.transform import from_bounds

transform = from_bounds(west, south, east, north, width, height)

pf = PathFinder(
    dataset_source=my_array,
    crs="EPSG:25832",
    transform=transform,
    source_coords=(472000, 5593400),
    target_coords=(472800, 5594000),
)
```

---

(geo-dataset-windowed)=
## Windowed Raster Reads

```{pyorps-status} unreleased source
```

`LocalRasterDataset` can read only a part of a raster, so a very large GeoTIFF does not have to fit in memory.

- `load_metadata()` reads no pixels; afterwards the raster geometry (transform, shape, CRS) is known, which is all that is needed to compute a search window.
- `read_window(window, **kwargs)` returns the cells of a `rasterio.windows.Window` without touching `dataset.data`. The cells are exactly those a full read followed by slicing would give.
- `load_data(window=window)` loads only that window; `transform` and `shape` then describe the window instead of the file.

```python
from rasterio.windows import Window
from pyorps import LocalRasterDataset

ds = LocalRasterDataset("cost_raster.tiff").load_metadata()
tile = ds.read_window(Window(col_off=0, row_off=0, width=512, height=512))
```

`PathFinder` uses these methods by default (`windowed_read=True` in {doc}`raster_handler`).

(geo-dataset-geodataset-class-hierarchy)=
## GeoDataset Class Hierarchy

Internally, PYORPS normalizes all inputs through the `initialize_geo_dataset()` factory function. It inspects the input type and returns the appropriate `GeoDataset` subclass:

| Class | Input Type | Description |
|-------|-----------|-------------|
| `LocalRasterDataset` | File path (`.tiff`, `.tif`, `.jp2`, etc.) | Reads raster data via rasterio |
| `InMemoryRasterDataset` | Numpy array + CRS + transform | Wraps an existing array with spatial metadata |
| `LocalVectorDataset` | File path (`.shp`, `.geojson`, `.gpkg`, etc.) | Reads vector data via geopandas |
| `InMemoryVectorDataset` | `GeoDataFrame` or `GeoSeries` | Wraps an existing GeoDataFrame |
| `WFSVectorDataset` | `dict` with `url` and `layer` keys | Downloads vector data from a Web Feature Service |

The factory function is called automatically by `PathFinder`, so you rarely need to use it directly. If you do:

```python
from pyorps.io.geo_dataset import initialize_geo_dataset

dataset = initialize_geo_dataset("landuse.shp", crs="EPSG:25832")
dataset.load_data()
```

---

(geo-dataset-coordinate-input-formats)=
## Coordinate Input Formats

Source and target coordinates accept a variety of Python types:

| Format | Example |
|--------|---------|
| Tuple | `(472000, 5593400)` |
| List of tuples | `[(472000, 5593400), (472800, 5594000)]` |
| Shapely Point | `Point(472000, 5593400)` |
| Shapely MultiPoint | `MultiPoint([(472000, 5593400), (472800, 5594000)])` |
| GeoSeries | `gpd.GeoSeries([Point(...), Point(...)])` |
| GeoDataFrame | `gpd.GeoDataFrame(geometry=[Point(...), Point(...)])` |
| Numpy array | `np.array([[472000, 5593400], [472800, 5594000]])` |

A single coordinate (tuple or Point) defines one source or target. A collection defines multiple sources or targets for multi-point routing.

---

(geo-dataset-bounding-box-and-mask)=
## Bounding Box and Mask

You can restrict the area of interest using `bbox` or `mask` parameters.

**Bounding box** (`bbox`): A tuple `(xmin, ymin, xmax, ymax)`, a Shapely Polygon, or a GeoDataFrame.

```python
pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="costs.csv",
    source_coords=source,
    target_coords=target,
    bbox=(460000, 5580000, 480000, 5600000),
)
```

**Mask** (`mask`): A Shapely Polygon or a GeoDataFrame. Only data within the mask geometry is loaded.

```python
from shapely.geometry import Polygon

area = Polygon([(460000, 5580000), (480000, 5580000),
                (480000, 5600000), (460000, 5600000)])

pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="costs.csv",
    source_coords=source,
    target_coords=target,
    mask=area,
)
```

:::{note}
If the CRS of the bounding box or mask differs from the dataset CRS, PYORPS will automatically reproject it and emit a warning.
:::
