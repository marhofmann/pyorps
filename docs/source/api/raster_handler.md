---
title: "Raster Handler"
summary: "Cut the search window, apply the mask and map coordinates to indices."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.raster.handler"
api:
  - pyorps.raster.handler.RasterHandler
---
# Raster Handler

`RasterHandler` cuts the search window out of the cost raster, applies the mask and converts between coordinates and array indices. `PathFinder` creates one for you; use it directly when you want the window without a route search.

(raster-handler-signature)=
## Signature

```python
from pyorps.raster import RasterHandler

handler = RasterHandler(
    raster_source,               # a RasterDataset
    source_coords, target_coords,
    search_space_buffer_m=None,  # None: estimated, clamped to 200-4000 m
    input_crs=None,
    apply_mask=True,
    outside_value=None,
    bands=None,
    windowed_read=True,          # read only the window from disk
    copy_window=True,
)
```

`RasterHandler` is exported from `pyorps.raster`, not from the top-level package. `window_from_bounds` returns the raster window for a bounding box.

Parameter meanings are documented on the class: {py:class}`pyorps.raster.handler.RasterHandler`.

(raster-handler-window)=
## Windowed reads

With `windowed_read=True` (the default) only the search window is read from the file, so a very large GeoTIFF does not have to fit in memory. `copy_window=True` copies the window into its own array; set it to `False` only if you will not modify the array. Both arguments are new since 0.3.2 and are marked as such in {doc}`geo_dataset`.

(raster-handler-limits)=
## Limits

- Square cells are assumed: distances use the absolute cell width. See {doc}`../concepts/known_limitations`.
- A raster with more than 2^32 - 1 cells raises an error.
- How the window size is chosen is described in {doc}`../concepts/search_space`.
