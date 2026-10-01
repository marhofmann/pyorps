---
title: "Path Lengths and Units"
summary: "Route lengths are reported in CRS units; cost versus length and cell shape."
status: stable
since: "0.4.0"
available_in: pypi
module: "pyorps.core.path"
api: []
---
# Path Lengths and Units

(path-lengths-units)=
## Lengths are in CRS units

Every length that PYORPS reports for a route is measured in the units of the raster's coordinate reference system (CRS). For a projected CRS such as UTM or ETRS89 / UTM this is metres. For a geographic CRS (degrees) the numbers are degrees and are not meaningful as distances: reproject the raster first.

The route search itself runs on cells. A cell step of length one is scaled by the raster's cell size (`abs(transform.a)`) before it is reported, and diagonal steps are weighted by the square root of two. A route over the same terrain therefore has the same length at 1 m and at 5 m resolution, up to the discretisation error of the coarser grid.

:::{note}
Older releases (up to 0.3.2) reported route length in cell units, so lengths and costs changed with the raster resolution. The CRS-unit behaviour is described here for a source checkout of `main`; see {doc}`../getting_started/release_status`.
:::

(path-lengths-cost)=
## Cost versus length

- `total_length` is a geometric length in CRS units.
- `total_cost` is the sum of cell costs along the route, each scaled by the step length. With a DEM, or with an `objective` that has extra terms, the cost is not simply distance times cell size; see {doc}`cost_semantics`.
- On the `raster_fim` backend the route cost is the arrival time at the target, a continuous value; see {doc}`../api/eikonal_fim`.

(path-lengths-cells)=
## Cell shape

PYORPS assumes square cells when it converts cell counts into distances. Real rasters can have a cell width that differs from the cell height, or a cell size other than one. Check `raster.res` before you compare lengths across datasets, and see {doc}`known_limitations`.
