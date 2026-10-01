---
title: "Known Limitations"
summary: "Behaviour that surprises users: barriers, units, thin features, FIM costs, certificates, backend choice."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps"
api: []
---
# Known Limitations

This page lists behaviour that surprises users. Each entry says what you see, why, and what to do.

(limit-ignore-max-cost)=
## `ignore_max_cost=True` is a hard barrier

`ignore_max_cost=True` (the default) treats cells at the maximum cost as forbidden. `ignore_max_cost=False` does the opposite of what the name suggests: no cell is forbidden and the maximum cost is only very expensive (65535 per metre), so a fully enclosed target can be reached by punching through the wall.

*Do*: use `True` whenever a barrier must stay a barrier.

(limit-lengths-units)=
## Lengths are in CRS units

Route lengths are in the units of the raster's CRS (metres for a projected CRS). Releases up to 0.3.2 reported cell units. See {doc}`path_lengths_and_units`.

(limit-square-cells)=
## Cells are not always square metres

Distances are computed from the raster's cell width. A raster whose cell height differs from its width, or whose cell size is not one metre, gives distorted lengths in code that assumes square cells. Tower fields keep separate x and y spacings and refuse to work with a single spacing when the cells are anisotropic.

*Do*: resample to a square grid in a projected CRS before routing.

(limit-thin-features)=
## Thin forbidden features leak

Rasterisation burns a polygon into the cells whose centre it covers. A fence, a thin road or a water line narrower than a cell can vanish, and routes then cross it.

*Do*: rasterise at a resolution finer than the narrowest forbidden feature, widen features with `widen_thin_forbidden=True`, or check the result with `detect_forbidden_burn_defects`. See {doc}`../api/thin_forbidden_features`.

(limit-fim)=
## Eikonal (FIM) costs are continuous

The `raster_fim` backend returns the arrival time `T[target]`, which is never bit-equal to a Dijkstra cost on the same raster. Cost differences of a fraction of a percent are expected. A grade limit (`max_gradient_pct`) is only screened by this backend: a route that is returned satisfies the limit, but a failure to find one does not prove that none exists. Cells the solver did not reach hold a very large finite number, not infinity; use the readout functions that map it to infinity.

*Do*: quote `T[target]` as the cost, and use a discrete backend such as `cython` to decide grade-limit feasibility. See {doc}`../api/eikonal_fim`.

(limit-not-certificates)=
## Delta-stepping and GPU fields are not certificates

The `delta-stepping` algorithm and the GPU backends can return routes that are slightly suboptimal or differ between runs. Use Dijkstra fields whenever you need a proven-optimal result or a bound. Stored fields are kept in float32 by default, which rounds at the mantissa; tower fields are computed in float64.

(limit-backend-selection)=
## Backends are chosen by name

`graph_api` defaults to `"cython"`. PYORPS does not switch backends by grid size, and an unknown name raises `ValueError`. There is no `cugraph` backend. See {doc}`../api/graph_backends`.
