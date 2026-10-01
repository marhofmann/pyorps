---
title: "Release Status"
summary: "What pip install gives today, what is on main and what needs a source checkout."
status: stable
since: "0.3.2"
available_in: pypi
module: "pyorps"
api: []
---
# Release Status

This page tells you what you get from each way of installing PYORPS. Every page of this documentation carries a status banner and a `status` / `available_in` header that say the same thing.

(release-status-legend)=
## Status labels

| Label | Meaning |
|---|---|
| `stable` | In a PyPI release; the API is kept. |
| `experimental` | In a release; may change. |
| `unreleased` | Only on GitHub `main` or in a source checkout. |

| `available_in` | Where you can get it |
|---|---|
| `pypi` | `pip install pyorps` (latest release, 0.3.2). |
| `main` | The `main` branch on GitHub, not released yet. |
| `source` | A source checkout of the development tree only. |

(release-status-pypi)=
## `pip install pyorps` (0.3.2)

The release contains routing on raster cost surfaces: {doc}`../api/path_finder`, the {doc}`../api/geo_dataset` hierarchy, {doc}`../api/geo_rasterizer`, {doc}`../api/cost_assumptions`, the {doc}`../api/raster_handler` and the CPU and library backends in {doc}`../api/graph_backends`. Where a page of these describes a newer option, the option is marked in the page.

(release-status-main)=
## GitHub `main`

`main` is release 0.3.2 plus path simplification (`simplify` in `PathFinder.find_route`, see {doc}`../api/path_finder`). It is not in the 0.3.2 wheel.

(release-status-source)=
## Source checkout (everything else)

Install from a checkout with `pip install -e .` (see {doc}`installation`) to get the features that are in neither the wheel nor `main` yet:

| Area | Page |
|---|---|
| Route length in CRS units | {doc}`../concepts/path_lengths_and_units` |
| Several metrics, `metrics=`, `Objective`, DEM slope | {doc}`../api/cost_assumptions`, {doc}`../api/metric_stack_objective` |
| Thin forbidden feature tools | {doc}`../api/thin_forbidden_features` |
| Cost fields and search sessions | {doc}`../api/cost_fields` |
| Corridor graphs (shared trenches) | {doc}`../api/corridor_graphs` |
| Tower fields (overhead lines) | {doc}`../api/tower_fields` |
| Constrained path finding | {doc}`../api/constrained_path_finder` |
| GPU backends `raster_gpu` (newest kernels) and `raster_fim`, `.gpur` files | {doc}`../api/graph_backends`, {doc}`../api/eikonal_fim` |
| Interactive GUI | {doc}`../api/gui` |

Packages `pyorps.siting`, `pyorps.collector`, `pyorps.costmodel` and `pyorps.certify` are research code in the checkout. They are not exported, not documented and not part of any release.

See the {doc}`../reference/changelog` for the list of changes.
