---
title: "Release Status"
summary: "What pip install gives today, what is on main and what needs a source checkout."
status: stable
since: "0.4.0"
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
| `experimental` | In the release; may change in a later version. |
| `unreleased` | Only on GitHub `main` or in a source checkout. |

| `available_in` | Where you can get it |
|---|---|
| `pypi` | `pip install pyorps` (latest release, 0.4.0). |
| `main` | The `main` branch on GitHub, not released yet. |
| `source` | A source checkout of the development tree only. |

(release-status-pypi)=
## `pip install pyorps` (0.4.0)

Everything documented here is in the 0.4.0 wheel. Install extras for the optional parts (`gui`, `gpu`, `gpu-full`, `simplify`, `viz`, `graph`; see {doc}`installation`).

| Area | Status | Page |
|---|---|---|
| Routing on raster cost surfaces, data input, rasterization, cost assumptions, CPU and library backends | stable | {doc}`../api/path_finder`, {doc}`../api/geo_dataset`, {doc}`../api/geo_rasterizer`, {doc}`../api/cost_assumptions`, {doc}`../api/raster_handler`, {doc}`../api/graph_backends` |
| Route length in CRS units, cost semantics, known limitations | stable | {doc}`../concepts/path_lengths_and_units`, {doc}`../concepts/cost_semantics`, {doc}`../concepts/known_limitations` |
| Several metrics, `metrics=`, `Objective`, DEM slope | experimental | {doc}`../api/metric_stack_objective` |
| Thin forbidden feature tools | experimental | {doc}`../api/thin_forbidden_features` |
| Cost fields and search sessions | experimental | {doc}`../api/cost_fields` |
| Corridor graphs (shared trenches) | experimental | {doc}`../api/corridor_graphs` |
| Tower fields (overhead lines) | experimental | {doc}`../api/tower_fields` |
| Constrained path finding | experimental | {doc}`../api/constrained_path_finder` |
| GPU backends `raster_gpu` and `raster_fim`, `.gpur` files | experimental | {doc}`../api/graph_backends`, {doc}`../api/eikonal_fim` |
| Interactive GUI (`pyorps.gui`, extra `gui`) | experimental | {doc}`../api/gui` |

The packages `pyorps.siting`, `pyorps.collector`, `pyorps.costmodel` and `pyorps.certify` are research code. They are in the wheel, but they are not exported from `pyorps`, not documented and carry no stability promise.

(release-status-main)=
## GitHub `main`

`main` is the 0.4.0 release.

(release-status-source)=
## Source checkout

Install from a checkout with `pip install -e .` (see {doc}`installation`) to get the development tree. It can be ahead of the last release; the {doc}`../reference/changelog` lists the changes.
