---
title: "PYORPS Documentation"
summary: "Overview of PYORPS, least-cost power line routing on raster geodata, with a status table and links."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps"
api: []
---
# PYORPS Documentation

**Python for Optimal Routes in Power Systems**

```{image} _static/images/pyorps_planning_results_21_targets_22_5deg_1mxm.png
:alt: PYORPS routing results
:class: hero-image
:width: 100%
```

```{image} _static/generated/dijkstra_wavefront.gif
:width: 100%
:alt: Dijkstra wavefront expansion animation
```

PYORPS is an open-source tool for automated power line routing using least-cost path analysis on high-resolution raster geodata. It supports flexible geospatial data input, customizable cost assumptions, multiple graph backends, and high-performance Cython and GPU-accelerated algorithms.

---

::::{grid} 2 2 4 4
:gutter: 3

:::{grid-item-card} Quick Start
:link: getting_started/quickstart
:link-type: doc

Get routing in 6 lines of code.
:::

:::{grid-item-card} Data Input
:link: api/geo_dataset
:link-type: doc

Raster, vector, WFS, and in-memory data.
:::

:::{grid-item-card} Path Finding
:link: api/path_finder
:link-type: doc

Single, multi-source/target, pairwise modes.
:::

:::{grid-item-card} API Reference
:link: reference/api
:link-type: doc

Full class and function documentation.
:::

::::

---

(index-status)=
## What is available where

Each page states its status. `stable` pages describe the API that is kept; `experimental` pages describe features in the same release (`pip install pyorps`, 0.4.0) that may still change. See {doc}`getting_started/release_status`.

| Area | Status | Page |
|---|---|---|
| Routing, data input, rasterization, cost assumptions, backends | stable | {doc}`api/path_finder`, {doc}`api/geo_dataset`, {doc}`api/geo_rasterizer`, {doc}`api/cost_assumptions`, {doc}`api/graph_backends` |
| Route simplification | stable | {doc}`api/path_finder` |
| Multi-metric objectives, cost fields, corridor graphs, tower fields | experimental | {doc}`api/metric_stack_objective`, {doc}`api/cost_fields`, {doc}`api/corridor_graphs`, {doc}`api/tower_fields` |
| Thin forbidden features, eikonal FIM, constrained routing, GUI | experimental | {doc}`api/thin_forbidden_features`, {doc}`api/eikonal_fim`, {doc}`api/constrained_path_finder`, {doc}`api/gui` |

Machine-readable versions of these docs: `llms.txt`, `llms-full.txt` and `api-index.json` at the root of the built site.

(index-citation)=
## Citation

If you use PYORPS in your research, please cite:

> Hofmann, M., Stetz, T., Kammer, F., Repo, S.: *PYORPS: An Open-Source Tool for Automated Power Line Routing.* CIRED 2025 — 28th Conference and Exhibition on Electricity Distribution, Geneva, Switzerland.

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting_started/installation
getting_started/quickstart
getting_started/release_status
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/cost_semantics
concepts/search_space
concepts/neighborhoods
concepts/path_lengths_and_units
concepts/known_limitations
```

```{toctree}
:maxdepth: 2
:caption: API
:hidden:

api/path_finder
api/path_results
api/visualization
api/geo_dataset
api/geo_rasterizer
api/thin_forbidden_features
api/cost_assumptions
api/raster_handler
api/graph_backends
api/eikonal_fim
api/metric_stack_objective
api/cost_fields
api/corridor_graphs
api/tower_fields
api/constrained_path_finder
api/gui
api/exceptions
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/architecture
reference/api
reference/contributing
reference/changelog
reference/license
reference/citation
```
