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
:link: core_features/data_input
:link-type: doc

Raster, vector, WFS, and in-memory data.
:::

:::{grid-item-card} Path Finding
:link: core_features/path_finding
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

## Citation

If you use PYORPS in your research, please cite:

> Hofmann, M., Stetz, T., Kammer, F., Repo, S.: *PYORPS: An Open-Source Tool for Automated Power Line Routing.* CIRED 2025 — 28th Conference and Exhibition on Electricity Distribution, Geneva, Switzerland.

```{toctree}
:maxdepth: 2
:caption: 🚀 Getting Started
:hidden:

getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: 📦 Core Features
:hidden:

core_features/data_input
core_features/cost_assumptions
core_features/cost_semantics
core_features/rasterization
core_features/search_space
core_features/path_finding
core_features/neighborhoods
core_features/results
core_features/visualization
```

```{toctree}
:maxdepth: 2
:caption: 🔧 Advanced
:hidden:

advanced/graph_backends
advanced/algorithms
advanced/performance
```

```{toctree}
:maxdepth: 2
:caption: 📚 Reference
:hidden:

reference/architecture
reference/api
reference/contributing
reference/changelog
reference/license
reference/citation
```
