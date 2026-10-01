---
title: "Changelog"
summary: "User-visible changes per version, including the unreleased ones."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps"
api: []
---
# 📝 Changelog

(changelog-unreleased)=
## Unreleased

Available from a source checkout only; see {doc}`../getting_started/release_status`.

Changed:

- Route length is now reported in CRS units instead of cell units, so lengths and costs no longer change with the raster resolution ({doc}`../concepts/path_lengths_and_units`).

Added:

- `raster_fim` backend: eikonal fast-iterative-method solver on the GPU, and the raw `.gpur` raster format ({doc}`../api/eikonal_fim`).
- GPU backend `raster_gpu`: newer persistent delta-stepping kernels ({doc}`../api/graph_backends`).
- `CostAssumptions(metrics=...)`, `MetricStack`, `Objective`, `GradientOptions` and `RouteEnsemble` for multi-metric routing ({doc}`../api/cost_assumptions`, {doc}`../api/metric_stack_objective`).
- Cost fields and search sessions: solve one field once and price many endpoint pairs ({doc}`../api/cost_fields`).
- Corridor graphs that reduce many routes to shared trenches ({doc}`../api/corridor_graphs`).
- Tower fields for overhead-line siting ({doc}`../api/tower_fields`).
- Thin forbidden feature tools: detection, widening, repair-seal checks and resolution advice ({doc}`../api/thin_forbidden_features`).
- Windowed raster reads in `RasterHandler` and `LocalRasterDataset` ({doc}`../api/raster_handler`).
- Interactive GUI, `pyorps.gui` ({doc}`../api/gui`).

(changelog-on-github-main-not-in-a-release)=
## On GitHub `main`, not in a release

- `simplify` option of `PathFinder.find_route` for simplified route geometry ({doc}`../api/path_finder`).

(changelog-0-3-x)=
## 0.3.0 to 0.3.2 (latest release on PyPI: 0.3.2)

Released 2026-03-22 and 2026-03-23.

- Cython layer refactored into modules: `_dijkstra`, `_delta_stepping` (bucket-based delta-stepping), `_heap` and `_raster_context`, with a new `raster_loader` and a NumPy/Numba fallback for the traversal code.
- Overflow of source-target indices in the Cython API fixed.
- New documentation and an OSMSES tutorial (updated in 0.3.2).
- Build fixes for Linux and Windows wheels and for macOS, including OpenMP and atomic-operation portability on ARM.
- 0.3.2: code-quality clean-ups (lower complexity, ruff fixes).
- The optional extras `gpu` and `gpu-full` are declared in `pyproject.toml`. The GPU solvers themselves are not part of these releases; see {doc}`../getting_started/release_status`.

(changelog-0-2-x)=
## 0.2.1 to 0.2.3

Released 2025-09-03 to 2025-09-08.

- 0.2.1: Cython backend (`CythonAPI`) and its build setup, documentation completed, installation from PyPI corrected.
- 0.2.2: bug fixes, including the correction of maximum-cost positions, and further documentation.
- 0.2.3: bug fixes, Python 3.13 support, and a new example notebook on preparing data for distribution grid planning.

(changelog-0-1-x)=
## 0.1.0 to 0.1.4

Released 2025-06-03.

- First public releases: `PathFinder` with library graph backends (for example NetworkX), `GeoRasterizer` for vector-to-raster conversion and the `GeoDataset` hierarchy for vector and raster input.
