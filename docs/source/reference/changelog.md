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

Added:

- Python 3.14 is supported (wheels for Windows, Linux and macOS, and a test job).
- Python 3.15 is supported (wheels for Windows, Linux and macOS, built with cibuildwheel 4.3.5, and test jobs on all three systems). The supported versions are now 3.12 to 3.15.

Changed:

- Data in a geographic CRS (degrees, for example EPSG:4326) is now reprojected automatically to the local UTM zone, with one `CrsReprojectionWarning`, instead of being rejected. The raster and every route are in the UTM CRS (`GeoRasterizer.crs`): give source and target coordinates in it.
- The GitHub Actions shrink to two workflows: *Tests* (a ruff lint job, the test matrix with the `full` and `gui` extras, and a weekly run against the newest dependencies) and the release pipeline. The separate *Code Quality Check* and *Tests with the newest dependencies* workflows are gone; ruff replaces flake8 and Pylint, and the Pylint settings are removed from `pyproject.toml`. ruff is part of the `dev` and `full` extras.
- The macOS wheels are built on macOS 15 and need macOS 15 or newer (before: macOS 14), because GitHub retired its macOS 14 runners on 2 November 2026. On macOS 14, pip builds PYORPS from the source distribution.
- The `graph` and `full` extras no longer install networkit on macOS: it has wheels for macOS 26 only (and none for macOS 14), so pip would compile it. Install it by hand on macOS 26 or newer if you want the networkit backend.

- Python 3.11 is no longer supported: PYORPS requires Python 3.12 or newer. numpy 2.5, scipy 1.18 and rasterio 1.5 already require 3.12, so a Python 3.11 installation was stuck on older releases of its dependencies. Stay on PYORPS 0.4.0 if you need Python 3.11.

(changelog-0-4-0)=
## 0.4.0 (2026-10-01)

First release since 0.3.2. Everything below is in `pip install pyorps` 0.4.0; see {doc}`../getting_started/release_status` for what is stable and what is experimental.

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
- `simplify` option of `PathFinder.find_route` for simplified route geometry ({doc}`../api/path_finder`).
- Research packages `pyorps.siting`, `pyorps.collector`, `pyorps.costmodel` and `pyorps.certify` (free siting of a collector substation with trench sharing and a proven-optimality certificate). They are shipped in the wheel but are not exported from `pyorps`, not documented and carry no stability promise.


(changelog-0-3-x)=
## 0.3.0 to 0.3.2

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
