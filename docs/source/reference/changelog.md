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
- The research packages `pyorps.collector`, `pyorps.certify`, `pyorps.costmodel` and `pyorps.siting` have package exports (`__all__`), correct package descriptions and one API page each ({doc}`../api/collector`, {doc}`../api/certify`, {doc}`../api/costmodel`, {doc}`../api/siting`). They are experimental research code: tested on synthetic instances, the method is not complete, and the interfaces may change.
- New extra `yaml` (`pip install "pyorps[yaml]"`, also part of `case_studies` and `full`) for everything that reads or writes YAML outside the GUI: infrastructure profiles, the cost catalogue, the data manifest and the collector and voltage input files. Without PyYAML these raise `MissingDependencyError` and name the extra, instead of a bare `ModuleNotFoundError`.
- Two exception classes: `ConsistencyCheckError` (also an `AssertionError`) and `NumericalPrecisionError` (also an `ArithmeticError`).

Changed:

- Data in a geographic CRS (degrees, for example EPSG:4326) is now reprojected automatically to the local UTM zone, with one `CrsReprojectionWarning`, instead of being rejected. The raster and every route are in the UTM CRS (`GeoRasterizer.crs`): give source and target coordinates in it.
- The GitHub Actions shrink to two workflows: *Tests* (a ruff lint job, the test matrix with the `full` and `gui` extras, and a weekly run against the newest dependencies) and the release pipeline. The separate *Code Quality Check* and *Tests with the newest dependencies* workflows are gone; ruff replaces flake8 and Pylint, and the Pylint settings are removed from `pyproject.toml`. ruff is part of the `dev` and `full` extras.
- The macOS wheels are built on macOS 15 and need macOS 15 or newer (before: macOS 14), because GitHub retired its macOS 14 runners on 2 November 2026. On macOS 14, pip builds PYORPS from the source distribution.
- The `graph` and `full` extras no longer install networkit on macOS: it has wheels for macOS 26 only (and none for macOS 14), so pip would compile it. Install it by hand on macOS 26 or newer if you want the networkit backend.

- Python 3.11 is no longer supported: PYORPS requires Python 3.12 or newer. numpy 2.5, scipy 1.18 and rasterio 1.5 already require 3.12, so a Python 3.11 installation was stuck on older releases of its dependencies. Stay on PYORPS 0.4.0 if you need Python 3.11.
- The research packages raise only `PyorpsError` subclasses (`InvalidInputError`, `PyorpsRuntimeError`, `MissingKeyError`, `MissingDependencyError`, and their own `DesignError`, `LoadFlowError`, `ManifestError`). Each class still derives from the builtin it replaces, so existing `except ValueError` code keeps working.
- The packages have a fixed import order, checked by a test: the core imports no research package, `pyorps.collector` imports only the core, `pyorps.certify` and `pyorps.costmodel` may import the collector, and `pyorps.siting` imports none of the others. For this the raster helpers `drain`, `step_table`, `boundary_cells` and `EXCLUDED` moved to `pyorps.graph.drain` (the old import from `pyorps.certify.windows` still works), and `certify_voltage` and `VoltageCertificate` moved from `pyorps.collector.voltage_rv` to `pyorps.certify.voltage` (the old import path is gone).
- Country-specific values of the research packages are named constants and keyword parameters with unchanged defaults: the railway-traction markers of `verified_110kv_towers`, the collector voltage tables and the cable-class fallback of `voltage_model_from_yaml`, and the connection-point and protection defaults of `grid_rules_from_yaml` and `plant_model_from_yaml`.

Fixed:

- GUI: a raster layer's tile client no longer keeps a rasterio environment open in the calling thread. Before, an action that read a raster could fail now and then with "No GDAL environment exists" (seen with Python 3.14).
- Saving an infrastructure profile to a `.yaml` file without PyYAML no longer leaves an empty file behind.

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
