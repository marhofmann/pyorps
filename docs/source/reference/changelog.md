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
- New warning class `AlgorithmFallbackWarning` (a `PyorpsWarning`): the search ran with another algorithm or backend than the one asked for, because the requested one cannot evaluate slope terms. It is emitted once per finder or session.
- GUI: `python -m pyorps.gui --project FILE` opens a saved project (a `project.json` or a folder that holds one) at start, in the browser and with `--desktop`. A missing or broken file prints one error line and ends with exit code 1 before a server or window starts.
- GUI: new card "Objective & terrain" in the Routes tab for plain routing: weights for cost, length and slope, extra metric layers with hard limits, a terrain model (DEM) with the slope response, 16-bit or 32-bit weights, and three presets. Its status line names a conflict before a run (for example a terrain model with a delta-stepping algorithm, which the GUI refuses with a pointer to the Algorithm field). A route computed with an objective is priced on the plain cost raster; an edit of such a route uses the same objective, terrain model and layers again.
- GUI: the constrained-routing profile is a form (23 fields and three tables, with limits and units from one table of field definitions). Form edits keep the YAML text current and keep keys that the form does not show; the YAML editor stays as "Raw YAML (advanced)".
- GUI: new card "Cost field": the cost to reach every cell from the sources or from all control points, summed or as the lowest of the points, as a raster layer, with the cheapest cell. A run that needs more memory than the limit is refused with the numbers and with what to change; a fixed point on a forbidden cell is refused (the library would move it).
- GUI: new card "Corridor graph": the shared-trench corridor graph between the control points, with map layers for sections, shared runs and nodes, a summary, and an export as GeoPackage that replaces an existing file only after confirmation.
- GUI: background runs with a progress line and a Stop button for heavy rasterize runs (more than 20 million cells), heavy constrained runs, cost fields and corridor graphs (a corridor graph cannot be stopped). Each status line says where Stop takes effect. Smaller rasterize and constrained runs stay in the foreground. `build_cost_raster` takes optional `cancel` and `progress` arguments.
- GUI: the settings of all cards are saved in the project file (`gui_settings`) and restored after "Open project", "New project" and a page reload. Older project files open unchanged.

Changed:

- Data in a geographic CRS (degrees, for example EPSG:4326) is now reprojected automatically to the local UTM zone, with one `CrsReprojectionWarning`, instead of being rejected. The raster and every route are in the UTM CRS (`GeoRasterizer.crs`): give source and target coordinates in it.
- The GitHub Actions shrink to two workflows: *Tests* (a ruff lint job, the test matrix with the `full` and `gui` extras, and a weekly run against the newest dependencies) and the release pipeline. The separate *Code Quality Check* and *Tests with the newest dependencies* workflows are gone; ruff replaces flake8 and Pylint, and the Pylint settings are removed from `pyproject.toml`. ruff is part of the `dev` and `full` extras.
- The macOS wheels are built on macOS 15 and need macOS 15 or newer (before: macOS 14), because GitHub retired its macOS 14 runners on 2 November 2026. On macOS 14, pip builds PYORPS from the source distribution.
- The `graph` and `full` extras no longer install networkit on macOS: it has wheels for macOS 26 only (and none for macOS 14), so pip would compile it. Install it by hand on macOS 26 or newer if you want the networkit backend.

- Python 3.11 is no longer supported: PYORPS requires Python 3.12 or newer. numpy 2.5, scipy 1.18 and rasterio 1.5 already require 3.12, so a Python 3.11 installation was stuck on older releases of its dependencies. Stay on PYORPS 0.4.0 if you need Python 3.11.
- The research packages raise only `PyorpsError` subclasses (`InvalidInputError`, `PyorpsRuntimeError`, `MissingKeyError`, `MissingDependencyError`, and their own `DesignError`, `LoadFlowError`, `ManifestError`). Each class still derives from the builtin it replaces, so existing `except ValueError` code keeps working.
- The packages have a fixed import order, checked by a test: the core imports no research package, `pyorps.collector` imports only the core, `pyorps.certify` and `pyorps.costmodel` may import the collector, and `pyorps.siting` imports none of the others. For this the raster helpers `drain`, `step_table`, `boundary_cells` and `EXCLUDED` moved to `pyorps.graph.drain` (the old import from `pyorps.certify.windows` still works), and `certify_voltage` and `VoltageCertificate` moved from `pyorps.collector.voltage_rv` to `pyorps.certify.voltage` (the old import path is gone).
- Country-specific values of the research packages are named constants and keyword parameters with unchanged defaults: the railway-traction markers of `verified_110kv_towers`, the collector voltage tables and the cable-class fallback of `voltage_model_from_yaml`, and the connection-point and protection defaults of `grid_rules_from_yaml` and `plant_model_from_yaml`.
- `GeoRasterizer`: a `bounding_box` given together with data in a geographic CRS is read in the CRS of the data and reprojected with it (the raster covers the envelope of the reprojected box). Before, it was read as UTM metres, which gave an empty or wrong raster. With projected data nothing changes.
- `GeoRasterizer.rasterize(fill_value=...)` is checked against `dtype` before any work is done, and `modify_raster_from_dataset` and `modify_raster_with_geodataframe` check every cost, `forbidden_value`, `ignore_value` and multiplier: a value that is NaN, negative, fractional (integer rasters) or outside the range of the raster's data type raises `InvalidInputError` instead of wrapping around or being cut off. A product of `multiply=True` above the range still saturates at the maximum (65535 for `uint16`, the forbidden value).
- `pyorps.graph.tower_field`, `pyorps.utils.directional` and `pyorps.utils.linestring_simplify` raise `ConsistencyCheckError` (still an `AssertionError`) and `PyorpsRuntimeError` (still a `RuntimeError`) instead of the bare built-in classes.
- The Linux test jobs need no package from the Ubuntu servers any more: they use the compiler of the runner image (gcc brings its own OpenMP runtime); `apt-get` is only a fallback, with retries. `setup.py` regenerates the C++ sources when the installed Cython or numpy differs from the one of the last build (sources generated next to a newer numpy do not compile against an older one).
- A DEM without an objective (`dem=` without `objective=`) now means the same on every backend: 3-D edge lengths in metres, and the cost multiplied by the slope penalty `exp(2 s^2)` (s = rise over horizontal run, in 0.25 % bins up to 200 %, so the largest penalty is `exp(8)`, about 2981). This equals `objective={"cost": 1}` with `GradientOptions(multiplier="exponential")`. Before, `cython` and `raster_gpu` ignored a bare DEM and returned the flat route. `Path.total_length` is the 3-D length in this case.
- Slope terms with the delta-stepping algorithms (`find_route` and `SearchSession`), in a `SearchSession` on `raster_gpu`, and for a cost field on `raster_gpu`: the search runs with the Cython Dijkstra and emits one `AlgorithmFallbackWarning`. Before, a session returned the flat route without a word, and `find_route` with an objective raised. An explicit delta-stepping or GPU algorithm for a cost field with slope terms still raises `FeatureNotImplementedError`.
- `graph_api="raster_fim"` with a DEM falls back to the Cython Dijkstra with an `AlgorithmFallbackWarning` (its eikonal solution of the terrain problem is not the same answer as the discrete optimum of the other backends). The backend class `RasterFIMAPI` itself is unchanged.
- Refusals around `weight_precision="float32"` that used to surface as a raw `ValueError` or buffer error (no objective, a bare DEM, a float32 session on a tree that only takes `uint16`, an unknown value) are `InvalidInputError` with a message that names the setting. A vector dataset as `dem`, and `dem_data` without slope tables on the `cython` and `raster_gpu` backend classes, are refused instead of being ignored. With `weight_precision="float32"`, voids in the DEM are `+inf` (forbidden), not 65535.
- GUI: the "Go fix" button of a notice opens a closed card around the control it points at. A page reload shows the cost table, the modifier list and the coverage text of the loaded project instead of empty tables. Warnings of a background routing run reach the notices (before, they went to the console only). A failed save of a raster no longer leaves a half-written file.
- GUI: a fallback of the library to Dijkstra or to the `cython` backend, and a start or end point that was moved because the terrain model has no elevation there, each show a plain notice instead of the generic warning box. Cost-field layers are not offered under "Route on raster". After `--project`, "Open project" or "Save", a page reload shows "opened: <project file>". While the result of a finished background run is added to the project, the status line says so, Stop is hidden and no other run can start.
- Packaging: the GPU kernel headers `pyorps/utils/kernels/*.cuh` are part of the wheel and of the source distribution (before, only the `*.cu` files were, so the kernels that include the headers could not be built from an installed copy). `pyorps.gui.sections` and `pyorps.utils.kernels` are listed packages; two tests keep the package list and the data-file patterns complete.

Fixed:

- GUI: a raster layer's tile client no longer keeps a rasterio environment open in the calling thread. Before, an action that read a raster could fail now and then with "No GDAL environment exists" (seen with Python 3.14).
- Saving an infrastructure profile to a `.yaml` file without PyYAML no longer leaves an empty file behind.
- The Cython Dijkstra's three predecessor walks (`_reconstruct_path`, `_walk_to_seed`, `MultiSourceSolver.path_to_root`) are bounded by the number of cells and raise `PyorpsRuntimeError` on a predecessor cycle instead of looping without end. Results are unchanged. A new four-thread regression test checks that every delta-stepping entry point of the public API returns optimal routes on rasters with many ties.
- `ConstrainedPathFinder.find_route` resets the cached `path_gdf` when it adds a route. Before, the attribute still showed only the earlier routes after a second call.
- Graph-library backends (networkx, networkit, igraph, rustworkx) priced a bare DEM with the horizontal run measured in cells instead of metres, so slopes were overstated by the cell size (a 10 % slope on 10 m cells counted as 100 %), and capped the penalty at 10 000 and each edge at 65 535. They now use the formula above. **Routes of these backends over a bare DEM change at cell sizes other than 1 m**; at 1 m only steep or formerly capped edges are priced differently.
- A `SearchSession` on a graph-library backend searched only half of the neighbourhood steps and returned routes that cost clearly more than `find_route` (8.8 % on a test raster), with or without a DEM, and it ignored slope terms. A session route now costs the same as `find_route` on every backend.
- A `SearchSession` that rebuilds its window keeps every setting of its `PathFinder` (the DEM, `metric_layers`, `steps`, `weight_precision`, `graph_api_options`, cost assumptions, dataset overlays and raster read options). Before, a rebuild dropped the DEM, and with `metric_layers` it raised "unknown metric layer".
- A DEM in another coordinate system than the cost raster, or one that does not overlap the search window, raises `InvalidInputError` that names both coordinate systems or both extents. Before, it failed with rasterio's "Number of columns or rows must be non-negative". A DEM given as `(array, crs, transform)` keeps its own CRS and transform.
- Cells that a DEM does not cover, and the nodata cells of a DEM file, are forbidden cells, as documented. Before, they were read as elevation 0 (or as the nodata number), which put a wall of false slope at the edge of the DEM.
- A source or target on a cell without DEM elevation is moved to the nearest valid cell on the first route already, with a warning that names the DEM. Before, the first route raised `NoPathFoundError` and later routes moved the point without saying why. The point-correction warnings are attributed to the caller's line.
- A metric layer (file or array with its own transform) that covers the cost raster only partly, or has nodata or NaN cells, was read as 0 there without a word. It still counts as 0, so routes do not change, but `PathFinder` warns once per layer with the layer name and the number of cells. With `hard_max` or `hard_min` set, such cells are forbidden (before, they passed `hard_max` and failed `hard_min` by accident of the stored 0). A nodata value of 65535 keeps meaning "forbidden". `reproject_to_grid` returns NaN where the source has no data, as its docstring said, and takes `src_nodata`.
- `{"derive": "slope_from_dem"}` over a DEM that covers the grid only partly forbids the uncovered cells, like the finder's own DEM, instead of treating them as sea level.
- With a search buffer below half a cell, or a point on a cell corner, the search window could leave out the cell of a given point, and the route stopped with "Point ... lies outside the search window". The window now always contains the cell of every given point inside the raster; windows that already did are unchanged. `RasterHandler.window_from_bounds` has an optional `points=` argument.
- `dem_kwargs={"gradient_function": ...}` with a DEM given as a file raised `TypeError`, because the key was passed on to the raster read.
- An `AlgorithmFallbackWarning` that is turned into an error (`warnings.simplefilter("error")`) no longer makes the next call silent; a session's fallback warning is attributed to the caller's line.
- GUI: a raster layer releases its tile client, and with it the open raster file, when the layer is removed or replaced, when a project is cleared, opened or closed, and at exit. Before, the `.tif` stayed locked on Windows. A release that fails is logged and shown as a warning.
- GUI: the page fits the window on every tab; hidden form labels no longer make the document taller than the window, and the tab content scrolls in its own pane.
- GUI: the grids use the `rowSelection` object of AG Grid 32.2 and newer, so the browser console shows no deprecation warning. Selection behaves as before.
- GUI: the result of a background run (plain routing included) could be lost when adding it to the project took longer than the poll interval: the browser dropped that answer and the poll stopped, so the layer was in the project but not on the page. Results are kept on the server until the page has confirmed them and reach the page exactly once, also after a reload.
- GUI: a grid far below the visible part of a tab kept its old content after an update (AG Grid 35 lets the browser skip rendering grids outside the window), so the routes table could show "No Rows To Show" after an edit. Grids now fit their columns when they really become visible (also inside a card that was closed, or a tab that was hidden), no column is squeezed below 60 px, and the console warning about a grid of zero width is gone.
- GUI: stopping the shared tile server at exit no longer waits without limit for idle connections.
- `pyorps.webviz`: `RouteViewer.shutdown()` also closes the raster reader of each tile client. Before, the reader was closed only while the interpreter shut down, and Python printed an empty "Error in sys.excepthook" at exit (seen with dash 4.5).

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
