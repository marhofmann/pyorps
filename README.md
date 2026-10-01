# **PYORPS** - Python for Optimal Routes in Power Systems

[![PyPI version](https://img.shields.io/pypi/v/pyorps.svg)](https://pypi.org/project/pyorps/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyorps.svg)](https://pypi.org/project/pyorps/)
[![Documentation Status](https://readthedocs.org/projects/pyorps/badge/?version=latest)](https://pyorps.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/marhofmann/pyorps/branch/main/graph/badge.svg)](https://codecov.io/gh/marhofmann/pyorps)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1403393e905b419b8b3a5865cebe85a7)](https://app.codacy.com/gh/marhofmann/pyorps/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marhofmann/pyorps/main?filepath=examples)


**PYORPS** is an open-source tool designed to automate route planning for underground cables in power systems. It uses high-resolution raster geodata to perform least-cost path analyses, optimizing routes based on economic and environmental factors.

## Overview

Power line route planning is a complex and time-consuming process traditionally neglected in early grid planning. **PYORPS** addresses this by:

- Finding optimal routes between connection points using least-cost path analysis
- Supporting high-resolution raster data for precise planning
- Considering both economic costs and environmental constraints
- Allowing customization of neighborhood selection and search parameters
- Enabling easy integration into existing planning workflows

While tailored for distribution grids, it can be adapted for various infrastructures, optimizing routes for cost and environmental impact.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="https://raw.githubusercontent.com/marhofmann/pyorps/refs/heads/main/docs/source/_static/images/pyorps_planning_results_21_targets_22_5deg_1mxm.png" alt="ex." width="100%"/><br>
      <sub>
        <b>Figure 1:</b> Parallel computation of 21 paths from single source to multiple targets.<br>
        332 s total runtime on laptop with Intel(R) Core(TM) i7-8850H CPU @ 2.6 GHz and 32 GB memory
      </sub>
    </td>
  </tr>
</table>


## What's new in 0.4.0

- **Route length in CRS units.** Lengths and costs no longer change with the raster resolution (breaking change for
  results that were reported in cell units).
- **Several metrics at once.** `CostAssumptions(metrics=...)`, `MetricStack`, `Objective` and DEM-based slope for
  multi-objective routing.
- **GPU backends.** `raster_gpu` (persistent delta-stepping kernels) and `raster_fim` (eikonal fast-iterative-method
  solver, raw `.gpur` rasters); install with `pip install pyorps[gpu]`.
- **Cost fields and search sessions.** Solve one field once and price many endpoint pairs.
- **Corridor graphs and tower fields.** Shared trenches for many routes; precomputed fields for overhead lines.
- **Interactive GUI.** `pip install pyorps[gui]`, then `python -m pyorps.gui`.
- **Route simplification.** `simplify` option of `PathFinder.find_route`.

Everything is listed with its status (`stable` or `experimental`) in the
[release status](https://pyorps.readthedocs.io/en/latest/getting_started/release_status.html) and the
[changelog](https://pyorps.readthedocs.io/en/latest/reference/changelog.html).

### Planning substation sites with cost fields

Instead of routing from every candidate site to every terminal, the siting research code solves one cost field per
fixed terminal (grid connection points and turbines) and then prices each candidate substation site by lookup.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="https://raw.githubusercontent.com/marhofmann/pyorps/refs/heads/main/docs/source/_static/images/substation_siting_method.gif" alt="Animation: ten cost fields are settled on a cost raster and candidate substation sites are priced by lookup" width="100%"/><br>
      <sub>
        <b>Figure 2:</b> Method illustration on public geodata: ten cost fields rooted at the fixed terminals
        (<b>+</b> turbines, <b>&#9632;</b> grid connection points), then four illustrative candidate sites (<b>&#9675;</b>),
        one west, north, east and south of the wind farm, are read off the fields.<br>
        The siting packages (<code>pyorps.siting</code>, <code>pyorps.collector</code>, <code>pyorps.costmodel</code>,
        <code>pyorps.certify</code>) are research code: shipped, but not part of the stable API.
      </sub>
    </td>
  </tr>
</table>

How the workflow in Figure 2 runs:

- Start from a cost raster in which every cell has a price per metre of trench and forbidden cells are black.
- Solve one cost field per fixed terminal (seven turbines and three grid connection points). Each field is kept in memory, and the raster is not searched again.
- Read a route out of a field by following the predecessor chain from a candidate site back to its root. No new search runs.
- Move the candidate site and read all ten routes again (seven medium-voltage feeders and three high-voltage connections). The cheapest of each kind is the one that counts.
- Ten searches therefore answer any number of candidate sites, because the search is rooted at what is fixed and what moves is looked up.

## Features

- **Flexible Input Data**: Use local raster files directly or create custom rasterized geodata from  WFS services or 
  local vector files
- **Customizable Costs**: Define terrain-specific cost values based on installation expenses or environmental impacts
- **Multiple Path Finding**: Calculate optimal routes between multiple sources and targets in parallel
- **Performance Optimization**: Control search space parameters to balance accuracy and computational efficiency
- **Environmental Consideration**: Add cost modifiers for nature reserves, water protection zones, and other sensitive 
  regions
- **GIS Integration**: Export results as GeoJSON for further analysis in GIS applications
- **Multiple Metrics**: Route on cost, slope or other metrics at once with `metrics=` and `Objective`
- **GPU Backends**: `raster_gpu` and `raster_fim` for very large rasters (extra `gpu`)
- **Interactive GUI**: Draw an area, load data from WFS services, edit costs and compare routes (extra `gui`)


## Quick Start

Here's a minimal example to get you started:

```python
from pyorps import PathFinder

# Define a file path to a raster file! 
raster_path = r"<PATH>\<TO>\<YOUR>\<RASTER_FILE>.tiff" 

# Define your source and target coordinates (must be in the same CRS)
source = (..., ...)
target = (..., ...)

# Create PathFinder instance
path_finder = PathFinder(
    dataset_source=raster_path,
    source_coords=source,
    target_coords=target,
)

# Find optimal route
path_finder.find_route()

# Visualize results
path_finder.plot_paths()

# Export to GeoJSON
path_finder.save_paths(r"<PATH>\<TO>\<YOUR>\<RESULTS>.geojson" )
```

Please check out the [example](https://github.com/marhofmann/pyorps/blob/main/examples/create_rasterized_geodata.ipynb)
for creating and setting up a dedicated raster dataset for your planning task.

## Binder - Run Examples

You can quickly start testing the functionalities of **PYORPS** using Binder. 
Click the badge below to launch an interactive environment where you can run the 
example notebooks directly in your browser.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marhofmann/pyorps/master?filepath=examples)

This Binder connection allows you to explore the examples provided in the `examples` 
directory without needing to install anything on your local machine. It's a great way 
to get a hands-on experience with **PYORPS** and see how it can optimize route planning for 
power systems.

## Installation

You can easily install **PYORPS** from the Python Package Index (PyPI) using pip or other 
package management tools.

#### Using pip

You can install the base package using pip:

```bash
pip install pyorps
```

This command will install the core functionality of **PYORPS** along with its essential dependencies, including:

- [NumPy](https://github.com/numpy/numpy)
- [Pandas](https://github.com/pandas-dev/pandas)
- [GeoPandas](https://github.com/geopandas/geopandas)
- [Numba](https://github.com/numba/numba)
- [Rasterio](https://github.com/rasterio/rasterio)

> **macOS 14 or older with Python 3.12 or newer:** rasterio 1.5 ships macOS wheels for macOS 15 only, so pip would try to build it
> from source (this needs GDAL). Install `pip install "rasterio<1.5"` first.

#### Optional Dependencies

**PYORPS** offers several optional dependencies that enhance its functionality. You can install these extras by specifying them in square brackets:

- **Examples**: To include all dependencies for examples and case-studies:
  ```bash
  pip install pyorps[examples]
  ```

- **Case Studies**: To include case study scripts:
  ```bash
  pip install pyorps[case_studies]
  ```

- **Development and Testing**: To include testing tools and the tests directory:
  ```bash
  pip install pyorps[dev]
  ```

- **Interactive GUI**: Dash-based route-planning workbench (`python -m pyorps.gui`):
  ```bash
  pip install pyorps[gui]
  ```

- **GPU backends**: CuPy for `raster_gpu` and `raster_fim` (CUDA 12; `gpu-full` adds cuGraph and cuDF):
  ```bash
  pip install pyorps[gpu]
  ```

- **Other graph libraries**: Rustworkx, iGraph, NetworkX and NetworKit backends:
  ```bash
  pip install pyorps[graph]
  ```

- **Route simplification**: `pip install pyorps[simplify]`

- **Full Installation**: To install all optional dependencies at once:
  ```bash
  pip install pyorps[full]
  ```


## How It Works

**PYORPS** performs route planning through these key steps:

1. **Data Preparation**: Categorizes continuous land use data using GeoPandas
2. **Rasterization**: Converts categorized geodata to raster format with cost values using Rasterio
3. **Graph Creation**: Transforms rasterized dataset into a graph structure using NetworKit
4. **Path Analysis**: Performs least-cost path analysis on the graph to find optimal routes
5. **Result Export**: Exports results in GeoJSON format for further use in GIS applications

The process can be configured with different neighborhood selections (R0-R3) and search space parameters to balance accuracy and performance.

## Use Cases

- **Distribution Grid Planning**: Optimize new underground cable connections to increase grid capacity
- **Grid Integration of Renewable Energies**: Determine optimal point of common coupling (PCC) and find the most 
  economical route for grid integration
- **Environmental Impact Reduction**: Route underground cables to minimize impact on protected areas
- **Cost Optimization**: Balance construction costs with environmental considerations and other aspects
- **General Infrastructure Planning**: Adapt for other linear infrastructure planning tasks (e.g. fiber optic cables or
  pipes)


## Technical Details

### Search Space Control: Buffering & Masking

Efficient path finding in large rasters requires limiting the search space. **PYORPS** provides:

- **Buffering**: Define a buffer (in meters) around the source/target line or polygon, restricting the raster window and graph to relevant areas. Buffer size can be set manually or estimated automatically based on terrain complexity.
- **Masking**: Apply geometric masks (e.g., polygons, convex hulls) to further restrict the area considered for 
  routing. Pixels outside the mask are not considered.

This dramatically reduces memory and computation time, especially for high-resolution data.

<table>
  <tr>
    <td align="center" width="100%">
      <img src="https://raw.githubusercontent.com/marhofmann/pyorps/refs/heads/main/docs/source/_static/images/buffer_600.png" alt="search spaces" 
width="100%"/><br>
      <sub><b>Figure 3:</b> Various optimal paths for different search spaces on rasterised geodata with 1 m² resolution</sub>
    </td>
  </tr>
</table>


### Neighborhoods: Fine-Grained Connectivity

The raster-to-graph conversion supports customizable neighborhood definitions:

- **Predefined and tested neighborhoods**:  
  - `R0` (4-connectivity), `R1` (8-connectivity), `R2` (16-connectivity), `R3`  (32-connectivity),
- **Custom neighborhoods**:  
  - Specify arbitrary step sets for advanced use cases or anisotropic cost surfaces.
- **High-order neighborhoods**:  
  - For `k > 3`, arbitrary neighborhoods are supported, enabling long-range or non-local connections.

This allows you to balance accuracy (following real-world paths) and performance (sparser graphs).

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/marhofmann/pyorps/refs/heads/main/docs/source/_static/images/R3-complete.PNG" 
alt="R3 complete" width="79%"/><br>
      <sub><b>Figure 4a:</b> Steps for neighbourhoods R0 (blue), R1 (green), R2 (yellow), and R3 (red)</sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/marhofmann/pyorps/refs/heads/main/docs/source/_static/images/intermediate_steps.PNG" 
alt="intermediates" width="90%"/><br>
      <sub><b>Figure 4b:</b> Intermediate elements Ik for selected edges of vertex v<sub>5,5</sub>.</sub>
    </td>
  </tr>
</table>


### Data Input: Raster & Vector, Local & Remote

**PYORPS** is agnostic to data source and format:

- **Raster data**:  
  - Directly use high-resolution GeoTIFFs or similar formats (tested up to 0.25 m² per pixel).
- **Vector data**:  
  - Shapefiles, GeoJSON, GPKG, or remote WFS layers (e.g., land registry, nature reserves, water protection zones).
- **Hybrid workflows**:  
  - Rasterize vector data with custom cost assumptions, overlay multiple datasets, and apply complex modifications.

All data is internally harmonized to a common CRS and resolution.

### Cost Assumptions: From Simple to Complex

Routing is driven by a **cost raster**. **PYORPS** supports:

- **Simple costs**:  
  - Assign a single cost per land use class or feature.
- **Hierarchical/multi-attribute costs**:  
  - Use CSV, Excel, or JSON files to define costs based on multiple attributes (e.g., land use + soil type).
- **Dynamic overlays**:  
  - Overlay additional datasets (e.g., protected areas) with additive or multiplicative cost modifiers, or set areas as forbidden.
- **Custom logic**:  
  - Apply buffers, ignore fields, or use complex rules for cost assignment.


#### Example: Cost Assumptions Table

| land_use                            | category                   | cost  |
|--------------------------------------|----------------------------|-------|
| Forest                              | Coniferous                 | 365   |
| Forest                              | Mixed Deciduous/Coniferous | 402   |
| Forest                              | Deciduous                  | 438   |
| Forest                              |                            | 365   |
| Road traffic                        | State road                 | 196   |
| Road traffic                        | Federal road               | 231   |
| Road traffic                        | Highway                    | 267   |
| Path                                | Footpath                   | 107   |
| Agriculture                         | Arable land                | 107   |
| Agriculture                         | Grassland                  | 107   |
| Agriculture                         | Orchard meadow             | 139   |
| Flowing water                       |                            | 186   |
| Sports, leisure, recreation area     |                            | 65535 |
| Standing water                      |                            | 155   |
| Square                              | Parking lot                | 178   |
| Square                              | Rest area                  | 178   |
| Rail traffic                        |                            | 415   |
| Residential building area            |                            | 65535 |
| Industrial and commercial area       |                            | 65535 |
|...                                  |...                          |...    |

**How to use:**  
- Use as a CSV file with columns (e.g. `land_use`, `category`, `cost`)  
- The `cost` value can be interpreted as €/m or as a relative score.
- Use uint16 and set 65535 to indicate forbidden areas.


### Rasterization: High-Resolution, Multi-Layer, and Overlay

- **High-resolution rasterization**:  
  - Rasterize vector data at arbitrary resolutions (tested up to 0.25 m² per pixel) using [Rasterio](https://rasterio.readthedocs.io/).
- **Buffering and overlays**:  
  - Apply geometric buffers to features before rasterization.
  - Overlay multiple datasets, each with its own cost logic.
- **Selective masking**:  
  - Mask out fields or regions, set forbidden values, or combine multiple masks.

### Supported Data Types

- **Vector formats**: Shapefile, GeoJSON, GPKG, GML, KML, WFS (remote)
- **Raster formats**: GeoTIFF, IMG, JP2, BIL, DEM, in-memory numpy arrays
- **Data sources**: Land registry, nature reserves, water protection areas, custom user data


### Various Graph Backends & Path-Finding Algorithms

**PYORPS** supports multiple high-performance graph libraries as interchangeable backends for path finding:

- **Cython**: In PYORPS, specialized Cython implementations of path-finding algorithms 
 are integrated for efficient processing of raster data. This backend offers the 
 highest performance, as it operates directly on raster structures without requiring conversion to graph representations.
- **[NetworKit](https://networkit.github.io/)**: Fast C++/Python library for large-scale network analysis.
- **[Rustworkx](https://qiskit.org/documentation/rustworkx/)**: Pythonic, Rust-powered graph algorithms.
- **[NetworkX](https://networkx.org/)**: Widely-used, pure Python graph library.
- **[iGraph](https://igraph.org/python/)**: Efficient C-based graph library.
- **GPU** (experimental, extra `gpu`): `raster_gpu` (delta-stepping on the GPU) and `raster_fim` (eikonal fast-iterative-method solver).

You can select the backend by name via the `graph_api` parameter in `PathFinder` (default `"cython"`; there is no automatic choice by problem size). Each backend exposes a unified interface for shortest path computation, supporting:

- **Δ-stepping**: Parallel path-finding algorithm - highest performance on multicore 
  CPUs
- **Dijkstra**: Robust and efficient.
- **A***: Heuristic-based, faster for spatial graphs with good heuristics.
- **Bellman-Ford**: Handles negative weights (where supported).
- **Bidirectional Dijkstra**: Available in some backends for further speedup.

## Documentation

The documentation for **PYORPS**, including detailed explanations and usage instructions, can be found on 
https://pyorps.readthedocs.io. It is organised into concepts (cost semantics, neighborhoods, search space, path
lengths and units, known limitations), one API page per feature and a
[release status](https://pyorps.readthedocs.io/en/latest/getting_started/release_status.html) page. Every page
states whether it is `stable` or `experimental`. For AI assistants and tools, the built site also ships
`llms.txt`, `llms-full.txt` and `api-index.json` at its root.
Examples demonstrating the functionality of **PYORPS**, along with practical use cases, are included as jupyter notebooks
in the [examples directory](https://github.com/marhofmann/pyorps/blob/main/examples).

## Contributing

Contributions are welcome! If you want to contribute, please check out the [**PYORPS** contribution guidelines](https://github.com/marhofmann/pyorps/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](https://github.com/marhofmann/pyorps/blob/main/LICENSE).

## Citation

If you use **PYORPS** in your research, please cite:

```
Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for Automated Power Line Routing', CIRED 
2025 - 28th Conference and Exhibition on Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
```

## Contact

For questions and feedback, please open an issue on our GitHub repository.