# PYORPS - Python for Optimal Routes in Power Systems

[![PyPI version](https://img.shields.io/pypi/v/pyorps.svg)](https://pypi.org/project/pyorps/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyorps.svg)](https://pypi.org/project/pyorps/)
[![Documentation Status](https://readthedocs.org/projects/pyorps/badge/?version=latest)](https://pyorps.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/yourusername/your-repo/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/your-repo)
[![Code Quality](https://github.com/yourusername/your-repo/actions/workflows/lint.yml/badge.svg)](https://github.com/yourusername/your-repo/actions/workflows/lint.yml)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/your-repo/main?filepath=notebooks%2Fexample.ipynb)


PYORPS is an open-source tool designed to automate route planning for underground cables in power systems. It uses high-resolution raster geodata to perform least-cost path analyses, optimizing routes based on economic and environmental factors.

## Overview

Power line route planning is a complex and time-consuming process traditionally neglected in early grid planning. PYORPS addresses this by:

- Finding optimal routes between connection points using least-cost path analysis
- Supporting high-resolution raster data for precise planning
- Considering both economic costs and environmental constraints
- Allowing customization of neighborhood selection and search parameters
- Enabling easy integration into existing planning workflows

While tailored for distribution grids, it can be adapted for various infrastructures, optimizing routes for cost and environmental impact.

<figure style="text-align:center;">
  <img src="https://github.com/marhofmann/pyorps/blob/master/docs/images/pyorps_planning_results_21_targets_22_5deg_1mxm.png" alt="ex."/>
  <figcaption>Figure 1: Parallel computation of 21 paths from single source to multiple targets. 332 s total runtime 
on laptop with Intel(R) Core(TM) i7-8850H CPU @ 2.6 GHz and 32 GB memory</figcaption>
</figure>

## Features

- **Flexible Input Data**: Use local raster files, WFS services, or create custom rasterized geodata
- **Customizable Costs**: Define terrain-specific cost values based on installation expenses or environmental impacts
- **Multiple Path Finding**: Calculate routes between multiple sources and targets in parallel
- **Performance Optimization**: Control search space parameters to balance accuracy and computational efficiency
- **Environmental Consideration**: Add cost modifiers for protected areas, water zones, and other sensitive regions
- **GIS Integration**: Export results as GeoJSON for further analysis in GIS applications

## Installation

```bash
pip install pyorps
```

## Quick Start

Here's a minimal example to get you started:

```python
from pyorps import PathFinder

# Define source and target coordinates
source = (472000, 5593400)
target = (472800, 5594000)

# Create PathFinder instance
path_finder = PathFinder(
    dataset_source="data/raster/sample_raster.tiff",
    source_coords=source,
    target_coords=target,
    graph_api='networkit'
)

# Find optimal route
path_finder.find_route()

# Visualize results
path_finder.plot_paths()

# Export to GeoJSON
path_finder.save_paths("data/results/route.geojson")
```

## How It Works

PYORPS performs route planning through these key steps:

1. **Data Preparation**: Categorizes continuous land use data using GeoPandas
2. **Rasterization**: Converts categorized geodata to raster format with cost values using Rasterio
3. **Graph Creation**: Transforms rasterized dataset into a graph structure using NetworKit
4. **Path Analysis**: Performs least-cost path analysis on the graph to find optimal routes
5. **Result Export**: Exports results in GeoJSON format for further use in GIS applications

The process can be configured with different neighborhood selections (R0-R3) and search space parameters to balance accuracy and performance.

## Use Cases

- **Distribution Grid Planning**: Optimize new underground cable connections
- **PV Plant Integration**: Determine optimal point of common coupling (PCC)
- **Environmental Impact Reduction**: Route infrastructure to minimize impact on protected areas
- **Cost Optimization**: Balance construction costs with environmental considerations
- **General Infrastructure Planning**: Adapt for other linear infrastructure planning tasks


## Technical Details

### Various Graph Backends & Path-Finding Algorithms

PYORPS supports multiple high-performance graph libraries as interchangeable backends for path finding:

- **[NetworKit](https://networkit.github.io/)** (default): Fast C++/Python library for large-scale network analysis.
- **[Rustworkx](https://qiskit.org/documentation/rustworkx/)**: Pythonic, Rust-powered graph algorithms.
- **[NetworkX](https://networkx.org/)**: Widely-used, pure Python graph library.
- **[iGraph](https://igraph.org/python/)**: Efficient C-based graph library.
- **(Upcoming)**: GPU-accelerated backends (e.g., cuGraph, Dask-cuGraph).

You can select the backend via the `graph_api` parameter in `PathFinder`. Each backend exposes a unified interface for shortest path computation, supporting:

- **Dijkstra** (default): Robust and efficient.
- **A***: Heuristic-based, faster for spatial graphs with good heuristics.
- **Bellman-Ford**: Handles negative weights (where supported).
- **Bidirectional Dijkstra**: Available in some backends for further speedup.


### Search Space Control: Buffering & Masking

Efficient path finding in large rasters requires limiting the search space. PYORPS provides:

- **Buffering**: Define a buffer (in meters) around the source/target line or polygon, restricting the raster window and graph to relevant areas. Buffer size can be set manually or estimated automatically based on terrain complexity.
- **Masking**: Apply geometric masks (e.g., polygons, convex hulls) to further restrict the area considered for 
  routing. Pixels outside the mask are not considered.

This dramatically reduces memory and computation time, especially for high-resolution data.

<figure style="text-align:center;">
  <img src="https://github.com/marhofmann/pyorps/blob/master/docs/images/buffer_600.png" alt="search spaces"/>
  <figcaption>Figure 2: Various optimal paths for different search spaces on rasterised geodata with 1 m² resolution</figcaption>
</figure>


### Neighborhoods: Fine-Grained Connectivity

The raster-to-graph conversion supports customizable neighborhood definitions:

- **Predefined and tested neighborhoods**:  
  - `R0` (4-connectivity), `R1` (8-connectivity), `R2` (16-connectivity), `R3`  (32-connectivity),
- **Custom neighborhoods**:  
  - Specify arbitrary step sets for advanced use cases or anisotropic cost surfaces.
- **High-order neighborhoods**:  
  - For `k > 3`, arbitrary neighborhoods are supported, enabling long-range or non-local connections.

This allows you to balance accuracy (following real-world paths) and performance (sparser graphs).

<figure style="display:inline-block; margin:10px; text-align:center;">
  <img src="https://github.com/marhofmann/pyorps/blob/master/docs/images/R3-complete.PNG" alt="R3 complete" 
width="45%"/>
  <figcaption>Figure 3: a) Steps for neighbourhoods R0 blue, R1 green, R2 yellow and R3 red</figcaption>
</figure>
<figure style="display:inline-block; margin:10px; text-align:center;">
  <img src="path/to/image2.png" alt="intermediates" width="45%"/>
  <figcaption>Figure 3: b) Intermediate elements Ik for selected edges of vertex v5,5.</figcaption>
</figure>

### Data Input: Raster & Vector, Local & Remote

PYORPS is agnostic to data source and format:

- **Raster data**:  
  - Directly use high-resolution GeoTIFFs or similar formats (tested up to 0.25 m² per pixel).
- **Vector data**:  
  - Shapefiles, GeoJSON, GPKG, or remote WFS layers (e.g., land registry, nature reserves, water protection zones).
- **Hybrid workflows**:  
  - Rasterize vector data with custom cost assumptions, overlay multiple datasets, and apply complex modifications.

All data is internally harmonized to a common CRS and resolution.

### Cost Assumptions: From Simple to Complex

Routing is driven by a **cost raster**. PYORPS supports:

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
- Use as a CSV file with columns: `land_use`, `category`, `cost`  
- The `cost` value can be interpreted as €/m or as a relative score.
- 65535 indicate forbidden areas.


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


## Documentation

For comprehensive documentation, examples, and tutorials, visit our [GitHub repository](https://github.com/marhofmann/pyorps).

## Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please feel free to reach out or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/marhofmann/pyorps/blob/master/LICENSE).

## Citation

If you use PYORPS in your research, please cite:

```
Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for Automated Power Line Routing', CIRED 
2025 - 28th Conference and Exhibition on Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
```

## Contact

For questions and feedback, please open an issue on our GitHub repository.