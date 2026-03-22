# 🏛️ Architecture

## Data Flow

The PYORPS routing pipeline transforms geospatial input data into optimized power line
routes through a series of well-defined stages:

```text
Input (raster/vector/WFS/in-memory)
  -> GeoDataset (normalize CRS/resolution)
  -> GeoRasterizer (vector -> raster, if needed)
  -> PathFinder (main routing interface)
    -> RasterHandler (extract raster window)
    -> GraphAPI (build graph from raster cells)
    -> Shortest path algorithm (Dijkstra/delta-stepping)
  -> Path / PathCollection (results, exportable)
```

```{image} ../_static/generated/architecture_flow.png
:width: 100%
```

## Package Structure

| Package | Purpose |
|---------|---------|
| `pyorps/core/` | Type definitions, cost assumptions, path results, exceptions |
| `pyorps/io/` | Data loading: GeoDataset hierarchy (Local, WFS, InMemory) |
| `pyorps/raster/` | GeoRasterizer (vector-to-raster), RasterHandler (windowing) |
| `pyorps/graph/` | PathFinder + `api/` subpackage with pluggable backends |
| `pyorps/utils/` | Performance-critical code: Cython extensions, Numba JIT |

## Key Classes

`PathFinder`
: Main user-facing class (~800 lines). Orchestrates the full pipeline from data
  loading through graph construction to path extraction and export.

`GeoDataset`
: Abstract base for all data inputs. Concrete subclasses cover local files, WFS
  services, and in-memory arrays. Factory function: `initialize_geo_dataset()`.

`GeoRasterizer`
: Converts vector data (shapefiles, GeoPackages, etc.) to cost rasters using
  configurable cost assumptions.

`RasterHandler`
: Manages raster windowing, coordinate transforms, and search space extraction.

`GraphAPI`
: Abstract base for all graph backends. Each concrete implementation builds a graph
  from the raster grid and runs shortest-path queries.

`Path` / `PathCollection`
: Dataclass results carrying geometry, total cost, and routing metrics. Exportable
  as GeoJSON, Shapefile, or GeoPackage.

## Graph Backend System

All backends implement the abstract `GraphAPI` base class. `GraphLibraryAPI` is an
intermediate base for external-library backends.

| Backend | Module | Notes |
|---------|--------|-------|
| **CythonAPI** (default) | `cython_api.py` | Fastest CPU backend; operates directly on raster, no graph object |
| NetworkitAPI | `networkit_api.py` | C++/Python hybrid |
| RustworkxAPI | `rustworkx_api.py` | Rust-backed |
| NetworkXAPI | `networkx_api.py` | Pure Python reference implementation |
| iGraphAPI | `igraph_api.py` | C-backed |

## Cython Extensions

Seven compiled C++20 extensions live in `pyorps/utils/`, organized in dependency layers:

**Foundation layer:**

`_heap.pyx`
: Binary heaps (uint32/uint64 variants), index conversion, circular buffer utilities, system resource detection.

`_raster_context.pyx`
: `RasterContext` class holding raster data, precomputed directions, cached steps, and exclude mask. Geometry and cost factor calculations.

**Unconstrained solver layer:**

`_dijkstra.pyx`
: `DijkstraSolver` class with methods for single-pair, one-to-many, many-to-many, and pairwise Dijkstra.

`_delta_stepping.pyx`
: Parallel delta-stepping with OpenMP, lock-free atomic CAS updates, standard and persistent thread pool variants.

**Constrained solver layer:**

`_constrained_context.pyx`
: State encoding for extended Dijkstra states `(cell, direction, span_bin)`, angle/neighbor precomputation, intermediate path caches, gradient caches, and catenary clearance helpers.

`_constrained_dijkstra.pyx`
: Constrained Dijkstra with automatic dense (bucket-queue) / sparse (hash-map) dispatch based on state-space size.

`_constrained_delta.pyx`
: Constrained parallel delta-stepping variants: basic, fixed-height clearance, variable-height, and lazy hash-map.

**Backward compatibility:** The old import paths (`path_core`, `path_algorithms`, `constrained_path_algorithms`) are maintained via thin Python shim files that re-export from the new modules.

Compiler directives: `boundscheck=False`, `wraparound=False`, `cdivision=True`.

Platform-specific flags:

| Platform | Flags |
|----------|-------|
| Windows (MSVC) | `/O2 /fp:fast /EHsc /openmp /std:c++20` |
| Linux (GCC) | `-O3 -ffast-math -fopenmp -std=c++20` |
| macOS (Clang) | `-O3 -ffast-math -std=c++20` |

## Type System

Cost values use `uint16` representation where the maximum value 65535 denotes
forbidden/impassable terrain. Type aliases are defined in `core/types.py`.

## Exception Hierarchy

```text
PyorpsError (base)
+-- CostAssumptionsError
|   +-- FileLoadError
|   +-- InvalidSourceError
|   +-- FormatError
+-- FeatureColumnError
|   +-- NoSuitableColumnsError
|   +-- ColumnAnalysisError
+-- WFSError
|   +-- WFSConnectionError
|   +-- WFSResponseParsingError
|   +-- WFSLayerNotFoundError
+-- RasterShapeError
+-- NoPathFoundError
+-- AlgorithmNotImplementedError
+-- PairwiseError
```
