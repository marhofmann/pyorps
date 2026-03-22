# 🏗️ Graph Backends

PYORPS uses a pluggable backend architecture. All backends implement the abstract `GraphAPI` base class (`pyorps.graph.api.graph_api`), ensuring a consistent interface regardless of the underlying implementation. `GraphLibraryAPI` provides an intermediate base for backends that construct explicit graph objects from raster data.

## Backend Comparison

| Backend | Class | Type | Install | Notes |
|---------|-------|------|---------|-------|
| Cython (default) | `CythonAPI` | Direct raster | Built-in | Fastest. No graph construction. |
| NetworKit | `NetworkitAPI` | Graph library | `pip install pyorps[graph]` | C++/Python hybrid |
| Rustworkx | `RustworkxAPI` | Graph library | `pip install pyorps[graph]` | Rust-backed |
| NetworkX | `NetworkxAPI` | Graph library | `pip install pyorps[graph]` | Pure Python reference |
| iGraph | `IGraphAPI` | Graph library | `pip install pyorps[graph]` | C-backed |
| RasterGPU | `RasterGPUAPI` | GPU direct | `pip install pyorps[gpu]` | CUDA delta-stepping |

## Usage

Select a backend by passing the `graph_api` parameter to `PathFinder`:

```{code-block} python
from pyorps import PathFinder

# Default (Cython) -- best CPU performance
pf = PathFinder(..., graph_api="cython")

# Switch backend
pf = PathFinder(..., graph_api="networkit")
pf = PathFinder(..., graph_api="rustworkx")
pf = PathFinder(..., graph_api="networkx")
pf = PathFinder(..., graph_api="igraph")
pf = PathFinder(..., graph_api="raster_gpu")
```

## Backend Details

### CythonAPI (Default)

The Cython backend operates directly on raster data without constructing a graph object in memory. This eliminates graph construction overhead and minimizes memory usage. It supports both Dijkstra and delta-stepping algorithms.

This is the recommended backend for most use cases.

### Library Backends

Library backends (`NetworkitAPI`, `RustworkxAPI`, `NetworkxAPI`, `IGraphAPI`) construct an explicit graph from the raster, then delegate shortest-path computation to the respective library. This adds graph construction overhead but provides access to a wider range of algorithms (A\*, Bellman-Ford, bidirectional Dijkstra).

Use library backends when you need:

- A specific algorithm not available in CythonAPI
- Direct access to the graph object for custom analysis
- Compatibility with an existing graph library workflow

### RasterGPU

The GPU backend runs a persistent cooperative kernel directly on raster data in GPU memory. See {doc}`../experimental/gpu` for details.

## Selection Guide

- **Best CPU performance**: CythonAPI (default). No graph construction, lowest overhead.
- **Need A\* or Bellman-Ford**: Use a library backend (NetworkX, Rustworkx, NetworKit, or iGraph).
- **Maximum throughput on large rasters**: RasterGPU for 4--20x speedup over Cython.
- **Prototyping and debugging**: NetworkX is pure Python and easiest to inspect.

```{image} ../_static/generated/backend_benchmark.png
:alt: Benchmark comparison of graph backends
:width: 100%
:align: center
```
