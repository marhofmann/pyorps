---
title: "Graph Backends"
summary: "Choose a backend by name, and the algorithms and performance levers each offers."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.graph.api"
api:
  - pyorps.graph.api.graph_api.GraphAPI
  - pyorps.graph.api.cython_api.CythonAPI
  - pyorps.graph.api.graph_library_api.GraphLibraryAPI
  - pyorps.graph.api.raster_gpu_api.RasterGPUAPI
---
# 🏗️ Graph Backends

PYORPS uses a pluggable backend architecture. All backends implement the abstract `GraphAPI` base class (`pyorps.graph.api.graph_api`), ensuring a consistent interface regardless of the underlying implementation. `GraphLibraryAPI` provides an intermediate base for backends that construct explicit graph objects from raster data.

(graph-backends-backend-comparison)=
## Backend Comparison

| Backend | Class | Type | Install | Notes |
|---------|-------|------|---------|-------|
| Cython (default) | `CythonAPI` | Direct raster | Built-in | Fastest. No graph construction. |
| NetworKit | `NetworkitAPI` | Graph library | `pip install pyorps[graph]` | C++/Python hybrid |
| Rustworkx | `RustworkxAPI` | Graph library | `pip install pyorps[graph]` | Rust-backed |
| NetworkX | `NetworkxAPI` | Graph library | `pip install pyorps[graph]` | Pure Python reference |
| iGraph | `IGraphAPI` | Graph library | `pip install pyorps[graph]` | C-backed |
| RasterGPU | `RasterGPUAPI` | GPU direct | `pip install pyorps[gpu]` | CUDA delta-stepping |
| RasterFIM | `RasterFIMAPI` | GPU direct (eikonal) | `pip install pyorps[gpu]` | Continuous least-cost solve (source checkout only) |

The backend is chosen by name only. `graph_api` defaults to `"cython"`; PYORPS does not switch backend by raster size. A `cugraph` backend does not exist (`graph_api="cugraph"` raises `NotImplementedError`).

(graph-backends-usage)=
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
pf = PathFinder(..., graph_api="raster_fim")
```

(graph-backends-backend-details)=
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

```{pyorps-status} unreleased source
```

The GPU backend runs a persistent cooperative kernel directly on raster data in GPU memory. See {ref}`raster-gpu-backend` below for details.

### RasterFIM

```{pyorps-status} unreleased source
```

`RasterFIMAPI` (`graph_api="raster_fim"`) does not search a graph. It solves the continuous eikonal equation `|grad T| = c(x)` on the cost raster with a GPU block fast-iterative method (CuPy kernels) and traces each route by steepest descent on the arrival-time field `T`. One solve from a source serves all its targets.

- **Algorithms**: `"fim"` and `"eikonal"` name the eikonal solve; `"dijkstra"` (the `PathFinder` default) is accepted as an alias for it. Any other name raises `AlgorithmNotImplementedError`.
- **Neighborhood**: `steps` are accepted and ignored, because the equation has no neighborhood.
- **Cost**: the authoritative route cost is `T[target]`. It is a continuous value and is never bit-equal to the Dijkstra cost on the same raster; the cost that `PathFinder` recomputes over the returned cell path differs slightly.
- **Grade limits**: `max_gradient_pct` is only screened. The limit is enforced outside the solver by a solve, check, mask and re-solve loop. A route that is returned satisfies the limit, but a failure to find one is not proof that none exists. Use a discrete backend (for example `cython`) as the authority on grade limits.
- **Requirements**: an NVIDIA GPU and CuPy (`pip install pyorps[gpu]`). Both are optional for PYORPS as a whole. Keep the GPU working set (chunks) below about 2 GB.

See {doc}`eikonal_fim` for the `.gpur` raw raster format that loads cost rasters into GPU memory.

(graph-backends-selection-guide)=
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

(algorithms-section)=
## Algorithms

PYORPS supports several shortest-path algorithms. The default is Dijkstra on the CythonAPI backend, which provides guaranteed optimality with excellent performance.

### Dijkstra (Default)

Single-source shortest path. Guaranteed optimal for non-negative edge weights. Available on all backends.

```{code-block} python
result = path_finder.find_route(algorithm="dijkstra")
```

### Delta-Stepping

Parallel bucket-based algorithm. On Linux, delta-stepping uses OpenMP for multi-threaded execution, making it well suited for very large rasters.

```{code-block} python
result = path_finder.find_route(algorithm="delta-stepping")
```

:::{note}
Delta-stepping is available on the CythonAPI and RasterGPU backends only. On Windows and macOS, it runs single-threaded because OpenMP is not available. For best parallel performance, use Linux or the GPU backend.
:::

### A\*

Heuristic-based search that can be faster than Dijkstra for single source-to-target pairs. Only available on library backends.

```{code-block} python
# Requires a library backend
pf = PathFinder(..., graph_api="networkx")
result = pf.find_route(algorithm="astar")
```

### Bellman-Ford

Supports negative edge weights (where the backend allows). Slower than Dijkstra but useful for specialized cost models.

```{code-block} python
pf = PathFinder(..., graph_api="networkx")
result = pf.find_route(algorithm="bellman_ford")
```

### Bidirectional Dijkstra

Searches from both source and target simultaneously, meeting in the middle. Can halve the search space for single source-target pairs.

```{code-block} python
pf = PathFinder(..., graph_api="networkit")
result = pf.find_route(algorithm="bidirectional_dijkstra")
```

### Algorithm Availability

| Algorithm | Cython | NetworKit | Rustworkx | NetworkX | iGraph | GPU |
|-----------|--------|-----------|-----------|----------|--------|-----|
| Dijkstra | Yes | Yes | Yes | Yes | Yes | Yes |
| Delta-stepping | Yes | No | No | No | No | Yes |
| A\* | No | Yes | Yes | Yes | No | No |
| Bellman-Ford | No | No | No | Yes | Yes | No |
| Bidirectional Dijkstra | No | Yes | No | Yes | No | No |

### Selection Guide

Default: Dijkstra on CythonAPI
: Best balance of performance and correctness for most routing tasks. No configuration needed.

Large rasters on Linux
: Delta-stepping on CythonAPI. OpenMP parallelism scales well with raster size and available cores.

Single source-to-target, need speed
: A\* on a library backend. The heuristic prunes the search space when source and target are far apart.

Maximum performance
: GPU delta-stepping. See {ref}`raster-gpu-backend` for setup and benchmarks.

Negative edge weights
: Bellman-Ford on NetworkX or iGraph. Required when your cost model allows negative costs.

(performance-tuning)=
## Performance Tuning

This page covers the main levers for improving PYORPS runtime and memory usage.

### Search Space Buffer

The single biggest performance lever. The `search_space_buffer_m` parameter limits the routing area to a rectangle around the source and target coordinates, dramatically reducing the number of cells processed.

```{code-block} python
path_finder = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    search_space_buffer_m=500,  # 500 m buffer around source/target extent
)
```

:::{tip}
Start with 200--600 m for short routes and 2000--5000 m for long routes. Increase the buffer if the resulting path appears suboptimal (it may be forced through high-cost areas because cheaper alternatives lie outside the buffer).
:::

### Neighborhood Size

The neighborhood parameter controls how many adjacent cells each cell connects to, which affects both path smoothness and computation time.

| Neighborhood | Connections | Relative Speed | Path Quality |
|-------------|-------------|----------------|--------------|
| R0 | 4 | Fastest | Blocky, staircase artifacts |
| R1 | 8 | Fast | Acceptable for many use cases |
| R2 (default) | 16 | Moderate | Good balance, recommended |
| R3 | 32 | Slower | Smooth paths |
| R4+ | 48+ | Slowest | Very smooth, diminishing returns |

R2 is recommended for most use cases. Only increase beyond R2 if visual smoothness matters (e.g., presentation maps). R0/R1 are useful for quick exploratory runs.

### Backend Selection

Choose the right backend for your scale:

- **CythonAPI** (default): fastest CPU backend, no graph construction overhead.
- **RasterGPU**: 4--20x faster than Cython for rasters larger than 500x500 cells. See {ref}`raster-gpu-backend`.
- **RasterFIM** (`graph_api="raster_fim"`): GPU eikonal solver. It answers many targets from one solve and has no directional (metrication) bias, but its cost is `T[target]`, which is never bit-equal to the Dijkstra cost, and grade limits are only screened. It needs CuPy and an NVIDIA GPU; keep GPU chunks under about 2 GB. Speed depends on the raster, so measure on your own data.
- **Library backends**: slower due to graph construction, but offer additional algorithms.

The backend is chosen by name; `graph_api` defaults to `"cython"` and is never switched by raster size.

For GPU runs, `save_gpu_raster()` converts a GeoTIFF once into the raw `.gpur` format, which `load_gpu_raster()` reads straight into GPU memory without decompression. See {doc}`eikonal_fim`.

### Algorithm Selection

- **Dijkstra**: general purpose, always optimal. Good for all raster sizes.
- **Delta-stepping**: better for very large rasters on Linux where OpenMP parallelism is available.
- **A\***: faster for single source-to-target when using a library backend.

See {ref}`algorithms-section` for the full comparison.

### Memory Management

PYORPS uses `uint16` cost values (0--65535) for memory efficiency. Even so, very large rasters can consume significant memory.

- PYORPS tracks `MAX_SAFE_CELLS` and issues a warning when the raster exceeds safe limits.
- Use `search_space_buffer_m` to reduce the effective raster size.
- Use a bounding box (`bbox`) or mask to spatially subset the raster before routing.

### Runtime Profiling

After calling `find_route()`, inspect the timing breakdown:

```{code-block} python
path_finder.find_route()
print(path_finder.runtimes)
# {'raster_handler': 0.12, 'graph_creation': 0.45, 'shortest_path': 1.23, ...}
```

This helps identify the bottleneck: raster loading, graph creation, or the shortest-path computation itself.

### Summary of Tips

1. **Always set `search_space_buffer_m`** for large rasters. This is the most impactful setting.
2. **Use CythonAPI** (default) unless you need a specific algorithm from a library backend.
3. **For rasters larger than 1000x1000**, consider GPU acceleration.
4. **Set `calculate_metrics=False`** if you do not need path statistics (length, cost breakdown). This skips post-processing.
5. **Keep the neighborhood at R2** unless you have a specific reason to change it.
6. **Profile with `runtimes`** to find where time is spent before optimizing.

(raster-gpu-backend)=
## GPU Backend `raster_gpu`

```{pyorps-status} unreleased source
```

PYORPS includes a GPU-accelerated SSSP (single-source shortest path) implementation that runs entirely on NVIDIA GPUs. It uses a persistent cooperative kernel with custom barrier synchronization, operating directly on raster data in GPU memory without constructing a graph object.

### Requirements

- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- CUDA toolkit installed
- CuPy Python package

### Installation

```{code-block} bash
pip install pyorps[gpu]
```

This installs CuPy. You may need to select the correct CuPy variant for your CUDA version:

```{code-block} bash
# For CUDA 12.x
pip install cupy-cuda12x
```

### Usage

Select the GPU backend via the `graph_api` parameter:

```{code-block} python
from pyorps import PathFinder

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    graph_api="raster_gpu",
)
result = pf.find_route()
```

The GPU backend is a drop-in replacement for the default CythonAPI. The result format is identical.

### Performance

Benchmarks on an NVIDIA RTX PRO 500 (Blackwell, 14 SMs):

| Raster Size | GPU | Cython | Speedup |
|------------|-----|--------|---------|
| 500x500 | 26 ms | 68 ms | 2.6x |
| 1000x1000 | 54 ms | 285 ms | 5.3x |
| 2000x2000 | 133 ms | 1158 ms | 8.7x |
| 3000x3000 | 235 ms | 2672 ms | 11.4x |

Speedup increases with raster size. For rasters smaller than ~300x300, the overhead of GPU kernel launch and data transfer may negate the benefit.

### Architecture

The GPU implementation uses a raster-direct delta-stepping algorithm:

1. The cost raster is transferred to GPU global memory.
2. A persistent cooperative grid is launched with a fixed number of blocks (2 per SM).
3. Each iteration relaxes edges within the current delta bucket, using atomic operations for distance updates.
4. A custom atomic barrier with `__threadfence()` synchronizes between iterations (instead of `grid.sync()` for compatibility across GPU architectures).
5. The final distance array and predecessor array are copied back to the host for path reconstruction.

### Limitations

- Requires NVIDIA GPU (no AMD/Intel GPU support)
- Compute capability >= 7.0 required
- Currently supports single-source shortest path only
- CUDA toolkit must be accessible at compile time

### Troubleshooting

**Check GPU availability:**

```{code-block} python
import cupy
print(cupy.cuda.runtime.getDeviceCount())  # Should print >= 1
```

**CuPy import fails:**

Ensure the CuPy variant matches your installed CUDA version. Run `nvcc --version` to check.

**Fallback to CPU:**

If the GPU is unavailable, switch to the default Cython backend:

```{code-block} python
pf = PathFinder(..., graph_api="cython")  # CPU fallback
```

```{image} ../_static/generated/gpu_performance.png
:alt: GPU vs CPU performance comparison across raster sizes
:width: 100%
:align: center
```
