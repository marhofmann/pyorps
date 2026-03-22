# 🧮 Algorithms

PYORPS supports several shortest-path algorithms. The default is Dijkstra on the CythonAPI backend, which provides guaranteed optimality with excellent performance.

## Dijkstra (Default)

Single-source shortest path. Guaranteed optimal for non-negative edge weights. Available on all backends.

```{code-block} python
result = path_finder.find_route(algorithm="dijkstra")
```

## Delta-Stepping

Parallel bucket-based algorithm. On Linux, delta-stepping uses OpenMP for multi-threaded execution, making it well suited for very large rasters.

```{code-block} python
result = path_finder.find_route(algorithm="delta_stepping")
```

:::{note}
Delta-stepping is available on the CythonAPI and RasterGPU backends only. On Windows and macOS, it runs single-threaded because OpenMP is not available. For best parallel performance, use Linux or the GPU backend.
:::

## A\*

Heuristic-based search that can be faster than Dijkstra for single source-to-target pairs. Only available on library backends.

```{code-block} python
# Requires a library backend
pf = PathFinder(..., graph_api="networkx")
result = pf.find_route(algorithm="astar")
```

## Bellman-Ford

Supports negative edge weights (where the backend allows). Slower than Dijkstra but useful for specialized cost models.

```{code-block} python
pf = PathFinder(..., graph_api="networkx")
result = pf.find_route(algorithm="bellman_ford")
```

## Bidirectional Dijkstra

Searches from both source and target simultaneously, meeting in the middle. Can halve the search space for single source-target pairs.

```{code-block} python
pf = PathFinder(..., graph_api="networkit")
result = pf.find_route(algorithm="bidirectional_dijkstra")
```

## Algorithm Availability

| Algorithm | Cython | NetworKit | Rustworkx | NetworkX | iGraph | GPU |
|-----------|--------|-----------|-----------|----------|--------|-----|
| Dijkstra | Yes | Yes | Yes | Yes | Yes | Yes |
| Delta-stepping | Yes | No | No | No | No | Yes |
| A\* | No | Yes | Yes | Yes | No | No |
| Bellman-Ford | No | No | No | Yes | Yes | No |
| Bidirectional Dijkstra | No | Yes | No | Yes | No | No |

## Selection Guide

Default: Dijkstra on CythonAPI
: Best balance of performance and correctness for most routing tasks. No configuration needed.

Large rasters on Linux
: Delta-stepping on CythonAPI. OpenMP parallelism scales well with raster size and available cores.

Single source-to-target, need speed
: A\* on a library backend. The heuristic prunes the search space when source and target are far apart.

Maximum performance
: GPU delta-stepping. See {doc}`../experimental/gpu` for setup and benchmarks.

Negative edge weights
: Bellman-Ford on NetworkX or iGraph. Required when your cost model allows negative costs.
