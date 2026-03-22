# 🏎️ Performance Tuning

This page covers the main levers for improving PYORPS runtime and memory usage.

## Search Space Buffer

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

## Neighborhood Size

The neighborhood parameter controls how many adjacent cells each cell connects to, which affects both path smoothness and computation time.

| Neighborhood | Connections | Relative Speed | Path Quality |
|-------------|-------------|----------------|--------------|
| R0 | 4 | Fastest | Blocky, staircase artifacts |
| R1 | 8 | Fast | Acceptable for many use cases |
| R2 (default) | 16 | Moderate | Good balance, recommended |
| R3 | 32 | Slower | Smooth paths |
| R4+ | 48+ | Slowest | Very smooth, diminishing returns |

R2 is recommended for most use cases. Only increase beyond R2 if visual smoothness matters (e.g., presentation maps). R0/R1 are useful for quick exploratory runs.

## Backend Selection

Choose the right backend for your scale:

- **CythonAPI** (default): fastest CPU backend, no graph construction overhead.
- **RasterGPU**: 4--20x faster than Cython for rasters larger than 500x500 cells. See {doc}`../experimental/gpu`.
- **Library backends**: slower due to graph construction, but offer additional algorithms.

## Algorithm Selection

- **Dijkstra**: general purpose, always optimal. Good for all raster sizes.
- **Delta-stepping**: better for very large rasters on Linux where OpenMP parallelism is available.
- **A\***: faster for single source-to-target when using a library backend.

See {doc}`algorithms` for the full comparison.

## Memory Management

PYORPS uses `uint16` cost values (0--65535) for memory efficiency. Even so, very large rasters can consume significant memory.

- PYORPS tracks `MAX_SAFE_CELLS` and issues a warning when the raster exceeds safe limits.
- Use `search_space_buffer_m` to reduce the effective raster size.
- Use a bounding box (`bbox`) or mask to spatially subset the raster before routing.

## Runtime Profiling

After calling `find_route()`, inspect the timing breakdown:

```{code-block} python
path_finder.find_route()
print(path_finder.runtimes)
# {'raster_handler': 0.12, 'graph_creation': 0.45, 'shortest_path': 1.23, ...}
```

This helps identify the bottleneck: raster loading, graph creation, or the shortest-path computation itself.

## Summary of Tips

1. **Always set `search_space_buffer_m`** for large rasters. This is the most impactful setting.
2. **Use CythonAPI** (default) unless you need a specific algorithm from a library backend.
3. **For rasters larger than 1000x1000**, consider GPU acceleration.
4. **Set `calculate_metrics=False`** if you do not need path statistics (length, cost breakdown). This skips post-processing.
5. **Keep the neighborhood at R2** unless you have a specific reason to change it.
6. **Profile with `runtimes`** to find where time is spent before optimizing.
