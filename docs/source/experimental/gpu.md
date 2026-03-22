# 🧪 GPU Acceleration

:::{admonition} Experimental Feature
:class: warning

This feature is under active development. The API may change between releases.
Use in production with caution.
:::

PYORPS includes a GPU-accelerated SSSP (single-source shortest path) implementation that runs entirely on NVIDIA GPUs. It uses a persistent cooperative kernel with custom barrier synchronization, operating directly on raster data in GPU memory without constructing a graph object.

## Requirements

- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- CUDA toolkit installed
- CuPy Python package

## Installation

```{code-block} bash
pip install pyorps[gpu]
```

This installs CuPy. You may need to select the correct CuPy variant for your CUDA version:

```{code-block} bash
# For CUDA 12.x
pip install cupy-cuda12x
```

## Usage

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

## Performance

Benchmarks on an NVIDIA RTX PRO 500 (Blackwell, 14 SMs):

| Raster Size | GPU | Cython | Speedup |
|------------|-----|--------|---------|
| 500x500 | 26 ms | 68 ms | 2.6x |
| 1000x1000 | 54 ms | 285 ms | 5.3x |
| 2000x2000 | 133 ms | 1158 ms | 8.7x |
| 3000x3000 | 235 ms | 2672 ms | 11.4x |

Speedup increases with raster size. For rasters smaller than ~300x300, the overhead of GPU kernel launch and data transfer may negate the benefit.

## Architecture

The GPU implementation uses a raster-direct delta-stepping algorithm:

1. The cost raster is transferred to GPU global memory.
2. A persistent cooperative grid is launched with a fixed number of blocks (2 per SM).
3. Each iteration relaxes edges within the current delta bucket, using atomic operations for distance updates.
4. A custom atomic barrier with `__threadfence()` synchronizes between iterations (instead of `grid.sync()` for compatibility across GPU architectures).
5. The final distance array and predecessor array are copied back to the host for path reconstruction.

## Limitations

- Requires NVIDIA GPU (no AMD/Intel GPU support)
- Compute capability >= 7.0 required
- Currently supports single-source shortest path only
- CUDA toolkit must be accessible at compile time

## Troubleshooting

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
