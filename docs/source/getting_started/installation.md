# 💽 Installation

## Quick Install (recommended)

Pre-built binary wheels are available on PyPI for **Windows**, **Linux**, and **macOS** with **Python 3.11, 3.12, and 3.13**. No C++ compiler required:

```{code-block} bash
pip install pyorps
```

This installs the core package with all required dependencies and pre-compiled Cython extensions.

:::{note}
A C++ compiler is **only** required if you install from source (e.g., `pip install -e .` for development) or if no pre-built wheel is available for your platform. See [Building from Source](#building-from-source) below.
:::

## Pre-built Wheel Availability

| Platform | Architecture | Python Versions |
|----------|-------------|-----------------|
| **Windows** | x86_64 (AMD64) | 3.11, 3.12, 3.13 |
| **Linux (glibc)** | x86_64 | 3.11, 3.12, 3.13 |
| **Linux (musl/Alpine)** | x86_64 | 3.11, 3.12, 3.13 |
| **macOS** | x86_64, ARM64 (Apple Silicon) | 3.11, 3.12, 3.13 |

If `pip install pyorps` succeeds, the Cython extensions are already compiled — no further action needed.

## Optional Dependencies

PYORPS provides several optional dependency groups for different use cases:

| Group | Command | Includes |
|-------|---------|----------|
| `graph` | `pip install pyorps[graph]` | rustworkx, python-igraph, networkx, networkit |
| `gpu` | `pip install pyorps[gpu]` | cupy-cuda12x |
| `gpu-full` | `pip install pyorps[gpu-full]` | cupy-cuda12x, cugraph-cu12, cudf-cu12 |
| `dev` | `pip install pyorps[dev]` | coverage, pytest, cython |
| `examples` | `pip install pyorps[examples]` | notebook, fiona |
| `case_studies` | `pip install pyorps[case_studies]` | notebook, fiona, pandapower, contextily |
| `additionals` | `pip install pyorps[additionals]` | openpyxl |
| `full` | `pip install pyorps[full]` | All optional dependencies (except GPU) |

You can combine multiple groups:

```{code-block} bash
pip install pyorps[graph,examples]
```

:::{note}
The `full` group does not include GPU dependencies (`gpu` / `gpu-full`) because these require a compatible NVIDIA GPU and CUDA toolkit. Install them separately if needed.
:::

(building-from-source)=
## Building from Source

Building from source is only necessary for **development** or if no pre-built wheel exists for your platform. This requires a C++ compiler with **C++20 support** (or C++17 minimum).

### Development Install

```{code-block} bash
git clone https://github.com/marhofmann/pyorps.git
cd pyorps
pip install -e .[dev,full]
```

### Building Cython Extensions

After modifying any `.pyx` file or after a fresh clone, rebuild the extensions:

```{code-block} bash
python setup.py build_ext --inplace
```

This compiles seven C++ extensions:

- `_heap` -- binary heaps, index conversion, system resource detection
- `_raster_context` -- geometry, direction precomputation, path cost utilities
- `_dijkstra` -- Dijkstra solver (single-pair, 1-to-N, M-to-N, pairwise)
- `_delta_stepping` -- parallel delta-stepping with OpenMP
- `_constrained_context` -- constrained state encoding, precomputation, clearance
- `_constrained_dijkstra` -- constrained Dijkstra (dense + sparse)
- `_constrained_delta` -- constrained parallel delta-stepping variants

### C++ Compiler Requirements

The extensions are compiled with **C++20** by default for best performance. C++17 is the minimum supported standard.

| Compiler | Minimum Version | C++20 Support | Notes |
|----------|----------------|---------------|-------|
| **MSVC** (Visual Studio) | 19.29+ (VS 2019 16.10+) | Yes | Recommended on Windows |
| **GCC** | 10+ | Yes | Default on most Linux distributions |
| **Clang** | 12+ | Yes | Default on macOS (via Xcode) |

(platform-notes)=
### Platform Notes

::::{tab-set}

:::{tab-item} Windows
- **Compiler:** MSVC (Visual Studio Build Tools)
- **Flags:** `/O2 /fp:fast /EHsc /openmp /std:c++20`
- **OpenMP:** Supported

Install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload.
:::

:::{tab-item} Linux
- **Compiler:** GCC 10+
- **Flags:** `-O3 -ffast-math -fopenmp -std=c++20`
- **OpenMP:** Supported (parallel delta-stepping)

On Debian/Ubuntu:

```{code-block} bash
sudo apt install build-essential gcc g++
```
:::

:::{tab-item} macOS
- **Compiler:** Clang (Xcode Command Line Tools)
- **Flags:** `-O3 -ffast-math -std=c++20`
- **OpenMP:** Not supported (single-threaded only)

Install the Xcode Command Line Tools:

```{code-block} bash
xcode-select --install
```
:::

::::

## Verify Installation

After installation, verify that PYORPS is available:

```{code-block} python
from pyorps import PathFinder
print(PathFinder)
```

To confirm that the Cython extensions are compiled and available:

```{code-block} python
from pyorps.utils import path_algorithms
print("Cython extensions available")
```

## Troubleshooting

### `pip install pyorps` fails with "no matching distribution"

Your platform or Python version may not have a pre-built wheel. Check that you are using **Python 3.11, 3.12, or 3.13** on a supported platform (see [Pre-built Wheel Availability](#pre-built-wheel-availability)). If no wheel is available, pip will attempt to build from source, which requires a C++ compiler (see [C++ Compiler Requirements](#c-compiler-requirements)).

### Missing C++ compiler on Windows

```
error: Microsoft Visual C++ 14.0 or greater is required.
```

This only occurs when building from source. Install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the "Desktop development with C++" workload.

### Cython build failures

```
ImportError: Cython is required but not installed!
```

Install Cython before building from source:

```{code-block} bash
pip install cython>=3.0.0
```

### NumPy header not found

```
fatal error: numpy/arrayobject.h: No such file or directory
```

Ensure NumPy is installed before building the extensions:

```{code-block} bash
pip install numpy>=2.0.0
python setup.py build_ext --inplace
```

### Import error after install

```
ImportError: cannot import name 'path_algorithms' from 'pyorps.utils'
```

The Cython extensions have not been compiled. If you installed from source, run:

```{code-block} bash
python setup.py build_ext --inplace
```

If you installed via `pip install pyorps`, this should not happen — please [open an issue](https://github.com/marhofmann/pyorps/issues).

### OpenMP not available on macOS

OpenMP parallelism is not supported on macOS with the default Clang compiler. PYORPS will still work correctly but the parallel delta-stepping algorithm will run single-threaded. All other algorithms (including the default Dijkstra) are unaffected.
