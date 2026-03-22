import platform
from pathlib import Path
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def numpy_include():
    import numpy as np
    return np.get_include()


def get_cpp_standard():
    """Get appropriate C++ standard for the platform.

    C++20 provides ~1.5-2x speedup over C++17 on MSVC 19.50+.
    Falls back gracefully: C++20 is supported by GCC 10+, Clang 12+, MSVC 19.29+.
    """
    system = platform.system().lower()

    if system == "windows":
        return "/std:c++20"
    else:
        return "c++20"


def make_extensions():
    modules = [
        ("pyorps.utils._heap", "pyorps/utils/_heap"),
        ("pyorps.utils._raster_context", "pyorps/utils/_raster_context"),
        ("pyorps.utils._dijkstra", "pyorps/utils/_dijkstra"),
        ("pyorps.utils._delta_stepping", "pyorps/utils/_delta_stepping"),
    ]

    system = platform.system().lower()
    cpp_standard = get_cpp_standard()

    print(f"Building for {system} with C++ standard: {cpp_standard}")

    if system == "windows":
        extra_compile_args = [
            "/O2",
            "/fp:fast",
            "/EHsc",
            "/openmp",
            cpp_standard,
            "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []

    elif system == "darwin":
        extra_compile_args = [
            "-O3",
            f"-std={cpp_standard}",
            "-ffast-math",
            "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []

        # OpenMP via Homebrew libomp (required for _delta_stepping)
        try:
            import subprocess
            libomp_prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True
            ).strip()
            extra_compile_args.extend([
                "-Xpreprocessor", "-fopenmp",
                f"-I{libomp_prefix}/include",
            ])
            extra_link_args.extend([
                f"-L{libomp_prefix}/lib",
                "-lomp",
            ])
            print(f"Found libomp at {libomp_prefix}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("WARNING: libomp not found. Install with: brew install libomp")

    else:  # Linux
        extra_compile_args = [
            "-O3",
            f"-std={cpp_standard}",
            "-ffast-math",
            "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            "-fopenmp",  # Enable OpenMP on Linux
        ]
        extra_link_args = ["-fopenmp"]
        libraries = []

    include_dirs = [numpy_include(), "pyorps/utils/"]

    extensions = []
    need_cythonize = False

    for ext_name, base in modules:
        pyx = Path(f"{base}.pyx")
        cpp = Path(f"{base}.cpp")

        # Check what source files exist
        if pyx.exists():
            sources = [str(pyx)]
            need_cythonize = True
            print(f"Found {pyx} - will cythonize")
        elif cpp.exists():
            sources = [str(cpp)]
            print(f"Found {cpp} - using pre-generated C++ file")
        else:
            # Default to .pyx and let Cython generate it
            sources = [str(pyx)]
            need_cythonize = True
            print(f"Expecting {pyx} to exist")

        extensions.append(
            Extension(
                name=ext_name,
                sources=sources,
                include_dirs=include_dirs,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=libraries,
            )
        )

    if need_cythonize and HAS_CYTHON:
        print("Cythonizing extensions...")
        return cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "initializedcheck": False,
                "cdivision": True,
                "nonecheck": False,
                "embedsignature": True,
            },
            annotate=False,
            force=False,
        )
    elif need_cythonize and not HAS_CYTHON:
        raise ImportError("Cython is required but not installed!")

    return extensions


# Always build extensions
setup(ext_modules=make_extensions(), zip_safe=False)
