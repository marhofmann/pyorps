"""
Run setup with:
python setup.py build_ext --inplace
"""
import os
import platform
from pathlib import Path
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except Exception:
    HAS_CYTHON = False


def make_extensions():
    # Base names (without suffix) for the Cython/C++ sources
    modules = [
        ("pyorps.utils.path_core", "pyorps/utils/path_core"),
        ("pyorps.utils.path_algorithms", "pyorps/utils/path_algorithms"),
    ]

    system = platform.system().lower()

    # Platform-specific flags
    if system == "windows":
        extra_compile_args = [
            "/O2", "/fp:fast", "/EHsc", "/openmp",
            "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []
    elif system == "darwin":
        # Apple clang has no OpenMP by default; disabled unless explicitly enabled
        extra_compile_args = [
            "-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []

        if os.environ.get("ENABLE_OPENMP", "0") == "1":
            # Requires: brew install libomp and proper include/lib paths in CFLAGS/LDFLAGS
            extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
            libraries += ["omp"]
    else:
        # Linux: enable OpenMP
        extra_compile_args = [
            "-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing", "-fopenmp",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = ["-fopenmp"]
        libraries = []

    include_dirs = [np.get_include(), "pyorps/utils/"]

    extensions = []
    need_cythonize = False

    for ext_name, base in modules:
        pyx = Path(f"{base}.pyx")
        cpp = Path(f"{base}.cpp")

        if HAS_CYTHON and pyx.exists():
            sources = [str(pyx)]
            need_cythonize = True
        elif cpp.exists():
            sources = [str(cpp)]
        else:
            raise RuntimeError(f"Neither {pyx} nor {cpp} found for extension {ext_name}")

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

    if need_cythonize:
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
    return extensions


# Optional/test dependencies so CI can install them with CIBW_TEST_EXTRAS
extras = {
    "graph": [
        "networkx>=3",
        "rustworkx>=0.14",
        "python-igraph>=0.11",  # import name: igraph
        "networkit>=11",
    ],
    "test": [
        "pytest>=7",
    ],
}
extras["all"] = sorted(set(sum(extras.values(), [])))

setup(
    ext_modules=make_extensions(),
    zip_safe=False,
    extras_require=extras,
)
