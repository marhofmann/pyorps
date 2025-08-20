import os
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


def make_extensions():
    modules = [
        ("pyorps.utils.path_core", "pyorps/utils/path_core"),
        ("pyorps.utils.path_algorithms", "pyorps/utils/path_algorithms"),
        ("pyorps.utils.path_delta", "pyorps/utils/path_delta")
    ]

    system = platform.system().lower()

    if system == "windows":
        extra_compile_args = [
            "/O2", "/fp:fast", "/EHsc", "/openmp",
            "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []
    elif system == "darwin":
        extra_compile_args = [
            "-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []
        if os.environ.get("ENABLE_OPENMP", "0") == "1":
            extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
            libraries += ["omp"]
    else:  # Linux
        extra_compile_args = [
            "-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing", "-fopenmp",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = ["-fopenmp"]
        libraries = []

    include_dirs = [numpy_include(), "pyorps/utils/"]

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
            raise RuntimeError(f"Neither {pyx} nor {cpp} found for {ext_name}")

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


setup(ext_modules=make_extensions(), zip_safe=False)

