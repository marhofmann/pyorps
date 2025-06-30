# build with:
# python setup_find_path_cython.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys


# Check if we're on Windows
is_windows = sys.platform.startswith("win")

# Define compiler args based on platform
compiler_args = []
if is_windows:
    # Use MSVC-compatible flags
    compiler_args = ["/O2", "/fp:fast", '/openmp']
    extra_link_args = []
else:
    # Use GCC-compatible flags
    compiler_args = ["-O3", "-march=native", "-std=c++11", '-fopenmp']
    extra_link_args = ['-fopenmp']

compiler_args.append('-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION')

extensions = [
    Extension(
        "find_path_cython",
        ["find_path_cython.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=compiler_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="find_path_cython",
    version="0.1.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'nonecheck': False,
        }
    ),
    requires=["numpy"],
)
