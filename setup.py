import os
import sys
from setuptools import setup, Extension
import numpy as np

# Only try to import Cython if building extensions
try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def get_extensions():
    """Get list of extensions to build"""
    extensions = []

    # Check if Cython source file exists
    pyx_file = "pyorps/utils/find_path_cython.pyx"
    if os.path.exists(pyx_file) and HAS_CYTHON:
        # Platform-specific compiler arguments
        is_windows = sys.platform.startswith("win")

        if is_windows:
            extra_compile_args = ["/O2", "/fp:fast"]
            extra_link_args = []
            # Try to add OpenMP support
            try:
                extra_compile_args.append('/openmp')
            except:
                pass
        else:
            extra_compile_args = ["-O3", "-std=c++11"]
            extra_link_args = []
            # Try to add OpenMP support
            try:
                extra_compile_args.append('-fopenmp')
                extra_link_args.append('-fopenmp')
            except:
                pass

        # Add numpy API version
        extra_compile_args.append('-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION')

        # Create extension
        ext = Extension(
            "pyorps.utils.find_path_cython",
            [pyx_file],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )

        extensions.append(ext)

    return extensions


# Get extensions
extensions = get_extensions()

# Cythonize if we have extensions
if extensions and HAS_CYTHON:
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'nonecheck': False,
        }
    )
else:
    ext_modules = []

# Setup
setup(
    ext_modules=ext_modules,
    zip_safe=False,  # Important for Cython extensions
)
