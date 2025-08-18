"""
Run setup with:
python setup.py build_ext --inplace
"""

import os
import sys
import platform
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("Warning: Cython not available. Skipping Cython extensions.")


def get_extensions():
    extensions = []

    # Define individual .pyx files to compile as separate extensions
    pyx_files = [
        ("pyorps.utils.path_core", "pyorps/utils/path_core.pyx"),
        ("pyorps.utils.path_algorithms", "pyorps/utils/path_algorithms.pyx")
    ]

    # Check if all files exist
    missing_files = []
    for _, file_path in pyx_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files or not HAS_CYTHON:
        print(
            f"Warning: Missing files {missing_files} or Cython not available. Skipping extension.")
        return extensions

    system = platform.system().lower()
    print(f"Building for platform: {system}")

    # Set up platform-specific flags
    if system == "windows":
        extra_compile_args = [
            "/O2", "/fp:fast", "/EHsc", "/openmp", "/wd4551",
            "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ]
        extra_link_args = []
        libraries = []
    elif system == "darwin":
        # Apple clang has no OpenMP by default; build without it
        extra_compile_args = [
            "-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ]
        extra_link_args = []
        libraries = []
    else:
        # Linux (GCC/Clang with OpenMP)
        extra_compile_args = [
            "-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing", "-fopenmp",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ]
        extra_link_args = ["-fopenmp"]
        libraries = []

    print(f"Compiler args: {extra_compile_args}")
    print(f"Linker args: {extra_link_args}")

    # Create separate extensions for each .pyx file
    for ext_name, source_file in pyx_files:
        ext = Extension(
            name=ext_name,
            sources=[source_file],
            include_dirs=[np.get_include(), "pyorps/utils/"],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        )
        extensions.append(ext)
        print(f"Successfully created extension {ext_name} from {source_file}")

    return extensions


def main():
    print("Starting setup.py...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy version: {np.__version__}")

    # Clean up build artifacts
    for cleanup_dir in ['build', 'dist', 'pyorps.egg-info']:
        if os.path.exists(cleanup_dir):
            print(f"Cleaning up {cleanup_dir}")
            import shutil
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    extensions = get_extensions()
    if extensions and HAS_CYTHON:
        print("Cythonizing extensions...")
        try:
            ext_modules = cythonize(
                extensions,
                compiler_directives={
                    'language_level': 3,
                    'boundscheck': False,
                    'wraparound': False,
                    'initializedcheck': False,
                    'cdivision': True,
                    'nonecheck': False,
                    'embedsignature': True,
                },
                annotate=False,
                force=True,
                quiet=False,
            )
            print(f"Successfully cythonized {len(ext_modules)} extensions")
        except Exception as e:
            print(f"Error during cythonization: {e}")
            ext_modules = []
    else:
        ext_modules = []
        if extensions:
            print("Extensions found but Cython not available")
        else:
            print("No extensions to build")

    print("Running setup...")
    setup(
        ext_modules=ext_modules,
        zip_safe=False,
        include_dirs=[np.get_include(), "pyorps/utils/"]
    )
    print("Setup completed successfully!")


if __name__ == "__main__":
    main()
