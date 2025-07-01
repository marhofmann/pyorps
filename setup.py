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
    pyx_file = "pyorps/utils/find_path_cython.pyx"
    if not os.path.exists(pyx_file) or not HAS_CYTHON:
        print(f"Warning: Cython source file {pyx_file} not found or Cython not available. Skipping extension.")
        return extensions

    system = platform.system().lower()
    print(f"Building for platform: {system}")

    if system == "windows":
        # MSVC flags
        extra_compile_args = ["/O2", "/fp:fast", "/EHsc", "/openmp"]
        extra_link_args = []
    else:
        # GCC/Clang flags
        extra_compile_args = ["-O3", "-std=c++11", "-ffast-math", "-fno-strict-aliasing", "-fopenmp"]
        extra_link_args = ["-fopenmp"]

    # Common for all platforms
    extra_compile_args.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

    print(f"Compiler args: {extra_compile_args}")
    print(f"Linker args: {extra_link_args}")

    ext = Extension(
        name="pyorps.utils.find_path_cython",
        sources=[pyx_file],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    extensions.append(ext)
    print(f"Successfully created extension for {pyx_file}")
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
    )
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()
