import os
import sys
import platform
from setuptools import setup, Extension
import numpy as np

# Only try to import Cython if building extensions
try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("Warning: Cython not available. Skipping Cython extensions.")


def get_extensions():
    """Get list of extensions to build with platform-specific configuration"""

    extensions = []

    # Check if Cython source file exists
    pyx_file = "pyorps/utils/find_path_cython.pyx"
    if not os.path.exists(pyx_file):
        print(f"Warning: Cython source file {pyx_file} not found. Skipping extension.")
        return extensions

    if not HAS_CYTHON:
        print("Warning: Cython not available. Skipping extension.")
        return extensions

    # Detect platform
    system = platform.system().lower()
    print(f"Building for platform: {system}")

    # Platform-specific compiler and linker arguments
    if system == "windows":
        # Windows MSVC compiler flags
        extra_compile_args = [
            "/O2",  # Optimization level 2
            "/fp:fast",  # Fast floating point model
            "/EHsc",  # Exception handling
        ]
        extra_link_args = []

        # Try to add OpenMP support for Windows
        try:
            extra_compile_args.append("/openmp")
            print("Added OpenMP support for Windows")
        except Exception as e:
            print(f"Warning: Could not add OpenMP support for Windows: {e}")

    else:
        # Linux/macOS GCC compiler flags
        extra_compile_args = [
            "-O3",  # Optimization level 3
            "-std=c++11",  # C++11 standard
            "-ffast-math",  # Fast math operations
            "-fno-strict-aliasing",  # Avoid strict aliasing issues
        ]
        extra_link_args = []

        # Try to add OpenMP support for Linux/macOS
        try:
            extra_compile_args.extend(["-fopenmp"])
            extra_link_args.extend(["-fopenmp"])
            print("Added OpenMP support for Linux/macOS")
        except Exception as e:
            print(f"Warning: Could not add OpenMP support for Linux/macOS: {e}")

    # Common compiler definitions (platform-independent)
    extra_compile_args.extend([
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
    ])

    # Debug output
    print(f"Compiler args: {extra_compile_args}")
    print(f"Linker args: {extra_link_args}")

    # Create extension
    try:
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

    except Exception as e:
        print(f"Error creating extension: {e}")
        return []

    return extensions


def main():
    """Main setup function"""

    print("Starting setup.py...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy version: {np.__version__}")

    # Clean up any existing build artifacts
    cleanup_dirs = ['build', 'dist', 'pyorps.egg-info']
    for cleanup_dir in cleanup_dirs:
        if os.path.exists(cleanup_dir):
            print(f"Cleaning up {cleanup_dir}")
            import shutil

            shutil.rmtree(cleanup_dir, ignore_errors=True)

    # Get extensions
    extensions = get_extensions()

    # Cythonize if we have extensions
    if extensions and HAS_CYTHON:
        print("Cythonizing extensions...")
        try:
            ext_modules = cythonize(
                extensions,
                compiler_directives={
                    'language_level': 3,  # Python 3
                    'boundscheck': False,  # Disable bounds checking for performance
                    'wraparound': False,  # Disable negative index wrapping
                    'initializedcheck': False,  # Disable initialization checking
                    'cdivision': True,  # Use C division semantics
                    'nonecheck': False,  # Disable None checking
                    'embedsignature': True,  # Embed function signatures in docstrings
                },
                # Build options
                annotate=False,  # Set to True for HTML annotation files
                force=True,  # Force rebuild to avoid cached issues
                quiet=False,  # Verbose output
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

    # Run setup
    print("Running setup...")
    setup(
        ext_modules=ext_modules,
        zip_safe=False,  # Important for Cython extensions
    )
    print("Setup completed successfully!")


if __name__ == "__main__":
    main()
