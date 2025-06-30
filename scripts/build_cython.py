#!/usr/bin/env python
"""
Script to build Cython extensions for development
"""
import subprocess
import sys
import os


def build_cython():
    """Build Cython extensions in-place for development"""
    try:
        # Check if the .pyx file exists
        if not os.path.exists("pyorps/utils/find_path_cython.pyx"):
            print("✗ Cython source file not found: pyorps/utils/find_path_cython.pyx")
            return False

        # Build in-place for development
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Cython extensions built successfully")
        print("Build output:", result.stdout)

        # Test import
        try:
            from pyorps.utils.find_path_cython import dijkstra_2d_cython
            print("✓ Cython module can be imported")
            return True
        except ImportError as e:
            print(f"✗ Failed to import Cython module: {e}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        print("Build stderr:", e.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = build_cython()
    sys.exit(0 if success else 1)
