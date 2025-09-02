import os
import platform
import subprocess
import tempfile
from pathlib import Path
from setuptools import Extension, setup
import glob
import sys

try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def numpy_include():
    import numpy as np
    return np.get_include()


def find_vcvarsall():
    """Find vcvarsall.bat from Visual Studio installation."""

    # Common paths for VS 2017/2019/2022
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2017\*\VC\Auxiliary\Build\vcvarsall.bat",
    ]

    for pattern in vs_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # Try using vswhere.exe to find VS installation
    vswhere_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe",
    ]

    for vswhere_path in vswhere_paths:
        vswhere_path = os.path.expandvars(vswhere_path)
        if os.path.exists(vswhere_path):
            try:
                result = subprocess.run(
                    [vswhere_path, "-latest", "-property", "installationPath"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    vs_path = result.stdout.strip()
                    vcvarsall = Path(
                        vs_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                    if vcvarsall.exists():
                        return str(vcvarsall)
            except:
                pass

    return None


def get_msvc_env():
    """Get MSVC environment variables."""
    vcvarsall = find_vcvarsall()
    if not vcvarsall:
        # Try to proceed anyway - might be in Developer Command Prompt
        return os.environ.copy()

    # Determine architecture
    arch = "x64" if platform.machine().endswith('64') else "x86"

    # Create a batch script to get environment variables
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_file = Path(tmpdir) / "get_env.bat"
        batch_content = f'''@echo off
call "{vcvarsall}" {arch} >nul 2>&1
set
'''
        batch_file.write_text(batch_content)

        try:
            result = subprocess.run(
                [str(batch_file)],
                capture_output=True,
                text=True,
                shell=True,
                timeout=10
            )

            if result.returncode == 0:
                env = os.environ.copy()
                for line in result.stdout.splitlines():
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env[key] = value
                return env
        except:
            pass

    return os.environ.copy()


def detect_cpp_standard():
    """Detect the highest C++ standard supported by the compiler."""

    system = platform.system().lower()
    if system == "windows":
        return detect_msvc_cpp_standard()

    # For GCC/Clang
    cxx = os.environ.get("CXX", "g++" if system == "linux" else "c++")

    test_code = """
    int main() {
        return 0;
    }
    """

    # C++ standards to test (newest first)
    standards = ["c++23", "c++2b", "c++20", "c++2a", "c++17", "c++14", "c++11"]

    for std in standards:
        if test_cpp_standard(cxx, std, test_code):
            print(f"Detected C++ standard: {std}")
            return std

    print("Warning: Could not detect C++ standard, falling back to C++11")
    return "c++11"


def detect_msvc_cpp_standard():
    """Detect the highest C++ standard supported by MSVC."""

    # Get MSVC environment
    env = get_msvc_env()

    # MSVC standards to test (newest first)
    standards = [
        "/std:c++latest",  # Latest draft standard
        "/std:c++23",
        "/std:c++20",
        "/std:c++17",
        "/std:c++14"
    ]

    test_code = """
    #include <iostream>
    int main() {
        return 0;
    }
    """

    # First, try using distutils/setuptools MSVC detection
    try:
        from setuptools._distutils._msvccompiler import MSVCCompiler
        compiler = MSVCCompiler()
        compiler.initialize()

        for std in standards:
            if test_msvc_standard_with_env(std, test_code, env):
                print(f"Detected MSVC C++ standard: {std}")
                return std
    except ImportError:
        # Fallback to older distutils
        try:
            from distutils.msvccompiler import MSVCCompiler
            compiler = MSVCCompiler()
            compiler.initialize()

            for std in standards:
                if test_msvc_standard_with_env(std, test_code, env):
                    print(f"Detected MSVC C++ standard: {std}")
                    return std
        except:
            pass

    # If detection failed, try direct testing with environment
    for std in standards:
        if test_msvc_standard_with_env(std, test_code, env):
            print(f"Detected MSVC C++ standard: {std}")
            return std

    # Modern MSVC should support at least C++17
    print("Warning: Could not detect MSVC C++ standard, using C++20 as default")
    return "/std:c++20"


def test_cpp_standard(compiler, standard, code):
    """Test if a compiler supports a given C++ standard."""

    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = Path(tmpdir) / "test.cpp"
        source_file.write_text(code)

        try:
            result = subprocess.run(
                [compiler, f"-std={standard}", "-c", str(source_file),
                 "-o", str(Path(tmpdir) / "test.o")],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def test_msvc_standard(standard, code):
    """Test if MSVC supports a given C++ standard (legacy, kept for compatibility)."""
    return test_msvc_standard_with_env(standard, code, get_msvc_env())


def test_msvc_standard_with_env(standard, code, env):
    """Test if MSVC supports a given C++ standard with proper environment."""

    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = Path(tmpdir) / "test.cpp"
        source_file.write_text(code)
        obj_file = Path(tmpdir) / "test.obj"

        # Build command
        cmd = ["cl", "/nologo", "/c", standard, str(source_file), f"/Fo{obj_file}"]

        try:
            # Try to compile with the given standard
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
                env=env,
                shell=True  # Needed for cl.exe on Windows
            )

            # Check if compilation succeeded and object file was created
            return result.returncode == 0 and obj_file.exists()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False


def detect_compiler_features():
    """Detect various compiler features and capabilities."""

    system = platform.system().lower()
    features = {
        "cpp_standard": detect_cpp_standard(),
        "has_openmp": False,
        "has_parallel_stl": False,
        "compiler_type": "unknown"
    }

    if system == "windows":
        features["compiler_type"] = "msvc"
        features["has_openmp"] = True  # MSVC has OpenMP by default

        # MSVC has parallel STL with C++17+
        cpp_std = features["cpp_standard"]
        if cpp_std and any(ver in cpp_std for ver in ["17", "20", "23", "latest"]):
            features["has_parallel_stl"] = True

    else:
        # Detect compiler type
        cxx = os.environ.get("CXX", "g++" if system == "linux" else "c++")

        try:
            result = subprocess.run([cxx, "--version"], capture_output=True, text=True)
            output = result.stdout.lower()

            if "g++" in output or "gcc" in output:
                features["compiler_type"] = "gcc"
                features["has_openmp"] = True
                features["has_parallel_stl"] = True
            elif "clang" in output:
                features["compiler_type"] = "clang"
                # Check for OpenMP support
                test_omp = "#include <omp.h>\nint main() { return 0; }"
                features["has_openmp"] = test_cpp_standard(cxx, "c++11", test_omp)
                # Clang needs TBB for parallel STL
                features["has_parallel_stl"] = check_library_available("tbb")
            elif "icc" in output or "icpc" in output:
                features["compiler_type"] = "intel"
                features["has_openmp"] = True
                features["has_parallel_stl"] = True
        except:
            pass

    return features


def check_library_available(lib_name):
    """Check if a library is available for linking."""

    test_code = f"""
    int main() {{
        return 0;
    }}
    """

    system = platform.system().lower()
    if system == "windows":
        return False  # Skip library detection on Windows for now

    cxx = os.environ.get("CXX", "g++" if system == "linux" else "c++")

    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = Path(tmpdir) / "test.cpp"
        source_file.write_text(test_code)

        try:
            result = subprocess.run(
                [cxx, str(source_file), f"-l{lib_name}", "-o",
                 str(Path(tmpdir) / "test")],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False


def make_extensions():
    modules = [
        ("pyorps.utils.path_core", "pyorps/utils/path_core"),
        ("pyorps.utils.path_algorithms", "pyorps/utils/path_algorithms"),
    ]

    system = platform.system().lower()

    # Detect compiler features
    features = detect_compiler_features()
    cpp_standard = features["cpp_standard"]

    print(f"Compiler type: {features['compiler_type']}")
    print(f"C++ standard: {cpp_standard}")
    print(f"OpenMP support: {features['has_openmp']}")
    print(f"Parallel STL: {features['has_parallel_stl']}")

    if system == "windows":
        extra_compile_args = [
            "/O2", "/fp:fast", "/EHsc",
            "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]

        # Add C++ standard
        if cpp_standard:
            extra_compile_args.append(cpp_standard)
        else:
            # Force modern C++ if detection somehow failed
            extra_compile_args.append("/std:c++20")
            print("Forcing C++20 standard as fallback")

        # Add OpenMP if available
        if features["has_openmp"]:
            extra_compile_args.append("/openmp")

        # Enable parallel algorithms if C++17 or newer
        if features["has_parallel_stl"]:
            extra_compile_args.append("/D_PARALLEL_ALGORITHMS")

        extra_link_args = []
        libraries = []

    elif system == "darwin":
        extra_compile_args = [
            "-O3", f"-std={cpp_standard}", "-ffast-math", "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []

        # macOS OpenMP handling
        if os.environ.get("ENABLE_OPENMP", "0") == "1" or features["has_openmp"]:
            if features["compiler_type"] == "gcc":
                extra_compile_args.append("-fopenmp")
                extra_link_args.append("-fopenmp")
                if features["has_parallel_stl"]:
                    extra_compile_args.append("-D_GLIBCXX_PARALLEL")
            else:
                # Apple Clang needs special handling
                extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                libraries += ["omp"]
                if features["has_parallel_stl"] and check_library_available("tbb"):
                    libraries += ["tbb"]
                    extra_compile_args.append("-DUSE_TBB")

    else:  # Linux
        extra_compile_args = [
            "-O3", f"-std={cpp_standard}", "-ffast-math", "-fno-strict-aliasing",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        ]
        extra_link_args = []
        libraries = []

        # Add OpenMP if available
        if features["has_openmp"]:
            extra_compile_args.append("-fopenmp")
            extra_link_args.append("-fopenmp")

        # Enable parallel STL based on compiler
        if features["compiler_type"] == "gcc" and features["has_parallel_stl"]:
            extra_compile_args.append("-D_GLIBCXX_PARALLEL")
        elif features["compiler_type"] == "clang" and features["has_parallel_stl"]:
            if check_library_available("tbb"):
                libraries.append("tbb")
                extra_compile_args.append("-DUSE_TBB")

    # Add a macro to expose the detected C++ standard to the code
    if cpp_standard:
        # Extract version number (e.g., "17" from "c++17" or "/std:c++17")
        import re
        match = re.search(r'(\d+)', cpp_standard)
        if match:
            cpp_version = match.group(1)
            extra_compile_args.append(f"-DCPP_STANDARD={cpp_version}")

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


# Main setup call
if __name__ == "__main__":
    setup(ext_modules=make_extensions(), zip_safe=False)
