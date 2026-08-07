import hashlib
import json
import os
import platform
import time
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


PROVENANCE_FILENAME = "_build_provenance.json"
PROVENANCE_SCHEMA = 1

MODULES = [
    ("pyorps.utils._heap", "pyorps/utils/_heap"),
    ("pyorps.utils._raster_context", "pyorps/utils/_raster_context"),
    # cimports _heap and _raster_context, so it must be built with them: it was
    # absent here, which is how its .pyd survived as an orphan no build could
    # regenerate and no provenance check covered.
    ("pyorps.utils._traversal", "pyorps/utils/_traversal"),
    ("pyorps.utils._dijkstra", "pyorps/utils/_dijkstra"),
    ("pyorps.utils._delta_stepping", "pyorps/utils/_delta_stepping"),
    ("pyorps.utils._delta_stepping_fused", "pyorps/utils/_delta_stepping_fused"),
    ("pyorps.utils._constrained_context", "pyorps/utils/_constrained_context"),
    ("pyorps.utils._constrained_dijkstra", "pyorps/utils/_constrained_dijkstra"),
    ("pyorps.utils._constrained_delta", "pyorps/utils/_constrained_delta"),
]

COMPILER_DIRECTIVES = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
    "cdivision": True,
    "nonecheck": False,
    "embedsignature": True,
}

# Filled by make_extensions(), read back by BuildExtWithProvenance after the
# compiler has run (cythonize rewrites Extension.sources to the generated .cpp).
SOURCE_INFO = {}
BUILD_INFO = {}


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


def fast_math_requested():
    """True when ``PYORPS_FAST_MATH`` asks for the unsafe floating-point modes."""
    return os.environ.get("PYORPS_FAST_MATH", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def get_floating_point_args(system):
    """Get the flags governing floating-point semantics.

    The kernels use +inf as a sentinel (gradient LUTs mark forbidden edges,
    ``dist`` arrays initialise to inf and are compared against it), and the
    backend parity suites rest on bit-identical edge weights. ``-ffast-math``
    implies ``-ffinite-math-only``, which tells the compiler those infinities
    cannot occur, and it permits reassociation. The unsafe modes are therefore
    opt-in: set ``PYORPS_FAST_MATH=1`` to build the fast variant and compare the
    two builds.
    """
    if fast_math_requested():
        return ["/fp:fast"] if system == "windows" else ["-ffast-math"]

    if system == "windows":
        return ["/fp:precise"]

    # extra_compile_args are appended after CFLAGS, so these also neutralise an
    # -ffast-math inherited from the environment.
    return ["-fno-fast-math", "-fno-finite-math-only"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_build_signature(system, cpp_standard, compile_args, link_args):
    payload = json.dumps(
        {
            "schema": PROVENANCE_SCHEMA,
            "system": system,
            "cpp_standard": cpp_standard,
            "compile_args": compile_args,
            "link_args": link_args,
            "directives": COMPILER_DIRECTIVES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_previous_build_signature(directories):
    for directory in directories:
        manifest = directory / PROVENANCE_FILENAME
        if manifest.exists():
            try:
                with open(manifest, encoding="utf-8") as handle:
                    return json.load(handle).get("build_signature")
            except (OSError, ValueError):
                return None
    return None


class BuildExtWithProvenance(_build_ext):
    """Record which sources each compiled extension was actually built from.

    setuptools decides what to recompile from mtimes alone, so a binary can
    outlive the source it was generated from. The manifest binds the hash of
    every .pyx to the hash of the binary produced from it, which is what
    ``tests/test_build_provenance.py`` checks before any measurement is trusted.
    """

    def run(self):
        super().run()
        self.write_provenance()

    def write_provenance(self):
        manifests = {}

        for ext in self.extensions:
            info = SOURCE_INFO.get(ext.name)
            if info is None:
                continue
            binary = Path(self.get_ext_fullpath(ext.name))
            if not binary.exists():
                continue

            source = Path(info["path"])
            manifest = manifests.setdefault(
                binary.parent, {"source_dirs": set(), "modules": {}}
            )
            manifest["modules"][ext.name] = {
                "source": source.name,
                "source_kind": info["kind"],
                "source_sha256": sha256_file(source) if source.exists() else None,
                "binary": binary.name,
                "binary_sha256": sha256_file(binary),
            }
            manifest["source_dirs"].add(source.parent.resolve())

        for directory, collected in manifests.items():
            headers = {}
            for source_dir in sorted(collected["source_dirs"]):
                for pattern in ("*.pxd", "*.h"):
                    for header in sorted(source_dir.glob(pattern)):
                        headers[header.name] = sha256_file(header)

            payload = {
                "schema": PROVENANCE_SCHEMA,
                "build_signature": BUILD_INFO.get("signature"),
                "fast_math": BUILD_INFO.get("fast_math"),
                "compile_args": BUILD_INFO.get("compile_args"),
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "source_dirs": [str(p) for p in sorted(collected["source_dirs"])],
                "headers": headers,
                "modules": collected["modules"],
            }

            target = directory / PROVENANCE_FILENAME
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            print(
                f"Wrote build provenance for {len(collected['modules'])} "
                f"extensions to {target}"
            )


def make_extensions():
    modules = MODULES

    system = platform.system().lower()
    cpp_standard = get_cpp_standard()
    fp_args = get_floating_point_args(system)

    print(f"Building for {system} with C++ standard: {cpp_standard}")
    print(f"Floating-point flags: {' '.join(fp_args)}")

    if system == "windows":
        extra_compile_args = [
            "/O2",
            *fp_args,
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
            *fp_args,
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
            *fp_args,
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
            SOURCE_INFO[ext_name] = {"kind": "pyx", "path": pyx}
            print(f"Found {pyx} - will cythonize")
        elif cpp.exists():
            sources = [str(cpp)]
            SOURCE_INFO[ext_name] = {"kind": "cpp", "path": cpp}
            print(f"Found {cpp} - using pre-generated C++ file")
        else:
            # Default to .pyx and let Cython generate it
            sources = [str(pyx)]
            need_cythonize = True
            SOURCE_INFO[ext_name] = {"kind": "pyx", "path": pyx}
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

    signature = compute_build_signature(
        system, cpp_standard, extra_compile_args, extra_link_args
    )
    BUILD_INFO.update(
        signature=signature,
        fast_math=fast_math_requested(),
        compile_args=list(extra_compile_args),
    )

    # setuptools only compares source and object mtimes, so a flag change alone
    # would leave the previous binaries in place.
    previous = read_previous_build_signature(
        sorted({Path(base).parent for _, base in modules})
    )
    force = signature != previous
    if force and previous is not None:
        print("Compiler flags changed since the last build - forcing a rebuild")

    if need_cythonize and HAS_CYTHON:
        print("Cythonizing extensions...")
        return cythonize(
            extensions,
            compiler_directives=COMPILER_DIRECTIVES,
            annotate=False,
            force=force,
        )
    elif need_cythonize and not HAS_CYTHON:
        raise ImportError("Cython is required but not installed!")

    return extensions


# Always build extensions
setup(
    ext_modules=make_extensions(),
    cmdclass={"build_ext": BuildExtWithProvenance},
    zip_safe=False,
)
