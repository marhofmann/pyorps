"""The examples, tutorials and notebooks must keep working on every platform and Python version.

Two layers:

* Static checks for EVERY example script and every code cell of every notebook under ``examples/`` and
  ``tutorial/``: it compiles (an invalid escape such as ``"data\\raster"`` is an error), it holds no absolute
  Windows path or backslash path literal, and every ``pyorps`` name it imports or uses exists in the installed package.
* Execution of what can run unattended, offline and without a GPU (see ``TABLE``). Each run is a subprocess with
  the ``Agg`` backend, at most 4 numba threads and the GPU hidden, in a temporary folder, so no tracked file changes.

``TABLE`` is the single place that records the state of every example. A new example fails
``test_every_example_has_a_decision`` until it gets a row.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
TUTORIAL = ROOT / "tutorial"

pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="no source tree (installed-wheel test run)")

TIMEOUT_S = 900


@dataclass(frozen=True)
class Entry:
    """What the smoke test does with one example file.

    ``run``: executed (``reason`` says how much of it). ``import``: imported in a subprocess, so syntax, imports and the
    absence of work at import time are checked, but a real run is impossible here (``reason`` says why). ``skip``: not
    executed at all (``reason`` says why). ``needs`` lists optional packages (or ``pyorps`` modules that need an
    extra); the row is skipped when one of them cannot be imported.
    """

    mode: str
    reason: str
    needs: tuple[str, ...] = ()
    untracked: bool = False


#: example file (relative to the repository root) -> decision. Reasons follow the audit of 2026-11-10.
TABLE: dict[str, Entry] = {
    "examples/minimal_example.ipynb": Entry(
        "run", "offline; runs on a window cut from the shipped sample raster, writes to its own output folder"),
    "examples/feasibility_multi_objective_routing.ipynb": Entry(
        "run", "offline and CPU only (the GPU is hidden, so it falls back to cython); Part 4 skips itself without "
               "the case-study files"),
    "examples/preprocess_mv_connections.py": Entry(
        "run", "command line on a small synthetic manifest and station layer (uniform-cost fallback)"),
    "examples/_test_public_pyorps_compat.py": Entry(
        "run", "API surface check only; the routing part needs the owner's cost raster and pandapower grid model"),
    "examples/pyorps_gui.py": Entry("run", "--help only; it is interactive (starts a Dash server)"),
    "examples/interactive_route_viewer.py": Entry(
        "run", "--help only; it is interactive and legacy (pyorps.webviz), and needs the WFS without --offline",
        needs=("pyorps.webviz",)),
    "tutorial/build_notebooks.py": Entry(
        "run", "builds both tutorial notebooks in a temporary folder and validates them as nbformat 4.5"),
    "examples/batch_route_planning_pandapower.py": Entry(
        "import", "a real run needs a pandapower grid model outside this repository and WFS downloads",
        needs=("pandapower",)),
    "examples/mv_cablebox_to_station_routing.py": Entry(
        "import", "same inputs as the batch script (grid model outside this repository, WFS downloads)",
        needs=("pandapower",)),
    "examples/compare_constrained_neighborhoods.py": Entry(
        "import", "needs the git-ignored distribution-grid cost raster (about 1.5 GB); long-running"),
    "examples/constrained_routing_3d.py": Entry(
        "import", "needs the git-ignored cost raster plus the DEM and DSM files; long-running"),
    "examples/analyze_simplification.py": Entry(
        "import", "reads routing results of the batch script, which are not shipped"),
    "examples/advanced_example.ipynb": Entry(
        "skip", "network (WFS) and shapefiles that the user must download (copyright), not shipped"),
    "examples/create_rasterized_geodata.ipynb": Entry(
        "skip", "network (WFS download of a 30 x 20 km area) and it writes a raster of about 1.5 GB"),
    "examples/prepare_data_for_distribution_grid_planning.ipynb": Entry(
        "skip", "network (five WFS services) and long-running; the GPU backend is used only when a device exists"),
    "tutorial/OSMSES_2026_Tutorial_PYORPS_full.ipynb": Entry(
        "skip", "network (WFS) and the mv_oberrhein raster, which is not in the repository (cell 34)"),
    "tutorial/OSMSES_2026_Tutorial_PYORPS_exercises.ipynb": Entry(
        "skip", "exercise stubs, incomplete by design; also needs the network"),
    "examples/api_comparison.py": Entry(
        "skip", "untracked (git-ignored): the owner must decide whether to commit it; needs the graph extra",
        untracked=True),
}


# ----------------------------------------------------------------------------------------------------------------------
# Discovery and source extraction
# ----------------------------------------------------------------------------------------------------------------------

_SKIP_DIRS = {"__pycache__", "output", "data"}


def _discover() -> list[Path]:
    """Every .py and .ipynb file under examples/ and tutorial/ (hidden folders such as a venv are left out)."""
    found = []
    for folder in (EXAMPLES, TUTORIAL):
        if not folder.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
            found.extend(Path(dirpath, name) for name in sorted(filenames) if name.endswith((".py", ".ipynb")))
    return sorted(found)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


FILES = _discover()


def _strip_magics(source: str) -> str:
    """Blank out IPython magics and shell escapes (lines starting with % or !), keeping the line numbers."""
    return "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in source.split("\n"))


def _notebook_code_cells(path: Path) -> list[tuple[int, str]]:
    """(cell index, source) for every code cell, magics removed."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [(index, _strip_magics("".join(cell["source"])))
            for index, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]


def _code_units(path: Path) -> list[tuple[str, str]]:
    """(label, source) pairs to analyse: the script itself, or each code cell of a notebook."""
    if path.suffix == ".ipynb":
        return [(f"{path.name} code cell {index}", source) for index, source in _notebook_code_cells(path)]
    return [(path.name, path.read_text(encoding="utf-8"))]


def _parse(source: str, label: str) -> ast.AST:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # invalid escapes are reported by the compile test, not here
        return ast.parse(source, label)


# ----------------------------------------------------------------------------------------------------------------------
# Static checks
# ----------------------------------------------------------------------------------------------------------------------

#: C:\..., C:/... (a drive letter that does not end a longer word, so "https://" does not match)
_WINDOWS_DRIVE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_UNC = re.compile(r"^\\\\[^\\/\s]+[\\/]")
#: dir\sub\file or dir\file.ext as it appears in the VALUE of a string (so "\n" and regex classes do not match)
_BACKSLASH_PATH = re.compile(r"\w{2,}(?:\\\w[\w.\-]*){2,}|\w{2,}\\\w[\w\-]*\.[A-Za-z0-9]{2,5}\b")
#: what the audit tooling leaves behind where an absolute path was removed
_REDACTED_PATH = "<local-path>"


def _string_values(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


@pytest.mark.parametrize("path", FILES, ids=_rel)
def test_compiles(path):
    """A syntax error, or an escape that a newer Python rejects (``"data\\raster"``), fails here on every platform."""
    for label, source in _code_units(path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            warnings.simplefilter("error", DeprecationWarning)
            compile(source, label, "exec")


@pytest.mark.parametrize("path", FILES, ids=_rel)
def test_no_absolute_or_backslash_paths(path):
    """Paths must be relative to the file (scripts) or to the notebook folder, with forward slashes or pathlib."""
    offenders = []
    for label, source in _code_units(path):
        for lineno, value in _string_values(_parse(source, label)):
            for rule, hit in (
                    ("absolute Windows path", _WINDOWS_DRIVE.search(value)),
                    ("UNC path", _UNC.search(value)),
                    ("backslash path", _BACKSLASH_PATH.search(value)),
                    ("redacted absolute path", _REDACTED_PATH in value),
            ):
                if hit:
                    offenders.append(f"{label}, line {lineno}: {rule}: {value[:70]!r}")
    assert not offenders, "OS-specific paths:\n  " + "\n  ".join(offenders)


@cache
def _load(module_name: str):
    """Import a module of the package. ``None`` when it needs an optional package that is not installed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pyorps.webviz warns about its own deprecation
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            if exc.name and exc.name.split(".")[0] != "pyorps":
                return None
            raise


def _is_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _check_from_import(module_name: str, name: str) -> str | None:
    module = _load(module_name)
    if module is None or name == "*" or hasattr(module, name) or _is_module(f"{module_name}.{name}"):
        return None
    return f"from {module_name} import {name}: no such name"


def _dotted(node: ast.expr) -> list[str] | None:
    """['pyorps', 'a', 'b'] for the expression pyorps.a.b, or None when it is anything else."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(names)]


def _pyorps_problems(tree: ast.AST) -> list[str]:
    try:
        return _collect_pyorps_problems(tree)
    except ImportError as exc:  # a pyorps module that does not exist, or fails to import
        return [f"cannot import: {exc}"]


def _collect_pyorps_problems(tree: ast.AST) -> list[str]:
    problems = []
    aliases: dict[str, str] = {}  # local name -> the pyorps module it is bound to
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                and (node.module == "pyorps" or node.module.startswith("pyorps.")):
            problems.extend(p for p in (_check_from_import(node.module, a.name) for a in node.names) if p)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyorps" or alias.name.startswith("pyorps."):
                    if _load(alias.name) is None and not _is_module(alias.name):
                        problems.append(f"import {alias.name}: no such module")
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        aliases["pyorps"] = "pyorps"
    for node in ast.walk(tree):
        chain = _dotted(node) if isinstance(node, ast.Attribute) else None
        if not chain or chain[0] not in aliases:
            continue
        path, obj = aliases[chain[0]], _load(aliases[chain[0]])
        for attr in chain[1:]:
            if obj is None:
                break
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
                if not isinstance(obj, type(sys)):
                    break  # the rest are attributes of an object, not of the package
                path = obj.__name__
            elif _is_module(f"{path}.{attr}"):
                path, obj = f"{path}.{attr}", _load(f"{path}.{attr}")
            else:
                problems.append(f"{'.'.join(chain)}: {path} has no attribute {attr!r}")
                break
    return sorted(set(problems))


@pytest.mark.parametrize("path", FILES, ids=_rel)
def test_pyorps_names_exist(path):
    """``from pyorps... import name`` and ``pyorps.name`` must exist in the installed package (API drift guard)."""
    problems = []
    for label, source in _code_units(path):
        problems.extend(f"{label}: {p}" for p in _pyorps_problems(_parse(source, label)))
    assert not problems, "\n".join(problems)


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------

def _env() -> dict[str, str]:
    threads = str(min(4, os.cpu_count() or 4))
    env = dict(os.environ)
    env.update(
        MPLBACKEND="Agg",
        NUMBA_NUM_THREADS=threads,
        OMP_NUM_THREADS=threads,
        PYTHONUTF8="1",  # some examples print emoji; a cp1252 pipe on Windows cannot encode them
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",  # never leave a __pycache__ in the source tree
        CUDA_VISIBLE_DEVICES="",  # no GPU: examples must fall back to the CPU, and CI machines have none
    )
    return env


def _run(args: list[str], cwd: Path, what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(args, cwd=cwd, env=_env(), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{what} did not finish within {TIMEOUT_S} s: mark it 'long-running' in TABLE", pytrace=False)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
        stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
        pytest.fail(f"{what} exited with code {proc.returncode}\n--- stderr (tail) ---\n{stderr_tail}"
                    f"\n--- stdout (tail) ---\n{stdout_tail}", pytrace=False)
    return proc


@pytest.fixture(scope="session")
def sample_workdir(tmp_path_factory) -> Path:
    """A folder laid out like ``examples/``, with a window cut from the shipped sample raster.

    The 44 MB original is only read; the window (about 3 MB) covers the source and target of the minimal example with
    300 m to spare. Notebooks run with this folder as their working directory.
    """
    import rasterio
    from rasterio.windows import from_bounds

    source = EXAMPLES / "data" / "raster" / "sample_raster.tiff"
    if not source.exists():
        pytest.skip("the sample raster is not in the source tree")
    workdir = tmp_path_factory.mktemp("examples_cwd")
    target = workdir / "data" / "raster"
    target.mkdir(parents=True)
    with rasterio.open(source) as dataset:
        window = from_bounds(471700, 5593100, 473100, 5594300, dataset.transform).round_offsets().round_lengths()
        data = dataset.read(1, window=window)
        profile = dataset.profile
        profile.update(height=data.shape[0], width=data.shape[1], transform=dataset.window_transform(window))
        with rasterio.open(target / "sample_raster.tiff", "w", **profile) as out:
            out.write(data, 1)
    return workdir


#: Runs the code cells of a notebook in order in one namespace, like a kernel would (no jupyter needed).
_NOTEBOOK_RUNNER = '''\
import builtins
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:  # IPython is not a core dependency: print what the notebook would show
    def display(*objects, **kwargs):
        for obj in objects:
            print(obj)

builtins.display = display
with open(sys.argv[1], encoding="utf-8") as handle:
    cells = json.load(handle)
namespace = {"__name__": "__main__"}
for number, source in cells:
    try:
        exec(compile(source, f"<code cell {number}>", "exec"), namespace)
    except BaseException:
        print(f"FAILED in code cell {number}", file=sys.stderr)
        raise
    finally:
        plt.close("all")
print(f"ran {len(cells)} code cells")
'''


def _run_notebook(rel: str, tmp_path: Path, request) -> None:
    workdir = request.getfixturevalue("sample_workdir")
    path = ROOT / rel
    cells_file = tmp_path / f"{path.stem}.cells.json"
    cells_file.write_text(json.dumps(_notebook_code_cells(path)), encoding="utf-8")
    runner = tmp_path / "run_notebook.py"
    runner.write_text(_NOTEBOOK_RUNNER, encoding="utf-8")
    _run([sys.executable, str(runner), str(cells_file)], workdir, rel)


def _run_help(rel: str, tmp_path: Path, request) -> None:
    proc = _run([sys.executable, str(ROOT / rel), "--help"], tmp_path, rel)
    assert "usage" in proc.stdout.lower()


def _run_compat_api_check(rel: str, tmp_path: Path, request) -> None:
    code = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='compat_check')['check_api_surface']()"
    _run([sys.executable, "-c", code, str(ROOT / rel)], tmp_path, rel)


def _run_preprocess_cli(rel: str, tmp_path: Path, request) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    crs = "EPSG:25832"
    stations_xy = {10: (470000.0, 5600000.0), 11: (471000.0, 5600000.0), 12: (472000.0, 5600000.0),
                   13: (470000.0, 5601000.0), 14: (471000.0, 5601000.0)}
    elements = [("PV-1", "sgen", "openfield_pv", 3.5, 10, (470100.0, 5600100.0)),
                ("LOAD-1", "load", "large_load", 6.0, 11, (471100.0, 5600050.0))]
    stations = gpd.GeoDataFrame(
        {"bus_id": list(stations_xy), "name": [f"ST-{b}" for b in stations_xy], "vn_kv": [20.0] * len(stations_xy)},
        geometry=[Point(xy) for xy in stations_xy.values()], crs=crs)
    manifest = gpd.GeoDataFrame(
        {"element_id": [e[0] for e in elements], "kind": [e[1] for e in elements], "tech": [e[2] for e in elements],
         "s_mva": [e[3] for e in elements], "default_pcc_bus": [e[4] for e in elements],
         "year": [2025] * len(elements), "scenario_id": ["base"] * len(elements), "seed": [42] * len(elements)},
        geometry=[Point(e[5]) for e in elements], crs=crs)
    manifest_path, stations_path, out_dir = tmp_path / "manifest.gpkg", tmp_path / "stations.gpkg", tmp_path / "out"
    manifest.to_file(manifest_path, layer="mv_connection_requests", driver="GPKG")
    stations.to_file(stations_path, layer="mv_stations", driver="GPKG")
    _run([sys.executable, str(ROOT / rel), "--manifest", str(manifest_path), "--stations", str(stations_path),
          "--out-dir", str(out_dir), "--k", "2"], tmp_path, rel)
    assert list(out_dir.glob("*.gpkg")), "the command line produced no candidate file"


def _run_builder(rel: str, tmp_path: Path, request) -> None:
    """The builder writes into ``--out-dir``, so the committed notebooks are never touched."""
    out_dir = tmp_path / "notebooks"
    _run([sys.executable, str(ROOT / rel), "--out-dir", str(out_dir)], tmp_path, rel)
    built = sorted(out_dir.glob("*.ipynb"))
    assert len(built) == 2, f"expected the full and the exercise notebook, got {[p.name for p in built]}"
    for path in built:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] <= 5
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids), f"{path.name}: nbformat 4.5 needs an id in every cell"
        assert len(set(ids)) == len(ids), f"{path.name}: cell ids must be unique"
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                assert "outputs" not in cell and "execution_count" not in cell
        if importlib.util.find_spec("nbformat"):  # the schema check is a bonus: nbformat is not a core dependency
            import nbformat

            nbformat.validate(nbformat.reads(path.read_text(encoding="utf-8"), as_version=4))


RUNNERS: dict[str, Callable[[str, Path, object], None]] = {
    "examples/minimal_example.ipynb": _run_notebook,
    "examples/feasibility_multi_objective_routing.ipynb": _run_notebook,
    "examples/preprocess_mv_connections.py": _run_preprocess_cli,
    "examples/_test_public_pyorps_compat.py": _run_compat_api_check,
    "examples/pyorps_gui.py": _run_help,
    "examples/interactive_route_viewer.py": _run_help,
    "tutorial/build_notebooks.py": _run_builder,
}

#: Imports a script from its folder without running it; every such script keeps its work behind main().
_IMPORT_CHECK = (
    "import importlib, sys; sys.path.insert(0, sys.argv[1]); module = importlib.import_module(sys.argv[2]); "
    "assert callable(getattr(module, 'main', None)), sys.argv[2] + ' has no main(): its code would run on import'"
)


def _import_script(rel: str, tmp_path: Path, request) -> None:
    path = ROOT / rel
    _run([sys.executable, "-c", _IMPORT_CHECK, str(path.parent), path.stem], tmp_path, rel)


def _available(name: str) -> bool:
    if name.split(".")[0] == "pyorps":
        return _load(name) is not None
    return importlib.util.find_spec(name) is not None


def _execution_params():
    for rel, entry in TABLE.items():
        marks = []
        if entry.mode == "skip":
            marks.append(pytest.mark.skip(reason=entry.reason))
        if entry.untracked and not (ROOT / rel).exists():
            marks.append(pytest.mark.skip(reason="untracked file is not in this checkout"))
        yield pytest.param(rel, id=rel, marks=marks)


@pytest.mark.parametrize("rel", _execution_params())
def test_example_runs_unattended(rel, tmp_path, request):
    entry = TABLE[rel]
    missing = [name for name in entry.needs if not _available(name)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)} (optional extra): {entry.reason}")
    runner = RUNNERS[rel] if entry.mode == "run" else _import_script
    runner(rel, tmp_path, request)


# ----------------------------------------------------------------------------------------------------------------------
# The table itself
# ----------------------------------------------------------------------------------------------------------------------

def test_every_example_has_a_decision():
    """A new example, tutorial or notebook needs a row in TABLE: run it, import it, or say why it cannot run."""
    undecided = sorted({_rel(p) for p in FILES} - TABLE.keys())
    assert not undecided, f"add these files to TABLE in {Path(__file__).name} with a mode and a reason: {undecided}"


def test_table_has_no_stale_rows():
    stale = sorted(rel for rel, entry in TABLE.items() if not entry.untracked and not (ROOT / rel).exists())
    assert not stale, f"these files are gone, remove their rows from TABLE: {stale}"


def test_table_modes_and_runners_agree():
    assert {entry.mode for entry in TABLE.values()} <= {"run", "import", "skip"}
    assert {rel for rel, entry in TABLE.items() if entry.mode == "run"} == set(RUNNERS)
    assert all(entry.reason for entry in TABLE.values()), "every row needs a reason"
