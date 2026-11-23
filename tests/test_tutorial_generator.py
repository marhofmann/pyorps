"""The two tutorial notebooks are generated: ``tutorial/build_notebooks.py`` is the single source of both.

The committed notebooks must not drift away from the script. Every test here only generates (into a temporary folder)
and compares; no notebook is executed, so nothing needs the network or the tutorial's data downloads.

* the generated cells equal the committed ones (number, order, type, id, source text), for both notebooks;
* ``--check`` passes on the current tree, fails when a cell changed, and writes nothing;
* the generated notebooks are valid nbformat 4.5 and claim no Python older than the project minimum;
* every exercise has its solution at the same position of the full notebook (the generator's pairing rule);
* a rebuild over an existing notebook keeps the outputs of the cells whose source did not change.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TUTORIAL = ROOT / "tutorial"
GENERATOR = TUTORIAL / "build_notebooks.py"
FULL = "OSMSES_2026_Tutorial_PYORPS_full.ipynb"
EXERCISES = "OSMSES_2026_Tutorial_PYORPS_exercises.ipynb"
NOTEBOOK_NAMES = (FULL, EXERCISES)
#: The project's ``requires-python``: a notebook must not claim an older Python in its metadata.
MIN_PYTHON = (3, 12)
#: What marks a gap in an exercise cell.
GAP_MARKERS = ("TODO", "YOUR CODE HERE")

pytestmark = pytest.mark.skipif(
    not TUTORIAL.is_dir(), reason="no tutorial/ folder next to the tests (installed-wheel test run)")


def _strict(constant):
    raise ValueError(f"{constant} is not valid JSON: GitHub cannot render the notebook")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_strict)


def _text(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _first_difference(name: str, generated: dict, committed: dict) -> str | None:
    """A message that names the first cell whose type, id or source differs (None: the cells are equal)."""
    if len(generated["cells"]) != len(committed["cells"]):
        return f"{name}: {len(generated['cells'])} cells generated, {len(committed['cells'])} committed"
    for index, (new, old) in enumerate(zip(generated["cells"], committed["cells"])):
        for field in ("cell_type", "id"):
            if new[field] != old[field]:
                return (f"{name}: first difference in cell {index}: {field} is {old[field]!r} in the committed "
                        f"notebook, {new[field]!r} generated")
        new_text, old_text = _text(new), _text(old)
        if new_text != old_text:
            new_lines, old_lines = new_text.split("\n"), old_text.split("\n")
            line = next((n for n, (a, b) in enumerate(zip(new_lines, old_lines)) if a != b),
                        min(len(new_lines), len(old_lines)))
            got = new_lines[line] if line < len(new_lines) else "<end of cell>"
            was = old_lines[line] if line < len(old_lines) else "<end of cell>"
            return (f"{name}: first difference in cell {index} ({old['cell_type']}, id {old['id']}), source line "
                    f"{line + 1}: committed {was!r}, generated {got!r}")
    return None


def _python(metadata: dict) -> tuple[int, ...] | None:
    version = (metadata.get("language_info") or {}).get("version")
    return tuple(int(part) for part in str(version).split(".")[:2]) if version else None


def _run_check(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(GENERATOR), "--check", *args], capture_output=True, text=True,
                          encoding="utf-8", env=env, timeout=120)


@pytest.fixture(scope="module")
def generator():
    """The generator module, loaded from its file (tutorial/ is not a package)."""
    spec = importlib.util.spec_from_file_location("tutorial_build_notebooks", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated_dir(generator, tmp_path_factory) -> Path:
    """Both notebooks, generated into an empty folder (so nothing is carried over from the committed ones)."""
    out_dir = tmp_path_factory.mktemp("generated")
    assert generator.main(["--out-dir", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(NOTEBOOK_NAMES)
    return out_dir


# ----------------------------------------------------------------------------------------------------------------------
# (a) the drift guard
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_generated_cells_equal_the_committed_notebook(name, generated_dir):
    generated, committed = _load(generated_dir / name), _load(TUTORIAL / name)
    message = _first_difference(name, generated, committed)
    assert message is None, f"{message}\nrun: python tutorial/build_notebooks.py   (or fix the script)"
    assert generated["nbformat"] == committed["nbformat"]
    assert generated["nbformat_minor"] == committed["nbformat_minor"]
    assert generated["metadata"]["kernelspec"] == committed["metadata"]["kernelspec"]


def test_a_changed_character_is_reported_with_its_cell():
    """The comparison itself must fail on a one-character change, and name the cell."""
    committed = _load(TUTORIAL / FULL)
    code_cells = [i for i, c in enumerate(committed["cells"]) if c["cell_type"] == "code" and _text(c)]
    index = code_cells[len(code_cells) // 2]
    changed = copy.deepcopy(committed)
    text = _text(changed["cells"][index])
    changed["cells"][index]["source"] = [text[:-1] + ("X" if text[-1] != "X" else "Y")]

    message = _first_difference(FULL, committed, changed)

    assert message is not None, "a changed character in a cell went unnoticed"
    assert f"cell {index} " in message and committed["cells"][index]["id"] in message


# ----------------------------------------------------------------------------------------------------------------------
# (b) --check
# ----------------------------------------------------------------------------------------------------------------------

def test_check_passes_on_the_current_tree():
    result = _run_check()
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_check_fails_when_a_cell_changed_and_writes_nothing(tmp_path):
    for name in NOTEBOOK_NAMES:
        shutil.copyfile(TUTORIAL / name, tmp_path / name)
    notebook = _load(tmp_path / FULL)
    index = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "markdown" and _text(c))
    notebook["cells"][index]["source"] = [_text(notebook["cells"][index]) + "!"]
    (tmp_path / FULL).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    result = _run_check("--out-dir", str(tmp_path))

    assert result.returncode == 1, f"{result.stdout}\n{result.stderr}"
    assert f"cell {index} " in result.stdout, result.stdout
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before, "--check must not write"


def test_check_without_notebooks_fails_and_creates_nothing(generator, tmp_path, capsys):
    assert generator.main(["--check", "--out-dir", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------------------------------------------------
# (c) valid nbformat 4.5
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_generated_notebook_is_valid_nbformat_4_5(name, generated_dir):
    """The checks of tests/test_notebooks.py, applied to what the generator writes."""
    path = generated_dir / name
    nb = _load(path)
    assert nb["nbformat"] == 4, f"nbformat {nb['nbformat']} does not exist; GitHub renders 4.x"
    assert nb["nbformat_minor"] == 5, "cell ids need nbformat 4.5"
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids), "nbformat 4.5 needs an id in every cell"
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            assert "outputs" not in cell and "execution_count" not in cell, "only code cells carry outputs"
    claimed = _python(nb["metadata"])
    assert claimed is None or claimed >= MIN_PYTHON, f"the metadata claims Python {claimed}, older than {MIN_PYTHON}"

    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.reads(path.read_text(encoding="utf-8"), as_version=4))


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_committed_notebook_claims_no_python_older_than_the_minimum(name):
    claimed = _python(_load(TUTORIAL / name)["metadata"])
    assert claimed is None or claimed >= MIN_PYTHON, f"{name} claims Python {claimed}, older than {MIN_PYTHON}"


# ----------------------------------------------------------------------------------------------------------------------
# (d) every exercise has its solution
# ----------------------------------------------------------------------------------------------------------------------

def test_every_exercise_has_its_solution_at_the_same_position(generator, generated_dir):
    """The generator's rule: ``code(solution, exercise_source=gap)`` is one cell definition, so the solution of an
    exercise is the cell at the same position of the full notebook (the cell ids differ between the notebooks)."""
    full, exercises = _load(generated_dir / FULL), _load(generated_dir / EXERCISES)
    definitions = generator.cells
    assert len(full["cells"]) == len(exercises["cells"]) == len(definitions)

    gaps = [i for i, c in enumerate(exercises["cells"])
            if c["cell_type"] == "code" and any(marker in _text(c) for marker in GAP_MARKERS)]
    assert gaps, "the exercises notebook holds no exercise"

    for index in gaps:
        exercise, solution, definition = exercises["cells"][index], full["cells"][index], definitions[index]
        where = f"cell {index} (exercises id {exercise['id']})"
        assert "_exercise_source" in definition, f"{where} is a gap but the generator defines no solution for it"
        assert solution["cell_type"] == exercise["cell_type"] == "code", where
        assert _text(solution) == definition["source"] and _text(exercise) == definition["_exercise_source"], where
        assert _text(solution).strip(), f"{where}: the solution is empty"
        assert not any(marker in _text(solution) for marker in GAP_MARKERS), f"{where}: the solution still has a gap"

    for index, definition in enumerate(definitions):
        if "_exercise_source" in definition:
            types = (definition["cell_type"], full["cells"][index]["cell_type"], exercises["cells"][index]["cell_type"])
            assert len(set(types)) == 1, f"cell {index}: the pair has different cell types {types}"


# ----------------------------------------------------------------------------------------------------------------------
# a rebuild keeps what was executed
# ----------------------------------------------------------------------------------------------------------------------

def test_rebuild_keeps_outputs_of_unchanged_cells_only(generator, tmp_path):
    for name in NOTEBOOK_NAMES:
        shutil.copyfile(TUTORIAL / name, tmp_path / name)
    notebook = _load(tmp_path / FULL)
    with_output = [i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code" and c["outputs"]]
    assert len(with_output) >= 2, "the committed full notebook should carry rendered outputs"
    kept, changed = with_output[0], with_output[1]
    notebook["cells"][changed]["source"] = [_text(notebook["cells"][changed]) + "\n# edited by hand\n"]
    (tmp_path / FULL).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    expected = _load(TUTORIAL / FULL)["cells"]
    # The line endings of the file that is rebuilt: write_text() gave it this platform's, whatever the checkout has.
    before_bytes = (tmp_path / FULL).read_bytes()

    assert generator.main(["--out-dir", str(tmp_path)]) == 0

    rebuilt = _load(tmp_path / FULL)["cells"]
    assert rebuilt[kept]["outputs"] == expected[kept]["outputs"]
    assert rebuilt[kept]["execution_count"] == expected[kept]["execution_count"]
    assert _text(rebuilt[changed]) == _text(expected[changed]), "the script's source must replace the edited one"
    assert rebuilt[changed]["outputs"] == [] and rebuilt[changed]["execution_count"] is None
    assert (b"\r\n" in (tmp_path / FULL).read_bytes()) == (b"\r\n" in before_bytes), "line endings are kept"
    assert (tmp_path / FULL).read_bytes().endswith(b"\n")
