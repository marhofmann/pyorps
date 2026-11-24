"""The German workshop notebook is generated: ``workshop/build_workshop_notebook.py`` and ``workshop/sections/*.py`` are
its single source.

Every test here only builds the notebook in memory and compares; nothing is executed, so no network, no GPU and no
search is needed. (A full run from top to bottom is a manual step: ``jupyter nbconvert --execute`` in the ``workshop``
folder.)

* the generated cells equal the committed notebook (number, order, type, source text);
* every code cell is valid Python;
* the data the notebook reads offline is in the repository (cost rasters, terrain clip, layer cache, profiles);
* the notebook carries no outputs and no execution counts (it is committed clean) and claims no Python older than the
  project minimum.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKSHOP = ROOT / "workshop"
BUILDER = WORKSHOP / "build_workshop_notebook.py"
NOTEBOOK = WORKSHOP / "pyorps_workshop_de.ipynb"
MIN_PYTHON = (3, 12)

pytestmark = pytest.mark.skipif(
    not BUILDER.is_file(), reason="no workshop/ folder next to the tests (installed-wheel test run)")


def _builder():
    spec = importlib.util.spec_from_file_location("build_workshop_notebook", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources(notebook):
    return [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def test_committed_notebook_matches_its_sections():
    committed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    generated = _builder().build()
    assert _sources(committed) == _sources(generated), (
        "pyorps_workshop_de.ipynb differs from workshop/sections: run python workshop/build_workshop_notebook.py")


def test_every_code_cell_is_valid_python():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            try:
                ast.parse("".join(cell["source"]))
            except SyntaxError as error:
                pytest.fail(f"code cell {index} is not valid Python: {error}")


def test_notebook_is_committed_without_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, f"cell {index} carries output"
    claimed = tuple(int(v) for v in notebook["metadata"]["language_info"]["version"].split(".")[:2])
    assert claimed >= MIN_PYTHON


def test_every_section_of_the_builder_exists():
    builder = _builder()
    missing = [name for name in builder.ORDER if not (WORKSHOP / "sections" / f"{name}.py").is_file()]
    assert not missing, f"sections missing: {missing}"


@pytest.mark.parametrize("relative", [
    "data/cost_raster_a.tiff",
    "data/cost_raster_b.tiff",
    "data/dem_clip.tif",
    "data/profiles/overhead_line_110kv.yaml",
    "data/profiles/overhead_line_220kv.yaml",
    "data/profiles/overhead_line_380kv.yaml",
])
def test_offline_data_is_in_the_repository(relative):
    assert (WORKSHOP / relative).is_file(), f"workshop/{relative} is missing (is it git-ignored?)"


def test_layer_cache_is_complete():
    names = ("alkis", "biotop", "ffh", "wasserschutz", "ueberschwemmung", "naturschutz")
    layers = WORKSHOP / "data" / "layers"
    for name in names:
        assert list(layers.glob(f"{name}_*.gpkg")), f"no cached layer {name}_*.gpkg (is it git-ignored?)"
