"""The example and tutorial notebooks must stay renderable on GitHub.

GitHub shows "Invalid Notebook" for broken JSON (a text replacement once removed the backslash of an escaped quote) and
"An error occurred" for a notebook that nbformat does not know (a tool wrote nbformat 5.9, which does not exist) or that
carries keys outside the schema (``jetTransient`` from an IDE). All of it is checked here without running a notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(p for folder in ("examples", "tutorial", "docs/source") for p in (ROOT / folder).rglob("*.ipynb")
                   if ".ipynb_checkpoints" not in p.parts)

pytestmark = pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks next to the tests (installed-wheel test run)")


def _strict(constant):
    raise ValueError(f"{constant} is not valid JSON: GitHub cannot render the notebook")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_is_strict_json_nbformat_4_5(path):
    nb = json.loads(path.read_text(encoding="utf-8"), parse_constant=_strict)
    assert nb["nbformat"] == 4, f"nbformat {nb['nbformat']} does not exist; GitHub renders 4.x"
    assert nb["nbformat_minor"] <= 5
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids), "nbformat 4.5 needs an id in every cell"
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            assert "outputs" not in cell and "execution_count" not in cell, "only code cells carry outputs"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_validates_against_the_nbformat_schema(path):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    nbformat.validate(nb)
