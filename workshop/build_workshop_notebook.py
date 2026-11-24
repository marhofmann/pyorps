"""Build the German PYORPS workshop notebook from the section modules in ``workshop/sections/``.

This script and the section modules are the single source of ``pyorps_workshop_de.ipynb``. Edit the section modules,
then run this script; do not edit the notebook by hand (the next rebuild would revert the change).

Every section module defines ``CELLS``: a list of ``("md", text)`` and ``("code", text)`` items.

Usage (paths are anchored on this file, so the working directory does not matter)::

    python workshop/build_workshop_notebook.py                  # write pyorps_workshop_de.ipynb next to this script
    python workshop/build_workshop_notebook.py --check          # write nothing; exit 1 if the notebook differs
    python workshop/build_workshop_notebook.py --allow-missing  # skip sections that do not exist yet (development)

The notebook is written without outputs; a run from top to bottom fills them (see ``README.md`` in this folder).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = HERE / "sections"
NOTEBOOK = HERE / "pyorps_workshop_de.ipynb"

#: The sections in notebook order (file names without ``.py``).
ORDER = (
    "sec1_installation",
    "sec1b_grundidee",
    "sec2_kostenraster",
    "sec4a_algorithmen_erklaerung",
    "sec4b_algorithmen_laeufe",
    "sec5_dgm_3d",
    "sec6_freileitung",
    "sec7_gui",
    "sec8_zusammenfassung",
)

#: The oldest Python a notebook may claim in its metadata (the project's ``requires-python``).
MIN_PYTHON = "3.12"

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": MIN_PYTHON,
    },
}


def load_cells(name):
    """Return the ``CELLS`` list of one section module."""
    path = SECTIONS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CELLS


def to_lines(text):
    """Notebook JSON stores a cell source as a list of lines that keep their line ends."""
    return text.strip("\n").splitlines(keepends=True)


def build(allow_missing=False):
    """Return the notebook as a dict."""
    cells = []
    for name in ORDER:
        if not (SECTIONS / f"{name}.py").exists():
            if allow_missing:
                print(f"skipped (missing): {name}", file=sys.stderr)
                continue
            raise SystemExit(f"missing section module: {SECTIONS / (name + '.py')}")
        for kind, text in load_cells(name):
            cell = {"id": f"ws-{len(cells):03d}", "metadata": {}, "source": to_lines(text)}
            if kind == "md":
                cell["cell_type"] = "markdown"
            elif kind == "code":
                cell["cell_type"] = "code"
                cell["execution_count"] = None
                cell["outputs"] = []
            else:
                raise SystemExit(f"{name}: unknown cell kind {kind!r}")
            cells.append(cell)
    return {"cells": cells, "metadata": NOTEBOOK_METADATA, "nbformat": 4, "nbformat_minor": 5}


def serialise(notebook):
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def sources(notebook):
    return [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if the notebook is out of date")
    parser.add_argument("--allow-missing", action="store_true", help="skip section modules that do not exist yet")
    args = parser.parse_args()

    new = build(allow_missing=args.allow_missing)
    if args.check:
        if not NOTEBOOK.exists():
            raise SystemExit(f"{NOTEBOOK} does not exist")
        old = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        if sources(old) != sources(new):
            raise SystemExit("pyorps_workshop_de.ipynb differs from its sections: run build_workshop_notebook.py")
        print("ok: notebook and sections agree")
        return
    NOTEBOOK.write_text(serialise(new), encoding="utf-8", newline="\n")
    n_code = sum(c["cell_type"] == "code" for c in new["cells"])
    print(f"wrote {NOTEBOOK}: {len(new['cells'])} cells ({n_code} code)")


if __name__ == "__main__":
    main()
