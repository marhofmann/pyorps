"""Run the release test job locally, from tracked files only, before a tag is pushed.

Why: most problems of the 0.4.0 release were differences between a developer machine and git
(ignored or untracked files, repo-relative paths, optional packages, newer dependencies). This script
exports exactly what git tracks and runs the tests the way the wheel-test job does: from a temporary
folder next to a copy of ``profiles/``, with the GUI and GPU packages hidden.

    python tools/release_check.py                 # quick: tests of the committed tree, current environment
    python tools/release_check.py --tree index    # test the staged files instead of HEAD
    python tools/release_check.py --build         # also build the wheel and test it in a clean virtual environment

Uncommitted, unstaged changes are never tested (that is the point). Nothing is written inside the repository.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIDE = ("dash", "dash_leaflet", "dash_ag_grid", "dash_bootstrap_components", "dash_extensions", "localtileserver",
        "webview", "geobuf", "flask_compress", "waitress", "cupy", "cupyx", "cugraph", "cudf", "playwright")

BLOCKER = '''import sys, importlib.abc, importlib.util
BLOCK = %r
class _B(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
sys.meta_path.insert(0, _B())
_orig = importlib.util.find_spec
def _fs(name, package=None):
    return None if name.split(".")[0] in BLOCK else _orig(name, package)
importlib.util.find_spec = _fs
''' % (set(HIDE),)


def git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8", "replace").strip()


def export_tree(tree: str, dest: Path) -> None:
    data = git("archive", "--format=tar", tree, binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", choices=("head", "index"), default="head", help="what to test (default: HEAD)")
    parser.add_argument("--build", action="store_true", help="build the wheel and test it in a clean virtual environment")
    parser.add_argument("--extra", default="dev", help="extra installed with the wheel in --build mode (default dev)")
    parser.add_argument("--clean", action="store_true", help="delete the temporary folder after a pass (kept by default: mass deletes can trip endpoint protection)")
    parser.add_argument("pytest_args", nargs="*", help="extra arguments for pytest (after --)")
    args = parser.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and args.tree == "head":
        print("note: uncommitted changes exist and are NOT part of this check:\n" + dirty + "\n")
    tree = git("write-tree") if args.tree == "index" else git("rev-parse", "HEAD")

    tmp = Path(tempfile.mkdtemp(prefix="pyorps_release_check_"))
    print(f"exporting {args.tree} ({tree[:10]}) to {tmp}")
    src = tmp / "src"
    export_tree(tree, src)

    work = tmp / "work"
    work.mkdir()
    (work / "tests_run").mkdir()
    # the layout of the wheel-test job: tests/ copied to tests_run/, profiles/ next to it and inside it
    for name in ("tests", "profiles"):
        if not (src / name).exists():
            print(f"FAIL: {name}/ is not tracked by git")
            return 1
    import shutil
    shutil.copytree(src / "tests", work / "tests_run", dirs_exist_ok=True)
    shutil.copytree(src / "profiles", work / "profiles")
    shutil.copytree(src / "profiles", work / "tests_run" / "profiles")

    python = sys.executable
    if args.build:
        wheelhouse = tmp / "wheelhouse"
        if run([python, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheelhouse)], cwd=src):
            print("FAIL: the wheel does not build")
            return 1
        env_dir = tmp / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
        python = str(env_dir / ("Scripts" if os.name == "nt" else "bin") / "python")
        wheel = next(wheelhouse.glob("pyorps-*.whl"))
        if run([python, "-m", "pip", "install", f"{wheel}[{args.extra}]", "pytest"], cwd=tmp):
            print("FAIL: the wheel does not install")
            return 1
        smoke = "import pyorps, pyorps.utils._dijkstra as d; print('pyorps', pyorps.__version__, 'from', pyorps.__file__)"
        if run([python, "-c", smoke], cwd=work):
            print("FAIL: the installed wheel does not import")
            return 1

    blocker = tmp / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(BLOCKER, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(blocker) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS", "4"))
    cmd = [python, "-m", "pytest", ".", "-q", "--tb=short", "-p", "no:cacheprovider", "--import-mode=importlib",
           "--ignore=test_io/test_vector_loader.py", *args.pytest_args]
    code = run(cmd, cwd=work / "tests_run", env=env)
    print("\nRELEASE CHECK", "PASSED" if code == 0 else f"FAILED (pytest exit code {code})")
    if args.clean and code == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("temporary folder kept:", tmp)
    return code


if __name__ == "__main__":
    sys.exit(main())
