"""Release gate: refuse to publish a tag that does not match the repository.

Used by the release workflow before anything is built. A tag ``vX.Y.Z`` (or a pre-release such as
``vX.Y.ZrcN``) is accepted only when

* the version in ``pyproject.toml`` and ``pyorps/__init__.py`` equals the tag (PEP 440 normalised),
* the tagged commit is part of ``origin/main`` (so an unmerged branch can never be released).

Prints ``prerelease=true|false`` so the workflow can send release candidates to TestPyPI.
Exit code 0 = ok, 1 = refused.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PEP440_PRE = re.compile(r"(a|b|rc|\.dev)\d*", re.IGNORECASE)


def normalise(version: str) -> str:
    """The PEP 440 spelling the wheel metadata will carry (``0.4.1-rc.1`` -> ``0.4.1rc1``)."""
    version = version.strip().lstrip("vV")
    version = re.sub(r"[-_.]?(alpha|a)[-_.]?(\d*)$", r"a\2", version, flags=re.IGNORECASE)
    version = re.sub(r"[-_.]?(beta|b)[-_.]?(\d*)$", r"b\2", version, flags=re.IGNORECASE)
    version = re.sub(r"[-_.]?(rc|c|pre|preview)[-_.]?(\d*)$", r"rc\2", version, flags=re.IGNORECASE)
    return version


def project_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def package_version() -> str:
    text = (ROOT / "pyorps" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE)
    if not match:
        raise SystemExit("pyorps/__init__.py has no __version__")
    return match.group(1)


def on_main(commit: str) -> bool:
    for ref in ("origin/main", "main"):
        probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref], cwd=ROOT,
                               capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, ref], cwd=ROOT, check=False)
            return result.returncode == 0
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="the tag name, e.g. v0.4.1 or v0.4.1rc1")
    parser.add_argument("--commit", default="HEAD", help="the tagged commit (default HEAD)")
    parser.add_argument("--skip-main-check", action="store_true", help="local use without a full clone")
    args = parser.parse_args()

    wanted = normalise(args.tag)
    problems = []
    for label, found in (("pyproject.toml", project_version()), ("pyorps/__init__.py", package_version())):
        if normalise(found) != wanted:
            problems.append(f"{label} says {found!r} but the tag says {wanted!r}")
    if not args.skip_main_check and not on_main(args.commit):
        problems.append(f"commit {args.commit} is not part of origin/main (merge first, tag last)")

    prerelease = bool(PEP440_PRE.search(wanted))
    if problems:
        print("RELEASE GATE REFUSED:", *problems, sep="\n  - ")
        return 1
    print(f"release gate ok: version {wanted}, prerelease={str(prerelease).lower()}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"prerelease={str(prerelease).lower()}\nversion={wanted}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
