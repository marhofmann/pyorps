"""Page status: validate the front matter of every Markdown page and render
the availability banner from it.

Every Markdown page must start with a MyST front matter block holding the keys
in ``REQUIRED_KEYS``. The build fails when a key is missing or a value is
outside its allowed set. The banner is generated here; never type it by hand.
"""

from __future__ import annotations

import re

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("title", "summary", "status", "since", "available_in",
                 "module", "api")
STATUS_VALUES = ("stable", "experimental", "unreleased")
AVAILABLE_VALUES = ("pypi", "main", "source")

_FRONT = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_FENCE = re.compile(r"^\s*(```|~~~|:::)")

_TEXT_UNRELEASED = (
    "Not in a PyPI release. Available from a source checkout of main only. "
    "pip install pyorps ({version}) does not include it."
)
_TEXT_MAIN = (
    "Available on the main branch on GitHub. It is not in the PyPI release "
    "{version}; pip install pyorps does not include it yet."
)
_TEXT_EXPERIMENTAL = "Experimental. The API may change between releases."


def parse_front_matter(text: str):
    """Return ``(dict | None, rest_of_text)`` for a Markdown source."""
    match = _FRONT.match(text)
    if not match:
        return None, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text[match.end():]
    return (data if isinstance(data, dict) else {}), text[match.end():]


def validate(meta) -> list[str]:
    """Return a list of problems for one front matter mapping."""
    if meta is None:
        return ["no front matter block"]
    problems = []
    for key in REQUIRED_KEYS:
        if key not in meta:
            problems.append(f"missing key '{key}'")
    if problems:
        return problems
    for key in ("title", "summary", "since", "module"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            problems.append(f"'{key}' must be a non-empty string")
    if meta["status"] not in STATUS_VALUES:
        problems.append(f"status '{meta['status']}' not in {STATUS_VALUES}")
    if meta["available_in"] not in AVAILABLE_VALUES:
        problems.append(
            f"available_in '{meta['available_in']}' not in {AVAILABLE_VALUES}")
    api = meta["api"]
    if not (isinstance(api, list)
            and all(isinstance(item, str) for item in api)):
        problems.append("'api' must be a list of dotted names")
    if not problems:
        if meta["status"] == "stable" and meta["available_in"] != "pypi":
            problems.append("status 'stable' requires available_in 'pypi'")
        if meta["status"] == "unreleased" and meta["available_in"] == "pypi":
            problems.append("status 'unreleased' cannot be available_in 'pypi'")
    return problems


def banner_text(status: str, available_in: str, version: str):
    if status == "unreleased" and available_in == "main":
        return _TEXT_MAIN.format(version=version)
    if status == "unreleased":
        return _TEXT_UNRELEASED.format(version=version)
    if status == "experimental":
        return _TEXT_EXPERIMENTAL
    return None


def make_banner(status: str, available_in: str, version: str):
    """Build the banner admonition node, or None when no banner is needed."""
    text = banner_text(status, available_in, version)
    if text is None:
        return None
    node = nodes.admonition()
    node["classes"] += ["warning", f"pyorps-status-{status}"]
    node += nodes.title("Availability", "Availability")
    node += nodes.paragraph(text, text)
    return node


class StatusDirective(Directive):
    """``pyorps-status <status> <available_in>``: render the banner."""

    required_arguments = 2
    optional_arguments = 0
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        status, available_in = self.arguments
        node = make_banner(status, available_in,
                           env.config.pyorps_pypi_version)
        return [node] if node is not None else []


def pages(env) -> dict:
    if not hasattr(env, "pyorps_pages"):
        env.pyorps_pages = {}
        env.pyorps_errors = {}
    return env.pyorps_pages


def _inject_banner(text: str, meta: dict) -> str:
    """Insert the banner directive right after the first level-one heading."""
    if banner_text(meta["status"], meta["available_in"], "") is None:
        return text
    lines = text.split("\n")
    fenced = False
    for index, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
        if not fenced and line.startswith("# "):
            directive = [
                "",
                f"```{{pyorps-status}} {meta['status']} {meta['available_in']}",
                "```",
            ]
            return "\n".join(lines[:index + 1] + directive + lines[index + 1:])
    return text


def on_source_read(app, docname, source):
    env = app.env
    store = pages(env)
    path = env.doc2path(docname, base=False)
    if not str(path).endswith(".md") or docname.startswith("generated/"):
        return
    meta, rest = parse_front_matter(source[0])
    problems = validate(meta)
    if problems:
        env.pyorps_errors[docname] = problems
        return
    env.pyorps_errors.pop(docname, None)
    store[docname] = {key: meta[key] for key in REQUIRED_KEYS}
    front_length = len(source[0]) - len(rest)
    source[0] = source[0][:front_length] + _inject_banner(rest, meta)


def on_purge(app, env, docname):
    pages(env).pop(docname, None)
    env.pyorps_errors.pop(docname, None)


def on_env_updated(app, env):
    pages(env)
    if env.pyorps_errors:
        lines = [f"  {doc}: {'; '.join(errs)}"
                 for doc, errs in sorted(env.pyorps_errors.items())]
        raise ExtensionError(
            "Invalid page front matter:\n" + "\n".join(lines))
    return []


def setup(app):
    app.add_config_value("pyorps_pypi_version", "0.4.0", "env")
    app.add_directive("pyorps-status", StatusDirective)
    app.connect("source-read", on_source_read)
    app.connect("env-purge-doc", on_purge)
    app.connect("env-updated", on_env_updated)
    return {"version": "1.0", "parallel_read_safe": False,
            "parallel_write_safe": True}
