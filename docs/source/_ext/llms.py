"""Write ``llms.txt`` and ``llms-full.txt`` (llmstxt.org format) into the build
output from the page front matter. Both files are generated on every build and
never committed.
"""

from __future__ import annotations

import os

from sphinx.util import logging

from status import pages, parse_front_matter

logger = logging.getLogger(__name__)

SECTIONS = (
    ("Start", "getting_started/"),
    ("API", "api/"),
    ("Concepts", "concepts/"),
    ("Optional", "reference/"),
)


def _toctree_order(env) -> list[str]:
    """Documents in toctree order, starting at the master document."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(doc: str) -> None:
        if doc in seen:
            return
        seen.add(doc)
        order.append(doc)
        for child in env.toctree_includes.get(doc, []):
            visit(child)

    visit(env.config.root_doc)
    return order


def _base_url(app) -> str:
    base = app.config.html_baseurl or os.environ.get(
        "READTHEDOCS_CANONICAL_URL", "")
    return base.rstrip("/") + "/" if base else ""


def _line(app, docname: str, meta: dict, base: str) -> str:
    link = f"{base}_sources/{docname}.md.txt"
    return (f"- [{meta['title']}]({link}): {meta['summary']} "
            f"(status: {meta['status']}, since: {meta['since']})")


def on_build_finished(app, exception):
    if exception is not None:
        return
    env = app.env
    store = pages(env)
    order = [doc for doc in _toctree_order(env) if doc in store]
    for doc in sorted(store):
        if doc not in order:
            order.append(doc)
    base = _base_url(app)
    version = app.config.pyorps_pypi_version

    head = [
        "# PYORPS",
        "",
        f"> Python for Optimal Routes in Power Systems: least-cost power line "
        f"routing on raster cost surfaces. Documentation version "
        f"{app.config.release}; latest PyPI release {version}.",
        "",
        "Status legend: `stable` is in a PyPI release and its API is kept; "
        "`experimental` is in a release and may change; `unreleased` is only "
        "on main or in a source checkout.",
        "",
        f"Machine-readable API index: [{base}api-index.json]"
        f"({base}api-index.json)",
        "",
    ]
    body = []
    for title, prefix in SECTIONS:
        docs = [d for d in order if d.startswith(prefix)]
        if not docs:
            continue
        body.append(f"## {title}")
        body.append("")
        body.extend(_line(app, d, store[d], base) for d in docs)
        body.append("")
    intro = [d for d in order if "/" not in d]
    if intro:
        body.append("## Overview")
        body.append("")
        body.extend(_line(app, d, store[d], base) for d in intro)
        body.append("")
    outdir = app.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "llms.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + body) + "\n")

    full = list(head)
    signatures = _signatures_by_page(outdir)
    for doc in order:
        path = env.doc2path(doc)
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        meta, rest = parse_front_matter(text)
        info = store[doc]
        full.append("---")
        full.append(f"page: {doc}")
        for key in ("title", "summary", "status", "since", "available_in",
                    "module"):
            full.append(f"{key}: {info[key]}")
        full.append("---")
        full.append("")
        full.append(rest.strip())
        full.append("")
        for entry in signatures.get(doc, []):
            if entry.get("signature"):
                full.append(f"Signature: `{entry['name']}"
                            f"{entry['signature']}`")
        full.append("")
    with open(os.path.join(outdir, "llms-full.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(full) + "\n")


def _signatures_by_page(outdir: str) -> dict:
    import json

    path = os.path.join(outdir, "api-index.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle)["entries"]
    except (OSError, ValueError, KeyError):
        return {}
    grouped: dict[str, list] = {}
    for entry in entries:
        grouped.setdefault(entry["page"], []).append(entry)
    return grouped


def setup(app):
    app.setup_extension("status")
    app.setup_extension("api_index")
    app.connect("build-finished", on_build_finished)
    return {"version": "1.0", "parallel_read_safe": False,
            "parallel_write_safe": True}
