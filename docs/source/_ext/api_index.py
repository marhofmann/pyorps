"""Machine-readable API index.

Collects the signatures that autodoc renders (``autodoc-process-signature``)
and the ``api`` lists of the page front matter, and writes ``api-index.json``
into the build output. Warns when a name in ``pyorps.__all__`` belongs to no
page or to more than one page.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os

from sphinx.util import logging

from status import pages

logger = logging.getLogger(__name__)

_SIGNATURES: dict[str, str] = {}


def on_signature(app, what, name, obj, options, signature, return_annotation):
    if signature is not None or return_annotation:
        _SIGNATURES[name] = f"{signature or '()'}" + (
            f" -> {return_annotation}" if return_annotation else "")
    return None


def _resolve(dotted: str):
    """Import ``pyorps.X.Y`` and return the object, or None."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:  # noqa: BLE001 - any import failure means unknown
            continue
        try:
            for attr in parts[cut:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def _entry(name: str, page: str, meta: dict) -> dict:
    obj = _resolve(name)
    module, qualname = name.rsplit(".", 1)[0], name.rsplit(".", 1)[-1]
    signature = None
    summary = None
    if obj is not None:
        module = getattr(obj, "__module__", None) or module
        canonical = f"{module}.{getattr(obj, '__qualname__', qualname)}"
        signature = _SIGNATURES.get(canonical) or _SIGNATURES.get(name)
        if signature is None and callable(obj):
            try:
                signature = str(inspect.signature(obj))
            except (TypeError, ValueError):
                signature = None
        doc = inspect.getdoc(obj) if callable(obj) else None
        if doc:
            summary = doc.strip().splitlines()[0]
    return {
        "name": name,
        "module": module,
        "signature": signature,
        "summary": summary,
        "status": meta["status"],
        "since": meta["since"],
        "page": page,
    }


def _check_exports(store: dict) -> None:
    try:
        import pyorps
    except Exception:  # noqa: BLE001
        logger.warning("api_index: cannot import pyorps, export check skipped")
        return
    owners: dict[str, list[str]] = {}
    for page, meta in store.items():
        for name in meta["api"]:
            owners.setdefault(name, []).append(page)
    for export in getattr(pyorps, "__all__", []):
        pages_for = owners.get(f"pyorps.{export}", [])
        if len(pages_for) == 0:
            logger.warning("api_index: pyorps.%s belongs to no page", export)
        elif len(pages_for) > 1:
            logger.warning("api_index: pyorps.%s belongs to %d pages: %s",
                           export, len(pages_for), ", ".join(pages_for))


def on_build_finished(app, exception):
    if exception is not None:
        return
    store = pages(app.env)
    entries = []
    for page in sorted(store):
        for name in store[page]["api"]:
            entries.append(_entry(name, page, store[page]))
    _check_exports(store)
    payload = {
        "pyorps_version": app.config.pyorps_pypi_version,
        "entries": entries,
    }
    os.makedirs(app.outdir, exist_ok=True)
    with open(os.path.join(app.outdir, "api-index.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def setup(app):
    app.setup_extension("status")
    app.connect("autodoc-process-signature", on_signature)
    app.connect("build-finished", on_build_finished)
    return {"version": "1.0", "parallel_read_safe": False,
            "parallel_write_safe": True}
