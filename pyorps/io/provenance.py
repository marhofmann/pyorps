"""Provenance records for persisted fields and results (plan Phase A1).

The proven-optimal siting study
(``docs/superpowers/plans/2026-09-23-proven-optimal-substation-full-cost.md``,
revision 5, Phase A1) reuses expensive fields across runs. A reused field
is only valid if it came from the same raster window, the same step
table, the same algorithm and weights, the same seeds and the same cost
parameters as the run that reads it. Before this module none of that
was written down: a cache hit was decided by a file name, or by an
origin within 0.5 m, and one cached HV field was found to have been
settled on a different crop of the raster than the run that read it.

A provenance record is a plain JSON-able dict made of named sections::

    {"schema": 1,
     "raster":     {...},   # source sha256, window, window sha256, shape
     "graph":      {...},   # steps, neighborhood, ignore_max_cost, api
     "algorithm":  {...},   # name, num_threads, delta
     "seeds":      {...},   # seed_hash of the sorted (cell, label) pairs
     "weights":    {...},   # length_rate, weight_mult, no-transit cells
     "storage":    {...},   # dtype and rounding of the stored labels
     "cost_model": {...},   # cost-model hash and parameter vector
     "dw":         {...},   # collector DW parameters, when relevant
     "code":       {...},   # sha256 of every source file that wrote it
     "extra":      {...}}   # free-form, never compared

Writers build a record with :func:`record`; readers build the record they
EXPECT with the same functions and call :func:`require_match`, which
compares every key of both records -- a key on one side only is a
mismatch too -- and raises :class:`ProvenanceMismatch` listing all
differences at once. Floats compare exactly: a record is a statement of
what was computed, not a measurement.

Hashing a 2 GB raster takes seconds, so :func:`sha256_file` caches its
result per process under ``(path, size, mtime)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PROVENANCE_SCHEMA",
    "SECTIONS",
    "ProvenanceMismatch",
    "canonical_json",
    "cells_hash",
    "diff",
    "parameter_hash",
    "read_sidecar",
    "record",
    "require_match",
    "seed_hash",
    "sha256_array",
    "sha256_file",
    "sidecar_path",
    "source_file_hashes",
    "write_sidecar",
]

#: Bump when the meaning of a key changes.
PROVENANCE_SCHEMA = 1

#: The sections a record may carry, in the order they are documented.
SECTIONS = ("raster", "graph", "algorithm", "seeds", "weights", "storage",
            "cost_model", "dw", "code", "extra")

#: Sections :func:`require_match` skips unless told otherwise.
_DEFAULT_IGNORE = ("extra",)

_CHUNK = 1 << 20
_file_hash_cache: dict[tuple[str, int, int], str] = {}


class ProvenanceMismatch(ValueError):
    """A persisted object does not match what its reader expects.

    ``differences`` holds one line per differing key, as dotted paths.
    """

    def __init__(self, what: str, differences: Sequence[str]):
        self.what = what
        self.differences = list(differences)
        shown = "\n  ".join(self.differences[:20])
        more = len(self.differences) - 20
        tail = f"\n  ... and {more} more" if more > 0 else ""
        super().__init__(
            f"{what} does not match the expected provenance "
            f"({len(self.differences)} difference(s)):\n  {shown}{tail}")


# --------------------------------------------------------------- hashing


def sha256_file(path, *, use_cache: bool = True) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB chunks.

    Cached per process under ``(resolved path, size, mtime_ns, ctime_ns,
    inode)``, so a raster that is hashed by every field of a run is read
    once. The change time and inode catch a same-size replacement that
    kept its modification time (``shutil.copy2``, ``rsync -t``). On
    Windows the "change time" is the creation time, so an in-place
    rewrite that restores the mtime is still not seen; pass
    ``use_cache=False`` where that matters.
    """
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), int(st.st_size), int(st.st_mtime_ns),
           int(st.st_ctime_ns), int(st.st_ino))
    if use_cache and key in _file_hash_cache:
        return _file_hash_cache[key]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if use_cache:
        _file_hash_cache[key] = digest
    return digest


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and C-order bytes.

    Two arrays hash equal iff they have the same dtype string, the same
    shape and the same values bit for bit. Large arrays are hashed in
    row blocks, so a non-contiguous view costs no whole-array copy.
    """
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{tuple(int(s) for s in a.shape)}|".encode())
    if a.ndim == 0 or a.size == 0:
        h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    flat_rows = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(-1, 1)
    row_bytes = max(1, flat_rows[0].nbytes)
    block = max(1, (64 << 20) // row_bytes)
    for start in range(0, flat_rows.shape[0], block):
        h.update(np.ascontiguousarray(flat_rows[start:start + block]).tobytes())
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Convert numpy scalars/arrays and tuples to plain JSON types."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        # JSON has no inf/nan; keep them readable and comparable.
        return repr(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no spaces, exact float repr."""
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def parameter_hash(params: Mapping | Sequence | Any) -> str:
    """SHA-256 of the canonical JSON of a parameter object."""
    return hashlib.sha256(canonical_json(params).encode()).hexdigest()


def seed_hash(cells: Iterable[int], labels: Iterable[float] | None = None
              ) -> str:
    """SHA-256 of the seed set, independent of the order it was given in.

    The pairs ``(cell, label)`` are sorted by cell, then label; cells as
    int64 and labels as float64, so the hash changes if a single label
    changes by one ulp. ``labels=None`` means every label is ``0.0``.
    """
    c = np.asarray(list(cells) if not isinstance(cells, np.ndarray)
                   else cells, dtype=np.int64).ravel()
    if labels is None:
        v = np.zeros(c.size, dtype=np.float64)
    else:
        v = np.asarray(list(labels) if not isinstance(labels, np.ndarray)
                       else labels, dtype=np.float64).ravel()
    if v.size != c.size:
        raise ValueError(f"{c.size} seed cells but {v.size} labels")
    order = np.lexsort((v, c))
    h = hashlib.sha256()
    h.update(b"seeds|")
    h.update(np.ascontiguousarray(c[order]).tobytes())
    h.update(np.ascontiguousarray(v[order]).tobytes())
    return h.hexdigest()


def cells_hash(cells: Iterable[int]) -> str:
    """SHA-256 of a cell set (sorted, de-duplicated, int64)."""
    c = np.unique(np.asarray(list(cells) if not isinstance(cells, np.ndarray)
                             else cells, dtype=np.int64).ravel())
    return hashlib.sha256(b"cells|" + c.tobytes()).hexdigest()


def source_file_hashes(sources: Iterable[Any], *, root=None
                       ) -> dict[str, str]:
    """``{path: sha256}`` for source files or imported modules.

    Parameters:
        sources: Paths, or modules (their ``__file__`` is used).
        root: Paths are stored relative to this directory when they lie
            under it, so a record does not change when the repository is
            checked out elsewhere. Defaults to the pyorps repository root.
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    root = Path(root).resolve()
    out: dict[str, str] = {}
    for src in sources:
        path = Path(getattr(src, "__file__", src)).resolve()
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            key = path.as_posix()
        out[key] = sha256_file(path)
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- records


def record(**sections: Mapping[str, Any] | None) -> dict[str, Any]:
    """Assemble a provenance record from named sections.

    Unknown section names raise, so a typo cannot silently create a key
    no reader compares. ``None`` sections are dropped. Values are
    normalised to plain JSON types.
    """
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown provenance section(s) {sorted(unknown)}; "
                         f"known: {SECTIONS}")
    rec: dict[str, Any] = {"schema": PROVENANCE_SCHEMA}
    for name in SECTIONS:
        body = sections.get(name)
        if body is None:
            continue
        if not isinstance(body, Mapping):
            raise TypeError(f"section {name!r} must be a mapping, "
                            f"got {type(body).__name__}")
        rec[name] = _jsonable(dict(body))
    # Round-trip once: what is compared later is what JSON stores.
    return json.loads(canonical_json(rec))


def diff(expected: Mapping[str, Any], actual: Mapping[str, Any], *,
         ignore: Iterable[str] = _DEFAULT_IGNORE) -> list[str]:
    """Every differing key of two records, as ``"dotted.key: a != b"``.

    Compares the union of keys: a key present on one side only is a
    difference. Sections named in ``ignore`` are skipped at top level.
    """
    skip = set(ignore)
    out: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for k in sorted(set(a) | set(b), key=str):
                if not path and k in skip:
                    continue
                sub = f"{path}.{k}" if path else str(k)
                if k not in a:
                    out.append(f"{sub}: missing in expected, stored "
                               f"{_short(b[k])}")
                elif k not in b:
                    out.append(f"{sub}: expected {_short(a[k])}, "
                               f"missing in stored")
                else:
                    walk(a[k], b[k], sub)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{path}: expected {len(a)} items, stored "
                           f"{len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
            return
        # 1 and 1.0 are the same number; True and 1 are not the same value.
        numeric = (isinstance(a, (int, float)) and isinstance(b, (int, float))
                   and not isinstance(a, bool) and not isinstance(b, bool))
        if a != b or (type(a) is not type(b) and not numeric):
            out.append(f"{path}: expected {_short(a)}, stored {_short(b)}")

    walk(_jsonable(expected), _jsonable(actual), "")
    return out


def _short(v: Any, limit: int = 80) -> str:
    s = canonical_json(v) if not isinstance(v, str) else repr(v)
    return s if len(s) <= limit else s[:limit - 3] + "..."


def require_match(expected: Mapping[str, Any], actual: Mapping[str, Any] | None,
                  *, what: str = "persisted object",
                  ignore: Iterable[str] = _DEFAULT_IGNORE) -> None:
    """Raise :class:`ProvenanceMismatch` unless the records agree.

    ``actual=None`` -- an object written before provenance records
    existed -- is always a mismatch: such objects are excluded from reuse
    (plan A3 renames them ``*_pre_v3``).
    """
    if actual is None:
        raise ProvenanceMismatch(what, [
            "no provenance record stored (written before plan Phase A1); " +
            "such objects are not reused"])
    differences = diff(expected, actual, ignore=ignore)
    if differences:
        raise ProvenanceMismatch(what, differences)


# --------------------------------------------------------------- sidecars


def sidecar_path(path) -> Path:
    """``<file>.prov.json`` beside ``path``."""
    p = Path(path)
    return p.with_name(p.name + ".prov.json")


def write_sidecar(path, rec: Mapping[str, Any]) -> Path:
    """Write ``rec`` as ``<path>.prov.json``, atomically.

    For outputs that cannot carry metadata themselves (GeoTIFFs, CSV,
    GeoJSON). Returns the sidecar path.
    """
    dest = sidecar_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.partial-{os.getpid()}")
    text = json.dumps(_jsonable(rec), sort_keys=True, indent=1)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


def read_sidecar(path) -> dict[str, Any] | None:
    """The record written by :func:`write_sidecar`, or ``None`` if absent."""
    src = sidecar_path(path)
    if not src.exists():
        return None
    return json.loads(src.read_text(encoding="utf-8"))
