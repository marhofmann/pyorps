"""The data manifest of a siting case study (plan rev. 5, Phase B, G4).

Every input layer is listed with its source, licence, capture date and the
SHA-256 of its file. A layer is ``pending`` until the file is on disk and its
entry is filled; a ``present`` layer is verified against its hash before it is
read, so a changed or replaced file is refused instead of silently changing a
result (plan Phase B acceptance: a data manifest with hashes). Downloads are
never done here: the user fetches files by hand, then registers them with
:meth:`DataManifest.register`.

The manifest is YAML::

    schema: 1
    layers:
      dem:
        kind: dem                  # one of KINDS
        format: raster             # raster or vector
        path: dem/dgm1.tif         # relative to the manifest file
        status: present            # or pending
        source: "..."
        licence: "..."
        captured: 2026-10-01       # ISO date; required when present
        sha256: "..."              # required when present
        crs: EPSG:25832            # optional, checked on read when given
        captured_at: "2026-10-01T14:45:09+02:00"   # optional ISO 8601 with offset
        retrieved_with: "..."      # optional: how it was fetched (script, versions)
        used_for: ["..."]          # optional: what it is used for, and whether it has been

Layers are read as a :class:`RasterWindow` (array, affine transform, CRS, the
layer's hash) or a GeoDataFrame clipped to a bounding box.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from pyorps.core.exceptions import InvalidInputError
from pyorps.io.provenance import ProvenanceMismatch, sha256_file

__all__ = [
    "KINDS",
    "DataManifest",
    "LayerEntry",
    "ManifestError",
    "RasterWindow",
]

SCHEMA = 1

#: The layer kinds of plan Phase B.
KINDS = ("site_raster", "access_raster", "lines", "roads", "dem",
         "protected_areas", "plan_layers", "terminals", "turbine_evidence",
         "other")

_FORMATS = ("raster", "vector")
_STATUS = ("pending", "present")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(InvalidInputError):
    """The manifest is malformed, or a layer cannot be used."""


@dataclass(frozen=True)
class LayerEntry:
    """One input layer.

    Attributes:
        id: The key in the manifest.
        kind: One of :data:`KINDS`.
        format: ``"raster"`` or ``"vector"``.
        path: File path relative to the manifest, or ``None`` while pending.
        status: ``"pending"`` (not on disk yet) or ``"present"``.
        source: Where it came from (a name and a URL or document).
        licence: Its licence, or ``"to be confirmed"``.
        captured: ISO date of the download or capture, ``None`` while pending.
        sha256: Hash of the file, ``None`` while pending.
        crs: Expected CRS (e.g. ``"EPSG:25832"``), optional.
        notes: Free text.
        captured_at: Exact capture time, ISO 8601 with a UTC offset (optional;
            its date must equal ``captured``).
        retrieved_with: How the file was fetched: script, tool versions.
        used_for: What the layer is used for, one entry per use, each saying
            whether the use has happened ("planned", "done: <output>").
    """
    id: str
    kind: str
    format: str
    path: str | None
    status: str
    source: str
    licence: str
    captured: str | None = None
    sha256: str | None = None
    crs: str | None = None
    notes: str | None = None
    captured_at: str | None = None
    retrieved_with: str | None = None
    used_for: tuple[str, ...] = ()

    def __post_init__(self):
        where = f"layer {self.id!r}"
        if self.kind not in KINDS:
            raise ManifestError(f"{where}: kind {self.kind!r} not in {KINDS}")
        if self.format not in _FORMATS:
            raise ManifestError(f"{where}: format must be one of {_FORMATS}")
        if self.status not in _STATUS:
            raise ManifestError(f"{where}: status must be one of {_STATUS}")
        if not self.source or not self.licence:
            raise ManifestError(f"{where}: source and licence are required")
        if self.status == "present":
            if not self.path:
                raise ManifestError(f"{where}: a present layer needs a path")
            if not self.sha256 or not _SHA.match(self.sha256):
                raise ManifestError(f"{where}: sha256 must be 64 hex digits")
            try:
                _dt.date.fromisoformat(str(self.captured))
            except ValueError as exc:
                raise ManifestError(
                    f"{where}: captured must be an ISO date") from exc
        if self.captured_at is not None:
            try:
                stamp = _dt.datetime.fromisoformat(str(self.captured_at))
            except ValueError as exc:
                raise ManifestError(
                    f"{where}: captured_at must be an ISO 8601 time") from exc
            if stamp.tzinfo is None:
                raise ManifestError(
                    f"{where}: captured_at needs a UTC offset")
            if self.captured and stamp.date().isoformat() != str(self.captured):
                raise ManifestError(
                    f"{where}: captured_at is not on the captured date")
        if isinstance(self.used_for, str) or not all(
                isinstance(u, str) for u in self.used_for):
            raise ManifestError(f"{where}: used_for must be a list of strings")

    def as_dict(self) -> dict[str, Any]:
        out = {"kind": self.kind, "format": self.format, "path": self.path,
               "status": self.status, "source": self.source,
               "licence": self.licence}
        for key in ("captured", "captured_at", "sha256", "crs", "retrieved_with",
                    "notes"):
            val = getattr(self, key)
            if val is not None:
                out[key] = str(val) if key in ("captured", "captured_at") else val
        if self.used_for:
            out["used_for"] = list(self.used_for)
        return out


@dataclass
class RasterWindow:
    """A window of a raster layer.

    Attributes:
        array: ``(rows, cols)`` values.
        transform: Affine transform of the window.
        crs: CRS as a string, or ``None``.
        row_off, col_off: Offset of the window in the layer's pixel grid.
        layer_id, sha256: The layer and its verified file hash.
    """
    array: np.ndarray
    transform: Any
    crs: str | None
    row_off: int
    col_off: int
    layer_id: str
    sha256: str


@dataclass
class DataManifest:
    """The layers of one case study, read from a YAML file."""
    layers: dict[str, LayerEntry]
    path: Path
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- I/O

    @classmethod
    def load(cls, path) -> DataManifest:
        """Read and validate a manifest."""
        from pyorps.core.optional_deps import import_yaml

        yaml = import_yaml()

        path = Path(path)
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise ManifestError(f"{path}: not a schema-{SCHEMA} manifest")
        raw = doc.get("layers")
        if not isinstance(raw, dict) or not raw:
            raise ManifestError(f"{path}: no layers")
        layers = {}
        for lid, body in raw.items():
            if not isinstance(body, dict):
                raise ManifestError(f"layer {lid!r}: not a mapping")
            body = dict(body)
            if isinstance(body.get("captured"), (_dt.date, _dt.datetime)):
                body["captured"] = body["captured"].isoformat()[:10]
            if isinstance(body.get("captured_at"), _dt.datetime):
                body["captured_at"] = body["captured_at"].isoformat()
            unknown = set(body) - {"kind", "format", "path", "status",
                                   "source", "licence", "captured", "sha256",
                                   "crs", "notes", "captured_at",
                                   "retrieved_with", "used_for"}
            if unknown:
                raise ManifestError(
                    f"layer {lid!r}: unknown field(s) {sorted(unknown)}")
            try:
                layers[str(lid)] = LayerEntry(id=str(lid), **{
                    "kind": body.get("kind"), "format": body.get("format"),
                    "path": body.get("path"),
                    "status": body.get("status", "pending"),
                    "source": body.get("source"),
                    "licence": body.get("licence"),
                    "captured": body.get("captured"),
                    "sha256": body.get("sha256"),
                    "crs": body.get("crs"), "notes": body.get("notes"),
                    "captured_at": body.get("captured_at"),
                    "retrieved_with": body.get("retrieved_with"),
                    "used_for": tuple(body.get("used_for") or ())})
            except TypeError as exc:                    # pragma: no cover
                raise ManifestError(f"layer {lid!r}: {exc}") from exc
        return cls(layers=layers, path=path, meta=dict(doc.get("meta") or {}))

    def save(self, path=None) -> Path:
        """Write the manifest as YAML (to ``path`` or where it was read)."""
        from pyorps.core.optional_deps import import_yaml

        yaml = import_yaml()

        target = Path(path) if path else self.path
        doc = {"schema": SCHEMA, "meta": self.meta,
               "layers": {k: v.as_dict() for k, v in self.layers.items()}}
        target.write_text(yaml.safe_dump(doc, sort_keys=False,
                                         allow_unicode=True), encoding="utf-8")
        return target

    # ------------------------------------------------------- bookkeeping

    def layer(self, layer_id: str) -> LayerEntry:
        try:
            return self.layers[layer_id]
        except KeyError:
            raise ManifestError(f"no layer {layer_id!r}; known: "
                                f"{sorted(self.layers)}") from None

    def pending(self) -> list[str]:
        """Ids of the layers whose file is not on disk yet."""
        return sorted(k for k, v in self.layers.items()
                      if v.status == "pending")

    def file_of(self, layer_id: str) -> Path:
        """Absolute path of a present layer's file."""
        e = self.layer(layer_id)
        if e.status != "present":
            raise ManifestError(f"layer {layer_id!r} is pending: the file has "
                                "not been downloaded and registered")
        return (self.path.parent / e.path).resolve()

    def register(self, layer_id: str, path, *, captured: str,
                 source: str | None = None, licence: str | None = None,
                 crs: str | None = None) -> LayerEntry:
        """Mark a layer present: hash its file and record the date.

        ``path`` is stored relative to the manifest when it lies under it.
        """
        e = self.layer(layer_id)
        p = Path(path).resolve()
        if not p.is_file():
            raise ManifestError(f"{p} is not a file")
        try:
            rel = p.relative_to(self.path.parent.resolve()).as_posix()
        except ValueError:
            rel = p.as_posix()
        new = replace(e, path=rel, status="present", captured=str(captured),
                      captured_at=None,
                      sha256=sha256_file(p, use_cache=False),
                      source=source or e.source, licence=licence or e.licence,
                      crs=crs or e.crs)
        self.layers[layer_id] = new
        return new

    def annotate(self, layer_id: str, *, captured_at: str | None = None,
                 retrieved_with: str | None = None, used_for=None,
                 notes: str | None = None) -> LayerEntry:
        """Add provenance details to a layer (time, method, uses) without
        touching its file or hash. ``used_for`` replaces the list."""
        e = self.layer(layer_id)
        new = replace(
            e,
            captured_at=captured_at if captured_at is not None else e.captured_at,
            retrieved_with=(retrieved_with if retrieved_with is not None
                            else e.retrieved_with),
            used_for=tuple(used_for) if used_for is not None else e.used_for,
            notes=notes if notes is not None else e.notes)
        self.layers[layer_id] = new
        return new

    # ----------------------------------------------------- verification

    def verify(self, layer_ids=None) -> dict[str, str]:
        """Recompute the hash of every present layer (or of ``layer_ids``).

        Returns ``{id: sha256}``. A pending layer in ``layer_ids``, a missing
        file or a changed hash raises :class:`ProvenanceMismatch` /
        :class:`ManifestError`.
        """
        ids = list(layer_ids) if layer_ids is not None else [
            k for k, v in self.layers.items() if v.status == "present"]
        diffs, out = [], {}
        for lid in ids:
            f = self.file_of(lid)            # raises for a pending layer
            if not f.is_file():
                diffs.append(f"{lid}: file missing ({f})")
                continue
            got = sha256_file(f, use_cache=False)
            want = self.layers[lid].sha256
            if got != want:
                diffs.append(f"{lid}: sha256 {got[:12]} != manifest "
                             f"{str(want)[:12]}")
            out[lid] = got
        if diffs:
            raise ProvenanceMismatch("the data manifest", diffs)
        return out

    def provenance(self, layer_ids=None) -> dict[str, dict]:
        """``{id: {sha256, captured, source, licence}}`` for the A1 record's
        ``extra`` section."""
        ids = layer_ids if layer_ids is not None else [
            k for k, v in self.layers.items() if v.status == "present"]
        return {k: {"sha256": self.layers[k].sha256,
                    "captured": self.layers[k].captured,
                    "source": self.layers[k].source,
                    "licence": self.layers[k].licence} for k in ids}

    # ------------------------------------------------------------ reads

    def read_window(self, layer_id: str, bounds=None, *, band: int = 1,
                    verify: bool = True) -> RasterWindow:
        """Read a raster layer, or the window covering ``bounds``.

        ``bounds`` is ``(left, bottom, right, top)`` in the layer's CRS; the
        window is the smallest whole-pixel window that contains it. With
        ``verify`` (the default) the file's hash is checked first.
        """
        import rasterio
        from rasterio.windows import Window, from_bounds, intersection

        e = self.layer(layer_id)
        if e.format != "raster":
            raise ManifestError(f"layer {layer_id!r} is a {e.format}, not a raster")
        digest = self.verify([layer_id])[layer_id] if verify else e.sha256
        with rasterio.open(self.file_of(layer_id)) as src:
            self._check_crs(e, src.crs)
            if bounds is None:
                win = Window(0, 0, src.width, src.height)
            else:
                win = from_bounds(*bounds, transform=src.transform)
                win = win.round_offsets(op="floor").round_lengths(op="ceil")
                win = intersection(win, Window(0, 0, src.width, src.height))
            arr = src.read(band, window=win)
            return RasterWindow(
                array=arr, transform=src.window_transform(win),
                crs=src.crs.to_string() if src.crs else None,
                row_off=int(win.row_off), col_off=int(win.col_off),
                layer_id=layer_id, sha256=digest)

    def read_vector(self, layer_id: str, bbox=None, *, verify: bool = True):
        """Read a vector layer as a GeoDataFrame, clipped to ``bbox`` by
        feature extent ``(left, bottom, right, top)`` when given."""
        import geopandas as gpd

        e = self.layer(layer_id)
        if e.format != "vector":
            raise ManifestError(f"layer {layer_id!r} is a {e.format}, not a vector")
        if verify:
            self.verify([layer_id])
        gdf = gpd.read_file(self.file_of(layer_id), bbox=bbox)
        self._check_crs(e, gdf.crs)
        return gdf

    @staticmethod
    def _check_crs(entry: LayerEntry, crs) -> None:
        if entry.crs is None:
            return
        from pyproj import CRS

        if crs is None or CRS.from_user_input(crs) != CRS.from_user_input(
                entry.crs):
            raise ManifestError(
                f"layer {entry.id!r}: file CRS {crs} is not the manifest's "
                f"{entry.crs}")
