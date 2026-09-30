"""Fixed terminals (grid connection points, turbines) named by their id.

Plan rev. 5, item A2. The retired free siting keyed its PCC fields by the
ROW of ``PCCs.shp`` (``enumerate(pccs.geometry)``); that file's rows hold
ids 1, 2, 0, so ``hv_field_PCC0`` was the connection point with id 1 and
every PCC-dependent number was joined to the wrong station. Here a
terminal's label comes from its id attribute and nothing else: row order
never reaches a name, and a missing, empty or repeated id is an error
rather than a silent renumbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Terminal", "load_terminals", "terminals_from_frame"]


@dataclass(frozen=True)
class Terminal:
    """One fixed terminal.

    Attributes:
        label: ``prefix + str(id)``, e.g. ``"PCC0"``; what a field is
            saved under and what every result refers to.
        id: The id attribute's value, as stored.
        xy: ``(x, y)`` in the requested CRS.
    """
    label: str
    id: Any
    xy: tuple[float, float]


def _normalise_id(value: Any, row: int, id_field: str) -> Any:
    """An integer-valued id becomes an ``int`` (``1.0`` from a shapefile
    reads as ``1``); anything else must be a non-empty string."""
    import math

    import numpy as np

    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"row {row} has no {id_field!r}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"row {row}: {id_field!r} is a boolean ({value!r})")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"row {row}: {id_field!r} = {value!r} is not an "
                         f"integer")
    text = str(value).strip()
    if not text:
        raise ValueError(f"row {row} has an empty {id_field!r}")
    return text


def terminals_from_frame(frame, *, id_field: str = "id", prefix: str = "",
                         crs=None) -> list[Terminal]:
    """Terminals from a GeoDataFrame of points, labelled by ``id_field``.

    Parameters:
        frame: A GeoDataFrame with one POINT per row.
        id_field: The attribute that names a terminal.
        prefix: Put in front of the id, e.g. ``"PCC"``.
        crs: Reproject to this CRS first (the routing raster's); ``None``
            keeps the frame's own.

    Returns:
        The terminals sorted by id (numbers before text), so the order is
        a property of the ids, not of the file.

    Raises:
        ValueError: no such attribute, a missing/empty/non-integer id, a
            repeated id or label, or a geometry that is not one point.
    """
    if id_field not in frame.columns:
        raise ValueError(f"no {id_field!r} attribute; have "
                         f"{[c for c in frame.columns if c != 'geometry']}")
    if crs is not None:
        if frame.crs is None:
            raise ValueError("the terminals have no CRS to reproject from")
        frame = frame.to_crs(crs)
    out: list[Terminal] = []
    seen: dict[str, int] = {}
    for row, (value, geom) in enumerate(zip(frame[id_field],
                                            frame.geometry)):
        ident = _normalise_id(value, row, id_field)
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            kind = None if geom is None else geom.geom_type
            raise ValueError(f"terminal {ident!r} (row {row}) is not a "
                             f"point: {kind}")
        label = f"{prefix}{ident}"
        if label in seen:
            raise ValueError(f"label {label!r} appears twice (rows "
                             f"{seen[label]} and {row})")
        seen[label] = row
        out.append(Terminal(label, ident, (float(geom.x), float(geom.y))))
    out.sort(key=lambda t: (1, t.id) if isinstance(t.id, str) else (0, t.id))
    return out


def load_terminals(path, *, id_field: str = "id", prefix: str = "",
                   crs=None, layer=None) -> list[Terminal]:
    """:func:`terminals_from_frame` on a vector file (shapefile, GPKG,
    GeoJSON, ...)."""
    import geopandas as gpd

    kwargs = {} if layer is None else {"layer": layer}
    return terminals_from_frame(gpd.read_file(path, **kwargs),
                                id_field=id_field, prefix=prefix, crs=crs)
