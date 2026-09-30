"""Bulk shapely -> GeoJSON conversion for rasterio's scan converter.

``rasterio.features.rasterize``/``geometry_mask`` do not burn shapely objects
directly: they call ``__geo_interface__`` on every shape, which builds one
Python dict per geometry in the interpreter. Measured on 302,500 land-use
parcels burned into 36 M cells, that dict building is 85-95 % of the wall
time (15.6 s total, 1.56 s of actual scan conversion at 52.9 us/geometry).

``shapely.to_ragged_array`` extracts every coordinate of a whole array in ONE
vectorized C call (0.44 s for 300k features vs 19.35 s for per-geometry
``__geo_interface__``); rebuilding the GeoJSON ring nesting from its offsets
is numpy slicing plus a single ``.tolist()``. Feeding rasterio the resulting
dicts is bit-identical -- it is exactly what it would have built itself --
but skips the per-geometry Python round trip.

Measured through ``GeoRasterizer._burn_index_band`` at EPSG:25832
magnitudes, every output bit-identical (parity is asserted in
``tests/test_raster/test_geojson_bulk.py``; the timings are not, because
wall time is machine and load dependent):

======================================  =========  ========  =====
fixture                                 shapely    bulk      gain
======================================  =========  ========  =====
302,500 parcels, 36 M cells, sparse       10.43 s    2.88 s  3.62x
80,089 holed + multipart, 10 M cells       3.69 s    1.02 s  3.63x
302,500 parcels, 36 M cells, 100 % cover   8.86 s    3.46 s  2.56x
======================================  =========  ========  =====

The gain shrinks as the polygons get larger because only the dict building
is removed -- the scan conversion underneath is untouched, and it is the
floor of what the bulk path can reach.

Two invariants govern everything here:

*Painting order is load-bearing.* pyorps burns features in ascending-cost
order so the most expensive one wins an overlap; a permutation silently
changes the optimal route. ``bulk_geojson`` therefore converts the WHOLE
array in a single ``to_ragged_array`` call and returns dicts in the original
positions. That is safe for the Polygon/MultiPolygon mix because
``to_ragged_array`` promotes Polygon -> MultiPolygon when both appear in one
call (verified: a mixed batch comes back as MULTIPOLYGON with three offset
levels), so no per-type grouping -- and hence no scatter-back permutation to
get wrong -- is needed. It is also the only formulation that keeps ``None``
geometries working: ``to_ragged_array`` carries a missing geometry through
POSITIONALLY as a zero-ring entry when real geometries share the call, but
RAISES on an array that is all-missing.

*Fallback, never raise.* Real overlays contain GeometryCollection,
LineString, Point, ``nan`` and all-missing arrays. On anything the bulk path
cannot represent, ``bulk_geojson`` returns ``None`` and the caller passes the
shapely objects straight through, preserving today's behaviour exactly --
including rasterio's own ``ShapeSkipWarning`` skipping of invalid shapes.
"""

import numpy as np
import shapely

# Geometry type ids the ragged-array round trip can represent. LineString,
# Point and GeometryCollection are deliberately excluded: rasterio HAPPILY
# BURNS all three, so they must reach it untouched rather than be dropped.
_POLYGON = int(shapely.GeometryType.POLYGON)          # 3
_MULTIPOLYGON = int(shapely.GeometryType.MULTIPOLYGON)  # 6
_SUPPORTED = (_POLYGON, _MULTIPOLYGON)

# shapely.get_type_id(None) -> -1 (GeometryType.MISSING). Missing geometries
# ride along inside a mixed call instead of triggering the fallback: they are
# common in WFS/ALKIS output, and gating on them would keep the fast path
# from ever firing on real data.
_MISSING = -1

# Pin the ordinate set instead of relying on the include_z=None AUTO default,
# which promotes 2D geometries to 3D and pads their z with NaN as soon as any
# geometry in the batch carries z. rasterio reads only x/y, so dropping the
# extra ordinates is burn-identical while thirding the .tolist() payload.
# include_m only exists from shapely 2.1, hence the capability probe.
try:
    shapely.to_ragged_array(np.empty(0, dtype=object), include_z=False,
                            include_m=False)
except TypeError:  # pragma: no cover - shapely 2.0
    _RAGGED_KWARGS = {"include_z": False}
except ValueError:  # empty array is rejected; the signature is fine
    _RAGGED_KWARGS = {"include_z": False, "include_m": False}
else:  # pragma: no cover - unreachable, the empty probe always raises
    _RAGGED_KWARGS = {"include_z": False, "include_m": False}


def _polygon_dicts(coords, ring_off, poly_off):
    """GeoJSON Polygon dicts from two ragged-array offset levels."""
    return [{"type": "Polygon",
             "coordinates": [coords[ring_off[r]:ring_off[r + 1]]
                             for r in range(poly_off[p], poly_off[p + 1])]}
            for p in range(len(poly_off) - 1)]


def _multipolygon_dicts(coords, ring_off, poly_off, multi_off):
    """GeoJSON MultiPolygon dicts from three ragged-array offset levels."""
    return [{"type": "MultiPolygon",
             "coordinates": [[coords[ring_off[r]:ring_off[r + 1]]
                              for r in range(poly_off[p], poly_off[p + 1])]
                             for p in range(multi_off[m], multi_off[m + 1])]}
            for m in range(len(multi_off) - 1)]


def bulk_geojson(geometries):
    """GeoJSON dicts for ``geometries``, in the ORIGINAL order, or ``None``.

    ``None`` means "not representable, use the shapely objects" -- see the
    module docstring. ``geometries`` must be a materialized 1-D sequence of
    shapely objects (GeoSeries, GeometryArray or object ndarray); a generator
    cannot be typed in bulk and is rejected via the ndim guard.
    """
    array = np.asarray(geometries, dtype=object)
    if array.ndim != 1 or array.size == 0:
        # 0-d means a generator slipped through; empty means nothing to gain
        # and to_ragged_array raises on it.
        return None

    try:
        type_ids = shapely.get_type_id(array)
    except TypeError:
        # A float nan where geopandas would store None. rasterio skips it
        # with a ShapeSkipWarning; keep that path untouched.
        return None

    present = np.unique(type_ids)
    present = present[present != _MISSING]
    if present.size == 0 or not np.isin(present, _SUPPORTED).all():
        # All-missing raises inside to_ragged_array; anything unsupported
        # (LineString/Point/GeometryCollection) must still reach rasterio.
        return None

    try:
        geom_type, coords, offsets = shapely.to_ragged_array(
            array, **_RAGGED_KWARGS)
    except (ValueError, TypeError):  # pragma: no cover - gated above
        return None

    # Branch on the RETURNED type, not on the input types: a batch holding
    # both Polygon and MultiPolygon comes back promoted to MULTIPOLYGON with
    # an extra offset level.
    coords = coords.tolist()
    offsets = [offset.tolist() for offset in offsets]
    if geom_type == shapely.GeometryType.POLYGON and len(offsets) == 2:
        return _polygon_dicts(coords, *offsets)
    if geom_type == shapely.GeometryType.MULTIPOLYGON and len(offsets) == 3:
        return _multipolygon_dicts(coords, *offsets)
    return None  # pragma: no cover - gated above


def geojson_shapes(geometries):
    """``bulk_geojson`` with the graceful fallback already applied.

    Returns something rasterio can burn in the SAME order it was given:
    GeoJSON dicts on the fast path, the untouched input otherwise.
    """
    converted = bulk_geojson(geometries)
    return geometries if converted is None else converted
