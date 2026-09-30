"""DEPRECATED shim: re-exports :mod:`pyorps.gui.services.tiles` (Section 15)."""
from pyorps.gui.services.tiles import (  # noqa: F401
    RasterTileLayer,
    _band_hw,
    _resolve_source,
    _valid_range,
    _write_tiled_geotiff,
    build_tile_layer,
)
