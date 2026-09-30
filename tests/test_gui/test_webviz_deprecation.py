"""Phase 9: pyorps.webviz is deprecated and re-exports the gui services."""
import importlib
import sys
import warnings

import pytest


def test_webviz_import_warns_deprecation():
    for module in list(sys.modules):
        if module.startswith("pyorps.webviz"):
            del sys.modules[module]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("pyorps.webviz")
    deprecations = [w for w in caught
                    if issubclass(w.category, DeprecationWarning)
                    and "pyorps.gui" in str(w.message)]
    assert deprecations, "importing pyorps.webviz must point to pyorps.gui"


def test_webviz_reexports_gui_services():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pyorps.gui.services import cost as gui_cost
        from pyorps.gui.services import geo as gui_geo
        from pyorps.gui.services import tiles as gui_tiles
        from pyorps.gui.services.routing import (
            route_through_points as gui_rtp,
        )
        from pyorps.webviz import builder, cost, geo, reroute, tiles

    assert geo.gdf_to_wgs84_geojson is gui_geo.gdf_to_wgs84_geojson
    assert cost.evaluate_route_cost is gui_cost.evaluate_route_cost
    assert tiles.build_tile_layer is gui_tiles.build_tile_layer
    assert reroute.route_through_points is gui_rtp
    # the v1 builder API shape is preserved (finder, results)
    assert builder.resolve_backend("dijkstra", "cpu") == \
        ("cython", "dijkstra")


def test_viz_extra_aliases_gui_extra():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    extras = config["project"]["optional-dependencies"]
    assert set(extras["viz"]) == set(extras["gui"])
