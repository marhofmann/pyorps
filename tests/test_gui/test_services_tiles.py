"""Unit tests: pyorps.gui.services.tiles — tile serving obeys C6-C9."""
import math

import pytest

requests = pytest.importorskip("requests")

from pyorps.gui.services.tiles import build_tile_layer


def test_tile_layer_serves_png(raster_path, tmp_path):
    layer = build_tile_layer(raster_path, name="cost", work_dir=tmp_path)
    try:
        # C6: URL advertised via localhost, never a raw ::1
        assert layer.tile_url.startswith("http://localhost:")
        # C7: bounds are [[south, west], [north, east]]
        (s, w), (n, e) = layer.bounds
        assert s < n and w < e
        lat, lon = (s + n) / 2, (w + e) / 2
        z = 15
        nn = 2 ** z
        x = int((lon + 180) / 360 * nn)
        y = int((1 - math.log(math.tan(math.radians(lat)) +
                              1 / math.cos(math.radians(lat))) / math.pi)
                / 2 * nn)
        url = (layer.tile_url.replace("{z}", str(z)).replace("{x}", str(x))
               .replace("{y}", str(y)))
        resp = requests.get(url, timeout=30)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")
    finally:
        shutdown = getattr(layer.tile_client, "shutdown", None)
        if callable(shutdown):
            shutdown()


def test_tile_layer_keeps_no_raster_environment_open(raster_path, tmp_path):
    """A tile layer must not leave a rasterio environment open in the calling thread.

    Seen on CI (Linux, Python 3.14): the tile client's reader kept one open; when the client was garbage
    collected inside another raster read, that read failed with "No GDAL environment exists".
    """
    import gc

    import rasterio
    import rasterio.env

    assert not rasterio.env.hasenv(), "something before this test left a rasterio environment open"
    layer = build_tile_layer(raster_path, name="cost", work_dir=tmp_path)
    assert not rasterio.env.hasenv()
    # the failing sequence: the client goes away while a raster read is in progress
    shutdown = getattr(layer.tile_client, "shutdown", None)
    if callable(shutdown):
        shutdown()
    with rasterio.Env():
        del layer, shutdown
        gc.collect()
        assert rasterio.env.hasenv()


def test_numpy_source_requires_crs_and_transform(tmp_path):
    import numpy as np

    with pytest.raises(ValueError, match="crs"):
        build_tile_layer(np.ones((10, 10), dtype="uint16"),
                         work_dir=tmp_path)
