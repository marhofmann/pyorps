"""PathFinder.release_device_resources: give VRAM back without dropping the finder.

Plan item 1.2 keeps a device session alive between queries, which is worth
roughly 18 B/cell. Refcounting frees it when the finder is dropped - but a
long-lived owner never drops one, so an idle interactive session pins VRAM on
a shared card. This is the escape hatch, and these tests check it actually
reclaims memory rather than merely returning.
"""
import numpy as np
import pytest
from rasterio.transform import from_origin

from pyorps import PathFinder

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU = True
    except Exception:
        GPU = False
except ImportError:
    GPU = False

gpu_only = pytest.mark.skipif(not GPU, reason="CUDA GPU not available")

CRS = "EPSG:25832"


def make_finder(graph_api="cython", n=240):
    """The transform must follow n: a fixed origin puts the corner
    coordinates outside a differently-sized raster."""
    rng = np.random.default_rng(4)
    raster = rng.integers(1, 400, (n, n)).astype(np.uint16)
    return PathFinder(
        dataset_source=raster, crs=CRS,
        transform=from_origin(0.0, float(n), 1.0, 1.0),
        source_coords=(2.5, float(n) - 2.5),
        target_coords=(float(n) - 2.5, 2.5),
        search_space_buffer_m=1000, graph_api=graph_api,
    )


class TestBackendsWithoutDeviceState:
    """A no-op on backends that hold nothing, and never an error."""

    def test_cython_backend_is_a_noop(self):
        finder = make_finder("cython")
        finder.find_route()
        finder.release_device_resources()          # must not raise
        assert finder.find_route() is not None     # still usable

    def test_before_any_graph_is_built(self):
        """Called on a fresh finder there is no backend at all yet."""
        make_finder("cython").release_device_resources()


@gpu_only
class TestGpuBackendReleasesMemory:
    def _pool_bytes(self):
        return int(cp.get_default_memory_pool().used_bytes())

    def test_release_reclaims_the_session(self):
        finder = make_finder("raster_gpu", n=600)
        finder.find_route()
        held = self._pool_bytes()
        assert held > 0, "the session should hold device memory after a route"

        finder.release_device_resources()
        after = self._pool_bytes()
        assert after < held, (
            f"release_device_resources freed nothing: {held} -> {after} bytes")

    def test_the_finder_still_works_afterwards(self):
        """Releasing costs one re-upload; it must not break the finder."""
        finder = make_finder("raster_gpu", n=400)
        first = finder.find_route()
        finder.release_device_resources()
        second = finder.find_route()

        assert list(second.path_indices) == list(first.path_indices)
        assert second.total_cost == first.total_cost

    def test_release_is_idempotent(self):
        finder = make_finder("raster_gpu", n=400)
        finder.find_route()
        finder.release_device_resources()
        finder.release_device_resources()          # must not raise

    def test_keeping_the_pool_is_available(self):
        """free_pool=False releases the session but keeps blocks cached."""
        finder = make_finder("raster_gpu", n=400)
        finder.find_route()
        finder.release_device_resources(free_pool=False)
        assert finder.find_route() is not None


if __name__ == "__main__":
    pytest.main([__file__])
