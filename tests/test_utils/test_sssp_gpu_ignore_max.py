"""Regression test: GPU raster SSSP with ignore_max=False must not treat
value-0 cells as impassable.

Bug (fixed in Phase 3 of the feasibility plan): the kernels compared
``sv == (unsigned short)max_cost`` — with ignore_max=False the host passes
max_cost = 65536, which truncates to 0 in the cast, silently forbidding
every 0-cost cell. The comparison now happens in int domain.
"""
import unittest

import numpy as np
import pytest
from rasterio.transform import from_origin

cupy = pytest.importorskip("cupy")

from pyorps import PathFinder  # noqa: E402

CRS = "EPSG:25832"
TRANSFORM = from_origin(0.0, 10.0, 1.0, 1.0)


class TestIgnoreMaxFalseZeroCells(unittest.TestCase):
    def test_zero_value_wall_is_traversable(self):
        """A full-height wall of 0-cost cells separates source and target.

        Under the truncation bug the wall is impassable and no path
        exists; with the fix the route crosses it (0-cost cells are the
        cheapest possible)."""
        raster = np.full((10, 20), 100, dtype=np.uint16)
        raster[:, 10] = 0  # zero-cost wall, full height

        finder = PathFinder(
            dataset_source=raster,
            crs=CRS,
            transform=TRANSFORM,
            source_coords=(2.5, 5.5),
            target_coords=(17.5, 5.5),
            search_space_buffer_m=50,
            graph_api="raster_gpu",
            ignore_max_cost=False,
        )
        path = finder.find_route()

        cols = np.array(path.path_indices) % raster.shape[1]
        self.assertIn(10, cols.tolist())  # the route crosses the wall


if __name__ == "__main__":
    unittest.main()
