"""Backward-compatibility shim — imports from new OO modules."""
from pyorps.utils._heap import PyBinaryHeap64, py_ravel_index, py_unravel_index
from pyorps.utils._raster_context import (
    create_exclude_mask,
    path_cost,
    path_cost_uint32,
)
