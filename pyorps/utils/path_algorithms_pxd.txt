# path_algorithms.pxd

# Import core data structures and utilities from path_core
from pyorps.utils.path_core cimport (
    int8_t, uint8_t, uint16_t, uint32_t, int32_t, float64_t, npy_intp,
    StepData, BinaryHeap, heap_init, heap_empty, heap_top, heap_push,
    heap_pop, ravel_index, unravel_index, check_path,
    precompute_directions
)

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

# Internal Dijkstra implementation for reuse across different algorithms
cdef np.ndarray[uint32_t, ndim=1] _dijkstra_2d_cython_internal(
    uint16_t[:, :] raster, uint8_t[:, :] exclude_mask,
    vector[StepData] directions, uint32_t source_idx,
    uint32_t target_idx, int rows, int cols
)


cdef double path_cost(
    np.ndarray[uint32_t, ndim=1] path,
    np.ndarray[uint16_t, ndim=2] raster_arr, int cols
)
