# _constrained_context.pxd
# Shared constrained pathfinding infrastructure: structs, state encoding,
# and precomputation used by both constrained Dijkstra and delta-stepping.
# Extracted from constrained_path_algorithms.pyx.

import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
from libcpp.vector cimport vector

from pyorps.utils._heap cimport StepData, CachedStepData, IntermediatePoint, npy_intp

# ==================== STRUCT DEFINITIONS ====================

# Precomputed valid neighbor for a given incoming direction.
cdef struct ValidNeighbor:
    uint8_t d_out
    int dr
    int dc
    float cost_factor
    float step_distance
    float angle_cost
    float tower_angle_cost

# AoS state data: all per-state fields in one struct for cache locality.
# sizeof = 24 bytes (8+8+4+1+1 + 2 padding), fits in one 64-byte cache line.
cdef struct StateData:
    double dist
    int64_t pred
    float span_dist
    uint8_t visited
    uint8_t touched

# Thread-local buffer for collecting relaxation results.
cdef struct CRelaxBuf:
    uint64_t* states
    double* dists
    int64_t* preds
    float* span_dists
    int count
    int capacity

# Lazy state for parallel delta-stepping with clearance.
cdef struct LazyState:
    double dist
    int64_t pred
    float span_dist
    uint8_t visited

# ==================== CONSTANTS ====================

cdef uint8_t FLAG_TOUCHED
cdef uint8_t FLAG_VISITED

# ==================== STATE PACKING ====================

cdef uint64_t _pack_state(uint32_t cell, uint8_t direction,
                           uint16_t span_bin, int n_dirs,
                           int n_span_bins) noexcept nogil

cdef void _unpack_state(uint64_t state, int n_dirs, int n_span_bins,
                         uint32_t* cell, uint8_t* direction,
                         uint16_t* span_bin) noexcept nogil

cdef uint64_t _pack_state_h(uint32_t cell, uint8_t direction,
                              uint16_t span_bin, uint8_t height_class,
                              int n_dirs, int n_span_bins,
                              int n_heights) noexcept nogil

cdef void _unpack_state_h(uint64_t state, int n_dirs, int n_span_bins,
                            int n_heights,
                            uint32_t* cell, uint8_t* direction,
                            uint16_t* span_bin,
                            uint8_t* height_class) noexcept nogil

# ==================== PRECOMPUTATION FUNCTIONS ====================

cdef int _check_intermediates_ptr(
    const vector[IntermediatePoint]& intermediates,
    int current_row, int current_col,
    const uint8_t* mask_ptr, const uint16_t* raster_ptr,
    int rows, int cols, float* total_cost,
) noexcept nogil

cdef vector[vector[ValidNeighbor]] _build_valid_neighbors(
    vector[StepData]& directions,
    float[:, :] angle_cost_view,
    uint8_t[:, :] angle_valid_view,
    float[:] step_dist_view,
    float[:, :] tower_angle_view,
    int n_dirs,
)

cdef void _flatten_valid_neighbors(
    vector[vector[ValidNeighbor]]& nested,
    int n_dirs,
    vector[ValidNeighbor]& flat_out,
    vector[int]& offsets_out,
)

cdef void _precompute_intermediate_cache(
    int total_cells, int n_dirs, int rows, int cols,
    const uint8_t* mask_ptr, const uint16_t* raster_ptr,
    vector[StepData]& directions,
    vector[CachedStepData]& cached_steps,
    uint8_t* icache_status, float* icache_cost,
) noexcept

cdef void _precompute_gradient_cache(
    int total_cells, int n_dirs, int rows, int cols,
    const float* dem_ptr, float cell_size,
    float max_gradient_pct, float gradient_scale,
    vector[StepData]& directions,
    uint8_t* icache_status,
    float* grad_penalty,
) noexcept

# ==================== TOWER AREA COST ====================

cdef double _area_terrain_cost(
    int row, int col, int rows, int cols,
    const uint16_t* raster_ptr,
    const float* tower_terrain_ptr,
    const int32_t* offsets,
    int n_offsets,
) noexcept nogil

cdef double _tower_terrain(
    int use_area, int row, int col,
    int d_in, int d_out, int n_dirs,
    int rows, int cols,
    const uint16_t* raster_ptr, const float* tower_terrain_ptr,
    double fallback,
    const int32_t* offsets, const int32_t* starts, const int32_t* counts,
    const float* dem_ptr, float cell_size, float gradient_scale,
) noexcept nogil

cdef double _area_terrain_cost_cell(
    int row, int col, int rows, int cols,
    const float* tower_cost_ptr,
    const int32_t* offsets,
    int n_offsets,
) noexcept nogil

cdef double _tower_terrain_cell(
    int use_area, int row, int col,
    int d_in, int d_out, int n_dirs,
    int rows, int cols,
    const float* tower_cost_ptr,
    double fallback,
    const int32_t* offsets, const int32_t* starts, const int32_t* counts,
    const float* dem_ptr, float cell_size, float gradient_scale,
) noexcept nogil

# ==================== CLEARANCE HELPERS ====================

cdef int _check_span_clearance(
    int cur_row, int cur_col,
    float span_dist_m,
    uint8_t direction,
    float tower_height, float cond_w, float cond_t, float min_clearance,
    const float* dem_ptr, const float* obstacle_ptr,
    int rows, int cols,
    vector[StepData]& directions,
    vector[CachedStepData]& cached_steps,
    float step_distance,
) noexcept nogil

cdef int _check_span_clearance_vh(
    int cur_row, int cur_col,
    float span_dist_m,
    uint8_t direction,
    float height_a, float height_b,
    float cond_w, float cond_t, float min_clearance,
    const float* dem_ptr, const float* obstacle_ptr,
    int rows, int cols,
    vector[StepData]& directions,
    vector[CachedStepData]& cached_steps,
    float step_distance,
) noexcept nogil
