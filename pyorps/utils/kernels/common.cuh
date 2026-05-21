#pragma once
#include <cooperative_groups.h>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

// Control buffer indices
#define CTL_COUNT_A 0
#define CTL_COUNT_B 1
#define CTL_SETTLED 2
#define CTL_PENDING 3
#define CTL_NEAR    4
#define CTL_FAR     5
#define CTL_BUCKET  6
#define CTL_DONE    7
#define CTL_EARLY_CTR 8
#define CTL_MIN_DIST 9
#define CTL_BARRIER_CNT 10
#define CTL_BARRIER_SENSE 11
#define CTL_TOWER_COUNT 12
#define CTL_QUEUE_OVERFLOW 13
#define CTL_FULL_SCANS 14
#define MAX_FULL_SCANS 3

// TowerRecord: 24 bytes, 8-byte aligned
struct __align__(8) TowerRecord {
    long long state;        // state where tower was placed (after move)
    long long pred_state;   // predecessor state (before tower placement)
    __half span_dist;       // span distance at tower placement
    __half tower_height;    // tower height used
    // 4 bytes padding implicit from align(8) on 24-byte struct
};
