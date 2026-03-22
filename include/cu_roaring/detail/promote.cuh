#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"

namespace cu_roaring {

// ============================================================================
// Container promotion policy
//
// Controls whether array/run containers are promoted to bitmap format for
// faster point queries. The tradeoff: bitmap containers use 8 KB each
// regardless of cardinality, but membership tests are a single word read
// instead of a binary search.
//
// PROMOTE_NONE (4096): Keep CRoaring's native container choices. Arrays
//   with <= 4096 elements stay as sorted arrays. Minimum memory.
//
// PROMOTE_ALL (0): Force all containers to bitmap. Maximum query speed
//   (3-8x faster than array containers), but uses more memory for sparse
//   filters.
//
// PROMOTE_AUTO (UINT32_MAX): Let the library choose based on the GPU's
//   L2 cache size and the bitmap's memory footprint. This is the default.
//   The heuristic:
//     - If the equivalent flat bitset fits in L2 cache: keep compressed
//       (the bitset path would be faster, but the user chose roaring for
//       memory savings or set operations — don't over-promote)
//     - If the flat bitset exceeds L2 cache: promote all containers to
//       bitmap, because roaring's __ldg-cached reads outperform the
//       cache-thrashing flat bitset at this scale
//
// Any other value N: containers with cardinality > N use bitmap format.
// ============================================================================

static constexpr uint32_t PROMOTE_NONE = 4096;
static constexpr uint32_t PROMOTE_ALL  = 0;
static constexpr uint32_t PROMOTE_AUTO = UINT32_MAX;

// Convert all array and run containers to bitmap format.
// Returns a NEW GpuRoaring; the original is not modified or freed.
GpuRoaring promote_to_bitmap(const GpuRoaring& bm, cudaStream_t stream = 0);

// Cache-aware promotion: queries the current GPU's L2 cache size and
// promotes containers to bitmap only when the data is large enough that
// flat bitset access would thrash the cache.
//
// Returns a NEW GpuRoaring if promotion was applied, or a deep copy if
// no promotion is needed. The original is not modified or freed.
//
// device_id: CUDA device ordinal (default: current device)
GpuRoaring promote_auto(const GpuRoaring& bm, cudaStream_t stream = 0,
                         int device_id = -1);

// Resolve PROMOTE_AUTO to a concrete threshold for a given universe size.
// Returns PROMOTE_ALL (0) if the flat bitset would exceed L2 cache,
// or PROMOTE_NONE (4096) if it fits.
//
// This is exposed so advanced users can inspect the decision:
//   uint32_t threshold = cu_roaring::resolve_auto_threshold(universe_size);
//   printf("Auto-selected threshold: %u\n", threshold);
uint32_t resolve_auto_threshold(uint32_t universe_size, int device_id = -1);

}  // namespace cu_roaring
