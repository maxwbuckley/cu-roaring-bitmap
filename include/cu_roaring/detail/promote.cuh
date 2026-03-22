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
// PROMOTE_AUTO (UINT32_MAX): Let the library choose based on the universe
//   size. This is the default. The heuristic:
//     - Universe <= ~4M (<=64 containers): keep arrays. The entire
//       structure fits in L1/L2 cache and array queries are fast.
//     - Universe > ~4M (>64 containers): promote all to bitmap. The key
//       binary search (7+ steps) combined with array binary search
//       (up to 12 steps) makes array queries 4-10x slower than bitmap.
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
