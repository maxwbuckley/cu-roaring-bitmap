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
// A threshold value N means "promote any container whose cardinality is > N
// to a bitmap." Named constants:
//
// PROMOTE_KEEP_DEFAULT (4096): Match CRoaring's native array-vs-bitmap
//   cutoff. Arrays with cardinality <= 4096 stay as sorted arrays. This is
//   CRoaring's own default, so upload() with this threshold preserves every
//   CRoaring container choice.
//
// PROMOTE_ALL (0): Force every container to bitmap. Maximum query speed
//   (3-8x faster than array containers), but uses more memory for sparse
//   filters (8 KB per container regardless of cardinality).
//
// PROMOTE_AUTO (UINT32_MAX): Let the library choose based on how the flat
//   bitset for this universe compares to the current GPU's L2 cache. The
//   heuristic (implemented by resolve_auto_threshold):
//     - If the flat bitset fits in L2 (flat_bytes * 2 <= L2 size), arrays
//       are fast enough — return PROMOTE_KEEP_DEFAULT.
//     - Otherwise the flat bitset would thrash L2 and the promoted Roaring
//       is the right choice — return PROMOTE_ALL.
//
// Any other value N: containers with cardinality > N use bitmap format.
// ============================================================================

static constexpr uint32_t PROMOTE_KEEP_DEFAULT = 4096;
static constexpr uint32_t PROMOTE_ALL          = 0;
static constexpr uint32_t PROMOTE_AUTO         = UINT32_MAX;

// Convert all array and run containers in `bm` to bitmap format. Returns a
// NEW GpuRoaring that owns its memory; the original is not modified or freed.
//
// Stream-ordered: all work is enqueued on `stream` and no host sync is
// forced. Callers that need host-side visibility must sync `stream`
// themselves.
GpuRoaring promote_to_bitmap(const GpuRoaring& bm, cudaStream_t stream = 0);

// Cache-aware promotion: queries the current GPU's L2 cache size via
// cudaDeviceGetAttribute(cudaDevAttrL2CacheSize) and promotes every
// container to bitmap only when the equivalent flat bitset would overflow
// L2.
//
// Returns a NEW GpuRoaring. If promotion was applied, the result is an
// all-bitmap Roaring. If no promotion was needed, the result is a deep
// copy of the input (same shape, independently owned device memory) so
// the caller's ownership model is unchanged.
//
// Stream-ordered: no implicit host sync.
//
// device_id: CUDA device ordinal (-1 = current device).
GpuRoaring promote_auto(const GpuRoaring& bm, cudaStream_t stream = 0,
                         int device_id = -1);

// Resolve PROMOTE_AUTO to a concrete threshold for a given universe size
// on the target device. Returns PROMOTE_ALL (0) if the flat bitset for
// `universe_size` would overflow L2, or PROMOTE_KEEP_DEFAULT (4096)
// otherwise.
//
// Exposed so advanced users can inspect the decision:
//   uint32_t t = cu_roaring::resolve_auto_threshold(universe_size);
//
// device_id: CUDA device ordinal (-1 = current device).
uint32_t resolve_auto_threshold(uint32_t universe_size, int device_id = -1);

// Deep copy a GpuRoaring on a stream. The result owns independent memory
// and can be freed with gpu_roaring_free(). Stream-ordered, no host sync.
//
// Requires that the source's `array_pool_bytes` / `run_pool_bytes` fields
// are populated if the corresponding pools are non-null. Every code path
// that allocates those pools sets the tracking fields.
GpuRoaring gpu_roaring_deep_copy(const GpuRoaring& bm, cudaStream_t stream = 0);

}  // namespace cu_roaring
