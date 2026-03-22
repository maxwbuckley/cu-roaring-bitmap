#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"

namespace cu_roaring {

// Array/run → bitmap promotion threshold.
// Containers with cardinality above this value use bitmap format.
// Default (4096) matches CRoaring's native threshold.
// Set to 0 to promote ALL containers to bitmap (fastest queries, more memory).
static constexpr uint32_t PROMOTE_NONE = 4096;
static constexpr uint32_t PROMOTE_ALL  = 0;

// Convert all array and run containers to bitmap format.
// Returns a NEW GpuRoaring; the original is not modified or freed.
// The resulting bitmap has zero array and run containers — every container
// is an 8 KB bitmap, giving O(1) membership tests with no binary search
// inside the container.
GpuRoaring promote_to_bitmap(const GpuRoaring& bm, cudaStream_t stream = 0);

}  // namespace cu_roaring
