#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"
#include "cu_roaring/detail/promote.cuh"

// Include roaring.h in all compilation modes.
// For CUDA files that also need this, we use a pre-include header to
// suppress GCC 13 intrinsics that nvcc cannot parse.
#include <roaring/roaring.h>

namespace cu_roaring {

// Upload a CRoaring bitmap to GPU.
//
// bitmap_threshold controls array-to-bitmap promotion:
//   PROMOTE_NONE (4096) — keep CRoaring's native container choices (default)
//   PROMOTE_ALL  (0)    — promote all containers to bitmap for fastest queries
//   Any value N         — promote containers with cardinality > N to bitmap
GpuRoaring upload(const roaring_bitmap_t* cpu_bitmap,
                  cudaStream_t stream = 0,
                  uint32_t bitmap_threshold = PROMOTE_AUTO);

void gpu_roaring_free(GpuRoaring& bitmap);

GpuRoaringMeta get_meta(const roaring_bitmap_t* cpu_bitmap);

}  // namespace cu_roaring
