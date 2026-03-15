#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"

// Include roaring.h in all compilation modes.
// For CUDA files that also need this, we use a pre-include header to
// suppress GCC 13 intrinsics that nvcc cannot parse.
#include <roaring/roaring.h>

namespace cu_roaring {

GpuRoaring upload(const roaring_bitmap_t* cpu_bitmap,
                  cudaStream_t stream = 0);

void gpu_roaring_free(GpuRoaring& bitmap);

GpuRoaringMeta get_meta(const roaring_bitmap_t* cpu_bitmap);

}  // namespace cu_roaring
