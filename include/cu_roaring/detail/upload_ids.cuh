#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"

namespace cu_roaring {

// Create GPU Roaring from sorted, deduplicated uint32_t IDs on host.
// Does NOT depend on CRoaring. Containers with <= 4096 elements use array
// format; denser containers use bitmap format.
GpuRoaring upload_from_sorted_ids(const uint32_t* sorted_ids,
                                  uint32_t n_ids,
                                  uint32_t universe_size,
                                  cudaStream_t stream = 0);

}  // namespace cu_roaring
