#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"

namespace cu_roaring {

// Decompress GpuRoaring to a flat bitset compatible with cuvs::core::bitset.
// Returns device pointer to uint32_t array of ceil(universe_size / 32) words.
// Caller frees with cudaFree.
uint32_t* decompress_to_bitset(const GpuRoaring& bitmap,
                               cudaStream_t stream = 0);

// Decompress into a pre-allocated buffer.
void decompress_to_bitset(const GpuRoaring& bitmap,
                          uint32_t* output,
                          uint32_t output_size_words,
                          cudaStream_t stream = 0);

}  // namespace cu_roaring
