#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"

namespace cu_roaring {

enum class SetOp : uint8_t {
    AND     = 0,
    OR      = 1,
    ANDNOT  = 2,  // A AND NOT B
    XOR     = 3
};

// Pairwise set operation: result = op(a, b).
// Allocates device memory for result. Caller frees with gpu_roaring_free().
GpuRoaring set_operation(const GpuRoaring& a,
                         const GpuRoaring& b,
                         SetOp op,
                         cudaStream_t stream = 0);

// Multi-bitmap AND: result = a[0] AND a[1] AND ... AND a[n-1].
GpuRoaring multi_and(const GpuRoaring* bitmaps,
                     uint32_t count,
                     cudaStream_t stream = 0);

// Multi-bitmap OR: result = a[0] OR a[1] OR ... OR a[n-1].
GpuRoaring multi_or(const GpuRoaring* bitmaps,
                    uint32_t count,
                    cudaStream_t stream = 0);

}  // namespace cu_roaring
