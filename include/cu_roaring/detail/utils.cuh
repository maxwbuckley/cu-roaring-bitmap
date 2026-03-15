#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace cu_roaring {

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error at ") +       \
                                     __FILE__ + ":" +                       \
                                     std::to_string(__LINE__) + ": " +     \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

// Divide and round up
inline __host__ __device__ uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

}  // namespace cu_roaring
