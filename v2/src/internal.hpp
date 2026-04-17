#pragma once
// Private header for the v2 library. Not installed; not part of the public API.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cu_roaring::v2::detail {

#define CU_ROARING_V2_CHECK_CUDA(expr)                                           \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            throw std::runtime_error(                                            \
                std::string(#expr) + " failed: " +                               \
                cudaGetErrorString(_err) + " at " + __FILE__ + ":" +             \
                std::to_string(__LINE__));                                       \
        }                                                                        \
    } while (0)

constexpr size_t align_up(size_t v, size_t a) {
    return (v + a - 1u) & ~(a - 1u);
}

} // namespace cu_roaring::v2::detail
