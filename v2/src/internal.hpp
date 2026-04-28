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

// Layout of the host-side per-bitmap metadata block. Computed identically by
// upload_batch (which builds it) and promote_batch (which clones + adjusts it),
// so the two paths must agree byte-for-byte.
struct HostMetaLayout {
    size_t total_card_off;
    size_t universe_off;
    size_t cstart_off;
    size_t kstart_off;
    size_t n_bitmap_off;
    size_t total_bytes;
};

inline HostMetaLayout compute_host_meta_layout(uint32_t n) {
    constexpr size_t A = 8;
    HostMetaLayout M{};
    size_t off = 0;
    M.total_card_off = off; off = align_up(off + n * sizeof(uint64_t), A);
    M.universe_off   = off; off = align_up(off + n * sizeof(uint32_t), A);
    M.cstart_off     = off; off = align_up(off + (n + 1u) * sizeof(uint32_t), A);
    M.kstart_off     = off; off = align_up(off + (n + 1u) * sizeof(uint32_t), A);
    M.n_bitmap_off   = off; off = align_up(off + n * sizeof(uint32_t), A);
    M.total_bytes    = (off == 0) ? A : off;
    return M;
}

} // namespace cu_roaring::v2::detail
