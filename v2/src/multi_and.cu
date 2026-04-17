#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace cu_roaring::v2 {

namespace {

// One block per common container. Each block reads the corresponding bitmap
// from every input, AND-reduces word-by-word into the output, and block-reduces
// the popcount to write the per-container cardinality and the global total.
__global__ void multi_and_kernel(
    const uint64_t* const* __restrict__ input_ptrs,  // [n_common * count]
    uint32_t       count,
    uint64_t*      dst_bitmap_pool,
    uint16_t*      dst_cards,
    unsigned long long* total_card)
{
    __shared__ uint32_t warp_sum[8];

    const uint32_t k = blockIdx.x;
    uint64_t*       dst = dst_bitmap_pool + static_cast<size_t>(k) * 1024u;
    const uint64_t* const* my_inputs =
        input_ptrs + static_cast<size_t>(k) * count;

    uint32_t local_pop = 0;
    for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
        uint64_t acc = ~0ULL;
        #pragma unroll 1
        for (uint32_t i = 0; i < count; ++i) {
            acc &= my_inputs[i][w];
        }
        dst[w] = acc;
        local_pop += static_cast<uint32_t>(__popcll(acc));
    }

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_pop += __shfl_down_sync(~0u, local_pop, offset);
    }
    if (lane == 0) warp_sum[warp] = local_pop;
    __syncthreads();

    if (warp == 0) {
        uint32_t v = (lane < 8u) ? warp_sum[lane] : 0u;
        for (int offset = 4; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(~0u, v, offset);
        }
        if (lane == 0) {
            dst_cards[k] = static_cast<uint16_t>(v > 65535u ? 0u : v);
            atomicAdd(total_card, static_cast<unsigned long long>(v));
        }
    }
}

struct Layout {
    size_t keys_off, types_off, offsets_off, cards_off, kidx_off, bmp_off;
    size_t total_bytes;
    uint32_t key_index_len;
};

Layout compute_layout(uint32_t n_common, uint16_t max_key) {
    using detail::align_up;
    Layout L{};
    L.key_index_len = static_cast<uint32_t>(max_key) + 1u;

    constexpr size_t A = 8;
    size_t off = 0;
    L.keys_off    = off; off = align_up(off + n_common * sizeof(uint16_t), A);
    L.types_off   = off; off = align_up(off + n_common * sizeof(ContainerType), A);
    L.offsets_off = off; off = align_up(off + n_common * sizeof(uint32_t), A);
    L.cards_off   = off; off = align_up(off + n_common * sizeof(uint16_t), A);
    L.kidx_off    = off; off = align_up(off + L.key_index_len * sizeof(uint16_t), A);
    L.bmp_off     = off; off += static_cast<size_t>(n_common) * 1024u * sizeof(uint64_t);
    L.total_bytes = align_up(off, A);
    return L;
}

void validate_all_bitmap(const GpuRoaring* inputs, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (inputs[i].n_array_containers > 0 || inputs[i].n_run_containers > 0) {
            throw std::invalid_argument(
                "multi_and: input " + std::to_string(i) +
                " has non-bitmap containers; call promote_to_bitmap() first");
        }
    }
}

} // namespace

GpuRoaring multi_and(const GpuRoaring* inputs, uint32_t count, cudaStream_t stream) {
    if (count == 0) {
        throw std::invalid_argument("multi_and: count must be >= 1");
    }
    validate_all_bitmap(inputs, count);

    // Short-circuit: if any input is empty, the intersection is empty.
    for (uint32_t i = 0; i < count; ++i) {
        if (inputs[i].n_containers == 0) return GpuRoaring{};
    }

    // D2H each input's keys[] for the host-side intersection. Sizes are small
    // (n_containers * 2 bytes); even 8 inputs × 10K keys is 160 KB total.
    std::vector<std::vector<uint16_t>> host_keys(count);
    for (uint32_t i = 0; i < count; ++i) {
        host_keys[i].resize(inputs[i].n_containers);
        CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
            host_keys[i].data(), inputs[i].keys,
            inputs[i].n_containers * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, stream));
    }
    CU_ROARING_V2_CHECK_CUDA(cudaStreamSynchronize(stream));

    // Seed the intersection with the input that has the fewest containers — any
    // common key must appear there, so this minimises subsequent set_intersection
    // work.
    uint32_t seed = 0;
    for (uint32_t i = 1; i < count; ++i) {
        if (host_keys[i].size() < host_keys[seed].size()) seed = i;
    }
    std::vector<uint16_t> common = host_keys[seed];
    for (uint32_t i = 0; i < count; ++i) {
        if (i == seed) continue;
        std::vector<uint16_t> next;
        next.reserve(common.size());
        std::set_intersection(common.begin(), common.end(),
                              host_keys[i].begin(), host_keys[i].end(),
                              std::back_inserter(next));
        common = std::move(next);
        if (common.empty()) break;
    }

    if (common.empty()) return GpuRoaring{};

    const uint32_t n_common = static_cast<uint32_t>(common.size());
    const uint16_t max_key  = common.back();
    const uint32_t universe = (static_cast<uint32_t>(max_key) + 1u) << 16;
    const Layout   L        = compute_layout(n_common, max_key);

    // Build the pointer table on host: for each (common_key, input) pair, resolve
    // the device pointer to that input's 8 KB bitmap container. Binary search on
    // the already-downloaded host_keys[i] — O(log n) per lookup.
    std::vector<const uint64_t*> h_ptrs(static_cast<size_t>(n_common) * count);
    for (uint32_t k = 0; k < n_common; ++k) {
        const uint16_t key = common[k];
        for (uint32_t i = 0; i < count; ++i) {
            auto it = std::lower_bound(host_keys[i].begin(), host_keys[i].end(), key);
            const uint32_t idx = static_cast<uint32_t>(it - host_keys[i].begin());
            h_ptrs[static_cast<size_t>(k) * count + i] =
                inputs[i].bitmap_data + static_cast<size_t>(idx) * 1024u;
        }
    }

    // Output layout + metadata built on host and H2D'd in one batch.
    std::vector<ContainerType> h_types(n_common, ContainerType::BITMAP);
    std::vector<uint32_t>      h_offs(n_common);
    for (uint32_t k = 0; k < n_common; ++k) h_offs[k] = k * 8192u;
    std::vector<uint16_t>      h_kidx(L.key_index_len, 0xFFFFu);
    for (uint32_t k = 0; k < n_common; ++k) h_kidx[common[k]] = static_cast<uint16_t>(k);

    char* d_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_buf),
                                             L.total_bytes, stream));

    auto* d_keys    = reinterpret_cast<uint16_t*>(d_buf + L.keys_off);
    auto* d_types   = reinterpret_cast<ContainerType*>(d_buf + L.types_off);
    auto* d_offs    = reinterpret_cast<uint32_t*>(d_buf + L.offsets_off);
    auto* d_cards   = reinterpret_cast<uint16_t*>(d_buf + L.cards_off);
    auto* d_kidx    = reinterpret_cast<uint16_t*>(d_buf + L.kidx_off);
    auto* d_bmp     = reinterpret_cast<uint64_t*>(d_buf + L.bmp_off);

    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_keys,  common.data(),
        n_common * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_types, h_types.data(),
        n_common * sizeof(ContainerType), cudaMemcpyHostToDevice, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_offs,  h_offs.data(),
        n_common * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_kidx,  h_kidx.data(),
        L.key_index_len * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    // Pointer table + scalar total-cardinality counter live outside the result
    // allocation and are freed before multi_and returns.
    const uint64_t** d_ptrs    = nullptr;
    unsigned long long* d_total = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_ptrs),
        h_ptrs.size() * sizeof(const uint64_t*), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_total),
        sizeof(unsigned long long), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_ptrs, h_ptrs.data(),
        h_ptrs.size() * sizeof(const uint64_t*), cudaMemcpyHostToDevice, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemsetAsync(d_total, 0,
        sizeof(unsigned long long), stream));

    multi_and_kernel<<<n_common, 256, 0, stream>>>(
        d_ptrs, count, d_bmp, d_cards, d_total);
    CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());

    unsigned long long total_card_host = 0;
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(&total_card_host, d_total,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaStreamSynchronize(stream));

    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_ptrs, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_total, stream));

    GpuRoaring r{};
    r._alloc_base         = d_buf;
    r.n_containers        = n_common;
    r.n_bitmap_containers = n_common;
    r.universe_size       = universe;
    r.max_key             = max_key;
    r.total_cardinality   = static_cast<uint64_t>(total_card_host);
    r.keys          = d_keys;
    r.types         = d_types;
    r.offsets       = d_offs;
    r.cardinalities = d_cards;
    r.key_index     = d_kidx;
    r.bitmap_data   = d_bmp;
    return r;
}

} // namespace cu_roaring::v2
