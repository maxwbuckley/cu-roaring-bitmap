#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <cstring>
#include <vector>

namespace cu_roaring {

GpuRoaring promote_to_bitmap(const GpuRoaring& bm, cudaStream_t stream)
{
    const uint32_t n = bm.n_containers;
    if (n == 0) return GpuRoaring{};

    // If already all-bitmap, just copy the structure (shallow — shares data)
    if (bm.n_array_containers == 0 && bm.n_run_containers == 0) {
        // Need a deep copy since caller may free the original
        GpuRoaring result{};
        result.n_containers        = n;
        result.n_bitmap_containers = n;
        result.n_array_containers  = 0;
        result.n_run_containers    = 0;
        result.universe_size       = bm.universe_size;

        CUDA_CHECK(cudaMalloc(&result.keys, n * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&result.types, n * sizeof(ContainerType)));
        CUDA_CHECK(cudaMalloc(&result.offsets, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&result.cardinalities, n * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&result.bitmap_data, n * 1024 * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpyAsync(result.keys, bm.keys,
                                   n * sizeof(uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.types, bm.types,
                                   n * sizeof(ContainerType),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.offsets, bm.offsets,
                                   n * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, bm.cardinalities,
                                   n * sizeof(uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.bitmap_data, bm.bitmap_data,
                                   n * 1024 * sizeof(uint64_t),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }

    // Download metadata from device
    std::vector<uint16_t> h_keys(n);
    std::vector<ContainerType> h_types(n);
    std::vector<uint32_t> h_offsets(n);
    std::vector<uint16_t> h_cards(n);

    CUDA_CHECK(cudaMemcpyAsync(h_keys.data(), bm.keys,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_types.data(), bm.types,
                               n * sizeof(ContainerType),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_offsets.data(), bm.offsets,
                               n * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_cards.data(), bm.cardinalities,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Download data pools
    std::vector<uint64_t> h_bitmap_data;
    if (bm.n_bitmap_containers > 0 && bm.bitmap_data) {
        h_bitmap_data.resize(
            static_cast<size_t>(bm.n_bitmap_containers) * 1024);
        CUDA_CHECK(cudaMemcpyAsync(
            h_bitmap_data.data(), bm.bitmap_data,
            h_bitmap_data.size() * sizeof(uint64_t),
            cudaMemcpyDeviceToHost, stream));
    }

    // Compute total array elements for download
    uint32_t total_array_elems = 0;
    uint32_t total_run_pairs   = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (h_types[i] == ContainerType::ARRAY)
            total_array_elems += h_cards[i];
        else if (h_types[i] == ContainerType::RUN)
            total_run_pairs += h_cards[i];
    }

    std::vector<uint16_t> h_array_data;
    if (total_array_elems > 0 && bm.array_data) {
        h_array_data.resize(total_array_elems);
        CUDA_CHECK(cudaMemcpyAsync(
            h_array_data.data(), bm.array_data,
            total_array_elems * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, stream));
    }

    std::vector<uint16_t> h_run_data;
    if (total_run_pairs > 0 && bm.run_data) {
        h_run_data.resize(static_cast<size_t>(total_run_pairs) * 2);
        CUDA_CHECK(cudaMemcpyAsync(
            h_run_data.data(), bm.run_data,
            h_run_data.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Build all-bitmap pool: every container becomes 1024 uint64_t words
    std::vector<uint64_t> all_bitmap(static_cast<size_t>(n) * 1024, 0);
    std::vector<uint32_t> new_offsets(n);
    std::vector<ContainerType> new_types(n, ContainerType::BITMAP);

    for (uint32_t i = 0; i < n; ++i) {
        uint64_t* dst = all_bitmap.data() + static_cast<size_t>(i) * 1024;
        new_offsets[i] = static_cast<uint32_t>(
            static_cast<size_t>(i) * 1024 * sizeof(uint64_t));

        if (h_types[i] == ContainerType::BITMAP) {
            uint32_t src_idx = h_offsets[i] / sizeof(uint64_t);
            std::memcpy(dst, h_bitmap_data.data() + src_idx,
                        1024 * sizeof(uint64_t));
        } else if (h_types[i] == ContainerType::ARRAY) {
            uint32_t src_idx = h_offsets[i] / sizeof(uint16_t);
            for (uint32_t j = 0; j < h_cards[i]; ++j) {
                uint16_t val = h_array_data[src_idx + j];
                dst[val / 64] |= 1ULL << (val % 64);
            }
        } else if (h_types[i] == ContainerType::RUN) {
            uint32_t src_idx = h_offsets[i] / sizeof(uint16_t);
            for (uint32_t r = 0; r < h_cards[i]; ++r) {
                uint16_t start = h_run_data[src_idx + r * 2];
                uint16_t len   = h_run_data[src_idx + r * 2 + 1];
                for (uint32_t v = start;
                     v <= static_cast<uint32_t>(start) + len; ++v) {
                    dst[v / 64] |= 1ULL << (v % 64);
                }
            }
        }
    }

    // Upload the new all-bitmap structure
    GpuRoaring result{};
    result.n_containers        = n;
    result.n_bitmap_containers = n;
    result.n_array_containers  = 0;
    result.n_run_containers    = 0;
    result.universe_size       = bm.universe_size;

    CUDA_CHECK(cudaMalloc(&result.keys, n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&result.types, n * sizeof(ContainerType)));
    CUDA_CHECK(cudaMalloc(&result.offsets, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&result.cardinalities, n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&result.bitmap_data,
                          static_cast<size_t>(n) * 1024 * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpyAsync(result.keys, h_keys.data(),
                               n * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.types, new_types.data(),
                               n * sizeof(ContainerType),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.offsets, new_offsets.data(),
                               n * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, h_cards.data(),
                               n * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.bitmap_data, all_bitmap.data(),
                               static_cast<size_t>(n) * 1024 * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

// ============================================================================
// Cache-aware automatic promotion
// ============================================================================

uint32_t resolve_auto_threshold(uint32_t universe_size, int device_id)
{
    if (device_id < 0) {
        CUDA_CHECK(cudaGetDevice(&device_id));
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    size_t l2_bytes          = static_cast<size_t>(prop.l2CacheSize);
    size_t flat_bitset_bytes = (static_cast<size_t>(universe_size) + 7) / 8;

    // If the flat bitset exceeds L2 cache, scattered bitset reads will
    // thrash the cache. Roaring with all-bitmap containers wins here
    // because __ldg routes through the read-only texture cache and the
    // key-indexed access pattern has better spatial locality than random
    // bitset lookups across a 100+ MB array.
    //
    // If the flat bitset fits in L2, bitset point queries are essentially
    // free (single L2-cached read), so there's no query speed reason to
    // promote. The user chose roaring for memory savings or set operations,
    // not for query speed at this scale.
    if (flat_bitset_bytes > l2_bytes) {
        return PROMOTE_ALL;
    }
    return PROMOTE_NONE;
}

GpuRoaring promote_auto(const GpuRoaring& bm, cudaStream_t stream, int device_id)
{
    uint32_t threshold = resolve_auto_threshold(bm.universe_size, device_id);

    if (threshold == PROMOTE_ALL &&
        (bm.n_array_containers > 0 || bm.n_run_containers > 0)) {
        return promote_to_bitmap(bm, stream);
    }

    // No promotion needed — return a deep copy so the caller has
    // consistent ownership semantics (always gets a new GpuRoaring).
    return promote_to_bitmap(bm, stream);  // handles the all-bitmap copy path
}

}  // namespace cu_roaring
