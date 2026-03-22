#include "cu_roaring/detail/upload.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <roaring/roaring.h>
#include <roaring/containers/array.h>
#include <roaring/containers/bitset.h>
#include <roaring/containers/run.h>

#include <algorithm>
#include <cstring>
#include <vector>

// CRoaring v4.x puts internal types in roaring::internal namespace
using roaring::internal::array_container_t;
using roaring::internal::bitset_container_t;
using roaring::internal::run_container_t;
using roaring::internal::rle16_t;

// CRoaring internal type codes
static constexpr uint8_t CROARING_BITSET = 1;
static constexpr uint8_t CROARING_ARRAY  = 2;
static constexpr uint8_t CROARING_RUN    = 3;

namespace cu_roaring {

GpuRoaringMeta get_meta(const roaring_bitmap_t* cpu_bitmap) {
    GpuRoaringMeta meta{};
    const roaring_array_t& ra = cpu_bitmap->high_low_container;
    meta.n_containers = static_cast<uint32_t>(ra.size);

    size_t total_array_elems = 0;
    size_t total_run_pairs   = 0;

    for (int32_t i = 0; i < ra.size; ++i) {
        uint8_t tc = ra.typecodes[i];
        if (tc == CROARING_BITSET) {
            ++meta.n_bitmap_containers;
        } else if (tc == CROARING_ARRAY) {
            ++meta.n_array_containers;
            const auto* ac =
                static_cast<const array_container_t*>(ra.containers[i]);
            total_array_elems += ac->cardinality;
        } else if (tc == CROARING_RUN) {
            ++meta.n_run_containers;
            const auto* rc =
                static_cast<const run_container_t*>(ra.containers[i]);
            total_run_pairs += rc->n_runs;
        }
    }

    if (meta.n_containers > 0) {
        uint16_t max_key = ra.keys[ra.size - 1];
        meta.universe_size = (static_cast<uint32_t>(max_key) + 1) << 16;
    }

    meta.total_bytes =
        meta.n_containers * sizeof(uint16_t) +
        meta.n_containers * sizeof(ContainerType) +
        meta.n_containers * sizeof(uint32_t) +
        meta.n_containers * sizeof(uint16_t) +
        meta.n_bitmap_containers * 1024 * sizeof(uint64_t) +
        total_array_elems * sizeof(uint16_t) +
        total_run_pairs * 2 * sizeof(uint16_t);

    return meta;
}

GpuRoaring upload(const roaring_bitmap_t* cpu_bitmap, cudaStream_t stream,
                  uint32_t bitmap_threshold) {
    const roaring_array_t& ra = cpu_bitmap->high_low_container;
    const uint32_t n = static_cast<uint32_t>(ra.size);

    if (n == 0) {
        return GpuRoaring{};
    }

    // Phase 1: Scan containers and build host-side SoA buffers
    std::vector<uint16_t>      h_keys(n);
    std::vector<ContainerType> h_types(n);
    std::vector<uint32_t>      h_offsets(n);
    std::vector<uint16_t>      h_cardinalities(n);

    size_t total_bitmap_words = 0;
    size_t total_array_elems  = 0;
    size_t total_run_pairs    = 0;
    uint32_t n_bitmap = 0, n_array = 0, n_run = 0;

    for (uint32_t i = 0; i < n; ++i) {
        uint8_t tc = ra.typecodes[i];
        if (tc == CROARING_BITSET) {
            total_bitmap_words += 1024;
            ++n_bitmap;
        } else if (tc == CROARING_ARRAY) {
            const auto* ac =
                static_cast<const array_container_t*>(ra.containers[i]);
            total_array_elems += ac->cardinality;
            ++n_array;
        } else if (tc == CROARING_RUN) {
            const auto* rc =
                static_cast<const run_container_t*>(ra.containers[i]);
            total_run_pairs += rc->n_runs;
            ++n_run;
        }
    }

    // Allocate host staging buffers (pinned for async transfer)
    uint64_t* h_bitmap_pool = nullptr;
    uint16_t* h_array_pool  = nullptr;
    uint16_t* h_run_pool    = nullptr;

    if (total_bitmap_words > 0) {
        CUDA_CHECK(cudaMallocHost(&h_bitmap_pool,
                                  total_bitmap_words * sizeof(uint64_t)));
    }
    if (total_array_elems > 0) {
        CUDA_CHECK(cudaMallocHost(&h_array_pool,
                                  total_array_elems * sizeof(uint16_t)));
    }
    if (total_run_pairs > 0) {
        CUDA_CHECK(cudaMallocHost(&h_run_pool,
                                  total_run_pairs * 2 * sizeof(uint16_t)));
    }

    // Phase 2: Fill host buffers
    size_t bitmap_offset = 0;
    size_t array_offset  = 0;
    size_t run_offset    = 0;

    for (uint32_t i = 0; i < n; ++i) {
        h_keys[i] = ra.keys[i];
        uint8_t tc = ra.typecodes[i];

        if (tc == CROARING_BITSET) {
            h_types[i] = ContainerType::BITMAP;
            h_offsets[i] = static_cast<uint32_t>(bitmap_offset * sizeof(uint64_t));
            const auto* bc =
                static_cast<const bitset_container_t*>(ra.containers[i]);
            h_cardinalities[i] = static_cast<uint16_t>(
                bc->cardinality > 65535 ? 0 : bc->cardinality);
            std::memcpy(h_bitmap_pool + bitmap_offset, bc->words,
                        1024 * sizeof(uint64_t));
            bitmap_offset += 1024;
        } else if (tc == CROARING_ARRAY) {
            h_types[i] = ContainerType::ARRAY;
            h_offsets[i] = static_cast<uint32_t>(array_offset * sizeof(uint16_t));
            const auto* ac =
                static_cast<const array_container_t*>(ra.containers[i]);
            h_cardinalities[i] = static_cast<uint16_t>(ac->cardinality);
            std::memcpy(h_array_pool + array_offset, ac->array,
                        ac->cardinality * sizeof(uint16_t));
            array_offset += ac->cardinality;
        } else if (tc == CROARING_RUN) {
            h_types[i] = ContainerType::RUN;
            h_offsets[i] = static_cast<uint32_t>(run_offset * 2 * sizeof(uint16_t));
            const auto* rc =
                static_cast<const run_container_t*>(ra.containers[i]);
            h_cardinalities[i] = static_cast<uint16_t>(rc->n_runs);
            std::memcpy(h_run_pool + run_offset * 2, rc->runs,
                        rc->n_runs * sizeof(rle16_t));
            run_offset += rc->n_runs;
        }
    }

    // Phase 3: Allocate device memory and transfer
    GpuRoaring result{};
    result.n_containers        = n;
    result.n_bitmap_containers = n_bitmap;
    result.n_array_containers  = n_array;
    result.n_run_containers    = n_run;

    if (n > 0) {
        uint16_t max_key = ra.keys[ra.size - 1];
        result.universe_size = (static_cast<uint32_t>(max_key) + 1) << 16;
    }

    CUDA_CHECK(cudaMalloc(&result.keys, n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&result.types, n * sizeof(ContainerType)));
    CUDA_CHECK(cudaMalloc(&result.offsets, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&result.cardinalities, n * sizeof(uint16_t)));

    if (total_bitmap_words > 0) {
        CUDA_CHECK(cudaMalloc(&result.bitmap_data,
                              total_bitmap_words * sizeof(uint64_t)));
    }
    if (total_array_elems > 0) {
        CUDA_CHECK(cudaMalloc(&result.array_data,
                              total_array_elems * sizeof(uint16_t)));
    }
    if (total_run_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&result.run_data,
                              total_run_pairs * 2 * sizeof(uint16_t)));
    }

    CUDA_CHECK(cudaMemcpyAsync(result.keys, h_keys.data(),
                               n * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.types, h_types.data(),
                               n * sizeof(ContainerType),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.offsets, h_offsets.data(),
                               n * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, h_cardinalities.data(),
                               n * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));

    if (total_bitmap_words > 0) {
        CUDA_CHECK(cudaMemcpyAsync(result.bitmap_data, h_bitmap_pool,
                                   total_bitmap_words * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (total_array_elems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(result.array_data, h_array_pool,
                                   total_array_elems * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (total_run_pairs > 0) {
        CUDA_CHECK(cudaMemcpyAsync(result.run_data, h_run_pool,
                                   total_run_pairs * 2 * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_bitmap_pool) cudaFreeHost(h_bitmap_pool);
    if (h_array_pool)  cudaFreeHost(h_array_pool);
    if (h_run_pool)    cudaFreeHost(h_run_pool);

    // Resolve auto threshold based on GPU L2 cache size
    uint32_t effective_threshold = bitmap_threshold;
    if (bitmap_threshold == PROMOTE_AUTO) {
        effective_threshold = resolve_auto_threshold(result.universe_size);
    }

    // Promote containers to bitmap if requested or auto-selected
    if (effective_threshold < PROMOTE_NONE &&
        (result.n_array_containers > 0 || result.n_run_containers > 0)) {
        GpuRoaring promoted = promote_to_bitmap(result, stream);
        gpu_roaring_free(result);
        return promoted;
    }

    return result;
}

void gpu_roaring_free(GpuRoaring& bitmap) {
    if (bitmap.keys)          cudaFree(bitmap.keys);
    if (bitmap.types)         cudaFree(bitmap.types);
    if (bitmap.offsets)       cudaFree(bitmap.offsets);
    if (bitmap.cardinalities) cudaFree(bitmap.cardinalities);
    if (bitmap.bitmap_data)   cudaFree(bitmap.bitmap_data);
    if (bitmap.array_data)    cudaFree(bitmap.array_data);
    if (bitmap.run_data)      cudaFree(bitmap.run_data);
    bitmap = GpuRoaring{};
}

}  // namespace cu_roaring
