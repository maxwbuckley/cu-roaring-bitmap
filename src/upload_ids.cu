#include "cu_roaring/detail/upload_ids.cuh"
#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cstring>
#include <vector>

namespace cu_roaring {

// Below this threshold, use CPU sort. The GPU-native pipeline wins at all
// scales above ~1K IDs on RTX 5090 (CUB launch overhead ~0.3ms vs CPU
// sort + upload). Conservative at 1024 to account for smaller GPUs.
static constexpr uint32_t GPU_SORT_THRESHOLD = 1024;

// ============================================================================
// GPU kernels for fully device-resident upload pipeline
// ============================================================================

// Extract high-16 key from each sorted ID
__global__ void extract_keys_kernel(const uint32_t* sorted_ids,
                                     uint16_t* keys_out,
                                     uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys_out[i] = static_cast<uint16_t>(sorted_ids[i] >> 16);
    }
}

// Scatter sorted IDs into bitmap containers.
// Each block handles one container. d_offsets[container] is the start index
// into sorted_ids for that container, d_counts[container] is the count.
__global__ void scatter_to_bitmaps_kernel(const uint32_t* sorted_ids,
                                           const uint32_t* d_offsets,
                                           const uint32_t* d_counts,
                                           uint64_t* bitmap_pool,
                                           uint32_t n_containers)
{
    uint32_t cid = blockIdx.x;
    if (cid >= n_containers) return;

    uint32_t start = d_offsets[cid];
    uint32_t count = d_counts[cid];
    uint64_t* dst  = bitmap_pool + static_cast<size_t>(cid) * 1024;

    // Zero the bitmap
    for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x) {
        dst[i] = 0;
    }
    __syncthreads();

    // Set bits
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        uint16_t low = static_cast<uint16_t>(sorted_ids[start + i] & 0xFFFF);
        atomicOr(reinterpret_cast<unsigned long long*>(&dst[low / 64]),
                 static_cast<unsigned long long>(1ULL << (low % 64)));
    }
}

// Build metadata arrays: keys, types (all BITMAP), offsets, cardinalities
__global__ void build_metadata_kernel(const uint16_t* unique_keys,
                                       const uint32_t* counts,
                                       uint16_t* out_keys,
                                       ContainerType* out_types,
                                       uint32_t* out_offsets,
                                       uint16_t* out_cardinalities,
                                       uint32_t n_containers)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_containers) return;

    out_keys[i]          = unique_keys[i];
    out_types[i]         = ContainerType::BITMAP;
    out_offsets[i]       = static_cast<uint32_t>(static_cast<size_t>(i) * 1024 * sizeof(uint64_t));
    out_cardinalities[i] = static_cast<uint16_t>(counts[i] > 65535 ? 0 : counts[i]);
}

// ============================================================================
// GPU-native upload: sort + unique + partition + build, all on GPU
// ============================================================================
static GpuRoaring gpu_upload(const uint32_t* host_ids,
                              uint32_t n_ids,
                              uint32_t universe_size,
                              cudaStream_t stream)
{
    GpuRoaring result{};
    result.universe_size = universe_size;

    // 1. Upload unsorted IDs to GPU
    uint32_t* d_in = nullptr;
    uint32_t* d_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n_ids * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted, n_ids * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_in, host_ids, n_ids * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    // 2. CUB RadixSort
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes,
                                   d_in, d_sorted, n_ids, 0, 32, stream);
    void* d_sort_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));
    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_temp_bytes,
                                   d_in, d_sorted, n_ids, 0, 32, stream);
    cudaFree(d_sort_temp);
    cudaFree(d_in);

    // 3. CUB Unique (deduplicate)
    uint32_t* d_unique = nullptr;
    uint32_t* d_num_unique = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique, n_ids * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_unique, sizeof(uint32_t)));

    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, unique_temp_bytes,
                              d_sorted, d_unique, d_num_unique, n_ids, stream);
    void* d_unique_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique_temp, unique_temp_bytes));
    cub::DeviceSelect::Unique(d_unique_temp, unique_temp_bytes,
                              d_sorted, d_unique, d_num_unique, n_ids, stream);
    cudaFree(d_unique_temp);
    cudaFree(d_sorted);

    // Download unique count (small sync — 4 bytes)
    uint32_t h_num_unique = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_unique, d_num_unique, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_num_unique);
    result.total_cardinality = h_num_unique;

    if (h_num_unique == 0) {
        cudaFree(d_unique);
        return result;
    }

    // 4. Extract high-16 keys from sorted unique IDs
    uint16_t* d_all_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_all_keys, h_num_unique * sizeof(uint16_t)));
    {
        uint32_t blocks = (h_num_unique + 255) / 256;
        extract_keys_kernel<<<blocks, 256, 0, stream>>>(
            d_unique, d_all_keys, h_num_unique);
    }

    // 5. CUB RunLengthEncode on keys → unique container keys + counts + n_containers
    uint16_t* d_container_keys = nullptr;
    uint32_t* d_counts = nullptr;
    uint32_t* d_n_containers = nullptr;
    CUDA_CHECK(cudaMalloc(&d_container_keys, h_num_unique * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_counts, h_num_unique * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_n_containers, sizeof(uint32_t)));

    size_t rle_temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_temp_bytes,
                                       d_all_keys, d_container_keys,
                                       d_counts, d_n_containers,
                                       h_num_unique, stream);
    void* d_rle_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rle_temp, rle_temp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_rle_temp, rle_temp_bytes,
                                       d_all_keys, d_container_keys,
                                       d_counts, d_n_containers,
                                       h_num_unique, stream);
    cudaFree(d_rle_temp);
    cudaFree(d_all_keys);

    // Download n_containers (small sync — 4 bytes)
    uint32_t h_n_containers = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_n_containers, d_n_containers, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_n_containers);

    result.n_containers        = h_n_containers;
    result.n_bitmap_containers = h_n_containers;
    result.n_array_containers  = 0;
    result.n_run_containers    = 0;

    // 6. Exclusive prefix sum on counts → container start offsets into d_unique
    uint32_t* d_offsets_into_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets_into_ids, h_n_containers * sizeof(uint32_t)));

    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                  d_counts, d_offsets_into_ids,
                                  h_n_containers, stream);
    void* d_scan_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
    cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
                                  d_counts, d_offsets_into_ids,
                                  h_n_containers, stream);
    cudaFree(d_scan_temp);

    // 7. Allocate bitmap pool and scatter IDs into containers (all on GPU)
    size_t bitmap_pool_size = static_cast<size_t>(h_n_containers) * 1024;
    CUDA_CHECK(cudaMalloc(&result.bitmap_data, bitmap_pool_size * sizeof(uint64_t)));

    scatter_to_bitmaps_kernel<<<h_n_containers, 256, 0, stream>>>(
        d_unique, d_offsets_into_ids, d_counts,
        result.bitmap_data, h_n_containers);

    cudaFree(d_unique);
    cudaFree(d_offsets_into_ids);

    // 8. Build metadata arrays on GPU
    CUDA_CHECK(cudaMalloc(&result.keys, h_n_containers * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&result.types, h_n_containers * sizeof(ContainerType)));
    CUDA_CHECK(cudaMalloc(&result.offsets, h_n_containers * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&result.cardinalities, h_n_containers * sizeof(uint16_t)));

    {
        uint32_t blocks = (h_n_containers + 255) / 256;
        build_metadata_kernel<<<blocks, 256, 0, stream>>>(
            d_container_keys, d_counts,
            result.keys, result.types, result.offsets, result.cardinalities,
            h_n_containers);
    }

    cudaFree(d_container_keys);
    cudaFree(d_counts);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}

// ============================================================================
// CPU path for small inputs
// ============================================================================
static GpuRoaring cpu_upload(const uint32_t* raw_ids,
                              uint32_t n_ids,
                              uint32_t universe_size,
                              uint32_t effective_threshold,
                              cudaStream_t stream)
{
    GpuRoaring result{};
    result.universe_size = universe_size;

    // Sort and deduplicate on CPU
    std::vector<uint32_t> ids(raw_ids, raw_ids + n_ids);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    result.total_cardinality = ids.size();

    // Partition into containers
    struct Container {
        uint16_t key;
        std::vector<uint16_t> values;
    };
    std::vector<Container> containers;

    uint16_t cur_key = static_cast<uint16_t>(ids[0] >> 16);
    containers.push_back({cur_key, {}});
    for (size_t i = 0; i < ids.size(); ++i) {
        uint16_t key = static_cast<uint16_t>(ids[i] >> 16);
        uint16_t val = static_cast<uint16_t>(ids[i] & 0xFFFF);
        if (key != cur_key) {
            cur_key = key;
            containers.push_back({key, {}});
        }
        containers.back().values.push_back(val);
    }

    uint32_t n = static_cast<uint32_t>(containers.size());
    result.n_containers = n;

    std::vector<uint16_t> h_keys(n);
    std::vector<ContainerType> h_types(n);
    std::vector<uint32_t> h_offsets(n);
    std::vector<uint16_t> h_cards(n);
    std::vector<uint64_t> h_bitmap_pool;
    std::vector<uint16_t> h_array_pool;
    uint32_t n_bitmap = 0, n_array = 0;

    for (uint32_t i = 0; i < n; ++i) {
        h_keys[i] = containers[i].key;
        uint32_t card = static_cast<uint32_t>(containers[i].values.size());
        h_cards[i] = static_cast<uint16_t>(card > 65535 ? 0 : card);

        if (card > effective_threshold) {
            h_types[i]  = ContainerType::BITMAP;
            h_offsets[i] = static_cast<uint32_t>(h_bitmap_pool.size() * sizeof(uint64_t));
            size_t base  = h_bitmap_pool.size();
            h_bitmap_pool.resize(base + 1024, 0);
            for (uint16_t v : containers[i].values)
                h_bitmap_pool[base + v / 64] |= 1ULL << (v % 64);
            ++n_bitmap;
        } else {
            h_types[i]  = ContainerType::ARRAY;
            h_offsets[i] = static_cast<uint32_t>(h_array_pool.size() * sizeof(uint16_t));
            h_array_pool.insert(h_array_pool.end(),
                                containers[i].values.begin(),
                                containers[i].values.end());
            ++n_array;
        }
    }

    result.n_bitmap_containers = n_bitmap;
    result.n_array_containers  = n_array;
    result.n_run_containers    = 0;

    CUDA_CHECK(cudaMalloc(&result.keys, n * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&result.types, n * sizeof(ContainerType)));
    CUDA_CHECK(cudaMalloc(&result.offsets, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&result.cardinalities, n * sizeof(uint16_t)));

    CUDA_CHECK(cudaMemcpyAsync(result.keys, h_keys.data(),
                               n * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.types, h_types.data(),
                               n * sizeof(ContainerType), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.offsets, h_offsets.data(),
                               n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, h_cards.data(),
                               n * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

    if (!h_bitmap_pool.empty()) {
        CUDA_CHECK(cudaMalloc(&result.bitmap_data,
                              h_bitmap_pool.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(result.bitmap_data, h_bitmap_pool.data(),
                                   h_bitmap_pool.size() * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (!h_array_pool.empty()) {
        CUDA_CHECK(cudaMalloc(&result.array_data,
                              h_array_pool.size() * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpyAsync(result.array_data, h_array_pool.data(),
                                   h_array_pool.size() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

// ============================================================================
// Public API
// ============================================================================

GpuRoaring upload_from_ids(const uint32_t* raw_ids,
                           uint32_t n_ids,
                           uint32_t universe_size,
                           cudaStream_t stream,
                           uint32_t bitmap_threshold)
{
    uint32_t effective_threshold = bitmap_threshold;
    if (bitmap_threshold == PROMOTE_AUTO) {
        effective_threshold = resolve_auto_threshold(universe_size);
    }

    if (n_ids == 0) {
        GpuRoaring result{};
        result.universe_size = universe_size;
        return result;
    }

    // Large inputs: fully GPU-native pipeline (always produces all-bitmap)
    if (n_ids > GPU_SORT_THRESHOLD) {
        return gpu_upload(raw_ids, n_ids, universe_size, stream);
    }

    // Small inputs: CPU sort + flexible container types
    return cpu_upload(raw_ids, n_ids, universe_size, effective_threshold, stream);
}

}  // namespace cu_roaring
