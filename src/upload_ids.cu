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

// Build direct-map key index: key_index[key] = container index, 0xFFFF = absent
__global__ void build_key_index_kernel(const uint16_t* unique_keys,
                                        uint16_t* key_index,
                                        uint32_t n_containers,
                                        uint32_t max_key_plus_one)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // First: zero-fill with 0xFFFF (absent sentinel)
    if (i < max_key_plus_one) {
        key_index[i] = 0xFFFF;
    }
    // Barrier not needed — separate kernel launch for the write pass
}

__global__ void populate_key_index_kernel(const uint16_t* unique_keys,
                                           uint16_t* key_index,
                                           uint32_t n_containers)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_containers) {
        key_index[unique_keys[i]] = static_cast<uint16_t>(i);
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
// GPU-native complement kernels
//
// When density > 50%, we compute the complement set (the "gaps" between
// existing IDs) on-GPU and build the Roaring bitmap from that instead.
// This guarantees the stored bitmap always has density <= 50%.
// ============================================================================

// Compute the size of each gap between consecutive sorted unique IDs.
// gap[0]     = d_unique[0]                        (IDs before first element)
// gap[i]     = d_unique[i] - d_unique[i-1] - 1    (interior gaps)
// gap[n]     = universe_size - d_unique[n-1] - 1   (IDs after last element)
__global__ void compute_gap_sizes_kernel(const uint32_t* d_unique,
                                          uint32_t* gap_sizes,
                                          uint32_t n_unique,
                                          uint32_t universe_size)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // n_unique + 1 gaps total
    if (i > n_unique) return;

    if (i == 0) {
        gap_sizes[0] = d_unique[0];
    } else if (i == n_unique) {
        gap_sizes[n_unique] = universe_size - d_unique[n_unique - 1] - 1;
    } else {
        gap_sizes[i] = d_unique[i] - d_unique[i - 1] - 1;
    }
}

// Expand gaps into complement IDs using precomputed offsets.
// Each thread handles one gap and writes sequential IDs.
__global__ void expand_gaps_kernel(const uint32_t* d_unique,
                                    const uint32_t* gap_offsets,
                                    const uint32_t* gap_sizes,
                                    uint32_t* complement_ids,
                                    uint32_t n_unique,
                                    uint32_t universe_size)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_unique) return;

    uint32_t size = gap_sizes[i];
    if (size == 0) return;

    uint32_t out_offset = gap_offsets[i];
    uint32_t start_id;

    if (i == 0) {
        start_id = 0;
    } else {
        start_id = d_unique[i - 1] + 1;
    }

    for (uint32_t j = 0; j < size; ++j) {
        complement_ids[out_offset + j] = start_id + j;
    }
}

// ============================================================================
// Helper: build a GpuRoaring from sorted unique IDs already on GPU.
// Shared by both the direct and complement paths.
// ============================================================================
static GpuRoaring build_from_device_ids(uint32_t* d_unique,
                                         uint32_t h_num_unique,
                                         uint32_t universe_size,
                                         cudaStream_t stream)
{
    GpuRoaring result{};
    result.universe_size = universe_size;
    result.total_cardinality = h_num_unique;

    if (h_num_unique == 0) {
        cudaFree(d_unique);
        return result;
    }

    // 1. Extract high-16 keys from sorted unique IDs
    uint16_t* d_all_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_all_keys, h_num_unique * sizeof(uint16_t)));
    {
        uint32_t blocks = (h_num_unique + 255) / 256;
        extract_keys_kernel<<<blocks, 256, 0, stream>>>(
            d_unique, d_all_keys, h_num_unique);
    }

    // 2. CUB RunLengthEncode on keys → unique container keys + counts + n_containers
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

    // 3. Exclusive prefix sum on counts → container start offsets into d_unique
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

    // 4. Allocate bitmap pool and scatter IDs into containers (all on GPU)
    size_t bitmap_pool_size = static_cast<size_t>(h_n_containers) * 1024;
    CUDA_CHECK(cudaMalloc(&result.bitmap_data, bitmap_pool_size * sizeof(uint64_t)));

    scatter_to_bitmaps_kernel<<<h_n_containers, 256, 0, stream>>>(
        d_unique, d_offsets_into_ids, d_counts,
        result.bitmap_data, h_n_containers);

    cudaFree(d_unique);
    cudaFree(d_offsets_into_ids);

    // 5. Build metadata arrays on GPU
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

    // 6. Build direct-map key index on GPU
    // Get max_key (last element of d_container_keys)
    uint16_t h_max_key = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_max_key,
                               d_container_keys + h_n_containers - 1,
                               sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    result.max_key = h_max_key;

    uint32_t index_size = static_cast<uint32_t>(h_max_key) + 1;
    CUDA_CHECK(cudaMalloc(&result.key_index, index_size * sizeof(uint16_t)));
    // Fill with 0xFFFF
    CUDA_CHECK(cudaMemsetAsync(result.key_index, 0xFF,
                               index_size * sizeof(uint16_t), stream));
    // Write container indices
    {
        uint32_t blocks = (h_n_containers + 255) / 256;
        populate_key_index_kernel<<<blocks, 256, 0, stream>>>(
            d_container_keys, result.key_index, h_n_containers);
    }

    cudaFree(d_container_keys);
    cudaFree(d_counts);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}

// ============================================================================
// GPU-native upload: sort + unique + partition + build, all on GPU.
// When density > 50%, automatically computes the complement on-GPU
// (gap-expand pipeline) so the stored bitmap has density <= 50%.
// ============================================================================
static GpuRoaring gpu_upload(const uint32_t* host_ids,
                              uint32_t n_ids,
                              uint32_t universe_size,
                              cudaStream_t stream)
{
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

    if (h_num_unique == 0) {
        cudaFree(d_unique);
        GpuRoaring result{};
        result.universe_size = universe_size;
        return result;
    }

    // 4. Complement decision: if density > 50%, build from complement IDs.
    //    The complement is computed entirely on-GPU via gap expansion —
    //    no additional host↔device transfers.
    bool should_negate = (static_cast<uint64_t>(h_num_unique) > static_cast<uint64_t>(universe_size) / 2);

    if (should_negate) {
        uint32_t n_complement = universe_size - h_num_unique;

        // 4a. Compute gap sizes: n_unique + 1 gaps
        uint32_t n_gaps = h_num_unique + 1;
        uint32_t* d_gap_sizes = nullptr;
        CUDA_CHECK(cudaMalloc(&d_gap_sizes, n_gaps * sizeof(uint32_t)));
        {
            uint32_t blocks = (n_gaps + 255) / 256;
            compute_gap_sizes_kernel<<<blocks, 256, 0, stream>>>(
                d_unique, d_gap_sizes, h_num_unique, universe_size);
        }

        // 4b. Prefix sum on gap sizes → output offsets
        uint32_t* d_gap_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_gap_offsets, n_gaps * sizeof(uint32_t)));

        size_t scan_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                      d_gap_sizes, d_gap_offsets,
                                      n_gaps, stream);
        void* d_scan_temp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
                                      d_gap_sizes, d_gap_offsets,
                                      n_gaps, stream);
        cudaFree(d_scan_temp);

        // 4c. Expand gaps into complement IDs
        uint32_t* d_complement = nullptr;
        CUDA_CHECK(cudaMalloc(&d_complement, n_complement * sizeof(uint32_t)));
        {
            uint32_t blocks = (n_gaps + 255) / 256;
            expand_gaps_kernel<<<blocks, 256, 0, stream>>>(
                d_unique, d_gap_offsets, d_gap_sizes,
                d_complement, h_num_unique, universe_size);
        }

        cudaFree(d_unique);
        cudaFree(d_gap_sizes);
        cudaFree(d_gap_offsets);

        // 4d. Build Roaring from complement IDs (reuses shared helper)
        GpuRoaring result = build_from_device_ids(
            d_complement, n_complement, universe_size, stream);
        result.negated = true;
        // total_cardinality is the logical cardinality (the original set)
        result.total_cardinality = h_num_unique;
        return result;
    }

    // 5. Normal path: build directly from unique IDs
    GpuRoaring result = build_from_device_ids(
        d_unique, h_num_unique, universe_size, stream);
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

    // Complement decision: if density > 50%, store the complement instead
    bool should_negate = (ids.size() > static_cast<size_t>(universe_size) / 2);
    if (should_negate) {
        // Sorted set-difference: {0..universe_size-1} \ ids
        std::vector<uint32_t> complement;
        complement.reserve(universe_size - ids.size());
        size_t j = 0;
        for (uint32_t v = 0; v < universe_size; ++v) {
            if (j < ids.size() && ids[j] == v) {
                ++j;
            } else {
                complement.push_back(v);
            }
        }
        ids = std::move(complement);
        result.negated = true;
    }

    if (ids.empty()) {
        return result;
    }

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

    // Build direct-map key index
    if (n > 0) {
        result.max_key = h_keys[n - 1];
        std::vector<uint16_t> h_key_index(result.max_key + 1, 0xFFFF);
        for (uint32_t i = 0; i < n; ++i) {
            h_key_index[h_keys[i]] = static_cast<uint16_t>(i);
        }
        CUDA_CHECK(cudaMalloc(&result.key_index,
                              (result.max_key + 1) * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpyAsync(result.key_index, h_key_index.data(),
                                   (result.max_key + 1) * sizeof(uint16_t),
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

// ============================================================================
// Upload from flat bitset — direct container extraction, no sort/dedupe.
//
// A flat bitset of N bits maps directly to ceil(N/65536) potential Roaring
// containers, each 2048 uint32_t words (= 1024 uint64_t = 8 KB). We just
// need to identify which chunks are non-empty (popcount > 0), compact them
// into the bitmap pool, and build metadata.
//
// GPU kernel: one block per chunk, popcounts + compaction.
// ============================================================================

// Per-chunk popcount: each block handles one 65536-bit chunk (2048 uint32_t words).
// Writes the chunk's popcount to d_chunk_popcounts[chunk_idx].
__global__ void chunk_popcount_kernel(const uint32_t* bitset,
                                       uint32_t* chunk_popcounts,
                                       uint32_t n_chunks,
                                       uint32_t n_words_total)
{
    uint32_t chunk = blockIdx.x;
    if (chunk >= n_chunks) return;

    uint32_t base = chunk * 2048u;
    uint32_t local_count = 0;

    for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
        uint32_t idx = base + i;
        if (idx < n_words_total) {
            local_count += __popc(bitset[idx]);
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
    }

    __shared__ uint32_t warp_counts[8];
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_counts[warp_id] = local_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t total = 0;
        uint32_t n_warps = (blockDim.x + 31) / 32;
        for (uint32_t w = 0; w < n_warps; ++w) total += warp_counts[w];
        chunk_popcounts[chunk] = total;
    }
}

// Copy non-empty chunks into the compacted bitmap pool and build metadata.
// d_chunk_map[i] = output container index for chunk i, or 0xFFFFFFFF if empty.
__global__ void compact_chunks_kernel(const uint32_t* bitset,
                                       const uint32_t* chunk_map,
                                       const uint32_t* chunk_popcounts,
                                       uint64_t* bitmap_pool,
                                       uint16_t* out_keys,
                                       ContainerType* out_types,
                                       uint32_t* out_offsets,
                                       uint16_t* out_cardinalities,
                                       uint32_t n_chunks,
                                       uint32_t n_words_total)
{
    uint32_t chunk = blockIdx.x;
    if (chunk >= n_chunks) return;

    uint32_t out_idx = chunk_map[chunk];
    if (out_idx == 0xFFFFFFFF) return;  // empty chunk

    uint32_t base_word = chunk * 2048u;

    // Copy bitset words into bitmap pool as uint64_t.
    // Read individual uint32_t words with bounds checking to handle
    // partial chunks where n_words_total is odd.
    uint64_t* dst = bitmap_pool + static_cast<size_t>(out_idx) * 1024;

    for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x) {
        uint32_t w0_idx = base_word + i * 2;
        uint32_t w1_idx = base_word + i * 2 + 1;
        uint32_t w0 = (w0_idx < n_words_total) ? bitset[w0_idx] : 0u;
        uint32_t w1 = (w1_idx < n_words_total) ? bitset[w1_idx] : 0u;
        dst[i] = static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32);
    }

    // Build metadata (one thread per block)
    if (threadIdx.x == 0) {
        out_keys[out_idx] = static_cast<uint16_t>(chunk);
        out_types[out_idx] = ContainerType::BITMAP;
        out_offsets[out_idx] = static_cast<uint32_t>(
            static_cast<size_t>(out_idx) * 1024 * sizeof(uint64_t));
        uint32_t pc = chunk_popcounts[chunk];
        out_cardinalities[out_idx] = static_cast<uint16_t>(pc > 65535 ? 0 : pc);
    }
}

// Invert a bitset in-place for complement (uint32_t words)
__global__ void invert_bitset_u32_kernel(uint32_t* data, uint32_t n_words)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) {
        data[i] = ~data[i];
    }
}

// Mask the tail word to universe_size
__global__ void mask_tail_kernel(uint32_t* data, uint32_t word_idx, uint32_t mask)
{
    data[word_idx] &= mask;
}

// Core implementation: build GpuRoaring from a device-resident bitset.
static GpuRoaring build_from_device_bitset(const uint32_t* d_bitset,
                                            uint32_t n_words,
                                            uint32_t universe_size,
                                            bool should_negate,
                                            uint64_t original_cardinality,
                                            cudaStream_t stream)
{
    uint32_t n_chunks = (n_words + 2047) / 2048;

    // 1. Popcount each chunk
    uint32_t* d_popcounts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_popcounts, n_chunks * sizeof(uint32_t)));
    chunk_popcount_kernel<<<n_chunks, 256, 0, stream>>>(
        d_bitset, d_popcounts, n_chunks, n_words);
    CUDA_CHECK(cudaGetLastError());

    // Download popcounts (small: n_chunks * 4 bytes, typically < 1 KB)
    std::vector<uint32_t> h_popcounts(n_chunks);
    CUDA_CHECK(cudaMemcpyAsync(h_popcounts.data(), d_popcounts,
                               n_chunks * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Identify non-empty chunks and build compaction map
    std::vector<uint32_t> h_chunk_map(n_chunks, 0xFFFFFFFF);
    uint32_t n_containers = 0;
    uint64_t total_card = 0;
    for (uint32_t i = 0; i < n_chunks; ++i) {
        total_card += h_popcounts[i];
        if (h_popcounts[i] > 0) {
            h_chunk_map[i] = n_containers++;
        }
    }

    GpuRoaring result{};
    result.universe_size = universe_size;
    result.total_cardinality = should_negate ? original_cardinality : total_card;
    result.negated = should_negate;
    result.n_containers = n_containers;
    result.n_bitmap_containers = n_containers;
    result.n_array_containers = 0;
    result.n_run_containers = 0;

    if (n_containers == 0) {
        cudaFree(d_popcounts);
        return result;
    }

    // 3. Upload chunk map and run compaction kernel
    uint32_t* d_chunk_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chunk_map, n_chunks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_chunk_map, h_chunk_map.data(),
                               n_chunks * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMalloc(&result.bitmap_data,
                          static_cast<size_t>(n_containers) * 1024 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&result.keys, n_containers * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&result.types, n_containers * sizeof(ContainerType)));
    CUDA_CHECK(cudaMalloc(&result.offsets, n_containers * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&result.cardinalities, n_containers * sizeof(uint16_t)));

    compact_chunks_kernel<<<n_chunks, 256, 0, stream>>>(
        d_bitset, d_chunk_map, d_popcounts,
        result.bitmap_data, result.keys, result.types,
        result.offsets, result.cardinalities,
        n_chunks, n_words);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_chunk_map);
    cudaFree(d_popcounts);

    // 4. Build key_index
    // Find max_key from the non-empty chunks
    uint16_t max_key = 0;
    for (uint32_t i = n_chunks; i > 0; --i) {
        if (h_chunk_map[i - 1] != 0xFFFFFFFF) {
            max_key = static_cast<uint16_t>(i - 1);
            break;
        }
    }
    result.max_key = max_key;

    uint32_t index_size = static_cast<uint32_t>(max_key) + 1;
    std::vector<uint16_t> h_key_index(index_size, 0xFFFF);
    for (uint32_t i = 0; i < n_chunks; ++i) {
        if (h_chunk_map[i] != 0xFFFFFFFF) {
            h_key_index[i] = static_cast<uint16_t>(h_chunk_map[i]);
        }
    }
    CUDA_CHECK(cudaMalloc(&result.key_index, index_size * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyAsync(result.key_index, h_key_index.data(),
                               index_size * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

// ============================================================================
// Public API: upload from device bitset
// ============================================================================
GpuRoaring upload_from_device_bitset(const uint32_t* d_bitset,
                                      uint32_t n_words,
                                      uint32_t universe_size,
                                      cudaStream_t stream)
{
    if (n_words == 0) {
        GpuRoaring result{};
        result.universe_size = universe_size;
        return result;
    }

    // Popcount the full bitset to decide on complement
    // We'll compute per-chunk popcounts inside build_from_device_bitset anyway,
    // but we need the total now for the complement decision.
    // Quick total via chunk_popcount_kernel + host sum:
    uint32_t n_chunks = (n_words + 2047) / 2048;
    uint32_t* d_popcounts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_popcounts, n_chunks * sizeof(uint32_t)));
    chunk_popcount_kernel<<<n_chunks, 256, 0, stream>>>(
        d_bitset, d_popcounts, n_chunks, n_words);

    std::vector<uint32_t> h_popcounts(n_chunks);
    CUDA_CHECK(cudaMemcpyAsync(h_popcounts.data(), d_popcounts,
                               n_chunks * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_popcounts);

    uint64_t total_card = 0;
    for (uint32_t i = 0; i < n_chunks; ++i) total_card += h_popcounts[i];

    bool should_negate = (total_card > static_cast<uint64_t>(universe_size) / 2);

    if (should_negate) {
        // Invert the bitset on GPU, then build from the inverted copy
        uint32_t* d_inverted = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inverted, n_words * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_inverted, d_bitset,
                                   n_words * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, stream));

        uint32_t inv_blocks = (n_words + 255) / 256;
        invert_bitset_u32_kernel<<<inv_blocks, 256, 0, stream>>>(d_inverted, n_words);

        // Mask tail bits beyond universe_size
        uint32_t tail_bits = universe_size % 32;
        if (tail_bits > 0) {
            uint32_t tail_word_idx = universe_size / 32;
            if (tail_word_idx < n_words) {
                uint32_t tail_mask = (1u << tail_bits) - 1u;
                mask_tail_kernel<<<1, 1, 0, stream>>>(d_inverted, tail_word_idx, tail_mask);
            }
        }

        GpuRoaring result = build_from_device_bitset(
            d_inverted, n_words, universe_size, true, total_card, stream);
        cudaFree(d_inverted);
        return result;
    }

    return build_from_device_bitset(d_bitset, n_words, universe_size, false, 0, stream);
}

// ============================================================================
// Public API: upload from host bitset
// ============================================================================
GpuRoaring upload_from_bitset(const uint32_t* host_bitset,
                               uint32_t n_words,
                               uint32_t universe_size,
                               cudaStream_t stream)
{
    if (n_words == 0) {
        GpuRoaring result{};
        result.universe_size = universe_size;
        return result;
    }

    // Upload to GPU, then use the device path
    uint32_t* d_bitset = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bitset, n_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_bitset, host_bitset,
                               n_words * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    GpuRoaring result = upload_from_device_bitset(d_bitset, n_words, universe_size, stream);
    cudaFree(d_bitset);
    return result;
}

}  // namespace cu_roaring
