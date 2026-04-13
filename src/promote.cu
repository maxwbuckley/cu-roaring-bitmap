#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/utils.cuh"

namespace cu_roaring {

// ============================================================================
// Device-side promotion kernels
// ============================================================================
//
// One block per source container. Each block reads its own metadata
// (type, offset, cardinality) directly from device memory, then either copies
// (BITMAP), scatters (ARRAY), or expands runs (RUN) into the destination
// bitmap pool at slot `cid * 1024`. This replaces the old D2H → CPU rebuild
// → H2D round-trip, which at 100M+ scale could move many megabytes of data
// across PCIe for a purely structural transform.

__global__ void promote_to_bitmap_kernel(
    const ContainerType* src_types,
    const uint32_t* src_offsets,
    const uint16_t* src_cards,
    const uint64_t* src_bitmap_pool,
    const uint16_t* src_array_pool,
    const uint16_t* src_run_pool,
    uint64_t* dst_bitmap_pool,
    ContainerType* dst_types,
    uint32_t* dst_offsets,
    uint32_t n)
{
    uint32_t cid = blockIdx.x;
    if (cid >= n) return;

    uint64_t* dst = dst_bitmap_pool + static_cast<size_t>(cid) * 1024;
    ContainerType type = src_types[cid];

    if (type == ContainerType::BITMAP) {
        // Straight copy of 1024 words.
        const uint64_t* src =
            src_bitmap_pool + (src_offsets[cid] / sizeof(uint64_t));
        for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
            dst[w] = src[w];
        }
    } else {
        // ARRAY or RUN: zero the destination first, then scatter/expand.
        for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
            dst[w] = 0ULL;
        }
        __syncthreads();

        uint16_t card = src_cards[cid];

        if (type == ContainerType::ARRAY) {
            const uint16_t* arr =
                src_array_pool + (src_offsets[cid] / sizeof(uint16_t));
            for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
                uint16_t val = arr[i];
                uint32_t word_idx = val / 64u;
                uint64_t bit_mask = 1ULL << (val % 64u);
                atomicOr(reinterpret_cast<unsigned long long*>(&dst[word_idx]),
                         static_cast<unsigned long long>(bit_mask));
            }
        } else {  // RUN
            const uint16_t* runs =
                src_run_pool + (src_offsets[cid] / sizeof(uint16_t));
            for (uint32_t r = threadIdx.x; r < card; r += blockDim.x) {
                uint16_t start = runs[r * 2];
                uint16_t length = runs[r * 2 + 1];
                uint32_t end = static_cast<uint32_t>(start) + length;
                if (end > 0xFFFFu) end = 0xFFFFu;
                for (uint32_t v = start; v <= end; ++v) {
                    uint32_t word_idx = v / 64u;
                    uint64_t bit_mask = 1ULL << (v % 64u);
                    atomicOr(
                        reinterpret_cast<unsigned long long*>(&dst[word_idx]),
                        static_cast<unsigned long long>(bit_mask));
                }
            }
        }
    }

    // Emit the output metadata for this container (one thread per block).
    if (threadIdx.x == 0) {
        dst_types[cid]   = ContainerType::BITMAP;
        dst_offsets[cid] = static_cast<uint32_t>(
            static_cast<size_t>(cid) * 1024 * sizeof(uint64_t));
    }
}

// Build key_index[k] = i where keys[i] == k; cells with no matching
// container are already set to 0xFFFF by the preceding cudaMemsetAsync(0xFF).
__global__ void build_key_index_kernel(const uint16_t* keys,
                                       uint32_t n,
                                       uint16_t* key_index)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key_index[keys[i]] = static_cast<uint16_t>(i);
}

// ============================================================================
// promote_to_bitmap — fully device-resident
// ============================================================================

GpuRoaring promote_to_bitmap(const GpuRoaring& bm, cudaStream_t stream)
{
    const uint32_t n = bm.n_containers;
    if (n == 0) return GpuRoaring{};

    GpuRoaring result{};
    result.n_containers        = n;
    result.n_bitmap_containers = n;
    result.n_array_containers  = 0;
    result.n_run_containers    = 0;
    result.universe_size       = bm.universe_size;
    result.total_cardinality   = bm.total_cardinality;
    result.negated             = bm.negated;

    const size_t bitmap_pool_bytes =
        static_cast<size_t>(n) * 1024 * sizeof(uint64_t);

    CUDA_CHECK(cudaMallocAsync(&result.keys,          n * sizeof(uint16_t),       stream));
    CUDA_CHECK(cudaMallocAsync(&result.types,         n * sizeof(ContainerType),  stream));
    CUDA_CHECK(cudaMallocAsync(&result.offsets,       n * sizeof(uint32_t),       stream));
    CUDA_CHECK(cudaMallocAsync(&result.cardinalities, n * sizeof(uint16_t),       stream));
    CUDA_CHECK(cudaMallocAsync(&result.bitmap_data,   bitmap_pool_bytes,          stream));

    // keys and cardinalities are unchanged by promotion — D2D copy.
    CUDA_CHECK(cudaMemcpyAsync(result.keys, bm.keys,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, bm.cardinalities,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));

    if (bm.n_array_containers == 0 && bm.n_run_containers == 0) {
        // Already all-bitmap: the only work is a D2D copy of the bitmap pool
        // and filling types/offsets with the trivial sequence. Use the
        // promotion kernel anyway — its BITMAP branch is a plain copy, and
        // it writes the contiguous types/offsets for us.
    }

    // Launch one block per container. The kernel self-dispatches on
    // source container type.
    if (n > 0) {
        promote_to_bitmap_kernel<<<n, 256, 0, stream>>>(
            bm.types,
            bm.offsets,
            bm.cardinalities,
            bm.bitmap_data,
            bm.array_data,
            bm.run_data,
            result.bitmap_data,
            result.types,
            result.offsets,
            n);
    }

    // Build key_index on device. The width is determined by universe_size,
    // not by reading max_key back to the host.
    if (bm.universe_size > 0) {
        // max_key for a universe of size U is (U - 1) >> 16. Clamp to uint16.
        uint32_t max_key32 =
            (bm.universe_size == 0) ? 0u : ((bm.universe_size - 1u) >> 16);
        if (max_key32 > 0xFFFFu) max_key32 = 0xFFFFu;
        result.max_key = static_cast<uint16_t>(max_key32);
        size_t idx_bytes =
            (static_cast<size_t>(result.max_key) + 1u) * sizeof(uint16_t);

        CUDA_CHECK(cudaMallocAsync(&result.key_index, idx_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(result.key_index, 0xFF, idx_bytes, stream));
        uint32_t block = 256;
        uint32_t grid  = (n + block - 1) / block;
        if (grid > 0) {
            build_key_index_kernel<<<grid, block, 0, stream>>>(
                result.keys, n, result.key_index);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

// ============================================================================
// Cache-aware automatic promotion
// ============================================================================

uint32_t resolve_auto_threshold(uint32_t universe_size, int device_id)
{
    (void)device_id;

    // Promote all containers to bitmap whenever the universe has more than
    // ~64 potential containers (universe > ~4M). At this scale the key
    // binary search (7+ steps) combined with array container binary search
    // (up to 12 steps) makes array queries 4-10x slower than bitmap queries
    // (which replace the inner search with a single word read).
    //
    // Below ~4M universe (<=64 containers, <=6 key search steps), array
    // containers are fast enough and the memory savings (2*card bytes vs
    // 8 KB per container) are worthwhile — a 1M/0.1% bitmap uses 2 KB
    // compressed vs 128 KB if promoted.
    //
    // Users who want minimum memory at any scale: pass PROMOTE_NONE.
    // Users who want maximum speed at any scale: pass PROMOTE_ALL.
    constexpr uint32_t CONTAINER_THRESHOLD = 64;
    uint32_t n_possible_containers = (universe_size + 65535) / 65536;

    if (n_possible_containers > CONTAINER_THRESHOLD) {
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
