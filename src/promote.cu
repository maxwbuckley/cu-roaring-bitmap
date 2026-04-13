#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/utils.cuh"

namespace cu_roaring {

// ============================================================================
// Device-side promotion kernel
// ============================================================================
//
// One block per source container. Each block reads its own metadata
// (type, offset, cardinality) from device memory, self-dispatches on the
// container type, and writes the 1024-word output bitmap:
//
//   BITMAP  — plain global->global copy, no shared memory.
//   ARRAY   — staged in 8 KB of shared memory via atomicOr (smem atomics
//             cost a handful of cycles vs hundreds for global atomics),
//             then copied to global.
//   RUN     — word-level expansion in shared memory. For each run, the
//             owning thread emits at most (tail_word - head_word + 1)
//             64-bit writes. A run of length L becomes ceil((L+1)/64)
//             atomic word-writes instead of (L+1) bit-writes: for typical
//             runs that is 64x fewer atomics.
//
// The kernel also writes this container's types[] and offsets[] entries
// inline from thread 0 — the output metadata is uniform (every entry is
// BITMAP at offset cid * 8192), so there is no need for a separate pass.

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
    __shared__ uint64_t smem[1024];

    const uint32_t cid = blockIdx.x;
    if (cid >= n) return;

    uint64_t* dst = dst_bitmap_pool + static_cast<size_t>(cid) * 1024;
    const ContainerType type = src_types[cid];

    if (type == ContainerType::BITMAP) {
        // Plain copy: the only work is moving 8 KB from one global pool to
        // another. Skip shared memory entirely.
        const uint64_t* src =
            src_bitmap_pool + (src_offsets[cid] / sizeof(uint64_t));
        for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
            dst[w] = src[w];
        }
    } else {
        // ARRAY or RUN: zero the shared buffer, scatter/expand into it,
        // then write the whole buffer out to global in one pass.
        for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
            smem[w] = 0ULL;
        }
        __syncthreads();

        const uint16_t card = src_cards[cid];

        if (type == ContainerType::ARRAY) {
            const uint16_t* arr =
                src_array_pool + (src_offsets[cid] / sizeof(uint16_t));
            for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
                const uint16_t val = arr[i];
                const uint32_t word_idx = val / 64u;
                const uint64_t bit_mask = 1ULL << (val % 64u);
                atomicOr(
                    reinterpret_cast<unsigned long long*>(&smem[word_idx]),
                    static_cast<unsigned long long>(bit_mask));
            }
        } else {  // RUN
            const uint16_t* runs =
                src_run_pool + (src_offsets[cid] / sizeof(uint16_t));
            // One thread per run; each thread emits whole 64-bit words
            // rather than iterating value-by-value.
            for (uint32_t r = threadIdx.x; r < card; r += blockDim.x) {
                const uint32_t start  = runs[r * 2];
                const uint32_t length = runs[r * 2 + 1];
                uint32_t end = start + length;
                if (end > 0xFFFFu) end = 0xFFFFu;

                const uint32_t head_word = start / 64u;
                const uint32_t tail_word = end   / 64u;
                const uint32_t head_bit  = start % 64u;
                const uint32_t tail_bit  = end   % 64u;

                if (head_word == tail_word) {
                    // Whole run fits inside one 64-bit word.
                    const uint32_t nbits = tail_bit - head_bit + 1u;
                    const uint64_t mask  = (nbits == 64u)
                        ? ~0ULL
                        : (((1ULL << nbits) - 1ULL) << head_bit);
                    atomicOr(
                        reinterpret_cast<unsigned long long*>(&smem[head_word]),
                        static_cast<unsigned long long>(mask));
                } else {
                    // Head partial word: bits [head_bit .. 63].
                    const uint64_t head_mask = ~0ULL << head_bit;
                    atomicOr(
                        reinterpret_cast<unsigned long long*>(&smem[head_word]),
                        static_cast<unsigned long long>(head_mask));
                    // Middle full words.
                    for (uint32_t w = head_word + 1u; w < tail_word; ++w) {
                        atomicOr(
                            reinterpret_cast<unsigned long long*>(&smem[w]),
                            static_cast<unsigned long long>(~0ULL));
                    }
                    // Tail partial word: bits [0 .. tail_bit].
                    const uint64_t tail_mask = (tail_bit == 63u)
                        ? ~0ULL
                        : ((1ULL << (tail_bit + 1u)) - 1ULL);
                    atomicOr(
                        reinterpret_cast<unsigned long long*>(&smem[tail_word]),
                        static_cast<unsigned long long>(tail_mask));
                }
            }
        }

        __syncthreads();
        // Flush shared buffer to the destination bitmap pool.
        for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
            dst[w] = smem[w];
        }
    }

    // Emit this container's output metadata (one thread per block).
    if (threadIdx.x == 0) {
        dst_types[cid]   = ContainerType::BITMAP;
        dst_offsets[cid] = static_cast<uint32_t>(
            static_cast<size_t>(cid) * 1024u * sizeof(uint64_t));
    }
}

// Write offsets[i] = i * stride. Used by the already-all-bitmap fast path
// to fill the output offsets array without launching the promotion kernel.
__global__ void fill_sequential_offsets_kernel(uint32_t* out,
                                               uint32_t n,
                                               uint32_t stride)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i * stride;
}

// Scatter keys[i] -> key_index[keys[i]] = i. Cells with no matching
// container remain at the 0xFFFF sentinel (set by the caller's memset).
//
// Defensive: the kernel refuses to write when keys[i] > max_key. This
// catches miscalibrated `universe_size` before it corrupts the allocation
// that follows key_index in the memory pool.
__global__ void build_key_index_kernel(const uint16_t* keys,
                                       uint32_t n,
                                       uint16_t* key_index,
                                       uint32_t max_key)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t k = keys[i];
    if (static_cast<uint32_t>(k) > max_key) return;
    key_index[k] = static_cast<uint16_t>(i);
}

// ============================================================================
// Deep copy helper (stream-ordered, no host sync)
// ============================================================================

GpuRoaring gpu_roaring_deep_copy(const GpuRoaring& bm, cudaStream_t stream)
{
    GpuRoaring r{};
    r.n_containers         = bm.n_containers;
    r.n_bitmap_containers  = bm.n_bitmap_containers;
    r.n_array_containers   = bm.n_array_containers;
    r.n_run_containers     = bm.n_run_containers;
    r.universe_size        = bm.universe_size;
    r.total_cardinality    = bm.total_cardinality;
    r.negated              = bm.negated;
    r.max_key              = bm.max_key;
    r.array_pool_bytes     = bm.array_pool_bytes;
    r.run_pool_bytes       = bm.run_pool_bytes;

    const uint32_t n = bm.n_containers;
    if (n == 0) return r;

    CUDA_CHECK(cudaMallocAsync(&r.keys,          n * sizeof(uint16_t),      stream));
    CUDA_CHECK(cudaMallocAsync(&r.types,         n * sizeof(ContainerType), stream));
    CUDA_CHECK(cudaMallocAsync(&r.offsets,       n * sizeof(uint32_t),      stream));
    CUDA_CHECK(cudaMallocAsync(&r.cardinalities, n * sizeof(uint16_t),      stream));

    CUDA_CHECK(cudaMemcpyAsync(r.keys,          bm.keys,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(r.types,         bm.types,
                               n * sizeof(ContainerType),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(r.offsets,       bm.offsets,
                               n * sizeof(uint32_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(r.cardinalities, bm.cardinalities,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));

    if (bm.bitmap_data != nullptr && bm.n_bitmap_containers > 0) {
        const size_t bytes =
            static_cast<size_t>(bm.n_bitmap_containers) * 1024 * sizeof(uint64_t);
        CUDA_CHECK(cudaMallocAsync(&r.bitmap_data, bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(r.bitmap_data, bm.bitmap_data, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (bm.array_data != nullptr && bm.array_pool_bytes > 0) {
        CUDA_CHECK(cudaMallocAsync(&r.array_data, bm.array_pool_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(r.array_data, bm.array_data,
                                   bm.array_pool_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (bm.run_data != nullptr && bm.run_pool_bytes > 0) {
        CUDA_CHECK(cudaMallocAsync(&r.run_data, bm.run_pool_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(r.run_data, bm.run_data,
                                   bm.run_pool_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    if (bm.key_index != nullptr && bm.universe_size > 0) {
        const size_t idx_bytes =
            (static_cast<size_t>(bm.max_key) + 1u) * sizeof(uint16_t);
        CUDA_CHECK(cudaMallocAsync(&r.key_index, idx_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(r.key_index, bm.key_index, idx_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    return r;
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
    result.array_pool_bytes    = 0;
    result.run_pool_bytes      = 0;
    result.universe_size       = bm.universe_size;
    result.total_cardinality   = bm.total_cardinality;
    result.negated             = bm.negated;

    const size_t bitmap_pool_bytes =
        static_cast<size_t>(n) * 1024 * sizeof(uint64_t);

    CUDA_CHECK(cudaMallocAsync(&result.keys,          n * sizeof(uint16_t),      stream));
    CUDA_CHECK(cudaMallocAsync(&result.types,         n * sizeof(ContainerType), stream));
    CUDA_CHECK(cudaMallocAsync(&result.offsets,       n * sizeof(uint32_t),      stream));
    CUDA_CHECK(cudaMallocAsync(&result.cardinalities, n * sizeof(uint16_t),      stream));
    CUDA_CHECK(cudaMallocAsync(&result.bitmap_data,   bitmap_pool_bytes,         stream));

    // keys and cardinalities are unchanged by promotion — D2D copy.
    CUDA_CHECK(cudaMemcpyAsync(result.keys, bm.keys,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, bm.cardinalities,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));

    if (bm.n_array_containers == 0 && bm.n_run_containers == 0) {
        // Fast path: input is already all-bitmap. A single D2D copy of
        // the whole pool replaces n block launches that would each run
        // the promotion kernel's BITMAP branch (plain copy). Types and
        // offsets are uniform sequences, so memset + a tiny kernel is
        // enough to populate them.
        CUDA_CHECK(cudaMemcpyAsync(result.bitmap_data, bm.bitmap_data,
                                   bitmap_pool_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        // ContainerType is a 1-byte enum with BITMAP == 1; memset to
        // 0x01 yields an array of BITMAP tags.
        CUDA_CHECK(cudaMemsetAsync(result.types, 0x01,
                                   n * sizeof(ContainerType), stream));
        const uint32_t block = 256;
        const uint32_t grid  = (n + block - 1) / block;
        fill_sequential_offsets_kernel<<<grid, block, 0, stream>>>(
            result.offsets, n,
            static_cast<uint32_t>(1024u * sizeof(uint64_t)));
    } else {
        // Mixed input: one block per container, kernel self-dispatches.
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

    // Build key_index on device. The table width is determined by
    // universe_size, not by reading max_key back to the host.
    if (bm.universe_size > 0) {
        uint32_t max_key32 = (bm.universe_size - 1u) >> 16;
        if (max_key32 > 0xFFFFu) max_key32 = 0xFFFFu;
        result.max_key = static_cast<uint16_t>(max_key32);
        const size_t idx_bytes =
            (static_cast<size_t>(result.max_key) + 1u) * sizeof(uint16_t);

        CUDA_CHECK(cudaMallocAsync(&result.key_index, idx_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(result.key_index, 0xFF, idx_bytes, stream));
        const uint32_t block = 256;
        const uint32_t grid  = (n + block - 1) / block;
        build_key_index_kernel<<<grid, block, 0, stream>>>(
            result.keys, n, result.key_index, max_key32);
    }

    // No host sync — caller syncs `stream` when it actually needs host
    // visibility. Everything above is stream-ordered.
    return result;
}

// ============================================================================
// Cache-aware automatic promotion
// ============================================================================

uint32_t resolve_auto_threshold(uint32_t universe_size, int device_id)
{
    if (universe_size == 0) return PROMOTE_KEEP_DEFAULT;

    // Query the real L2 cache size for the target device. Fall back to a
    // conservative 4 MB estimate if the query fails (e.g. pre-CUDA 11 or
    // an unusual device). 4 MB is smaller than any modern data-center or
    // consumer GPU's L2, so the fallback errs on the side of "promote
    // sooner", which matches the previous hardcoded constant-threshold
    // behaviour.
    int dev = device_id;
    if (dev < 0) {
        if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    }
    int l2_bytes_i = 0;
    if (cudaDeviceGetAttribute(&l2_bytes_i, cudaDevAttrL2CacheSize, dev)
            != cudaSuccess) {
        l2_bytes_i = 4 * 1024 * 1024;
    }
    const size_t l2_bytes = static_cast<size_t>(l2_bytes_i);

    // Flat bitset size for this universe. Using `(U + 7) / 8` matches the
    // layout produced by decompress_to_bitset().
    const size_t flat_bytes = (static_cast<size_t>(universe_size) + 7u) / 8u;

    // Promote when the flat bitset would consume more than half of L2.
    // Keeping at least half the cache for query working set (candidate IDs
    // being tested) avoids evicting the bitset on every query while still
    // honouring the "does it fit?" intent.
    if (flat_bytes * 2u > l2_bytes) {
        return PROMOTE_ALL;
    }
    return PROMOTE_KEEP_DEFAULT;
}

GpuRoaring promote_auto(const GpuRoaring& bm, cudaStream_t stream, int device_id)
{
    const uint32_t threshold =
        resolve_auto_threshold(bm.universe_size, device_id);

    const bool has_non_bitmap =
        (bm.n_array_containers > 0) || (bm.n_run_containers > 0);

    if (threshold == PROMOTE_ALL && has_non_bitmap) {
        return promote_to_bitmap(bm, stream);
    }

    // No promotion needed — either the flat bitset still fits in L2, or
    // the input is already all-bitmap. Return a deep copy so the caller's
    // ownership model is consistent (always gets a new owned GpuRoaring).
    return gpu_roaring_deep_copy(bm, stream);
}

}  // namespace cu_roaring
