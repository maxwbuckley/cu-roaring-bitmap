#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <cstring>

namespace cu_roaring::v2 {

namespace {

constexpr size_t ALIGN = 8;

// Output device layout for a fully-promoted batch. n_bitmap_containers_total
// equals total_containers, and there are no ARRAY/RUN pools. The CSR start
// arrays and key_indices keep the same shape as the input batch — promotion
// preserves keys and per-bitmap container ordering.
struct DeviceLayout {
    size_t cstart_off, kstart_off;
    size_t keys_off, types_off, offsets_off, cards_off;
    size_t kidx_off;
    size_t bmp_off;
    size_t total_bytes;
};

DeviceLayout compute_layout(uint32_t n_bitmaps,
                            uint32_t total_containers,
                            uint64_t total_kidx_len)
{
    using detail::align_up;
    DeviceLayout L{};
    size_t off = 0;
    L.cstart_off  = off;
    off = align_up(off + (n_bitmaps + 1u) * sizeof(uint32_t), ALIGN);
    L.kstart_off  = off;
    off = align_up(off + (n_bitmaps + 1u) * sizeof(uint32_t), ALIGN);
    L.keys_off    = off;
    off = align_up(off + total_containers * sizeof(uint16_t), ALIGN);
    L.types_off   = off;
    off = align_up(off + total_containers * sizeof(ContainerType), ALIGN);
    L.offsets_off = off;
    off = align_up(off + total_containers * sizeof(uint32_t), ALIGN);
    L.cards_off   = off;
    off = align_up(off + total_containers * sizeof(uint16_t), ALIGN);
    L.kidx_off    = off;
    off = align_up(off + total_kidx_len * sizeof(uint16_t), ALIGN);
    L.bmp_off     = off;
    off += static_cast<size_t>(total_containers) * 1024u * sizeof(uint64_t);
    L.total_bytes = align_up(off, ALIGN);
    if (L.total_bytes == 0) L.total_bytes = ALIGN;
    return L;
}

// One block per global container. Reads the source container (selected by
// src_types[cid]) into shared memory as a 1024 × uint64 dense bitmap, copies it
// out to the output bitmap pool slot, and reduces popcount for the per-container
// cardinality. Also writes types[cid] = BITMAP and offsets[cid] = cid * 8192.
//
// Keys, key_indices, container_starts, key_index_starts are NOT touched here —
// they're D2D-copied from the source batch since promotion doesn't change them.
__global__ void promote_kernel(
    const ContainerType* src_types,
    const uint32_t*      src_offsets,
    const uint16_t*      src_cards,
    const uint64_t*      src_bitmap_pool,
    const uint16_t*      src_array_pool,
    const uint16_t*      src_run_pool,
    ContainerType*       dst_types,
    uint32_t*            dst_offsets,
    uint16_t*            dst_cards,
    uint64_t*            dst_bitmap_pool,
    uint32_t             total_containers)
{
    __shared__ uint64_t smem[1024];
    __shared__ uint32_t warp_sum[8];

    const uint32_t cid = blockIdx.x;
    if (cid >= total_containers) return;

    for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
        smem[w] = 0ULL;
    }
    __syncthreads();

    const ContainerType type = src_types[cid];
    const uint32_t      off  = src_offsets[cid];
    const uint16_t      card = src_cards[cid];

    if (type == ContainerType::BITMAP) {
        const uint64_t* src = reinterpret_cast<const uint64_t*>(
            reinterpret_cast<const char*>(src_bitmap_pool) + off);
        for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
            smem[w] = src[w];
        }
    } else if (type == ContainerType::ARRAY) {
        const uint16_t* arr = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(src_array_pool) + off);
        // Threads scatter their assigned values into shared bits. Word-level
        // collisions go through atomicOr on shared memory (a few cycles each).
        for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
            const uint16_t val = arr[i];
            atomicOr(reinterpret_cast<unsigned long long*>(&smem[val / 64u]),
                     static_cast<unsigned long long>(1ULL << (val % 64u)));
        }
    } else {  // RUN
        const uint16_t* runs = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(src_run_pool) + off);
        for (uint32_t r = threadIdx.x; r < card; r += blockDim.x) {
            const uint32_t start  = runs[r * 2u];
            const uint32_t length = runs[r * 2u + 1u];
            uint32_t end = start + length;
            if (end > 0xFFFFu) end = 0xFFFFu;

            const uint32_t head_word = start / 64u;
            const uint32_t tail_word = end   / 64u;
            const uint32_t head_bit  = start % 64u;
            const uint32_t tail_bit  = end   % 64u;

            if (head_word == tail_word) {
                const uint32_t nbits = tail_bit - head_bit + 1u;
                const uint64_t mask  = (nbits == 64u)
                    ? ~0ULL
                    : (((1ULL << nbits) - 1ULL) << head_bit);
                atomicOr(reinterpret_cast<unsigned long long*>(&smem[head_word]),
                         static_cast<unsigned long long>(mask));
            } else {
                atomicOr(reinterpret_cast<unsigned long long*>(&smem[head_word]),
                         static_cast<unsigned long long>(~0ULL << head_bit));
                for (uint32_t w = head_word + 1u; w < tail_word; ++w) {
                    atomicOr(reinterpret_cast<unsigned long long*>(&smem[w]),
                             static_cast<unsigned long long>(~0ULL));
                }
                const uint64_t tail_mask = (tail_bit == 63u)
                    ? ~0ULL
                    : ((1ULL << (tail_bit + 1u)) - 1ULL);
                atomicOr(reinterpret_cast<unsigned long long*>(&smem[tail_word]),
                         static_cast<unsigned long long>(tail_mask));
            }
        }
    }
    __syncthreads();

    uint64_t* dst = dst_bitmap_pool + static_cast<size_t>(cid) * 1024u;
    uint32_t  local_pop = 0;
    for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
        const uint64_t word = smem[w];
        dst[w] = word;
        local_pop += static_cast<uint32_t>(__popcll(word));
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
            dst_types[cid]   = ContainerType::BITMAP;
            dst_offsets[cid] = cid * 8192u;
            dst_cards[cid]   = static_cast<uint16_t>(v > 65535u ? 0u : v);
        }
    }
}

// Clone the source's host metadata block into a fresh allocation and patch
// host_n_bitmap_containers[b] = n_containers[b] (everything is BITMAP after
// promotion). Other fields are unchanged.
char* clone_and_patch_host_meta(const GpuRoaringBatch&        src,
                                const detail::HostMetaLayout& M)
{
    const uint32_t n = src.n_bitmaps;
    char* meta = new char[M.total_bytes];
    std::memset(meta, 0, M.total_bytes);

    if (n > 0) {
        std::memcpy(meta + M.total_card_off, src.host_total_cardinalities,
                    n * sizeof(uint64_t));
        std::memcpy(meta + M.universe_off, src.host_universe_sizes,
                    n * sizeof(uint32_t));
    }
    std::memcpy(meta + M.cstart_off, src.host_container_starts,
                (n + 1u) * sizeof(uint32_t));
    std::memcpy(meta + M.kstart_off, src.host_key_index_starts,
                (n + 1u) * sizeof(uint32_t));

    auto* nbm = reinterpret_cast<uint32_t*>(meta + M.n_bitmap_off);
    for (uint32_t b = 0; b < n; ++b) {
        nbm[b] = src.host_container_starts[b + 1] - src.host_container_starts[b];
    }
    return meta;
}

} // namespace

GpuRoaringBatch promote_batch(const GpuRoaringBatch& batch, cudaStream_t stream)
{
    if (batch.n_bitmaps == 0) return GpuRoaringBatch{};

    const uint32_t n          = batch.n_bitmaps;
    const uint32_t total      = batch.total_containers;
    const uint64_t kidx_total =
        static_cast<uint64_t>(batch.host_key_index_starts[n]);

    const DeviceLayout            L = compute_layout(n, total, kidx_total);
    const detail::HostMetaLayout  M = detail::compute_host_meta_layout(n);

    char* d_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_buf),
                                             L.total_bytes, stream));

    auto* d_cstart = reinterpret_cast<uint32_t*>(d_buf + L.cstart_off);
    auto* d_kstart = reinterpret_cast<uint32_t*>(d_buf + L.kstart_off);
    auto* d_keys   = reinterpret_cast<uint16_t*>(d_buf + L.keys_off);
    auto* d_types  = reinterpret_cast<ContainerType*>(d_buf + L.types_off);
    auto* d_offs   = reinterpret_cast<uint32_t*>(d_buf + L.offsets_off);
    auto* d_cards  = reinterpret_cast<uint16_t*>(d_buf + L.cards_off);
    auto* d_kidx   = reinterpret_cast<uint16_t*>(d_buf + L.kidx_off);
    auto* d_bmp    = reinterpret_cast<uint64_t*>(d_buf + L.bmp_off);

    // D2D copies of the sections promotion preserves verbatim. All async on
    // the stream — no sync, no host round-trip.
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
        d_cstart, batch.container_starts,
        (n + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
        d_kstart, batch.key_index_starts,
        (n + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    if (total > 0) {
        CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
            d_keys, batch.keys,
            total * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
    }
    if (kidx_total > 0) {
        CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
            d_kidx, batch.key_indices,
            kidx_total * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
    }

    if (total > 0) {
        promote_kernel<<<total, 256, 0, stream>>>(
            batch.types, batch.offsets, batch.cardinalities,
            batch.bitmap_data, batch.array_data, batch.run_data,
            d_types, d_offs, d_cards, d_bmp,
            total);
        CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());
    }

    char* h_meta = clone_and_patch_host_meta(batch, M);

    GpuRoaringBatch out{};
    out.n_bitmaps                 = n;
    out.total_containers          = total;
    out.n_bitmap_containers_total = total;
    out.array_pool_bytes          = 0;
    out.run_pool_bytes            = 0;

    out.container_starts = d_cstart;
    out.key_index_starts = d_kstart;
    out.keys             = d_keys;
    out.types            = d_types;
    out.offsets          = d_offs;
    out.cardinalities    = d_cards;
    out.key_indices      = d_kidx;
    out.bitmap_data      = (total > 0) ? d_bmp : nullptr;
    out.array_data       = nullptr;
    out.run_data         = nullptr;

    out._alloc_base     = d_buf;
    out._host_meta_base = h_meta;

    out.host_total_cardinalities  =
        reinterpret_cast<uint64_t*>(h_meta + M.total_card_off);
    out.host_universe_sizes       =
        reinterpret_cast<uint32_t*>(h_meta + M.universe_off);
    out.host_container_starts     =
        reinterpret_cast<uint32_t*>(h_meta + M.cstart_off);
    out.host_key_index_starts     =
        reinterpret_cast<uint32_t*>(h_meta + M.kstart_off);
    out.host_n_bitmap_containers  =
        reinterpret_cast<uint32_t*>(h_meta + M.n_bitmap_off);

    return out;
}

} // namespace cu_roaring::v2
