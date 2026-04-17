#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

namespace cu_roaring::v2 {

namespace {

// One block per source container. Each block fully materialises its 8 KB output
// bitmap in shared memory, copies it out, and block-reduces the popcount to
// write the per-container cardinality. The kernel also writes keys, types,
// offsets, and scatters the direct-map key_index entry for this container.
__global__ void promote_kernel(
    const uint16_t*      src_keys,
    const ContainerType* src_types,
    const uint32_t*      src_offsets,
    const uint16_t*      src_cards,
    const uint64_t*      src_bitmap_pool,  // may be nullptr when n_bitmap == 0
    const uint16_t*      src_array_pool,   // may be nullptr
    const uint16_t*      src_run_pool,     // may be nullptr
    uint16_t*            dst_keys,
    ContainerType*       dst_types,
    uint32_t*            dst_offsets,
    uint16_t*            dst_cards,
    uint64_t*            dst_bitmap_pool,
    uint16_t*            dst_key_index,
    uint32_t             n)
{
    __shared__ uint64_t smem[1024];
    __shared__ uint32_t warp_sum[8];  // 256 threads / 32 lanes = 8 warps

    const uint32_t cid = blockIdx.x;
    if (cid >= n) return;

    // Zero the shared output accumulator for every container (including BITMAP,
    // which will overwrite every word — the zero is a cheap no-op path).
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
        // Thread-strided scatter. Collisions on the same 64-bit word require
        // atomicOr on shared memory (cheap: a handful of cycles on smem).
        for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
            const uint16_t val = arr[i];
            atomicOr(reinterpret_cast<unsigned long long*>(&smem[val / 64u]),
                     static_cast<unsigned long long>(1ULL << (val % 64u)));
        }
    } else {  // RUN
        const uint16_t* runs = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(src_run_pool) + off);
        // One thread per run; each thread emits whole 64-bit words instead of
        // bit-by-bit. A run of length L costs ceil((L+1)/64) atomicOrs rather
        // than L+1 individual bit sets.
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

    // Copy smem → global bitmap pool while accumulating popcount for cardinality.
    uint64_t* dst = dst_bitmap_pool + static_cast<size_t>(cid) * 1024u;
    uint32_t  local_pop = 0;
    for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
        const uint64_t word = smem[w];
        dst[w] = word;
        local_pop += static_cast<uint32_t>(__popcll(word));
    }

    // Warp-shuffle reduce, then 8-warp aggregation.
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
            const uint16_t key = src_keys[cid];
            dst_keys[cid]          = key;
            dst_types[cid]         = ContainerType::BITMAP;
            dst_offsets[cid]       = cid * 8192u;  // bytes
            dst_cards[cid]         = static_cast<uint16_t>(v > 65535u ? 0u : v);
            dst_key_index[key]     = static_cast<uint16_t>(cid);
        }
    }
}

struct Layout {
    size_t keys_off, types_off, offsets_off, cards_off, kidx_off, bmp_off;
    size_t total_bytes;
    uint32_t key_index_len;
};

Layout compute_layout(uint32_t n, uint16_t max_key) {
    using detail::align_up;
    Layout L{};
    L.key_index_len = static_cast<uint32_t>(max_key) + 1u;

    constexpr size_t A = 8;
    size_t off = 0;
    L.keys_off    = off; off = align_up(off + n * sizeof(uint16_t), A);
    L.types_off   = off; off = align_up(off + n * sizeof(ContainerType), A);
    L.offsets_off = off; off = align_up(off + n * sizeof(uint32_t), A);
    L.cards_off   = off; off = align_up(off + n * sizeof(uint16_t), A);
    L.kidx_off    = off; off = align_up(off + L.key_index_len * sizeof(uint16_t), A);
    L.bmp_off     = off; off += static_cast<size_t>(n) * 1024u * sizeof(uint64_t);
    L.total_bytes = align_up(off, A);
    return L;
}

} // namespace

GpuRoaring promote_to_bitmap(const GpuRoaring& bm, cudaStream_t stream) {
    if (bm.n_containers == 0) {
        GpuRoaring r{};
        r.universe_size = bm.universe_size;
        return r;
    }

    const Layout L = compute_layout(bm.n_containers, bm.max_key);

    char* d_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_buf),
                                             L.total_bytes, stream));
    // key_index defaults to 0xFFFF per entry; the kernel will scatter the present
    // containers in a single pass.
    CU_ROARING_V2_CHECK_CUDA(cudaMemsetAsync(d_buf + L.kidx_off, 0xFF,
                                             L.key_index_len * sizeof(uint16_t),
                                             stream));

    auto* d_keys    = reinterpret_cast<uint16_t*>(d_buf + L.keys_off);
    auto* d_types   = reinterpret_cast<ContainerType*>(d_buf + L.types_off);
    auto* d_offs    = reinterpret_cast<uint32_t*>(d_buf + L.offsets_off);
    auto* d_cards   = reinterpret_cast<uint16_t*>(d_buf + L.cards_off);
    auto* d_kidx    = reinterpret_cast<uint16_t*>(d_buf + L.kidx_off);
    auto* d_bmp     = reinterpret_cast<uint64_t*>(d_buf + L.bmp_off);

    promote_kernel<<<bm.n_containers, 256, 0, stream>>>(
        bm.keys, bm.types, bm.offsets, bm.cardinalities,
        bm.bitmap_data, bm.array_data, bm.run_data,
        d_keys, d_types, d_offs, d_cards, d_bmp, d_kidx,
        bm.n_containers);
    CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());

    GpuRoaring r{};
    r._alloc_base         = d_buf;
    r.n_containers        = bm.n_containers;
    r.n_bitmap_containers = bm.n_containers;
    r.universe_size       = bm.universe_size;
    r.max_key             = bm.max_key;
    r.total_cardinality   = bm.total_cardinality;  // promotion preserves the set
    r.keys          = d_keys;
    r.types         = d_types;
    r.offsets       = d_offs;
    r.cardinalities = d_cards;
    r.key_index     = d_kidx;
    r.bitmap_data   = d_bmp;
    return r;
}

} // namespace cu_roaring::v2
