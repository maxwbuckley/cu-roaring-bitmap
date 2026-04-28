#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <cub/device/device_scan.cuh>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace cu_roaring::v2 {

namespace {

constexpr size_t ALIGN = 8;

// Output layout for a 1-bitmap batch. We over-allocate to anchor_n containers
// so we never need to know n_common host-side before building the output buffer
// — the actual n_common only flows to the host once at the very end (single
// 12-byte D2H, fused with total_cardinality).
struct OutLayout {
    size_t cstart_off;
    size_t kstart_off;
    size_t keys_off;
    size_t types_off;
    size_t offsets_off;
    size_t cards_off;
    size_t kidx_off;
    size_t bmp_off;
    size_t total_bytes;
};

OutLayout compute_out_layout(uint32_t n_upper, uint32_t kidx_len) {
    using detail::align_up;
    OutLayout L{};
    size_t off = 0;
    L.cstart_off  = off; off = align_up(off + 2u * sizeof(uint32_t), ALIGN);
    L.kstart_off  = off; off = align_up(off + 2u * sizeof(uint32_t), ALIGN);
    L.keys_off    = off; off = align_up(off + n_upper * sizeof(uint16_t), ALIGN);
    L.types_off   = off; off = align_up(off + n_upper * sizeof(ContainerType), ALIGN);
    L.offsets_off = off; off = align_up(off + n_upper * sizeof(uint32_t), ALIGN);
    L.cards_off   = off; off = align_up(off + n_upper * sizeof(uint16_t), ALIGN);
    L.kidx_off    = off; off = align_up(off + kidx_len * sizeof(uint16_t), ALIGN);
    L.bmp_off     = off; off += static_cast<size_t>(n_upper) * 1024u * sizeof(uint64_t);
    L.total_bytes = align_up(off, ALIGN);
    if (L.total_bytes == 0) L.total_bytes = ALIGN;
    return L;
}

// Pure-GPU key intersection. One thread per anchor container. For each
// non-anchor input, the thread does an O(1) direct-map lookup into that input's
// key_indices slice; if any input is missing the key, flag = 0. Anchor's own
// key is trivially present.
//
// O(anchor_n * n_inputs) device work, no D2H.
__global__ void intersect_anchor_kernel(
    const uint32_t* __restrict__ batch_container_starts,
    const uint32_t* __restrict__ batch_key_index_starts,
    const uint16_t* __restrict__ batch_keys,
    const uint16_t* __restrict__ batch_key_indices,
    const uint32_t* __restrict__ d_input_indices,
    uint32_t                     n_inputs,
    uint32_t                     anchor_b,
    uint32_t                     anchor_cstart,
    uint32_t                     anchor_n,
    uint32_t*       __restrict__ flags)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= anchor_n) return;

    const uint16_t key = batch_keys[anchor_cstart + i];
    const uint32_t key_u = static_cast<uint32_t>(key);

    for (uint32_t j = 0; j < n_inputs; ++j) {
        const uint32_t b = d_input_indices[j];
        if (b == anchor_b) continue;

        const uint32_t kstart   = batch_key_index_starts[b];
        const uint32_t kend     = batch_key_index_starts[b + 1];
        const uint32_t kidx_len = kend - kstart;
        if (key_u >= kidx_len) {
            flags[i] = 0u;
            return;
        }
        const uint16_t local = batch_key_indices[kstart + key_u];
        if (local == 0xFFFFu) {
            flags[i] = 0u;
            return;
        }
        // batch_container_starts is otherwise unused here; it will be re-read
        // by build_and_kernel for the matched anchors. Reading it twice is
        // cheaper than threading a per-(i,j) intermediate buffer through the
        // pipeline.
        (void)batch_container_starts;
    }
    flags[i] = 1u;
}

// Final-count kernel. Runs after the exclusive-sum scan; reads flags[n-1] and
// compacted_pos[n-1] to produce the inclusive total. One thread, one block.
__global__ void compute_count_kernel(const uint32_t* flags,
                                     const uint32_t* compacted_pos,
                                     uint32_t        n,
                                     uint32_t*       d_n_common)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (n == 0u) { *d_n_common = 0u; return; }
    *d_n_common = compacted_pos[n - 1u] + (flags[n - 1u] != 0u ? 1u : 0u);
}

// Pulls the count from the device counter into container_starts[1] without
// any host round-trip. Also fills the static [0, kidx_len] for key_index_starts.
__global__ void write_starts_kernel(const uint32_t* d_n_common,
                                    uint32_t        out_kidx_len,
                                    uint32_t*       out_cstart,
                                    uint32_t*       out_kstart)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out_cstart[0] = 0u;
    out_cstart[1] = *d_n_common;
    out_kstart[0] = 0u;
    out_kstart[1] = out_kidx_len;
}

// One block per anchor container (n_anchor blocks). Flagged blocks resolve
// per-input source bitmap pointers via a single cooperative pass into shared
// memory, then AND-reduce 1024 words while accumulating popcount.
//
// Dynamic shared memory size: n_inputs * sizeof(const uint64_t*).
__global__ void build_and_kernel(
    const uint32_t*     __restrict__ batch_container_starts,
    const uint32_t*     __restrict__ batch_key_index_starts,
    const uint16_t*     __restrict__ batch_keys,
    const uint32_t*     __restrict__ batch_offsets,
    const uint16_t*     __restrict__ batch_key_indices,
    const uint64_t*     __restrict__ batch_bitmap_data,
    const uint32_t*     __restrict__ d_input_indices,
    uint32_t                         n_inputs,
    uint32_t                         anchor_cstart,
    uint32_t                         anchor_n,
    const uint32_t*     __restrict__ flags,
    const uint32_t*     __restrict__ compacted_pos,
    uint16_t*           __restrict__ out_keys,
    ContainerType*      __restrict__ out_types,
    uint32_t*           __restrict__ out_offsets,
    uint16_t*           __restrict__ out_cards,
    uint16_t*           __restrict__ out_key_indices,
    uint64_t*           __restrict__ out_bitmap_data,
    unsigned long long* __restrict__ d_total_card)
{
    extern __shared__ unsigned char shmem[];
    auto** src_ptrs = reinterpret_cast<const uint64_t**>(shmem);
    __shared__ uint32_t warp_sum[8];

    const uint32_t i = blockIdx.x;
    if (i >= anchor_n) return;
    if (flags[i] == 0u) return;

    const uint32_t k   = compacted_pos[i];
    const uint16_t key = batch_keys[anchor_cstart + i];

    // Cooperative pointer table fill — one strided pass; each lookup is the
    // same direct-map cost as contains(): 1 read for key_index, 1 for offsets.
    for (uint32_t j = threadIdx.x; j < n_inputs; j += blockDim.x) {
        const uint32_t b      = d_input_indices[j];
        const uint32_t kstart = batch_key_index_starts[b];
        const uint16_t local  = batch_key_indices[kstart + key];
        const uint32_t gcid   = batch_container_starts[b]
                                + static_cast<uint32_t>(local);
        const uint32_t off    = batch_offsets[gcid];
        src_ptrs[j] = reinterpret_cast<const uint64_t*>(
            reinterpret_cast<const char*>(batch_bitmap_data) + off);
    }
    __syncthreads();

    uint64_t* dst = out_bitmap_data + static_cast<size_t>(k) * 1024u;
    uint32_t  local_pop = 0;
    for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
        uint64_t acc = ~0ULL;
        #pragma unroll 1
        for (uint32_t j = 0; j < n_inputs; ++j) {
            acc &= src_ptrs[j][w];
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
            out_keys[k]          = key;
            out_types[k]         = ContainerType::BITMAP;
            out_offsets[k]       = k * 8192u;
            out_cards[k]         = static_cast<uint16_t>(v > 65535u ? 0u : v);
            out_key_indices[key] = static_cast<uint16_t>(k);
            atomicAdd(d_total_card, static_cast<unsigned long long>(v));
        }
    }
}

GpuRoaringBatch allocate_empty_one_batch(cudaStream_t stream) {
    // Minimal device-side state for a 1-bitmap batch whose single bitmap is
    // empty: just the two CSR start arrays of length 2, all zero. No pools,
    // no per-container metadata.
    constexpr size_t bytes = 2u * 2u * sizeof(uint32_t);  // cstart + kstart
    char* d_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_buf),
                                             bytes, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemsetAsync(d_buf, 0, bytes, stream));

    const detail::HostMetaLayout M = detail::compute_host_meta_layout(1);
    auto h_meta = std::make_unique<char[]>(M.total_bytes);
    std::memset(h_meta.get(), 0, M.total_bytes);

    GpuRoaringBatch B{};
    B.n_bitmaps        = 1;
    B.total_containers = 0;
    B.container_starts = reinterpret_cast<uint32_t*>(d_buf);
    B.key_index_starts = reinterpret_cast<uint32_t*>(d_buf + 2u * sizeof(uint32_t));
    B._alloc_base      = d_buf;
    B._host_meta_base  = h_meta.get();
    B.host_total_cardinalities  =
        reinterpret_cast<uint64_t*>(h_meta.get() + M.total_card_off);
    B.host_universe_sizes       =
        reinterpret_cast<uint32_t*>(h_meta.get() + M.universe_off);
    B.host_container_starts     =
        reinterpret_cast<uint32_t*>(h_meta.get() + M.cstart_off);
    B.host_key_index_starts     =
        reinterpret_cast<uint32_t*>(h_meta.get() + M.kstart_off);
    B.host_n_bitmap_containers  =
        reinterpret_cast<uint32_t*>(h_meta.get() + M.n_bitmap_off);
    h_meta.release();
    return B;
}

void validate_inputs(const GpuRoaringBatch& batch,
                     const uint32_t*        input_indices,
                     uint32_t               n_inputs)
{
    if (n_inputs == 0u) {
        throw std::invalid_argument("multi_and: n_inputs must be >= 1");
    }
    if (input_indices == nullptr) {
        throw std::invalid_argument("multi_and: input_indices is null");
    }
    for (uint32_t k = 0; k < n_inputs; ++k) {
        const uint32_t b = input_indices[k];
        if (b >= batch.n_bitmaps) {
            throw std::out_of_range(
                "multi_and: input_indices[" + std::to_string(k) +
                "] = " + std::to_string(b) +
                " out of range for batch of size " +
                std::to_string(batch.n_bitmaps));
        }
        const uint32_t n_b =
            batch.host_container_starts[b + 1] - batch.host_container_starts[b];
        if (batch.host_n_bitmap_containers[b] != n_b) {
            throw std::invalid_argument(
                "multi_and: input_indices[" + std::to_string(k) +
                "] = " + std::to_string(b) +
                " has non-bitmap containers; call promote_batch first");
        }
    }
}

} // namespace

GpuRoaringBatch multi_and(const GpuRoaringBatch& batch,
                          const uint32_t*        input_indices,
                          uint32_t               n_inputs,
                          cudaStream_t           stream)
{
    validate_inputs(batch, input_indices, n_inputs);

    // Short-circuit: any selected bitmap is empty → result is empty.
    for (uint32_t k = 0; k < n_inputs; ++k) {
        const uint32_t b = input_indices[k];
        if (batch.host_container_starts[b + 1] ==
            batch.host_container_starts[b]) {
            return allocate_empty_one_batch(stream);
        }
    }

    // Pick anchor on host: smallest n_containers across selected inputs.
    uint32_t anchor_pos = 0;
    uint32_t best_n = batch.host_container_starts[input_indices[0] + 1] -
                      batch.host_container_starts[input_indices[0]];
    for (uint32_t k = 1; k < n_inputs; ++k) {
        const uint32_t b = input_indices[k];
        const uint32_t n_b = batch.host_container_starts[b + 1] -
                             batch.host_container_starts[b];
        if (n_b < best_n) { best_n = n_b; anchor_pos = k; }
    }
    const uint32_t anchor_b      = input_indices[anchor_pos];
    const uint32_t anchor_cstart = batch.host_container_starts[anchor_b];
    const uint32_t anchor_n      = best_n;
    const uint32_t anchor_kstart = batch.host_key_index_starts[anchor_b];
    const uint32_t anchor_kend   = batch.host_key_index_starts[anchor_b + 1];
    const uint32_t anchor_kidx_len = anchor_kend - anchor_kstart;

    // --- Allocate temp buffers for the intersect/scan pipeline. ------------
    uint32_t*           d_input_indices = nullptr;
    uint32_t*           d_flags         = nullptr;
    uint32_t*           d_compacted_pos = nullptr;
    uint32_t*           d_n_common      = nullptr;
    unsigned long long* d_total_card    = nullptr;
    void*               d_scan_temp     = nullptr;

    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void**>(&d_input_indices),
        n_inputs * sizeof(uint32_t), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void**>(&d_flags),
        anchor_n * sizeof(uint32_t), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void**>(&d_compacted_pos),
        anchor_n * sizeof(uint32_t), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void**>(&d_n_common), sizeof(uint32_t), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void**>(&d_total_card),
        sizeof(unsigned long long), stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemsetAsync(
        d_total_card, 0, sizeof(unsigned long long), stream));

    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
        d_input_indices, input_indices,
        n_inputs * sizeof(uint32_t),
        cudaMemcpyHostToDevice, stream));

    // --- Kernel A: per-anchor key intersection. -----------------------------
    {
        const uint32_t blocks = (anchor_n + 255u) / 256u;
        intersect_anchor_kernel<<<blocks, 256, 0, stream>>>(
            batch.container_starts, batch.key_index_starts, batch.keys,
            batch.key_indices,
            d_input_indices, n_inputs,
            anchor_b, anchor_cstart, anchor_n,
            d_flags);
        CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());
    }

    // --- CUB exclusive-sum scan over flags → compacted_pos. -----------------
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                  d_flags, d_compacted_pos,
                                  static_cast<int>(anchor_n), stream);
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(&d_scan_temp, scan_temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes,
                                  d_flags, d_compacted_pos,
                                  static_cast<int>(anchor_n), stream);
    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_scan_temp, stream));

    // n_common = compacted_pos[anchor_n - 1] + flags[anchor_n - 1].
    compute_count_kernel<<<1, 1, 0, stream>>>(
        d_flags, d_compacted_pos, anchor_n, d_n_common);
    CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());

    // --- Allocate output, over-sized to anchor_n containers (upper bound). --
    const uint32_t out_kidx_len = anchor_kidx_len;
    const OutLayout L = compute_out_layout(anchor_n, out_kidx_len);

    char* d_out_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(
        reinterpret_cast<void**>(&d_out_buf), L.total_bytes, stream));

    auto* d_cstart = reinterpret_cast<uint32_t*>(d_out_buf + L.cstart_off);
    auto* d_kstart = reinterpret_cast<uint32_t*>(d_out_buf + L.kstart_off);
    auto* d_keys   = reinterpret_cast<uint16_t*>(d_out_buf + L.keys_off);
    auto* d_types  = reinterpret_cast<ContainerType*>(d_out_buf + L.types_off);
    auto* d_offs   = reinterpret_cast<uint32_t*>(d_out_buf + L.offsets_off);
    auto* d_cards  = reinterpret_cast<uint16_t*>(d_out_buf + L.cards_off);
    auto* d_kidx   = reinterpret_cast<uint16_t*>(d_out_buf + L.kidx_off);
    auto* d_bmp    = reinterpret_cast<uint64_t*>(d_out_buf + L.bmp_off);

    // Preset output key_indices to the 0xFFFF "absent" sentinel.
    if (out_kidx_len > 0) {
        CU_ROARING_V2_CHECK_CUDA(cudaMemsetAsync(
            d_kidx, 0xFF, out_kidx_len * sizeof(uint16_t), stream));
    }

    // CSR start arrays. Pulls the count from d_n_common — pure GPU, no H2D.
    write_starts_kernel<<<1, 1, 0, stream>>>(
        d_n_common, out_kidx_len, d_cstart, d_kstart);
    CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());

    // --- Kernel B: build the output containers + AND. ----------------------
    {
        const size_t shared_bytes =
            static_cast<size_t>(n_inputs) * sizeof(const uint64_t*);
        build_and_kernel<<<anchor_n, 256, shared_bytes, stream>>>(
            batch.container_starts, batch.key_index_starts, batch.keys,
            batch.offsets, batch.key_indices, batch.bitmap_data,
            d_input_indices, n_inputs,
            anchor_cstart, anchor_n,
            d_flags, d_compacted_pos,
            d_keys, d_types, d_offs, d_cards,
            d_kidx, d_bmp,
            d_total_card);
        CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());
    }

    // --- Single sync, two scalars: n_common (4B) + total_card (8B). ---------
    uint32_t           n_common_host   = 0u;
    unsigned long long total_card_host = 0ull;
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
        &n_common_host, d_n_common,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(
        &total_card_host, d_total_card,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaStreamSynchronize(stream));

    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_input_indices, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_flags,         stream));
    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_compacted_pos, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_n_common,      stream));
    CU_ROARING_V2_CHECK_CUDA(cudaFreeAsync(d_total_card,    stream));

    // --- Build the result GpuRoaringBatch struct. ---------------------------
    const detail::HostMetaLayout M = detail::compute_host_meta_layout(1);
    auto h_meta = std::make_unique<char[]>(M.total_bytes);
    std::memset(h_meta.get(), 0, M.total_bytes);

    auto* h_total_card = reinterpret_cast<uint64_t*>(h_meta.get() + M.total_card_off);
    auto* h_universe   = reinterpret_cast<uint32_t*>(h_meta.get() + M.universe_off);
    auto* h_cstart_h   = reinterpret_cast<uint32_t*>(h_meta.get() + M.cstart_off);
    auto* h_kstart_h   = reinterpret_cast<uint32_t*>(h_meta.get() + M.kstart_off);
    auto* h_nbm        = reinterpret_cast<uint32_t*>(h_meta.get() + M.n_bitmap_off);

    h_total_card[0] = static_cast<uint64_t>(total_card_host);
    // out_kidx_len is an upper bound on max_key + 1; the actual max may be
    // smaller, but using the bound is safe — extra key_indices entries are
    // 0xFFFF sentinels and contains() returns false for them.
    h_universe[0]  = out_kidx_len << 16;
    h_cstart_h[0]  = 0u;
    h_cstart_h[1]  = n_common_host;
    h_kstart_h[0]  = 0u;
    h_kstart_h[1]  = out_kidx_len;
    h_nbm[0]       = n_common_host;

    GpuRoaringBatch out{};
    out.n_bitmaps                 = 1;
    out.total_containers          = n_common_host;
    out.n_bitmap_containers_total = n_common_host;
    out.array_pool_bytes          = 0;
    out.run_pool_bytes            = 0;

    out.container_starts = d_cstart;
    out.key_index_starts = d_kstart;
    out.keys             = d_keys;
    out.types            = d_types;
    out.offsets          = d_offs;
    out.cardinalities    = d_cards;
    out.key_indices      = d_kidx;
    out.bitmap_data      = (n_common_host > 0) ? d_bmp : nullptr;
    out.array_data       = nullptr;
    out.run_data         = nullptr;

    out._alloc_base     = d_out_buf;
    out._host_meta_base = h_meta.get();
    out.host_total_cardinalities  = h_total_card;
    out.host_universe_sizes       = h_universe;
    out.host_container_starts     = h_cstart_h;
    out.host_key_index_starts     = h_kstart_h;
    out.host_n_bitmap_containers  = h_nbm;
    h_meta.release();
    return out;
}

} // namespace cu_roaring::v2
