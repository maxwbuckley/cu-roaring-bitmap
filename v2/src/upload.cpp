#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <roaring/roaring.h>
#include <roaring/containers/array.h>
#include <roaring/containers/bitset.h>
#include <roaring/containers/run.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// CRoaring v4.x exposes container internals under roaring::internal.
using roaring::internal::array_container_t;
using roaring::internal::bitset_container_t;
using roaring::internal::run_container_t;

namespace cu_roaring::v2 {

namespace {

// CRoaring internal type codes.
constexpr uint8_t CROARING_BITSET = 1;
constexpr uint8_t CROARING_ARRAY  = 2;
constexpr uint8_t CROARING_RUN    = 3;

constexpr size_t ALIGN = 8;

struct DeviceLayout {
    size_t cstart_off, kstart_off;
    size_t keys_off, types_off, offsets_off, cards_off;
    size_t kidx_off;
    size_t bmp_off, arr_off, run_off;
    size_t total_bytes;
};

// Per-bitmap rollup computed in the first pass over the input array.
struct PerBitmap {
    uint32_t n_containers;
    uint32_t n_bitmap_containers;
    uint32_t key_index_len;     // = max_key + 1, or 0 if empty
    uint32_t universe_size;
    uint64_t total_cardinality;
};

void compute_per_bitmap(const roaring_bitmap_t* const* cpus,
                        uint32_t                       n,
                        std::vector<PerBitmap>&        out,
                        size_t&                        total_bmp_words,
                        size_t&                        total_arr_elems,
                        size_t&                        total_run_pairs,
                        uint32_t&                      total_containers,
                        uint32_t&                      total_n_bitmap,
                        uint64_t&                      total_key_index_len)
{
    out.resize(n);
    total_bmp_words     = 0;
    total_arr_elems     = 0;
    total_run_pairs     = 0;
    total_containers    = 0;
    total_n_bitmap      = 0;
    total_key_index_len = 0;

    for (uint32_t b = 0; b < n; ++b) {
        if (cpus[b] == nullptr) {
            throw std::invalid_argument("upload_batch: cpus[" +
                                        std::to_string(b) + "] is null");
        }
        const roaring_array_t& ra = cpus[b]->high_low_container;
        PerBitmap& p = out[b];
        p.n_containers        = static_cast<uint32_t>(ra.size);
        p.n_bitmap_containers = 0;
        p.total_cardinality   =
            static_cast<uint64_t>(roaring_bitmap_get_cardinality(cpus[b]));

        if (p.n_containers == 0) {
            p.key_index_len = 0;
            p.universe_size = 0;
        } else {
            const uint16_t max_key = ra.keys[ra.size - 1];
            p.key_index_len = static_cast<uint32_t>(max_key) + 1u;
            p.universe_size = (static_cast<uint32_t>(max_key) + 1u) << 16;
        }

        for (uint32_t i = 0; i < p.n_containers; ++i) {
            const uint8_t tc = ra.typecodes[i];
            if (tc == CROARING_BITSET) {
                ++p.n_bitmap_containers;
                total_bmp_words += 1024;
            } else if (tc == CROARING_ARRAY) {
                const auto* ac =
                    static_cast<const array_container_t*>(ra.containers[i]);
                total_arr_elems += static_cast<size_t>(ac->cardinality);
            } else if (tc == CROARING_RUN) {
                const auto* rc =
                    static_cast<const run_container_t*>(ra.containers[i]);
                total_run_pairs += static_cast<size_t>(rc->n_runs);
            } else {
                throw std::runtime_error(
                    "upload_batch: unknown CRoaring typecode " +
                    std::to_string(tc));
            }
        }

        total_containers    += p.n_containers;
        total_n_bitmap      += p.n_bitmap_containers;
        total_key_index_len += p.key_index_len;
    }
}

DeviceLayout compute_device_layout(uint32_t n_bitmaps,
                                   uint32_t total_containers,
                                   uint64_t total_kidx_len,
                                   size_t   total_bmp_words,
                                   size_t   total_arr_elems,
                                   size_t   total_run_pairs)
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
    off += total_bmp_words * sizeof(uint64_t);
    L.arr_off     = off;
    off = align_up(off + total_arr_elems * sizeof(uint16_t), ALIGN);
    L.run_off     = off;
    off += total_run_pairs * 2u * sizeof(uint16_t);
    L.total_bytes = align_up(off, ALIGN);
    if (L.total_bytes == 0) L.total_bytes = ALIGN;
    return L;
}

void pack_host_buffer(char*                          buf,
                      const DeviceLayout&            L,
                      const std::vector<PerBitmap>&  per,
                      const roaring_bitmap_t* const* cpus,
                      uint32_t                       n,
                      uint64_t                       total_kidx_len)
{
    auto* h_cstart = reinterpret_cast<uint32_t*>(buf + L.cstart_off);
    auto* h_kstart = reinterpret_cast<uint32_t*>(buf + L.kstart_off);
    auto* h_keys   = reinterpret_cast<uint16_t*>(buf + L.keys_off);
    auto* h_types  = reinterpret_cast<ContainerType*>(buf + L.types_off);
    auto* h_offs   = reinterpret_cast<uint32_t*>(buf + L.offsets_off);
    auto* h_cards  = reinterpret_cast<uint16_t*>(buf + L.cards_off);
    auto* h_kidx   = reinterpret_cast<uint16_t*>(buf + L.kidx_off);
    auto* h_bmp    = reinterpret_cast<uint64_t*>(buf + L.bmp_off);
    auto* h_arr    = reinterpret_cast<uint16_t*>(buf + L.arr_off);
    auto* h_run    = reinterpret_cast<uint16_t*>(buf + L.run_off);

    // 0xFF the entire concatenated key_indices region in one shot — cheaper
    // than a per-bitmap memset, and the empty-slice case (zero-length) is a no-op.
    if (total_kidx_len > 0) {
        std::memset(h_kidx, 0xFF,
                    static_cast<size_t>(total_kidx_len) * sizeof(uint16_t));
    }

    h_cstart[0] = 0;
    h_kstart[0] = 0;
    for (uint32_t b = 0; b < n; ++b) {
        h_cstart[b + 1] = h_cstart[b] + per[b].n_containers;
        h_kstart[b + 1] = h_kstart[b] + per[b].key_index_len;
    }

    size_t bmp_cursor = 0;
    size_t arr_cursor = 0;
    size_t run_cursor = 0;

    for (uint32_t b = 0; b < n; ++b) {
        const PerBitmap& p = per[b];
        if (p.n_containers == 0) continue;

        const roaring_array_t& ra = cpus[b]->high_low_container;
        const uint32_t cstart = h_cstart[b];
        uint16_t* kidx_slice  = h_kidx + h_kstart[b];

        for (uint32_t i = 0; i < p.n_containers; ++i) {
            const uint32_t gid = cstart + i;
            const uint16_t key = ra.keys[i];
            const uint8_t  tc  = ra.typecodes[i];

            h_keys[gid]     = key;
            kidx_slice[key] = static_cast<uint16_t>(i);

            if (tc == CROARING_BITSET) {
                const auto* bc =
                    static_cast<const bitset_container_t*>(ra.containers[i]);
                h_types[gid] = ContainerType::BITMAP;
                h_offs[gid]  = static_cast<uint32_t>(bmp_cursor * sizeof(uint64_t));
                // CRoaring stores cardinality=-1 for the all-ones case. We use 0 as
                // a "full or empty" sentinel; callers disambiguate via popcount if
                // they care.
                const int32_t card = bc->cardinality;
                h_cards[gid] = static_cast<uint16_t>(card > 65535 || card < 0
                                                     ? 0 : card);
                std::memcpy(h_bmp + bmp_cursor, bc->words,
                            1024u * sizeof(uint64_t));
                bmp_cursor += 1024;
            } else if (tc == CROARING_ARRAY) {
                const auto* ac =
                    static_cast<const array_container_t*>(ra.containers[i]);
                h_types[gid] = ContainerType::ARRAY;
                h_offs[gid]  = static_cast<uint32_t>(arr_cursor * sizeof(uint16_t));
                h_cards[gid] = static_cast<uint16_t>(ac->cardinality);
                std::memcpy(h_arr + arr_cursor, ac->array,
                            static_cast<size_t>(ac->cardinality)
                            * sizeof(uint16_t));
                arr_cursor += static_cast<size_t>(ac->cardinality);
            } else {  // CROARING_RUN
                const auto* rc =
                    static_cast<const run_container_t*>(ra.containers[i]);
                h_types[gid] = ContainerType::RUN;
                h_offs[gid]  = static_cast<uint32_t>(run_cursor * 2u * sizeof(uint16_t));
                // For RUN, cardinalities[] holds n_runs (the iteration bound the
                // device kernels use). True popcount isn't needed by any API
                // function; promote_batch recomputes it during the bitmap fill.
                h_cards[gid] = static_cast<uint16_t>(rc->n_runs);
                std::memcpy(h_run + run_cursor * 2u, rc->runs,
                            static_cast<size_t>(rc->n_runs) * 2u * sizeof(uint16_t));
                run_cursor += static_cast<size_t>(rc->n_runs);
            }
        }
    }
}

void pack_host_meta(char*                          meta,
                    const detail::HostMetaLayout&  M,
                    const std::vector<PerBitmap>&  per,
                    uint32_t                       n,
                    const uint32_t*                cstart,
                    const uint32_t*                kstart)
{
    auto* total_card = reinterpret_cast<uint64_t*>(meta + M.total_card_off);
    auto* universe   = reinterpret_cast<uint32_t*>(meta + M.universe_off);
    auto* cs         = reinterpret_cast<uint32_t*>(meta + M.cstart_off);
    auto* ks         = reinterpret_cast<uint32_t*>(meta + M.kstart_off);
    auto* nbm        = reinterpret_cast<uint32_t*>(meta + M.n_bitmap_off);

    for (uint32_t b = 0; b < n; ++b) {
        total_card[b] = per[b].total_cardinality;
        universe[b]   = per[b].universe_size;
        nbm[b]        = per[b].n_bitmap_containers;
    }
    std::memcpy(cs, cstart, (n + 1u) * sizeof(uint32_t));
    std::memcpy(ks, kstart, (n + 1u) * sizeof(uint32_t));
}

GpuRoaringBatch finalise(char*                         d_buf,
                         const DeviceLayout&           L,
                         char*                         h_meta,
                         const detail::HostMetaLayout& M,
                         uint32_t                      n_bitmaps,
                         uint32_t              total_containers,
                         uint32_t              total_n_bitmap,
                         size_t                total_arr_elems,
                         size_t                total_run_pairs)
{
    GpuRoaringBatch B{};
    B.n_bitmaps                 = n_bitmaps;
    B.total_containers          = total_containers;
    B.n_bitmap_containers_total = total_n_bitmap;
    B.array_pool_bytes          =
        static_cast<uint32_t>(total_arr_elems * sizeof(uint16_t));
    B.run_pool_bytes            =
        static_cast<uint32_t>(total_run_pairs * 2u * sizeof(uint16_t));

    B.container_starts = reinterpret_cast<uint32_t*>(d_buf + L.cstart_off);
    B.key_index_starts = reinterpret_cast<uint32_t*>(d_buf + L.kstart_off);
    B.keys             = reinterpret_cast<uint16_t*>(d_buf + L.keys_off);
    B.types            = reinterpret_cast<ContainerType*>(d_buf + L.types_off);
    B.offsets          = reinterpret_cast<uint32_t*>(d_buf + L.offsets_off);
    B.cardinalities    = reinterpret_cast<uint16_t*>(d_buf + L.cards_off);
    B.key_indices      = reinterpret_cast<uint16_t*>(d_buf + L.kidx_off);
    if (total_n_bitmap > 0) {
        B.bitmap_data = reinterpret_cast<uint64_t*>(d_buf + L.bmp_off);
    }
    if (total_arr_elems > 0) {
        B.array_data = reinterpret_cast<uint16_t*>(d_buf + L.arr_off);
    }
    if (total_run_pairs > 0) {
        B.run_data = reinterpret_cast<uint16_t*>(d_buf + L.run_off);
    }

    B._alloc_base     = d_buf;
    B._host_meta_base = h_meta;

    B.host_total_cardinalities  =
        reinterpret_cast<uint64_t*>(h_meta + M.total_card_off);
    B.host_universe_sizes       =
        reinterpret_cast<uint32_t*>(h_meta + M.universe_off);
    B.host_container_starts     =
        reinterpret_cast<uint32_t*>(h_meta + M.cstart_off);
    B.host_key_index_starts     =
        reinterpret_cast<uint32_t*>(h_meta + M.kstart_off);
    B.host_n_bitmap_containers  =
        reinterpret_cast<uint32_t*>(h_meta + M.n_bitmap_off);
    return B;
}

} // namespace

GpuRoaringBatch upload_batch(const roaring_bitmap_t* const* cpus,
                             uint32_t                       n,
                             cudaStream_t                   stream)
{
    if (n == 0) return GpuRoaringBatch{};
    if (cpus == nullptr) {
        throw std::invalid_argument("upload_batch: cpus is null with n > 0");
    }

    std::vector<PerBitmap> per;
    size_t   total_bmp_words  = 0;
    size_t   total_arr_elems  = 0;
    size_t   total_run_pairs  = 0;
    uint32_t total_containers = 0;
    uint32_t total_n_bitmap   = 0;
    uint64_t total_kidx_len   = 0;
    compute_per_bitmap(cpus, n, per, total_bmp_words, total_arr_elems,
                       total_run_pairs, total_containers, total_n_bitmap,
                       total_kidx_len);

    const DeviceLayout            L = compute_device_layout(n, total_containers,
                                                             total_kidx_len,
                                                             total_bmp_words,
                                                             total_arr_elems,
                                                             total_run_pairs);
    const detail::HostMetaLayout  M = detail::compute_host_meta_layout(n);

    auto h_buf = std::make_unique<char[]>(L.total_bytes);
    std::memset(h_buf.get(), 0, L.total_bytes);
    pack_host_buffer(h_buf.get(), L, per, cpus, n, total_kidx_len);

    auto* h_cstart = reinterpret_cast<const uint32_t*>(h_buf.get() + L.cstart_off);
    auto* h_kstart = reinterpret_cast<const uint32_t*>(h_buf.get() + L.kstart_off);

    auto h_meta_owned = std::make_unique<char[]>(M.total_bytes);
    std::memset(h_meta_owned.get(), 0, M.total_bytes);
    pack_host_meta(h_meta_owned.get(), M, per, n, h_cstart, h_kstart);

    char* d_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_buf),
                                             L.total_bytes, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_buf, h_buf.get(),
                                             L.total_bytes,
                                             cudaMemcpyHostToDevice, stream));
    // Sync once so h_buf can be safely freed when this function returns. Upload
    // is a one-shot ingest path — this is the only D2{H,2D}-bound operation in
    // the public API; everything downstream stays stream-asynchronous.
    CU_ROARING_V2_CHECK_CUDA(cudaStreamSynchronize(stream));

    return finalise(d_buf, L, h_meta_owned.release(), M, n,
                    total_containers, total_n_bitmap,
                    total_arr_elems, total_run_pairs);
}

void free_batch(GpuRoaringBatch& batch) {
    if (batch._alloc_base != nullptr) {
        // Synchronous free: safe even if the original stream has been destroyed.
        cudaFree(batch._alloc_base);
    }
    if (batch._host_meta_base != nullptr) {
        delete[] reinterpret_cast<char*>(batch._host_meta_base);
    }
    batch = GpuRoaringBatch{};
}

GpuRoaringView make_view(const GpuRoaringBatch& batch, uint32_t b) {
    if (b >= batch.n_bitmaps) {
        throw std::out_of_range(
            "make_view: bitmap_idx " + std::to_string(b) +
            " out of range for batch of size " +
            std::to_string(batch.n_bitmaps));
    }
    GpuRoaringView v{};
    if (batch.host_container_starts == nullptr) return v;  // empty batch

    const uint32_t cstart = batch.host_container_starts[b];
    const uint32_t cend   = batch.host_container_starts[b + 1];
    const uint32_t kstart = batch.host_key_index_starts[b];
    const uint32_t kend   = batch.host_key_index_starts[b + 1];

    v.n_containers = cend - cstart;
    v.max_key      = (kend > kstart) ? (kend - kstart - 1u) : 0u;

    if (v.n_containers > 0) {
        v.keys          = batch.keys          + cstart;
        v.types         = batch.types         + cstart;
        v.offsets       = batch.offsets       + cstart;
        v.cardinalities = batch.cardinalities + cstart;
        v.key_index     = batch.key_indices   + kstart;
    }

    // Pools are shared across the batch; offsets[] are absolute byte offsets
    // into them, so the view sees the data each container was packed into at
    // upload time without any redirection.
    v.bitmap_data = batch.bitmap_data;
    v.array_data  = batch.array_data;
    v.run_data    = batch.run_data;
    return v;
}

} // namespace cu_roaring::v2
