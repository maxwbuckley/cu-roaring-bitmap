#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <roaring/roaring.h>
#include <roaring/containers/array.h>
#include <roaring/containers/bitset.h>
#include <roaring/containers/run.h>

#include <cstring>
#include <memory>
#include <stdexcept>

// CRoaring v4.x exposes container internals under roaring::internal.
using roaring::internal::array_container_t;
using roaring::internal::bitset_container_t;
using roaring::internal::run_container_t;

namespace cu_roaring::v2 {

namespace {

// CRoaring internal type codes (from roaring/misc/configreport.h).
constexpr uint8_t CROARING_BITSET = 1;
constexpr uint8_t CROARING_ARRAY  = 2;
constexpr uint8_t CROARING_RUN    = 3;

struct Layout {
    size_t keys_off    = 0;
    size_t types_off   = 0;
    size_t offsets_off = 0;
    size_t cards_off   = 0;
    size_t kidx_off    = 0;
    size_t bmp_off     = 0;
    size_t arr_off     = 0;
    size_t run_off     = 0;
    size_t total_bytes = 0;

    uint32_t n                  = 0;
    uint32_t n_bitmap           = 0;
    uint32_t n_array            = 0;
    uint32_t n_run              = 0;
    uint32_t key_index_len      = 0;
    uint16_t max_key            = 0;
    uint32_t universe_size      = 0;
    size_t   total_bitmap_words = 0;
    size_t   total_array_elems  = 0;
    size_t   total_run_pairs    = 0;
};

Layout compute_layout(const roaring_bitmap_t* cpu) {
    using detail::align_up;
    const roaring_array_t& ra = cpu->high_low_container;
    Layout L;
    L.n = static_cast<uint32_t>(ra.size);

    L.max_key       = ra.keys[ra.size - 1];
    L.universe_size = (static_cast<uint32_t>(L.max_key) + 1u) << 16;
    L.key_index_len = static_cast<uint32_t>(L.max_key) + 1u;

    for (uint32_t i = 0; i < L.n; ++i) {
        const uint8_t tc = ra.typecodes[i];
        if (tc == CROARING_BITSET) {
            ++L.n_bitmap;
            L.total_bitmap_words += 1024;
        } else if (tc == CROARING_ARRAY) {
            ++L.n_array;
            L.total_array_elems +=
                static_cast<size_t>(
                    static_cast<const array_container_t*>(ra.containers[i])->cardinality);
        } else if (tc == CROARING_RUN) {
            ++L.n_run;
            L.total_run_pairs +=
                static_cast<size_t>(
                    static_cast<const run_container_t*>(ra.containers[i])->n_runs);
        }
    }

    constexpr size_t A = 8;
    size_t off = 0;
    L.keys_off    = off; off = align_up(off + L.n * sizeof(uint16_t), A);
    L.types_off   = off; off = align_up(off + L.n * sizeof(ContainerType), A);
    L.offsets_off = off; off = align_up(off + L.n * sizeof(uint32_t), A);
    L.cards_off   = off; off = align_up(off + L.n * sizeof(uint16_t), A);
    L.kidx_off    = off; off = align_up(off + L.key_index_len * sizeof(uint16_t), A);
    L.bmp_off     = off; off += L.total_bitmap_words * sizeof(uint64_t);
    L.arr_off     = off; off = align_up(off + L.total_array_elems * sizeof(uint16_t), A);
    L.run_off     = off; off += L.total_run_pairs * 2u * sizeof(uint16_t);
    L.total_bytes = align_up(off, A);
    if (L.total_bytes == 0) L.total_bytes = A;
    return L;
}

void pack_host_buffer(char* buf, const Layout& L, const roaring_bitmap_t* cpu) {
    const roaring_array_t& ra = cpu->high_low_container;

    auto* h_keys  = reinterpret_cast<uint16_t*>(buf + L.keys_off);
    auto* h_types = reinterpret_cast<ContainerType*>(buf + L.types_off);
    auto* h_offs  = reinterpret_cast<uint32_t*>(buf + L.offsets_off);
    auto* h_cards = reinterpret_cast<uint16_t*>(buf + L.cards_off);
    auto* h_kidx  = reinterpret_cast<uint16_t*>(buf + L.kidx_off);
    auto* h_bmp   = reinterpret_cast<uint64_t*>(buf + L.bmp_off);
    auto* h_arr   = reinterpret_cast<uint16_t*>(buf + L.arr_off);
    auto* h_run   = reinterpret_cast<uint16_t*>(buf + L.run_off);

    std::memset(h_kidx, 0xFF, L.key_index_len * sizeof(uint16_t));

    size_t bmp_cursor = 0, arr_cursor = 0, run_cursor = 0;
    for (uint32_t i = 0; i < L.n; ++i) {
        h_keys[i] = ra.keys[i];
        h_kidx[ra.keys[i]] = static_cast<uint16_t>(i);
        const uint8_t tc = ra.typecodes[i];

        if (tc == CROARING_BITSET) {
            const auto* bc = static_cast<const bitset_container_t*>(ra.containers[i]);
            h_types[i] = ContainerType::BITMAP;
            h_offs[i]  = static_cast<uint32_t>(bmp_cursor * sizeof(uint64_t));
            // CRoaring stores cardinality=-1 for the all-ones case (bc->cardinality
            // is int32_t). The struct uses 0 as the "full or empty" sentinel; callers
            // disambiguate by checking n_containers and popcount if they care.
            const int32_t card = bc->cardinality;
            h_cards[i] = static_cast<uint16_t>(card > 65535 || card < 0 ? 0 : card);
            std::memcpy(h_bmp + bmp_cursor, bc->words, 1024u * sizeof(uint64_t));
            bmp_cursor += 1024;
        } else if (tc == CROARING_ARRAY) {
            const auto* ac = static_cast<const array_container_t*>(ra.containers[i]);
            h_types[i] = ContainerType::ARRAY;
            h_offs[i]  = static_cast<uint32_t>(arr_cursor * sizeof(uint16_t));
            h_cards[i] = static_cast<uint16_t>(ac->cardinality);
            std::memcpy(h_arr + arr_cursor, ac->array,
                        static_cast<size_t>(ac->cardinality) * sizeof(uint16_t));
            arr_cursor += static_cast<size_t>(ac->cardinality);
        } else if (tc == CROARING_RUN) {
            const auto* rc = static_cast<const run_container_t*>(ra.containers[i]);
            h_types[i] = ContainerType::RUN;
            h_offs[i]  = static_cast<uint32_t>(run_cursor * 2u * sizeof(uint16_t));
            h_cards[i] = static_cast<uint16_t>(rc->n_runs);
            // rle16_t is two uint16s: {value, length}. Layout-compatible with our
            // (start, length) pair packing.
            std::memcpy(h_run + run_cursor * 2u, rc->runs,
                        static_cast<size_t>(rc->n_runs) * 2u * sizeof(uint16_t));
            run_cursor += static_cast<size_t>(rc->n_runs);
        } else {
            throw std::runtime_error("upload_from_croaring: unknown CRoaring typecode");
        }
    }
}

GpuRoaring build_result(char* d_buf, const Layout& L, uint64_t total_card) {
    GpuRoaring r{};
    r._alloc_base         = d_buf;
    r.n_containers        = L.n;
    r.n_bitmap_containers = L.n_bitmap;
    r.n_array_containers  = L.n_array;
    r.n_run_containers    = L.n_run;
    r.universe_size       = L.universe_size;
    r.max_key             = L.max_key;
    r.total_cardinality   = total_card;

    r.keys          = reinterpret_cast<uint16_t*>(d_buf + L.keys_off);
    r.types         = reinterpret_cast<ContainerType*>(d_buf + L.types_off);
    r.offsets       = reinterpret_cast<uint32_t*>(d_buf + L.offsets_off);
    r.cardinalities = reinterpret_cast<uint16_t*>(d_buf + L.cards_off);
    r.key_index     = reinterpret_cast<uint16_t*>(d_buf + L.kidx_off);

    if (L.total_bitmap_words > 0) {
        r.bitmap_data = reinterpret_cast<uint64_t*>(d_buf + L.bmp_off);
    }
    if (L.total_array_elems > 0) {
        r.array_data       = reinterpret_cast<uint16_t*>(d_buf + L.arr_off);
        r.array_pool_bytes = static_cast<uint32_t>(L.total_array_elems * sizeof(uint16_t));
    }
    if (L.total_run_pairs > 0) {
        r.run_data       = reinterpret_cast<uint16_t*>(d_buf + L.run_off);
        r.run_pool_bytes = static_cast<uint32_t>(L.total_run_pairs * 2u * sizeof(uint16_t));
    }
    return r;
}

} // namespace

GpuRoaring upload_from_croaring(const roaring_bitmap_t* cpu, cudaStream_t stream) {
    if (cpu == nullptr) {
        throw std::invalid_argument("upload_from_croaring: cpu is null");
    }

    const roaring_array_t& ra = cpu->high_low_container;
    if (ra.size == 0) {
        return GpuRoaring{};
    }

    const Layout L = compute_layout(cpu);

    // Pageable host staging. cudaMemcpyAsync does the pinning internally via the
    // driver; for one-shot uploads that's within a few percent of an explicit
    // cudaMallocHost path and saves the pinned-alloc round-trip.
    auto h_buf = std::make_unique<char[]>(L.total_bytes);
    std::memset(h_buf.get(), 0, L.total_bytes);
    pack_host_buffer(h_buf.get(), L, cpu);

    char* d_buf = nullptr;
    CU_ROARING_V2_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_buf),
                                             L.total_bytes, stream));
    CU_ROARING_V2_CHECK_CUDA(cudaMemcpyAsync(d_buf, h_buf.get(), L.total_bytes,
                                             cudaMemcpyHostToDevice, stream));
    // Sync so that h_buf (unique_ptr) can be safely freed as this function returns.
    CU_ROARING_V2_CHECK_CUDA(cudaStreamSynchronize(stream));

    return build_result(d_buf, L, roaring_bitmap_get_cardinality(cpu));
}

void free_bitmap(GpuRoaring& bm) {
    if (bm._alloc_base != nullptr) {
        cudaFree(bm._alloc_base);
    }
    bm = GpuRoaring{};
}

} // namespace cu_roaring::v2
