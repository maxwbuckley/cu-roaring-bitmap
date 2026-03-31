#include "cu_roaring/detail/upload.cuh"
#include "cu_roaring/detail/utils.cuh"
#include "cu_roaring/upload_pool.hpp"

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

static size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

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

// ============================================================================
// Shared packing logic — used by both upload_impl() and UploadPool::upload()
// ============================================================================

struct PackedLayout {
    size_t keys_off, types_off, offsets_off, cards_off, kidx_off;
    size_t bmp_off, arr_off, run_off;
    size_t total_bytes;
    uint32_t n, n_bitmap, n_array, n_run;
    uint32_t key_index_len;
    uint16_t max_key;
    size_t total_bitmap_words, total_array_elems, total_run_pairs;
    uint32_t universe_size;
    bool promote_all;
};

static PackedLayout compute_layout(const roaring_bitmap_t* cpu_bitmap,
                                   uint32_t explicit_universe_size,
                                   uint32_t bitmap_threshold) {
    const roaring_array_t& ra = cpu_bitmap->high_low_container;
    PackedLayout L{};
    L.n = static_cast<uint32_t>(ra.size);

    L.universe_size = explicit_universe_size;
    if (L.universe_size == 0) {
        uint16_t mk = ra.keys[ra.size - 1];
        L.universe_size = (static_cast<uint32_t>(mk) + 1) << 16;
    }

    uint32_t eff = bitmap_threshold;
    if (eff == PROMOTE_AUTO) eff = resolve_auto_threshold(L.universe_size);
    L.promote_all = (eff < PROMOTE_NONE);

    // Guard: skip promotion when it would eliminate compression.
    // Only applies to PROMOTE_AUTO — an explicit PROMOTE_ALL is respected.
    // If n_containers × 8 KB >= half the flat bitset, the promoted Roaring
    // uses as much memory as the bitset with none of the query benefit.
    // Benchmark (1B universe, uniform/power_law): 15 K containers × 8 KB =
    // 125 MB = same as bitset, 28-47× slower upload, identical query speed.
    if (L.promote_all && bitmap_threshold == PROMOTE_AUTO) {
        size_t promoted_bytes = static_cast<size_t>(L.n) * 1024 * sizeof(uint64_t);
        size_t bitset_bytes   = (static_cast<size_t>(L.universe_size) + 7) / 8;
        if (promoted_bytes * 2 >= bitset_bytes) {
            L.promote_all = false;
        }
    }

    for (uint32_t i = 0; i < L.n; ++i) {
        uint8_t tc = ra.typecodes[i];
        if (tc == CROARING_BITSET) {
            ++L.n_bitmap; L.total_bitmap_words += 1024;
        } else if (tc == CROARING_ARRAY) {
            if (L.promote_all) { L.total_bitmap_words += 1024; }
            else { ++L.n_array; L.total_array_elems +=
                static_cast<const array_container_t*>(ra.containers[i])->cardinality; }
        } else if (tc == CROARING_RUN) {
            if (L.promote_all) { L.total_bitmap_words += 1024; }
            else { ++L.n_run; L.total_run_pairs +=
                static_cast<const run_container_t*>(ra.containers[i])->n_runs; }
        }
    }
    if (L.promote_all) { L.n_bitmap = L.n; L.n_array = 0; L.n_run = 0; }

    L.max_key = ra.keys[ra.size - 1];
    L.key_index_len = static_cast<uint32_t>(L.max_key) + 1;

    constexpr size_t A = 8;
    size_t off = 0;
    L.keys_off    = off; off = align_up(off + L.n * sizeof(uint16_t), A);
    L.types_off   = off; off = align_up(off + L.n * sizeof(ContainerType), A);
    L.offsets_off = off; off = align_up(off + L.n * sizeof(uint32_t), A);
    L.cards_off   = off; off = align_up(off + L.n * sizeof(uint16_t), A);
    L.kidx_off    = off; off = align_up(off + L.key_index_len * sizeof(uint16_t), A);
    L.bmp_off     = off; off += L.total_bitmap_words * sizeof(uint64_t);
    L.arr_off     = off; off = align_up(off + L.total_array_elems * sizeof(uint16_t), A);
    L.run_off     = off; off += L.total_run_pairs * 2 * sizeof(uint16_t);
    L.total_bytes = align_up(off, A);
    if (L.total_bytes == 0) L.total_bytes = A;
    return L;
}

// Pack CRoaring data into h_buf according to layout L.
static void pack_into_buffer(char* h_buf, const PackedLayout& L,
                             const roaring_bitmap_t* cpu_bitmap) {
    const roaring_array_t& ra = cpu_bitmap->high_low_container;

    auto* h_keys  = reinterpret_cast<uint16_t*>(h_buf + L.keys_off);
    auto* h_types = reinterpret_cast<ContainerType*>(h_buf + L.types_off);
    auto* h_offs  = reinterpret_cast<uint32_t*>(h_buf + L.offsets_off);
    auto* h_cards = reinterpret_cast<uint16_t*>(h_buf + L.cards_off);
    auto* h_kidx  = reinterpret_cast<uint16_t*>(h_buf + L.kidx_off);
    auto* h_bmp   = reinterpret_cast<uint64_t*>(h_buf + L.bmp_off);
    auto* h_arr   = reinterpret_cast<uint16_t*>(h_buf + L.arr_off);
    auto* h_run   = reinterpret_cast<uint16_t*>(h_buf + L.run_off);

    if (L.promote_all && L.total_bitmap_words > 0)
        std::memset(h_bmp, 0, L.total_bitmap_words * sizeof(uint64_t));
    std::memset(h_kidx, 0xFF, L.key_index_len * sizeof(uint16_t));

    size_t bmp_cursor = 0, arr_cursor = 0, run_cursor = 0;
    for (uint32_t i = 0; i < L.n; ++i) {
        h_keys[i] = ra.keys[i];
        h_kidx[ra.keys[i]] = static_cast<uint16_t>(i);
        uint8_t tc = ra.typecodes[i];

        if (L.promote_all) {
            h_types[i] = ContainerType::BITMAP;
            h_offs[i]  = static_cast<uint32_t>(bmp_cursor * sizeof(uint64_t));
            uint64_t* dst = h_bmp + bmp_cursor;
            if (tc == CROARING_BITSET) {
                const auto* bc = static_cast<const bitset_container_t*>(ra.containers[i]);
                h_cards[i] = static_cast<uint16_t>(bc->cardinality > 65535 ? 0 : bc->cardinality);
                std::memcpy(dst, bc->words, 1024 * sizeof(uint64_t));
            } else if (tc == CROARING_ARRAY) {
                const auto* ac = static_cast<const array_container_t*>(ra.containers[i]);
                h_cards[i] = static_cast<uint16_t>(ac->cardinality);
                for (int32_t j = 0; j < ac->cardinality; ++j) {
                    uint16_t val = ac->array[j]; dst[val/64] |= 1ULL << (val%64); }
            } else if (tc == CROARING_RUN) {
                const auto* rc = static_cast<const run_container_t*>(ra.containers[i]);
                h_cards[i] = static_cast<uint16_t>(rc->n_runs);
                for (int32_t r = 0; r < rc->n_runs; ++r) {
                    uint16_t start = rc->runs[r].value; uint16_t len = rc->runs[r].length;
                    for (uint32_t v = start; v <= static_cast<uint32_t>(start)+len; ++v)
                        dst[v/64] |= 1ULL << (v%64); }
            }
            bmp_cursor += 1024;
        } else {
            if (tc == CROARING_BITSET) {
                h_types[i] = ContainerType::BITMAP;
                h_offs[i]  = static_cast<uint32_t>(bmp_cursor * sizeof(uint64_t));
                const auto* bc = static_cast<const bitset_container_t*>(ra.containers[i]);
                h_cards[i] = static_cast<uint16_t>(bc->cardinality > 65535 ? 0 : bc->cardinality);
                std::memcpy(h_bmp + bmp_cursor, bc->words, 1024 * sizeof(uint64_t));
                bmp_cursor += 1024;
            } else if (tc == CROARING_ARRAY) {
                h_types[i] = ContainerType::ARRAY;
                h_offs[i]  = static_cast<uint32_t>(arr_cursor * sizeof(uint16_t));
                const auto* ac = static_cast<const array_container_t*>(ra.containers[i]);
                h_cards[i] = static_cast<uint16_t>(ac->cardinality);
                std::memcpy(h_arr + arr_cursor, ac->array, ac->cardinality * sizeof(uint16_t));
                arr_cursor += ac->cardinality;
            } else if (tc == CROARING_RUN) {
                h_types[i] = ContainerType::RUN;
                h_offs[i]  = static_cast<uint32_t>(run_cursor * 2 * sizeof(uint16_t));
                const auto* rc = static_cast<const run_container_t*>(ra.containers[i]);
                h_cards[i] = static_cast<uint16_t>(rc->n_runs);
                std::memcpy(h_run + run_cursor*2, rc->runs, rc->n_runs * sizeof(rle16_t));
                run_cursor += rc->n_runs;
            }
        }
    }
}

// Build a GpuRoaring pointing into d_buf at the offsets described by L.
static GpuRoaring build_result(char* d_buf, const PackedLayout& L,
                               const roaring_bitmap_t* cpu_bitmap,
                               void* alloc_base) {
    GpuRoaring result{};
    result._alloc_base         = alloc_base;
    result.n_containers        = L.n;
    result.n_bitmap_containers = L.n_bitmap;
    result.n_array_containers  = L.n_array;
    result.n_run_containers    = L.n_run;
    result.universe_size       = L.universe_size;
    result.total_cardinality   = roaring_bitmap_get_cardinality(cpu_bitmap);
    result.keys          = reinterpret_cast<uint16_t*>(d_buf + L.keys_off);
    result.types         = reinterpret_cast<ContainerType*>(d_buf + L.types_off);
    result.offsets       = reinterpret_cast<uint32_t*>(d_buf + L.offsets_off);
    result.cardinalities = reinterpret_cast<uint16_t*>(d_buf + L.cards_off);
    result.max_key       = L.max_key;
    result.key_index     = reinterpret_cast<uint16_t*>(d_buf + L.kidx_off);
    if (L.total_bitmap_words > 0)
        result.bitmap_data = reinterpret_cast<uint64_t*>(d_buf + L.bmp_off);
    if (L.total_array_elems > 0)
        result.array_data = reinterpret_cast<uint16_t*>(d_buf + L.arr_off);
    if (L.total_run_pairs > 0)
        result.run_data = reinterpret_cast<uint16_t*>(d_buf + L.run_off);
    return result;
}

// ============================================================================
// upload_impl: allocating path (cudaMallocHost + cudaMalloc per call)
// ============================================================================
static GpuRoaring upload_impl(const roaring_bitmap_t* cpu_bitmap,
                                uint32_t explicit_universe_size,
                                cudaStream_t stream,
                                uint32_t bitmap_threshold) {
    const roaring_array_t& ra = cpu_bitmap->high_low_container;
    if (ra.size == 0) {
        GpuRoaring result{};
        result.universe_size = explicit_universe_size;
        return result;
    }

    auto L = compute_layout(cpu_bitmap, explicit_universe_size, bitmap_threshold);

    char* h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, L.total_bytes));
    pack_into_buffer(h_buf, L, cpu_bitmap);

    char* d_buf = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_buf, L.total_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, L.total_bytes,
                               cudaMemcpyHostToDevice, stream));
    cudaFreeHost(h_buf);

    return build_result(d_buf, L, cpu_bitmap, d_buf);
}

// Original overload: infer universe from max key, no auto-complement
GpuRoaring upload(const roaring_bitmap_t* cpu_bitmap, cudaStream_t stream,
                  uint32_t bitmap_threshold) {
    return upload_impl(cpu_bitmap, 0, stream, bitmap_threshold);
}

// New overload with explicit universe_size: enables auto-complement
GpuRoaring upload(const roaring_bitmap_t* cpu_bitmap,
                  uint32_t universe_size,
                  cudaStream_t stream,
                  uint32_t bitmap_threshold) {
    uint64_t card = roaring_bitmap_get_cardinality(cpu_bitmap);

    if (card > static_cast<uint64_t>(universe_size) / 2) {
        // Complement: flip the bitmap, upload that, set negated=true
        roaring_bitmap_t* complement = roaring_bitmap_copy(cpu_bitmap);
        roaring_bitmap_flip_inplace(complement, 0, static_cast<uint64_t>(universe_size));

        GpuRoaring result = upload_impl(complement, universe_size, stream, bitmap_threshold);
        result.negated = true;
        // total_cardinality is the logical cardinality (original set, not complement)
        result.total_cardinality = card;
        roaring_bitmap_free(complement);
        return result;
    }

    return upload_impl(cpu_bitmap, universe_size, stream, bitmap_threshold);
}

void gpu_roaring_free(GpuRoaring& bitmap) {
    if (bitmap._alloc_base) {
        // Packed allocation: all device pointers are offsets into one block
        cudaFree(bitmap._alloc_base);
    } else {
        // Individual allocations (legacy path, upload_from_ids, set_ops, etc.)
        if (bitmap.keys)          cudaFree(bitmap.keys);
        if (bitmap.types)         cudaFree(bitmap.types);
        if (bitmap.offsets)       cudaFree(bitmap.offsets);
        if (bitmap.cardinalities) cudaFree(bitmap.cardinalities);
        if (bitmap.bitmap_data)   cudaFree(bitmap.bitmap_data);
        if (bitmap.array_data)    cudaFree(bitmap.array_data);
        if (bitmap.run_data)      cudaFree(bitmap.run_data);
        if (bitmap.key_index)     cudaFree(bitmap.key_index);
    }
    bitmap = GpuRoaring{};
}

void gpu_roaring_free_async(GpuRoaring& bitmap, cudaStream_t stream) {
    if (bitmap._alloc_base) {
        CUDA_CHECK(cudaFreeAsync(bitmap._alloc_base, stream));
    } else {
        if (bitmap.keys)          CUDA_CHECK(cudaFreeAsync(bitmap.keys, stream));
        if (bitmap.types)         CUDA_CHECK(cudaFreeAsync(bitmap.types, stream));
        if (bitmap.offsets)       CUDA_CHECK(cudaFreeAsync(bitmap.offsets, stream));
        if (bitmap.cardinalities) CUDA_CHECK(cudaFreeAsync(bitmap.cardinalities, stream));
        if (bitmap.bitmap_data)   CUDA_CHECK(cudaFreeAsync(bitmap.bitmap_data, stream));
        if (bitmap.array_data)    CUDA_CHECK(cudaFreeAsync(bitmap.array_data, stream));
        if (bitmap.run_data)      CUDA_CHECK(cudaFreeAsync(bitmap.run_data, stream));
        if (bitmap.key_index)     CUDA_CHECK(cudaFreeAsync(bitmap.key_index, stream));
    }
    bitmap = GpuRoaring{};
}

// ============================================================================
// UploadPool — zero-allocation upload path
// ============================================================================

UploadPool::UploadPool(size_t capacity_bytes, cudaStream_t /*stream*/)
  : capacity_(capacity_bytes)
{
    CUDA_CHECK(cudaMallocHost(&h_pinned_, capacity_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf_, capacity_bytes));
}

UploadPool::~UploadPool()
{
    if (h_pinned_) cudaFreeHost(h_pinned_);
    if (d_buf_)    cudaFree(d_buf_);
}

UploadPool::UploadPool(UploadPool&& o) noexcept
  : h_pinned_(o.h_pinned_), d_buf_(o.d_buf_),
    capacity_(o.capacity_), last_bytes_(o.last_bytes_)
{
    o.h_pinned_ = nullptr; o.d_buf_ = nullptr;
    o.capacity_ = 0; o.last_bytes_ = 0;
}

UploadPool& UploadPool::operator=(UploadPool&& o) noexcept
{
    if (this != &o) {
        if (h_pinned_) cudaFreeHost(h_pinned_);
        if (d_buf_)    cudaFree(d_buf_);
        h_pinned_ = o.h_pinned_; d_buf_ = o.d_buf_;
        capacity_ = o.capacity_; last_bytes_ = o.last_bytes_;
        o.h_pinned_ = nullptr; o.d_buf_ = nullptr;
        o.capacity_ = 0; o.last_bytes_ = 0;
    }
    return *this;
}

GpuRoaring UploadPool::upload(const roaring_bitmap_t* cpu_bitmap,
                              uint32_t universe_size,
                              cudaStream_t stream,
                              uint32_t bitmap_threshold)
{
    const roaring_array_t& ra = cpu_bitmap->high_low_container;
    if (ra.size == 0) {
        GpuRoaring result{};
        result.universe_size = universe_size;
        last_bytes_ = 0;
        return result;
    }

    // Handle complement (density > 50%)
    uint64_t card = roaring_bitmap_get_cardinality(cpu_bitmap);
    if (universe_size > 0 && card > static_cast<uint64_t>(universe_size) / 2) {
        roaring_bitmap_t* complement = roaring_bitmap_copy(cpu_bitmap);
        roaring_bitmap_flip_inplace(complement, 0, static_cast<uint64_t>(universe_size));
        GpuRoaring result = this->upload(complement, universe_size, stream, bitmap_threshold);
        result.negated = true;
        result.total_cardinality = card;
        roaring_bitmap_free(complement);
        return result;
    }

    auto L = compute_layout(cpu_bitmap, universe_size, bitmap_threshold);
    last_bytes_ = L.total_bytes;

    // Fall back to allocating path if too large for pool
    if (L.total_bytes > capacity_) {
        return upload_impl(cpu_bitmap, universe_size, stream, bitmap_threshold);
    }

    // Pack into pre-allocated pinned buffer, single memcpy to pre-allocated device buffer
    pack_into_buffer(h_pinned_, L, cpu_bitmap);
    CUDA_CHECK(cudaMemcpyAsync(d_buf_, h_pinned_, L.total_bytes,
                               cudaMemcpyHostToDevice, stream));

    // Build result — _alloc_base = nullptr so gpu_roaring_free() is a no-op
    // (pool owns the memory)
    return build_result(d_buf_, L, cpu_bitmap, nullptr);
}

}  // namespace cu_roaring
