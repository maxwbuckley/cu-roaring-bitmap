#pragma once
#include <cstddef>
#include <cstdint>

namespace cu_roaring {

enum class ContainerType : uint8_t {
    ARRAY  = 0,
    BITMAP = 1,
    RUN    = 2
};

// GPU-resident Roaring bitmap in Structure-of-Arrays layout.
// All pointers are device pointers allocated with cudaMalloc.
struct GpuRoaring {
    // Top-level index (one entry per container, sorted by key)
    uint16_t*      keys         = nullptr;  // [n_containers] high-16-bit keys
    ContainerType* types        = nullptr;  // [n_containers] container type tag
    uint32_t*      offsets      = nullptr;  // [n_containers] byte offset into per-type data pool
    uint16_t*      cardinalities = nullptr; // [n_containers] element count
    uint32_t       n_containers      = 0;
    uint32_t       universe_size     = 0;   // max representable ID + 1
    uint64_t       total_cardinality = 0;   // total number of set bits

    // Per-type data pools (contiguous, type-homogeneous)
    uint64_t* bitmap_data        = nullptr; // [n_bitmap_containers * 1024]
    uint32_t  n_bitmap_containers = 0;

    uint16_t* array_data         = nullptr; // [total_array_elements]
    uint32_t  n_array_containers  = 0;

    uint16_t* run_data           = nullptr; // [total_run_pairs * 2] packed (start, length)
    uint32_t  n_run_containers    = 0;

    // Direct-map key index: key_index[high16] = container index, or 0xFFFF.
    // Replaces O(log n) binary search with O(1) table lookup.
    // Built automatically during upload. Memory: (max_key + 1) * 2 bytes.
    uint16_t* key_index          = nullptr; // [max_key + 1] or nullptr
    uint32_t  max_key            = 0;       // highest container key value

    // Complement optimization: when true, the stored set is the complement
    // of the logical set. contains() results are flipped at query time.
    // This guarantees the stored bitmap always has density <= 50%, making
    // Roaring compression symmetric around 50% (e.g., a 99% pass-rate
    // filter stores only the 1% rejects → 59x compression).
    bool      negated            = false;

    // Internal: base of a single packed device allocation. When non-null,
    // all device pointers above are offsets into this block, and
    // gpu_roaring_free() frees only this pointer.
    void*     _alloc_base        = nullptr;
};

struct GpuRoaringMeta {
    uint32_t n_containers         = 0;
    uint32_t n_bitmap_containers  = 0;
    uint32_t n_array_containers   = 0;
    uint32_t n_run_containers     = 0;
    uint32_t universe_size        = 0;
    size_t   total_bytes          = 0;
};

}  // namespace cu_roaring
