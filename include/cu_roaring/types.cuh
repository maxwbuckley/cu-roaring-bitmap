#pragma once
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
