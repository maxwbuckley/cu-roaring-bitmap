#pragma once
#include <cstddef>
#include <cstdint>

namespace cu_roaring::v2 {

enum class ContainerType : uint8_t {
    ARRAY  = 0,
    BITMAP = 1,
    RUN    = 2
};

// GPU-resident batch of N Roaring bitmaps, packed into a single device
// allocation. v2's primary type. Every operation in the API takes or returns a
// batch; n=1 is a degenerate case that still flows through the batch code path.
//
// CSR-style layout so each bitmap's data is contiguous:
//
//   container_starts[b] .. container_starts[b+1]
//       → the slice of containers owned by bitmap b
//   key_index_starts[b] .. key_index_starts[b+1]
//       → the slice of `key_indices` owned by bitmap b (length =
//         max_key_of_b + 1, or 0 if the bitmap is empty)
//
//   keys / types / offsets / cardinalities
//       → per-container metadata, concatenated across all bitmaps in
//         bitmap-then-key order, indexed by GLOBAL container id.
//
//   key_indices[key_index_starts[b] + high16]
//       → LOCAL container index within bitmap b (0 .. bitmap's n_containers)
//         or 0xFFFF for absent. Add container_starts[b] to get the global id.
//
//   bitmap_data / array_data / run_data
//       → per-type pools SHARED across the batch; offsets[cid] is a byte
//         offset into the pool selected by types[cid].
//
// All pointers below are device pointers. Every allocation lives inside the
// single block pointed to by _alloc_base. Per-bitmap scalars (total_cardinality,
// universe_size) are mirrored in host-side arrays so callers can read them
// without a D2H.
struct GpuRoaringBatch {
    uint32_t n_bitmaps                 = 0;
    uint32_t total_containers          = 0;
    uint32_t n_bitmap_containers_total = 0;  // # BITMAP-type containers in the batch
    uint32_t array_pool_bytes          = 0;
    uint32_t run_pool_bytes            = 0;

    // CSR indices, both length n_bitmaps + 1.
    uint32_t* container_starts = nullptr;
    uint32_t* key_index_starts = nullptr;

    // Concatenated per-container metadata, indexed by global container id.
    uint16_t*      keys          = nullptr;
    ContainerType* types         = nullptr;
    uint32_t*      offsets       = nullptr;
    uint16_t*      cardinalities = nullptr;

    // Concatenated direct-map key indices. Per-bitmap slices hold LOCAL cids.
    uint16_t* key_indices = nullptr;

    // Shared per-type data pools.
    uint64_t* bitmap_data = nullptr;
    uint16_t* array_data  = nullptr;
    uint16_t* run_data    = nullptr;

    // Host-side mirrors of per-bitmap scalars. Readable without a D2H; never
    // dereferenced from device. All point into _host_meta_base.
    //
    //   host_total_cardinalities  : length n_bitmaps          (popcount per bitmap)
    //   host_universe_sizes       : length n_bitmaps          ((max_key+1) << 16)
    //   host_container_starts     : length n_bitmaps + 1      (mirror of device CSR)
    //   host_key_index_starts     : length n_bitmaps + 1      (mirror of device CSR)
    //   host_n_bitmap_containers  : length n_bitmaps          (per-bitmap BITMAP-type count;
    //                                                          used to validate "all-bitmap"
    //                                                          inputs on the host without
    //                                                          touching device memory)
    const uint64_t* host_total_cardinalities = nullptr;
    const uint32_t* host_universe_sizes      = nullptr;
    const uint32_t* host_container_starts    = nullptr;
    const uint32_t* host_key_index_starts    = nullptr;
    const uint32_t* host_n_bitmap_containers = nullptr;

    // Ownership. free_batch() releases both blocks and nulls every field.
    void* _alloc_base     = nullptr;  // device
    void* _host_meta_base = nullptr;  // host
};

// Thin POD view onto one bitmap within a batch. Extracted on the host via
// make_view() and passed by value to kernels. Pointers alias the batch's pools
// and are valid until free_batch() is called on the underlying batch.
//
// Use a view when a warp / block / kernel focuses on one specific bitmap: the
// hot contains() path becomes 2 reads (key_index + bitmap word), matching a
// single-bitmap Roaring. Falling through to contains(batch, b, id) instead
// costs 2 extra reads per call for the CSR slice bounds.
struct GpuRoaringView {
    uint32_t       n_containers = 0;
    uint32_t       max_key      = 0;  // highest high-16 key present; key_index length = max_key + 1

    uint16_t*      keys          = nullptr;
    ContainerType* types         = nullptr;
    uint32_t*      offsets       = nullptr;
    uint16_t*      cardinalities = nullptr;
    uint16_t*      key_index     = nullptr;

    uint64_t* bitmap_data = nullptr;
    uint16_t* array_data  = nullptr;
    uint16_t* run_data    = nullptr;
};

} // namespace cu_roaring::v2
