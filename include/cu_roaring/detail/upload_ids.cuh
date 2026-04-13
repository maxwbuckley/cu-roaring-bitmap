#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"
#include "cu_roaring/detail/promote.cuh"

namespace cu_roaring {

// Create GPU Roaring from an array of uint32_t IDs on host.
// Does NOT depend on CRoaring.
//
// IDs do not need to be sorted or deduplicated — the function handles both
// internally. Duplicate IDs are silently ignored.
//
// bitmap_threshold controls array-to-bitmap promotion:
//   PROMOTE_AUTO (default) — query GPU L2 cache size, pick optimal strategy
//   PROMOTE_KEEP_DEFAULT (4096)    — containers with <= 4096 elements use array format
//   PROMOTE_ALL  (0)       — all containers use bitmap format
//   Any value N            — containers with cardinality > N use bitmap format
GpuRoaring upload_from_ids(const uint32_t* ids,
                           uint32_t n_ids,
                           uint32_t universe_size,
                           cudaStream_t stream = 0,
                           uint32_t bitmap_threshold = PROMOTE_AUTO);

// Backwards-compatible alias
inline GpuRoaring upload_from_sorted_ids(const uint32_t* ids,
                                          uint32_t n_ids,
                                          uint32_t universe_size,
                                          cudaStream_t stream = 0,
                                          uint32_t bitmap_threshold = PROMOTE_AUTO)
{
  return upload_from_ids(ids, n_ids, universe_size, stream, bitmap_threshold);
}

// ============================================================================
// Upload from flat bitset — skips sort/dedupe/scatter entirely.
//
// The bitset words ARE the bitmap containers. This path just partitions
// into 65536-bit chunks, identifies non-empty containers, builds metadata,
// and applies the complement optimization if density > 50%.
//
// Accepts uint32_t words (C++ bitset convention): bit i is set if
// (words[i/32] >> (i%32)) & 1. n_words = ceil(universe_size / 32).
//
// Host version: bitset is in host memory, transferred to GPU internally.
// Device version: bitset is already in GPU memory (zero-copy).
// ============================================================================
GpuRoaring upload_from_bitset(const uint32_t* host_bitset,
                               uint32_t n_words,
                               uint32_t universe_size,
                               cudaStream_t stream = 0);

GpuRoaring upload_from_device_bitset(const uint32_t* d_bitset,
                                      uint32_t n_words,
                                      uint32_t universe_size,
                                      cudaStream_t stream = 0);

}  // namespace cu_roaring
