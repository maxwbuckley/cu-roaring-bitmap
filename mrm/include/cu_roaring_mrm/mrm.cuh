/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Masked Roaring Matrix (MRM): the transpose-with-payload of k roaring
 * bitmaps. One roaring-shaped structure keyed by row id whose per-id value
 * is a k-bit query-membership mask (k <= 64, mask_t = uint64_t).
 *
 * Per 64K chunk the merged container is one of:
 *   ARRAY_MASKED  — sorted uint16 ids[n] + uint64 masks[n] (SoA), n < 4096
 *   BITMAP_MASKED — 8 KB union bitmap + uint64 masks[n] indexed by rank
 *   RUN_MASKED    — uint16 starts[n] + uint16 lens[n] + uint64 masks[n];
 *                   mask constant per run (len = run length - 1, CRoaring
 *                   convention: run covers [start, start+len])
 *
 * Construction is a single kernel launch with no device->host syncs:
 * capacity comes from host-side per-container cardinality bounds (known at
 * filter construction), not from count kernels. Exact union cardinalities
 * are written into device-side descriptors and never required on the host
 * for search.
 */
#pragma once

#include <cu_roaring/types.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace cu_roaring::mrm {

using mask_t = uint64_t;
static constexpr uint32_t kMaxLanes = 64;

enum class MrmContainerType : uint8_t {
  ARRAY_MASKED  = 0,
  BITMAP_MASKED = 1,
  RUN_MASKED    = 2,
};

// Device-side per-chunk descriptor (written by the construction kernel).
struct MrmContainerDesc {
  uint16_t key;           // high 16 bits of row ids in this chunk
  uint8_t type;           // MrmContainerType
  uint8_t _pad;
  uint32_t n;             // ARRAY/BITMAP: union cardinality; RUN: run count
  uint64_t byte_offset;   // offset of this chunk's payload within the pool
};

// Plain device view (pass by value to kernels).
struct MrmView {
  const MrmContainerDesc* descs;  // [n_chunks], ascending key order
  const uint8_t* pool;            // payload pool
  uint32_t n_chunks;
  uint32_t n_lanes;               // k

  // Payload accessors -------------------------------------------------------
  __host__ __device__ inline const uint16_t* array_ids(const MrmContainerDesc& d) const
  {
    return reinterpret_cast<const uint16_t*>(pool + d.byte_offset);
  }
  __host__ __device__ inline const mask_t* array_masks(const MrmContainerDesc& d) const
  {
    uint64_t ids_bytes = (static_cast<uint64_t>(d.n) * 2 + 7) & ~7ull;
    return reinterpret_cast<const mask_t*>(pool + d.byte_offset + ids_bytes);
  }
  __host__ __device__ inline const uint64_t* bitmap_words(const MrmContainerDesc& d) const
  {
    return reinterpret_cast<const uint64_t*>(pool + d.byte_offset);
  }
  __host__ __device__ inline const mask_t* bitmap_masks(const MrmContainerDesc& d) const
  {
    return reinterpret_cast<const mask_t*>(pool + d.byte_offset + 8192);
  }
  __host__ __device__ inline const uint16_t* run_starts(const MrmContainerDesc& d) const
  {
    return reinterpret_cast<const uint16_t*>(pool + d.byte_offset);
  }
  __host__ __device__ inline const uint16_t* run_lens(const MrmContainerDesc& d) const
  {
    return reinterpret_cast<const uint16_t*>(pool + d.byte_offset) + d.n;
  }
  __host__ __device__ inline const mask_t* run_masks(const MrmContainerDesc& d) const
  {
    uint64_t sl_bytes = (static_cast<uint64_t>(d.n) * 4 + 7) & ~7ull;
    return reinterpret_cast<const mask_t*>(pool + d.byte_offset + sl_bytes);
  }
};

// Owning handle.
struct Mrm {
  MrmView view{};
  void* _alloc_base = nullptr;     // single device allocation
  uint64_t pool_bytes = 0;         // payload pool size (capacity, by bound)
  uint64_t nnz_bound = 0;          // sum of input cardinality bounds
  std::vector<uint16_t> chunk_keys;  // host mirror, ascending
};

// Host-side metadata of one input bitmap (all arrays length n_containers;
// available for free at filter construction time, or via download_meta()).
struct HostBitmapMeta {
  std::vector<uint16_t> keys;
  std::vector<ContainerType> types;
  std::vector<uint32_t> offsets;        // byte offsets into per-type pools
  std::vector<uint16_t> cardinalities;  // ARRAY/BITMAP: cardinality; RUN: n_runs
};

// D2H helper for callers that don't track metadata host-side.
HostBitmapMeta download_meta(const GpuRoaring& bitmap, cudaStream_t stream = 0);

// Build an MRM from k <= 64 device roaring bitmaps. `metas[i]` must describe
// `bitmaps[i]`. Negated inputs are not supported. Single kernel launch; no
// device->host synchronization.
Mrm mrm_build(const GpuRoaring* bitmaps,
              const HostBitmapMeta* metas,
              uint32_t k,
              cudaStream_t stream = 0);

// Convenience overload: downloads metadata first (k small syncs; prefer the
// overload above in hot paths).
Mrm mrm_build(const GpuRoaring* bitmaps, uint32_t k, cudaStream_t stream = 0);

void mrm_free(Mrm& m, cudaStream_t stream = 0);

// Test/debug: decode the MRM into (row id, mask) pairs, ascending by id.
// Returns pairs via out parameters (host vectors). Synchronizes.
void mrm_decode(const Mrm& m,
                std::vector<uint32_t>& ids,
                std::vector<mask_t>& masks,
                cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// Fused filtered search (Phase 3): exact top-k inner products of each lane's
// query against exactly the rows its filter selects, straight off the MRM.
// No dense bitmap, no CSR, no nnz-sized value materialization.
//
// dataset: [n_rows, dim] fp32 row-major (device). queries: [n_lanes, dim]
// fp32 row-major (device). out_ids/out_dists: [n_lanes, topk] (device),
// descending by inner product; out_ids padded with -1 when a lane selects
// fewer than topk rows. dim <= 512 (queries staged in shared memory).
// topk <= 32. Reusable scratch is allocated/freed per call.
// ----------------------------------------------------------------------------
void mrm_search(const Mrm& m,
                const float* dataset,
                uint32_t n_rows,
                uint32_t dim,
                const float* queries,
                uint32_t topk,
                int64_t* out_ids,
                float* out_dists,
                cudaStream_t stream = 0);

// Dense-tile variant (design doc §4.2): 32-row x 64-lane GEMM tiles over the
// compacted selected rows with a mask epilogue. Same contract as mrm_search;
// dim must be 128 (template instantiation), topk <= 16.
void mrm_search_tile(const Mrm& m,
                     const float* dataset,
                     uint32_t n_rows,
                     uint32_t dim,
                     const float* queries,
                     uint32_t topk,
                     int64_t* out_ids,
                     float* out_dists,
                     cudaStream_t stream = 0);

}  // namespace cu_roaring::mrm
