/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM fused filtered search (Phase 3, v1: fp32 inner product).
 *
 * Stage 1 (scan): one CTA per chunk, one warp per (lane, chunk) work item.
 * The warp walks the chunk's container, skips rows whose mask lacks its
 * lane bit (RUN_MASKED skips whole runs — the multitenant fast path),
 * computes the dot product cooperatively (32 threads across dim), and
 * keeps a warp-local top-k. Per-(chunk, lane) candidates go to global.
 * Row data reuse across lanes within a chunk is served by L1/L2 (the
 * chunk's selected rows are cache-resident), which converts the
 * multiplicity win into bandwidth without cross-lane coordination.
 *
 * Stage 2 (merge): one warp per lane folds its per-chunk candidate lists
 * into the final top-k.
 */

#include <cu_roaring_mrm/mrm.cuh>

#include <cfloat>
#include <cstdio>

namespace cu_roaring::mrm {

static constexpr uint32_t BLOCK = 256;
static constexpr uint32_t WARPS = BLOCK / 32;
static constexpr uint32_t MAX_TOPK = 32;

#define MRM_CUDA_CHECK(call)                                                      \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "MRM CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                                \
      abort();                                                                    \
    }                                                                             \
  } while (0)

// Warp-cooperative inner product: row and query read with stride-32 access.
__device__ static float warp_dot(const float* __restrict__ row,
                                 const float* __restrict__ query,
                                 uint32_t dim)
{
  float acc      = 0.0f;
  uint32_t lane  = threadIdx.x & 31u;
  for (uint32_t j = lane; j < dim; j += 32)
    acc += row[j] * query[j];
  for (uint32_t off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  return acc;  // valid on lane 0
}

// Insert (dist, id) into the warp leader's sorted-descending top-k arrays.
__device__ static void topk_insert(float* dist, int64_t* id, uint32_t topk,
                                   float d, int64_t i)
{
  if (d <= dist[topk - 1]) return;
  uint32_t pos = topk - 1;
  while (pos > 0 && dist[pos - 1] < d) {
    dist[pos] = dist[pos - 1];
    id[pos]   = id[pos - 1];
    --pos;
  }
  dist[pos] = d;
  id[pos]   = i;
}

__global__ static void mrm_scan_kernel(MrmView view,
                                       const float* __restrict__ dataset,
                                       const float* __restrict__ queries,
                                       uint32_t dim,
                                       uint32_t topk,
                                       float* __restrict__ cand_dists,
                                       int64_t* __restrict__ cand_ids)
{
  uint32_t chunk = blockIdx.x;
  if (chunk >= view.n_chunks) return;
  uint32_t warp = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 31u;

  const MrmContainerDesc d = view.descs[chunk];
  uint64_t base_id         = static_cast<uint64_t>(d.key) << 16;
  auto type                = static_cast<MrmContainerType>(d.type);

  for (uint32_t L = warp; L < view.n_lanes; L += WARPS) {
    mask_t lane_bit      = mask_t{1} << L;
    const float* query   = queries + static_cast<uint64_t>(L) * dim;
    float top_d[MAX_TOPK];
    int64_t top_i[MAX_TOPK];
    for (uint32_t t = 0; t < topk; ++t) {
      top_d[t] = -FLT_MAX;
      top_i[t] = -1;
    }

    if (type == MrmContainerType::ARRAY_MASKED) {
      const uint16_t* ids = view.array_ids(d);
      const mask_t* masks = view.array_masks(d);
      for (uint32_t i = 0; i < d.n; ++i) {
        if ((masks[i] & lane_bit) == 0) continue;
        uint64_t row_id = base_id | ids[i];
        float dot = warp_dot(dataset + row_id * dim, query, dim);
        if (lane == 0) topk_insert(top_d, top_i, topk, dot, static_cast<int64_t>(row_id));
      }
    } else if (type == MrmContainerType::BITMAP_MASKED) {
      const uint64_t* words = view.bitmap_words(d);
      const mask_t* masks   = view.bitmap_masks(d);
      uint32_t rank         = 0;
      for (uint32_t w = 0; w < 1024; ++w) {
        uint64_t word = words[w];
        while (word != 0) {
          uint32_t bit = static_cast<uint32_t>(__ffsll(static_cast<long long>(word))) - 1;
          if (masks[rank] & lane_bit) {
            uint64_t row_id = base_id | (w * 64 + bit);
            float dot = warp_dot(dataset + row_id * dim, query, dim);
            if (lane == 0)
              topk_insert(top_d, top_i, topk, dot, static_cast<int64_t>(row_id));
          }
          ++rank;
          word &= word - 1;
        }
      }
    } else {  // RUN_MASKED: constant mask per run -> whole-run take or skip
      const uint16_t* starts = view.run_starts(d);
      const uint16_t* lens   = view.run_lens(d);
      const mask_t* masks    = view.run_masks(d);
      for (uint32_t r = 0; r < d.n; ++r) {
        if ((masks[r] & lane_bit) == 0) continue;
        uint32_t start = starts[r];
        uint32_t end   = start + lens[r];
        for (uint32_t v = start; v <= end; ++v) {
          uint64_t row_id = base_id | v;
          float dot = warp_dot(dataset + row_id * dim, query, dim);
          if (lane == 0) topk_insert(top_d, top_i, topk, dot, static_cast<int64_t>(row_id));
        }
      }
    }

    if (lane == 0) {
      uint64_t slot = (static_cast<uint64_t>(chunk) * view.n_lanes + L) * topk;
      for (uint32_t t = 0; t < topk; ++t) {
        cand_dists[slot + t] = top_d[t];
        cand_ids[slot + t]   = top_i[t];
      }
    }
  }
}

__global__ static void mrm_merge_kernel(uint32_t n_chunks,
                                        uint32_t n_lanes,
                                        uint32_t topk,
                                        const float* __restrict__ cand_dists,
                                        const int64_t* __restrict__ cand_ids,
                                        float* __restrict__ out_dists,
                                        int64_t* __restrict__ out_ids)
{
  uint32_t L = blockIdx.x;
  if (L >= n_lanes || threadIdx.x != 0) return;
  float top_d[MAX_TOPK];
  int64_t top_i[MAX_TOPK];
  for (uint32_t t = 0; t < topk; ++t) {
    top_d[t] = -FLT_MAX;
    top_i[t] = -1;
  }
  for (uint32_t c = 0; c < n_chunks; ++c) {
    uint64_t slot = (static_cast<uint64_t>(c) * n_lanes + L) * topk;
    for (uint32_t t = 0; t < topk; ++t) {
      int64_t id = cand_ids[slot + t];
      if (id < 0) break;  // candidate lists are sorted descending
      topk_insert(top_d, top_i, topk, cand_dists[slot + t], id);
    }
  }
  for (uint32_t t = 0; t < topk; ++t) {
    out_dists[static_cast<uint64_t>(L) * topk + t] = top_d[t];
    out_ids[static_cast<uint64_t>(L) * topk + t]   = top_i[t];
  }
}

void mrm_search(const Mrm& m,
                const float* dataset,
                uint32_t n_rows,
                uint32_t dim,
                const float* queries,
                uint32_t topk,
                int64_t* out_ids,
                float* out_dists,
                cudaStream_t stream)
{
  (void)n_rows;
  if (topk > MAX_TOPK) {
    fprintf(stderr, "mrm_search: topk=%u exceeds MAX_TOPK=%u\n", topk, MAX_TOPK);
    abort();
  }
  uint32_t C = m.view.n_chunks;
  uint32_t k = m.view.n_lanes;
  if (C == 0) return;

  float* cand_dists = nullptr;
  int64_t* cand_ids = nullptr;
  uint64_t slots    = static_cast<uint64_t>(C) * k * topk;
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&cand_dists), slots * 4, stream));
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&cand_ids), slots * 8, stream));

  mrm_scan_kernel<<<C, BLOCK, 0, stream>>>(
    m.view, dataset, queries, dim, topk, cand_dists, cand_ids);
  MRM_CUDA_CHECK(cudaGetLastError());
  mrm_merge_kernel<<<k, 32, 0, stream>>>(
    C, k, topk, cand_dists, cand_ids, out_dists, out_ids);
  MRM_CUDA_CHECK(cudaGetLastError());

  MRM_CUDA_CHECK(cudaFreeAsync(cand_dists, stream));
  MRM_CUDA_CHECK(cudaFreeAsync(cand_ids, stream));
}

}  // namespace cu_roaring::mrm
