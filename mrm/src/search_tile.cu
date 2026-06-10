/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM fused filtered search, dense-tile kernel (Phase 3 v2, design doc §4.2).
 *
 * The v1 lane-major scan (search.cu) re-traverses each container per lane
 * and pays a warp reduction per (row, lane) pair — measured 2.2-78x slower
 * than the SDDMM MVP. This kernel instead treats each 32-row tile of the
 * chunk's *compacted* selected rows as a dense GEMM tile against the full
 * 64-lane query block:
 *
 *   - queries [64, 128] staged once per CTA in shared memory
 *   - a serial producer (thread 0) emits the next 32 (row id, mask) pairs
 *     from the container (array slice / bitmap rank walk / run slice)
 *   - all threads cooperatively load the 32 rows into padded smem
 *   - thread (row_slot r, lane_group g) computes 8 dots (lanes 8g..8g+7)
 *     with 8 FMAs per smem row element — no shuffles, no re-reads
 *   - mask epilogue drops non-selected (row, lane) pairs at insert time;
 *     wasted FLOPs are bounded by tile mask density (zero waste when all
 *     lanes share the filter)
 *   - per-thread top-k registers; end-of-segment in-CTA merge produces one
 *     top-k list per (chunk-segment, lane), folded by the same merge
 *     kernel as v1
 *
 * fp32 inner product, dim == 128 (template), n_lanes <= 64, topk <= 32.
 */

#include <cu_roaring_mrm/mrm.cuh>

#include <cfloat>
#include <cstdio>

namespace cu_roaring::mrm {

static constexpr uint32_t TILE_ROWS = 32;
static constexpr uint32_t LANE_GROUPS = 8;   // 8 lanes per thread
static constexpr uint32_t LANES_PER_THREAD = 8;
static constexpr uint32_t TBLOCK = 256;      // TILE_ROWS * LANE_GROUPS
static constexpr uint32_t MAX_TOPK_T = 16;   // per-thread top-k capacity

#define MRM_CUDA_CHECK(call)                                                      \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "MRM CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                                \
      abort();                                                                    \
    }                                                                             \
  } while (0)

__device__ static void tk_insert(float* dist, int32_t* id, uint32_t topk,
                                 float d, int32_t i)
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

// Producer state for "next TILE_ROWS (low16 id, mask) pairs of this segment".
struct Producer {
  // ARRAY: [lo, hi) index range. BITMAP: word range + running rank.
  // RUN: per-run element slices.
  uint32_t a_i, a_end;
  uint32_t b_w, b_wend, b_rank;
  uint64_t b_word;
  uint32_t r_run, r_off;  // current run index, offset within its slice
};

template <int DIM>
__global__ static void mrm_tile_kernel(MrmView view,
                                       const float* __restrict__ dataset,
                                       const float* __restrict__ queries,
                                       uint32_t topk,
                                       uint32_t n_segs,
                                       float* __restrict__ cand_dists,
                                       int64_t* __restrict__ cand_ids)
{
  uint32_t chunk = blockIdx.x;
  uint32_t seg   = blockIdx.y;
  if (chunk >= view.n_chunks) return;
  uint32_t tid      = threadIdx.x;
  uint32_t row_slot = tid & (TILE_ROWS - 1);
  uint32_t group    = tid >> 5;  // tid / TILE_ROWS

  const MrmContainerDesc dsc = view.descs[chunk];
  uint64_t base_id           = static_cast<uint64_t>(dsc.key) << 16;
  auto type                  = static_cast<MrmContainerType>(dsc.type);

  extern __shared__ float smem[];
  float* sq    = smem;                          // [64][DIM]
  float* srows = sq + 64 * DIM;                 // [TILE_ROWS][DIM+1] padded
  auto srow    = [&](uint32_t r) { return srows + r * (DIM + 1); };
  uint32_t* sid   = reinterpret_cast<uint32_t*>(srows + TILE_ROWS * (DIM + 1));
  mask_t* smask   = reinterpret_cast<mask_t*>(sid + TILE_ROWS);  // 8B aligned ok
  uint32_t* scount = reinterpret_cast<uint32_t*>(smask + TILE_ROWS);
  // merge staging reuses the srows region after the segment loop
  float* stage_d  = srows;
  int32_t* stage_i = reinterpret_cast<int32_t*>(srows + TILE_ROWS * MAX_TOPK_T);

  // stage queries (lanes beyond n_lanes are zero -> their dots are 0 and
  // never inserted because no mask bit can be set for them)
  for (uint32_t idx = tid; idx < 64 * DIM; idx += TBLOCK)
    sq[idx] = (idx / DIM) < view.n_lanes ? queries[idx] : 0.0f;

  // per-thread top-k for 8 lanes (local memory)
  float tk_d[LANES_PER_THREAD][MAX_TOPK_T];
  int32_t tk_i[LANES_PER_THREAD][MAX_TOPK_T];
  for (uint32_t l = 0; l < LANES_PER_THREAD; ++l)
    for (uint32_t t = 0; t < topk; ++t) {
      tk_d[l][t] = -FLT_MAX;
      tk_i[l][t] = -1;
    }

  // segment bounds + producer init (thread 0 owns producer state in smem? it
  // is cheaper to recompute in registers of thread 0 only; communicated via
  // sid/smask/scount each tile)
  __shared__ Producer prod;
  if (tid == 0) {
    if (type == MrmContainerType::ARRAY_MASKED) {
      prod.a_i   = seg * dsc.n / n_segs;
      prod.a_end = (seg + 1) * dsc.n / n_segs;
    } else if (type == MrmContainerType::BITMAP_MASKED) {
      prod.b_w    = seg * 1024 / n_segs;
      prod.b_wend = (seg + 1) * 1024 / n_segs;
      const uint64_t* words = view.bitmap_words(dsc);
      uint32_t rank         = 0;
      for (uint32_t w = 0; w < prod.b_w; ++w)
        rank += static_cast<uint32_t>(__popcll(words[w]));
      prod.b_rank = rank;
      prod.b_word = prod.b_w < prod.b_wend ? words[prod.b_w] : 0;
    } else {
      prod.r_run = 0;
      prod.r_off = 0;
    }
  }
  __syncthreads();

  while (true) {
    // ---- produce next tile (thread 0, serial; tiny vs the GEMM tile) ----
    if (tid == 0) {
      uint32_t count = 0;
      if (type == MrmContainerType::ARRAY_MASKED) {
        const uint16_t* ids = view.array_ids(dsc);
        const mask_t* masks = view.array_masks(dsc);
        while (count < TILE_ROWS && prod.a_i < prod.a_end) {
          sid[count]   = ids[prod.a_i];
          smask[count] = masks[prod.a_i];
          ++prod.a_i;
          ++count;
        }
      } else if (type == MrmContainerType::BITMAP_MASKED) {
        const uint64_t* words = view.bitmap_words(dsc);
        const mask_t* masks   = view.bitmap_masks(dsc);
        while (count < TILE_ROWS && prod.b_w < prod.b_wend) {
          if (prod.b_word == 0) {
            ++prod.b_w;
            if (prod.b_w >= prod.b_wend) break;
            prod.b_word = words[prod.b_w];
            continue;
          }
          uint32_t bit =
            static_cast<uint32_t>(__ffsll(static_cast<long long>(prod.b_word))) - 1;
          sid[count]   = prod.b_w * 64 + bit;
          smask[count] = masks[prod.b_rank];
          ++prod.b_rank;
          ++count;
          prod.b_word &= prod.b_word - 1;
        }
      } else {  // RUN: each segment takes a slice of every run
        const uint16_t* starts = view.run_starts(dsc);
        const uint16_t* lens   = view.run_lens(dsc);
        const mask_t* masks    = view.run_masks(dsc);
        while (count < TILE_ROWS && prod.r_run < dsc.n) {
          uint32_t total = static_cast<uint32_t>(lens[prod.r_run]) + 1;
          uint32_t e_lo  = seg * total / n_segs;
          uint32_t e_hi  = (seg + 1) * total / n_segs;
          if (prod.r_off < e_lo) prod.r_off = e_lo;
          if (prod.r_off >= e_hi) {
            ++prod.r_run;
            prod.r_off = 0;
            continue;
          }
          sid[count]   = static_cast<uint32_t>(starts[prod.r_run]) + prod.r_off;
          smask[count] = masks[prod.r_run];
          ++prod.r_off;
          ++count;
        }
      }
      *scount = count;
    }
    __syncthreads();
    uint32_t count = *scount;
    if (count == 0) break;

    // ---- cooperative row load (padded to kill bank conflicts) ----
    for (uint32_t idx = tid; idx < count * DIM; idx += TBLOCK) {
      uint32_t r = idx / DIM;
      uint32_t j = idx % DIM;
      srow(r)[j] = dataset[(base_id | sid[r]) * DIM + j];
    }
    __syncthreads();

    // ---- 32x64 dense tile: 8 dots per thread ----
    if (row_slot < count) {
      float acc[LANES_PER_THREAD];
#pragma unroll
      for (uint32_t l = 0; l < LANES_PER_THREAD; ++l)
        acc[l] = 0.0f;
      const float* rp = srow(row_slot);
#pragma unroll 4
      for (uint32_t j = 0; j < DIM; ++j) {
        float rv = rp[j];
#pragma unroll
        for (uint32_t l = 0; l < LANES_PER_THREAD; ++l)
          acc[l] += rv * sq[(group * LANES_PER_THREAD + l) * DIM + j];
      }
      mask_t mask = smask[row_slot];
      int32_t low = static_cast<int32_t>(sid[row_slot]);
#pragma unroll
      for (uint32_t l = 0; l < LANES_PER_THREAD; ++l) {
        uint32_t L = group * LANES_PER_THREAD + l;
        if (mask & (mask_t{1} << L)) tk_insert(tk_d[l], tk_i[l], topk, acc[l], low);
      }
    }
    __syncthreads();  // before producer overwrites sid/smask
  }

  // ---- in-CTA per-lane merge ----
  // For each (group, lane-within-group): the group's 32 threads stage their
  // per-thread lists into the (now idle) row region; one thread selects the
  // CTA-level top-k for that lane. 64 short iterations.
  for (uint32_t g = 0; g < LANE_GROUPS; ++g) {
    for (uint32_t l = 0; l < LANES_PER_THREAD; ++l) {
      __syncthreads();
      if (group == g) {
        for (uint32_t t = 0; t < topk; ++t) {
          stage_d[row_slot * MAX_TOPK_T + t] = tk_d[l][t];
          stage_i[row_slot * MAX_TOPK_T + t] = tk_i[l][t];
        }
      }
      __syncthreads();
      uint32_t L = g * LANES_PER_THREAD + l;
      if (tid == 0 && L < view.n_lanes) {
        float od[MAX_TOPK_T];
        int32_t oi[MAX_TOPK_T];
        for (uint32_t t = 0; t < topk; ++t) {
          od[t] = -FLT_MAX;
          oi[t] = -1;
        }
        for (uint32_t r = 0; r < TILE_ROWS; ++r)
          for (uint32_t t = 0; t < topk; ++t) {
            int32_t i = stage_i[r * MAX_TOPK_T + t];
            if (i < 0) break;
            tk_insert(od, oi, topk, stage_d[r * MAX_TOPK_T + t], i);
          }
        uint64_t slot =
          ((static_cast<uint64_t>(chunk) * n_segs + seg) * view.n_lanes + L) * topk;
        for (uint32_t t = 0; t < topk; ++t) {
          cand_dists[slot + t] = od[t];
          cand_ids[slot + t] =
            oi[t] < 0 ? -1 : static_cast<int64_t>(base_id | static_cast<uint32_t>(oi[t]));
        }
      }
    }
  }
}

// Same merge as v1 (duplicated here to keep TUs independent).
__global__ static void tile_merge_kernel(uint32_t n_lists,
                                         uint32_t n_lanes,
                                         uint32_t topk,
                                         const float* __restrict__ cand_dists,
                                         const int64_t* __restrict__ cand_ids,
                                         float* __restrict__ out_dists,
                                         int64_t* __restrict__ out_ids)
{
  uint32_t L = blockIdx.x;
  if (L >= n_lanes || threadIdx.x != 0) return;
  float top_d[MAX_TOPK_T];
  int64_t top_i[MAX_TOPK_T];
  for (uint32_t t = 0; t < topk; ++t) {
    top_d[t] = -FLT_MAX;
    top_i[t] = -1;
  }
  for (uint32_t c = 0; c < n_lists; ++c) {
    uint64_t slot = (static_cast<uint64_t>(c) * n_lanes + L) * topk;
    for (uint32_t t = 0; t < topk; ++t) {
      int64_t id = cand_ids[slot + t];
      if (id < 0) break;
      float d      = cand_dists[slot + t];
      if (d <= top_d[topk - 1]) continue;
      uint32_t pos = topk - 1;
      while (pos > 0 && top_d[pos - 1] < d) {
        top_d[pos] = top_d[pos - 1];
        top_i[pos] = top_i[pos - 1];
        --pos;
      }
      top_d[pos] = d;
      top_i[pos] = id;
    }
  }
  for (uint32_t t = 0; t < topk; ++t) {
    out_dists[static_cast<uint64_t>(L) * topk + t] = top_d[t];
    out_ids[static_cast<uint64_t>(L) * topk + t]   = top_i[t];
  }
}

void mrm_search_tile(const Mrm& m,
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
  if (dim != 128) {
    fprintf(stderr, "mrm_search_tile: only dim=128 instantiated (got %u)\n", dim);
    abort();
  }
  if (topk > MAX_TOPK_T) {
    fprintf(stderr, "mrm_search_tile: topk=%u exceeds %u\n", topk, MAX_TOPK_T);
    abort();
  }
  uint32_t C = m.view.n_chunks;
  uint32_t k = m.view.n_lanes;
  if (C == 0) return;

  uint32_t n_segs = 4096 / C;
  if (n_segs < 1) n_segs = 1;
  if (n_segs > 64) n_segs = 64;

  float* cand_dists = nullptr;
  int64_t* cand_ids = nullptr;
  uint64_t slots    = static_cast<uint64_t>(C) * n_segs * k * topk;
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&cand_dists), slots * 4, stream));
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&cand_ids), slots * 8, stream));

  constexpr int DIM = 128;
  size_t smem_bytes = (64 * DIM + TILE_ROWS * (DIM + 1)) * sizeof(float) +
                      TILE_ROWS * (sizeof(uint32_t) + sizeof(mask_t)) + 64;
  static bool attr_set = false;
  if (!attr_set) {
    MRM_CUDA_CHECK(cudaFuncSetAttribute(mrm_tile_kernel<DIM>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes)));
    attr_set = true;
  }

  dim3 grid(C, n_segs);
  mrm_tile_kernel<DIM><<<grid, TBLOCK, smem_bytes, stream>>>(
    m.view, dataset, queries, topk, n_segs, cand_dists, cand_ids);
  MRM_CUDA_CHECK(cudaGetLastError());
  tile_merge_kernel<<<k, 32, 0, stream>>>(
    C * n_segs, k, topk, cand_dists, cand_ids, out_dists, out_ids);
  MRM_CUDA_CHECK(cudaGetLastError());

  MRM_CUDA_CHECK(cudaFreeAsync(cand_dists, stream));
  MRM_CUDA_CHECK(cudaFreeAsync(cand_ids, stream));
}

}  // namespace cu_roaring::mrm
