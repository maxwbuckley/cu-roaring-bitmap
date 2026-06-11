/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM fused filtered search, dense-tile kernel (Phase 3 v3).
 *
 * v2 ablation (see mrm/README.md) attributed 78% of kernel time to top-k
 * reduction (serial in-CTA merge + single-thread final merge) and showed
 * every phase latency-bound at 2 CTAs/SM (50 KB smem). v3 changes:
 *
 *  - queries are no longer staged in shared memory: the 64xDIM block
 *    (<=32 KB) is L1-resident and read with warp-uniform addresses; smem
 *    drops to ~17 KB -> ~3x more resident CTAs. The host pads queries to
 *    64 rows so dead lanes read zeros instead of out-of-bounds.
 *  - per-thread top-k is a fully unrolled predicated bubble insert over
 *    MAX_TOPK_T register slots (no dynamic indexing -> stays in
 *    registers), which makes warp shuffle merging possible.
 *  - the in-CTA merge exploits lane ownership: warp g exclusively owns
 *    lanes 8g..8g+7, so its 32 per-thread lists fold with a 5-round
 *    __shfl tree, zero shared memory, zero block barriers (v2 spent
 *    11.3 ms here on 64 barrier-separated serial selections).
 *  - the final merge kernel uses one warp per lane (strided scan +
 *    shuffle tree) instead of one thread per lane (v2: 6.9 ms).
 *
 * Tile compute is unchanged from v2: a serial producer emits 32-row
 * (id, mask) tiles from the container, rows are cooperatively staged in
 * padded smem, thread (row_slot, group) computes 8 lanes' dots with 8
 * FMAs per smem element, mask epilogue at insert time.
 *
 * fp32 inner product, dim == 128 (template), n_lanes <= 64, topk <= 16.
 */

#include <cu_roaring_mrm/mrm.cuh>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace cu_roaring::mrm {

static constexpr uint32_t TILE_ROWS = 32;
static constexpr uint32_t LANE_GROUPS = 8;   // warps per CTA
static constexpr uint32_t LANES_PER_THREAD = 8;
static constexpr uint32_t TBLOCK = 256;
static_assert(TBLOCK == TILE_ROWS * LANE_GROUPS, "one warp per lane group");
static constexpr int MAX_TOPK_T = 16;        // register top-k slots

#define MRM_CUDA_CHECK(call)                                                      \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "MRM CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                                \
      abort();                                                                    \
    }                                                                             \
  } while (0)

// Fully unrolled descending-order insert: compile-time indices only, so the
// arrays live in registers. The candidate bubbles down; the previous last
// element falls off.
__device__ static inline void reg_insert(float (&d)[MAX_TOPK_T],
                                         int32_t (&i)[MAX_TOPK_T],
                                         float nd,
                                         int32_t ni)
{
  if (nd <= d[MAX_TOPK_T - 1]) return;
#pragma unroll
  for (int t = 0; t < MAX_TOPK_T; ++t) {
    bool sw     = nd > d[t];
    float td    = sw ? nd : d[t];
    int32_t ti  = sw ? ni : i[t];
    nd          = sw ? d[t] : nd;
    ni          = sw ? i[t] : ni;
    d[t]        = td;
    i[t]        = ti;
  }
}

// Fold the 32 sorted register lists of a warp into lane 0's list via a
// shuffle tree. All threads participate in the shuffles; inserts are
// predicated on being a receiving thread.
__device__ static inline void warp_topk_merge(float (&d)[MAX_TOPK_T],
                                              int32_t (&i)[MAX_TOPK_T],
                                              uint32_t lane)
{
#pragma unroll
  for (uint32_t step = 16; step > 0; step >>= 1) {
#pragma unroll
    for (int j = 0; j < MAX_TOPK_T; ++j) {
      float pd   = __shfl_down_sync(0xffffffffu, d[j], step);
      int32_t pi = __shfl_down_sync(0xffffffffu, i[j], step);
      if (lane < step && pi >= 0) reg_insert(d, i, pd, pi);
    }
  }
}

// Producer state for "next TILE_ROWS (low16 id, mask) pairs of this segment".
struct Producer {
  uint32_t a_i, a_end;
  uint32_t b_w, b_wend, b_rank;
  uint64_t b_word;
  uint32_t r_run, r_off;
};

// ablate: 0 = full; 1 = skip merges/output (timing); 2 = + skip compute;
// 3 = + skip row loads; 4 = + skip producer body.
template <int DIM>
__global__ static void mrm_tile_kernel(MrmView view,
                                       const float* __restrict__ dataset,
                                       const float* __restrict__ queries,  // [64, DIM] padded
                                       uint32_t topk,
                                       uint32_t n_segs,
                                       uint32_t ablate,
                                       float* __restrict__ cand_dists,
                                       int64_t* __restrict__ cand_ids)
{
  uint32_t chunk = blockIdx.x;
  uint32_t seg   = blockIdx.y;
  if (chunk >= view.n_chunks) return;
  uint32_t tid      = threadIdx.x;
  uint32_t row_slot = tid & (TILE_ROWS - 1);
  uint32_t group    = tid >> 5;

  const MrmContainerDesc dsc = view.descs[chunk];
  uint64_t base_id           = static_cast<uint64_t>(dsc.key) << 16;
  auto type                  = static_cast<MrmContainerType>(dsc.type);

  extern __shared__ float smem[];
  float* srows = smem;  // [TILE_ROWS][DIM+1] padded
  auto srow    = [&](uint32_t r) { return srows + r * (DIM + 1); };
  uint32_t* sid    = reinterpret_cast<uint32_t*>(srows + TILE_ROWS * (DIM + 1));
  mask_t* smask    = reinterpret_cast<mask_t*>(sid + TILE_ROWS);
  uint32_t* scount = reinterpret_cast<uint32_t*>(smask + TILE_ROWS);

  // per-thread top-k for 8 lanes (local memory; folded via registers at end)
  float tk_d[LANES_PER_THREAD][MAX_TOPK_T];
  int32_t tk_i[LANES_PER_THREAD][MAX_TOPK_T];
  for (uint32_t l = 0; l < LANES_PER_THREAD; ++l)
    for (int t = 0; t < MAX_TOPK_T; ++t) {
      tk_d[l][t] = -FLT_MAX;
      tk_i[l][t] = -1;
    }

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
    if (ablate >= 4) {
      prod.a_i = prod.a_end = prod.b_w = prod.b_wend = 0;
      prod.r_run = 0xFFFFFFFFu;
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
    if (ablate < 3) {
      for (uint32_t idx = tid; idx < count * DIM; idx += TBLOCK) {
        uint32_t r = idx / DIM;
        uint32_t j = idx % DIM;
        srow(r)[j] = dataset[(base_id | sid[r]) * DIM + j];
      }
    }
    __syncthreads();

    // ---- 32x64 dense tile: 8 dots per thread; queries read via L1 ----
    if (row_slot < count && ablate < 2) {
      float acc[LANES_PER_THREAD];
#pragma unroll
      for (uint32_t l = 0; l < LANES_PER_THREAD; ++l)
        acc[l] = 0.0f;
      const float* rp = srow(row_slot);
      const float* qp = queries + group * LANES_PER_THREAD * DIM;
#pragma unroll 4
      for (uint32_t j = 0; j < DIM; ++j) {
        float rv = rp[j];
#pragma unroll
        for (uint32_t l = 0; l < LANES_PER_THREAD; ++l)
          acc[l] += rv * __ldg(&qp[l * DIM + j]);
      }
      mask_t mask = smask[row_slot];
      int32_t low = static_cast<int32_t>(sid[row_slot]);
#pragma unroll
      for (uint32_t l = 0; l < LANES_PER_THREAD; ++l) {
        uint32_t L = group * LANES_PER_THREAD + l;
        if (mask & (mask_t{1} << L)) reg_insert(tk_d[l], tk_i[l], acc[l], low);
      }
    }
    __syncthreads();  // before producer overwrites sid/smask
  }

  // ---- per-lane reduction: warp g owns lanes 8g..8g+7 exclusively, so a
  // shuffle tree folds its 32 lists with no smem and no block barriers ----
  if (ablate >= 1) return;
  for (uint32_t l = 0; l < LANES_PER_THREAD; ++l) {
    float rd[MAX_TOPK_T];
    int32_t ri[MAX_TOPK_T];
#pragma unroll
    for (int t = 0; t < MAX_TOPK_T; ++t) {
      rd[t] = tk_d[l][t];
      ri[t] = tk_i[l][t];
    }
    warp_topk_merge(rd, ri, row_slot);
    if (row_slot == 0) {
      uint32_t L = group * LANES_PER_THREAD + l;
      if (L < view.n_lanes) {
        uint64_t slot =
          ((static_cast<uint64_t>(chunk) * n_segs + seg) * view.n_lanes + L) * topk;
        for (uint32_t t = 0; t < topk; ++t) {
          cand_dists[slot + t] = rd[t];
          cand_ids[slot + t] =
            ri[t] < 0 ? -1 : static_cast<int64_t>(base_id | static_cast<uint32_t>(ri[t]));
        }
      }
    }
  }
}

// One warp per lane: threads scan candidate lists strided, then fold via
// the same shuffle tree. (v2 used one thread per lane: 6.9 ms of 23.4.)
__global__ static void tile_merge_kernel(uint32_t n_lists,
                                         uint32_t n_lanes,
                                         uint32_t topk,
                                         const float* __restrict__ cand_dists,
                                         const int64_t* __restrict__ cand_ids,
                                         float* __restrict__ out_dists,
                                         int64_t* __restrict__ out_ids)
{
  uint32_t L = blockIdx.x;
  if (L >= n_lanes) return;
  uint32_t lane = threadIdx.x & 31u;

  float rd[MAX_TOPK_T];
  int32_t ri[MAX_TOPK_T];
  // ids are 64-bit; track them via an index into the candidate array instead
  // of truncating: store the *position* in ri and resolve on output.
#pragma unroll
  for (int t = 0; t < MAX_TOPK_T; ++t) {
    rd[t] = -FLT_MAX;
    ri[t] = -1;
  }
  for (uint32_t c = lane; c < n_lists; c += 32) {
    uint64_t slot = (static_cast<uint64_t>(c) * n_lanes + L) * topk;
    for (uint32_t t = 0; t < topk; ++t) {
      int64_t id = cand_ids[slot + t];
      if (id < 0) break;  // lists are sorted descending
      reg_insert(rd, ri, cand_dists[slot + t], static_cast<int32_t>(slot + t));
    }
  }
  warp_topk_merge(rd, ri, lane);
  if (lane == 0) {
    for (uint32_t t = 0; t < topk; ++t) {
      out_dists[static_cast<uint64_t>(L) * topk + t] = rd[t];
      out_ids[static_cast<uint64_t>(L) * topk + t] =
        ri[t] < 0 ? -1 : cand_ids[ri[t]];
    }
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
  if (topk > static_cast<uint32_t>(MAX_TOPK_T)) {
    fprintf(stderr, "mrm_search_tile: topk=%u exceeds %d\n", topk, MAX_TOPK_T);
    abort();
  }
  uint32_t C = m.view.n_chunks;
  uint32_t k = m.view.n_lanes;
  if (C == 0) return;

  // Fixed per-CTA costs (producer init, merge rounds, launch) dominate when
  // segments get thin: measured optimum is ~128 CTAs total (SEGS=8 at C=16,
  // SEGS=1 at C=153 on RTX 5090), not maximal segmentation.
  uint32_t n_segs = 128 / C;
  if (n_segs < 1) n_segs = 1;
  if (n_segs > 64) n_segs = 64;
  static const uint32_t segs_override = [] {
    const char* e = getenv("MRM_TILE_SEGS");
    return e ? static_cast<uint32_t>(atoi(e)) : 0u;
  }();
  if (segs_override >= 1 && segs_override <= 64) n_segs = segs_override;

  constexpr int DIM = 128;

  // pad queries to 64 rows so dead lanes read zeros, not out-of-bounds
  float* q_pad      = nullptr;
  float* cand_dists = nullptr;
  int64_t* cand_ids = nullptr;
  uint64_t slots    = static_cast<uint64_t>(C) * n_segs * k * topk;
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&q_pad), 64 * DIM * 4, stream));
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&cand_dists), slots * 4, stream));
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&cand_ids), slots * 8, stream));
  if (k < 64) MRM_CUDA_CHECK(cudaMemsetAsync(q_pad, 0, 64 * DIM * 4, stream));
  MRM_CUDA_CHECK(cudaMemcpyAsync(
    q_pad, queries, static_cast<size_t>(k) * DIM * 4, cudaMemcpyDeviceToDevice, stream));

  static const uint32_t ablate = [] {
    const char* e = getenv("MRM_TILE_ABLATE");
    return e ? static_cast<uint32_t>(atoi(e)) : 0u;
  }();

  size_t smem_bytes = TILE_ROWS * (DIM + 1) * sizeof(float) +
                      TILE_ROWS * (sizeof(uint32_t) + sizeof(mask_t)) + 64;

  dim3 grid(C, n_segs);
  mrm_tile_kernel<DIM><<<grid, TBLOCK, smem_bytes, stream>>>(
    m.view, dataset, q_pad, topk, n_segs, ablate, cand_dists, cand_ids);
  MRM_CUDA_CHECK(cudaGetLastError());
  tile_merge_kernel<<<k, 32, 0, stream>>>(
    C * n_segs, k, topk, cand_dists, cand_ids, out_dists, out_ids);
  MRM_CUDA_CHECK(cudaGetLastError());

  MRM_CUDA_CHECK(cudaFreeAsync(q_pad, stream));
  MRM_CUDA_CHECK(cudaFreeAsync(cand_dists, stream));
  MRM_CUDA_CHECK(cudaFreeAsync(cand_ids, stream));
}

}  // namespace cu_roaring::mrm
