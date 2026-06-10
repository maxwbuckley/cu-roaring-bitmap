/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM construction: k-way masked merge of roaring containers.
 *
 * One CTA per 64K chunk, single kernel launch, no device->host syncs:
 *  1. OR all source containers into an 8 KB shared-memory union bitmap
 *  2. per-word popcount prefix (rank table)
 *  3. emit ARRAY_MASKED (union < 4096) or BITMAP_MASKED payload
 *  4. scatter per-lane mask bits via rank lookup
 *  5. detect maximal constant-mask runs; rewrite as RUN_MASKED if smaller
 *
 * Capacity is allocated from host-side per-container cardinality bounds
 * (known at filter construction) — deliberately overallocating instead of
 * running a count kernel + sync (the cuvs#1960 anti-pattern).
 */

#include <cu_roaring_mrm/mrm.cuh>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>

namespace cu_roaring::mrm {

static constexpr uint32_t BLOCK = 256;
static constexpr uint32_t WORDS = 1024;            // 64K bits / 64
static constexpr uint32_t ARRAY_LIMIT = 4096;      // union < limit -> ARRAY_MASKED
static constexpr uint32_t RUN_STAGE_MAX = 1024;    // max runs rewritten in-block

__host__ __device__ static inline uint64_t align8(uint64_t v) { return (v + 7) & ~7ull; }

#define MRM_CUDA_CHECK(call)                                                      \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "MRM CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                                \
      abort();                                                                    \
    }                                                                             \
  } while (0)

// ----------------------------------------------------------------------------
// Construction kernel
// ----------------------------------------------------------------------------
struct BuildArgs {
  // per chunk (n_chunks entries, ascending key)
  const uint16_t* chunk_keys;
  const uint64_t* region_off;   // byte offset of chunk payload region in pool
  const uint32_t* work_starts;  // [n_chunks + 1] into the source arrays
  // flattened sources
  const uint8_t* src_lane;
  const uint8_t* src_type;      // cu_roaring::ContainerType
  const void* const* src_ptr;   // element pointer (typed by src_type)
  const uint32_t* src_card;     // ARRAY/BITMAP: cardinality; RUN: n_runs
  // outputs
  MrmContainerDesc* descs;
  uint8_t* pool;
};

__device__ static void block_scan_exclusive_256(uint32_t* vals, uint32_t tid, uint32_t* total)
{
  // Hillis-Steele inclusive scan on 256 elements, then convert to exclusive.
  for (uint32_t stride = 1; stride < BLOCK; stride *= 2) {
    uint32_t v = (tid >= stride) ? vals[tid - stride] : 0;
    __syncthreads();
    vals[tid] += v;
    __syncthreads();
  }
  if (tid == 0 && total != nullptr) *total = vals[BLOCK - 1];
  uint32_t mine = (tid > 0) ? vals[tid - 1] : 0;
  __syncthreads();
  vals[tid] = mine;
  __syncthreads();
}

__global__ static void mrm_build_kernel(BuildArgs a, uint32_t n_chunks)
{
  uint32_t chunk = blockIdx.x;
  if (chunk >= n_chunks) return;
  uint32_t tid = threadIdx.x;

  __shared__ uint64_t uwords[WORDS];     // union bitmap (8 KB)
  __shared__ uint32_t wprefix[WORDS];    // exclusive rank prefix per word (4 KB)
  __shared__ uint32_t partials[BLOCK];   // scan scratch
  __shared__ uint32_t s_total;           // union cardinality
  __shared__ uint32_t s_runs;            // run count
  __shared__ uint16_t s_run_idx[RUN_STAGE_MAX];     // run start indices
  __shared__ uint16_t s_run_start[RUN_STAGE_MAX];   // staged run starts
  __shared__ uint16_t s_run_len[RUN_STAGE_MAX];     // staged run lens (count-1)
  __shared__ mask_t s_run_mask[RUN_STAGE_MAX];      // staged run masks

  // 1. union of all sources ---------------------------------------------------
  for (uint32_t w = tid; w < WORDS; w += BLOCK)
    uwords[w] = 0;
  __syncthreads();

  uint32_t e_begin = a.work_starts[chunk];
  uint32_t e_end   = a.work_starts[chunk + 1];
  for (uint32_t e = e_begin; e < e_end; ++e) {
    auto ctype = static_cast<ContainerType>(a.src_type[e]);
    if (ctype == ContainerType::BITMAP) {
      const uint64_t* src = static_cast<const uint64_t*>(a.src_ptr[e]);
      for (uint32_t w = tid; w < WORDS; w += BLOCK)
        uwords[w] |= src[w];  // word w owned by one thread per pass: no atomics
    } else if (ctype == ContainerType::ARRAY) {
      const uint16_t* src = static_cast<const uint16_t*>(a.src_ptr[e]);
      uint32_t card       = a.src_card[e];
      for (uint32_t i = tid; i < card; i += BLOCK) {
        uint32_t v = src[i];
        atomicOr(reinterpret_cast<unsigned long long*>(&uwords[v >> 6]),
                 1ull << (v & 63u));
      }
    } else {  // RUN
      // runs are few but can be 64K elements long: serialize over runs,
      // parallelize words within each run across the block
      const uint16_t* runs = static_cast<const uint16_t*>(a.src_ptr[e]);
      uint32_t n_runs      = a.src_card[e];
      for (uint32_t r = 0; r < n_runs; ++r) {
        uint32_t start  = runs[r * 2];
        uint32_t end    = start + runs[r * 2 + 1];  // inclusive
        uint32_t w_lo   = start >> 6;
        uint32_t w_hi   = end >> 6;
        for (uint32_t w = w_lo + tid; w <= w_hi; w += BLOCK) {
          uint32_t lo   = (w << 6) < start ? (start & 63u) : 0;
          uint32_t hi   = ((w << 6) + 63u) > end ? (end & 63u) : 63u;
          uint64_t mask = (hi - lo == 63u) ? ~0ull : (((1ull << (hi - lo + 1)) - 1) << lo);
          atomicOr(reinterpret_cast<unsigned long long*>(&uwords[w]), mask);
        }
      }
    }
    __syncthreads();
  }

  // 2. rank table -------------------------------------------------------------
  {
    uint32_t mine = 0;
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w)
      mine += static_cast<uint32_t>(__popcll(uwords[w]));
    partials[tid] = mine;
    __syncthreads();
    block_scan_exclusive_256(partials, tid, &s_total);
    uint32_t running = partials[tid];
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w) {
      wprefix[w] = running;
      running += static_cast<uint32_t>(__popcll(uwords[w]));
    }
    __syncthreads();
  }
  uint32_t U = s_total;

  // 3. emit union payload -----------------------------------------------------
  uint64_t off    = a.region_off[chunk];
  bool emit_array = (U < ARRAY_LIMIT);
  uint16_t* arr_ids = reinterpret_cast<uint16_t*>(a.pool + off);
  mask_t* masks     = emit_array
                        ? reinterpret_cast<mask_t*>(a.pool + off + align8((uint64_t)U * 2))
                        : reinterpret_cast<mask_t*>(a.pool + off + 8192);

  if (emit_array) {
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w) {
      uint64_t word = uwords[w];
      uint32_t rank = wprefix[w];
      while (word != 0) {
        uint32_t bit    = static_cast<uint32_t>(__ffsll(static_cast<long long>(word))) - 1;
        arr_ids[rank++] = static_cast<uint16_t>((w << 6) | bit);
        word &= word - 1;
      }
    }
  } else {
    uint64_t* out_words = reinterpret_cast<uint64_t*>(a.pool + off);
    for (uint32_t w = tid; w < WORDS; w += BLOCK)
      out_words[w] = uwords[w];
  }
  for (uint32_t i = tid; i < U; i += BLOCK)
    masks[i] = 0;
  __syncthreads();

  // 4. mask scatter -----------------------------------------------------------
  for (uint32_t e = e_begin; e < e_end; ++e) {
    mask_t lane_bit = mask_t{1} << a.src_lane[e];
    auto ctype      = static_cast<ContainerType>(a.src_type[e]);
    auto rank_of    = [&](uint32_t v) -> uint32_t {
      uint64_t below = (v & 63u) ? (uwords[v >> 6] & ((1ull << (v & 63u)) - 1)) : 0;
      return wprefix[v >> 6] + static_cast<uint32_t>(__popcll(below));
    };
    if (ctype == ContainerType::ARRAY) {
      const uint16_t* src = static_cast<const uint16_t*>(a.src_ptr[e]);
      uint32_t card       = a.src_card[e];
      for (uint32_t i = tid; i < card; i += BLOCK)
        atomicOr(reinterpret_cast<unsigned long long*>(&masks[rank_of(src[i])]),
                 static_cast<unsigned long long>(lane_bit));
    } else if (ctype == ContainerType::BITMAP) {
      const uint64_t* src = static_cast<const uint64_t*>(a.src_ptr[e]);
      for (uint32_t w = tid; w < WORDS; w += BLOCK) {
        uint64_t word = src[w];
        while (word != 0) {
          uint32_t bit = static_cast<uint32_t>(__ffsll(static_cast<long long>(word))) - 1;
          atomicOr(reinterpret_cast<unsigned long long*>(&masks[rank_of((w << 6) | bit)]),
                   static_cast<unsigned long long>(lane_bit));
          word &= word - 1;
        }
      }
    } else {  // RUN
      // serialize over runs, parallelize elements within each run
      const uint16_t* runs = static_cast<const uint16_t*>(a.src_ptr[e]);
      uint32_t n_runs      = a.src_card[e];
      for (uint32_t r = 0; r < n_runs; ++r) {
        uint32_t start = runs[r * 2];
        uint32_t end   = start + runs[r * 2 + 1];
        for (uint32_t v = start + tid; v <= end; v += BLOCK)
          atomicOr(reinterpret_cast<unsigned long long*>(&masks[rank_of(v)]),
                   static_cast<unsigned long long>(lane_bit));
      }
    }
  }
  __syncthreads();

  // 5. run detection (ARRAY only): rewrite as RUN_MASKED when clearly smaller
  uint8_t out_type = emit_array ? static_cast<uint8_t>(MrmContainerType::ARRAY_MASKED)
                                : static_cast<uint8_t>(MrmContainerType::BITMAP_MASKED);
  uint32_t out_n = U;

  if (emit_array && U > 0) {
    // contiguous partition of [0, U)
    uint32_t per  = (U + BLOCK - 1) / BLOCK;
    uint32_t lo   = tid * per;
    uint32_t hi   = lo + per < U ? lo + per : U;
    uint32_t mine = 0;
    for (uint32_t i = lo; i < hi; ++i) {
      bool is_start = (i == 0) || (arr_ids[i] != static_cast<uint16_t>(arr_ids[i - 1] + 1)) ||
                      (masks[i] != masks[i - 1]);
      mine += is_start ? 1u : 0u;
    }
    partials[tid] = mine;
    __syncthreads();
    block_scan_exclusive_256(partials, tid, &s_runs);
    uint32_t R = s_runs;

    // worth it when run payload (12 B/run) clearly beats array payload
    // (10 B/id) and fits the staging buffer
    if (R > 0 && R <= RUN_STAGE_MAX && static_cast<uint64_t>(R) * 12 < static_cast<uint64_t>(U) * 10 / 2) {
      uint32_t slot = partials[tid];
      for (uint32_t i = lo; i < hi; ++i) {
        bool is_start = (i == 0) || (arr_ids[i] != static_cast<uint16_t>(arr_ids[i - 1] + 1)) ||
                        (masks[i] != masks[i - 1]);
        if (is_start) s_run_idx[slot++] = static_cast<uint16_t>(i);
      }
      __syncthreads();
      for (uint32_t s = tid; s < R; s += BLOCK) {
        uint32_t i0    = s_run_idx[s];
        uint32_t i1    = (s + 1 < R) ? s_run_idx[s + 1] : U;
        s_run_start[s] = arr_ids[i0];
        s_run_len[s]   = static_cast<uint16_t>(i1 - i0 - 1);
        s_run_mask[s]  = masks[i0];
      }
      __syncthreads();
      // overwrite region with RUN layout: starts[R] | lens[R] | align8 | masks[R]
      uint16_t* out_starts = reinterpret_cast<uint16_t*>(a.pool + off);
      uint16_t* out_lens   = out_starts + R;
      mask_t* out_masks    = reinterpret_cast<mask_t*>(a.pool + off + align8((uint64_t)R * 4));
      for (uint32_t s = tid; s < R; s += BLOCK) {
        out_starts[s] = s_run_start[s];
        out_lens[s]   = s_run_len[s];
        out_masks[s]  = s_run_mask[s];
      }
      out_type = static_cast<uint8_t>(MrmContainerType::RUN_MASKED);
      out_n    = R;
    }
  }
  __syncthreads();

  if (tid == 0) {
    MrmContainerDesc d;
    d.key         = a.chunk_keys[chunk];
    d.type        = out_type;
    d._pad        = 0;
    d.n           = out_n;
    d.byte_offset = off;
    a.descs[chunk] = d;
  }
}

// ----------------------------------------------------------------------------
// Host orchestration
// ----------------------------------------------------------------------------
HostBitmapMeta download_meta(const GpuRoaring& bitmap, cudaStream_t stream)
{
  HostBitmapMeta m;
  uint32_t n = bitmap.n_containers;
  m.keys.resize(n);
  m.types.resize(n);
  m.offsets.resize(n);
  m.cardinalities.resize(n);
  if (n == 0) return m;
  MRM_CUDA_CHECK(cudaMemcpyAsync(
    m.keys.data(), bitmap.keys, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
  MRM_CUDA_CHECK(cudaMemcpyAsync(
    m.types.data(), bitmap.types, n * sizeof(ContainerType), cudaMemcpyDeviceToHost, stream));
  MRM_CUDA_CHECK(cudaMemcpyAsync(
    m.offsets.data(), bitmap.offsets, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  MRM_CUDA_CHECK(cudaMemcpyAsync(m.cardinalities.data(),
                                 bitmap.cardinalities,
                                 n * sizeof(uint16_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
  MRM_CUDA_CHECK(cudaStreamSynchronize(stream));
  return m;
}

Mrm mrm_build(const GpuRoaring* bitmaps,
              const HostBitmapMeta* metas,
              uint32_t k,
              cudaStream_t stream)
{
  if (k < 1 || k > kMaxLanes) {
    fprintf(stderr, "mrm_build: k=%u out of range [1, %u]\n", k, kMaxLanes);
    abort();
  }

  struct Src {
    uint8_t lane;
    uint8_t type;
    const void* ptr;
    uint32_t card;
    uint32_t bound;  // element-count upper bound
  };
  std::map<uint16_t, std::vector<Src>> chunks;  // ordered by key

  for (uint32_t l = 0; l < k; ++l) {
    const GpuRoaring& b    = bitmaps[l];
    const HostBitmapMeta& m = metas[l];
    if (b.negated) {
      // upload paths apply a complement optimization above 50% density;
      // a negated input here would silently merge the complement.
      fprintf(stderr, "mrm_build: lane %u is negated (unsupported)\n", l);
      abort();
    }
    for (uint32_t c = 0; c < b.n_containers; ++c) {
      Src s;
      s.lane = static_cast<uint8_t>(l);
      s.type = static_cast<uint8_t>(m.types[c]);
      s.card = m.cardinalities[c];
      switch (m.types[c]) {
        case ContainerType::ARRAY:
          s.ptr   = b.array_data + m.offsets[c] / 2;
          s.bound = s.card;
          break;
        case ContainerType::BITMAP:
          s.ptr = b.bitmap_data + m.offsets[c] / 8;
          // uint16 cardinality wraps for a full container: 65536 -> 0
          s.bound = (s.card == 0) ? 65536u : s.card;
          break;
        case ContainerType::RUN:
        default:
          s.ptr   = b.run_data + m.offsets[c] / 2;
          s.bound = 65536;  // element count unknown host-side; conservative
          break;
      }
      chunks[m.keys[c]].push_back(s);
    }
  }

  uint32_t C = static_cast<uint32_t>(chunks.size());

  // flatten worklist + region offsets
  std::vector<uint16_t> h_keys;
  std::vector<uint64_t> h_off;
  std::vector<uint32_t> h_starts{0};
  std::vector<uint8_t> h_lane, h_type;
  std::vector<const void*> h_ptr;
  std::vector<uint32_t> h_card;
  uint64_t pool_bytes = 0;
  uint64_t nnz_bound  = 0;
  for (auto& [key, srcs] : chunks) {
    uint64_t bound = 0;
    for (auto& s : srcs) {
      bound += s.bound;
      nnz_bound += s.bound;
      h_lane.push_back(s.lane);
      h_type.push_back(s.type);
      h_ptr.push_back(s.ptr);
      h_card.push_back(s.card);
    }
    if (bound > 65536) bound = 65536;
    uint64_t region = (bound < ARRAY_LIMIT) ? align8(bound * 2) + bound * 8
                                            : 8192 + bound * 8;
    h_keys.push_back(key);
    h_off.push_back(pool_bytes);
    h_starts.push_back(static_cast<uint32_t>(h_lane.size()));
    pool_bytes += align8(region);
  }

  // single device allocation: descs | keys | off | starts | lane | type | ptr | card | pool
  uint32_t S          = static_cast<uint32_t>(h_lane.size());
  uint64_t descs_b    = align8(static_cast<uint64_t>(C) * sizeof(MrmContainerDesc));
  uint64_t keys_b     = align8(static_cast<uint64_t>(C) * 2);
  uint64_t off_b      = align8(static_cast<uint64_t>(C) * 8);
  uint64_t starts_b   = align8(static_cast<uint64_t>(C + 1) * 4);
  uint64_t lane_b     = align8(S);
  uint64_t type_b     = align8(S);
  uint64_t ptr_b      = align8(static_cast<uint64_t>(S) * sizeof(void*));
  uint64_t card_b     = align8(static_cast<uint64_t>(S) * 4);
  uint64_t meta_total = descs_b + keys_b + off_b + starts_b + lane_b + type_b + ptr_b + card_b;

  uint8_t* base = nullptr;
  MRM_CUDA_CHECK(cudaMallocAsync(
    reinterpret_cast<void**>(&base), meta_total + pool_bytes, stream));

  uint8_t* p           = base;
  auto* d_descs        = reinterpret_cast<MrmContainerDesc*>(p);  p += descs_b;
  auto* d_keys         = reinterpret_cast<uint16_t*>(p);          p += keys_b;
  auto* d_off          = reinterpret_cast<uint64_t*>(p);          p += off_b;
  auto* d_starts       = reinterpret_cast<uint32_t*>(p);          p += starts_b;
  auto* d_lane         = p;                                       p += lane_b;
  auto* d_type         = p;                                       p += type_b;
  auto* d_ptr          = reinterpret_cast<const void**>(p);       p += ptr_b;
  auto* d_card         = reinterpret_cast<uint32_t*>(p);          p += card_b;
  uint8_t* d_pool      = p;

  if (C > 0) {
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_keys, h_keys.data(), C * 2, cudaMemcpyHostToDevice, stream));
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_off, h_off.data(), C * 8, cudaMemcpyHostToDevice, stream));
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_starts, h_starts.data(), (C + 1) * 4, cudaMemcpyHostToDevice, stream));
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_lane, h_lane.data(), S, cudaMemcpyHostToDevice, stream));
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_type, h_type.data(), S, cudaMemcpyHostToDevice, stream));
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_ptr, h_ptr.data(), S * sizeof(void*), cudaMemcpyHostToDevice, stream));
    MRM_CUDA_CHECK(cudaMemcpyAsync(
      d_card, h_card.data(), S * 4, cudaMemcpyHostToDevice, stream));

    BuildArgs args;
    args.chunk_keys  = d_keys;
    args.region_off  = d_off;
    args.work_starts = d_starts;
    args.src_lane    = d_lane;
    args.src_type    = d_type;
    args.src_ptr     = d_ptr;
    args.src_card    = d_card;
    args.descs       = d_descs;
    args.pool        = d_pool;
    mrm_build_kernel<<<C, BLOCK, 0, stream>>>(args, C);
    MRM_CUDA_CHECK(cudaGetLastError());
  }

  Mrm m;
  m.view.descs    = d_descs;
  m.view.pool     = d_pool;
  m.view.n_chunks = C;
  m.view.n_lanes  = k;
  m._alloc_base   = base;
  m.pool_bytes    = pool_bytes;
  m.nnz_bound     = nnz_bound;
  m.chunk_keys    = std::move(h_keys);
  return m;
}

Mrm mrm_build(const GpuRoaring* bitmaps, uint32_t k, cudaStream_t stream)
{
  std::vector<HostBitmapMeta> metas(k);
  for (uint32_t l = 0; l < k; ++l)
    metas[l] = download_meta(bitmaps[l], stream);
  return mrm_build(bitmaps, metas.data(), k, stream);
}

void mrm_free(Mrm& m, cudaStream_t stream)
{
  if (m._alloc_base != nullptr) {
    MRM_CUDA_CHECK(cudaFreeAsync(m._alloc_base, stream));
    m._alloc_base = nullptr;
  }
  m.view = MrmView{};
}

}  // namespace cu_roaring::mrm
