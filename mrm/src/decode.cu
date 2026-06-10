/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM decode: expand an MRM back into (row id, mask) pairs. Test/debug
 * utility — not a hot path. One block per chunk into fixed 64K-slot
 * regions; the host compacts and sorts.
 */

#include <cu_roaring_mrm/mrm.cuh>

#include <algorithm>
#include <cstdio>

namespace cu_roaring::mrm {

static constexpr uint32_t BLOCK = 256;

#define MRM_CUDA_CHECK(call)                                                      \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "MRM CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                                \
      abort();                                                                    \
    }                                                                             \
  } while (0)

__device__ static void scan_exclusive_256(uint32_t* vals, uint32_t tid, uint32_t* total)
{
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

__global__ static void mrm_decode_kernel(MrmView view,
                                         uint32_t* out_ids,
                                         mask_t* out_masks,
                                         uint32_t* out_counts)
{
  uint32_t chunk = blockIdx.x;
  if (chunk >= view.n_chunks) return;
  uint32_t tid               = threadIdx.x;
  const MrmContainerDesc d   = view.descs[chunk];
  uint32_t base_id           = static_cast<uint32_t>(d.key) << 16;
  uint64_t out_base          = static_cast<uint64_t>(chunk) * 65536;
  __shared__ uint32_t partials[BLOCK];
  __shared__ uint32_t s_total;

  auto type = static_cast<MrmContainerType>(d.type);
  if (type == MrmContainerType::ARRAY_MASKED) {
    const uint16_t* ids = view.array_ids(d);
    const mask_t* masks = view.array_masks(d);
    for (uint32_t i = tid; i < d.n; i += BLOCK) {
      out_ids[out_base + i]   = base_id | ids[i];
      out_masks[out_base + i] = masks[i];
    }
    if (tid == 0) out_counts[chunk] = d.n;
  } else if (type == MrmContainerType::BITMAP_MASKED) {
    const uint64_t* words = view.bitmap_words(d);
    const mask_t* masks   = view.bitmap_masks(d);
    uint32_t mine         = 0;
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w)
      mine += static_cast<uint32_t>(__popcll(words[w]));
    partials[tid] = mine;
    __syncthreads();
    scan_exclusive_256(partials, tid, &s_total);
    uint32_t rank = partials[tid];
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w) {
      uint64_t word = words[w];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(__ffsll(static_cast<long long>(word))) - 1;
        out_ids[out_base + rank]   = base_id | (w << 6) | bit;
        out_masks[out_base + rank] = masks[rank];
        ++rank;
        word &= word - 1;
      }
    }
    if (tid == 0) out_counts[chunk] = s_total;
  } else {  // RUN_MASKED
    const uint16_t* starts = view.run_starts(d);
    const uint16_t* lens   = view.run_lens(d);
    const mask_t* masks    = view.run_masks(d);
    uint32_t mine          = 0;
    for (uint32_t r = tid; r < d.n; r += BLOCK)
      mine += static_cast<uint32_t>(lens[r]) + 1;
    partials[tid] = mine;
    __syncthreads();
    scan_exclusive_256(partials, tid, &s_total);
    uint32_t pos = partials[tid];
    for (uint32_t r = tid; r < d.n; r += BLOCK) {
      uint32_t start = starts[r];
      for (uint32_t v = 0; v <= lens[r]; ++v) {
        out_ids[out_base + pos]   = base_id | (start + v);
        out_masks[out_base + pos] = masks[r];
        ++pos;
      }
    }
    if (tid == 0) out_counts[chunk] = s_total;
  }
}

void mrm_decode(const Mrm& m,
                std::vector<uint32_t>& ids,
                std::vector<mask_t>& masks,
                cudaStream_t stream)
{
  ids.clear();
  masks.clear();
  uint32_t C = m.view.n_chunks;
  if (C == 0) return;

  uint32_t* d_ids   = nullptr;
  mask_t* d_masks   = nullptr;
  uint32_t* d_count = nullptr;
  uint64_t slots    = static_cast<uint64_t>(C) * 65536;
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ids), slots * 4, stream));
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_masks), slots * 8, stream));
  MRM_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_count), C * 4, stream));

  mrm_decode_kernel<<<C, BLOCK, 0, stream>>>(m.view, d_ids, d_masks, d_count);
  MRM_CUDA_CHECK(cudaGetLastError());

  std::vector<uint32_t> h_counts(C);
  MRM_CUDA_CHECK(
    cudaMemcpyAsync(h_counts.data(), d_count, C * 4, cudaMemcpyDeviceToHost, stream));
  MRM_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<uint32_t> h_ids(slots);
  std::vector<mask_t> h_masks(slots);
  MRM_CUDA_CHECK(
    cudaMemcpyAsync(h_ids.data(), d_ids, slots * 4, cudaMemcpyDeviceToHost, stream));
  MRM_CUDA_CHECK(
    cudaMemcpyAsync(h_masks.data(), d_masks, slots * 8, cudaMemcpyDeviceToHost, stream));
  MRM_CUDA_CHECK(cudaStreamSynchronize(stream));

  struct Pair {
    uint32_t id;
    mask_t mask;
  };
  std::vector<Pair> pairs;
  for (uint32_t c = 0; c < C; ++c)
    for (uint32_t i = 0; i < h_counts[c]; ++i)
      pairs.push_back({h_ids[static_cast<uint64_t>(c) * 65536 + i],
                       h_masks[static_cast<uint64_t>(c) * 65536 + i]});
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
    return a.id < b.id;
  });
  ids.reserve(pairs.size());
  masks.reserve(pairs.size());
  for (auto& pr : pairs) {
    ids.push_back(pr.id);
    masks.push_back(pr.mask);
  }

  MRM_CUDA_CHECK(cudaFreeAsync(d_ids, stream));
  MRM_CUDA_CHECK(cudaFreeAsync(d_masks, stream));
  MRM_CUDA_CHECK(cudaFreeAsync(d_count, stream));
}

}  // namespace cu_roaring::mrm
