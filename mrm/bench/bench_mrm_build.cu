/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM construction benchmark (Phase 2 deliverable): mrm_build cost across
 * k / selectivity / clustering, with and without host-metadata reuse.
 * Acceptance context: construction must stay <= 10-15% of end-to-end
 * search time in the target regime (k>=256 ... here capped at 64 lanes per
 * tile; multi-tile batching is Phase 4).
 *
 * Env: MRM_BUILD_Q (default 1e6), MRM_BUILD_ITERS (20).
 */

#include <cu_roaring/cu_roaring.cuh>
#include <cu_roaring_mrm/mrm.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

using cu_roaring::GpuRoaring;
using cu_roaring::mrm::HostBitmapMeta;
using cu_roaring::mrm::Mrm;

static std::vector<uint32_t> gen_ids(std::mt19937_64& rng,
                                     uint32_t q,
                                     uint32_t p,
                                     const char* mode)
{
  std::vector<uint32_t> ids;
  if (std::string(mode) == "contiguous") {
    uint32_t start = static_cast<uint32_t>(rng() % (q - p));
    ids.resize(p);
    for (uint32_t i = 0; i < p; ++i)
      ids[i] = start + i;
    return ids;
  }
  std::map<uint32_t, bool> seen;
  while (ids.size() < p) {
    uint32_t v = static_cast<uint32_t>(rng() % q);
    if (!seen.count(v)) {
      seen[v] = true;
      ids.push_back(v);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

int main()
{
  const char* qe   = std::getenv("MRM_BUILD_Q");
  uint32_t q       = qe ? static_cast<uint32_t>(atof(qe)) : 1000000u;
  const char* ie   = std::getenv("MRM_BUILD_ITERS");
  int iters        = ie ? atoi(ie) : 20;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  printf("MRM construction bench: q=%u iters=%d\n", q, iters);
  printf("%6s %8s %12s %4s | %10s %12s %12s %10s\n",
         "k", "s", "mode", "m", "build_ms", "meta_dl_ms", "pool_MB", "chunks");

  for (uint32_t k : {8u, 32u, 64u})
    for (double s : {0.001, 0.01, 0.1})
      for (const char* mode : {"uniform", "contiguous"})
        for (uint32_t m : {1u, k}) {
          std::mt19937_64 rng(42 + k + static_cast<uint64_t>(s * 1000) + m);
          uint32_t p      = static_cast<uint32_t>(s * q);
          uint32_t n_base = (k + m - 1) / m;

          std::vector<GpuRoaring> bitmaps;
          for (uint32_t b = 0; b < n_base; ++b) {
            auto ids = gen_ids(rng, q, p, mode);
            for (uint32_t j = 0; j < m && bitmaps.size() < k; ++j)
              bitmaps.push_back(cu_roaring::upload_from_sorted_ids(
                ids.data(), static_cast<uint32_t>(ids.size()), q));
          }

          // metadata download (one-time, amortizable; would be free if the
          // builder tracked host mirrors like v2 GpuRoaringBatch)
          auto t0 = std::chrono::steady_clock::now();
          std::vector<HostBitmapMeta> metas(k);
          for (uint32_t l = 0; l < k; ++l)
            metas[l] = cu_roaring::mrm::download_meta(bitmaps[l], stream);
          double meta_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();

          // warm + measure construction (metas reused)
          Mrm warm = cu_roaring::mrm::mrm_build(bitmaps.data(), metas.data(), k, stream);
          cudaStreamSynchronize(stream);
          uint64_t pool_bytes = warm.pool_bytes;
          uint32_t n_chunks   = warm.view.n_chunks;
          cu_roaring::mrm::mrm_free(warm, stream);

          std::vector<double> times;
          for (int it = 0; it < iters; ++it) {
            auto b0 = std::chrono::steady_clock::now();
            Mrm mm  = cu_roaring::mrm::mrm_build(bitmaps.data(), metas.data(), k, stream);
            cudaStreamSynchronize(stream);
            times.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - b0)
                              .count());
            cu_roaring::mrm::mrm_free(mm, stream);
          }
          std::sort(times.begin(), times.end());
          double med = times[times.size() / 2];

          printf("%6u %8.3f %12s %4u | %10.3f %12.2f %12.2f %10u\n",
                 k, s, mode, m, med, meta_ms,
                 static_cast<double>(pool_bytes) / 1048576.0, n_chunks);
          fflush(stdout);

          for (auto& b : bitmaps)
            cu_roaring::gpu_roaring_free(b);
        }

  cudaStreamDestroy(stream);
  return 0;
}
