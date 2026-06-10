/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM fused search tests: exact filtered top-k inner products vs a
 * double-precision CPU oracle (tie-tolerant: every returned id must be a
 * member of its lane's filter, its recomputed IP must match the GPU
 * distance, and must reach the oracle's kth value within tolerance).
 */

#include <gtest/gtest.h>

#include <cu_roaring/cu_roaring.cuh>
#include <cu_roaring_mrm/mrm.cuh>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

using cu_roaring::GpuRoaring;
using cu_roaring::mrm::mask_t;
using cu_roaring::mrm::Mrm;

namespace {

struct SearchCase {
  uint32_t q;
  uint32_t dim;
  uint32_t k;
  uint32_t topk;
  std::vector<std::vector<uint32_t>> lanes;
  std::vector<float> dataset;  // host [q, dim]
  std::vector<float> queries;  // host [k, dim]
};

void run_case(const SearchCase& sc)
{
  // device buffers
  float *d_data = nullptr, *d_q = nullptr, *d_out = nullptr;
  int64_t* d_ids = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_data, sc.dataset.size() * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_q, sc.queries.size() * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, (size_t)sc.k * sc.topk * 4));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_ids, (size_t)sc.k * sc.topk * 8));
  cudaMemcpy(d_data, sc.dataset.data(), sc.dataset.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_q, sc.queries.data(), sc.queries.size() * 4, cudaMemcpyHostToDevice);

  std::vector<GpuRoaring> bitmaps;
  for (auto& l : sc.lanes)
    bitmaps.push_back(cu_roaring::upload_from_sorted_ids(
      l.data(), static_cast<uint32_t>(l.size()), sc.q));

  Mrm m = cu_roaring::mrm::mrm_build(bitmaps.data(), sc.k);
  cu_roaring::mrm::mrm_search(m, d_data, sc.q, sc.dim, d_q, sc.topk, d_ids, d_out);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<int64_t> h_ids((size_t)sc.k * sc.topk);
  std::vector<float> h_out((size_t)sc.k * sc.topk);
  cudaMemcpy(h_ids.data(), d_ids, h_ids.size() * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * 4, cudaMemcpyDeviceToHost);

  for (uint32_t L = 0; L < sc.k; ++L) {
    // oracle: top-k true IPs over this lane's filter
    std::vector<double> ips;
    for (uint32_t id : sc.lanes[L]) {
      double acc = 0;
      for (uint32_t j = 0; j < sc.dim; ++j)
        acc += (double)sc.dataset[(size_t)id * sc.dim + j] *
               (double)sc.queries[(size_t)L * sc.dim + j];
      ips.push_back(acc);
    }
    std::sort(ips.begin(), ips.end(), std::greater<double>());
    uint32_t kk = std::min<uint32_t>(sc.topk, (uint32_t)ips.size());
    double kth  = kk > 0 ? ips[kk - 1] : 0;
    double tol  = 1e-4 * std::abs(kth) + 1e-4;

    for (uint32_t t = 0; t < sc.topk; ++t) {
      int64_t id = h_ids[(size_t)L * sc.topk + t];
      if (t >= kk) {
        ASSERT_EQ(id, -1) << "lane " << L << " slot " << t << " should be padded";
        continue;
      }
      ASSERT_GE(id, 0) << "lane " << L << " slot " << t << " missing";
      // membership
      ASSERT_TRUE(std::binary_search(sc.lanes[L].begin(), sc.lanes[L].end(),
                                     (uint32_t)id))
        << "lane " << L << " returned non-member id " << id;
      // value matches recomputation and reaches the oracle threshold
      double acc = 0;
      for (uint32_t j = 0; j < sc.dim; ++j)
        acc += (double)sc.dataset[(size_t)id * sc.dim + j] *
               (double)sc.queries[(size_t)L * sc.dim + j];
      ASSERT_NEAR(h_out[(size_t)L * sc.topk + t], acc, 1e-2)
        << "lane " << L << " dist mismatch for id " << id;
      ASSERT_GE(acc, kth - tol) << "lane " << L << " id " << id << " below oracle kth";
    }
  }

  cu_roaring::mrm::mrm_free(m);
  for (auto& b : bitmaps)
    cu_roaring::gpu_roaring_free(b);
  cudaFree(d_data);
  cudaFree(d_q);
  cudaFree(d_out);
  cudaFree(d_ids);
}

SearchCase make_case(uint32_t q, uint32_t dim, uint32_t k, uint32_t topk,
                     double s, const char* mode, uint32_t m_mult, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> fd(-1.0f, 1.0f);
  SearchCase sc;
  sc.q    = q;
  sc.dim  = dim;
  sc.k    = k;
  sc.topk = topk;
  sc.dataset.resize((size_t)q * dim);
  for (auto& v : sc.dataset)
    v = fd(rng);
  sc.queries.resize((size_t)k * dim);
  for (auto& v : sc.queries)
    v = fd(rng);

  uint32_t p      = static_cast<uint32_t>(s * q);
  uint32_t n_base = (k + m_mult - 1) / m_mult;
  for (uint32_t b = 0; b < n_base; ++b) {
    std::vector<uint32_t> ids;
    if (std::string(mode) == "contiguous") {
      uint32_t start = rng() % (q - p);
      for (uint32_t i = 0; i < p; ++i)
        ids.push_back(start + i);
    } else {
      std::map<uint32_t, bool> seen;
      while (ids.size() < p) {
        uint32_t v = static_cast<uint32_t>(rng() % q);
        if (!seen.count(v)) {
          seen[v] = true;
          ids.push_back(v);
        }
      }
      std::sort(ids.begin(), ids.end());
    }
    for (uint32_t j = 0; j < m_mult && sc.lanes.size() < k; ++j)
      sc.lanes.push_back(ids);
  }
  return sc;
}

}  // namespace

TEST(MrmSearch, UniformSparse)
{
  run_case(make_case(300000, 64, 16, 10, 0.005, "uniform", 1, 21));
}

TEST(MrmSearch, UniformDenseChunks)
{
  run_case(make_case(200000, 64, 8, 10, 0.05, "uniform", 1, 22));
}

TEST(MrmSearch, ContiguousShared)
{
  // all lanes identical contiguous range -> RUN_MASKED fast path
  run_case(make_case(300000, 64, 16, 10, 0.01, "contiguous", 16, 23));
}

TEST(MrmSearch, ContiguousDistinct)
{
  run_case(make_case(300000, 32, 8, 10, 0.02, "contiguous", 1, 24));
}

TEST(MrmSearch, FewerThanTopk)
{
  // each lane selects ~5 rows < topk=10 -> padding required
  run_case(make_case(100000, 32, 4, 10, 0.00005, "uniform", 1, 25));
}

TEST(MrmSearch, SixtyFourLanes)
{
  run_case(make_case(200000, 32, 64, 10, 0.01, "uniform", 8, 26));
}
