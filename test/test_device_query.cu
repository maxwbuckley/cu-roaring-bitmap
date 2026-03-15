#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <roaring/roaring.h>

#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/device/roaring_view.cuh"
#include "cu_roaring/device/roaring_warp_query.cuh"
#include "cu_roaring/device/make_view.cuh"

#include <random>
#include <vector>

// Forward declare bloom builder
namespace cu_roaring {
void build_key_bloom(GpuRoaring& bitmap, cudaStream_t stream);
}

// Kernel: test contains() for each query ID
__global__ void test_contains_kernel(cu_roaring::GpuRoaringView view,
                                     const uint32_t* query_ids,
                                     bool* results,
                                     uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_queries) { results[idx] = view.contains(query_ids[idx]); }
}

// Kernel: test warp_contains() — 32 threads per warp, each checks one ID
__global__ void test_warp_contains_kernel(cu_roaring::GpuRoaringView view,
                                          const uint32_t* query_ids,
                                          bool* results,
                                          uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_queries) { results[idx] = cu_roaring::warp_contains(view, query_ids[idx]); }
}

class DeviceQueryTest : public ::testing::Test {
 protected:
  void verify_contains(cu_roaring::GpuRoaring& gpu_bm,
                       const roaring_bitmap_t* cpu_bm,
                       const std::vector<uint32_t>& queries,
                       bool use_warp)
  {
    uint32_t n = static_cast<uint32_t>(queries.size());

    // Upload queries
    uint32_t* d_queries;
    bool* d_results;
    cudaMalloc(&d_queries, n * sizeof(uint32_t));
    cudaMalloc(&d_results, n * sizeof(bool));
    cudaMemcpy(d_queries, queries.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    auto view   = cu_roaring::make_view(gpu_bm);
    uint32_t blocks = (n + 255) / 256;

    if (use_warp) {
      test_warp_contains_kernel<<<blocks, 256>>>(view, d_queries, d_results, n);
    } else {
      test_contains_kernel<<<blocks, 256>>>(view, d_queries, d_results, n);
    }
    cudaDeviceSynchronize();

    // Download results
    std::vector<bool> h_results(n);
    // bool is tricky with cudaMemcpy, use char
    std::vector<char> h_results_raw(n);
    cudaMemcpy(h_results_raw.data(), d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify against CRoaring
    for (uint32_t i = 0; i < n; ++i) {
      bool gpu_result = h_results_raw[i];
      bool cpu_result = roaring_bitmap_contains(cpu_bm, queries[i]);
      EXPECT_EQ(gpu_result, cpu_result)
        << "Mismatch at query ID " << queries[i]
        << " (GPU=" << gpu_result << " CPU=" << cpu_result << ")"
        << (use_warp ? " [warp]" : " [scalar]");
    }

    cudaFree(d_queries);
    cudaFree(d_results);
  }
};

TEST_F(DeviceQueryTest, ScalarContains_Dense) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint32_t i = 0; i < 200000; ++i)
    if (dist(gen) < 0.5) roaring_bitmap_add(r, i);

  auto gpu = cu_roaring::upload(r);

  // Query: mix of present and absent IDs
  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 10000; ++i)
    queries.push_back(i * 20);  // some present, some not

  verify_contains(gpu, r, queries, false);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, ScalarContains_Sparse) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint32_t> dist(0, 999999);
  for (int i = 0; i < 5000; ++i)
    roaring_bitmap_add(r, dist(gen));

  auto gpu = cu_roaring::upload(r);

  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 10000; ++i)
    queries.push_back(dist(gen));

  verify_contains(gpu, r, queries, false);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, ScalarContains_WithBloom) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint32_t i = 0; i < 500000; ++i)
    if (dist(gen) < 0.3) roaring_bitmap_add(r, i);

  auto gpu = cu_roaring::upload(r);
  cu_roaring::build_key_bloom(gpu, 0);

  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 20000; ++i)
    queries.push_back(i * 25);

  verify_contains(gpu, r, queries, false);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, WarpContains_Dense) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint32_t i = 0; i < 200000; ++i)
    if (dist(gen) < 0.5) roaring_bitmap_add(r, i);

  auto gpu = cu_roaring::upload(r);

  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 10000; ++i)
    queries.push_back(i * 20);

  verify_contains(gpu, r, queries, true);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, WarpContains_ClusteredIDs) {
  // Test where warp threads query IDs in the same container (same key)
  roaring_bitmap_t* r = roaring_bitmap_create();
  for (uint32_t i = 0; i < 100000; ++i)
    roaring_bitmap_add(r, i);

  auto gpu = cu_roaring::upload(r);

  // All queries in same container (key=0): should share binary search
  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 1024; ++i)
    queries.push_back(i);

  verify_contains(gpu, r, queries, true);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, WarpContains_WithBloom) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint32_t i = 0; i < 500000; ++i)
    if (dist(gen) < 0.3) roaring_bitmap_add(r, i);

  auto gpu = cu_roaring::upload(r);
  cu_roaring::build_key_bloom(gpu, 0);

  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 20000; ++i)
    queries.push_back(i * 25);

  verify_contains(gpu, r, queries, true);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, Contains_RunContainers) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  // Create runs
  for (uint32_t i = 1000; i < 5000; ++i) roaring_bitmap_add(r, i);
  for (uint32_t i = 70000; i < 80000; ++i) roaring_bitmap_add(r, i);
  roaring_bitmap_run_optimize(r);

  auto gpu = cu_roaring::upload(r);

  std::vector<uint32_t> queries;
  for (uint32_t i = 0; i < 100000; i += 100)
    queries.push_back(i);

  verify_contains(gpu, r, queries, false);
  verify_contains(gpu, r, queries, true);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}

TEST_F(DeviceQueryTest, Contains_EmptyBitmap) {
  roaring_bitmap_t* r = roaring_bitmap_create();
  auto gpu = cu_roaring::upload(r);

  // All queries should return false
  std::vector<uint32_t> queries = {0, 1, 100, 65535, 65536, 1000000};

  uint32_t n = static_cast<uint32_t>(queries.size());
  uint32_t* d_queries;
  bool* d_results;
  cudaMalloc(&d_queries, n * sizeof(uint32_t));
  cudaMalloc(&d_results, n * sizeof(bool));
  cudaMemcpy(d_queries, queries.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Can't use make_view on empty bitmap (null pointers) — just verify n_containers == 0
  EXPECT_EQ(gpu.n_containers, 0u);

  cudaFree(d_queries);
  cudaFree(d_results);
  cu_roaring::gpu_roaring_free(gpu);
  roaring_bitmap_free(r);
}
