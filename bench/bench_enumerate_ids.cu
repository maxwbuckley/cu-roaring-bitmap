/*
 * B9: enumerate_ids (CSR Export) Benchmark
 *
 * Measures enumerate_ids() latency for extracting sorted element IDs
 * directly from compressed Roaring bitmaps, and compares against the
 * two-step alternative: decompress_to_bitset() + bitset-to-CSR scan.
 *
 * The two-step baseline represents what a real system would do without
 * enumerate_ids: decompress the Roaring bitmap to a flat bitset, then
 * scan the entire bitset to extract set bit positions into sorted int64_t
 * array (CSR column indices).
 *
 * Test matrix:
 *   Universe sizes: 1M, 10M, 100M, 1B
 *   Densities:      0.1%, 1%, 10%, 50%
 *
 * Outputs JSON to results/raw/bench9_enumerate_ids.json
 */

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/detail/to_csr.cuh"
#include "cu_roaring/detail/decompress.cuh"
#include "data_gen.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

// ============================================================================
// Baseline: bitset → sorted int64_t CSR column indices
//
// Two-pass GPU pipeline:
//   Pass 1: popcount each uint32_t word
//   CUB ExclusiveSum on popcounts → per-word write offsets
//   Pass 2: extract set bit positions at computed offsets
// ============================================================================

__global__ void popcount_words_kernel(const uint32_t* __restrict__ bitset,
                                      uint32_t* __restrict__ counts,
                                      uint32_t n_words)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_words) {
    counts[i] = __popc(bitset[i]);
  }
}

__global__ void extract_bits_kernel(const uint32_t* __restrict__ bitset,
                                    const uint32_t* __restrict__ offsets,
                                    int64_t* __restrict__ output,
                                    uint32_t n_words)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_words) return;

  uint32_t word     = bitset[i];
  uint32_t base_bit = i * 32;
  uint32_t out_idx  = offsets[i];

  while (word != 0) {
    uint32_t bit = __ffs(word) - 1;
    output[out_idx++] = static_cast<int64_t>(base_bit + bit);
    word &= word - 1;
  }
}

// Run the full bitset→CSR pipeline on a pre-decompressed bitset.
// Assumes d_bitset is already populated. Allocates scratch internally.
static void bitset_to_csr(const uint32_t* d_bitset,
                           uint32_t n_words,
                           int64_t* d_output,
                           cudaStream_t stream)
{
  constexpr uint32_t BLOCK = 256;
  uint32_t grid = (n_words + BLOCK - 1) / BLOCK;

  // Allocate popcount + offset arrays
  uint32_t* d_counts  = nullptr;
  uint32_t* d_offsets = nullptr;
  cudaMallocAsync(&d_counts, n_words * sizeof(uint32_t), stream);
  cudaMallocAsync(&d_offsets, n_words * sizeof(uint32_t), stream);

  // Pass 1: popcount
  popcount_words_kernel<<<grid, BLOCK, 0, stream>>>(d_bitset, d_counts, n_words);

  // CUB exclusive prefix sum
  void* d_temp  = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_counts, d_offsets,
                                static_cast<int>(n_words), stream);
  cudaMallocAsync(&d_temp, temp_bytes, stream);
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_counts, d_offsets,
                                static_cast<int>(n_words), stream);
  cudaFreeAsync(d_temp, stream);

  // Pass 2: extract bits
  extract_bits_kernel<<<grid, BLOCK, 0, stream>>>(d_bitset, d_offsets, d_output, n_words);

  cudaFreeAsync(d_counts, stream);
  cudaFreeAsync(d_offsets, stream);
}

// ============================================================================
// Statistics
// ============================================================================

struct Stats {
  double median, mean, min_v, max_v, std_dev;
};

static Stats compute_stats(std::vector<double>& t)
{
  std::sort(t.begin(), t.end());
  int n = static_cast<int>(t.size());
  double sum = 0;
  for (auto v : t) sum += v;
  double mean = sum / n;
  double var  = 0;
  for (auto v : t) var += (v - mean) * (v - mean);
  return {t[static_cast<size_t>(n / 2)], mean, t[0], t[static_cast<size_t>(n - 1)],
          std::sqrt(var / n)};
}

// ============================================================================
// Main benchmark
// ============================================================================

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB L2)\n\n",
         prop.name, prop.multiProcessorCount,
         prop.l2CacheSize / (1024.0 * 1024.0));

  FILE* jf = fopen("results/raw/bench9_enumerate_ids.json", "w");
  if (!jf) jf = fopen("bench9_enumerate_ids.json", "w");

  fprintf(jf, "{\n  \"benchmark\": \"enumerate_ids\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n", prop.name);
  fprintf(jf, "  \"results\": [\n");

  struct Config {
    uint32_t universe;
    double   density;      // fraction, e.g. 0.001 = 0.1%
    int      warmup;
    int      iters;
  };

  // Scale iterations inversely with universe size to keep total runtime
  // reasonable while maintaining n>=30 for smaller configs.
  Config configs[] = {
    // universe,   density, warmup, iters
    {   1000000,   0.001,    10,    50},  // 1M, 0.1%  →    ~1K IDs
    {   1000000,   0.01,     10,    50},  // 1M, 1%    →   10K IDs
    {   1000000,   0.10,     10,    50},  // 1M, 10%   →  100K IDs
    {   1000000,   0.50,     10,    50},  // 1M, 50%   →  500K IDs
    {  10000000,   0.001,    10,    50},  // 10M, 0.1% →   10K IDs
    {  10000000,   0.01,     10,    50},  // 10M, 1%   →  100K IDs
    {  10000000,   0.10,     10,    50},  // 10M, 10%  →    1M IDs
    {  10000000,   0.50,     10,    50},  // 10M, 50%  →    5M IDs
    { 100000000,   0.01,      5,    30},  // 100M, 1%  →    1M IDs
    { 100000000,   0.10,      5,    30},  // 100M, 10% →   10M IDs
    { 100000000,   0.50,      5,    30},  // 100M, 50% →   50M IDs
    {1000000000u,  0.001,     3,    30},  // 1B, 0.1%  →    1M IDs
    {1000000000u,  0.01,      3,    30},  // 1B, 1%    →   10M IDs
    {1000000000u,  0.10,      3,    15},  // 1B, 10%   →  100M IDs
  };
  int n_configs = sizeof(configs) / sizeof(configs[0]);

  bool first_json = true;

  for (int ci = 0; ci < n_configs; ++ci) {
    auto& cfg = configs[ci];
    uint64_t expected_card = static_cast<uint64_t>(cfg.universe * cfg.density);

    printf("=== Universe=%uM  Density=%.1f%%  (~%luK IDs) ===\n",
           cfg.universe / 1000000, cfg.density * 100.0,
           static_cast<unsigned long>(expected_card / 1000));
    fflush(stdout);

    // Generate bitmap
    auto* cpu_bm = cu_roaring::bench::generate_bitmap(cfg.universe, cfg.density, 42);
    auto gpu_bm  = cu_roaring::upload(cpu_bm, cfg.universe);

    uint64_t actual_card = gpu_bm.total_cardinality;
    uint32_t n_words     = (gpu_bm.universe_size + 31) / 32;

    printf("  Actual cardinality: %lu  (%u containers: %u array, %u bitmap, %u run)\n",
           static_cast<unsigned long>(actual_card),
           gpu_bm.n_containers,
           gpu_bm.n_array_containers,
           gpu_bm.n_bitmap_containers,
           gpu_bm.n_run_containers);

    // Pre-allocate output buffers (reused across iterations)
    int64_t*  d_csr_output   = nullptr;
    uint32_t* d_bitset       = nullptr;
    int64_t*  d_baseline_out = nullptr;

    cudaMalloc(&d_csr_output, actual_card * sizeof(int64_t));
    cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));
    cudaMalloc(&d_baseline_out, actual_card * sizeof(int64_t));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // ---- Benchmark: enumerate_ids (direct CSR export) ----
    // Warmup
    for (int i = 0; i < cfg.warmup; ++i) {
      cu_roaring::enumerate_ids(gpu_bm, d_csr_output);
      cudaDeviceSynchronize();
    }

    std::vector<double> enum_times;
    for (int i = 0; i < cfg.iters; ++i) {
      cudaEventRecord(ev_start);
      cu_roaring::enumerate_ids(gpu_bm, d_csr_output);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);

      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      enum_times.push_back(static_cast<double>(ms));
    }
    auto enum_stats = compute_stats(enum_times);

    // ---- Benchmark: decompress_to_bitset (step 1 of baseline) ----
    for (int i = 0; i < cfg.warmup; ++i) {
      cu_roaring::decompress_to_bitset(gpu_bm, d_bitset, n_words);
      cudaDeviceSynchronize();
    }

    std::vector<double> decomp_times;
    for (int i = 0; i < cfg.iters; ++i) {
      cudaMemsetAsync(d_bitset, 0, n_words * sizeof(uint32_t));
      cudaEventRecord(ev_start);
      cu_roaring::decompress_to_bitset(gpu_bm, d_bitset, n_words);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);

      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      decomp_times.push_back(static_cast<double>(ms));
    }
    auto decomp_stats = compute_stats(decomp_times);

    // ---- Benchmark: bitset → CSR (step 2 of baseline) ----
    // First decompress once so d_bitset is populated
    cu_roaring::decompress_to_bitset(gpu_bm, d_bitset, n_words);
    cudaDeviceSynchronize();

    for (int i = 0; i < cfg.warmup; ++i) {
      bitset_to_csr(d_bitset, n_words, d_baseline_out, 0);
      cudaDeviceSynchronize();
    }

    std::vector<double> b2c_times;
    for (int i = 0; i < cfg.iters; ++i) {
      cudaEventRecord(ev_start);
      bitset_to_csr(d_bitset, n_words, d_baseline_out, 0);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);

      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      b2c_times.push_back(static_cast<double>(ms));
    }
    auto b2c_stats = compute_stats(b2c_times);

    // ---- Benchmark: full two-step baseline (decompress + bitset→CSR) ----
    for (int i = 0; i < cfg.warmup; ++i) {
      cu_roaring::decompress_to_bitset(gpu_bm, d_bitset, n_words);
      bitset_to_csr(d_bitset, n_words, d_baseline_out, 0);
      cudaDeviceSynchronize();
    }

    std::vector<double> full_baseline_times;
    for (int i = 0; i < cfg.iters; ++i) {
      cudaMemsetAsync(d_bitset, 0, n_words * sizeof(uint32_t));
      cudaEventRecord(ev_start);
      cu_roaring::decompress_to_bitset(gpu_bm, d_bitset, n_words);
      bitset_to_csr(d_bitset, n_words, d_baseline_out, 0);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);

      float ms = 0;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      full_baseline_times.push_back(static_cast<double>(ms));
    }
    auto baseline_stats = compute_stats(full_baseline_times);

    // ---- Correctness check: verify enumerate_ids output matches baseline ----
    // Run both once and compare on host
    cu_roaring::enumerate_ids(gpu_bm, d_csr_output);
    cudaMemsetAsync(d_bitset, 0, n_words * sizeof(uint32_t));
    cu_roaring::decompress_to_bitset(gpu_bm, d_bitset, n_words);
    bitset_to_csr(d_bitset, n_words, d_baseline_out, 0);
    cudaDeviceSynchronize();

    std::vector<int64_t> h_enum(actual_card);
    std::vector<int64_t> h_base(actual_card);
    cudaMemcpy(h_enum.data(), d_csr_output,
               actual_card * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_base.data(), d_baseline_out,
               actual_card * sizeof(int64_t), cudaMemcpyDeviceToHost);

    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < actual_card; ++i) {
      if (h_enum[i] != h_base[i]) ++mismatches;
    }

    // ---- Derived metrics ----
    double enum_median_ms     = enum_stats.median;
    double baseline_median_ms = baseline_stats.median;
    double speedup            = baseline_median_ms / enum_median_ms;

    // Throughput: IDs per second
    double enum_throughput_gids = (actual_card / 1e9) / (enum_median_ms / 1e3);
    double base_throughput_gids = (actual_card / 1e9) / (baseline_median_ms / 1e3);

    // Effective output bandwidth: bytes written per second
    double enum_bw_gbs  = (actual_card * 8.0 / 1e9) / (enum_median_ms / 1e3);
    double base_bw_gbs  = (actual_card * 8.0 / 1e9) / (baseline_median_ms / 1e3);

    // Bitset size that the baseline must scan
    double bitset_mb = n_words * sizeof(uint32_t) / (1024.0 * 1024.0);

    // Roaring compressed size (approximate: metadata + data pools)
    double roaring_mb = (gpu_bm.n_bitmap_containers * 1024 * 8
                        + gpu_bm.n_array_containers * 4096 * 2  // worst case
                        + gpu_bm.n_containers * (2 + 1 + 4 + 2))
                        / (1024.0 * 1024.0);

    printf("  enumerate_ids:      %8.3f ms (median)  ± %.3f ms\n",
           enum_median_ms, enum_stats.std_dev);
    printf("  decompress_bitset:  %8.3f ms (median)  ± %.3f ms\n",
           decomp_stats.median, decomp_stats.std_dev);
    printf("  bitset_to_csr:      %8.3f ms (median)  ± %.3f ms\n",
           b2c_stats.median, b2c_stats.std_dev);
    printf("  full baseline:      %8.3f ms (median)  ± %.3f ms\n",
           baseline_median_ms, baseline_stats.std_dev);
    printf("  speedup:            %8.2fx\n", speedup);
    printf("  enum throughput:    %8.2f G IDs/s\n", enum_throughput_gids);
    printf("  enum bandwidth:     %8.1f GB/s\n", enum_bw_gbs);
    printf("  baseline bandwidth: %8.1f GB/s\n", base_bw_gbs);
    printf("  bitset scan size:   %8.1f MB\n", bitset_mb);
    printf("  correctness:        %lu mismatches / %lu IDs\n",
           static_cast<unsigned long>(mismatches),
           static_cast<unsigned long>(actual_card));
    printf("\n");
    fflush(stdout);

    // JSON output
    if (!first_json) fprintf(jf, ",\n");
    first_json = false;

    fprintf(jf, "    {\n");
    fprintf(jf, "      \"universe\": %u,\n", cfg.universe);
    fprintf(jf, "      \"density\": %.4f,\n", cfg.density);
    fprintf(jf, "      \"cardinality\": %lu,\n",
            static_cast<unsigned long>(actual_card));
    fprintf(jf, "      \"n_containers\": %u,\n", gpu_bm.n_containers);
    fprintf(jf, "      \"n_array_containers\": %u,\n", gpu_bm.n_array_containers);
    fprintf(jf, "      \"n_bitmap_containers\": %u,\n", gpu_bm.n_bitmap_containers);
    fprintf(jf, "      \"n_run_containers\": %u,\n", gpu_bm.n_run_containers);
    fprintf(jf, "      \"negated\": %s,\n", gpu_bm.negated ? "true" : "false");
    fprintf(jf, "      \"bitset_size_mb\": %.2f,\n", bitset_mb);
    fprintf(jf, "      \"enumerate_ids_median_ms\": %.4f,\n", enum_stats.median);
    fprintf(jf, "      \"enumerate_ids_mean_ms\": %.4f,\n", enum_stats.mean);
    fprintf(jf, "      \"enumerate_ids_std_ms\": %.4f,\n", enum_stats.std_dev);
    fprintf(jf, "      \"enumerate_ids_min_ms\": %.4f,\n", enum_stats.min_v);
    fprintf(jf, "      \"enumerate_ids_max_ms\": %.4f,\n", enum_stats.max_v);
    fprintf(jf, "      \"decompress_median_ms\": %.4f,\n", decomp_stats.median);
    fprintf(jf, "      \"decompress_mean_ms\": %.4f,\n", decomp_stats.mean);
    fprintf(jf, "      \"decompress_std_ms\": %.4f,\n", decomp_stats.std_dev);
    fprintf(jf, "      \"bitset_to_csr_median_ms\": %.4f,\n", b2c_stats.median);
    fprintf(jf, "      \"bitset_to_csr_mean_ms\": %.4f,\n", b2c_stats.mean);
    fprintf(jf, "      \"bitset_to_csr_std_ms\": %.4f,\n", b2c_stats.std_dev);
    fprintf(jf, "      \"full_baseline_median_ms\": %.4f,\n", baseline_stats.median);
    fprintf(jf, "      \"full_baseline_mean_ms\": %.4f,\n", baseline_stats.mean);
    fprintf(jf, "      \"full_baseline_std_ms\": %.4f,\n", baseline_stats.std_dev);
    fprintf(jf, "      \"speedup\": %.2f,\n", speedup);
    fprintf(jf, "      \"enum_throughput_gids\": %.3f,\n", enum_throughput_gids);
    fprintf(jf, "      \"baseline_throughput_gids\": %.3f,\n", base_throughput_gids);
    fprintf(jf, "      \"enum_bandwidth_gbs\": %.1f,\n", enum_bw_gbs);
    fprintf(jf, "      \"baseline_bandwidth_gbs\": %.1f,\n", base_bw_gbs);
    fprintf(jf, "      \"mismatches\": %lu,\n",
            static_cast<unsigned long>(mismatches));
    fprintf(jf, "      \"iters\": %d\n", cfg.iters);
    fprintf(jf, "    }");

    // Cleanup
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_csr_output);
    cudaFree(d_bitset);
    cudaFree(d_baseline_out);
    cu_roaring::gpu_roaring_free(gpu_bm);
    roaring_bitmap_free(cpu_bm);
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);
  printf("=== B9 COMPLETE ===\n");
  printf("Results written to results/raw/bench9_enumerate_ids.json\n");
  return 0;
}
