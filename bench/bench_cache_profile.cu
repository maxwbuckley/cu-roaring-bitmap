/*
 * B10: Cache Hierarchy Profiling Benchmark
 *
 * Compares flat bitset vs roaring contains() vs roaring warp_contains()
 * at 1B universe across densities {0.1%, 1%, 5%, 10%, 25%, 50%}.
 * Random query pattern only (realistic for filtered vector search).
 *
 * Phase 1 — Runtime: median latency and Gq/s over 50 iterations.
 * Phase 2 — Nsight Compute: run with --ncu flag and profile with ncu CLI
 *           to collect L1/L2/DRAM metrics per kernel.
 *
 * The --ncu flag runs each kernel exactly once with 1M queries (enough for
 * stable per-SM counters while keeping profile time short).
 *
 * The --density flag selects a single density point for Nsight profiling,
 * so each ncu invocation profiles exactly one (density, kernel) triple:
 *   bench_cache_profile --ncu --density 0.01
 *
 * Outputs JSON to results/raw/bench10_cache_profile.json
 */

#include <cuda_runtime.h>
#include <roaring/roaring.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/device/make_view.cuh"
#include "cu_roaring/device/roaring_view.cuh"
#include "cu_roaring/device/roaring_warp_query.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

// ============================================================================
// Query generation — random uniform only (the realistic case)
// ============================================================================
static std::vector<uint32_t> gen_random_queries(uint32_t universe,
                                                 uint32_t n_queries,
                                                 uint64_t seed)
{
  std::vector<uint32_t> q(n_queries);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dist(0, universe - 1);
  for (uint32_t i = 0; i < n_queries; ++i)
    q[i] = dist(gen);
  return q;
}

// ============================================================================
// GPU kernels — separate functions so Nsight can attribute metrics independently
// ============================================================================

__global__ void __launch_bounds__(256)
bitset_query_kernel(const uint32_t* __restrict__ bitset,
                    const uint32_t* __restrict__ queries,
                    uint32_t* __restrict__ results,
                    uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  uint32_t id   = queries[idx];
  uint32_t word = __ldg(&bitset[id >> 5]);
  results[idx]  = (word >> (id & 31)) & 1u;
}

__global__ void __launch_bounds__(256)
roaring_contains_kernel(cu_roaring::GpuRoaringView view,
                        const uint32_t* __restrict__ queries,
                        uint32_t* __restrict__ results,
                        uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

__global__ void __launch_bounds__(256)
roaring_warp_contains_kernel(cu_roaring::GpuRoaringView view,
                             const uint32_t* __restrict__ queries,
                             uint32_t* __restrict__ results,
                             uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  results[idx] = cu_roaring::warp_contains(view, queries[idx]) ? 1u : 0u;
}

// ============================================================================
// Bitmap construction
// ============================================================================
static roaring_bitmap_t* make_bitmap(uint32_t universe, double density,
                                      uint64_t seed)
{
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(seed);

  if (density >= 0.5) {
    roaring_bitmap_add_range(r, 0, universe);
    double remove_rate = 1.0 - density;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (uint32_t key = 0; key < (universe + 65535) / 65536; ++key) {
      uint32_t base = key * 65536u;
      uint32_t end  = std::min(base + 65536u, universe);
      for (uint32_t i = base; i < end; ++i) {
        if (dist(gen) < remove_rate) roaring_bitmap_remove(r, i);
      }
    }
  } else {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (uint32_t i = 0; i < universe; ++i) {
      if (dist(gen) < density) roaring_bitmap_add(r, i);
    }
  }
  roaring_bitmap_run_optimize(r);
  return r;
}

// ============================================================================
// Flat bitset on GPU
// ============================================================================
static uint32_t* make_gpu_bitset(const roaring_bitmap_t* bm, uint32_t universe)
{
  uint32_t n_words = (universe + 31) / 32;
  std::vector<uint32_t> h_bitset(n_words, 0);

  roaring_uint32_iterator_t* iter = roaring_iterator_create(bm);
  while (iter->has_value) {
    uint32_t v = iter->current_value;
    h_bitset[v / 32] |= (1u << (v % 32));
    roaring_uint32_iterator_advance(iter);
  }
  roaring_uint32_iterator_free(iter);

  uint32_t* d_bitset;
  cudaMalloc(&d_bitset, static_cast<size_t>(n_words) * sizeof(uint32_t));
  cudaMemcpy(d_bitset, h_bitset.data(),
             static_cast<size_t>(n_words) * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  return d_bitset;
}

// ============================================================================
// GPU memory accounting for GpuRoaring (actual device footprint)
// ============================================================================
static size_t compute_gpu_roaring_bytes(const cu_roaring::GpuRoaring& bm)
{
  size_t bytes = 0;
  uint32_t n = bm.n_containers;
  bytes += n * sizeof(uint16_t);             // keys
  bytes += n * sizeof(cu_roaring::ContainerType); // types
  bytes += n * sizeof(uint32_t);             // offsets
  bytes += n * sizeof(uint16_t);             // cardinalities
  bytes += static_cast<size_t>(bm.n_bitmap_containers) * 1024 * sizeof(uint64_t); // bitmap_data
  // array_data and run_data sizes aren't directly available from the struct,
  // but after PROMOTE_AUTO at 1B they're zero. Use container counts as proxy.
  // For safety, just count bitmap pool + metadata + key_index.
  if (bm.key_index) {
    bytes += (static_cast<size_t>(bm.max_key) + 1) * sizeof(uint16_t);
  }
  return bytes;
}

// ============================================================================
// L2 cache flush: write to a buffer larger than L2 to evict all lines.
// This lets us compare L2-warm (default) vs L2-cold (flushed) latency
// and compute an effective L2 hit rate without Nsight Compute.
// ============================================================================
static uint32_t* d_l2_flush_buf = nullptr;
static size_t    l2_flush_bytes = 0;

__global__ void flush_l2_kernel(uint32_t* buf, uint32_t n_words)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_words) buf[i] = i;
}

static void init_l2_flush(size_t l2_size_bytes)
{
  // Allocate 2x L2 to ensure full eviction
  l2_flush_bytes = l2_size_bytes * 2;
  cudaMalloc(&d_l2_flush_buf, l2_flush_bytes);
}

static void flush_l2()
{
  if (!d_l2_flush_buf) return;
  uint32_t n = static_cast<uint32_t>(l2_flush_bytes / sizeof(uint32_t));
  flush_l2_kernel<<<(n + 255) / 256, 256>>>(d_l2_flush_buf, n);
  cudaDeviceSynchronize();
}

static void cleanup_l2_flush()
{
  if (d_l2_flush_buf) {
    cudaFree(d_l2_flush_buf);
    d_l2_flush_buf = nullptr;
  }
}

// ============================================================================
// Statistics
// ============================================================================
struct Stats {
  double median, mean, p5, p95, std_dev, min_val, max_val;
};

static Stats compute_stats(std::vector<double>& times)
{
  std::sort(times.begin(), times.end());
  int n      = static_cast<int>(times.size());
  double sum = 0;
  for (auto t : times)
    sum += t;
  double mean = sum / n;
  double var  = 0;
  for (auto t : times)
    var += (t - mean) * (t - mean);
  return {times[n / 2],
          mean,
          times[std::max(0, static_cast<int>(n * 0.05))],
          times[std::min(n - 1, static_cast<int>(n * 0.95))],
          std::sqrt(var / n),
          times[0],
          times[n - 1]};
}

static Stats bench_gpu_kernel(int warmup, int iters, std::function<void()> fn,
                              bool flush_l2_before = false)
{
  cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) {
    if (flush_l2_before) flush_l2();
    fn();
  }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  std::vector<double> times(iters);
  for (int i = 0; i < iters; ++i) {
    if (flush_l2_before) flush_l2();
    cudaEventRecord(s);
    fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    times[i] = ms;
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return compute_stats(times);
}

// ============================================================================
// Correctness
// ============================================================================
static uint32_t verify_results(const uint32_t* d_a, const uint32_t* d_b,
                                uint32_t n)
{
  std::vector<uint32_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < n; ++i)
    if (ha[i] != hb[i]) ++mismatches;
  return mismatches;
}

static uint32_t count_hits(const uint32_t* d_results, uint32_t n)
{
  std::vector<uint32_t> h(n);
  cudaMemcpy(h.data(), d_results, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  uint32_t hits = 0;
  for (uint32_t i = 0; i < n; ++i)
    hits += h[i];
  return hits;
}

// ============================================================================
// Run one density point (shared by normal and ncu modes)
// ============================================================================
struct DensityResult {
  double density;
  const char* label;
  uint64_t cardinality;
  double hit_rate;
  uint32_t n_containers, n_bitmap, n_array, n_run;
  size_t bitset_bytes, roaring_transfer_bytes, roaring_gpu_bytes;
  // L2-warm (normal) measurements
  Stats bs, ct, wc;
  // L2-cold (flushed before each iteration) measurements
  Stats bs_cold, ct_cold, wc_cold;
  bool correct;
};

static DensityResult run_density(double density, const char* label,
                                  uint32_t universe,
                                  const uint32_t* d_queries, uint32_t n_queries,
                                  uint32_t* d_results_bitset,
                                  uint32_t* d_results_contains,
                                  uint32_t* d_results_warp,
                                  int warmup, int iters)
{
  DensityResult res{};
  res.density = density;
  res.label = label;

  printf("Building bitmap 1B @ %s...", label);
  fflush(stdout);

  auto t0 = std::chrono::high_resolution_clock::now();
  roaring_bitmap_t* cpu_bm = make_bitmap(universe, density, 42);
  auto t1 = std::chrono::high_resolution_clock::now();
  double gen_sec = std::chrono::duration<double>(t1 - t0).count();

  res.cardinality = roaring_bitmap_get_cardinality(cpu_bm);
  printf(" card=%lluM (%.1fs)\n",
         static_cast<unsigned long long>(res.cardinality / 1000000), gen_sec);

  // Upload roaring to GPU (PROMOTE_AUTO → all-bitmap at 1B)
  auto gpu_bm = cu_roaring::upload(cpu_bm, universe);
  auto view   = cu_roaring::make_view(gpu_bm);

  res.n_containers = gpu_bm.n_containers;
  res.n_bitmap = gpu_bm.n_bitmap_containers;
  res.n_array = gpu_bm.n_array_containers;
  res.n_run = gpu_bm.n_run_containers;

  // Flat bitset
  uint32_t* d_bitset = make_gpu_bitset(cpu_bm, universe);
  res.bitset_bytes = (static_cast<size_t>(universe) + 31) / 32 * sizeof(uint32_t);

  // CRoaring compressed size (transfer size)
  auto meta = cu_roaring::get_meta(cpu_bm);
  res.roaring_transfer_bytes = meta.total_bytes;

  // Actual GPU device memory footprint
  res.roaring_gpu_bytes = compute_gpu_roaring_bytes(gpu_bm);

  printf("  Memory: bitset=%.1f MB  roaring_gpu=%.1f MB  roaring_transfer=%.1f MB  (%.1fx transfer compression)\n",
         res.bitset_bytes / 1e6, res.roaring_gpu_bytes / 1e6,
         res.roaring_transfer_bytes / 1e6,
         static_cast<double>(res.bitset_bytes) / res.roaring_transfer_bytes);
  printf("  Containers: %u (bmp=%u arr=%u run=%u)\n",
         res.n_containers, res.n_bitmap, res.n_array, res.n_run);

  dim3 block(256);
  dim3 grid((n_queries + 255) / 256);

  // --- Benchmark: L2-warm (normal) ---
  res.bs = bench_gpu_kernel(warmup, iters, [&]() {
    bitset_query_kernel<<<grid, block>>>(
      d_bitset, d_queries, d_results_bitset, n_queries);
  });

  res.ct = bench_gpu_kernel(warmup, iters, [&]() {
    roaring_contains_kernel<<<grid, block>>>(
      view, d_queries, d_results_contains, n_queries);
  });

  res.wc = bench_gpu_kernel(warmup, iters, [&]() {
    roaring_warp_contains_kernel<<<grid, block>>>(
      view, d_queries, d_results_warp, n_queries);
  });

  // --- Benchmark: L2-cold (flush before each iteration) ---
  res.bs_cold = bench_gpu_kernel(warmup, iters, [&]() {
    bitset_query_kernel<<<grid, block>>>(
      d_bitset, d_queries, d_results_bitset, n_queries);
  }, true);

  res.ct_cold = bench_gpu_kernel(warmup, iters, [&]() {
    roaring_contains_kernel<<<grid, block>>>(
      view, d_queries, d_results_contains, n_queries);
  }, true);

  res.wc_cold = bench_gpu_kernel(warmup, iters, [&]() {
    roaring_warp_contains_kernel<<<grid, block>>>(
      view, d_queries, d_results_warp, n_queries);
  }, true);

  // Correctness
  uint32_t mm_ct = verify_results(d_results_bitset, d_results_contains, n_queries);
  uint32_t mm_wc = verify_results(d_results_bitset, d_results_warp, n_queries);
  uint32_t hits  = count_hits(d_results_bitset, n_queries);
  res.correct = (mm_ct == 0 && mm_wc == 0);
  res.hit_rate = static_cast<double>(hits) / n_queries;

  if (!res.correct) {
    printf("  *** CORRECTNESS FAILURE: contains=%u warp=%u ***\n", mm_ct, mm_wc);
  }

  cudaFree(d_bitset);
  cu_roaring::gpu_roaring_free(gpu_bm);
  roaring_bitmap_free(cpu_bm);

  return res;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
  bool ncu_mode = false;
  double density_filter = -1.0;  // -1 means run all

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--ncu") == 0) {
      ncu_mode = true;
    } else if (strcmp(argv[i], "--density") == 0 && i + 1 < argc) {
      density_filter = atof(argv[++i]);
    }
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB, L2=%.0f KB)\n",
         prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0),
         prop.l2CacheSize / 1024.0);

  constexpr uint32_t UNIVERSE = 1000000000u;  // 1B

  const uint32_t N_QUERIES = ncu_mode ? 1000000u : 10000000u;
  const int WARMUP = ncu_mode ? 1 : 10;
  const int ITERS  = ncu_mode ? 1 : 50;

  double   densities[]     = {0.001, 0.01, 0.05, 0.10, 0.25, 0.50};
  const char* density_names[] = {"0.1%", "1%", "5%", "10%", "25%", "50%"};
  constexpr int N_DENSITIES = 6;

  if (ncu_mode) {
    printf("\n*** NCU MODE: 1 iter, %uK queries ***\n", N_QUERIES / 1000);
  }

  // Initialize L2 flush buffer (2x L2 size)
  init_l2_flush(static_cast<size_t>(prop.l2CacheSize));

  // Pre-generate queries (same across all densities)
  printf("Generating %uM random queries for universe %uM...\n",
         N_QUERIES / 1000000, UNIVERSE / 1000000);
  auto h_queries = gen_random_queries(UNIVERSE, N_QUERIES, 777);

  uint32_t* d_queries;
  uint32_t* d_results_bitset;
  uint32_t* d_results_contains;
  uint32_t* d_results_warp;
  cudaMalloc(&d_queries, N_QUERIES * sizeof(uint32_t));
  cudaMalloc(&d_results_bitset, N_QUERIES * sizeof(uint32_t));
  cudaMalloc(&d_results_contains, N_QUERIES * sizeof(uint32_t));
  cudaMalloc(&d_results_warp, N_QUERIES * sizeof(uint32_t));
  cudaMemcpy(d_queries, h_queries.data(),
             N_QUERIES * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Collect results
  std::vector<DensityResult> results;

  for (int di = 0; di < N_DENSITIES; ++di) {
    // If density filter is set, skip non-matching entries
    if (density_filter >= 0.0 && fabs(densities[di] - density_filter) > 1e-6) {
      continue;
    }

    auto res = run_density(densities[di], density_names[di], UNIVERSE,
                           d_queries, N_QUERIES,
                           d_results_bitset, d_results_contains, d_results_warp,
                           WARMUP, ITERS);
    results.push_back(res);
  }

  // Print summary table: L2-warm throughput
  printf("\n=== L2-WARM (steady-state, data cached from prior iterations) ===\n");
  printf("%-8s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-9s %-9s\n",
         "Density", "Bitset", "Gq/s",
         "contains", "Gq/s",
         "warp", "Gq/s",
         "Roar GPU", "Roar Xfer");
  printf("---------|");
  printf("--------------------|");
  printf("--------------------|");
  printf("--------------------|");
  printf("--------------------\n");

  for (auto& r : results) {
    double bs_gqps = N_QUERIES / (r.bs.median * 1e-3) / 1e9;
    double ct_gqps = N_QUERIES / (r.ct.median * 1e-3) / 1e9;
    double wc_gqps = N_QUERIES / (r.wc.median * 1e-3) / 1e9;

    printf("%-8s | %7.3f ms %5.1f | %7.3f ms %5.1f | %7.3f ms %5.1f | %6.1f MB %6.1f MB\n",
           r.label,
           r.bs.median, bs_gqps,
           r.ct.median, ct_gqps,
           r.wc.median, wc_gqps,
           r.roaring_gpu_bytes / 1e6,
           r.roaring_transfer_bytes / 1e6);
  }

  // Print L2-cold throughput
  printf("\n=== L2-COLD (L2 flushed before every iteration) ===\n");
  printf("%-8s | %-10s %-6s | %-10s %-6s | %-10s %-6s\n",
         "Density", "Bitset", "Gq/s",
         "contains", "Gq/s",
         "warp", "Gq/s");
  printf("---------|");
  printf("--------------------|");
  printf("--------------------|");
  printf("--------------------\n");

  for (auto& r : results) {
    double bs_gqps = N_QUERIES / (r.bs_cold.median * 1e-3) / 1e9;
    double ct_gqps = N_QUERIES / (r.ct_cold.median * 1e-3) / 1e9;
    double wc_gqps = N_QUERIES / (r.wc_cold.median * 1e-3) / 1e9;

    printf("%-8s | %7.3f ms %5.1f | %7.3f ms %5.1f | %7.3f ms %5.1f\n",
           r.label,
           r.bs_cold.median, bs_gqps,
           r.ct_cold.median, ct_gqps,
           r.wc_cold.median, wc_gqps);
  }

  // Print effective L2 benefit (warm vs cold speedup)
  // Effective L2 hit rate estimate:
  //   T_warm = T_l1 * hit_l2 + T_dram * (1 - hit_l2)
  //   T_cold ≈ T_dram (all misses)
  //   hit_l2 ≈ 1 - T_warm / T_cold  (when T_l1 << T_dram)
  printf("\n=== CACHE ANALYSIS (warm/cold ratio → effective L2 benefit) ===\n");
  printf("%-8s | %-14s %-10s | %-14s %-10s | %-14s %-10s\n",
         "Density", "Bitset warm/c", "L2 save",
         "contains w/c", "L2 save",
         "warp w/c", "L2 save");
  printf("---------|");
  printf("--------------------------|");
  printf("--------------------------|");
  printf("--------------------------\n");

  for (auto& r : results) {
    double bs_ratio  = r.bs_cold.median / r.bs.median;
    double ct_ratio  = r.ct_cold.median / r.ct.median;
    double wc_ratio  = r.wc_cold.median / r.wc.median;
    // L2 savings: fraction of time saved by having warm L2
    double bs_l2save = 1.0 - r.bs.median / r.bs_cold.median;
    double ct_l2save = 1.0 - r.ct.median / r.ct_cold.median;
    double wc_l2save = 1.0 - r.wc.median / r.wc_cold.median;

    printf("%-8s | %10.2fx     %6.0f%%   | %10.2fx     %6.0f%%   | %10.2fx     %6.0f%%\n",
           r.label,
           bs_ratio, bs_l2save * 100,
           ct_ratio, ct_l2save * 100,
           wc_ratio, wc_l2save * 100);
  }

  // JSON output (skip in ncu mode)
  if (!ncu_mode) {
    const char* path = "results/raw/bench10_cache_profile.json";
    FILE* f = fopen(path, "w");
    if (!f) {
      fprintf(stderr, "Cannot open %s for writing\n", path);
    } else {
      fprintf(f, "{\n  \"benchmark\": \"cache_profile\",\n");
      fprintf(f, "  \"gpu\": \"%s\",\n", prop.name);
      fprintf(f, "  \"n_sms\": %d,\n", prop.multiProcessorCount);
      fprintf(f, "  \"l2_cache_bytes\": %d,\n", prop.l2CacheSize);
      fprintf(f, "  \"universe\": %u,\n", UNIVERSE);
      fprintf(f, "  \"n_queries\": %u,\n", N_QUERIES);
      fprintf(f, "  \"results\": [\n");

      for (size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        double bs_gqps = N_QUERIES / (r.bs.median * 1e-3) / 1e9;
        double ct_gqps = N_QUERIES / (r.ct.median * 1e-3) / 1e9;
        double wc_gqps = N_QUERIES / (r.wc.median * 1e-3) / 1e9;

        if (i > 0) fprintf(f, ",\n");
        fprintf(f, "    {\n");
        fprintf(f, "      \"density\": %.4f, \"density_label\": \"%s\",\n",
                r.density, r.label);
        fprintf(f, "      \"cardinality\": %llu, \"hit_rate\": %.6f,\n",
                static_cast<unsigned long long>(r.cardinality), r.hit_rate);
        fprintf(f, "      \"n_containers\": %u, \"n_bitmap\": %u, \"n_array\": %u, \"n_run\": %u,\n",
                r.n_containers, r.n_bitmap, r.n_array, r.n_run);
        fprintf(f, "      \"bitset_bytes\": %zu, \"roaring_gpu_bytes\": %zu, \"roaring_transfer_bytes\": %zu,\n",
                r.bitset_bytes, r.roaring_gpu_bytes, r.roaring_transfer_bytes);
        fprintf(f, "      \"bitset_ms\": %.4f, \"bitset_gqps\": %.4f,\n",
                r.bs.median, bs_gqps);
        fprintf(f, "      \"contains_ms\": %.4f, \"contains_gqps\": %.4f,\n",
                r.ct.median, ct_gqps);
        fprintf(f, "      \"warp_ms\": %.4f, \"warp_gqps\": %.4f,\n",
                r.wc.median, wc_gqps);
        fprintf(f, "      \"contains_vs_bitset\": %.4f, \"warp_vs_bitset\": %.4f,\n",
                r.bs.median / r.ct.median, r.bs.median / r.wc.median);
        // L2-cold measurements
        double bsc_gqps = N_QUERIES / (r.bs_cold.median * 1e-3) / 1e9;
        double ctc_gqps = N_QUERIES / (r.ct_cold.median * 1e-3) / 1e9;
        double wcc_gqps = N_QUERIES / (r.wc_cold.median * 1e-3) / 1e9;
        fprintf(f, "      \"bitset_cold_ms\": %.4f, \"bitset_cold_gqps\": %.4f,\n",
                r.bs_cold.median, bsc_gqps);
        fprintf(f, "      \"contains_cold_ms\": %.4f, \"contains_cold_gqps\": %.4f,\n",
                r.ct_cold.median, ctc_gqps);
        fprintf(f, "      \"warp_cold_ms\": %.4f, \"warp_cold_gqps\": %.4f,\n",
                r.wc_cold.median, wcc_gqps);
        fprintf(f, "      \"bitset_warm_cold_ratio\": %.4f,\n",
                r.bs_cold.median / r.bs.median);
        fprintf(f, "      \"contains_warm_cold_ratio\": %.4f,\n",
                r.ct_cold.median / r.ct.median);
        fprintf(f, "      \"warp_warm_cold_ratio\": %.4f,\n",
                r.wc_cold.median / r.wc.median);
        fprintf(f, "      \"correctness\": %s\n",
                r.correct ? "true" : "false");
        fprintf(f, "    }");
      }

      fprintf(f, "\n  ]\n}\n");
      fclose(f);
      printf("\nJSON written to %s\n", path);
    }
  }

  cudaFree(d_queries);
  cudaFree(d_results_bitset);
  cudaFree(d_results_contains);
  cudaFree(d_results_warp);
  cleanup_l2_flush();

  // Print Nsight Compute instructions
  printf("\n");
  printf("=== Nsight Compute profiling ===\n");
  printf("Profile a single density + kernel with ncu:\n");
  printf("\n");
  printf("  # 0.1%% density, all kernels:\n");
  printf("  ncu --set full -o results/ncu/d0.1pct \\\n");
  printf("      ./bench/bench_cache_profile --ncu --density 0.001\n");
  printf("\n");
  printf("  # 10%% density, bitset kernel only:\n");
  printf("  ncu --set full --kernel-name bitset_query_kernel \\\n");
  printf("      --launch-skip 1 --launch-count 1 \\\n");
  printf("      -o results/ncu/d10pct_bitset \\\n");
  printf("      ./bench/bench_cache_profile --ncu --density 0.10\n");
  printf("\n");
  printf("Key metrics to check in ncu-ui:\n");
  printf("  l1tex__t_sector_hit_rate.pct    — L1 hit rate\n");
  printf("  lts__t_sector_hit_rate.pct      — L2 hit rate\n");
  printf("  dram__bytes_read.sum            — DRAM traffic\n");
  printf("  dram__throughput.pct            — DRAM bandwidth utilization\n");
  printf("\n");
  printf("Or use the automated script:\n");
  printf("  scripts/ncu_cache_profile.sh\n");

  return 0;
}
