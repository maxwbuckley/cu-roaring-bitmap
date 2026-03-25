/*
 * B11: Selectivity Sweep Benchmark
 *
 * Sweeps selectivity from 0.1% to 99% comparing Roaring bitmaps vs plain
 * bitsets for point query throughput. The key insight this benchmark targets
 * is the crossover point: at what selectivity does the memory savings of
 * Roaring outweigh its per-query overhead vs a flat bitset?
 *
 * Test matrix:
 *   Universe:     {1M, 10M, 100M, 1B}
 *   Selectivity:  {0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%, 30%, 50%, 70%, 90%, 95%, 99%}
 *   Method:       {flat_bitset, contains, warp_contains}
 *   Query pattern: random (uniform) only -- isolates selectivity effect
 *
 * For each (universe, selectivity) pair:
 *   - Creates a roaring bitmap with complement optimization (upload with universe_size)
 *   - Creates an equivalent flat bitset
 *   - Measures point query throughput for all three methods
 *   - Verifies correctness (all methods must agree)
 *   - Reports mean, median, stdev, p5/p95, throughput (Gq/s), speedup ratios
 *
 * Outputs JSON to results/raw/bench11_selectivity_sweep.json
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
#include <functional>
#include <random>
#include <vector>


// ============================================================================
// Query generation (random uniform only -- we want to isolate selectivity)
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
// GPU Kernels
// ============================================================================

// Baseline: flat bitset point query
__global__ void bitset_query_kernel(const uint32_t* __restrict__ bitset,
                                     const uint32_t* __restrict__ queries,
                                     uint32_t* __restrict__ results,
                                     uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  uint32_t id = queries[idx];
  uint32_t word = bitset[id >> 5];
  results[idx] = (word >> (id & 31)) & 1u;
}

// Roaring contains() -- per-thread, no warp cooperation
__global__ void roaring_contains_kernel(cu_roaring::GpuRoaringView view,
                                         const uint32_t* __restrict__ queries,
                                         uint32_t* __restrict__ results,
                                         uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

// Roaring warp_contains() -- warp-cooperative
__global__ void roaring_warp_contains_kernel(cu_roaring::GpuRoaringView view,
                                              const uint32_t* __restrict__ queries,
                                              uint32_t* __restrict__ results,
                                              uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  results[idx] = cu_roaring::warp_contains(view, queries[idx]) ? 1u : 0u;
}

// ============================================================================
// Bitmap generation
// ============================================================================

static roaring_bitmap_t* make_bitmap(uint32_t universe, double density,
                                      uint64_t seed)
{
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(seed);

  if (density >= 0.5) {
    // For high density: start full, remove elements
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
    // For low density: add elements
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (uint32_t i = 0; i < universe; ++i) {
      if (dist(gen) < density) roaring_bitmap_add(r, i);
    }
  }
  roaring_bitmap_run_optimize(r);
  return r;
}

// ============================================================================
// Flat bitset construction from Roaring
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

  uint32_t* d_bitset = nullptr;
  cudaMalloc(&d_bitset, static_cast<size_t>(n_words) * sizeof(uint32_t));
  cudaMemcpy(d_bitset, h_bitset.data(),
             static_cast<size_t>(n_words) * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  return d_bitset;
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

static void write_stats(FILE* f, const char* name, const Stats& s)
{
  fprintf(f,
          "\"%s\": {\"median\": %.4f, \"mean\": %.4f, \"p5\": %.4f, "
          "\"p95\": %.4f, \"std\": %.4f, \"min\": %.4f, \"max\": %.4f}",
          name, s.median, s.mean, s.p5, s.p95, s.std_dev, s.min_val, s.max_val);
}

// GPU timing helper: returns Stats in milliseconds
static Stats bench_gpu_kernel(int warmup, int iters, std::function<void()> fn)
{
  cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i)
    fn();
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  std::vector<double> times(iters);
  for (int i = 0; i < iters; ++i) {
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
// Correctness verification
// ============================================================================

static uint32_t verify_results(const uint32_t* d_results_a,
                                const uint32_t* d_results_b,
                                uint32_t n)
{
  std::vector<uint32_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_results_a, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_results_b, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
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
// Main benchmark
// ============================================================================

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB)\n",
         prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0));

  // Ensure output directory exists
  const char* path = "results/raw/bench11_selectivity_sweep.json";
  FILE* f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Cannot open %s for writing\n", path);
    return 1;
  }

  fprintf(f, "{\n  \"benchmark\": \"selectivity_sweep\",\n");
  fprintf(f, "  \"gpu\": \"%s\",\n", prop.name);
  fprintf(f, "  \"n_sms\": %d,\n", prop.multiProcessorCount);
  fprintf(f, "  \"description\": \"Point query throughput vs selectivity (Roaring vs bitset)\",\n");
  fprintf(f, "  \"results\": [\n");

  // --- Test matrix ---

  uint32_t universes[] = {1000000, 10000000, 100000000, 1000000000};
  const int n_universes = sizeof(universes) / sizeof(universes[0]);

  struct SelectivityPoint {
    double   density;
    const char* label;
  };

  SelectivityPoint selectivities[] = {
    {0.001,  "0.1%"},
    {0.005,  "0.5%"},
    {0.01,   "1%"},
    {0.02,   "2%"},
    {0.05,   "5%"},
    {0.10,   "10%"},
    {0.20,   "20%"},
    {0.30,   "30%"},
    {0.50,   "50%"},
    {0.70,   "70%"},
    {0.90,   "90%"},
    {0.95,   "95%"},
    {0.99,   "99%"},
  };
  const int n_selectivities = sizeof(selectivities) / sizeof(selectivities[0]);

  constexpr uint32_t N_QUERIES = 10000000;  // 10M queries per measurement
  constexpr int WARMUP = 10;
  constexpr int ITERS  = 50;

  // Determine available GPU memory to skip configurations that won't fit.
  // The bitset for 1B universe is 125MB alone; we need query + result buffers too.
  size_t free_mem = 0;
  size_t total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("GPU memory: %.0f MB free / %.0f MB total\n\n",
         free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));

  bool first_result = true;

  for (int ui = 0; ui < n_universes; ++ui) {
    uint32_t U = universes[ui];

    // Estimate minimum memory: bitset + queries + 3 result buffers
    size_t bitset_bytes_est = (static_cast<size_t>(U) + 31) / 32 * sizeof(uint32_t);
    size_t query_bytes = static_cast<size_t>(N_QUERIES) * sizeof(uint32_t);
    size_t result_bytes = 3 * query_bytes;  // 3 result buffers
    size_t min_needed = bitset_bytes_est + query_bytes + result_bytes;
    // Add ~50% headroom for roaring structures
    min_needed = min_needed * 3 / 2;

    if (min_needed > free_mem) {
      printf("=== SKIP U=%uM: needs ~%.0fMB, only %.0fMB free ===\n\n",
             U / 1000000, min_needed / (1024.0 * 1024.0),
             free_mem / (1024.0 * 1024.0));
      continue;
    }

    printf("=== Universe=%uM ===\n", U / 1000000);

    // Pre-generate queries (same for all selectivities at this universe size)
    auto h_queries = gen_random_queries(U, N_QUERIES, 777);
    uint32_t* d_queries = nullptr;
    cudaMalloc(&d_queries, query_bytes);
    cudaMemcpy(d_queries, h_queries.data(), query_bytes, cudaMemcpyHostToDevice);

    // Allocate result buffers (reused across selectivities)
    uint32_t* d_results_bitset   = nullptr;
    uint32_t* d_results_contains = nullptr;
    uint32_t* d_results_warp     = nullptr;
    cudaMalloc(&d_results_bitset,   N_QUERIES * sizeof(uint32_t));
    cudaMalloc(&d_results_contains, N_QUERIES * sizeof(uint32_t));
    cudaMalloc(&d_results_warp,     N_QUERIES * sizeof(uint32_t));

    dim3 block(256);
    dim3 grid((N_QUERIES + 255) / 256);

    for (int si = 0; si < n_selectivities; ++si) {
      double d = selectivities[si].density;
      const char* label = selectivities[si].label;

      printf("  selectivity=%s: ", label);
      fflush(stdout);

      // Generate bitmap
      auto t0 = std::chrono::high_resolution_clock::now();
      roaring_bitmap_t* cpu_bm = make_bitmap(U, d, 42);
      auto t1 = std::chrono::high_resolution_clock::now();
      double gen_sec = std::chrono::duration<double>(t1 - t0).count();

      uint64_t card = roaring_bitmap_get_cardinality(cpu_bm);
      double actual_density = static_cast<double>(card) / U;

      // Upload to GPU with complement optimization (passes universe size)
      auto gpu_bm = cu_roaring::upload(cpu_bm, U);
      auto view = cu_roaring::make_view(gpu_bm);

      // Create flat bitset
      uint32_t* d_bitset = make_gpu_bitset(cpu_bm, U);
      size_t bitset_bytes = (static_cast<size_t>(U) + 31) / 32 * sizeof(uint32_t);

      // Roaring GPU memory estimate
      auto meta = cu_roaring::get_meta(cpu_bm);
      size_t roaring_bytes = meta.total_bytes;

      printf("card=%llu (%.2f%%) gen=%.1fs negated=%s ",
             (unsigned long long)card, actual_density * 100.0, gen_sec,
             gpu_bm.negated ? "yes" : "no");
      printf("containers=%u (bmp=%u arr=%u run=%u)\n",
             gpu_bm.n_containers, gpu_bm.n_bitmap_containers,
             gpu_bm.n_array_containers, gpu_bm.n_run_containers);

      // --- Flat bitset ---
      auto bs_stats = bench_gpu_kernel(WARMUP, ITERS, [&]() {
        bitset_query_kernel<<<grid, block>>>(
          d_bitset, d_queries, d_results_bitset, N_QUERIES);
      });

      // --- Roaring contains() ---
      auto ct_stats = bench_gpu_kernel(WARMUP, ITERS, [&]() {
        roaring_contains_kernel<<<grid, block>>>(
          view, d_queries, d_results_contains, N_QUERIES);
      });

      // --- Roaring warp_contains() ---
      auto wc_stats = bench_gpu_kernel(WARMUP, ITERS, [&]() {
        roaring_warp_contains_kernel<<<grid, block>>>(
          view, d_queries, d_results_warp, N_QUERIES);
      });

      // Correctness: all methods must match bitset
      uint32_t mm_ct = verify_results(d_results_bitset, d_results_contains, N_QUERIES);
      uint32_t mm_wc = verify_results(d_results_bitset, d_results_warp, N_QUERIES);
      uint32_t hits  = count_hits(d_results_bitset, N_QUERIES);

      if (mm_ct > 0 || mm_wc > 0) {
        printf("    *** CORRECTNESS FAILURE: contains=%u warp=%u ***\n",
               mm_ct, mm_wc);
      }

      double hit_rate = static_cast<double>(hits) / N_QUERIES;

      // Throughput in billion queries/sec
      double bs_gqps = N_QUERIES / (bs_stats.median * 1e-3) / 1e9;
      double ct_gqps = N_QUERIES / (ct_stats.median * 1e-3) / 1e9;
      double wc_gqps = N_QUERIES / (wc_stats.median * 1e-3) / 1e9;

      // Speedup: >1 means roaring is faster than bitset
      double contains_speedup = bs_stats.median / ct_stats.median;
      double warp_speedup     = bs_stats.median / wc_stats.median;

      printf("    bitset:         %.3f ms (%.2f Gq/s) std=%.4f\n",
             bs_stats.median, bs_gqps, bs_stats.std_dev);
      printf("    contains:       %.3f ms (%.2f Gq/s) %.2fx vs bitset  std=%.4f\n",
             ct_stats.median, ct_gqps, contains_speedup, ct_stats.std_dev);
      printf("    warp_contains:  %.3f ms (%.2f Gq/s) %.2fx vs bitset  std=%.4f\n",
             wc_stats.median, wc_gqps, warp_speedup, wc_stats.std_dev);
      printf("    hit_rate=%.4f compression=%.1fx mem: bitset=%.2fMB roaring=%.2fMB\n",
             hit_rate,
             static_cast<double>(bitset_bytes) / std::max(roaring_bytes, static_cast<size_t>(1)),
             bitset_bytes / 1e6, roaring_bytes / 1e6);

      // Write JSON
      if (!first_result) fprintf(f, ",\n");
      first_result = false;

      fprintf(f, "    {\n");
      fprintf(f, "      \"universe\": %u,\n", U);
      fprintf(f, "      \"selectivity\": %.4f, \"selectivity_label\": \"%s\",\n",
              d, label);
      fprintf(f, "      \"n_queries\": %u,\n", N_QUERIES);
      fprintf(f, "      \"cardinality\": %llu, \"actual_density\": %.6f,\n",
              (unsigned long long)card, actual_density);
      fprintf(f, "      \"hit_rate\": %.6f,\n", hit_rate);
      fprintf(f, "      \"negated\": %s,\n", gpu_bm.negated ? "true" : "false");
      fprintf(f, "      \"n_containers\": %u, \"n_bitmap\": %u, "
              "\"n_array\": %u, \"n_run\": %u,\n",
              gpu_bm.n_containers, gpu_bm.n_bitmap_containers,
              gpu_bm.n_array_containers, gpu_bm.n_run_containers);
      fprintf(f, "      \"bitset_bytes\": %zu, \"roaring_bytes\": %zu,\n",
              bitset_bytes, roaring_bytes);
      fprintf(f, "      \"compression_ratio\": %.2f,\n",
              static_cast<double>(bitset_bytes) /
                std::max(roaring_bytes, static_cast<size_t>(1)));
      fprintf(f, "      ");
      write_stats(f, "bitset_ms", bs_stats);
      fprintf(f, ",\n      ");
      write_stats(f, "contains_ms", ct_stats);
      fprintf(f, ",\n      ");
      write_stats(f, "warp_contains_ms", wc_stats);
      fprintf(f, ",\n");
      fprintf(f, "      \"bitset_gqps\": %.4f, \"contains_gqps\": %.4f, "
              "\"warp_gqps\": %.4f,\n",
              bs_gqps, ct_gqps, wc_gqps);
      fprintf(f, "      \"contains_vs_bitset\": %.4f, \"warp_vs_bitset\": %.4f,\n",
              ct_stats.median / bs_stats.median,
              wc_stats.median / bs_stats.median);
      fprintf(f, "      \"contains_speedup\": %.4f, \"warp_speedup\": %.4f,\n",
              contains_speedup, warp_speedup);
      fprintf(f, "      \"warp_vs_contains\": %.4f,\n",
              ct_stats.median / wc_stats.median);
      fprintf(f, "      \"bitmap_gen_sec\": %.2f,\n", gen_sec);
      fprintf(f, "      \"correctness\": %s\n",
              (mm_ct == 0 && mm_wc == 0) ? "true" : "false");
      fprintf(f, "    }");

      // Cleanup per-selectivity allocations
      cudaFree(d_bitset);
      cu_roaring::gpu_roaring_free(gpu_bm);
      roaring_bitmap_free(cpu_bm);
    }

    // Cleanup per-universe allocations
    cudaFree(d_queries);
    cudaFree(d_results_bitset);
    cudaFree(d_results_contains);
    cudaFree(d_results_warp);

    printf("\n");
  }

  fprintf(f, "\n  ]\n}\n");
  fclose(f);

  printf("=== B11 COMPLETE -- results written to %s ===\n", path);
  return 0;
}
