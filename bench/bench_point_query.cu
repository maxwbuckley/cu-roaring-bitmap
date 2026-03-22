/*
 * B6: Point Query Throughput Benchmark
 *
 * Measures the core operation for cuVS/CAGRA integration: per-thread and
 * warp-cooperative membership queries against GPU-resident Roaring bitmaps,
 * compared against a flat bitset baseline.
 *
 * Test matrix:
 *   Universe:  {1M, 10M, 100M, 1B}
 *   Density:   {0.1%, 1%, 10%, 50%}
 *   Pattern:   {random, clustered, strided}
 *   Method:    {flat_bitset, contains, warp_contains}
 *
 * Outputs JSON to results/raw/bench6_point_query.json
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
#include <numeric>
#include <random>
#include <vector>


// ============================================================================
// Query generation
// ============================================================================

// Random uniform queries
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

// Clustered queries: groups of 32 consecutive IDs from the same container.
// This is the best case for warp_contains() — all 32 lanes in a warp share
// the same high-16 key, so only 1 binary search per warp instead of 32.
static std::vector<uint32_t> gen_clustered_queries(uint32_t universe,
                                                    uint32_t n_queries,
                                                    uint64_t seed)
{
  std::vector<uint32_t> q(n_queries);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> container_dist(0, (universe >> 16));
  std::uniform_int_distribution<uint16_t> low_dist(0, 65535u);

  for (uint32_t i = 0; i < n_queries; i += 32) {
    uint32_t container_key = container_dist(gen);
    uint32_t base = container_key << 16;
    uint16_t start_low = low_dist(gen);
    for (uint32_t j = 0; j < 32 && (i + j) < n_queries; ++j) {
      // Consecutive low bits within the same container
      uint16_t low = static_cast<uint16_t>((start_low + j) & 0xFFFF);
      q[i + j] = std::min(base | low, universe - 1);
    }
  }
  return q;
}

// Strided queries: simulate graph traversal access pattern.
// Neighbors in a graph index tend to be numerically close (locality) but not
// perfectly consecutive. We pick a random anchor, then generate neighbors
// within a stride window (~1% of universe), with some inter-container spread.
static std::vector<uint32_t> gen_strided_queries(uint32_t universe,
                                                  uint32_t n_queries,
                                                  uint64_t seed)
{
  std::vector<uint32_t> q(n_queries);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> anchor_dist(0, universe - 1);

  // Window size: ~1% of universe, but at least 1 container (65536)
  uint32_t window = std::max(universe / 100u, 65536u);

  for (uint32_t i = 0; i < n_queries; i += 32) {
    uint32_t anchor = anchor_dist(gen);
    uint32_t lo = (anchor > window / 2) ? anchor - window / 2 : 0;
    uint32_t hi = std::min(lo + window, universe - 1);
    std::uniform_int_distribution<uint32_t> nbr_dist(lo, hi);
    for (uint32_t j = 0; j < 32 && (i + j) < n_queries; ++j) {
      q[i + j] = nbr_dist(gen);
    }
  }
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

// Roaring contains() — per-thread, no warp cooperation
__global__ void roaring_contains_kernel(cu_roaring::GpuRoaringView view,
                                         const uint32_t* __restrict__ queries,
                                         uint32_t* __restrict__ results,
                                         uint32_t n_queries)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_queries) return;

  results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

// Roaring warp_contains() — warp-cooperative
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
// Bitmap generation (same as bench_comprehensive but capped for safety)
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

  uint32_t* d_bitset;
  cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));
  cudaMemcpy(d_bitset, h_bitset.data(), n_words * sizeof(uint32_t),
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
          times[std::max(0, (int)(n * 0.05))],
          times[std::min(n - 1, (int)(n * 0.95))],
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

  const char* path = "results/raw/bench6_point_query.json";
  FILE* f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Cannot open %s for writing\n", path);
    return 1;
  }

  fprintf(f, "{\n  \"benchmark\": \"point_query\",\n");
  fprintf(f, "  \"gpu\": \"%s\",\n", prop.name);
  fprintf(f, "  \"n_sms\": %d,\n", prop.multiProcessorCount);
  fprintf(f, "  \"results\": [\n");

  uint32_t universes[]  = {1000000, 10000000, 100000000, 1000000000};
  double   densities[]  = {0.001, 0.01, 0.10, 0.50};
  const char* density_names[] = {"0.1%", "1%", "10%", "50%"};

  struct PatternDef {
    const char* name;
    std::vector<uint32_t> (*gen)(uint32_t, uint32_t, uint64_t);
  };
  PatternDef patterns[] = {
    {"random",    gen_random_queries},
    {"clustered", gen_clustered_queries},
    {"strided",   gen_strided_queries},
  };

  constexpr uint32_t N_QUERIES = 10000000;  // 10M queries
  constexpr int WARMUP = 10;
  constexpr int ITERS  = 50;

  bool first_result = true;

  for (auto U : universes) {
    for (int di = 0; di < 4; ++di) {
      double d = densities[di];

      // Skip 1B + 50% — that's 125MB just for bitset, fine, but bitmap gen takes ages
      // Actually at 1B even 10% takes a while. Let's keep it but warn.
      printf("\n=== U=%uM d=%s ===\n", U / 1000000, density_names[di]);
      fflush(stdout);

      // Generate bitmap
      auto t0 = std::chrono::high_resolution_clock::now();
      roaring_bitmap_t* cpu_bm = make_bitmap(U, d, 42);
      auto t1 = std::chrono::high_resolution_clock::now();
      double gen_sec = std::chrono::duration<double>(t1 - t0).count();

      uint64_t card = roaring_bitmap_get_cardinality(cpu_bm);
      printf("  Bitmap: card=%llu (%.2f%%) gen=%.1fs\n",
             (unsigned long long)card, 100.0 * card / U, gen_sec);

      // Upload to GPU
      auto gpu_bm = cu_roaring::upload(cpu_bm);
      printf("  Containers: %u (bmp=%u arr=%u run=%u)\n",
             gpu_bm.n_containers, gpu_bm.n_bitmap_containers,
             gpu_bm.n_array_containers, gpu_bm.n_run_containers);

      // Create view
      auto view = cu_roaring::make_view(gpu_bm);

      // Create flat bitset
      uint32_t* d_bitset = make_gpu_bitset(cpu_bm, U);
      size_t bitset_bytes = ((size_t)U + 31) / 32 * sizeof(uint32_t);

      // Estimate roaring GPU bytes
      auto meta = cu_roaring::get_meta(cpu_bm);
      size_t roaring_bytes = meta.total_bytes;

      printf("  Memory: bitset=%.2fMB roaring=%.2fMB (%.1fx compression)\n",
             bitset_bytes / 1e6, roaring_bytes / 1e6,
             (double)bitset_bytes / roaring_bytes);

      // Allocate device query/result buffers
      uint32_t* d_queries;
      uint32_t* d_results_bitset;
      uint32_t* d_results_contains;
      uint32_t* d_results_warp;
      cudaMalloc(&d_queries, N_QUERIES * sizeof(uint32_t));
      cudaMalloc(&d_results_bitset, N_QUERIES * sizeof(uint32_t));
      cudaMalloc(&d_results_contains, N_QUERIES * sizeof(uint32_t));
      cudaMalloc(&d_results_warp, N_QUERIES * sizeof(uint32_t));

      dim3 block(256);
      dim3 grid((N_QUERIES + 255) / 256);

      for (auto& pat : patterns) {
        printf("  Pattern: %s\n", pat.name);
        fflush(stdout);

        // Generate queries on CPU, upload
        auto h_queries = pat.gen(U, N_QUERIES, 777);
        cudaMemcpy(d_queries, h_queries.data(),
                   N_QUERIES * sizeof(uint32_t), cudaMemcpyHostToDevice);

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
          printf("  *** CORRECTNESS FAILURE: contains=%u warp=%u ***\n",
                 mm_ct, mm_wc);
        }

        double hit_rate = (double)hits / N_QUERIES;

        // Throughput in billion queries/sec
        double bs_gqps = N_QUERIES / (bs_stats.median * 1e-3) / 1e9;
        double ct_gqps = N_QUERIES / (ct_stats.median * 1e-3) / 1e9;
        double wc_gqps = N_QUERIES / (wc_stats.median * 1e-3) / 1e9;

        printf("    bitset:         %.3f ms (%.2f Gq/s)\n", bs_stats.median, bs_gqps);
        printf("    contains:       %.3f ms (%.2f Gq/s) %.1fx vs bitset\n",
               ct_stats.median, ct_gqps, bs_stats.median / ct_stats.median);
        printf("    warp_contains:  %.3f ms (%.2f Gq/s) %.1fx vs bitset\n",
               wc_stats.median, wc_gqps, bs_stats.median / wc_stats.median);
        printf("    hit_rate=%.3f\n", hit_rate);

        // Write JSON
        if (!first_result) fprintf(f, ",\n");
        first_result = false;

        fprintf(f, "    {\n");
        fprintf(f, "      \"universe\": %u, \"density\": %.4f, \"density_label\": \"%s\",\n",
                U, d, density_names[di]);
        fprintf(f, "      \"pattern\": \"%s\", \"n_queries\": %u,\n", pat.name, N_QUERIES);
        fprintf(f, "      \"cardinality\": %llu, \"hit_rate\": %.6f,\n",
                (unsigned long long)card, hit_rate);
        fprintf(f, "      \"n_containers\": %u, \"n_bitmap\": %u, \"n_array\": %u, \"n_run\": %u,\n",
                gpu_bm.n_containers, gpu_bm.n_bitmap_containers,
                gpu_bm.n_array_containers, gpu_bm.n_run_containers);
        fprintf(f, "      \"bitset_bytes\": %zu, \"roaring_bytes\": %zu,\n",
                bitset_bytes, roaring_bytes);
        fprintf(f, "      \"compression_ratio\": %.2f,\n",
                (double)bitset_bytes / roaring_bytes);
        fprintf(f, "      ");
        write_stats(f, "bitset_ms", bs_stats);
        fprintf(f, ",\n      ");
        write_stats(f, "contains_ms", ct_stats);
        fprintf(f, ",\n      ");
        write_stats(f, "warp_contains_ms", wc_stats);
        fprintf(f, ",\n");

        // Derived metrics
        fprintf(f, "      \"bitset_gqps\": %.4f, \"contains_gqps\": %.4f, "
                "\"warp_gqps\": %.4f,\n",
                bs_gqps, ct_gqps, wc_gqps);
        fprintf(f, "      \"contains_vs_bitset\": %.4f, \"warp_vs_bitset\": %.4f,\n",
                ct_stats.median / bs_stats.median,
                wc_stats.median / bs_stats.median);
        fprintf(f, "      \"warp_vs_contains\": %.4f,\n",
                ct_stats.median / wc_stats.median);
        fprintf(f, "      \"correctness\": %s\n",
                (mm_ct == 0 && mm_wc == 0) ? "true" : "false");
        fprintf(f, "    }");
      }

      // Cleanup for this (universe, density) pair
      cudaFree(d_queries);
      cudaFree(d_results_bitset);
      cudaFree(d_results_contains);
      cudaFree(d_results_warp);
      cudaFree(d_bitset);
      cu_roaring::gpu_roaring_free(gpu_bm);
      roaring_bitmap_free(cpu_bm);
    }
  }

  fprintf(f, "\n  ]\n}\n");
  fclose(f);

  printf("\n=== B6 COMPLETE — results written to %s ===\n", path);
  return 0;
}
