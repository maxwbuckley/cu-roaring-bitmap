/*
 * Comprehensive GPU Roaring Bitmap benchmark suite.
 * Covers: kernel microbenchmarks (B5), filter construction (B1),
 * memory footprint (B3), E2E latency (B4).
 * Outputs JSON to results/raw/ for figure generation.
 */

#include <cuda_runtime.h>
#include <roaring/roaring.h>
#include "cu_roaring/cu_roaring.cuh"

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
// Generate a Roaring bitmap with given density (efficient for large universe)
// ============================================================================
static roaring_bitmap_t* make_bitmap(uint32_t universe, double density, uint64_t seed)
{
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(seed);

  if (density >= 0.5) {
    // For high density: start full, remove elements
    roaring_bitmap_add_range(r, 0, universe);
    double remove_rate = 1.0 - density;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    // Remove in batches per container for efficiency
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

// Generate Zipfian tag bitmaps efficiently: each tag gets a random density
// drawn from a Zipfian distribution, then we generate a bitmap at that density.
struct TagSet {
  std::vector<roaring_bitmap_t*> bitmaps;
  uint32_t universe;
  ~TagSet()
  {
    for (auto* b : bitmaps)
      if (b) roaring_bitmap_free(b);
  }
};

static TagSet make_zipf_tags(uint32_t universe, uint32_t n_tags, double alpha, uint64_t seed)
{
  TagSet ts;
  ts.universe = universe;
  ts.bitmaps.resize(n_tags);

  // Zipfian densities: tag rank i gets density proportional to 1/i^alpha
  // Normalize so most popular tag has ~80% density, least popular ~0.01%
  std::vector<double> densities(n_tags);
  double max_rank_weight = 1.0;
  double min_rank_weight = 1.0 / std::pow(static_cast<double>(n_tags), alpha);
  for (uint32_t i = 0; i < n_tags; ++i) {
    double weight  = 1.0 / std::pow(static_cast<double>(i + 1), alpha);
    densities[i]   = 0.0001 + (0.80 - 0.0001) * (weight - min_rank_weight) /
                                                    (max_rank_weight - min_rank_weight);
  }

  for (uint32_t i = 0; i < n_tags; ++i) {
    ts.bitmaps[i] = make_bitmap(universe, densities[i], seed + i);
  }
  return ts;
}

// ============================================================================
// Timing utilities
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

static Stats bench_gpu(int warmup, int iters, std::function<void()> fn)
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

static Stats bench_cpu(int warmup, int iters, std::function<void()> fn)
{
  for (int i = 0; i < warmup; ++i)
    fn();
  std::vector<double> times(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0  = std::chrono::high_resolution_clock::now();
    fn();
    auto t1  = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return compute_stats(times);
}

static void write_stats(FILE* f, const char* name, const Stats& s)
{
  fprintf(f,
          "      \"%s\": {\"median\": %.4f, \"mean\": %.4f, \"p5\": %.4f, "
          "\"p95\": %.4f, \"std\": %.4f, \"min\": %.4f, \"max\": %.4f}",
          name, s.median, s.mean, s.p5, s.p95, s.std_dev, s.min_val, s.max_val);
}

// ============================================================================
// BENCHMARK 5: Kernel Microbenchmarks
// ============================================================================
static void run_bench5(const char* path)
{
  printf("\n=== B5: Kernel Microbenchmarks ===\n");
  FILE* f = fopen(path, "w");
  fprintf(f, "{\n  \"benchmark\": \"kernel_micro\",\n  \"results\": [\n");

  uint32_t universes[] = {10000000, 100000000, 1000000000};
  double densities[]   = {0.01, 0.10, 0.50};
  bool first           = true;

  for (auto U : universes) {
    for (auto d : densities) {
      printf("  U=%uM d=%.0f%%: ", U / 1000000, d * 100);
      fflush(stdout);

      auto* cpu_a = make_bitmap(U, d, 42);
      auto* cpu_b = make_bitmap(U, d * 0.7, 123);
      auto gpu_a  = cu_roaring::upload(cpu_a, U);
      auto gpu_b  = cu_roaring::upload(cpu_b, U);

      uint32_t nw = (U + 31) / 32;
      uint32_t* d_out;
      cudaMalloc(&d_out, nw * sizeof(uint32_t));

      auto ds = bench_gpu(10, 100, [&]() {
        cu_roaring::decompress_to_bitset(gpu_a, d_out, nw);
      });

      double bw = (nw * 4.0) / (ds.median * 1e-3) / 1e9;

      auto as = bench_gpu(10, 50, [&]() {
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cu_roaring::gpu_roaring_free(r);
      });

      printf("decomp=%.3fms (%.0f GB/s, %.0f%% peak)  AND=%.3fms  cont=%u\n",
             ds.median, bw, bw / 1800 * 100, as.median, gpu_a.n_containers);

      if (!first) fprintf(f, ",\n");
      first = false;
      fprintf(f, "    {\"universe\": %u, \"density\": %.4f, ", U, d);
      fprintf(f, "\"n_containers\": %u, \"n_bmp\": %u, \"n_arr\": %u, ",
              gpu_a.n_containers, gpu_a.n_bitmap_containers, gpu_a.n_array_containers);
      fprintf(f, "\"decomp_median_ms\": %.4f, \"decomp_bw_gbs\": %.1f, ", ds.median, bw);
      fprintf(f, "\"and_median_ms\": %.4f}", as.median);

      cudaFree(d_out);
      cu_roaring::gpu_roaring_free(gpu_a);
      cu_roaring::gpu_roaring_free(gpu_b);
      roaring_bitmap_free(cpu_a);
      roaring_bitmap_free(cpu_b);
    }
  }
  fprintf(f, "\n  ]\n}\n");
  fclose(f);
}

// ============================================================================
// BENCHMARK 1: Filter Construction Latency vs Predicate Count
// ============================================================================
static void run_bench1(const char* path)
{
  printf("\n=== B1: Filter Construction vs Predicate Count ===\n");
  FILE* f = fopen(path, "w");
  fprintf(f, "{\n  \"benchmark\": \"filter_construction\",\n  \"results\": [\n");

  uint32_t universes[]   = {10000000, 100000000, 1000000000};
  uint32_t preds[]       = {1, 2, 4, 6, 8};
  constexpr int W = 10, N = 100;
  bool first = true;

  for (auto U : universes) {
    printf("\n  Universe=%uM: generating 8 tag bitmaps (Zipf)...\n", U / 1000000);
    fflush(stdout);

    auto t0   = std::chrono::high_resolution_clock::now();
    auto tags = make_zipf_tags(U, 8, 1.2, 42);
    auto t1   = std::chrono::high_resolution_clock::now();
    printf("  Generated in %.1fs\n",
           std::chrono::duration<double>(t1 - t0).count());

    for (int i = 0; i < 8; ++i) {
      printf("    Tag %d: card=%llu (%.1f%%)\n", i,
             (unsigned long long)roaring_bitmap_get_cardinality(tags.bitmaps[i]),
             100.0 * roaring_bitmap_get_cardinality(tags.bitmaps[i]) / U);
    }

    // Upload to GPU
    std::vector<cu_roaring::GpuRoaring> gpu;
    for (int i = 0; i < 8; ++i)
      gpu.push_back(cu_roaring::upload(tags.bitmaps[i], U));

    for (auto np : preds) {
      printf("  np=%u: ", np);
      fflush(stdout);

      // CPU
      auto cs = bench_cpu(W, N, [&]() {
        roaring_bitmap_t* r = roaring_bitmap_copy(tags.bitmaps[0]);
        for (uint32_t i = 1; i < np; ++i)
          roaring_bitmap_and_inplace(r, tags.bitmaps[i]);
        roaring_bitmap_free(r);
      });

      // GPU pre-resident (set ops only)
      auto gs = bench_gpu(W, N, [&]() {
        if (np == 1) {
          // Nothing to do for single predicate
          return;
        }
        auto r = cu_roaring::set_operation(gpu[0], gpu[1], cu_roaring::SetOp::AND);
        for (uint32_t i = 2; i < np; ++i) {
          auto n = cu_roaring::set_operation(r, gpu[i], cu_roaring::SetOp::AND);
          cu_roaring::gpu_roaring_free(r);
          r = n;
        }
        cu_roaring::gpu_roaring_free(r);
      });

      // GPU full pipeline (set ops + decompress)
      auto fs = bench_gpu(W, N, [&]() {
        cu_roaring::GpuRoaring r;
        if (np == 1) {
          // Just decompress
          auto* bs    = cu_roaring::decompress_to_bitset(gpu[0]);
          cudaFree(bs);
          return;
        }
        r = cu_roaring::set_operation(gpu[0], gpu[1], cu_roaring::SetOp::AND);
        for (uint32_t i = 2; i < np; ++i) {
          auto n = cu_roaring::set_operation(r, gpu[i], cu_roaring::SetOp::AND);
          cu_roaring::gpu_roaring_free(r);
          r = n;
        }
        auto* bs = cu_roaring::decompress_to_bitset(r);
        cudaFree(bs);
        cu_roaring::gpu_roaring_free(r);
      });

      // Correctness
      roaring_bitmap_t* cr = roaring_bitmap_copy(tags.bitmaps[0]);
      for (uint32_t i = 1; i < np; ++i)
        roaring_bitmap_and_inplace(cr, tags.bitmaps[i]);
      uint64_t card = roaring_bitmap_get_cardinality(cr);
      roaring_bitmap_free(cr);

      double spd = cs.median / (np == 1 ? fs.median : gs.median);
      printf("CPU=%.2f GPU_ops=%.2f GPU_full=%.2f spd=%.1fx card=%llu\n",
             cs.median, gs.median, fs.median, spd, (unsigned long long)card);

      if (!first) fprintf(f, ",\n");
      first = false;
      fprintf(f, "    {\n      \"universe\": %u, \"n_preds\": %u, \"card\": %llu,\n",
              U, np, (unsigned long long)card);
      write_stats(f, "cpu_ms", cs);
      fprintf(f, ",\n");
      write_stats(f, "gpu_ops_ms", gs);
      fprintf(f, ",\n");
      write_stats(f, "gpu_full_ms", fs);
      fprintf(f, ",\n      \"speedup\": %.2f\n    }", spd);
    }

    for (auto& g : gpu)
      cu_roaring::gpu_roaring_free(g);
  }

  fprintf(f, "\n  ]\n}\n");
  fclose(f);
}

// ============================================================================
// BENCHMARK 3: Memory Footprint
// ============================================================================
static void run_bench3(const char* path)
{
  printf("\n=== B3: Memory Footprint ===\n");
  FILE* f = fopen(path, "w");
  fprintf(f, "{\n  \"benchmark\": \"memory_footprint\",\n  \"results\": [\n");

  // At 1B scale, compute theoretical sizes for various tag counts
  // Use measured compression ratios from real bitmaps at different densities
  struct DensityBucket {
    const char* name;
    double density;
    double compression_ratio;  // measured from roaring
  };

  // Generate real bitmaps at 100M to measure actual compression
  uint32_t U            = 100000000;
  DensityBucket buckets[] = {
    {"rare (0.1%)", 0.001, 0},
    {"uncommon (1%)", 0.01, 0},
    {"medium (10%)", 0.10, 0},
    {"common (30%)", 0.30, 0},
    {"very_common (50%)", 0.50, 0},
    {"dominant (80%)", 0.80, 0},
  };

  printf("  Measuring compression at 100M:\n");
  for (auto& b : buckets) {
    auto* bm       = make_bitmap(U, b.density, 42);
    size_t flat     = (static_cast<size_t>(U) + 7) / 8;
    size_t roaring  = roaring_bitmap_portable_size_in_bytes(bm);
    b.compression_ratio = static_cast<double>(flat) / roaring;
    printf("    %s: flat=%.1fMB roaring=%.1fMB ratio=%.1fx\n",
           b.name, flat / 1e6, roaring / 1e6, b.compression_ratio);
    roaring_bitmap_free(bm);
  }

  bool first          = true;
  uint32_t tag_counts[] = {10, 100, 500, 1000, 5000};
  uint32_t scale_U      = 1000000000;  // 1B

  printf("\n  Projected memory at 1B (using measured compression):\n");
  for (auto n : tag_counts) {
    // Assume Zipfian: ~20% rare, 30% uncommon, 25% medium, 15% common, 8% very_common, 2% dominant
    double fractions[] = {0.20, 0.30, 0.25, 0.15, 0.08, 0.02};
    size_t flat_bytes = static_cast<size_t>(n) * ((scale_U + 7) / 8);
    size_t roaring_bytes = 0;
    for (int b = 0; b < 6; ++b) {
      uint32_t count = static_cast<uint32_t>(n * fractions[b]);
      size_t per_flat = (scale_U + 7) / 8;
      roaring_bytes += static_cast<size_t>(count * per_flat / buckets[b].compression_ratio);
    }

    bool flat_fits    = flat_bytes < 32ULL * 1024 * 1024 * 1024;
    bool roaring_fits = roaring_bytes < 32ULL * 1024 * 1024 * 1024;
    printf("  %5u tags: flat=%.1fGB roaring=%.1fGB ratio=%.1fx flat_32GB=%s roaring_32GB=%s\n",
           n, flat_bytes / 1e9, roaring_bytes / 1e9,
           (double)flat_bytes / roaring_bytes,
           flat_fits ? "yes" : "NO", roaring_fits ? "yes" : "NO");

    if (!first) fprintf(f, ",\n");
    first = false;
    fprintf(f,
            "    {\"universe\": %u, \"n_tags\": %u, \"flat_bytes\": %zu, "
            "\"roaring_bytes\": %zu, \"ratio\": %.2f, \"flat_fits_32gb\": %s, "
            "\"roaring_fits_32gb\": %s}",
            scale_U, n, flat_bytes, roaring_bytes,
            (double)flat_bytes / roaring_bytes,
            flat_fits ? "true" : "false", roaring_fits ? "true" : "false");
  }

  fprintf(f, "\n  ]\n}\n");
  fclose(f);
}

// ============================================================================
// BENCHMARK 4: End-to-End Latency Breakdown
// ============================================================================
static void run_bench4(const char* path)
{
  printf("\n=== B4: E2E Latency Breakdown (4 predicates) ===\n");
  FILE* f = fopen(path, "w");
  fprintf(f, "{\n  \"benchmark\": \"e2e_latency\",\n  \"results\": [\n");

  uint32_t universes[] = {10000000, 100000000, 1000000000};
  constexpr int W = 10, N = 100;
  constexpr double SEARCH_MS = 10.0;
  bool first = true;

  for (auto U : universes) {
    printf("\n  U=%uM, 4 predicates:\n", U / 1000000);
    fflush(stdout);

    // Generate 4 bitmaps with varied density
    double dens[] = {0.60, 0.20, 0.10, 0.05};
    std::vector<roaring_bitmap_t*> cpus;
    std::vector<cu_roaring::GpuRoaring> gpus;
    for (int i = 0; i < 4; ++i) {
      auto* bm = make_bitmap(U, dens[i], 42 + i * 100);
      cpus.push_back(bm);
      gpus.push_back(cu_roaring::upload(bm, U));
    }

    uint32_t nw = (U + 31) / 32;

    // Pipeline A: CPU filter construction
    auto t_cpu = bench_cpu(W, N, [&]() {
      auto* r = roaring_bitmap_copy(cpus[0]);
      for (int i = 1; i < 4; ++i)
        roaring_bitmap_and_inplace(r, cpus[i]);
      roaring_bitmap_free(r);
    });

    // Pipeline A: PCIe transfer of flat bitset
    std::vector<uint32_t> h_flat(nw, 0);
    uint32_t* d_flat;
    cudaMalloc(&d_flat, nw * sizeof(uint32_t));
    auto t_xfer = bench_gpu(W, N, [&]() {
      cudaMemcpy(d_flat, h_flat.data(), nw * sizeof(uint32_t), cudaMemcpyHostToDevice);
    });

    // Pipeline C: GPU kernel (set ops)
    auto t_kern = bench_gpu(W, N, [&]() {
      auto r = cu_roaring::set_operation(gpus[0], gpus[1], cu_roaring::SetOp::AND);
      for (int i = 2; i < 4; ++i) {
        auto n = cu_roaring::set_operation(r, gpus[i], cu_roaring::SetOp::AND);
        cu_roaring::gpu_roaring_free(r);
        r = n;
      }
      cu_roaring::gpu_roaring_free(r);
    });

    // Pipeline C: decompress
    auto t_dec = bench_gpu(W, N, [&]() {
      cu_roaring::decompress_to_bitset(gpus[0], d_flat, nw);
    });

    double A = t_cpu.median + t_xfer.median + SEARCH_MS;
    double C = t_kern.median + t_dec.median + SEARCH_MS;

    printf("  Pipeline A: cpu=%.2f + xfer=%.2f + search=%.1f = %.2f ms\n",
           t_cpu.median, t_xfer.median, SEARCH_MS, A);
    printf("  Pipeline C: kern=%.2f + decomp=%.2f + search=%.1f = %.2f ms\n",
           t_kern.median, t_dec.median, SEARCH_MS, C);
    printf("  Speedup: %.2fx\n", A / C);

    cudaFree(d_flat);

    if (!first) fprintf(f, ",\n");
    first = false;
    fprintf(f, "    {\n      \"universe\": %u,\n", U);
    write_stats(f, "cpu_filter_ms", t_cpu);
    fprintf(f, ",\n");
    write_stats(f, "pcie_transfer_ms", t_xfer);
    fprintf(f, ",\n");
    write_stats(f, "gpu_kernel_ms", t_kern);
    fprintf(f, ",\n");
    write_stats(f, "gpu_decompress_ms", t_dec);
    fprintf(f, ",\n");
    fprintf(f, "      \"sim_search_ms\": %.1f,\n", SEARCH_MS);
    fprintf(f, "      \"total_A_ms\": %.4f, \"total_C_ms\": %.4f, \"speedup\": %.2f\n    }",
            A, C, A / C);

    for (auto& g : gpus)
      cu_roaring::gpu_roaring_free(g);
    for (auto* c : cpus)
      roaring_bitmap_free(c);
  }

  fprintf(f, "\n  ]\n}\n");
  fclose(f);
}

// ============================================================================
int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs, %.0f MB)\n",
         prop.name, prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024.0));

  run_bench5("results/raw/bench5_kernel_micro.json");
  run_bench1("results/raw/bench1_filter_construction.json");
  run_bench3("results/raw/bench3_memory_footprint.json");
  run_bench4("results/raw/bench4_e2e_latency.json");

  printf("\n=== ALL BENCHMARKS COMPLETE ===\n");
  return 0;
}
