/*
 * B9: Fused vs Pairwise multi-predicate AND
 *
 * Compares multi_and() (now fused for all-bitmap inputs) against
 * the pairwise set_operation chain at 2, 4, 6, 8 predicates.
 */

#include <cuda_runtime.h>
#include <roaring/roaring.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/detail/promote.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

struct Stats {
  double median, mean, std_dev;
};

static Stats compute_stats(std::vector<double>& t)
{
  std::sort(t.begin(), t.end());
  int n = static_cast<int>(t.size());
  double sum = 0;
  for (auto v : t) sum += v;
  double mean = sum / n;
  double var = 0;
  for (auto v : t) var += (v - mean) * (v - mean);
  return {t[n / 2], mean, std::sqrt(var / n)};
}

static Stats bench_gpu(int warmup, int iters, std::function<void()> fn)
{
  cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) fn();
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

static roaring_bitmap_t* make_bitmap(uint32_t universe, double density, uint64_t seed)
{
  roaring_bitmap_t* r = roaring_bitmap_create();
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint32_t i = 0; i < universe; ++i)
    if (dist(gen) < density) roaring_bitmap_add(r, i);
  roaring_bitmap_run_optimize(r);
  return r;
}

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

  FILE* jf = fopen("results/raw/bench9_multi_and.json", "w");
  if (!jf) jf = fopen("bench9_multi_and.json", "w");
  fprintf(jf, "{\n  \"benchmark\": \"multi_and\",\n");
  fprintf(jf, "  \"gpu\": \"%s\",\n  \"results\": [\n", prop.name);

  struct Config {
    uint32_t universe;
    int iters;
  };
  Config configs[] = {
    {  10000000, 50},
    { 100000000, 30},
    {1000000000, 10},
  };
  uint32_t pred_counts[] = {2, 4, 6, 8};

  // Zipfian densities for predicates
  double densities[] = {0.60, 0.30, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005};

  bool first = true;

  for (auto& cfg : configs) {
    printf("=== Universe=%uM ===\n", cfg.universe / 1000000);

    // Generate CPU bitmaps and upload as all-bitmap (for fused path)
    constexpr int MAX_PREDS = 8;
    std::vector<roaring_bitmap_t*> cpu_bms(MAX_PREDS);
    std::vector<cu_roaring::GpuRoaring> gpu_bms(MAX_PREDS);

    for (int i = 0; i < MAX_PREDS; ++i) {
      printf("  Generating predicate %d (density=%.1f%%)...\n", i, densities[i] * 100);
      fflush(stdout);
      cpu_bms[i] = make_bitmap(cfg.universe, densities[i], 42 + i * 100);
      gpu_bms[i] = cu_roaring::upload(cpu_bms[i]);
      printf("    card=%llu containers=%u (bmp=%u arr=%u)\n",
             (unsigned long long)roaring_bitmap_get_cardinality(cpu_bms[i]),
             gpu_bms[i].n_containers,
             gpu_bms[i].n_bitmap_containers,
             gpu_bms[i].n_array_containers);
    }

    for (auto np : pred_counts) {
      printf("  %u predicates:\n", np);
      fflush(stdout);

      // --- Pairwise chain (old path) ---
      auto s_pairwise = bench_gpu(5, cfg.iters, [&]() {
        auto r = cu_roaring::set_operation(gpu_bms[0], gpu_bms[1], cu_roaring::SetOp::AND);
        for (uint32_t i = 2; i < np; ++i) {
          auto n = cu_roaring::set_operation(r, gpu_bms[i], cu_roaring::SetOp::AND);
          cu_roaring::gpu_roaring_free(r);
          r = n;
        }
        cu_roaring::gpu_roaring_free(r);
      });

      // --- Fused multi_and (new path) ---
      auto s_fused = bench_gpu(5, cfg.iters, [&]() {
        auto r = cu_roaring::multi_and(gpu_bms.data(), np);
        cu_roaring::gpu_roaring_free(r);
      });

      // --- CPU baseline ---
      std::vector<double> cpu_times;
      for (int i = 0; i < std::min(cfg.iters, 20); ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto* r = roaring_bitmap_copy(cpu_bms[0]);
        for (uint32_t j = 1; j < np; ++j)
          roaring_bitmap_and_inplace(r, cpu_bms[j]);
        roaring_bitmap_free(r);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      auto s_cpu = compute_stats(cpu_times);

      double speedup = s_pairwise.median / s_fused.median;
      double vs_cpu = s_cpu.median / s_fused.median;

      printf("    CPU:      %8.2f ms\n", s_cpu.median);
      printf("    Pairwise: %8.2f ms\n", s_pairwise.median);
      printf("    Fused:    %8.2f ms  (%.1fx vs pairwise, %.1fx vs CPU)\n\n",
             s_fused.median, speedup, vs_cpu);

      if (!first) fprintf(jf, ",\n");
      first = false;
      fprintf(jf, "    {\"universe\": %u, \"n_preds\": %u,\n", cfg.universe, np);
      fprintf(jf, "     \"cpu_median_ms\": %.4f,\n", s_cpu.median);
      fprintf(jf, "     \"pairwise_median_ms\": %.4f, \"pairwise_std_ms\": %.4f,\n",
              s_pairwise.median, s_pairwise.std_dev);
      fprintf(jf, "     \"fused_median_ms\": %.4f, \"fused_std_ms\": %.4f,\n",
              s_fused.median, s_fused.std_dev);
      fprintf(jf, "     \"fused_vs_pairwise\": %.2f, \"fused_vs_cpu\": %.2f}",
              speedup, vs_cpu);
    }

    for (int i = 0; i < MAX_PREDS; ++i) {
      cu_roaring::gpu_roaring_free(gpu_bms[i]);
      roaring_bitmap_free(cpu_bms[i]);
    }
  }

  fprintf(jf, "\n  ]\n}\n");
  fclose(jf);
  printf("=== B9 COMPLETE ===\n");
  return 0;
}
