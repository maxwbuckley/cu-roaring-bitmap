#include <benchmark/benchmark.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "data_gen.cuh"

static void BM_GPU_Decompress(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    double density = state.range(1) / 1000.0;  // e.g., 100 → 0.1

    auto* cpu_bm = cu_roaring::bench::generate_bitmap(universe, density, 42);
    auto gpu_bm = cu_roaring::upload(cpu_bm);

    uint32_t n_words = (gpu_bm.universe_size + 31) / 32;
    uint32_t* d_output = nullptr;
    cudaMalloc(&d_output, n_words * sizeof(uint32_t));

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cu_roaring::decompress_to_bitset(gpu_bm, d_output, n_words);
        cudaDeviceSynchronize();
    }

    for (auto _ : state) {
        cudaEventRecord(start_ev);
        cu_roaring::decompress_to_bitset(gpu_bm, d_output, n_words);
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);
    }

    // Report metrics
    double flat_size_mb = n_words * sizeof(uint32_t) / (1024.0 * 1024.0);
    state.counters["flat_MB"] = flat_size_mb;
    state.counters["density"] = density;
    state.counters["n_containers"] = gpu_bm.n_containers;

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cudaFree(d_output);
    cu_roaring::gpu_roaring_free(gpu_bm);
    roaring_bitmap_free(cpu_bm);
}

// Args: {universe_size, density_per_mille}
BENCHMARK(BM_GPU_Decompress)
    ->Args({1000000, 1})       // 1M, 0.1%
    ->Args({1000000, 10})      // 1M, 1%
    ->Args({1000000, 100})     // 1M, 10%
    ->Args({1000000, 500})     // 1M, 50%
    ->Args({10000000, 1})      // 10M, 0.1%
    ->Args({10000000, 10})     // 10M, 1%
    ->Args({10000000, 100})    // 10M, 10%
    ->Args({10000000, 500})    // 10M, 50%
    ->Args({100000000, 10})    // 100M, 1%
    ->Args({100000000, 100})   // 100M, 10%
    ->Args({100000000, 500})   // 100M, 50%
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();
