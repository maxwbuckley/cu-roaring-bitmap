#include <benchmark/benchmark.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "data_gen.cuh"

#include <cstdio>

// Benchmark state holder to avoid re-uploading every iteration
struct BenchState {
    roaring_bitmap_t* cpu_a = nullptr;
    roaring_bitmap_t* cpu_b = nullptr;
    cu_roaring::GpuRoaring gpu_a{};
    cu_roaring::GpuRoaring gpu_b{};
    cudaEvent_t start_ev{};
    cudaEvent_t stop_ev{};
    bool initialized = false;

    void init(uint32_t universe, double density_a, double density_b,
              uint64_t seed_a = 42, uint64_t seed_b = 123) {
        if (initialized) return;
        cpu_a = cu_roaring::bench::generate_bitmap(universe, density_a, seed_a);
        cpu_b = cu_roaring::bench::generate_bitmap(universe, density_b, seed_b);
        gpu_a = cu_roaring::upload(cpu_a);
        gpu_b = cu_roaring::upload(cpu_b);
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);
        initialized = true;
    }

    ~BenchState() {
        if (!initialized) return;
        cu_roaring::gpu_roaring_free(gpu_a);
        cu_roaring::gpu_roaring_free(gpu_b);
        roaring_bitmap_free(cpu_a);
        roaring_bitmap_free(cpu_b);
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }
};

// ============================================================================
// CPU baselines
// ============================================================================
static void BM_CPU_AND_DenseDense(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    auto* a = cu_roaring::bench::generate_bitmap(universe, 0.5, 42);
    auto* b = cu_roaring::bench::generate_bitmap(universe, 0.3, 123);

    for (auto _ : state) {
        roaring_bitmap_t* result = roaring_bitmap_and(a, b);
        benchmark::DoNotOptimize(result);
        roaring_bitmap_free(result);
    }

    state.SetItemsProcessed(state.iterations() *
                            roaring_bitmap_get_cardinality(a));
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

static void BM_CPU_AND_SparseSparse(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    auto* a = cu_roaring::bench::generate_bitmap(universe, 0.01, 42);
    auto* b = cu_roaring::bench::generate_bitmap(universe, 0.005, 123);

    for (auto _ : state) {
        roaring_bitmap_t* result = roaring_bitmap_and(a, b);
        benchmark::DoNotOptimize(result);
        roaring_bitmap_free(result);
    }

    state.SetItemsProcessed(state.iterations() *
                            roaring_bitmap_get_cardinality(a));
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

// ============================================================================
// GPU benchmarks
// ============================================================================
static void BM_GPU_AND_DenseDense(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    auto* cpu_a = cu_roaring::bench::generate_bitmap(universe, 0.5, 42);
    auto* cpu_b = cu_roaring::bench::generate_bitmap(universe, 0.3, 123);
    auto gpu_a = cu_roaring::upload(cpu_a);
    auto gpu_b = cu_roaring::upload(cpu_b);

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cu_roaring::gpu_roaring_free(r);
    }

    for (auto _ : state) {
        cudaEventRecord(start_ev);
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);

        cu_roaring::gpu_roaring_free(r);
    }

    state.SetItemsProcessed(state.iterations() *
                            roaring_bitmap_get_cardinality(cpu_a));

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(cpu_a);
    roaring_bitmap_free(cpu_b);
}

static void BM_GPU_AND_SparseSparse(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    auto* cpu_a = cu_roaring::bench::generate_bitmap(universe, 0.01, 42);
    auto* cpu_b = cu_roaring::bench::generate_bitmap(universe, 0.005, 123);
    auto gpu_a = cu_roaring::upload(cpu_a);
    auto gpu_b = cu_roaring::upload(cpu_b);

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    for (int i = 0; i < 5; ++i) {
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cu_roaring::gpu_roaring_free(r);
    }

    for (auto _ : state) {
        cudaEventRecord(start_ev);
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);

        cu_roaring::gpu_roaring_free(r);
    }

    state.SetItemsProcessed(state.iterations() *
                            roaring_bitmap_get_cardinality(cpu_a));

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(cpu_a);
    roaring_bitmap_free(cpu_b);
}

static void BM_GPU_AND_DenseSparse(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    auto* cpu_a = cu_roaring::bench::generate_bitmap(universe, 0.5, 42);
    auto* cpu_b = cu_roaring::bench::generate_bitmap(universe, 0.01, 123);
    auto gpu_a = cu_roaring::upload(cpu_a);
    auto gpu_b = cu_roaring::upload(cpu_b);

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    for (int i = 0; i < 5; ++i) {
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cu_roaring::gpu_roaring_free(r);
    }

    for (auto _ : state) {
        cudaEventRecord(start_ev);
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);

        cu_roaring::gpu_roaring_free(r);
    }

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(cpu_a);
    roaring_bitmap_free(cpu_b);
}

static void BM_CPU_AND_DenseSparse(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    auto* a = cu_roaring::bench::generate_bitmap(universe, 0.5, 42);
    auto* b = cu_roaring::bench::generate_bitmap(universe, 0.01, 123);

    for (auto _ : state) {
        roaring_bitmap_t* result = roaring_bitmap_and(a, b);
        benchmark::DoNotOptimize(result);
        roaring_bitmap_free(result);
    }

    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

// Register benchmarks with multiple universe sizes
BENCHMARK(BM_CPU_AND_DenseDense)
    ->Arg(1000000)->Arg(10000000)->Arg(100000000)->Arg(1000000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_AND_DenseDense)
    ->Arg(1000000)->Arg(10000000)->Arg(100000000)->Arg(1000000000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_CPU_AND_SparseSparse)
    ->Arg(1000000)->Arg(10000000)->Arg(100000000)->Arg(1000000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_AND_SparseSparse)
    ->Arg(1000000)->Arg(10000000)->Arg(100000000)->Arg(1000000000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_CPU_AND_DenseSparse)
    ->Arg(1000000)->Arg(10000000)->Arg(100000000)->Arg(1000000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_AND_DenseSparse)
    ->Arg(1000000)->Arg(10000000)->Arg(100000000)->Arg(1000000000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();
