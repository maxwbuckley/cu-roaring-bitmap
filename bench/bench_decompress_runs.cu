// bench_decompress_runs.cu
//
// Micro-benchmark for decompress_to_bitset() on RUN-container-dominated
// bitmaps -- the code path changed by the word-at-a-time run-expansion
// optimization in src/decompress.cu.
//
// The optimization replaces O(run_length) atomicOr calls per run with
// ~2 atomicOr (partial boundary words) + O(run_length / 32) plain stores
// (interior full words). The speedup therefore scales with average run
// length, so this benchmark sweeps run length at a fixed universe size and
// density.
//
// Build:  add_cu_roaring_bench(bench_decompress_runs) is in bench/CMakeLists.txt
// Run:    ./bench/bench_decompress_runs
// Workflow + interpretation: see bench/DECOMPRESS_RUN_BENCHMARK.md

#include <benchmark/benchmark.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"

#include <algorithm>
#include <cstdint>

namespace {

// Build a RUN-container-dominated bitmap: alternating runs of `run_len` set
// bits and gaps of `gap_len` clear bits across [0, universe). After
// run_optimize, each 65536-bit container is stored as a RUN container
// provided it holds fewer than ~2048 runs (above that CRoaring prefers a
// bitmap container, since 4 bytes/run would exceed the 8 KB bitmap).
roaring_bitmap_t* make_run_bitmap(uint32_t universe,
                                  uint32_t run_len,
                                  uint32_t gap_len) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    uint32_t pos = 0;
    while (pos < universe) {
        uint32_t end = std::min(pos + run_len, universe);
        if (end > pos) {
            roaring_bitmap_add_range(r, pos, end);  // adds [pos, end)
        }
        pos = end + gap_len;
    }
    roaring_bitmap_run_optimize(r);
    return r;
}

}  // namespace

// Args: {universe_size, run_len}. Gap is fixed at 2x run_len (~33% density)
// so the input stays comfortably RUN-encoded and complement ("negated")
// storage is never triggered -- run length is the only variable.
static void BM_GPU_Decompress_Runs(benchmark::State& state) {
    const uint32_t universe = static_cast<uint32_t>(state.range(0));
    const uint32_t run_len  = static_cast<uint32_t>(state.range(1));
    const uint32_t gap_len  = 2u * run_len;

    roaring_bitmap_t* cpu_bm = make_run_bitmap(universe, run_len, gap_len);
    auto gpu_bm = cu_roaring::upload(cpu_bm, universe);

    // Guard: if CRoaring did not actually choose RUN containers, this
    // benchmark would be measuring the array/bitmap path instead. Fail
    // loudly rather than report a misleading number.
    if (gpu_bm.n_run_containers == 0) {
        state.SkipWithError(
            "input produced no RUN containers - adjust run_len/gap_len");
        cu_roaring::gpu_roaring_free(gpu_bm);
        roaring_bitmap_free(cpu_bm);
        return;
    }

    const uint32_t n_words = (gpu_bm.universe_size + 31) / 32;
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

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);
    }

    state.counters["out_MB"] =
        static_cast<double>(n_words) * sizeof(uint32_t) / (1024.0 * 1024.0);
    state.counters["run_len"]          = run_len;
    state.counters["n_containers"]     = gpu_bm.n_containers;
    state.counters["n_run_containers"] = gpu_bm.n_run_containers;
    state.counters["cardinality"] =
        static_cast<double>(gpu_bm.total_cardinality);

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cudaFree(d_output);
    cu_roaring::gpu_roaring_free(gpu_bm);
    roaring_bitmap_free(cpu_bm);
}

// Sweep run length from ~1 word (32 bits) to ~512 words (16384 bits). The run
// path's per-run cost drops from O(run_len) atomics to ~O(run_len/32) plain
// stores, so the expected speedup vs the bit-at-a-time baseline grows with
// run_len and saturates near 32x. Repetitions(10) makes every row report
// median + stddev + cv.
BENCHMARK(BM_GPU_Decompress_Runs)
    ->ArgNames({"universe", "run_len"})
    ->Args({64000000, 32})
    ->Args({64000000, 128})
    ->Args({64000000, 512})
    ->Args({64000000, 2048})
    ->Args({64000000, 16384})
    ->Args({256000000, 512})  // scale check at a mid run length
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Repetitions(10)
    ->DisplayAggregatesOnly(true);
