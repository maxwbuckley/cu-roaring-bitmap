// bench_filtered_search.cu
//
// Dense-masked vs schedule-driven (container-dispatch) roaring-filtered
// brute-force search. Measures end-to-end search latency (top-k inner
// product) with the filter/schedule pre-built -- the design's stated
// assumption is a filter shared across a query batch.
//
//   mode 0 = dense   : full Q x N GEMM + decompressed-bitset mask + top-k
//   mode 1 = roaring : build_schedule() dispatch -> GEMM only over eligible
//
// Sweeps: selectivity, query batch size, vector dimension, and run-count
// fragmentation (to locate the shape-gate crossover).
//
// Build:  add_cu_roaring_bench(bench_filtered_search) in bench/CMakeLists.txt
// Run:    ./bench/bench_filtered_search

#include <benchmark/benchmark.h>
#include <roaring/roaring.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/detail/filtered_search.cuh"

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kK = 10;

// ---- shared device dataset, cached by (N, D) -------------------------------
struct Dataset {
    float* d_db = nullptr;
    float* d_q  = nullptr;   // kMaxQ query rows
    uint32_t N = 0, D = 0;
};
constexpr uint32_t kMaxQ = 128;

Dataset& dataset(uint32_t N, uint32_t D) {
    static std::map<uint64_t, Dataset> cache;
    uint64_t key = (static_cast<uint64_t>(N) << 16) | D;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    Dataset ds;
    ds.N = N;
    ds.D = D;
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> h(static_cast<size_t>(N) * D);
    for (auto& x : h) x = nd(rng);
    cudaMalloc(&ds.d_db, h.size() * sizeof(float));
    cudaMemcpy(ds.d_db, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> hq(static_cast<size_t>(kMaxQ) * D);
    for (auto& x : hq) x = nd(rng);
    cudaMalloc(&ds.d_q, hq.size() * sizeof(float));
    cudaMemcpy(ds.d_q, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice);

    return cache.emplace(key, ds).first->second;
}

cublasHandle_t cublas() {
    static cublasHandle_t h = [] {
        cublasHandle_t handle;
        cublasCreate(&handle);
        return handle;
    }();
    return h;
}

// Filter selecting `sel_permille` of [0,N), spread over `n_runs` contiguous
// runs (the favourable shape for the run dispatch).
roaring_bitmap_t* make_run_filter(uint32_t N, uint32_t sel_permille,
                                  uint32_t n_runs) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    uint64_t total = static_cast<uint64_t>(N) * sel_permille / 1000u;
    if (n_runs < 1) n_runs = 1;
    uint64_t width  = std::max<uint64_t>(2, total / n_runs);
    uint64_t stride = std::max<uint64_t>(width, N / n_runs);
    for (uint32_t i = 0; i < n_runs; ++i) {
        uint64_t s = static_cast<uint64_t>(i) * stride;
        if (s >= N) break;
        uint64_t e = std::min<uint64_t>(s + width, N);
        roaring_bitmap_add_range(r, s, e);
    }
    roaring_bitmap_run_optimize(r);
    return r;
}

// ---- the benchmark ---------------------------------------------------------
// Args: {N_kilo, D, Q, sel_permille, n_runs, mode}
void BM_Search(benchmark::State& state) {
    const uint32_t N   = static_cast<uint32_t>(state.range(0)) * 1024u;
    const uint32_t D   = static_cast<uint32_t>(state.range(1));
    const uint32_t Q   = static_cast<uint32_t>(state.range(2));
    const uint32_t sel = static_cast<uint32_t>(state.range(3));
    const uint32_t nr  = static_cast<uint32_t>(state.range(4));
    const int      mode = static_cast<int>(state.range(5));

    Dataset& ds = dataset(N, D);
    cublasHandle_t h = cublas();

    roaring_bitmap_t* cpu = make_run_filter(N, sel, nr);
    auto gpu = cu_roaring::upload(cpu, N);

    cu_roaring::SearchSchedule sched;
    if (mode == 1) sched = cu_roaring::build_schedule(gpu, N, ds.d_db, D);

    uint32_t* d_ids = nullptr;
    float*    d_sc  = nullptr;
    cudaMalloc(&d_ids, static_cast<size_t>(Q) * kK * sizeof(uint32_t));
    cudaMalloc(&d_sc,  static_cast<size_t>(Q) * kK * sizeof(float));

    // A non-default stream so the executor can CUDA-graph-capture the search.
    cudaStream_t st;
    cudaStreamCreate(&st);

    auto do_search = [&] {
        if (mode == 0)
            cu_roaring::dense_filtered_search(h, ds.d_q, Q, ds.d_db, N, D,
                                              gpu, kK, d_ids, d_sc, st);
        else
            cu_roaring::roaring_filtered_search(h, ds.d_q, Q, ds.d_db, N, D,
                                                sched, kK, d_ids, d_sc, st);
    };

    for (int i = 0; i < 8; ++i) { do_search(); }  // warm cuBLAS + capture graph
    cudaStreamSynchronize(st);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (auto _ : state) {
        cudaEventRecord(e0, st);
        do_search();
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        state.SetIterationTime(ms / 1000.0);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(st);

    // GEMM columns actually processed (the O(.) work) and FLOPs.
    uint64_t cols = (mode == 0) ? N : sched.total_cols;
    state.counters["cardinality"] = static_cast<double>(gpu.total_cardinality);
    state.counters["gemm_cols"]   = static_cast<double>(cols);
    state.counters["GFLOP"] =
        2.0 * static_cast<double>(Q) * static_cast<double>(cols) * D / 1e9;
    state.counters["fallback"] = (mode == 1 && sched.used_fallback) ? 1 : 0;
    state.counters["negated"]  = gpu.negated ? 1 : 0;

    if (mode == 1) cu_roaring::free_schedule(sched);
    cudaFree(d_ids);
    cudaFree(d_sc);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(cpu);
}

// N=2048Ki (2M rows). k=10 throughout. mode: 0=dense 1=roaring.

// (1) Selectivity sweep: contiguous filter (8 runs), Q=32, D=128.
#define SEL_ROW(sel)                          \
    Args({2048, 128, 32, sel, 8, 0})          \
    ->Args({2048, 128, 32, sel, 8, 1})
BENCHMARK(BM_Search)
    ->ArgNames({"N_Ki", "D", "Q", "sel_pm", "runs", "mode"})
    ->SEL_ROW(1)->SEL_ROW(10)->SEL_ROW(50)
    ->SEL_ROW(250)->SEL_ROW(500)->SEL_ROW(900)
    ->Unit(benchmark::kMicrosecond)->UseManualTime()
    ->Repetitions(30)->DisplayAggregatesOnly(true);

// (2) Query-batch sweep: sel=5%, D=128, 8 runs.
#define Q_ROW(q)                              \
    Args({2048, 128, q, 50, 8, 0})            \
    ->Args({2048, 128, q, 50, 8, 1})
BENCHMARK(BM_Search)
    ->ArgNames({"N_Ki", "D", "Q", "sel_pm", "runs", "mode"})
    ->Q_ROW(1)->Q_ROW(8)->Q_ROW(32)->Q_ROW(128)
    ->Unit(benchmark::kMicrosecond)->UseManualTime()
    ->Repetitions(30)->DisplayAggregatesOnly(true);

// (3) Dimension sweep: sel=5%, Q=32, 8 runs.
#define D_ROW(d)                              \
    Args({2048, d, 32, 50, 8, 0})             \
    ->Args({2048, d, 32, 50, 8, 1})
BENCHMARK(BM_Search)
    ->ArgNames({"N_Ki", "D", "Q", "sel_pm", "runs", "mode"})
    ->D_ROW(64)->D_ROW(128)->D_ROW(768)
    ->Unit(benchmark::kMicrosecond)->UseManualTime()
    ->Repetitions(30)->DisplayAggregatesOnly(true);

// (4) Fragmentation sweep: sel=10%, Q=32, D=128, run count 8 -> 90k.
#define FRAG_ROW(runs)                        \
    Args({2048, 128, 32, 100, runs, 0})       \
    ->Args({2048, 128, 32, 100, runs, 1})
BENCHMARK(BM_Search)
    ->ArgNames({"N_Ki", "D", "Q", "sel_pm", "runs", "mode"})
    ->FRAG_ROW(8)->FRAG_ROW(512)->FRAG_ROW(16384)->FRAG_ROW(90000)
    ->Unit(benchmark::kMicrosecond)->UseManualTime()
    ->Repetitions(30)->DisplayAggregatesOnly(true);

}  // namespace

BENCHMARK_MAIN();
