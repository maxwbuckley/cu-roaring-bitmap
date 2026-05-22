/*
 * Benchmark: Run container performance vs flat bitset.
 *
 * Creates filters made of contiguous ID ranges (the ideal case for
 * Roaring run containers). Measures point query latency and compares
 * against flat bitset. This simulates realistic filtered search where
 * the filter selects a few contiguous ranges (e.g., "tenant A's rows",
 * "IDs inserted between timestamps T1-T2").
 *
 * Filter shapes tested:
 *   - 1 range   (single contiguous block)
 *   - 3 ranges  (e.g., 3 geographic regions)
 *   - 10 ranges (e.g., 10 time windows)
 *   - 100 ranges (e.g., 100 micro-segments)
 *
 * At selectivities: 0.001%, 0.01%, 0.1%, 1%
 * Universe: 1B
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
// GPU Kernels
// ============================================================================

__global__ void bitset_query_kernel(const uint32_t* __restrict__ bitset,
                                     const uint32_t* __restrict__ queries,
                                     uint32_t* __restrict__ results,
                                     uint32_t n_queries) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    uint32_t id = queries[idx];
    results[idx] = (bitset[id >> 5] >> (id & 31)) & 1u;
}

__global__ void roaring_contains_kernel(cu_roaring::GpuRoaringView view,
                                         const uint32_t* __restrict__ queries,
                                         uint32_t* __restrict__ results,
                                         uint32_t n_queries) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

__global__ void roaring_warp_contains_kernel(cu_roaring::GpuRoaringView view,
                                              const uint32_t* __restrict__ queries,
                                              uint32_t* __restrict__ results,
                                              uint32_t n_queries) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = cu_roaring::warp_contains(view, queries[idx]) ? 1u : 0u;
}

// ============================================================================
// Helpers
// ============================================================================

struct Stats {
    double median, mean, std_dev;
};

static Stats compute_stats(std::vector<double>& t) {
    std::sort(t.begin(), t.end());
    int n = static_cast<int>(t.size());
    double sum = 0;
    for (auto v : t) sum += v;
    double mean = sum / n;
    double var = 0;
    for (auto v : t) var += (v - mean) * (v - mean);
    return {t[n / 2], mean, std::sqrt(var / n)};
}

static Stats bench_gpu(int warmup, int iters, std::function<void()> fn) {
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

// Create a bitmap with n_ranges contiguous ranges, total cardinality = target_card
static roaring_bitmap_t* make_range_bitmap(uint32_t universe, uint64_t target_card,
                                            uint32_t n_ranges, uint64_t seed) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 gen(seed);

    uint64_t per_range = target_card / n_ranges;
    if (per_range == 0) per_range = 1;

    // Place ranges randomly within the universe
    std::uniform_int_distribution<uint32_t> pos_dist(0, universe - 1);
    for (uint32_t i = 0; i < n_ranges; ++i) {
        uint32_t start = pos_dist(gen);
        uint64_t end = std::min(static_cast<uint64_t>(start) + per_range,
                                static_cast<uint64_t>(universe));
        roaring_bitmap_add_range(r, start, end);
    }
    roaring_bitmap_run_optimize(r);
    return r;
}


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    constexpr uint32_t UNIVERSE = 1000000000;
    constexpr uint32_t N_QUERIES = 10000000;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 50;

    const size_t bitset_bytes = (static_cast<size_t>(UNIVERSE) + 31) / 32 * sizeof(uint32_t);

    // Pre-generate random queries
    std::vector<uint32_t> h_queries(N_QUERIES);
    {
        std::mt19937 gen(777);
        std::uniform_int_distribution<uint32_t> dist(0, UNIVERSE - 1);
        for (uint32_t i = 0; i < N_QUERIES; ++i) h_queries[i] = dist(gen);
    }

    uint32_t* d_queries = nullptr;
    cudaMalloc(&d_queries, N_QUERIES * sizeof(uint32_t));
    cudaMemcpy(d_queries, h_queries.data(), N_QUERIES * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    uint32_t* d_results_bs = nullptr;
    uint32_t* d_results_ct = nullptr;
    uint32_t* d_results_wc = nullptr;
    cudaMalloc(&d_results_bs, N_QUERIES * sizeof(uint32_t));
    cudaMalloc(&d_results_ct, N_QUERIES * sizeof(uint32_t));
    cudaMalloc(&d_results_wc, N_QUERIES * sizeof(uint32_t));

    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, bitset_bytes);

    dim3 block(256);
    dim3 grid((N_QUERIES + 255) / 256);

    struct Config {
        double selectivity;
        uint32_t n_ranges;
    };

    Config configs[] = {
        {0.00001, 1}, {0.00001, 3}, {0.00001, 10}, {0.00001, 100},
        {0.0001,  1}, {0.0001,  3}, {0.0001,  10}, {0.0001,  100},
        {0.001,   1}, {0.001,   3}, {0.001,   10}, {0.001,   100},
        {0.01,    1}, {0.01,    3}, {0.01,    10}, {0.01,    100},
    };

    printf("Universe: %u (bitset = %.0f MB)\n", UNIVERSE, bitset_bytes / 1e6);
    printf("Queries: %u random uniform\n\n", N_QUERIES);

    printf("%-8s %6s %12s  %7s %10s  %10s %10s %10s  %8s %8s\n",
           "Sel%", "Ranges", "Card", "GPU KB", "Compress",
           "Bitset", "Contains", "Warp", "Speedup", "Type");
    printf("%-8s %6s %12s  %7s %10s  %10s %10s %10s  %8s %8s\n",
           "", "", "", "", "ratio",
           "(ms)", "(ms)", "(ms)", "vs bs", "breakdown");
    printf("%s\n", std::string(110, '-').c_str());

    for (const auto& cfg : configs) {
        uint64_t target_card = static_cast<uint64_t>(UNIVERSE * cfg.selectivity);
        if (target_card == 0) target_card = 1;

        roaring_bitmap_t* cpu_bm = make_range_bitmap(UNIVERSE, target_card,
                                                      cfg.n_ranges, 42);
        roaring_bitmap_run_optimize(cpu_bm);

        uint64_t card = roaring_bitmap_get_cardinality(cpu_bm);
        auto meta = cu_roaring::get_meta(cpu_bm);

        // Upload Roaring (PROMOTE_KEEP_DEFAULT to keep run containers)
        auto gpu_bm = cu_roaring::upload(cpu_bm, UNIVERSE, 0, cu_roaring::PROMOTE_KEEP_DEFAULT);
        auto view = cu_roaring::make_view(gpu_bm);

        // Compute actual GPU size
        size_t gpu_bytes = meta.total_bytes;
        gpu_bytes += gpu_bm.n_containers * (2 + 1 + 4 + 2);
        if (gpu_bm.key_index)
            gpu_bytes += (static_cast<size_t>(gpu_bm.max_key) + 1) * 2;

        // Build flat bitset
        {
            uint32_t n_words = (UNIVERSE + 31) / 32;
            std::vector<uint32_t> h(n_words, 0);
            roaring_uint32_iterator_t* iter = roaring_iterator_create(cpu_bm);
            while (iter->has_value) {
                uint32_t v = iter->current_value;
                if (v / 32 < n_words) h[v / 32] |= (1u << (v % 32));
                roaring_uint32_iterator_advance(iter);
            }
            roaring_uint32_iterator_free(iter);
            cudaMemcpy(d_bitset, h.data(), bitset_bytes, cudaMemcpyHostToDevice);
        }

        // Benchmark
        auto bs = bench_gpu(WARMUP, ITERS, [&]() {
            bitset_query_kernel<<<grid, block>>>(d_bitset, d_queries, d_results_bs, N_QUERIES);
        });
        auto ct = bench_gpu(WARMUP, ITERS, [&]() {
            roaring_contains_kernel<<<grid, block>>>(view, d_queries, d_results_ct, N_QUERIES);
        });
        auto wc = bench_gpu(WARMUP, ITERS, [&]() {
            roaring_warp_contains_kernel<<<grid, block>>>(view, d_queries, d_results_wc, N_QUERIES);
        });

        double best_roaring = std::min(ct.median, wc.median);
        double speedup = bs.median / best_roaring;
        double compress = static_cast<double>(bitset_bytes) / gpu_bytes;

        // Container type breakdown
        char types[64];
        snprintf(types, sizeof(types), "B%u A%u R%u",
                 gpu_bm.n_bitmap_containers, gpu_bm.n_array_containers,
                 gpu_bm.n_run_containers);

        printf("%-8.4f %6u %12llu  %7.1f %9.0fx  %10.3f %10.3f %10.3f  %7.1fx %s\n",
               cfg.selectivity * 100.0,
               cfg.n_ranges,
               (unsigned long long)card,
               gpu_bytes / 1024.0,
               compress,
               bs.median, ct.median, wc.median,
               speedup, types);

        cu_roaring::gpu_roaring_free(gpu_bm);
        roaring_bitmap_free(cpu_bm);
    }

    cudaFree(d_queries);
    cudaFree(d_results_bs);
    cudaFree(d_results_ct);
    cudaFree(d_results_wc);
    cudaFree(d_bitset);

    return 0;
}
