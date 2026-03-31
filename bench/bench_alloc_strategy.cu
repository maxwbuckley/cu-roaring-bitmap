// bench_alloc_strategy.cu — Measures the impact of stream-ordered allocation
// (cudaMallocAsync/cudaFreeAsync) on set_operation and upload_from_ids.
//
// Tests two CUDA memory pool configurations:
//   1. Default pool (OS may reclaim freed memory between iterations)
//   2. Tuned pool  (release_threshold = UINT64_MAX, mimics RMM pool behavior)
//
// Protocol: 10 warmup + 50 timed iterations per condition.
// Reports: mean, median, stdev, min, max, and Welch's t-test p-value.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include "cu_roaring/cu_roaring.cuh"
#include "data_gen.cuh"

// ============================================================================
// Statistics
// ============================================================================
struct Stats {
    double mean;
    double median;
    double stdev;
    double min_val;
    double max_val;
};

static Stats compute_stats(std::vector<double>& v) {
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min_val = v.front();
    s.max_val = v.back();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / static_cast<double>(v.size());
    size_t mid = v.size() / 2;
    s.median = (v.size() % 2 == 0) ? (v[mid - 1] + v[mid]) / 2.0 : v[mid];
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - s.mean) * (x - s.mean);
    s.stdev = std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
    return s;
}

// Welch's t-test (two-sample, unequal variance)
static double welch_t_pvalue(const std::vector<double>& a,
                              const std::vector<double>& b) {
    double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(a.size());
    double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(b.size());
    double var_a = 0.0, var_b = 0.0;
    for (double x : a) var_a += (x - mean_a) * (x - mean_a);
    for (double x : b) var_b += (x - mean_b) * (x - mean_b);
    var_a /= static_cast<double>(a.size() - 1);
    var_b /= static_cast<double>(b.size() - 1);

    double se = std::sqrt(var_a / static_cast<double>(a.size()) +
                          var_b / static_cast<double>(b.size()));
    if (se < 1e-15) return 1.0;  // identical distributions

    double t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom
    double va_n = var_a / static_cast<double>(a.size());
    double vb_n = var_b / static_cast<double>(b.size());
    double df = (va_n + vb_n) * (va_n + vb_n) /
                (va_n * va_n / (static_cast<double>(a.size()) - 1.0) +
                 vb_n * vb_n / (static_cast<double>(b.size()) - 1.0));

    // Approximate p-value using the normal distribution for large df
    // (good enough for df > 30, which we always have with n=50)
    double z = std::abs(t);
    // Two-tailed p-value approximation (Abramowitz & Stegun 26.2.17)
    double p = std::exp(-0.5 * z * z) / (z * std::sqrt(2.0 * M_PI));
    if (df > 30.0) {
        // Use normal approximation
        p = 2.0 * 0.5 * std::erfc(z / std::sqrt(2.0));
    } else {
        // Rough Student's t approximation
        p = 2.0 * 0.5 * std::erfc(z / std::sqrt(2.0));
    }
    return p;
}

// ============================================================================
// Pool configuration
// ============================================================================
static void set_pool_release_threshold(uint64_t threshold) {
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

static void trim_pool() {
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    cudaMemPoolTrimTo(pool, 0);
}

// ============================================================================
// Benchmark runner
// ============================================================================
static constexpr int WARMUP = 10;
static constexpr int ITERS  = 50;

struct BenchConfig {
    const char* name;
    uint32_t    universe;
    double      density_a;
    double      density_b;
    cu_roaring::SetOp op;
};

static const char* op_name(cu_roaring::SetOp op) {
    switch (op) {
        case cu_roaring::SetOp::AND:    return "AND";
        case cu_roaring::SetOp::OR:     return "OR";
        case cu_roaring::SetOp::ANDNOT: return "ANDNOT";
        case cu_roaring::SetOp::XOR:    return "XOR";
    }
    return "?";
}

static void print_stats(const char* label, Stats& s) {
    std::printf("  %-20s  mean=%8.1f us  median=%8.1f us  stdev=%7.1f us  "
                "min=%8.1f us  max=%8.1f us\n",
                label, s.mean, s.median, s.stdev, s.min_val, s.max_val);
}

// Run set_operation benchmark with a given pool config
static std::vector<double> bench_set_op(
    const cu_roaring::GpuRoaring& gpu_a,
    const cu_roaring::GpuRoaring& gpu_b,
    cu_roaring::SetOp op,
    cudaStream_t stream)
{
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, op, stream);
        cu_roaring::gpu_roaring_free(r);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    std::vector<double> times(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        cudaEventRecord(start_ev, stream);
        auto r = cu_roaring::set_operation(gpu_a, gpu_b, op, stream);
        cu_roaring::gpu_roaring_free(r);
        cudaEventRecord(stop_ev, stream);
        cudaEventSynchronize(stop_ev);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        times[i] = static_cast<double>(ms) * 1000.0;  // convert to microseconds
    }

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    return times;
}

int main() {
    // Print GPU info
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
                props.name, props.major, props.minor,
                props.multiProcessorCount);

    // Check that stream-ordered allocation is supported
    int supports_pool = 0;
    cudaDeviceGetAttribute(&supports_pool,
                           cudaDevAttrMemoryPoolsSupported, 0);
    if (!supports_pool) {
        std::fprintf(stderr, "ERROR: Device does not support memory pools "
                             "(cudaMallocAsync). Need CUDA 11.2+ and compute >= 7.0\n");
        return 1;
    }
    std::printf("Stream-ordered memory pools: SUPPORTED\n\n");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ========================================================================
    // Test matrix
    // ========================================================================
    std::vector<BenchConfig> configs = {
        // Dense x Dense (many bitmap containers, heavy temp allocs)
        {"dense_AND_1M",   1'000'000,  0.50, 0.30, cu_roaring::SetOp::AND},
        {"dense_AND_10M",  10'000'000, 0.50, 0.30, cu_roaring::SetOp::AND},
        {"dense_AND_100M", 100'000'000,0.50, 0.30, cu_roaring::SetOp::AND},
        {"dense_OR_10M",   10'000'000, 0.50, 0.30, cu_roaring::SetOp::OR},

        // Sparse x Sparse (few containers, less alloc overhead)
        {"sparse_AND_1M",  1'000'000,  0.01, 0.005, cu_roaring::SetOp::AND},
        {"sparse_AND_10M", 10'000'000, 0.01, 0.005, cu_roaring::SetOp::AND},

        // Mixed density (bitmap x array containers)
        {"mixed_AND_10M",  10'000'000, 0.50, 0.01, cu_roaring::SetOp::AND},
        {"mixed_OR_10M",   10'000'000, 0.50, 0.01, cu_roaring::SetOp::OR},
    };

    std::printf("=== set_operation Benchmark ===\n");
    std::printf("Protocol: %d warmup + %d timed iterations per condition\n", WARMUP, ITERS);
    std::printf("Pool configs: default (release_threshold=0) vs tuned (release_threshold=MAX)\n\n");

    for (auto& cfg : configs) {
        std::printf("--- %s (universe=%uM, density=%.3f x %.3f, op=%s) ---\n",
                    cfg.name, cfg.universe / 1'000'000,
                    cfg.density_a, cfg.density_b, op_name(cfg.op));

        // Generate test data
        auto* cpu_a = cu_roaring::bench::generate_bitmap(cfg.universe, cfg.density_a, 42);
        auto* cpu_b = cu_roaring::bench::generate_bitmap(cfg.universe, cfg.density_b, 123);
        auto gpu_a = cu_roaring::upload(cpu_a, cfg.universe);
        auto gpu_b = cu_roaring::upload(cpu_b, cfg.universe);

        std::printf("  A: %u containers (%u bitmap, %u array, %u run)\n",
                    gpu_a.n_containers, gpu_a.n_bitmap_containers,
                    gpu_a.n_array_containers, gpu_a.n_run_containers);
        std::printf("  B: %u containers (%u bitmap, %u array, %u run)\n",
                    gpu_b.n_containers, gpu_b.n_bitmap_containers,
                    gpu_b.n_array_containers, gpu_b.n_run_containers);

        // Config 1: Default pool (threshold = 0)
        trim_pool();
        set_pool_release_threshold(0);
        auto times_default = bench_set_op(gpu_a, gpu_b, cfg.op, stream);
        auto stats_default = compute_stats(times_default);

        // Config 2: Tuned pool (threshold = MAX, like RMM pool)
        trim_pool();
        set_pool_release_threshold(UINT64_MAX);
        auto times_tuned = bench_set_op(gpu_a, gpu_b, cfg.op, stream);
        auto stats_tuned = compute_stats(times_tuned);

        // Reset pool to default
        trim_pool();
        set_pool_release_threshold(0);

        print_stats("default_pool", stats_default);
        print_stats("tuned_pool", stats_tuned);

        double speedup = stats_default.median / stats_tuned.median;
        double p = welch_t_pvalue(times_default, times_tuned);
        std::printf("  Speedup (tuned/default): %.2fx median  (p=%.4f)\n\n",
                    speedup, p);

        // Cleanup
        cu_roaring::gpu_roaring_free(gpu_a);
        cu_roaring::gpu_roaring_free(gpu_b);
        roaring_bitmap_free(cpu_a);
        roaring_bitmap_free(cpu_b);
    }

    // ========================================================================
    // upload_from_ids benchmark
    // ========================================================================
    std::printf("\n=== upload_from_ids Benchmark ===\n\n");

    struct UploadConfig {
        const char* name;
        uint32_t universe;
        double density;
    };
    std::vector<UploadConfig> upload_configs = {
        {"upload_1M_10pct",   1'000'000,  0.10},
        {"upload_10M_10pct",  10'000'000, 0.10},
        {"upload_10M_50pct",  10'000'000, 0.50},
        {"upload_100M_10pct", 100'000'000,0.10},
    };

    for (auto& cfg : upload_configs) {
        std::printf("--- %s (universe=%uM, density=%.2f) ---\n",
                    cfg.name, cfg.universe / 1'000'000, cfg.density);

        // Generate IDs
        auto* cpu_bmp = cu_roaring::bench::generate_bitmap(cfg.universe, cfg.density, 42);
        uint64_t card = roaring_bitmap_get_cardinality(cpu_bmp);
        std::vector<uint32_t> ids(card);
        roaring_bitmap_to_uint32_array(cpu_bmp, ids.data());
        roaring_bitmap_free(cpu_bmp);

        std::printf("  n_ids=%zu\n", ids.size());

        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        auto bench_upload = [&]() -> std::vector<double> {
            // Warmup
            for (int i = 0; i < WARMUP; ++i) {
                auto r = cu_roaring::upload_from_ids(ids.data(),
                    static_cast<uint32_t>(ids.size()), cfg.universe, stream);
                cu_roaring::gpu_roaring_free(r);
            }
            cudaDeviceSynchronize();

            std::vector<double> times(ITERS);
            for (int i = 0; i < ITERS; ++i) {
                cudaEventRecord(start_ev, stream);
                auto r = cu_roaring::upload_from_ids(ids.data(),
                    static_cast<uint32_t>(ids.size()), cfg.universe, stream);
                cu_roaring::gpu_roaring_free(r);
                cudaEventRecord(stop_ev, stream);
                cudaEventSynchronize(stop_ev);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_ev, stop_ev);
                times[i] = static_cast<double>(ms) * 1000.0;
            }
            return times;
        };

        // Default pool
        trim_pool();
        set_pool_release_threshold(0);
        auto times_default = bench_upload();
        auto stats_default = compute_stats(times_default);

        // Tuned pool
        trim_pool();
        set_pool_release_threshold(UINT64_MAX);
        auto times_tuned = bench_upload();
        auto stats_tuned = compute_stats(times_tuned);

        trim_pool();
        set_pool_release_threshold(0);

        print_stats("default_pool", stats_default);
        print_stats("tuned_pool", stats_tuned);

        double speedup = stats_default.median / stats_tuned.median;
        double p = welch_t_pvalue(times_default, times_tuned);
        std::printf("  Speedup (tuned/default): %.2fx median  (p=%.4f)\n\n",
                    speedup, p);

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    cudaStreamDestroy(stream);

    std::printf("\n=== Notes ===\n");
    std::printf("- 'default_pool': CUDA async pool with release_threshold=0 (OS reclaims freed memory)\n");
    std::printf("- 'tuned_pool': CUDA async pool with release_threshold=UINT64_MAX (RMM-equivalent behavior)\n");
    std::printf("- Both configs use cudaMallocAsync/cudaFreeAsync (stream-ordered, no device sync on free)\n");
    std::printf("- To compare against the old cudaMalloc/cudaFree baseline, build the repo\n");
    std::printf("  from the commit before this change and run bench_set_ops.\n");

    return 0;
}
