// bench_vs_bitset — performance sweep for cu_roaring::v2.
//
// Sweeps n_bitmaps × universe × selectivity. Each cell measures:
//   - upload_batch / promote_batch / decompress_batch latency
//   - contains() Gq/s (roaring batch vs flat bitset of identical shape)
//   - multi_and() ms (roaring batch vs flat bitset of identical shape)
//   - memory footprint (roaring vs bitset)
//
// All timings are taken with cudaEvent over N iters after a warmup; we report
// median and stdev so noisy cells (small kernels, async-engine jitter) show up
// as wider distributions instead of being averaged into a misleading mean.
//
// Both baselines query the same logical bitmaps — the bitset is built from the
// roaring batch via decompress_batch, so any correctness bug in either path
// manifests as a value mismatch in the contains step.

#include "cu_roaring_v2/api.hpp"
#include "cu_roaring_v2/query.cuh"

#include <roaring/roaring.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace crv2 = cu_roaring::v2;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// --- timing utilities ------------------------------------------------------

struct Stats {
    double mean_ms;
    double median_ms;
    double stdev_ms;
};

template <typename Launch>
Stats time_kernel(Launch&& launch, int warmup, int iters) {
    cudaEvent_t s, e;
    check(cudaEventCreate(&s), "event create");
    check(cudaEventCreate(&e), "event create");

    for (int i = 0; i < warmup; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync");

    std::vector<float> samples(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(s), "event rec s");
        launch();
        check(cudaEventRecord(e), "event rec e");
        check(cudaEventSynchronize(e), "event sync e");
        check(cudaEventElapsedTime(&samples[static_cast<size_t>(i)], s, e), "elapsed");
    }
    check(cudaEventDestroy(s), "event destroy");
    check(cudaEventDestroy(e), "event destroy");

    double sum = 0.0;
    for (auto x : samples) sum += static_cast<double>(x);
    const double mean = sum / static_cast<double>(iters);
    double sq = 0.0;
    for (auto x : samples) {
        const double d = static_cast<double>(x) - mean;
        sq += d * d;
    }
    const double stdev = std::sqrt(sq / static_cast<double>(iters));
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median =
        static_cast<double>(sorted[static_cast<size_t>(iters / 2)]);
    return {mean, median, stdev};
}

// --- query kernels ---------------------------------------------------------

__global__ void roaring_batch_contains_kernel(
    crv2::GpuRoaringBatch         batch,
    const uint32_t* __restrict__  bitmap_indices,
    const uint32_t* __restrict__  ids,
    uint32_t                      n,
    uint8_t*        __restrict__  out)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = crv2::contains(batch, bitmap_indices[i], ids[i]) ? 1u : 0u;
}

__global__ void bitset_batch_contains_kernel(
    const uint64_t* __restrict__  bitsets,
    uint64_t                      words_each,
    const uint32_t* __restrict__  bitmap_indices,
    const uint32_t* __restrict__  ids,
    uint32_t                      n,
    uint8_t*        __restrict__  out)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t b  = bitmap_indices[i];
    const uint32_t id = ids[i];
    const uint64_t w  = id >> 6;
    if (w >= words_each) { out[i] = 0u; return; }
    out[i] = ((__ldg(&bitsets[static_cast<size_t>(b) * words_each + w])
              >> (id & 63u)) & 1ULL) ? 1u : 0u;
}

__global__ void bitset_multi_and_kernel(
    const uint64_t* __restrict__  bitsets,
    uint64_t                      words_each,
    const uint32_t* __restrict__  indices,
    uint32_t                      count,
    uint64_t*       __restrict__  out)
{
    const uint64_t w = static_cast<uint64_t>(blockIdx.x) * blockDim.x
                     + threadIdx.x;
    if (w >= words_each) return;
    uint64_t acc = ~0ULL;
    #pragma unroll 1
    for (uint32_t k = 0; k < count; ++k) {
        acc &= bitsets[static_cast<size_t>(indices[k]) * words_each + w];
    }
    out[w] = acc;
}

// --- synthetic dataset ------------------------------------------------------

roaring_bitmap_t* build_cpu_bitmap(uint64_t universe,
                                   double   selectivity,
                                   uint64_t seed)
{
    roaring_bitmap_t* r = roaring_bitmap_create();
    const uint64_t target = static_cast<uint64_t>(
        static_cast<double>(universe) * selectivity);
    std::mt19937_64 rng(seed);
    constexpr size_t BATCH = 8192;
    std::vector<uint32_t> batch(BATCH);
    size_t produced = 0;
    while (produced < target) {
        const size_t take = std::min(BATCH, static_cast<size_t>(target - produced));
        for (size_t i = 0; i < take; ++i) {
            batch[i] = static_cast<uint32_t>(rng() % universe);
        }
        roaring_bitmap_add_many(r, take, batch.data());
        produced += take;
    }
    roaring_bitmap_run_optimize(r);
    return r;
}

size_t roaring_total_bytes(const crv2::GpuRoaringBatch& B) {
    // Mirrors the layout produced by upload.cpp: every section of the single
    // packed device allocation, summed in nominal element size.
    size_t b = 0;
    b += (B.n_bitmaps + 1u) * 2u * sizeof(uint32_t);                       // CSR starts
    b += B.total_containers * (sizeof(uint16_t)                            // keys
                              + sizeof(crv2::ContainerType)                // types
                              + sizeof(uint32_t)                           // offsets
                              + sizeof(uint16_t));                         // cardinalities
    b += B.host_key_index_starts[B.n_bitmaps] * sizeof(uint16_t);          // key_indices
    b += static_cast<size_t>(B.n_bitmap_containers_total) * 1024u
       * sizeof(uint64_t);                                                  // bitmap pool
    b += B.array_pool_bytes;
    b += B.run_pool_bytes;
    return b;
}

// --- per-cell driver -------------------------------------------------------

struct Cell {
    uint32_t n_bitmaps;
    uint64_t universe;
    double   selectivity;
};

struct CellResult {
    Cell c;

    // Setup latencies (one-shot per-cell, single call timed).
    double upload_ms;
    double promote_ms;
    double decompress_ms;

    // Memory.
    size_t roaring_bytes_raw;     // before promote
    size_t roaring_bytes_promoted;
    size_t bitset_bytes;

    // Query throughput. Roaring uses the raw mixed-type batch (apples-to-apples
    // for "what does the user actually pay"); the bitset baseline uses a flat
    // [n_bitmaps × words_each] array materialised from the same data.
    Stats contains_roaring;
    Stats contains_bitset;
    double n_queries;

    // multi_and (roaring uses the *promoted* batch; bitset is naturally flat).
    Stats and_roaring;
    Stats and_bitset;
    uint32_t and_count;
};

CellResult run_cell(const Cell& c, uint32_t n_queries, int warmup, int iters) {
    CellResult R{c, 0, 0, 0, 0, 0, 0, {}, {}, 0, {}, {}, 0};

    cudaStream_t stream = 0;

    // --- build N CRoaring bitmaps on host -------------------------------------
    std::vector<roaring_bitmap_t*> cpu(c.n_bitmaps);
    for (uint32_t b = 0; b < c.n_bitmaps; ++b) {
        cpu[b] = build_cpu_bitmap(c.universe, c.selectivity, 17u * b + 1u);
    }
    std::vector<const roaring_bitmap_t*> cpu_ptrs(c.n_bitmaps);
    for (uint32_t b = 0; b < c.n_bitmaps; ++b) cpu_ptrs[b] = cpu[b];

    // --- upload (timed once) --------------------------------------------------
    cudaEvent_t s, e;
    check(cudaEventCreate(&s), "event create");
    check(cudaEventCreate(&e), "event create");
    check(cudaEventRecord(s, stream), "rec s");
    crv2::GpuRoaringBatch batch =
        crv2::upload_batch(cpu_ptrs.data(), c.n_bitmaps, stream);
    check(cudaEventRecord(e, stream), "rec e");
    check(cudaEventSynchronize(e), "sync e");
    {
        float ms = 0.f;
        check(cudaEventElapsedTime(&ms, s, e), "elapsed upload");
        R.upload_ms = ms;
    }
    R.roaring_bytes_raw = roaring_total_bytes(batch);

    // --- promote (timed once) -------------------------------------------------
    check(cudaEventRecord(s, stream), "rec s");
    crv2::GpuRoaringBatch promoted = crv2::promote_batch(batch, stream);
    check(cudaEventRecord(e, stream), "rec e");
    check(cudaEventSynchronize(e), "sync e");
    {
        float ms = 0.f;
        check(cudaEventElapsedTime(&ms, s, e), "elapsed promote");
        R.promote_ms = ms;
    }
    R.roaring_bytes_promoted = roaring_total_bytes(promoted);

    // --- materialise flat bitsets (also acts as decompress timing) -----------
    // host_universe_sizes is rounded up to the next 65536 (= (max_key+1) << 16)
    // because Roaring containers are key-aligned. The bitset baseline must use
    // the same bound so decompress_batch's precondition (`words_each >= ceil(
    // max(universe) / 64)`) holds and so contains() comparisons remain
    // apples-to-apples.
    uint32_t max_universe = 0u;
    for (uint32_t b = 0; b < c.n_bitmaps; ++b) {
        if (batch.host_universe_sizes[b] > max_universe) {
            max_universe = batch.host_universe_sizes[b];
        }
    }
    const uint64_t words_each = (static_cast<uint64_t>(max_universe) + 63ull)
                              / 64ull;
    const size_t   total_words =
        static_cast<size_t>(c.n_bitmaps) * static_cast<size_t>(words_each);

    uint64_t* d_bitsets = nullptr;
    check(cudaMalloc(&d_bitsets, total_words * sizeof(uint64_t)),
          "alloc bitsets");
    check(cudaMemset(d_bitsets, 0, total_words * sizeof(uint64_t)),
          "zero bitsets");

    check(cudaEventRecord(s, stream), "rec s");
    crv2::decompress_batch(batch, d_bitsets, words_each, stream);
    check(cudaEventRecord(e, stream), "rec e");
    check(cudaEventSynchronize(e), "sync e");
    {
        float ms = 0.f;
        check(cudaEventElapsedTime(&ms, s, e), "elapsed decompress");
        R.decompress_ms = ms;
    }
    R.bitset_bytes = total_words * sizeof(uint64_t);

    // --- query set: random (bitmap_idx, id) ----------------------------------
    std::vector<uint32_t> h_bidx(n_queries), h_ids(n_queries);
    {
        std::mt19937_64 rng(0xCAFEBABEull ^ static_cast<uint64_t>(c.n_bitmaps));
        for (uint32_t i = 0; i < n_queries; ++i) {
            h_bidx[i] = static_cast<uint32_t>(rng() % c.n_bitmaps);
            h_ids[i]  = static_cast<uint32_t>(rng() % c.universe);
        }
    }
    uint32_t* d_bidx = nullptr;
    uint32_t* d_ids  = nullptr;
    uint8_t*  d_out  = nullptr;
    check(cudaMalloc(&d_bidx, n_queries * sizeof(uint32_t)), "alloc bidx");
    check(cudaMalloc(&d_ids,  n_queries * sizeof(uint32_t)), "alloc ids");
    check(cudaMalloc(&d_out,  n_queries),                    "alloc out");
    check(cudaMemcpy(d_bidx, h_bidx.data(), n_queries * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy bidx");
    check(cudaMemcpy(d_ids,  h_ids.data(),  n_queries * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy ids");

    const uint32_t blocks = (n_queries + 255u) / 256u;
    auto roaring_q = [&]() {
        roaring_batch_contains_kernel<<<blocks, 256, 0, stream>>>(
            batch, d_bidx, d_ids, n_queries, d_out);
    };
    auto bitset_q = [&]() {
        bitset_batch_contains_kernel<<<blocks, 256, 0, stream>>>(
            d_bitsets, words_each, d_bidx, d_ids, n_queries, d_out);
    };
    R.contains_roaring = time_kernel(roaring_q, warmup, iters);
    R.contains_bitset  = time_kernel(bitset_q,  warmup, iters);
    R.n_queries        = static_cast<double>(n_queries);

    // --- multi_and: pick first min(n_bitmaps, 8) inputs ----------------------
    const uint32_t and_count = std::min(c.n_bitmaps, 8u);
    R.and_count = and_count;

    std::vector<uint32_t> h_and_indices(and_count);
    for (uint32_t i = 0; i < and_count; ++i) h_and_indices[i] = i;
    uint32_t* d_and_indices = nullptr;
    check(cudaMalloc(&d_and_indices, and_count * sizeof(uint32_t)),
          "alloc and_indices");
    check(cudaMemcpy(d_and_indices, h_and_indices.data(),
                     and_count * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy and_indices");

    uint64_t* d_and_out = nullptr;
    check(cudaMalloc(&d_and_out, words_each * sizeof(uint64_t)),
          "alloc and_out");

    const uint32_t and_blocks = static_cast<uint32_t>((words_each + 255ull) / 256ull);
    auto bitset_and = [&]() {
        bitset_multi_and_kernel<<<and_blocks, 256, 0, stream>>>(
            d_bitsets, words_each, d_and_indices, and_count, d_and_out);
    };

    // multi_and allocates / frees its own result; we keep a slot so the launch
    // lambda owns the result of the prior iter and frees it before the next.
    crv2::GpuRoaringBatch last_and{};
    auto roaring_and = [&]() {
        if (last_and._alloc_base != nullptr) crv2::free_batch(last_and);
        last_and = crv2::multi_and(promoted, h_and_indices.data(),
                                   and_count, stream);
    };

    // multi_and is heavier than contains; cap iters for the larger cells so we
    // don't blow up wall time.
    const int and_iters  = std::max(8, iters / 3);
    const int and_warmup = 2;
    R.and_roaring = time_kernel(roaring_and, and_warmup, and_iters);
    R.and_bitset  = time_kernel(bitset_and,  and_warmup, and_iters);

    if (last_and._alloc_base != nullptr) crv2::free_batch(last_and);

    // --- cleanup -------------------------------------------------------------
    cudaFree(d_and_indices);
    cudaFree(d_and_out);
    cudaFree(d_bidx);
    cudaFree(d_ids);
    cudaFree(d_out);
    cudaFree(d_bitsets);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    crv2::free_batch(promoted);
    crv2::free_batch(batch);
    for (uint32_t b = 0; b < c.n_bitmaps; ++b) roaring_bitmap_free(cpu[b]);

    return R;
}

void print_header() {
    std::printf(
        "n_bitmaps,universe,selectivity,"
        "upload_ms,promote_ms,decompress_ms,"
        "roaring_raw_MB,roaring_bmp_MB,bitset_MB,compression_x,"
        "contains_roaring_gqs_med,contains_roaring_gqs_stdev,"
        "contains_bitset_gqs_med,contains_bitset_gqs_stdev,"
        "contains_speedup,"
        "and_count,and_roaring_ms_med,and_roaring_ms_stdev,"
        "and_bitset_ms_med,and_bitset_ms_stdev,"
        "and_speedup\n");
}

void print_row(const CellResult& R) {
    auto gqs = [&](double median_ms) {
        return R.n_queries / (median_ms * 1e6);
    };
    auto gqs_stdev = [&](const Stats& st) {
        // Convert stdev_ms into a stdev in Gq/s by perturbing around the median.
        const double centre = R.n_queries / (st.median_ms * 1e6);
        const double low    = R.n_queries / ((st.median_ms + st.stdev_ms) * 1e6);
        return std::abs(centre - low);
    };

    const double cx_raw = static_cast<double>(R.bitset_bytes) /
                          static_cast<double>(R.roaring_bytes_raw == 0 ? 1
                                              : R.roaring_bytes_raw);
    const double contains_speedup = gqs(R.contains_roaring.median_ms) /
                                    gqs(R.contains_bitset.median_ms);
    const double and_speedup =
        R.and_bitset.median_ms / R.and_roaring.median_ms;

    std::printf(
        "%u,%llu,%.4f,"
        "%.3f,%.3f,%.3f,"
        "%.2f,%.2f,%.2f,%.2fx,"
        "%.3f,%.3f,"
        "%.3f,%.3f,"
        "%.2fx,"
        "%u,%.4f,%.4f,"
        "%.4f,%.4f,"
        "%.2fx\n",
        R.c.n_bitmaps,
        static_cast<unsigned long long>(R.c.universe),
        R.c.selectivity,
        R.upload_ms, R.promote_ms, R.decompress_ms,
        static_cast<double>(R.roaring_bytes_raw)      / (1024.0 * 1024.0),
        static_cast<double>(R.roaring_bytes_promoted) / (1024.0 * 1024.0),
        static_cast<double>(R.bitset_bytes)           / (1024.0 * 1024.0),
        cx_raw,
        gqs(R.contains_roaring.median_ms), gqs_stdev(R.contains_roaring),
        gqs(R.contains_bitset.median_ms),  gqs_stdev(R.contains_bitset),
        contains_speedup,
        R.and_count,
        R.and_roaring.median_ms, R.and_roaring.stdev_ms,
        R.and_bitset.median_ms,  R.and_bitset.stdev_ms,
        and_speedup);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    }

    cudaDeviceProp prop{};
    int dev = 0;
    check(cudaGetDevice(&dev), "get device");
    check(cudaGetDeviceProperties(&prop, dev), "get props");
    std::printf("# device: %s (SMs=%d, L2=%d KB, GMem=%.1f GB)\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024,
                static_cast<double>(prop.totalGlobalMem)
                    / (1024.0 * 1024.0 * 1024.0));

    std::vector<Cell> grid;
    if (quick) {
        grid = {
            {4u,  1'000'000ull, 0.01},
            {16u, 1'000'000ull, 0.10},
        };
    } else {
        const uint32_t Ns[]   = {1u, 4u, 16u, 64u};
        const uint64_t Us[]   = {1'000'000ull, 10'000'000ull};
        const double   Ss[]   = {0.001, 0.01, 0.10};
        for (auto u : Us) for (auto s : Ss) for (auto n : Ns) {
            grid.push_back({n, u, s});
        }
    }

    const uint32_t n_queries = quick ? 200'000u : 1'000'000u;
    const int      warmup    = 5;
    const int      iters     = 30;

    std::printf("# queries per contains trial: %u, warmup=%d, iters=%d\n",
                n_queries, warmup, iters);

    // PTX → SASS JIT pass: the first time each kernel runs on this GPU, the
    // driver compiles SM-specific code from our embedded SM_89 PTX (we ship
    // PTX for forward-compat to e.g. SM_120 on RTX 5090). That JIT step shows
    // up as ~100ms on the first call. Run one tiny cell here so the timed
    // sweep below sees nothing but cache-warm SASS.
    {
        const Cell warmup_cell{4u, 1'000'000ull, 0.01};
        (void)run_cell(warmup_cell, 50'000u, 1, 3);
    }

    print_header();
    for (const auto& c : grid) {
        CellResult R = run_cell(c, n_queries, warmup, iters);
        print_row(R);
    }
    return 0;
}
