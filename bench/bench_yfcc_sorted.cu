/*
 * Benchmark: YFCC-10M tag filters — unsorted bitset vs unsorted Roaring vs
 * lex-sorted Roaring. Measures query latency, GPU memory, and batch capacity.
 *
 * 1. Loads all tag bitmaps from bench/yfcc_data/tags/
 * 2. Builds lex-sort mapping (sort docs by their sorted tag tuple)
 * 3. For a sample of tags at various densities:
 *    a. Flat bitset from original IDs
 *    b. Unsorted CRoaring from original IDs
 *    c. Sorted CRoaring from remapped IDs
 *    d. Benchmarks 10M random point queries for each
 * 4. Reports memory, query latency, batch capacity
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
#include <dirent.h>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
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

static std::string fmt_size(size_t bytes) {
    char buf[32];
    if (bytes < 1024)
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    else if (bytes < 1024 * 1024)
        snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
    return buf;
}

// ============================================================================
// Load tags from disk
// ============================================================================

struct TagData {
    uint32_t tag_id;
    std::vector<uint32_t> doc_ids;  // sorted ascending
};

static std::vector<TagData> load_tags(const std::string& dir) {
    std::vector<TagData> tags;
    DIR* d = opendir(dir.c_str());
    if (!d) return tags;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".bin") continue;
        if (name.substr(0, 4) != "tag_") continue;

        uint32_t tag_id = std::stoul(name.substr(4, name.size() - 8));
        std::string path = dir + "/" + name;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t n = sz / sizeof(uint32_t);
        std::vector<uint32_t> ids(n);
        if (fread(ids.data(), sizeof(uint32_t), n, f) == n) {
            std::sort(ids.begin(), ids.end());
            tags.push_back({tag_id, std::move(ids)});
        }
        fclose(f);
    }
    closedir(d);
    return tags;
}

// ============================================================================
// Build lex-sort mapping
// ============================================================================

// Returns old_id → new_id mapping
static std::vector<uint32_t> build_lex_sort(const std::vector<TagData>& tags,
                                              uint32_t n_docs) {
    // Build per-doc tag lists
    std::vector<std::vector<uint32_t>> doc_tags(n_docs);
    for (const auto& t : tags) {
        for (uint32_t doc : t.doc_ids) {
            if (doc < n_docs) doc_tags[doc].push_back(t.tag_id);
        }
    }
    // Sort tags within each doc
    for (auto& dt : doc_tags) std::sort(dt.begin(), dt.end());

    // Sort doc IDs by their tag tuple
    std::vector<uint32_t> order(n_docs);
    for (uint32_t i = 0; i < n_docs; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return doc_tags[a] < doc_tags[b];
    });

    // Build old→new mapping
    std::vector<uint32_t> remap(n_docs);
    for (uint32_t new_id = 0; new_id < n_docs; ++new_id) {
        remap[order[new_id]] = new_id;
    }
    return remap;
}

// ============================================================================
// Roaring GPU memory estimate
// ============================================================================

static size_t roaring_gpu_size(const cu_roaring::GpuRoaring& bm,
                                const cu_roaring::GpuRoaringMeta& meta) {
    size_t bytes;
    if (bm.n_array_containers == 0 && bm.n_run_containers == 0) {
        bytes = static_cast<size_t>(bm.n_bitmap_containers) * 1024 * sizeof(uint64_t);
    } else {
        bytes = meta.total_bytes;
    }
    bytes += bm.n_containers * (2 + 1 + 4 + 2);
    if (bm.key_index)
        bytes += (static_cast<size_t>(bm.max_key) + 1) * 2;
    return bytes;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    constexpr uint32_t N_DOCS = 10000000;
    constexpr uint32_t N_QUERIES = 10000000;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 50;

    const size_t bitset_bytes = (static_cast<size_t>(N_DOCS) + 31) / 32 * sizeof(uint32_t);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Universe: %u  Bitset: %s\n\n", N_DOCS, fmt_size(bitset_bytes).c_str());

    // ---- Load tags ----
    printf("Loading tags...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tags = load_tags("bench/yfcc_data/tags");
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Loaded %zu tags in %.1fs\n", tags.size(),
           std::chrono::duration<double>(t1 - t0).count());

    // Sort tags by cardinality descending for ranking
    std::sort(tags.begin(), tags.end(),
              [](const TagData& a, const TagData& b) {
                  return a.doc_ids.size() > b.doc_ids.size();
              });

    // ---- Build lex-sort mapping ----
    printf("Building lex-sort mapping...\n");
    t0 = std::chrono::high_resolution_clock::now();
    auto remap = build_lex_sort(tags, N_DOCS);
    t1 = std::chrono::high_resolution_clock::now();
    printf("  Built in %.1fs\n\n", std::chrono::duration<double>(t1 - t0).count());

    // ---- Pre-generate queries ----
    std::vector<uint32_t> h_queries(N_QUERIES);
    {
        std::mt19937 gen(777);
        std::uniform_int_distribution<uint32_t> dist(0, N_DOCS - 1);
        for (uint32_t i = 0; i < N_QUERIES; ++i) h_queries[i] = dist(gen);
    }
    uint32_t* d_queries = nullptr;
    cudaMalloc(&d_queries, N_QUERIES * sizeof(uint32_t));
    cudaMemcpy(d_queries, h_queries.data(), N_QUERIES * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    uint32_t* d_results = nullptr;
    cudaMalloc(&d_results, N_QUERIES * sizeof(uint32_t));

    // Bitset device buffer (reused)
    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, bitset_bytes);

    dim3 block(256);
    dim3 grid((N_QUERIES + 255) / 256);

    // ---- Select sample tags at various ranks ----
    std::vector<int> sample_ranks = {0, 1, 4, 9, 19, 29, 43, 49, 99, 199, 499, 999};
    // Remove ranks beyond available tags
    while (!sample_ranks.empty() &&
           sample_ranks.back() >= static_cast<int>(tags.size()))
        sample_ranks.pop_back();

    // ---- Header ----
    printf("%-6s %6s %10s  %10s %9s %9s  %10s %9s %9s  %10s %9s %9s  %9s %9s\n",
           "Rank", "TagID", "Card",
           "BS mem", "BS q(ms)", "BS Gq/s",
           "UR mem", "UR q(ms)", "UR Gq/s",
           "SR mem", "SR q(ms)", "SR Gq/s",
           "SR/BS", "SR/UR");
    printf("%s\n", std::string(145, '-').c_str());

    // Accumulate for batch summary
    struct Result {
        int rank;
        uint32_t tag_id;
        uint32_t card;
        size_t bs_mem, ur_mem, sr_mem;
        double bs_ms, ur_ms, sr_ms;
    };
    std::vector<Result> results;

    for (int rank : sample_ranks) {
        const auto& tag = tags[rank];
        uint32_t card = static_cast<uint32_t>(tag.doc_ids.size());
        double density = static_cast<double>(card) / N_DOCS;

        // ---- 1. Flat bitset ----
        {
            uint32_t n_words = (N_DOCS + 31) / 32;
            std::vector<uint32_t> h_bs(n_words, 0);
            for (uint32_t id : tag.doc_ids)
                if (id / 32 < n_words) h_bs[id / 32] |= (1u << (id % 32));
            cudaMemcpy(d_bitset, h_bs.data(), bitset_bytes, cudaMemcpyHostToDevice);
        }
        auto bs_stats = bench_gpu(WARMUP, ITERS, [&]() {
            bitset_query_kernel<<<grid, block>>>(d_bitset, d_queries, d_results, N_QUERIES);
        });

        // ---- 2. Unsorted Roaring ----
        roaring_bitmap_t* cpu_unsorted = roaring_bitmap_create();
        for (uint32_t id : tag.doc_ids) roaring_bitmap_add(cpu_unsorted, id);
        roaring_bitmap_run_optimize(cpu_unsorted);

        auto gpu_unsorted = cu_roaring::upload(cpu_unsorted, N_DOCS);
        auto meta_unsorted = cu_roaring::get_meta(cpu_unsorted);
        size_t ur_mem = roaring_gpu_size(gpu_unsorted, meta_unsorted);

        auto ur_view = cu_roaring::make_view(gpu_unsorted);
        auto ur_stats = bench_gpu(WARMUP, ITERS, [&]() {
            roaring_contains_kernel<<<grid, block>>>(ur_view, d_queries, d_results, N_QUERIES);
        });

        // ---- 3. Sorted Roaring ----
        roaring_bitmap_t* cpu_sorted = roaring_bitmap_create();
        for (uint32_t id : tag.doc_ids) {
            if (id < N_DOCS) roaring_bitmap_add(cpu_sorted, remap[id]);
        }
        roaring_bitmap_run_optimize(cpu_sorted);

        auto gpu_sorted = cu_roaring::upload(cpu_sorted, N_DOCS);
        auto meta_sorted = cu_roaring::get_meta(cpu_sorted);
        size_t sr_mem = roaring_gpu_size(gpu_sorted, meta_sorted);

        auto sr_view = cu_roaring::make_view(gpu_sorted);
        auto sr_stats = bench_gpu(WARMUP, ITERS, [&]() {
            roaring_contains_kernel<<<grid, block>>>(sr_view, d_queries, d_results, N_QUERIES);
        });

        // ---- Print row ----
        double bs_gqps = N_QUERIES / (bs_stats.median * 1e-3) / 1e9;
        double ur_gqps = N_QUERIES / (ur_stats.median * 1e-3) / 1e9;
        double sr_gqps = N_QUERIES / (sr_stats.median * 1e-3) / 1e9;

        printf("%-6d %6u %10u  %10s %9.3f %8.2f  %10s %9.3f %8.2f  %10s %9.3f %8.2f  %8.1fx %8.1fx\n",
               rank + 1, tag.tag_id, card,
               fmt_size(bitset_bytes).c_str(), bs_stats.median, bs_gqps,
               fmt_size(ur_mem).c_str(), ur_stats.median, ur_gqps,
               fmt_size(sr_mem).c_str(), sr_stats.median, sr_gqps,
               bs_stats.median / sr_stats.median,
               ur_stats.median / sr_stats.median);

        results.push_back({rank + 1, tag.tag_id, card,
                           bitset_bytes, ur_mem, sr_mem,
                           bs_stats.median, ur_stats.median, sr_stats.median});

        cu_roaring::gpu_roaring_free(gpu_unsorted);
        cu_roaring::gpu_roaring_free(gpu_sorted);
        roaring_bitmap_free(cpu_unsorted);
        roaring_bitmap_free(cpu_sorted);
    }

    // ---- Batch capacity summary ----
    printf("\n");
    printf("BATCH CAPACITY: How many concurrent filters fit in 8 GB?\n");
    printf("=========================================================\n");
    constexpr size_t BUDGET = 8ULL * 1024 * 1024 * 1024;
    printf("%-6s %10s  %12s %12s %12s  %12s %12s\n",
           "Rank", "Card",
           "Bitset", "Unsorted R", "Sorted R",
           "SR/BS ratio", "SR/UR ratio");
    printf("%s\n", std::string(85, '-').c_str());

    for (const auto& r : results) {
        uint64_t bs_batch = BUDGET / r.bs_mem;
        uint64_t ur_batch = BUDGET / r.ur_mem;
        uint64_t sr_batch = BUDGET / r.sr_mem;
        printf("%-6d %10u  %12llu %12llu %12llu  %11.0fx %11.0fx\n",
               r.rank, r.card,
               (unsigned long long)bs_batch,
               (unsigned long long)ur_batch,
               (unsigned long long)sr_batch,
               static_cast<double>(sr_batch) / bs_batch,
               static_cast<double>(sr_batch) / ur_batch);
    }

    cudaFree(d_queries);
    cudaFree(d_results);
    cudaFree(d_bitset);

    printf("\n=== COMPLETE ===\n");
    return 0;
}
