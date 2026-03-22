/*
 * B10: YFCC-10M Filtered Track Benchmark
 *
 * Uses the real Big-ANN NeurIPS'23 filtered track dataset:
 * - 10M vectors, 192-dim uint8 CLIP, 200K-vocab tags
 * - 100K queries, each requiring 1-2 tags
 * - Real-world Zipfian tag distribution (0.0006% to 33.9%)
 *
 * Measures:
 * 1. Per-tag bitmap upload latency (upload_from_ids)
 * 2. Multi-predicate AND latency (fused multi_and)
 * 3. Memory usage: roaring vs flat bitset
 * 4. Point query throughput on the resulting filter
 *
 * Prerequisites: run bench/yfcc_export.py first.
 */

#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/device/make_view.cuh"
#include "cu_roaring/device/roaring_view.cuh"
#include "cu_roaring/device/roaring_warp_query.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

// ============================================================================
// Data loading
// ============================================================================

struct TagData {
    uint32_t tag_id;
    std::vector<uint32_t> ids;
};

static TagData load_tag(const std::string& path)
{
    TagData td;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return td; }
    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, f) != 2) { fclose(f); return td; }
    td.tag_id = header[1];
    td.ids.resize(header[0]);
    if (fread(td.ids.data(), sizeof(uint32_t), header[0], f) != header[0]) {
        td.ids.clear();
    }
    fclose(f);
    return td;
}

struct Query {
    std::vector<uint32_t> tag_ids;
};

static std::vector<Query> load_queries(const std::string& path)
{
    std::vector<Query> queries;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return queries;
    uint32_t n_queries;
    if (fread(&n_queries, sizeof(uint32_t), 1, f) != 1) { fclose(f); return queries; }
    queries.resize(n_queries);
    for (uint32_t i = 0; i < n_queries; ++i) {
        uint32_t n_tags;
        if (fread(&n_tags, sizeof(uint32_t), 1, f) != 1) break;
        queries[i].tag_ids.resize(n_tags);
        if (fread(queries[i].tag_ids.data(), sizeof(uint32_t), n_tags, f) != n_tags) break;
    }
    fclose(f);
    return queries;
}

// ============================================================================
// Helpers
// ============================================================================

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

// Point query kernel
__global__ void query_kernel(cu_roaring::GpuRoaringView view,
                              const uint32_t* queries,
                              uint32_t* results,
                              uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

__global__ void warp_query_kernel(cu_roaring::GpuRoaringView view,
                                   const uint32_t* queries,
                                   uint32_t* results,
                                   uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = cu_roaring::warp_contains(view, queries[idx]) ? 1u : 0u;
}

__global__ void bitset_query_kernel(const uint32_t* bitset,
                                     const uint32_t* queries,
                                     uint32_t* results,
                                     uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t id = queries[idx];
    results[idx] = (bitset[id >> 5] >> (id & 31)) & 1u;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    const char* data_dir = "bench/yfcc_data";
    if (argc > 1) data_dir = argv[1];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("Dataset: YFCC-10M (Big-ANN NeurIPS'23 filtered track)\n\n");

    constexpr uint32_t UNIVERSE = 10000000;

    // Load queries
    auto queries = load_queries(std::string(data_dir) + "/queries.bin");
    printf("Loaded %zu queries\n", queries.size());

    // Collect all unique tags needed
    std::map<uint32_t, TagData> tag_cache;
    for (auto& q : queries) {
        for (auto tid : q.tag_ids) {
            if (tag_cache.find(tid) == tag_cache.end()) {
                char path[256];
                snprintf(path, sizeof(path), "%s/tags/tag_%u.bin", data_dir, tid);
                tag_cache[tid] = load_tag(path);
            }
        }
    }
    printf("Loaded %zu unique tags\n\n", tag_cache.size());

    // ================================================================
    // 1. Upload latency: build GPU roaring bitmaps for all tags
    // ================================================================
    printf("=== Tag Upload Latency ===\n");

    // Categorize tags by density
    struct DensityBucket {
        const char* name;
        double lo, hi;
        std::vector<uint32_t> tag_ids;
        double total_upload_ms;
        size_t total_roaring_bytes;
    };
    DensityBucket buckets[] = {
        {"rare (<0.1%)",      0.0,    0.001, {}, 0, 0},
        {"uncommon (0.1-1%)", 0.001,  0.01,  {}, 0, 0},
        {"medium (1-10%)",    0.01,   0.10,  {}, 0, 0},
        {"common (10-50%)",   0.10,   0.50,  {}, 0, 0},
        {"dominant (>50%)",   0.50,   1.01,  {}, 0, 0},
    };

    std::map<uint32_t, cu_roaring::GpuRoaring> gpu_tags;
    for (auto& [tid, td] : tag_cache) {
        double density = static_cast<double>(td.ids.size()) / UNIVERSE;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto bm = cu_roaring::upload_from_ids(
            td.ids.data(), static_cast<uint32_t>(td.ids.size()), UNIVERSE);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        size_t roaring_bytes = static_cast<size_t>(bm.n_bitmap_containers) * 1024 * sizeof(uint64_t)
                             + bm.n_containers * (sizeof(uint16_t) * 2 + sizeof(uint8_t) + sizeof(uint32_t));

        for (auto& b : buckets) {
            if (density >= b.lo && density < b.hi) {
                b.tag_ids.push_back(tid);
                b.total_upload_ms += ms;
                b.total_roaring_bytes += roaring_bytes;
                break;
            }
        }

        gpu_tags[tid] = std::move(bm);
    }

    size_t total_bitset_bytes = tag_cache.size() * ((UNIVERSE + 31) / 32) * sizeof(uint32_t);
    size_t total_roaring_bytes = 0;

    printf("%-25s %8s %12s %12s %10s\n",
           "Bucket", "Tags", "Avg Upload", "Roaring MB", "Compress");
    for (auto& b : buckets) {
        if (b.tag_ids.empty()) continue;
        double avg_ms = b.total_upload_ms / b.tag_ids.size();
        double roaring_mb = b.total_roaring_bytes / 1e6;
        double flat_mb = b.tag_ids.size() * ((UNIVERSE + 31.0) / 32) * sizeof(uint32_t) / 1e6;
        printf("%-25s %8zu %10.2f ms %10.2f MB %9.1fx\n",
               b.name, b.tag_ids.size(), avg_ms, roaring_mb, flat_mb / std::max(0.001, roaring_mb));
        total_roaring_bytes += b.total_roaring_bytes;
    }
    printf("\nTotal memory: roaring=%.1f MB  bitset=%.1f MB  compression=%.1fx\n",
           total_roaring_bytes / 1e6, total_bitset_bytes / 1e6,
           static_cast<double>(total_bitset_bytes) / std::max(static_cast<size_t>(1), total_roaring_bytes));

    // ================================================================
    // 2. Multi-predicate AND latency for sample queries
    // ================================================================
    printf("\n=== Multi-Predicate AND Latency (sample of 1000 queries) ===\n");

    std::vector<double> and_1pred_times, and_2pred_times;
    int n_sample = std::min(static_cast<int>(queries.size()), 1000);

    for (int q = 0; q < n_sample; ++q) {
        auto& qry = queries[q];

        auto t0 = std::chrono::high_resolution_clock::now();
        if (qry.tag_ids.size() == 1) {
            // Single tag: no AND needed, just use the tag bitmap directly
            auto t1 = std::chrono::high_resolution_clock::now();
            and_1pred_times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        } else {
            // Multi-tag AND
            std::vector<cu_roaring::GpuRoaring*> ptrs;
            for (auto tid : qry.tag_ids) {
                ptrs.push_back(&gpu_tags[tid]);
            }
            std::vector<cu_roaring::GpuRoaring> refs;
            for (auto* p : ptrs) refs.push_back(*p);  // shallow copy for multi_and
            auto result = cu_roaring::multi_and(refs.data(),
                                                 static_cast<uint32_t>(refs.size()));
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            and_2pred_times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
            cu_roaring::gpu_roaring_free(result);
        }
    }

    if (!and_1pred_times.empty()) {
        auto s1 = compute_stats(and_1pred_times);
        printf("  1-predicate: median=%.3f ms (n=%zu)\n", s1.median, and_1pred_times.size());
    }
    if (!and_2pred_times.empty()) {
        auto s2 = compute_stats(and_2pred_times);
        printf("  2-predicate AND: median=%.3f ms (n=%zu)\n", s2.median, and_2pred_times.size());
    }

    // ================================================================
    // 3. Point query throughput on combined filters
    // ================================================================
    printf("\n=== Point Query Throughput (10M random queries) ===\n");

    // Pick a representative 2-tag query
    uint32_t sample_tid1 = 0, sample_tid2 = 0;
    for (auto& q : queries) {
        if (q.tag_ids.size() == 2) {
            sample_tid1 = q.tag_ids[0];
            sample_tid2 = q.tag_ids[1];
            break;
        }
    }

    printf("  Sample query: tag %u (%.2f%%) AND tag %u (%.2f%%)\n",
           sample_tid1,
           100.0 * tag_cache[sample_tid1].ids.size() / UNIVERSE,
           sample_tid2,
           100.0 * tag_cache[sample_tid2].ids.size() / UNIVERSE);

    // Build combined filter
    cu_roaring::GpuRoaring inputs[] = {gpu_tags[sample_tid1], gpu_tags[sample_tid2]};
    auto combined = cu_roaring::multi_and(inputs, 2);
    auto view = cu_roaring::make_view(combined);

    printf("  Combined filter: %u containers (bmp=%u), cardinality=%llu\n",
           combined.n_containers, combined.n_bitmap_containers,
           (unsigned long long)combined.total_cardinality);

    // Build flat bitset for comparison
    auto* d_bitset_flat = cu_roaring::decompress_to_bitset(combined);

    // Generate random query IDs
    constexpr uint32_t NQ = 10000000;
    std::vector<uint32_t> h_qids(NQ);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, UNIVERSE - 1);
    for (auto& id : h_qids) id = dist(gen);

    uint32_t* d_qids;
    uint32_t* d_results;
    cudaMalloc(&d_qids, NQ * sizeof(uint32_t));
    cudaMalloc(&d_results, NQ * sizeof(uint32_t));
    cudaMemcpy(d_qids, h_qids.data(), NQ * sizeof(uint32_t), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((NQ + 255) / 256);

    auto s_bitset = bench_gpu(10, 50, [&]() {
        bitset_query_kernel<<<grid, block>>>(d_bitset_flat, d_qids, d_results, NQ);
    });

    auto s_contains = bench_gpu(10, 50, [&]() {
        query_kernel<<<grid, block>>>(view, d_qids, d_results, NQ);
    });

    auto s_warp = bench_gpu(10, 50, [&]() {
        warp_query_kernel<<<grid, block>>>(view, d_qids, d_results, NQ);
    });

    double bs_gqps = NQ / (s_bitset.median * 1e-3) / 1e9;
    double ct_gqps = NQ / (s_contains.median * 1e-3) / 1e9;
    double wc_gqps = NQ / (s_warp.median * 1e-3) / 1e9;

    printf("  bitset:         %.3f ms (%.2f Gq/s)\n", s_bitset.median, bs_gqps);
    printf("  contains:       %.3f ms (%.2f Gq/s) %.2fx vs bitset\n",
           s_contains.median, ct_gqps, s_bitset.median / s_contains.median);
    printf("  warp_contains:  %.3f ms (%.2f Gq/s) %.2fx vs bitset\n",
           s_warp.median, wc_gqps, s_bitset.median / s_warp.median);

    // Cleanup
    cudaFree(d_qids);
    cudaFree(d_results);
    cudaFree(d_bitset_flat);
    cu_roaring::gpu_roaring_free(combined);
    for (auto& [tid, bm] : gpu_tags) cu_roaring::gpu_roaring_free(bm);

    printf("\n=== B10 COMPLETE ===\n");
    return 0;
}
