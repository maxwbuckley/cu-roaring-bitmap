/*
 * Benchmark: Load real CRoaring bitmaps from disk and compare
 *   Roaring upload vs flat bitset transfer — copy time, memory, query latency.
 *
 * Input: directory of .bin files in CRoaring portable serialization format,
 *        produced by roaring-benchmark's `sweep --save-bitmaps`.
 *
 * Filename convention:  {distribution}_s{selectivity:.6f}_t{trial}.bin
 *   e.g.  clustered_s0.010000_t0.bin
 *
 * Usage:
 *   bench_from_disk <bitmap_dir> [universe_size]
 *
 * Default universe_size = 1000000000 (1B).
 *
 * For each bitmap file, measures:
 *   - Roaring upload latency   (cu_roaring::upload from CRoaring)
 *   - Flat bitset H→D latency  (cudaMemcpy of universe/8 bytes)
 *   - Roaring GPU memory       (compressed)
 *   - Flat bitset GPU memory   (universe/8 bytes)
 *   - Point query: flat_bitset vs contains vs warp_contains
 *
 * Outputs JSON to results/raw/bench_from_disk.json
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
#include <cstring>
#include <dirent.h>
#include <functional>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

// ============================================================================
// File I/O
// ============================================================================

static std::vector<char> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != static_cast<size_t>(sz)) {
        fclose(f);
        return {};
    }
    fclose(f);
    return buf;
}

struct BitmapFile {
    std::string path;
    std::string filename;
    std::string distribution;
    double      selectivity;
    int         trial;
};

static bool parse_filename(const std::string& fname, BitmapFile& out) {
    // Format: {dist}_s{sel:.6f}_t{trial}.bin
    auto pos_s = fname.rfind("_s");
    auto pos_t = fname.rfind("_t");
    auto pos_dot = fname.rfind(".bin");
    if (pos_s == std::string::npos || pos_t == std::string::npos ||
        pos_dot == std::string::npos)
        return false;

    out.distribution = fname.substr(0, pos_s);
    out.selectivity = std::stod(fname.substr(pos_s + 2, pos_t - pos_s - 2));
    out.trial = std::stoi(fname.substr(pos_t + 2, pos_dot - pos_t - 2));
    out.filename = fname;
    return true;
}

static std::vector<BitmapFile> list_bitmap_files(const std::string& dir) {
    std::vector<BitmapFile> files;
    DIR* d = opendir(dir.c_str());
    if (!d) return files;

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".bin")
            continue;
        BitmapFile bf;
        if (parse_filename(name, bf)) {
            bf.path = dir + "/" + name;
            files.push_back(bf);
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end(), [](const BitmapFile& a, const BitmapFile& b) {
        if (a.distribution != b.distribution) return a.distribution < b.distribution;
        if (a.selectivity != b.selectivity) return a.selectivity < b.selectivity;
        return a.trial < b.trial;
    });
    return files;
}

// ============================================================================
// Statistics
// ============================================================================

struct Stats {
    double median, mean, p5, p95, std_dev, min_val, max_val;
};

static Stats compute_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    int n = static_cast<int>(times.size());
    double sum = 0;
    for (auto t : times) sum += t;
    double mean = sum / n;
    double var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    return {times[n / 2],
            mean,
            times[std::max(0, static_cast<int>(n * 0.05))],
            times[std::min(n - 1, static_cast<int>(n * 0.95))],
            std::sqrt(var / n),
            times[0],
            times[n - 1]};
}

static void write_stats(FILE* f, const char* name, const Stats& s) {
    fprintf(f,
            "\"%s\": {\"median\": %.4f, \"mean\": %.4f, \"p5\": %.4f, "
            "\"p95\": %.4f, \"std\": %.4f, \"min\": %.4f, \"max\": %.4f}",
            name, s.median, s.mean, s.p5, s.p95, s.std_dev, s.min_val,
            s.max_val);
}

// ============================================================================
// GPU timing helpers
// ============================================================================

// Returns Stats in milliseconds for a GPU kernel/operation
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

// Returns Stats in milliseconds for a host-timed operation (includes CPU work)
static Stats bench_host(int warmup, int iters, std::function<void()> fn) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return compute_stats(times);
}

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
    uint32_t word = bitset[id >> 5];
    results[idx] = (word >> (id & 31)) & 1u;
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
// Flat bitset construction
// ============================================================================

static std::vector<uint32_t> make_host_bitset(const roaring_bitmap_t* bm,
                                               uint32_t universe) {
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> h_bitset(n_words, 0);
    roaring_uint32_iterator_t* iter = roaring_iterator_create(bm);
    while (iter->has_value) {
        uint32_t v = iter->current_value;
        if (v / 32 < n_words)
            h_bitset[v / 32] |= (1u << (v % 32));
        roaring_uint32_iterator_advance(iter);
    }
    roaring_uint32_iterator_free(iter);
    return h_bitset;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <bitmap_dir> [universe_size]\n", argv[0]);
        return 1;
    }

    const std::string bitmap_dir = argv[1];
    const uint32_t universe = (argc >= 3) ? static_cast<uint32_t>(std::stoul(argv[2]))
                                          : 1000000000u;

    auto files = list_bitmap_files(bitmap_dir);
    if (files.empty()) {
        fprintf(stderr, "No .bin files found in %s\n", bitmap_dir.c_str());
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.0f MB)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Universe: %u (%.0f MB flat bitset)\n",
           universe,
           (static_cast<size_t>(universe) + 31) / 32 * sizeof(uint32_t) / (1024.0 * 1024.0));
    printf("Found %zu bitmap files in %s\n\n", files.size(), bitmap_dir.c_str());

    // Ensure output directory exists
    mkdir("results", 0755);
    mkdir("results/raw", 0755);

    const char* json_path = "results/raw/bench_from_disk.json";
    FILE* jf = fopen(json_path, "w");
    if (!jf) {
        fprintf(stderr, "Cannot open %s for writing\n", json_path);
        return 1;
    }

    fprintf(jf, "{\n  \"benchmark\": \"from_disk\",\n");
    fprintf(jf, "  \"gpu\": \"%s\",\n", prop.name);
    fprintf(jf, "  \"universe\": %u,\n", universe);
    fprintf(jf, "  \"n_files\": %zu,\n", files.size());

    const size_t bitset_bytes = (static_cast<size_t>(universe) + 31) / 32 * sizeof(uint32_t);
    fprintf(jf, "  \"flat_bitset_bytes\": %zu,\n", bitset_bytes);
    fprintf(jf, "  \"results\": [\n");

    // Pre-generate random queries
    constexpr uint32_t N_QUERIES = 10000000;
    std::vector<uint32_t> h_queries(N_QUERIES);
    {
        std::mt19937 gen(777);
        std::uniform_int_distribution<uint32_t> dist(0, universe - 1);
        for (uint32_t i = 0; i < N_QUERIES; ++i)
            h_queries[i] = dist(gen);
    }

    uint32_t* d_queries = nullptr;
    cudaMalloc(&d_queries, N_QUERIES * sizeof(uint32_t));
    cudaMemcpy(d_queries, h_queries.data(), N_QUERIES * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    // Result buffers
    uint32_t* d_results_bs = nullptr;
    uint32_t* d_results_ct = nullptr;
    uint32_t* d_results_wc = nullptr;
    cudaMalloc(&d_results_bs, N_QUERIES * sizeof(uint32_t));
    cudaMalloc(&d_results_ct, N_QUERIES * sizeof(uint32_t));
    cudaMalloc(&d_results_wc, N_QUERIES * sizeof(uint32_t));

    // Pre-allocate flat bitset on device (reused — 125 MB for 1B universe)
    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, bitset_bytes);

    // Pre-allocate pinned host bitset buffer (reused)
    uint32_t* h_bitset_pinned = nullptr;
    cudaMallocHost(&h_bitset_pinned, bitset_bytes);

    dim3 block(256);
    dim3 grid((N_QUERIES + 255) / 256);

    constexpr int UPLOAD_WARMUP = 3;
    constexpr int UPLOAD_ITERS  = 30;
    constexpr int QUERY_WARMUP  = 10;
    constexpr int QUERY_ITERS   = 50;

    bool first = true;

    for (size_t fi = 0; fi < files.size(); ++fi) {
        const auto& bf = files[fi];

        // Load CRoaring bitmap from portable serialization
        auto buf = read_file(bf.path);
        if (buf.empty()) {
            fprintf(stderr, "  SKIP %s (cannot read)\n", bf.filename.c_str());
            continue;
        }

        roaring_bitmap_t* cpu_bm = roaring_bitmap_portable_deserialize_safe(
            buf.data(), buf.size());
        if (!cpu_bm) {
            fprintf(stderr, "  SKIP %s (deserialize failed)\n", bf.filename.c_str());
            continue;
        }
        roaring_bitmap_run_optimize(cpu_bm);

        uint64_t card = roaring_bitmap_get_cardinality(cpu_bm);
        double actual_density = static_cast<double>(card) / universe;
        size_t serialized_bytes = buf.size();

        auto meta = cu_roaring::get_meta(cpu_bm);

        printf("[%3zu/%zu] %-45s card=%12llu (%.4f%%) containers=%u  ",
               fi + 1, files.size(), bf.filename.c_str(),
               (unsigned long long)card, actual_density * 100.0,
               meta.n_containers);
        fflush(stdout);

        // ----------------------------------------------------------------
        // 1. Roaring upload latency (includes CPU scan + H→D transfer)
        // ----------------------------------------------------------------
        cu_roaring::GpuRoaring gpu_bm{};  // keep from last iteration for queries
        auto upload_stats = bench_host(UPLOAD_WARMUP, UPLOAD_ITERS, [&]() {
            if (gpu_bm.keys || gpu_bm._alloc_base)
                cu_roaring::gpu_roaring_free(gpu_bm);
            gpu_bm = cu_roaring::upload(cpu_bm, universe);
        });

        // Compute actual GPU footprint AFTER upload (may or may not be promoted).
        // When all-bitmap (promoted): n_bitmap * 8 KB. When mixed: use
        // get_meta() which sums actual CRoaring container sizes.
        size_t roaring_data_bytes;
        if (gpu_bm.n_array_containers == 0 && gpu_bm.n_run_containers == 0) {
            // All bitmap — compute directly from container count
            roaring_data_bytes =
                static_cast<size_t>(gpu_bm.n_bitmap_containers) * 1024 * sizeof(uint64_t);
        } else {
            // Mixed containers (not promoted) — meta.total_bytes is accurate
            roaring_data_bytes = meta.total_bytes;
        }
        // Add metadata overhead: keys, types, offsets, cardinalities, key_index
        size_t roaring_gpu_bytes = roaring_data_bytes +
            gpu_bm.n_containers * (sizeof(uint16_t) + sizeof(uint8_t) +
                                    sizeof(uint32_t) + sizeof(uint16_t));
        if (gpu_bm.key_index)
            roaring_gpu_bytes += (static_cast<size_t>(gpu_bm.max_key) + 1) * sizeof(uint16_t);
        size_t roaring_pre_promotion_bytes = meta.total_bytes;

        // ----------------------------------------------------------------
        // 2. Flat bitset transfer latency (H→D memcpy only, host bitset pre-built)
        // ----------------------------------------------------------------
        // Build host bitset once (into pinned memory)
        {
            auto h_bs = make_host_bitset(cpu_bm, universe);
            std::memcpy(h_bitset_pinned, h_bs.data(), bitset_bytes);
        }

        auto bitset_stats = bench_gpu(UPLOAD_WARMUP, UPLOAD_ITERS, [&]() {
            cudaMemcpyAsync(d_bitset, h_bitset_pinned, bitset_bytes,
                            cudaMemcpyHostToDevice);
        });

        // ----------------------------------------------------------------
        // 3. Point query latency
        // ----------------------------------------------------------------
        auto view = cu_roaring::make_view(gpu_bm);

        auto bs_query = bench_gpu(QUERY_WARMUP, QUERY_ITERS, [&]() {
            bitset_query_kernel<<<grid, block>>>(
                d_bitset, d_queries, d_results_bs, N_QUERIES);
        });

        auto ct_query = bench_gpu(QUERY_WARMUP, QUERY_ITERS, [&]() {
            roaring_contains_kernel<<<grid, block>>>(
                view, d_queries, d_results_ct, N_QUERIES);
        });

        auto wc_query = bench_gpu(QUERY_WARMUP, QUERY_ITERS, [&]() {
            roaring_warp_contains_kernel<<<grid, block>>>(
                view, d_queries, d_results_wc, N_QUERIES);
        });

        // Correctness
        uint32_t mm_ct = 0, mm_wc = 0;
        {
            std::vector<uint32_t> hbs(N_QUERIES), hct(N_QUERIES), hwc(N_QUERIES);
            cudaMemcpy(hbs.data(), d_results_bs, N_QUERIES * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(hct.data(), d_results_ct, N_QUERIES * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(hwc.data(), d_results_wc, N_QUERIES * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < N_QUERIES; ++i) {
                if (hbs[i] != hct[i]) ++mm_ct;
                if (hbs[i] != hwc[i]) ++mm_wc;
            }
        }

        // Print summary line
        double upload_speedup = bitset_stats.median > 0
                                    ? upload_stats.median / bitset_stats.median
                                    : 0;
        double mem_ratio = roaring_gpu_bytes > 0
                               ? static_cast<double>(bitset_bytes) / roaring_gpu_bytes
                               : 0;

        printf("upload: roaring=%.2fms bitset=%.2fms (%.1fx)  "
               "gpu_mem: roaring=%.2fMB bitset=%.2fMB (%.1fx)  "
               "query(ms): bs=%.3f ct=%.3f wc=%.3f",
               upload_stats.median, bitset_stats.median, upload_speedup,
               roaring_gpu_bytes / 1e6, bitset_bytes / 1e6, mem_ratio,
               bs_query.median, ct_query.median, wc_query.median);
        if (mm_ct > 0 || mm_wc > 0)
            printf("  *** MISMATCH ct=%u wc=%u ***", mm_ct, mm_wc);
        printf("\n");

        // ----------------------------------------------------------------
        // Write JSON entry
        // ----------------------------------------------------------------
        if (!first) fprintf(jf, ",\n");
        first = false;

        fprintf(jf, "    {\n");
        fprintf(jf, "      \"file\": \"%s\",\n", bf.filename.c_str());
        fprintf(jf, "      \"distribution\": \"%s\",\n", bf.distribution.c_str());
        fprintf(jf, "      \"selectivity\": %.6f,\n", bf.selectivity);
        fprintf(jf, "      \"trial\": %d,\n", bf.trial);
        fprintf(jf, "      \"cardinality\": %llu,\n", (unsigned long long)card);
        fprintf(jf, "      \"actual_density\": %.8f,\n", actual_density);
        fprintf(jf, "      \"negated\": %s,\n", gpu_bm.negated ? "true" : "false");
        fprintf(jf, "      \"n_containers\": %u,\n", gpu_bm.n_containers);
        fprintf(jf, "      \"n_bitmap\": %u, \"n_array\": %u, \"n_run\": %u,\n",
                gpu_bm.n_bitmap_containers, gpu_bm.n_array_containers,
                gpu_bm.n_run_containers);
        fprintf(jf, "      \"n_containers_original\": %u,\n", meta.n_containers);
        fprintf(jf, "      \"n_bitmap_original\": %u, \"n_array_original\": %u, \"n_run_original\": %u,\n",
                meta.n_bitmap_containers, meta.n_array_containers,
                meta.n_run_containers);
        fprintf(jf, "      \"serialized_bytes\": %zu,\n", serialized_bytes);
        fprintf(jf, "      \"roaring_pre_promotion_bytes\": %zu,\n", roaring_pre_promotion_bytes);
        fprintf(jf, "      \"roaring_gpu_bytes\": %zu,\n", roaring_gpu_bytes);
        fprintf(jf, "      \"flat_bitset_bytes\": %zu,\n", bitset_bytes);
        fprintf(jf, "      \"gpu_compression_ratio\": %.2f,\n", mem_ratio);
        fprintf(jf, "      \"pre_promotion_compression_ratio\": %.2f,\n",
                roaring_pre_promotion_bytes > 0
                    ? static_cast<double>(bitset_bytes) / roaring_pre_promotion_bytes : 0.0);
        fprintf(jf, "      \"n_queries\": %u,\n", N_QUERIES);
        fprintf(jf, "      ");
        write_stats(jf, "upload_roaring_ms", upload_stats);
        fprintf(jf, ",\n      ");
        write_stats(jf, "upload_bitset_ms", bitset_stats);
        fprintf(jf, ",\n      ");
        write_stats(jf, "query_bitset_ms", bs_query);
        fprintf(jf, ",\n      ");
        write_stats(jf, "query_contains_ms", ct_query);
        fprintf(jf, ",\n      ");
        write_stats(jf, "query_warp_contains_ms", wc_query);
        fprintf(jf, ",\n");
        fprintf(jf, "      \"upload_speedup\": %.4f,\n", upload_speedup);
        fprintf(jf, "      \"correctness\": %s\n",
                (mm_ct == 0 && mm_wc == 0) ? "true" : "false");
        fprintf(jf, "    }");

        cu_roaring::gpu_roaring_free(gpu_bm);
        roaring_bitmap_free(cpu_bm);
    }

    fprintf(jf, "\n  ]\n}\n");
    fclose(jf);

    cudaFree(d_queries);
    cudaFree(d_results_bs);
    cudaFree(d_results_ct);
    cudaFree(d_results_wc);
    cudaFree(d_bitset);
    cudaFreeHost(h_bitset_pinned);

    printf("\n=== COMPLETE — results written to %s ===\n", json_path);
    return 0;
}
