// bench_vs_bitset — compare cu_roaring::v2 against flat uint64 bitsets on the
// two operations a filtered-vector-search pipeline cares about:
//
//   1. contains()      — called from inside CAGRA traversal, millions of times
//                        per query batch. Throughput (Gq/s) is the figure.
//   2. multi_and()     — filter composition across N predicates. Latency (ms)
//                        and output size are the figures.
//
// Sweeps universe size × selectivity and prints a CSV-style table on stdout.
// All roaring bitmaps are upload_from_croaring-built from a CPU CRoaring bitmap
// constructed with uniform random IDs. For multi_and, inputs are pre-promoted
// (v2 requires that explicitly) so the comparison is apples-to-apples: each
// side ANDs all-bitmap roaring containers vs full bitsets.

#include "cu_roaring_v2/api.hpp"
#include "cu_roaring_v2/query.cuh"

#include <roaring/roaring.h>
#include <cuda_runtime.h>

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

// --- query kernels ---------------------------------------------------------

__global__ void roaring_contains_kernel(crv2::GpuRoaring bm,
                                        const uint32_t* ids,
                                        uint32_t        n,
                                        uint8_t*        out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = crv2::contains(bm, ids[i]) ? 1u : 0u;
}

__global__ void bitset_contains_kernel(const uint64_t* __restrict__ bitset,
                                       uint64_t        words,
                                       const uint32_t* __restrict__ ids,
                                       uint32_t        n,
                                       uint8_t*        out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t id = ids[i];
    const uint64_t w  = id >> 6;
    if (w >= words) { out[i] = 0u; return; }
    out[i] = ((__ldg(&bitset[w]) >> (id & 63u)) & 1ULL) ? 1u : 0u;
}

// --- bitset multi-AND ------------------------------------------------------

__global__ void bitset_multi_and_kernel(const uint64_t* const* __restrict__ inputs,
                                        uint32_t       count,
                                        uint64_t*      __restrict__ out,
                                        uint64_t       words) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words) return;
    uint64_t acc = ~0ULL;
    #pragma unroll 1
    for (uint32_t j = 0; j < count; ++j) acc &= inputs[j][i];
    out[i] = acc;
}

// --- synthetic bitmap construction ----------------------------------------

roaring_bitmap_t* build_cpu_bitmap(uint64_t universe, double selectivity,
                                   uint64_t seed) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    const uint64_t target = static_cast<uint64_t>(
        static_cast<double>(universe) * selectivity);
    std::mt19937_64 rng(seed);
    // Add in batches to reduce per-call overhead.
    constexpr size_t BATCH = 4096;
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

// Build a device-resident flat bitset with the same logical contents as `bm`.
// Uses v2's decompress kernel, which is already differentially tested, so any
// correctness bug shows up in the bitset baseline too (fair comparison).
uint64_t* build_device_bitset(const crv2::GpuRoaring& bm, uint64_t words) {
    uint64_t* d = nullptr;
    check(cudaMalloc(&d, words * sizeof(uint64_t)), "alloc bitset");
    check(cudaMemset(d, 0, words * sizeof(uint64_t)), "zero bitset");
    crv2::decompress_to_bitset(bm, d, words);
    check(cudaDeviceSynchronize(), "decompress sync");
    return d;
}

size_t roaring_bytes(const crv2::GpuRoaring& bm) {
    // Exactly matches what free_bitmap() owns: the single packed allocation.
    // Count the nominal bytes of each section since we don't retain total_bytes.
    size_t b = 0;
    b += bm.n_containers * sizeof(uint16_t);        // keys
    b += bm.n_containers * sizeof(crv2::ContainerType);
    b += bm.n_containers * sizeof(uint32_t);        // offsets
    b += bm.n_containers * sizeof(uint16_t);        // cardinalities
    b += (static_cast<size_t>(bm.max_key) + 1u) * sizeof(uint16_t);  // key_index
    b += bm.n_bitmap_containers * 1024u * sizeof(uint64_t);
    b += bm.array_pool_bytes;
    b += bm.run_pool_bytes;
    return b;
}

// --- timing ----------------------------------------------------------------

template <typename F>
float time_ms(F&& launch, int warmup, int iters) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop),  "event create");

    for (int i = 0; i < warmup; ++i) launch();
    check(cudaDeviceSynchronize(), "warmup sync");

    check(cudaEventRecord(start), "event record");
    for (int i = 0; i < iters; ++i) launch();
    check(cudaEventRecord(stop),  "event record");
    check(cudaEventSynchronize(stop), "event sync");

    float total_ms = 0.0f;
    check(cudaEventElapsedTime(&total_ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / static_cast<float>(iters);
}

// --- one cell of the sweep -------------------------------------------------

struct ContainsResult {
    double roaring_gqs;
    double bitset_gqs;
    size_t roaring_bytes;
    size_t bitset_bytes;
};

ContainsResult bench_contains(uint64_t universe, double selectivity,
                              uint32_t n_queries, uint64_t seed) {
    roaring_bitmap_t* cpu = build_cpu_bitmap(universe, selectivity, seed);
    crv2::GpuRoaring  bm  = crv2::upload_from_croaring(cpu);

    const uint64_t bitset_words = (universe + 63u) / 64u;
    uint64_t* d_bitset = build_device_bitset(bm, bitset_words);

    // Random queries drawn from [0, universe). Matches a CAGRA traversal where
    // neighbours are scattered across the universe.
    std::vector<uint32_t> queries(n_queries);
    {
        std::mt19937_64 rng(seed ^ 0xABCDEFull);
        for (uint32_t i = 0; i < n_queries; ++i) {
            queries[i] = static_cast<uint32_t>(rng() % universe);
        }
    }
    uint32_t* d_ids = nullptr;
    uint8_t*  d_out = nullptr;
    check(cudaMalloc(&d_ids, n_queries * sizeof(uint32_t)), "alloc ids");
    check(cudaMalloc(&d_out, n_queries),                    "alloc out");
    check(cudaMemcpy(d_ids, queries.data(), n_queries * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "memcpy ids");

    const uint32_t blocks = (n_queries + 255u) / 256u;

    auto roaring_launch = [&]() {
        roaring_contains_kernel<<<blocks, 256>>>(bm, d_ids, n_queries, d_out);
    };
    auto bitset_launch = [&]() {
        bitset_contains_kernel<<<blocks, 256>>>(
            d_bitset, bitset_words, d_ids, n_queries, d_out);
    };

    const float roaring_ms = time_ms(roaring_launch, 3, 10);
    const float bitset_ms  = time_ms(bitset_launch,  3, 10);

    ContainsResult r;
    r.roaring_gqs   = static_cast<double>(n_queries) / (roaring_ms * 1e6);
    r.bitset_gqs    = static_cast<double>(n_queries) / (bitset_ms  * 1e6);
    r.roaring_bytes = roaring_bytes(bm);
    r.bitset_bytes  = bitset_words * sizeof(uint64_t);

    cudaFree(d_ids);
    cudaFree(d_out);
    cudaFree(d_bitset);
    crv2::free_bitmap(bm);
    roaring_bitmap_free(cpu);
    return r;
}

struct MultiAndResult {
    float  roaring_ms;
    float  bitset_ms;
    size_t roaring_result_bytes;
    size_t bitset_result_bytes;
};

MultiAndResult bench_multi_and(uint64_t universe, double selectivity,
                               uint32_t count, uint64_t seed) {
    std::vector<roaring_bitmap_t*> cpu_inputs(count);
    std::vector<crv2::GpuRoaring>  gpu_inputs(count);
    std::vector<crv2::GpuRoaring>  gpu_inputs_bmp(count);
    std::vector<uint64_t*>         d_bitsets(count);

    const uint64_t bitset_words = (universe + 63u) / 64u;

    for (uint32_t i = 0; i < count; ++i) {
        cpu_inputs[i]     = build_cpu_bitmap(universe, selectivity, seed + i);
        gpu_inputs[i]     = crv2::upload_from_croaring(cpu_inputs[i]);
        gpu_inputs_bmp[i] = crv2::promote_to_bitmap(gpu_inputs[i]);
        d_bitsets[i]      = build_device_bitset(gpu_inputs[i], bitset_words);
    }

    // Bitset baseline: device-side array of input pointers, one kernel.
    std::vector<const uint64_t*> h_input_ptrs(count);
    for (uint32_t i = 0; i < count; ++i) h_input_ptrs[i] = d_bitsets[i];
    const uint64_t** d_input_ptrs = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_input_ptrs),
                     count * sizeof(const uint64_t*)),
          "alloc input_ptrs");
    check(cudaMemcpy(d_input_ptrs, h_input_ptrs.data(),
                     count * sizeof(const uint64_t*),
                     cudaMemcpyHostToDevice),
          "memcpy input_ptrs");

    uint64_t* d_bitset_out = nullptr;
    check(cudaMalloc(&d_bitset_out, bitset_words * sizeof(uint64_t)),
          "alloc bitset_out");

    const uint32_t bs_blocks = static_cast<uint32_t>(
        (bitset_words + 255u) / 256u);
    auto bitset_launch = [&]() {
        bitset_multi_and_kernel<<<bs_blocks, 256>>>(
            d_input_ptrs, count, d_bitset_out, bitset_words);
    };

    // Roaring multi_and allocates its result each call — we free the previous
    // result inside the launch lambda to avoid leaking across iters.
    crv2::GpuRoaring last_result{};
    auto roaring_launch = [&]() {
        if (last_result._alloc_base != nullptr) crv2::free_bitmap(last_result);
        last_result = crv2::multi_and(gpu_inputs_bmp.data(), count);
    };

    const float roaring_ms = time_ms(roaring_launch, 2, 5);
    const float bitset_ms  = time_ms(bitset_launch,  2, 5);

    MultiAndResult r;
    r.roaring_ms           = roaring_ms;
    r.bitset_ms            = bitset_ms;
    r.roaring_result_bytes = roaring_bytes(last_result);
    r.bitset_result_bytes  = bitset_words * sizeof(uint64_t);

    cudaFree(d_input_ptrs);
    cudaFree(d_bitset_out);
    if (last_result._alloc_base != nullptr) crv2::free_bitmap(last_result);
    for (uint32_t i = 0; i < count; ++i) {
        cudaFree(d_bitsets[i]);
        crv2::free_bitmap(gpu_inputs_bmp[i]);
        crv2::free_bitmap(gpu_inputs[i]);
        roaring_bitmap_free(cpu_inputs[i]);
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    // Defaults: three universe sizes, five selectivities. Override on the CLI
    // with --quick for a single small configuration (for smoke-testing).
    std::vector<uint64_t> universes   = {1'000'000ull, 10'000'000ull, 100'000'000ull};
    std::vector<double>   sels        = {0.001, 0.01, 0.1, 0.5, 0.9};
    uint32_t              n_queries   = 1'000'000u;
    uint32_t              and_count   = 4u;

    if (argc > 1 && std::strcmp(argv[1], "--quick") == 0) {
        universes = {1'000'000ull};
        sels      = {0.01, 0.1};
        n_queries = 100'000u;
    }

    cudaDeviceProp prop{};
    int dev = 0;
    check(cudaGetDevice(&dev), "get device");
    check(cudaGetDeviceProperties(&prop, dev), "get props");
    std::printf("# device: %s (SMs=%d, L2=%d KB)\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024);
    std::printf("# queries per contains trial: %u ; multi_and count: %u\n",
                n_queries, and_count);
    std::printf("# universe, selectivity, "
                "roaring_contains_gqs, bitset_contains_gqs, contains_speedup, "
                "roaring_bytes, bitset_bytes, compression_x, "
                "roaring_and_ms, bitset_and_ms, and_speedup\n");

    for (uint64_t u : universes) {
        for (double s : sels) {
            const ContainsResult  c = bench_contains(u, s, n_queries, 42u);
            const MultiAndResult  a = bench_multi_and(u, s, and_count, 1000u);

            const double compress =
                static_cast<double>(c.bitset_bytes) /
                static_cast<double>(c.roaring_bytes == 0 ? 1 : c.roaring_bytes);
            const double c_speedup = c.roaring_gqs / c.bitset_gqs;
            const double a_speedup = static_cast<double>(a.bitset_ms) /
                                     static_cast<double>(a.roaring_ms);

            std::printf(
                "%llu, %.3f, "
                "%.3f, %.3f, %.2fx, "
                "%zu, %zu, %.2fx, "
                "%.3f, %.3f, %.2fx\n",
                static_cast<unsigned long long>(u), s,
                c.roaring_gqs, c.bitset_gqs, c_speedup,
                c.roaring_bytes, c.bitset_bytes, compress,
                static_cast<double>(a.roaring_ms),
                static_cast<double>(a.bitset_ms), a_speedup);
            std::fflush(stdout);
        }
    }
    return 0;
}
