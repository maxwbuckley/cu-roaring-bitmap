#include <benchmark/benchmark.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "data_gen.cuh"

#include <cstring>
#include <vector>

// Baseline: upload a flat bitset (uncompressed) to GPU
static void BM_FlatBitsetTransfer(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    double density = state.range(1) / 1000.0;

    auto* cpu_bm = cu_roaring::bench::generate_bitmap(universe, density, 42);

    // Create flat bitset on host
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> h_bitset(n_words, 0);
    roaring_uint32_iterator_t* iter = roaring_iterator_create(cpu_bm);
    while (iter->has_value) {
        uint32_t v = iter->current_value;
        h_bitset[v / 32] |= (1u << (v % 32));
        roaring_uint32_iterator_advance(iter);
    }
    roaring_uint32_iterator_free(iter);

    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    for (auto _ : state) {
        cudaEventRecord(start_ev);
        cudaMemcpy(d_bitset, h_bitset.data(), n_words * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);
    }

    double size_mb = n_words * sizeof(uint32_t) / (1024.0 * 1024.0);
    state.counters["transfer_MB"] = size_mb;

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cudaFree(d_bitset);
    roaring_bitmap_free(cpu_bm);
}

// Our approach: upload compressed Roaring + decompress on GPU
static void BM_CompressedTransferAndDecompress(benchmark::State& state) {
    uint32_t universe = static_cast<uint32_t>(state.range(0));
    double density = state.range(1) / 1000.0;

    auto* cpu_bm = cu_roaring::bench::generate_bitmap(universe, density, 42);

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    // Pre-allocate output buffer
    uint32_t n_words = (universe + 31) / 32;
    uint32_t* d_output = nullptr;
    cudaMalloc(&d_output, n_words * sizeof(uint32_t));

    for (auto _ : state) {
        cudaEventRecord(start_ev);

        // Upload compressed
        auto gpu_bm = cu_roaring::upload(cpu_bm);
        // Decompress on GPU
        cu_roaring::decompress_to_bitset(gpu_bm, d_output, n_words);
        cudaDeviceSynchronize();

        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        state.SetIterationTime(ms / 1000.0);

        cu_roaring::gpu_roaring_free(gpu_bm);
    }

    auto meta = cu_roaring::get_meta(cpu_bm);
    double compressed_mb = meta.total_bytes / (1024.0 * 1024.0);
    double flat_mb = n_words * sizeof(uint32_t) / (1024.0 * 1024.0);
    state.counters["compressed_MB"] = compressed_mb;
    state.counters["flat_MB"] = flat_mb;
    state.counters["ratio"] = flat_mb / compressed_mb;

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cudaFree(d_output);
    roaring_bitmap_free(cpu_bm);
}

BENCHMARK(BM_FlatBitsetTransfer)
    ->Args({1000000, 100})
    ->Args({1000000, 500})
    ->Args({10000000, 100})
    ->Args({10000000, 500})
    ->Args({100000000, 100})
    ->Args({100000000, 500})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_CompressedTransferAndDecompress)
    ->Args({1000000, 100})
    ->Args({1000000, 500})
    ->Args({10000000, 100})
    ->Args({10000000, 500})
    ->Args({100000000, 100})
    ->Args({100000000, 500})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();
