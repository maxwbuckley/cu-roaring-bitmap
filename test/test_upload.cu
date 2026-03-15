#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"

#include <cstring>
#include <random>
#include <vector>

class UploadTest : public ::testing::Test {
 protected:
    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(UploadTest, EmptyBitmap) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    auto gpu = cu_roaring::upload(r);
    EXPECT_EQ(gpu.n_containers, 0u);
    EXPECT_EQ(gpu.keys, nullptr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(UploadTest, SingleElement) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add(r, 42);

    auto meta = cu_roaring::get_meta(r);
    EXPECT_EQ(meta.n_containers, 1u);
    EXPECT_EQ(meta.n_array_containers, 1u);
    EXPECT_EQ(meta.n_bitmap_containers, 0u);

    auto gpu = cu_roaring::upload(r);
    EXPECT_EQ(gpu.n_containers, 1u);
    EXPECT_EQ(gpu.n_array_containers, 1u);

    // Download key and verify
    uint16_t key = 0;
    cudaMemcpy(&key, gpu.keys, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(key, 0u);  // element 42 has high-16 key = 0

    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(UploadTest, MixedContainerTypes) {
    roaring_bitmap_t* r = roaring_bitmap_create();

    // Create an array container (few elements in key=0 range)
    for (uint32_t i = 0; i < 100; ++i) {
        roaring_bitmap_add(r, i);
    }

    // Create a bitmap container (many elements in key=1 range)
    for (uint32_t i = 65536; i < 65536 + 50000; ++i) {
        roaring_bitmap_add(r, i);
    }

    // Create a run container (run optimization)
    roaring_bitmap_t* run_bm = roaring_bitmap_create();
    for (uint32_t i = 131072; i < 131072 + 10000; ++i) {
        roaring_bitmap_add(run_bm, i);
    }
    roaring_bitmap_run_optimize(run_bm);
    roaring_bitmap_or_inplace(r, run_bm);
    roaring_bitmap_run_optimize(r);
    roaring_bitmap_free(run_bm);

    auto meta = cu_roaring::get_meta(r);
    EXPECT_EQ(meta.n_containers, 3u);
    EXPECT_GE(meta.n_bitmap_containers + meta.n_array_containers + meta.n_run_containers, 3u);

    auto gpu = cu_roaring::upload(r);
    EXPECT_EQ(gpu.n_containers, 3u);
    EXPECT_EQ(gpu.n_containers,
              gpu.n_bitmap_containers + gpu.n_array_containers + gpu.n_run_containers);

    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(UploadTest, RoundTrip) {
    // Create bitmap, upload to GPU, decompress, verify against CRoaring
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 gen(12345);
    std::uniform_int_distribution<uint32_t> dist(0, 999999);

    for (int i = 0; i < 50000; ++i) {
        roaring_bitmap_add(r, dist(gen));
    }

    uint64_t cpu_card = roaring_bitmap_get_cardinality(r);
    auto gpu = cu_roaring::upload(r);

    // Decompress to flat bitset
    uint32_t universe = gpu.universe_size;
    uint32_t n_words = (universe + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    // Copy back to host
    std::vector<uint32_t> h_bitset(n_words);
    cudaMemcpy(h_bitset.data(), d_bitset, n_words * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // Verify each bit
    uint64_t gpu_card = 0;
    for (uint32_t w = 0; w < n_words; ++w) {
        gpu_card += __builtin_popcount(h_bitset[w]);
    }
    EXPECT_EQ(gpu_card, cpu_card);

    // Check specific elements
    roaring_uint32_iterator_t* iter = roaring_iterator_create(r);
    while (iter->has_value) {
        uint32_t val = iter->current_value;
        uint32_t word = val / 32;
        uint32_t bit = val % 32;
        EXPECT_TRUE(h_bitset[word] & (1u << bit))
            << "Missing bit for value " << val;
        roaring_uint32_iterator_advance(iter);
    }
    roaring_uint32_iterator_free(iter);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(UploadTest, LargeBitmap) {
    roaring_bitmap_t* r = roaring_bitmap_create();

    // Add ranges to create many containers
    for (uint32_t key = 0; key < 100; ++key) {
        uint32_t base = key * 65536;
        for (uint32_t i = 0; i < 1000; ++i) {
            roaring_bitmap_add(r, base + i * 50);
        }
    }

    auto gpu = cu_roaring::upload(r);
    EXPECT_EQ(gpu.n_containers, 100u);

    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}
