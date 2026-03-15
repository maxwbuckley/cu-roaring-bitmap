#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"

#include <random>
#include <vector>

class DecompressTest : public ::testing::Test {
 protected:
    void TearDown() override {
        cudaDeviceReset();
    }

    // Helper: verify GPU bitset matches CRoaring bitmap exactly
    void verify_bitset(const roaring_bitmap_t* cpu_bm,
                       const uint32_t* d_bitset,
                       uint32_t n_words,
                       uint32_t universe_size)
    {
        std::vector<uint32_t> h_bitset(n_words, 0);
        cudaMemcpy(h_bitset.data(), d_bitset, n_words * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        uint64_t cpu_card = roaring_bitmap_get_cardinality(cpu_bm);

        // Count GPU popcount
        uint64_t gpu_card = 0;
        for (uint32_t w = 0; w < n_words; ++w) {
            gpu_card += __builtin_popcount(h_bitset[w]);
        }
        EXPECT_EQ(gpu_card, cpu_card) << "Cardinality mismatch";

        // Verify every CPU bit is set in GPU output
        roaring_uint32_iterator_t* iter = roaring_iterator_create(cpu_bm);
        while (iter->has_value) {
            uint32_t val = iter->current_value;
            if (val < universe_size) {
                uint32_t word = val / 32;
                uint32_t bit = val % 32;
                EXPECT_TRUE(h_bitset[word] & (1u << bit))
                    << "Missing bit for value " << val;
            }
            roaring_uint32_iterator_advance(iter);
        }
        roaring_uint32_iterator_free(iter);

        // Verify no extra bits set
        for (uint32_t w = 0; w < n_words; ++w) {
            uint32_t word = h_bitset[w];
            while (word) {
                uint32_t bit = __builtin_ctz(word);
                uint32_t val = w * 32 + bit;
                EXPECT_TRUE(roaring_bitmap_contains(cpu_bm, val))
                    << "Extra bit set for value " << val;
                word &= word - 1;  // clear lowest set bit
            }
        }
    }
};

TEST_F(DecompressTest, EmptyBitmap) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    auto gpu = cu_roaring::upload(r);

    // Empty bitmap should return nullptr
    uint32_t* bitset = cu_roaring::decompress_to_bitset(gpu);
    EXPECT_EQ(bitset, nullptr);

    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, SingleElement) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add(r, 42);

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, ArrayContainers) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    // Sparse data → array containers
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, 999999);
    for (int i = 0; i < 5000; ++i) {
        roaring_bitmap_add(r, dist(gen));
    }

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, BitmapContainers) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    // Dense data → bitmap containers
    for (uint32_t i = 0; i < 50000; ++i) {
        roaring_bitmap_add(r, i);
    }

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, RunContainers) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    // Contiguous ranges → run containers after optimization
    for (uint32_t i = 1000; i < 5000; ++i) {
        roaring_bitmap_add(r, i);
    }
    for (uint32_t i = 70000; i < 80000; ++i) {
        roaring_bitmap_add(r, i);
    }
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, MixedContainerTypes) {
    roaring_bitmap_t* r = roaring_bitmap_create();

    // Array container (key=0): sparse
    for (uint32_t i = 0; i < 100; i += 3) {
        roaring_bitmap_add(r, i);
    }

    // Bitmap container (key=1): dense
    for (uint32_t i = 65536; i < 65536 + 40000; ++i) {
        roaring_bitmap_add(r, i);
    }

    // Run container (key=2): contiguous
    for (uint32_t i = 131072; i < 131072 + 20000; ++i) {
        roaring_bitmap_add(r, i);
    }
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, PreAllocatedBuffer) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    for (uint32_t i = 0; i < 10000; ++i) {
        roaring_bitmap_add(r, i * 7);
    }

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;

    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));

    cu_roaring::decompress_to_bitset(gpu, d_bitset, n_words);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, BoundaryElement) {
    // Test element at max value within a container (65535)
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add(r, 65535);
    roaring_bitmap_add(r, 65536);  // First element of key=1

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(DecompressTest, FullContainer) {
    // All 65536 bits set in one container
    roaring_bitmap_t* r = roaring_bitmap_create();
    for (uint32_t i = 0; i < 65536; ++i) {
        roaring_bitmap_add(r, i);
    }

    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu);

    verify_bitset(r, d_bitset, n_words, gpu.universe_size);

    cudaFree(d_bitset);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}
