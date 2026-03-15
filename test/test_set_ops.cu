#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"

#include <random>
#include <vector>

class SetOpsTest : public ::testing::Test {
 protected:
    void TearDown() override {
        cudaDeviceReset();
    }

    roaring_bitmap_t* make_random(uint32_t universe, double density, uint64_t seed) {
        roaring_bitmap_t* r = roaring_bitmap_create();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (uint32_t i = 0; i < universe; ++i) {
            if (dist(gen) < density) {
                roaring_bitmap_add(r, i);
            }
        }
        return r;
    }

    // Verify GPU result matches CPU result exactly by decompressing both to bitsets
    void verify_match(const roaring_bitmap_t* cpu_result,
                      const cu_roaring::GpuRoaring& gpu_result) {
        uint64_t cpu_card = roaring_bitmap_get_cardinality(cpu_result);

        if (cpu_card == 0 && gpu_result.n_containers == 0) {
            return;  // Both empty — OK
        }

        // Determine universe for comparison
        uint32_t universe = gpu_result.universe_size;
        if (universe == 0) {
            // GPU result is empty, check CPU is too
            EXPECT_EQ(cpu_card, 0u);
            return;
        }

        // Get CPU bitset
        uint32_t n_words = (universe + 31) / 32;
        std::vector<uint32_t> cpu_bitset(n_words, 0);

        roaring_uint32_iterator_t* iter = roaring_iterator_create(cpu_result);
        while (iter->has_value) {
            uint32_t val = iter->current_value;
            if (val / 32 < n_words) {
                cpu_bitset[val / 32] |= (1u << (val % 32));
            }
            roaring_uint32_iterator_advance(iter);
        }
        roaring_uint32_iterator_free(iter);

        // Get GPU bitset
        uint32_t* d_bitset = cu_roaring::decompress_to_bitset(gpu_result);
        std::vector<uint32_t> gpu_bitset(n_words, 0);
        cudaMemcpy(gpu_bitset.data(), d_bitset, n_words * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_bitset);

        // Compare
        uint64_t gpu_card = 0;
        for (uint32_t w = 0; w < n_words; ++w) {
            gpu_card += __builtin_popcount(gpu_bitset[w]);
        }
        EXPECT_EQ(gpu_card, cpu_card) << "Cardinality mismatch: GPU=" << gpu_card
                                      << " CPU=" << cpu_card;

        for (uint32_t w = 0; w < n_words; ++w) {
            EXPECT_EQ(gpu_bitset[w], cpu_bitset[w])
                << "Mismatch at word " << w
                << " (bits " << (w * 32) << "-" << (w * 32 + 31) << ")";
        }
    }
};

// ============================================================================
// AND tests
// ============================================================================
TEST_F(SetOpsTest, AND_BitmapBitmap) {
    // Dense bitmaps → bitmap containers
    auto* a = make_random(200000, 0.6, 100);
    auto* b = make_random(200000, 0.4, 200);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);

    roaring_bitmap_t* cpu_result = roaring_bitmap_and(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

TEST_F(SetOpsTest, AND_ArrayArray) {
    // Sparse bitmaps → array containers
    auto* a = make_random(200000, 0.02, 300);
    auto* b = make_random(200000, 0.01, 400);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);

    roaring_bitmap_t* cpu_result = roaring_bitmap_and(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

TEST_F(SetOpsTest, AND_BitmapArray) {
    // One dense, one sparse → bitmap x array pairs
    auto* a = make_random(200000, 0.5, 500);
    auto* b = make_random(200000, 0.02, 600);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);

    roaring_bitmap_t* cpu_result = roaring_bitmap_and(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

TEST_F(SetOpsTest, AND_EmptyResult) {
    // Non-overlapping ranges
    roaring_bitmap_t* a = roaring_bitmap_create();
    roaring_bitmap_t* b = roaring_bitmap_create();
    for (uint32_t i = 0; i < 1000; ++i) roaring_bitmap_add(a, i);
    for (uint32_t i = 65536; i < 66536; ++i) roaring_bitmap_add(b, i);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);

    EXPECT_EQ(gpu_result.n_containers, 0u);

    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

TEST_F(SetOpsTest, AND_IdenticalBitmaps) {
    auto* a = make_random(100000, 0.3, 700);
    auto* b = roaring_bitmap_copy(a);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);

    // AND of identical bitmaps should equal the input
    verify_match(a, gpu_result);

    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

// ============================================================================
// OR tests
// ============================================================================
TEST_F(SetOpsTest, OR_BitmapBitmap) {
    auto* a = make_random(200000, 0.5, 800);
    auto* b = make_random(200000, 0.3, 900);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::OR);

    roaring_bitmap_t* cpu_result = roaring_bitmap_or(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

TEST_F(SetOpsTest, OR_NonOverlapping) {
    roaring_bitmap_t* a = roaring_bitmap_create();
    roaring_bitmap_t* b = roaring_bitmap_create();
    for (uint32_t i = 0; i < 1000; ++i) roaring_bitmap_add(a, i);
    for (uint32_t i = 65536; i < 66536; ++i) roaring_bitmap_add(b, i);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::OR);

    roaring_bitmap_t* cpu_result = roaring_bitmap_or(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

// ============================================================================
// ANDNOT tests
// ============================================================================
TEST_F(SetOpsTest, ANDNOT_BitmapBitmap) {
    auto* a = make_random(200000, 0.5, 1000);
    auto* b = make_random(200000, 0.3, 1100);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::ANDNOT);

    roaring_bitmap_t* cpu_result = roaring_bitmap_andnot(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

// ============================================================================
// XOR tests
// ============================================================================
TEST_F(SetOpsTest, XOR_BitmapBitmap) {
    auto* a = make_random(200000, 0.5, 1200);
    auto* b = make_random(200000, 0.3, 1300);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::XOR);

    roaring_bitmap_t* cpu_result = roaring_bitmap_xor(a, b);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

TEST_F(SetOpsTest, XOR_IdenticalBitmaps) {
    auto* a = make_random(100000, 0.4, 1400);
    auto* b = roaring_bitmap_copy(a);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::XOR);

    // XOR of identical bitmaps should be empty
    EXPECT_EQ(gpu_result.n_containers, 0u);

    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
}

// ============================================================================
// Multi-AND / Multi-OR tests
// ============================================================================
TEST_F(SetOpsTest, MultiAND) {
    auto* a = make_random(200000, 0.6, 1500);
    auto* b = make_random(200000, 0.5, 1600);
    auto* c = make_random(200000, 0.4, 1700);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_c = cu_roaring::upload(c);

    cu_roaring::GpuRoaring bitmaps[] = {gpu_a, gpu_b, gpu_c};
    auto gpu_result = cu_roaring::multi_and(bitmaps, 3);

    roaring_bitmap_t* temp = roaring_bitmap_and(a, b);
    roaring_bitmap_t* cpu_result = roaring_bitmap_and(temp, c);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(temp);
    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    cu_roaring::gpu_roaring_free(gpu_c);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
    roaring_bitmap_free(c);
}

TEST_F(SetOpsTest, MultiOR) {
    auto* a = make_random(200000, 0.1, 1800);
    auto* b = make_random(200000, 0.1, 1900);
    auto* c = make_random(200000, 0.1, 2000);

    auto gpu_a = cu_roaring::upload(a);
    auto gpu_b = cu_roaring::upload(b);
    auto gpu_c = cu_roaring::upload(c);

    cu_roaring::GpuRoaring bitmaps[] = {gpu_a, gpu_b, gpu_c};
    auto gpu_result = cu_roaring::multi_or(bitmaps, 3);

    roaring_bitmap_t* temp = roaring_bitmap_or(a, b);
    roaring_bitmap_t* cpu_result = roaring_bitmap_or(temp, c);
    verify_match(cpu_result, gpu_result);

    roaring_bitmap_free(temp);
    roaring_bitmap_free(cpu_result);
    cu_roaring::gpu_roaring_free(gpu_result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
    cu_roaring::gpu_roaring_free(gpu_c);
    roaring_bitmap_free(a);
    roaring_bitmap_free(b);
    roaring_bitmap_free(c);
}
