#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <roaring/roaring.h>

#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/device/roaring_view.cuh"
#include "cu_roaring/device/roaring_warp_query.cuh"
#include "cu_roaring/device/make_view.cuh"

#include <algorithm>
#include <random>
#include <vector>

namespace cu_roaring {
void gpu_roaring_free(GpuRoaring& bitmap);
}

// ============================================================================
// Kernels for testing contains() and warp_contains() on negated bitmaps
// ============================================================================

__global__ void test_contains_kernel(cu_roaring::GpuRoaringView view,
                                     const uint32_t* query_ids,
                                     bool* results,
                                     uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_queries) { results[idx] = view.contains(query_ids[idx]); }
}

__global__ void test_warp_contains_kernel(cu_roaring::GpuRoaringView view,
                                          const uint32_t* query_ids,
                                          bool* results,
                                          uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_queries) { results[idx] = cu_roaring::warp_contains(view, query_ids[idx]); }
}

// ============================================================================
// Test fixture
// ============================================================================

class ComplementTest : public ::testing::Test {
 protected:
    void TearDown() override {
        cudaDeviceReset();
    }

    // Run GPU contains() or warp_contains() and compare against expected bools
    void verify_queries(cu_roaring::GpuRoaring& gpu_bm,
                        const std::vector<uint32_t>& queries,
                        const std::vector<bool>& expected,
                        bool use_warp)
    {
        uint32_t n = static_cast<uint32_t>(queries.size());
        ASSERT_EQ(n, static_cast<uint32_t>(expected.size()));

        uint32_t* d_queries;
        bool* d_results;
        cudaMalloc(&d_queries, n * sizeof(uint32_t));
        cudaMalloc(&d_results, n * sizeof(bool));
        cudaMemcpy(d_queries, queries.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);

        auto view = cu_roaring::make_view(gpu_bm);
        uint32_t blocks = (n + 255) / 256;

        if (use_warp) {
            test_warp_contains_kernel<<<blocks, 256>>>(view, d_queries, d_results, n);
        } else {
            test_contains_kernel<<<blocks, 256>>>(view, d_queries, d_results, n);
        }
        cudaDeviceSynchronize();

        std::vector<char> h_results_raw(n);
        cudaMemcpy(h_results_raw.data(), d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i < n; ++i) {
            bool gpu_result = h_results_raw[i];
            EXPECT_EQ(gpu_result, expected[i])
                << "Mismatch at query ID " << queries[i]
                << " (GPU=" << gpu_result << " expected=" << expected[i] << ")"
                << (use_warp ? " [warp]" : " [scalar]");
        }

        cudaFree(d_queries);
        cudaFree(d_results);
    }

    // Verify GPU result matches CPU CRoaring result via decompression
    void verify_match(const roaring_bitmap_t* cpu_result,
                      const cu_roaring::GpuRoaring& gpu_result,
                      uint32_t universe_size) {
        uint32_t n_words = (universe_size + 31) / 32;

        // CPU bitset
        std::vector<uint32_t> cpu_bitset(n_words, 0);
        roaring_uint32_iterator_t* iter = roaring_iterator_create(cpu_result);
        while (iter->has_value) {
            uint32_t val = iter->current_value;
            if (val < universe_size) {
                cpu_bitset[val / 32] |= (1u << (val % 32));
            }
            roaring_uint32_iterator_advance(iter);
        }
        roaring_uint32_iterator_free(iter);

        // GPU bitset (decompress handles negation)
        uint32_t* d_bitset = nullptr;
        cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));
        cu_roaring::decompress_to_bitset(gpu_result, d_bitset, n_words);
        std::vector<uint32_t> gpu_bitset(n_words, 0);
        cudaMemcpy(gpu_bitset.data(), d_bitset, n_words * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_bitset);

        for (uint32_t w = 0; w < n_words; ++w) {
            EXPECT_EQ(gpu_bitset[w], cpu_bitset[w])
                << "Bitset mismatch at word " << w
                << " (bits " << (w * 32) << "-" << (w * 32 + 31) << ")";
        }
    }
};

// ============================================================================
// 1. Negated contains() — per-thread
// ============================================================================

TEST_F(ComplementTest, NegatedContains_Small) {
    // Universe = 10, IDs = {0,1,2,3,4,5,6,8,9} (90% density)
    // Should store complement {7} with negated=true
    uint32_t universe = 10;
    std::vector<uint32_t> ids = {0, 1, 2, 3, 4, 5, 6, 8, 9};
    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    EXPECT_TRUE(gpu.negated);

    // Query all IDs in universe
    std::vector<uint32_t> queries;
    std::vector<bool> expected;
    for (uint32_t i = 0; i < universe; ++i) {
        queries.push_back(i);
        expected.push_back(std::find(ids.begin(), ids.end(), i) != ids.end());
    }

    verify_queries(gpu, queries, expected, false);
    verify_queries(gpu, queries, expected, true);
    cu_roaring::gpu_roaring_free(gpu);
}

TEST_F(ComplementTest, NegatedContains_AbsentKey) {
    // Universe spans multiple 65536 key ranges. The negated bitmap stores
    // the complement, which has FEWER containers. Absent keys in the stored
    // complement should return TRUE (the entire range is in the logical set).
    uint32_t universe = 200000;  // keys 0, 1, 2
    // Set almost everything: 90% density
    std::vector<uint32_t> ids;
    std::mt19937 gen(42);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 10 != 0) ids.push_back(i);  // ~90% density
    }

    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    EXPECT_TRUE(gpu.negated);

    // Build truth set for fast lookup
    std::vector<bool> truth(universe, false);
    for (uint32_t id : ids) truth[id] = true;

    // Query a sample across the full range
    std::vector<uint32_t> queries;
    std::vector<bool> expected;
    for (uint32_t i = 0; i < universe; i += 7) {
        queries.push_back(i);
        expected.push_back(truth[i]);
    }

    verify_queries(gpu, queries, expected, false);
    verify_queries(gpu, queries, expected, true);
    cu_roaring::gpu_roaring_free(gpu);
}

TEST_F(ComplementTest, NegatedContains_LargeGPUPath) {
    // Force GPU path (> 1024 IDs), 80% density
    uint32_t universe = 100000;
    std::vector<uint32_t> ids;
    std::mt19937 gen(123);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 5 != 0) ids.push_back(i);  // ~80%
    }

    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    EXPECT_TRUE(gpu.negated);

    std::vector<bool> truth(universe, false);
    for (uint32_t id : ids) truth[id] = true;

    std::vector<uint32_t> queries;
    std::vector<bool> expected;
    for (uint32_t i = 0; i < universe; i += 3) {
        queries.push_back(i);
        expected.push_back(truth[i]);
    }

    verify_queries(gpu, queries, expected, false);
    verify_queries(gpu, queries, expected, true);
    cu_roaring::gpu_roaring_free(gpu);
}

TEST_F(ComplementTest, NoNegate_LowDensity) {
    // 10% density — should NOT negate
    uint32_t universe = 100000;
    std::vector<uint32_t> ids;
    std::mt19937 gen(42);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 10 == 0) ids.push_back(i);
    }

    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    EXPECT_FALSE(gpu.negated);
    cu_roaring::gpu_roaring_free(gpu);
}

TEST_F(ComplementTest, NoNegate_ExactlyHalf) {
    // Exactly 50% — should NOT negate (only > 50% triggers)
    uint32_t universe = 100;
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < 50; ++i) ids.push_back(i);

    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    EXPECT_FALSE(gpu.negated);
    cu_roaring::gpu_roaring_free(gpu);
}

// ============================================================================
// 2. CRoaring upload with complement
// ============================================================================

TEST_F(ComplementTest, CRoaringUpload_Complement) {
    uint32_t universe = 100000;
    roaring_bitmap_t* r = roaring_bitmap_create();
    // 90% density
    for (uint32_t i = 0; i < universe; ++i) {
        if (i % 10 != 0) roaring_bitmap_add(r, i);
    }

    auto gpu = cu_roaring::upload(r, universe);
    EXPECT_TRUE(gpu.negated);

    // Verify via decompression
    verify_match(r, gpu, universe);

    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(ComplementTest, CRoaringUpload_NoComplement) {
    uint32_t universe = 100000;
    roaring_bitmap_t* r = roaring_bitmap_create();
    // 10% density
    for (uint32_t i = 0; i < universe; i += 10) {
        roaring_bitmap_add(r, i);
    }

    auto gpu = cu_roaring::upload(r, universe);
    EXPECT_FALSE(gpu.negated);

    verify_match(r, gpu, universe);

    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

// ============================================================================
// 3. Decompress negated bitmap
// ============================================================================

TEST_F(ComplementTest, Decompress_Negated) {
    uint32_t universe = 50000;
    std::vector<uint32_t> ids;
    // 80% density
    for (uint32_t i = 0; i < universe; ++i) {
        if (i % 5 != 0) ids.push_back(i);
    }

    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    ASSERT_TRUE(gpu.negated);

    // Decompress and verify against truth
    uint32_t n_words = (universe + 31) / 32;
    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));
    cu_roaring::decompress_to_bitset(gpu, d_bitset, n_words);

    std::vector<uint32_t> h_bitset(n_words, 0);
    cudaMemcpy(h_bitset.data(), d_bitset, n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_bitset);

    std::vector<bool> truth(universe, false);
    for (uint32_t id : ids) truth[id] = true;

    for (uint32_t i = 0; i < universe; ++i) {
        bool in_bitset = (h_bitset[i / 32] >> (i % 32)) & 1;
        EXPECT_EQ(in_bitset, truth[i]) << "Decompress mismatch at ID " << i;
    }

    cu_roaring::gpu_roaring_free(gpu);
}

// ============================================================================
// 4. DeMorgan set operations — all 16 combinations
//    Tests your exact example: NOT({7}) AND {4,6,7} = {4,6}
// ============================================================================

TEST_F(ComplementTest, SetOp_UserExample) {
    // NOT({7}) AND {4,6,7} should give {4,6} in a universe of 10
    uint32_t universe = 10;

    // Build A = {7}, mark as negated → logically NOT({7}) = {0,1,2,3,4,5,6,8,9}
    std::vector<uint32_t> a_ids = {7};
    auto gpu_a = cu_roaring::upload_from_ids(a_ids.data(), 1, universe);
    // Force negation for this test (density is only 10%, won't auto-negate)
    // We need to manually create the negated version
    // NOT({7}) in universe 10 = {0,1,2,3,4,5,6,8,9}
    std::vector<uint32_t> nota_ids = {0, 1, 2, 3, 4, 5, 6, 8, 9};
    cu_roaring::gpu_roaring_free(gpu_a);
    gpu_a = cu_roaring::upload_from_ids(nota_ids.data(),
                                         static_cast<uint32_t>(nota_ids.size()),
                                         universe);
    // This is >50% density, should auto-negate
    EXPECT_TRUE(gpu_a.negated);

    // B = {4, 6, 7}
    std::vector<uint32_t> b_ids = {4, 6, 7};
    auto gpu_b = cu_roaring::upload_from_ids(b_ids.data(),
                                              static_cast<uint32_t>(b_ids.size()),
                                              universe);
    EXPECT_FALSE(gpu_b.negated);

    // AND(negated_a, b) should give {4, 6} (NOT({7}) AND {4,6,7} = {4,6})
    auto result = cu_roaring::set_operation(gpu_a, gpu_b, cu_roaring::SetOp::AND);

    // Build expected on CPU
    roaring_bitmap_t* cpu_expected = roaring_bitmap_create();
    roaring_bitmap_add(cpu_expected, 4);
    roaring_bitmap_add(cpu_expected, 6);

    verify_match(cpu_expected, result, universe);

    roaring_bitmap_free(cpu_expected);
    cu_roaring::gpu_roaring_free(result);
    cu_roaring::gpu_roaring_free(gpu_a);
    cu_roaring::gpu_roaring_free(gpu_b);
}

// Helper: build GPU bitmap from IDs, optionally forcing negation
struct TestBitmap {
    cu_roaring::GpuRoaring gpu;
    roaring_bitmap_t* cpu;  // logical meaning (after negation)
};

class DeMorganTest : public ComplementTest {
 protected:
    static constexpr uint32_t UNIVERSE = 100000;

    // Build a random bitmap with given density; if force_negate, store complement
    TestBitmap make_bitmap(double density, uint64_t seed, bool force_negate) {
        TestBitmap tb{};
        // Build the logical set
        roaring_bitmap_t* logical = roaring_bitmap_create();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (uint32_t i = 0; i < UNIVERSE; ++i) {
            if (dist(gen) < density) roaring_bitmap_add(logical, i);
        }
        tb.cpu = logical;

        if (force_negate) {
            // Upload the complement, set negated=true
            roaring_bitmap_t* complement = roaring_bitmap_copy(logical);
            roaring_bitmap_flip_inplace(complement, 0, UNIVERSE);
            tb.gpu = cu_roaring::upload(complement, UNIVERSE);
            tb.gpu.negated = true;
            tb.gpu.total_cardinality = roaring_bitmap_get_cardinality(logical);
            roaring_bitmap_free(complement);
        } else {
            tb.gpu = cu_roaring::upload(logical, UNIVERSE);
        }
        return tb;
    }

    void free_bitmap(TestBitmap& tb) {
        cu_roaring::gpu_roaring_free(tb.gpu);
        roaring_bitmap_free(tb.cpu);
    }

    // Compute CPU reference for a set operation
    roaring_bitmap_t* cpu_set_op(const roaring_bitmap_t* a,
                                  const roaring_bitmap_t* b,
                                  cu_roaring::SetOp op) {
        switch (op) {
        case cu_roaring::SetOp::AND:
            return roaring_bitmap_and(a, b);
        case cu_roaring::SetOp::OR:
            return roaring_bitmap_or(a, b);
        case cu_roaring::SetOp::ANDNOT:
            return roaring_bitmap_andnot(a, b);
        case cu_roaring::SetOp::XOR:
            return roaring_bitmap_xor(a, b);
        }
        return roaring_bitmap_create();
    }

    // Test one DeMorgan combination
    void test_demorgan(cu_roaring::SetOp op, bool negate_a, bool negate_b,
                        const char* label) {
        auto a = make_bitmap(0.3, 42, negate_a);
        auto b = make_bitmap(0.4, 99, negate_b);

        auto gpu_result = cu_roaring::set_operation(a.gpu, b.gpu, op);
        roaring_bitmap_t* cpu_result = cpu_set_op(a.cpu, b.cpu, op);

        verify_match(cpu_result, gpu_result, UNIVERSE);

        roaring_bitmap_free(cpu_result);
        cu_roaring::gpu_roaring_free(gpu_result);
        free_bitmap(a);
        free_bitmap(b);
    }
};

// AND × {FF, TF, FT, TT}
TEST_F(DeMorganTest, AND_FF) { test_demorgan(cu_roaring::SetOp::AND, false, false, "AND_FF"); }
TEST_F(DeMorganTest, AND_TF) { test_demorgan(cu_roaring::SetOp::AND, true,  false, "AND_TF"); }
TEST_F(DeMorganTest, AND_FT) { test_demorgan(cu_roaring::SetOp::AND, false, true,  "AND_FT"); }
TEST_F(DeMorganTest, AND_TT) { test_demorgan(cu_roaring::SetOp::AND, true,  true,  "AND_TT"); }

// OR × {FF, TF, FT, TT}
TEST_F(DeMorganTest, OR_FF) { test_demorgan(cu_roaring::SetOp::OR, false, false, "OR_FF"); }
TEST_F(DeMorganTest, OR_TF) { test_demorgan(cu_roaring::SetOp::OR, true,  false, "OR_TF"); }
TEST_F(DeMorganTest, OR_FT) { test_demorgan(cu_roaring::SetOp::OR, false, true,  "OR_FT"); }
TEST_F(DeMorganTest, OR_TT) { test_demorgan(cu_roaring::SetOp::OR, true,  true,  "OR_TT"); }

// ANDNOT × {FF, TF, FT, TT}
TEST_F(DeMorganTest, ANDNOT_FF) { test_demorgan(cu_roaring::SetOp::ANDNOT, false, false, "ANDNOT_FF"); }
TEST_F(DeMorganTest, ANDNOT_TF) { test_demorgan(cu_roaring::SetOp::ANDNOT, true,  false, "ANDNOT_TF"); }
TEST_F(DeMorganTest, ANDNOT_FT) { test_demorgan(cu_roaring::SetOp::ANDNOT, false, true,  "ANDNOT_FT"); }
TEST_F(DeMorganTest, ANDNOT_TT) { test_demorgan(cu_roaring::SetOp::ANDNOT, true,  true,  "ANDNOT_TT"); }

// XOR × {FF, TF, FT, TT}
TEST_F(DeMorganTest, XOR_FF) { test_demorgan(cu_roaring::SetOp::XOR, false, false, "XOR_FF"); }
TEST_F(DeMorganTest, XOR_TF) { test_demorgan(cu_roaring::SetOp::XOR, true,  false, "XOR_TF"); }
TEST_F(DeMorganTest, XOR_FT) { test_demorgan(cu_roaring::SetOp::XOR, false, true,  "XOR_FT"); }
TEST_F(DeMorganTest, XOR_TT) { test_demorgan(cu_roaring::SetOp::XOR, true,  true,  "XOR_TT"); }

// ============================================================================
// 5. Promote preserves negation
// ============================================================================

TEST_F(ComplementTest, Promote_PreservesNegated) {
    uint32_t universe = 100000;
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < universe; ++i) {
        if (i % 5 != 0) ids.push_back(i);  // 80%
    }

    auto gpu = cu_roaring::upload_from_ids(ids.data(), static_cast<uint32_t>(ids.size()),
                                           universe);
    ASSERT_TRUE(gpu.negated);

    auto promoted = cu_roaring::promote_to_bitmap(gpu);
    EXPECT_TRUE(promoted.negated);
    EXPECT_EQ(promoted.n_array_containers, 0u);

    // Verify queries still work
    std::vector<bool> truth(universe, false);
    for (uint32_t id : ids) truth[id] = true;

    std::vector<uint32_t> queries;
    std::vector<bool> expected;
    for (uint32_t i = 0; i < universe; i += 13) {
        queries.push_back(i);
        expected.push_back(truth[i]);
    }
    verify_queries(promoted, queries, expected, false);

    cu_roaring::gpu_roaring_free(promoted);
    cu_roaring::gpu_roaring_free(gpu);
}

// ============================================================================
// 6. Multi-way ops with negated inputs
// ============================================================================

TEST_F(ComplementTest, MultiAnd_WithNegated) {
    // multi_and with negated inputs uses the fused kernel (single pass)
    uint32_t universe = 50000;

    roaring_bitmap_t* r1 = roaring_bitmap_create();
    roaring_bitmap_t* r2 = roaring_bitmap_create();
    roaring_bitmap_t* r3 = roaring_bitmap_create();

    std::mt19937 gen(42);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 3 == 0) roaring_bitmap_add(r1, i);
        if (gen() % 2 == 0) roaring_bitmap_add(r2, i);
        if (gen() % 4 != 0) roaring_bitmap_add(r3, i);  // ~75% → will negate
    }

    auto gpu1 = cu_roaring::upload(r1, universe);
    auto gpu2 = cu_roaring::upload(r2, universe);
    auto gpu3 = cu_roaring::upload(r3, universe);  // likely negated

    cu_roaring::GpuRoaring bitmaps[] = {gpu1, gpu2, gpu3};
    auto result = cu_roaring::multi_and(bitmaps, 3);

    // CPU reference
    roaring_bitmap_t* cpu12 = roaring_bitmap_and(r1, r2);
    roaring_bitmap_t* cpu123 = roaring_bitmap_and(cpu12, r3);

    verify_match(cpu123, result, universe);

    roaring_bitmap_free(cpu12);
    roaring_bitmap_free(cpu123);
    cu_roaring::gpu_roaring_free(result);
    cu_roaring::gpu_roaring_free(gpu1);
    cu_roaring::gpu_roaring_free(gpu2);
    cu_roaring::gpu_roaring_free(gpu3);
    roaring_bitmap_free(r1);
    roaring_bitmap_free(r2);
    roaring_bitmap_free(r3);
}

TEST_F(ComplementTest, MultiAnd_AllNegated) {
    // AND(~A, ~B, ~C) = ~(A|B|C) — all inputs negated, fused kernel handles
    // via full-universe key set (no non-negated inputs to filter).
    // Use a universe spanning multiple 65536-key ranges to exercise absent-key logic.
    uint32_t universe = 200000;  // keys 0, 1, 2

    roaring_bitmap_t* r1 = roaring_bitmap_create();
    roaring_bitmap_t* r2 = roaring_bitmap_create();
    roaring_bitmap_t* r3 = roaring_bitmap_create();

    std::mt19937 gen(77);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 4 != 0) roaring_bitmap_add(r1, i);  // ~75%
        if (gen() % 3 != 0) roaring_bitmap_add(r2, i);  // ~67%
        if (gen() % 5 != 0) roaring_bitmap_add(r3, i);  // ~80%
    }

    auto gpu1 = cu_roaring::upload(r1, universe);
    auto gpu2 = cu_roaring::upload(r2, universe);
    auto gpu3 = cu_roaring::upload(r3, universe);

    // All should be negated (>50% density)
    EXPECT_TRUE(gpu1.negated);
    EXPECT_TRUE(gpu2.negated);
    EXPECT_TRUE(gpu3.negated);

    cu_roaring::GpuRoaring bitmaps[] = {gpu1, gpu2, gpu3};
    auto result = cu_roaring::multi_and(bitmaps, 3);

    // CPU reference: AND(r1, r2, r3)
    roaring_bitmap_t* cpu12 = roaring_bitmap_and(r1, r2);
    roaring_bitmap_t* cpu123 = roaring_bitmap_and(cpu12, r3);

    verify_match(cpu123, result, universe);

    roaring_bitmap_free(cpu12);
    roaring_bitmap_free(cpu123);
    cu_roaring::gpu_roaring_free(result);
    cu_roaring::gpu_roaring_free(gpu1);
    cu_roaring::gpu_roaring_free(gpu2);
    cu_roaring::gpu_roaring_free(gpu3);
    roaring_bitmap_free(r1);
    roaring_bitmap_free(r2);
    roaring_bitmap_free(r3);
}
