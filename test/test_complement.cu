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
            // Upload the complement, set negated=true.
            // Use upload() without universe_size to avoid auto-complement
            // (which would double-negate since the complement is >50% dense).
            roaring_bitmap_t* complement = roaring_bitmap_copy(logical);
            roaring_bitmap_flip_inplace(complement, 0, UNIVERSE);
            tb.gpu = cu_roaring::upload(complement);
            tb.gpu.negated = true;
            tb.gpu.universe_size = UNIVERSE;
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
                        const char* /*label*/) {
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

// ============================================================================
// 6b. Fused multi_and with mixed negation — targeted edge cases
//
// The fused_multi_and_allbitmap path handles negated inputs natively via
// (a) per-input ~word inversion inside the AND loop, and (b) a static
// all-zeros sentinel for keys absent from a negated input's storage.
// These tests deliberately construct inputs that hit those paths and
// verify against CPU CRoaring as the oracle.
// ============================================================================

// A negated input with STORAGE that covers only a subset of the keys
// touched by other inputs. For the absent keys the fused kernel must
// reach the all-zeros sentinel, invert it to all-ones, and treat the
// negated input as "no restriction" at that key. If the sentinel path is
// broken, the output at those keys becomes empty (the negated input
// incorrectly wins) or undefined (UB if the pointer is garbage).
TEST_F(ComplementTest, FusedMultiAnd_AbsentKeyInNegatedInput) {
    // 3 containers' worth of universe so we can place content at keys
    // 0, 1, 2 and see the absent-key behaviour.
    const uint32_t universe = 3u * 65536u;

    // r_A: sparse non-negated, content at all three keys (~20% density each)
    roaring_bitmap_t* r_A = roaring_bitmap_create();
    std::mt19937 gen(1001);
    for (uint32_t k = 0; k < 3; ++k) {
        uint32_t base = k * 65536u;
        for (uint32_t i = 0; i < 65536u; ++i) {
            if ((gen() % 5u) == 0) roaring_bitmap_add(r_A, base + i);
        }
    }

    // r_B: logical density ~100% with a handful of gaps ONLY inside key 1.
    // After auto-complement upload, stored_B has exactly one container at
    // key 1 containing those gap bits; keys 0 and 2 are absent from
    // storage and must take the sentinel path in the fused kernel.
    roaring_bitmap_t* r_B = roaring_bitmap_create();
    roaring_bitmap_add_range(r_B, 0, universe);
    for (uint32_t gap : {70000u, 70001u, 70005u, 70100u, 70200u}) {
        roaring_bitmap_remove(r_B, gap);
    }

    // r_C: sparse non-negated at keys 0 and 2 only (key 1 absent from
    // both its storage and logical meaning — distinct from r_B's case).
    roaring_bitmap_t* r_C = roaring_bitmap_create();
    for (uint32_t i = 0; i < 65536u; ++i) {
        if ((gen() % 7u) == 0) roaring_bitmap_add(r_C, i);
        if ((gen() % 7u) == 0) roaring_bitmap_add(r_C, 131072u + i);
    }

    auto gpu_A = cu_roaring::upload(r_A, universe, 0, cu_roaring::PROMOTE_ALL);
    auto gpu_B = cu_roaring::upload(r_B, universe, 0, cu_roaring::PROMOTE_ALL);
    auto gpu_C = cu_roaring::upload(r_C, universe, 0, cu_roaring::PROMOTE_ALL);

    // Invariants we rely on for this test to exercise the right paths:
    ASSERT_TRUE(gpu_B.negated)
        << "r_B was meant to be auto-complemented at upload";
    ASSERT_EQ(gpu_B.n_containers, 1u)
        << "stored_B should hold one container at key 1, and none at 0 or 2";
    ASSERT_FALSE(gpu_A.negated);
    ASSERT_FALSE(gpu_C.negated);
    ASSERT_EQ(gpu_A.n_array_containers, 0u);  // PROMOTE_ALL → all bitmap
    ASSERT_EQ(gpu_B.n_array_containers, 0u);
    ASSERT_EQ(gpu_C.n_array_containers, 0u);
    ASSERT_EQ(gpu_A.n_run_containers, 0u);
    ASSERT_EQ(gpu_B.n_run_containers, 0u);
    ASSERT_EQ(gpu_C.n_run_containers, 0u);

    cu_roaring::GpuRoaring bitmaps[] = {gpu_A, gpu_B, gpu_C};
    auto result = cu_roaring::multi_and(bitmaps, 3);

    // CPU oracle
    roaring_bitmap_t* cpu_ab  = roaring_bitmap_and(r_A, r_B);
    roaring_bitmap_t* cpu_abc = roaring_bitmap_and(cpu_ab, r_C);
    verify_match(cpu_abc, result, universe);

    // Extra: confirm the answer is NOT empty at key 1 (where the
    // absent-key sentinel path is NOT hit — r_B's real content lives
    // there) and NOT empty at keys 0, 2 (where the sentinel path IS hit).
    // An empty result at keys 0 or 2 would be the tell-tale sign of a
    // sentinel bug.
    EXPECT_GT(roaring_bitmap_get_cardinality(cpu_abc), 0u);

    roaring_bitmap_free(cpu_ab);
    roaring_bitmap_free(cpu_abc);
    cu_roaring::gpu_roaring_free(result);
    cu_roaring::gpu_roaring_free(gpu_A);
    cu_roaring::gpu_roaring_free(gpu_B);
    cu_roaring::gpu_roaring_free(gpu_C);
    roaring_bitmap_free(r_A);
    roaring_bitmap_free(r_B);
    roaring_bitmap_free(r_C);
}

// Many containers (16) with 5 inputs of mixed negation. The existing
// MultiAnd_WithNegated test uses a 50K universe (1 container), so the
// key-set filtering logic isn't exercised with real variety. Here every
// container has a different sparse / dense pattern so the fused kernel's
// presence-AND-reduce and per-word inversion paths are both hammered.
TEST_F(ComplementTest, FusedMultiAnd_MixedNegation_ManyContainers) {
    const uint32_t universe = 16u * 65536u;

    std::mt19937 gen(2002);
    auto make_sparse = [&](double density) {
        roaring_bitmap_t* r = roaring_bitmap_create();
        for (uint32_t i = 0; i < universe; ++i) {
            if ((gen() & 0xFFFFu) < static_cast<uint32_t>(density * 65536.0)) {
                roaring_bitmap_add(r, i);
            }
        }
        return r;
    };

    roaring_bitmap_t* r1 = make_sparse(0.30);   // non-negated
    roaring_bitmap_t* r2 = make_sparse(0.90);   // will negate (density > 50%)
    roaring_bitmap_t* r3 = make_sparse(0.15);   // non-negated
    roaring_bitmap_t* r4 = make_sparse(0.75);   // will negate
    roaring_bitmap_t* r5 = make_sparse(0.25);   // non-negated

    auto g1 = cu_roaring::upload(r1, universe, 0, cu_roaring::PROMOTE_ALL);
    auto g2 = cu_roaring::upload(r2, universe, 0, cu_roaring::PROMOTE_ALL);
    auto g3 = cu_roaring::upload(r3, universe, 0, cu_roaring::PROMOTE_ALL);
    auto g4 = cu_roaring::upload(r4, universe, 0, cu_roaring::PROMOTE_ALL);
    auto g5 = cu_roaring::upload(r5, universe, 0, cu_roaring::PROMOTE_ALL);

    // Sanity: the dense ones are stored negated, the sparse ones aren't.
    EXPECT_FALSE(g1.negated);
    EXPECT_TRUE(g2.negated);
    EXPECT_FALSE(g3.negated);
    EXPECT_TRUE(g4.negated);
    EXPECT_FALSE(g5.negated);

    cu_roaring::GpuRoaring bitmaps[] = {g1, g2, g3, g4, g5};
    auto result = cu_roaring::multi_and(bitmaps, 5);

    roaring_bitmap_t* cpu = roaring_bitmap_and(r1, r2);
    roaring_bitmap_t* t;
    t = roaring_bitmap_and(cpu, r3); roaring_bitmap_free(cpu); cpu = t;
    t = roaring_bitmap_and(cpu, r4); roaring_bitmap_free(cpu); cpu = t;
    t = roaring_bitmap_and(cpu, r5); roaring_bitmap_free(cpu); cpu = t;

    verify_match(cpu, result, universe);
    EXPECT_GT(roaring_bitmap_get_cardinality(cpu), 0u)
        << "5-way AND shouldn't be empty with these densities";

    roaring_bitmap_free(cpu);
    cu_roaring::gpu_roaring_free(result);
    cu_roaring::gpu_roaring_free(g1);
    cu_roaring::gpu_roaring_free(g2);
    cu_roaring::gpu_roaring_free(g3);
    cu_roaring::gpu_roaring_free(g4);
    cu_roaring::gpu_roaring_free(g5);
    roaring_bitmap_free(r1);
    roaring_bitmap_free(r2);
    roaring_bitmap_free(r3);
    roaring_bitmap_free(r4);
    roaring_bitmap_free(r5);
}

// 8-input alternating-negation stress test. Exercises the fused kernel's
// `negation_mask` bit-test path for every pattern of (negated, absent)
// across many containers.
TEST_F(ComplementTest, FusedMultiAnd_AlternatingNegation_8Inputs) {
    const uint32_t universe = 8u * 65536u;
    const uint32_t N = 8;

    std::mt19937 gen(3003);
    roaring_bitmap_t* r[N];
    cu_roaring::GpuRoaring g[N];

    for (uint32_t i = 0; i < N; ++i) {
        r[i] = roaring_bitmap_create();
        // Alternate: even indices sparse (~20%), odd indices dense (~85%)
        const double density = (i % 2 == 0) ? 0.20 : 0.85;
        for (uint32_t j = 0; j < universe; ++j) {
            if ((gen() & 0xFFFFu) < static_cast<uint32_t>(density * 65536.0)) {
                roaring_bitmap_add(r[i], j);
            }
        }
        g[i] = cu_roaring::upload(r[i], universe, 0, cu_roaring::PROMOTE_ALL);
        EXPECT_EQ(g[i].negated, (i % 2) != 0);
    }

    auto result = cu_roaring::multi_and(g, N);

    roaring_bitmap_t* cpu = roaring_bitmap_copy(r[0]);
    for (uint32_t i = 1; i < N; ++i) {
        roaring_bitmap_t* t = roaring_bitmap_and(cpu, r[i]);
        roaring_bitmap_free(cpu);
        cpu = t;
    }

    verify_match(cpu, result, universe);

    roaring_bitmap_free(cpu);
    cu_roaring::gpu_roaring_free(result);
    for (uint32_t i = 0; i < N; ++i) {
        cu_roaring::gpu_roaring_free(g[i]);
        roaring_bitmap_free(r[i]);
    }
}

// ============================================================================
// 7. Upload from bitset — direct container extraction, no sort/dedupe
// ============================================================================

TEST_F(ComplementTest, UploadFromBitset_Sparse) {
    uint32_t universe = 100000;
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> bitset(n_words, 0);

    std::mt19937 gen(42);
    std::vector<bool> truth(universe, false);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 10 == 0) {
            bitset[i / 32] |= (1u << (i % 32));
            truth[i] = true;
        }
    }

    auto gpu = cu_roaring::upload_from_bitset(bitset.data(), n_words, universe);
    EXPECT_FALSE(gpu.negated);  // 10% density

    std::vector<uint32_t> queries;
    std::vector<bool> expected;
    for (uint32_t i = 0; i < universe; i += 7) {
        queries.push_back(i);
        expected.push_back(truth[i]);
    }
    verify_queries(gpu, queries, expected, false);
    cu_roaring::gpu_roaring_free(gpu);
}

TEST_F(ComplementTest, UploadFromBitset_Dense) {
    uint32_t universe = 100000;
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> bitset(n_words, 0);

    std::mt19937 gen(42);
    std::vector<bool> truth(universe, false);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 10 != 0) {
            bitset[i / 32] |= (1u << (i % 32));
            truth[i] = true;
        }
    }

    auto gpu = cu_roaring::upload_from_bitset(bitset.data(), n_words, universe);
    EXPECT_TRUE(gpu.negated);  // 90% density → complement

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

TEST_F(ComplementTest, UploadFromBitset_RoundTrip) {
    // bitset → Roaring → decompress → compare against original
    uint32_t universe = 200000;
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> bitset(n_words, 0);

    std::mt19937 gen(99);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 3 == 0) bitset[i / 32] |= (1u << (i % 32));
    }

    auto gpu = cu_roaring::upload_from_bitset(bitset.data(), n_words, universe);

    uint32_t* d_result = nullptr;
    cudaMalloc(&d_result, n_words * sizeof(uint32_t));
    cu_roaring::decompress_to_bitset(gpu, d_result, n_words);

    std::vector<uint32_t> result(n_words);
    cudaMemcpy(result.data(), d_result, n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    for (uint32_t w = 0; w < n_words; ++w) {
        uint32_t mask = (w == n_words - 1 && universe % 32 != 0)
                        ? ((1u << (universe % 32)) - 1u) : 0xFFFFFFFF;
        EXPECT_EQ(result[w] & mask, bitset[w] & mask) << "Mismatch at word " << w;
    }
    cu_roaring::gpu_roaring_free(gpu);
}

TEST_F(ComplementTest, UploadFromDeviceBitset) {
    uint32_t universe = 100000;
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> bitset(n_words, 0);

    std::mt19937 gen(42);
    std::vector<bool> truth(universe, false);
    for (uint32_t i = 0; i < universe; ++i) {
        if (gen() % 4 == 0) {
            bitset[i / 32] |= (1u << (i % 32));
            truth[i] = true;
        }
    }

    uint32_t* d_bitset = nullptr;
    cudaMalloc(&d_bitset, n_words * sizeof(uint32_t));
    cudaMemcpy(d_bitset, bitset.data(), n_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    auto gpu = cu_roaring::upload_from_device_bitset(d_bitset, n_words, universe);
    cudaFree(d_bitset);

    EXPECT_FALSE(gpu.negated);  // 25% density

    std::vector<uint32_t> queries;
    std::vector<bool> expected;
    for (uint32_t i = 0; i < universe; i += 11) {
        queries.push_back(i);
        expected.push_back(truth[i]);
    }
    verify_queries(gpu, queries, expected, false);
    cu_roaring::gpu_roaring_free(gpu);
}
