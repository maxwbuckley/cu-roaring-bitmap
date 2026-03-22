#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/device/make_view.cuh"
#include "cu_roaring/device/roaring_view.cuh"

#include <random>
#include <vector>

class PromoteTest : public ::testing::Test {
 protected:
    void TearDown() override {
        cudaDeviceReset();
    }

    // Helper: decompress a GpuRoaring to a host-side bitset
    std::vector<uint32_t> to_host_bitset(const cu_roaring::GpuRoaring& gpu) {
        uint32_t n_words = (gpu.universe_size + 31) / 32;
        uint32_t* d_bs = cu_roaring::decompress_to_bitset(gpu);
        std::vector<uint32_t> h_bs(n_words);
        cudaMemcpy(h_bs.data(), d_bs, n_words * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_bs);
        return h_bs;
    }

    // Helper: count set bits in a host bitset
    static uint64_t popcount(const std::vector<uint32_t>& bs) {
        uint64_t count = 0;
        for (auto w : bs)
            count += static_cast<uint64_t>(__builtin_popcount(w));
        return count;
    }
};

// ============================================================================
// promote_to_bitmap()
// ============================================================================

TEST_F(PromoteTest, PromoteMixedContainers) {
    roaring_bitmap_t* r = roaring_bitmap_create();

    // Array container (key=0): 100 sparse elements
    for (uint32_t i = 0; i < 100; ++i)
        roaring_bitmap_add(r, i * 10);

    // Bitmap container (key=1): 50K dense elements
    for (uint32_t i = 65536; i < 65536 + 50000; ++i)
        roaring_bitmap_add(r, i);

    // Run container (key=2): contiguous range
    roaring_bitmap_add_range(r, 131072, 141072);
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_containers, 3u);
    EXPECT_GT(gpu.n_array_containers + gpu.n_run_containers, 0u);

    // Decompress original for comparison
    auto original_bs = to_host_bitset(gpu);
    uint64_t original_card = popcount(original_bs);

    // Promote
    auto promoted = cu_roaring::promote_to_bitmap(gpu);
    EXPECT_EQ(promoted.n_containers, 3u);
    EXPECT_EQ(promoted.n_bitmap_containers, 3u);
    EXPECT_EQ(promoted.n_array_containers, 0u);
    EXPECT_EQ(promoted.n_run_containers, 0u);

    // Decompress promoted — must be identical
    auto promoted_bs = to_host_bitset(promoted);
    uint64_t promoted_card = popcount(promoted_bs);
    EXPECT_EQ(promoted_card, original_card);
    EXPECT_EQ(original_bs, promoted_bs);

    cu_roaring::gpu_roaring_free(promoted);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(PromoteTest, PromoteAlreadyAllBitmap) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    // Dense, aligned to container boundaries → all bitmap containers.
    // 3 full containers: 3 * 65536 = 196608 elements, all above 4096 threshold.
    for (uint32_t i = 0; i < 3 * 65536; ++i)
        roaring_bitmap_add(r, i);

    auto gpu = cu_roaring::upload(r);
    EXPECT_EQ(gpu.n_containers, 3u);
    EXPECT_EQ(gpu.n_array_containers, 0u);
    EXPECT_EQ(gpu.n_run_containers, 0u);
    EXPECT_EQ(gpu.n_bitmap_containers, 3u);

    auto promoted = cu_roaring::promote_to_bitmap(gpu);
    EXPECT_EQ(promoted.n_containers, gpu.n_containers);
    EXPECT_EQ(promoted.n_bitmap_containers, gpu.n_containers);

    auto orig_bs = to_host_bitset(gpu);
    auto prom_bs = to_host_bitset(promoted);
    EXPECT_EQ(orig_bs, prom_bs);

    cu_roaring::gpu_roaring_free(promoted);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(PromoteTest, PromoteEmpty) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    auto gpu = cu_roaring::upload(r);
    auto promoted = cu_roaring::promote_to_bitmap(gpu);
    EXPECT_EQ(promoted.n_containers, 0u);
    cu_roaring::gpu_roaring_free(promoted);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(PromoteTest, PromoteAllArrays) {
    // Sparse bitmap with only array containers
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 gen(42);
    for (uint32_t key = 0; key < 50; ++key) {
        uint32_t base = key * 65536u;
        std::uniform_int_distribution<uint32_t> dist(0, 65535);
        for (int j = 0; j < 100; ++j)
            roaring_bitmap_add(r, base + dist(gen));
    }

    auto gpu = cu_roaring::upload(r);
    EXPECT_EQ(gpu.n_bitmap_containers, 0u);
    EXPECT_GT(gpu.n_array_containers, 0u);

    auto original_bs = to_host_bitset(gpu);

    auto promoted = cu_roaring::promote_to_bitmap(gpu);
    EXPECT_EQ(promoted.n_bitmap_containers, promoted.n_containers);
    EXPECT_EQ(promoted.n_array_containers, 0u);

    auto promoted_bs = to_host_bitset(promoted);
    EXPECT_EQ(original_bs, promoted_bs);

    cu_roaring::gpu_roaring_free(promoted);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

// ============================================================================
// upload() with bitmap_threshold
// ============================================================================

TEST_F(PromoteTest, UploadWithPromoteAll) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    // Mix of array and bitmap containers
    for (uint32_t i = 0; i < 100; ++i)
        roaring_bitmap_add(r, i * 10);        // array in key=0
    for (uint32_t i = 65536; i < 115536; ++i)
        roaring_bitmap_add(r, i);              // bitmap in key=1

    // Default upload: mixed
    auto mixed = cu_roaring::upload(r);
    EXPECT_GT(mixed.n_array_containers, 0u);

    // Upload with PROMOTE_ALL: everything should be bitmap
    auto all_bmp = cu_roaring::upload(r, 0, cu_roaring::PROMOTE_ALL);
    EXPECT_EQ(all_bmp.n_bitmap_containers, all_bmp.n_containers);
    EXPECT_EQ(all_bmp.n_array_containers, 0u);
    EXPECT_EQ(all_bmp.n_run_containers, 0u);

    // Both must produce same decompressed bitset
    auto bs1 = to_host_bitset(mixed);
    auto bs2 = to_host_bitset(all_bmp);
    EXPECT_EQ(bs1, bs2);

    cu_roaring::gpu_roaring_free(mixed);
    cu_roaring::gpu_roaring_free(all_bmp);
    roaring_bitmap_free(r);
}

// ============================================================================
// upload_from_sorted_ids() with bitmap_threshold
// ============================================================================

TEST_F(PromoteTest, UploadFromIdsWithThreshold) {
    // 500 IDs spread across 5 containers — default threshold makes them arrays
    std::vector<uint32_t> ids;
    for (uint32_t key = 0; key < 5; ++key) {
        uint32_t base = key * 65536u;
        for (uint32_t j = 0; j < 100; ++j)
            ids.push_back(base + j * 100);
    }
    std::sort(ids.begin(), ids.end());

    uint32_t universe = 5 * 65536;

    // Explicit PROMOTE_NONE: array containers
    auto arr = cu_roaring::upload_from_sorted_ids(
        ids.data(), static_cast<uint32_t>(ids.size()), universe, 0,
        cu_roaring::PROMOTE_NONE);
    EXPECT_EQ(arr.n_containers, 5u);
    EXPECT_EQ(arr.n_array_containers, 5u);
    EXPECT_EQ(arr.n_bitmap_containers, 0u);

    // Threshold=0: all bitmap
    auto bmp = cu_roaring::upload_from_sorted_ids(
        ids.data(), static_cast<uint32_t>(ids.size()), universe, 0,
        cu_roaring::PROMOTE_ALL);
    EXPECT_EQ(bmp.n_containers, 5u);
    EXPECT_EQ(bmp.n_bitmap_containers, 5u);
    EXPECT_EQ(bmp.n_array_containers, 0u);

    // Both must decompress to the same bitset
    auto bs1 = to_host_bitset(arr);
    auto bs2 = to_host_bitset(bmp);
    EXPECT_EQ(bs1, bs2);

    cu_roaring::gpu_roaring_free(arr);
    cu_roaring::gpu_roaring_free(bmp);
}

TEST_F(PromoteTest, UploadFromIdsCustomThreshold) {
    // 200 IDs per container — with threshold=100, they should become bitmap
    std::vector<uint32_t> ids;
    for (uint32_t key = 0; key < 3; ++key) {
        uint32_t base = key * 65536u;
        for (uint32_t j = 0; j < 200; ++j)
            ids.push_back(base + j);
    }

    uint32_t universe = 3 * 65536;

    // Threshold 100: card=200 > 100, so all become bitmap
    auto bmp = cu_roaring::upload_from_sorted_ids(
        ids.data(), static_cast<uint32_t>(ids.size()), universe, 0, 100);
    EXPECT_EQ(bmp.n_bitmap_containers, 3u);
    EXPECT_EQ(bmp.n_array_containers, 0u);

    // Threshold 4096 (PROMOTE_NONE): card=200 <= 4096, so all stay array
    auto arr = cu_roaring::upload_from_sorted_ids(
        ids.data(), static_cast<uint32_t>(ids.size()), universe, 0,
        cu_roaring::PROMOTE_NONE);
    EXPECT_EQ(arr.n_array_containers, 3u);
    EXPECT_EQ(arr.n_bitmap_containers, 0u);

    // Same bitset
    auto bs1 = to_host_bitset(bmp);
    auto bs2 = to_host_bitset(arr);
    EXPECT_EQ(bs1, bs2);

    cu_roaring::gpu_roaring_free(bmp);
    cu_roaring::gpu_roaring_free(arr);
}

// ============================================================================
// PROMOTE_AUTO (cache-aware policy)
// ============================================================================

TEST_F(PromoteTest, ResolveAutoThresholdSmallUniverse) {
    // Small universe: flat bitset easily fits in L2 → PROMOTE_NONE
    uint32_t threshold = cu_roaring::resolve_auto_threshold(1000000);  // 125 KB
    EXPECT_EQ(threshold, cu_roaring::PROMOTE_NONE);
}

TEST_F(PromoteTest, ResolveAutoThresholdLargeUniverse) {
    // 1B universe: flat bitset = 125 MB, exceeds any GPU's L2 → PROMOTE_ALL
    uint32_t threshold = cu_roaring::resolve_auto_threshold(1000000000);
    EXPECT_EQ(threshold, cu_roaring::PROMOTE_ALL);
}

TEST_F(PromoteTest, PromoteAutoSmallUniverse) {
    // Small universe: auto should keep arrays (bitset fits in L2)
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < 500; ++i)
        ids.push_back(i * 100);

    auto bm = cu_roaring::upload_from_sorted_ids(
        ids.data(), static_cast<uint32_t>(ids.size()), 100000);
    // At 100K universe (12.5 KB bitset), auto keeps PROMOTE_NONE
    // Containers have 500 elements each → array containers
    EXPECT_GT(bm.n_array_containers, 0u);

    auto promoted = cu_roaring::promote_auto(bm);
    // Auto should NOT promote (small universe fits in L2)
    // But promote_auto always returns a valid copy, verify correctness
    auto bs1 = to_host_bitset(bm);
    auto bs2 = to_host_bitset(promoted);
    EXPECT_EQ(bs1, bs2);

    cu_roaring::gpu_roaring_free(promoted);
    cu_roaring::gpu_roaring_free(bm);
}

TEST_F(PromoteTest, PromoteAutoLargeUniverse) {
    // Large universe: auto should promote (bitset exceeds L2)
    // We can't create a real 1B bitmap in a test, but we can set
    // universe_size large and verify resolve_auto_threshold behavior
    uint32_t threshold = cu_roaring::resolve_auto_threshold(1000000000);
    EXPECT_EQ(threshold, cu_roaring::PROMOTE_ALL);

    // Small dataset but large universe_size → auto promotes
    std::vector<uint32_t> ids = {0, 65536, 131072};  // 3 containers, 1 element each
    auto bm = cu_roaring::upload_from_sorted_ids(
        ids.data(), 3, 1000000000);  // universe=1B
    // Auto default: should have promoted to bitmap
    EXPECT_EQ(bm.n_bitmap_containers, 3u);
    EXPECT_EQ(bm.n_array_containers, 0u);

    cu_roaring::gpu_roaring_free(bm);
}

TEST_F(PromoteTest, UploadDefaultIsAuto) {
    // Verify that the default parameter is PROMOTE_AUTO by checking
    // that small universes produce arrays and large universes produce bitmaps.
    std::vector<uint32_t> ids = {0, 100, 200};

    // Small universe: default auto → PROMOTE_NONE → arrays
    auto small = cu_roaring::upload_from_sorted_ids(ids.data(), 3, 10000);
    EXPECT_EQ(small.n_array_containers, 1u);
    EXPECT_EQ(small.n_bitmap_containers, 0u);

    // Large universe: default auto → PROMOTE_ALL → bitmaps
    auto large = cu_roaring::upload_from_sorted_ids(ids.data(), 3, 1000000000);
    EXPECT_EQ(large.n_bitmap_containers, 1u);
    EXPECT_EQ(large.n_array_containers, 0u);

    // Both must decompress to the same content
    // (different universe_size but same IDs present)
    auto bs_s = to_host_bitset(small);
    auto bs_l = to_host_bitset(large);
    // Check the first few words match (IDs 0, 100, 200 are all in word range 0-6)
    for (int w = 0; w < 7 && w < static_cast<int>(bs_s.size()) && w < static_cast<int>(bs_l.size()); ++w) {
        EXPECT_EQ(bs_s[w], bs_l[w]) << "Mismatch at word " << w;
    }

    cu_roaring::gpu_roaring_free(small);
    cu_roaring::gpu_roaring_free(large);
}
