/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Correctness tests for enumerate_runs(): run extraction and cross-container
 * coalescing. Each test builds a RUN-only bitmap (asserted) so the coalesced
 * ranges equal the contiguous regions that were inserted.
 */

#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/detail/enumerate_runs.cuh"

#include <cstdint>
#include <vector>

namespace {

class EnumerateRunsTest : public ::testing::Test {
 protected:
    void TearDown() override { cudaDeviceReset(); }

    // Copy a RunRanges result to host. enumerate_runs() returns ranges already
    // sorted by start, so no host-side sort is applied (the tests verify order).
    static std::vector<cu_roaring::IdRange> to_host(const cu_roaring::RunRanges& rr) {
        std::vector<cu_roaring::IdRange> h(rr.count);
        if (rr.count > 0) {
            cudaMemcpy(h.data(), rr.ranges,
                       rr.count * sizeof(cu_roaring::IdRange),
                       cudaMemcpyDeviceToHost);
        }
        return h;
    }
};

TEST_F(EnumerateRunsTest, EmptyBitmap) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    auto gpu = cu_roaring::upload(r);

    auto rr = cu_roaring::enumerate_runs(gpu);
    EXPECT_EQ(rr.count, 0u);
    EXPECT_EQ(rr.ranges, nullptr);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, NoRunContainers) {
    // Scattered sparse bits stay ARRAY containers even after run_optimize.
    roaring_bitmap_t* r = roaring_bitmap_create();
    for (uint32_t i = 0; i < 100; ++i) {
        roaring_bitmap_add(r, i * 1000);
    }
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, 0u);  // input has no RUN containers

    auto rr = cu_roaring::enumerate_runs(gpu);
    EXPECT_EQ(rr.count, 0u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, SingleRun) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 100, 500);  // [100, 500)
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, gpu.n_containers);  // input is all-RUN

    auto rr = cu_roaring::enumerate_runs(gpu);
    auto h  = to_host(rr);
    ASSERT_EQ(rr.count, 1u);
    EXPECT_EQ(h[0].start, 100u);
    EXPECT_EQ(h[0].end, 500u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, MultipleRunsOneContainer) {
    // Three disjoint runs within container 0, separated by gaps -> 3 ranges.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 100, 200);
    roaring_bitmap_add_range(r, 1000, 1500);
    roaring_bitmap_add_range(r, 5000, 5001);  // single element
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, gpu.n_containers);

    auto rr = cu_roaring::enumerate_runs(gpu);
    auto h  = to_host(rr);
    ASSERT_EQ(rr.count, 3u);
    EXPECT_EQ(h[0].start, 100u);   EXPECT_EQ(h[0].end, 200u);
    EXPECT_EQ(h[1].start, 1000u);  EXPECT_EQ(h[1].end, 1500u);
    EXPECT_EQ(h[2].start, 5000u);  EXPECT_EQ(h[2].end, 5001u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, CoalesceAcrossContainers) {
    // A contiguous range spanning several 64K container boundaries must
    // coalesce into ONE range despite being stored as multiple RUN containers.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 0, 200000);  // spans containers 0,1,2,3
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, gpu.n_containers);
    ASSERT_GE(gpu.n_containers, 3u);  // genuinely multi-container

    auto rr = cu_roaring::enumerate_runs(gpu);
    auto h  = to_host(rr);
    ASSERT_EQ(rr.count, 1u) << "runs touching across container boundaries must coalesce";
    EXPECT_EQ(h[0].start, 0u);
    EXPECT_EQ(h[0].end, 200000u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, AdjacentContainersExactBoundary) {
    // A range crossing exactly one container boundary (60000..70000 crosses
    // 65536): the last run of container 0 ends at the boundary and the first
    // run of container 1 starts at it -> one coalesced range.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 60000, 70000);
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, gpu.n_containers);
    ASSERT_GE(gpu.n_containers, 2u);

    auto rr = cu_roaring::enumerate_runs(gpu);
    auto h  = to_host(rr);
    ASSERT_EQ(rr.count, 1u);
    EXPECT_EQ(h[0].start, 60000u);
    EXPECT_EQ(h[0].end, 70000u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, NoCoalesceAcrossGap) {
    // Two runs in non-adjacent containers (an empty container 1 between them)
    // must NOT coalesce.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 0, 65536);               // container 0, full
    roaring_bitmap_add_range(r, 131072, 131072 + 1000);  // container 2
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, gpu.n_containers);

    auto rr = cu_roaring::enumerate_runs(gpu);
    auto h  = to_host(rr);
    ASSERT_EQ(rr.count, 2u);
    EXPECT_EQ(h[0].start, 0u);       EXPECT_EQ(h[0].end, 65536u);
    EXPECT_EQ(h[1].start, 131072u);  EXPECT_EQ(h[1].end, 132072u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

TEST_F(EnumerateRunsTest, MixedRunsAndGapsAcrossContainers) {
    // One range spanning two containers, then a gap, then another range:
    // [10000, 70000) coalesces (crosses 65536), [200000, 210000) is separate.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 10000, 70000);
    roaring_bitmap_add_range(r, 200000, 210000);
    roaring_bitmap_run_optimize(r);

    auto gpu = cu_roaring::upload(r);
    ASSERT_EQ(gpu.n_run_containers, gpu.n_containers);

    auto rr = cu_roaring::enumerate_runs(gpu);
    auto h  = to_host(rr);
    ASSERT_EQ(rr.count, 2u);
    EXPECT_EQ(h[0].start, 10000u);   EXPECT_EQ(h[0].end, 70000u);
    EXPECT_EQ(h[1].start, 200000u);  EXPECT_EQ(h[1].end, 210000u);

    cu_roaring::free_run_ranges(rr);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

}  // namespace
