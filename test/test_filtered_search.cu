/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Correctness tests for roaring-filtered brute-force search.
 *
 * Each test builds a filter that lands in one dispatch regime (run / array /
 * dense-bitmap / sparse-bitmap / mixed / negated / fragmentation fallback),
 * asserts the schedule took that path, then verifies BOTH the schedule-driven
 * search and the dense-masked baseline against an exhaustive CPU reference.
 */

#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/detail/filtered_search.cuh"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kN = 262144;  // 4 full 64K container blocks
constexpr uint32_t kD = 64;
constexpr uint32_t kQ = 4;
constexpr uint32_t kK = 8;

class FilteredSearchTest : public ::testing::Test {
 protected:
    std::vector<float> h_db, h_q;
    float* d_db = nullptr;
    float* d_q  = nullptr;
    cublasHandle_t handle{};
    cudaStream_t   stream{};

    void SetUp() override {
        std::mt19937 rng(12345);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        h_db.resize(static_cast<size_t>(kN) * kD);
        h_q.resize(static_cast<size_t>(kQ) * kD);
        for (auto& x : h_db) x = nd(rng);
        for (auto& x : h_q)  x = nd(rng);

        cudaMalloc(&d_db, h_db.size() * sizeof(float));
        cudaMalloc(&d_q,  h_q.size()  * sizeof(float));
        cudaMemcpy(d_db, h_db.data(), h_db.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q,  h_q.data(),  h_q.size()  * sizeof(float), cudaMemcpyHostToDevice);
        cublasCreate(&handle);
        cudaStreamCreate(&stream);
    }

    void TearDown() override {
        // No cudaDeviceReset(): the filtered-search executor keeps a
        // process-lifetime scratch cache, which a reset would invalidate.
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        cudaFree(d_db);
        cudaFree(d_q);
    }

    float dot(uint32_t q, uint32_t n) const {
        float s = 0.0f;
        for (uint32_t d = 0; d < kD; ++d)
            s += h_q[q * kD + d] * h_db[static_cast<size_t>(n) * kD + d];
        return s;
    }

    // Exhaustive CPU top-k over the eligible rows of `cpu_bm`.
    std::vector<std::vector<float>> cpu_reference(const roaring_bitmap_t* cpu_bm) {
        std::vector<std::vector<float>> out(kQ);
        for (uint32_t q = 0; q < kQ; ++q) {
            std::vector<float> sc;
            roaring_uint32_iterator_t* it = roaring_iterator_create(cpu_bm);
            while (it->has_value) {
                if (it->current_value < kN) sc.push_back(dot(q, it->current_value));
                roaring_uint32_iterator_advance(it);
            }
            roaring_uint32_iterator_free(it);
            std::sort(sc.begin(), sc.end(), std::greater<float>());
            sc.resize(std::min<size_t>(kK, sc.size()));
            out[q] = sc;
        }
        return out;
    }

    // Verify a GPU result against the CPU reference. Robust to score ties:
    // checks (a) every returned id is eligible and its score is the true dot,
    // (b) the sorted returned scores equal the CPU top-k scores.
    void verify(const char* label, const roaring_bitmap_t* cpu_bm,
                const std::vector<uint32_t>& ids,
                const std::vector<float>& scores,
                const std::vector<std::vector<float>>& ref) {
        const float tol = 2e-2f;
        for (uint32_t q = 0; q < kQ; ++q) {
            size_t valid = ref[q].size();
            std::vector<float> got;
            for (uint32_t i = 0; i < kK; ++i) {
                uint32_t id = ids[q * kK + i];
                float    sc = scores[q * kK + i];
                if (i < valid) {
                    EXPECT_TRUE(roaring_bitmap_contains(cpu_bm, id))
                        << label << ": q" << q << " slot" << i
                        << " returned ineligible id " << id;
                    EXPECT_NEAR(sc, dot(q, id), tol)
                        << label << ": q" << q << " slot" << i << " score/id mismatch";
                    got.push_back(sc);
                }
            }
            std::sort(got.begin(), got.end(), std::greater<float>());
            ASSERT_EQ(got.size(), valid) << label << ": q" << q;
            for (size_t i = 0; i < valid; ++i)
                EXPECT_NEAR(got[i], ref[q][i], tol)
                    << label << ": q" << q << " rank" << i << " not the true top-k";
        }
    }

    // Run both search paths on `cpu_bm` and verify them. `expect` is an
    // optional schedule-shape assertion.
    void run_case(const char* label, roaring_bitmap_t* cpu_bm,
                  void (*expect)(const cu_roaring::SearchSchedule&)) {
        auto gpu = cu_roaring::upload(cpu_bm, kN);
        auto ref = cpu_reference(cpu_bm);

        uint32_t* d_ids = nullptr;
        float*    d_sc  = nullptr;
        cudaMalloc(&d_ids, kQ * kK * sizeof(uint32_t));
        cudaMalloc(&d_sc,  kQ * kK * sizeof(float));
        std::vector<uint32_t> ids(kQ * kK);
        std::vector<float>    sc(kQ * kK);

        auto fetch = [&] {
            cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(sc.data(), d_sc, sc.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        };

        // Schedule-driven. Run twice: the second call exercises the
        // executor's CUDA-graph capture+replay path.
        auto sched = cu_roaring::build_schedule(gpu, kN, d_db, kD, stream);
        if (expect) expect(sched);
        for (int pass = 0; pass < 2; ++pass) {
            cu_roaring::roaring_filtered_search(handle, d_q, kQ, d_db, kN, kD,
                                                sched, kK, d_ids, d_sc, stream);
            cudaStreamSynchronize(stream);
            fetch();
            verify((std::string(label) + (pass ? " [roaring/graph]"
                                               : " [roaring]")).c_str(),
                   cpu_bm, ids, sc, ref);
        }
        cu_roaring::free_schedule(sched);

        // Dense-masked baseline, likewise twice.
        for (int pass = 0; pass < 2; ++pass) {
            cu_roaring::dense_filtered_search(handle, d_q, kQ, d_db, kN, kD,
                                              gpu, kK, d_ids, d_sc, stream);
            cudaStreamSynchronize(stream);
            fetch();
            verify((std::string(label) + (pass ? " [dense/graph]"
                                               : " [dense]")).c_str(),
                   cpu_bm, ids, sc, ref);
        }

        cudaFree(d_ids);
        cudaFree(d_sc);
        cu_roaring::gpu_roaring_free(gpu);
    }
};

TEST_F(FilteredSearchTest, RunContainers) {
    // A wide run (>= the direct-GEMM gate, and crossing a 64K container
    // boundary so enumerate_runs' cross-container coalescing is exercised)
    // plus a narrow run. The wide run stays a direct unmasked range task;
    // the narrow one is gathered. Total cardinality < kN/2 so upload() does
    // not auto-complement.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 1000, 80000);     // wide, crosses 65536
    roaring_bitmap_add_range(r, 200000, 240000);  // narrow -> gathered
    roaring_bitmap_run_optimize(r);

    auto gpu_probe = cu_roaring::upload(r, kN);
    ASSERT_FALSE(gpu_probe.negated);
    ASSERT_EQ(gpu_probe.n_run_containers, gpu_probe.n_containers);
    cu_roaring::gpu_roaring_free(gpu_probe);

    run_case("RunContainers", r, [](const cu_roaring::SearchSchedule& s) {
        bool direct_range = false;
        for (const auto& t : s.tasks)
            if (t.kind == cu_roaring::GemmTask::kRange && t.mask == nullptr)
                direct_range = true;
        EXPECT_TRUE(direct_range) << "wide run should keep a direct range GEMM";
    });
    roaring_bitmap_free(r);
}

TEST_F(FilteredSearchTest, ArrayContainers) {
    // Scattered sparse ids -> ARRAY containers -> gather-GEMM.
    roaring_bitmap_t* r = roaring_bitmap_create();
    for (uint32_t i = 0; i < kN; i += 53) roaring_bitmap_add(r, i);

    auto gpu_probe = cu_roaring::upload(r, kN);
    ASSERT_EQ(gpu_probe.n_array_containers, gpu_probe.n_containers);  // all ARRAY
    cu_roaring::gpu_roaring_free(gpu_probe);

    run_case("ArrayContainers", r, [](const cu_roaring::SearchSchedule& s) {
        bool has_gather = false;
        for (const auto& t : s.tasks)
            if (t.kind == cu_roaring::GemmTask::kGather) has_gather = true;
        EXPECT_TRUE(has_gather);
    });
    roaring_bitmap_free(r);
}

TEST_F(FilteredSearchTest, BitmapDense) {
    // One ~70%-dense scattered block -> BITMAP container, density >= gate
    // -> range task carrying the container's own bitset as a mask.
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 rng(7);
    for (uint32_t i = 0; i < 65536; ++i)
        if (rng() % 100 < 70) roaring_bitmap_add(r, i);

    auto gpu_probe = cu_roaring::upload(r, kN);
    ASSERT_EQ(gpu_probe.n_bitmap_containers, 1u);
    cu_roaring::gpu_roaring_free(gpu_probe);

    run_case("BitmapDense", r, [](const cu_roaring::SearchSchedule& s) {
        bool masked_range = false;
        for (const auto& t : s.tasks)
            if (t.kind == cu_roaring::GemmTask::kRange && t.mask) masked_range = true;
        EXPECT_TRUE(masked_range);
    });
    roaring_bitmap_free(r);
}

TEST_F(FilteredSearchTest, BitmapSparse) {
    // One ~20%-dense block: > 4096 elems (so BITMAP) but < 50% (so gather).
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 rng(9);
    for (uint32_t i = 0; i < 65536; ++i)
        if (rng() % 100 < 20) roaring_bitmap_add(r, i);

    auto gpu_probe = cu_roaring::upload(r, kN);
    ASSERT_EQ(gpu_probe.n_bitmap_containers, 1u);
    cu_roaring::gpu_roaring_free(gpu_probe);

    run_case("BitmapSparse", r, [](const cu_roaring::SearchSchedule& s) {
        bool has_gather = false;
        for (const auto& t : s.tasks)
            if (t.kind == cu_roaring::GemmTask::kGather) has_gather = true;
        EXPECT_TRUE(has_gather);
    });
    roaring_bitmap_free(r);
}

TEST_F(FilteredSearchTest, MixedContainers) {
    // A run, a scattered-array region, and a dense block in one filter.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 0, 50000);                 // -> RUN
    for (uint32_t i = 70000; i < 130000; i += 41)          // -> ARRAY
        roaring_bitmap_add(r, i);
    std::mt19937 rng(3);
    for (uint32_t i = 131072; i < 196608; ++i)             // dense block -> BITMAP
        if (rng() % 100 < 65) roaring_bitmap_add(r, i);
    roaring_bitmap_run_optimize(r);

    run_case("MixedContainers", r, nullptr);
    roaring_bitmap_free(r);
}

TEST_F(FilteredSearchTest, Negated) {
    // > 50% dense -> upload() stores the complement and sets negated=true.
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 0, 90000);
    roaring_bitmap_add_range(r, 100000, kN);
    roaring_bitmap_run_optimize(r);

    auto gpu_probe = cu_roaring::upload(r, kN);
    ASSERT_TRUE(gpu_probe.negated) << "expected auto-complement for a >50% filter";
    cu_roaring::gpu_roaring_free(gpu_probe);

    run_case("Negated", r, [](const cu_roaring::SearchSchedule& s) {
        EXPECT_GT(s.tasks.size(), 0u);
    });
    roaring_bitmap_free(r);
}

TEST_F(FilteredSearchTest, FragmentationFallback) {
    // Thousands of width-12 runs: mean range far below the shape gate
    // -> the schedule falls back to a single gather.
    roaring_bitmap_t* r = roaring_bitmap_create();
    for (uint32_t p = 0; p + 12 < kN; p += 600)
        roaring_bitmap_add_range(r, p, p + 12);
    roaring_bitmap_run_optimize(r);

    run_case("FragmentationFallback", r, [](const cu_roaring::SearchSchedule& s) {
        EXPECT_TRUE(s.used_fallback);
        bool has_gather = false;
        for (const auto& t : s.tasks)
            if (t.kind == cu_roaring::GemmTask::kGather) has_gather = true;
        EXPECT_TRUE(has_gather);
    });
    roaring_bitmap_free(r);
}

// fp16 path correctness — mixed-container filter exercising gather + masked
// range, both kinds of GEMM tiles converted to cublasGemmEx (CUDA_R_16F inputs,
// fp32 accumulation). Verifies the top-k IDs against the same CPU reference;
// scores are loosely compared because fp16 inputs introduce per-element
// quantisation noise that aliases into the dot-product result.
TEST_F(FilteredSearchTest, Fp16MixedContainers) {
    // Same filter shape as MixedContainers above (dense bitmap + array).
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 0, 50000);            // dense bitmap container
    for (uint32_t i = 65536; i < 131072; i += 30)     // sparse array container
        roaring_bitmap_add(r, i);

    // Convert dataset + queries to __half on device.
    std::vector<__half> h_db_h(h_db.size()), h_q_h(h_q.size());
    for (size_t i = 0; i < h_db.size(); ++i) h_db_h[i] = __float2half(h_db[i]);
    for (size_t i = 0; i < h_q.size();  ++i) h_q_h[i]  = __float2half(h_q[i]);
    __half* d_db_h = nullptr; __half* d_q_h = nullptr;
    cudaMalloc(&d_db_h, h_db_h.size() * sizeof(__half));
    cudaMalloc(&d_q_h,  h_q_h.size()  * sizeof(__half));
    cudaMemcpy(d_db_h, h_db_h.data(), h_db_h.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_h,  h_q_h.data(),  h_q_h.size()  * sizeof(__half),
               cudaMemcpyHostToDevice);

    auto gpu = cu_roaring::upload(r, kN);
    auto ref = cpu_reference(r);
    auto sched = cu_roaring::build_schedule(gpu, kN, d_db_h, kD, stream);

    uint32_t* d_ids = nullptr;
    float*    d_sc  = nullptr;
    cudaMalloc(&d_ids, kQ * kK * sizeof(uint32_t));
    cudaMalloc(&d_sc,  kQ * kK * sizeof(float));
    cu_roaring::roaring_filtered_search_fp16(handle, d_q_h, kQ, d_db_h, kN, kD,
                                             sched, kK, d_ids, d_sc, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint32_t> ids(kQ * kK);
    std::vector<float>    sc(kQ * kK);
    cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), d_sc, sc.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // The set of top-k IDs should agree with the fp32 reference. Score
    // tolerance is widened to account for fp16 input quantisation: per-element
    // error ~5e-4 relative, total dot-product error ~sqrt(D)*5e-4 ≈ 4e-3 on
    // unit-variance inputs, which is ~0.04 absolute for typical |dot| ~10.
    const float fp16_tol = 0.10f;
    for (uint32_t q = 0; q < kQ; ++q) {
        std::vector<uint32_t> ref_ids;
        roaring_uint32_iterator_t* it = roaring_iterator_create(r);
        std::vector<std::pair<float, uint32_t>> all;
        while (it->has_value) {
            all.emplace_back(dot(q, it->current_value), it->current_value);
            roaring_uint32_iterator_advance(it);
        }
        roaring_uint32_iterator_free(it);
        std::sort(all.begin(), all.end(), std::greater<>());
        for (size_t i = 0; i < std::min<size_t>(kK, all.size()); ++i)
            ref_ids.push_back(all[i].second);

        // Allow neighbouring score-tie permutations: every returned id should
        // be eligible and its fp32 score should match the top-k frontier
        // within fp16_tol.
        float frontier = all[std::min<size_t>(kK, all.size()) - 1].first;
        for (uint32_t i = 0; i < kK && i < ref_ids.size(); ++i) {
            uint32_t id = ids[q * kK + i];
            EXPECT_TRUE(roaring_bitmap_contains(r, id))
                << "fp16: q" << q << " slot" << i << " returned ineligible id " << id;
            float true_score = dot(q, id);
            EXPECT_GE(true_score, frontier - fp16_tol)
                << "fp16: q" << q << " slot" << i
                << " returned id whose fp32 score (" << true_score
                << ") is well below the top-k frontier (" << frontier << ")";
        }
    }

    cudaFree(d_ids);
    cudaFree(d_sc);
    cudaFree(d_db_h);
    cudaFree(d_q_h);
    cu_roaring::free_schedule(sched);
    cu_roaring::gpu_roaring_free(gpu);
    roaring_bitmap_free(r);
}

}  // namespace
