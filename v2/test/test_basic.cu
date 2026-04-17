// Minimal smoke + differential test for cu_roaring::v2.
// Exercises every public function and cross-checks results against CPU CRoaring.

#include "cu_roaring_v2/api.hpp"
#include "cu_roaring_v2/query.cuh"

#include <roaring/roaring.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace crv2 = cu_roaring::v2;

// Pass GpuRoaring by value: the struct is POD with device pointers, the copy
// preserves the pointers and the underlying device memory outlives the launch.
__global__ void contains_kernel(crv2::GpuRoaring bm,
                                const uint32_t* ids,
                                uint32_t        n,
                                uint8_t*        out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = crv2::contains(bm, ids[i]) ? 1u : 0u;
}

static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    // Build a CRoaring bitmap with content that forces all three container types.
    roaring_bitmap_t* cpu_a = roaring_bitmap_create();

    // Sparse — expected ARRAY container at high16 = 0.
    for (uint32_t i = 0; i < 100; ++i) roaring_bitmap_add(cpu_a, i * 17u);

    // Dense — expected BITMAP container at high16 = 1.
    for (uint32_t i = 65536; i < 65536u + 60000u; ++i) roaring_bitmap_add(cpu_a, i);

    // Single long run — expected RUN container at high16 = 2 after run_optimize.
    for (uint32_t i = 131072; i < 131072u + 50000u; ++i) roaring_bitmap_add(cpu_a, i);
    roaring_bitmap_run_optimize(cpu_a);

    crv2::GpuRoaring bm_a = crv2::upload_from_croaring(cpu_a);

    if (bm_a.n_containers != 3) return fail("upload: expected 3 containers");
    if (bm_a.n_array_containers == 0) return fail("upload: expected at least one ARRAY container");
    if (bm_a.n_run_containers == 0)   return fail("upload: expected at least one RUN container");
    if (bm_a.key_index == nullptr)    return fail("upload: key_index not built");

    // Probe a spread of IDs covering hits, near-misses, and gaps between containers.
    std::vector<uint32_t> test_ids;
    test_ids.reserve(10000);
    for (uint32_t i = 0; i < 200000u; i += 37u) test_ids.push_back(i);

    const uint32_t n = static_cast<uint32_t>(test_ids.size());
    uint32_t* d_ids = nullptr;
    uint8_t*  d_out = nullptr;
    check_cuda(cudaMalloc(&d_ids, n * sizeof(uint32_t)), "alloc d_ids");
    check_cuda(cudaMalloc(&d_out, n),                    "alloc d_out");
    check_cuda(cudaMemcpy(d_ids, test_ids.data(), n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "memcpy ids");

    auto run_contains = [&](const crv2::GpuRoaring& bm, std::vector<uint8_t>& out) {
        contains_kernel<<<(n + 255u) / 256u, 256>>>(bm, d_ids, n, d_out);
        check_cuda(cudaDeviceSynchronize(), "contains sync");
        out.resize(n);
        check_cuda(cudaMemcpy(out.data(), d_out, n, cudaMemcpyDeviceToHost),
                   "memcpy out");
    };

    // contains() on the mixed-type bitmap must match CRoaring bit-for-bit.
    std::vector<uint8_t> got;
    run_contains(bm_a, got);
    for (uint32_t i = 0; i < n; ++i) {
        const bool gpu = got[i] != 0u;
        const bool cpu = roaring_bitmap_contains(cpu_a, test_ids[i]);
        if (gpu != cpu) {
            std::fprintf(stderr, "  id=%u gpu=%d cpu=%d\n", test_ids[i], gpu, cpu);
            return fail("contains mismatch (mixed types)");
        }
    }

    // promote_to_bitmap must preserve contains() semantics and turn every
    // container into BITMAP.
    crv2::GpuRoaring bm_a_prom = crv2::promote_to_bitmap(bm_a);
    if (bm_a_prom.n_containers        != bm_a.n_containers) return fail("promote: n_containers");
    if (bm_a_prom.n_bitmap_containers != bm_a.n_containers) return fail("promote: n_bitmap");
    if (bm_a_prom.n_array_containers  != 0)                 return fail("promote: n_array");
    if (bm_a_prom.n_run_containers    != 0)                 return fail("promote: n_run");
    if (bm_a_prom.total_cardinality   != bm_a.total_cardinality)
        return fail("promote: total_cardinality drift");

    run_contains(bm_a_prom, got);
    for (uint32_t i = 0; i < n; ++i) {
        const bool gpu = got[i] != 0u;
        const bool cpu = roaring_bitmap_contains(cpu_a, test_ids[i]);
        if (gpu != cpu) return fail("contains mismatch (after promote)");
    }

    // multi_and: build a second bitmap, promote both, intersect, verify.
    roaring_bitmap_t* cpu_b = roaring_bitmap_create();
    for (uint32_t i = 0; i < 250000u; i += 3u) roaring_bitmap_add(cpu_b, i);
    roaring_bitmap_run_optimize(cpu_b);

    crv2::GpuRoaring bm_b      = crv2::upload_from_croaring(cpu_b);
    crv2::GpuRoaring bm_b_prom = crv2::promote_to_bitmap(bm_b);

    crv2::GpuRoaring inputs[2] = {bm_a_prom, bm_b_prom};
    crv2::GpuRoaring anded     = crv2::multi_and(inputs, 2u);

    roaring_bitmap_t* expected = roaring_bitmap_and(cpu_a, cpu_b);
    const uint64_t expected_card = roaring_bitmap_get_cardinality(expected);
    if (anded.total_cardinality != expected_card) {
        std::fprintf(stderr, "  got=%llu expected=%llu\n",
                     static_cast<unsigned long long>(anded.total_cardinality),
                     static_cast<unsigned long long>(expected_card));
        return fail("multi_and: cardinality mismatch");
    }

    run_contains(anded, got);
    for (uint32_t i = 0; i < n; ++i) {
        const bool gpu = got[i] != 0u;
        const bool cpu = roaring_bitmap_contains(expected, test_ids[i]);
        if (gpu != cpu) return fail("contains mismatch (after multi_and)");
    }

    // multi_and must refuse mixed-type inputs (here: bm_a still has ARRAY/RUN).
    bool caught = false;
    try {
        crv2::GpuRoaring bad_inputs[2] = {bm_a, bm_b_prom};
        crv2::GpuRoaring bad_result = crv2::multi_and(bad_inputs, 2u);
        crv2::free_bitmap(bad_result);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    if (!caught) return fail("multi_and should reject mixed-type inputs");

    // decompress_to_bitset: flat bitset agrees with CRoaring on every probed id.
    const size_t bitset_words = (static_cast<size_t>(bm_a.universe_size) + 63u) / 64u;
    uint64_t* d_bitset = nullptr;
    check_cuda(cudaMalloc(&d_bitset, bitset_words * sizeof(uint64_t)), "alloc bitset");
    check_cuda(cudaMemset(d_bitset, 0, bitset_words * sizeof(uint64_t)),
               "zero bitset");
    crv2::decompress_to_bitset(bm_a, d_bitset, bitset_words);
    check_cuda(cudaDeviceSynchronize(), "decompress sync");

    std::vector<uint64_t> h_bitset(bitset_words);
    check_cuda(cudaMemcpy(h_bitset.data(), d_bitset,
                          bitset_words * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "memcpy bitset");
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t id = test_ids[i];
        if (id >= bm_a.universe_size) continue;
        const bool bs  = ((h_bitset[id / 64u] >> (id % 64u)) & 1ULL) != 0ULL;
        const bool cpu = roaring_bitmap_contains(cpu_a, id);
        if (bs != cpu) return fail("decompress_to_bitset mismatch");
    }

    // Cleanup. free_bitmap must be safe to call in any order and on empty bitmaps.
    crv2::free_bitmap(bm_a);
    crv2::free_bitmap(bm_a_prom);
    crv2::free_bitmap(bm_b);
    crv2::free_bitmap(bm_b_prom);
    crv2::free_bitmap(anded);
    crv2::GpuRoaring empty{};
    crv2::free_bitmap(empty);

    cudaFree(d_ids);
    cudaFree(d_out);
    cudaFree(d_bitset);
    roaring_bitmap_free(cpu_a);
    roaring_bitmap_free(cpu_b);
    roaring_bitmap_free(expected);

    std::printf("cu_roaring_v2 test_basic: OK\n");
    return 0;
}
