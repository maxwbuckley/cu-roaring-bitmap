// Differential test for cu_roaring::v2's batch-first API.
//
// Builds an n=4 batch spanning ARRAY / BITMAP / RUN container mixes, exercises
// every public function (upload_batch, make_view, contains, promote_batch,
// multi_and, decompress_batch), and verifies bit-for-bit against CPU CRoaring.
// Includes an n=1 case to confirm the degenerate-batch path still works.

#include "cu_roaring_v2/api.hpp"
#include "cu_roaring_v2/query.cuh"

#include <roaring/roaring.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace crv2 = cu_roaring::v2;

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

[[nodiscard]] int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

// Pass GpuRoaringView / GpuRoaringBatch by value — both are POD with device
// pointers; the copy preserves the pointers and the underlying memory outlives
// the launch.
__global__ void contains_view_kernel(crv2::GpuRoaringView view,
                                     const uint32_t* __restrict__ ids,
                                     uint32_t        n,
                                     uint8_t*        __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = crv2::contains(view, ids[i]) ? 1u : 0u;
}

__global__ void contains_batch_kernel(crv2::GpuRoaringBatch batch,
                                      uint32_t              b,
                                      const uint32_t* __restrict__ ids,
                                      uint32_t              n,
                                      uint8_t*        __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = crv2::contains(batch, b, ids[i]) ? 1u : 0u;
}

// Build a CPU CRoaring bitmap whose contents cover all three container types.
// The mix is parameterised so each of the four bitmaps in the batch differs
// (for non-trivial multi_and intersections later).
roaring_bitmap_t* build_mixed_bitmap(uint32_t variant) {
    roaring_bitmap_t* r = roaring_bitmap_create();

    // ARRAY at high16 = 0: ~100 sparse points, stride depends on variant.
    const uint32_t stride = 13u + variant;
    for (uint32_t i = 0; i < 200u; ++i) {
        roaring_bitmap_add(r, (i * stride) % 65536u);
    }

    // BITMAP at high16 = 1: dense block, bounds depend on variant.
    const uint32_t lo = 65536u + variant * 1000u;
    const uint32_t hi = 65536u + 60000u;
    for (uint32_t i = lo; i < hi; ++i) roaring_bitmap_add(r, i);

    // RUN at high16 = 2: long contiguous range, length depends on variant.
    const uint32_t run_start = 131072u + variant * 100u;
    const uint32_t run_end   = 131072u + 50000u;
    for (uint32_t i = run_start; i < run_end; ++i) roaring_bitmap_add(r, i);

    // Add a small extra to high16 = 3 only for some variants — exercises the
    // intersection logic dropping keys that aren't in every input.
    if (variant >= 2u) {
        for (uint32_t i = 0; i < 50u; ++i) {
            roaring_bitmap_add(r, 196608u + i * 7u);
        }
    }

    roaring_bitmap_run_optimize(r);
    return r;
}

std::vector<uint32_t> build_probe_set() {
    std::vector<uint32_t> probes;
    probes.reserve(8192);
    for (uint32_t i = 0; i < 280000u; i += 37u) probes.push_back(i);
    return probes;
}

bool diff_contains(const std::vector<uint8_t>& got,
                   const roaring_bitmap_t*     cpu,
                   const std::vector<uint32_t>& probes,
                   const char*                  ctx)
{
    for (size_t i = 0; i < probes.size(); ++i) {
        const bool gpu = got[i] != 0u;
        const bool ref = roaring_bitmap_contains(cpu, probes[i]);
        if (gpu != ref) {
            std::fprintf(stderr,
                "  %s mismatch at probe[%zu]=%u: gpu=%d ref=%d\n",
                ctx, i, probes[i], gpu, ref);
            return false;
        }
    }
    return true;
}

// Run contains(batch, b, id) in a kernel and pull results back to host.
void run_contains_batch(const crv2::GpuRoaringBatch& batch,
                        uint32_t                     b,
                        const uint32_t*              d_ids,
                        uint32_t                     n,
                        uint8_t*                     d_out,
                        std::vector<uint8_t>&        out)
{
    contains_batch_kernel<<<(n + 255u) / 256u, 256>>>(batch, b, d_ids, n, d_out);
    check_cuda(cudaDeviceSynchronize(), "contains_batch sync");
    out.resize(n);
    check_cuda(cudaMemcpy(out.data(), d_out, n, cudaMemcpyDeviceToHost),
               "contains_batch d2h");
}

void run_contains_view(const crv2::GpuRoaringView& view,
                       const uint32_t*             d_ids,
                       uint32_t                    n,
                       uint8_t*                    d_out,
                       std::vector<uint8_t>&       out)
{
    contains_view_kernel<<<(n + 255u) / 256u, 256>>>(view, d_ids, n, d_out);
    check_cuda(cudaDeviceSynchronize(), "contains_view sync");
    out.resize(n);
    check_cuda(cudaMemcpy(out.data(), d_out, n, cudaMemcpyDeviceToHost),
               "contains_view d2h");
}

int run_n1_smoke() {
    // Degenerate n=1 batch: single mixed bitmap, exercise upload, contains
    // (both overloads), promote, decompress.
    roaring_bitmap_t* cpu = build_mixed_bitmap(0u);
    const roaring_bitmap_t* arr[1] = {cpu};
    crv2::GpuRoaringBatch batch = crv2::upload_batch(arr, 1u);

    if (batch.n_bitmaps != 1u) return fail("n=1: n_bitmaps");
    if (batch.host_total_cardinalities[0]
            != roaring_bitmap_get_cardinality(cpu)) {
        return fail("n=1: total_cardinality mirror");
    }

    const auto probes = build_probe_set();
    const uint32_t n  = static_cast<uint32_t>(probes.size());
    uint32_t* d_ids = nullptr;
    uint8_t*  d_out = nullptr;
    check_cuda(cudaMalloc(&d_ids, n * sizeof(uint32_t)), "alloc d_ids");
    check_cuda(cudaMalloc(&d_out, n),                    "alloc d_out");
    check_cuda(cudaMemcpy(d_ids, probes.data(), n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "memcpy ids");

    std::vector<uint8_t> got;
    run_contains_batch(batch, 0u, d_ids, n, d_out, got);
    if (!diff_contains(got, cpu, probes, "n=1 contains(batch)")) {
        return fail("n=1: contains(batch) mismatch");
    }

    crv2::GpuRoaringView v = crv2::make_view(batch, 0u);
    run_contains_view(v, d_ids, n, d_out, got);
    if (!diff_contains(got, cpu, probes, "n=1 contains(view)")) {
        return fail("n=1: contains(view) mismatch");
    }

    crv2::GpuRoaringBatch promoted = crv2::promote_batch(batch);
    if (promoted.n_bitmap_containers_total != promoted.total_containers) {
        return fail("n=1 promote: not all-bitmap");
    }
    run_contains_batch(promoted, 0u, d_ids, n, d_out, got);
    if (!diff_contains(got, cpu, probes, "n=1 contains after promote")) {
        return fail("n=1: contains after promote");
    }

    // multi_and with n_inputs=1 must produce a copy.
    const uint32_t one_idx[1] = {0u};
    crv2::GpuRoaringBatch self = crv2::multi_and(promoted, one_idx, 1u);
    if (self.host_total_cardinalities[0] != promoted.host_total_cardinalities[0]) {
        std::fprintf(stderr, "  got=%llu expected=%llu\n",
            (unsigned long long)self.host_total_cardinalities[0],
            (unsigned long long)promoted.host_total_cardinalities[0]);
        return fail("n=1 multi_and(self) cardinality");
    }
    run_contains_batch(self, 0u, d_ids, n, d_out, got);
    if (!diff_contains(got, cpu, probes, "n=1 multi_and(self) contains")) {
        return fail("n=1 multi_and(self) contains");
    }

    crv2::free_batch(self);
    crv2::free_batch(promoted);
    crv2::free_batch(batch);
    cudaFree(d_ids);
    cudaFree(d_out);
    roaring_bitmap_free(cpu);
    return 0;
}

int run_n4_full() {
    constexpr uint32_t N = 4u;

    std::vector<roaring_bitmap_t*> cpu(N);
    for (uint32_t b = 0; b < N; ++b) cpu[b] = build_mixed_bitmap(b);

    std::vector<const roaring_bitmap_t*> cpu_ptrs(N);
    for (uint32_t b = 0; b < N; ++b) cpu_ptrs[b] = cpu[b];

    crv2::GpuRoaringBatch batch =
        crv2::upload_batch(cpu_ptrs.data(), N);

    if (batch.n_bitmaps != N)
        return fail("n=4: n_bitmaps");
    if (batch.host_container_starts[N] != batch.total_containers)
        return fail("n=4: container_starts terminator");

    // Verify per-bitmap host metadata against CRoaring.
    for (uint32_t b = 0; b < N; ++b) {
        const uint64_t expected_card = roaring_bitmap_get_cardinality(cpu[b]);
        if (batch.host_total_cardinalities[b] != expected_card) {
            std::fprintf(stderr, "  bitmap %u: host_total_cardinality=%llu vs %llu\n",
                b,
                (unsigned long long)batch.host_total_cardinalities[b],
                (unsigned long long)expected_card);
            return fail("n=4: host_total_cardinalities mismatch");
        }
        const uint32_t n_b = batch.host_container_starts[b + 1] -
                             batch.host_container_starts[b];
        const uint32_t n_b_ref =
            static_cast<uint32_t>(cpu[b]->high_low_container.size);
        if (n_b != n_b_ref) {
            std::fprintf(stderr, "  bitmap %u: n_containers=%u vs %u\n",
                b, n_b, n_b_ref);
            return fail("n=4: container_starts slice width");
        }
    }

    const auto probes = build_probe_set();
    const uint32_t n  = static_cast<uint32_t>(probes.size());
    uint32_t* d_ids = nullptr;
    uint8_t*  d_out = nullptr;
    check_cuda(cudaMalloc(&d_ids, n * sizeof(uint32_t)), "alloc d_ids");
    check_cuda(cudaMalloc(&d_out, n),                    "alloc d_out");
    check_cuda(cudaMemcpy(d_ids, probes.data(), n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "memcpy ids");

    std::vector<uint8_t> got;
    for (uint32_t b = 0; b < N; ++b) {
        run_contains_batch(batch, b, d_ids, n, d_out, got);
        char ctx[64];
        std::snprintf(ctx, sizeof(ctx), "n=4 contains(batch, b=%u)", b);
        if (!diff_contains(got, cpu[b], probes, ctx)) {
            return fail("n=4: contains(batch, b)");
        }

        crv2::GpuRoaringView v = crv2::make_view(batch, b);
        run_contains_view(v, d_ids, n, d_out, got);
        std::snprintf(ctx, sizeof(ctx), "n=4 contains(view, b=%u)", b);
        if (!diff_contains(got, cpu[b], probes, ctx)) {
            return fail("n=4: contains(view, b)");
        }
    }

    // promote_batch — every container becomes BITMAP; semantics preserved.
    crv2::GpuRoaringBatch promoted = crv2::promote_batch(batch);
    if (promoted.n_bitmaps != N) return fail("n=4 promote: n_bitmaps");
    if (promoted.total_containers != batch.total_containers)
        return fail("n=4 promote: total_containers drift");
    if (promoted.n_bitmap_containers_total != promoted.total_containers)
        return fail("n=4 promote: not all-bitmap");
    if (promoted.array_pool_bytes != 0u || promoted.run_pool_bytes != 0u)
        return fail("n=4 promote: residual ARRAY/RUN pools");
    for (uint32_t b = 0; b < N; ++b) {
        if (promoted.host_total_cardinalities[b]
                != batch.host_total_cardinalities[b]) {
            return fail("n=4 promote: total_cardinality drift");
        }
    }
    for (uint32_t b = 0; b < N; ++b) {
        run_contains_batch(promoted, b, d_ids, n, d_out, got);
        char ctx[64];
        std::snprintf(ctx, sizeof(ctx), "n=4 contains after promote, b=%u", b);
        if (!diff_contains(got, cpu[b], probes, ctx)) {
            return fail("n=4: contains after promote");
        }
    }

    // multi_and: AND across {0, 1, 2}.
    {
        const uint32_t indices[3] = {0u, 1u, 2u};
        crv2::GpuRoaringBatch anded =
            crv2::multi_and(promoted, indices, 3u);

        roaring_bitmap_t* expected = roaring_bitmap_and(cpu[0], cpu[1]);
        roaring_bitmap_t* exp2     = roaring_bitmap_and(expected, cpu[2]);
        roaring_bitmap_free(expected);
        expected = exp2;

        const uint64_t exp_card = roaring_bitmap_get_cardinality(expected);
        if (anded.host_total_cardinalities[0] != exp_card) {
            std::fprintf(stderr, "  multi_and(0,1,2): got=%llu expected=%llu\n",
                (unsigned long long)anded.host_total_cardinalities[0],
                (unsigned long long)exp_card);
            roaring_bitmap_free(expected);
            return fail("n=4 multi_and(0,1,2) cardinality");
        }
        run_contains_batch(anded, 0u, d_ids, n, d_out, got);
        bool ok = diff_contains(got, expected, probes, "multi_and(0,1,2)");
        roaring_bitmap_free(expected);
        if (!ok) return fail("n=4 multi_and(0,1,2) contains");
        crv2::free_batch(anded);
    }

    // multi_and: subset {2, 3} where 3 has the high16=3 ARRAY only common to
    // variants >= 2 — exercises a different anchor selection.
    {
        const uint32_t indices[2] = {2u, 3u};
        crv2::GpuRoaringBatch anded =
            crv2::multi_and(promoted, indices, 2u);

        roaring_bitmap_t* expected = roaring_bitmap_and(cpu[2], cpu[3]);
        const uint64_t exp_card = roaring_bitmap_get_cardinality(expected);
        if (anded.host_total_cardinalities[0] != exp_card) {
            std::fprintf(stderr, "  multi_and(2,3): got=%llu expected=%llu\n",
                (unsigned long long)anded.host_total_cardinalities[0],
                (unsigned long long)exp_card);
            roaring_bitmap_free(expected);
            return fail("n=4 multi_and(2,3) cardinality");
        }
        run_contains_batch(anded, 0u, d_ids, n, d_out, got);
        bool ok = diff_contains(got, expected, probes, "multi_and(2,3)");
        roaring_bitmap_free(expected);
        if (!ok) return fail("n=4 multi_and(2,3) contains");
        crv2::free_batch(anded);
    }

    // multi_and must reject non-all-bitmap inputs (raw batch still has ARRAY/RUN).
    {
        const uint32_t indices[2] = {0u, 1u};
        bool caught = false;
        try {
            crv2::GpuRoaringBatch bad =
                crv2::multi_and(batch, indices, 2u);
            crv2::free_batch(bad);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) return fail("n=4 multi_and must reject non-all-bitmap");
    }

    // decompress_batch: produce per-bitmap flat bitsets, verify per-id.
    {
        uint64_t max_universe = 0;
        for (uint32_t b = 0; b < N; ++b) {
            if (batch.host_universe_sizes[b] > max_universe) {
                max_universe = batch.host_universe_sizes[b];
            }
        }
        const uint64_t words_each = (max_universe + 63u) / 64u;
        const size_t   total_words = static_cast<size_t>(N) * words_each;

        uint64_t* d_bitsets = nullptr;
        check_cuda(cudaMalloc(&d_bitsets, total_words * sizeof(uint64_t)),
                   "alloc bitsets");
        check_cuda(cudaMemset(d_bitsets, 0, total_words * sizeof(uint64_t)),
                   "zero bitsets");

        crv2::decompress_batch(batch, d_bitsets, words_each);
        check_cuda(cudaDeviceSynchronize(), "decompress sync");

        std::vector<uint64_t> h_bitsets(total_words);
        check_cuda(cudaMemcpy(h_bitsets.data(), d_bitsets,
                              total_words * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost),
                   "memcpy bitsets");

        for (uint32_t b = 0; b < N; ++b) {
            const uint64_t* slot = h_bitsets.data() + b * words_each;
            for (size_t k = 0; k < probes.size(); ++k) {
                const uint32_t id = probes[k];
                if (id >= batch.host_universe_sizes[b]) continue;
                const bool bs  = ((slot[id / 64u] >> (id & 63u)) & 1ULL) != 0ULL;
                const bool ref = roaring_bitmap_contains(cpu[b], id);
                if (bs != ref) {
                    std::fprintf(stderr,
                        "  decompress mismatch b=%u id=%u: bs=%d ref=%d\n",
                        b, id, bs, ref);
                    cudaFree(d_bitsets);
                    return fail("n=4 decompress_batch mismatch");
                }
            }
        }
        cudaFree(d_bitsets);
    }

    // Cleanup.
    crv2::free_batch(promoted);
    crv2::free_batch(batch);
    cudaFree(d_ids);
    cudaFree(d_out);
    for (uint32_t b = 0; b < N; ++b) roaring_bitmap_free(cpu[b]);
    return 0;
}

} // namespace

int main() {
    if (int rc = run_n4_full(); rc != 0) return rc;
    if (int rc = run_n1_smoke(); rc != 0) return rc;

    // Empty-batch / free-batch idempotency.
    crv2::GpuRoaringBatch empty{};
    crv2::free_batch(empty);
    crv2::GpuRoaringBatch zero = crv2::upload_batch(nullptr, 0u);
    if (zero.n_bitmaps != 0u) return fail("upload_batch(n=0) should be empty");
    crv2::free_batch(zero);

    std::printf("cu_roaring_v2 test_basic: OK\n");
    return 0;
}
