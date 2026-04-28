#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <stdexcept>
#include <string>

namespace cu_roaring::v2 {

namespace {

// Find the bitmap index `b` that owns global container `cid`, defined by
// container_starts[b] <= cid < container_starts[b+1]. The CSR start array is
// monotone non-decreasing with length n_bitmaps + 1, so this is upper_bound on
// the suffix [1, n_bitmaps+1). Pure device-side, O(log n_bitmaps), one read of
// container_starts per binary-search step.
__device__ __forceinline__
uint32_t find_owning_bitmap(const uint32_t* container_starts,
                            uint32_t        n_bitmaps,
                            uint32_t        cid)
{
    uint32_t lo = 0u;
    uint32_t hi = n_bitmaps;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (container_starts[mid + 1u] > cid) hi = mid;
        else                                  lo = mid + 1u;
    }
    return lo;
}

// One block per global container. Each block resolves its owning bitmap b,
// then writes its 1024-word window into device_bitsets[b * words_each + key *
// 1024 .. + 1024]. Per Roaring's invariants, distinct global containers within
// a single bitmap own disjoint key prefixes (different `key`), so the per-block
// windows never overlap and the kernel needs no cross-block synchronisation.
//
// Caller owns zero-initialisation of device_bitsets. ARRAY / RUN paths use
// atomicOr to set bits within their assigned word and rely on the pre-zeroed
// state.
__global__ void decompress_kernel(
    const uint32_t*      container_starts,
    uint32_t             n_bitmaps,
    const uint16_t*      keys,
    const ContainerType* types,
    const uint32_t*      offsets,
    const uint16_t*      cards,
    const uint64_t*      bitmap_pool,
    const uint16_t*      array_pool,
    const uint16_t*      run_pool,
    uint32_t             total_containers,
    uint64_t*            out_bitsets,
    uint64_t             words_each)
{
    __shared__ uint32_t s_b;

    const uint32_t cid = blockIdx.x;
    if (cid >= total_containers) return;

    if (threadIdx.x == 0) {
        s_b = find_owning_bitmap(container_starts, n_bitmaps, cid);
    }
    __syncthreads();
    const uint32_t b = s_b;

    const size_t bitmap_base = static_cast<size_t>(b) * words_each;
    const size_t bitmap_end  = bitmap_base + words_each;
    const size_t word_base   = bitmap_base
                             + static_cast<size_t>(keys[cid]) * 1024u;
    if (word_base >= bitmap_end) return;

    const size_t window = (bitmap_end - word_base < 1024u)
        ? (bitmap_end - word_base) : 1024u;
    uint64_t* dst = out_bitsets + word_base;

    const ContainerType type = types[cid];
    const uint32_t      off  = offsets[cid];
    const uint16_t      card = cards[cid];

    if (type == ContainerType::BITMAP) {
        const uint64_t* src = reinterpret_cast<const uint64_t*>(
            reinterpret_cast<const char*>(bitmap_pool) + off);
        for (uint32_t w = threadIdx.x; w < window; w += blockDim.x) {
            dst[w] = src[w];
        }
        return;
    }

    if (type == ContainerType::ARRAY) {
        const uint16_t* arr = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(array_pool) + off);
        for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
            const uint16_t val      = arr[i];
            const uint32_t word_idx = val / 64u;
            if (word_idx < window) {
                atomicOr(reinterpret_cast<unsigned long long*>(&dst[word_idx]),
                         static_cast<unsigned long long>(1ULL << (val % 64u)));
            }
        }
        return;
    }

    // RUN: emit whole 64-bit words per run. Within a container the runs are
    // disjoint per Roaring invariants, so threads writing different runs never
    // target the same word. AtomicOr is kept as a defensive measure: if either
    // the CRoaring builder or the upload path ever produced overlapping runs,
    // we'd degrade to correct-but-slower rather than silent data races.
    const uint16_t* runs = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const char*>(run_pool) + off);
    for (uint32_t r = threadIdx.x; r < card; r += blockDim.x) {
        const uint32_t start  = runs[r * 2u];
        const uint32_t length = runs[r * 2u + 1u];
        uint32_t end = start + length;
        if (end > 0xFFFFu) end = 0xFFFFu;

        const uint32_t head_word = start / 64u;
        const uint32_t tail_word = end   / 64u;
        const uint32_t head_bit  = start % 64u;
        const uint32_t tail_bit  = end   % 64u;

        if (head_word == tail_word) {
            if (head_word >= window) continue;
            const uint32_t nbits = tail_bit - head_bit + 1u;
            const uint64_t mask  = (nbits == 64u)
                ? ~0ULL
                : (((1ULL << nbits) - 1ULL) << head_bit);
            atomicOr(reinterpret_cast<unsigned long long*>(&dst[head_word]),
                     static_cast<unsigned long long>(mask));
        } else {
            if (head_word < window) {
                atomicOr(reinterpret_cast<unsigned long long*>(&dst[head_word]),
                         static_cast<unsigned long long>(~0ULL << head_bit));
            }
            for (uint32_t w = head_word + 1u; w < tail_word && w < window; ++w) {
                atomicOr(reinterpret_cast<unsigned long long*>(&dst[w]),
                         static_cast<unsigned long long>(~0ULL));
            }
            if (tail_word < window) {
                const uint64_t tail_mask = (tail_bit == 63u)
                    ? ~0ULL
                    : ((1ULL << (tail_bit + 1u)) - 1ULL);
                atomicOr(reinterpret_cast<unsigned long long*>(&dst[tail_word]),
                         static_cast<unsigned long long>(tail_mask));
            }
        }
    }
}

} // namespace

void decompress_batch(const GpuRoaringBatch& batch,
                      uint64_t*              device_bitsets,
                      uint64_t               words_each,
                      cudaStream_t           stream)
{
    if (batch.n_bitmaps == 0 || batch.total_containers == 0) return;
    if (device_bitsets == nullptr) {
        throw std::invalid_argument("decompress_batch: device_bitsets is null");
    }

    // Validate words_each on the host using the cached universe sizes; this
    // turns a potential silent OOB write inside the kernel into an immediate,
    // diagnosable error at the API boundary.
    uint32_t max_universe = 0u;
    for (uint32_t b = 0; b < batch.n_bitmaps; ++b) {
        if (batch.host_universe_sizes[b] > max_universe) {
            max_universe = batch.host_universe_sizes[b];
        }
    }
    const uint64_t required_words = (static_cast<uint64_t>(max_universe) + 63ull) / 64ull;
    if (words_each < required_words) {
        throw std::invalid_argument(
            "decompress_batch: words_each " + std::to_string(words_each) +
            " < required " + std::to_string(required_words));
    }

    decompress_kernel<<<batch.total_containers, 256, 0, stream>>>(
        batch.container_starts, batch.n_bitmaps,
        batch.keys, batch.types, batch.offsets, batch.cardinalities,
        batch.bitmap_data, batch.array_data, batch.run_data,
        batch.total_containers,
        device_bitsets, words_each);
    CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());
}

} // namespace cu_roaring::v2
