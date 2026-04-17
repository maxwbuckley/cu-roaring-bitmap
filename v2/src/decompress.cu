#include "cu_roaring_v2/api.hpp"
#include "internal.hpp"

#include <stdexcept>
#include <string>

namespace cu_roaring::v2 {

namespace {

// One block per container. Each block writes exactly the 1024-word window
// belonging to its container (word_base = key * 1024), so work across blocks
// never races. Within a block, ARRAY uses atomicOr for the natural
// word-collision case; RUN and BITMAP do plain stores since they partition
// words across threads without overlap.
__global__ void decompress_kernel(
    const uint16_t*      keys,
    const ContainerType* types,
    const uint32_t*      offsets,
    const uint16_t*      cards,
    const uint64_t*      bitmap_pool,
    const uint16_t*      array_pool,
    const uint16_t*      run_pool,
    uint32_t             n_containers,
    uint64_t*            out_bitset,
    size_t               out_words)
{
    const uint32_t cid = blockIdx.x;
    if (cid >= n_containers) return;

    const size_t word_base = static_cast<size_t>(keys[cid]) * 1024u;
    if (word_base >= out_words) return;

    const size_t window = (out_words - word_base < 1024u)
        ? (out_words - word_base) : 1024u;
    uint64_t* dst = out_bitset + word_base;

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

    // ARRAY / RUN: zero the window first so we can OR in on top.
    for (uint32_t w = threadIdx.x; w < window; w += blockDim.x) {
        dst[w] = 0ULL;
    }
    __syncthreads();

    if (type == ContainerType::ARRAY) {
        const uint16_t* arr = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(array_pool) + off);
        for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
            const uint16_t val      = arr[i];
            const uint32_t word_idx = val / 64u;
            if (word_idx < window) {
                atomicOr(
                    reinterpret_cast<unsigned long long*>(&dst[word_idx]),
                    static_cast<unsigned long long>(1ULL << (val % 64u)));
            }
        }
        return;
    }

    // RUN: emit whole 64-bit words per run. Runs within a container are disjoint
    // by Roaring's invariants, so different threads never target the same word.
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

        // Still use atomicOr as defence-in-depth: the disjoint-runs invariant
        // lives in the CPU CRoaring builder and in our RLE path. If either ever
        // produces overlapping runs we degrade to correct-but-slower rather
        // than silent data races.
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

void decompress_to_bitset(const GpuRoaring& bm,
                          uint64_t*         device_bitset,
                          size_t            words,
                          cudaStream_t      stream)
{
    if (bm.n_containers == 0) return;
    if (device_bitset == nullptr) {
        throw std::invalid_argument("decompress_to_bitset: device_bitset is null");
    }

    const size_t required = (static_cast<size_t>(bm.universe_size) + 63u) / 64u;
    if (words < required) {
        throw std::invalid_argument(
            "decompress_to_bitset: bitset capacity " + std::to_string(words) +
            " words < required " + std::to_string(required) + " words");
    }

    decompress_kernel<<<bm.n_containers, 256, 0, stream>>>(
        bm.keys, bm.types, bm.offsets, bm.cardinalities,
        bm.bitmap_data, bm.array_data, bm.run_data,
        bm.n_containers,
        device_bitset, words);
    CU_ROARING_V2_CHECK_CUDA(cudaGetLastError());
}

} // namespace cu_roaring::v2
