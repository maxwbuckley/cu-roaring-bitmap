#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "cu_roaring_v2/types.cuh"

namespace cu_roaring::v2 {

namespace detail {

// Shared container-dispatch body. Used by both contains() overloads once the
// global container id has been resolved.
__device__ __forceinline__
bool contains_in_container(const ContainerType* types,
                           const uint32_t*      offsets,
                           const uint16_t*      cardinalities,
                           const uint64_t*      bitmap_data,
                           const uint16_t*      array_data,
                           const uint16_t*      run_data,
                           uint32_t             cid,
                           uint16_t             low)
{
    const ContainerType type = types[cid];
    const uint32_t      off  = __ldg(&offsets[cid]);

    if (type == ContainerType::BITMAP) {
        const uint64_t* data = reinterpret_cast<const uint64_t*>(
            reinterpret_cast<const char*>(bitmap_data) + off);
        return ((__ldg(&data[low >> 6]) >> (low & 63u)) & 1ULL) != 0ULL;
    }

    if (type == ContainerType::ARRAY) {
        const uint16_t* arr = reinterpret_cast<const uint16_t*>(
            reinterpret_cast<const char*>(array_data) + off);
        const uint16_t  card = __ldg(&cardinalities[cid]);
        uint32_t lo = 0, hi = card;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            const uint16_t v   = __ldg(&arr[mid]);
            if (v < low) lo = mid + 1;
            else         hi = mid;
        }
        return (lo < card) && (__ldg(&arr[lo]) == low);
    }

    // RUN: find the last run whose start <= low, then verify low <= start+length.
    const uint16_t* runs  = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const char*>(run_data) + off);
    const uint16_t  nruns = __ldg(&cardinalities[cid]);
    uint32_t lo = 0, hi = nruns;
    while (lo < hi) {
        const uint32_t mid   = (lo + hi) >> 1;
        const uint16_t start = __ldg(&runs[mid * 2]);
        if (start <= low) lo = mid + 1;
        else              hi = mid;
    }
    if (lo == 0) return false;
    const uint32_t r      = lo - 1;
    const uint16_t start  = __ldg(&runs[r * 2]);
    const uint16_t length = __ldg(&runs[r * 2 + 1]);
    return static_cast<uint32_t>(low) <= static_cast<uint32_t>(start) + length;
}

} // namespace detail

// Single-bitmap view — 2 global reads on the BITMAP hot path (key_index entry
// plus bitmap word). Use this for kernels where one specific bitmap is
// interrogated many times (e.g. a CAGRA traversal parameterised on one filter).
// Extract the view once on the host via make_view(), then pass it by value to
// the kernel.
__device__ __forceinline__
bool contains(const GpuRoaringView& view, uint32_t id) {
    const uint16_t high = static_cast<uint16_t>(id >> 16);
    if (static_cast<uint32_t>(high) > view.max_key) return false;

    const uint16_t local_cid = __ldg(&view.key_index[high]);
    if (local_cid == 0xFFFFu) return false;

    const uint16_t low = static_cast<uint16_t>(id & 0xFFFFu);
    return detail::contains_in_container(
        view.types, view.offsets, view.cardinalities,
        view.bitmap_data, view.array_data, view.run_data,
        static_cast<uint32_t>(local_cid), low);
}

// Batch query — 2 extra reads vs the view path, for the CSR slice bounds
// (key_index_starts[b], key_index_starts[b+1], container_starts[b]). Use this
// when different threads / warps of a kernel query different bitmaps of the
// batch; for single-bitmap hot loops, extract a view first.
__device__ __forceinline__
bool contains(const GpuRoaringBatch& batch, uint32_t b, uint32_t id) {
    if (b >= batch.n_bitmaps) return false;

    const uint32_t kidx_start = __ldg(&batch.key_index_starts[b]);
    const uint32_t kidx_end   = __ldg(&batch.key_index_starts[b + 1]);

    const uint16_t high = static_cast<uint16_t>(id >> 16);
    if (static_cast<uint32_t>(high) >= kidx_end - kidx_start) return false;

    const uint16_t local_cid = __ldg(&batch.key_indices[kidx_start + high]);
    if (local_cid == 0xFFFFu) return false;

    const uint32_t container_start = __ldg(&batch.container_starts[b]);
    const uint32_t global_cid = container_start + static_cast<uint32_t>(local_cid);
    const uint16_t low = static_cast<uint16_t>(id & 0xFFFFu);

    return detail::contains_in_container(
        batch.types, batch.offsets, batch.cardinalities,
        batch.bitmap_data, batch.array_data, batch.run_data,
        global_cid, low);
}

} // namespace cu_roaring::v2
