#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "cu_roaring_v2/types.cuh"

// CRoaring's C++ build wraps the C types in roaring::api, so a plain
// extern-"C" forward-decl of roaring_bitmap_t would clash with the real type
// once roaring/roaring.h is included downstream. Pull the real header in here
// and let any cu_roaring::v2 user pick it up transitively, matching v1.
#include <roaring/roaring.h>

namespace cu_roaring::v2 {

// Upload N CPU Roaring bitmaps as one device-resident batch. Single packed
// cudaMallocAsync + single H2D memcpy for the whole batch — this is the scalar
// ingest path the rest of the library is designed around.
//
// Container types are preserved exactly as CRoaring has them (ARRAY/BITMAP/RUN);
// no reshape, no complement, no heuristics. n == 0 returns an empty batch. Any
// null entry in cpus[] raises std::invalid_argument.
GpuRoaringBatch upload_batch(const roaring_bitmap_t* const* cpus,
                             uint32_t                       n,
                             cudaStream_t                   stream = 0);

// Release all device memory and host-side scalar mirrors held by the batch.
// Safe on a default-constructed batch.
void free_batch(GpuRoaringBatch& batch);

// Promote every bitmap in the batch to all-bitmap format (every ARRAY / RUN
// container becomes a BITMAP container). Returns a new batch; the input is not
// modified and not freed. This is the only function in v2 that changes
// container types, and nothing else calls it implicitly — callers are in
// control of when promotion happens.
GpuRoaringBatch promote_batch(const GpuRoaringBatch& batch,
                              cudaStream_t           stream = 0);

// Host-side helper: build a thin view onto a single bitmap within the batch.
// O(1), no allocations. The returned view aliases the batch's pools and is
// valid until free_batch() is called. Pass the view by value to kernels that
// focus on one specific bitmap to get the fast (2-read) contains() path.
GpuRoaringView make_view(const GpuRoaringBatch& batch, uint32_t bitmap_idx);

// AND together the bitmaps in `batch` identified by input_indices[0..n_inputs).
// Every selected bitmap must be all-bitmap (callers typically run promote_batch
// over the whole batch once up front). input_indices is a host pointer;
// n_inputs must be >= 1. Returns a 1-bitmap batch.
//
// The n=1 case (a single index) degenerates to a copy of that bitmap into a new
// 1-batch — useful when the API needs a batch-shaped output but the caller has
// only one input.
GpuRoaringBatch multi_and(const GpuRoaringBatch& batch,
                          const uint32_t*        input_indices,
                          uint32_t               n_inputs,
                          cudaStream_t           stream = 0);

// Decompress every bitmap in the batch into a contiguous device bitset array
// of shape [n_bitmaps, words_each]. Bitmap b writes to
//   device_bitsets[b * words_each .. (b + 1) * words_each).
// words_each must be >= ceil(max(host_universe_sizes) / 64). Callers own the
// output allocation and are responsible for initialising it (typically to
// zeros): only the container windows actually present in each bitmap are
// written.
void decompress_batch(const GpuRoaringBatch& batch,
                      uint64_t*              device_bitsets,
                      uint64_t               words_each,
                      cudaStream_t           stream = 0);

} // namespace cu_roaring::v2
