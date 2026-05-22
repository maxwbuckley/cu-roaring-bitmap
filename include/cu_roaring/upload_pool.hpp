#pragma once
#include <cuda_runtime.h>
#include "cu_roaring/types.cuh"
#include "cu_roaring/detail/promote.cuh"
#include <roaring/roaring.h>

namespace cu_roaring {

/// Pre-allocated host+device buffer pool for zero-allocation uploads.
///
/// Eliminates cudaMallocHost + cudaMalloc from the upload hot path.
/// The pool owns one pinned host buffer and one device buffer, both sized
/// at construction. upload() packs into the host buffer and issues a
/// single cudaMemcpyAsync — no allocations, no syncs.
///
/// Usage:
/// @code
///   // Once at init (pre-allocates 4 MB each on host + device)
///   cu_roaring::UploadPool pool(4 * 1024 * 1024);
///
///   // Per-query (zero alloc, ~0.01 ms overhead)
///   GpuRoaring gpu = pool.upload(cpu_bitmap, universe_size, stream);
///   auto filter = cuvs::neighbors::filtering::roaring_filter(gpu);
///   cagra::search(res, params, index, queries, nb, dist, filter);
///   // No gpu_roaring_free needed — pool owns the memory.
///   // gpu is valid until the next pool.upload() call.
/// @endcode
class UploadPool {
 public:
  /// Create a pool with the given capacity in bytes.
  /// Allocates one pinned host buffer and one device buffer.
  explicit UploadPool(size_t capacity_bytes, cudaStream_t stream = 0);

  ~UploadPool();

  // Non-copyable, movable
  UploadPool(const UploadPool&) = delete;
  UploadPool& operator=(const UploadPool&) = delete;
  UploadPool(UploadPool&& other) noexcept;
  UploadPool& operator=(UploadPool&& other) noexcept;

  /// Upload a CRoaring bitmap into the pool's pre-allocated buffers.
  /// Returns a GpuRoaring whose device pointers point into the pool's device
  /// buffer.  Valid until the next upload() call (overwrites the buffer).
  ///
  /// If the bitmap is too large for the pool, falls back to the standard
  /// upload() path (with allocation).
  GpuRoaring upload(const roaring_bitmap_t* cpu_bitmap,
                    uint32_t universe_size,
                    cudaStream_t stream = 0,
                    uint32_t bitmap_threshold = PROMOTE_KEEP_DEFAULT);

  /// Query pool capacity and last upload size.
  size_t capacity() const { return capacity_; }
  size_t last_upload_bytes() const { return last_bytes_; }

 private:
  char*  h_pinned_  = nullptr;  // pinned host buffer
  char*  d_buf_     = nullptr;  // device buffer
  size_t capacity_  = 0;
  size_t last_bytes_ = 0;
};

}  // namespace cu_roaring
