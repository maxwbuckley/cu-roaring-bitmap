#pragma once
#include "roaring_view.cuh"

namespace cu_roaring {

// Warp-cooperative contains: threads sharing the same high-16 key
// amortise the binary search over the container key array.
__device__ __forceinline__ bool warp_contains(const GpuRoaringView& r, uint32_t id)
{
  const uint16_t key = static_cast<uint16_t>(id >> 16);
  const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);
  const unsigned lane = threadIdx.x & 31;

  // Use the active mask instead of FULL_MASK to avoid deadlocks when
  // not all warp lanes participate (e.g. inside CAGRA search kernels).
  const unsigned active_mask = __activemask();

  // Find the leader: for each thread, find the lowest lane with the same key.
  // Use __match_any_sync to get a mask of all lanes with the same key,
  // then find the lowest set bit as the leader.
#if __CUDA_ARCH__ >= 700
  unsigned match_mask = __match_any_sync(active_mask, static_cast<unsigned>(key));
  unsigned leader_lane = __ffs(match_mask) - 1;  // lowest set bit (0-indexed)
#else
  // Fallback for older architectures: every thread is its own leader
  unsigned leader_lane = lane;
#endif

  bool is_leader = (lane == leader_lane);

  // Leaders do the binary search
  int container_idx = -1;
  if (is_leader) {
    if (r.bloom_n_hashes > 0 && !r.bloom_may_contain(key)) {
      container_idx = -2;
    } else {
      container_idx = r.binary_search_keys(key);
    }
  }

  // Broadcast result from leader to all threads with the same key
  container_idx = __shfl_sync(active_mask, container_idx, leader_lane);

  if (container_idx < 0) return false;

  // Each thread does its own low-bits check
  ContainerTypeD type = r.types[container_idx];
  uint32_t offset     = r.offsets[container_idx];

  switch (type) {
    case ContainerTypeD::BITMAP:
      return r.bitmap_contains(offset, low);
    case ContainerTypeD::ARRAY:
      return r.array_contains(offset, r.cardinalities[container_idx], low);
    case ContainerTypeD::RUN:
      return r.run_contains(offset, r.cardinalities[container_idx], low);
    default: return false;
  }
}

}  // namespace cu_roaring
