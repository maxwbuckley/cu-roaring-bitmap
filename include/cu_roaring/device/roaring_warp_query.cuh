#pragma once
#include "roaring_view.cuh"

namespace cu_roaring {

// Warp-cooperative contains: threads sharing the same high-16 key
// amortise the binary search over the container key array.
// The leader also reads container metadata and broadcasts it via
// __shfl_sync, so follower lanes never touch global memory for
// type/offset/cardinality lookups.
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

  // Leaders do the binary search + metadata read.
  // Followers get results via __shfl_sync (no global memory access).
  int container_idx = -1;
  uint32_t meta_type   = 0;
  uint32_t meta_offset = 0;
  uint32_t meta_card   = 0;

  if (is_leader) {
    container_idx = r.lookup_key(key);

    if (container_idx >= 0) {
      meta_type   = static_cast<uint32_t>(r.load_type(container_idx));
      meta_offset = r.load_offset(container_idx);
      meta_card   = r.load_cardinality(container_idx);
    }
  }

  // Broadcast everything from leader — followers do zero global reads
  container_idx = __shfl_sync(active_mask, container_idx, leader_lane);
  if (container_idx < 0) return false;

  meta_type   = __shfl_sync(active_mask, meta_type, leader_lane);
  meta_offset = __shfl_sync(active_mask, meta_offset, leader_lane);
  meta_card   = __shfl_sync(active_mask, meta_card, leader_lane);

  // Each thread does its own low-bits check using the broadcast metadata
  switch (static_cast<ContainerTypeD>(meta_type)) {
    case ContainerTypeD::BITMAP:
      return r.bitmap_contains(meta_offset, low);
    case ContainerTypeD::ARRAY:
      return r.array_contains(meta_offset, static_cast<uint16_t>(meta_card), low);
    case ContainerTypeD::RUN:
      return r.run_contains(meta_offset, static_cast<uint16_t>(meta_card), low);
    default: return false;
  }
}

}  // namespace cu_roaring
