#pragma once
#include "roaring_view.cuh"

namespace cu_roaring {

// Warp-cooperative contains: threads sharing the same high-16 key
// amortise the key lookup across the warp.
//
// Fast path (all_bitmap + key_index): leader does 1 read (key_index),
// broadcasts container idx, each thread does 1 read (bitmap word).
// Total: 2 global reads, same as per-thread contains() fast path but
// with the key lookup amortised across matching lanes.
//
// Complement support: when r.negated is true, absent containers return
// true and membership results are flipped via XOR (1 cycle, 0 extra reads).
__device__ __forceinline__ bool warp_contains(const GpuRoaringView& r, uint32_t id)
{
  const uint16_t key = static_cast<uint16_t>(id >> 16);
  const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);
  const unsigned lane = threadIdx.x & 31;

  const unsigned active_mask = __activemask();

#if __CUDA_ARCH__ >= 700
  unsigned match_mask = __match_any_sync(active_mask, static_cast<unsigned>(key));
  unsigned leader_lane = __ffs(match_mask) - 1;
#else
  unsigned leader_lane = lane;
#endif

  bool is_leader = (lane == leader_lane);

  // Fast path: all-bitmap + key_index → leader does 1 read, everyone does 1 read
  if (r.all_bitmap && r.key_index) {
    int container_idx = -1;
    if (is_leader) {
      if (key <= r.max_key) {
        uint16_t idx = __ldg(r.key_index + key);
        container_idx = (idx == 0xFFFF) ? -1 : static_cast<int>(idx);
      }
    }
    container_idx = __shfl_sync(active_mask, container_idx, leader_lane);
    if (container_idx < 0) return r.negated;

    // Each thread reads its own bitmap word — offset is implicit
    uint64_t word = __ldg(r.bitmap_data +
                          static_cast<uint32_t>(container_idx) * 1024 + (low >> 6));
    return static_cast<bool>((word >> (low & 63)) & 1) ^ r.negated;
  }

  // General path: leader reads key + metadata, broadcasts everything
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

  container_idx = __shfl_sync(active_mask, container_idx, leader_lane);
  if (container_idx < 0) return r.negated;

  meta_type   = __shfl_sync(active_mask, meta_type, leader_lane);
  meta_offset = __shfl_sync(active_mask, meta_offset, leader_lane);
  meta_card   = __shfl_sync(active_mask, meta_card, leader_lane);

  bool result;
  switch (static_cast<ContainerTypeD>(meta_type)) {
    case ContainerTypeD::BITMAP:
      result = r.bitmap_contains(meta_offset, low);
      break;
    case ContainerTypeD::ARRAY:
      result = r.array_contains(meta_offset, static_cast<uint16_t>(meta_card), low);
      break;
    case ContainerTypeD::RUN:
      result = r.run_contains(meta_offset, static_cast<uint16_t>(meta_card), low);
      break;
    default: result = false;
  }
  return result ^ r.negated;
}

}  // namespace cu_roaring
