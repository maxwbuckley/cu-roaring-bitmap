#pragma once
#include <cstdint>

namespace cu_roaring {

enum class ContainerTypeD : uint8_t {
  ARRAY  = 0,
  BITMAP = 1,
  RUN    = 2
};

// Lightweight, trivially-copyable device view of a GPU Roaring bitmap.
// All pointers are device pointers. Safe to capture by value in kernels.
struct GpuRoaringView {
  const uint16_t*       keys;
  const ContainerTypeD* types;
  const uint32_t*       offsets;
  const uint16_t*       cardinalities;
  uint32_t              n_containers;

  const uint64_t* bitmap_data;
  const uint16_t* array_data;
  const uint16_t* run_data;

  __device__ __forceinline__ bool contains(uint32_t id) const
  {
    const uint16_t key = static_cast<uint16_t>(id >> 16);
    const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);

    int idx = binary_search_keys(key);
    if (idx < 0) return false;

    auto type = load_type(idx);
    uint32_t off = load_offset(idx);

    switch (type) {
      case ContainerTypeD::BITMAP:
        return bitmap_contains(off, low);
      case ContainerTypeD::ARRAY:
        return array_contains(off, load_cardinality(idx), low);
      case ContainerTypeD::RUN:
        return run_contains(off, load_cardinality(idx), low);
      default: return false;
    }
  }

  // ---- Metadata loads via __ldg (read-only texture cache) ----

  __device__ __forceinline__ ContainerTypeD load_type(int idx) const
  {
    return static_cast<ContainerTypeD>(
        __ldg(reinterpret_cast<const uint8_t*>(types) + idx));
  }

  __device__ __forceinline__ uint32_t load_offset(int idx) const
  {
    return __ldg(offsets + idx);
  }

  __device__ __forceinline__ uint16_t load_cardinality(int idx) const
  {
    return __ldg(cardinalities + idx);
  }

  // ---- Key binary search (uses __ldg for each probe) ----

  __device__ __forceinline__ int binary_search_keys(uint16_t key) const
  {
    int lo = 0;
    int hi = static_cast<int>(n_containers) - 1;
    while (lo <= hi) {
      int mid          = (lo + hi) >> 1;
      uint16_t mid_key = __ldg(keys + mid);
      if (mid_key == key) return mid;
      if (mid_key < key)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    return -1;
  }

  // NOTE: all offsets are BYTE offsets into the respective pools.

  __device__ __forceinline__ bool bitmap_contains(uint32_t byte_offset, uint16_t low) const
  {
    uint64_t word = __ldg(bitmap_data + byte_offset / sizeof(uint64_t) + (low >> 6));
    return (word >> (low & 63)) & 1;
  }

  __device__ __forceinline__ bool array_contains(uint32_t byte_offset,
                                                 uint16_t card,
                                                 uint16_t low) const
  {
    const uint16_t* arr = array_data + byte_offset / sizeof(uint16_t);
    int lo = 0;
    int hi = static_cast<int>(card) - 1;
    while (lo <= hi) {
      int mid      = (lo + hi) >> 1;
      uint16_t val = __ldg(arr + mid);
      if (val == low) return true;
      if (val < low)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    return false;
  }

  __device__ __forceinline__ bool run_contains(uint32_t byte_offset,
                                               uint16_t n_runs,
                                               uint16_t low) const
  {
    const uint16_t* runs = run_data + byte_offset / sizeof(uint16_t);
    int lo = 0;
    int hi = static_cast<int>(n_runs) - 1;
    while (lo <= hi) {
      int mid        = (lo + hi) >> 1;
      uint16_t start = __ldg(runs + mid * 2);
      uint16_t len   = __ldg(runs + mid * 2 + 1);
      if (low < start)
        hi = mid - 1;
      else if (low > start + len)
        lo = mid + 1;
      else
        return true;
    }
    return false;
  }
};

}  // namespace cu_roaring
