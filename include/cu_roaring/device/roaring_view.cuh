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

  // Direct-map key index: O(1) lookup replaces O(log n) binary search
  const uint16_t* key_index;    // key_index[high16] = container idx, 0xFFFF = absent
  uint32_t        max_key;      // highest valid key value

  // When true, all containers are bitmap and offsets are sequential
  // (idx * 1024). The fast path skips type/offset/cardinality reads
  // entirely: 2 global reads instead of 4.
  bool            all_bitmap;

  // Complement flag: when true, the stored set is the complement of the
  // logical set. Absent containers represent "all ones" (full 65536-element
  // range), and membership test results are flipped via XOR.
  bool            negated;

  __device__ __forceinline__ bool contains(uint32_t id) const
  {
    const uint16_t key = static_cast<uint16_t>(id >> 16);
    const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);

    // Fast path: key_index + all-bitmap → 2 reads total
    if (all_bitmap && key_index) {
      if (key > max_key) return negated;
      uint16_t idx = __ldg(key_index + key);
      if (idx == 0xFFFF) return negated;
      // Offset is implicit: idx * 1024 words
      uint64_t word = __ldg(bitmap_data + static_cast<uint32_t>(idx) * 1024 + (low >> 6));
      return static_cast<bool>((word >> (low & 63)) & 1) ^ negated;
    }

    // General path: lookup key, read metadata, dispatch by type
    int idx = lookup_key(key);
    if (idx < 0) return negated;

    auto type = load_type(idx);
    uint32_t off = load_offset(idx);

    bool result;
    switch (type) {
      case ContainerTypeD::BITMAP:
        result = bitmap_contains(off, low);
        break;
      case ContainerTypeD::ARRAY:
        result = array_contains(off, load_cardinality(idx), low);
        break;
      case ContainerTypeD::RUN:
        result = run_contains(off, load_cardinality(idx), low);
        break;
      default: result = false;
    }
    return result ^ negated;
  }

  // ---- Key lookup: O(1) direct-map or O(log n) binary search fallback ----

  __device__ __forceinline__ int lookup_key(uint16_t key) const
  {
    if (key_index) {
      if (key > max_key) return -1;
      uint16_t idx = __ldg(key_index + key);
      return (idx == 0xFFFF) ? -1 : static_cast<int>(idx);
    }
    return binary_search_keys(key);
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

  // ---- Binary search fallback ----

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
