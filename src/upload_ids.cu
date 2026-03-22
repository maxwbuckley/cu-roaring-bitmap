#include "cu_roaring/detail/upload_ids.cuh"
#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

namespace cu_roaring {

GpuRoaring upload_from_sorted_ids(const uint32_t* sorted_ids,
                                  uint32_t n_ids,
                                  uint32_t universe_size,
                                  cudaStream_t stream,
                                  uint32_t bitmap_threshold)
{
  // Resolve PROMOTE_AUTO to a concrete threshold
  uint32_t effective_threshold = bitmap_threshold;
  if (bitmap_threshold == PROMOTE_AUTO) {
    effective_threshold = resolve_auto_threshold(universe_size);
  }

  GpuRoaring result{};
  result.universe_size = universe_size;
  if (n_ids == 0) return result;

  // Partition IDs into containers by high 16 bits
  struct Container {
    uint16_t key;
    std::vector<uint16_t> values;
  };
  std::vector<Container> containers;

  uint16_t cur_key = static_cast<uint16_t>(sorted_ids[0] >> 16);
  containers.push_back({cur_key, {}});
  for (uint32_t i = 0; i < n_ids; ++i) {
    uint16_t key = static_cast<uint16_t>(sorted_ids[i] >> 16);
    uint16_t val = static_cast<uint16_t>(sorted_ids[i] & 0xFFFF);
    if (key != cur_key) {
      cur_key = key;
      containers.push_back({key, {}});
    }
    containers.back().values.push_back(val);
  }

  uint32_t n = static_cast<uint32_t>(containers.size());
  result.n_containers = n;

  std::vector<uint16_t> h_keys(n);
  std::vector<ContainerType> h_types(n);
  std::vector<uint32_t> h_offsets(n);
  std::vector<uint16_t> h_cards(n);
  std::vector<uint64_t> h_bitmap_pool;
  std::vector<uint16_t> h_array_pool;
  uint32_t n_bitmap = 0, n_array = 0;

  for (uint32_t i = 0; i < n; ++i) {
    h_keys[i]  = containers[i].key;
    uint32_t card = static_cast<uint32_t>(containers[i].values.size());
    h_cards[i] = static_cast<uint16_t>(card > 65535 ? 0 : card);

    if (card > effective_threshold) {
      h_types[i]  = ContainerType::BITMAP;
      h_offsets[i] = static_cast<uint32_t>(h_bitmap_pool.size() * sizeof(uint64_t));
      size_t base  = h_bitmap_pool.size();
      h_bitmap_pool.resize(base + 1024, 0);
      for (uint16_t v : containers[i].values)
        h_bitmap_pool[base + v / 64] |= 1ULL << (v % 64);
      ++n_bitmap;
    } else {
      h_types[i]  = ContainerType::ARRAY;
      h_offsets[i] = static_cast<uint32_t>(h_array_pool.size() * sizeof(uint16_t));
      h_array_pool.insert(h_array_pool.end(),
                          containers[i].values.begin(),
                          containers[i].values.end());
      ++n_array;
    }
  }

  result.n_bitmap_containers = n_bitmap;
  result.n_array_containers  = n_array;
  result.n_run_containers    = 0;

  // Upload index arrays
  CUDA_CHECK(cudaMalloc(&result.keys, n * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&result.types, n * sizeof(ContainerType)));
  CUDA_CHECK(cudaMalloc(&result.offsets, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&result.cardinalities, n * sizeof(uint16_t)));

  CUDA_CHECK(cudaMemcpyAsync(result.keys, h_keys.data(),
                             n * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(result.types, h_types.data(),
                             n * sizeof(ContainerType), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(result.offsets, h_offsets.data(),
                             n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, h_cards.data(),
                             n * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));

  if (!h_bitmap_pool.empty()) {
    CUDA_CHECK(cudaMalloc(&result.bitmap_data,
                          h_bitmap_pool.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(result.bitmap_data, h_bitmap_pool.data(),
                               h_bitmap_pool.size() * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, stream));
  }
  if (!h_array_pool.empty()) {
    CUDA_CHECK(cudaMalloc(&result.array_data,
                          h_array_pool.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyAsync(result.array_data, h_array_pool.data(),
                               h_array_pool.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return result;
}

}  // namespace cu_roaring
