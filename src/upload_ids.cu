#include "cu_roaring/detail/upload_ids.cuh"
#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cstring>
#include <vector>

namespace cu_roaring {

// Below this threshold, CPU std::sort is faster than GPU sort due to
// kernel launch overhead and H2D/D2H transfer cost.
static constexpr uint32_t GPU_SORT_THRESHOLD = 65536;

// GPU sort + unique via CUB. Returns sorted, deduplicated IDs on host.
static std::vector<uint32_t> gpu_sort_unique(const uint32_t* ids,
                                              uint32_t n_ids,
                                              cudaStream_t stream)
{
    // Upload unsorted IDs to GPU
    uint32_t* d_in = nullptr;
    uint32_t* d_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n_ids * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted, n_ids * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_in, ids, n_ids * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    // CUB DeviceRadixSort
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes,
                                   d_in, d_sorted, n_ids, 0, 32, stream);
    void* d_sort_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));
    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_temp_bytes,
                                   d_in, d_sorted, n_ids, 0, 32, stream);
    cudaFree(d_sort_temp);
    cudaFree(d_in);

    // CUB DeviceSelect::Unique (in-place output)
    uint32_t* d_unique = nullptr;
    uint32_t* d_num_unique = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique, n_ids * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_unique, sizeof(uint32_t)));

    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, unique_temp_bytes,
                              d_sorted, d_unique, d_num_unique, n_ids, stream);
    void* d_unique_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique_temp, unique_temp_bytes));
    cub::DeviceSelect::Unique(d_unique_temp, unique_temp_bytes,
                              d_sorted, d_unique, d_num_unique, n_ids, stream);
    cudaFree(d_unique_temp);
    cudaFree(d_sorted);

    // Download unique count
    uint32_t h_num_unique = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_unique, d_num_unique, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_num_unique);

    // Download sorted unique IDs
    std::vector<uint32_t> result(h_num_unique);
    CUDA_CHECK(cudaMemcpy(result.data(), d_unique,
                          h_num_unique * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_unique);

    return result;
}

// CPU sort + unique (faster for small inputs)
static std::vector<uint32_t> cpu_sort_unique(const uint32_t* raw_ids,
                                              uint32_t n_ids)
{
    std::vector<uint32_t> ids(raw_ids, raw_ids + n_ids);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

GpuRoaring upload_from_ids(const uint32_t* raw_ids,
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

  // Sort and deduplicate — GPU for large inputs, CPU for small
  std::vector<uint32_t> ids = (n_ids > GPU_SORT_THRESHOLD)
      ? gpu_sort_unique(raw_ids, n_ids, stream)
      : cpu_sort_unique(raw_ids, n_ids);
  result.total_cardinality = ids.size();

  // Partition IDs into containers by high 16 bits
  struct Container {
    uint16_t key;
    std::vector<uint16_t> values;
  };
  std::vector<Container> containers;

  uint16_t cur_key = static_cast<uint16_t>(ids[0] >> 16);
  containers.push_back({cur_key, {}});
  for (size_t i = 0; i < ids.size(); ++i) {
    uint16_t key = static_cast<uint16_t>(ids[i] >> 16);
    uint16_t val = static_cast<uint16_t>(ids[i] & 0xFFFF);
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
