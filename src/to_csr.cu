/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cu_roaring/detail/to_csr.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <cub/device/device_scan.cuh>

namespace cu_roaring {

// ============================================================================
// Kernel: enumerate set element IDs from roaring containers into sorted output.
// One block per container, 256 threads per block.
//
// Array containers:  direct copy (O(cardinality), already sorted)
// Bitmap containers: block-cooperative bit extraction with shared-mem prefix sum
// Run containers:    expand runs with prefix sum on run lengths
// ============================================================================

static constexpr uint32_t BLOCK_SIZE = 256;

// Inclusive prefix sum in shared memory (Hillis-Steele, O(n log n) work).
// 256 elements is small enough that this is fine.
__device__ void block_inclusive_scan(uint32_t* smem, uint32_t tid)
{
  for (uint32_t stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    uint32_t val = (tid >= stride) ? smem[tid - stride] : 0;
    __syncthreads();
    smem[tid] += val;
    __syncthreads();
  }
}

__global__ void enumerate_ids_kernel(
    const uint16_t*      keys,
    const ContainerType* types,
    const uint32_t*      offsets,
    const uint16_t*      cardinalities,
    const uint32_t*      container_output_offsets,  // exclusive prefix sum of cardinalities
    uint32_t             n_containers,
    const uint64_t*      bitmap_data,
    const uint16_t*      array_data,
    const uint16_t*      run_data,
    int64_t*             output)
{
  uint32_t cid = blockIdx.x;
  if (cid >= n_containers) return;

  uint32_t key       = keys[cid];
  uint32_t base_id   = static_cast<uint32_t>(key) << 16;
  uint32_t out_start = container_output_offsets[cid];

  ContainerType ctype = types[cid];
  uint32_t offset     = offsets[cid];
  uint32_t tid        = threadIdx.x;

  if (ctype == ContainerType::ARRAY) {
    // Array container: sorted uint16_t values. Direct copy with widening.
    const uint16_t* arr = array_data + (offset / sizeof(uint16_t));
    uint16_t card       = cardinalities[cid];

    for (uint32_t i = tid; i < card; i += BLOCK_SIZE) {
      output[out_start + i] = static_cast<int64_t>(base_id | arr[i]);
    }

  } else if (ctype == ContainerType::BITMAP) {
    // Bitmap container: 1024 uint64_t words. Extract set bit positions in sorted order.
    //
    // Each thread handles a contiguous chunk of words so that output is sorted.
    // 1024 words / 256 threads = 4 words per thread.
    const uint64_t* bmp = bitmap_data + (offset / sizeof(uint64_t));

    // Pass 1: count set bits in my words
    __shared__ uint32_t counts[BLOCK_SIZE];

    uint32_t w_start    = tid * 4;
    uint32_t my_popcount = 0;
    for (uint32_t w = w_start; w < w_start + 4 && w < 1024; ++w) {
      my_popcount += __popcll(bmp[w]);
    }
    counts[tid] = my_popcount;
    __syncthreads();

    // Inclusive scan → exclusive offset per thread
    block_inclusive_scan(counts, tid);
    uint32_t my_offset = (tid > 0) ? counts[tid - 1] : 0;

    // Pass 2: extract set bits at the correct sorted position
    for (uint32_t w = w_start; w < w_start + 4 && w < 1024; ++w) {
      uint64_t word = bmp[w];
      while (word != 0) {
        uint32_t bit = __ffsll(word) - 1;
        uint32_t low = w * 64 + bit;
        output[out_start + my_offset] = static_cast<int64_t>(base_id | low);
        ++my_offset;
        word &= word - 1;  // clear lowest set bit
      }
    }

  } else if (ctype == ContainerType::RUN) {
    // Run container: pairs of (start, length) as uint16_t.
    // Expand each run into individual IDs.
    const uint16_t* runs = run_data + (offset / sizeof(uint16_t));
    uint16_t n_runs      = cardinalities[cid];

    // Pass 1: count elements per run handled by this thread
    __shared__ uint32_t counts[BLOCK_SIZE];
    uint32_t my_count = 0;
    for (uint32_t r = tid; r < n_runs; r += BLOCK_SIZE) {
      my_count += static_cast<uint32_t>(runs[r * 2 + 1]) + 1;
    }
    counts[tid] = my_count;
    __syncthreads();

    // Inclusive scan
    block_inclusive_scan(counts, tid);
    uint32_t my_offset = (tid > 0) ? counts[tid - 1] : 0;

    // Pass 2: expand runs at sorted positions
    for (uint32_t r = tid; r < n_runs; r += BLOCK_SIZE) {
      uint16_t start  = runs[r * 2];
      uint16_t length = runs[r * 2 + 1];
      for (uint32_t v = 0; v <= length; ++v) {
        output[out_start + my_offset] = static_cast<int64_t>(base_id | (start + v));
        ++my_offset;
      }
    }
  }
}

// Compute per-container element counts on device.
// For array/bitmap containers, cardinality IS the element count.
// For run containers, cardinality is n_runs — expand to element count
// by summing (length + 1) for each run pair.
__global__ void compute_element_counts_kernel(
    const ContainerType* types,
    const uint16_t*      cardinalities,
    const uint32_t*      offsets,
    const uint16_t*      run_data,
    uint32_t*            element_counts,
    uint32_t             n_containers)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_containers) return;

  if (types[i] != ContainerType::RUN) {
    element_counts[i] = cardinalities[i];
  } else {
    uint16_t n_runs      = cardinalities[i];
    const uint16_t* runs = run_data + (offsets[i] / sizeof(uint16_t));
    uint32_t count       = 0;
    for (uint16_t r = 0; r < n_runs; ++r) {
      count += static_cast<uint32_t>(runs[r * 2 + 1]) + 1;
    }
    element_counts[i] = count;
  }
}

void enumerate_ids(const GpuRoaring& bitmap, int64_t* output, cudaStream_t stream)
{
  if (bitmap.n_containers == 0 || bitmap.total_cardinality == 0) return;

  uint32_t n = bitmap.n_containers;

  // Step 1: compute per-container element counts entirely on device.
  uint32_t* d_counts  = nullptr;
  uint32_t* d_offsets = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_counts, n * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaMallocAsync(&d_offsets, n * sizeof(uint32_t), stream));

  uint32_t count_grid = (n + 255) / 256;
  compute_element_counts_kernel<<<count_grid, 256, 0, stream>>>(
      bitmap.types, bitmap.cardinalities, bitmap.offsets,
      bitmap.run_data, d_counts, n);
  CUDA_CHECK(cudaGetLastError());

  // Step 2: CUB exclusive prefix sum on device → container output offsets.
  void*  d_temp     = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_counts, d_offsets,
                                static_cast<int>(n), stream);
  CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_counts, d_offsets,
                                static_cast<int>(n), stream);
  CUDA_CHECK(cudaFreeAsync(d_temp, stream));
  CUDA_CHECK(cudaFreeAsync(d_counts, stream));

  // Step 3: extract IDs.
  dim3 grid(n);
  dim3 block(BLOCK_SIZE);

  enumerate_ids_kernel<<<grid, block, 0, stream>>>(
      bitmap.keys,
      bitmap.types,
      bitmap.offsets,
      bitmap.cardinalities,
      d_offsets,
      n,
      bitmap.bitmap_data,
      bitmap.array_data,
      bitmap.run_data,
      output);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFreeAsync(d_offsets, stream));
}

int64_t* enumerate_ids(const GpuRoaring& bitmap, cudaStream_t stream)
{
  if (bitmap.total_cardinality == 0) return nullptr;

  int64_t* output = nullptr;
  CUDA_CHECK(cudaMallocAsync(&output,
                              bitmap.total_cardinality * sizeof(int64_t), stream));

  enumerate_ids(bitmap, output, stream);
  return output;
}

}  // namespace cu_roaring
