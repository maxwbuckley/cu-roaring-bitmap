/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pipeline:
 *   1. count_runs_kernel     -> runs per container (0 for non-RUN)
 *   2. CUB ExclusiveSum      -> per-container output offset into a flat array
 *   3. emit_intervals_kernel -> absolute half-open [lo, hi) intervals, sorted
 *   4. mark_starts_kernel    -> 1 where an interval begins a new merged range
 *   5. CUB InclusiveSum      -> range index per interval
 *   6. scatter_ranges_kernel -> write coalesced [start, end) ranges
 *
 * The intervals are globally sorted and disjoint by construction (containers
 * sorted by key, runs sorted within a container), so two intervals coalesce
 * iff interval[i].lo == interval[i-1].hi (half-open adjacency).
 */

#include "cu_roaring/detail/enumerate_runs.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <cub/device/device_scan.cuh>

namespace cu_roaring {

static constexpr uint32_t kBlock = 256;

// run_counts[i] = number of runs in container i (0 for non-RUN containers).
__global__ void count_runs_kernel(const ContainerType* types,
                                   const uint16_t*      cardinalities,
                                   uint32_t*            run_counts,
                                   uint32_t             n_containers)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_containers) return;
    run_counts[i] = (types[i] == ContainerType::RUN)
                        ? static_cast<uint32_t>(cardinalities[i])
                        : 0u;
}

// One block per container: emit each RUN container's runs as absolute
// half-open intervals [lo, hi) into a globally-sorted flat array.
__global__ void emit_intervals_kernel(const uint16_t*      keys,
                                       const ContainerType* types,
                                       const uint32_t*      offsets,
                                       const uint16_t*      cardinalities,
                                       const uint32_t*      run_offsets,
                                       const uint16_t*      run_data,
                                       uint32_t             n_containers,
                                       uint32_t*            interval_lo,
                                       uint32_t*            interval_hi)
{
    uint32_t cid = blockIdx.x;
    if (cid >= n_containers) return;
    if (types[cid] != ContainerType::RUN) return;

    uint32_t key     = keys[cid];
    uint32_t n_runs  = cardinalities[cid];
    uint32_t base    = run_offsets[cid];
    uint32_t base_id = key << 16;
    const uint16_t* runs = run_data + (offsets[cid] / sizeof(uint16_t));

    for (uint32_t r = threadIdx.x; r < n_runs; r += blockDim.x) {
        uint32_t start  = runs[r * 2];
        uint32_t length = runs[r * 2 + 1];
        uint32_t lo     = base_id + start;
        uint32_t hi     = lo + length + 1u;  // run covers [start, start+length]
        interval_lo[base + r] = lo;
        interval_hi[base + r] = hi;
    }
}

// is_start[i] = 1 if interval i begins a new coalesced range.
__global__ void mark_starts_kernel(const uint32_t* interval_lo,
                                    const uint32_t* interval_hi,
                                    uint32_t*       is_start,
                                    uint32_t        total_runs)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_runs) return;
    // Intervals are globally sorted and disjoint, so interval i touches the
    // previous one iff its lo equals the previous hi (half-open adjacency).
    is_start[i] = (i == 0u || interval_lo[i] != interval_hi[i - 1]) ? 1u : 0u;
}

// Write coalesced ranges: the first interval of a group writes .start, the
// last writes .end. inclusive_starts[i] = number of range-starts in [0, i],
// so the range index of interval i is inclusive_starts[i] - 1.
__global__ void scatter_ranges_kernel(const uint32_t* interval_lo,
                                       const uint32_t* interval_hi,
                                       const uint32_t* is_start,
                                       const uint32_t* inclusive_starts,
                                       uint32_t        total_runs,
                                       IdRange*        ranges)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_runs) return;

    uint32_t g = inclusive_starts[i] - 1u;  // range index for interval i
    if (is_start[i] != 0u) { ranges[g].start = interval_lo[i]; }

    bool is_last = (i + 1u == total_runs) || (is_start[i + 1] != 0u);
    if (is_last) { ranges[g].end = interval_hi[i]; }
}

RunRanges enumerate_runs(const GpuRoaring& bitmap, cudaStream_t stream)
{
    RunRanges result;  // {nullptr, 0}

    uint32_t n = bitmap.n_containers;
    if (n == 0) return result;

    // Step 1+2: per-container run counts -> exclusive prefix sum (output offsets).
    uint32_t* d_run_counts  = nullptr;
    uint32_t* d_run_offsets = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_run_counts, n * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_run_offsets, n * sizeof(uint32_t), stream));

    count_runs_kernel<<<div_ceil(n, kBlock), kBlock, 0, stream>>>(
        bitmap.types, bitmap.cardinalities, d_run_counts, n);
    CUDA_CHECK(cudaGetLastError());

    void*  d_temp     = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_run_counts,
                                  d_run_offsets, static_cast<int>(n), stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_run_counts,
                                  d_run_offsets, static_cast<int>(n), stream);
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));

    // total_runs = exclusive_sum[n-1] + run_counts[n-1]. Read both back.
    uint32_t h_last_offset = 0;
    uint32_t h_last_count  = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_last_offset, d_run_offsets + (n - 1),
                               sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_last_count, d_run_counts + (n - 1),
                               sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    uint32_t total_runs = h_last_offset + h_last_count;

    CUDA_CHECK(cudaFreeAsync(d_run_counts, stream));

    if (total_runs == 0) {
        CUDA_CHECK(cudaFreeAsync(d_run_offsets, stream));
        return result;  // no RUN containers
    }

    // Step 3: emit absolute half-open intervals, globally sorted.
    uint32_t* d_lo = nullptr;
    uint32_t* d_hi = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_lo, total_runs * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_hi, total_runs * sizeof(uint32_t), stream));

    emit_intervals_kernel<<<n, kBlock, 0, stream>>>(
        bitmap.keys, bitmap.types, bitmap.offsets, bitmap.cardinalities,
        d_run_offsets, bitmap.run_data, n, d_lo, d_hi);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_run_offsets, stream));

    // Step 4+5: mark range starts, inclusive-scan -> range index per interval.
    uint32_t* d_is_start = nullptr;
    uint32_t* d_incl     = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_is_start, total_runs * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_incl, total_runs * sizeof(uint32_t), stream));

    mark_starts_kernel<<<div_ceil(total_runs, kBlock), kBlock, 0, stream>>>(
        d_lo, d_hi, d_is_start, total_runs);
    CUDA_CHECK(cudaGetLastError());

    void*  d_temp2     = nullptr;
    size_t temp2_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp2, temp2_bytes, d_is_start, d_incl,
                                  static_cast<int>(total_runs), stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp2, temp2_bytes, stream));
    cub::DeviceScan::InclusiveSum(d_temp2, temp2_bytes, d_is_start, d_incl,
                                  static_cast<int>(total_runs), stream);
    CUDA_CHECK(cudaFreeAsync(d_temp2, stream));

    // count = number of coalesced ranges = last inclusive-sum value.
    uint32_t count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_incl + (total_runs - 1),
                               sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 6: scatter coalesced ranges into the result buffer.
    IdRange* d_ranges = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_ranges, count * sizeof(IdRange), stream));

    scatter_ranges_kernel<<<div_ceil(total_runs, kBlock), kBlock, 0, stream>>>(
        d_lo, d_hi, d_is_start, d_incl, total_runs, d_ranges);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_lo, stream));
    CUDA_CHECK(cudaFreeAsync(d_hi, stream));
    CUDA_CHECK(cudaFreeAsync(d_is_start, stream));
    CUDA_CHECK(cudaFreeAsync(d_incl, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // result ready for the caller

    result.ranges = d_ranges;
    result.count  = count;
    return result;
}

void free_run_ranges(RunRanges& ranges)
{
    if (ranges.ranges != nullptr) { CUDA_CHECK(cudaFree(ranges.ranges)); }
    ranges.ranges = nullptr;
    ranges.count  = 0;
}

}  // namespace cu_roaring
