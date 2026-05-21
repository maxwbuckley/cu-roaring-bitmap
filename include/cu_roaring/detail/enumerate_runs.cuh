/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Enumerate coalesced run ranges from a GPU roaring bitmap.
 *
 * A RUN container stores contiguous ranges of set bits as (start, length)
 * pairs within a 64K key block. enumerate_runs() extracts every run, converts
 * it to an absolute half-open [start, end) ID range, and coalesces runs that
 * touch across container boundaries into maximal ranges.
 *
 * The intended consumer is a filter-driven search: each range is a contiguous
 * slab of eligible database rows, i.e. a GEMM column tile. This turns a dense
 * O(universe) masked search into O(n_runs) sub-GEMMs over only the rows that
 * pass the filter. See ENUMERATE_RUNS.md.
 */

#pragma once

#include "cu_roaring/types.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace cu_roaring {

/**
 * @brief A maximal contiguous range of set bits, as a half-open interval.
 *
 * `end - start` is the run length. For a GEMM tile, `start` is the database
 * row/column offset and `end - start` is the tile width.
 */
struct IdRange {
    uint32_t start;  ///< first set ID (inclusive)
    uint32_t end;    ///< one past the last set ID (exclusive)
};

/**
 * @brief Device array of coalesced ID ranges produced by enumerate_runs().
 *
 * Ranges are sorted by `start`, disjoint, and non-adjacent (any two ranges
 * have at least one clear bit between them).
 */
struct RunRanges {
    IdRange* ranges = nullptr;  ///< device pointer, [count] entries (nullptr if count == 0)
    uint32_t count  = 0;        ///< number of coalesced ranges
};

/**
 * @brief Extract coalesced run ranges from the RUN containers of a bitmap.
 *
 * Every run is converted to an absolute [start, end) range; runs that touch
 * across 64K container boundaries are merged into one range.
 *
 * Allocates the device range array — free it with free_run_ranges().
 * Synchronizes `stream` internally (run and range counts are read back to host
 * to size intermediate buffers). This is a one-time schedule-build step.
 *
 * Contract: reads RUN containers only. For bitmaps produced with
 * roaring_bitmap_run_optimize(), every contiguous region of more than a few
 * elements is stored as RUN container(s), so this captures them. ARRAY and
 * BITMAP containers hold sparse/scattered data by construction and are not
 * enumerated here (use enumerate_ids() / to_csr for those). If
 * `bitmap.negated` is true the stored runs describe the complement (the
 * excluded regions) — the caller must account for that.
 *
 * @param bitmap  GPU roaring bitmap
 * @param stream  CUDA stream
 * @return        RunRanges owning a device array (empty if no RUN containers)
 */
RunRanges enumerate_runs(const GpuRoaring& bitmap, cudaStream_t stream = 0);

/**
 * @brief Free a RunRanges returned by enumerate_runs(). Safe on an empty result.
 */
void free_run_ranges(RunRanges& ranges);

}  // namespace cu_roaring
