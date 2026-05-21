/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Roaring-filtered brute-force vector search.
 *
 * A brute-force search scores a batch of queries against a database of N
 * row vectors (inner product) and keeps the top-k per query. A Roaring
 * bitmap acts as a row filter. There are two ways to use it:
 *
 *   - dense_filtered_search(): expand the filter to a flat bitset, run the
 *     full Q x N GEMM, mask, top-k.  Cost scales with the *universe* N.
 *
 *   - roaring_filtered_search(): never expand.  build_schedule() turns the
 *     filter's containers into a work schedule and the GEMM runs only over
 *     eligible rows.  Cost scales with *cardinality* and filter shape.
 *
 * The schedule honours the full container -> schedule dispatch table:
 *   absent  -> skipped entirely
 *   run     -> contiguous range, GEMM directly on db[start:end] (no copy)
 *   array   -> scattered ids, gather rows -> compact GEMM
 *   bitmap  -> density gate: near-full -> range + bitset mask
 *                            sparse    -> bit-scan -> gather
 *   negated -> the schedule is the complement (eligible = the gaps)
 *
 * A fragmentation fallback (the "selectivity / shape gate") converts an
 * over-fragmented range schedule into a single gather when the mean range
 * is too thin to amortise a per-range GEMM launch.
 *
 * See analysis/filter_driven_search.md for the design rationale.
 */

#pragma once

#include "cu_roaring/types.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace cu_roaring {

/// Largest k supported by the top-k merge kernel.
constexpr uint32_t kMaxK = 32;

/**
 * @brief One GEMM tile of the search schedule.
 *
 * A task covers `n_cols` database rows. `kind` selects where the rows live:
 *   - kRange : rows [start, start+n_cols) of the database, GEMM'd in place.
 *   - kGather: rows gather_ids[start .. start+n_cols), pre-gathered into the
 *              schedule's compact buffer.
 * `mask` (kRange only) is an optional eligibility bitset: column c is
 * eligible iff bit (mask_bit0 + c) is set (XOR `mask_invert`).
 */
struct GemmTask {
    enum Kind : uint32_t { kRange = 0, kGather = 1 };
    Kind            kind     = kRange;
    uint32_t        start    = 0;        ///< db row (kRange) or gather offset (kGather)
    uint32_t        n_cols   = 0;        ///< number of eligible-or-masked columns
    const uint32_t* mask     = nullptr;  ///< device bitset, or null = all eligible
    uint32_t        mask_bit0 = 0;       ///< global bit index of column 0
    uint8_t         mask_invert = 0;     ///< 1 = eligible when bit is clear
};

/**
 * @brief A built search schedule: GEMM tasks plus owned device buffers.
 *
 * Produced by build_schedule(), released by free_schedule().
 */
struct SearchSchedule {
    std::vector<GemmTask> tasks;                ///< host list of GEMM tiles
    uint32_t*             gather_ids  = nullptr;///< device [n_gather] global row ids
    float*                gather_db   = nullptr;///< device [n_gather*dim] gathered rows
    uint32_t              n_gather    = 0;
    uint32_t*             mask_pool   = nullptr;///< device, owned bitset masks
    uint64_t              total_cols  = 0;      ///< sum of task n_cols (the GEMM work)
    bool                  used_fallback = false;///< true if the shape gate forced a gather
};

/**
 * @brief Build a search schedule from a Roaring filter.
 *
 * Dispatches every container by type/density into range or gather tasks,
 * gathers the scattered rows, and applies the fragmentation fallback.
 *
 * @param filter  row filter over [0, n_rows)
 * @param n_rows  database row count
 * @param d_db    device database, row-major [n_rows * dim]
 * @param dim     vector dimension
 * @param stream  CUDA stream
 */
SearchSchedule build_schedule(const GpuRoaring& filter, uint32_t n_rows,
                              const float* d_db, uint32_t dim,
                              cudaStream_t stream = 0);

/// Release the device buffers owned by a SearchSchedule.
void free_schedule(SearchSchedule& schedule);

/**
 * @brief Schedule-driven filtered search (top-k inner product).
 *
 * @param handle      cuBLAS handle (stream set by the caller)
 * @param d_queries   device queries, row-major [q * dim]
 * @param q           query batch size
 * @param d_db        device database, row-major [n_rows * dim]
 * @param n_rows      database row count
 * @param dim         vector dimension
 * @param schedule    schedule from build_schedule()
 * @param k           neighbours per query (<= kMaxK)
 * @param d_out_ids   device output [q * k] global row ids, descending score
 * @param d_out_scores device output [q * k] scores
 * @param stream      CUDA stream
 */
void roaring_filtered_search(cublasHandle_t handle, const float* d_queries,
                             uint32_t q, const float* d_db, uint32_t n_rows,
                             uint32_t dim, const SearchSchedule& schedule,
                             uint32_t k, uint32_t* d_out_ids,
                             float* d_out_scores, cudaStream_t stream = 0);

/**
 * @brief Dense-masked baseline: full GEMM + decompressed-bitset mask + top-k.
 *
 * Same signature semantics as roaring_filtered_search(); takes the filter
 * directly (it expands it to a flat bitset internally).
 */
void dense_filtered_search(cublasHandle_t handle, const float* d_queries,
                           uint32_t q, const float* d_db, uint32_t n_rows,
                           uint32_t dim, const GpuRoaring& filter, uint32_t k,
                           uint32_t* d_out_ids, float* d_out_scores,
                           cudaStream_t stream = 0);

}  // namespace cu_roaring
