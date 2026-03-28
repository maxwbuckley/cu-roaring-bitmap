/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Export roaring bitmap contents as sorted element IDs for CSR construction.
 *
 * Array containers: direct copy (already sorted uint16_t → int64_t widening)
 * Bitmap containers: popcount + prefix scan → extract set bit positions
 * Absent containers: skip (zero work)
 *
 * This is much faster than bitset→CSR because:
 *   - Array containers are already a sorted ID list (no bit scanning)
 *   - Bitmap containers are 8KB each (L1-resident) vs full bitset
 *   - Absent containers are skipped entirely
 */

#pragma once

#include "cu_roaring/types.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace cu_roaring {

/**
 * @brief Export all set element IDs as a sorted int64_t array on GPU.
 *
 * Writes exactly bitmap.total_cardinality elements to output.
 * Output must be pre-allocated with at least total_cardinality int64_t's.
 *
 * @param bitmap     GPU roaring bitmap
 * @param output     Device pointer to output array (pre-allocated)
 * @param stream     CUDA stream
 */
void enumerate_ids(const GpuRoaring& bitmap, int64_t* output, cudaStream_t stream = 0);

/**
 * @brief Allocate and export all set element IDs as a sorted int64_t array.
 *
 * Caller must cudaFree the returned pointer.
 *
 * @param bitmap     GPU roaring bitmap
 * @param stream     CUDA stream
 * @return           Device pointer to sorted ID array (caller owns)
 */
int64_t* enumerate_ids(const GpuRoaring& bitmap, cudaStream_t stream = 0);

}  // namespace cu_roaring
