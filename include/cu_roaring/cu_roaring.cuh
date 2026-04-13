#pragma once

// Public umbrella header for cu_roaring_bitmap
#include "cu_roaring/types.cuh"
#include "cu_roaring/detail/utils.cuh"
#include "cu_roaring/detail/promote.cuh"
#include "cu_roaring/detail/upload.cuh"
#include "cu_roaring/detail/decompress.cuh"
#include "cu_roaring/detail/set_ops.cuh"
#include "cu_roaring/detail/upload_ids.cuh"
// Note: scoped_gpu_roaring.hpp and upload_pool.hpp are NOT included here
// because they pull in <roaring/roaring.h> which cannot be compiled by nvcc.
// Include them directly from .cpp files that need them.
