# cu-roaring-filter

GPU-accelerated Roaring Bitmap library for NVIDIA CUDA. Enables compressed set membership queries, boolean set operations, and integration with GPU vector search kernels — all without decompressing to flat bitsets.

## Why

Filtered vector search at scale requires checking billions of candidate IDs against a set of valid IDs. Flat bitsets waste memory and bandwidth when the valid set is sparse. CPU Roaring bitmaps (CRoaring) compress well but live on the wrong side of the PCIe bus.

cu-roaring-filter brings Roaring bitmaps to the GPU:

- **6-59x memory compression** vs flat bitsets (depending on set density)
- **17x faster filter construction** than CPU CRoaring for multi-predicate queries at 1B scale
- **Direct kernel integration** — `contains()` and `warp_contains()` callable from any CUDA kernel
- **Sub-millisecond set operations** (AND/OR/ANDNOT/XOR) on GPU-resident bitmaps

## Quick Start

```bash
git clone --recursive https://github.com/maxwbuckley/cu-roaring-filter.git
cd cu-roaring-filter
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)
ctest --output-on-failure   # 34 tests
```

Requires: CUDA 12.4+, CMake 3.25+, GCC 13+ (or any C++17 compiler).

## Core API

### Upload a CRoaring bitmap to GPU

```cpp
#include <cu_roaring/cu_roaring.cuh>
#include <roaring/roaring.h>

// Build bitmap on CPU (or load from storage)
roaring_bitmap_t* cpu_bm = roaring_bitmap_create();
roaring_bitmap_add_range(cpu_bm, 0, 500000);

// Transfer to GPU in compressed SoA format
cu_roaring::GpuRoaring gpu_bm = cu_roaring::upload(cpu_bm, stream);
cu_roaring::build_key_bloom(gpu_bm, stream);  // optional Bloom filter

roaring_bitmap_free(cpu_bm);  // CPU copy no longer needed
```

### Build directly from sorted IDs (no CRoaring dependency)

```cpp
#include <cu_roaring/detail/upload_ids.cuh>

std::vector<uint32_t> pass_ids = {0, 5, 12, 99, 1042, ...};
auto gpu_bm = cu_roaring::upload_from_sorted_ids(
    pass_ids.data(), pass_ids.size(), universe_size, stream);
```

### Set operations on GPU

```cpp
#include <cu_roaring/detail/set_ops.cuh>

// Single operations
auto intersection = cu_roaring::set_operation(a, b, cu_roaring::SetOp::AND, stream);
auto union_       = cu_roaring::set_operation(a, b, cu_roaring::SetOp::OR, stream);
auto difference   = cu_roaring::set_operation(a, b, cu_roaring::SetOp::ANDNOT, stream);
auto sym_diff     = cu_roaring::set_operation(a, b, cu_roaring::SetOp::XOR, stream);

// Multi-way operations
cu_roaring::GpuRoaring bitmaps[] = {color_red, price_lt_50, in_stock};
auto result = cu_roaring::multi_and(bitmaps, 3, stream);

cu_roaring::gpu_roaring_free(intersection);
cu_roaring::gpu_roaring_free(result);
```

### Point queries inside CUDA kernels

```cpp
#include <cu_roaring/device/roaring_view.cuh>
#include <cu_roaring/device/roaring_warp_query.cuh>
#include <cu_roaring/device/make_view.cuh>

auto view = cu_roaring::make_view(gpu_bm);  // lightweight, trivially copyable

// Per-thread query (any kernel)
__global__ void my_kernel(cu_roaring::GpuRoaringView view, ...) {
    uint32_t candidate_id = ...;
    if (view.contains(candidate_id)) {
        // candidate passes filter
    }
}

// Warp-cooperative query (amortizes binary search across 32 lanes)
__global__ void my_kernel_warp(cu_roaring::GpuRoaringView view, ...) {
    uint32_t candidate_id = ...;
    if (cu_roaring::warp_contains(view, candidate_id)) {
        // candidate passes filter
    }
}
```

### Decompress to flat bitset (when needed)

```cpp
#include <cu_roaring/detail/decompress.cuh>

// Allocating version
uint32_t* flat = cu_roaring::decompress_to_bitset(gpu_bm, stream);
cudaFree(flat);

// Into pre-allocated buffer
uint32_t n_words = (universe_size + 31) / 32;
cu_roaring::decompress_to_bitset(gpu_bm, my_buffer, n_words, stream);
```

## Integration with NVIDIA cuVS (CAGRA)

cu-roaring-filter integrates natively with CAGRA's filtered search. The Roaring bitmap is queried directly during graph traversal — no decompression step.

```cpp
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

// Build filter from arbitrary predicates
auto color_bm   = cu_roaring::upload(color_red_bitmap, stream);
auto price_bm   = cu_roaring::upload(price_lt_50_bitmap, stream);
auto filter_bm   = cu_roaring::set_operation(color_bm, price_bm, cu_roaring::SetOp::AND, stream);
cu_roaring::build_key_bloom(filter_bm, stream);

auto view = cu_roaring::make_view(filter_bm);
auto filter = cuvs::neighbors::filtering::roaring_filter_warp(
    view, cardinality, n_rows);

// Search — roaring membership checks happen inside CAGRA's graph traversal kernel
cuvs::neighbors::cagra::search(res, params, index, queries, neighbors, distances, filter);
```

## Benchmark Results

All benchmarks on NVIDIA RTX 5090 (170 SMs, 32 GB). Median of 30 iterations with 10 warmup.

### Filter Construction (1B universe, Zipfian tag distribution)

| Predicates | CPU CRoaring | GPU Roaring | Speedup |
|-----------|-------------|-------------|---------|
| 1 | 46.7 ms | 2.2 ms | **21x** |
| 4 | 180.6 ms | 10.4 ms | **17x** |
| 8 | 180.1 ms | 15.5 ms | **12x** |

### CAGRA Filtered Search (1M vectors, 128-dim, k=10, batch=100)

| Pass Rate | cuVS bitset | roaring_warp | Speedup | Recall |
|-----------|------------|-------------|---------|--------|
| 50% | 0.981 ms | 0.751 ms | **1.31x** | 98.2% |
| 10% | 1.602 ms | 1.327 ms | **1.21x** | 96.2% |
| 1% | 3.596 ms | 3.584 ms | 1.00x | 97.2% |

### Throughput (1M vectors, batch=10K)

| Filter | QPS |
|--------|-----|
| No filter | 2.6M |
| Bitset | 922K |
| Roaring warp | **960K** |

### Memory Compression

| Set Density | Flat Bitset | Roaring | Compression |
|------------|------------|---------|-------------|
| 0.1% (rare) | 125 MB | 2.1 MB | **59x** |
| 1% (uncommon) | 125 MB | 20 MB | **6.2x** |
| 50% (common) | 125 MB | ~125 MB | 1x |

### Decompress Kernel Throughput

| Universe Size | Latency | Bandwidth |
|--------------|---------|-----------|
| 1B | 0.23 ms | 538 GB/s |
| 100M | 0.018 ms | 713 GB/s |

## Architecture

### GPU Memory Layout

Roaring bitmaps are stored in Structure-of-Arrays (SoA) format for coalesced GPU memory access:

```
GpuRoaring
├── keys[]           uint16_t   sorted container keys (high 16 bits)
├── types[]          uint8_t    ARRAY=0, BITMAP=1, RUN=2
├── offsets[]        uint32_t   byte offset into per-type data pool
├── cardinalities[]  uint16_t   elements per container
├── bitmap_data[]    uint64_t   bitmap containers (1024 words each)
├── array_data[]     uint16_t   sorted array containers
├── run_data[]       uint16_t   run containers (start, length pairs)
└── key_bloom[]      uint32_t   optional Bloom filter (8 KB)
```

### Container Types

Following the CRoaring format, each 16-bit key range uses the most compact representation:

- **Array**: up to 4096 elements stored as sorted `uint16_t` values
- **Bitmap**: 4097+ elements stored as 1024-word (8 KB) bitset
- **Run**: consecutive ranges stored as (start, length) pairs

### Query Strategies

**`contains(id)`** — Per-thread point query:
1. Extract high-16 key, check Bloom filter (if present)
2. Binary search over sorted keys array
3. Container-type-specific membership test

**`warp_contains(id)`** — Warp-cooperative query:
1. `__match_any_sync` groups threads with the same high-16 key
2. Leader thread performs one binary search
3. `__shfl_sync` broadcasts container index to group
4. Each thread does its own low-16-bit membership test

The warp variant reduces binary searches by up to 32x when neighboring threads query IDs in the same container (common in graph traversal).

## Project Structure

```
cu-roaring-filter/
├── include/cu_roaring/
│   ├── cu_roaring.cuh              umbrella header
│   ├── types.cuh                   GpuRoaring, enums
│   ├── detail/
│   │   ├── upload.cuh              CPU CRoaring → GPU
│   │   ├── upload_ids.cuh          sorted IDs → GPU (no CRoaring dep)
│   │   ├── decompress.cuh          GPU → flat bitset
│   │   ├── set_ops.cuh             AND/OR/ANDNOT/XOR
│   │   └── utils.cuh               CUDA_CHECK, helpers
│   └── device/
│       ├── roaring_view.cuh        device-side contains()
│       ├── roaring_warp_query.cuh  warp-cooperative contains()
│       └── make_view.cuh           GpuRoaring → GpuRoaringView
├── src/                            implementation (.cpp, .cu)
├── test/                           34 Google Tests
├── bench/                          benchmarks (Google Benchmark + standalone)
├── third_party/CRoaring/           git submodule
└── results/                        benchmark outputs + figures
```

## Comparison with Other Approaches

| | cu-roaring-filter | Flat Bitset | VecFlow (label-centric IVF) |
|---|---|---|---|
| **Filter type** | Any boolean predicate | Any boolean predicate | Pre-indexed labels only |
| **Memory** | 6-59x compressed | Baseline | 3.5-10.8x overhead |
| **Set ops on GPU** | AND/OR/ANDNOT/XOR | Bitwise only | N/A (built into index) |
| **Kernel integration** | `contains()` / `warp_contains()` | Bit test | N/A (custom kernels) |
| **Ad-hoc queries** | Yes | Yes | No (requires re-indexing) |
| **Peak filtered QPS** | 960K (CAGRA, 1M, batch=10K) | 922K | 5M (custom index, 10M) |
| **Construction** | Upload + set ops (~10ms for 4 preds @ 1B) | Trivial | Index build (minutes) |

## License

Apache 2.0
