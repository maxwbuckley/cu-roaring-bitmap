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
ctest --output-on-failure   # 5 test suites
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

// For hot filters queried millions of times (e.g., CAGRA graph traversal),
// promote all containers to bitmap for 3-8x faster point queries:
auto fast_bm = cu_roaring::upload(cpu_bm, stream, cu_roaring::PROMOTE_ALL);

roaring_bitmap_free(cpu_bm);  // CPU copy no longer needed
```

### Build directly from sorted IDs (no CRoaring dependency)

```cpp
#include <cu_roaring/detail/upload_ids.cuh>

std::vector<uint32_t> pass_ids = {0, 5, 12, 99, 1042, ...};
auto gpu_bm = cu_roaring::upload_from_sorted_ids(
    pass_ids.data(), pass_ids.size(), universe_size, stream);

// Or with all-bitmap promotion for fastest queries:
auto fast_bm = cu_roaring::upload_from_sorted_ids(
    pass_ids.data(), pass_ids.size(), universe_size, stream,
    cu_roaring::PROMOTE_ALL);
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

### Promote containers to bitmap (faster queries)

Array containers require a binary search inside the container, which is 3-8x slower than bitmap containers at scale. For filters that will be queried many times (e.g., during CAGRA graph traversal), promote all containers to bitmap:

```cpp
#include <cu_roaring/detail/promote.cuh>

// Post-upload promotion (returns a new GpuRoaring; original is not modified)
auto promoted = cu_roaring::promote_to_bitmap(gpu_bm, stream);
cu_roaring::gpu_roaring_free(gpu_bm);  // free the original
gpu_bm = promoted;

// Or use a custom threshold: containers with >256 elements become bitmap
auto custom = cu_roaring::upload(cpu_bm, stream, 256);
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

auto view = cu_roaring::make_view(filter_bm);
auto filter = cuvs::neighbors::filtering::roaring_filter_warp(
    view, cardinality, n_rows);

// Search — roaring membership checks happen inside CAGRA's graph traversal kernel
cuvs::neighbors::cagra::search(res, params, index, queries, neighbors, distances, filter);
```

## Benchmark Results

All benchmarks on NVIDIA RTX 5090 (170 SMs, 32 GB). Median of 50 iterations with 10 warmup.

### Filter Construction (1B universe, Zipfian tag distribution)

| Predicates | CPU CRoaring | GPU Roaring | Speedup |
|-----------|-------------|-------------|---------|
| 1 | 46.7 ms | 2.2 ms | **21x** |
| 4 | 180.6 ms | 10.4 ms | **17x** |
| 8 | 180.1 ms | 15.5 ms | **12x** |

### Point Query Throughput (10M random queries)

Standalone point query benchmark comparing `contains()`, `warp_contains()`, and flat bitset across universe sizes and set densities.

| Universe | Density | Containers | Bitset | `contains()` | `warp_contains()` | Roaring vs Bitset |
|----------|---------|------------|--------|-------------|-------------------|-------------------|
| 1M | 0.1% | 16 (all array) | 137 Gq/s | 167 Gq/s | 151 Gq/s | **1.2x faster** |
| 10M | 0.1% | 153 (all array) | 100 Gq/s | 136 Gq/s | 115 Gq/s | **1.4x faster** |
| 100M | 10% | 1526 (all bitmap) | 101 Gq/s | 95 Gq/s | 94 Gq/s | 0.9x |
| 100M | 1% | 1526 (all array) | 101 Gq/s | 13 Gq/s | 13 Gq/s | 0.1x |
| 1B | 10% | 15259 (all bitmap) | 15 Gq/s | 15 Gq/s | **24 Gq/s** | **1.6x faster** |
| 1B | 50% | 15259 (all bitmap) | 27 Gq/s | 26 Gq/s | 24 Gq/s | 1.0x |

Key findings:
- **Bitmap containers match or beat bitset speed** — at 1B/10%, `warp_contains()` is 1.6x faster than flat bitset because the 125 MB bitset thrashes L2 cache while Roaring accesses less data per query via the `__ldg` read-only cache path.
- **Array containers are slow** — the double binary search (key lookup + in-container search) causes 3-8x slowdown at 100M+ scale. See [Optimization Analysis](#optimization-analysis) for solutions.
- **Small universes favor Roaring** — at 1M (16 containers), the entire structure fits in L2 cache and `contains()` is faster than bitset.

### Memory Compression

| Set Density | Flat Bitset | Roaring | Compression |
|------------|------------|---------|-------------|
| 0.1% (rare) | 125 MB | 2.1 MB | **59x** |
| 1% (uncommon) | 125 MB | 20 MB | **6.2x** |
| 50% (common) | 125 MB | ~125 MB | 1x |

### End-to-End Latency (4 predicates, Zipfian density)

Compares two filter pipelines: (A) CPU filter + PCIe transfer vs (C) GPU-resident Roaring set ops.

| Universe | Pipeline A (CPU + PCIe + search) | Pipeline C (GPU set ops + search) | Speedup |
|----------|--------------------------------|-----------------------------------|---------|
| 10M | 10.9 ms | 12.1 ms | 0.9x |
| 100M | 20.7 ms | 12.4 ms | **1.7x** |
| 1B | 134.3 ms | 18.1 ms | **7.4x** |

GPU-resident filtering breaks even at ~50M and dominates at 100M+ where CPU filter construction and PCIe transfer become the bottleneck.

### Decompress Kernel Throughput

| Universe Size | Latency | Bandwidth |
|--------------|---------|-----------|
| 1B | 0.23 ms | 538 GB/s |
| 100M | 0.018 ms | 713 GB/s |

### CAGRA Filtered Search (1M vectors, 128-dim, k=10, batch=100)

| Pass Rate | cuVS bitset | roaring_warp | Speedup | Recall |
|-----------|------------|-------------|---------|--------|
| 50% | 0.981 ms | 0.751 ms | **1.31x** | 98.2% |
| 10% | 1.602 ms | 1.327 ms | **1.21x** | 96.2% |
| 1% | 3.596 ms | 3.584 ms | 1.00x | 97.2% |

## Optimization Analysis

Detailed benchmarks in `bench/bench_optimized_query.cu` (B7) measured three optimization strategies against the baseline. Results on 10M random queries, RTX 5090.

### The Array Container Problem

Array containers (used when a container has fewer than 4096 elements) require a binary search inside the container *on top of* the key binary search. This compounds to 18-28 global memory reads per query vs 1 for a flat bitset:

```
Flat bitset:   id → bitset[id/32]                    = 1 global read
Roaring array: id → key search (11 reads) →
                     type/offset/card (3 reads) →
                     array search (10 reads)          = 24 global reads
Roaring bitmap: id → key search (11 reads) →
                      type/offset (2 reads) →
                      bitmap word (1 read)            = 14 global reads
```

### Optimization Results

| Strategy | What it does | Best speedup | Status |
|----------|-------------|-------------|--------|
| **All-bitmap promotion** | Convert array containers to bitmap at upload time | **7.4x** (100M/1%) | `upload(bm, stream, PROMOTE_ALL)` or `promote_to_bitmap()` |
| **`__ldg()` + warp broadcast** | Read-only cache for global loads; leader broadcasts metadata | **1.6x** (1B/10%) | Applied in library (`roaring_view.cuh`, `roaring_warp_query.cuh`) |
| **Direct-map key index** | Replace O(log n) key binary search with O(1) table lookup | **1.75x** (1B/10%) | Prototyped in `bench_optimized_query.cu` |

### Recommendations for cuVS Integration

1. **Hot filters** (queried during CAGRA graph traversal): Promote all array containers to bitmap at upload time. The 3-8x query speedup justifies the extra memory. At 0.1% density this means using 125 MB instead of 2.1 MB — but you're querying this filter millions of times per search batch.

2. **Cold filters** (stored in GPU memory, used for set operations): Keep compressed Roaring. The 59x memory savings lets you store far more concurrent filters in GPU memory.

3. **Memory is the real value proposition** at scale. With 100 filter attributes at 1B scale: flat bitsets require 12.5 GB, Roaring requires ~0.7 GB for sparse attributes.

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
1. Extract high-16 key
2. Binary search over sorted keys array (via `__ldg` read-only cache)
3. Load container metadata (type, offset, cardinality) via `__ldg`
4. Container-type-specific membership test

**`warp_contains(id)`** — Warp-cooperative query:
1. `__match_any_sync` groups threads with the same high-16 key
2. Leader thread performs one binary search + reads metadata
3. `__shfl_sync` broadcasts container index, type, offset, and cardinality to group
4. Each thread does its own low-16-bit membership test

The warp variant reduces binary searches by up to 32x when neighboring threads query IDs in the same container (common in graph traversal). At 1B scale with bitmap containers, this makes `warp_contains()` 1.6x faster than a flat bitset.

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
│   │   ├── promote.cuh             array/run → bitmap promotion
│   │   └── utils.cuh               CUDA_CHECK, helpers
│   └── device/
│       ├── roaring_view.cuh        device-side contains() with __ldg
│       ├── roaring_warp_query.cuh  warp-cooperative contains() with broadcast
│       └── make_view.cuh           GpuRoaring → GpuRoaringView
├── src/                            implementation (.cpp, .cu)
├── test/                           5 test suites (Google Test)
├── bench/                          benchmarks (B1-B7)
│   ├── bench_comprehensive.cu      B1/B3/B4/B5: construction, memory, E2E
│   ├── bench_point_query.cu        B6: point query throughput
│   ├── bench_optimized_query.cu    B7: optimization analysis
│   ├── bench_set_ops.cu            set operation microbenchmarks
│   ├── bench_decompress.cu         decompression microbenchmarks
│   └── bench_transfer.cu           PCIe transfer comparison
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
| **Point query (1B/10%)** | **24 Gq/s** (warp) | 15 Gq/s | N/A |
| **Construction** | Upload + set ops (~10ms for 4 preds @ 1B) | Trivial | Index build (minutes) |

## Development

### Compiler Warnings

All targets (library, tests, benchmarks) compile with `-Werror` and a strict warning set. No warning suppressions are allowed — if it warns, it doesn't merge.

**CXX flags**: `-Wall -Wextra -Werror -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual -Wcast-align -Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation -Wnull-dereference -Wdouble-promotion -Wunused -Wpedantic`

**CUDA flags**: `--Werror all-warnings` with the same flags forwarded to the host compiler via `-Xcompiler`. A few flags (`-Wold-style-cast`, `-Wconversion`, `-Wpedantic`, `-Wdouble-promotion`) are excluded from the CUDA forwarding set because they fire in CUDA toolkit headers.

Warnings are applied per-target via `cu_roaring_target_strict_warnings()` defined in the root `CMakeLists.txt`. Third-party code (CRoaring, Google Test, Google Benchmark) is excluded.

### Running Benchmarks

```bash
cd build
./bench/bench_point_query           # B6: point query throughput
./bench/bench_optimized_query       # B7: optimization analysis
./bench/bench_comprehensive         # B1/B3/B4/B5: construction, memory, E2E
./bench/bench_set_ops               # set operation microbenchmarks
./bench/bench_decompress            # decompression microbenchmarks
./bench/bench_transfer              # PCIe transfer comparison
```

Results are written to `results/raw/*.json`. Generate figures with:

```bash
cd results
python3 plot_figures.py             # B1/B3/B4/B5 figures
python3 plot_point_query.py         # B6 figures + summary table
```

## License

Apache 2.0
