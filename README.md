# cu-roaring-filter

GPU-accelerated Roaring Bitmap library for NVIDIA CUDA. Enables compressed set membership queries, boolean set operations, and integration with GPU vector search kernels — all without decompressing to flat bitsets.

## Why

Filtered vector search at scale requires checking billions of candidate IDs against a set of valid IDs. Flat bitsets waste memory and bandwidth when the valid set is sparse. CPU Roaring bitmaps (CRoaring) compress well but live on the wrong side of the PCIe bus.

cu-roaring-filter brings Roaring bitmaps to the GPU:

- **6-59x memory compression** vs flat bitsets (depending on set density)
- **66x faster filter upload** at 100M IDs via GPU-native sort/partition pipeline
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
ctest --output-on-failure   # 6 test suites
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

// Transfer to GPU — PROMOTE_AUTO (default) queries the GPU's L2 cache size
// and automatically selects the optimal container format:
//   - Small universe (bitset fits in L2): keeps compressed arrays (saves memory)
//   - Large universe (bitset exceeds L2): promotes all to bitmap (faster queries)
cu_roaring::GpuRoaring gpu_bm = cu_roaring::upload(cpu_bm, stream);

// Manual overrides if you know your workload:
auto compressed = cu_roaring::upload(cpu_bm, stream, cu_roaring::PROMOTE_NONE);  // min memory
auto fast       = cu_roaring::upload(cpu_bm, stream, cu_roaring::PROMOTE_ALL);   // max query speed

roaring_bitmap_free(cpu_bm);  // CPU copy no longer needed
```

### Build from ID array (no CRoaring dependency)

```cpp
#include <cu_roaring/detail/upload_ids.cuh>

// IDs can be unsorted and contain duplicates — handled internally
std::vector<uint32_t> pass_ids = {1042, 5, 99, 12, 0, 5, ...};
auto gpu_bm = cu_roaring::upload_from_ids(
    pass_ids.data(), pass_ids.size(), universe_size, stream);

// Override the cache-aware default if needed:
auto fast_bm = cu_roaring::upload_from_ids(
    pass_ids.data(), pass_ids.size(), universe_size, stream,
    cu_roaring::PROMOTE_ALL);  // force all-bitmap for max query speed
```

### Build from flat bitset (fastest path)

```cpp
#include <cu_roaring/detail/upload_ids.cuh>

// From a host-side bitset (uint32_t words, bit i = words[i/32] >> (i%32) & 1)
uint32_t n_words = (universe_size + 31) / 32;
auto gpu_bm = cu_roaring::upload_from_bitset(
    bitset_words, n_words, universe_size, stream);

// From a device-side bitset (already on GPU — zero-copy, no H→D transfer)
auto gpu_bm = cu_roaring::upload_from_device_bitset(
    d_bitset, n_words, universe_size, stream);
```

This is the fastest upload path: the bitset words map directly to Roaring bitmap containers with no sort, dedupe, or scatter. Just popcount each 65536-bit chunk, compact non-empty chunks, and build metadata. Complement optimization applies automatically when density > 50%.

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

### Container promotion (automatic and manual)

Array containers require a binary search inside the container, which is 3-8x slower than bitmap containers at scale. The library automatically handles this via **`PROMOTE_AUTO`** (the default), which queries the GPU's L2 cache size:

- **Small universe** (<=~4M, <=64 containers): keeps compressed arrays — structure fits in cache, queries are fast
- **Larger universe** (>~4M, >64 containers): promotes all containers to bitmap — eliminates the 4-10x array binary search overhead

```cpp
#include <cu_roaring/detail/promote.cuh>

// Check what the auto policy chose for your data:
uint32_t threshold = cu_roaring::resolve_auto_threshold(universe_size);
// Returns PROMOTE_ALL (0) at 10M+ scale, PROMOTE_NONE (4096) at 1M scale

// Post-upload cache-aware promotion:
auto optimized = cu_roaring::promote_auto(gpu_bm, stream);

// Or force all-bitmap manually:
auto promoted = cu_roaring::promote_to_bitmap(gpu_bm, stream);

// Or set a custom threshold: containers with >256 elements become bitmap
auto custom = cu_roaring::upload(cpu_bm, stream, 256);
```

### Complement optimization (automatic)

Roaring compression is asymmetric: a 1% density filter compresses 59x, but a 99% density filter gets ~1x (no savings). The complement optimization fixes this by transparently storing the *complement* (the rejects) when density exceeds 50%, then flipping the `contains()` result at query time.

This makes compression **symmetric around 50%**: a 99% pass-rate filter stores only the 1% rejects, achieving the same 59x compression.

| Pass Rate | Without Complement | With Complement | Stored Density |
|-----------|-------------------|-----------------|---------------|
| 1% | 59x | 59x | 1% |
| 10% | 6.2x | 6.2x | 10% |
| 50% | ~1x | ~1x | 50% |
| 70% | ~1x | **3.3x** | 30% |
| 90% | ~1x | **6.2x** | 10% |
| 99% | ~1x | **59x** | 1% |

**Automatic**: `upload_from_ids()` detects density > 50% and stores the complement on-GPU (no additional host transfers). The complement is computed via a GPU-native gap-expansion pipeline using CUB.

```cpp
// Automatic: >50% density triggers complement storage
auto gpu_bm = cu_roaring::upload_from_ids(
    pass_ids.data(), pass_ids.size(), universe_size, stream);

// gpu_bm.negated == true if complement was stored
// contains() and warp_contains() handle this transparently
```

**From CRoaring**: Use the `universe_size` overload to enable complement detection:

```cpp
// With explicit universe_size: enables auto-complement
auto gpu_bm = cu_roaring::upload(cpu_bm, universe_size, stream);
```

**Set operations**: DeMorgan's laws are applied automatically. `AND(~A, B)` becomes `ANDNOT(B, A)`, etc. All 16 combinations of `{AND,OR,ANDNOT,XOR} × {negated,normal}` inputs are handled correctly.

**Decompression**: `decompress_to_bitset()` inverts the output when `negated == true`, producing the correct logical bitset.

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

### Enumerate IDs / CSR export

`enumerate_ids` exports all set element IDs as a sorted `int64_t` array on the GPU in a single kernel launch. This is the natural output format for CSR column indices and avoids the round-trip through a flat bitset when downstream consumers need explicit ID lists.

```cpp
#include <cu_roaring/detail/to_csr.cuh>

// Auto-allocating: returns a device pointer the caller must cudaFree
int64_t* ids = cu_roaring::enumerate_ids(bitmap, stream);
// ids now contains bitmap.total_cardinality sorted int64_t values
// Use as CSR column indices with trivial indptr construction

// Or with a pre-allocated buffer:
int64_t* output;  // pre-allocated with at least total_cardinality elements
cu_roaring::enumerate_ids(bitmap, output, stream);
```

**Per-container extraction strategy:**

- **Array containers**: Direct copy. The stored `uint16_t` values are already sorted, so each element is widened to `int64_t` with the container's 16-bit key prefix OR'd in. Work is O(cardinality).
- **Bitmap containers**: Block-cooperative bit extraction. Each block (256 threads) processes one container's 1024 `uint64_t` words (4 words per thread). A shared-memory prefix sum of per-thread popcounts determines each thread's write offset, ensuring sorted output without post-sort. Work is O(65536) per container regardless of density (popcount + scan all words), but actual writes are O(cardinality).
- **Run containers**: Run expansion via shared-memory prefix sum on run lengths. Each (start, length) pair is expanded to individual IDs at the correct sorted position.
- **Absent containers**: Skipped entirely (zero work, zero output).

**Motivation**: This enables a fused roaring-to-SDDMM path in cuVS brute-force search. Instead of decompressing to a flat bitset (which requires scanning all N bits) and then converting bitset-to-CSR (another full scan), `enumerate_ids` produces the CSR column indices directly from the compressed representation. This bypasses the bitset intermediate entirely and reduces the kernel launch count from 7 to 4 in the brute-force filtered search pipeline.

## Integration with NVIDIA cuVS (CAGRA)

cu-roaring-filter integrates natively with CAGRA's filtered search. The Roaring bitmap is queried directly during graph traversal — no decompression step.

```cpp
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/roaring_filter.cuh>

// Build filter from arbitrary predicates
auto color_bm  = cu_roaring::upload(color_red_bitmap, stream);
auto price_bm  = cu_roaring::upload(price_lt_50_bitmap, stream);
auto filter_bm = cu_roaring::set_operation(color_bm, price_bm, cu_roaring::SetOp::AND, stream);

// One line: GpuRoaring → ready-to-search filter
auto filter = cuvs::neighbors::filtering::roaring_filter(filter_bm);
cuvs::neighbors::cagra::search(res, params, index, queries, neighbors, distances, filter);
```

The `roaring_filter` constructor takes a `GpuRoaring` directly — no `make_view()`, no cardinality tracking, no choosing between filter variants. It uses warp-cooperative queries internally (always faster for CAGRA's graph traversal pattern).

## Benchmark Results

All benchmarks on NVIDIA RTX 5090 (170 SMs, 32 GB). Median of 50 iterations with 10 warmup.

### Filter Construction (1B universe, Zipfian tag distribution)

| Predicates | CPU CRoaring | GPU Roaring | Speedup |
|-----------|-------------|-------------|---------|
| 1 | 46.7 ms | 2.2 ms | **21x** |
| 4 | 180.6 ms | 10.4 ms | **17x** |
| 8 | 180.1 ms | 15.5 ms | **12x** |

### Upload Latency at Scale (B8)

`upload_from_ids()` uses a fully GPU-native pipeline for large inputs: CUB RadixSort + Unique + on-device container construction. Data never returns to the host after the initial upload.

| IDs | CPU-only | GPU-native | Speedup |
|-----|---------|-----------|---------|
| 100K | 5.3 ms | 0.9 ms | **6x** |
| 1M | 57 ms | 4.7 ms | **11x** |
| 10M | 629 ms | 11 ms | **52x** |
| 100M | 7,464 ms | 99 ms | **66x** |

At search engine scale (100M IDs), filter construction takes **99 ms** — fast enough to rebuild on every query.

### Point Query Throughput (10M random queries)

Standalone point query benchmark comparing `contains()`, `warp_contains()`, and flat bitset across universe sizes and set densities.

All results with `PROMOTE_AUTO` + O(1) direct-map key index. Random access pattern (realistic for search).

| Universe | Density | Bitset | `contains()` | `warp_contains()` | vs Bitset | Compression |
|----------|---------|--------|-------------|-------------------|-----------|-------------|
| 1M | 0.1% | 156 Gq/s | **214 Gq/s** | 190 Gq/s | **1.4x faster** | 59.7x |
| 10M | 1% | 101 Gq/s | 99 Gq/s | 99 Gq/s | **1.0x** | **6.2x** |
| 100M | 1% | 102 Gq/s | 95 Gq/s | 96 Gq/s | 0.9x | **6.2x** |
| 1B | 0.1% | 15 Gq/s | 15 Gq/s | 15 Gq/s | **1.0x** | **58.4x** |
| 1B | 1% | 15 Gq/s | 15 Gq/s | **25 Gq/s** | **1.6x faster** | **6.2x** |
| 1B | 10% | 15 Gq/s | 15 Gq/s | 15 Gq/s | 1.0x | 1.0x |

Key findings:
- **Query speed at parity or better for random access.** Across random/strided patterns, roaring matches bitset within 10% at all scales. At 1M/0.1% it's 1.4x faster; at 1B/1% it's 1.6x faster.
- **Memory compression is the value proposition.** At 1B/0.1%, roaring uses 2.1 MB vs bitset's 125 MB (58.4x) — same query speed, 59x less memory.
- **Clustered access patterns favor bitset** (1.2-2.3x faster) due to cache-line prefetching. CAGRA's graph traversal pattern is closer to random/strided than clustered.
- **`warp_contains()` wins at 1B/1%.** The 125 MB flat bitset thrashes L2 cache while roaring's `__ldg`-cached reads achieve 1.6x higher throughput.

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

### enumerate_ids (CSR Export) (B9)

`enumerate_ids()` extracts sorted element IDs directly from the compressed Roaring bitmap, producing CSR column indices without decompressing to a flat bitset intermediate. Compared against the two-step alternative: `decompress_to_bitset()` + bitset-to-CSR scan (popcount + CUB prefix sum + bit extraction).

| Universe | Density | Cardinality | enumerate_ids | Baseline (decomp+scan) | Speedup |
|----------|---------|-------------|--------------|----------------------|---------|
| 1M | 0.1% | 975 | 0.567 ms | 0.557 ms | 1.0x |
| 10M | 0.1% | 10K | 0.533 ms | 0.604 ms | **1.1x** |
| 10M | 50% | 5M | 0.588 ms | 0.633 ms | **1.1x** |
| 100M | 1% | 1M | 0.549 ms | 0.590 ms | **1.1x** |
| 100M | 10% | 10M | 0.624 ms | 0.683 ms | **1.1x** |
| **1B** | **0.1%** | **1M** | **0.692 ms** | **3.089 ms** | **4.5x** |
| **1B** | **1%** | **10M** | **0.726 ms** | **3.103 ms** | **4.3x** |
| **1B** | **10%** | **100M** | **3.590 ms** | **3.829 ms** | **1.1x** |

Key findings:
- **4-4.5x faster at 1B scale with sparse filters** (0.1-1% density). The baseline must scan the full 119 MB bitset regardless of how many bits are set; `enumerate_ids` processes only non-empty containers.
- **At parity or slightly faster at all other scales.** The device-side prefix sum (CUB `ExclusiveSum`) avoids the host roundtrip overhead that would otherwise dominate at small scales.
- **Crossover to write-bound at high density.** At 100M/50% (50M output IDs), the baseline's bitset-to-CSR kernel distributes writes across more thread blocks, achieving better write parallelism.

### CAGRA Filtered Search (1M vectors, 128-dim, k=10, batch=100)

| Pass Rate | cuVS bitset | roaring_warp | Speedup | Agreement* |
|-----------|------------|-------------|---------|------------|
| 50% | 0.981 ms | 0.751 ms | **1.31x** | 98.2% |
| 10% | 1.602 ms | 1.327 ms | **1.21x** | 96.2% |
| 1% | 3.596 ms | 3.584 ms | 1.00x | 97.2% |

*\*Result agreement between roaring and bitset filters. Both produce approximate results (CAGRA is ANN). The 2-4% difference reflects different graph traversal paths, not missing valid neighbors.*

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
| **Cache-aware auto promotion** | Query GPU L2 cache size, promote when bitset would thrash cache | **7.4x** (100M/1%) | `PROMOTE_AUTO` (default), `promote_auto()`, or manual `PROMOTE_ALL` |
| **`__ldg()` + warp broadcast** | Read-only cache for global loads; leader broadcasts metadata | **1.6x** (1B/10%) | Applied in library (`roaring_view.cuh`, `roaring_warp_query.cuh`) |
| **Direct-map key index** | Replace O(log n) key binary search with O(1) table lookup | **1.75x** (1B/10%) | Prototyped in `bench_optimized_query.cu` |

### Recommendations for cuVS Integration

1. **Default (PROMOTE_AUTO)**: Just call `upload()` — the library queries your GPU's L2 cache size and picks the optimal strategy. At 1B scale it promotes to all-bitmap (faster queries); at 1M it keeps compressed arrays (saves memory). No tuning required.

2. **Cold filters** (stored in GPU memory, used for set operations): Use `PROMOTE_NONE` explicitly. The 59x memory savings lets you store far more concurrent filters in GPU memory. Promote only the final combined result before search.

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
└── run_data[]       uint16_t   run containers (start, length pairs)
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
│   │   ├── upload_ids.cuh          IDs → GPU (GPU-native sort/partition)
│   │   ├── decompress.cuh          GPU → flat bitset
│   │   ├── set_ops.cuh             AND/OR/ANDNOT/XOR
│   │   ├── promote.cuh             array/run → bitmap promotion
│   │   ├── to_csr.cuh             enumerate_ids() → sorted int64_t export
│   │   └── utils.cuh               CUDA_CHECK, helpers
│   └── device/
│       ├── roaring_view.cuh        device-side contains() with __ldg
│       ├── roaring_warp_query.cuh  warp-cooperative contains() with broadcast
│       └── make_view.cuh           GpuRoaring → GpuRoaringView
├── src/                            implementation (.cpp, .cu)
├── test/                           6 test suites (Google Test)
├── bench/                          benchmarks (B1-B9)
│   ├── bench_comprehensive.cu      B1/B3/B4/B5: construction, memory, E2E
│   ├── bench_point_query.cu        B6: point query throughput
│   ├── bench_optimized_query.cu    B7: optimization analysis
│   ├── bench_upload_scale.cu       B8: upload latency at XL scale
│   ├── bench_enumerate_ids.cu      B9: enumerate_ids / CSR export
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
| **Upload (100M IDs)** | **99 ms** (GPU-native) | N/A (pre-allocated) | Minutes (index build) |
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
./bench/bench_upload_scale          # B8: upload latency at XL scale
./bench/bench_enumerate_ids         # B9: enumerate_ids / CSR export
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
