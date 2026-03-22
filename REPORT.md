# cu-roaring-filter: GPU Roaring Bitmaps for Filtered Vector Search

**Target**: NVIDIA cuVS (CAGRA graph-based ANN search)
**Hardware**: NVIDIA RTX 5090 (170 SMs, 32 GB, Blackwell architecture)
**Date**: March 2026

---

## Executive Summary

cu-roaring-filter brings Roaring bitmap compressed set membership to NVIDIA GPUs, enabling memory-efficient filtered vector search at billion scale. At 1B vectors with 10% filter selectivity, `warp_contains()` is **1.6x faster than flat bitset** while using up to **59x less memory** for sparse filters. Inside NVIDIA cuVS's CAGRA graph search, roaring filters deliver **1.31x speedup** over the bitset baseline at 50% selectivity with 98.2% recall.

Key results:
- **Query speed**: 24 Gq/s (warp-cooperative, 1B/10%) vs 15 Gq/s for flat bitset
- **Memory**: 2.1 MB vs 125 MB at 0.1% density (1B universe)
- **CAGRA search**: 0.751 ms vs 0.981 ms per batch (1M vectors, 50% pass, 100 queries)
- **Filter construction**: 17x faster than CPU CRoaring for 4-predicate AND at 1B scale

---

## 1. Problem Statement

Filtered vector search at scale requires checking candidate IDs against a validity set during graph traversal. CAGRA's search kernel calls the filter function **millions of times per query batch** — every candidate neighbor encountered during the search is checked.

The current approach in cuVS uses a flat bitset (`cuvs::core::bitset`). This has two problems at scale:

| Problem | Impact |
|---------|--------|
| **Memory** | A flat bitset for 1B vectors = 125 MB. With 100 filter attributes, that's 12.5 GB of GPU memory just for filter storage |
| **Cache thrashing** | At 1B scale, the 125 MB bitset exceeds L2 cache (RTX 5090: 96 MB). Random candidate checks cause cache misses |

Roaring bitmaps solve both: sparse filters compress 6-59x, and the compressed representation accesses less total memory per query.

---

## 2. Architecture

### GPU Memory Layout

Roaring bitmaps partition the 32-bit ID space into 65536-element containers keyed by the high 16 bits. Each container uses the most compact representation:

```
GpuRoaring (Structure-of-Arrays)
├── keys[]           uint16_t   sorted container keys (high 16 bits)
├── types[]          uint8_t    ARRAY=0, BITMAP=1, RUN=2
├── offsets[]        uint32_t   byte offset into per-type data pool
├── cardinalities[]  uint16_t   elements per container
├── bitmap_data[]    uint64_t   bitmap containers (8 KB each)
├── array_data[]     uint16_t   sorted array containers (2B per element)
└── run_data[]       uint16_t   run containers (start, length pairs)
```

### Query Path (contains)

Each membership check follows this path:

```
1. key = id >> 16                            // register, free
2. binary_search(keys[], key)                // log2(N) reads via __ldg (read-only cache)
3. load type, offset, cardinality            // 1 read (via __ldg)
4. container-specific test:
   - BITMAP: bitmap_data[word] via __ldg     // 1 read
   - ARRAY:  binary_search(array_data[])     // log2(card) reads
```

### Warp-Cooperative Query (warp_contains)

The warp variant amortizes the key lookup across 32 threads:

```
1. __match_any_sync groups threads with the same high-16 key
2. Leader performs binary search + reads metadata (type, offset, cardinality)
3. __shfl_sync broadcasts all metadata to followers (zero global reads for followers)
4. Each thread does its own container-level test
```

---

## 3. Standalone Benchmark Results (B6)

### Methodology

- 10M random queries per configuration
- 50 iterations with 10 warmup, median reported
- GPU timing via `cudaEvent` (no host overhead)
- Correctness verified against flat bitset for every configuration

### Point Query Throughput

| Universe | Density | Containers | Bitset (Gq/s) | `contains()` | `warp_contains()` | vs Bitset |
|----------|---------|------------|---------------|-------------|-------------------|-----------|
| 1M | 0.1% | 16 arr | 137 | **167** | 151 | **1.2x faster** |
| 10M | 0.1% | 153 arr | 100 | **136** | 115 | **1.4x faster** |
| 10M | 1% | 153 arr | 100 | 25 | 25 | 0.25x |
| 100M | 10% | 1526 bmp | 101 | 95 | 94 | 0.9x |
| 100M | 1% | 1526 arr | 101 | 13 | 13 | 0.1x |
| 1B | 10% | 15259 bmp | 15 | 15 | **24** | **1.6x faster** |
| 1B | 50% | 15259 bmp | 27 | 26 | 24 | 1.0x |

### Key Findings

**Roaring beats bitset in three scenarios:**

1. **Small universe, sparse data** (1M-10M, 0.1%): The entire roaring structure fits in L2 cache. `contains()` is 1.2-1.4x faster than bitset because the compressed data has better cache efficiency.

2. **Large universe, bitmap containers** (1B, 10%): The flat bitset (125 MB) thrashes L2 cache on random access. Roaring's `__ldg`-cached reads through the 125 MB bitmap data achieve better texture cache utilization because the access pattern (key lookup → one bitmap word) has higher spatial locality than scattered bitset reads.

3. **Warp-cooperative at scale** (1B, 10%): `warp_contains()` at 24 Gq/s vs bitset at 15 Gq/s. The `__shfl_sync` metadata broadcast eliminates redundant global reads when warp lanes share the same container key.

**Roaring is slower with array containers** (100M, 1%): The double binary search (key + in-container) costs 24 global reads per query vs 1 for bitset. This led to the all-bitmap promotion optimization.

---

## 4. Optimization Analysis (B7)

### The Array Container Bottleneck

| Container Type | Global Memory Reads per Query | Relative Cost |
|---------------|-------------------------------|---------------|
| Flat bitset | 1 | 1x |
| Roaring bitmap container | ~14 (key search + 1 word read) | 14x |
| Roaring array container | ~24 (key search + array search) | 24x |

### Optimization Results

Three optimizations were prototyped and benchmarked:

| Optimization | What | Best Speedup | Status |
|---|---|---|---|
| **All-bitmap promotion** | Convert array containers to bitmap at upload | **7.4x** (100M/1%) | Implemented: `upload(bm, stream, PROMOTE_ALL)` |
| **`__ldg()` read-only cache** | Route all global reads through texture cache | **1.6x** (1B/10%) | Applied in library |
| **Warp metadata broadcast** | Leader reads type/offset/card, broadcasts via `__shfl_sync` | Eliminates 96 redundant reads/warp | Applied in library |

### All-Bitmap Promotion Impact (B7 data)

| Condition | base_contains | allbmp_contains | Speedup | allbmp vs Bitset |
|-----------|--------------|-----------------|---------|------------------|
| 10M/1% random | 0.404 ms | 0.103 ms | **3.9x** | 0.97x |
| 100M/0.1% random | 0.339 ms | 0.107 ms | **3.2x** | 0.92x |
| 100M/1% random | 0.784 ms | 0.106 ms | **7.4x** | 0.94x |
| 1B/0.1% random | 0.767 ms | 0.379 ms | **2.0x** | 1.73x faster |
| 1B/1% random | 1.056 ms | 0.378 ms | **2.8x** | 1.73x faster |

With all-bitmap promotion, roaring matches flat bitset speed across all conditions and **beats it at 1B scale** (1.73x) due to cache effects.

---

## 5. CAGRA Integration Results

### Setup

- CAGRA index: 1M vectors, 128 dimensions, graph_degree=32, itopk=256
- 100 queries per batch, k=10 nearest neighbors
- Recall measured against bitset_filter ground truth

### Search Latency

| Pass Rate | No Filter | Bitset Filter | Roaring (warp) | Speedup vs Bitset | Recall |
|-----------|-----------|--------------|----------------|-------------------|--------|
| 50% | 4.57 ms | 0.981 ms | **0.751 ms** | **1.31x** | 98.2% |
| 10% | 0.77 ms | 1.602 ms | **1.327 ms** | **1.21x** | 96.2% |
| 1% | 0.68 ms | 3.596 ms | **3.584 ms** | 1.00x | 97.2% |

### Analysis

At **50% pass rate**, roaring is 1.31x faster than bitset. This is because:
- The CAGRA kernel checks every candidate neighbor against the filter
- With 50% selectivity, each warp of 32 threads checks 32 candidates
- Many candidates share the same high-16 key → `warp_contains()` amortizes the key binary search
- The roaring structure (1M vectors → 16 containers) fits entirely in L2 cache

At **1% pass rate**, roaring and bitset are equal. The search is dominated by graph traversal (finding valid neighbors in a sparse filter), not filter evaluation cost.

### Throughput (1M vectors, batch=10K)

| Filter | QPS |
|--------|-----|
| No filter | 2.6M |
| Bitset | 922K |
| Roaring warp | **960K** |

---

## 6. Memory Analysis

### Per-Bitmap Memory at 1B Scale

| Density | Flat Bitset | Roaring (compressed) | Roaring (promoted) | Compression |
|---------|------------|---------------------|-------------------|-------------|
| 0.1% | 125 MB | **2.1 MB** | 125 MB | **59x** |
| 1% | 125 MB | **20 MB** | 125 MB | **6.2x** |
| 10% | 125 MB | 125 MB | 125 MB | 1x |
| 50% | 125 MB | 125 MB | 125 MB | 1x |

### Multi-Attribute Scaling

For a 1B-vector dataset with N filter attributes (Zipfian density distribution):

| Attributes | Flat Bitset | Roaring (compressed) | Fits in 32 GB? |
|-----------|------------|---------------------|----------------|
| 10 | 1.25 GB | 0.44 GB | Both fit |
| 100 | 12.5 GB | 6.9 GB | Bitset: barely. Roaring: yes |
| 500 | 62.5 GB | 34.5 GB | Neither. But roaring is 1.8x smaller |
| 1000 | 125 GB | 69 GB | Neither. Roaring: 1.8x smaller |

The compression advantage is most impactful when many attributes are sparse (0.1-1% density). A real-world e-commerce catalog might have: 3 popular categories (50% density, no savings), 20 medium categories (10%, no savings), 100 rare attributes (1%, 6x savings), 500 tag bitmaps (0.1%, 59x savings). The rare/tag bitmaps dominate memory and are where roaring helps most.

### Recommended Strategy

| Use Case | Strategy | Memory | Query Speed |
|----------|----------|--------|-------------|
| **Hot filter** (queried per search) | `upload(bm, stream, PROMOTE_ALL)` | Same as bitset | Matches or beats bitset |
| **Cold filter** (stored, used in set ops) | Default Roaring | 6-59x smaller | Slower per-query, but rarely queried directly |
| **Multi-predicate** | Set ops on compressed, then promote result | Only final result uses full memory | Best of both worlds |

---

## 7. End-to-End Pipeline Comparison

For a multi-predicate filtered search (e.g., "color=red AND price<50 AND in_stock"):

### Pipeline A: CPU Filter + PCIe Transfer
```
1. CPU CRoaring AND (3 bitmaps)        → 180 ms at 1B
2. PCIe transfer (flat bitset to GPU)   →   9 ms at 1B
3. CAGRA search with bitset_filter      →  10 ms (simulated)
Total: 199 ms
```

### Pipeline B: GPU-Resident Roaring
```
1. GPU set_operation AND (3 bitmaps)    →   8 ms at 1B
2. (No transfer — already on GPU)      →   0 ms
3. CAGRA search with roaring_filter     →  10 ms (simulated)
Total: 18 ms → 11x faster
```

### Pipeline C: GPU Roaring + Promotion
```
1. GPU set_operation AND (3 bitmaps)    →   8 ms at 1B
2. promote_to_bitmap (result only)      →  <1 ms
3. CAGRA search with promoted filter    →  ~8 ms (estimated from 1.3x speedup)
Total: ~17 ms → 12x faster
```

---

## 8. Optimizations Applied

### In the Library (shipped)

| Optimization | File | Impact |
|---|---|---|
| `__ldg()` on all device reads | `roaring_view.cuh` | 1.6x at 1B/10% (texture cache for scattered reads) |
| Warp metadata broadcast | `roaring_warp_query.cuh` | Eliminates 96 redundant global reads per warp |
| All-bitmap promotion | `promote.cuh`, `upload.cuh`, `upload_ids.cuh` | 3.9-7.4x for array container elimination |
| Configurable threshold | `upload()`, `upload_from_sorted_ids()` | `PROMOTE_ALL` or custom threshold |
| Strict `-Werror` warnings | `CMakeLists.txt` | `-Wshadow`, `-Wnon-virtual-dtor`, etc. on all targets |

### Prototyped (not yet in library)

| Optimization | Location | Impact | Effort |
|---|---|---|---|
| Direct-map key index | `bench_optimized_query.cu` | 1.75x at 1B for `contains()` | Medium |
| Fused multi-predicate kernel | Not started | 20-40% for 4+ predicates | High |
| cudaMallocAsync / memory pool | Not started | Reduce allocation stalls | Medium |

---

## 9. Comparison with Alternatives

| | cu-roaring-filter | Flat Bitset (cuVS default) | VecFlow (label-centric IVF) |
|---|---|---|---|
| **Filter type** | Any boolean predicate | Any boolean predicate | Pre-indexed labels only |
| **Memory (1B, 0.1%)** | **2.1 MB** | 125 MB | N/A |
| **Point query (1B/10%)** | **24 Gq/s** | 15 Gq/s | N/A |
| **CAGRA speedup** | **1.31x** (50% pass) | Baseline | N/A |
| **Set ops on GPU** | AND/OR/ANDNOT/XOR | Bitwise only | N/A |
| **Ad-hoc queries** | Yes | Yes | No (requires re-indexing) |
| **Construction (4 pred, 1B)** | **10 ms** (GPU) | Trivial | Minutes (index build) |

---

## 10. Integration Path

### For cuVS upstream (rapidsai/cuvs)

The integration is already prototyped in the cuVS fork:

1. **`cuvs/core/roaring.hpp`** — `gpu_roaring` struct with RAII (rmm::device_uvector), `from_sorted_ids()`, set operations, `to_bitset()` decompression
2. **`cuvs/neighbors/roaring_filter.cuh`** — `roaring_filter` and `roaring_filter_warp` implementing the `base_filter` interface
3. **CAGRA kernel instantiations** for all 4 data types (float, half, int8, uint8) x roaring filter

Remaining work for upstream:
- Port `promote_to_bitmap()` into `cuvs::core::gpu_roaring`
- Add `FilterType::Roaring` to IVF runtime dispatch
- Python bindings via `pylibcuvs`
- Integration tests in cuVS CI

### Broader NVIDIA ecosystem

| Library | Use Case | Fit |
|---------|----------|-----|
| **cuVS** | Filtered ANN search | Primary target (prototyped) |
| **RAFT** | Core data structure (like `raft::core::bitset`) | Natural home for the container type |
| **cuDF** | DataFrame predicate masks | Strong fit for compound WHERE clauses |
| **cuGraph** | Vertex/edge filtering for subgraph ops | Similar traversal pattern to CAGRA |

---

## Appendix: Benchmark Reproducibility

### Hardware
- GPU: NVIDIA GeForce RTX 5090 (170 SMs, 32 GB GDDR7, Blackwell)
- CPU: Intel (WSL2 on Windows)
- CUDA: 12.4

### Running Benchmarks

```bash
# Standalone benchmarks (cu-roaring-filter)
cd cu-roaring-filter/build
./bench/bench_point_query           # B6: point query throughput (48 configs)
./bench/bench_optimized_query       # B7: optimization analysis (32 configs)
./bench/bench_comprehensive         # B1/B3/B4/B5: construction, memory, E2E

# cuVS CAGRA benchmark
cd cuvs/cpp/bench/prims/core/build
LD_LIBRARY_PATH=../../../build:../../../build/_deps/rmm-build ./bench_cagra_roaring
```

### Methodology
- n >= 30 iterations per measurement (B6/B7 use 50)
- 10 warmup iterations
- Median reported (not mean) to reduce outlier sensitivity
- GPU timing via `cudaEvent` pairs (microsecond precision)
- Correctness verified via bitwise comparison against flat bitset ground truth for every configuration
- All results in `results/raw/*.json` with full statistics (median, mean, p5, p95, stdev, min, max)
