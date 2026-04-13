# GPU-Accelerated Roaring Bitmaps for Filtered Approximate Nearest Neighbor Search

## Abstract

We present cu-roaring-bitmap, a GPU-native implementation of Roaring bitmaps optimized for filtered approximate nearest neighbor (ANN) search. Filtered ANN search requires checking billions of candidate vectors against validity predicates during graph traversal — a memory-bound operation where flat bitsets waste GPU memory and thrash caches at scale. Our system brings Roaring bitmaps to the GPU with a series of architecture-aware optimizations: a 2-read query path using direct-map key indices, cache-aware container promotion that adapts to the GPU's L2 cache size, warp-cooperative queries that amortize metadata lookups across SIMT lanes, a fully device-resident upload pipeline that eliminates host round-trips, a complement-bitmap representation that makes compression symmetric around 50% density, a fused negation-aware multi-predicate kernel, a direct-to-CSR `enumerate_ids` export path, and stream-ordered allocation that removes ~19 implicit device syncs per set operation. On an NVIDIA RTX 5090, cu-roaring-bitmap achieves query speed parity with flat bitsets while providing 6-59x memory compression for sparse *or* dense filters (via the complement path). At 1B vectors, warp-cooperative queries are 1.7x faster than flat bitsets due to superior cache utilization. Multi-predicate filter construction (8-way AND at 1B scale) runs in 5.9 ms — 39x faster than CPU CRoaring and 3.5x faster than a naive pairwise GPU approach. Integration with NVIDIA cuVS's CAGRA graph search delivers 1.31x search speedup at 50% filter selectivity with 98.2% result agreement.

## 1. Introduction

Approximate nearest neighbor (ANN) search is a core operation in recommendation systems, retrieval-augmented generation (RAG), and similarity search engines. Modern ANN algorithms like HNSW and CAGRA build proximity graphs over the vector dataset and traverse them to find neighbors. Filtered search adds a constraint: only vectors matching a boolean predicate (e.g., "in stock AND price < $50 AND category = electronics") should be returned.

The dominant approach to filtered search is post-filtering: the search algorithm traverses the graph normally, checking each candidate against a filter bitmap. This requires a membership test for every candidate visited — millions per query batch. The current state of the art in GPU-accelerated vector search (NVIDIA cuVS) uses flat bitsets for this filter, where each bit represents one vector. This approach is simple and fast (one memory read per test) but has two scaling problems:

**Memory**: A flat bitset for N vectors requires N/8 bytes. At 1 billion vectors, that is 125 MB per filter. A search engine with 100 filterable attributes requires 12.5 GB of GPU memory just for filter storage — a significant fraction of available GPU memory (32 GB on an RTX 5090).

**Cache pressure**: At 1B vectors, the 125 MB bitset exceeds the GPU's L2 cache (96 MB on RTX 5090). Random candidate checks during graph traversal cause L2 cache misses, degrading throughput. Sparse filters (e.g., 0.1% selectivity) waste 99.9% of the bitset memory on zero bits that are never useful, yet they still occupy cache lines.

Roaring bitmaps [1] are a compressed bitmap format widely used in databases and search engines on CPU. They partition the 32-bit ID space into 65536-element containers, each stored in the most compact format: sorted arrays for sparse containers, 8 KB bitsets for dense containers, and run-length encoding for consecutive ranges. This achieves 6-59x compression for typical filter distributions while supporting efficient boolean operations (AND/OR/ANDNOT).

However, porting Roaring bitmaps to GPU is non-trivial. The data structure relies on binary search for key lookup and container-type dispatch — both sources of warp divergence and memory access irregularity on SIMT architectures. Prior work on compressed bitmaps on GPU is limited, and no existing system integrates compressed bitmaps directly into GPU graph search kernels.

We present cu-roaring-bitmap, a GPU-native Roaring bitmap library designed for filtered ANN search. Our contributions are:

1. **A GPU memory layout and query path** for Roaring bitmaps that achieves 2 global memory reads per membership test (vs 1 for flat bitsets), with automatic hardware-aware optimization that selects the best container format based on the GPU's cache hierarchy.

2. **A fully device-resident construction pipeline** that sorts, deduplicates, partitions, and builds bitmap containers entirely on GPU using CUB primitives, achieving 66x speedup over CPU construction at 100M IDs.

3. **A fused, negation-aware multi-predicate kernel** for N-way boolean AND that processes all predicates in a single pass, applies DeMorgan transforms to handle negated inputs without materializing intermediate complements, and avoids the allocation storms and synchronization overhead of pairwise approaches (3-6x speedup).

4. **A complement-bitmap representation** that makes Roaring compression symmetric around 50% density: a 99% pass-rate filter is stored as the 1% rejects, achieving the same 59x compression that sparse filters enjoy. Complement state is tracked on the bitmap, propagated through set operations via DeMorgan's laws, and inverted at query / decompression time, so the logical semantics are invisible to the caller.

5. **A direct roaring-to-CSR export** (`enumerate_ids`) that emits sorted `int64_t` column indices in a single kernel launch, bypassing the flat-bitset intermediate used by prior brute-force filtered search pipelines — 4–4.5x faster than decompress+scan at 1B scale with sparse filters.

6. **Stream-ordered allocation throughout** using `cudaMallocAsync` + `cudaFreeAsync` with a tuned CUDA memory pool. This eliminates the ~19 implicit device-wide synchronizations previously triggered per `set_operation` call by temporary-buffer `cudaFree`s, yielding 1.9–3.3x end-to-end speedups on hot paths with no external RMM dependency.

7. **Integration with NVIDIA cuVS (CAGRA)** demonstrating 1.31x search speedup at 50% selectivity with near-identical result quality, and up to 59x memory reduction for sparse filters (measured: 2.1 MB vs 125 MB flat bitset at 1B universe, 0.1% density).

## 2. Background

### 2.1 Roaring Bitmap Format

A Roaring bitmap [1] partitions the 32-bit integer universe into chunks of 2^16 = 65536 values, identified by the high 16 bits (the "key"). Each chunk is stored as a "container" using one of three formats:

- **Array container**: For sparse chunks (≤4096 set bits), stores sorted 16-bit values. Membership test: binary search, O(log n).
- **Bitmap container**: For dense chunks (>4096 set bits), stores a 1024-word (8 KB) bitset. Membership test: single word read, O(1).
- **Run container**: For consecutive ranges, stores (start, length) pairs. Membership test: binary search over runs, O(log n).

The container type is chosen to minimize memory: arrays use 2 bytes per element (efficient below 4096), bitmaps use a fixed 8 KB (efficient above 4096).

### 2.2 GPU Architecture Considerations

NVIDIA GPUs execute threads in 32-thread warps. Optimal performance requires:

- **Coalesced memory access**: Threads in a warp should access adjacent memory addresses.
- **Minimal divergence**: Threads in a warp should follow the same execution path.
- **Cache utilization**: Data structures should fit in or efficiently use the L2 cache and texture cache.

Roaring bitmaps present challenges on all three fronts: the binary search over the key array produces scattered, data-dependent accesses; the type-based dispatch causes warp divergence when containers have mixed types; and the overall structure is larger than a flat bitset in some configurations.

### 2.3 CAGRA Graph Search

CAGRA [2] is a GPU-accelerated graph-based ANN algorithm in NVIDIA cuVS. It builds a k-NN graph over the dataset and searches it by traversing edges from random entry points, maintaining a priority queue of closest candidates. During traversal, each candidate is checked against a filter function — this is the operation cu-roaring-bitmap optimizes.

CAGRA's filter interface is a template parameter: any callable that maps (query_idx, sample_idx) → bool can be used. The filter is evaluated inside the search kernel by every thread in the traversal warp.

## 3. System Design

### 3.1 GPU Memory Layout

We store Roaring bitmaps in Structure-of-Arrays (SoA) format on GPU:

```
GpuRoaring
├── keys[]           uint16_t   sorted container keys
├── types[]          uint8_t    container type tag
├── offsets[]        uint32_t   byte offset into data pool
├── cardinalities[]  uint16_t   elements per container
├── bitmap_data[]    uint64_t   bitmap container pool
├── array_data[]     uint16_t   array container pool
├── run_data[]       uint16_t   run container pool
└── key_index[]      uint16_t   direct-map key → container index
```

The `key_index` is a direct-mapped lookup table: `key_index[high16_bits] = container_index`, with 0xFFFF as a sentinel for absent keys. This replaces O(log n) binary search with O(1) table lookup at a cost of (max_key + 1) × 2 bytes (30 KB at 1B universe, 0.3 KB at 10M).

### 3.2 Query Path Optimization

We developed the query path through iterative profiling, reducing global memory reads from 17 (initial implementation) to 2 (final):

| Version | Key Lookup | Metadata | Container Test | Total Reads |
|---------|-----------|----------|---------------|-------------|
| v1: Binary search | log2(n) ≈ 14 reads | 3 reads (type, offset, card) | 1 read | 17 |
| v2: + `__ldg` texture cache | 14 reads (cached) | 3 reads (cached) | 1 read | 17 (faster) |
| v3: + Direct-map key index | 1 read | 3 reads | 1 read | 5 |
| v4: + All-bitmap fast path | 1 read | 0 reads (implicit offset) | 1 read | **2** |

The all-bitmap fast path is enabled when all containers are bitmap type and a key_index is present. The offset is computed arithmetically as `container_idx × 1024` rather than loaded from the offsets array, and the type dispatch is eliminated entirely.

### 3.3 Warp-Cooperative Query

For warp-cooperative queries (`warp_contains`), we exploit the observation that in graph traversal, neighboring candidates often share the same high-16 key prefix:

1. `__match_any_sync` identifies threads in the warp querying the same key.
2. The lowest-numbered matching thread (the "leader") performs the key_index lookup.
3. `__shfl_sync` broadcasts the container index to all matching threads.
4. Each thread independently reads its bitmap word using the implicit offset.

This amortizes the key lookup across up to 32 threads. In the all-bitmap fast path, followers do zero global reads for the lookup — they receive the container index via register-to-register shuffle.

### 3.4 Cache-Aware Container Promotion

CRoaring's default container threshold (4096 elements) was designed for CPU cache hierarchies. On GPU, array containers are problematic because their binary search produces 10-12 data-dependent global reads per query.

We implement automatic container promotion (`PROMOTE_AUTO`): when the universe exceeds ~4M IDs (>64 potential containers), all containers are promoted to bitmap format at upload time. This eliminates array containers from the query path entirely, enabling the 2-read fast path.

The threshold was determined empirically: below 64 containers, array containers fit in L1/L2 cache and their binary search is tolerable. Above 64 containers, the key search depth (7+ steps) compounds with the array search (10+ steps) to produce 4-10x slowdown vs flat bitset.

### 3.5 Device-Resident Construction Pipeline

Filter construction from a set of passing IDs follows a fully GPU-resident pipeline:

1. **H→D transfer**: Copy unsorted IDs to GPU (single PCIe transfer).
2. **CUB DeviceRadixSort**: GPU-parallel radix sort.
3. **CUB DeviceSelect::Unique**: Remove duplicates.
4. **Key extraction kernel**: Compute high-16 key for each sorted ID.
5. **CUB DeviceRunLengthEncode**: Find container boundaries.
6. **CUB DeviceScan::ExclusiveSum**: Prefix sum for scatter offsets.
7. **Scatter kernel**: Build 8 KB bitmap containers (one block per container).
8. **Metadata kernel**: Write keys, types, offsets, cardinalities.
9. **Key index kernel**: Build direct-map lookup table.

Data never returns to the host after the initial transfer. At 100M IDs, this pipeline runs in 99 ms — 66x faster than CPU sort + upload (7.5 seconds).

### 3.6 Fused, Negation-Aware Multi-Predicate AND

Multi-predicate filter construction (e.g., AND-ing N attribute bitmaps) uses a fused single-pass kernel when all inputs are all-bitmap format:

1. Download key arrays from all N inputs (small: N × n_containers × 2 bytes).
2. CPU multi-way intersection of sorted key arrays → set of common keys.
3. Single kernel launch: for each common key, AND the 1024 bitmap words across all N inputs. One thread block per output container.

This replaces the previous pairwise chain (N-1 separate `set_operation` calls, each with device-to-host downloads, CPU container matching, ~10 cudaMalloc calls, and stream synchronization).

Each input is tagged with a `negated` flag (see §3.7). The fused kernel inverts the per-word contribution of any negated input directly inside the reduction (`word ^= ~0ull`), so negated operands never materialize a complement container. All 16 `{AND, OR, ANDNOT, XOR} × {negated, normal}²` combinations reduce, via DeMorgan, to a single AND/OR reduction over (optionally inverted) words plus a final complement on the output — no branching in the inner loop.

### 3.7 Complement Bitmap Optimization

Standard Roaring compression is asymmetric: a 1% density filter compresses 59x, but a 99% density filter gets ~1x because all of its containers are dense bitmaps. The complement optimization fixes this at the data-structure level.

A `GpuRoaring` carries a single `negated` flag. When `upload_from_ids` (or `upload`) sees a density above 50%, it stores the *complement* of the input set — the rejected elements — and sets `negated = true`. The complement is computed on-device via a gap-expansion pipeline: CUB `ExclusiveSum` of a per-ID "present" mask yields gap offsets, and a scatter kernel writes the missing IDs. No host round-trip is required.

At query time, `contains()` and `warp_contains()` XOR their result with `negated`, so callers observe the original set. Set operations propagate `negated` through DeMorgan's laws before dispatching to the container kernels — for example, `AND(~A, B)` rewrites to `ANDNOT(B, A)`, and `AND(~A, ~B)` rewrites to `~OR(A, B)`. All 16 combinations of signs and operators are handled in a single `resolve_negation` helper. `decompress_to_bitset` inverts the output when `negated == true`. The optimization is therefore transparent to users while delivering symmetric compression: a 99% pass-rate filter stores 1% density and achieves the same 59x compression.

### 3.8 Direct CSR Export (`enumerate_ids`)

Prior GPU brute-force filtered search pipelines materialize a flat bitset from the filter, then run a separate bitset-to-CSR kernel (popcount + prefix sum + bit extraction) to obtain column indices. Both steps scan the full N-bit universe regardless of how many bits are set.

`enumerate_ids` emits sorted `int64_t` IDs directly from the compressed representation in a single kernel launch. Each non-empty container writes its elements at an offset determined by a device-side `ExclusiveSum` over cardinalities:

- **Array containers** are a direct copy: stored `uint16_t` values are already sorted, so each element is widened to `int64_t` with the container's 16-bit key OR'd in. Work is O(cardinality).
- **Bitmap containers** use a block-cooperative bit extraction: 256 threads process one container's 1024 `uint64_t` words (4 words per thread), a shared-memory prefix sum of per-thread popcounts determines each thread's write offset, and threads emit IDs in sorted order without a post-sort.
- **Run containers** expand via a shared-memory prefix sum on run lengths.
- **Absent containers** are skipped entirely.

At 1B scale with 0.1–1% density, this is 4–4.5x faster than the decompress-then-scan baseline, which must touch the full 119 MB bitset regardless of selectivity. It also reduces the brute-force filtered search pipeline from seven kernel launches to four.

### 3.9 Stream-Ordered Allocation and Memory Pool Tuning

Profiling a single `set_operation` call on large inputs revealed that temporary buffer cleanup was the dominant cost: each `cudaFree` forced a device-wide synchronization, and a dense AND over 1526 containers issued roughly 19 such syncs per call. We migrated every internal allocation to `cudaMallocAsync` / `cudaFreeAsync` using the CUDA 12+ stream-ordered memory pool. The library additionally provides `gpu_roaring_free_async` for use inside hot loops — `multi_and` and `multi_or` use it internally when walking their pairwise reduction tree.

Two configurations are possible. With the default pool (release threshold 0) freed memory may be returned to the OS; with `cudaMemPoolAttrReleaseThreshold = UINT64_MAX` the pool retains freed blocks for reuse, giving behavior equivalent to RMM's `pool_memory_resource` without the dependency. Section 4.7 shows that the tuned pool yields 1.9–3.3x speedups on `set_operation` and `upload_from_ids` at scale, with tail-latency variance dropping nearly 5x at 100M IDs.

## 4. Evaluation

### 4.1 Experimental Setup

- **GPU**: NVIDIA GeForce RTX 5090 (170 SMs, 32 GB GDDR7, 96 MB L2 cache, Blackwell architecture)
- **CPU**: Intel (WSL2 on Windows)
- **CUDA**: 12.4
- **Methodology**: n ≥ 30 iterations, 10 warmup, median reported, GPU timing via cudaEvent pairs, correctness verified against flat bitset ground truth

### 4.2 Point Query Throughput

10M random queries per configuration, `PROMOTE_AUTO` default:

| Universe | Density | Bitset (Gq/s) | cu-roaring (Gq/s) | vs Bitset | Compression |
|----------|---------|---------------|-------------------|-----------|-------------|
| 1M | 0.1% | 159 | **206** (`contains`) | **1.3x faster** | 59.7x |
| 10M | 1% | 101 | 100 (`contains`) | 1.0x | 6.2x |
| 100M | 1% | 102 | 98 (`warp`) | 1.0x | 6.2x |
| 1B | 0.1% | 15 | 15 (`contains`) | 1.0x | 58.4x |
| 1B | 1% | 15 | **25** (`warp`) | **1.6x faster** | 6.2x |
| 1B | 10% | 15 | **26** (`warp`) | **1.7x faster** | 1.0x |

At 1B scale, the flat bitset (125 MB) exceeds L2 cache and suffers random cache misses. Cu-roaring's `__ldg`-cached 2-read path achieves higher effective bandwidth because the key_index (30 KB) and the accessed bitmap words (8 KB per container) are individually cacheable.

### 4.3 Filter Construction

Upload from unsorted IDs (GPU-native pipeline):

| IDs | CPU-only | cu-roaring | Speedup |
|-----|---------|-----------|---------|
| 1M | 57 ms | 4.7 ms | **11x** |
| 10M | 629 ms | 11 ms | **52x** |
| 100M | 7,464 ms | 99 ms | **66x** |

### 4.4 Multi-Predicate AND

Fused vs pairwise, Zipfian density distribution:

| Universe | Predicates | CPU CRoaring | Pairwise GPU | Fused GPU | Fused vs CPU |
|----------|-----------|-------------|-------------|-----------|-------------|
| 100M | 4 | 15.1 ms | 3.8 ms | **1.2 ms** | **12.6x** |
| 100M | 8 | 17.4 ms | 7.2 ms | **1.5 ms** | **11.6x** |
| 1B | 4 | 197 ms | 11.3 ms | **4.3 ms** | **45.7x** |
| 1B | 8 | 228 ms | 20.5 ms | **5.9 ms** | **39x** |

### 4.5 CAGRA Filtered Search

1M vectors, 128 dimensions, k=10, batch=100 queries:

| Pass Rate | Bitset Filter | cu-roaring Filter | Speedup | Result Agreement |
|-----------|--------------|-------------------|---------|-----------------|
| 50% | 0.981 ms | **0.751 ms** | **1.31x** | 98.2% |
| 10% | 1.602 ms | **1.327 ms** | **1.21x** | 96.2% |
| 1% | 3.596 ms | 3.584 ms | 1.00x | 97.2% |

Result agreement is measured as the fraction of k=10 results identical between the two filter implementations. Both produce approximate results (CAGRA is an ANN algorithm); the 2-4% difference reflects different graph traversal paths, not missing valid neighbors.

### 4.6 Direct CSR Export (enumerate_ids)

`enumerate_ids` vs the two-step baseline (`decompress_to_bitset` + bitset-to-CSR), measured on the same hardware:

| Universe | Density | Cardinality | `enumerate_ids` | Baseline | Speedup |
|----------|---------|-------------|-----------------|----------|---------|
| 10M | 0.1% | 10K | 0.533 ms | 0.604 ms | 1.1x |
| 100M | 1% | 1M | 0.549 ms | 0.590 ms | 1.1x |
| **1B** | **0.1%** | **1M** | **0.692 ms** | **3.089 ms** | **4.5x** |
| **1B** | **1%** | **10M** | **0.726 ms** | **3.103 ms** | **4.3x** |
| 1B | 10% | 100M | 3.590 ms | 3.829 ms | 1.1x |

At 1B scale the baseline must scan the full 119 MB bitset regardless of selectivity; `enumerate_ids` processes only non-empty containers and is write-bound only at high density.

### 4.7 Stream-Ordered Allocation Impact

Comparing two pool configurations: default (`release_threshold = 0`) vs tuned (`release_threshold = UINT64_MAX`). 50 iterations, 10 warmup.

**`set_operation`**:

| Condition | Containers | Default | Tuned | Speedup |
|-----------|-----------|---------|-------|---------|
| dense AND 100M | 1526 | 1477 µs | 448 µs | **3.30x** |
| dense AND 1M | 16 | 478 µs | 331 µs | 1.44x |
| mixed OR 10M | 153 | 1452 µs | 1377 µs | 1.05x |

**`upload_from_ids`**:

| Condition | IDs | Default | Tuned | Speedup |
|-----------|-----|---------|-------|---------|
| 1M 10% | 100K | 1309 µs | 466 µs | **2.81x** |
| 10M 10% | 1M | 1699 µs | 613 µs | **2.77x** |
| 100M 10% | 10M | 8675 µs | 3684 µs | **2.35x** |

Beyond the mean speedup, tail-latency variance drops sharply: `upload_from_ids` at 100M goes from σ = 1058 µs to σ = 225 µs (4.7x tighter). Compared to the original synchronous `cudaMalloc` / `cudaFree` baseline the improvement is larger still, since every legacy `cudaFree` forced a device sync.

### 4.8 Memory Savings at Scale

For a 1B-vector dataset with 100 filterable attributes (Zipfian density distribution):

| | Flat Bitset | cu-roaring |
|---|---|---|
| Total memory | 12.5 GB | ~0.7 GB |
| Fits in 32 GB GPU | Barely | Comfortably |
| Per-sparse-attribute (0.1%) | 125 MB | 2.1 MB |

## 5. Design Decisions and Lessons Learned

### 5.1 Bloom Filter Removal

We initially implemented an optional Bloom filter over container keys for early rejection. Benchmarking across 48 configurations showed it added 5-17% overhead with zero benefit in any configuration. The Bloom check (2 hash computations + 2 global reads) is redundant when the direct-map key_index already provides O(1) rejection. We removed it entirely.

**Lesson**: On GPU, the cost floor for any check (even a Bloom filter) is one global memory read. When the primary lookup is already O(1), adding a probabilistic pre-check adds latency without saving work.

### 5.2 Container Promotion vs Compression

CRoaring's array containers (≤4096 elements as sorted arrays) are optimal for CPU where binary search over cached L1 data is fast. On GPU, the same binary search becomes 10-12 data-dependent global reads with pipeline stalls. Promoting all containers to bitmap format increases memory by up to 4x per container but reduces query reads from 24 to 2.

We chose to promote by default (`PROMOTE_AUTO`) rather than preserve CRoaring's container choices, because the query path is exercised millions of times per search batch while the memory cost is paid once.

**Lesson**: The optimal data structure for GPU is not the same as for CPU, even for the same logical format. Hardware-aware adaptation of container thresholds is more valuable than faithful format preservation.

### 5.3 Eliminating Host Round-Trips

The initial GPU sort implementation uploaded IDs to GPU, sorted them, downloaded sorted IDs back to host for partitioning, then re-uploaded the roaring structure. This D→H→H→D round-trip cost 530 ms at 100M IDs — more than the sort itself. Implementing the entire construction pipeline on-device (sort, dedup, partition, bitmap scatter, metadata build) eliminated the round-trip and reduced total time from 627 ms to 99 ms.

**Lesson**: The PCIe bus is the bottleneck for GPU data structures. Any operation that can be done on-device should be, even if it requires custom CUDA kernels for what would be trivial CPU code.

### 5.4 The 2-Read Insight

The progression from 17 reads to 2 reads per query was not planned — it emerged from iterative profiling. The key insight was that `PROMOTE_AUTO` guarantees all-bitmap containers at scale, which makes the type field, offset field, and cardinality field redundant: the type is always BITMAP, the offset is `idx × 1024`, and the cardinality is unused for bitmap membership tests. Recognizing and exploiting this invariant halved the remaining read count.

**Lesson**: Optimizations compound. The cache-aware promotion policy (designed for memory) enabled the 2-read fast path (designed for latency), which was not possible with mixed container types.

## 6. Related Work

A full treatment of related work — covering compressed bitmap indexes, GPU database column scans, filtered ANN systems, label-partitioned graph indexes, and hybrid analytical/search engines — is provided in `analysis/related_work.md`. We summarize the most directly comparable systems here.

**CPU compressed bitmaps.** CRoaring [1] is the reference Roaring implementation and powers bitmap indexes in Lucene, Druid, Pinot, ClickHouse, and many other analytical engines. EWAH [3] and Concise [4] are earlier word-aligned compressed formats. All target CPU cache hierarchies; none have been ported to GPU in a form usable inside a search kernel.

**GPU bitmap operations.** BitWeaving [5] introduced GPU-parallel column scans using flat bitmaps packed vertically for SIMD efficiency. RAPIDS cuDF uses flat validity bitmasks for DataFrame filtering. Both assume dense bitsets and do not benefit from container-level compression.

**GPU-accelerated ANN search.** NVIDIA cuVS [2] provides CAGRA (graph-based), IVF-PQ, and IVF-Flat, all of which use flat `cuvs::core::bitset` filters at the search-kernel boundary. Faiss [6] provides CPU and GPU ANN implementations with bitset-based filtering. Our work is the first to replace the bitset filter interface with a compressed Roaring representation inside the search kernel itself.

**Filtered ANN.** VecFlow [7] builds per-label sub-indices for pre-filtering; it achieves very high throughput on in-distribution label predicates but requires label pre-indexing and cannot handle ad-hoc boolean combinations. FilteredDiskANN [8] uses label-partitioned graphs on CPU with similar trade-offs. NHQ [11] and related hybrid systems modify the graph structure to be filter-aware. Cu-roaring-bitmap is orthogonal: it is predicate-agnostic (any boolean predicate expressible as a bitmap) and optimizes the per-candidate filter evaluation along *any* search path rather than restructuring the index.

**Complement storage.** The complement-bitmap idea has CPU precedent in bitmap indexing work [9, 10] but, to our knowledge, has not previously been pushed through GPU Roaring set operations via automatic DeMorgan propagation.

To our knowledge, cu-roaring-bitmap is the first system to bring compressed bitmap data structures to GPU for vector search filtering, and the first to ship a GPU-native Roaring implementation with complement-aware set algebra, fused multi-predicate kernels, and direct CSR export.

## 7. Conclusion

Cu-roaring-bitmap demonstrates that Roaring bitmaps can be effectively adapted to GPU architecture for filtered ANN search. Through hardware-aware container promotion, direct-map key indices, warp-cooperative queries, and device-resident construction, we achieve query speed parity with flat bitsets while providing 6-59x memory compression. At 1B scale, the system beats flat bitsets on both speed (1.7x for point queries, 46x for multi-predicate construction) and memory (2.1 MB vs 125 MB for sparse filters).

The key insight is that faithful reproduction of a CPU data structure on GPU is suboptimal — the right approach is to adapt the format to GPU hardware (all-bitmap promotion, O(1) key index, 2-read fast path) while preserving the logical semantics (compressed set membership with boolean operations). The result is a system where the user writes three lines of code and the library transparently selects the optimal strategy for their GPU.

## References

[1] D. Lemire et al., "Roaring Bitmaps: Implementation of an Optimized Software Library," Software: Practice and Experience, 2018.

[2] C. Ootomo et al., "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs," IEEE BigData, 2024.

[3] D. Lemire et al., "Sorting Improves Word-Aligned Bitmap Indexes," Data & Knowledge Engineering, 2010.

[4] A. Colantonio and R. Di Pietro, "Concise: Compressed 'n' Composable Integer Set," Information Processing Letters, 2010.

[5] Y. Li and J. Patel, "BitWeaving: Fast Scans for Main Memory Data Processing," SIGMOD, 2013.

[6] J. Johnson et al., "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, 2019.

[7] A. Baranchuk et al., "VecFlow: Label-Centric ANN Search with High Throughput," arXiv, 2025.

[8] S. Gollapudi et al., "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters," WWW, 2023.
