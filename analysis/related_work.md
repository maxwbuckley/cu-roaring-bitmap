# Related Work

Filtered approximate nearest neighbor (ANN) search sits at the intersection of
several research threads: compressed bitmap data structures, GPU-accelerated
vector search, and predicate-aware index designs. We survey each area and
position cu-roaring-bitmap relative to the landscape.

## 6.1 Compressed Bitmap Data Structures

Bitmap indices have a long history in database systems.  O'Neil and Quass
[O'Neil and Quass, SIGMOD 1997] introduced variant bitmap indices for data
warehousing workloads, demonstrating that read-mostly environments permit more
elaborate index structures than traditional B-trees.  Wu et al. [Wu et al., TODS
2006] proposed Word-Aligned Hybrid (WAH) compression, which aligns run-length
encoded segments to machine word boundaries so that boolean operations can
process one word per CPU cycle.  Lemire et al. [Lemire et al., Data & Knowledge
Engineering 2010] introduced Enhanced WAH (EWAH), which adds a marker-word
scheme that avoids decompressing dirty words unnecessarily, achieving better
throughput on skewed distributions.  Colantonio and Di Pietro [Colantonio and Di
Pietro, Information Processing Letters 2010] proposed Concise, an alternative
word-aligned scheme that reduces memory by up to 50% compared to WAH through
mixed fill/literal encoding, at the cost of slightly more complex iteration.

Roaring bitmaps [Chambi et al., Software: Practice and Experience 2016] departed
from run-length encoding entirely, instead partitioning the integer universe into
64K-element chunks and selecting among three container formats (sorted arrays,
uncompressed bitsets, and run-length encoded sequences) per chunk.  This
format-per-chunk adaptivity yields both superior compression and faster boolean
operations than WAH, EWAH, and Concise across a wide range of densities.
Roaring has been adopted in production systems including Apache Lucene, Apache
Spark, Apache Druid, and Apache Kylin.  The CRoaring library [Lemire et al.,
Software: Practice and Experience 2018] provides the reference C implementation
and serves as the CPU baseline in our evaluation.

All of these formats were designed for CPU execution.  Cu-roaring-bitmap is, to
our knowledge, the first implementation of Roaring bitmaps on GPU.  Our key
insight is that faithful reproduction of CPU container selection heuristics is
suboptimal on GPU: promoting all containers to bitmap format (Section 3.4)
sacrifices some compression but enables a 2-read query path that eliminates
branch divergence and data-dependent memory accesses.

## 6.2 GPU Bitmap Processing

There is limited prior work on GPU-accelerated compressed bitmap operations.
Andrzejewski and Wrembel [Andrzejewski and Wrembel, DEXA 2010] proposed
GPU-WAH, which parallelizes WAH compression and decompression on early CUDA
hardware.  Their approach maps fill words and literal words to separate GPU
threads for parallel boolean operations, demonstrating speedups over CPU WAH for
large bitmaps.  However, GPU-WAH operates on WAH-compressed bitmaps and targets
bulk compression/decompression rather than random-access membership queries.

Tran et al. [Tran et al., DASFAA 2020] improved GPU bitmap index query
processing with three enhancements: data structure reuse to reduce memory
allocation calls, metadata precomputation to skip intermediate processing steps,
and preallocated memory pools.  These optimizations achieved 33--113x speedup
over unenhanced GPU WAH implementations for range queries.  Like GPU-WAH, this
work targets WAH-compressed bitmaps for analytical column scans, not the
point-membership queries required by ANN graph traversal.

Cu-roaring-bitmap differs from both lines of work in three respects.  First, we
implement Roaring bitmaps rather than WAH, gaining the format's inherent
adaptivity.  Second, our query path is optimized for single-element membership
tests (the operation performed millions of times per ANN search batch), not bulk
boolean operations on entire bitmaps.  Third, we integrate directly into the CAGRA
search kernel via a filter functor, whereas prior GPU bitmap work operates as
standalone processing steps.

## 6.3 GPU-Accelerated Vector Search

Faiss [Johnson et al., IEEE Transactions on Big Data 2019] pioneered
billion-scale similarity search on GPU, introducing an optimized k-selection
algorithm that achieves 55% of theoretical peak GPU throughput.  Faiss provides
GPU implementations of IVF-Flat, IVF-PQ, and brute-force search, but its
filtering support is limited to flat bitset masks applied after distance
computation.

CAGRA [Ootomo et al., ICDE 2024] is a GPU-native graph-based ANN algorithm in
NVIDIA's cuVS library.  CAGRA constructs a pruned k-NN graph and searches it
via warp-parallel beam traversal, achieving 33--77x higher throughput than
CPU HNSW at equivalent recall.  CAGRA's filter interface accepts any callable
`(query_idx, sample_idx) -> bool`, evaluated inside the search kernel.
Cu-roaring-bitmap implements this interface, replacing CAGRA's default flat
bitset filter with a compressed Roaring bitmap that reduces memory by 6--59x
while matching or exceeding flat bitset query throughput.

VecFlow [Mo et al., SIGMOD 2025] is a GPU-native filtered search system that
adopts a label-centric IVF design: data points are partitioned by shared labels
rather than spatial proximity, with dual-structured posting lists for
high-specificity and low-specificity label groups.  VecFlow achieves 5 million
QPS at 90% recall on an NVIDIA A100, outperforming CPU-based Filtered-DiskANN by
up to 135x.  However, VecFlow requires labels to be known at index time and
builds per-label structures, making it a pre-filtering approach.  In contrast,
cu-roaring-bitmap is predicate-agnostic: any boolean predicate expressible as a
set of passing IDs can be uploaded at query time without rebuilding the index.
This makes our approach complementary to VecFlow -- one could use VecFlow's
label-centric indexing for high-frequency label predicates and cu-roaring-bitmap
for ad-hoc or compound predicates that were not anticipated at index time.

## 6.4 Filtered ANN: Label and Predicate-Aware Indices

A growing body of work modifies the ANN index structure itself to accelerate
filtered search.  These approaches can be categorized by the type of predicate
they target.

**Label-based filtering.**  Filtered-DiskANN [Gollapudi et al., WWW 2023]
introduced two algorithms, FilteredVamana and StitchedVamana, that build graph
edges considering both vector proximity and label overlap.  StitchedVamana
constructs per-label subgraphs and stitches them into a unified index, while
FilteredVamana directly incorporates label-aware edge selection during greedy
search.  Both approaches support conjunctive-OR label predicates natively but
cannot handle arbitrary boolean expressions or predicates unknown at index time.

ACORN [Patel et al., SIGMOD 2024] takes a predicate-agnostic approach on the
CPU side: it modifies HNSW construction to increase graph connectivity so that
the subgraph induced by any predicate remains navigable.  ACORN achieves
2--1000x higher throughput than prior methods by traversing only the
predicate-satisfying subgraph during search.  Weaviate adopted ACORN as its
default filter strategy starting in v1.34 [Weaviate, 2024].  While ACORN is
predicate-agnostic like cu-roaring-bitmap, it operates on CPU and modifies the
graph structure.  Our work is orthogonal: cu-roaring-bitmap optimizes the filter
evaluation step on GPU without modifying the graph, and could in principle be
combined with an ACORN-style GPU graph.

UNG [Cai et al., SIGMOD 2024] constructs a Label Navigating Graph (LNG) as a
directed acyclic graph over distinct label sets, with a proximity graph built per
LNG vertex.  This structure supports any query label size and achieves 10x
speedups by searching only vectors with matching label sets.  Like
Filtered-DiskANN, UNG requires labels to be indexed at build time.

PathFinder [arXiv 2025] extends filtered ANN to complex boolean predicates with
conjunctions and disjunctions.  It builds per-attribute HNSW indices and employs
a cost-based optimizer to select which indices to search and how to combine
results.  Its "out-of-range search" strategy considers non-matching neighbors for
graph expansion while maintaining a separate result queue for matching vectors,
achieving up to 9.8x higher throughput at 95% recall.  PathFinder's approach is
complementary to ours: it optimizes which graph paths to explore, while
cu-roaring-bitmap optimizes the per-candidate filter evaluation along any path.

SIEVE [Li et al., PVLDB 2025] builds a collection of HNSW indices tailored to
an observed query workload.  A three-dimensional analytical model captures the
relationship among index size, search time, and recall, guiding the selection of
which indices to build under a memory budget.  At query time, a router selects
the index with the best expected latency-recall trade-off for each filter,
achieving up to 8x speedup.  SIEVE assumes a known workload distribution and
requires significant pre-computation, whereas cu-roaring-bitmap supports
arbitrary ad-hoc predicates with no workload assumptions.

**Range filtering.**  SeRF [Zuo et al., SIGMOD 2024] introduced the segment
graph for range-filtered ANN, where queries restrict to vectors whose scalar
attribute falls within a range [l, h].  SeRF compresses O(n) segment-specific
ANNS indices into a single structure with O(n log n) index size, breaking the
quadratic barrier of naive range partitioning.  UNIFY [Yao et al., VLDB 2025]
extends this with a Segmented Inclusive Graph (SIG) that ensures any segment
combination's proximity graph is a subgraph of the unified index, and adds a
hierarchical variant (HSIG) inspired by HNSW for logarithmic filtering
complexity.  DIGRA [SIGMOD 2025] further adds dynamic update support via a
multi-way tree structure with lazy weight-based updates, addressing the practical
requirement of evolving datasets.  Range-filtered methods are orthogonal to
cu-roaring-bitmap: they optimize for continuous attribute ranges, while we
optimize for arbitrary set-membership predicates.

**Hybrid query systems.**  VBase [Zhang et al., OSDI 2023] unifies vector
similarity search with relational queries through the concept of relaxed
monotonicity, which allows vector indices to be used within a standard query
optimizer.  VBase integrates into PostgreSQL and supports HNSW, IVFFlat, and
SPANN indices, achieving up to three orders of magnitude higher performance than
prior vector database systems on complex queries.  VBase targets CPU-based
analytical queries over moderate-scale data, whereas cu-roaring-bitmap targets
high-throughput GPU filtering at billion scale.

## 6.5 Production Vector Search Systems

Several production systems have adopted filtered ANN search with varying
strategies.  Milvus [Zilliz] uses Knowhere as its vector execution engine,
integrating Faiss, HNSW, and DiskANN indices with bitset-based filtering.  Milvus
2.4 added GPU indexing via NVIDIA CAGRA, using flat bitset masks for filtering.
Knowhere 2.0 improved filtered HNSW performance by 6--80x through optimized
bitset evaluation, but still relies on uncompressed bitsets that scale linearly
with dataset size.

Weaviate [Weaviate] adopted ACORN-inspired filtering in v1.27 (October 2024)
and made it the default strategy in v1.34.  The ACORN-1 variant checks filters
before computing distances, avoiding expensive distance calculations for
non-matching candidates.  This approach dramatically improves performance for
negatively correlated filters (where matching vectors are far from the query
in the graph) but operates on CPU and does not address GPU memory constraints.

Vespa [Vespa.ai] implements a dynamic strategy selector for HNSW-based filtered
search, choosing among pre-filtering, post-filtering, and ACORN-1 based on
estimated filter selectivity.  When the filter matches fewer than ~5% of
vectors, Vespa falls back to exact brute-force search over the filtered subset.
The selectivity thresholds are configurable per deployment.

All three production systems use flat (uncompressed) bitmaps or bitsets for
filter representation.  At billion-vector scale, this requires 125 MB per filter
attribute, consuming a significant fraction of available memory.  Cu-roaring-bitmap
addresses this gap by providing compressed filter storage (2.1 MB for a 0.1%
selectivity filter on 1B vectors) that integrates directly into the GPU search
kernel without query performance degradation.

## 6.6 Benchmarks for Filtered ANN Search

The Big-ANN NeurIPS'23 competition [Simhadri et al., NeurIPS 2023] introduced
the first large-scale benchmark for practical ANN variants, including a filtered
search track alongside sparse, out-of-distribution, and streaming tracks.  The
competition used ~10M-point datasets on normalized hardware (8 vCPUs, 16 GB DRAM)
and measured throughput at 90% 10-recall@10, establishing a standard evaluation
protocol for filtered ANN.

Shi et al. [Shi et al., PVLDB 2025] presented a unified benchmark and systematic
experimental study of filtered ANN methods, evaluating 10 algorithms across six
real-world datasets (arXiv, TripClick, LAION, YFCC, YouTube-Audio,
YouTube-Video) with over 41,000 parameter combinations.  Their key finding is
that no single method dominates all filter types: UNG excels at containment and
fixed-length equality, ACORN-1 and Stitched-DiskANN provide balanced performance
for overlap filters, and several methods degrade significantly as dataset size
increases.  The study also revealed that Filtered-DiskANN, CAPS, and UNG fail to
reach 25% recall on their largest dataset, highlighting scalability challenges.

Iff et al. [Iff et al., arXiv 2025] contributed a benchmark using
transformer-based embedding vectors from 2.7 million arXiv paper abstracts with
11 real-world attributes, providing the first publicly available dataset
combining modern embeddings with abundant metadata.  Their evaluation of 11
algorithms distilled eight key observations guiding method selection by filter
type, selectivity, and scale.

These benchmarks focus exclusively on CPU methods.  None evaluates GPU-accelerated
filtered search, and none measures the memory cost of filter representation --
a critical concern at billion scale where filter storage can exceed the GPU's
total memory.  Cu-roaring-bitmap fills both gaps: it is a GPU-native filter that
compresses representation while maintaining query throughput, evaluated on
datasets up to 1B vectors.

## 6.7 Positioning of Cu-Roaring-Bitmap

Table 1 summarizes how cu-roaring-bitmap relates to the closest prior work.

| System | Platform | Approach | Predicate-Agnostic | Compressed Filter | GPU-Native |
|--------|----------|----------|--------------------|-------------------|------------|
| Flat bitset (cuVS) | GPU | Post-filter | Yes | No | Yes |
| CRoaring | CPU | Bitmap ops | Yes | Yes | No |
| GPU-WAH | GPU | Bulk bitmap | Yes | Yes (WAH) | Partial |
| ACORN | CPU | Modified graph | Yes | N/A | No |
| Filtered-DiskANN | CPU | Label-partitioned graph | No | N/A | No |
| VecFlow | GPU | Label-centric IVF | No | N/A | Yes |
| SIEVE | CPU | Index collection | No (workload) | N/A | No |
| **cu-roaring-bitmap** | **GPU** | **Post-filter** | **Yes** | **Yes (Roaring)** | **Yes** |

Cu-roaring-bitmap occupies a unique position: it is the only system that is
simultaneously (1) GPU-native, (2) predicate-agnostic (any set of passing IDs
can be uploaded at query time), and (3) uses compressed filter representation.
Predicate-aware methods (Filtered-DiskANN, UNG, VecFlow, SIEVE) achieve higher
throughput for their supported predicate types by baking filter information into
the index, but cannot handle ad-hoc or compound predicates without index
rebuilding.  Cu-roaring-bitmap pays a modest cost for generality -- evaluating a
filter bitmap at each candidate -- but compensates through GPU-optimized data
layout (2-read query path, warp-cooperative queries) and gains substantial
memory savings that enable scaling to billion-vector datasets with hundreds of
filter attributes.
