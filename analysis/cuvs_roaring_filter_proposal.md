# A case for Roaring-bitmap-backed filtering in cuVS

**TL;DR** — cuVS today serves filtered ANN search through
`raft::core::bitset` + `bitset_filter`. For filter selectivities **above
~50%** the flat-bitset path is excellent — it's a tight fused
brute_force-plus-mask kernel and is hard to beat. For everything **below
that** (the dominant production regime: tag predicates, recency cutoffs,
permission filters), the bitset path leaves substantial performance on the
floor — both because the per-query bitset is large (`N/8` bytes, redundant
when most of those bits are zero) and because cuVS's sparse-path
implementation is materialisation-heavy (`make_device_csr_matrix +
csr_to_coo + masked_matmul`).

We've shipped a working Roaring-bitmap implementation
(`github.com/maxwbuckley/cu-roaring-bitmap`, `run-optimizations` branch)
with a schedule-driven brute_force-filtered search that addresses this
gap. Measured on **RTX 5090, N=10M, D=512, Q=64, k=10, recall@10 = 1.000
across every cell of every config**:

| selectivity | uniform-random filters | **clustered filters** (sorted tables, range predicates) |
|---|---|---|
| 0.01% | 1.54× faster than cuVS bitset | 1.07× |
| 1%    | 6.06× | 6.21× |
| 10%   | **12.50×** (fp32) / **17.11×** (fp16) | **18.75×** |
| 30%   | 1.48× | 2.22× |
| 50%   | 0.63× (cuVS faster) | **1.36×** |
| 90%   | 0.66× | 0.72× |

The case for a contribution to cuVS rests on six arguments. Each is backed
by data in `results/raw/2026-05-23/` and a reproducible bench.

## 1. The memory argument: filter representation matters at scale

A `raft::core::bitset` for `N=10M` is **1.25 MB** regardless of how many
bits are set. A search engine with 100 filterable attributes pays 125 MB
of permanent device memory for filter storage alone; at `N=1B` that's
12.5 GB out of a 5090's 32 GB before any vector data is loaded.

Roaring container-based encoding scales with cardinality + structure, not
universe:

| filter | bitset size (N=10M) | roaring portable bytes |
|---|---:|---:|
| 0.01% selectivity, random | 1.25 MB | **3.16 KB** (400× smaller) |
| 1% selectivity, random | 1.25 MB | **196 KB** (6.4× smaller) |
| 10% selectivity, random | 1.25 MB | 1.22 MB (parity — all bitmap containers) |
| 10% selectivity, **clustered** | 1.25 MB | **0.22 KB** (~5,700× smaller!) |
| 90% selectivity, clustered | 1.25 MB | **1.92 KB** (~640× smaller) |

For clustered or near-clustered filters (which describes most production
shapes — tag tables sorted by tag, range filters on sorted columns), the
H2D upload effectively becomes free. cuVS users today pay a constant 1.25
MB H2D per query regardless of filter shape.

This argument doesn't require any change in search throughput to be
compelling — it's a pure memory-and-bandwidth argument that scales with
universe size and number of distinct filters.

## 2. The speedup argument: real per-query gains, especially at sparse-to-moderate selectivity

Below sel ≈ 40%, the schedule approach **never gathers more rows than
necessary**: array containers feed into a single big gather buffer of
size `card × D × 4 bytes`, then a `Q × card` skinny GEMM. cuVS's bitset
path does a full `Q × N` GEMM at sel ≥ 15% (the dense path) or
materialises an `NQ × N` CSR + masked matmul at sel ≤ 10% (the sparse
path) — both proportional to N, not card.

Speedup curve, end-to-end (per-call H2D filter + setup + search + D2H
top-k, both paths) on uniform-random filters:

```
sel %    cuvs ms    roar ms    speedup   path used by roaring
0.01     1.62       0.99       1.54x     gather (1K of 153 array containers)
0.1      3.21       1.04       3.01x     gather
1        12.55      1.98       6.06x     gather (100K of 153 array)
3        34.10      3.54       9.61x     gather (300K of 153 array)
5        55.61      4.90      11.32x     gather
10       111.06     8.88      12.50x     gather (1M of 152 bitmap)  ← peak
15       34.19      12.56      2.71x     gather (cuVS switches to dense)
30       34.40      23.11      1.48x     gather
50       34.47      54.69      0.63x     76 masked + 2.5M gather
90       33.15      50.37      0.66x     153 → 10 masked tasks (post-coalescing)
```

**Three caveats to read this honestly:**

(a) Some of the peak comes from cuVS's `sparsity < 0.9` threshold being
mis-tuned — cuVS picks its sparse CSR path at sel ≤ 10% when its dense
path would actually be faster for sel ≥ 3%. We measured the crossover at
**~sel=3% (sparsity≈0.97)**, not the 0.9 cuVS picks today. If that
threshold is fixed (one-line change), the peak speedup at sel=10% drops
from 12.50× → **~3.8× intrinsic**. Still meaningful — that 3.8× is
schedule-vs-bitset on cuVS's own best code path.

(b) The popcount fairness adjustment (cuVS calls `view().count(res)`
inside every search to decide which path to take) is small — 0.1–0.25
ms per cell, only meaningful at sel ≤ 0.01% (where it's 13% of cuVS
time). At sel ≥ 1% it's <2% of cuVS time and the "fair" speedup tracks
the raw speedup within rounding.

(c) Above sel ~ 40% on uniform random filters, the schedule is slower
than cuVS bitset (0.5–0.7×). Not algorithmically — both paths do
range+mask GEMM tiles — but the schedule's tile dispatch issues 3
kernel launches per tile (sgemm + apply_mask + topk_pass1) where cuVS's
fused brute_force kernel does the equivalent work in fewer launches.
Closing this needs kernel-level fusion work (separately from the
schedule layer), or a hybrid that dispatches to cuVS's brute_force above
some selectivity threshold.

## 3. The shape argument: clustered filters are where production lives

Production filter distributions are rarely uniform-random. Tag-style
filters on sorted tables, recency cutoffs on time-sorted data, range
predicates on sorted columns — these all produce **clustered IDs**, which
roaring encodes via RUN containers. After `run_optimize` + cross-container
coalescing the schedule emits one `kRange`-direct task spanning the whole
run: no gather, no mask, one direct GEMM over a contiguous slice of the
dataset.

Same sweep, clustered filter (single contiguous run of `card` IDs):

```
sel %    cuvs ms    roar ms    speedup    schedule shape
1        12.54      2.00       6.21x      1 direct (100K cols)
3        33.48      2.96      11.24x      1 direct (300K)
5        54.63      3.58      15.21x      1 direct (500K)
10       110.10     5.87      18.75x      1 direct (1M, 16 RUN containers coalesced)
15       30.63      8.09       3.77x      1 direct (1.5M)
30       31.09      13.93      2.22x      1 direct (3M)
50       31.08      22.77      1.36x      1 direct (5M)  ← above parity (was 0.63x random)
90       31.11      42.92      0.72x      1 direct (9M) + 3 trailing masked
```

Two material changes vs the random shape:

- **Peak speedup grows 12.50× → 18.75× at sel=10%.** The single direct
  GEMM skips the gather kernel (saving ~3 ms at 1M rows × 512 dim) that
  the random-shape gather path has to run.
- **The 0.6× regression at sel=50% disappears.** Random's 39 small masked
  tile-tasks become one big direct task; roaring flips from 0.63× →
  **1.36× above parity**. The crossover with cuVS shifts from ~40% to
  ~75% selectivity.

The point is not that clustered is the only shape — it's that **a roaring
implementation gets to use the right algorithm per filter shape, and
cuVS's flat bitset can't.** A bitset has no "this row range is contiguous"
information by construction.

## 4. The fp16 / TensorCore argument: the cuVS misstep gets worse, not better

Re-running the same sweep with fp16 datasets/queries (cuBLAS `GemmEx` with
CUDA_R_16F inputs, CUDA_R_32F accumulation through TensorCores; top-k
stays fp32 so ranking precision is preserved):

| sel % | fp32 D=512 | fp16 D=512 | fp16 D=1024 |
|---|---:|---:|---:|
| 1 | 6.06× | 6.66× | 8.27× |
| 10 | 12.50× | **17.11×** | **17.08×** |
| 30 | 1.48× | 1.30× | 1.04× |
| 90 | 0.66× | 0.65× | 0.71× |

cu_roaring's small GEMM is compute-bound at sel ≤ 10% — TensorCores
deliver. cuVS's sparse path is memory-bound on the CSR materialisation —
TensorCores don't help. So the **peak speedup widens from 12.50× to
17.11×** under fp16. The `sparsity < 0.9` mistuning is correspondingly
more costly in fp16 because the dense path (TensorCore-accelerated)
would have been even cheaper.

fp16 D=1024 was the configuration the project couldn't reach in fp32 on a
single 5090 (40 GB > 32 GB). In fp16 it fits, runs at full TensorCore
throughput, and the 17.08× peak is essentially the same as fp16 D=512 —
the relative gap doesn't shrink with dimension.

## 5. The correctness argument: recall@k = 1.000 across every measured cell

This is the boring argument and the most important one. Across the
combined sweep:

- 11 selectivities × 2 shapes (random, clustered) × 3 (dtype, D) configs
  (fp32 D=512, fp16 D=512, fp16 D=1024) = **66 cells**
- 15 iterations per cell, 5-iter warmup, A/B interleaved
- Top-k IDs compared against the cuVS bitset reference for the same query
  set
- **All 66 cells: recall@10 = 1.000** (bitwise-identical top-k IDs)
- Distances are not bit-identical (fp16 input quantisation; gather-then-
  GEMM accumulation order differs from masked-GEMM order) — but ranking
  is preserved across the dense range tested

The library also has unit-test coverage for every dispatch path:
`test_filtered_search.cu` has 8 tests (RunContainers, ArrayContainers,
BitmapDense, BitmapSparse, MixedContainers, Negated, FragmentationFallback,
Fp16MixedContainers) verifying both schedule and `dense_filtered_search`
baseline against an exhaustive CPU reference; all 8 pass on the
`run-optimizations` HEAD.

## 6. The real-world argument: YFCC-10M end-to-end

Synthetic data is suspicious. We measured the same comparison on YFCC-10M
(real `D=192` SIFT-style features, real per-query tag tuples,
256-query sample) — fair end-to-end timing (cuVS bitset alloc + H2D
inside the timed region, roaring through `upload(filt, N, stream)`):

| | unsorted base | sorted-by-tag base |
|---|---:|---:|
| cuVS bitset (per-query) | 1.10 ms | 1.10 ms |
| roaring END-TO-END | 0.26 ms | **0.23 ms** |
| **speedup** | **4.30×** | **4.87×** |
| recall@10 | 1.000 | 1.000 |

Same shape, same recall, **4–5× faster end-to-end on real data**. This is
the production-realistic number — not a synthetic best case.

(Note: the per-query roaring build path here is
`upload_from_device_bitset`, which silently discards RUN container info.
The CRoaring-preserving `upload(filt, N)` path is currently slower due to
host-side serialisation, but offers a cleaner integration model. See §8.)

## 7. Where this is NOT the right tool

Honest accounting of the regimes where the bitset path wins or where the
schedule approach needs more work before it's competitive:

- **Selectivity > 50% on uniform-random filters**: cuVS bitset is 1.4–
  1.5× faster. Both paths do the same thing (range + mask GEMM), but
  cuVS's fused kernel beats the schedule's per-tile launch overhead.
  Kernel-fusion work in the schedule's masked path would close this.
- **Very small per-query batches at very small D**: the YFCC measurement
  is `Q=1, D=192`. The schedule build cost (1–2 ms via the host
  CRoaring path) dominates over the saved GEMM cost (~50 µs search
  difference). The faster `upload_from_device_bitset` path makes this
  competitive (4.9×) but throws away run info.
- **Tiny clustered filters (sel ≤ 0.1% with a single narrow run)**: hit
  the schedule's fragmentation fallback (runs < 64K wide get gathered
  anyway), pay ~0.2 ms of enumerate_runs overhead the array path
  doesn't pay. Easy follow-up optimisation.

For any of these, the right answer is to **dispatch to cuVS's existing
brute_force** — exactly as today. The roaring filter complements
`bitset_filter`, it doesn't replace it.

## 8. What an integration would look like

The cleanest cuVS integration is a new filter type that mirrors
`bitset_filter`:

```cpp
namespace cuvs::neighbors::filtering {

// New filter type alongside bitset_filter and bitmap_filter.
template <typename IdxT = int64_t>
struct roaring_filter : public base_filter {
    explicit roaring_filter(const cu_roaring::GpuRoaring& bitmap);

    FilterType get_filter_type() const override { return FilterType::Roaring; }

    // Pre-built schedule (caller may build_schedule() once and reuse).
    const cu_roaring::SearchSchedule* schedule = nullptr;
};

}  // namespace cuvs::neighbors::filtering
```

`brute_force::search`'s filter dispatch (currently a `dynamic_cast`-tree at
`knn_brute_force.cuh:621`) gets one more case:

```cpp
} else if (filter_type == FilterType::Roaring) {
    auto* rf = dynamic_cast<const roaring_filter<IdxT>*>(filter);
    cu_roaring::roaring_filtered_search(
        cublas_handle_from_resources(res),
        queries.data_handle(), n_queries,
        idx.dataset().data_handle(), n_dataset, dim,
        *rf->schedule, k,
        neighbors.data_handle(), distances.data_handle(),
        raft::resource::get_cuda_stream(res));
    return;
}
```

Roughly: one new filter class (50 lines), one new dispatch case in
`brute_force::search` (15 lines), one new linker dependency
(`libcu_roaring_bitmap.a`).

`cu_roaring` is currently a standalone library and its build is C++17 /
CUDA 12.x. It can be vendored into cuVS's third_party or pulled via CPM
the same way RAFT pulls thrust.

**fp16** has the same dispatch shape — one extra case for
`roaring_filtered_search_fp16` keyed on the index's `T` template
parameter.

**Filter cardinality is known on the roaring side** without a popcount —
`gpu.cardinality` is materialised at upload time. The dispatch can pass
this directly into the sparse-vs-dense decision instead of running
`raft::popc`, which sidesteps the existing `bitset_view::count()`
overhead entirely. (Independently, the threshold should be raised to
~0.97 — see §2(a) — but with a known cardinality there's no need for the
runtime popc at all.)

## 9. Open questions

These would need cuVS-team buy-in to resolve:

- **Filter-build location.** The cleanest API is "user gives a
  `cu_roaring::GpuRoaring` they already built". But upstream of cuVS,
  many users would want a helper that takes a host CRoaring or a vector
  of IDs. Whose responsibility?
- **`upload(filt, N)` performance.** The host CRoaring path is currently
  1–2 ms on YFCC's ~100-container filters because of `cudaMallocHost`
  pinning cost and host-side container packing. A pre-pinned buffer
  pool and a GPU-side roaring AND (for intersecting tag bitmaps without
  round-tripping to host) would push this under 0.5 ms. This is
  cu_roaring work, not cuVS work, but informs the integration design.
- **`gather_db` lifetime.** The schedule currently owns a `void*`
  gather buffer keyed off the build_scratch arena. For repeated searches
  against the same filter the schedule can be cached and reused;
  building this caching layer into cuVS or leaving it to the user?
- **CAGRA / IVF integration.** This proposal is brute_force-only — the
  filter shows up in CAGRA's graph search and IVF's posting-list scan
  too. Earlier cu-roaring work integrated with CAGRA via a custom
  `roaring_filter` template (in `include/cu_roaring/scoped_gpu_roaring.hpp`);
  the schedule-driven brute_force is a separate and complementary path.
  Brute_force is the obvious first integration target because the
  speedups are largest.

## 10. What's already done

- Library: `cu-roaring-bitmap` on the `run-optimizations` branch
  (`origin/run-optimizations`, `7ca7c7f`). 8 unit tests covering every
  dispatch path including fp16. Recall=1.000 across every cell of every
  benchmark.
- Bench harness: `~/Development/cuvs/cpp/bench/prims/core/bench_e2e_sweep.cu`
  drives a 11-cell selectivity sweep, supports `E2E_DTYPE=fp16`,
  `E2E_SHAPE=clustered`, fair-framing baked in. Bench source snapshot
  committed at `results/raw/2026-05-23/bench_e2e_sweep.cu.snapshot`.
- Reproducer: every cell in every table above is one
  `./bench_e2e_sweep` invocation, env vars documented in
  `results/raw/2026-05-23/README.md`.
- Real-data validation: YFCC-10M, 256-query sample, recall=1.000,
  4.30–4.87× end-to-end speedup; `results/raw/2026-05-23/bench_yfcc_*`.
- Two cu-roaring lib improvements landed during the investigation that
  matter for the integration story: masked-task coalescing (commit
  `c19f60a`) and fp16 support (commit `b841802`).

## 11. The ask

For the cuVS team:

1. **Fix the `sparsity < 0.9f` threshold in `knn_brute_force.cuh:643`.**
   Independent of any roaring contribution — this is leaving real
   performance on the floor today. Recommended value: 0.97 (measured
   crossover on RTX 5090 at the configs we tested). Right answer is a
   measured cost model, but 0.97 as a hardcoded improvement is an
   unambiguous win for ANY user with sel ∈ [3%, 10%] filters.
2. **Land the popcount-skipping path** (your in-flight PR). Frees up
   the small-card regime regardless of what happens with roaring.
3. **Discuss the roaring integration shape** above. We're happy to do
   the work upstream — split the library out cleanly, add the
   `roaring_filter` class, write the integration tests. The bigger
   question is whether cuVS wants this as a native filter type or as
   an extension. Our preference is native because that's where the
   memory + bandwidth wins compound across the whole search stack.

For everyone reading this without write access to cuVS: the library is
usable today as a sibling brute_force pipeline. The
`bench_e2e_sweep` harness is the easiest way to verify any of the
numbers above on your own hardware and shape.
