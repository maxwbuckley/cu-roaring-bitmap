# Filter-driven search — implementation & experiment

*Implements the container→schedule dispatch sketched in
[`filter_driven_search.md`](filter_driven_search.md): a roaring-filtered
brute-force vector search where the filter becomes the loop bounds of the GEMM
instead of a post-hoc mask. Measured on an RTX 5090 (sm_120), 2026-05-22.
Comparison uses cuVS (`v25.12.00a`) `brute_force::search` + raft
`bitset_filter` as the production baseline.*

## What was built

A complete container-type dispatch + search executor in `src/filtered_search.cu`
(API: `include/cu_roaring/detail/filtered_search.cuh`).

| Container in the filter | Schedule | Notes |
|---|---|---|
| absent (no set bits in the 64K block) | **skipped** | 0 cost |
| run, width ≥ 64K | **direct range GEMM** on `db[start:end]` | no copy |
| run, width < 64K | **gathered** | shape gate |
| array (sparse scattered) | **gather** rows → one compact GEMM | |
| bitmap, density ≥ 50% | **range GEMM + bitset mask** | mask = container's own words |
| bitmap, density < 50% | **gather** (bit-scan → ids) | |
| `negated` bitmap | complement schedule | absent blocks → full ranges, excluded containers → range + inverted mask |

`build_schedule()` walks the roaring containers and emits a `SearchSchedule`
— direct range tasks + at most one gather buffer. `roaring_filtered_search()`
runs each task: cuBLAS GEMM of the queries against that task's database
columns → optional mask → column-parallel top-k merged into a running
per-query top-k. The whole task sequence is **CUDA-graph-captured and
replayed** (a filtered search issues O(tasks) kernels; per-launch host
overhead would otherwise dominate). The dense baseline is the same executor
over one `[0,N)` range with the decompressed filter bitset as the mask —
the design doc's "option 1".

## Correctness

`test/test_filtered_search.cu` — 7 cases (run / array / dense-bitmap /
sparse-bitmap / mixed / negated / fragmentation-fallback). Each builds a
filter that lands in one dispatch regime, asserts the schedule took that
path, and verifies both the schedule-driven search and the dense baseline
against an exhaustive CPU top-k reference — twice, so the CUDA-graph replay
path is also covered. All pass. **In the cuVS comparison below, schedule-
driven results match cuVS's bitset-filtered brute force at `recall@10 = 1.000`
for every config.**

## Result 1 — vs cuVS bitset filter, main grid

Inner-product top-10, D=128, Q=64. Median of 3 replicates × 20 interleaved
A/B iterations (cv < 2%). Filter shape is either uniform Bernoulli
("scattered") or 4 equally-spaced contiguous runs ("clustered"). For
clustered, the filter is built through CRoaring with `run_optimize()` then
uploaded preserving RUN containers — that's what fires the direct-range
dispatch path.

### Scattered (Bernoulli — produces BITMAP containers, no RUN)

| N | sel | card | containers (run/arr/bmp) | schedule (D/M/G) | cuVS bitset | roaring | speedup | bitset QPS | roaring QPS |
|---|---|---|---|---|---|---|---|---|---|
| 1M | 1% | 9,922 | 0/0/16 | 0/0/1 | 2.6 ms | 0.10 ms | **27×** | 25 k | 673 k |
| 1M | 5% | 49,923 | 0/0/16 | 0/0/1 | 4.6 ms | 0.36 ms | **13×** | 14 k | 179 k |
| 1M | 25% | 249,738 | 0/0/16 | 0/0/1 | 4.5 ms | 1.13 ms | 4.0× | 14 k | 56 k |
| 1M | 50% | 500,275 | 0/0/16 | 0/16/0 | 5.1 ms | 6.9 ms | 0.74× | 13 k | 9 k |
| 5M | 5% | 249,054 | 0/0/77 | 0/0/1 | 15.9 ms | 1.13 ms | **14×** | 4 k | 56 k |
| 5M | 10% | 499,607 | 0/0/77 | 0/0/1 | 32.2 ms | 1.65 ms | **20×** | 2 k | 39 k |
| 5M | 50% | 2,499,161 | 0/0/77 | 0/37/1 | 12.8 ms | 20.3 ms | 0.63× | 5 k | 3 k |
| 10M | 1% | 99,572 | 0/0/153 | 0/0/1 | 7.3 ms | 0.66 ms | **11×** | 9 k | 97 k |
| 10M | 5% | 499,833 | 0/0/153 | 0/0/1 | 32.2 ms | 1.64 ms | **20×** | 2 k | 39 k |
| 10M | 10% | 998,671 | 0/0/153 | 0/0/1 | 65.3 ms | 2.46 ms | **27×** | 1 k | 26 k |
| 10M | 50% | 5,001,379 | 0/0/153 | 0/153/0 | 18.7 ms | 68.9 ms | **0.27×** | 3 k | 0.9 k |
| 25M | 5% | 1,250,015 | 0/0/382 | 0/0/1 | 81.9 ms | 3.53 ms | **23×** | 0.8 k | 18 k |
| 25M | 50% | 12,499,860 | 0/0/382 | 0/194/1 | 34.1 ms | 102.6 ms | 0.33× | 2 k | 0.6 k |
| 50M | 1% | 501,047 | 0/0/763 | 0/0/1 | 32.3 ms | 1.65 ms | **19×** | 2 k | 39 k |
| 50M | 5% | 2,501,961 | 0/0/763 | 0/0/1 | 165.3 ms | 6.49 ms | **25×** | 0.4 k | 10 k |
| 50M | 10% | 4,997,866 | 0/0/763 | 0/0/1 | 3,295 ms | 12.2 ms | **271×** ✱ | 0.02 k | 5 k |

✱ At 10% selectivity (sparsity = 0.9, exactly cuVS's CSR↔dense dispatch
threshold) cuVS picks CSR/SpGEMM, which scales catastrophically at 50M.

### Clustered (4 equally-spaced contiguous runs — RUN containers preserved)

| N | sel | card | containers (run/arr/bmp) | schedule (D/M/G) | cuVS bitset | roaring | speedup | bitset QPS | roaring QPS |
|---|---|---|---|---|---|---|---|---|---|
| 1M | 1% | 10,000 | **4**/0/0 | 0/0/1 | 2.6 ms | 0.10 ms | 27× | 25 k | 669 k |
| 1M | 25% | 250,000 | 7/0/0 | 0/0/1 | 4.3 ms | 1.14 ms | 3.8× | 15 k | 56 k |
| 1M | 50% | 500,000 | 11/0/0 | **4/0/0** | 4.7 ms | 3.3 ms | **1.4×** | 14 k | 19 k |
| 5M | 10% | 500,000 | 10/0/0 | **4/0/0** | 31.4 ms | 3.3 ms | **9.6×** | 2 k | 19 k |
| 5M | 25% | 1,250,000 | 20/0/0 | **4/0/0** | 11.3 ms | 5.1 ms | 2.2× | 6 k | 13 k |
| 5M | 50% | 2,500,000 | 40/0/0 | **4/0/0** | 11.4 ms | 7.4 ms | **1.5×** | 6 k | 9 k |
| 10M | 5% | 500,000 | 11/0/0 | **4/0/0** | 32.0 ms | 3.3 ms | 9.8× | 2 k | 19 k |
| 10M | 10% | 1,000,000 | 18/0/0 | **4/0/0** | 65.8 ms | 4.5 ms | **15×** | 1 k | 14 k |
| 10M | 25% | 2,500,000 | 40/0/0 | **4/0/0** | 15.9 ms | 7.4 ms | 2.2× | 4 k | 9 k |
| 10M | 50% | 5,000,000 | 80/0/0 | **4/0/0** | 15.7 ms | 14.1 ms | **1.1×** | 4 k | 5 k |
| 25M | 50% | 12,500,000 | 194/0/0 | **4/0/0** | 27.9 ms | 29.8 ms | 0.93× | 2 k | 2 k |
| 50M | 10% | 5,000,000 | 80/0/0 | **4/0/0** | 1,599 ms | 14.1 ms | **114×** | 0.04 k | 5 k |

**Container dispatch confirmed.** Scattered → 100% BITMAP containers, schedule
collapses to one gather task at low/moderate selectivity, one masked-range
task per BITMAP at 50% selectivity. Clustered → 100% RUN containers; once
runs are ≥ 64K wide the schedule's `direct-range` (D) bucket fires — that's
the no-copy direct-GEMM path. The "fallback" column is 1 when narrow runs
got gathered, 0 when direct ranges fired.

## Result 2 — D sweep (N=2M, sel=5%, Q=64)

| D | shape | cuVS bitset | roaring | speedup | bitset QPS | roaring QPS |
|---|---|---|---|---|---|---|
| 128 | scattered | 7.3 ms | 0.66 ms | 11× | 9 k | 97 k |
| 128 | clustered | 7.0 ms | 0.66 ms | 11× | 9 k | 97 k |
| 384 | scattered | 10.1 ms | 0.76 ms | 13× | 6 k | 84 k |
| 384 | clustered | 10.0 ms | 0.76 ms | 13× | 6 k | 84 k |
| 768 | scattered | 14.8 ms | 0.91 ms | **16×** | 4 k | 70 k |
| 768 | clustered | 14.6 ms | 0.91 ms | **16×** | 4 k | 71 k |
| 1536 | scattered | 23.4 ms | 1.19 ms | **20×** | 3 k | 54 k |
| 1536 | clustered | 23.1 ms | 1.18 ms | **20×** | 3 k | 54 k |

The schedule-driven advantage **grows with D**: at D=128 it's 11×, at
D=1536 (typical for modern embeddings) it's 20×. The dense path's GEMM cost
scales linearly with `N · D`; the schedule-driven path's GEMM scales with
`card · D`. So the bigger D gets, the bigger the proportional saving from
not scanning the universe. Shape doesn't matter here — at card=100k both
shapes collapse to one gather task.

## Result 3 — Q sweep (N=10M, sel=5%, D=128, scattered)

| Q | cuVS bitset | roaring | speedup | bitset QPS | roaring QPS |
|---|---|---|---|---|---|
| 1 | 3.6 ms | 0.22 ms | 17× | 0.28 k | 4.6 k |
| 8 | 5.5 ms | 0.41 ms | 14× | 1.4 k | 20 k |
| 32 | 21.6 ms | 0.95 ms | 23× | 1.5 k | 34 k |
| 64 | 32.2 ms | 1.64 ms | 20× | 2 k | 39 k |
| 128 | 41.5 ms | 3.10 ms | 13× | 3 k | 41 k |

The speedup is roughly flat across the GEMV→GEMM transition (13–23×) — the
schedule-driven path saves work in proportion to `card / N`, and that ratio
doesn't depend on Q. The headline throughput is at Q=128: **41k QPS on a
single 5090** for a 5%-selective filter over 10M-row inner-product top-10.

## Investigation: the 50% scattered regression

At ≥40% selectivity on scattered data, schedule-driven slows down to
0.27–0.74× of the bitset baseline (the design doc's own caveat: "doing
less work is not faster if it fragments into many small awkward GEMMs"). I
profiled `10M / 50% / scattered` with `nsys` (mode A = bitset, mode B =
roaring, 90 timed iterations each). Per-kernel time, schedule-driven side:

| kernel | instances | avg | total | % |
|---|---|---|---|---|
| `partial_topk_kernel` (mine) | 153 | 383 µs | 58.7 ms | **42%** |
| `cutlass_simt_sgemm_128x64` (mine) | 152 | 34 µs | 5.2 ms | 4% |
| `reduce_into_running_kernel` (mine) | 153 | 22 µs | 3.4 ms | 2% |
| `apply_mask_kernel` (mine) | 153 | 8.7 µs | 1.3 ms | 1% |

The GEMM itself isn't the bottleneck — at 10M/50% the schedule does *fewer*
total GEMM columns than cuVS (~7.6M vs 10M). The bottleneck is **my top-k
fires once per task**: at 50% scattered the schedule fragments into ~76
small masked-range tasks (one per dense BITMAP container), each pays a
`partial_topk_kernel` call. cuVS's side:

| kernel | instances | avg | total |
|---|---|---|---|
| `magma_sgemmEx` (cuVS GEMM) | 505 | 984 µs | 497 ms |
| `cub for_each` (cuVS mask) | 505 | 726 µs | 367 ms |
| `raft warpsort block_kernel` (cuVS top-k) | 1010 | 170 µs | **172 ms** |

cuVS runs one big tiled GEMM + raft's `warpsort` top-k once per search.
**Clustered (RUN containers) recovers most of the regression.** Same N×sel
but 4 wide runs instead of 76 small masked blocks:

| config | scattered roaring | clustered roaring | improvement |
|---|---|---|---|
| 5M / 50% | 20.3 ms | 7.4 ms | 2.7× |
| 10M / 50% | 68.9 ms | 14.1 ms | **4.9×** |
| 25M / 50% | 102.6 ms | 29.8 ms | 3.4× |

4 direct-range tasks + 4 top-ks beats 76 masked-range tasks + 76 top-ks
cleanly. The 10M/50% case goes from **0.27× to 1.12× vs cuVS** — the
regression is *data-layout-dependent*, not fundamental.

## Investigation: cuVS's bitset_filter pays a popcount kernel per search

cuVS's `raft::core::bitset` does **not** store cardinality.
`knn_brute_force.cuh:629` calls `bitset_view::count(res)` on every filtered
search, which runs a popcount kernel (`raft::popc`) over the whole bitset to
compute sparsity, then dispatches CSR/SpGEMM at sparsity ≥ 0.9 vs
dense+mask below. This is the cause of the non-monotonic shape of the
bitset baseline's cost: it's not one curve, it's two paths joined at the
sparsity-0.9 threshold. Most starkly visible at 50M / 10% (sparsity exactly
0.9 — picks CSR — collapses to **3.3 seconds**).

The roaring filter, by contrast, stores cardinality as a field
(`cu_roaring::GpuRoaring::total_cardinality`), populated at upload. cuVS's
brute_force path reads it directly (`brute_force.cu:46`), so the
schedule-driven path computes sparsity for free.

## Optimisation analysis

Two issues, both quantified above.

1. **Top-k is local-memory-latency bound + per-task overhead.**
   `partial_topk_kernel` keeps each thread's running top-k in a 32-slot
   array that is dynamically indexed, so ptxas places it in a 1 KB
   per-thread stack frame; the kernel runs at ~31% occupancy. Fix: a
   register-resident or warp-distributed top-k (each lane owns k/32 of the
   list, bitonic merge over shuffles), or just adopt raft's `warpsort`. The
   stand-alone per-call speed isn't the only issue — the **per-task multiplier
   at fragmented schedules** is.

2. **Search-level selectivity gate.** Above ~30% selectivity on scattered
   data the schedule-driven path loses cleanly. The fix is to fall back to
   `dense_filtered_search` (one big GEMM + one warpsort) when
   `cardinality / N` exceeds a learned threshold. The benchmark above
   locates the crossover exactly; the gate is a 5-line change in
   `roaring_filtered_search`. Not implemented in this branch; documented as
   the next step.

A third opportunity is **per-task gather coalescing for clustered data with
many medium runs**: as Result 1 shows, scattered (1 gather task) beats
clustered (4 direct tasks) at moderate selectivity because the per-task
top-k × n_tasks bill outweighs the no-copy advantage of direct GEMMs. The
right rule is probably "use direct only when each run is big enough that
the GEMM is the bulk of the work" — formally a function of D and Q.

## Result 4 — synthetic filter generators

Filters loaded from the `roaring-benchmark` library's five distribution
generators (CRoaring portable `.bin` files), each at N=10M. Same database
+ harness as Result 1 (D=128, Q=64, k=10, interleaved A/B, recall=1.000
throughout). Cells are `bitset_ms / roaring_ms = speedup`.

| sel | uniform | clustered | multi_tenant | power_law | temporal |
|---|---|---|---|---|---|
| 0.01% | 0.8 / 0.05 = **17×** | 1.1 / 0.05 = **22×** | 1.0 / 0.05 = **21×** | 1.0 / 0.05 = **21×** | 0.9 / 0.05 = **19×** |
| 0.1% | 2.4 / 0.11 = **21×** | 2.5 / 0.11 = **22×** | 2.2 / 0.10 = **22×** | 2.2 / 0.10 = **23×** | 2.0 / 0.10 = **21×** |
| 1% | 7.2 / 0.66 = 11× | 7.5 / 0.66 = 11× | 6.3 / 0.58 = 11× | 6.9 / 0.64 = 11× | 7.1 / 0.66 = 11× |
| 5% | 32.4 / 1.66 = **20×** | 31.3 / 2.63 = 12× | 10.9 / 0.91 = 12× | 36.6 / 1.76 = **21×** | 32.1 / 1.65 = **19×** |
| 10% | 65.9 / 2.47 = **27×** | 66.2 / 5.21 = 13× | 10.8 / 0.91 = 12× | 59.1 / 2.34 = **25×** | 18.1 / 2.49 = 7× |
| 25% | 18.4 / 6.50 = 2.8× | 16.2 / 11.2 = 1.5× | 10.7 / 0.91 = **12×** | 18.3 / 5.05 = 3.6× | 18.3 / 13.9 = 1.3× |
| 50% | 18.5 / 41.4 = **0.45×** | 15.9 / 20.3 = 0.78× | 10.8 / 0.91 = **12×** | 18.2 / 5.05 = 3.6× | 18.2 / 16.5 = 1.1× |

What the container-type breakdown shows, by shape:

- **uniform** — pure Bernoulli, 100% BITMAP containers (153 of them at N=10M),
  one gather task at sel ≤ 10%, masked-range per BITMAP at 50% — fragments
  badly above sparsity 0.9 (the cuVS dense+mask threshold).
- **clustered** — 100% RUN containers; `enumerate_runs` coalesces into 4
  direct-range GEMMs once runs are wide enough (sel ≥ 1%), and the schedule
  stays compact through 50%. Recovers the regression to 0.78× vs uniform's
  0.45× — direct-range dispatch doing exactly what the design said it
  should.
- **multi_tenant** — the "target selectivity" caps at card ≈ 152K (the
  generator's tenant has a fixed size); roaring time is **constant ~0.9 ms**
  across 5–50% target, so the speedup *grows* with target selectivity to
  **12× even at 50%**. Practical lesson: when the *real* eligible set is
  small, schedule-driven is a clean win regardless of how the caller phrases
  selectivity.
- **power_law** — ARRAY containers at low sel, ARRAY→BITMAP transition at
  10%, BITMAP-dominated above. Crosses to dense+mask at sparsity 0.75 so
  cuVS levels off at ~18 ms; roaring also levels off near 5 ms because the
  power-law distribution concentrates the cardinality. Speedup stays at
  3.6× through 50%.
- **temporal** — autocorrelated (recent activity is correlated), at 50% it
  produces a mix of RUN + ARRAY + BITMAP containers (30/40/83); the
  schedule fragments into 1 direct + 20 masked + 1 gather. Comes out at
  1.1× (parity with cuVS), better than uniform's 0.45× because of the RUN
  components but not as good as pure clustered.

The headline reading: at low-to-moderate selectivity the schedule-driven
path wins by **10–27× across every shape**. Above ~10% the bitset
crossover to dense+mask compresses the gap. At ≥25% the *shape* of the
filter starts to matter — pure-Bernoulli is the worst case, and the
multi-tenant case (where the apparent and real selectivity disagree) is the
best.

## Result 5 — YFCC-10M (real per-query filters, unsorted)

NeurIPS'23 Big-ANN filtered track: 10M base vectors (192-d u8), 100K queries
each carrying 1–2 tag predicates, 7910 distinct query-relevant tags. The
per-query filter is the intersection of that query's tag bitmaps — i.e.
**the filter changes every query**, so the "schedule built once, reused
across the batch" assumption from the synthetic results doesn't apply. 256
sampled queries, Q=1, inner-product top-10.

| measurement | cuVS bitset | roaring schedule |
|---|---|---|
| **search only** (filter pre-built) | 0.641 ms / 1561 QPS | **0.046 ms / 21512 QPS** |
| schedule build (per query) | — | 2.868 ms |
| **end-to-end** (build + search) | 0.641 ms | **2.920 ms / 342 QPS** |
| | — | 0.22× vs cuVS |

`card` per query: median 15,238 (mean 193,204 — long tail).

**Two opposing forces.** When the filter is pre-built, schedule-driven is
**13.8× faster than cuVS** on real YFCC queries — comparable to the
synthetic numbers at similar selectivity (~0.1% median sparsity here). But
the **per-query schedule build is ~2.9 ms**, dominating the 0.046 ms search
and pushing end-to-end to **0.22× of cuVS** — i.e. 4.5× slower overall.
This is the per-query-filter regime: cuVS pays essentially nothing to
construct a `bitset_filter` view from an already-built bitset; cu_roaring
pays the upload (sort + container build + key-index) plus
`build_schedule()` (enumerate_runs + container-dispatch kernels + the
gather pre-pass).

**Where this matters.** For "shared filter across a query batch" workloads
(saved searches, ACL filters that apply to whole result pages, the
synthetic configs above) the search-only number is the right one. For
"each user's query has its own filter" workloads — and YFCC is exactly
that — the build cost is in the hot path, so the right number is the
end-to-end one, and **the schedule-driven path is not competitive with
cuVS bitset on this workload as currently implemented**. Concrete next
steps: (a) reduce upload cost (the YFCC tags here build a fresh GpuRoaring
+ direct-map key index per query — most of the 2.9 ms is upload, not
schedule), (b) reuse a tag-bitmap GPU cache across queries (each YFCC
tag bitmap is read by many queries — caching the upload once amortises
it), (c) build the per-query intersection directly on the GPU from cached
tag bitmaps (multi_and kernel already exists in cu_roaring).

### YFCC-10M sorted by tag tuple

I built a lex-by-tag-tuple permutation of YFCC-10M (5.33M unique tag tuples
across 10M items; preprocessing in `tools/yfcc_sort_by_tags.py`), re-emitted
the base vectors and all 7910 tag bitmaps in the new order, and re-ran the
bench. The per-tag bitmaps compress dramatically on the sorted layout:

| tag | unsorted serialized | sorted serialized (+ run_optimize) | ratio |
|---|---|---|---|
| 1 (266K items)   | 535 KB | **75 bytes** | 7100× |
| 5 (1.24M items)  | 1.25 MB | **287 bytes** | 4500× |
| 28 (706K items)  | 1.25 MB | **1.6 KB** | 770× |
| 100 (509K items) | 694 KB | 16 KB | 43× |

Same harness, same 256 sampled queries, `run_optimize`'d filter (so RUN
containers would survive upload). End-to-end median:

| measurement | unsorted | sorted | Δ |
|---|---|---|---|
| cuVS bitset                | 0.641 ms | 0.672 ms | ≈ |
| roaring SEARCH (pre-built) | 0.046 ms | 0.046 ms | ≈ |
| roaring BUILD              | 2.868 ms | 2.960 ms | ≈ |
| roaring END-TO-END         | 2.920 ms | 3.073 ms | ≈ |
| speedup search-only        | 13.8×    | 14.5×    | +5% |
| speedup end-to-end         | 0.22×    | 0.22×    | — |

The sort makes tag bitmaps thousands of times smaller (the result above
is real and not in doubt) — but neither end of the search changes
meaningfully. Why:

- **Search path doesn't differ.** Typical YFCC per-query filter has card
  ≈ 15K (median), so the AND-of-tag-bitmaps result is narrower than the
  64K direct-range threshold. Both layouts route to the gather path, with
  the same single small GEMM, so the search itself is the same ~0.046 ms.
  To exercise the direct-range path on YFCC you'd need either many
  large-card queries or a lower direct-range threshold (a tunable).
- **Build cost is CUDA-call-bound, not shape-bound.** The 2.9–3.0 ms
  per-query build is dominated by ~15–20 separate `cudaMalloc /
  cudaMemcpyAsync / cudaStreamSynchronize` calls inside
  `cu_roaring::upload(bm, N)` and `build_schedule()`. On WSL2 each call is
  50–200 µs; the schedule-build pipeline accumulates ~3 ms of host-side
  CUDA overhead regardless of how compact the input is. Sorting doesn't
  remove those calls.

**What would move the needle.** The schedule-driven path is built for
"filter shared across a batch" — that's where the 10–27× speedups live
(Results 1–4). For per-query-filter workloads like YFCC, the architectural
change required is to keep the *tag* bitmaps GPU-resident and uploaded
once at startup, and build per-query filters with a **GPU-side multi-AND**
(`cu_roaring::fused_multi_and` already exists) — bypassing the per-query
host→device upload entirely. With cached tag bitmaps the per-query work
collapses to one device-side AND + one `build_schedule` call. That should
make end-to-end competitive on YFCC; not implemented in this branch, but
the diagnostic above shows where the cost actually lives.

## Reproduce

```bash
# cu-roaring side: implementation, tests, dense baseline
cd build && cmake --build . --target test_filtered_search bench_filtered_search -j
ctest -R filtered_search --output-on-failure
./bench/bench_filtered_search

# cuVS comparison: bench_schedule_driven_roaring.cu (synthetic shapes,
# scattered + clustered) and bench_synth_filters.cu (loads .bin filters
# from roaring-benchmark's sweep_10M output) both live in the cuVS bench
# tree (cpp/bench/prims/core).  Each needs cu_roaring built with the cuVS
# bench toolchain (CUDA 12.4 / sm_89) so the static library device-links.
#
# To reproduce the synthetic-generator table:
#   python3 -m roaring_bench sweep -n 10000000 --num-trials 1 \
#     --output-dir output/sweep_10M --save-bitmaps --skip-access
#   SYNTH_FILTER_DIR=.../sweep_10M/bitmaps SYNTH_N=10000000 \
#     ./bench_synth_filters
```

Raw results: [`bench_cuvs_comparison.json`](bench_cuvs_comparison.json),
[`bench_synth_generators.json`](bench_synth_generators.json).
