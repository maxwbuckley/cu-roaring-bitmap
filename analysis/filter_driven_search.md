# Filter-driven search — design notes

*Design rationale behind `enumerate_runs` and the run-aware filtered-search
path. Companion to [`ENUMERATE_RUNS.md`](../ENUMERATE_RUNS.md) (API + tests) and
[`bench/DECOMPRESS_RUN_BENCHMARK.md`](../bench/DECOMPRESS_RUN_BENCHMARK.md)
(decompress benchmark). Written for posterity — the "why" for branch
`run-optimizations`.*

## Summary

A GPU vector search with a sparse filter has two ways to use a roaring-bitmap
filter:

1. **Expand to a dense bitset**, then run standard bitset filtering. Roaring is
   only a compression format for the host→device transfer; once expanded, the
   filter costs whatever a dense bitset costs.
2. **Schedule-driven** — never expand. Treat the roaring structure as a *work
   schedule*: the filter becomes the loop bounds of the search itself.

This repo is built on (2). These notes record why, and what landed toward it.

## The cost argument

The dense-bitset filter for a universe of N IDs is N/8 bytes — **independent of
selectivity**. For N = 200M that is 25 MB; a batch of 32 per-query filters is
~800 MB. A brute-force search with such a filter:

- computes the full `queries × N` distance GEMM and *then* masks — the filter
  saves nothing on the matmul;
- reads O(N) filter bytes regardless of how few bits are set;
- overflows L2 once N exceeds ~150–200M IDs on datacenter GPUs (A100 40 MB,
  H100/H200 50 MB, L40S 48 MB L2) — the bitset stops being cache-resident;
- cannot skip an empty region without reading it.

Cost scales with the **universe**. For a sparse filter, almost all of it is
waste. Expanding roaring → bitset is a low-*risk* change (it only optimizes the
PCIe transfer, and the GPU-side filtering is unchanged and well-trodden) but it
is not low-*cost*: it permanently commits to the dense-bitset cost model.

The schedule-driven approach makes cost scale with **cardinality and filter
shape** instead. A roaring bitmap only stores containers for populated 64K
blocks, and each container's type is a hint for the cheapest way to turn it
into search work:

| Container | Meaning | Search schedule |
|---|---|---|
| absent | 64K block, no set bits | **skip** — zero work, never iterated |
| run | contiguous range | a **dense GEMM tile** on `db[start:end]` |
| array | sparse scattered IDs | **CSR** — gather rows → dense GEMM, or SDDMM |
| bitmap | dense-but-scattered block | block GEMM + mask, *or* gather if not near-full |

For a filter that is "10 contiguous runs covering 20M of a 200M universe," the
run path does 10 sub-GEMMs over 20M rows instead of one GEMM over 200M — the
180M filtered-out rows are never read. The saving is `≈ 1/selectivity`
(estimated ~10× here, ~100× at 1%, ~1000× at 0.1%), bounded at the very sparse
end by per-tile launch overhead.

## What is skipped, what is wasted

- **Absent containers** — pure skip. Zero GEMM, zero memory, never iterated.
  This is the bulk of the saving for a sparse filter.
- **Run containers** — zero waste. GEMM the coalesced `[start, end)` exactly;
  every row in the range is eligible. (Requires GEMMing coalesced ranges, not
  whole containers — see `enumerate_runs`.)
- **Array containers** — no wasted *FLOPs*: a gather produces a compact
  `card × D` matrix and the GEMM is `Q × card`, every column eligible. The cost
  is the gather's *scattered* row reads — access-pattern overhead, not wasted
  computation.
- **Bitmap containers** — the only real FLOP waste, *if* handled by whole-block
  GEMM + mask (`65536 − card` wasted dot products). But the container type is a
  *storage* decision (it crossed the 4096-element size threshold), not a search
  mandate: a sparse-ish bitmap container can be bit-scanned and gathered like an
  array. The honest dispatch is on **density/contiguity**, not stored type —
  contiguous → range GEMM, sparse → gather/CSR, near-full → block GEMM + mask.

Net for sparse data: GEMM work goes O(universe) → O(cardinality); the waste is
second-order (gather inefficiency, near-full-block FLOPs).

## Lookups become enumeration — the binary searches go away

The point-query model (`contains(id)`) is *candidate-driven*: it iterates
candidates and **probes** the filter, and locating an ID needs a key binary
search plus an in-array binary search — an `O(log structure)` factor per
candidate.

The schedule-driven model **inverts** this. It never asks "is X in the set?" It
**enumerates** the set: `enumerate_runs` / `enumerate_ids` linearly scan the
(sorted) roaring structure and emit ranges / CSR indices. Enumeration visits
everything in order — there is nothing to search for. The emitted schedule then
*is* the GEMM's loop bounds; the search kernel reads `db[start:end]` or
`db[gathered_ids]` and never touches the filter again.

So in the schedule-driven brute-force path there is **no binary search at all**:
`enumerate_*` are linear scans + CUB prefix sums, the GEMM is a matmul, the
gather is an indexed read, the threshold filter is a compare, select_k is a
top-k reduction. The `O(log)` factor does not shrink — it disappears, and
filter work is decoupled from the candidate count.

**Boundary:** this works when the candidate set is enumerable ahead of time —
brute-force, IVF list scans. Graph traversal (CAGRA) discovers candidates
dynamically and *must* still point-query the filter, so `contains()` /
`warp_contains()` and their binary searches remain necessary for that path. The
library wants both.

## Reusing the progressive top-k threshold

The progressive-threshold technique — process tile 0 with a full select_k to
establish a per-query threshold (the worst distance currently in the top-k),
then filter every later tile against it and merge only survivors — drops
straight in: the runs *are* the tiles. The run path therefore reuses the
standard primitives (parallel threshold extraction, warp-aggregated survivor
compaction, merge-buffer assembly, select_k).

Two savings stack and are complementary:

- **run-tiling** cuts the GEMM — O(cardinality) eligible rows, not O(universe);
- **the threshold** cuts the select_k / merge on top — once it is tight, later
  run-tiles contribute almost no survivors.

The threshold does not cut GEMM FLOPs (distances are data-dependent — you must
compute a row to compare it). A run-aware Phase 1 can also *sample across all
runs* for the initial threshold, avoiding the prefix-sample bias that a
fixed "first 200K columns" Phase 1 suffers on clustered data.

## What landed (branch `run-optimizations`)

1. **Word-at-a-time RUN-container decompress.** `decompress_kernel` expanded
   each run bit-at-a-time — one `atomicOr` per set bit, with the 32 atomics per
   output word serialized by hardware. It now expands word-at-a-time: interior
   fully-covered words are plain `0xFFFFFFFF` stores (a word strictly inside a
   run is owned exclusively by that run), only partial boundary words use
   `atomicOr`. Per run: `~2 atomics + O(L/32) stores` instead of `O(L) atomics`;
   output bit-identical. Ships with `bench_decompress_runs` and a before/after
   measurement guide.

2. **`enumerate_runs()`.** Extracts every run from RUN containers, converts each
   to an absolute half-open `[start, end)` range, and coalesces runs that touch
   across 64K container boundaries into maximal ranges (a 10M-element region
   stored as ~153 full RUN containers comes back as one range). Pipeline:
   count runs → exclusive scan → emit sorted intervals → mark range starts →
   inclusive scan → scatter. This is the foundational schedule-builder: each
   range is a GEMM tile.

`enumerate_runs` came first because the run path is the cleanest, highest-
leverage win (contiguous, zero-waste, tensor-core-friendly), and because the
`enumerate → schedule` pattern it establishes is what the array/CSR and bitmap
paths will follow.

## Caveats

- **Per-query distinct filters** fragment GEMM batching — each query's eligible
  set differs, so one tiling cannot be shared across a query batch. The clean
  wins assume a filter shared across a batch of queries.
- **The dense-uniform GEMM has real virtues** — peak tensor-core utilization,
  zero dispatch overhead, simplicity. "Doing less work" is not "faster" if the
  less-work path fragments into many small, awkward GEMMs. The design needs a
  **selectivity/shape gate**: fall back to the monolithic dense GEMM + bitset
  mask when the filter is dense or its containers are tiny and numerous. This
  is `PROMOTE_AUTO`'s cache-aware philosophy extended from storage format to
  search schedule.
- **`bitmap.negated`** — when the bitmap stores the complement, the runs are the
  *excluded* regions; `enumerate_runs` currently returns stored runs as-is and
  the caller must account for this.
- **Provenance:** the `run-optimizations` code was authored without a local
  CUDA toolchain and is **uncompiled**. Tonight's build is the first real
  validation.

## Next steps

1. **Range-driven search driver** — iterate `RunRanges`, one (sub-)GEMM per
   range (or grouped GEMM for many small runs), remap local→global IDs, feed
   each into the progressive top-k merge.
2. **Array containers → CSR** — `enumerate_ids` already produces sorted IDs;
   wire them as CSR column indices into gather+dense-GEMM or SDDMM.
3. **Density-driven dispatch + selectivity gate** — pick range-GEMM /
   gather-CSR / block-GEMM-mask per container by density, with a dense fallback.
4. **`negated` handling.**
5. **End-to-end benchmark** — run-driven vs dense-masked filtered search.

## Tonight's benchmark plan

The schedule-driven *search* is not built yet, so tonight validates the two
pieces that landed:

1. **Build + correctness.** Configure, build all targets (also the `-Werror`
   lint gate), run `ctest --output-on-failure`. Confirm the new
   `test_enumerate_runs` (8 cases: coalescing, gaps, container boundaries)
   passes, and that nothing else regressed.
2. **Decompress optimization — before/after.** Follow
   `bench/DECOMPRESS_RUN_BENCHMARK.md`: stash `src/decompress.cu`, build, run
   `bench_decompress_runs` → baseline; restore, rebuild, run → optimized;
   compare with `compare.py`. Expect the run-decompress speedup to grow with
   run length (~2× at short runs toward ~25–32× for long runs), and the
   array/bitmap rows of `bench_decompress` to be unchanged (the control).
3. Report median **and** stddev for every row.
