# v2 Batch-First Rewrite — Plan

## Goal

Reshape v2 around `GpuRoaringBatch` as the primary type. Every API function
takes or returns a batch; `n_bitmaps = 1` is a degenerate case, not the common
path. Motivation: CUDA throughput is driven by keeping many blocks in flight,
and the single-bitmap API leaves parallelism on the table for any workload
with more than one filter (YFCC per-tag bitmaps, multi-tenant filtering,
batched predicate composition).

## Current state — DO NOT build v2 yet

**Updated to the batch-first shape:**

- `include/cu_roaring_v2/types.cuh` — `GpuRoaringBatch` + `GpuRoaringView`
- `include/cu_roaring_v2/api.hpp` — batch-centric 6-function surface
- `include/cu_roaring_v2/query.cuh` — device `contains` for view + batch
- `README.md` — describes the batch-first design

**Stale (still target the pre-batch single-bitmap API):**

- `src/upload.cpp`
- `src/promote.cu`
- `src/multi_and.cu`
- `src/decompress.cu`
- `test/test_basic.cu`
- `bench/bench_vs_bitset.cu`

The stale sources will not compile against the new headers. Until they are
rewritten, `add_subdirectory(v2)` must stay out of the repo-root `CMakeLists.txt`.
Kernels themselves (promote, fused multi-AND, decompress) are largely reusable;
the host-side orchestration is what changes.

## Remaining work

1. **`src/upload.cpp`** — walk N CRoaring bitmaps, compute the packed layout
   (`container_starts`, `key_index_starts`, per-type pool totals), fill a
   single host staging buffer, one H2D to the device batch allocation. Build
   the host-side mirror of per-bitmap scalars.
2. **`src/promote.cu`** — one kernel, `total_containers` blocks. Each block
   promotes its global container. A small second pass (or thread-0 work inside
   the promote kernel, with a binary search over `container_starts`) populates
   the per-bitmap `key_indices` slices after a `cudaMemsetAsync(… 0xFF)`.
3. **`src/multi_and.cu`** — batch + host `input_indices` → 1-bitmap batch.
   Download the selected bitmaps' `keys` slices (one coalesced D2H per input,
   bounded by `container_starts`), intersect on host, build the pointer table
   referencing `batch.bitmap_data + global_cid * 1024`, single fused AND kernel.
4. **`src/decompress.cu`** — one kernel, `total_containers` blocks. Each block
   resolves its owning bitmap via binary search on `container_starts`, writes
   to `device_bitsets + b * words_each + key * 1024`.
5. **`test/test_basic.cu`** — build a batch with n=4 bitmaps spanning
   ARRAY/BITMAP/RUN mixes, exercise every API call, verify bit-for-bit against
   CPU CRoaring. Include an explicit `n=1` case to confirm the degenerate
   path works.
6. **`bench/bench_vs_bitset.cu`** — sweep `n_bitmaps × selectivity × universe`.
   Batch roaring vs a batch of flat bitsets packed identically (same
   `[n_bitmaps, words_each]` contiguous layout so memory footprint and query
   kernels compare apples-to-apples).

## Locked design decisions

- **CSR indexing.** Bitmap b's containers are `[container_starts[b],
  container_starts[b+1])`. Bitmap b's key_index is `key_indices[
  key_index_starts[b] .. key_index_starts[b+1])`. Keeps per-container arrays
  dense and coalesced across the batch.
- **Local container indices in `key_indices`.** `contains` adds
  `container_starts[b]` to resolve. Preserves the `0xFFFF` sentinel and the
  `uint16_t` width regardless of how many bitmaps are in the batch.
- **No auto-promotion, anywhere.** `promote_batch` is the only function that
  changes container types, and it's explicit.
- **Two `contains` overloads.** `contains(batch, b, id)` is the convenience
  path (2 extra reads for CSR bounds). `contains(view, id)` is the fast path
  (2-read BITMAP hot path, same as the old single-bitmap API). Kernels that
  query one specific bitmap heavily call `make_view()` once at entry and
  pass the view by value.
- **`multi_and` takes a subset by host-side `input_indices`.** `n_inputs=1`
  degenerates to a copy. Grouped `multi_and` (M groups → M outputs in one
  launch) is deferred.
- **`decompress_batch` output is contiguous `[n_bitmaps, words_each]`.**
  Callers pick `words_each` once from `max(host_universe_sizes) / 64`. A
  pointer-array variant is a future option if heterogeneous universes matter.
- **CRoaring ingest only.** `upload_from_sorted_ids` stays out of scope.

## Open questions

- Should `multi_and`'s `input_indices` become a device pointer for fully
  device-resident pipelines (future cuVS integration)? Current call: host
  pointer for this rewrite; revisit once `bench_vs_bitset` quantifies the
  overhead.
- Is `GpuRoaringView` enough for the CAGRA single-filter workload, or do we
  need a warp-broadcast `warp_contains` like v1 had? Defer until there are
  real CAGRA numbers.
- Do we need a `select_batch(source, indices)` primitive to materialise a
  sub-batch from a larger one (e.g. registering 7,910 YFCC tag bitmaps once
  and then building per-query filter sub-batches)? Defer.

## Future benchmark — YFCC-10M with ID reorder

Target dataset: YFCC-10M from the Big-ANN NeurIPS'23 filtered track — 10M ×
192-dim uint8 CLIP vectors plus a sparse 10M × 200K tag matrix (~108M
assignments). Average per-tag selectivity is 0.0054% (~540 docs per tag),
but the distribution is Zipfian; the 7,910 tags that appear in the public
query set are what drives the batch-filter memory story. v1's
`bench/yfcc_export.py` already materialises per-tag ID lists from the
Big-ANN metadata matrix — reusable as-is.

### Why this dataset tests the batch-first design

Flat-bitset filter memory per Q-query batch is `Q × N / 8`. At Q=1,000 and
N=10M that's already 1.25 GB; at N=1B it's 125 GB — past any single GPU.
Roaring swaps the N factor for per-filter cardinality K. A tag with 540
docs is ~1 KB of ARRAY containers; 7,910 such tags are ~8 MB vs ~10 GB of
bitsets. That's the scaling regime the batch API was built for.

### Three-row story we want the benchmark to produce

| Row | Per-tag storage | 7,910-tag batch | Notes |
|---|---|---|---|
| Flat bitset | 1.25 MB (N/8, constant) | ~10 GB | Doesn't scale with N, Q, or clustering |
| Roaring, natural IDs | ~1 KB avg (ARRAY-dominated small tags) | ~8 MB | ~1,000× over bitset on raw ingest |
| Roaring, presorted IDs | medium-to-large tags drop to handfuls of runs | single-digit MB | BITMAP → RUN flip on the Zipfian head |

"Presorted": reorder doc IDs so docs sharing popular tags get consecutive
numbers. Tiny tags already fit in small ARRAY containers and don't benefit
much; the real win is on medium-to-large tags that would otherwise cross
into BITMAP (8 KB) and instead stay RUN (hundreds of bytes). Because the
Zipfian head dominates both memory and query traffic, this matters more
than the averages suggest.

### Reorder strategy

Recursive graph bisection / BP (Dhulipala et al., "Compressing Graphs and
Indexes with Recursive Graph Bisection", KDD 2016) is the canonical
IR-literature method for this reordering.

First pass: **frequency-sorted reorder** — number docs by descending
membership in the top-K tags. Cheap, deterministic, probably captures most
of the compression for the Zipfian head. Full BP is a later polish if the
first-pass numbers aren't enough.

### Benchmark slot

After the v2 `.cu` rewrites land and `bench_vs_bitset.cu` is running, add
`bench/bench_yfcc.cu` with a `--reorder` flag that applies the
frequency-sorted reorder before building the CRoaring bitmaps. The library
itself needs no changes — reorder is a data-pipeline step; the library
just sees CRoaring bitmaps with whatever IDs were assigned.

## Not in scope for this rewrite

- Wiring `add_subdirectory(v2)` into the repo-root `CMakeLists.txt`.
- Any cuVS / CAGRA integration.
- Backporting the batch design to v1, or deleting v1.
- The `negated` / complement optimisation.
