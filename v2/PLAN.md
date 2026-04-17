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

## Not in scope for this rewrite

- Wiring `add_subdirectory(v2)` into the repo-root `CMakeLists.txt`.
- Any cuVS / CAGRA integration.
- Backporting the batch design to v1, or deleting v1.
- The `negated` / complement optimisation.
