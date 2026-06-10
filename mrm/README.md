# cu_roaring MRM — Masked Roaring Matrix

The transpose-with-payload of `k ≤ 64` roaring bitmaps: one roaring-shaped
structure keyed by row id whose per-id payload is a k-bit query-membership
mask (`uint64_t`). Equivalent views: a compressed CSC of the `k×q` SDDMM
sparsity pattern; a bit-sliced index with a roaring-compressed key dimension.
This is the core data structure for multi-query multi-filter prefiltered
brute-force search (see the MRM design doc; benchmark harness lives in the
cuVS checkout under `cpp/bench/prims/core/bench_mrm_baselines.cu`).

Per 64K chunk, the merged container is one of:

| Type | Layout | Chosen when |
|---|---|---|
| `ARRAY_MASKED` | sorted `uint16 ids[n]` + `uint64 masks[n]` (SoA) | union < 4096 |
| `BITMAP_MASKED` | 8 KB union bitmap + `uint64 masks[n]` (rank-indexed) | union ≥ 4096 |
| `RUN_MASKED` | `uint16 starts[n]` + `uint16 lens[n]` + `uint64 masks[n]` | constant-mask runs clearly smaller than the array form |

## Construction

`mrm_build(bitmaps, metas, k, stream)` is a **single kernel launch with no
device→host syncs**: one CTA per chunk ORs all source containers into an 8 KB
shared-memory union bitmap, builds a per-word popcount rank table, emits the
union payload, scatters per-lane mask bits via rank lookup, then detects
maximal constant-mask runs in-block and rewrites the region as `RUN_MASKED`
when the run form is at most half the array bytes. Capacity comes from
host-side per-container cardinality bounds (known at filter construction) —
deliberate overallocation instead of a count kernel + sync. Exact union
cardinalities live in device-side descriptors; search kernels never need
them on the host.

`HostBitmapMeta` is the per-bitmap container metadata (keys/types/offsets/
cardinalities). Pass it from your filter-construction pipeline if you have
it; `download_meta()` fetches it (k small D2H syncs) if you don't.

## API

```cpp
#include <cu_roaring_mrm/mrm.cuh>
using namespace cu_roaring::mrm;

Mrm m = mrm_build(bitmaps, metas, k, stream);   // or mrm_build(bitmaps, k)
MrmView v = m.view;                              // pass by value to kernels
// v.descs[c] -> {key, type, n, byte_offset}; payload via v.array_ids(d), ...
mrm_free(m, stream);
```

`mrm_decode()` expands back to `(id, mask)` pairs (test/debug only).

## Build & test

Part of the main CMake build (`CU_ROARING_BUILD_MRM=ON` by default):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target test_mrm bench_mrm_build -j
ctest --test-dir build -R test_mrm --output-on-failure
./build/mrm/bench_mrm_build
```

Tests are CPU-reference property tests over every input container-type
pairing (array/bitmap/run), boundary ids, empty lanes, full chunks, 64
lanes, and the shared-filter case that must collapse to `RUN_MASKED`.

## Status / next

- [x] Phase 2: construction + tests + construction-cost bench
- [ ] Phase 3: fused compute kernels (RUN_MASKED slab GEMM, BITMAP_MASKED
      dense tile, ARRAY_MASKED data-major scan) + top-k reduce
- [ ] k > 64 via query tiles (Phase 4 grouping)
