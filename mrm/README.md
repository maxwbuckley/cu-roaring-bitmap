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

## Search kernels: measured status (both lose to the SDDMM MVP)

`mrm_search` (v1 lane-major scan) and `mrm_search_tile` (v2 dense 32×64
smem tile) are oracle-correct but slower than the batched-CSR + cusparse
SDDMM + sparse select_k pipeline in every benchmarked config.

Ablation attribution for the tile kernel at 1M rows / s=0.1 / 64 shared
lanes (23.4 ms total; `MRM_TILE_ABLATE=1..4` skips phases cumulatively —
ncu is unavailable on this WSL box, ERR_NVGPUCTRPERM):

| phase | ms | share |
|---|---|---|
| in-CTA per-lane merge (64 serial selections, 2 barriers each) | 11.3 | 48% |
| final merge kernel (1 thread/lane over 1024 lists) + launch overhead | 6.9 | 30% |
| cooperative row loads into smem | 2.6 | 11% |
| dot products + top-k inserts | 2.1 | 9% |
| serial tile producer | 0.4 | 2% |

**~78% is top-k reduction, not math.** v3 requirements, in order: (1)
collapse the reduction — far fewer segment lists (size CTAs to fill the
GPU, not 64×chunks), parallel warp-level merges in-CTA, one warp (not one
thread) per lane in the final merge; (2) occupancy — 50 KB smem caps the
kernel at 2 CTAs/SM and every phase is latency-bound (row loads run 20x
over the DRAM floor); shrink smem (queries fit L2 — consider not staging
them) and double-buffer tiles; (3) only then revisit the FMA micro-kernel
(register-block the query operand) and WMMA on RUN slabs.

## Phase 3 v3 (reduction collapse + occupancy) — implemented, still loses

v3 applied the ablation work order: queries no longer staged in smem
(~17 KB/CTA, ~3x occupancy), register-resident top-k (fully unrolled
predicated bubble insert), in-CTA reduction as a per-warp `__shfl` tree
(warp g exclusively owns lanes 8g..8g+7 — zero barriers, zero smem),
final merge one warp per lane, and CTA count sized to ~128 (measured
optimum; `MRM_TILE_SEGS` overrides).

Result: 2–14x faster than v2 on uniform/zipf shared-filter configs
(best: 10M/s=0.001/m=64 25.7→1.85 ms; 10M/s=0.1/m=64 140→43.9 ms, now
1.68x from P1), oracle-correct throughout — but **0/48 wins vs P1**.
Contiguous-clustered configs regressed under the new segment default
(few dense chunks want many segments; sparse many-chunk workloads want
few): the right segmentation depends on per-chunk union cardinality,
which only the device knows — a host heuristic cannot pick one value for
all shapes. Proper fix: device-side adaptive work partitioning
(persistent CTAs / work stealing), plus ncu once GPU counters are
enabled (Windows toggle) for the remaining latency-bound compute.

## Status / next

- [x] Phase 2: construction + tests + construction-cost bench
- [x] Phase 3 v1 (scan) + v2 (dense tile) + v3 (occupancy + shuffle
      reduction): correct, benchmarked; best-case gap to the SDDMM MVP
      now 1.35–1.7x, but no config where the fused kernel wins
- [ ] Phase 3 v4: device-side adaptive segmentation + ncu-guided compute
- [ ] k > 64 via query tiles (Phase 4 grouping)
