# enumerate_runs — filter-driven search via run ranges

## Goal

A dense bitset filter is a **post-hoc mask**: a brute-force search computes the
full `queries × N` distance matrix and then discards the rows whose filter bit
is 0. The filter only helps in the top-k stage — the GEMM is full regardless of
selectivity.

Roaring lets the filter become the **loop bounds of the GEMM instead**. A RUN
container stores a contiguous range of eligible IDs, and a contiguous range of
eligible IDs is a contiguous **slab of database rows** — i.e. a GEMM column
tile. If a 200M-row filter is "10 contiguous runs covering 20M rows," a
run-aware search does **10 sub-GEMMs over 20M rows** instead of one GEMM over
200M: ~10× less compute and ~10× less HBM traffic, and the 180M filtered-out
rows are never read. Work becomes **O(n_runs) / O(cardinality)** instead of
O(universe).

`enumerate_runs()` is the first concrete piece of that path: it turns the RUN
containers of a `GpuRoaring` into a device array of coalesced `[start, end)`
ranges that a search driver can consume directly as tiles.

## What this delivers

```cpp
#include <cu_roaring/detail/enumerate_runs.cuh>

cu_roaring::RunRanges rr = cu_roaring::enumerate_runs(gpu_bm, stream);
// rr.ranges : device IdRange[rr.count], sorted, disjoint, non-adjacent
// each IdRange is a half-open [start, end) -> GEMM tile: offset start, width end-start
cu_roaring::free_run_ranges(rr);
```

**Algorithm** (`src/enumerate_runs.cu`): count runs per container → CUB
exclusive scan (per-container output offsets) → emit absolute half-open
intervals, one block per container → mark range starts → CUB inclusive scan
(range index per interval) → scatter coalesced ranges.

The coalescing step is the point: a run is stored *per 64K key block*, so a
10M-element contiguous region is physically ~153 separate full RUN containers.
Because the emitted intervals are globally sorted and disjoint, two intervals
belong to the same logical range iff `interval[i].lo == interval[i-1].hi`
(half-open adjacency) — so those 153 containers collapse back into one range.

**Contract.** `enumerate_runs` reads **RUN containers only**. For bitmaps built
with `roaring_bitmap_run_optimize()`, every contiguous region of more than a
few elements is stored as RUN container(s), so this captures them. ARRAY and
BITMAP containers hold sparse/scattered data by construction — those go through
`enumerate_ids()` / `to_csr` (CSR column indices) instead. If `bitmap.negated`
is true the stored runs describe the *complement*; the caller must account for
that. `enumerate_runs` synchronizes the stream internally (it reads run/range
counts back to host) — it is a one-time schedule-build step.

## Files

| File | Role |
|---|---|
| `include/cu_roaring/detail/enumerate_runs.cuh` | `IdRange` / `RunRanges` API |
| `src/enumerate_runs.cu` | count → emit → mark → scatter pipeline |
| `test/test_enumerate_runs.cu` | 8 correctness tests (coalescing, gaps, boundaries) |

## How to test (today, on a GPU box)

Requires CUDA 12.4+, CMake 3.25+, an SM_89+ GPU. This was authored without a
local CUDA toolchain — **it has not been compiled**; first build will be the
real check.

```bash
cd cu-roaring-bitmap
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build . --target test_enumerate_runs -j
ctest -R enumerate_runs --output-on-failure
```

The suite (`EnumerateRunsTest`) checks:

| Test | Verifies |
|---|---|
| `EmptyBitmap` | empty bitmap → `count == 0`, `ranges == nullptr` |
| `NoRunContainers` | scattered ARRAY-only input → `count == 0` (RUN-only contract) |
| `SingleRun` | one run in one container → one exact range |
| `MultipleRunsOneContainer` | three gapped runs in one container → three ranges, **not** merged |
| `CoalesceAcrossContainers` | `[0, 200000)` stored as 4 RUN containers → **one** coalesced range |
| `AdjacentContainersExactBoundary` | range crossing exactly the 65536 boundary → one range |
| `NoCoalesceAcrossGap` | runs in non-adjacent containers (empty container between) → **not** merged |
| `MixedRunsAndGapsAcrossContainers` | a cross-container range + a separate range → two ranges |

Every test first asserts the input is all-RUN (`n_run_containers == n_containers`,
or `== 0` for `NoRunContainers`) — so if CRoaring's `run_optimize` ever
classifies a container differently, the test fails loudly at the assert rather
than silently measuring the wrong path.

The build is also the lint gate: the repo compiles every target with
`-Werror -Wall -Wextra -Wshadow …` (CXX) and `nvcc --Werror all-warnings`, so a
clean build of `test_enumerate_runs` and `cu_roaring_bitmap` confirms the new
code passes the strict warning policy.

## Status & next steps

**Done:** `enumerate_runs` — RUN containers → coalesced device range list.

**Not done (the rest of the filter-driven search path):**

1. **Range-driven GEMM driver** — iterate `RunRanges`, run one (sub-)GEMM per
   range on `database[start:end, :]`, remap local→global IDs (`+start`), feed
   each range into a progressive top-k merge. Big fat runs → per-run GEMM;
   many tiny runs → batch/grouped GEMM or fall back.
2. **Reuse the progressive top-k threshold.** Each range is a tile, so the
   tile-0-builds-a-threshold / later-tiles-filter-against-it scheme (à la the
   Exa `optimized_brute_force` progressive path) applies directly.
3. **Array containers → CSR.** Scattered IDs via `enumerate_ids` → CSR column
   indices → gather+dense-GEMM or SDDMM.
4. **`negated` handling** — interpret runs as excluded regions when the bitmap
   stores the complement.
5. **Benchmark** — `enumerate_runs` cost (should be negligible: O(n_runs)) and
   the end-to-end run-driven vs dense-masked search comparison.
