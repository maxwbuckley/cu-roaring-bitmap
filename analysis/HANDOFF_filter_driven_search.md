# Filter-driven search — handoff

*Index + state pointer. The substantive report is in
[`filter_driven_search_results.md`](filter_driven_search_results.md);
the design rationale is in [`filter_driven_search.md`](filter_driven_search.md).
This file is the pick-up-where-we-left-off summary.*

## TL;DR

A complete container→schedule dispatch for roaring-filtered brute-force
vector search lives on branch **`run-optimizations`** (pushed to
`origin/run-optimizations` on `maxwbuckley/cu-roaring-bitmap`). Verified
on RTX 5090. **Search-only is 14–27× faster than cuVS `brute_force` +
raft `bitset_filter`** across the synthetic configs; **end-to-end on
real YFCC-10M is 1.24× (unsorted) / 1.39× (sorted) faster than cuVS**
with `recall@10 = 1.000`. Two more optimisation levers are identified
and unimplemented; see "Next" below.

## Branch state (GitHub)

```
b82b9ce filtered_search: skip enumerate_runs + persistent scratch -> YFCC beats cuVS
df46f1e YFCC: GPU-side construction via upload_from_device_bitset cuts build 2.6x
e420e3e Add YFCC sorted-by-tag-tuple result: tag bitmaps shrink 4000-7000x ...
d593b9e Add YFCC-10M unsorted result: per-query schedule build dominates
51e19ef Add synthetic-generator sweep across 5 filter distributions
6b317b6 Expand filter-driven search writeup with clustered + D/Q sweeps + cuVS
e1eed24 Commit files referenced by the branch but never tracked
9d9390a Add roaring-filtered search with container-type dispatch
09443a8 Add filter-driven search design notes              (pre-existing)
2385ac9 Add enumerate_runs: coalesced run ranges...         (pre-existing)
de025ad Optimize RUN-container decompress...                (pre-existing)
```

Everything ≥ `9d9390a` was added during the filter-driven-search work.

## Files added on the branch

| location | role |
|---|---|
| `include/cu_roaring/detail/filtered_search.cuh` | API: `GemmTask`, `SearchSchedule`, `build_schedule`, `roaring_filtered_search`, `dense_filtered_search`. |
| `src/filtered_search.cu` | Implementation: container dispatch, cuBLAS executor, CUDA-graph cache, two-pass top-k, persistent build-scratch. |
| `test/test_filtered_search.cu` | 7 correctness cases (every dispatch path + negated + fragmentation-fallback), exhaustive CPU reference, both eager and graph-replay paths. |
| `bench/bench_filtered_search.cu` | Schedule-driven vs dense-masked sweeps (selectivity / batch / dim / fragmentation). |
| `analysis/filter_driven_search_results.md` | The detailed writeup with all six results. **Start here.** |
| `analysis/bench_*.json` | Raw JSON for every result (cuVS comparison, synth generators, YFCC unsorted+sorted). |
| `tools/yfcc_sort_by_tags.py` | Lex-by-tag-tuple permutation preprocessor for the sorted YFCC variant. |

## cuVS-side files (uncommitted in `~/Development/cuvs`)

Three files live in your `cuvs` working tree but were *not* committed,
because the cuVS `cpp/bench/prims/core/CMakeLists.txt` already has
~150 lines of your own in-progress local-build-tree path rework that I
shouldn't fold into a roaring commit:

```
cpp/bench/prims/core/bench_schedule_driven_roaring.cu  (NEW — synthetic scattered/clustered + D/Q sweeps)
cpp/bench/prims/core/bench_synth_filters.cu            (NEW — loads roaring-benchmark .bin filters)
cpp/bench/prims/core/bench_yfcc_search.cu              (NEW — real YFCC, per-query filters, host vs GPU build path)
cpp/bench/prims/core/CMakeLists.txt                    (MODIFIED — adds the three targets above; also has your unrelated WIP)
```

All three benches **build and run** as committed locally. They link
`cu_roaring_bitmap.a` built at **CUDA 12.4 / sm_89** (to device-link with
cuVS's own toolchain — cuVS uses cuda-12.4 which doesn't support sm_120).

To commit the cuVS-side files cleanly: split your `CMakeLists.txt` WIP
into its own commit first, then add my block as a separate commit (or
`git add -p` to stage just the bench-target block).

## Ephemeral state (was in `/tmp`, may not survive)

| path | purpose |
|---|---|
| `/tmp/cu-roaring-ro/` | git worktree on `run-optimizations`. Recreate with `git worktree add /tmp/cu-roaring-ro run-optimizations`. |
| `/tmp/cu-roaring-ro/build89/` | sm_89 / CUDA 12.4 build of `cu_roaring_bitmap` (needed for cuVS integration). |
| `/tmp/yfcc_sorted/` | Permuted YFCC base vectors + tag bitmaps. Regenerate with `python3 tools/yfcc_sort_by_tags.py`. |
| `/tmp/yfcc_sorted/perm.bin` | Old→new id permutation. |

## Key numbers (RTX 5090, branch HEAD `b82b9ce`)

**vs cuVS bitset, synthetic generators** (`bench_synth_filters`, N=10M, D=128, Q=64):

| sel | uniform | clustered | multi_tenant | power_law | temporal |
|---|---|---|---|---|---|
| 0.01% | 17× | 22× | 21× | 21× | 19× |
| 0.1%  | 21× | 22× | 22× | 23× | 21× |
| 1%    | 11× | 11× | 11× | 11× | 11× |
| 5%    | 20× | 12× | 12× | 21× | 19× |
| 10%   | 27× | 13× | 12× | 25× |  7× |
| 25%   | 2.8× | 1.5× | **12×** | 3.6× | 1.3× |
| 50%   | 0.45× | 0.78× | **12×** | 3.6× | 1.1× |

Recall = 1.000 everywhere.

**YFCC-10M (real per-query filters, 256 sampled queries)**:

| | unsorted | sorted |
|---|---|---|
| cuVS bitset            | 0.76 ms | 0.82 ms |
| roaring search only    | 0.049 ms (**15.5×**) | 0.049 ms (**16.6×**) |
| roaring build (host)   | 2.60 ms | 2.95 ms |
| roaring build (GPU+opts1+2) | **0.55 ms** | **0.53 ms** |
| roaring end-to-end (GPU+opts1+2) | **0.61 ms (1.24× cuVS)** | **0.59 ms (1.39× cuVS)** |
| recall@10              | 1.000 | 1.000 |

## Implementation summary

`build_schedule()` walks the roaring containers and emits a
`SearchSchedule` (host vector of `GemmTask` + at most one device gather
buffer). Dispatch table:

| container | path | notes |
|---|---|---|
| absent | skip | 0 cost |
| run ≥ 64K wide | direct range-GEMM on `db[start:end]` | no copy |
| run < 64K | gathered | width gate |
| array | gathered | |
| bitmap, density ≥ 50% | range + bitset mask | mask = container's own words |
| bitmap, density < 50% | gathered (bit-scan) | |
| `negated` | complement schedule | absent blocks → full ranges, excluded → range + inverted mask |

`roaring_filtered_search()` executes each task as cuBLAS GEMM → optional
mask → column-parallel two-pass top-k merged into a running per-query
top-k. The whole task sequence is **CUDA-graph-captured and replayed**
on the second call with a matching signature.

`dense_filtered_search()` is the same executor over one `[0,N)` range
with the decompressed filter bitset as the mask (the "expand-to-bitset"
baseline from the design doc).

## Build / reproduce

```bash
# Standard sm_120 build for tests and bench_filtered_search:
cd cu-roaring-bitmap
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target test_filtered_search bench_filtered_search -j
ctest -R filtered_search --output-on-failure          # 7/7 expected
./build/bench/bench_filtered_search                   # internal sweeps

# sm_89 / CUDA 12.4 build for cuVS integration:
cmake -S . -B build89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build89 --target cu_roaring_bitmap -j

# cuVS benches (after copying the three .cu's and the CMakeLists block):
cd ~/Development/cuvs/cpp/bench/prims/core/build
cmake .. -DSD_ROARING_DIR=$(realpath ../../../../../cu-roaring-bitmap) \
         -DSD_ROARING_LIB=...build89/libcu_roaring_bitmap.a \
         -DSD_CROARING_LIB=...build89/third_party/CRoaring/src/libroaring.a
make -j bench_schedule_driven_roaring bench_synth_filters bench_yfcc_search

# Set LD_LIBRARY_PATH to libcuvs.so + librmm.so when running:
LD_LIBRARY_PATH=~/Development/cuvs/cpp/build:~/Development/cuvs/cpp/build/_deps/rmm-build \
    ./bench_schedule_driven_roaring

# YFCC: needs base.10M.u8bin (from big-ann-benchmarks/data/yfcc100M)
# and yfcc_data/{queries.bin, tags/} (exported via bench/yfcc_export.py).
YFCC_DATA=cu-roaring-bitmap/bench/yfcc_data \
    YFCC_VEC=big-ann-benchmarks/data/yfcc100M/base.10M.u8bin \
    YFCC_SAMPLE=256 \
    LD_LIBRARY_PATH=... ./bench_yfcc_search

# Sorted YFCC: regenerate the layout first
python3 tools/yfcc_sort_by_tags.py
YFCC_VEC=/tmp/yfcc_sorted/base.10M.u8bin \
    YFCC_TAG_DIR=/tmp/yfcc_sorted/tags \
    ... ./bench_yfcc_search

# Synthetic 5-generator sweep:
cd ~/Development/roaring-benchmark
python3 -m roaring_bench sweep -n 10000000 --num-trials 1 \
    --output-dir output/sweep_10M --save-bitmaps --skip-access
SYNTH_FILTER_DIR=$PWD/output/sweep_10M/bitmaps SYNTH_N=10000000 \
    LD_LIBRARY_PATH=... ./bench_synth_filters
```

## Build-environment gotchas

- The cuVS bench toolchain is **CUDA 12.4** (path
  `/usr/local/cuda-12.4/bin/nvcc`, see
  `cuvs/cpp/bench/prims/core/build/CMakeCache.txt`). CUDA 12.4 does not
  support sm_120, so all cuVS-linked builds are sm_89 + PTX-JIT to sm_120
  at runtime on the 5090. `cu_roaring_bitmap.a` for cuVS linkage must be
  built at sm_89 (the `build89/` tree above).
- The repo's own tests/benches use **CUDA 13.2 / sm_120** natively; that
  build is the `build/` tree.
- Two pre-existing repo build bugs (already fixed on the branch in commit
  `9d9390a` and `e1eed24`): (a) `CMakeLists.txt` linked `CUDA::cudart`
  (shared) while separable compilation pulled `-lcudart_static` →
  segfault from double-linked cudart on every test/bench (fix:
  `CUDA::cudart_static` throughout). (b) committed code referenced
  uncommitted files (`upload_pool.hpp`, four `bench_*.cu` files) — they
  are committed now.

## What's left — next-step optimisations, ordered by payoff

1. **Search-level selectivity gate (≈5 lines).** When
   `filter.total_cardinality / n_rows > ~0.30`, route to
   `dense_filtered_search` directly. Makes the `< 1.0×` rows in Result 1
   clamp to ≥ 1.0× (parity with cuVS) — never lose to bitset for any
   selectivity. Doesn't help YFCC (its sel is sub-1% throughout); helps
   "shared filter across batch" workloads at high sel.

2. **GPU-side `run_optimize`.** The `upload_from_device_bitset` path
   emits only ARRAY/BITMAP. Add a per-block transition-count scan + emit
   RUN when low (single extra branch in the existing upload kernel), or
   a separate `run_optimize_gpu(GpuRoaring&)` post-pass. Sorted-layout
   YFCC's tag bitmaps compress 4000–7000× via host `run_optimize` but
   that benefit doesn't currently propagate to the GPU search. With
   GPU-side run detection the direct-range dispatch should fire on
   sorted YFCC and push the win further.

3. **Pre-uploaded tag bitmaps + GPU-side `multi_and`.** For YFCC's 7910
   query-relevant tags, upload all of them once at startup
   (memory: tens of MB on sorted layout, hundreds on unsorted). Per
   query: `cu_roaring::multi_and(query_tags…)` → already a `GpuRoaring`.
   Skips both the bitset materialisation *and* the
   `upload_from_device_bitset` step — should cut per-query build below
   0.2 ms.

4. **Faster top-k.** `partial_topk_kernel` is the biggest single GPU
   cost at fragmented schedules (Result 2's nsys profile: 42% of GPU
   time at 10M/50% scattered). Per-thread top-k array lives in a 1 KB
   stack frame (local memory) at ~31% occupancy. Fix: warp-distributed
   register-resident top-k, or just adopt raft's `warpsort`. Closes the
   remaining high-sel fragmentation gap.

5. **Kernel fusion in the construction pipeline.** Popcount-per-block,
   classify, write metadata, emit container data — all touch the same
   per-block data once. Collapsing them into one cooperative-group
   kernel cuts ~5 launches. Diminishing return after #1–#4.

Order of impact: **#1 ships almost-no-regressions for synthetic workloads
with one if-statement; #2 + #3 unlock further YFCC gains; #4 closes the
50%-selectivity fragmentation cliff.**

## Open questions / things I deliberately didn't do

- **Cross-platform / non-WSL2 numbers.** Everything here is one
  RTX 5090 on WSL2. Per-kernel launch latency on WSL2 is 2–4× native
  Linux; some of the per-query build cost is WSL2 overhead. Worth
  re-measuring on bare Linux before publishing absolute QPS numbers.
- **The popc-kernel-per-search finding** in `knn_brute_force.cuh:629`
  (cuVS recomputes filter cardinality on every brute-force search via
  `bitset_view::count(res)`) is documented in the writeup and is a
  small but real per-query cost cuVS pays that we don't.
- **Pushing the cuVS-side files** — left for you to do, because of the
  unrelated WIP in `cuvs/cpp/bench/prims/core/CMakeLists.txt`.

## Resume checklist

If picking this up cold:

1. `git fetch origin && git checkout run-optimizations && git pull`.
2. Verify HEAD is `b82b9ce` (or whatever's newest).
3. Read [`filter_driven_search_results.md`](filter_driven_search_results.md)
   end-to-end.
4. Rebuild sm_89 lib if `/tmp/cu-roaring-ro/build89/` is gone (see
   "Build / reproduce" above).
5. The first optimisation to ship is the **selectivity gate (#1 above)**
   — it's a 5-line change to `roaring_filtered_search()` in
   `src/filtered_search.cu`, and would let you publish "schedule-driven
   never loses to cuVS bitset" cleanly.
