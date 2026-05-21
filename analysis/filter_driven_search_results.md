# Filter-driven search — implementation & experiment

*Implements the container→schedule dispatch sketched in
[`filter_driven_search.md`](filter_driven_search.md): a roaring-filtered
brute-force vector search where the filter becomes the loop bounds of the GEMM
instead of a post-hoc mask. Measured on an RTX 5090 (sm_120), 2026-05-22.*

## What was built

A complete container-type dispatch + search executor in `src/filtered_search.cu`
(API: `include/cu_roaring/detail/filtered_search.cuh`).

| Container in the filter | Schedule | GEMM cost |
|---|---|---|
| absent (no set bits in the 64K block) | **skipped** | 0 |
| run, width ≥ 64K | **direct range GEMM** on `db[start:end]`, no copy | run width |
| run, width < 64K | **gathered** into the compact GEMM (see shape gate) | cardinality |
| array (sparse scattered) | **gather** rows → one compact GEMM | cardinality |
| bitmap, density ≥ 50% | **range GEMM + bitset mask** (the container's own words) | 64K |
| bitmap, density < 50% | **gather** (bit-scan → ids) | popcount |
| `negated` bitmap | complement schedule: absent blocks → full ranges, excluded containers → range + inverted mask | universe − excluded |

`build_schedule()` walks the roaring containers and emits a `SearchSchedule` —
direct range tasks + one gather buffer. `roaring_filtered_search()` runs each
task: cuBLAS GEMM of the queries against that task's database columns → optional
mask → column-parallel two-pass top-k merged into a running per-query top-k.
The whole task sequence is **CUDA-graph-captured and replayed** (a filtered
search issues O(tasks) kernels; per-launch host overhead would otherwise
dominate). The dense baseline is the same executor over one `[0,N)` range with
the decompressed filter bitset as the mask — the design doc's "option 1".

## Correctness

`test/test_filtered_search.cu` — 7 cases (run / array / dense-bitmap /
sparse-bitmap / mixed / negated / fragmentation-fallback). Each builds a filter
that lands in one dispatch regime, asserts the schedule took that path, and
verifies both the schedule-driven search and the dense baseline against an
exhaustive CPU top-k reference — twice, so the CUDA-graph replay path is also
covered. All pass. In the cuVS comparison below, schedule-driven results match
the production cuVS bitset path at **recall@10 = 1.000** for every config.

## Result 1 — vs cuVS production bitset filter

`bench_schedule_driven_roaring.cu`, wired into the cuVS benchmark tree:
cuVS `brute_force::search` + raft `bitset_filter` (the production dense-bitset
prefilter) vs `cu_roaring::roaring_filtered_search`. Inner-product top-10,
D=128, 64 queries, random-scattered filter, median of 30 interleaved A/B reps.

| N | selectivity | cuVS bitset | roaring-schedule | speedup |
|---|---|---|---|---|
| 100K | 1% | 1.28 ms | 0.058 ms | **22.1×** |
| 100K | 5% | 2.54 ms | 0.075 ms | **34.0×** |
| 100K | 25% | 0.89 ms | 0.229 ms | 3.9× |
| 100K | 50% | 0.86 ms | 0.369 ms | 2.3× |
| 1M | 1% | 2.17 ms | 0.116 ms | **18.8×** |
| 1M | 5% | 4.38 ms | 0.376 ms | **11.7×** |
| 1M | 25% | 5.02 ms | 1.14 ms | 4.4× |
| 1M | 50% | 4.88 ms | 3.21 ms | 1.5× |
| 5M | 1% | 4.38 ms | 0.374 ms | **11.7×** |
| 5M | 5% | 15.68 ms | 1.14 ms | **13.8×** |
| 5M | 25% | 12.79 ms | 3.55 ms | 3.6× |
| 5M | 50% | 13.19 ms | 34.48 ms | 0.38× |
| 10M | 1% | 6.96 ms | 0.655 ms | **10.6×** |
| 10M | 5% | 31.76 ms | 1.65 ms | **19.2×** |
| 10M | 25% | 18.62 ms | 6.49 ms | 2.9× |
| 10M | 50% | 18.62 ms | 37.28 ms | 0.50× |

**Typical case 10–20× below ~10% selectivity, 3–4× at 25%, crossing to a loss
above ~40%.** Schedule-driven search makes GEMM work scale with filter
cardinality; the cuVS bitset path runs the full Q×N GEMM and masks. At ≥50%
selectivity the gather copies most of the database — more traffic than the
dense masked GEMM — so the dense path wins; this is the documented next step
(a search-level selectivity gate, see below).

## Result 2 — internal sweep (schedule-driven vs dense-masked)

`bench_filtered_search.cu`, N=2M, inner-product top-10, median of 30 reps
(cv < 2.5%), CUDA-graph replay, schedule pre-built.

| Sweep | Schedule-driven speedup vs dense-masked |
|---|---|
| Selectivity (Q=32, 8 runs): 0.1 / 1 / 5 / 25 / 50 / 90 % | **33.8× / 17.4× / 4.6×** / 0.93× / 0.61× / 0.44× |
| Batch size (5%, 8 runs): Q = 1 / 8 / 32 / 128 | 10.4× / 7.4× / 4.6× / 5.0× |
| Dimension (5%, Q=32): D = 64 / 128 / 768 | 3.9× / 4.6× / **11.2×** |
| Fragmentation (10%, Q=32): 8 / 512 / 16384 / 90000 runs | 3.0× / 3.1× / 4.4× / 4.8× |

## Optimization analysis

Profiled with `nsys` + `ptxas -v`. Two findings, one fixed in this branch.

**1. Shape-gate mis-calibration — found by the benchmark, fixed.** The first
implementation gated the gather fallback on *mean* range width (< 256). The
fragmentation sweep then exposed a pathological config: 512 runs of ~390
columns each passed the gate and issued **512 separate skinny GEMMs**, running
**10× slower than the dense baseline** (0.10×). cuBLAS runs a 390-column GEMM
at a tiny fraction of peak. The fix is a **per-range partition**: a range keeps
its own direct (no-copy) GEMM only if it is ≥ 64K columns wide — wide enough to
saturate the GPU — otherwise it joins the single compact gather GEMM. A
schedule may mix a few huge direct ranges with a gather of the rest. This is
the single highest-leverage optimization: it lifted 0.1%-selectivity from 5.9×
to **33.8×** and turned the 512-run regression into a 3.1× win.

**2. Top-k is local-memory-latency bound (remaining, ~40% of GPU time).**
`partial_topk_kernel` keeps each thread's running top-k in a 32-slot array that
is dynamically indexed (insertion shifts), so ptxas places it in a 1 KB
per-thread stack frame (local memory) — 0 register spills, but every insert
touches local memory at ~31% occupancy, too low to hide L2 latency. *Fix:* a
register-resident or warp-distributed top-k (each lane owns k/32 of the list,
bitonic merge over shuffles).

**3. High-selectivity crossover.** Above ~40% selectivity the gather copies
most of the database and schedule-driven loses to the dense masked GEMM. A
search-level **selectivity gate** — fall back to `dense_filtered_search` when
`cardinality / N` exceeds ~0.3 — would make the search never lose. Not yet
implemented; the data above shows exactly where the crossover sits.

Schedule build (`enumerate_runs` + dispatch kernels) is O(n_containers+n_runs),
sub-millisecond, amortised across the query batch — not a bottleneck.

## Files

| File | Role |
|---|---|
| `include/cu_roaring/detail/filtered_search.cuh` | `GemmTask` / `SearchSchedule` API |
| `src/filtered_search.cu` | dispatch, executor, CUDA-graph cache, top-k |
| `test/test_filtered_search.cu` | 7 correctness cases vs CPU reference |
| `bench/bench_filtered_search.cu` | schedule-driven vs dense sweeps |
| `cuvs/.../bench_schedule_driven_roaring.cu` | vs cuVS bitset filter (lives in the cuVS tree) |

## Reproduce

```bash
cd build && cmake --build . --target test_filtered_search bench_filtered_search -j
ctest -R filtered_search --output-on-failure
./bench/bench_filtered_search                      # internal sweeps

# cuVS comparison: cu_roaring must be built with the cuVS CUDA toolchain
# (CUDA 12.4 / sm_89) so it device-links with the cuVS benchmark tree.
```
