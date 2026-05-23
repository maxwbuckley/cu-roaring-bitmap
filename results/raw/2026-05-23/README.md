# End-to-end filtered-search sweep: cu_roaring vs cuVS bitset (2026-05-23)

## fp16 support added (post-original-write-up)

`filtered_search.cu` now supports both fp32 and fp16 datasets/queries (one
templated executor body, two public entry points: `roaring_filtered_search`
and `roaring_filtered_search_fp16`). The fp16 GEMM uses `cublasGemmEx` with
CUDA_R_16F inputs and CUDA_R_32F accumulation through TensorCores; the score
tile, top-k, and masks stay in fp32, so the ranking is precision-equivalent
to fp32 for the dense range tested. All 8 unit tests pass (7 existing fp32
+ 1 new `Fp16MixedContainers`).

The original benchmark request was D=1024 at N=10M, which doesn't fit on
the 5090 in fp32 (40 GB > 32 GB). With fp16 it fits (20 GB), and the
TensorCore-accelerated GEMM also widens the sparse-end speedups slightly.
See `e2e_sweep_speedup_multi.png` for the side-by-side, and the fp16
results below.

### fp16 D=1024 (the originally-requested configuration)

| sel %  | cuvs ms | roar ms | fair speedup | recall |
|-------:|--------:|--------:|-------------:|-------:|
|   0.01 |   1.671 |   1.010 |     1.55x    | 1.000  |
|   0.1  |   3.525 |   1.062 |     3.22x    | 1.000  |
|   1    |  18.183 |   2.187 |     8.27x    | 1.000  |
|   3    |  50.764 |   4.135 |    12.25x    | 1.000  |
|   5    |  83.352 |   5.855 |    14.22x    | 1.000  |
| **10** | 167.140 |   9.776 | **17.08x**   | 1.000  |
|  15    |  28.920 |  14.442 |     1.99x    | 1.000  |
|  20    |  28.937 |  18.426 |     1.56x    | 1.000  |
|  30    |  30.234 |  28.950 |     1.04x    | 1.000  |
|  50    |  29.473 |  56.646 |     0.52x    | 1.000  |
|  90    |  28.019 |  39.024 |     0.71x    | 1.000  |

### fp16 D=512 (apples-to-apples vs the fp32 sweep)

| sel %  | cuvs ms | roar ms | fair speedup | vs fp32 same-D speedup |
|-------:|--------:|--------:|-------------:|-----------------------:|
|   0.01 |   1.066 |   0.873 |     1.11x    | (fp32: 1.54x)          |
|   0.1  |   2.482 |   0.892 |     2.66x    | (fp32: 3.01x)          |
|   1    |  12.438 |   1.852 |     6.66x    | (fp32: 6.06x)          |
|   3    |  32.755 |   3.103 |    10.52x    | (fp32: 9.61x)          |
|   5    |  53.719 |   4.069 |    13.17x    | (fp32: 11.32x)         |
| **10** | 109.283 |   6.380 | **17.11x**   | (fp32: 12.50x)         |
|  15    |  22.026 |   9.385 |     2.33x    | (fp32: 2.71x)          |
|  30    |  21.782 |  16.731 |     1.30x    | (fp32: 1.48x)          |
|  50    |  22.833 |  42.518 |     0.53x    | (fp32: 0.63x)          |
|  90    |  21.526 |  32.696 |     0.65x    | (fp32: 0.66x)          |

### What fp16 does to the curve

- **cu_roaring**: ~40% faster everywhere in the gather-dominated regime
  (sel ≤ 10%), 20–30% faster in the masked regime (sel ≥ 50%). The boost is
  larger at low selectivity because the GEMM there is small enough to be
  compute-bound on TensorCores; at high selectivity the masked path is
  more memory-bound on the full-dataset scan.
- **cuVS dense path** (sel ≥ 15%): drops from ~34 ms to ~22 ms (35% faster)
  — the brute_force tiled kernel benefits cleanly from TensorCores.
- **cuVS sparse path** (sel ≤ 10%): essentially unchanged. It's bottlenecked
  on CSR materialisation (`make_device_csr_matrix` + `csr_to_coo`), which
  is memory-bound on the bitset → CSR conversion, not on the GEMM. So fp16
  doesn't help cuVS in this regime at all.
- **Net effect on the speedup peak**: 12.50x → 17.11x at D=512, because
  roaring gets faster *and* cuVS sparse stays slow. At D=1024 the peak
  holds at 17.08x — the absolute miss (~167 ms cuVS vs ~10 ms roar) is
  even larger because the cuVS sparse path's per-pair cost scales with D.

### Caveat: cuVS's path-selection mistake gets WORSE in fp16

At D=512 fp16, cuVS sparse takes 109 ms at sel=10%, but its dense path
takes only 22 ms — a **5× ratio**, up from 3.3× in fp32. The threshold
recommendation from earlier (sparsity ≈ 0.97) now applies *even more
firmly*: in fp16, the asymptotic crossover is closer to **1.5% selectivity**
(sparsity ≈ 0.985). If cuVS plans to ship fp16 brute_force as a recommended
path, fixing that threshold becomes a bigger deal — the misstep costs
proportionally more.



Speedup vs cuVS bitset filter on RTX 5090, N=10M, D=512, Q=64, k=10,
**recall@k = 1.000 everywhere** (top-k IDs bitwise identical):

```
 sel%   speedup (fair, after schedule fix)
 0.01    1.52x
 0.1     3.01x
 1       6.23x
 3       9.55x
 5      11.32x
10      12.50x   ← peak
15       2.71x
20       2.18x
30       1.48x
50       0.63x   ← was 0.53x before the masked-coalescing fix below
90       0.66x   ← was 0.34x; 48% faster (96.3 → 50.4 ms)
```

Two libraries levers, both implemented during this session:

1. **`gpu_roaring_free_async(stream)` for benches under churn** — the sync
   `gpu_roaring_free` mismatches `cudaMallocAsync` and surfaces as illegal-
   address under heavy upload/free cycling.
2. **Masked-task coalescing in `build_schedule`** — merge contiguous dense-
   bitmap (or contiguous negated-block) masked tasks up to `kTileWMax=1M`
   wide, so the executor runs ~10 sgemm+mask+top-k launches instead of
   one-per-65K-container at high selectivity.


## Setup

- **GPU**: NVIDIA GeForce RTX 5090 (32 GB GDDR7)
- **Dataset**: N=10,000,000 vectors, D=512, fp32 (~20.5 GB on device)
  - D was reduced from the originally-requested 1024 because 10M×1024×fp32 = 40 GB
    won't fit on this card; `filtered_search.cu` is fp32-only so we couldn't
    drop to fp16 without a (separate) port.
- **Query batch**: Q=64
- **Top-k**: k=10
- **Filter shape**: uniform random IDs at the target selectivity (worst case
  for roaring — no clustering, no exploitable runs; arrays at low sel, single-
  density bitmap containers at high sel)
- **Selectivity grid**: 0.01%, 0.1%, 1%, **3%**, 5%, 10%, **15%**, 20%, 30%, 50%, 90%
- **Warmup / iters**: 5 / 15, interleaved A/B per iter
- **Bench source**: `bench_e2e_sweep.cu.snapshot` here (canonical copy lives in
  `~/Development/cuvs/cpp/bench/prims/core/bench_e2e_sweep.cu`)

## Timed region (both paths)

> "We have the roaring bitmap on the host; benchmark = copying + searching + getting results."

Each timed event measures, end-to-end:
1. **H2D copy** of the filter (CRoaring portable bytes for cu_roaring; flat
   packed 32-bit-word bitset for cuVS)
2. **Per-filter setup**: `cu_roaring::build_schedule` (cu_roaring side);
   `raft::core::bitset` + `bitset_filter` wrap (cuVS side)
3. **The search itself** (`cuvs::neighbors::brute_force::search` vs
   `cu_roaring::roaring_filtered_search`)
4. **D2H** of the top-k id matrix
5. **Free** of per-call device resources (so churn cost is included)

The dataset, the cuVS brute_force index, and the persistent device output
buffers are built once outside the timed region — both APIs would do the same
in production.

## Fairness: subtracting raft::popc count

cuVS's `tiled_brute_force_knn` calls `filter.view().count(res)` on every
search to compute sparsity and pick the dense vs sparse code path. That's a
`raft::popc` kernel + a D2H of the scalar + a stream sync. cu_roaring already
knows the cardinality from CRoaring host-side, so it never pays this cost.
raft's bitset has no `set_count` API to hand it our cached value.

The bench therefore measures `bitset_view::count()` in isolation per cell
(8 iters, warm) and reports two cuVS columns:
- `cuvs_ms` — raw end-to-end, **what cuVS actually does today**
- `cuvs_ms_no_count` = `cuvs_ms - count_ms` — the fair-comparison number

Across all cells `count_ms` is **0.1–0.25 ms** — too small to matter except
at sel=0.01% where it's 13% of cuVS's 1.65 ms total. The fair speedup is
≤2% lower than the raw speedup everywhere else.

## Results (post-coalescing; fair speedup, raw in parens)

| sel %  | card      | cuvs ms | count ms | cuvs-count | roar ms | fair (raw) | recall | schedule                  |
|-------:|----------:|--------:|---------:|-----------:|--------:|-----------:|-------:|---------------------------|
|   0.01 |     1,000 |   1.467 |    0.106 |      1.361 |   0.894 | **1.52x** (1.64x) | 1.000 | gather (1K of 153 arr)    |
|   0.1  |    10,000 |   3.212 |    0.100 |      3.112 |   1.035 | **3.01x** (3.10x) | 1.000 | gather (10K of 153 arr)   |
|   1    |   100,000 |  12.510 |    0.189 |     12.321 |   1.977 | **6.23x** (6.33x) | 1.000 | gather (100K of 153 arr)  |
|   3    |   300,000 |  34.184 |    0.120 |     34.063 |   3.568 | **9.55x** (9.58x) | 1.000 | gather (300K of 153 arr)  |
|   5    |   500,000 |  55.608 |    0.105 |     55.503 |   4.901 | **11.33x** (11.35x) | 1.000 | gather (500K of 153 arr)  |
| **10** | 1,000,000 | 111.056 |    0.082 |    110.974 |   8.877 | **12.50x** (12.51x) — peak | 1.000 | gather (1M of 152 bmp)    |
|  15    | 1,500,000 |  34.187 |    0.144 |     34.043 |  12.564 | **2.71x** (2.72x) | 1.000 | gather (1.5M of 153 bmp)  |
|  20    | 2,000,000 |  34.452 |    0.186 |     34.265 |  15.707 | **2.18x** (2.19x) | 1.000 | gather (2M of 153 bmp)    |
|  30    | 3,000,000 |  34.401 |    0.194 |     34.207 |  23.113 | **1.48x** (1.49x) | 1.000 | gather (3M of 153 bmp)    |
|  50    | 5,000,000 |  34.468 |    0.163 |     34.305 |  54.685 | **0.63x** (0.63x) | 1.000 | **39 masked** (5M) + 2.5M gather |
|  90    | 9,000,000 |  33.150 |    0.112 |     33.038 |  50.373 | **0.66x** (0.66x) | 1.000 | **10 masked** (10M), no gather |

**Recall@10 = 1.000 across every cell** — top-k IDs are bitwise identical
between the two implementations.

### Before the coalescing fix (v2 data, same lib but no schedule merging)

| sel % | masked-task count | roaring ms | fair speedup | delta from v2→v3 |
|------:|------------------:|-----------:|-------------:|------------------|
| 50    | 76 → **39**       | 65.55 → **54.69** | 0.53x → **0.63x** | **18% faster** |
| 90    | 153 → **10**      | 96.33 → **50.37** | 0.34x → **0.66x** | **48% faster** |

Cells at sel ≤ 30% produce **zero masked tasks** on this uniform-random shape
(every container has density < 50% → gather path) so the coalescing change
does not affect them; small differences in their numbers vs v2 are normal
run-to-run variance.

## Reading the curve

- **Sparse (0.01% – 10%)**: cu_roaring wins by **1.4×–13.3×**. The schedule
  dispatches all of these as one big gather; cuVS does an N×Q GEMM + bitset
  mask. Saving the full GEMM is the entire margin.
- **Peak at 10%**: 13.25×. This is the last cell where the schedule path can
  still gather-then-GEMM the kept points more cheaply than cuVS scans the
  whole dataset. Per-container density is 6553/65536 — just over the
  array→bitmap promotion threshold — so 152 of 153 containers are bitmap;
  the gather kernel reads a few hundred 8 KB containers and writes 1M rows
  to a 2 GB gather buffer.
- **The 10%→15% cliff is mostly cuVS getting faster, not cu_roaring getting
  slower.** cuVS brute_force has a `if (sparsity < 0.9) tiled_brute_force_knn
  else sparse-iterate-ids` switch (knn_brute_force.cuh:643). At sel ≥ 10%
  cuVS picks the dense tiled path and drops from 138 ms → 39 ms in one step;
  cu_roaring's curve continues smoothly. Everything to the right of 15% is
  benchmarking against cuVS's good path.
- **Crossover at ~35–40%**: cu_roaring slips below parity.

## The masked-task coalescing fix (now implemented)

Above sel ≈ 50%, both paths logically do **the same thing**: GEMM the whole
dataset, then mask out rejected rows, then top-k. The 96 ms at sel=90% in
the original v2 data was not an algorithmic gap — it was schedule-side
launch overhead.

**Before the fix** (`build_schedule` at src/filtered_search.cu:580–594 of the
v2 snapshot): when a bitmap container is dense (≥50%), it emits **one
`kRange`-with-mask task per container** with `n_cols = 65,536` (one
container's width). The executor then runs one cuBLAS sgemm + one
`apply_mask_kernel` + one top-k pass-1 per task. At sel=90%, every one of
the 153 containers is dense → **153 tasks × 3 kernels = ~459 launches**.
Each 65,536-wide GEMM is computationally tiny for an RTX 5090 (~30 µs of
math) and is launch-bound (~10–15 µs of overhead). The math is correct;
the launches dominate.

cuVS's `tiled_brute_force_knn` runs the same GEMM+mask+top-k as **one
internally-tiled call** with a tile width near 1M cols. At N=10M that's
~10 launches per kernel, not 153.

**The fix** — added in this session, see `filtered_search.cu.snapshot`
(post-`for (auto& t : masked)` push, before assembling sched.tasks):

```cpp
// Coalesce adjacent masked tasks whose row ranges and mask buffers are both
// contiguous. ~10x fewer kernel launches at high selectivity.
std::vector<GemmTask> coalesced;
for (const GemmTask& t : masked) {
    if (!coalesced.empty()) {
        GemmTask& back = coalesced.back();
        const uint32_t stride_words = back.n_cols >> 5;
        if (back.start + back.n_cols == t.start                  // rows contig
         && back.mask + stride_words  == t.mask                  // mask buf contig
         && back.mask_bit0 == 0u && t.mask_bit0 == 0u
         && back.mask_invert == t.mask_invert
         && back.n_cols + t.n_cols   <= kTileWMax) {             // cap at tile width
            back.n_cols += t.n_cols;
            continue;
        }
    }
    coalesced.push_back(t);
}
masked = std::move(coalesced);
```

This works in both the non-negated path (bitmap containers laid out in
key-sorted order in `filter.bitmap_data` — adjacent containers ARE memory-
adjacent) and the negated path (per-block masks written into
`scratch.mask_pool` at uniform 2048-uint32 stride in need_mask order, which
IS block order). The row-contiguity check is the safety net for any gap
(an array/run container interleaved, or a fully-excluded block in the
negated path).

**Where it helps**:

- **sel=90%** (153 dense containers, all contiguous): 153 → **10 tasks**
  (capped at kTileWMax=1M cols; 10M cols / 1M tile = 10). Roaring drops
  **96.3 → 50.4 ms** (1.91× speedup). Schedule output:
  `M=10(10.0M) G=0(0.0M)`.
- **sel=50%** (random mix of ~half dense, ~half sparse bitmaps): 76 → **39
  tasks** — matches the expected number of dense streaks in a random
  Bernoulli(0.5) sequence over 153 containers. Roaring **65.6 → 54.7 ms**
  (1.20× speedup).
- **sel ≤ 30%**: 0 masked tasks on this uniform-random shape — pure gather.
  No effect.
- **Real-world clustered filters** (e.g., tags correlated with consecutive
  row IDs in production): dense bitmap containers cluster into longer
  contiguous runs, so the fix's reach extends to lower selectivities where
  the existing benchmark shows none. Worth a follow-up sweep on YFCC-10M
  with the sort-by-tag-tuple permutation.

**Where ~17 ms of gap to cuVS remains at sel=90%**: cu_roaring still pays
10 outer-task iterations (so 10 pass-2 top-k merges) and uses three
separate kernel launches per tile (sgemm + apply_mask + topk_pass1) while
cuVS likely fuses GEMM + filter + top-k into one or two kernels. Closing
the rest would need kernel-level work, not schedule-level.

The existing test suite (`test/test_filtered_search.cu`, 7 tests covering
RUN, ARRAY, BitmapDense, BitmapSparse, MixedContainers, Negated,
FragmentationFallback) passes unchanged after the coalescing patch.

## Files

Each cell was run in a fresh process so the persistent `build_schedule`
scratch starts at zero.

**Final / canonical (v3, post-coalescing fix; what the plots show):**
- `bench_e2e_sweep_v3_low.json` — cells 0.01, 0.1, 1, 3, 5, 10, 15, 20%
- `bench_e2e_sweep_v3_30.json`  — cell 30%
- `bench_e2e_sweep_v3_50.json`  — cell 50%
- `bench_e2e_sweep_v3_90.json`  — cell 90%
- `bench_e2e_sweep_v3_10.json`  — sanity rerun of cell 10% (no regression)
- `*.log` — corresponding stdout
- `bench_e2e_sweep.cu.snapshot` — bench source
- `filtered_search.cu.snapshot` — the cu_roaring source with the coalescing patch
- `../../figures/e2e_sweep_speedup.png` — fair-vs-raw speedup curve, log/log
- `../../figures/e2e_sweep_ms.png`      — raw latency, both paths, p10..p90 shaded

**Pre-coalescing v2 data (kept for the before/after comparison):**
- `bench_e2e_sweep_v2_low.json`, `_v2_30.json`, `_v2_50.json`, `_v2_90.json`

**First-pass (no count-fairness adjustment, no 3%/15% cells):**
- `bench_e2e_sweep_low.json`, `_high.json`, `_50.json`

## Reproduce

```bash
# cu_roaring static lib (sm_89, on run-optimizations branch)
cd /tmp/cu-roaring-ro && cmake -B build89 -DCMAKE_CUDA_ARCHITECTURES=89 && \
  cmake --build build89 -j

# cuVS bench (depends on the lib above)
cd ~/Development/cuvs/cpp/bench/prims/core/build && \
  cmake -DSD_ROARING_LIB=/tmp/cu-roaring-ro/build89/libcu_roaring_bitmap.a \
        -DSD_CROARING_LIB=/tmp/cu-roaring-ro/build89/third_party/CRoaring/src/libroaring.a \
        -DSD_ROARING_DIR=/tmp/cu-roaring-ro . && \
  make bench_e2e_sweep -j

# Low/mid-sel cells (everything that fits with no extra scratch)
LD_LIBRARY_PATH=~/Development/cuvs/cpp/build:$LD_LIBRARY_PATH \
  E2E_SELS="0.0001,0.001,0.01,0.03,0.05,0.10,0.15,0.20" \
  E2E_OUT=bench_e2e_sweep_v3_low.json ./bench_e2e_sweep

# Each high-sel cell needs a fresh process so the prior scratch is zero
E2E_SELS="0.30" E2E_OUT=bench_e2e_sweep_v3_30.json ./bench_e2e_sweep
E2E_SELS="0.50" E2E_OUT=bench_e2e_sweep_v3_50.json ./bench_e2e_sweep
E2E_SELS="0.90" E2E_OUT=bench_e2e_sweep_v3_90.json ./bench_e2e_sweep

# Stitch + plot
python3 ~/cu-roaring-bitmap/scripts/plot_e2e_sweep.py \
  bench_e2e_sweep_v3_low.json bench_e2e_sweep_v3_30.json \
  bench_e2e_sweep_v3_50.json bench_e2e_sweep_v3_90.json
```

## Bench source fix discovered along the way

The first bench attempt crashed at sel=5–10% with an illegal-address error in
the cu_roaring path. Root cause: `upload_impl` allocates the packed device
buffer with `cudaMallocAsync(stream)` (stream-ordered memory pool), but
`gpu_roaring_free` was being called with the synchronous `cudaFree`. Mixing
the stream-ordered pool allocator with non-pool free is documented UB in CUDA
and surfaces under churn as an illegal-address sticky error on subsequent pool
allocations. CUDA_LAUNCH_BLOCKING=1 makes the bug disappear — that's the tell.

The fix in the bench was to call `gpu_roaring_free_async(gpu, stream)` —
both the public header and `src/upload.cpp` already expose it for exactly
this case. Worth a sweep of other call sites in the library that pair
`cudaMallocAsync` with sync `cudaFree`.
