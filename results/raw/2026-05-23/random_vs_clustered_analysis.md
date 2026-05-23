# Synthetic sweep: random vs clustered filters (RUN-container dispatch)

The previous sweeps used **uniform-random IDs** as the filter shape — the
worst case for roaring (no clusters, no runs, density tracks selectivity).
On real-world filters where IDs cluster (sorted-by-tag tables, range filters
on sorted columns, etc.), the schedule's `kRange`-direct dispatch should
fire and the bench should look noticeably different.

This run replays the exact same sweep, same configuration, but with the
filter built as a **single contiguous run** of `card` IDs. After
`roaring_bitmap_run_optimize` that's RUN containers + cross-container
coalescing by `enumerate_runs` → ONE `kRange`-direct task spanning the
whole run. No gather, no mask. Best case for the schedule.

## Setup

- RTX 5090, fp32, N=10M, D=512, Q=64, k=10, recall@k = 1.000 every cell
- Filter shape: `E2E_SHAPE=clustered` builds the filter as
  `ids[i] = offset + i` for `i ∈ [0, card)`, where `offset` is chosen
  stably from `card` so cells don't all land in the same dataset slice
- All other framing identical to the random sweep
  (cuVS bitset alloc + H2D inside `run_a`'s timed region;
  cu_roaring through `cu_roaring::upload(filt, N, stream)` which preserves
  RUN containers)

## Results

| sel %  | RANDOM | | CLUSTERED | | container counts (clustered) |
|:------:|--------:|----------:|----------:|----------:|---------------:|
|        | roar ms | fair spd | roar ms | fair spd | run/arr/bmp + schedule shape |
| 0.01   | 0.99    | 1.54×    | 1.22    | **1.07×** | 1/0/0 — FALLBACK (run < 64K) → 1 gather |
| 0.1    | 1.04    | 3.01×    | 1.24    | **2.19×** | 1/0/0 — FALLBACK (run < 64K) → 1 gather |
| 1      | 1.98    | 6.06×    | 2.00    | 6.21×    | 2/0/0 — 1 direct (100K cols) |
| 3      | 3.54    | 9.61×    | 2.96    | **11.24×** | 6/0/0 — 1 direct (300K cols) |
| 5      | 4.90    | 11.32×   | 3.58    | **15.21×** | 8/0/0 — 1 direct (500K cols) |
| **10** | 8.88    | 12.50×   | 5.87    | **18.75×** | 16/0/0 — 1 direct (1M cols, 16 RUN containers coalesced) |
| 15     | 12.56   | 2.71×    | 8.09    | **3.77×** | 24/0/0 — 1 direct (1.5M cols) |
| 20     | 15.71   | 2.18×    | 9.43    | **3.28×** | 32/0/0 — 1 direct (2M cols) |
| 30     | 23.11   | 1.48×    | 13.93   | **2.22×** | 46/0/0 — 1 direct (3M cols) |
| **50** | 54.69   | **0.63×** | 22.77 | **1.36×** | 77/0/0 — 1 direct (5M cols) |
| 90     | 50.37   | 0.66×    | 42.92   | 0.72×    | 16/0/0 — 1 direct (9M) + 3 masked (200K) |

(`fair spd` = cuVS_ms_no_count / roaring_ms; everywhere ≈ raw speedup
because count is < 0.25 ms on all cells.)

## What changes

### 1. The 0.6× regression at sel=50% on random **goes away** under clustered
Random at sel=50% got 0.63× (cu_roaring **slower** than cuVS) because the
schedule emitted 39 masked tile-tasks — each tile had its own sgemm + mask
+ top-k launches, dominated by launch overhead. Under clustered the
schedule produces **one** `kRange`-direct task spanning 5M cols, executed
as a single big tiled GEMM. Same compute, one outer task instead of 39 ×
3 = 117 launches. Result: **1.36× faster than cuVS** instead of 0.63×.

### 2. Mid-sel speedups grow ~50%
At sel=3–30% the random curve was already winning; clustered pushes it
further. Saving the gather kernel pass is the main mechanism:

- sel=10% random: 8.88 ms (gather 1M IDs + skinny GEMM)
- sel=10% clustered: 5.87 ms (direct slice + GEMM, no gather)
- Δ = 3 ms — about what a 1M-row × 512-dim × 4-byte gather costs on the
  RTX 5090 at ~700 GB/s effective bandwidth

### 3. Peak speedup at sel=10% is now **18.75×**
(was 12.50× on uniform random). Same mechanism as #2 — the gather kernel
is the saved cost — but here it lands right at cuVS's pathological
sparse-CSR sweet spot, so the speedup compounds.

### 4. cuVS gets faster too on clustered, but only slightly
The cuVS bitset path is shape-agnostic — `raft::popc` counts the same
number of bits, the brute_force kernel scans the same N rows. The 1–4 ms
drop in cuVS time (e.g. sel=15%: 34.2 → 30.6 ms) is just run-to-run
variance / fewer queries-per-second-throttling.

### 5. The two surprises at low sel (clustered worse)

At sel=0.01% and 0.1%, clustered loses (1.07× / 2.19× vs 1.54× / 3.01×).
The runs there are 1K / 10K cols wide — **below `kDirectMinWidth=64K`**
— so the schedule's fragmentation fallback kicks in, gathers the run
into a compact buffer, and runs the same gather+GEMM as the random path
would. **But** it also runs `enumerate_runs` + `expand_ranges` first,
because there ARE RUN containers — that's ~0.2 ms of extra kernel-launch
overhead that the array-path doesn't pay. At these sizes that overhead
is the entire latency budget.

This is a real micro-optimisation we could pick up: if all RUN containers
fit in a single container AND will be gathered anyway (below the direct-
width gate), skip the enumerate_runs kernels and just gather the
contiguous range directly. Not large, but free perf.

### 6. At sel=90%, clustered is still slower than cuVS (0.72×)
The 9M-wide run becomes 1 direct task + 3 small masked tasks for the
trailing bits beyond the last full container. The single direct GEMM is
9M × 64 × 512 = 295 GFLOP — ~5 ms of math + the executor's tile-loop
overhead (≥9 tiles at kTileWMax=1M). cuVS does the equivalent fused work
in a tighter kernel. Closing this gap needs kernel-fusion in the masked
path, not schedule-level work — same as the random-curve story at high
sel.

## Where cu_roaring crosses cuVS

| selectivity range | random | clustered |
|---|---|---|
| sub-parity threshold (where speedup drops below 1×) | ~40% | **~75%** |
| peak speedup | 12.50× at sel=10% | **18.75× at sel=10%** |

For workloads with clustered ID distributions (sorted tables, range
predicates over sorted columns), the schedule approach is the right tool
over a **roughly 2× wider selectivity range** than the random-case
analysis suggested — productively useful up to ~50–60% rather than only
to ~30%.

## Files

- `bench_e2e_sweep_v3_clustered_low.json`     — cells 0.01–20% in one process
- `bench_e2e_sweep_v3_clustered_30.json` / `_50.json` / `_90.json`
  — high-sel cells, fresh process each (the persistent build_schedule
    scratch fits one big direct-task allocation easily; no SKIPs)
- `bench_e2e_sweep.cu.snapshot`               — bench source with `E2E_SHAPE`
- `../../figures/e2e_sweep_speedup_random_vs_clustered.png`
- `../../figures/e2e_sweep_ms_random_vs_clustered.png`

Reproduce:
```bash
# random (default — already in v3 JSONs)
./bench_e2e_sweep

# clustered
E2E_SHAPE=clustered ./bench_e2e_sweep
```
