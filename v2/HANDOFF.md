# v2 Handoff — 2026-04-28

Session ended with steps 1–6 of `v2/PLAN.md` complete and benched on RTX 5090.
Nothing committed yet — all changes are in the working tree.

## What landed today

### Files written (or rewritten)

```
v2/include/cu_roaring_v2/types.cuh    extended GpuRoaringBatch with three
                                       host metadata mirrors so make_view and
                                       multi_and avoid D2H entirely
v2/include/cu_roaring_v2/query.cuh    empty-view guard in contains(view, id)
v2/include/cu_roaring_v2/api.hpp      include <roaring/roaring.h> directly to
                                       avoid the namespace clash between the
                                       extern-"C" forward decl and CRoaring's
                                       roaring::api typedef
v2/src/internal.hpp                   shared HostMetaLayout used by upload +
                                       promote + multi_and (one source of
                                       truth for the host meta byte layout)
v2/src/upload.cpp                     full rewrite: N CRoaring → packed CSR
                                       device buffer, single H2D, one terminal
                                       cudaStreamSynchronize so host staging
                                       can free; also implements free_batch
                                       and make_view
v2/src/promote.cu                     full rewrite: total_containers blocks,
                                       D2D copies of the unchanged sections
                                       (keys/key_indices/CSR starts), kernel
                                       writes types/offsets/cardinalities and
                                       expanded bitmap pool
v2/src/multi_and.cu                   full rewrite, pure-GPU intersection.
                                       Anchor pick on host via
                                       host_container_starts. Kernel A direct-
                                       map lookups in every other input's
                                       key_index. CUB ExclusiveSum compacts.
                                       Kernel B (anchor_n blocks, flagged
                                       only) hoists per-input src pointers
                                       into dynamic shared memory, AND-reduces
                                       1024 words, atomicAdd to total_card.
                                       Single 12-byte D2H + sync at the end —
                                       nothing else leaves the GPU.
v2/src/decompress.cu                  full rewrite: total_containers blocks,
                                       block resolves owning bitmap via
                                       binary search on container_starts,
                                       writes to device_bitsets[b*words_each
                                       + key*1024]
v2/test/test_basic.cu                 full rewrite for batch API. n=4 mixed-
                                       type differential vs CRoaring covers
                                       upload, make_view, both contains
                                       overloads, promote, multi_and on two
                                       subsets, multi_and's all-bitmap reject,
                                       decompress. n=1 smoke covers the
                                       degenerate batch.
v2/bench/bench_vs_bitset.cu           full rewrite. Sweep n_bitmaps × universe
                                       × selectivity. Median + stdev across
                                       30 iters with 5 warmup. Reports upload
                                       / promote / decompress latencies, both
                                       contains overloads, multi_and roaring
                                       vs flat-bitset baseline of identical
                                       memory shape. JIT pre-warm cell at
                                       startup so cell #1 isn't polluted by
                                       the SM_120 PTX→SASS compile.
v2/CMakeLists.txt                     CU_ROARING_V2_BUILD_BENCHMARKS default
                                       returned to ON
CMakeLists.txt (root)                 added option CU_ROARING_BUILD_V2 (ON)
                                       and add_subdirectory(v2)
v2/README.md                          one-paragraph update mentioning the
                                       three new host mirror fields
```

### Test status

```
$ ctest -R cu_roaring_v2_basic --output-on-failure
1/1 Test #7: cu_roaring_v2_basic ............ Passed   0.25 sec
```

### Bench artefact

`results/raw/2026-04-28/v2_bench_vs_bitset.csv` — 24 cells on RTX 5090.

## Architectural decisions worth remembering

1. **`multi_and` does the key intersection on the GPU**, overriding the
   PLAN.md sketch that said "intersect on host". The user's directive was
   "no transferring back and forth, no slow code paths". The cost was a
   slightly more complex kernel pipeline but the benefit is a fully
   stream-async multi_and with one 12-byte D2H at the very end.
2. **Output buffer is over-allocated to `anchor_n` containers** so we don't
   need to know `n_common` host-side before the build kernel runs. Wasted
   space in the worst case is `(anchor_n - n_common) * 1024 * 8` bytes;
   trivial for typical filter sizes.
3. **Per-bitmap state goes through host mirrors** (`host_container_starts`,
   `host_key_index_starts`, `host_n_bitmap_containers`) so `make_view`,
   anchor selection, and all-bitmap validation are all O(1) reads with no
   device contact. The host meta block is a single packed allocation
   alongside `_alloc_base`.
4. **`promote_batch` D2D-copies what doesn't change** (keys, key_indices,
   both CSR start arrays) and only re-derives types / offsets /
   cardinalities / bitmap_data. `cardinalities` is recomputed via popcount
   in the kernel for *all* containers — input ARRAY/RUN had different
   semantics for that field, so blanket recomputation is the correct
   behaviour.
5. **Empty-view guard lives in `contains(view, id)`**, not in `make_view`,
   because trying to make `max_key` a sentinel doesn't work cleanly with
   the existing `>` bound check. One register read added to the hot path;
   branch predicts perfectly for non-empty views.

## Bench findings — RTX 5090

96 MB L2, 170 SMs, 31.8 GB GMem. Sweep cells: n_bitmaps ∈ {1, 4, 16, 64} ×
universe ∈ {1M, 10M} × selectivity ∈ {0.001, 0.01, 0.10}. 1M queries, 30 iters.

### Compression — works
- sel=0.001 → 60× over bitset
- sel=0.01  → 6.5×
- sel=0.10  → 1.0×
After `promote_batch`, footprint matches bitset 1:1 — promotion intentionally
trades compression for the all-bitmap precondition `multi_and` requires.

### contains throughput — bitset wins most cells
Roaring competes only at the smallest, sparsest cells (1.16× at n=1, sel=0.001).
Otherwise bitset is 1.2–8× faster. Structural cause: roaring batch contains is
6+ global reads (key_index_starts × 2, key_indices, container_starts, types,
offsets, then container payload) versus bitset's single uint64 read + bit test.
**The crossover happens when the bitset spills L2** — and the 5090's 96 MB L2
fits this entire sweep without spilling. Need universe ≥ 100M or n_bitmaps ≥
256 to see roaring's expected win.

### multi_and — bitset wins by 20–50× across the board
Roaring multi_and floor is 0.15–0.35 ms regardless of cell; bitset baseline is
0.005–0.008 ms. **Overhead-bound, not data-bound.** Each multi_and call runs:
5 cudaMallocAsyncs, intersect kernel, CUB scan (allocates its own temp), count
kernel (1 thread), output cudaMallocAsync, key_indices memset, write_starts
kernel (1 thread), build_and kernel, 5 cudaFreeAsyncs, 12-byte D2H sync. The
actual AND work is fast — orchestration is the bottleneck for typical filter
sizes (tens of containers per anchor).

## Where to pick up tomorrow

In order of expected information-per-effort:

### 1. Push the sweep past L2 (~30 min)
Add `universe = 100M` and `n_bitmaps = 256` cells to
`v2/bench/bench_vs_bitset.cu`. Memory budget: 256 × 12.5 MB bitset = 3.2 GB,
fits the 5090. Expectation: roaring compresses to fit in L2 while bitset
spills, contains crossover should appear.

If it does NOT appear: investigate whether the `contains(batch, b, id)` kernel
is actually memory-bound or whether it's instruction-bound on the indirection
chain. Use ncu (`scripts/ncu_cache_profile.sh` is already in the tree from
v1) to look at L2 hit rate and the read-dependency chain.

### 2. Fused `multi_and` (~half-day)
Collapse the current 9-kernel pipeline into one kernel:
- One block per anchor cid.
- Block-level intersect: thread 0 (or warp 0) does the per-input key_index
  lookups, broadcasts the "match" bool via shared memory.
- If matched: atomicAdd to a global counter to claim a compacted output slot.
  This loses the sorted output property — but for v2's all-bitmap output
  with a direct-map key_indices, sortedness isn't required for correctness
  (key_indices acts as the sorted index). Confirm that.
- Same block then writes metadata + AND-reduces 1024 words.
- Reduce total_card via atomicAdd as today.
- Single D2H of {n_common, total_card} at the end.

Eliminates: scan, count kernel, write_starts kernel, 4 of the 5 temp
allocations. Would collapse the 0.15 ms floor to one launch ≈ 5–10 µs.
**This is the change that would make the README's positioning honest** —
right now the 20–50× regression at typical cell sizes is the user's first
impression of v2 multi_and.

### 3. Type-aware AND for sparse filters (architectural)
Skip `promote_batch` for `multi_and` when inputs are small. ARRAY ∩ ARRAY =
sort-merge into ARRAY (output stays small). BITMAP ∩ ARRAY = filter ARRAY
through BITMAP. Lets sparse filters keep their 60× compression through the
intersection. Bigger change — affects API, kernel set, and tests. Defer until
(2) lands.

### 4. YFCC-10M benchmark (`v2/PLAN.md` §"Future benchmark")
Once (1) and (2) are in, add `bench/bench_yfcc.cu`. v1's
`bench/yfcc_export.py` already materialises per-tag ID lists, reusable as-is.
Three-row story is the deliverable: flat bitset / roaring natural / roaring
presorted.

## Build cheatsheet

```bash
# WSL2 cu-roaring-bitmap repo — rebuild tomorrow.
# Note: the existing build/ directory has a stale cache from before the
# project rename (cu-roaring-filter → cu-roaring-bitmap). Use build_v2/.

cmake --build /mnt/c/Users/maxwb/Development/cu-roaring-bitmap/build_v2 \
      --target cu_roaring_v2_test_basic -j

cmake --build /mnt/c/Users/maxwb/Development/cu-roaring-bitmap/build_v2 \
      --target cu_roaring_v2_bench_vs_bitset -j

# Test:
/mnt/c/Users/maxwb/Development/cu-roaring-bitmap/build_v2/v2/test/cu_roaring_v2_test_basic

# Bench:
/mnt/c/Users/maxwb/Development/cu-roaring-bitmap/build_v2/v2/bench/cu_roaring_v2_bench_vs_bitset

# Quick smoke (2 cells, ~5s):
/mnt/c/Users/maxwb/Development/cu-roaring-bitmap/build_v2/v2/bench/cu_roaring_v2_bench_vs_bitset --quick
```

If `cmake` says "no rule to make target cu_roaring_v2_bench_vs_bitset", the
benchmarks option is sticky-OFF in cache; reconfigure with
`-DCU_ROARING_V2_BUILD_BENCHMARKS=ON`.

## Open questions for tomorrow

1. Does `multi_and`'s output need sorted keys, or does direct-map
   `key_indices` make sortedness redundant? Affects the fused-kernel design
   above.
2. The first cell of the bench used to show ~117 ms `promote_batch`; the JIT
   warmup fixed it but it confirms the ship-as-PTX strategy has real
   first-launch cost on Blackwell. Worth thinking about whether to also ship
   SM_120 SASS once we're on a CUDA toolkit that supports it (CUDA 12.4
   doesn't).
3. The current 0xFFFF sentinel in `key_indices` means `multi_and` output's
   `key_indices` reserves up to 65536 entries × 2 bytes = 128 KB per result.
   For batch outputs of multiple ANDs (deferred grouped multi_and),
   that becomes 128 KB × M outputs. Consider whether this needs revisiting.
