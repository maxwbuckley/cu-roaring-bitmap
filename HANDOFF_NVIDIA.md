# Handoff — RTX 5090 Validation (2026-04-13, late session)

Session notes for picking this up tonight on the NVIDIA box. Everything below is already committed and pushed to `master`. **None of it has been built or tested** — macOS dev box, no `nvcc`.

## What changed today

On top of `509a9f7` (stream-ordered alloc):

1. **`311971b` — Rename repo** `cu-roaring-filter` → `cu-roaring-bitmap`. GitHub is now public at `github.com/maxwbuckley/cu-roaring-bitmap`. Local directory is still `cu-roaring-filter/`.
2. **`2938ea9` — Backfill docs** (docs only, no code risk).
3. **`9c9185b` — Fix fused multi_and cardinality; clamp run expansion end** (two correctness fixes).
4. **`84e82ba` — Eliminate D2H round-trips in `promote_to_bitmap` and `fused_multi_and_allbitmap`** (two high-impact transfer eliminations — the session's main work).

## First steps on the NVIDIA box

```bash
cd ~/Development/cu-roaring-filter          # or wherever the WSL checkout lives
git pull --rebase
cd build
cmake --build . -j 2>&1 | tee /tmp/build.log
```

**Expect some build friction.** I wrote ~400 lines of new CUDA code without a compiler. Typical things that may need fixing:

- Missing or redundant `#include`s in `src/promote.cu` (I removed `<cstring>` and `<vector>` usage there — if something else references them, add back). The file no longer needs `std::vector` or `std::memcpy` at all.
- `FusedInputMeta` struct visibility — I declared it in `src/set_ops.cu` at namespace scope *before* `fused_multi_and_allbitmap`. If nvcc gripes about layout-compatibility between host init and device access, add `__align__` or move the struct above the first kernel that uses it.
- `__shfl_up_sync` / `__shfl_xor_sync` — if `-Werror` flags a shadowing warning in the manual block scan inside `enumerate_presence_keys_kernel`, rename the `v` variable.
- `static_assert(sizeof(ContainerType) == 1, ...)` in `fused_multi_and_allbitmap` — should succeed (the enum is `: uint8_t`), but if for some reason the type changed, the assert will catch it.

If the build fails, **fix forward, don't revert the commit.** Every logical change is correct in isolation; the likely failures are syntax / include issues I couldn't catch without a compiler.

## What the two new commits actually do

### `9c9185b` — correctness fixes (from the earlier session)

Already documented above: fused multi_and now populates cardinalities + total, and run-expansion loops clamp their end at `0xFFFF` as defence in depth.

### `84e82ba` — D2H / H2D elimination

**`src/promote.cu` — `promote_to_bitmap` is now fully device-resident.**

Old path: D2H of every metadata array + all three data pools → CPU rebuild → H2D of the new all-bitmap structure. At 100M containers, that's tens of MB round-tripping PCIe for a pure structural transform.

New path: one block per source container, kernel self-dispatches on source type:
- `BITMAP` → plain word copy (256 threads × 4 words each).
- `ARRAY` → zero dst then atomicOr-scatter (threads stride the source array).
- `RUN` → zero dst then expand each (start, length) pair, clamping end to `0xFFFF`.

The kernel also writes `types[cid] = BITMAP` and `offsets[cid] = cid * 8192` inline from thread 0. `keys` and `cardinalities` are D2D copies (unchanged by promotion). `max_key` is derived from `universe_size` — **no D2H scalar read**. `key_index` is allocated, memset 0xFF, then populated by a tiny `build_key_index_kernel` (one thread per container, scatters `key_index[keys[i]] = i`).

Net traffic: 0 bytes D2H, 0 bytes H2D, just D2D and kernel launches.

**`src/set_ops.cu::fused_multi_and_allbitmap` — now fully device-resident apart from two 4-byte control-flow reads.**

Old path (roughly the same story as promote): D2H every input's `keys[]`, CPU `std::set_intersection` across all non-negated inputs, CPU `std::lower_bound` loop to build an `n_common × count` pointer table, H2D the pointer table. That fires on every all-bitmap `multi_and` call, which is the hot path for multi-predicate filter construction.

New pipeline (all kernels are in `src/set_ops.cu`, just above `fused_multi_and_kernel`):

| Step | Kernel | Work |
|------|--------|------|
| 1 | `presence_set_bits_kernel` | For each non-negated input, scatter its `keys[]` into a 65536-bit presence bitmap (1024 × `uint64_t`). Called once per non-negated input. |
| 2 | `presence_and_reduce_kernel` | AND-reduce all non-negated presence bitmaps into a single 1024-word result. If every input is negated, fall back to the universe mask (bits `[0, max_key_plus_1)` set). Launch config: `<<<4, 256>>>` — one thread per word. |
| 3 | `presence_popcount_kernel` | Atomically popcount the result bitmap into a single `uint32_t`. Launch config: `<<<4, 256>>>`. **The one unavoidable D2H**: we need `n_common` on host to size downstream allocations. |
| 4 | `enumerate_presence_keys_kernel` | Block-cooperative enumeration: each of 256 threads processes 4 words, warp-shuffle scan of per-thread popcounts gives each thread its write offset, set bits are emitted as sorted `uint16_t` keys directly into `result.keys`. |
| 5 | `build_pointer_table_kernel` | 2D block (`count × 256/count`), resolves each `(common_key, input)` pair via `key_index` (O(1)) or binary search on device `keys[]`, writes either `bitmap_data + idx*1024` or the shared all-zeros sentinel. |
| 6 | `fused_multi_and_kernel` | Unchanged — emits bitmap data + per-container cardinalities + atomicAdds into a global total counter. |
| 7 | `cudaMemsetAsync(types, BITMAP)` + `fused_fill_offsets_kernel` + `fused_build_key_index_kernel` | Fill output metadata. `types[]` is a memset (enum is `uint8_t`, `BITMAP == 1`). `offsets[]` is `i * 8192`. `key_index` is memset 0xFF then scatter-populated. |
| 8 | 4-byte D2H of `total_cardinality` | Final sync, write into `result.total_cardinality`. |

The only host↔device traffic on the hot path:
- ~256 bytes H2D of the `FusedInputMeta` descriptor array (count × 32 bytes).
- 4 bytes D2H of `n_common` (step 3 — unavoidable, sizes downstream mallocs).
- 4 bytes D2H of `total_cardinality` (step 8).

Everything else is kernel launches + D2D. **No more `std::set_intersection`, no more `std::lower_bound`, no more key-array download.**

## What is specifically at risk

### Correctness risks in `fused_multi_and_allbitmap`

1. **`build_pointer_table_kernel` launch config.** I use `dim3 block(count, 256/count)` with a defensive `if (block.y < 1) block.y = 1;`. For `count == 1` this gives `block = (1, 256)`, for `count == 32` → `(32, 8)`, for `count == 2` → `(2, 128)`. Inside the kernel, `k = blockIdx.x * blockDim.y + threadIdx.y`, `i = threadIdx.x`. **Verify with `cuda-memcheck` that all `(k, i)` pairs are visited exactly once** for several counts (1, 2, 4, 8, 16, 32).

2. **`enumerate_presence_keys_kernel` block scan.** I wrote the warp-shuffle inclusive scan by hand (two passes: warp-level + 8-warp-total reduction). The bug-prone part is the exclusive offset computation: `excl_in_warp = v - local_count; warp_prefix = (warp > 0) ? warp_totals[warp - 1] : 0;`. If any thread's `write_pos` is wrong, `out_keys` will have duplicate or out-of-order values. **Easy cross-check**: after the kernel runs, verify that `out_keys[]` is strictly monotonic and that the count matches the popcount from step 3. The test suite's existing multi_and correctness asserts should catch this indirectly.

3. **All-negated universe mask.** When `n_non_negated == 0`, `presence_and_reduce_kernel` writes bits `[0, max_key_plus_1)` into the result. Edge cases: `max_key_plus_1 == 0` (empty universe — shouldn't happen), `max_key_plus_1 == 65536` (full universe — should produce all-ones). Verify with a multi_and of two fully-negated empty bitmaps.

4. **`FusedInputMeta.max_key_plus_1 = 0` when `key_index == nullptr`.** The kernel's guard `if (meta.key_index != nullptr && key < meta.max_key_plus_1)` short-circuits correctly when `key_index` is null — it falls through to binary search. But double-check that every caller in the current codebase actually populates `key_index` (upload.cpp, upload_ids.cu, promote.cu all should). If any path leaves `key_index == nullptr`, the kernel will hit the binary-search branch on every call, which is correct but slower.

### Correctness risks in `promote_to_bitmap`

1. **`max_key` derivation**. I use `(universe_size - 1) >> 16` instead of reading the highest key from device. For a bitmap whose largest *actual* key is less than the universe max, this allocates a slightly larger `key_index`. Functionally correct (unused cells stay 0xFFFF), just a tiny memory overhead. **Not a bug**, but if a test asserts `result.max_key == old_max_key`, that assertion needs to loosen.

2. **Empty container (`n == 0`)**. Early-return at top of function returns a default-constructed `GpuRoaring{}`. Same as old behaviour.

3. **Already-all-bitmap fast path**. Old code had an explicit branch that did D2D copies and returned. I folded this into the same kernel (the `BITMAP` branch inside `promote_to_bitmap_kernel` is a plain word copy). Performance should be identical or slightly better (one kernel launch instead of multiple D2D memcpy calls). **Verify**: round-trip a known all-bitmap `GpuRoaring` through `promote_auto` and confirm bitwise equality.

## Required testing

Run in this order; stop if anything goes red.

### Stage 1 — build clean
```bash
cmake --build . -j 2>&1 | tee /tmp/build.log
grep -E '(error|warning)' /tmp/build.log | head -40
```
`-Werror` is on everywhere. Any warning is a blocker. Watch especially for:
- unused variables in the host-side of `fused_multi_and_allbitmap` (I removed `std::vector<std::vector<uint16_t>>`, several locals)
- signed/unsigned comparison in the manual scans
- `__shfl_up_sync` lane-mask correctness

### Stage 2 — existing test suite
```bash
ctest --output-on-failure 2>&1 | tee /tmp/ctest.log
```

Everything should pass. If a test fails, grep the test name and correlate to one of these files:
- Tests that exercise `multi_and` / `fused` / `set_operation` / `promote` are the load-bearing ones.
- `enumerate_ids` and auto-promotion tests indirectly exercise `total_cardinality` — they were broken before the cardinality fix from commit `9c9185b`, so if they start *passing* where they used to pass-by-luck, great; if they newly fail, investigate the reduction in `fused_multi_and_kernel`.

### Stage 3 — targeted correctness spot-checks
Even if `ctest` is green, run a focused sanity check on the two new paths.

```bash
# multi_and with diverse container counts
ctest -R multi_and --output-on-failure -V
ctest -R fused    --output-on-failure -V
ctest -R promote  --output-on-failure -V

# If there are cuda-memcheck-enabled test targets, run them on multi_and:
cuda-memcheck ./test/test_set_ops --gtest_filter='*multi_and*'
cuda-memcheck ./test/test_promote
```

Look specifically for:
- out-of-bounds writes in `build_pointer_table_kernel` (bad launch config)
- race conditions in `presence_set_bits_kernel` (should be atomicOr-safe)
- uninitialised reads in `key_index` (memset 0xFF must complete before `fused_build_key_index_kernel`)

### Stage 4 — add a new regression test (strongly recommended)
There's no existing test that specifically asserts the transfer-elimination behaviour. Drop a small one at `test/test_set_ops.cpp` (or wherever multi_and tests live):

```cpp
TEST(FusedMultiAnd, DeviceResidentTransferBudget) {
    // 8 inputs, each with ~1000 containers, Zipfian density.
    // Build on device via upload_from_ids.
    // multi_and them.
    // Assert: result.total_cardinality matches sum of popcounts on device.
    // Assert: result.n_containers > 0, result.keys is sorted on device,
    //         result.cardinalities[i] > 0 for every i.
}

TEST(FusedMultiAnd, AllNegatedUniverseMask) {
    // Two fully-negated tiny bitmaps. Their AND should cover the entire
    // universe. Verify via decompress_to_bitset + popcount.
}

TEST(Promote, AllBitmapFastPath) {
    // Upload an all-bitmap GpuRoaring, call promote_to_bitmap,
    // assert bitwise equality of bitmap_data and metadata.
}
```

### Stage 5 — benchmarks

Order matters: run the micro benchmarks first (tight signals), then the end-to-end ones.

```bash
# Micro: the two paths we just rewrote
./bench/bench_multi_and         # expect fused path to be FASTER than before
                                # (no host sync mid-pipeline, no CPU intersection)
./bench/bench_set_ops           # pairwise set_operation — should be unchanged
./bench/bench_comprehensive     # B1/B3/B4/B5 — no regression expected
./bench/bench_alloc_strategy    # B10 — stream-ordered alloc still at 1.9-3.3x?
./bench/bench_enumerate_ids     # B9 — should be unchanged
./bench/bench_selectivity_sweep # B11 — baseline
./bench/bench_point_query       # B6 — unchanged
./bench/bench_upload_scale      # B8 — unchanged

# Real-data: if yfcc_data is present
./bench/bench_yfcc bench/yfcc_data
```

**Expected wins** (what to write down and brag about in the next PAPER.md update):
- `bench_multi_and` large-count case should show a **notable speedup** at count=8+ with large per-input container counts. The old path spent tens of ms on CPU `set_intersection`; the new path spends microseconds on device kernels. On the RTX 5090 at 1B/1% with 8 predicates, I'd expect the total-call time to drop by several ms.
- `bench_alloc_strategy` should ALSO benefit because we eliminated several temporary host allocations and reduced stream sync count in the fused path.
- `promote_to_bitmap` doesn't have its own dedicated bench, but any bench that exercises `PROMOTE_AUTO` on large inputs will implicitly benchmark it. Watch `bench_upload_scale` at 100M+ IDs.

**Red flags** that mean something is wrong:
- **`bench_multi_and` slower than before.** The new path has more kernel launches (7-8 vs 1-2) but no host sync. At small container counts the launch overhead could actually make it slower. If so, the fix is to fuse the auxiliary kernels (presence scatter + reduce + popcount + enumerate) into one larger kernel, or keep the old CPU path as a small-count fallback. Threshold to consider: `count × avg_n_containers < 1000`.
- **Recall regression in CAGRA benchmarks.** Would indicate a correctness bug in the fused kernel.
- **Tail latency variance up.** The device pipeline should *reduce* variance (fewer host-side mallocs, fewer syncs). If it spikes, check for redundant allocations in the hot loop.

Save any changed numbers under `results/raw/2026-04-13/` and update PAPER.md / REPORT.md accordingly.

## Remaining code backlog

Full list in `PAPER_TODOS.md → Code Optimization Backlog`. Priority after tonight's work lands:

1. **atomicOr contention in scatter/decompress** (`src/decompress.cu`, `src/upload_ids.cu`, `src/set_ops.cu::scatter_to_bitmaps_kernel`). Stage into per-thread `uint64_t` accumulators or partition the word space. Highest remaining win.
2. **Pairwise `set_operation` container matching** (`src/set_ops.cu::download_index` + the whole host-side classification loop). Only hits when inputs have mixed container types, which is uncommon once `promote_auto` runs, so it's **medium impact**. A proper on-device port is ~500 lines of new kernels and careful classification — treat as a standalone branch.
3. `array_array_and_kernel` block-cooperative merge + small-size CPU fallback.
4. `bitmap_array_and_kernel` ballot + warp-prefix-sum instead of per-thread atomics.
5. Batch / defer the 4 cardinality D2H copies in `set_operation` into one.
6. `upload_ids.cu` scalar readback batching (3 separate 4-byte D2H → 1).

## Doc state

Already caught up after commit `2938ea9`:
- `PAPER.md` §3.6–3.9 + §4.6–4.7 cover negation-aware AND, complement, enumerate_ids, stream-ordered alloc.
- `REPORT.md` Shipped table includes everything through commit `84e82ba`.
- `README.md` project structure + bench list is consistent.
- `PAPER_TODOS.md` status tracker reflects the current state, and the Code Optimization Backlog section captures what's still open.

**Not yet written up**: the content of commit `84e82ba`. After validation tonight, the PAPER / REPORT / README should gain a short note that the device-resident pipeline is now the default. That's a 15-minute follow-up after the build is green.

Still not inlined into PAPER.md: the full roofline derivation from `analysis/roofline_model.md`. PAPER.md cites it but doesn't reproduce the algebra. PVLDB-submission task, not tonight.

## Sanity checklist before going back to PAPER drafting

- [ ] `git pull --rebase` clean
- [ ] `cmake --build . -j` clean, zero warnings with `-Werror`
- [ ] `ctest` green — especially multi_and, fused, promote, enumerate_ids
- [ ] `cuda-memcheck` clean on multi_and and promote tests
- [ ] `bench_multi_and` faster than the previous baseline at count ≥ 8
- [ ] `bench_alloc_strategy` still shows 1.9-3.3x pool tuning win
- [ ] Fresh benchmark JSONs saved under `results/raw/2026-04-13/`
- [ ] README and PAPER updated with the new multi_and numbers
- [ ] `PAPER_TODOS.md` backlog reviewed — pick item (1) as the next commit

If any bench regresses, the first suspect is the extra kernel-launch overhead in `fused_multi_and_allbitmap` — the old path had one big kernel; the new path has 7-8 small ones. Mitigations in order of invasiveness:

1. Fuse the presence set/reduce/popcount into a single kernel (easy, ~50 lines).
2. Reinstate the host-side fast path as an opt-in for `count × avg_containers < threshold` (medium).
3. Profile with `nsys` or `ncu` and see where the real time is going before touching anything.

If any test fails, the first suspect is `enumerate_presence_keys_kernel` — the block scan is the most bug-prone piece of new code. Compare its output against a simple single-thread reference enumeration to isolate.
