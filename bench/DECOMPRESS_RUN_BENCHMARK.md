# Decompress: run-container expansion — test & benchmark guide

How to verify and measure the **word-at-a-time run-container optimization** in
`src/decompress.cu` (`decompress_kernel`, the `ContainerType::RUN` branch).

## What changed

Previously each RUN container was expanded **bit-at-a-time**: the owning thread
issued one `atomicOr` per set bit, so a run of `L` bits cost `L` atomics — and
the 32 atomics that landed on each output word were serialized by the hardware.

Now each run is expanded **word-at-a-time**:

- **interior fully-covered words** → a plain `output[w] = 0xFFFFFFFFu` store (no atomic);
- **first / last partially-covered words** → one `atomicOr` with a bit mask.

Per run this is `~2 atomics + O(L/32) plain stores` instead of `O(L) atomics`.
The output bitset is **bit-identical** — only the write pattern changed.

**Performance hypothesis.** The run path speeds up by roughly
`1 / (1/32 + 2/avg_run_len)` — about **2x at very short runs, ~10x near 32-bit
runs, 25–32x for long runs**. Array- and bitmap-container decompress is
**untouched** and must not change.

## TL;DR

```bash
cd build
ctest -R Decompress --output-on-failure          # correctness gate
cmake --build . --target bench_decompress_runs -j
./bench/bench_decompress_runs                     # median + stddev table
```

For the before/after speedup number, see [§2](#2-benchmark-beforeafter).

## 1. Correctness

The optimization is exact, so the existing decompress suite is the gate.
`test/test_decompress.cu` verifies every bit against CRoaring (`verify_bitset`
checks cardinality, every CPU bit present, no extra bits). Run-relevant cases:

| Test | Exercises |
|---|---|
| `RunContainers` | multi-word runs, unaligned start (`[1000,5000)`), cross-container |
| `MixedContainerTypes` | a 20k-element run container beside array/bitmap |
| `FullContainer` | a single 65536-bit run — all-interior, both boundaries full |
| `BoundaryElement` | last bit of a container (65535) |

```bash
cd build
cmake --build . --target test_decompress -j
ctest -R Decompress --output-on-failure
```

A failure here means the port is wrong — **do not benchmark a failing build.**

**Optional extra coverage.** The one case the suite does not isolate is *two
runs in the same container sharing one output word* — the reason boundary
words still use `atomicOr` rather than a plain store. To pin it, add this to
`test_decompress.cu`:

```cpp
TEST_F(DecompressTest, RunsSharingBoundaryWord) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    roaring_bitmap_add_range(r, 10, 20);   // ends in word 0
    roaring_bitmap_add_range(r, 25, 40);   // starts in word 0, ends in word 1
    roaring_bitmap_run_optimize(r);
    auto gpu = cu_roaring::upload(r);
    uint32_t n_words = (gpu.universe_size + 31) / 32;
    uint32_t* d = cu_roaring::decompress_to_bitset(gpu);
    verify_bitset(r, d, n_words, gpu.universe_size);
    cudaFree(d); cu_roaring::gpu_roaring_free(gpu); roaring_bitmap_free(r);
}
```

## 2. Benchmark: before/after

`bench/bench_decompress_runs.cu` builds RUN-container-dominated bitmaps and
times `decompress_to_bitset`, sweeping average run length (32 → 16384 bits) at
fixed universe/density. It bakes in `Repetitions(10)`, so every row reports
`median` + `stddev` + `cv`, and it `SkipWithError`s if CRoaring did not
actually pick RUN containers — you can never silently measure the wrong path.

Because this is a **kernel change**, you measure the binary twice — once with
the old run code, once with the new — and diff.

### Step 1 — baseline (old bit-at-a-time run code)

If the optimization is still uncommitted, stash just that one file (the new
benchmark and this doc stay in place):

```bash
git stash push src/decompress.cu          # remove the optimization only
cd build && cmake --build . --target bench_decompress_runs -j
./bench/bench_decompress_runs \
    --benchmark_out=baseline.json --benchmark_out_format=json
cd .. && git stash pop                    # restore the optimization
```

If it is already committed, swap the stash for
`git checkout <parent-commit> -- src/decompress.cu` (baseline) and
`git checkout HEAD -- src/decompress.cu` (restore).

### Step 2 — optimized (new word-at-a-time run code)

```bash
cd build && cmake --build . --target bench_decompress_runs -j
./bench/bench_decompress_runs \
    --benchmark_out=optimized.json --benchmark_out_format=json
```

### Step 3 — compare

Google Benchmark ships a comparison tool (fetched with the dependency):

```bash
python3 build/_deps/googlebenchmark-src/tools/compare.py \
    benchmarks baseline.json optimized.json
```

It prints the per-row time delta and a U-test p-value (`pip install scipy` if
missing). A negative `time` delta is a speedup.

## 3. Reading the results

Per `BM_GPU_Decompress_Runs/universe/run_len` row:

- **`*_median` time** — the headline; lower is better.
- **`*_stddev` / `*_cv`** — spread. If `cv > ~5%`, rerun with
  `--benchmark_repetitions=30` (CLI overrides the baked-in 10), or check for a
  busy GPU / clock throttling.
- **`n_run_containers`** — sanity check it equals `n_containers` (the input is
  entirely RUN-encoded; if not, the row is not testing this path).

Expected shape after the optimization: median time **falls as `run_len`
grows**, and the speedup vs baseline climbs from ~2x (`run_len=32`, the
single-word case) toward ~25–32x (`run_len=16384`, deep interior loop).

> Report median **with** stddev when writing this up — a point estimate hides
> whether a "2x" at short runs clears the run-to-run noise envelope.

## 4. Control — non-run paths must be unchanged

The change only touches the RUN branch. Confirm it by running the existing
`bench_decompress` (array- and bitmap-dominated inputs) before and after:

```bash
./bench/bench_decompress \
    --benchmark_out=ctrl_baseline.json  --benchmark_out_format=json   # baseline build
./bench/bench_decompress \
    --benchmark_out=ctrl_optimized.json --benchmark_out_format=json   # optimized build
python3 build/_deps/googlebenchmark-src/tools/compare.py \
    benchmarks ctrl_baseline.json ctrl_optimized.json
```

Every row there should be flat within noise. A real delta in `bench_decompress`
means the edit leaked outside the RUN branch.

## 5. Optional — confirm the mechanism with ncu

To see atomic traffic actually drop, not just wall time:

```bash
ncu --kernel-name decompress_kernel --launch-count 1 \
    --section MemoryWorkloadAnalysis \
    ./bench/bench_decompress_runs --benchmark_filter='run_len:2048'
```

Compare baseline vs optimized: for the run-heavy launch, global atomic /
reduction traffic on `decompress_kernel` should collapse and global store
throughput should rise. (Exact metric names vary by GPU and ncu version — the
Memory Workload section surfaces the atomic-vs-store split either way.)
