# PAPER.md — Reviewer Requirements & Action Items

## Session Summary (March 22-23, 2026)

### What We Built
- **cu-roaring-bitmap library**: GPU Roaring bitmaps with 2-read query path, direct-map key index, cache-aware PROMOTE_AUTO, fused multi-AND, GPU-native upload pipeline
- **10 benchmark suites** (B1-B10) across synthetic and real-world data
- **cuVS/CAGRA integration** with one-line filter constructor
- **PAPER.md** draft with systems contribution framing
- **YFCC-10M Big-ANN benchmark** (B10) — downloaded and running on real data

### Key Results
| Metric | Value |
|--------|-------|
| Point query reads per lookup | 2 (down from 17) |
| vs bitset (10M, random) | 1.08x faster |
| vs bitset (1B/1%, warp) | 1.6x faster |
| Memory compression (1B/0.1%) | 58.4x (2.1 MB vs 125 MB) |
| Upload 100M IDs | 99 ms (66x vs CPU) |
| 8-way AND at 1B | 5.9 ms (39x vs CPU) |
| CAGRA search speedup (50% pass) | 1.31x |

### What's Downloaded
- **YFCC-10M dataset**: `/mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M/`
  - `base.10M.u8bin` (1.8 GB) — 10M × 192-dim uint8 CLIP vectors
  - `base.metadata.10M.spmat` (902 MB) — sparse tag matrix (10M × 200K, 108M assignments)
  - `query.public.100K.u8bin` (19 MB) — 100K query vectors
  - `query.metadata.public.100K.spmat` (1.9 MB) — query tag requirements
  - `GT.public.ibin` (7.7 MB) — ground truth (100K × 10 neighbors)
- **Exported data**: `cu-roaring-bitmap/bench/yfcc_data/` (316 MB)
  - 7,910 per-tag ID lists, query metadata, ground truth

---

## Benchmarks Needed

### 1. Big-ANN Filtered Track — DONE (standalone) / Needs CAGRA Integration
- **Status**: B10 benchmark runs on exported tag data (upload, AND, point queries)
- **Still needed**: Full CAGRA search on YFCC vectors with roaring filter vs bitset filter
- **Blocker**: cuVS standalone benchmark build has ABI mismatch (`std::experimental::extents`). Options:
  - (a) Full `./build.sh libcuvs` rebuild (~45 min) to pick up latest cu-roaring-bitmap headers
  - (b) Fix the `ivf_pq.hpp` header issue in the standalone build
- **Data location**: `big-ann-benchmarks/data/yfcc100M/`
- **Export script**: `bench/yfcc_export.py`
- **Commands**:
  ```bash
  # Re-export if needed
  python3 bench/yfcc_export.py \
    --data-dir /mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M \
    --out-dir bench/yfcc_data

  # Run standalone benchmark
  cd build && ./bench/bench_yfcc ../bench/yfcc_data
  ```

### 2. System Baselines (Critical)
- **ACORN**: CPU SOTA. Default in Weaviate and Vespa.
  - Paper: https://dl.acm.org/doi/10.1145/3654923
  - Need to clone, build, run on same YFCC-10M data
  - Compare: throughput (QPS) vs recall at various selectivities
- **VecFlow**: Only other GPU filtered search system. 5M QPS at 90% recall on A100.
  - Paper: https://arxiv.org/abs/2506.00812
  - Code: https://github.com/Supercomputing-System-AI-Lab/VecFlow
  - Need to build and run on RTX 5090 for fair comparison
  - Key difference: VecFlow uses label-centric IVF (pre-indexed), we use predicate-agnostic post-filtering

### 3. Scale the CAGRA Eval (Critical)
- **Current**: 1M vectors, batch=100, 3 selectivity points
- **Need**:
  - 10M vectors (YFCC data is ready)
  - Batch sizes: 100, 1K, 10K
  - Selectivity sweep: 0.1%, 0.5%, 1%, 5%, 10%, 20%, 50%, 70%, 90%, 99%
- **How**: Update `bench_cagra_roaring.cu` configs, rebuild cuVS benchmark

---

## Analysis Needed

### 4. Roofline Model (Important)
- Follow Crystal (SIGMOD 2020) template
- **Analytical prediction**: crossover at N = L2_size × 8 bits/byte
  - RTX 5090 (96 MB L2): ~768M vectors
  - A100 (40 MB L2): ~320M vectors
  - H100 (50 MB L2): ~400M vectors
- **Validate against B6 data**: the 1B results should match the prediction
- **Analysis framework**:
  ```
  Bitset:  1 read × (L2 hit if N < L2×8, else DRAM miss)
  Roaring: 2 reads × (key_index always L2, bitmap word depends on locality)
  Crossover: when bitset DRAM miss cost > roaring 2× L2 hit cost
  ```

---

## Writing Needed

### 5. Related Work Expansion (Critical)
- Current: 6 sentences, 8 references
- **Need**: ~2 pages, 15+ papers

**Filtered ANN algorithms:**
- **ACORN** — Predicate subgraph traversal on HNSW. SIGMOD 2024. https://dl.acm.org/doi/10.1145/3654923
- **Filtered-DiskANN** — Label-partitioned graphs. WWW 2023. (already cited)
- **VBase** — Relaxed monotonicity. OSDI 2023. https://www.usenix.org/conference/osdi23/presentation/zhang-qianxi
- **SeRF** — Segment graph for range-filtering. SIGMOD 2024.
- **UNG** — Unified Navigating Graph. arXiv 2024.
- **SIEVE** — Collection of indexes. arXiv 2025. https://arxiv.org/abs/2507.11907
- **DIGRA** — Dynamic graph for range-filtered ANN. arXiv 2025.
- **UNIFY** — Unified range filter index. VLDB 2025. https://www.vldb.org/pvldb/vol18/p1118-yao.pdf
- **PathFinder** — Conjunctions/disjunctions. arXiv 2025. https://arxiv.org/abs/2511.00995

**GPU systems:**
- **VecFlow** — Label-centric GPU IVF. SIGMOD 2025. https://dl.acm.org/doi/10.1145/3749189
- **GPU-WAH** — GPU compressed bitmaps. DEXA 2010. https://link.springer.com/chapter/10.1007/978-3-642-15251-1_26
- **GPU bitmap enhancements** — DASFAA 2020. https://dl.acm.org/doi/abs/10.1007/978-3-030-59419-0_21

**Benchmarks:**
- **Big-ANN NeurIPS'23** — https://big-ann-benchmarks.com/neurips23.html
- **Filtered ANN Benchmark** — arXiv 2025. https://arxiv.org/abs/2509.07789
- **ETH Benchmark** — 2025. http://htor.inf.ethz.ch/publications/img/2025_iff_fanns_benchmark.pdf

**Production systems:**
- **Milvus/Knowhere**, **Weaviate** (ACORN), **Vespa** (HNSW+filter)

---

## Venue Targeting
- **PVLDB rolling**: 1st of each month. Target **July 1, 2026**.
- **SIGMOD 2027 Industrial Track Round 3**: July 17, 2026 deadline. Strong option given NVIDIA ecosystem angle.

---

## Session Summary (March 24, 2026)

### What Was Produced
- **Selectivity sweep benchmark (B11)**: `bench/bench_selectivity_sweep.cu` — 13 selectivity points (0.1%–99%) × 4 universe sizes (1M–1B), comparing flat_bitset vs roaring::contains vs roaring::warp_contains. Added to `bench/CMakeLists.txt`. **Not yet built or run.**
- **Roofline model analysis**: `analysis/roofline_model.md` — full algebraic derivation of bitset vs roaring crossover points (RTX 5090 ~805M, A100 ~336M, H100 ~419M vectors), validated against B6 empirical data, analysis of 1B/10% warp anomaly. **Ready for integration into PAPER.md.**
- **Related work draft**: `analysis/related_work.md` — ~2,270 words, 7 subsections, 24 references covering filtered ANN algorithms (ACORN, DiskANN, VBase, SeRF, UNG, SIEVE, DIGRA, UNIFY, PathFinder), GPU systems (VecFlow, GPU-WAH), benchmarks, and production systems. **Ready for integration into PAPER.md.**
- **Baseline setup scripts**: `scripts/setup_baselines.sh` (ACORN + VecFlow clone/build), `scripts/run_baselines.py` (evaluation on YFCC-10M), `scripts/compare_results.py` (QPS-recall comparison tables). **Not yet run.**

### Known Issues (from March 24 early session)
- VecFlow only ships a CUDA 11 wheel — building from source may be needed on CUDA 12
- ACORN's filter_ids_map is O(nq × N) bytes — run script batches queries to stay under 500MB

---

## Session Summary (March 24-25, 2026 — cuVS Integration)

### What Was Done
1. **cuVS ABI investigation** — root cause identified: libcuvs.so uses Kokkos `std::experimental::mdspan`, conda libs use `cuda::std::mdspan`, system CUDA 12.4 uses `cuda::__4::mr::resource_ref` vs conda's `cuda::mr::__4::basic_resource_ref`. All three are incompatible.
2. **Working build configuration** — conda `cuvs` env headers + local libcuvs.so + shared cu-roaring-bitmap lib. See memory file `project_cuvs_abi.md` for full build commands.
3. **CUB ODR crash fixed** — cu-roaring-bitmap must be built as SHARED library (`-DBUILD_SHARED_LIBS=ON`) to isolate CUB device symbols from conda's CUB.
4. **Comprehensive benchmark** (`bench_cagra_roaring_comprehensive.cu`) — 15 search configs (1M + 10M, 0.1%-99% selectivity) + 3 multi-AND configs on RTX 5090.
5. **Report generated** — `cuvs/cpp/bench/prims/core/ROARING_BENCHMARK_REPORT.md`
6. **cuVS RAII wrapper updated** — `roaring.hpp` (key_index, negated, total_cardinality fields), `roaring.cu` (builds key_index in from_sorted_ids)

### Benchmark Results (decompress-to-bitset path)
| Metric | Result |
|--------|--------|
| Search perf (roaring vs bitset) | ~identical (both use bitset_filter after decompression) |
| Recall@10 | 0.95-1.00 across all configs |
| Compression at 1M | 0.9x (no advantage — all containers promoted to bitmap) |
| Compression at 10M | 1.0x (same — 153 bitmap containers) |
| Build time (roaring vs bitset) | 0.5-6ms vs 0.08ms (roaring slower due to GPU sort) |
| Multi-AND (roaring vs bitset) | 0.2ms vs 0.007ms (roaring slower at 1M) |
| Complement optimization | Works correctly at >50% selectivity |

### What the Benchmark Does NOT Show (needs rebuilt libcuvs)
- **Direct roaring_filter kernel** — warp-cooperative 2-read path without decompression. Expected to show 10-30% search speedup (from cu-roaring-bitmap's own B6 benchmarks).
- **Billion-scale compression** — 1B/0.1% → 59x compression (2.1MB vs 125MB). Current benchmark maxes at 10M.
- **Multi-AND on compressed data** — at billion scale, roaring AND avoids full O(N/8) scans.

### Key Technical Discoveries
- **RTTI mismatch** — `dynamic_cast<roaring_filter&>` fails when benchmark and libcuvs are compiled with different headers, causing silent fallback to bitset_filter (reads wrong memory → recall=0). Workaround: decompress to bitset.
- **CUB ODR violation** — static linking cu-roaring-bitmap.a mixes CUB device symbols from two CCCL versions. Fix: shared library.
- **RMM header migration** — new RMM moves `rmm/mr/device/*.hpp` to `rmm/mr/*.hpp`, breaking cuVS source.

---

## Status Tracker

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| YFCC-10M data download | **DONE** | Critical | 2.8 GB in `big-ann-benchmarks/data/yfcc100M/` |
| YFCC export script | **DONE** | Critical | `bench/yfcc_export.py` → 7,910 tag files |
| B10 standalone benchmark | **DONE** | Critical | Upload + AND + query on real tags |
| CAGRA benchmark (decompress path) | **DONE** | Critical | 15 configs + 3 multi-AND on RTX 5090 |
| CAGRA benchmark (direct roaring path) | **Blocked** | Critical | Needs libcuvs rebuild with matching headers |
| ACORN baseline | **Scripts ready** | Critical | `scripts/setup_baselines.sh` |
| VecFlow baseline | **Scripts ready** | Critical | CUDA 11 wheel caveat |
| Baseline comparison | **Scripts ready** | Critical | `scripts/run_baselines.py` + `scripts/compare_results.py` |
| Selectivity sweep (B11) | **Code written** | Critical | `bench/bench_selectivity_sweep.cu` — needs build + run |
| Roofline model | **Draft done** | Important | `analysis/roofline_model.md` — needs PAPER.md integration |
| Related work (2 pages) | **Draft done** | Critical | `analysis/related_work.md` — needs PAPER.md integration |
| cuVS RAII wrapper | **Updated** | Medium | roaring.hpp + roaring.cu (key_index, negated, total_cardinality) |
| cuVS benchmark report | **DONE** | Critical | `ROARING_BENCHMARK_REPORT.md` (decompress path) |
| Paper draft revision | In progress | Critical | PAPER.md needs eval + related work sections |
| PVLDB July 1 submission | Target | — | ~3 months from now |

---

## File Locations

```
cu-roaring-bitmap/
├── PAPER.md                    — Draft paper
├── PAPER_TODOS.md              — This file
├── REPORT.md                   — Technical report
├── README.md                   — Library documentation
├── analysis/
│   ├── roofline_model.md       — Roofline model analysis (NEW, ready for PAPER.md)
│   └── related_work.md         — Related work draft (NEW, ready for PAPER.md)
├── bench/
│   ├── yfcc_export.py          — YFCC data export script
│   ├── bench_yfcc.cu           — B10: YFCC standalone benchmark
│   ├── bench_selectivity_sweep.cu — B11: selectivity sweep (NEW, needs build+run)
│   ├── bench_point_query.cu    — B6: point query throughput
│   ├── bench_optimized_query.cu — B7: optimization analysis
│   ├── bench_upload_scale.cu   — B8: upload latency at scale
│   ├── bench_multi_and.cu      — B9: fused multi-AND
│   ├── bench_comprehensive.cu  — B1/B3/B4/B5
│   └── yfcc_data/              — Exported YFCC tag data (gitignored)
├── scripts/
│   ├── setup_baselines.sh      — Clone/build ACORN + VecFlow (NEW)
│   ├── run_baselines.py        — Evaluate baselines on YFCC-10M (NEW)
│   └── compare_results.py      — Generate comparison tables (NEW)
└── results/raw/                — All benchmark JSON results

big-ann-benchmarks/             — NeurIPS'23 benchmark framework
└── data/yfcc100M/              — Downloaded YFCC-10M dataset (2.8 GB)

cuvs/                           — cuVS fork with roaring integration
├── cpp/include/cuvs/core/roaring.hpp          — Updated (key_index, negated, total_cardinality)
├── cpp/include/cuvs/neighbors/roaring_filter.cuh
├── cpp/src/core/roaring/roaring.cu            — Updated (builds key_index in from_sorted_ids)
└── cpp/bench/prims/core/
    ├── bench_cagra_roaring.cu                 — Original benchmark (5 configs)
    ├── bench_cagra_roaring_comprehensive.cu   — NEW: 15 search + 3 multi-AND configs
    ├── generate_report.py                     — NEW: JSON → markdown report
    ├── ROARING_BENCHMARK_REPORT.md            — NEW: Generated comparison report
    ├── bench_cagra_roaring_comprehensive.json — NEW: Raw benchmark data
    └── CMakeLists.txt                         — Updated (conda paths, shared lib, CRoaring link)
```
