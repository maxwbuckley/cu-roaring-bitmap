# PAPER.md — Reviewer Requirements & Action Items

## Benchmarks Needed

### 1. Big-ANN Filtered Track (Critical)
- **Dataset**: 10M YFCC, 192-dim CLIP embeddings (uint8), 200K-vocab tags
- **Why**: Gold standard since 2023. Every filtered ANN paper uses it. Without it, VLDB/SIGMOD reviewers won't take eval seriously.
- **Source**: https://big-ann-benchmarks.com/neurips23.html (filtered track)
- **Repo**: https://github.com/harsha-simhadri/big-ann-benchmarks
- **Download**: `python create_dataset.py --dataset yfcc-10M`
- **Run**: `python run.py --neurips23track filter --algorithm faiss --dataset yfcc-10M`
- **Details**: 10M vectors, 192-dim uint8 (CLIP), L2 distance, 100K queries, each query has 1-2 required tags from 200K vocab
- **Also need**: SIFT + synthetic filters at varying selectivity

### 2. System Baselines (Critical)
- **ACORN**: CPU SOTA for filtered ANN. Default in Weaviate and Vespa. Must compare.
- **VecFlow**: Only other GPU filtered search system. Direct competitor.
- **Current gap**: We only compare against CAGRA's native bitset filter.

### 3. Scale the CAGRA Eval (Critical)
- Current: 1M vectors, batch=100 — too small for a GPU paper
- **Need**: 10M+ vectors, 1K+ batch
- **Selectivity sweep**: 8-10 points from 0.1% to 99% (currently only 3 points: 1%, 10%, 50%)

## Analysis Needed

### 4. Roofline Model (Important)
- Follow Crystal (SIGMOD 2020) template
- Model analytically when cu-roaring's 2-read path beats flat bitset's 1-read path

**Analysis framework:**

```
Bitset cost per query:
  If bitset ≤ L2:  ~4 cycles (L2 hit)
  If bitset > L2:  ~400 cycles (DRAM access, ~200ns at 5 GHz)

Roaring cost per query (2-read path):
  Read 1: key_index[key] — 30KB table, always L2-resident → ~4 cycles
  Read 2: bitmap_data[idx*1024 + low/64] — 8KB per container
    If hot container in L2: ~4 cycles
    If cold container:      ~400 cycles (DRAM)
  Total:  ~8 cycles (L2) to ~404 cycles (cold)

Crossover: bitset DRAM (~400 cyc) > roaring L2+DRAM (~404 cyc)
  → When bitset > L2 AND queries have enough container locality
  → N > L2_size * 8 = 96MB * 8 = 768M vectors
```

**Variables to sweep:**
- N (universe size): 1M to 10B
- L2 cache size: 40MB (A100), 50MB (H100), 96MB (RTX 5090)
- Query locality: random vs graph-traversal (CAGRA-like)

**Expected result:** Crossover at ~768M on RTX 5090, ~400M on A100/H100. Below crossover, bitset wins by ~2x (1 read vs 2). Above crossover, roaring wins by 1.5-2x (L2 hits vs DRAM misses).

## Writing Needed

### 5. Related Work Expansion (Critical)
- Current: 6 sentences, 8 references
- **Need**: ~2 pages, 15+ filtered ANN papers since 2022

Papers to cover (with venues and URLs):

**Filtered ANN algorithms:**
- **ACORN** — Predicate-agnostic search via predicate subgraph traversal on HNSW. CPU SOTA, default in Weaviate+Vespa. SIGMOD 2024. https://dl.acm.org/doi/10.1145/3654923
- **Filtered-DiskANN** — Label-partitioned graphs on disk. WWW 2023. (already cited)
- **VBase** — Relaxed monotonicity for vector+attribute queries. OSDI 2023 (not SIGMOD). https://www.usenix.org/conference/osdi23/presentation/zhang-qianxi
- **SeRF** — Segment graph for range-filtering ANN, compresses multiple HNSWs. SIGMOD 2024. Proc. ACM Manag. Data 2, 1, Article 69.
- **UNG** — Unified Navigating Graph for filtered search. arXiv 2024.
- **SIEVE** — Collection of indexes for effective filtered vector search. arXiv 2025 (July). https://arxiv.org/abs/2507.11907
- **DIGRA** — Dynamic incremental graph for range-filtered ANN. arXiv 2025 (December).
- **UNIFY** — Unified index for range filtered ANN. VLDB 2025. https://www.vldb.org/pvldb/vol18/p1118-yao.pdf
- **PathFinder** — Conjunctions and disjunctions for filtered ANN. arXiv 2025 (November). https://arxiv.org/abs/2511.00995

**GPU systems:**
- **VecFlow** — Label-centric IVF on GPU. 5M QPS at 90% recall. SIGMOD 2025. https://arxiv.org/abs/2506.00812 / https://dl.acm.org/doi/10.1145/3749189
- **GPU-WAH** — GPU word-aligned hybrid compressed bitmaps. Andrzejewski & Wrembel, DEXA 2010. 54x speedup over CPU. https://link.springer.com/chapter/10.1007/978-3-642-15251-1_26
- **GPU bitmap index enhancements** — Tran et al., DASFAA 2020. Metadata creation, memory pools. https://dl.acm.org/doi/abs/10.1007/978-3-030-59419-0_21

**Benchmarks:**
- **Big-ANN NeurIPS'23** — Filtered track with YFCC-10M. https://big-ann-benchmarks.com/neurips23.html
- **Filtered ANN Benchmark** — Unified benchmark and systematic study. arXiv 2025 (September). https://arxiv.org/abs/2509.07789
- **ETH Benchmark** — Benchmarking filtered ANN on transformer embeddings. Htor et al., 2025. http://htor.inf.ethz.ch/publications/img/2025_iff_fanns_benchmark.pdf

**Production systems:**
- **Milvus/Knowhere** — Production filtered search system
- **Weaviate** — ACORN integration for predicate-agnostic search
- **Vespa** — HNSW+filter production deployment

### 6. Venue Targeting
- **PVLDB rolling**: 1st of each month. Target June 1 or July 1 2026 after adding benchmarks.
- **SIGMOD 2027 Industrial Track Round 3**: July 17, 2026 deadline. Strong option given NVIDIA ecosystem angle.

## Status

| Item | Status | Priority |
|------|--------|----------|
| Big-ANN filtered track benchmark | Not started | Critical |
| ACORN baseline comparison | Not started | Critical |
| VecFlow baseline comparison | Not started | Critical |
| Scale CAGRA to 10M+, batch 1K+ | Not started | Critical |
| Selectivity sweep (0.1% to 99%) | Not started | Critical |
| Roofline model | Not started | Important |
| Related work expansion | Not started | Critical |
| PVLDB June 1 submission | Target date | — |
