#!/usr/bin/env python3
"""
run_baselines.py -- Evaluate ACORN and VecFlow on YFCC-10M filtered track.

Runs each baseline system with the same queries and filters as our cu-roaring-bitmap
benchmarks, measuring QPS vs recall at various selectivities.

Usage:
    # Run all baselines:
    python3 scripts/run_baselines.py \
        --data-dir /mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M \
        --yfcc-export-dir bench/yfcc_data \
        --output-dir results/baselines

    # Run only ACORN:
    python3 scripts/run_baselines.py --acorn-only ...

    # Run only VecFlow:
    conda activate vecflow
    python3 scripts/run_baselines.py --vecflow-only ...

Prerequisites:
    - Run scripts/setup_baselines.sh first to build ACORN and install VecFlow
    - Run bench/yfcc_export.py first to export per-tag ID lists
    - YFCC-10M dataset downloaded to big-ann-benchmarks/data/yfcc100M/

Output:
    results/baselines/acorn_results.json
    results/baselines/vecflow_results.json
    Each file contains QPS-recall curves at multiple selectivity points.

Data format notes:
    - YFCC-10M base vectors: uint8, 192-dim. Both baselines require float32.
    - YFCC-10M tags: sparse matrix (CSR). Per-tag ID lists in bench/yfcc_data/tags/
    - Queries: 100K queries, each requiring 1-2 tags (AND semantics)
    - Ground truth: 100K x 10 nearest neighbors (with filter applied)
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Data Loading Utilities
# =============================================================================

def read_u8bin(filename: str) -> np.ndarray:
    """Read uint8 binary vectors in Big-ANN format: [nvecs:i32, dim:i32, data...]."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        arr = np.fromfile(f, dtype=np.uint8, count=int(nvecs) * int(dim))
    return arr.reshape(int(nvecs), int(dim))


def read_ibin(filename: str) -> np.ndarray:
    """Read int32 binary vectors in Big-ANN format."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        arr = np.fromfile(f, dtype=np.int32, count=int(nvecs) * int(dim))
    return arr.reshape(int(nvecs), int(dim))


def read_sparse_matrix(fname: str):
    """Read CSR sparse matrix in .spmat format (Big-ANN convention)."""
    from scipy.sparse import csr_matrix
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=int(nrow) + 1)
        indices = np.fromfile(f, dtype="int32", count=int(nnz))
        data = np.fromfile(f, dtype="float32", count=int(nnz))
    return csr_matrix((data, indices, indptr), shape=(int(nrow), int(ncol)))


def load_exported_queries(query_file: str) -> list:
    """Load queries exported by yfcc_export.py (binary format)."""
    queries = []
    with open(query_file, "rb") as f:
        (n_queries,) = struct.unpack("I", f.read(4))
        for _ in range(n_queries):
            (n_tags,) = struct.unpack("I", f.read(4))
            tag_ids = struct.unpack(f"{n_tags}I", f.read(4 * n_tags))
            queries.append(list(tag_ids))
    return queries


def load_exported_groundtruth(gt_file: str) -> np.ndarray:
    """Load ground truth exported by yfcc_export.py: [nq:u32, k:u32, ids...]."""
    with open(gt_file, "rb") as f:
        nq, k = struct.unpack("II", f.read(8))
        gt = np.frombuffer(f.read(nq * k * 4), dtype=np.uint32).reshape(nq, k)
    return gt


def load_tag_ids(tag_dir: str, tag_id: int) -> np.ndarray:
    """Load per-tag ID list from binary file (exported by yfcc_export.py)."""
    path = os.path.join(tag_dir, f"tag_{tag_id}.bin")
    with open(path, "rb") as f:
        n_ids, stored_tag = struct.unpack("II", f.read(8))
        ids = np.frombuffer(f.read(n_ids * 4), dtype=np.uint32)
    return ids


def compute_recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Compute recall@k.
    results: (nq, k_result) array of retrieved neighbor IDs
    ground_truth: (nq, k_gt) array of true neighbor IDs
    Returns: average fraction of true top-k neighbors found.
    """
    nq = results.shape[0]
    k_gt = min(k, ground_truth.shape[1])
    k_res = min(k, results.shape[1])
    recall_sum = 0.0
    for i in range(nq):
        gt_set = set(ground_truth[i, :k_gt].tolist())
        # Remove sentinel values (-1 or very large values indicating no neighbor)
        gt_set.discard(-1)
        gt_set.discard(0xFFFFFFFF)
        if len(gt_set) == 0:
            continue
        found = len(gt_set.intersection(results[i, :k_res].tolist()))
        recall_sum += found / len(gt_set)
    return recall_sum / nq


def build_per_vector_labels(base_meta_path: str) -> list:
    """
    Build per-vector label lists from the sparse metadata matrix.
    Returns: list of lists, labels[i] = [tag_id_0, tag_id_1, ...] for vector i.

    WARNING: This is memory-intensive for 10M vectors. ~2-4 GB.
    """
    print("  Loading sparse metadata matrix...")
    meta = read_sparse_matrix(base_meta_path)
    n_vectors = meta.shape[0]
    print(f"  Building per-vector label lists for {n_vectors} vectors...")
    labels = []
    # CSR format: iterate rows efficiently
    for i in range(n_vectors):
        start, end = meta.indptr[i], meta.indptr[i + 1]
        labels.append(meta.indices[start:end].tolist())
        if (i + 1) % 1_000_000 == 0:
            print(f"    {i+1}/{n_vectors}...")
    return labels


def select_queries_by_selectivity(
    queries: list,
    tag_dir: str,
    n_vectors: int,
    target_bins: list,
) -> dict:
    """
    Group queries into selectivity bins.
    Returns: dict mapping bin_label -> list of (query_idx, tag_ids, selectivity).

    target_bins: list of (label, lo_pct, hi_pct) tuples defining selectivity ranges.
    """
    # Cache tag sizes
    tag_size_cache = {}

    def get_selectivity(tag_ids):
        """Compute selectivity for an AND of tags (approximate: product of individual)."""
        # For exact selectivity we'd need to intersect, but the tag files
        # give individual counts. We use the minimum tag's selectivity
        # as an upper bound (since AND can only reduce).
        min_count = n_vectors
        for tid in tag_ids:
            if tid not in tag_size_cache:
                try:
                    ids = load_tag_ids(tag_dir, tid)
                    tag_size_cache[tid] = len(ids)
                except FileNotFoundError:
                    tag_size_cache[tid] = 0
            min_count = min(min_count, tag_size_cache[tid])
        return min_count / n_vectors

    bins = {label: [] for label, _, _ in target_bins}
    for qi, tag_ids in enumerate(queries):
        sel = get_selectivity(tag_ids)
        for label, lo, hi in target_bins:
            if lo <= sel < hi:
                bins[label].append((qi, tag_ids, sel))
                break

    return bins


# =============================================================================
# ACORN Evaluation
# =============================================================================

def run_acorn_evaluation(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    queries: list,
    ground_truth: np.ndarray,
    base_meta_path: str,
    tag_dir: str,
    output_file: str,
    n_vectors: int,
    max_queries_per_bin: int = 500,
):
    """
    Run ACORN (FAISS IndexACORNFlat) on YFCC-10M with filtered queries.

    ACORN's filtered search API requires a dense boolean filter_ids_map of shape
    (n_queries, n_vectors). For 10M vectors this is 10MB per query, so we must
    batch queries carefully.
    """
    print("\n" + "=" * 70)
    print("ACORN EVALUATION")
    print("=" * 70)

    try:
        import faiss
    except ImportError:
        print("ERROR: ACORN/FAISS not importable.")
        print("  Build ACORN first: bash scripts/setup_baselines.sh --acorn-only")
        print("  Then ensure the Python bindings are installed:")
        print("    cd baselines/ACORN/build/faiss/python && pip install -e .")
        return None

    # Check that this is the ACORN fork (has IndexACORNFlat)
    if not hasattr(faiss, "IndexACORNFlat"):
        print("ERROR: This FAISS installation does not have IndexACORNFlat.")
        print("  You need the ACORN fork: https://github.com/stanford-futuredata/ACORN")
        print("  Standard FAISS does not include the ACORN index type.")
        return None

    dim = base_vectors.shape[1]
    n_base = base_vectors.shape[0]
    k = 10  # top-10, matching ground truth

    # Convert uint8 -> float32 (ACORN/FAISS requires float32)
    print(f"Converting {n_base} base vectors from uint8 to float32...")
    t0 = time.time()
    base_f32 = base_vectors.astype(np.float32)
    print(f"  Conversion: {time.time() - t0:.1f}s, {base_f32.nbytes / 1e9:.1f} GB")

    query_f32 = query_vectors.astype(np.float32)

    # ---- Build ACORN index ----
    # Parameters from the ACORN paper:
    #   M=32 (HNSW degree), gamma=12 (ACORN expansion), M_beta=64
    # For 10M vectors, this takes several minutes.
    M = 32
    gamma = 12
    M_beta = 64

    print(f"\nBuilding ACORN index: dim={dim}, M={M}, gamma={gamma}, M_beta={M_beta}")
    print(f"  This will take several minutes for {n_base/1e6:.0f}M vectors...")
    t0 = time.time()
    index = faiss.IndexACORNFlat(dim, M, gamma, M_beta)
    index.add(n_base, base_f32)
    build_time = time.time() - t0
    print(f"  Index built in {build_time:.1f}s")

    # ---- Load tag membership for filter construction ----
    # We need to build filter_ids_map: for each query, which base vectors pass.
    # Strategy: precompute per-tag bitsets, then AND them per query.
    print("\nPrecomputing per-tag membership sets for filter construction...")
    tag_sets = {}  # tag_id -> set of vector IDs
    all_tags_needed = set()
    for tag_ids in queries:
        all_tags_needed.update(tag_ids)

    for tid in sorted(all_tags_needed):
        try:
            ids = load_tag_ids(tag_dir, tid)
            tag_sets[tid] = set(ids.tolist())
        except FileNotFoundError:
            print(f"  WARNING: tag_{tid}.bin not found, skipping")
            tag_sets[tid] = set()
    print(f"  Loaded {len(tag_sets)} tag membership sets")

    # ---- Group queries by selectivity ----
    selectivity_bins = [
        ("0.1%",  0.0,    0.005),
        ("1%",    0.005,  0.05),
        ("5%",    0.03,   0.08),
        ("10%",   0.05,   0.15),
        ("20%",   0.15,   0.30),
        ("50%",   0.30,   0.60),
        (">50%",  0.60,   1.01),
    ]
    query_bins = select_queries_by_selectivity(
        queries, tag_dir, n_vectors, selectivity_bins
    )

    # ---- Sweep efSearch for recall-QPS curves ----
    ef_values = [10, 20, 40, 80, 120, 200, 400, 800]

    results = {
        "system": "ACORN",
        "version": "SIGMOD2024",
        "index_params": {"M": M, "gamma": gamma, "M_beta": M_beta, "dim": dim},
        "build_time_s": build_time,
        "n_vectors": n_base,
        "k": k,
        "curves": [],
    }

    for bin_label, bin_queries in query_bins.items():
        if not bin_queries:
            print(f"\n  Selectivity bin '{bin_label}': no queries, skipping")
            continue

        # Limit queries per bin to keep runtime reasonable
        sample = bin_queries[:max_queries_per_bin]
        nq = len(sample)
        avg_sel = np.mean([s for _, _, s in sample])
        print(f"\n  Selectivity bin '{bin_label}': {nq} queries (avg selectivity {avg_sel:.3%})")

        # Build filter_ids_map for this batch
        # filter_ids_map[q * n_base + v] = 1 if vector v passes query q's filter
        # Memory: nq * n_base bytes. For 500 queries * 10M = 5 GB. Batch further.
        BATCH = min(nq, max(1, 500_000_000 // n_base))  # ~500MB per batch
        print(f"    Processing in batches of {BATCH} queries ({BATCH * n_base / 1e9:.1f} GB filter map)")

        for ef in ef_values:
            index.acorn.efSearch = ef

            all_neighbors = np.zeros((nq, k), dtype=np.int64)
            all_distances = np.zeros((nq, k), dtype=np.float32)
            total_search_us = 0.0

            for batch_start in range(0, nq, BATCH):
                batch_end = min(batch_start + BATCH, nq)
                batch_nq = batch_end - batch_start
                batch_queries_f32 = query_f32[[s[0] for s in sample[batch_start:batch_end]]]

                # Build filter map
                filter_map = np.zeros(batch_nq * n_base, dtype=np.int8)
                for bqi, (qi, tag_ids, _) in enumerate(sample[batch_start:batch_end]):
                    # Compute intersection of tag sets (AND semantics)
                    if len(tag_ids) == 1:
                        passing = tag_sets.get(tag_ids[0], set())
                    else:
                        passing = tag_sets.get(tag_ids[0], set())
                        for tid in tag_ids[1:]:
                            passing = passing & tag_sets.get(tid, set())
                    for vid in passing:
                        filter_map[bqi * n_base + vid] = 1

                # Search with filter
                # The ACORN FAISS fork overloads index.search() with an extra
                # char* filter_id_map parameter. The SWIG Python binding may
                # expose this in several ways depending on the fork version:
                #   1. index.search(x, k, filter_id_map)  -- extra positional arg
                #   2. index.search_with_filter(x, k, filter_id_map)
                #   3. Standard FAISS IDSelector mechanism
                # We try each in order and cache the working method.
                filter_map_bytes = filter_map.tobytes()

                t0 = time.perf_counter()
                search_succeeded = False

                # Method 1: ACORN's overloaded search with filter_id_map
                if not search_succeeded:
                    try:
                        D, I = index.search(batch_queries_f32, k, filter_map_bytes)
                        search_succeeded = True
                    except TypeError:
                        pass

                # Method 2: Separate search_with_filter method
                if not search_succeeded:
                    try:
                        D, I = index.search_with_filter(
                            batch_queries_f32, k, filter_map_bytes
                        )
                        search_succeeded = True
                    except (TypeError, AttributeError):
                        pass

                # Method 3: Pass as numpy array directly (some SWIG wrappers)
                if not search_succeeded:
                    try:
                        D, I = index.search(batch_queries_f32, k, filter_map)
                        search_succeeded = True
                    except TypeError:
                        pass

                # Method 4: Fall back to unfiltered search + post-filter
                if not search_succeeded:
                    print("      WARNING: Could not invoke ACORN filtered search.")
                    print("      Falling back to over-retrieval + post-filter.")
                    print("      This gives weaker results than ACORN's native filtering.")
                    k_over = k * 100
                    D_raw, I_raw = index.search(batch_queries_f32, k_over)
                    # Post-filter
                    D = np.full((batch_nq, k), float("inf"), dtype=np.float32)
                    I = np.full((batch_nq, k), -1, dtype=np.int64)
                    for qi_local in range(batch_nq):
                        count = 0
                        for j in range(k_over):
                            vid = I_raw[qi_local, j]
                            if vid >= 0 and filter_map[qi_local * n_base + vid]:
                                I[qi_local, count] = vid
                                D[qi_local, count] = D_raw[qi_local, j]
                                count += 1
                                if count >= k:
                                    break

                elapsed_us = (time.perf_counter() - t0) * 1e6
                total_search_us += elapsed_us

                all_neighbors[batch_start:batch_end] = I
                all_distances[batch_start:batch_end] = D

            # Compute recall against ground truth
            gt_subset = ground_truth[[s[0] for s in sample]]
            recall = compute_recall_at_k(all_neighbors.astype(np.uint32), gt_subset, k)
            qps = nq / (total_search_us / 1e6)

            print(f"    efSearch={ef:4d}: recall@{k}={recall:.4f}, QPS={qps:.0f}")
            results["curves"].append({
                "selectivity_bin": bin_label,
                "avg_selectivity": float(avg_sel),
                "n_queries": nq,
                "efSearch": ef,
                "recall_at_k": float(recall),
                "k": k,
                "qps": float(qps),
                "total_search_us": float(total_search_us),
            })

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ACORN results saved to {output_file}")
    return results


# =============================================================================
# VecFlow Evaluation
# =============================================================================

def run_vecflow_evaluation(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    queries: list,
    ground_truth: np.ndarray,
    base_meta_path: str,
    tag_dir: str,
    output_file: str,
    n_vectors: int,
    max_queries_per_bin: int = 1000,
):
    """
    Run VecFlow on YFCC-10M with filtered queries.

    VecFlow uses a label-centric IVF approach: it builds per-label subindices
    at index construction time. This is fundamentally different from our
    predicate-agnostic post-filtering approach.
    """
    print("\n" + "=" * 70)
    print("VECFLOW EVALUATION")
    print("=" * 70)

    try:
        from vecflow import VecFlow
    except ImportError:
        print("ERROR: VecFlow not importable.")
        print("  Install VecFlow first:")
        print("    conda activate vecflow")
        print("    bash scripts/setup_baselines.sh --vecflow-only")
        return None

    dim = base_vectors.shape[1]
    n_base = base_vectors.shape[0]
    k = 10  # top-10, matching ground truth

    # Convert uint8 -> float32
    print(f"Converting {n_base} base vectors from uint8 to float32...")
    t0 = time.time()
    base_f32 = base_vectors.astype(np.float32)
    print(f"  Conversion: {time.time() - t0:.1f}s, {base_f32.nbytes / 1e9:.1f} GB")

    query_f32 = query_vectors.astype(np.float32)

    # ---- Build per-vector label lists ----
    # VecFlow needs: data_labels = list[list[int]], one entry per vector
    print("\nBuilding per-vector label lists from sparse metadata...")
    data_labels = build_per_vector_labels(base_meta_path)
    print(f"  Built labels for {len(data_labels)} vectors")

    # ---- Build VecFlow index ----
    # VecFlow parameters:
    #   graph_degree: CAGRA graph degree (16 default)
    #   specificity_threshold: labels with >= T vectors use IVF-CAGRA, else IVF-BFS
    #     For YFCC with Zipfian distribution, 2000 is a reasonable default.
    graph_degree = 16
    specificity_threshold = 2000

    print(f"\nBuilding VecFlow index: graph_degree={graph_degree}, "
          f"specificity_threshold={specificity_threshold}")
    print(f"  This builds per-label subindices for all labels. May take a while...")

    # VecFlow saves index to files. Use a temp directory.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vecflow_") as tmpdir:
        graph_fname = os.path.join(tmpdir, "graph.bin")
        bfs_fname = os.path.join(tmpdir, "bfs.bin")

        t0 = time.time()
        vf = VecFlow()
        vf.build(
            dataset=base_f32,
            data_labels=data_labels,
            graph_degree=graph_degree,
            specificity_threshold=specificity_threshold,
            ivf_graph_fname=graph_fname,
            ivf_bfs_fname=bfs_fname,
        )
        build_time = time.time() - t0
        print(f"  VecFlow index built in {build_time:.1f}s")

        # Free the label lists -- no longer needed
        del data_labels

        # ---- Group queries by selectivity ----
        selectivity_bins = [
            ("0.1%",  0.0,    0.005),
            ("1%",    0.005,  0.05),
            ("5%",    0.03,   0.08),
            ("10%",   0.05,   0.15),
            ("20%",   0.15,   0.30),
            ("50%",   0.30,   0.60),
            (">50%",  0.60,   1.01),
        ]
        query_bins = select_queries_by_selectivity(
            queries, tag_dir, n_vectors, selectivity_bins
        )

        # ---- Sweep itopk_size for recall-QPS curves ----
        itopk_values = [16, 32, 64, 128, 256, 512]

        results = {
            "system": "VecFlow",
            "version": "v0.0.1_SIGMOD2026",
            "index_params": {
                "graph_degree": graph_degree,
                "specificity_threshold": specificity_threshold,
                "dim": dim,
            },
            "build_time_s": build_time,
            "n_vectors": n_base,
            "k": k,
            "curves": [],
        }

        for bin_label, bin_queries in query_bins.items():
            if not bin_queries:
                print(f"\n  Selectivity bin '{bin_label}': no queries, skipping")
                continue

            sample = bin_queries[:max_queries_per_bin]
            nq = len(sample)
            avg_sel = np.mean([s for _, _, s in sample])
            print(f"\n  Selectivity bin '{bin_label}': {nq} queries (avg selectivity {avg_sel:.3%})")

            batch_queries_f32 = query_f32[[s[0] for s in sample]]
            batch_query_labels = [tag_ids for _, tag_ids, _ in sample]

            for itopk in itopk_values:
                # Warmup (3 runs)
                for _ in range(3):
                    vf.search(
                        queries=batch_queries_f32,
                        query_labels=batch_query_labels,
                        itopk_size=itopk,
                    )

                # Timed runs (10 iterations for GPU stability)
                times = []
                all_neighbors = None
                for _ in range(10):
                    t0 = time.perf_counter()
                    neighbors, distances = vf.search(
                        queries=batch_queries_f32,
                        query_labels=batch_query_labels,
                        itopk_size=itopk,
                    )
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    if all_neighbors is None:
                        all_neighbors = neighbors

                median_time = np.median(times)
                mean_time = np.mean(times)
                std_time = np.std(times)

                # Compute recall
                gt_subset = ground_truth[[s[0] for s in sample]]
                recall = compute_recall_at_k(
                    np.asarray(all_neighbors, dtype=np.uint32)[:, :k],
                    gt_subset,
                    k,
                )
                qps = nq / median_time

                print(f"    itopk={itopk:4d}: recall@{k}={recall:.4f}, "
                      f"QPS={qps:.0f} (median={median_time*1e3:.1f}ms, "
                      f"std={std_time*1e3:.2f}ms)")

                results["curves"].append({
                    "selectivity_bin": bin_label,
                    "avg_selectivity": float(avg_sel),
                    "n_queries": nq,
                    "itopk_size": itopk,
                    "recall_at_k": float(recall),
                    "k": k,
                    "qps": float(qps),
                    "median_time_ms": float(median_time * 1e3),
                    "mean_time_ms": float(mean_time * 1e3),
                    "std_time_ms": float(std_time * 1e3),
                })

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  VecFlow results saved to {output_file}")
    return results


# =============================================================================
# FAISS HNSW Baseline (standard FAISS, no ACORN -- useful if ACORN build fails)
# =============================================================================

def run_faiss_hnsw_baseline(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    queries: list,
    ground_truth: np.ndarray,
    tag_dir: str,
    output_file: str,
    n_vectors: int,
    max_queries_per_bin: int = 500,
):
    """
    Fallback: Standard FAISS HNSW with post-filtering (brute-force re-ranking).

    If the ACORN fork cannot be built, this provides a reasonable CPU baseline
    using stock FAISS. It uses HNSW search with over-retrieval followed by
    filter-based re-ranking.

    This is weaker than ACORN (no predicate-aware graph traversal) but
    still a valid CPU baseline representing the "HNSW + post-filter" approach.
    """
    print("\n" + "=" * 70)
    print("FAISS HNSW POST-FILTER BASELINE (fallback)")
    print("=" * 70)

    try:
        import faiss
    except ImportError:
        print("ERROR: FAISS not installed.")
        print("  Install: pip install faiss-cpu  (or conda install -c pytorch faiss-cpu)")
        return None

    dim = base_vectors.shape[1]
    n_base = base_vectors.shape[0]
    k = 10

    base_f32 = base_vectors.astype(np.float32)
    query_f32 = query_vectors.astype(np.float32)

    # Build standard HNSW index
    M = 32
    print(f"\nBuilding FAISS HNSW index: dim={dim}, M={M}")
    t0 = time.time()
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 200
    index.add(n_base, base_f32)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")

    # Load tag sets
    tag_sets = {}
    all_tags_needed = set()
    for tag_ids in queries:
        all_tags_needed.update(tag_ids)
    for tid in sorted(all_tags_needed):
        try:
            ids = load_tag_ids(tag_dir, tid)
            tag_sets[tid] = set(ids.tolist())
        except FileNotFoundError:
            tag_sets[tid] = set()

    selectivity_bins = [
        ("0.1%",  0.0,    0.005),
        ("1%",    0.005,  0.05),
        ("10%",   0.05,   0.15),
        ("50%",   0.30,   0.60),
    ]
    query_bins = select_queries_by_selectivity(
        queries, tag_dir, n_vectors, selectivity_bins
    )

    # Over-retrieval factors: retrieve k*factor candidates, then filter
    factors = [10, 50, 100, 500]
    ef_values = [64, 200, 400]

    results = {
        "system": "FAISS_HNSW_PostFilter",
        "version": "stock_faiss",
        "index_params": {"M": M, "efConstruction": 200, "dim": dim},
        "build_time_s": build_time,
        "n_vectors": n_base,
        "k": k,
        "curves": [],
    }

    for bin_label, bin_queries in query_bins.items():
        if not bin_queries:
            continue
        sample = bin_queries[:max_queries_per_bin]
        nq = len(sample)
        avg_sel = np.mean([s for _, _, s in sample])
        print(f"\n  Selectivity '{bin_label}': {nq} queries (avg sel {avg_sel:.3%})")

        batch_queries_f32 = query_f32[[s[0] for s in sample]]

        for ef in ef_values:
            index.hnsw.efSearch = ef
            for factor in factors:
                k_over = k * factor

                t0 = time.perf_counter()
                D, I = index.search(batch_queries_f32, k_over)
                search_time = time.perf_counter() - t0

                # Post-filter
                filtered_results = np.full((nq, k), -1, dtype=np.int64)
                for qi_local, (qi, tag_ids, _) in enumerate(sample):
                    if len(tag_ids) == 1:
                        valid = tag_sets.get(tag_ids[0], set())
                    else:
                        valid = tag_sets.get(tag_ids[0], set())
                        for tid in tag_ids[1:]:
                            valid = valid & tag_sets.get(tid, set())
                    count = 0
                    for j in range(k_over):
                        if I[qi_local, j] in valid:
                            filtered_results[qi_local, count] = I[qi_local, j]
                            count += 1
                            if count >= k:
                                break

                total_time = time.perf_counter() - t0
                gt_subset = ground_truth[[s[0] for s in sample]]
                recall = compute_recall_at_k(
                    filtered_results.astype(np.uint32), gt_subset, k
                )
                qps = nq / total_time

                print(f"    ef={ef}, over={factor}x: recall@{k}={recall:.4f}, QPS={qps:.0f}")
                results["curves"].append({
                    "selectivity_bin": bin_label,
                    "avg_selectivity": float(avg_sel),
                    "n_queries": nq,
                    "efSearch": ef,
                    "over_retrieval_factor": factor,
                    "recall_at_k": float(recall),
                    "k": k,
                    "qps": float(qps),
                })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  HNSW post-filter results saved to {output_file}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ACORN and VecFlow baselines on YFCC-10M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation (both baselines):
    python3 scripts/run_baselines.py \\
        --data-dir /mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M \\
        --yfcc-export-dir bench/yfcc_data

    # ACORN only:
    python3 scripts/run_baselines.py --acorn-only \\
        --data-dir /path/to/yfcc100M --yfcc-export-dir bench/yfcc_data

    # VecFlow only (must be in vecflow conda env):
    conda activate vecflow
    python3 scripts/run_baselines.py --vecflow-only \\
        --data-dir /path/to/yfcc100M --yfcc-export-dir bench/yfcc_data

    # FAISS HNSW fallback (if ACORN build fails):
    python3 scripts/run_baselines.py --faiss-hnsw-only \\
        --data-dir /path/to/yfcc100M --yfcc-export-dir bench/yfcc_data
""",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to big-ann-benchmarks/data/yfcc100M with raw dataset files",
    )
    parser.add_argument(
        "--yfcc-export-dir", default="bench/yfcc_data",
        help="Path to exported YFCC data (from yfcc_export.py)",
    )
    parser.add_argument(
        "--output-dir", default="results/baselines",
        help="Directory for output JSON files",
    )
    parser.add_argument("--acorn-only", action="store_true")
    parser.add_argument("--vecflow-only", action="store_true")
    parser.add_argument("--faiss-hnsw-only", action="store_true",
                        help="Run stock FAISS HNSW post-filter baseline (fallback)")
    parser.add_argument(
        "--max-queries-per-bin", type=int, default=500,
        help="Max queries to evaluate per selectivity bin (default: 500)",
    )
    args = parser.parse_args()

    # Determine which baselines to run
    run_acorn = True
    run_vecflow = True
    run_hnsw = False
    if args.acorn_only:
        run_vecflow = False
    if args.vecflow_only:
        run_acorn = False
    if args.faiss_hnsw_only:
        run_acorn = False
        run_vecflow = False
        run_hnsw = True

    # ---- Load shared data ----
    print("=" * 70)
    print("Loading YFCC-10M dataset")
    print("=" * 70)

    base_path = os.path.join(args.data_dir, "base.10M.u8bin")
    query_path = os.path.join(args.data_dir, "query.public.100K.u8bin")
    meta_path = os.path.join(args.data_dir, "base.metadata.10M.spmat")

    # Check files exist
    for p, desc in [
        (base_path, "base vectors"),
        (query_path, "query vectors"),
        (meta_path, "base metadata"),
    ]:
        if not os.path.exists(p):
            print(f"ERROR: {desc} not found at {p}")
            print(f"  Download YFCC-10M: see PAPER_TODOS.md for instructions")
            sys.exit(1)

    tag_dir = os.path.join(args.yfcc_export_dir, "tags")
    query_file = os.path.join(args.yfcc_export_dir, "queries.bin")
    gt_file = os.path.join(args.yfcc_export_dir, "groundtruth.bin")

    for p, desc in [
        (tag_dir, "tag directory"),
        (query_file, "query metadata"),
        (gt_file, "ground truth"),
    ]:
        if not os.path.exists(p):
            print(f"ERROR: {desc} not found at {p}")
            print(f"  Run: python3 bench/yfcc_export.py --data-dir {args.data_dir}")
            sys.exit(1)

    print(f"Loading base vectors: {base_path}")
    base_vectors = read_u8bin(base_path)
    n_vectors = base_vectors.shape[0]
    print(f"  Shape: {base_vectors.shape}, dtype: {base_vectors.dtype}")

    print(f"Loading query vectors: {query_path}")
    query_vectors = read_u8bin(query_path)
    print(f"  Shape: {query_vectors.shape}")

    print(f"Loading queries: {query_file}")
    queries = load_exported_queries(query_file)
    print(f"  {len(queries)} queries")

    print(f"Loading ground truth: {gt_file}")
    ground_truth = load_exported_groundtruth(gt_file)
    print(f"  Shape: {ground_truth.shape}")

    # ---- Run evaluations ----
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    if run_acorn:
        acorn_out = os.path.join(args.output_dir, "acorn_results.json")
        r = run_acorn_evaluation(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            queries=queries,
            ground_truth=ground_truth,
            base_meta_path=meta_path,
            tag_dir=tag_dir,
            output_file=acorn_out,
            n_vectors=n_vectors,
            max_queries_per_bin=args.max_queries_per_bin,
        )
        if r:
            all_results["acorn"] = r

    if run_vecflow:
        vecflow_out = os.path.join(args.output_dir, "vecflow_results.json")
        r = run_vecflow_evaluation(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            queries=queries,
            ground_truth=ground_truth,
            base_meta_path=meta_path,
            tag_dir=tag_dir,
            output_file=vecflow_out,
            n_vectors=n_vectors,
            max_queries_per_bin=args.max_queries_per_bin,
        )
        if r:
            all_results["vecflow"] = r

    if run_hnsw:
        hnsw_out = os.path.join(args.output_dir, "faiss_hnsw_results.json")
        r = run_faiss_hnsw_baseline(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            queries=queries,
            ground_truth=ground_truth,
            tag_dir=tag_dir,
            output_file=hnsw_out,
            n_vectors=n_vectors,
            max_queries_per_bin=args.max_queries_per_bin,
        )
        if r:
            all_results["faiss_hnsw"] = r

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    for name, r in all_results.items():
        print(f"\n  {r['system']}:")
        print(f"    Build time: {r['build_time_s']:.1f}s")
        print(f"    Curves: {len(r['curves'])} data points")
        if r["curves"]:
            best = max(r["curves"], key=lambda c: c["recall_at_k"])
            print(f"    Best recall@{k}: {best['recall_at_k']:.4f} at QPS={best['qps']:.0f}")
    print(f"\n  Results in: {args.output_dir}/")
    print(f"  Next: python3 scripts/compare_results.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()
