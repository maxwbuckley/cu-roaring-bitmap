#!/usr/bin/env python3
"""
Export YFCC-10M filtered track data for C++ benchmarks.

Reads the Big-ANN benchmark YFCC-10M dataset (10M vectors, 192-dim uint8,
200K-vocab tags) and exports:
  1. Per-tag Roaring-compatible sorted ID lists (binary uint32 arrays)
  2. Query metadata (which tags each query requires)
  3. Base vectors and query vectors in fbin format
  4. Ground truth

Usage:
  python3 bench/yfcc_export.py \
    --data-dir /path/to/big-ann-benchmarks/data/yfcc100M \
    --out-dir bench/yfcc_data
"""

import argparse
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix

def read_u8bin(filename):
    """Read uint8 binary vectors (Big-ANN format)."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        arr = np.fromfile(f, dtype=np.uint8, count=nvecs * dim)
    return arr.reshape(nvecs, dim)

def read_ibin(filename):
    """Read int32 binary vectors."""
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        arr = np.fromfile(f, dtype=np.int32, count=nvecs * dim)
    return arr.reshape(nvecs, dim)

def read_sparse_matrix(fname):
    """Read CSR matrix in spmat format."""
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        indices = np.fromfile(f, dtype='int32', count=nnz)
        data = np.fromfile(f, dtype='float32', count=nnz)
    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Path to big-ann-benchmarks/data/yfcc100M')
    parser.add_argument('--out-dir', default='bench/yfcc_data',
                        help='Output directory for exported data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load metadata
    print("Loading base metadata...")
    base_meta = read_sparse_matrix(
        os.path.join(args.data_dir, 'base.metadata.10M.spmat'))
    n_vectors, n_tags = base_meta.shape
    print(f"  {n_vectors} vectors, {n_tags} tags, {base_meta.nnz} tag assignments")
    print(f"  Avg tags per vector: {base_meta.nnz / n_vectors:.1f}")

    print("Loading query metadata...")
    query_meta = read_sparse_matrix(
        os.path.join(args.data_dir, 'query.metadata.public.100K.spmat'))
    n_queries = query_meta.shape[0]
    print(f"  {n_queries} queries")

    # Analyze tag distribution
    tag_counts = np.array(base_meta.sum(axis=0)).flatten()
    print(f"\nTag distribution:")
    print(f"  Total unique tags with >0 vectors: {np.sum(tag_counts > 0)}")
    print(f"  Min/median/max tag frequency: {tag_counts[tag_counts > 0].min()} / "
          f"{np.median(tag_counts[tag_counts > 0]):.0f} / {tag_counts.max()}")

    # Find tags used in queries
    query_tags = set()
    for q in range(n_queries):
        row = query_meta.getrow(q)
        query_tags.update(row.indices.tolist())
    print(f"  Tags used in queries: {len(query_tags)}")

    # Export per-tag ID lists for the tags used in queries
    print(f"\nExporting per-tag ID lists for {len(query_tags)} query-relevant tags...")
    tag_dir = os.path.join(args.out_dir, 'tags')
    os.makedirs(tag_dir, exist_ok=True)

    # Transpose to get tag → vector mapping
    tag_to_vectors = base_meta.T.tocsr()  # shape: (n_tags, n_vectors)

    tag_stats = []
    for tag_id in sorted(query_tags):
        row = tag_to_vectors.getrow(tag_id)
        ids = np.sort(row.indices).astype(np.uint32)
        density = len(ids) / n_vectors * 100

        # Write as raw uint32 array with header (n_ids, tag_id)
        out_path = os.path.join(tag_dir, f'tag_{tag_id}.bin')
        with open(out_path, 'wb') as f:
            np.array([len(ids), tag_id], dtype=np.uint32).tofile(f)
            ids.tofile(f)

        tag_stats.append((tag_id, len(ids), density))

    # Sort by frequency for reporting
    tag_stats.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top 10 most common query tags:")
    for tag_id, count, density in tag_stats[:10]:
        print(f"    tag {tag_id}: {count} vectors ({density:.2f}%)")
    print(f"  Bottom 10 rarest query tags:")
    for tag_id, count, density in tag_stats[-10:]:
        print(f"    tag {tag_id}: {count} vectors ({density:.4f}%)")

    # Export query tag requirements
    print(f"\nExporting query metadata...")
    query_file = os.path.join(args.out_dir, 'queries.bin')
    with open(query_file, 'wb') as f:
        np.array([n_queries], dtype=np.uint32).tofile(f)
        for q in range(n_queries):
            row = query_meta.getrow(q)
            tags = row.indices.astype(np.uint32)
            np.array([len(tags)], dtype=np.uint32).tofile(f)
            tags.tofile(f)

    # Analyze query predicate counts
    preds_per_query = np.array([query_meta.getrow(q).nnz for q in range(n_queries)])
    print(f"  Predicates per query: min={preds_per_query.min()} "
          f"max={preds_per_query.max()} mean={preds_per_query.mean():.1f}")
    for np_val in sorted(set(preds_per_query)):
        count = np.sum(preds_per_query == np_val)
        print(f"    {np_val} predicates: {count} queries ({100*count/n_queries:.1f}%)")

    # Export ground truth
    print(f"\nExporting ground truth...")
    gt = read_ibin(os.path.join(args.data_dir, 'GT.public.ibin'))
    gt_file = os.path.join(args.out_dir, 'groundtruth.bin')
    with open(gt_file, 'wb') as f:
        np.array([gt.shape[0], gt.shape[1]], dtype=np.uint32).tofile(f)
        gt.astype(np.uint32).tofile(f)
    print(f"  {gt.shape[0]} queries × {gt.shape[1]} neighbors")

    # Summary
    print(f"\nExport complete:")
    print(f"  {args.out_dir}/tags/tag_*.bin  — {len(query_tags)} per-tag ID lists")
    print(f"  {args.out_dir}/queries.bin      — {n_queries} query tag requirements")
    print(f"  {args.out_dir}/groundtruth.bin  — {gt.shape[0]}×{gt.shape[1]} ground truth")
    print(f"\n  Universe size: {n_vectors}")
    print(f"  Vector dim: 192 (uint8)")
    print(f"  Unique query tags: {len(query_tags)}")

    # Write summary JSON
    import json
    summary = {
        'n_vectors': int(n_vectors),
        'n_tags': int(n_tags),
        'n_queries': int(n_queries),
        'dim': 192,
        'dtype': 'uint8',
        'n_query_tags': len(query_tags),
        'tag_stats': [{'tag_id': int(t), 'count': int(c), 'density_pct': round(d, 4)}
                      for t, c, d in tag_stats],
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
