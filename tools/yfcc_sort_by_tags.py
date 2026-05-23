#!/usr/bin/env python3
"""Compute a lex-by-tag-tuple permutation of YFCC-10M, then write permuted
base vectors and per-tag id-list bitmaps so the existing bench harness
(bench_yfcc_search.cu) can run on the sorted layout."""
import os, sys, time
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

BASE_META = '/mnt/c/Users/maxwb/Development/cu-roaring-bitmap/data/yfcc100M/base.metadata.10M.spmat'
BASE_VEC  = '/mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin'
QVEC      = '/mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M/query.public.100K.u8bin'
QUERIES   = '/mnt/c/Users/maxwb/Development/cu-roaring-bitmap/bench/yfcc_data/queries.bin'
TAG_DIR_IN = '/mnt/c/Users/maxwb/Development/cu-roaring-bitmap/bench/yfcc_data/tags'

OUT_DIR    = '/tmp/yfcc_sorted'
OUT_TAGS   = os.path.join(OUT_DIR, 'tags')
OUT_VEC    = os.path.join(OUT_DIR, 'base.10M.u8bin')
PERM_FILE  = os.path.join(OUT_DIR, 'perm.bin')

os.makedirs(OUT_TAGS, exist_ok=True)

def read_sparse_matrix(fname):
    with open(fname, 'rb') as f:
        nrow, ncol, nnz = np.fromfile(f, dtype='int64', count=3)
        indptr  = np.fromfile(f, dtype='int64', count=nrow + 1)
        indices = np.fromfile(f, dtype='int32', count=nnz)
        data    = np.fromfile(f, dtype='float32', count=nnz)
    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))

def read_u8bin(fname):
    with open(fname, 'rb') as f:
        n, d = np.fromfile(f, dtype='int32', count=2)
        a = np.fromfile(f, dtype=np.uint8).reshape(n, d)
    return n, d, a

def write_u8bin(fname, arr):
    with open(fname, 'wb') as f:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.int32).tofile(f)
        arr.tofile(f)

t0 = time.time()
print('loading base metadata ...', flush=True)
M = read_sparse_matrix(BASE_META)
N = M.shape[0]
indptr  = M.indptr
indices = M.indices
print(f'  N={N}, tags={M.shape[1]}, nnz={M.nnz}  ({time.time()-t0:.1f}s)', flush=True)

t1 = time.time()
print('grouping items by tag tuple ...', flush=True)
# Bytes-key trick: cast the row's int32 indices to bytes -> the byte string
# is a hashable lex key (numpy int32 little-endian preserves lex order for
# non-negative values), and we avoid creating 10M Python tuples.
groups = defaultdict(list)
indices_bytes = indices.view(np.uint8).reshape(-1, 4)  # for slicing
for i in range(N):
    s = indptr[i]
    e = indptr[i + 1]
    # take indices[s:e] as raw bytes
    key = indices[s:e].tobytes()
    groups[key].append(i)
print(f'  {len(groups)} unique tag tuples  ({time.time()-t1:.1f}s)', flush=True)

t2 = time.time()
print('sorting unique tuples ...', flush=True)
sorted_keys = sorted(groups.keys())
print(f'  done  ({time.time()-t2:.1f}s)', flush=True)

t3 = time.time()
print('assigning permutation ...', flush=True)
perm_old_to_new = np.empty(N, dtype=np.uint32)  # perm[old] = new
new_id = 0
for key in sorted_keys:
    for old in groups[key]:
        perm_old_to_new[old] = new_id
        new_id += 1
assert new_id == N
perm_old_to_new.tofile(PERM_FILE)
print(f'  perm dumped  ({time.time()-t3:.1f}s)', flush=True)

t4 = time.time()
print('permuting base vectors ...', flush=True)
nb, db, base = read_u8bin(BASE_VEC)
assert nb == N
new_base = np.empty_like(base)
new_base[perm_old_to_new] = base   # row at new_pos = old row
write_u8bin(OUT_VEC, new_base)
del base, new_base
print(f'  wrote {OUT_VEC}  ({time.time()-t4:.1f}s)', flush=True)

t5 = time.time()
print('permuting tag bitmaps ...', flush=True)
tag_files = sorted(f for f in os.listdir(TAG_DIR_IN) if f.endswith('.bin'))
for k, fn in enumerate(tag_files):
    p = os.path.join(TAG_DIR_IN, fn)
    with open(p, 'rb') as f:
        hdr = np.fromfile(f, dtype=np.uint32, count=2)
        ids = np.fromfile(f, dtype=np.uint32, count=hdr[0])
    new_ids = np.sort(perm_old_to_new[ids])
    out = os.path.join(OUT_TAGS, fn)
    with open(out, 'wb') as f:
        hdr.tofile(f)
        new_ids.tofile(f)
    if k and k % 1000 == 0:
        print(f'  {k}/{len(tag_files)}', flush=True)
print(f'  {len(tag_files)} tag bitmaps permuted  ({time.time()-t5:.1f}s)', flush=True)

print(f'TOTAL: {time.time()-t0:.1f}s')
print(f'sorted base vec : {OUT_VEC}')
print(f'sorted tags     : {OUT_TAGS}')
print(f'permutation     : {PERM_FILE}')
