# cu-roaring v2 — minimal, batch-first GPU roaring bitmap

A stripped-down rewrite focused on one use case: **filtered vector search on GPU**.

v2 is **batch-first**. The primary type is `GpuRoaringBatch` — N Roaring bitmaps
packed into a single device allocation. Every API function takes or returns a
batch; `n = 1` is a degenerate case, not the common path. This matches how the
GPU wants to work: launch one kernel that does parallel work across thousands
of (bitmap, container) pairs, not one kernel per bitmap.

## Scope

Six public functions, all batched:

```cpp
namespace cu_roaring::v2 {

// Ingest — N CRoaring bitmaps, single packed device allocation, single H2D.
GpuRoaringBatch upload_batch(const roaring_bitmap_t* const* cpus,
                             uint32_t n,
                             cudaStream_t stream = 0);
void            free_batch(GpuRoaringBatch& batch);

// Explicit format transform on the whole batch.
GpuRoaringBatch promote_batch(const GpuRoaringBatch& batch,
                              cudaStream_t stream = 0);

// Thin per-bitmap accessor; pass by value into kernels that focus on one
// specific bitmap to get the fast 2-read contains() path.
GpuRoaringView  make_view(const GpuRoaringBatch& batch, uint32_t bitmap_idx);

// Device query — overloads for the view (2-read hot path) and the full batch
// (2 extra reads for CSR slice bounds).
__device__ bool contains(const GpuRoaringView& view, uint32_t id);
__device__ bool contains(const GpuRoaringBatch& batch, uint32_t b, uint32_t id);

// AND a subset of bitmaps from the batch. input_indices is host-side.
// Selected bitmaps must be all-bitmap. Returns a 1-bitmap batch.
GpuRoaringBatch multi_and(const GpuRoaringBatch& batch,
                          const uint32_t*        input_indices,
                          uint32_t               n_inputs,
                          cudaStream_t stream = 0);

// Decompress every bitmap in the batch into a contiguous [n_bitmaps, words_each]
// device array. Each bitmap b writes to the b-th slot.
void decompress_batch(const GpuRoaringBatch& batch,
                      uint64_t* device_bitsets,
                      uint64_t  words_each,
                      cudaStream_t stream = 0);

} // namespace cu_roaring::v2
```

## Why batch-first

CUDA throughput is about keeping thousands of blocks in flight. A single
Roaring bitmap with 100 containers gives you 100 blocks — fine for one kernel
launch, but the launch-overhead-to-work ratio is bad once you need to run
anything over N bitmaps sequentially. The batch layout changes that:

- **Upload:** one H2D memcpy for N bitmaps instead of N memcpys.
- **Promote:** one kernel with `total_containers` blocks (tens to hundreds of
  thousands), saturating the GPU regardless of how small individual bitmaps are.
- **Decompress:** one kernel launch writing all N bitsets in parallel.
- **contains():** a kernel that queries different bitmaps per warp (e.g. one
  warp per query vector, each with a different filter) just works — no loop
  over N launches.

For workloads with genuinely one filter per query (a common CAGRA shape),
`n_bitmaps = 1` still goes through the same code path and stays cheap. That's
the degenerate case, not the common one.

## Layout of a GpuRoaringBatch

CSR indexing so each bitmap's data is contiguous:

```
container_starts[b .. b+1]   → slice of containers owned by bitmap b
key_index_starts[b .. b+1]   → slice of key_indices owned by bitmap b

keys[], types[], offsets[],
cardinalities[]              → per-container metadata, concatenated across the
                                batch in bitmap-then-key order

key_indices[]                → concatenated direct-map high-16 → local container
                                index arrays; 0xFFFF sentinels for absent keys

bitmap_data[], array_data[],
run_data[]                   → per-type pools SHARED across the batch;
                                offsets[cid] is a byte offset into the pool
                                selected by types[cid].
```

All device pointers alias into a single `cudaMallocAsync` block. Per-bitmap
scalars (`total_cardinality`, `universe_size`) are mirrored in host-side
arrays so callers can read them without a D2H.

## What's kept from the single-bitmap design

- **All three container types** (ARRAY / BITMAP / RUN).
- **Direct-map key index** for O(1) high-16 → container lookup inside a view.
- **Single packed `cudaMallocAsync`** per batch.
- **No auto-promotion, anywhere.** `promote_batch` is the only function that
  changes container types, and it only runs when explicitly called.

## What's cut

| Removed | Why |
|---|---|
| XOR, ANDNOT, pairwise `set_operation` on mixed types | Filtered search only needs AND. |
| `enumerate_ids` / CSR export | Downstream consumers can `decompress_batch` and scan. |
| `negated` / complement optimisation | Orthogonal, re-add later. |
| Any auto-promotion heuristic | One explicit `promote_batch` function; callers decide. |
| Single-bitmap `upload_from_croaring` path | Call `upload_batch({cpu}, 1)` for the n=1 case. |
| `upload_from_sorted_ids`, `upload_from_bitset` | CRoaring is the only ingest path in v2. |
| 9 of 11 benchmarks | Keep `bench_vs_bitset` (batch contains + batch multi_and). |

## Promotion policy

- `upload_batch` preserves whatever container type CRoaring picked for each
  bitmap. What you give us is what lives on the device.
- `multi_and` refuses non-all-bitmap inputs. Callers run `promote_batch` first.
- `promote_batch` is the only function that mutates container types, and only
  when explicitly called.

## Layout

```
v2/
├── README.md
├── CMakeLists.txt
├── include/cu_roaring_v2/
│   ├── types.cuh           GpuRoaringBatch + GpuRoaringView
│   ├── api.hpp             host-side API signatures
│   └── query.cuh           device-inline contains() for view + batch
├── src/
│   ├── internal.hpp        private: CUDA check macro + alignment helper
│   ├── upload.cpp          N CRoaring bitmaps → GpuRoaringBatch
│   ├── promote.cu          batch ARRAY/RUN → BITMAP transform
│   ├── multi_and.cu        AND a subset of a batch, returning a 1-batch
│   └── decompress.cu       batch → flat uint64 bitsets
├── test/
│   └── test_basic.cu       differential tests on n=4 (common) and n=1 (edge case)
└── bench/
    └── bench_vs_bitset.cu  batch contains + batch multi_and vs packed bitsets
```
