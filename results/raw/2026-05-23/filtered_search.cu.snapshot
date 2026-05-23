/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Roaring-filtered brute-force vector search — implementation.
 * See include/cu_roaring/detail/filtered_search.cuh and
 * analysis/filter_driven_search.md.
 */

#include "cu_roaring/detail/filtered_search.cuh"
#include "cu_roaring/detail/enumerate_runs.cuh"
#include "cu_roaring/detail/decompress.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>

namespace cu_roaring {

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t st = (call);                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(std::string("cuBLAS error at ") +       \
                                     __FILE__ + ":" +                        \
                                     std::to_string(__LINE__) + " code " +   \
                                     std::to_string(static_cast<int>(st)));  \
        }                                                                    \
    } while (0)

namespace {

constexpr uint32_t kTileWMax         = 1u << 20; // GEMM column tile cap
constexpr uint32_t kTileWMin         = 1u << 15; // GEMM column tile floor
constexpr uint64_t kTileBudget       = 64u << 20;// score-tile budget in floats
constexpr uint32_t kBlockTK          = 256;    // threads per top-k block (8 warps)
constexpr uint32_t kColBlkMax        = 64;     // max column blocks per query, top-k pass 1
constexpr float    kBitmapDenseThr   = 0.5f;   // bitmap density gate
constexpr uint32_t kDirectMinWidth   = 1u << 16; // shape gate: min width for a direct per-range GEMM
constexpr uint32_t kBlock1D          = 256;

// 1-D grid size for `n` elements at kBlock1D threads/block.
inline uint32_t grid1d(uint64_t n)
{
    return static_cast<uint32_t>((n + kBlock1D - 1) / kBlock1D);
}

// ---------------------------------------------------------------------------
// Top-k device helpers — arrays are kept sorted descending by score.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void topk_insert(float* s, uint32_t* id, uint32_t k,
                                             float ns, uint32_t nid)
{
    if (ns <= s[k - 1]) return;
    int p = static_cast<int>(k) - 1;
    while (p > 0 && s[p - 1] < ns) {
        s[p] = s[p - 1];
        id[p] = id[p - 1];
        --p;
    }
    s[p]  = ns;
    id[p] = nid;
}

// as/ai := top-k of (as/ai) U (bs/bi); all inputs sorted descending.
__device__ __forceinline__ void topk_merge(float* as, uint32_t* ai,
                                           const float* bs, const uint32_t* bi,
                                           uint32_t k)
{
    float    ts[kMaxK];
    uint32_t ti[kMaxK];
    int ia = 0, ib = 0;
    for (uint32_t o = 0; o < k; ++o) {
        if (as[ia] >= bs[ib]) { ts[o] = as[ia]; ti[o] = ai[ia]; ++ia; }
        else                  { ts[o] = bs[ib]; ti[o] = bi[ib]; ++ib; }
    }
    for (uint32_t o = 0; o < k; ++o) { as[o] = ts[o]; ai[o] = ti[o]; }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void init_topk_kernel(float* score, uint32_t* id, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { score[i] = -INFINITY; id[i] = 0u; }
}

// Set masked-out columns of a score tile to -inf. Column c is eligible iff
// bit (mask_bit0 + c) of `mask` is set, XOR `invert`.
__global__ void apply_mask_kernel(float* tile, uint32_t q, uint32_t w,
                                  const uint32_t* mask, uint32_t mask_bit0,
                                  uint8_t invert)
{
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<uint64_t>(q) * w) return;
    uint32_t col = static_cast<uint32_t>(idx % w);
    uint32_t bit = mask_bit0 + col;
    uint32_t e   = (mask[bit >> 5] >> (bit & 31u)) & 1u;
    if (invert) e ^= 1u;
    if (!e) tile[idx] = -INFINITY;
}

// XOR-reduce a per-lane sorted-descending top-k across a warp; on return
// every lane holds the warp's combined top-k. ~5 * k shuffles, no shared mem.
__device__ __forceinline__ void warp_reduce_topk(float* s, uint32_t* id,
                                                 uint32_t k)
{
    for (int stride = 16; stride >= 1; stride >>= 1) {
        float    os[kMaxK];
        uint32_t oid[kMaxK];
        for (uint32_t i = 0; i < k; ++i) {
            os[i]  = __shfl_xor_sync(0xFFFFFFFFu, s[i],  stride);
            oid[i] = __shfl_xor_sync(0xFFFFFFFFu, id[i], stride);
        }
        topk_merge(s, id, os, oid, k);
    }
}

// Combine a block's per-thread top-k into out[0..k): warp-shuffle reduction
// then a tiny 8-list merge by thread 0. Block is fixed at kBlockTK threads
// (8 warps); shared use is only 8*kMaxK entries.
__device__ __forceinline__ void block_topk(float* ps, uint32_t* pid, uint32_t k,
                                           float* out_s, uint32_t* out_id)
{
    constexpr uint32_t kNWarp = kBlockTK / 32u;
    __shared__ float    sh_s[kNWarp * kMaxK];
    __shared__ uint32_t sh_i[kNWarp * kMaxK];

    warp_reduce_topk(ps, pid, k);
    uint32_t warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0u) {
        for (uint32_t i = 0; i < k; ++i) {
            sh_s[warp * kMaxK + i] = ps[i];
            sh_i[warp * kMaxK + i] = pid[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float    rs[kMaxK];
        uint32_t ri[kMaxK];
        for (uint32_t i = 0; i < k; ++i) { rs[i] = sh_s[i]; ri[i] = sh_i[i]; }
        for (uint32_t w = 1; w < kNWarp; ++w)
            topk_merge(rs, ri, &sh_s[w * kMaxK], &sh_i[w * kMaxK], k);
        for (uint32_t i = 0; i < k; ++i) { out_s[i] = rs[i]; out_id[i] = ri[i]; }
    }
}

// Pass 1: column-parallel partial top-k. grid = (n_colblk, q); each block
// scans one column slice of the score tile and writes one k-list to
// partials[(q*n_colblk + cb)*kMaxK ..].
__global__ void partial_topk_kernel(const float* tile, uint32_t w, uint32_t k,
                                    uint32_t n_colblk, uint32_t id_base,
                                    const uint32_t* id_map, float* part_s,
                                    uint32_t* part_i)
{
    uint32_t cb = blockIdx.x;
    uint32_t q  = blockIdx.y;
    uint32_t chunk = (w + n_colblk - 1u) / n_colblk;
    uint32_t c0 = cb * chunk;
    uint32_t c1 = c0 + chunk < w ? c0 + chunk : w;

    float    ps[kMaxK];
    uint32_t pid[kMaxK];
    for (uint32_t i = 0; i < k; ++i) { ps[i] = -INFINITY; pid[i] = 0u; }

    const float* row = tile + static_cast<uint64_t>(q) * w;
    for (uint32_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        uint32_t id = id_map ? id_map[col] : (id_base + col);
        topk_insert(ps, pid, k, row[col], id);
    }
    uint32_t base = (q * n_colblk + cb) * kMaxK;
    block_topk(ps, pid, k, part_s + base, part_i + base);
}

// Pass 2: running[q] := top-k of running[q] U partials[q]. grid = q.
__global__ void reduce_into_running_kernel(float* run_score, uint32_t* run_id,
                                           const float* part_s,
                                           const uint32_t* part_i,
                                           uint32_t n_colblk, uint32_t k)
{
    uint32_t q   = blockIdx.x;
    uint32_t tid = threadIdx.x;

    float    ps[kMaxK];
    uint32_t pid[kMaxK];
    for (uint32_t i = 0; i < k; ++i) { ps[i] = -INFINITY; pid[i] = 0u; }

    for (uint32_t j = tid; j < n_colblk; j += blockDim.x) {
        uint32_t base = (q * n_colblk + j) * kMaxK;
        for (uint32_t i = 0; i < k; ++i)
            topk_insert(ps, pid, k, part_s[base + i], part_i[base + i]);
    }
    if (tid < k) {
        topk_insert(ps, pid, k, run_score[q * k + tid], run_id[q * k + tid]);
    }
    __shared__ float    out_s[kMaxK];
    __shared__ uint32_t out_id[kMaxK];
    block_topk(ps, pid, k, out_s, out_id);
    if (tid == 0) {
        for (uint32_t i = 0; i < k; ++i) {
            run_score[q * k + i] = out_s[i];
            run_id[q * k + i]    = out_id[i];
        }
    }
}

template <typename T>
__global__ void gather_rows_kernel(T* dst, const T* db,
                                   const uint32_t* ids, uint32_t n, uint32_t dim)
{
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<uint64_t>(n) * dim) return;
    uint32_t row = static_cast<uint32_t>(idx / dim);
    uint32_t d   = static_cast<uint32_t>(idx % dim);
    dst[idx] = db[static_cast<uint64_t>(ids[row]) * dim + d];
}

// Descriptor for one scattered-id source container (ARRAY or sparse BITMAP).
struct GatherSrc {
    uint32_t type;       // 0 = ARRAY, 1 = BITMAP
    uint32_t key;        // high-16 key
    uint32_t data_off;   // ARRAY: uint16 index into array_data; BITMAP: uint32 index into bitmap_data
    uint32_t card;       // element count
    uint32_t out_off;    // base offset into gather_ids
};

// Emit global ids for each gather source. One block per source.
__global__ void emit_gather_ids_kernel(const GatherSrc* srcs, uint32_t n_src,
                                       const uint16_t* array_data,
                                       const uint32_t* bitmap_data_u32,
                                       uint32_t* out_ids)
{
    uint32_t s = blockIdx.x;
    if (s >= n_src) return;
    GatherSrc src = srcs[s];
    uint32_t base = src.key << 16;

    if (src.type == 0u) {  // ARRAY: ids are already a compact list
        const uint16_t* arr = array_data + src.data_off;
        for (uint32_t j = threadIdx.x; j < src.card; j += blockDim.x) {
            out_ids[src.out_off + j] = base | static_cast<uint32_t>(arr[j]);
        }
    } else {               // BITMAP: bit-scan with atomic compaction
        __shared__ uint32_t bcount;
        if (threadIdx.x == 0) bcount = 0u;
        __syncthreads();
        const uint32_t* words = bitmap_data_u32 + src.data_off;
        for (uint32_t w = threadIdx.x; w < 2048u; w += blockDim.x) {
            uint32_t x = words[w];
            while (x) {
                uint32_t b   = static_cast<uint32_t>(__ffs(static_cast<int>(x)) - 1);
                x &= x - 1u;
                uint32_t pos = atomicAdd(&bcount, 1u);
                out_ids[src.out_off + pos] = base | (w * 32u + b);
            }
        }
    }
}

// Expand half-open ranges into a flat id list (fragmentation fallback).
__global__ void expand_ranges_kernel(const IdRange* ranges,
                                     const uint64_t* out_off, uint32_t n_ranges,
                                     uint32_t* out_ids)
{
    uint32_t r = blockIdx.x;
    if (r >= n_ranges) return;
    uint32_t lo = ranges[r].start;
    uint32_t hi = ranges[r].end;
    uint64_t base = out_off[r];
    for (uint32_t v = lo + threadIdx.x; v < hi; v += blockDim.x) {
        out_ids[base + (v - lo)] = v;
    }
}

// Build a 2048-word eligibility bitset for one excluded container (negated
// path). One block per masked container. Correctness-first, not tuned.
__global__ void build_container_mask_kernel(uint32_t* mask_pool,
                                            uint32_t type, uint32_t card,
                                            uint32_t data_off_u16,
                                            uint32_t data_off_u32,
                                            const uint16_t* array_data,
                                            const uint32_t* bitmap_data_u32,
                                            const uint16_t* run_data)
{
    uint32_t* mask = mask_pool;  // [2048]
    if (type == 1u) {            // BITMAP: direct copy
        const uint32_t* words = bitmap_data_u32 + data_off_u32;
        for (uint32_t w = threadIdx.x; w < 2048u; w += blockDim.x) mask[w] = words[w];
        return;
    }
    for (uint32_t w = threadIdx.x; w < 2048u; w += blockDim.x) mask[w] = 0u;
    __syncthreads();
    if (type == 0u) {            // ARRAY: scatter
        const uint16_t* arr = array_data + data_off_u16;
        for (uint32_t j = threadIdx.x; j < card; j += blockDim.x) {
            uint32_t v = arr[j];
            atomicOr(&mask[v >> 5], 1u << (v & 31u));
        }
    } else {                     // RUN: expand (card = run count)
        const uint16_t* runs = run_data + data_off_u16;
        for (uint32_t r = threadIdx.x; r < card; r += blockDim.x) {
            uint32_t start = runs[r * 2];
            uint32_t end   = start + runs[r * 2 + 1];
            if (end > 0xFFFFu) end = 0xFFFFu;
            for (uint32_t v = start; v <= end; ++v) {
                atomicOr(&mask[v >> 5], 1u << (v & 31u));
            }
        }
    }
}

// Identity of one execute() invocation — used to cache the CUDA graph so
// repeated searches replay instead of re-issuing every kernel launch.
struct ExecSig {
    const void*     db   = nullptr;
    const void*     gdb  = nullptr;
    const uint32_t* gids = nullptr;
    uint32_t*       oid  = nullptr;
    float*          osc  = nullptr;
    uint32_t        q = 0, k = 0, dim = 0, tile_w = 0;
    uint64_t        taskhash = 0;
};
bool sig_eq(const ExecSig& a, const ExecSig& b)
{
    return a.db == b.db && a.gdb == b.gdb && a.gids == b.gids &&
           a.oid == b.oid && a.osc == b.osc && a.q == b.q && a.k == b.k &&
           a.dim == b.dim && a.tile_w == b.tile_w && a.taskhash == b.taskhash;
}
uint64_t hash_tasks(const std::vector<GemmTask>& t)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(t.size());
    for (const GemmTask& x : t) {
        mix(x.kind); mix(x.start); mix(x.n_cols);
        mix(reinterpret_cast<uintptr_t>(x.mask));
        mix(x.mask_bit0); mix(x.mask_invert);
    }
    return h;
}

// ---------------------------------------------------------------------------
// Shared executor: run a list of GEMM tasks through GEMM -> mask -> top-k.
//
// The task loop is captured into a CUDA graph on the second call with a given
// signature and replayed thereafter — a filtered search issues O(tasks)
// kernels, so per-launch host overhead would otherwise dominate a small
// schedule. The first call runs eagerly (also warms cuBLAS before capture).
// ---------------------------------------------------------------------------
template <typename T>
void execute_impl(cublasHandle_t handle, const T* d_queries, uint32_t q,
                  const T* d_db, uint32_t dim,
                  const std::vector<GemmTask>& tasks, const T* d_gather_db,
                  const uint32_t* d_gather_ids, uint32_t k,
                  uint32_t* d_out_ids, float* d_out_scores, cudaStream_t stream)
{
    static_assert(std::is_same<T, float>::value || std::is_same<T, __half>::value,
                  "filtered_search supports float (fp32) and __half (fp16) only");
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Adaptive GEMM column tile: as wide as the score-tile budget allows, so
    // a big database is covered in a handful of tiles, not hundreds.
    uint32_t tile_w = static_cast<uint32_t>(
        std::min<uint64_t>(kTileWMax, std::max<uint64_t>(kTileWMin,
                                                         kTileBudget / q)));

    // Persistent grow-only scratch (freed at process exit) — repeated searches
    // must not pay a device allocation per call, and a graph capture must not
    // contain one.
    static float*    s_tile     = nullptr;
    static float*    s_part_s   = nullptr;
    static uint32_t* s_part_i   = nullptr;
    static size_t    s_tile_cap = 0;
    static size_t    s_part_cap = 0;
    size_t need_tile = static_cast<size_t>(q) * tile_w;
    size_t need_part = static_cast<size_t>(q) * kColBlkMax * kMaxK;
    if (need_tile > s_tile_cap) {
        if (s_tile) CUDA_CHECK(cudaFree(s_tile));
        CUDA_CHECK(cudaMalloc(&s_tile, need_tile * sizeof(float)));
        s_tile_cap = need_tile;
    }
    if (need_part > s_part_cap) {
        if (s_part_s) CUDA_CHECK(cudaFree(s_part_s));
        if (s_part_i) CUDA_CHECK(cudaFree(s_part_i));
        CUDA_CHECK(cudaMalloc(&s_part_s, need_part * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s_part_i, need_part * sizeof(uint32_t)));
        s_part_cap = need_part;
    }
    float*    d_tile   = s_tile;
    float*    d_part_s = s_part_s;
    uint32_t* d_part_i = s_part_i;

    const float alpha = 1.0f, beta = 0.0f;

    // The full launch sequence of one search.
    auto record = [&]() {
        init_topk_kernel<<<div_ceil(q * k, kBlock1D), kBlock1D, 0, stream>>>(
            d_out_scores, d_out_ids, q * k);
        for (const GemmTask& task : tasks) {
            const T* mat_base =
                (task.kind == GemmTask::kGather) ? d_gather_db : d_db;
            for (uint32_t off = 0; off < task.n_cols; off += tile_w) {
                uint32_t w = std::min(tile_w, task.n_cols - off);
                const T* mat = mat_base +
                    static_cast<uint64_t>(task.start + off) * dim;

                // tile (row-major q x w) = queries (q x dim) . mat^T (dim x w)
                if constexpr (std::is_same<T, float>::value) {
                    CUBLAS_CHECK(cublasSgemm(
                        handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w), static_cast<int>(q),
                        static_cast<int>(dim), &alpha, mat,
                        static_cast<int>(dim), d_queries,
                        static_cast<int>(dim), &beta, d_tile,
                        static_cast<int>(w)));
                } else {
                    // fp16 inputs, fp32 accumulation/output via TensorCores.
                    // alpha/beta are fp32 to match the compute type.
                    CUBLAS_CHECK(cublasGemmEx(
                        handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w), static_cast<int>(q),
                        static_cast<int>(dim),
                        &alpha,
                        mat,        CUDA_R_16F, static_cast<int>(dim),
                        d_queries,  CUDA_R_16F, static_cast<int>(dim),
                        &beta,
                        d_tile,     CUDA_R_32F, static_cast<int>(w),
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
                }

                if (task.kind == GemmTask::kRange && task.mask) {
                    uint64_t n = static_cast<uint64_t>(q) * w;
                    apply_mask_kernel<<<grid1d(n), kBlock1D, 0, stream>>>(
                        d_tile, q, w, task.mask, task.mask_bit0 + off,
                        task.mask_invert);
                }

                const uint32_t* id_map =
                    (task.kind == GemmTask::kGather)
                        ? (d_gather_ids + task.start + off)
                        : nullptr;
                uint32_t id_base = task.start + off;

                // Column-parallel two-pass top-k.
                uint32_t n_colblk = w / 2048u;
                if (n_colblk < 1u) n_colblk = 1u;
                if (n_colblk > kColBlkMax) n_colblk = kColBlkMax;
                partial_topk_kernel<<<dim3(n_colblk, q), kBlockTK, 0, stream>>>(
                    d_tile, w, k, n_colblk, id_base, id_map, d_part_s, d_part_i);
                reduce_into_running_kernel<<<q, kBlockTK, 0, stream>>>(
                    d_out_scores, d_out_ids, d_part_s, d_part_i, n_colblk, k);
            }
        }
    };

    // The default (NULL) stream cannot be stream-captured; without a real
    // stream, fall back to eager launches.
    if (stream == 0) {
        record();
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ExecSig sig{d_db, d_gather_db, d_gather_ids, d_out_ids, d_out_scores,
                q, k, dim, tile_w, hash_tasks(tasks)};
    static ExecSig         g_sig;
    static cudaGraphExec_t g_exec = nullptr;
    static bool            g_have = false;

    if (g_have && sig_eq(sig, g_sig)) {
        if (g_exec) {                       // captured already: replay
            CUDA_CHECK(cudaGraphLaunch(g_exec, stream));
            return;
        }
        // Second call with this signature: capture, then replay.
        CUDA_CHECK(cudaStreamBeginCapture(stream,
                                          cudaStreamCaptureModeThreadLocal));
        record();
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&g_exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphLaunch(g_exec, stream));
        return;
    }

    // New signature: drop any stale graph, run eagerly once.
    if (g_exec) { CUDA_CHECK(cudaGraphExecDestroy(g_exec)); g_exec = nullptr; }
    g_sig  = sig;
    g_have = true;
    record();
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// ---------------------------------------------------------------------------
// Schedule construction
// ---------------------------------------------------------------------------
// Process-lifetime grow-only scratch for build_schedule. A filter built on the
// hot path of a per-query workload (e.g. YFCC) hits build_schedule once per
// query; using these scratch buffers means each call pays at most a no-op
// growth check instead of three to five cudaMallocs + Frees through the async
// allocator. Buffers grow geometrically; total scratch peaks at the worst-case
// (largest filter) seen so far.
namespace {
struct BuildScratch {
    uint32_t* gather_ids = nullptr;  size_t gather_ids_cap = 0;
    void*     gather_db  = nullptr;  size_t gather_db_cap  = 0;  // bytes
    uint32_t* mask_pool  = nullptr;  size_t mask_pool_cap  = 0;  // bytes
    void*     d_srcs     = nullptr;  size_t d_srcs_cap     = 0;  // bytes
    void*     d_ranges   = nullptr;  size_t d_ranges_cap   = 0;  // bytes
    void*     d_roff     = nullptr;  size_t d_roff_cap     = 0;  // bytes

    template <typename T>
    void grow(T*& ptr, size_t& cap, size_t need_bytes)
    {
        if (need_bytes <= cap) return;
        if (ptr) { CUDA_CHECK(cudaFree(ptr)); }
        // Round up to next power of two for amortisation.
        size_t new_cap = cap > 0 ? cap : 1;
        while (new_cap < need_bytes) new_cap *= 2;
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, new_cap));
        ptr = static_cast<T*>(p);
        cap = new_cap;
    }
};
inline BuildScratch& build_scratch()
{
    static BuildScratch s;
    return s;
}
}  // namespace

namespace {

// Templated body of build_schedule — all schedule construction is
// dtype-independent; only the final gather_rows_kernel call differs between
// fp32 and fp16. Exposed via two thin overloads below.
template <typename T>
SearchSchedule build_schedule_impl(const GpuRoaring& filter, uint32_t n_rows,
                                   const T* d_db, uint32_t dim,
                                   cudaStream_t stream)
{
    SearchSchedule sched;
    sched.scratched = true;
    BuildScratch& sc = build_scratch();
    uint32_t nc = filter.n_containers;

    // Pull the container index to host (one-time, a few KB).
    std::vector<uint16_t>      keys(nc);
    std::vector<ContainerType> types(nc);
    std::vector<uint16_t>      cards(nc);
    std::vector<uint32_t>      offs(nc);
    if (nc > 0) {
        CUDA_CHECK(cudaMemcpyAsync(keys.data(), filter.keys,
                                   nc * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(types.data(), filter.types,
                                   nc * sizeof(ContainerType), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(cards.data(), filter.cardinalities,
                                   nc * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(offs.data(), filter.offsets,
                                   nc * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<IdRange>   ranges;       // range tasks (no mask)
    std::vector<GemmTask>  masked;       // near-full bitmap range tasks
    std::vector<GatherSrc> gsrc;         // scattered-id sources

    auto clamp_end = [&](uint32_t e) { return e < n_rows ? e : n_rows; };

    if (!filter.negated) {
        // RUN containers -> coalesced contiguous ranges. Skip the whole
        // enumerate_runs pipeline (4 kernels + 2 D2H scalar syncs, ~150-300us
        // of host overhead on WSL2) when the filter has no RUN containers --
        // e.g. anything built via upload_from_device_bitset or upload_from_ids.
        RunRanges rr{nullptr, 0};
        if (filter.n_run_containers > 0) {
            rr = enumerate_runs(filter, stream);
            std::vector<IdRange> hr(rr.count);
            if (rr.count > 0) {
                CUDA_CHECK(cudaMemcpy(hr.data(), rr.ranges,
                                      rr.count * sizeof(IdRange),
                                      cudaMemcpyDeviceToHost));
                for (auto& r : hr) {
                    if (r.start < n_rows) ranges.push_back({r.start, clamp_end(r.end)});
                }
            }
        }
        free_run_ranges(rr);

        // ARRAY and BITMAP containers.
        for (uint32_t c = 0; c < nc; ++c) {
            uint32_t base = static_cast<uint32_t>(keys[c]) << 16;
            if (base >= n_rows) continue;
            if (types[c] == ContainerType::ARRAY) {
                gsrc.push_back({0u, keys[c], offs[c] / 2u, cards[c], 0u});
            } else if (types[c] == ContainerType::BITMAP) {
                // card wraps to 0 for a full (65536) container -> treat as full.
                uint32_t card = cards[c] == 0u ? 65536u : cards[c];
                float density = static_cast<float>(card) / 65536.0f;
                if (density >= kBitmapDenseThr) {
                    GemmTask t;
                    t.kind   = GemmTask::kRange;
                    t.start  = base;
                    t.n_cols = clamp_end(base + 65536u) - base;
                    t.mask   = reinterpret_cast<const uint32_t*>(filter.bitmap_data)
                               + offs[c] / 4u;
                    t.mask_bit0   = 0u;
                    t.mask_invert = 0u;
                    masked.push_back(t);
                } else {
                    gsrc.push_back({1u, keys[c], offs[c] / 4u, card, 0u});
                }
            }
        }
    } else {
        // negated: stored containers are the EXCLUDED set; eligible = complement.
        uint32_t n_blocks = div_ceil(n_rows, 65536u);
        std::vector<int> block2cont(n_blocks, -1);
        for (uint32_t c = 0; c < nc; ++c)
            if (keys[c] < n_blocks) block2cont[keys[c]] = static_cast<int>(c);

        // Count excluded containers that need a per-block mask (skip RUN
        // containers that cover their whole block — fully excluded).
        std::vector<int> need_mask;  // container indices
        for (uint32_t b = 0; b < n_blocks; ++b) {
            int c = block2cont[b];
            if (c < 0) continue;
            bool full_block = false;
            if (types[c] == ContainerType::RUN && cards[c] == 1u) {
                uint16_t two[2];
                CUDA_CHECK(cudaMemcpy(two, filter.run_data + offs[c] / 2u,
                                      2 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                full_block = (two[0] == 0u && two[1] >= 0xFFFFu);
            } else if (types[c] == ContainerType::BITMAP && cards[c] == 0u) {
                full_block = true;  // wrapped 65536 -> all excluded
            }
            if (!full_block) need_mask.push_back(c);
        }

        // Build per-block exclusion masks (scratch).
        uint32_t* mask_pool = nullptr;
        if (!need_mask.empty()) {
            sc.grow(sc.mask_pool, sc.mask_pool_cap,
                    need_mask.size() * 2048u * sizeof(uint32_t));
            mask_pool = sc.mask_pool;
        }
        sched.mask_pool = mask_pool;
        std::vector<const uint32_t*> mask_ptr(nc, nullptr);
        for (size_t m = 0; m < need_mask.size(); ++m) {
            int c = need_mask[m];
            uint32_t* dst = mask_pool + m * 2048u;
            uint32_t type = (types[c] == ContainerType::ARRAY)  ? 0u
                          : (types[c] == ContainerType::BITMAP) ? 1u : 2u;
            build_container_mask_kernel<<<1, kBlock1D, 0, stream>>>(
                dst, type, cards[c],
                offs[c] / 2u, offs[c] / 4u,
                filter.array_data,
                reinterpret_cast<const uint32_t*>(filter.bitmap_data),
                filter.run_data);
            CUDA_CHECK(cudaGetLastError());
            mask_ptr[c] = dst;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Walk blocks: absent -> eligible range; present -> masked range.
        uint32_t run_lo = 0;
        bool in_run = false;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            int c = block2cont[b];
            bool eligible_block = (c < 0);
            bool full_excluded =
                (c >= 0 && mask_ptr[c] == nullptr);  // skipped == fully excluded
            if (eligible_block) {
                if (!in_run) { run_lo = b * 65536u; in_run = true; }
            } else {
                if (in_run) {
                    ranges.push_back({run_lo, clamp_end(b * 65536u)});
                    in_run = false;
                }
                if (!full_excluded) {
                    uint32_t base = b * 65536u;
                    GemmTask t;
                    t.kind   = GemmTask::kRange;
                    t.start  = base;
                    t.n_cols = clamp_end(base + 65536u) - base;
                    t.mask   = mask_ptr[c];
                    t.mask_bit0   = 0u;
                    t.mask_invert = 1u;  // eligible where the exclusion bit is clear
                    masked.push_back(t);
                }
            }
        }
        if (in_run) ranges.push_back({run_lo, n_rows});
    }

    // -- Shape gate ----------------------------------------------------------
    // A range keeps its own direct (no-copy) GEMM only if it is wide enough to
    // run one efficiently; narrower ranges issue a small skinny GEMM each, so
    // they are gathered into the single compact GEMM instead. The decision is
    // per range — a schedule may mix a few huge direct ranges with a gather of
    // all the rest.
    std::vector<IdRange> direct, narrow;
    for (const IdRange& r : ranges) {
        if ((r.end - r.start) >= kDirectMinWidth) direct.push_back(r);
        else                                     narrow.push_back(r);
    }
    uint64_t narrow_cols = 0;
    for (const IdRange& r : narrow) narrow_cols += (r.end - r.start);
    ranges = direct;  // only wide ranges remain as direct range tasks
    sched.used_fallback = !narrow.empty();

    // -- Collect gather ids: scattered sources + narrow ranges ---------------
    uint64_t n_gather = 0;
    for (auto& s : gsrc) n_gather += s.card;
    uint64_t fallback_off = n_gather;
    n_gather += narrow_cols;

    if (n_gather > 0) {
        sc.grow(sc.gather_ids, sc.gather_ids_cap, n_gather * sizeof(uint32_t));
        sched.gather_ids = sc.gather_ids;
        sched.n_gather   = static_cast<uint32_t>(n_gather);

        // Scattered-id sources.
        if (!gsrc.empty()) {
            uint32_t acc = 0;
            for (auto& s : gsrc) { s.out_off = acc; acc += s.card; }
            sc.grow(sc.d_srcs, sc.d_srcs_cap, gsrc.size() * sizeof(GatherSrc));
            GatherSrc* d_srcs = static_cast<GatherSrc*>(sc.d_srcs);
            CUDA_CHECK(cudaMemcpyAsync(d_srcs, gsrc.data(),
                                       gsrc.size() * sizeof(GatherSrc),
                                       cudaMemcpyHostToDevice, stream));
            emit_gather_ids_kernel<<<static_cast<uint32_t>(gsrc.size()),
                                     kBlock1D, 0, stream>>>(
                d_srcs, static_cast<uint32_t>(gsrc.size()), filter.array_data,
                reinterpret_cast<const uint32_t*>(filter.bitmap_data),
                sched.gather_ids);
            CUDA_CHECK(cudaGetLastError());
        }

        // Expand the narrow ranges into the gather id list.
        if (!narrow.empty()) {
            std::vector<uint64_t> roff;
            roff.reserve(narrow.size() + 1);
            uint64_t acc = fallback_off;
            for (const IdRange& r : narrow) {
                roff.push_back(acc);
                acc += (r.end - r.start);
            }
            roff.push_back(acc);
            sc.grow(sc.d_ranges, sc.d_ranges_cap, narrow.size() * sizeof(IdRange));
            sc.grow(sc.d_roff,   sc.d_roff_cap,   roff.size()   * sizeof(uint64_t));
            IdRange*  d_ranges = static_cast<IdRange*>(sc.d_ranges);
            uint64_t* d_roff   = static_cast<uint64_t*>(sc.d_roff);
            CUDA_CHECK(cudaMemcpyAsync(d_ranges, narrow.data(),
                                       narrow.size() * sizeof(IdRange),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_roff, roff.data(),
                                       roff.size() * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice, stream));
            expand_ranges_kernel<<<static_cast<uint32_t>(narrow.size()),
                                   kBlock1D, 0, stream>>>(
                d_ranges, d_roff, static_cast<uint32_t>(narrow.size()),
                sched.gather_ids);
            CUDA_CHECK(cudaGetLastError());
        }

        // Gather the rows into a compact buffer (scratched). Sized for T so
        // fp16 only uses half the bytes of an fp32 gather.
        sc.grow(sc.gather_db, sc.gather_db_cap, n_gather * dim * sizeof(T));
        sched.gather_db = sc.gather_db;
        gather_rows_kernel<T><<<grid1d(n_gather * dim), kBlock1D, 0, stream>>>(
            static_cast<T*>(sched.gather_db), d_db, sched.gather_ids,
            sched.n_gather, dim);
        CUDA_CHECK(cudaGetLastError());
    }

    // -- Coalesce adjacent masked tasks --------------------------------------
    // The bitmap branch above emits one masked task per 65536-wide container
    // (per-block in the negated case). At high selectivity that's one task
    // per container — at sel=90% with 153 dense bitmap containers, the
    // executor runs 153 separate sgemm + apply_mask + top-k passes, each
    // GEMM ~30us of math but ~10-15us of launch overhead. Coalescing pairs
    // (a,b) where (1) rows are contiguous (a.start+a.n_cols == b.start),
    // (2) mask buffers are memory-contiguous (a.mask + a.n_cols/32 == b.mask),
    // (3) mask_bit0 == 0 and (4) mask_invert matches collapses runs of
    // dense-bitmap or runs of negated-block masks into one wide task — same
    // mask kernel reads, one outer task iteration in the executor, ~10x
    // fewer kernel launches.
    //
    // Memory-contiguity holds by construction in both paths: bitmap_data is
    // packed in container-index (== key-sorted) order, and the negated path
    // writes its per-block masks into scratch.mask_pool at a uniform 2048-
    // uint32 stride in need_mask-iteration order, which IS block order.
    // The row-contiguity check is the safety net for any gap (an array/run
    // container interleaved, or a fully-excluded block in the negated path).
    //
    // Cap at kTileWMax so per-task scratch (~tile_w * q * 4 bytes) stays
    // bounded; the executor's inner tile loop handles any task >= tile_w.
    {
        std::vector<GemmTask> coalesced;
        coalesced.reserve(masked.size());
        for (const GemmTask& t : masked) {
            if (!coalesced.empty()) {
                GemmTask& back = coalesced.back();
                const uint32_t stride_words = back.n_cols >> 5;
                const bool rows_contig  = back.start + back.n_cols == t.start;
                const bool mask_contig  = back.mask + stride_words == t.mask;
                const bool both_word0   = back.mask_bit0 == 0u && t.mask_bit0 == 0u;
                const bool invert_match = back.mask_invert == t.mask_invert;
                const bool fits         =
                    static_cast<uint64_t>(back.n_cols) +
                    static_cast<uint64_t>(t.n_cols) <= kTileWMax;
                if (rows_contig && mask_contig && both_word0 && invert_match && fits) {
                    back.n_cols += t.n_cols;
                    continue;
                }
            }
            coalesced.push_back(t);
        }
        masked = std::move(coalesced);
    }

    // -- Assemble the task list ----------------------------------------------
    for (auto& r : ranges) {
        GemmTask t;
        t.kind   = GemmTask::kRange;
        t.start  = r.start;
        t.n_cols = r.end - r.start;
        sched.tasks.push_back(t);
    }
    for (auto& t : masked) sched.tasks.push_back(t);
    if (sched.n_gather > 0) {
        GemmTask t;
        t.kind   = GemmTask::kGather;
        t.start  = 0u;
        t.n_cols = sched.n_gather;
        sched.tasks.push_back(t);
    }
    for (auto& t : sched.tasks) sched.total_cols += t.n_cols;
    return sched;
}

}  // namespace (build_schedule_impl)

// Public overloads — thin wrappers over the templated body above.
SearchSchedule build_schedule(const GpuRoaring& filter, uint32_t n_rows,
                              const float* d_db, uint32_t dim,
                              cudaStream_t stream)
{
    return build_schedule_impl<float>(filter, n_rows, d_db, dim, stream);
}

SearchSchedule build_schedule(const GpuRoaring& filter, uint32_t n_rows,
                              const __half* d_db, uint32_t dim,
                              cudaStream_t stream)
{
    return build_schedule_impl<__half>(filter, n_rows, d_db, dim, stream);
}

void free_schedule(SearchSchedule& s)
{
    if (!s.scratched) {
        if (s.gather_ids) cudaFree(s.gather_ids);
        if (s.gather_db)  cudaFree(s.gather_db);
        if (s.mask_pool)  cudaFree(s.mask_pool);
    }
    s.gather_ids = nullptr;
    s.gather_db  = nullptr;
    s.mask_pool  = nullptr;
    s.n_gather   = 0;
    s.tasks.clear();
    s.total_cols = 0;
    s.scratched  = false;
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
void roaring_filtered_search(cublasHandle_t handle, const float* d_queries,
                             uint32_t q, const float* d_db, uint32_t n_rows,
                             uint32_t dim, const SearchSchedule& schedule,
                             uint32_t k, uint32_t* d_out_ids,
                             float* d_out_scores, cudaStream_t stream)
{
    if (k > kMaxK) throw std::runtime_error("filtered_search: k exceeds kMaxK");
    (void)n_rows;
    execute_impl<float>(handle, d_queries, q, d_db, dim, schedule.tasks,
                        static_cast<const float*>(schedule.gather_db),
                        schedule.gather_ids, k,
                        d_out_ids, d_out_scores, stream);
}

void roaring_filtered_search_fp16(cublasHandle_t handle,
                                  const __half* d_queries, uint32_t q,
                                  const __half* d_db, uint32_t n_rows,
                                  uint32_t dim, const SearchSchedule& schedule,
                                  uint32_t k, uint32_t* d_out_ids,
                                  float* d_out_scores, cudaStream_t stream)
{
    if (k > kMaxK) throw std::runtime_error("filtered_search_fp16: k exceeds kMaxK");
    (void)n_rows;
    execute_impl<__half>(handle, d_queries, q, d_db, dim, schedule.tasks,
                         static_cast<const __half*>(schedule.gather_db),
                         schedule.gather_ids, k,
                         d_out_ids, d_out_scores, stream);
}

void dense_filtered_search(cublasHandle_t handle, const float* d_queries,
                           uint32_t q, const float* d_db, uint32_t n_rows,
                           uint32_t dim, const GpuRoaring& filter, uint32_t k,
                           uint32_t* d_out_ids, float* d_out_scores,
                           cudaStream_t stream)
{
    if (k > kMaxK) throw std::runtime_error("dense_filtered_search: k exceeds kMaxK");

    // Option 1 of the design doc: expand the filter to a flat bitset, mask.
    // The bitset lands in a persistent grow-only buffer so its pointer is
    // stable across calls — that keeps execute()'s graph signature stable.
    uint32_t n_words = div_ceil(n_rows, 32u);
    static uint32_t* s_bits     = nullptr;
    static uint32_t  s_bits_cap = 0;
    if (n_words > s_bits_cap) {
        if (s_bits) CUDA_CHECK(cudaFree(s_bits));
        CUDA_CHECK(cudaMalloc(&s_bits, n_words * sizeof(uint32_t)));
        s_bits_cap = n_words;
    }
    decompress_to_bitset(filter, s_bits, n_words, stream);

    std::vector<GemmTask> tasks(1);
    tasks[0].kind        = GemmTask::kRange;
    tasks[0].start       = 0u;
    tasks[0].n_cols      = n_rows;
    tasks[0].mask        = s_bits;
    tasks[0].mask_bit0   = 0u;
    tasks[0].mask_invert = 0u;

    execute_impl<float>(handle, d_queries, q, d_db, dim, tasks, nullptr,
                        nullptr, k, d_out_ids, d_out_scores, stream);
}

}  // namespace cu_roaring
