#include "cu_roaring/detail/set_ops.cuh"
#include "cu_roaring/detail/upload.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

namespace cu_roaring {

// ============================================================================
// Work item structs with explicit constructors (nvcc doesn't support
// brace-enclosed init lists in push_back)
// ============================================================================
struct WorkItem {
    uint16_t key;
    int32_t  a_idx;
    int32_t  b_idx;
    ContainerType a_type;
    ContainerType b_type;

    __host__ WorkItem(uint16_t k, int32_t ai, int32_t bi,
                      ContainerType at, ContainerType bt)
        : key(k), a_idx(ai), b_idx(bi), a_type(at), b_type(bt) {}
};

struct BitmapBitmapWork {
    uint16_t key;
    uint32_t a_bitmap_offset;
    uint32_t b_bitmap_offset;

    __host__ BitmapBitmapWork(uint16_t k, uint32_t ao, uint32_t bo)
        : key(k), a_bitmap_offset(ao), b_bitmap_offset(bo) {}
};

struct BitmapArrayWork {
    uint16_t key;
    uint32_t bitmap_offset;
    uint32_t array_offset;
    uint16_t array_card;
    bool     bitmap_is_a;

    __host__ BitmapArrayWork(uint16_t k, uint32_t bo, uint32_t ao,
                             uint16_t card, bool bia)
        : key(k), bitmap_offset(bo), array_offset(ao),
          array_card(card), bitmap_is_a(bia) {}
};

struct ArrayArrayWork {
    uint16_t key;
    uint32_t a_offset;
    uint16_t a_card;
    uint32_t b_offset;
    uint16_t b_card;

    __host__ ArrayArrayWork(uint16_t k, uint32_t ao, uint16_t ac,
                            uint32_t bo, uint16_t bc)
        : key(k), a_offset(ao), a_card(ac), b_offset(bo), b_card(bc) {}
};

struct CopyWork {
    uint16_t      key;
    ContainerType type;
    uint32_t      offset;
    uint16_t      cardinality;
    bool          from_a;

    __host__ CopyWork(uint16_t k, ContainerType t, uint32_t o,
                      uint16_t c, bool fa)
        : key(k), type(t), offset(o), cardinality(c), from_a(fa) {}
};

// ============================================================================
// Phase 1: Container matching (CPU side)
// ============================================================================
static std::vector<WorkItem> match_containers(
    const uint16_t* h_keys_a, const ContainerType* h_types_a, uint32_t na,
    const uint16_t* h_keys_b, const ContainerType* h_types_b, uint32_t nb,
    SetOp op)
{
    std::vector<WorkItem> work;
    work.reserve(na + nb);

    uint32_t ia = 0, ib = 0;
    while (ia < na && ib < nb) {
        if (h_keys_a[ia] < h_keys_b[ib]) {
            if (op == SetOp::OR || op == SetOp::XOR || op == SetOp::ANDNOT) {
                work.emplace_back(h_keys_a[ia], static_cast<int32_t>(ia), -1,
                                  h_types_a[ia], ContainerType::ARRAY);
            }
            ++ia;
        } else if (h_keys_a[ia] > h_keys_b[ib]) {
            if (op == SetOp::OR || op == SetOp::XOR) {
                work.emplace_back(h_keys_b[ib], -1, static_cast<int32_t>(ib),
                                  ContainerType::ARRAY, h_types_b[ib]);
            }
            ++ib;
        } else {
            work.emplace_back(h_keys_a[ia],
                              static_cast<int32_t>(ia),
                              static_cast<int32_t>(ib),
                              h_types_a[ia], h_types_b[ib]);
            ++ia;
            ++ib;
        }
    }
    while (ia < na) {
        if (op == SetOp::OR || op == SetOp::XOR || op == SetOp::ANDNOT) {
            work.emplace_back(h_keys_a[ia], static_cast<int32_t>(ia), -1,
                              h_types_a[ia], ContainerType::ARRAY);
        }
        ++ia;
    }
    while (ib < nb) {
        if (op == SetOp::OR || op == SetOp::XOR) {
            work.emplace_back(h_keys_b[ib], -1, static_cast<int32_t>(ib),
                              ContainerType::ARRAY, h_types_b[ib]);
        }
        ++ib;
    }
    return work;
}

// ============================================================================
// Bitmap x Bitmap kernels (one per operation)
// ============================================================================
template <SetOp OP>
__device__ __forceinline__ uint64_t apply_op(uint64_t a, uint64_t b);

template <> __device__ __forceinline__ uint64_t apply_op<SetOp::AND>(uint64_t a, uint64_t b) { return a & b; }
template <> __device__ __forceinline__ uint64_t apply_op<SetOp::OR>(uint64_t a, uint64_t b) { return a | b; }
template <> __device__ __forceinline__ uint64_t apply_op<SetOp::ANDNOT>(uint64_t a, uint64_t b) { return a & ~b; }
template <> __device__ __forceinline__ uint64_t apply_op<SetOp::XOR>(uint64_t a, uint64_t b) { return a ^ b; }

template <SetOp OP>
__global__ void bitmap_bitmap_kernel(
    const BitmapBitmapWork* work,
    uint32_t n_pairs,
    const uint64_t* a_bitmap_data,
    const uint64_t* b_bitmap_data,
    uint64_t* out_bitmap_data,
    uint16_t* out_cardinalities)
{
    uint32_t pair_idx = blockIdx.x;
    if (pair_idx >= n_pairs) return;

    const BitmapBitmapWork& w = work[pair_idx];
    const uint64_t* a = a_bitmap_data + w.a_bitmap_offset;
    const uint64_t* b = b_bitmap_data + w.b_bitmap_offset;
    uint64_t* out = out_bitmap_data + pair_idx * 1024;

    uint32_t local_popcount = 0;

    for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x) {
        uint64_t val = apply_op<OP>(a[i], b[i]);
        out[i] = val;
        local_popcount += __popcll(val);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_popcount += __shfl_down_sync(0xFFFFFFFF, local_popcount, offset);
    }

    __shared__ uint32_t warp_counts[8];
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_counts[warp_id] = local_popcount;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t total = 0;
        for (uint32_t w = 0; w < blockDim.x / 32; ++w) {
            total += warp_counts[w];
        }
        out_cardinalities[pair_idx] = static_cast<uint16_t>(
            total > 65535 ? 0 : total);
    }
}

// ============================================================================
// Bitmap x Array AND kernel
// ============================================================================
__global__ void bitmap_array_and_kernel(
    const BitmapArrayWork* work,
    uint32_t n_pairs,
    const uint64_t* bitmap_pool_a,
    const uint64_t* bitmap_pool_b,
    const uint16_t* array_pool_a,
    const uint16_t* array_pool_b,
    uint16_t* out_array_data,
    const uint32_t* out_array_offsets,
    uint16_t* out_cardinalities)
{
    uint32_t pair_idx = blockIdx.x;
    if (pair_idx >= n_pairs) return;

    const BitmapArrayWork& w = work[pair_idx];

    const uint64_t* bmp = w.bitmap_is_a
        ? (bitmap_pool_a + w.bitmap_offset)
        : (bitmap_pool_b + w.bitmap_offset);
    const uint16_t* arr = w.bitmap_is_a
        ? (array_pool_b + w.array_offset)
        : (array_pool_a + w.array_offset);

    uint16_t* out = out_array_data + out_array_offsets[pair_idx];

    extern __shared__ uint16_t smem[];
    uint16_t* s_results = smem;

    __shared__ uint32_t s_count;
    __shared__ uint32_t warp_base[8];
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    for (uint32_t base = 0; base < w.array_card; base += blockDim.x) {
        uint32_t i = base + threadIdx.x;
        bool present = false;
        uint16_t val = 0;

        if (i < w.array_card) {
            val = arr[i];
            uint32_t word_idx = val / 64u;
            uint32_t bit_pos  = val % 64u;
            present = (bmp[word_idx] >> bit_pos) & 1u;
        }

        uint32_t warp_id = threadIdx.x / 32;
        uint32_t lane_id = threadIdx.x % 32;
        uint32_t mask = __ballot_sync(0xFFFFFFFF, present);
        uint32_t warp_count = __popc(mask);
        uint32_t lane_prefix = __popc(mask & ((1u << lane_id) - 1));

        if (lane_id == 0 && warp_count > 0) {
            warp_base[warp_id] = atomicAdd(&s_count, warp_count);
        }
        __syncwarp();

        if (present) {
            s_results[warp_base[warp_id] + lane_prefix] = val;
        }
        __syncthreads();
    }

    uint32_t total = s_count;
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        out[i] = s_results[i];
    }
    if (threadIdx.x == 0) {
        out_cardinalities[pair_idx] = static_cast<uint16_t>(total);
    }
}

// ============================================================================
// Array x Array AND kernel
// ============================================================================
__global__ void array_array_and_kernel(
    const ArrayArrayWork* work,
    uint32_t n_pairs,
    const uint16_t* array_pool_a,
    const uint16_t* array_pool_b,
    uint16_t* out_array_data,
    const uint32_t* out_array_offsets,
    uint16_t* out_cardinalities)
{
    uint32_t pair_idx = blockIdx.x;
    if (pair_idx >= n_pairs) return;

    const ArrayArrayWork& w = work[pair_idx];
    const uint16_t* a = array_pool_a + w.a_offset;
    const uint16_t* b = array_pool_b + w.b_offset;
    uint16_t* out = out_array_data + out_array_offsets[pair_idx];

    extern __shared__ uint16_t smem[];
    uint16_t* sa = smem;
    uint16_t* sb = sa + w.a_card;

    for (uint32_t i = threadIdx.x; i < w.a_card; i += blockDim.x) {
        sa[i] = a[i];
    }
    for (uint32_t i = threadIdx.x; i < w.b_card; i += blockDim.x) {
        sb[i] = b[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t ia = 0, ib = 0, ic = 0;
        while (ia < w.a_card && ib < w.b_card) {
            if (sa[ia] < sb[ib]) {
                ++ia;
            } else if (sa[ia] > sb[ib]) {
                ++ib;
            } else {
                out[ic++] = sa[ia];
                ++ia;
                ++ib;
            }
        }
        out_cardinalities[pair_idx] = static_cast<uint16_t>(ic);
    }
}

// ============================================================================
// Expansion kernels: expand array/run containers to bitmap format
// Used as fallback for non-AND operations on mixed type pairs
// ============================================================================
struct ExpandWork {
    uint32_t src_offset;   // offset in source pool (elements for array, pairs for run)
    uint16_t cardinality;  // num elements (array) or num runs (run)
    ContainerType type;

    __host__ ExpandWork(uint32_t so, uint16_t c, ContainerType t)
        : src_offset(so), cardinality(c), type(t) {}
};

// Expand an array or run container to a 1024-word bitmap.
// One block per container.
__global__ void expand_to_bitmap_kernel(
    const ExpandWork* work,
    uint32_t n_items,
    const uint16_t* array_pool,
    const uint16_t* run_pool,
    uint64_t* out_bitmaps)  // [n_items * 1024]
{
    uint32_t idx = blockIdx.x;
    if (idx >= n_items) return;

    const ExpandWork& w = work[idx];
    uint64_t* out = out_bitmaps + idx * 1024;

    // Zero the bitmap first
    for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x) {
        out[i] = 0;
    }
    __syncthreads();

    if (w.type == ContainerType::ARRAY) {
        const uint16_t* arr = array_pool + w.src_offset;
        for (uint32_t i = threadIdx.x; i < w.cardinality; i += blockDim.x) {
            uint16_t val = arr[i];
            uint32_t word_idx = val / 64u;
            uint64_t bit_mask = 1ULL << (val % 64u);
            atomicOr(reinterpret_cast<unsigned long long*>(&out[word_idx]),
                     static_cast<unsigned long long>(bit_mask));
        }
    } else if (w.type == ContainerType::RUN) {
        const uint16_t* runs = run_pool + w.src_offset;
        for (uint32_t r = threadIdx.x; r < w.cardinality; r += blockDim.x) {
            uint16_t start = runs[r * 2];
            uint16_t length = runs[r * 2 + 1];
            // Clamp end to 0xFFFF: writes must stay inside the 1024-word
            // per-container bitmap (`out`), so a malformed run that claims
            // start + length > 65535 must not escape this container.
            uint32_t end = static_cast<uint32_t>(start) + length;
            if (end > 0xFFFFu) end = 0xFFFFu;
            for (uint32_t v = start; v <= end; ++v) {
                uint32_t word_idx = v / 64u;
                uint64_t bit_mask = 1ULL << (v % 64u);
                atomicOr(reinterpret_cast<unsigned long long*>(&out[word_idx]),
                         static_cast<unsigned long long>(bit_mask));
            }
        }
    }
}

// ============================================================================
// Copy-through kernels
// ============================================================================
__global__ void copy_bitmap_kernel(
    const CopyWork* work,
    uint32_t n_items,
    const uint64_t* a_bitmap_data,
    const uint64_t* b_bitmap_data,
    uint64_t* out_bitmap_data,
    uint32_t out_bitmap_start_idx)
{
    uint32_t item_idx = blockIdx.x;
    if (item_idx >= n_items) return;

    const CopyWork& w = work[item_idx];
    const uint64_t* src = w.from_a
        ? (a_bitmap_data + w.offset / sizeof(uint64_t))
        : (b_bitmap_data + w.offset / sizeof(uint64_t));
    uint64_t* dst = out_bitmap_data + (out_bitmap_start_idx + item_idx) * 1024;

    for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void copy_array_kernel(
    const CopyWork* work,
    uint32_t n_items,
    const uint16_t* a_array_data,
    const uint16_t* b_array_data,
    uint16_t* out_array_data,
    const uint32_t* out_offsets)
{
    uint32_t item_idx = blockIdx.x;
    if (item_idx >= n_items) return;

    const CopyWork& w = work[item_idx];
    const uint16_t* src = w.from_a
        ? (a_array_data + w.offset / sizeof(uint16_t))
        : (b_array_data + w.offset / sizeof(uint16_t));
    uint16_t* dst = out_array_data + out_offsets[item_idx];

    for (uint32_t i = threadIdx.x; i < w.cardinality; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void copy_run_kernel(
    const CopyWork* work,
    uint32_t n_items,
    const uint16_t* a_run_data,
    const uint16_t* b_run_data,
    uint16_t* out_run_data,
    const uint32_t* out_offsets)
{
    uint32_t item_idx = blockIdx.x;
    if (item_idx >= n_items) return;

    const CopyWork& w = work[item_idx];
    const uint16_t* src = w.from_a
        ? (a_run_data + w.offset / sizeof(uint16_t))
        : (b_run_data + w.offset / sizeof(uint16_t));
    uint16_t* dst = out_run_data + out_offsets[item_idx];
    uint32_t n_vals = static_cast<uint32_t>(w.cardinality) * 2u;

    for (uint32_t i = threadIdx.x; i < n_vals; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ============================================================================
// Helper: download index arrays from GpuRoaring for CPU-side matching
// ============================================================================
struct HostIndex {
    std::vector<uint16_t>      keys;
    std::vector<ContainerType> types;
    std::vector<uint32_t>      offsets;
    std::vector<uint16_t>      cardinalities;
};

static HostIndex download_index(const GpuRoaring& g, cudaStream_t stream) {
    HostIndex h;
    uint32_t n = g.n_containers;
    h.keys.resize(n);
    h.types.resize(n);
    h.offsets.resize(n);
    h.cardinalities.resize(n);

    if (n == 0) return h;

    CUDA_CHECK(cudaMemcpyAsync(h.keys.data(), g.keys,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h.types.data(), g.types,
                               n * sizeof(ContainerType),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h.offsets.data(), g.offsets,
                               n * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h.cardinalities.data(), g.cardinalities,
                               n * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h;
}

// ============================================================================
// Dispatch helper for bitmap_bitmap_kernel
// ============================================================================
static void launch_bitmap_bitmap(
    SetOp op, uint32_t n_bb,
    const BitmapBitmapWork* d_work,
    const uint64_t* a_data, const uint64_t* b_data,
    uint64_t* out_data, uint16_t* out_cards,
    cudaStream_t stream)
{
    switch (op) {
        case SetOp::AND:
            bitmap_bitmap_kernel<SetOp::AND><<<n_bb, 256, 0, stream>>>(
                d_work, n_bb, a_data, b_data, out_data, out_cards);
            break;
        case SetOp::OR:
            bitmap_bitmap_kernel<SetOp::OR><<<n_bb, 256, 0, stream>>>(
                d_work, n_bb, a_data, b_data, out_data, out_cards);
            break;
        case SetOp::ANDNOT:
            bitmap_bitmap_kernel<SetOp::ANDNOT><<<n_bb, 256, 0, stream>>>(
                d_work, n_bb, a_data, b_data, out_data, out_cards);
            break;
        case SetOp::XOR:
            bitmap_bitmap_kernel<SetOp::XOR><<<n_bb, 256, 0, stream>>>(
                d_work, n_bb, a_data, b_data, out_data, out_cards);
            break;
    }
}

// ============================================================================
// Main set_operation implementation
// ============================================================================
// ============================================================================
// DeMorgan transform: rewrite set ops when inputs are negated.
//
// When a GpuRoaring has negated=true, its stored set is the complement of
// the logical set. We transform the operation + swap operands so existing
// kernels work on the physical (stored) sets, then set the result's negated
// flag accordingly.
//
// Truth table (all 16 cases):
//   AND(~A,B)  = B &~A       → ANDNOT(B,A), neg=F  [swap]
//   AND(A,~B)  = A &~B       → ANDNOT(A,B), neg=F
//   AND(~A,~B) = ~(A|B)      → OR(A,B),     neg=T
//   OR(~A,B)   = ~(A &~B)    → ANDNOT(A,B), neg=T
//   OR(A,~B)   = ~(B &~A)    → ANDNOT(B,A), neg=T  [swap]
//   OR(~A,~B)  = ~(A&B)      → AND(A,B),    neg=T
//   ANDNOT(~A,B) = ~A &~B = ~(A|B)  → OR(A,B),     neg=T
//   ANDNOT(A,~B) = A & ~~B = A&B    → AND(A,B),    neg=F
//   ANDNOT(~A,~B) = ~A&~~B = ~A&B = B&~A → ANDNOT(B,A), neg=F [swap]
//   XOR(~A,B)  = ~(A^B)      → XOR(A,B),    neg=T
//   XOR(A,~B)  = ~(A^B)      → XOR(A,B),    neg=T
//   XOR(~A,~B) = A^B         → XOR(A,B),    neg=F
// ============================================================================
static void resolve_negation(SetOp& op,
                              const GpuRoaring*& a,
                              const GpuRoaring*& b,
                              bool& result_negated)
{
    bool na = a->negated;
    bool nb = b->negated;
    result_negated = false;

    if (!na && !nb) return;  // fast path: nothing to transform

    switch (op) {
    case SetOp::AND:
        if (na && !nb)       { op = SetOp::ANDNOT; std::swap(a, b); }
        else if (!na && nb)  { op = SetOp::ANDNOT; }
        else /* na && nb */  { op = SetOp::OR;  result_negated = true; }
        break;
    case SetOp::OR:
        if (na && !nb)       { op = SetOp::ANDNOT; result_negated = true; }
        else if (!na && nb)  { op = SetOp::ANDNOT; std::swap(a, b); result_negated = true; }
        else /* na && nb */  { op = SetOp::AND; result_negated = true; }
        break;
    case SetOp::ANDNOT:
        if (na && !nb)       { op = SetOp::OR;  result_negated = true; }
        else if (!na && nb)  { op = SetOp::AND; }
        else /* na && nb */  { op = SetOp::ANDNOT; std::swap(a, b); }
        break;
    case SetOp::XOR:
        if (na && nb)        { /* op stays XOR, negated stays false */ }
        else                 { result_negated = true; /* op stays XOR */ }
        break;
    }
}

GpuRoaring set_operation(const GpuRoaring& a, const GpuRoaring& b,
                         SetOp op, cudaStream_t stream)
{
    // DeMorgan transform for negated inputs
    const GpuRoaring* pa = &a;
    const GpuRoaring* pb = &b;
    bool result_negated = false;
    resolve_negation(op, pa, pb, result_negated);

    HostIndex ha = download_index(*pa, stream);
    HostIndex hb = download_index(*pb, stream);

    auto work = match_containers(
        ha.keys.data(), ha.types.data(), pa->n_containers,
        hb.keys.data(), hb.types.data(), pb->n_containers,
        op);

    if (work.empty()) {
        GpuRoaring empty{};
        empty.universe_size = std::max(pa->universe_size, pb->universe_size);
        empty.negated = result_negated;
        return empty;
    }

    // Classify work items by type pair
    std::vector<BitmapBitmapWork> bb_work;
    std::vector<BitmapArrayWork> ba_work;   // AND only
    std::vector<ArrayArrayWork> aa_work;    // AND only
    std::vector<CopyWork> copy_bitmap_work;
    std::vector<CopyWork> copy_array_work;
    std::vector<CopyWork> copy_run_work;

    // For non-AND operations with non-bitmap×bitmap pairs: expand to bitmap
    // Each entry records which containers to expand and the output key
    struct MixedPair {
        uint16_t key;
        // Container A info
        ContainerType a_type;
        uint32_t a_offset;
        uint16_t a_card;
        bool a_is_bitmap;
        // Container B info
        ContainerType b_type;
        uint32_t b_offset;
        uint16_t b_card;
        bool b_is_bitmap;

        MixedPair(uint16_t k,
                  ContainerType at, uint32_t ao, uint16_t ac, bool ab,
                  ContainerType bt, uint32_t bo, uint16_t bc, bool bb_)
            : key(k), a_type(at), a_offset(ao), a_card(ac), a_is_bitmap(ab),
              b_type(bt), b_offset(bo), b_card(bc), b_is_bitmap(bb_) {}
    };
    std::vector<MixedPair> mixed_work;

    for (auto& wi : work) {
        if (wi.a_idx < 0) {
            auto& ho = hb;
            int32_t idx = wi.b_idx;
            CopyWork cw(wi.key, ho.types[idx], ho.offsets[idx],
                        ho.cardinalities[idx], false);
            if (ho.types[idx] == ContainerType::BITMAP)
                copy_bitmap_work.push_back(cw);
            else if (ho.types[idx] == ContainerType::ARRAY)
                copy_array_work.push_back(cw);
            else
                copy_run_work.push_back(cw);
        } else if (wi.b_idx < 0) {
            auto& ho = ha;
            int32_t idx = wi.a_idx;
            CopyWork cw(wi.key, ho.types[idx], ho.offsets[idx],
                        ho.cardinalities[idx], true);
            if (ho.types[idx] == ContainerType::BITMAP)
                copy_bitmap_work.push_back(cw);
            else if (ho.types[idx] == ContainerType::ARRAY)
                copy_array_work.push_back(cw);
            else
                copy_run_work.push_back(cw);
        } else {
            ContainerType ta = ha.types[wi.a_idx];
            ContainerType tb = hb.types[wi.b_idx];

            if (ta == ContainerType::BITMAP && tb == ContainerType::BITMAP) {
                bb_work.emplace_back(wi.key,
                                     ha.offsets[wi.a_idx] / static_cast<uint32_t>(sizeof(uint64_t)),
                                     hb.offsets[wi.b_idx] / static_cast<uint32_t>(sizeof(uint64_t)));
            } else if (op == SetOp::AND &&
                       ((ta == ContainerType::BITMAP && tb == ContainerType::ARRAY) ||
                        (ta == ContainerType::ARRAY && tb == ContainerType::BITMAP))) {
                // AND bitmap×array: use specialized kernel
                bool bmp_is_a = (ta == ContainerType::BITMAP);
                ba_work.emplace_back(wi.key,
                    bmp_is_a ? ha.offsets[wi.a_idx] / static_cast<uint32_t>(sizeof(uint64_t))
                             : hb.offsets[wi.b_idx] / static_cast<uint32_t>(sizeof(uint64_t)),
                    bmp_is_a ? hb.offsets[wi.b_idx] / static_cast<uint32_t>(sizeof(uint16_t))
                             : ha.offsets[wi.a_idx] / static_cast<uint32_t>(sizeof(uint16_t)),
                    bmp_is_a ? hb.cardinalities[wi.b_idx]
                             : ha.cardinalities[wi.a_idx],
                    bmp_is_a);
            } else if (op == SetOp::AND &&
                       ta == ContainerType::ARRAY && tb == ContainerType::ARRAY) {
                // AND array×array: use specialized kernel
                aa_work.emplace_back(wi.key,
                                     ha.offsets[wi.a_idx] / static_cast<uint32_t>(sizeof(uint16_t)),
                                     ha.cardinalities[wi.a_idx],
                                     hb.offsets[wi.b_idx] / static_cast<uint32_t>(sizeof(uint16_t)),
                                     hb.cardinalities[wi.b_idx]);
            } else {
                // Non-AND with non-bitmap×bitmap: expand to bitmap and use bb kernel
                auto get_offset = [](ContainerType t, uint32_t off) -> uint32_t {
                    if (t == ContainerType::BITMAP) return off / static_cast<uint32_t>(sizeof(uint64_t));
                    if (t == ContainerType::ARRAY)  return off / static_cast<uint32_t>(sizeof(uint16_t));
                    return off / static_cast<uint32_t>(sizeof(uint16_t));  // RUN: uint16_t pairs
                };
                mixed_work.emplace_back(wi.key,
                    ta, get_offset(ta, ha.offsets[wi.a_idx]), ha.cardinalities[wi.a_idx],
                    ta == ContainerType::BITMAP,
                    tb, get_offset(tb, hb.offsets[wi.b_idx]), hb.cardinalities[wi.b_idx],
                    tb == ContainerType::BITMAP);
            }
        }
    }

    // Count output containers
    uint32_t out_n_bitmap = static_cast<uint32_t>(bb_work.size() +
                                                   mixed_work.size() +
                                                   copy_bitmap_work.size());
    uint32_t out_n_array  = static_cast<uint32_t>(ba_work.size() +
                                                   aa_work.size() +
                                                   copy_array_work.size());
    uint32_t out_n_run    = static_cast<uint32_t>(copy_run_work.size());

    // Allocate output bitmap pool
    uint64_t* d_out_bitmap = nullptr;
    if (out_n_bitmap > 0) {
        CUDA_CHECK(cudaMallocAsync(&d_out_bitmap,
                              out_n_bitmap * 1024 * sizeof(uint64_t), stream));
    }

    // Execute Bitmap x Bitmap
    std::vector<uint16_t> out_bb_cards(bb_work.size());
    if (!bb_work.empty()) {
        BitmapBitmapWork* d_bb_work = nullptr;
        uint16_t* d_bb_cards = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_bb_work,
                              bb_work.size() * sizeof(BitmapBitmapWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_bb_cards,
                              bb_work.size() * sizeof(uint16_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_bb_work, bb_work.data(),
                                   bb_work.size() * sizeof(BitmapBitmapWork),
                                   cudaMemcpyHostToDevice, stream));

        uint32_t n_bb = static_cast<uint32_t>(bb_work.size());
        launch_bitmap_bitmap(op, n_bb, d_bb_work,
                             pa->bitmap_data, pb->bitmap_data,
                             d_out_bitmap, d_bb_cards, stream);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(out_bb_cards.data(), d_bb_cards,
                                   bb_work.size() * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(d_bb_work, stream));
        CUDA_CHECK(cudaFreeAsync(d_bb_cards, stream));
    }

    // Mixed-pair handling: expand non-bitmap containers to temp bitmaps,
    // then apply bitmap×bitmap operation
    std::vector<uint16_t> out_mixed_cards(mixed_work.size());
    uint64_t* d_temp_a_bitmaps = nullptr;
    uint64_t* d_temp_b_bitmaps = nullptr;
    if (!mixed_work.empty()) {
        uint32_t n_mixed = static_cast<uint32_t>(mixed_work.size());

        // Allocate temp bitmap pools (1024 uint64_t per container)
        CUDA_CHECK(cudaMallocAsync(&d_temp_a_bitmaps, n_mixed * 1024 * sizeof(uint64_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_temp_b_bitmaps, n_mixed * 1024 * sizeof(uint64_t), stream));

        // Build expansion work lists for A and B sides
        std::vector<ExpandWork> expand_a, expand_b;
        for (auto& mp : mixed_work) {
            expand_a.emplace_back(mp.a_offset, mp.a_card, mp.a_type);
            expand_b.emplace_back(mp.b_offset, mp.b_card, mp.b_type);
        }

        // Upload expand work
        ExpandWork* d_exp_a = nullptr;
        ExpandWork* d_exp_b = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_exp_a, n_mixed * sizeof(ExpandWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_exp_b, n_mixed * sizeof(ExpandWork), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_exp_a, expand_a.data(),
                                   n_mixed * sizeof(ExpandWork),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_exp_b, expand_b.data(),
                                   n_mixed * sizeof(ExpandWork),
                                   cudaMemcpyHostToDevice, stream));

        // For containers that are already bitmaps, copy directly instead of expanding
        // For array/run containers, expand to bitmap
        // We handle both uniformly: expansion kernel checks type

        // But for bitmap containers, the expand kernel needs access to bitmap pools
        // Approach: for bitmap type, we copy from the actual bitmap pool
        // For array/run, we expand using the expand kernel

        // First zero temp pools
        CUDA_CHECK(cudaMemsetAsync(d_temp_a_bitmaps, 0,
                                   n_mixed * 1024 * sizeof(uint64_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_temp_b_bitmaps, 0,
                                   n_mixed * 1024 * sizeof(uint64_t), stream));

        // Expand A-side non-bitmap containers
        expand_to_bitmap_kernel<<<n_mixed, 256, 0, stream>>>(
            d_exp_a, n_mixed, pa->array_data, pa->run_data, d_temp_a_bitmaps);
        CUDA_CHECK(cudaGetLastError());

        // Expand B-side non-bitmap containers
        expand_to_bitmap_kernel<<<n_mixed, 256, 0, stream>>>(
            d_exp_b, n_mixed, pb->array_data, pb->run_data, d_temp_b_bitmaps);
        CUDA_CHECK(cudaGetLastError());

        // For containers that are already bitmaps, copy from actual bitmap pool
        // We need a simple copy kernel for this
        for (uint32_t i = 0; i < n_mixed; ++i) {
            if (mixed_work[i].a_is_bitmap) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_temp_a_bitmaps + i * 1024,
                    pa->bitmap_data + mixed_work[i].a_offset,
                    1024 * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream));
            }
            if (mixed_work[i].b_is_bitmap) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_temp_b_bitmaps + i * 1024,
                    pb->bitmap_data + mixed_work[i].b_offset,
                    1024 * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }

        // Build BitmapBitmapWork for temp pools (sequential offsets)
        std::vector<BitmapBitmapWork> mixed_bb;
        for (uint32_t i = 0; i < n_mixed; ++i) {
            mixed_bb.emplace_back(mixed_work[i].key, i * 1024, i * 1024);
        }

        BitmapBitmapWork* d_mixed_bb = nullptr;
        uint16_t* d_mixed_cards = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_mixed_bb, n_mixed * sizeof(BitmapBitmapWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_mixed_cards, n_mixed * sizeof(uint16_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_mixed_bb, mixed_bb.data(),
                                   n_mixed * sizeof(BitmapBitmapWork),
                                   cudaMemcpyHostToDevice, stream));

        // Output goes after bb_work results in the output bitmap pool
        uint32_t mixed_out_offset = static_cast<uint32_t>(bb_work.size());
        // We need the output to go to the right place in d_out_bitmap
        // The bitmap_bitmap_kernel writes to out_bitmap_data + pair_idx * 1024
        // So we offset the output pointer
        launch_bitmap_bitmap(op, n_mixed, d_mixed_bb,
                             d_temp_a_bitmaps, d_temp_b_bitmaps,
                             d_out_bitmap + mixed_out_offset * 1024,
                             d_mixed_cards, stream);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(out_mixed_cards.data(), d_mixed_cards,
                                   n_mixed * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(d_exp_a, stream));
        CUDA_CHECK(cudaFreeAsync(d_exp_b, stream));
        CUDA_CHECK(cudaFreeAsync(d_mixed_bb, stream));
        CUDA_CHECK(cudaFreeAsync(d_mixed_cards, stream));
    }

    // Copy-through bitmap containers
    if (!copy_bitmap_work.empty()) {
        CopyWork* d_copy_work = nullptr;
        uint32_t n_copy = static_cast<uint32_t>(copy_bitmap_work.size());
        CUDA_CHECK(cudaMallocAsync(&d_copy_work, n_copy * sizeof(CopyWork), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_copy_work, copy_bitmap_work.data(),
                                   n_copy * sizeof(CopyWork),
                                   cudaMemcpyHostToDevice, stream));
        copy_bitmap_kernel<<<n_copy, 256, 0, stream>>>(
            d_copy_work, n_copy,
            pa->bitmap_data, pb->bitmap_data,
            d_out_bitmap,
            static_cast<uint32_t>(bb_work.size() + mixed_work.size()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFreeAsync(d_copy_work, stream));
    }

    // Compute output offsets for array containers
    std::vector<uint32_t> ba_offsets(ba_work.size());
    uint32_t ba_total_elems = 0;
    for (size_t i = 0; i < ba_work.size(); ++i) {
        ba_offsets[i] = ba_total_elems;
        ba_total_elems += ba_work[i].array_card;
    }

    std::vector<uint32_t> aa_offsets(aa_work.size());
    uint32_t aa_total_elems = 0;
    for (size_t i = 0; i < aa_work.size(); ++i) {
        aa_offsets[i] = ba_total_elems + aa_total_elems;
        aa_total_elems += std::min(aa_work[i].a_card, aa_work[i].b_card);
    }

    std::vector<uint32_t> ca_offsets(copy_array_work.size());
    uint32_t ca_total_elems = 0;
    for (size_t i = 0; i < copy_array_work.size(); ++i) {
        ca_offsets[i] = ba_total_elems + aa_total_elems + ca_total_elems;
        ca_total_elems += copy_array_work[i].cardinality;
    }

    uint32_t total_array_elems = ba_total_elems + aa_total_elems + ca_total_elems;
    uint16_t* d_out_array = nullptr;
    if (total_array_elems > 0) {
        CUDA_CHECK(cudaMallocAsync(&d_out_array,
                              total_array_elems * sizeof(uint16_t), stream));
    }

    // Execute Bitmap x Array AND
    std::vector<uint16_t> out_ba_cards(ba_work.size());
    if (!ba_work.empty() && op == SetOp::AND) {
        BitmapArrayWork* d_ba_work = nullptr;
        uint32_t* d_ba_offsets = nullptr;
        uint16_t* d_ba_cards = nullptr;
        uint32_t n_ba = static_cast<uint32_t>(ba_work.size());

        CUDA_CHECK(cudaMallocAsync(&d_ba_work, n_ba * sizeof(BitmapArrayWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_ba_offsets, n_ba * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_ba_cards, n_ba * sizeof(uint16_t), stream));

        CUDA_CHECK(cudaMemcpyAsync(d_ba_work, ba_work.data(),
                                   n_ba * sizeof(BitmapArrayWork),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ba_offsets, ba_offsets.data(),
                                   n_ba * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        uint32_t max_card = 0;
        for (auto& bw : ba_work) {
            max_card = std::max(max_card, static_cast<uint32_t>(bw.array_card));
        }
        size_t smem_size = max_card * sizeof(uint16_t);

        bitmap_array_and_kernel<<<n_ba, 256, smem_size, stream>>>(
            d_ba_work, n_ba,
            pa->bitmap_data, pb->bitmap_data,
            pa->array_data, pb->array_data,
            d_out_array, d_ba_offsets, d_ba_cards);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(out_ba_cards.data(), d_ba_cards,
                                   n_ba * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(d_ba_work, stream));
        CUDA_CHECK(cudaFreeAsync(d_ba_offsets, stream));
        CUDA_CHECK(cudaFreeAsync(d_ba_cards, stream));
    }

    // Execute Array x Array AND
    std::vector<uint16_t> out_aa_cards(aa_work.size());
    if (!aa_work.empty() && op == SetOp::AND) {
        ArrayArrayWork* d_aa_work = nullptr;
        uint32_t* d_aa_offsets = nullptr;
        uint16_t* d_aa_cards = nullptr;
        uint32_t n_aa = static_cast<uint32_t>(aa_work.size());

        CUDA_CHECK(cudaMallocAsync(&d_aa_work, n_aa * sizeof(ArrayArrayWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_aa_offsets, n_aa * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_aa_cards, n_aa * sizeof(uint16_t), stream));

        CUDA_CHECK(cudaMemcpyAsync(d_aa_work, aa_work.data(),
                                   n_aa * sizeof(ArrayArrayWork),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_aa_offsets, aa_offsets.data(),
                                   n_aa * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        uint32_t max_smem = 0;
        for (auto& aw : aa_work) {
            max_smem = std::max(max_smem,
                                static_cast<uint32_t>(aw.a_card + aw.b_card));
        }
        size_t smem_size = max_smem * sizeof(uint16_t);
        smem_size = std::min(smem_size, static_cast<size_t>(48 * 1024));

        array_array_and_kernel<<<n_aa, 256, smem_size, stream>>>(
            d_aa_work, n_aa,
            pa->array_data, pb->array_data,
            d_out_array, d_aa_offsets, d_aa_cards);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(out_aa_cards.data(), d_aa_cards,
                                   n_aa * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(d_aa_work, stream));
        CUDA_CHECK(cudaFreeAsync(d_aa_offsets, stream));
        CUDA_CHECK(cudaFreeAsync(d_aa_cards, stream));
    }

    // Copy-through array containers
    if (!copy_array_work.empty()) {
        CopyWork* d_copy = nullptr;
        uint32_t* d_offsets = nullptr;
        uint32_t n_ca = static_cast<uint32_t>(copy_array_work.size());

        CUDA_CHECK(cudaMallocAsync(&d_copy, n_ca * sizeof(CopyWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_offsets, n_ca * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_copy, copy_array_work.data(),
                                   n_ca * sizeof(CopyWork),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, ca_offsets.data(),
                                   n_ca * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        copy_array_kernel<<<n_ca, 256, 0, stream>>>(
            d_copy, n_ca, pa->array_data, pb->array_data,
            d_out_array, d_offsets);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFreeAsync(d_copy, stream));
        CUDA_CHECK(cudaFreeAsync(d_offsets, stream));
    }

    // Copy-through run containers
    uint16_t* d_out_run = nullptr;
    uint32_t total_run_pairs = 0;
    std::vector<uint32_t> cr_offsets(copy_run_work.size());
    if (!copy_run_work.empty()) {
        for (size_t i = 0; i < copy_run_work.size(); ++i) {
            cr_offsets[i] = total_run_pairs * 2;
            total_run_pairs += copy_run_work[i].cardinality;
        }
        CUDA_CHECK(cudaMallocAsync(&d_out_run,
                              total_run_pairs * 2 * sizeof(uint16_t), stream));

        CopyWork* d_copy = nullptr;
        uint32_t* d_offsets = nullptr;
        uint32_t n_cr = static_cast<uint32_t>(copy_run_work.size());

        CUDA_CHECK(cudaMallocAsync(&d_copy, n_cr * sizeof(CopyWork), stream));
        CUDA_CHECK(cudaMallocAsync(&d_offsets, n_cr * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_copy, copy_run_work.data(),
                                   n_cr * sizeof(CopyWork),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, cr_offsets.data(),
                                   n_cr * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        copy_run_kernel<<<n_cr, 256, 0, stream>>>(
            d_copy, n_cr, pa->run_data, pb->run_data,
            d_out_run, d_offsets);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFreeAsync(d_copy, stream));
        CUDA_CHECK(cudaFreeAsync(d_offsets, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Phase 4: Assemble output GpuRoaring
    std::vector<uint16_t>      out_keys;
    std::vector<ContainerType> out_types;
    std::vector<uint32_t>      out_offsets_vec;
    std::vector<uint16_t>      out_cards;

    uint32_t bitmap_idx = 0;
    uint32_t mixed_idx = 0;
    uint32_t ba_idx = 0;
    uint32_t aa_idx = 0;
    uint32_t copy_bmp_idx = 0;
    uint32_t copy_arr_idx = 0;
    uint32_t copy_run_idx = 0;

    for (auto& wi : work) {
        if (wi.a_idx < 0) {
            ContainerType ct = hb.types[wi.b_idx];
            if (ct == ContainerType::BITMAP) {
                uint32_t out_bmp_slot = static_cast<uint32_t>(
                    bb_work.size() + mixed_work.size()) + copy_bmp_idx;
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::BITMAP);
                out_offsets_vec.push_back(out_bmp_slot * 1024 *
                                          static_cast<uint32_t>(sizeof(uint64_t)));
                out_cards.push_back(hb.cardinalities[wi.b_idx]);
                ++copy_bmp_idx;
            } else if (ct == ContainerType::ARRAY) {
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::ARRAY);
                out_offsets_vec.push_back(ca_offsets[copy_arr_idx] *
                                          static_cast<uint32_t>(sizeof(uint16_t)));
                out_cards.push_back(copy_array_work[copy_arr_idx].cardinality);
                ++copy_arr_idx;
            } else {
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::RUN);
                out_offsets_vec.push_back(cr_offsets[copy_run_idx] *
                                          static_cast<uint32_t>(sizeof(uint16_t)));
                out_cards.push_back(copy_run_work[copy_run_idx].cardinality);
                ++copy_run_idx;
            }
        } else if (wi.b_idx < 0) {
            ContainerType ct = ha.types[wi.a_idx];
            if (ct == ContainerType::BITMAP) {
                uint32_t out_bmp_slot = static_cast<uint32_t>(
                    bb_work.size() + mixed_work.size()) + copy_bmp_idx;
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::BITMAP);
                out_offsets_vec.push_back(out_bmp_slot * 1024 *
                                          static_cast<uint32_t>(sizeof(uint64_t)));
                out_cards.push_back(ha.cardinalities[wi.a_idx]);
                ++copy_bmp_idx;
            } else if (ct == ContainerType::ARRAY) {
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::ARRAY);
                out_offsets_vec.push_back(ca_offsets[copy_arr_idx] *
                                          static_cast<uint32_t>(sizeof(uint16_t)));
                out_cards.push_back(copy_array_work[copy_arr_idx].cardinality);
                ++copy_arr_idx;
            } else {
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::RUN);
                out_offsets_vec.push_back(cr_offsets[copy_run_idx] *
                                          static_cast<uint32_t>(sizeof(uint16_t)));
                out_cards.push_back(copy_run_work[copy_run_idx].cardinality);
                ++copy_run_idx;
            }
        } else {
            ContainerType ta = ha.types[wi.a_idx];
            ContainerType tb = hb.types[wi.b_idx];

            if (ta == ContainerType::BITMAP && tb == ContainerType::BITMAP) {
                uint16_t card = out_bb_cards[bitmap_idx];
                if (card == 0 && op != SetOp::OR) {
                    ++bitmap_idx;
                    continue;
                }
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::BITMAP);
                out_offsets_vec.push_back(bitmap_idx * 1024 *
                                          static_cast<uint32_t>(sizeof(uint64_t)));
                out_cards.push_back(card);
                ++bitmap_idx;
            } else if (op == SetOp::AND &&
                       ((ta == ContainerType::BITMAP && tb == ContainerType::ARRAY) ||
                        (ta == ContainerType::ARRAY && tb == ContainerType::BITMAP))) {
                uint16_t card = out_ba_cards[ba_idx];
                if (card == 0) { ++ba_idx; continue; }
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::ARRAY);
                out_offsets_vec.push_back(ba_offsets[ba_idx] *
                                          static_cast<uint32_t>(sizeof(uint16_t)));
                out_cards.push_back(card);
                ++ba_idx;
            } else if (op == SetOp::AND &&
                       ta == ContainerType::ARRAY && tb == ContainerType::ARRAY) {
                uint16_t card = out_aa_cards[aa_idx];
                if (card == 0) { ++aa_idx; continue; }
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::ARRAY);
                out_offsets_vec.push_back(aa_offsets[aa_idx] *
                                          static_cast<uint32_t>(sizeof(uint16_t)));
                out_cards.push_back(card);
                ++aa_idx;
            } else {
                // Mixed pair (non-AND with non-bitmap×bitmap) — expanded to bitmap
                uint16_t card = out_mixed_cards[mixed_idx];
                if (card == 0 && op != SetOp::OR) {
                    ++mixed_idx;
                    continue;
                }
                uint32_t out_bmp_slot = static_cast<uint32_t>(bb_work.size()) +
                                        mixed_idx;
                out_keys.push_back(wi.key);
                out_types.push_back(ContainerType::BITMAP);
                out_offsets_vec.push_back(out_bmp_slot * 1024 *
                                          static_cast<uint32_t>(sizeof(uint64_t)));
                out_cards.push_back(card);
                ++mixed_idx;
            }
        }
    }

    // Build final GpuRoaring
    GpuRoaring result{};
    result.n_containers = static_cast<uint32_t>(out_keys.size());
    result.universe_size = std::max(pa->universe_size, pb->universe_size);
    result.negated = result_negated;
    result.bitmap_data = d_out_bitmap;
    result.n_bitmap_containers = out_n_bitmap;
    result.array_data = d_out_array;
    result.n_array_containers = out_n_array;
    result.array_pool_bytes =
        static_cast<uint32_t>(total_array_elems * sizeof(uint16_t));
    result.run_data = d_out_run;
    result.n_run_containers = out_n_run;
    result.run_pool_bytes =
        static_cast<uint32_t>(total_run_pairs * 2 * sizeof(uint16_t));

    if (result.n_containers > 0) {
        CUDA_CHECK(cudaMallocAsync(&result.keys,
                              result.n_containers * sizeof(uint16_t), stream));
        CUDA_CHECK(cudaMallocAsync(&result.types,
                              result.n_containers * sizeof(ContainerType), stream));
        CUDA_CHECK(cudaMallocAsync(&result.offsets,
                              result.n_containers * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMallocAsync(&result.cardinalities,
                              result.n_containers * sizeof(uint16_t), stream));

        CUDA_CHECK(cudaMemcpyAsync(result.keys, out_keys.data(),
                                   result.n_containers * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.types, out_types.data(),
                                   result.n_containers * sizeof(ContainerType),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.offsets, out_offsets_vec.data(),
                                   result.n_containers * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(result.cardinalities, out_cards.data(),
                                   result.n_containers * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Free temporary expansion bitmaps
    if (d_temp_a_bitmaps) CUDA_CHECK(cudaFreeAsync(d_temp_a_bitmaps, stream));
    if (d_temp_b_bitmaps) CUDA_CHECK(cudaFreeAsync(d_temp_b_bitmaps, stream));

    return result;
}

// ============================================================================
// Fused multi-AND kernel for all-bitmap inputs
// ============================================================================
//
// Support kernels for a fully-device-resident pipeline. The old
// implementation downloaded every input's key array to host, intersected on
// CPU with std::set_intersection, built the pointer table on host via
// std::lower_bound, and re-uploaded the pointer table. All of that runs on
// device now: a 65536-bit presence bitmap per non-negated input is scattered
// on-device, AND-reduced, popcounted, enumerated into sorted common_keys,
// and fed into a device-side pointer-table builder.

// Scatter: for each key k in keys[], set bit k in a 65536-bit presence
// bitmap (1024 × uint64_t). Contention is low because each input has at most
// a few thousand distinct keys.
__global__ void presence_set_bits_kernel(const uint16_t* keys,
                                          uint32_t n,
                                          uint64_t* presence_1024)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t k = keys[i];
    atomicOr(reinterpret_cast<unsigned long long*>(&presence_1024[k / 64u]),
             1ULL << (k % 64u));
}

// AND-reduce `n_non_negated` presence bitmaps (each 1024 words, laid out
// contiguously in `presences_nn`) into `result_1024`. If `n_non_negated == 0`
// (all inputs are negated), fall back to a universe mask: bits
// [0, max_key_plus_1) set, so every in-universe key survives the AND.
__global__ void presence_and_reduce_kernel(const uint64_t* presences_nn,
                                            uint32_t n_non_negated,
                                            uint32_t max_key_plus_1,
                                            uint64_t* result_1024)
{
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= 1024u) return;

    if (n_non_negated > 0) {
        uint64_t acc = ~0ULL;
        for (uint32_t i = 0; i < n_non_negated; ++i) {
            acc &= presences_nn[static_cast<size_t>(i) * 1024u + w];
        }
        result_1024[w] = acc;
    } else {
        uint32_t lo = w * 64u;
        if (lo >= max_key_plus_1) {
            result_1024[w] = 0ULL;
        } else if (lo + 64u <= max_key_plus_1) {
            result_1024[w] = ~0ULL;
        } else {
            uint32_t bits = max_key_plus_1 - lo;
            result_1024[w] = (bits >= 64u) ? ~0ULL : ((1ULL << bits) - 1ULL);
        }
    }
}

// Popcount a 1024-word presence bitmap into *out_count. Caller must zero
// *out_count first. 1 thread per word, 1024 atomicAdds on a single location —
// a few microseconds on modern GPUs.
__global__ void presence_popcount_kernel(const uint64_t* presence_1024,
                                          uint32_t* out_count)
{
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w < 1024u) {
        atomicAdd(out_count, __popcll(presence_1024[w]));
    }
}

// Block-cooperative enumeration: read a 1024-word presence bitmap, emit set
// bit indices as sorted uint16_t values into out_keys. Single block, 256
// threads, each handles 4 consecutive words. Per-thread popcount → warp scan
// → block scan gives each thread its write offset, so set bits are emitted
// in order without a post-sort.
__global__ void enumerate_presence_keys_kernel(const uint64_t* presence_1024,
                                                 uint16_t* out_keys)
{
    constexpr int WORDS_PER_THREAD = 4;
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid & 31u;
    uint32_t warp = tid >> 5;

    uint64_t my_words[WORDS_PER_THREAD];
    uint32_t local_count = 0;
    #pragma unroll
    for (int i = 0; i < WORDS_PER_THREAD; ++i) {
        my_words[i] = presence_1024[tid * WORDS_PER_THREAD + i];
        local_count += __popcll(my_words[i]);
    }

    // Warp inclusive scan of local_count.
    uint32_t v = local_count;
    for (int off = 1; off < 32; off <<= 1) {
        uint32_t up = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= static_cast<uint32_t>(off)) v += up;
    }
    __shared__ uint32_t warp_totals[8];
    if (lane == 31u) warp_totals[warp] = v;
    __syncthreads();

    // Warp 0 scans the 8 warp totals (inclusive).
    if (warp == 0) {
        uint32_t wt = (lane < 8u) ? warp_totals[lane] : 0u;
        for (int off = 1; off < 8; off <<= 1) {
            uint32_t up = __shfl_up_sync(0xffffffffu, wt, off);
            if (lane >= static_cast<uint32_t>(off)) wt += up;
        }
        if (lane < 8u) warp_totals[lane] = wt;
    }
    __syncthreads();

    // My exclusive offset across the block.
    uint32_t excl_in_warp = v - local_count;
    uint32_t warp_prefix  = (warp > 0) ? warp_totals[warp - 1] : 0u;
    uint32_t write_pos    = warp_prefix + excl_in_warp;

    // Emit set bits in sorted order.
    #pragma unroll
    for (int i = 0; i < WORDS_PER_THREAD; ++i) {
        uint64_t w = my_words[i];
        uint32_t base_bit = (tid * WORDS_PER_THREAD + i) * 64u;
        while (w != 0ULL) {
            int lsb = __ffsll(static_cast<long long>(w)) - 1;
            out_keys[write_pos++] = static_cast<uint16_t>(base_bit + lsb);
            w &= (w - 1ULL);
        }
    }
}

// Packed per-input descriptor used by the pointer-table builder kernel.
struct FusedInputMeta {
    const uint16_t* keys;         // sorted device keys (may be nullptr iff n==0)
    const uint64_t* bitmap_data;  // bitmap pool, 1024 words per container
    const uint16_t* key_index;    // direct-map index, or nullptr for fallback
    uint32_t        n_containers;
    uint32_t        max_key_plus_1;  // size of key_index (when present)
};

// For every (common_key, input) pair, resolve the bitmap pointer:
//   - prefer key_index[key] in O(1) when present and in range
//   - otherwise binary-search the input's sorted keys
//   - if the key is absent, point to `sentinel` (an all-zeros scratch
//     buffer; negated inputs use ~0 → all-ones, non-negated inputs never
//     reach this branch because their keys pruned the intersection)
__global__ void build_pointer_table_kernel(const uint16_t* common_keys,
                                             uint32_t n_common,
                                             const FusedInputMeta* inputs,
                                             uint32_t count,
                                             const uint64_t* sentinel,
                                             const uint64_t** out_ptrs)
{
    uint32_t k = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t i = threadIdx.x;
    if (k >= n_common || i >= count) return;

    uint16_t key = common_keys[k];
    const FusedInputMeta meta = inputs[i];

    uint32_t idx = 0xFFFFFFFFu;

    if (meta.n_containers > 0 && meta.keys != nullptr) {
        if (meta.key_index != nullptr && key < meta.max_key_plus_1) {
            uint16_t v = meta.key_index[key];
            if (v != 0xFFFFu) idx = v;
        } else {
            // Fallback: binary search on sorted keys.
            uint32_t lo = 0, hi = meta.n_containers;
            while (lo < hi) {
                uint32_t mid = (lo + hi) >> 1;
                uint16_t mk = meta.keys[mid];
                if (mk < key)      lo = mid + 1;
                else if (mk > key) hi = mid;
                else               { idx = mid; break; }
            }
        }
    }

    const uint64_t* ptr;
    if (idx == 0xFFFFFFFFu) {
        ptr = sentinel;
    } else {
        ptr = meta.bitmap_data + static_cast<size_t>(idx) * 1024u;
    }
    out_ptrs[static_cast<size_t>(k) * count + i] = ptr;
}

// Fill output metadata arrays for the fused result:
//   - types[i] = BITMAP (uint8_t == 1), done via cudaMemsetAsync by caller
//   - offsets[i] = i * 8192 (bytes)
//   - key_index built the same way as promote's helper
__global__ void fused_fill_offsets_kernel(uint32_t* offsets, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    offsets[i] = static_cast<uint32_t>(static_cast<size_t>(i) * 1024u * sizeof(uint64_t));
}

__global__ void fused_build_key_index_kernel(const uint16_t* common_keys,
                                              uint32_t n_common,
                                              uint16_t* key_index)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_common) return;
    key_index[common_keys[i]] = static_cast<uint16_t>(i);
}

// Each block handles one output container. For each common key,
// AND the 1024 words across all N input bitmaps.
// bitmap_ptrs[i] points to the start of input i's bitmap for this key.
// For negated inputs missing a key, the pointer points to a static all-zeros
// sentinel (absent key in negated bitmap = all ones after ~, which is the
// identity element for AND).
// negation_mask: bit i set = input i is negated (apply ~word before AND).
__global__ void fused_multi_and_kernel(
    const uint64_t* const* bitmap_ptrs,  // [n_common_keys][n_inputs]
    uint64_t* output,                     // [n_common_keys * 1024]
    uint16_t* out_cardinalities,          // [n_common_keys]
    uint32_t* out_total,                  // [1], atomicAdd target, may be null
    uint32_t n_inputs,
    uint32_t n_common_keys,
    uint32_t negation_mask)
{
    uint32_t key_idx = blockIdx.x;
    if (key_idx >= n_common_keys) return;

    const uint64_t* const* ptrs = bitmap_ptrs + key_idx * n_inputs;
    uint64_t* dst = output + static_cast<size_t>(key_idx) * 1024;

    uint32_t local_pop = 0;
    for (uint32_t w = threadIdx.x; w < 1024u; w += blockDim.x) {
        uint64_t val = ~0ULL;  // identity for AND
        for (uint32_t i = 0; i < n_inputs; ++i) {
            uint64_t word = ptrs[i][w];
            if ((negation_mask >> i) & 1) word = ~word;
            val &= word;
        }
        dst[w] = val;
        local_pop += __popcll(val);
    }

    // Block-wide reduction of local_pop → single value at thread 0.
    // 256 threads / 32 lanes = 8 warps.
    for (int off = 16; off > 0; off >>= 1) {
        local_pop += __shfl_xor_sync(0xffffffffu, local_pop, off);
    }
    __shared__ uint32_t warp_sums[8];
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = local_pop;
    __syncthreads();
    if (warp == 0) {
        uint32_t v = (lane < 8) ? warp_sums[lane] : 0u;
        for (int off = 4; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) {
            // A full container (65536 bits) can't be stored as uint16_t;
            // clamp to 0xFFFF. In practice the intersection of N inputs
            // is rarely fully dense, so this is defensive.
            uint32_t clamped = v > 65535u ? 65535u : v;
            out_cardinalities[key_idx] = static_cast<uint16_t>(clamped);
            if (out_total) atomicAdd(out_total, v);
        }
    }
}

// Fused multi-AND for all-bitmap inputs: one kernel launch instead of
// N-1 pairwise set_operation calls.
//
// Handles negated inputs natively: negated inputs contribute ~word to the AND,
// and absent keys in negated inputs contribute all-ones (no effect on AND).
// Only non-negated inputs can eliminate keys from the result.
//
// Fully device-resident as of April 2026: the key intersection, pointer
// table, and per-container metadata are all produced by device kernels.
// The only host↔device traffic on this path is:
//   - one 4-byte D2H to learn n_common (needed for subsequent allocations)
//   - one 4-byte D2H to read the final total_cardinality
//   - one 4-byte H2D of the max-universe scalar used by the all-negated path
// Everything else stays on-device.
static GpuRoaring fused_multi_and_allbitmap(const GpuRoaring* bitmaps,
                                             uint32_t count,
                                             cudaStream_t stream)
{
    // Build negation mask (bit i = input i is negated). Max 32 inputs.
    uint32_t neg_mask = 0;
    for (uint32_t i = 0; i < count && i < 32; ++i) {
        if (bitmaps[i].negated) neg_mask |= (1u << i);
    }

    // Compute result universe and the max_key_plus_1 needed for the
    // all-negated presence fallback. These are derived from struct fields
    // the caller already holds in host memory — no D2H required.
    uint32_t result_universe = bitmaps[0].universe_size;
    for (uint32_t i = 1; i < count; ++i) {
        result_universe = std::max(result_universe, bitmaps[i].universe_size);
    }
    uint32_t max_key_plus_1 = (result_universe + 65535u) / 65536u;
    if (max_key_plus_1 > 65536u) max_key_plus_1 = 65536u;

    GpuRoaring result{};
    result.universe_size       = result_universe;
    result.n_array_containers  = 0;
    result.n_run_containers    = 0;

    // ------------------------------------------------------------------
    // 1. Build per-input presence bitmaps (non-negated inputs only) on
    //    device, then AND-reduce them into a single 1024-word result.
    //    If every input is negated, the all-negated fallback writes the
    //    universe mask directly into the result bitmap.
    // ------------------------------------------------------------------
    uint32_t n_non_negated = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!bitmaps[i].negated) ++n_non_negated;
    }

    uint64_t* d_presences_nn = nullptr;  // [n_non_negated * 1024]
    if (n_non_negated > 0) {
        CUDA_CHECK(cudaMallocAsync(
            &d_presences_nn,
            static_cast<size_t>(n_non_negated) * 1024u * sizeof(uint64_t),
            stream));
        CUDA_CHECK(cudaMemsetAsync(
            d_presences_nn, 0,
            static_cast<size_t>(n_non_negated) * 1024u * sizeof(uint64_t),
            stream));

        uint32_t slot = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (bitmaps[i].negated) continue;
            uint32_t n = bitmaps[i].n_containers;
            if (n > 0) {
                uint32_t block = 256;
                uint32_t grid  = (n + block - 1) / block;
                presence_set_bits_kernel<<<grid, block, 0, stream>>>(
                    bitmaps[i].keys, n,
                    d_presences_nn + static_cast<size_t>(slot) * 1024u);
            }
            ++slot;
        }
    }

    uint64_t* d_presence_result = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_presence_result,
                                1024u * sizeof(uint64_t), stream));
    presence_and_reduce_kernel<<<4, 256, 0, stream>>>(
        d_presences_nn, n_non_negated, max_key_plus_1, d_presence_result);
    if (d_presences_nn) CUDA_CHECK(cudaFreeAsync(d_presences_nn, stream));

    // ------------------------------------------------------------------
    // 2. Popcount the presence bitmap → n_common. This is the single
    //    D2H we can't avoid because n_common determines the size of the
    //    allocations below.
    // ------------------------------------------------------------------
    uint32_t* d_n_common = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_n_common, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_n_common, 0, sizeof(uint32_t), stream));
    presence_popcount_kernel<<<4, 256, 0, stream>>>(
        d_presence_result, d_n_common);

    uint32_t h_n_common = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_n_common, d_n_common, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_n_common, stream));

    uint32_t n_common = h_n_common;
    result.n_containers        = n_common;
    result.n_bitmap_containers = n_common;

    if (n_common == 0) {
        CUDA_CHECK(cudaFreeAsync(d_presence_result, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }

    // ------------------------------------------------------------------
    // 3. Enumerate the set bits of the presence bitmap into a sorted
    //    common_keys array on device. Single block, 256 threads.
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMallocAsync(&result.keys, n_common * sizeof(uint16_t), stream));
    enumerate_presence_keys_kernel<<<1, 256, 0, stream>>>(
        d_presence_result, result.keys);
    CUDA_CHECK(cudaFreeAsync(d_presence_result, stream));

    // ------------------------------------------------------------------
    // 4. Build the device-side pointer table. One entry per
    //    (common_key, input) pair. Uses key_index when available,
    //    binary search otherwise, sentinel for absent-in-negated.
    // ------------------------------------------------------------------
    uint64_t* d_sentinel = nullptr;
    bool need_sentinel = (neg_mask != 0);
    if (need_sentinel) {
        CUDA_CHECK(cudaMallocAsync(&d_sentinel, 1024u * sizeof(uint64_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_sentinel, 0, 1024u * sizeof(uint64_t), stream));
    }

    // Upload per-input metadata (count × sizeof(FusedInputMeta) bytes —
    // at most ~1 KB, builds once per call).
    std::vector<FusedInputMeta> h_input_meta(count);
    for (uint32_t i = 0; i < count; ++i) {
        h_input_meta[i].keys         = bitmaps[i].keys;
        h_input_meta[i].bitmap_data  = bitmaps[i].bitmap_data;
        h_input_meta[i].key_index    = bitmaps[i].key_index;
        h_input_meta[i].n_containers = bitmaps[i].n_containers;
        h_input_meta[i].max_key_plus_1 =
            (bitmaps[i].key_index != nullptr)
                ? static_cast<uint32_t>(bitmaps[i].max_key) + 1u
                : 0u;
    }
    FusedInputMeta* d_input_meta = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_input_meta,
                                count * sizeof(FusedInputMeta), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input_meta, h_input_meta.data(),
                                count * sizeof(FusedInputMeta),
                                cudaMemcpyHostToDevice, stream));

    const uint64_t** d_ptrs = nullptr;
    CUDA_CHECK(cudaMallocAsync(
        &d_ptrs,
        static_cast<size_t>(n_common) * count * sizeof(const uint64_t*),
        stream));

    {
        // 2D block: x = input index (<= 32), y = common_keys chunk.
        dim3 block(count, 256u / count > 0 ? 256u / count : 1u);
        if (block.y < 1) block.y = 1;
        dim3 grid((n_common + block.y - 1) / block.y);
        build_pointer_table_kernel<<<grid, block, 0, stream>>>(
            result.keys, n_common, d_input_meta, count,
            d_sentinel, d_ptrs);
    }
    CUDA_CHECK(cudaFreeAsync(d_input_meta, stream));

    // ------------------------------------------------------------------
    // 5. Launch the fused AND kernel. It writes bitmap_data, the per-
    //    container cardinalities, and atomicAdds into a single total
    //    counter copied back at the end.
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMallocAsync(&result.bitmap_data,
                          static_cast<size_t>(n_common) * 1024 * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMallocAsync(&result.cardinalities, n_common * sizeof(uint16_t), stream));

    uint32_t* d_total = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_total, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_total, 0, sizeof(uint32_t), stream));

    fused_multi_and_kernel<<<n_common, 256, 0, stream>>>(
        d_ptrs, result.bitmap_data, result.cardinalities, d_total,
        count, n_common, neg_mask);

    CUDA_CHECK(cudaFreeAsync(d_ptrs, stream));
    if (d_sentinel) CUDA_CHECK(cudaFreeAsync(d_sentinel, stream));

    // ------------------------------------------------------------------
    // 6. Produce the remaining metadata arrays on-device:
    //      - types:   memset to BITMAP (enum value 1)
    //      - offsets: sequence kernel
    //      - key_index: memset 0xFF + scatter kernel
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMallocAsync(&result.types,   n_common * sizeof(ContainerType), stream));
    CUDA_CHECK(cudaMallocAsync(&result.offsets, n_common * sizeof(uint32_t),       stream));

    static_assert(sizeof(ContainerType) == 1,
                  "ContainerType must be one byte for memset fill to work");
    CUDA_CHECK(cudaMemsetAsync(result.types, static_cast<int>(ContainerType::BITMAP),
                                n_common * sizeof(ContainerType), stream));

    {
        uint32_t block = 256;
        uint32_t grid  = (n_common + block - 1) / block;
        fused_fill_offsets_kernel<<<grid, block, 0, stream>>>(
            result.offsets, n_common);
    }

    // key_index: max_key is derived from the *result universe*, not from a
    // D2H of common_keys. The scatter kernel reads common_keys (which live
    // on device in result.keys) and writes their positions directly.
    result.max_key = static_cast<uint16_t>(
        max_key_plus_1 == 0 ? 0u : (max_key_plus_1 - 1u));
    size_t key_index_bytes =
        (static_cast<size_t>(result.max_key) + 1u) * sizeof(uint16_t);
    CUDA_CHECK(cudaMallocAsync(&result.key_index, key_index_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(result.key_index, 0xFF, key_index_bytes, stream));
    {
        uint32_t block = 256;
        uint32_t grid  = (n_common + block - 1) / block;
        fused_build_key_index_kernel<<<grid, block, 0, stream>>>(
            result.keys, n_common, result.key_index);
    }

    // ------------------------------------------------------------------
    // 7. Single final D2H: the total cardinality.
    // ------------------------------------------------------------------
    uint32_t h_total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_total, d_total, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_total, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    result.total_cardinality = h_total;
    return result;
}

// ============================================================================
// Multi-AND / Multi-OR
// ============================================================================
GpuRoaring multi_and(const GpuRoaring* bitmaps, uint32_t count,
                     cudaStream_t stream)
{
    if (count == 0) return GpuRoaring{};
    if (count == 1) {
        return set_operation(bitmaps[0], bitmaps[0], SetOp::AND, stream);
    }

    // Fast path: if all inputs are all-bitmap, use fused kernel.
    // The fused kernel handles negated inputs natively via per-input ~word
    // and an all-zeros sentinel for absent keys — no pairwise fallback needed.
    bool all_bitmap = true;
    for (uint32_t i = 0; i < count; ++i) {
        if (bitmaps[i].n_array_containers > 0 || bitmaps[i].n_run_containers > 0) {
            all_bitmap = false;
            break;
        }
    }
    if (all_bitmap && count <= 32) {
        return fused_multi_and_allbitmap(bitmaps, count, stream);
    }

    // General path: pairwise chain
    GpuRoaring result = set_operation(bitmaps[0], bitmaps[1], SetOp::AND, stream);
    for (uint32_t i = 2; i < count; ++i) {
        GpuRoaring next = set_operation(result, bitmaps[i], SetOp::AND, stream);
        gpu_roaring_free_async(result, stream);
        result = next;
    }
    return result;
}

GpuRoaring multi_or(const GpuRoaring* bitmaps, uint32_t count,
                    cudaStream_t stream)
{
    if (count == 0) return GpuRoaring{};
    if (count == 1) {
        return set_operation(bitmaps[0], bitmaps[0], SetOp::AND, stream);
    }

    GpuRoaring result = set_operation(bitmaps[0], bitmaps[1], SetOp::OR, stream);
    for (uint32_t i = 2; i < count; ++i) {
        GpuRoaring next = set_operation(result, bitmaps[i], SetOp::OR, stream);
        gpu_roaring_free_async(result, stream);
        result = next;
    }
    return result;
}

}  // namespace cu_roaring
