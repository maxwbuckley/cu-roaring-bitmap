/*
 * B7: Optimized Point Query Benchmark
 *
 * Implements and measures three optimizations to Roaring GPU point queries:
 *
 * Opt 1: Direct-map key index — replace O(log n) binary search over sorted
 *        keys[] with O(1) table lookup. key_index[key] = container index.
 *        Cost: (max_key+1) * 2 bytes. At 1B universe: 30 KB.
 *
 * Opt 2: Packed AoS metadata — pack (type, cardinality, offset) into a
 *        single 8-byte struct per container. One cache line instead of 3
 *        separate array reads.
 *
 * Opt 3: Eliminated array containers — re-encode all array containers as
 *        bitmap containers at upload time. Trades memory for query speed
 *        by removing the in-container binary search.
 *
 * Outputs JSON to results/raw/bench7_optimized_query.json
 */

#include <cuda_runtime.h>
#include <roaring/roaring.h>
#include "cu_roaring/cu_roaring.cuh"
#include "cu_roaring/device/make_view.cuh"
#include "cu_roaring/device/roaring_view.cuh"
#include "cu_roaring/device/roaring_warp_query.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

// ============================================================================
// Optimization 1+2: Direct-map key index + Packed AoS metadata
// ============================================================================

// Packed per-container metadata: 8 bytes, naturally aligned
struct __align__(8) ContainerMeta {
    uint32_t offset;        // byte offset into data pool
    uint16_t cardinality;   // element count
    uint8_t  type;          // 0=ARRAY, 1=BITMAP, 2=RUN
    uint8_t  padding;
};

// Optimized device view with O(1) key lookup + packed metadata
struct GpuRoaringViewOpt {
    // Direct-map: key_index[high16] = container index, or 0xFFFF if absent
    // This replaces the entire binary search with a single global read.
    const uint16_t* key_index;       // [max_key + 1] entries
    uint32_t        max_key;         // highest valid key

    // Packed metadata (replaces separate types[], offsets[], cardinalities[])
    const ContainerMeta* meta;       // [n_containers]

    // Data pools (unchanged)
    const uint64_t* bitmap_data;
    const uint16_t* array_data;
    const uint16_t* run_data;

    __device__ __forceinline__ bool contains(uint32_t id) const
    {
        const uint16_t key = static_cast<uint16_t>(id >> 16);
        const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);

        // O(1) key lookup — single global read
        if (key > max_key) return false;
        uint16_t idx = key_index[key];
        if (idx == 0xFFFF) return false;

        // Single struct read — 8 bytes, one cache line
        ContainerMeta m = meta[idx];

        switch (m.type) {
            case 1:  // BITMAP
                return bitmap_contains(m.offset, low);
            case 0:  // ARRAY
                return array_contains(m.offset, m.cardinality, low);
            case 2:  // RUN
                return run_contains(m.offset, m.cardinality, low);
            default: return false;
        }
    }

    __device__ __forceinline__ bool bitmap_contains(uint32_t byte_offset,
                                                     uint16_t low) const
    {
        uint64_t word = bitmap_data[byte_offset / sizeof(uint64_t) + (low >> 6)];
        return (word >> (low & 63)) & 1;
    }

    __device__ __forceinline__ bool array_contains(uint32_t byte_offset,
                                                    uint16_t card,
                                                    uint16_t low) const
    {
        const uint16_t* arr = array_data + byte_offset / sizeof(uint16_t);
        int lo = 0, hi = static_cast<int>(card) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint16_t val = arr[mid];
            if (val == low) return true;
            if (val < low) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }

    __device__ __forceinline__ bool run_contains(uint32_t byte_offset,
                                                  uint16_t n_runs,
                                                  uint16_t low) const
    {
        const uint16_t* runs = run_data + byte_offset / sizeof(uint16_t);
        int lo = 0, hi = static_cast<int>(n_runs) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint16_t start = runs[mid * 2];
            uint16_t len = runs[mid * 2 + 1];
            if (low < start) hi = mid - 1;
            else if (low > start + len) lo = mid + 1;
            else return true;
        }
        return false;
    }
};

// ============================================================================
// Optimization 3: All-bitmap view (no array containers)
// Same as Opt1+2 but array containers have been promoted to bitmap at build time
// ============================================================================
struct GpuRoaringViewAllBitmap {
    const uint16_t* key_index;
    uint32_t        max_key;
    const ContainerMeta* meta;
    const uint64_t* bitmap_data;
    // No array_data or run_data needed — everything is bitmap

    __device__ __forceinline__ bool contains(uint32_t id) const
    {
        const uint16_t key = static_cast<uint16_t>(id >> 16);
        const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);

        if (key > max_key) return false;
        uint16_t idx = key_index[key];
        if (idx == 0xFFFF) return false;

        // We know everything is bitmap — no switch needed
        uint32_t byte_offset = meta[idx].offset;
        uint64_t word = bitmap_data[byte_offset / sizeof(uint64_t) + (low >> 6)];
        return (word >> (low & 63)) & 1;
    }
};


// ============================================================================
// Warp-cooperative variants
// ============================================================================
__device__ __forceinline__ bool warp_contains_opt(const GpuRoaringViewOpt& r,
                                                   uint32_t id)
{
    const uint16_t key = static_cast<uint16_t>(id >> 16);
    const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);
    const unsigned active_mask = __activemask();

#if __CUDA_ARCH__ >= 700
    unsigned match_mask = __match_any_sync(active_mask, static_cast<unsigned>(key));
    unsigned leader_lane = __ffs(match_mask) - 1;
#else
    unsigned leader_lane = threadIdx.x & 31;
#endif

    bool is_leader = ((threadIdx.x & 31) == leader_lane);

    int container_idx = -1;
    ContainerMeta m{};
    if (is_leader) {
        if (key > r.max_key) {
            container_idx = -1;
        } else {
            uint16_t idx = r.key_index[key];
            container_idx = (idx == 0xFFFF) ? -1 : static_cast<int>(idx);
        }
        if (container_idx >= 0) {
            m = r.meta[container_idx];
        }
    }

    container_idx = __shfl_sync(active_mask, container_idx, leader_lane);
    if (container_idx < 0) return false;

    // Broadcast metadata from leader
    m.offset      = __shfl_sync(active_mask, m.offset, leader_lane);
    m.cardinality = __shfl_sync(active_mask, m.cardinality, leader_lane);
    m.type        = __shfl_sync(active_mask, m.type, leader_lane);

    switch (m.type) {
        case 1:  return r.bitmap_contains(m.offset, low);
        case 0:  return r.array_contains(m.offset, m.cardinality, low);
        case 2:  return r.run_contains(m.offset, m.cardinality, low);
        default: return false;
    }
}

__device__ __forceinline__ bool warp_contains_allbitmap(
    const GpuRoaringViewAllBitmap& r, uint32_t id)
{
    const uint16_t key = static_cast<uint16_t>(id >> 16);
    const uint16_t low = static_cast<uint16_t>(id & 0xFFFF);
    const unsigned active_mask = __activemask();

#if __CUDA_ARCH__ >= 700
    unsigned match_mask = __match_any_sync(active_mask, static_cast<unsigned>(key));
    unsigned leader_lane = __ffs(match_mask) - 1;
#else
    unsigned leader_lane = threadIdx.x & 31;
#endif

    bool is_leader = ((threadIdx.x & 31) == leader_lane);

    uint32_t byte_offset = 0xFFFFFFFF;
    if (is_leader) {
        if (key <= r.max_key) {
            uint16_t idx = r.key_index[key];
            if (idx != 0xFFFF) {
                byte_offset = r.meta[idx].offset;
            }
        }
    }

    byte_offset = __shfl_sync(active_mask, byte_offset, leader_lane);
    if (byte_offset == 0xFFFFFFFF) return false;

    // Every thread does its own bitmap word read
    uint64_t word = r.bitmap_data[byte_offset / sizeof(uint64_t) + (low >> 6)];
    return (word >> (low & 63)) & 1;
}


// ============================================================================
// Host-side builders
// ============================================================================

struct OptView {
    GpuRoaringViewOpt view;
    uint16_t* d_key_index;
    ContainerMeta* d_meta;
    size_t total_bytes;  // extra memory used by the optimized index
};

static OptView build_opt_view(const cu_roaring::GpuRoaring& g)
{
    OptView result{};

    // Download keys from device
    std::vector<uint16_t> h_keys(g.n_containers);
    cudaMemcpy(h_keys.data(), g.keys, g.n_containers * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    std::vector<cu_roaring::ContainerType> h_types(g.n_containers);
    cudaMemcpy(h_types.data(), g.types, g.n_containers * sizeof(cu_roaring::ContainerType),
               cudaMemcpyDeviceToHost);

    std::vector<uint32_t> h_offsets(g.n_containers);
    cudaMemcpy(h_offsets.data(), g.offsets, g.n_containers * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    std::vector<uint16_t> h_cards(g.n_containers);
    cudaMemcpy(h_cards.data(), g.cardinalities, g.n_containers * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    // Build direct-map key index
    uint16_t max_key = h_keys.empty() ? 0 : h_keys.back();
    std::vector<uint16_t> h_key_index(max_key + 1, 0xFFFF);
    for (uint32_t i = 0; i < g.n_containers; ++i) {
        h_key_index[h_keys[i]] = static_cast<uint16_t>(i);
    }

    // Build packed metadata
    std::vector<ContainerMeta> h_meta(g.n_containers);
    for (uint32_t i = 0; i < g.n_containers; ++i) {
        h_meta[i].offset      = h_offsets[i];
        h_meta[i].cardinality = h_cards[i];
        h_meta[i].type        = static_cast<uint8_t>(h_types[i]);
        h_meta[i].padding     = 0;
    }

    // Upload
    size_t key_index_bytes = (max_key + 1) * sizeof(uint16_t);
    size_t meta_bytes = g.n_containers * sizeof(ContainerMeta);
    cudaMalloc(&result.d_key_index, key_index_bytes);
    cudaMalloc(&result.d_meta, meta_bytes);
    cudaMemcpy(result.d_key_index, h_key_index.data(), key_index_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_meta, h_meta.data(), meta_bytes,
               cudaMemcpyHostToDevice);

    result.view.key_index   = result.d_key_index;
    result.view.max_key     = max_key;
    result.view.meta        = result.d_meta;
    result.view.bitmap_data = g.bitmap_data;
    result.view.array_data  = g.array_data;
    result.view.run_data    = g.run_data;
    result.total_bytes      = key_index_bytes + meta_bytes;

    return result;
}

static void free_opt_view(OptView& v)
{
    cudaFree(v.d_key_index);
    cudaFree(v.d_meta);
}


// All-bitmap builder: re-encode array containers as bitmaps
struct AllBitmapView {
    GpuRoaringViewAllBitmap view;
    uint16_t* d_key_index;
    ContainerMeta* d_meta;
    uint64_t* d_bitmap_data;  // new expanded bitmap pool
    size_t total_bytes;
};

static AllBitmapView build_allbitmap_view(const cu_roaring::GpuRoaring& g)
{
    AllBitmapView result{};

    // Download container metadata
    std::vector<uint16_t> h_keys(g.n_containers);
    cudaMemcpy(h_keys.data(), g.keys, g.n_containers * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    std::vector<cu_roaring::ContainerType> h_types(g.n_containers);
    cudaMemcpy(h_types.data(), g.types, g.n_containers * sizeof(cu_roaring::ContainerType),
               cudaMemcpyDeviceToHost);

    std::vector<uint32_t> h_offsets(g.n_containers);
    cudaMemcpy(h_offsets.data(), g.offsets, g.n_containers * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    std::vector<uint16_t> h_cards(g.n_containers);
    cudaMemcpy(h_cards.data(), g.cardinalities, g.n_containers * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    // Download existing data pools if needed
    std::vector<uint64_t> h_bitmap_data;
    if (g.n_bitmap_containers > 0) {
        h_bitmap_data.resize(g.n_bitmap_containers * 1024);
        cudaMemcpy(h_bitmap_data.data(), g.bitmap_data,
                   h_bitmap_data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    std::vector<uint16_t> h_array_data;
    if (g.n_array_containers > 0) {
        // We need to figure out total array elements. Sum cardinalities of array containers.
        uint32_t total_arr_elems = 0;
        for (uint32_t i = 0; i < g.n_containers; ++i) {
            if (h_types[i] == cu_roaring::ContainerType::ARRAY) {
                total_arr_elems += h_cards[i];
            }
        }
        h_array_data.resize(total_arr_elems);
        cudaMemcpy(h_array_data.data(), g.array_data,
                   total_arr_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    std::vector<uint16_t> h_run_data;
    if (g.n_run_containers > 0) {
        uint32_t total_run_pairs = 0;
        for (uint32_t i = 0; i < g.n_containers; ++i) {
            if (h_types[i] == cu_roaring::ContainerType::RUN) {
                total_run_pairs += h_cards[i];
            }
        }
        h_run_data.resize(total_run_pairs * 2);
        cudaMemcpy(h_run_data.data(), g.run_data,
                   h_run_data.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }

    // Build all-bitmap pool: every container becomes a 1024-word bitmap
    std::vector<uint64_t> all_bitmap(g.n_containers * 1024, 0);

    for (uint32_t i = 0; i < g.n_containers; ++i) {
        uint64_t* dst = all_bitmap.data() + i * 1024;

        if (h_types[i] == cu_roaring::ContainerType::BITMAP) {
            uint32_t src_idx = h_offsets[i] / sizeof(uint64_t);
            memcpy(dst, h_bitmap_data.data() + src_idx, 1024 * sizeof(uint64_t));
        } else if (h_types[i] == cu_roaring::ContainerType::ARRAY) {
            uint32_t src_idx = h_offsets[i] / sizeof(uint16_t);
            for (uint32_t j = 0; j < h_cards[i]; ++j) {
                uint16_t val = h_array_data[src_idx + j];
                dst[val / 64] |= 1ULL << (val % 64);
            }
        } else if (h_types[i] == cu_roaring::ContainerType::RUN) {
            uint32_t src_idx = h_offsets[i] / sizeof(uint16_t);
            for (uint32_t r = 0; r < h_cards[i]; ++r) {
                uint16_t start = h_run_data[src_idx + r * 2];
                uint16_t len   = h_run_data[src_idx + r * 2 + 1];
                for (uint32_t v = start; v <= static_cast<uint32_t>(start) + len; ++v) {
                    dst[v / 64] |= 1ULL << (v % 64);
                }
            }
        }
    }

    // Build direct-map key index
    uint16_t max_key = h_keys.empty() ? 0 : h_keys.back();
    std::vector<uint16_t> h_key_index(max_key + 1, 0xFFFF);
    for (uint32_t i = 0; i < g.n_containers; ++i) {
        h_key_index[h_keys[i]] = static_cast<uint16_t>(i);
    }

    // Build metadata (all containers are now BITMAP type)
    std::vector<ContainerMeta> h_meta(g.n_containers);
    for (uint32_t i = 0; i < g.n_containers; ++i) {
        h_meta[i].offset      = static_cast<uint32_t>(i * 1024 * sizeof(uint64_t));
        h_meta[i].cardinality = h_cards[i];
        h_meta[i].type        = 1;  // BITMAP
        h_meta[i].padding     = 0;
    }

    // Upload
    size_t key_index_bytes = (max_key + 1) * sizeof(uint16_t);
    size_t meta_bytes = g.n_containers * sizeof(ContainerMeta);
    size_t bitmap_bytes = all_bitmap.size() * sizeof(uint64_t);

    cudaMalloc(&result.d_key_index, key_index_bytes);
    cudaMalloc(&result.d_meta, meta_bytes);
    cudaMalloc(&result.d_bitmap_data, bitmap_bytes);
    cudaMemcpy(result.d_key_index, h_key_index.data(), key_index_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_meta, h_meta.data(), meta_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_bitmap_data, all_bitmap.data(), bitmap_bytes,
               cudaMemcpyHostToDevice);

    result.view.key_index   = result.d_key_index;
    result.view.max_key     = max_key;
    result.view.meta        = result.d_meta;
    result.view.bitmap_data = result.d_bitmap_data;
    result.total_bytes      = key_index_bytes + meta_bytes + bitmap_bytes;

    return result;
}

static void free_allbitmap_view(AllBitmapView& v)
{
    cudaFree(v.d_key_index);
    cudaFree(v.d_meta);
    cudaFree(v.d_bitmap_data);
}


// ============================================================================
// Benchmark kernels
// ============================================================================

__global__ void bitset_query_kernel(const uint32_t* __restrict__ bitset,
                                     const uint32_t* __restrict__ queries,
                                     uint32_t* __restrict__ results,
                                     uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    uint32_t id = queries[idx];
    results[idx] = (bitset[id >> 5] >> (id & 31)) & 1u;
}

__global__ void baseline_contains_kernel(cu_roaring::GpuRoaringView view,
                                          const uint32_t* __restrict__ queries,
                                          uint32_t* __restrict__ results,
                                          uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

__global__ void baseline_warp_kernel(cu_roaring::GpuRoaringView view,
                                      const uint32_t* __restrict__ queries,
                                      uint32_t* __restrict__ results,
                                      uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = cu_roaring::warp_contains(view, queries[idx]) ? 1u : 0u;
}

__global__ void opt_contains_kernel(GpuRoaringViewOpt view,
                                     const uint32_t* __restrict__ queries,
                                     uint32_t* __restrict__ results,
                                     uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

__global__ void opt_warp_kernel(GpuRoaringViewOpt view,
                                 const uint32_t* __restrict__ queries,
                                 uint32_t* __restrict__ results,
                                 uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = warp_contains_opt(view, queries[idx]) ? 1u : 0u;
}

__global__ void allbmp_contains_kernel(GpuRoaringViewAllBitmap view,
                                        const uint32_t* __restrict__ queries,
                                        uint32_t* __restrict__ results,
                                        uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = view.contains(queries[idx]) ? 1u : 0u;
}

__global__ void allbmp_warp_kernel(GpuRoaringViewAllBitmap view,
                                    const uint32_t* __restrict__ queries,
                                    uint32_t* __restrict__ results,
                                    uint32_t n_queries)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_queries) return;
    results[idx] = warp_contains_allbitmap(view, queries[idx]) ? 1u : 0u;
}


// ============================================================================
// Utilities
// ============================================================================

static roaring_bitmap_t* make_bitmap(uint32_t universe, double density, uint64_t seed)
{
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 gen(seed);
    if (density >= 0.5) {
        roaring_bitmap_add_range(r, 0, universe);
        double remove_rate = 1.0 - density;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (uint32_t key = 0; key < (universe + 65535) / 65536; ++key) {
            uint32_t base = key * 65536u;
            uint32_t end = std::min(base + 65536u, universe);
            for (uint32_t i = base; i < end; ++i)
                if (dist(gen) < remove_rate) roaring_bitmap_remove(r, i);
        }
    } else {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (uint32_t i = 0; i < universe; ++i)
            if (dist(gen) < density) roaring_bitmap_add(r, i);
    }
    roaring_bitmap_run_optimize(r);
    return r;
}

static uint32_t* make_gpu_bitset(const roaring_bitmap_t* bm, uint32_t universe)
{
    uint32_t n_words = (universe + 31) / 32;
    std::vector<uint32_t> h(n_words, 0);
    roaring_uint32_iterator_t* iter = roaring_iterator_create(bm);
    while (iter->has_value) {
        uint32_t v = iter->current_value;
        h[v / 32] |= (1u << (v % 32));
        roaring_uint32_iterator_advance(iter);
    }
    roaring_uint32_iterator_free(iter);
    uint32_t* d;
    cudaMalloc(&d, n_words * sizeof(uint32_t));
    cudaMemcpy(d, h.data(), n_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    return d;
}

static std::vector<uint32_t> gen_random_queries(uint32_t universe,
                                                 uint32_t n, uint64_t seed)
{
    std::vector<uint32_t> q(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, universe - 1);
    for (uint32_t i = 0; i < n; ++i) q[i] = dist(gen);
    return q;
}

static std::vector<uint32_t> gen_strided_queries(uint32_t universe,
                                                  uint32_t n, uint64_t seed)
{
    std::vector<uint32_t> q(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> anchor_dist(0, universe - 1);
    uint32_t window = std::max(universe / 100u, 65536u);
    for (uint32_t i = 0; i < n; i += 32) {
        uint32_t anchor = anchor_dist(gen);
        uint32_t lo = (anchor > window / 2) ? anchor - window / 2 : 0;
        uint32_t hi = std::min(lo + window, universe - 1);
        std::uniform_int_distribution<uint32_t> nbr(lo, hi);
        for (uint32_t j = 0; j < 32 && (i + j) < n; ++j)
            q[i + j] = nbr(gen);
    }
    return q;
}

struct Stats {
    double median, mean, p5, p95, std_dev, min_val, max_val;
};

static Stats compute_stats(std::vector<double>& times)
{
    std::sort(times.begin(), times.end());
    int n = static_cast<int>(times.size());
    double sum = 0;
    for (auto t : times) sum += t;
    double mean = sum / n;
    double var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    return {times[n / 2], mean,
            times[std::max(0, (int)(n * 0.05))],
            times[std::min(n - 1, (int)(n * 0.95))],
            std::sqrt(var / n), times[0], times[n - 1]};
}

static Stats bench_gpu(int warmup, int iters, std::function<void()> fn)
{
    cudaDeviceSynchronize();
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(s);
        fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        times[i] = ms;
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return compute_stats(times);
}

static uint32_t verify(const uint32_t* a, const uint32_t* b, uint32_t n)
{
    std::vector<uint32_t> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t mm = 0;
    for (uint32_t i = 0; i < n; ++i) if (ha[i] != hb[i]) ++mm;
    return mm;
}

static void write_stats(FILE* f, const char* name, const Stats& s)
{
    fprintf(f, "\"%s\": {\"median\": %.4f, \"mean\": %.4f, \"p5\": %.4f, "
            "\"p95\": %.4f, \"std\": %.4f, \"min\": %.4f, \"max\": %.4f}",
            name, s.median, s.mean, s.p5, s.p95, s.std_dev, s.min_val, s.max_val);
}


// ============================================================================
// Main
// ============================================================================
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.0f MB)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0));

    const char* path = "results/raw/bench7_optimized_query.json";
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    fprintf(f, "{\n  \"benchmark\": \"optimized_query\",\n");
    fprintf(f, "  \"gpu\": \"%s\",\n  \"n_sms\": %d,\n",
            prop.name, prop.multiProcessorCount);
    fprintf(f, "  \"results\": [\n");

    uint32_t universes[] = {1000000, 10000000, 100000000, 1000000000};
    double densities[]   = {0.001, 0.01, 0.10, 0.50};
    const char* dlabels[] = {"0.1%", "1%", "10%", "50%"};

    struct PatternDef {
        const char* name;
        std::vector<uint32_t> (*gen)(uint32_t, uint32_t, uint64_t);
    };
    PatternDef patterns[] = {
        {"random",  gen_random_queries},
        {"strided", gen_strided_queries},
    };

    constexpr uint32_t NQ = 10000000;
    constexpr int W = 10, N = 50;
    bool first = true;

    for (auto U : universes) {
        for (int di = 0; di < 4; ++di) {
            double d = densities[di];
            printf("\n=== U=%uM d=%s ===\n", U / 1000000, dlabels[di]);
            fflush(stdout);

            auto* cpu_bm = make_bitmap(U, d, 42);
            uint64_t card = roaring_bitmap_get_cardinality(cpu_bm);
            printf("  card=%llu (%.2f%%)\n", (unsigned long long)card, 100.0*card/U);

            auto gpu_bm = cu_roaring::upload(cpu_bm);
            printf("  containers=%u (bmp=%u arr=%u run=%u)\n",
                   gpu_bm.n_containers, gpu_bm.n_bitmap_containers,
                   gpu_bm.n_array_containers, gpu_bm.n_run_containers);

            // Build views
            auto base_view = cu_roaring::make_view(gpu_bm);
            auto opt = build_opt_view(gpu_bm);
            auto abm = build_allbitmap_view(gpu_bm);
            auto* d_bitset = make_gpu_bitset(cpu_bm, U);

            size_t bitset_bytes = ((size_t)U + 31) / 32 * sizeof(uint32_t);
            auto meta = cu_roaring::get_meta(cpu_bm);

            printf("  Memory: bitset=%.2fMB base_roaring=%.2fMB opt_extra=%.2fKB allbmp=%.2fMB\n",
                   bitset_bytes / 1e6, meta.total_bytes / 1e6,
                   opt.total_bytes / 1e3, abm.total_bytes / 1e6);

            // Allocate query/result buffers
            uint32_t* d_queries;
            uint32_t* d_res_bitset;
            uint32_t* d_res_base;
            uint32_t* d_res_base_warp;
            uint32_t* d_res_opt;
            uint32_t* d_res_opt_warp;
            uint32_t* d_res_abm;
            uint32_t* d_res_abm_warp;
            cudaMalloc(&d_queries, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_bitset, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_base, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_base_warp, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_opt, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_opt_warp, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_abm, NQ * sizeof(uint32_t));
            cudaMalloc(&d_res_abm_warp, NQ * sizeof(uint32_t));

            dim3 block(256);
            dim3 grid((NQ + 255) / 256);

            for (auto& pat : patterns) {
                printf("  %s:\n", pat.name);
                fflush(stdout);

                auto h_q = pat.gen(U, NQ, 777);
                cudaMemcpy(d_queries, h_q.data(), NQ * sizeof(uint32_t),
                           cudaMemcpyHostToDevice);

                // 1. Flat bitset (baseline)
                auto s_bs = bench_gpu(W, N, [&]() {
                    bitset_query_kernel<<<grid, block>>>(
                        d_bitset, d_queries, d_res_bitset, NQ);
                });

                // 2. Original contains()
                auto s_base = bench_gpu(W, N, [&]() {
                    baseline_contains_kernel<<<grid, block>>>(
                        base_view, d_queries, d_res_base, NQ);
                });

                // 3. Original warp_contains()
                auto s_base_w = bench_gpu(W, N, [&]() {
                    baseline_warp_kernel<<<grid, block>>>(
                        base_view, d_queries, d_res_base_warp, NQ);
                });

                // 4. Optimized contains (direct-map + AoS)
                auto s_opt = bench_gpu(W, N, [&]() {
                    opt_contains_kernel<<<grid, block>>>(
                        opt.view, d_queries, d_res_opt, NQ);
                });

                // 5. Optimized warp
                auto s_opt_w = bench_gpu(W, N, [&]() {
                    opt_warp_kernel<<<grid, block>>>(
                        opt.view, d_queries, d_res_opt_warp, NQ);
                });

                // 6. All-bitmap contains
                auto s_abm = bench_gpu(W, N, [&]() {
                    allbmp_contains_kernel<<<grid, block>>>(
                        abm.view, d_queries, d_res_abm, NQ);
                });

                // 7. All-bitmap warp
                auto s_abm_w = bench_gpu(W, N, [&]() {
                    allbmp_warp_kernel<<<grid, block>>>(
                        abm.view, d_queries, d_res_abm_warp, NQ);
                });

                // Verify correctness
                uint32_t mm1 = verify(d_res_bitset, d_res_base, NQ);
                uint32_t mm2 = verify(d_res_bitset, d_res_opt, NQ);
                uint32_t mm3 = verify(d_res_bitset, d_res_abm, NQ);
                uint32_t mm4 = verify(d_res_bitset, d_res_base_warp, NQ);
                uint32_t mm5 = verify(d_res_bitset, d_res_opt_warp, NQ);
                uint32_t mm6 = verify(d_res_bitset, d_res_abm_warp, NQ);
                bool correct = (mm1 == 0 && mm2 == 0 && mm3 == 0 &&
                                mm4 == 0 && mm5 == 0 && mm6 == 0);
                if (!correct) {
                    printf("  *** MISMATCH: base=%u opt=%u abm=%u bw=%u ow=%u aw=%u ***\n",
                           mm1, mm2, mm3, mm4, mm5, mm6);
                }

                // Print results
                auto pr = [&](const char* name, const Stats& s, double ref_ms) {
                    double gqps = NQ / (s.median * 1e-3) / 1e9;
                    double vs = ref_ms / s.median;
                    printf("    %-20s %7.3f ms  %6.2f Gq/s  %5.2fx vs bitset\n",
                           name, s.median, gqps, vs);
                };
                pr("bitset", s_bs, s_bs.median);
                pr("base_contains", s_base, s_bs.median);
                pr("base_warp", s_base_w, s_bs.median);
                pr("opt_contains", s_opt, s_bs.median);
                pr("opt_warp", s_opt_w, s_bs.median);
                pr("allbmp_contains", s_abm, s_bs.median);
                pr("allbmp_warp", s_abm_w, s_bs.median);

                // Speedup of opt vs base
                printf("    >> opt speedup vs base: contains=%.2fx warp=%.2fx\n",
                       s_base.median / s_opt.median,
                       s_base_w.median / s_opt_w.median);
                printf("    >> allbmp speedup vs base: contains=%.2fx warp=%.2fx\n",
                       s_base.median / s_abm.median,
                       s_base_w.median / s_abm_w.median);

                // JSON output
                if (!first) fprintf(f, ",\n");
                first = false;
                fprintf(f, "    {\n");
                fprintf(f, "      \"universe\": %u, \"density\": %.4f, \"density_label\": \"%s\",\n",
                        U, d, dlabels[di]);
                fprintf(f, "      \"pattern\": \"%s\", \"n_queries\": %u,\n", pat.name, NQ);
                fprintf(f, "      \"n_containers\": %u, \"n_bitmap\": %u, \"n_array\": %u,\n",
                        gpu_bm.n_containers, gpu_bm.n_bitmap_containers, gpu_bm.n_array_containers);
                fprintf(f, "      \"bitset_bytes\": %zu, \"roaring_bytes\": %zu, "
                        "\"opt_extra_bytes\": %zu, \"allbmp_bytes\": %zu,\n",
                        bitset_bytes, meta.total_bytes, opt.total_bytes, abm.total_bytes);
                fprintf(f, "      ");
                write_stats(f, "bitset_ms", s_bs);
                fprintf(f, ",\n      ");
                write_stats(f, "base_contains_ms", s_base);
                fprintf(f, ",\n      ");
                write_stats(f, "base_warp_ms", s_base_w);
                fprintf(f, ",\n      ");
                write_stats(f, "opt_contains_ms", s_opt);
                fprintf(f, ",\n      ");
                write_stats(f, "opt_warp_ms", s_opt_w);
                fprintf(f, ",\n      ");
                write_stats(f, "allbmp_contains_ms", s_abm);
                fprintf(f, ",\n      ");
                write_stats(f, "allbmp_warp_ms", s_abm_w);
                fprintf(f, ",\n");
                fprintf(f, "      \"opt_vs_base_contains\": %.4f, \"opt_vs_base_warp\": %.4f,\n",
                        s_base.median / s_opt.median, s_base_w.median / s_opt_w.median);
                fprintf(f, "      \"allbmp_vs_base_contains\": %.4f, \"allbmp_vs_base_warp\": %.4f,\n",
                        s_base.median / s_abm.median, s_base_w.median / s_abm_w.median);
                fprintf(f, "      \"allbmp_vs_bitset\": %.4f,\n",
                        s_abm.median / s_bs.median);
                fprintf(f, "      \"correctness\": %s\n", correct ? "true" : "false");
                fprintf(f, "    }");
            }

            cudaFree(d_queries);
            cudaFree(d_res_bitset);
            cudaFree(d_res_base);
            cudaFree(d_res_base_warp);
            cudaFree(d_res_opt);
            cudaFree(d_res_opt_warp);
            cudaFree(d_res_abm);
            cudaFree(d_res_abm_warp);
            cudaFree(d_bitset);
            free_opt_view(opt);
            free_allbitmap_view(abm);
            cu_roaring::gpu_roaring_free(gpu_bm);
            roaring_bitmap_free(cpu_bm);
        }
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("\n=== B7 COMPLETE — results at %s ===\n", path);
    return 0;
}
