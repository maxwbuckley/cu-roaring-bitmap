#include "cu_roaring/detail/decompress.cuh"
#include "cu_roaring/detail/utils.cuh"

namespace cu_roaring {

// One block per container, 256 threads per block.
// Each block decompresses one container into the flat bitset output.
__global__ void decompress_kernel(
    const uint16_t*      keys,
    const ContainerType* types,
    const uint32_t*      offsets,
    const uint16_t*      cardinalities,
    uint32_t             n_containers,
    const uint64_t*      bitmap_data,
    const uint16_t*      array_data,
    const uint16_t*      run_data,
    uint32_t*            output,
    uint32_t             output_size_words)
{
    uint32_t cid = blockIdx.x;
    if (cid >= n_containers) return;

    uint32_t key  = keys[cid];
    // Each key covers 65536 IDs = 2048 uint32_t words in the output bitset
    uint32_t base_word = key * 2048u;

    ContainerType ctype = types[cid];
    uint32_t offset     = offsets[cid];

    if (ctype == ContainerType::BITMAP) {
        // Bitmap container: 1024 uint64_t words = 2048 uint32_t words
        // offset is in bytes into bitmap_data
        const uint32_t* src =
            reinterpret_cast<const uint32_t*>(bitmap_data) +
            (offset / sizeof(uint32_t));

        // 256 threads, each copies ceil(2048/256) = 8 words
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint32_t dst_idx = base_word + i;
            if (dst_idx < output_size_words) {
                output[dst_idx] = src[i];
            }
        }
    } else if (ctype == ContainerType::ARRAY) {
        // Array container: scatter-set individual bits
        // offset is in bytes into array_data
        const uint16_t* arr =
            array_data + (offset / sizeof(uint16_t));
        uint16_t card = cardinalities[cid];

        for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
            uint16_t val = arr[i];
            uint32_t abs_bit = (static_cast<uint32_t>(key) << 16) | val;
            uint32_t word_idx = abs_bit / 32u;
            uint32_t bit_pos  = abs_bit % 32u;
            if (word_idx < output_size_words) {
                atomicOr(&output[word_idx], 1u << bit_pos);
            }
        }
    } else if (ctype == ContainerType::RUN) {
        // Run container: each run is (start, length) pair of uint16_t
        // offset is in bytes into run_data
        const uint16_t* runs =
            run_data + (offset / sizeof(uint16_t));
        uint16_t n_runs = cardinalities[cid];

        // Distribute runs across threads
        for (uint32_t r = threadIdx.x; r < n_runs; r += blockDim.x) {
            uint16_t start  = runs[r * 2];
            uint16_t length = runs[r * 2 + 1];

            // Set bits [start, start + length] (inclusive) within this container
            for (uint32_t v = start; v <= static_cast<uint32_t>(start) + length; ++v) {
                uint32_t abs_bit = (static_cast<uint32_t>(key) << 16) | v;
                uint32_t word_idx = abs_bit / 32u;
                uint32_t bit_pos  = abs_bit % 32u;
                if (word_idx < output_size_words) {
                    atomicOr(&output[word_idx], 1u << bit_pos);
                }
            }
        }
    }
}

void decompress_to_bitset(const GpuRoaring& bitmap,
                          uint32_t* output,
                          uint32_t output_size_words,
                          cudaStream_t stream) {
    if (bitmap.n_containers == 0) return;

    // Zero the output buffer
    CUDA_CHECK(cudaMemsetAsync(output, 0,
                               output_size_words * sizeof(uint32_t), stream));

    dim3 grid(bitmap.n_containers);
    dim3 block(256);

    decompress_kernel<<<grid, block, 0, stream>>>(
        bitmap.keys,
        bitmap.types,
        bitmap.offsets,
        bitmap.cardinalities,
        bitmap.n_containers,
        bitmap.bitmap_data,
        bitmap.array_data,
        bitmap.run_data,
        output,
        output_size_words);

    CUDA_CHECK(cudaGetLastError());
}

uint32_t* decompress_to_bitset(const GpuRoaring& bitmap, cudaStream_t stream) {
    uint32_t n_words = div_ceil(bitmap.universe_size, 32u);
    if (n_words == 0) return nullptr;

    uint32_t* output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, n_words * sizeof(uint32_t)));

    decompress_to_bitset(bitmap, output, n_words, stream);
    return output;
}

}  // namespace cu_roaring
