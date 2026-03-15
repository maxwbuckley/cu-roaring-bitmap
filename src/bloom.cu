#include "cu_roaring/types.cuh"
#include "cu_roaring/detail/utils.cuh"

#include <cstring>
#include <vector>

namespace cu_roaring {

void build_key_bloom(GpuRoaring& bitmap, cudaStream_t stream)
{
  constexpr uint32_t BLOOM_WORDS = GpuRoaring::BLOOM_SIZE_WORDS;
  constexpr uint32_t BLOOM_BITS  = BLOOM_WORDS * 32;
  constexpr uint32_t N_HASHES    = 2;

  if (bitmap.n_containers == 0) return;

  // Download keys to host
  std::vector<uint16_t> h_keys(bitmap.n_containers);
  CUDA_CHECK(cudaMemcpyAsync(h_keys.data(), bitmap.keys,
                             bitmap.n_containers * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Build bloom on host
  std::vector<uint32_t> h_bloom(BLOOM_WORDS, 0);
  for (uint32_t i = 0; i < bitmap.n_containers; ++i) {
    uint32_t key = h_keys[i];
    uint32_t h1  = key * 0x9E3779B9u;
    uint32_t h2  = key * 0x517CC1B7u;
    for (uint32_t j = 0; j < N_HASHES; ++j) {
      uint32_t bit = (h1 + j * h2) % BLOOM_BITS;
      h_bloom[bit >> 5] |= (1u << (bit & 31));
    }
  }

  // Upload to device
  if (bitmap.key_bloom) cudaFree(bitmap.key_bloom);
  CUDA_CHECK(cudaMalloc(&bitmap.key_bloom, BLOOM_WORDS * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpyAsync(bitmap.key_bloom, h_bloom.data(),
                             BLOOM_WORDS * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cu_roaring
