add_executable(cu_roaring_v2_bench_vs_bitset bench_vs_bitset.cu)

target_link_libraries(cu_roaring_v2_bench_vs_bitset PRIVATE
    cu_roaring_v2
    roaring
    CUDA::cudart
)

set_target_properties(cu_roaring_v2_bench_vs_bitset PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(COMMAND cu_roaring_target_strict_warnings)
    cu_roaring_target_strict_warnings(cu_roaring_v2_bench_vs_bitset)
endif()
