include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW ON
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

function(add_cu_roaring_bench name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE cu_roaring_bitmap benchmark::benchmark benchmark::benchmark_main)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cu_roaring_target_strict_warnings(${name})
endfunction()

add_cu_roaring_bench(bench_set_ops)
add_cu_roaring_bench(bench_decompress)
add_cu_roaring_bench(bench_transfer)

# Comprehensive benchmark (no google benchmark dependency, standalone)
add_executable(bench_comprehensive bench_comprehensive.cu)
target_link_libraries(bench_comprehensive PRIVATE cu_roaring_bitmap)
set_target_properties(bench_comprehensive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_comprehensive)

# Point query benchmark (standalone, no google benchmark dependency)
add_executable(bench_point_query bench_point_query.cu)
target_link_libraries(bench_point_query PRIVATE cu_roaring_bitmap)
set_target_properties(bench_point_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_point_query)

# Optimized query benchmark (direct-map + AoS + all-bitmap variants)
add_executable(bench_optimized_query bench_optimized_query.cu)
target_link_libraries(bench_optimized_query PRIVATE cu_roaring_bitmap)
set_target_properties(bench_optimized_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_optimized_query)

# Upload scale benchmark (GPU sort vs CPU sort at XL scale)
add_executable(bench_upload_scale bench_upload_scale.cu)
target_link_libraries(bench_upload_scale PRIVATE cu_roaring_bitmap)
set_target_properties(bench_upload_scale PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_upload_scale)

# Multi-predicate AND benchmark (fused vs pairwise)
add_executable(bench_multi_and bench_multi_and.cu)
target_link_libraries(bench_multi_and PRIVATE cu_roaring_bitmap)
set_target_properties(bench_multi_and PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_multi_and)

# YFCC-10M filtered track benchmark
add_executable(bench_yfcc bench_yfcc.cu)
target_link_libraries(bench_yfcc PRIVATE cu_roaring_bitmap)
set_target_properties(bench_yfcc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_yfcc)

# Selectivity sweep benchmark (Roaring vs bitset across 0.1%-99% selectivity)
add_executable(bench_selectivity_sweep bench_selectivity_sweep.cu)
target_link_libraries(bench_selectivity_sweep PRIVATE cu_roaring_bitmap)
set_target_properties(bench_selectivity_sweep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_selectivity_sweep)

# enumerate_ids / CSR export benchmark (standalone)
add_executable(bench_enumerate_ids bench_enumerate_ids.cu)
target_link_libraries(bench_enumerate_ids PRIVATE cu_roaring_bitmap)
set_target_properties(bench_enumerate_ids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_enumerate_ids)

# Cache hierarchy profiling benchmark (standalone, companion to ncu_cache_profile.sh)
add_executable(bench_cache_profile bench_cache_profile.cu)
target_link_libraries(bench_cache_profile PRIVATE cu_roaring_bitmap)
set_target_properties(bench_cache_profile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_cache_profile)

# YFCC sorted vs unsorted benchmark (bitset, unsorted roaring, lex-sorted roaring)
add_executable(bench_yfcc_sorted bench_yfcc_sorted.cu)
target_link_libraries(bench_yfcc_sorted PRIVATE cu_roaring_bitmap)
set_target_properties(bench_yfcc_sorted PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_yfcc_sorted)

# Run container benchmark: contiguous ranges, ideal case for Roaring
add_executable(bench_run_containers bench_run_containers.cu)
target_link_libraries(bench_run_containers PRIVATE cu_roaring_bitmap)
set_target_properties(bench_run_containers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_run_containers)

# From-disk benchmark: load CRoaring .bin files, compare roaring vs bitset
add_executable(bench_from_disk bench_from_disk.cu)
target_link_libraries(bench_from_disk PRIVATE cu_roaring_bitmap)
set_target_properties(bench_from_disk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_from_disk)

# Stream-ordered allocation benchmark (default pool vs tuned pool)
add_executable(bench_alloc_strategy bench_alloc_strategy.cu)
target_link_libraries(bench_alloc_strategy PRIVATE cu_roaring_bitmap)
set_target_properties(bench_alloc_strategy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_alloc_strategy)