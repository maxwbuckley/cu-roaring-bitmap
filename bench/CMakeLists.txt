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
    target_link_libraries(${name} PRIVATE cu_roaring_filter benchmark::benchmark benchmark::benchmark_main)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cu_roaring_target_strict_warnings(${name})
endfunction()

add_cu_roaring_bench(bench_set_ops)
add_cu_roaring_bench(bench_decompress)
add_cu_roaring_bench(bench_transfer)

# Comprehensive benchmark (no google benchmark dependency, standalone)
add_executable(bench_comprehensive bench_comprehensive.cu)
target_link_libraries(bench_comprehensive PRIVATE cu_roaring_filter)
set_target_properties(bench_comprehensive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_comprehensive)

# Point query benchmark (standalone, no google benchmark dependency)
add_executable(bench_point_query bench_point_query.cu)
target_link_libraries(bench_point_query PRIVATE cu_roaring_filter)
set_target_properties(bench_point_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_point_query)

# Optimized query benchmark (direct-map + AoS + all-bitmap variants)
add_executable(bench_optimized_query bench_optimized_query.cu)
target_link_libraries(bench_optimized_query PRIVATE cu_roaring_filter)
set_target_properties(bench_optimized_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_optimized_query)

# Upload scale benchmark (GPU sort vs CPU sort at XL scale)
add_executable(bench_upload_scale bench_upload_scale.cu)
target_link_libraries(bench_upload_scale PRIVATE cu_roaring_filter)
set_target_properties(bench_upload_scale PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cu_roaring_target_strict_warnings(bench_upload_scale)
