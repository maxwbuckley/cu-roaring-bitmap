# Expected to be pulled in via add_subdirectory(mrm) from the repo-root
# CMakeLists (configures CUDA, CRoaring, warning flags, ctest, and — when
# tests are enabled — GoogleTest via test/CMakeLists.txt).

add_library(cu_roaring_mrm STATIC
    src/build.cu
    src/decode.cu
    src/search.cu
    src/search_tile.cu
)

target_include_directories(cu_roaring_mrm
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cu_roaring_mrm PUBLIC
    cu_roaring_bitmap
    CUDA::cudart
)

set_target_properties(cu_roaring_mrm PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(COMMAND cu_roaring_target_strict_warnings)
    cu_roaring_target_strict_warnings(cu_roaring_mrm)
endif()

option(CU_ROARING_MRM_BUILD_TESTS      "Build MRM tests"      ON)
option(CU_ROARING_MRM_BUILD_BENCHMARKS "Build MRM benchmarks" ON)

if(CU_ROARING_MRM_BUILD_TESTS AND TARGET GTest::gtest_main)
    foreach(t test_mrm test_mrm_search)
        add_executable(${t} test/${t}.cu)
        target_link_libraries(${t} PRIVATE cu_roaring_mrm GTest::gtest_main)
        set_target_properties(${t} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        if(COMMAND cu_roaring_target_strict_warnings)
            cu_roaring_target_strict_warnings(${t})
        endif()
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()

if(CU_ROARING_MRM_BUILD_BENCHMARKS)
    add_executable(bench_mrm_build bench/bench_mrm_build.cu)
    target_link_libraries(bench_mrm_build PRIVATE cu_roaring_mrm)
    set_target_properties(bench_mrm_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(COMMAND cu_roaring_target_strict_warnings)
        cu_roaring_target_strict_warnings(bench_mrm_build)
    endif()
endif()
