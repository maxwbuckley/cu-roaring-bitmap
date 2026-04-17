cmake_minimum_required(VERSION 3.24)

# Expected to be pulled in via add_subdirectory(v2) from the repo-root CMakeLists,
# which configures CUDA, CRoaring, warning flags, and the ctest harness. This file
# only declares v2's own source list and public include tree.

add_library(cu_roaring_v2 STATIC
    src/upload.cpp
    src/promote.cu
    src/multi_and.cu
    src/decompress.cu
)

target_include_directories(cu_roaring_v2
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cu_roaring_v2 PUBLIC
    CUDA::cudart
    roaring
)

set_target_properties(cu_roaring_v2 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Inherit the parent project's -Werror policy when available.
if(COMMAND cu_roaring_target_strict_warnings)
    cu_roaring_target_strict_warnings(cu_roaring_v2)
endif()

option(CU_ROARING_V2_BUILD_TESTS      "Build cu_roaring_v2 tests"      ON)
option(CU_ROARING_V2_BUILD_BENCHMARKS "Build cu_roaring_v2 benchmarks" ON)

if(CU_ROARING_V2_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(CU_ROARING_V2_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
