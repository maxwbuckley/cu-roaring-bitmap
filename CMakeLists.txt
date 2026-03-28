cmake_minimum_required(VERSION 3.25)
project(cu_roaring_filter LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architecture: SM_89 (Ada) + PTX for JIT to SM_120 (Blackwell)
# CUDA 12.4 doesn't support sm_120 natively, but PTX compiled for sm_89
# will JIT-compile to sm_120 at runtime on RTX 5090.
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")

# Find CUDA toolkit so CXX files can include cuda_runtime.h
find_package(CUDAToolkit REQUIRED)

# GCC 13 intrinsics fix: pre-include header to suppress builtins nvcc can't parse
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --pre-include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc_pre_include.h")
endif()

# CRoaring (add BEFORE our warning flags to avoid contamination)
set(ROARING_BUILD_LTO OFF CACHE BOOL "" FORCE)
set(ENABLE_ROARING_TESTS OFF CACHE BOOL "" FORCE)
set(ROARING_BUILD_C_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ROARING_BUILD_CXX_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/CRoaring EXCLUDE_FROM_ALL)

# ============================================================================
# Strict warning policy — applied to ALL our targets (library, tests, benches)
#
# CXX:  Full GCC/Clang warning set with -Werror
# CUDA: nvcc --Werror all-warnings, forwarding the same flags to the host
#       compiler via -Xcompiler.  -Wold-style-cast and -Wconversion are
#       excluded because CUDA/thrust headers and nvcc-generated code trigger
#       them in ways we cannot fix.
# ============================================================================
set(CU_ROARING_CXX_WARNINGS
    -Wall -Wextra -Werror
    -Wshadow
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wcast-align
    -Wformat=2
    -Wimplicit-fallthrough
    -Wmisleading-indentation
    -Wnull-dereference
    -Wdouble-promotion
    -Wunused
    -Wpedantic
)

# For CUDA files, forward a subset via -Xcompiler.  Flags that fire inside
# nvcc-generated host stubs or CUDA toolkit headers are omitted.
set(CU_ROARING_CUDA_XCOMPILER
    -Wall,-Wextra,-Wshadow,-Wnon-virtual-dtor,-Woverloaded-virtual,-Wcast-align,-Wformat=2,-Wimplicit-fallthrough,-Wunused,-Wnull-dereference)

# Helper: apply our strict warnings to a target
function(cu_roaring_target_strict_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CU_ROARING_CXX_WARNINGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings -Xcompiler=${CU_ROARING_CUDA_XCOMPILER}>
    )
endfunction()

# Main library
# upload.cpp is compiled by g++ (not nvcc) because it touches CRoaring internals
# which include GCC 13 intrinsics that nvcc 12.4 cannot parse.
add_library(cu_roaring_filter
    src/upload.cpp
    src/set_ops.cu
    src/decompress.cu
    src/upload_ids.cu
    src/promote.cu
    src/to_csr.cu
)

# Header-only interface target for device-side API (roaring_view, warp_query)
add_library(cu_roaring_device INTERFACE)
target_include_directories(cu_roaring_device INTERFACE include/)
target_include_directories(cu_roaring_filter PUBLIC include/)
target_link_libraries(cu_roaring_filter PUBLIC roaring CUDA::cudart)
set_target_properties(cu_roaring_filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

cu_roaring_target_strict_warnings(cu_roaring_filter)

# Tests
option(CU_ROARING_BUILD_TESTS "Build tests" ON)
if(CU_ROARING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmarks
option(CU_ROARING_BUILD_BENCHMARKS "Build benchmarks" ON)
if(CU_ROARING_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
