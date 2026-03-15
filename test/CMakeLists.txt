include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW ON
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

function(add_cu_roaring_test name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE cu_roaring_filter GTest::gtest_main)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_cu_roaring_test(test_upload)
add_cu_roaring_test(test_decompress)
add_cu_roaring_test(test_set_ops)
