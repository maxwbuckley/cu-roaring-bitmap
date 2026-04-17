add_executable(cu_roaring_v2_test_basic test_basic.cu)

target_link_libraries(cu_roaring_v2_test_basic PRIVATE
    cu_roaring_v2
    roaring
    CUDA::cudart
)

set_target_properties(cu_roaring_v2_test_basic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(COMMAND cu_roaring_target_strict_warnings)
    cu_roaring_target_strict_warnings(cu_roaring_v2_test_basic)
endif()

add_test(NAME cu_roaring_v2_basic COMMAND cu_roaring_v2_test_basic)
