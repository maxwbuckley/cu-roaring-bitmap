/*
 * Pre-include header for nvcc to prevent GCC 13+ intrinsics from being included.
 * These intrinsics (AMX, CMPccXADD) are not supported by nvcc's cudafe frontend.
 */

#ifdef __CUDACC__
/* Prevent inclusion of GCC 13+ intrinsic headers that nvcc doesn't support */
#undef __CMPCCXADD__
#undef __AMX_TILE__
#undef __AMX_INT8__
#undef __AMX_BF16__
#undef __AMX_FP16__

/* Also prevent the headers from being included if they somehow get defined later */
#define _CMPCCXADDINTRIN_H_INCLUDED 1
#define _AMXTILEINTRIN_H_INCLUDED 1
#define _AMXINT8INTRIN_H_INCLUDED 1
#define _AMXBF16INTRIN_H_INCLUDED 1
#define _AMXFP16INTRIN_H_INCLUDED 1
#endif
