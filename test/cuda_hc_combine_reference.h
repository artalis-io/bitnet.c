#ifndef BN_CUDA_HC_COMBINE_REFERENCE_H
#define BN_CUDA_HC_COMBINE_REFERENCE_H
#include <stdint.h>
/* Original llama.cpp 3d3d7c81813067fc8c185da017e0af03b4269b1e CUDA,
 * SM120: scale -> sigmoid -> scale -> repeat -> multiply -> add.
 * FNV-1a over float bit patterns; 36 cases, 117504 baseline output words. */
typedef struct {
    int dim, streams, seed;
    float magnitude;
    uint64_t hash;
} BnCudaHCCombineReference;
static const BnCudaHCCombineReference cuda_hc_combine_reference[] = {
    {64,2,0,9.99999978e-03f,UINT64_C(0x7a9ebf45039c3735)},
    {64,2,0,1.00000000e+00f,UINT64_C(0xa8410a0e712f21c1)},
    {64,2,0,4.00000000e+00f,UINT64_C(0xc7faeb3d472f21c1)},
    {64,2,23,9.99999978e-03f,UINT64_C(0x84681b924fec1324)},
    {64,2,23,1.00000000e+00f,UINT64_C(0x22e1186498954af6)},
    {64,2,23,4.00000000e+00f,UINT64_C(0x53f3b9086e954af6)},
    {64,4,0,9.99999978e-03f,UINT64_C(0x24495f292c8e6307)},
    {64,4,0,1.00000000e+00f,UINT64_C(0x87aa45575ff93ea6)},
    {64,4,0,4.00000000e+00f,UINT64_C(0xe116e79309f93ea6)},
    {64,4,23,9.99999978e-03f,UINT64_C(0x1f49148ca4ad4659)},
    {64,4,23,1.00000000e+00f,UINT64_C(0x1cdf23def79ca47b)},
    {64,4,23,4.00000000e+00f,UINT64_C(0x60873d56559ca47b)},
    {640,2,0,9.99999978e-03f,UINT64_C(0x9ed40ecc8c9598d2)},
    {640,2,0,1.00000000e+00f,UINT64_C(0x73dd1b17e04de7bc)},
    {640,2,0,4.00000000e+00f,UINT64_C(0x0e45d39dda4de7bc)},
    {640,2,23,9.99999978e-03f,UINT64_C(0xb820ed6df7b8707f)},
    {640,2,23,1.00000000e+00f,UINT64_C(0xef647e025cdb7d7f)},
    {640,2,23,4.00000000e+00f,UINT64_C(0x500957d17cdb7d7f)},
    {640,4,0,9.99999978e-03f,UINT64_C(0x43ab261bf68045b8)},
    {640,4,0,1.00000000e+00f,UINT64_C(0xac5d4f0949b74d63)},
    {640,4,0,4.00000000e+00f,UINT64_C(0x1c68046d9bb74d63)},
    {640,4,23,9.99999978e-03f,UINT64_C(0xeac2650c7470b6f4)},
    {640,4,23,1.00000000e+00f,UINT64_C(0x21cffafc68a6f5c6)},
    {640,4,23,4.00000000e+00f,UINT64_C(0xaa7d238a90a6f5c6)},
    {2560,2,0,9.99999978e-03f,UINT64_C(0x3ffe08e0cfc0adef)},
    {2560,2,0,1.00000000e+00f,UINT64_C(0x8359fb6bf1edacd5)},
    {2560,2,0,4.00000000e+00f,UINT64_C(0xf1e9d0ab29edacd5)},
    {2560,2,23,9.99999978e-03f,UINT64_C(0x04427e9c8b450b14)},
    {2560,2,23,1.00000000e+00f,UINT64_C(0x72ffda15bb18befc)},
    {2560,2,23,4.00000000e+00f,UINT64_C(0xaf53da244718befc)},
    {2560,4,0,9.99999978e-03f,UINT64_C(0x1b268108b807cf58)},
    {2560,4,0,1.00000000e+00f,UINT64_C(0xaed309958d80967b)},
    {2560,4,0,4.00000000e+00f,UINT64_C(0xe393a7a4c980967b)},
    {2560,4,23,9.99999978e-03f,UINT64_C(0xac903d1b9a69ea3b)},
    {2560,4,23,1.00000000e+00f,UINT64_C(0x7499ff4376402cff)},
    {2560,4,23,4.00000000e+00f,UINT64_C(0x7723a2f2d6402cff)},
};
#endif
