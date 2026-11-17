#ifndef BN_QUANT_FP8_E4M3_CUDA_CUH
#define BN_QUANT_FP8_E4M3_CUDA_CUH
#include <cuda_fp8.h>

/* One full warp, four strided features per lane. F32 butterflies and explicit
 * BF16 materialization precede E4M3FN RNE conversion. The scale is a positive
 * power of two, from absmax clamped to 1e-4, divided by finite FP8 max448.
 * Raw bytes live in resident I8 storage; this adds no GGUF weight capability.
 */
namespace bn_cuda_fp8 {
static __device__ void hadamard128_bf16(float *v) {
    for (int d = 1; d < 32; d *= 2) {
        for (int j = 0; j < 4; ++j) {
            float b = __shfl_xor_sync(0xffffffff, v[j], d);
            v[j] = (threadIdx.x & d) ? __fsub_rn(b, v[j]) : __fadd_rn(v[j], b);
        }
    }
    for (int j = 0; j < 4; j += 2) {
        float a = v[j], b = v[j + 1];
        v[j] = __fadd_rn(a, b);
        v[j + 1] = __fsub_rn(a, b);
    }
    for (int j = 0; j < 2; ++j) {
        float a = v[j], b = v[j + 2];
        v[j] = __fadd_rn(a, b);
        v[j + 2] = __fsub_rn(a, b);
    }
    for (int j = 0; j < 4; ++j)
        v[j] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(v[j], 0.08838834764831845f)));
}
static __device__ void quantize128(float *v, unsigned char *out, float *scales, int row) {
    float a = 0;
    for (int j = 0; j < 4; ++j) a = fmaxf(a, fabsf(v[j]));
    for (int d = 16; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
    a = fmaxf(a, 1e-4f);
    float scale = exp2f(ceilf(log2f(__fmul_rn(a, 1.f / 448.f))));
    if (threadIdx.x == 0) scales[row] = scale;
    for (int j = 0; j < 4; ++j) {
        float f = __fdividef(v[j], scale);
        out[row * 128 + threadIdx.x + j * 32] = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
    }
}
} /* namespace bn_cuda_fp8 */
#endif
