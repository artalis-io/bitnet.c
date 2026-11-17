#ifndef BN_GPU_CUDA_ATTENTION_GATE_CUH
#define BN_GPU_CUDA_ATTENTION_GATE_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace bn_cuda_attention_gate {

/* The compiled contract retains sigmoid in F32 through multiplication.
 * Preserve div.full: div.rn differs at a BF16 rounding boundary in the
 * independent finite-input fixture. */
template<bool Compiled>
static __device__ __nv_bfloat16 element(__nv_bfloat16 x,__nv_bfloat16 g) {
    float a = __bfloat162float(g), sigmoid;
    if (Compiled) {
        float exponent, scaled = __fmul_rn(a, -0x1.715476p+0f);
        asm("ex2.approx.f32 %0,%1;" : "=f"(exponent) : "f"(scaled));
        float denominator = __fadd_rn(1.0f, exponent);
        asm("div.full.f32 %0,%1,%2;" : "=f"(sigmoid)
            : "f"(1.0f), "f"(denominator));
    } else {
        sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-a)));
        sigmoid = __bfloat162float(__float2bfloat16_rn(sigmoid));
    }
    return __float2bfloat16_rn(__fmul_rn(__bfloat162float(x), sigmoid));
}

template<bool Compiled>
static __global__ void apply(__nv_bfloat16 *out, const __nv_bfloat16 *x,
                             const __nv_bfloat16 *g, size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=element<Compiled>(x[i],g[i]);
}

} // namespace bn_cuda_attention_gate
#endif
