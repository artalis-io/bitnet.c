#ifndef BN_GPU_CUDA_ACTIVATION_CUH
#define BN_GPU_CUDA_ACTIVATION_CUH
/* Torch SiLU opmath preserves results when exp(-x) exceeds 2^126. */
static __device__ __forceinline__ float cuda_bf16_silu_float(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
}
static __device__ __forceinline__ void silu_mul_bf16_core(__nv_bfloat16 *gate, const __nv_bfloat16 *up, size_t count, size_t i) {
    if (i >= count) return;
    float g = __bfloat162float(gate[i]);
    __nv_bfloat16 activation = __float2bfloat16_rn(cuda_bf16_silu_float(g));
    gate[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(activation),
        __bfloat162float(up[i])));
}

static __global__ void silu_mul_bf16_kernel(__nv_bfloat16 *gate,
    const __nv_bfloat16 *up, size_t count) {
    silu_mul_bf16_core(gate, up, count, (size_t)blockIdx.x * blockDim.x + threadIdx.x);
}

/* Original vLLM SiluAndMul keeps SiLU in F32 through the multiplication. */
static __global__ void silu_mul_fused_bf16_kernel(__nv_bfloat16 *gate,
    const __nv_bfloat16 *up, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    gate[i] = __float2bfloat16_rn(__fmul_rn(cuda_bf16_silu_float(__bfloat162float(gate[i])),
        __bfloat162float(up[i])));
}
/* Preserve the compiler's full-range approximate division contract. RN division
 * differs at BF16 boundaries on SM120 even with identical gate/up values. */
static __device__ __forceinline__ __nv_bfloat16 silu_mul_compiled_bf16_value(
    __nv_bfloat16 gate,__nv_bfloat16 up) {
    float g = __bfloat162float(gate);
    float denominator = __fadd_rn(1.0f, expf(-g)), activation;
    asm("div.full.f32 %0, %1, %2;" : "=f"(activation) : "f"(g), "f"(denominator));
    return __float2bfloat16_rn(__fmul_rn(activation, __bfloat162float(up)));
}
static __global__ void silu_mul_compiled_bf16_kernel(__nv_bfloat16 *gate,
    const __nv_bfloat16 *up, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gate[i]=silu_mul_compiled_bf16_value(gate[i],up[i]);
}
static __global__ void silu_mul_stacked_compiled_bf16_kernel(__nv_bfloat16 *out,
    const __nv_bfloat16 *merged,size_t count,int hidden) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    size_t row=i/(size_t)hidden,col=i%(size_t)hidden,base=row*(size_t)hidden*2+col;
    out[i]=silu_mul_compiled_bf16_value(merged[base],merged[base+hidden]);
}
#endif
