#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu_cuda_activation.cuh"
#include "cuda_bf16_silu_profiles_reference.h"
#include "cuda_compiled_silu_reference.h"

static const int count = 65536 * 8;
static __global__ void initialize(__nv_bfloat16 *gate, __nv_bfloat16 *up) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const float values[]={-2,-1,-0.5f,0,0.5f,1,2,3.140625f};
    gate[i]=__ushort_as_bfloat16((uint16_t)(i/8));up[i]=__float2bfloat16_rn(values[i%8]);
}
static uint64_t digest(const uint16_t *x) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(int i=0;i<count;i++){uint16_t u=x[i];if((u&0x7f80)==0x7f80&&(u&127))u=0x7fc0;h=(h^u)*UINT64_C(1099511628211);}
    return h;
}
int main(void) {
    __nv_bfloat16 *gate,*up;uint16_t*host=(uint16_t*)malloc((size_t)count*2);assert(host);
    assert(cudaMalloc((void**)&gate,(size_t)count*2)==cudaSuccess&&cudaMalloc((void**)&up,(size_t)count*2)==cudaSuccess);
    for(int mode=0;mode<3;mode++) {
        initialize<<<count/256,256>>>(gate,up);
        if(mode==2)silu_mul_compiled_bf16_kernel<<<count/256,256>>>(gate,up,count);
        else if(mode)silu_mul_fused_bf16_kernel<<<count/256,256>>>(gate,up,count);
        else silu_mul_bf16_kernel<<<count/256,256>>>(gate,up,count);
        assert(cudaGetLastError()==cudaSuccess&&cudaMemcpy(host,gate,(size_t)count*2,cudaMemcpyDeviceToHost)==cudaSuccess);
        assert(digest(host)==(mode==2?BN_SILU_COMPILED_FNV:mode?BN_SILU_FUSED_FNV:BN_SILU_TWO_FNV));
    }
    for(int j=0;j<2;j++) {
        uint16_t g=cuda_compiled_silu_edges[j][0],u=cuda_compiled_silu_edges[j][1],out;
        assert(cudaMemcpy(gate,&g,2,cudaMemcpyHostToDevice)==cudaSuccess);
        assert(cudaMemcpy(up,&u,2,cudaMemcpyHostToDevice)==cudaSuccess);
        silu_mul_compiled_bf16_kernel<<<1,256>>>(gate,up,1);
        assert(cudaMemcpy(&out,gate,2,cudaMemcpyDeviceToHost)==cudaSuccess);
        assert(out==cuda_compiled_silu_edges[j][2]);
    }
    cudaFree(gate);cudaFree(up);free(host);
    puts("CUDA BF16 SiLU profiles PASSED: 524288 inputs per profile");return 0;
}
