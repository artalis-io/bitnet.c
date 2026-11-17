/* Offline oracle: nvcc -Itest this.cu -lcuda.
 * Run ORIGINAL_CUBIN FA2_REFERENCE_DIR, or ORIGINAL_CUBIN --wide.
 * Only the original cached kernel supplies expected values. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "cuda_bf16_attention_reference.h"

static uint64_t hash(const std::vector<unsigned short>& values) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (unsigned short value : values) {
        h = (h ^ (value & 255)) * UINT64_C(1099511628211);
        h = (h ^ (value >> 8)) * UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t original_gate(CUfunction fun,
    const std::vector<unsigned short>& x, const std::vector<unsigned short>& gate) {
    assert(x.size() == gate.size());
    int count = (int)x.size();
    size_t bytes = x.size() * 2;
    __nv_bfloat16 *dx, *dg, *dy;
    assert(cudaMalloc(&dx,bytes) == cudaSuccess &&
           cudaMalloc(&dg,bytes) == cudaSuccess && cudaMalloc(&dy,bytes) == cudaSuccess);
    assert(cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(dg,gate.data(),bytes,cudaMemcpyHostToDevice) == cudaSuccess);
    void *scratch = NULL;
    void *args[] = {&dx,&dg,&dy,&count,&scratch,&scratch};
    assert(cuLaunchKernel(fun,(count+1023)/1024,1,1,128,1,1,0,0,args,0) == CUDA_SUCCESS);
    std::vector<unsigned short> out(x.size());
    assert(cudaMemcpy(out.data(),dy,bytes,cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(dx); cudaFree(dg); cudaFree(dy);
    return hash(out);
}

int main(int argc, char **argv) {
    assert(argc == 3 && cudaFree(0) == cudaSuccess);
    CUmodule module;
    CUfunction fun;
    assert(cuModuleLoad(&module,argv[1]) == CUDA_SUCCESS &&
           cuModuleGetFunction(&fun,module,"triton_poi_fused_mul_sigmoid_view_0") == CUDA_SUCCESS);
    if (!strcmp(argv[2],"--wide")) {
        const int count = 2097152;
        std::vector<unsigned short> x(count), gate(count);
        for (int seed=1; seed<=2; seed++) {
            for (int i=0; i<count; i++) {
                x[i] = (unsigned short)((unsigned)i*37+(unsigned)(i/17)*seed);
                gate[i] = (unsigned short)((unsigned)i*131+(unsigned)(i/31)*seed);
                if ((x[i]&0x7f80) == 0x7f80) x[i] &= 0xff7f;
                if ((gate[i]&0x7f80) == 0x7f80) gate[i] &= 0xff7f;
            }
            printf("{\"seed\":%d,\"words\":%d,\"hash\":%llu}\n",seed,count,
                   (unsigned long long)original_gate(fun,x,gate));
        }
    } else {
        int step=-1, chain=-1;
        for (size_t ci=0; ci<sizeof(cuda_bf16_attention_cases)/sizeof(cuda_bf16_attention_cases[0]); ci++) {
            const auto& ref = cuda_bf16_attention_cases[ci];
            step = chain == ref.chain ? step+1 : 0;
            chain = ref.chain;
            int count = ref.tokens*ref.heads*256;
            std::vector<unsigned short> x(count), gate(count);
            char path[1024];
            snprintf(path,sizeof(path),"%s/chain%d-%d-attention.bin",argv[2],chain,step);
            FILE *f = fopen(path,"rb");
            assert(f && fread(x.data(),2,count,f) == (size_t)count);
            fclose(f);
            assert(hash(x) == ref.attention_hash);
            size_t start = (size_t)ref.position*ref.heads*256;
            for (int j=0; j<count; j++) {
                size_t i = start+j;
                float value = (float)((int)((i*19+ref.seed*7)%509)-254)/16;
                const float special[] = {0.f,-0.f,-88.f,-89.f,-90.f,88.f,89.f,90.f};
                if (j<8) value = special[j];
                gate[j] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
            }
            uint64_t expected = ref.gated ? original_gate(fun,x,gate) : hash(x);
            printf("{\"case\":%zu,\"words\":%d,\"hash\":%llu}\n",ci,count,
                   (unsigned long long)expected);
        }
    }
    cuModuleUnload(module);
}
