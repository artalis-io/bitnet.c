#ifndef BN_GPU_CUDA_MLA_CUH
#define BN_GPU_CUDA_MLA_CUH
#include <cuda_fp8.h>
#include "gpu_cuda_mla_attention.h"
namespace bn_cuda_mla {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;
using bn_cuda_gdn::set_device;
static __global__ void cache_append_kernel(const __nv_bfloat16 *x,const int32_t *slots,
    unsigned char *cache,int capacity) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&15,group=tid/16,slot=slots[row];
    if(slot<0||slot>=capacity)return;
    float v[8],amax=0;
    for(int j=0;j<8;++j){v[j]=__bfloat162float(x[(size_t)row*512+tid*8+j]);amax=fmaxf(amax,fabsf(v[j]));}
    for(int off=8;off;off/=2)amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,off,16));
    float scale=exp2f(ceilf(log2f(fmaxf(__fdiv_rn(amax,448.f),1e-4f))));
    unsigned char *dst=cache+(size_t)slot*656;
    if(!lane)((float*)(dst+512))[group]=scale;
    uint64_t packed=0;
    for(int j=0;j<8;++j)packed|=(uint64_t)__nv_cvt_float_to_fp8(__fdiv_rn(v[j],scale),__NV_SATFINITE,__NV_E4M3)<<(j*8);
    ((uint64_t*)dst)[tid]=packed;
    if(tid<16)((uint64_t*)(dst+528))[tid]=0;
}
static int cache_append(void *ctx_ptr,void *ptr,const BnGPUMlaFP8CacheAppendResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->n_tokens<0||
       p->n_tokens>INT_MAX/512||p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||
       (size_t)p->num_cache_blocks>SIZE_MAX/(64u*656)||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens;BnGPUResidentView v[]={p->kv,p->slots,p->cache};
    size_t counts[]={n*512,n,(size_t)p->num_cache_blocks*64*656};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8};void *data[3];
    for(int j=0;j<3;++j){if(!counts[j]){if(v[j].request||v[j].count)return -1;data[j]=NULL;continue;}
        data[j]=view_data(r,v[j]);if(!data[j]||v[j].count!=counts[j]||v[j].type!=types[j])return -1;
        for(int k=0;k<j;++k)if(counts[k]&&overlap(v[j],v[k]))return -1;}
    if(((uintptr_t)data[0]&15u)||((uintptr_t)data[2]&15u))return -1;
    if(!n)return 0;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    cache_append_kernel<<<p->n_tokens,64,0,r->stream>>>((const __nv_bfloat16*)data[0],
        (const int32_t*)data[1],(unsigned char*)data[2],p->num_cache_blocks*64);
    if(cudaGetLastError()==cudaSuccess)return 0;r->failed=1;return -1;
}
/* Convert and validate before any vendor gather. Invalid GPU metadata cannot
 * escape the physical cache extent even when the caller neglects status. */
static __global__ void physical_indices_kernel(const int32_t *selection,const int32_t *table,
    int32_t *physical,int32_t *status,int position,int logical_blocks,int cache_blocks) {
    int row=blockIdx.x;
    for(int col=threadIdx.x;col<2176;col+=blockDim.x){
        int out=-1;
        if(col<2051){
            int logical=selection[(size_t)row*2051+col];
            if(logical!=-1){
                if(logical<0||logical>position+row||logical/64>=logical_blocks)atomicOr(status,1);
                else{int page=table[logical/64];
                    if(page<0||page>=cache_blocks)atomicOr(status,1);
                    else out=page*64+logical%64;}
            }
        }
        physical[(size_t)row*2176+col]=out;
    }
}
static int attention(void *ctx_ptr,void *ptr,const BnGPUMlaFP8AttentionResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||ctx->multiprocessors<1||
       p->n_tokens<1||p->n_tokens>INT_MAX/32768||p->position<0||p->position>INT_MAX-p->n_tokens||
       p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||p->logical_blocks<1||
       p->logical_blocks>INT_MAX/64||p->position+p->n_tokens>p->logical_blocks*64||
       (size_t)p->num_cache_blocks>SIZE_MAX/(64u*656)||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens,split=n<=64?n*64*34:0;
    BnGPUResidentView v[]={p->query,p->cache,p->selection,p->block_table,p->physical,p->out,
        p->lse,p->partial,p->partial_lse,p->status};
    size_t counts[]={n*64*512,(size_t)p->num_cache_blocks*64*656,n*2051,
        (size_t)p->logical_blocks,n*2176,n*64*512,n*64,split*512,split,1};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,
        BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_BF16,
        BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32};void *data[10];
    for(int j=0;j<10;++j){
        if(!counts[j]){if(v[j].request||v[j].count)return -1;data[j]=NULL;continue;}
        data[j]=view_data(r,v[j]);if(!data[j]||v[j].type!=types[j]||v[j].count!=counts[j])return -1;
        if(j!=2&&j!=3&&j!=9&&((uintptr_t)data[j]&15u))return -1;
        for(int k=0;k<j;++k)if(counts[k]&&overlap(v[j],v[k]))return -1;
    }
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    physical_indices_kernel<<<p->n_tokens,256,0,r->stream>>>((const int32_t*)data[2],
        (const int32_t*)data[3],(int32_t*)data[4],(int32_t*)data[9],p->position,
        p->logical_blocks,p->num_cache_blocks);
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    BnCudaMLAAttentionCall call={};call.query=data[0];call.cache=data[1];call.indices=(const int*)data[4];
    call.out=data[5];call.lse=(float*)data[6];call.partial=data[7];call.partial_lse=(float*)data[8];
    call.n_tokens=p->n_tokens;call.multiprocessors=ctx->multiprocessors;call.stream=r->stream;
    if(!bn_cuda_mla_attention_forward(&call))return 0;
    r->failed=1;return -1;
}

}
#endif
