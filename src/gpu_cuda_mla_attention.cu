#include "gpu_cuda_mla_attention.h"
#ifdef BN_CUDA_MXFP4_SM120
/* Pinned FlashInfer templates; licenses and revision in third_party/flashinfer.
 * No upstream host dispatch, mutable configuration cache, Torch or TVM runtime.
 * The host compilation pass warns about an unused NVFP4 helper's undefined
 * non-device return and unused template constants. Neither is instantiated
 * by this FP8-only adapter. Scope diagnostics to the vendor includes. */
#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 549
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv32_fp8/decode.cuh>
#include <flashinfer/attention/sparse_mla_sm120/execution/merge.cuh>
#include <flashinfer/attention/sparse_mla_sm120/compute/q_rope.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_swapab.cuh>
#pragma nv_diag_default 549
#pragma nv_diag_default 177
using namespace flashinfer::sparse_mla_sm120;
static constexpr ModelType mla_model = ModelType::GLM53_NOPE;
static constexpr int mla_splits = 34;
/* Original resolver's wave-gap heuristic. Scratch stride always keeps all
 * 34 splits; inactive splits write the original -1e30 LSE sentinel. */
static int mla_chunks_per_block(int tokens,int multiprocessors) {
    int cpb=1;float best_gap=2.f;
    for(int candidate=1;candidate<=mla_splits;++candidate){
        int effective=(mla_splits+candidate-1)/candidate;
        long long active=(long long)tokens*4*effective;
        long long ceiling=(active+multiprocessors-1)/multiprocessors;
        if(ceiling>3)continue;
        float gap=(float)ceiling-(float)active/(float)multiprocessors;
        if(gap<best_gap-1e-6f||(gap<best_gap+1e-6f&&candidate>cpb)){best_gap=gap;cpb=candidate;}
    }
    return cpb;
}
extern "C" int bn_cuda_mla_attention_prepare(void) {
    cudaError_t e=cudaFuncSetAttribute(sparse_mla_decode_dsv3_2_kernel<mla_model,64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,Dsv32DecodeSmem<mla_model>::LAUNCH_BYTES);
    if(e!=cudaSuccess)return -1;
    e=cudaFuncSetAttribute(sparse_mla_prefill_swapab_kernel<mla_model,64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,SmemLayoutSwapAB<mla_model>::TOTAL);
    return e==cudaSuccess?0:-1;
}
extern "C" int bn_cuda_mla_attention_forward(const BnCudaMLAAttentionCall *p) {
    if(!p||p->n_tokens<1||p->multiprocessors<1||!p->query||!p->cache||!p->indices||!p->out||!p->lse)return -1;
    if(p->n_tokens>64){
        PrefillColdParams cold{};cold.sm_scale=.0625f;cold.num_tokens=p->n_tokens;
        cold.kv_stride_bytes=656;cold.out_lse_stride_elems=64;cold.topk=2176;cold.page_block_size=64;cold.lse_scale=1.f;
        sparse_mla_prefill_swapab_kernel<mla_model,64><<<p->n_tokens,BLOCK_THREADS,
            SmemLayoutSwapAB<mla_model>::TOTAL,p->stream>>>((const bf16*)p->query,
            (const uint8_t*)p->cache,p->indices,nullptr,(bf16*)p->out,p->lse,cold);
        return cudaGetLastError()==cudaSuccess?0:-1;
    }
    if(!p->partial||!p->partial_lse)return -1;
    int cpb=mla_chunks_per_block(p->n_tokens,p->multiprocessors);
    sparse_mla_decode_dsv3_2_kernel<mla_model,64><<<dim3(p->n_tokens,4,mla_splits),
        DSV32_BLOCK_THREADS,Dsv32DecodeSmem<mla_model>::LAUNCH_BYTES,p->stream>>>(
        (const bf16*)p->query,(const uint8_t*)p->cache,p->indices,(bf16*)p->partial,
        p->partial_lse,nullptr,p->n_tokens,64,2176,mla_splits,cpb,.0625f,64*656,2176,656,
        flashinfer::uint_fastdiv(64));
    if(cudaGetLastError()!=cudaSuccess)return -1;
    sparse_mla_decode_dsv4_merge_kernel<64,512,64,8><<<dim3(p->n_tokens,64),64,
        mla_splits*sizeof(float),p->stream>>>((const bf16*)p->partial,p->partial_lse,
        (bf16*)p->out,p->lse,nullptr,p->n_tokens,mla_splits,64,64,64,1.f);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
#else
extern "C" int bn_cuda_mla_attention_prepare(void){return -1;}
extern "C" int bn_cuda_mla_attention_forward(const BnCudaMLAAttentionCall*){return -1;}
#endif
