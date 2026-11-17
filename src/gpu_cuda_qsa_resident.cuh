#ifndef BN_GPU_CUDA_QSA_RESIDENT_CUH
#define BN_GPU_CUDA_QSA_RESIDENT_CUH
#include "gpu_cuda_qsa_attention.cuh"

namespace bn_cuda_qsa_resident {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;
using bn_cuda_qsa_attention::partial_kernel;
using bn_cuda_qsa_attention::merge_kernel;
using bn_cuda_qsa_attention::PartialShared;

struct Profile { int block_n, reference_warps, splits; size_t partial, lse; };
static int profile(int tokens, int prefill, Profile *out) {
    if (!out || tokens <= 0 || tokens > 131070 || (prefill != 0 && prefill != 1)) return -1;
    int programs = tokens*2;
    Profile p = {64,2,1,0,0};
    if (programs > 2048) { if (prefill) p = {32,1,1,0,0}; }
    else if (programs <= 24) p = {32,4,64,0,0};
    else if (programs <= 32) p = {32,1,16,0,0};
    else if (programs <= 64) p = {32,1,8,0,0};
    else if (programs <= 128) p = {32,1,4,0,0};
    else if (programs <= 256) p = {32,1,8,0,0};
    else if (programs <= 512) p = {64,2,4,0,0};
    if (p.splits > 1) {
        p.partial = (size_t)tokens*p.splits*24*256;
        p.lse = (size_t)tokens*p.splits*24;
    }
    *out = p; return 0;
}
static int workspace_size(void *ptr, int max_tokens, size_t *bytes) {
    const BnCudaCtx *ctx = (const BnCudaCtx*)ptr;
    if (!ctx || ctx->compute_capability != 1200 || !bytes || max_tokens <= 0 || max_tokens > 131070) return -1;
    size_t peak = 0;
    /* Above 256 rows every profile is single-split, hence no larger scratch. */
    for (int n = 1; n <= max_tokens && n <= 256; n++) {
        Profile p; if (profile(n,1,&p)) return -1;
        size_t current = (p.partial+p.lse)*sizeof(float);
        if (current > peak) peak = current;
    }
    *bytes = peak; return 0;
}
template<int BN,int W,int S> static cudaError_t configure() {
    return cudaFuncSetAttribute(partial_kernel<BN,W,S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sizeof(PartialShared<BN>));
}
static int prepare(void *ptr) {
    /* Other architectures retain their existing callbacks. Runtime qualification
     * of this exact numerical profile is currently limited to SM120. */
    BnCudaCtx *ctx = (BnCudaCtx*)ptr;
    if (!ctx || ctx->compute_capability != 1200) return -1;
    if (ctx->qsa_attention_ready) return 0;
    if (cuda_ctx_set_device(ctx)) return -1;
    cudaError_t e = configure<32,4,64>();
    if (e == cudaSuccess) e = configure<32,1,16>();
    if (e == cudaSuccess) e = configure<32,1,8>();
    if (e == cudaSuccess) e = configure<32,1,4>();
    if (e == cudaSuccess) e = configure<64,2,4>();
    if (e == cudaSuccess) e = configure<32,1,1>();
    if (e == cudaSuccess) e = configure<64,2,1>();
    if (e != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    ctx->qsa_attention_ready = 1; return 0;
}
static int empty(BnGPUResidentView v) {
    return !v.request && !v.offset && !v.count && !v.slot && !v.type;
}
template<int BN,int W,int S> static void launch(ResidentRequest *r,
    const BnGPUQSAAttentionResidentPlan *p, void **data, float *scratch) {
    partial_kernel<BN,W,S><<<dim3(p->n_tokens,2,S),128,sizeof(PartialShared<BN>),r->stream>>>(
        (const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],
        (const __nv_bfloat16*)data[2],(const int*)data[3],(const int*)data[4],
        (const int*)data[5],S>1?scratch:NULL,S>1?scratch+(size_t)p->n_tokens*S*24*256:NULL,
        (const __nv_bfloat16*)data[6],(__nv_bfloat16*)data[7],p->n_tokens,
        p->table_width,p->num_cache_blocks,p->num_requests);
}
template<int S> static void merge(ResidentRequest *r,
    const BnGPUQSAAttentionResidentPlan *p, void **data, float *scratch) {
    merge_kernel<S><<<dim3(p->n_tokens,24),256,0,r->stream>>>(scratch,
        scratch+(size_t)p->n_tokens*S*24*256,(const __nv_bfloat16*)data[6],
        (__nv_bfloat16*)data[7],p->n_tokens);
}
static int execute(void *ptr, void *request, const BnGPUQSAAttentionResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx*)ptr; ResidentRequest *r = (ResidentRequest*)request;
    Profile config;
    if (!ctx || !r || !p || r->backend_owner != ctx || r->failed ||
        !ctx->qsa_attention_ready || profile(p->n_tokens,p->use_prefill_config,&config) ||
        p->num_cache_blocks <= 0 || p->num_requests <= 0 || p->table_width <= 0 ||
        (size_t)p->num_cache_blocks > SIZE_MAX/8192/2 ||
        (size_t)p->num_requests > SIZE_MAX/(size_t)p->table_width/4) return -1;
    size_t n = (size_t)p->n_tokens, q_count = n*24*256;
    BnGPUResidentView views[] = {p->q,p->k_cache,p->v_cache,p->indices,p->block_table,
        p->token_to_req,p->gate,p->out,p->workspace};
    size_t counts[] = {q_count,(size_t)p->num_cache_blocks*8192,
        (size_t)p->num_cache_blocks*8192,n*2052,(size_t)p->num_requests*p->table_width,
        n,q_count,q_count,config.partial+config.lse};
    int types[] = {BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,
        BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,
        BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F32};
    void *data[9] = {};
    int count = empty(p->workspace) ? 8 : 9;
    if (count == 8 && counts[8]) return -1;
    for (int i=0;i<count;i++) {
        data[i] = view_data(r,views[i]);
        unsigned alignment_mask = (i>=3 && i<=5) ? 3u : 15u;
        if (!data[i] || views[i].type != types[i] || ((uintptr_t)data[i]&alignment_mask) ||
            (i == 8 ? views[i].count < counts[i] : views[i].count != counts[i])) return -1;
        for (int j=0;j<i;j++) if ((i>=7 || j>=7) && overlap(views[i],views[j])) return -1;
    }
    if (bn_cuda_gdn::set_device(r)) return -1;
    if (cudaGetLastError()!=cudaSuccess) { r->failed=1;return -1; }
    float *scratch = (float*)data[8];
    if (config.block_n == 32) {
        if (config.splits == 64) launch<32,4,64>(r,p,data,scratch);
        else if (config.splits == 16) launch<32,1,16>(r,p,data,scratch);
        else if (config.splits == 8) launch<32,1,8>(r,p,data,scratch);
        else if (config.splits == 4) launch<32,1,4>(r,p,data,scratch);
        else launch<32,1,1>(r,p,data,scratch);
    } else {
        if (config.splits == 4) launch<64,2,4>(r,p,data,scratch);
        else launch<64,2,1>(r,p,data,scratch);
    }
    if (cudaGetLastError()!=cudaSuccess) { r->failed=1;return -1; }
    if (config.splits == 64) merge<64>(r,p,data,scratch);
    else if (config.splits == 16) merge<16>(r,p,data,scratch);
    else if (config.splits == 8) merge<8>(r,p,data,scratch);
    else if (config.splits == 4) merge<4>(r,p,data,scratch);
    if (cudaGetLastError()==cudaSuccess) return 0;
    r->failed=1;return -1;
}
} /* namespace bn_cuda_qsa_resident */
#endif
