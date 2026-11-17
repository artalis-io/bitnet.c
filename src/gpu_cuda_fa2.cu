/* FlashAttention 2 kernel adapter. Upstream BSD notices are distributed under
 * third_party/. Only kernel headers are included: no Torch host API or runtime. */
#include "gpu_cuda_fa2.h"
#include <cmath>
#include <cstdio>
#include "flash.h"
/* Upstream's paged-load helpers retain explicit flash:: qualifications even
 * when FLASH_NAMESPACE is overridden. Keep their alias local to this TU. */
namespace flash = bn_fa2;
#include "flash_fwd_kernel.h"

namespace bn_cuda_fa2_detail {
using Traits = Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define BN_FA2_GRID_CONST __grid_constant__
#else
#define BN_FA2_GRID_CONST
#endif

template <bool Causal, bool Split>
__global__ void forward_kernel(BN_FA2_GRID_CONST const bn_fa2::Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    bn_fa2::compute_attn_splitkv<Traits, Causal, false, false, false, true,
                                false, Split, false>(params);
#else
    (void)params;
#endif
}

template <int LogSplits>
__global__ void combine_kernel(BN_FA2_GRID_CONST const bn_fa2::Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    bn_fa2::combine_attn_seqk_parallel<Traits, 4, LogSplits, true>(params);
#else
    (void)params;
#endif
}

/* Same occupancy policy as the pinned FA2 host API. Decode groups Q heads
 * into query rows; prefill uses the non-split, paged-compatible forward tree. */
static int split_count(const BnCudaFA2Call &c) {
    int maximum=c.max_query_tokens?c.max_query_tokens:c.n_tokens;
    if (maximum != 1 || c.n_tokens != 1 || c.n_heads == c.n_kv_heads)
        return 1;
    int groups = c.n_heads / c.n_kv_heads;
    int batch = c.split_batch_size ? c.split_batch_size : 1;
    int work = batch * c.n_kv_heads * ((groups + 63) / 64), sms = 2 * c.multiprocessors;
    if (work >= 0.8f * sms)
        return 1;
    int kv = c.split_kv_tokens ? c.split_kv_tokens : c.kv_tokens;
    int blocks = (kv + 63) / 64;
    int limit = blocks < 128 ? blocks : 128;
    if (limit > sms) limit = sms;
    float efficiency[128] = {}, best = 0;
    for (int s = 1; s <= limit; s++) {
        if (s > 1 && (blocks + s - 1) / s == (blocks + s - 2) / (s - 1))
            continue;
        float waves = float(work * s) / sms;
        efficiency[s-1] = waves / std::ceil(waves);
        if (efficiency[s-1] > best) best = efficiency[s-1];
    }
    for (int s = 1; s <= limit; s++)
        if (efficiency[s-1] >= 0.85 * best) return s;
    return 1;
}

static __host__ __device__ bn_fa2::Flash_fwd_params params_for(const BnCudaFA2Call &c,int splits) {
    int maximum=c.max_query_tokens?c.max_query_tokens:c.n_tokens;
    bool swapped = maximum == 1 && c.n_tokens == 1 && c.n_heads > c.n_kv_heads;
    bn_fa2::Flash_fwd_params p = {};
    p.q_ptr = c.q; p.k_ptr = c.k; p.v_ptr = c.v; p.o_ptr = c.out;
    p.softmax_lse_ptr = c.lse;
    p.b = 1; p.h = swapped ? c.n_kv_heads : c.n_heads; p.h_k = c.n_kv_heads;
    p.h_h_k_ratio = p.h / p.h_k;
    p.d = p.d_rounded = 256;
    p.seqlen_q = swapped ? c.n_heads / c.n_kv_heads : c.n_tokens;
    p.seqlen_k = c.kv_tokens; p.total_q = p.seqlen_q;
    p.seqlen_q_rounded = (p.seqlen_q + 127) / 128 * 128;
    p.seqlen_k_rounded = (p.seqlen_k + 127) / 128 * 128;
    p.q_row_stride = p.o_row_stride = swapped ? 256 : c.n_heads * 256;
    p.q_head_stride = p.o_head_stride = swapped ? p.seqlen_q * 256 : 256;
    p.q_batch_stride = p.o_batch_stride = (int64_t)c.n_tokens * c.n_heads * 256;
    p.k_row_stride = p.v_row_stride = c.n_kv_heads * 256;
    p.k_head_stride = p.v_head_stride = 256;
    p.k_batch_stride = p.v_batch_stride = (int64_t)c.cache_capacity * c.n_kv_heads * 256;
    p.scale_softmax = p.scale_softmax_rp_dropout = c.scale;
    p.scale_softmax_log2 = c.scale * 1.4426950408889634074;
    p.p_dropout = p.rp_dropout = 1; p.p_dropout_in_uint8_t = 255;
    p.is_bf16 = true; p.is_causal = maximum > 1;
    p.is_seqlens_k_cumulative = true; p.unpadded_lse = true;
    p.seqlenq_ngroups_swapped = swapped;
    p.window_size_left = p.is_causal ? c.kv_tokens : -1;
    p.window_size_right = p.is_causal ? 0 : -1;
    p.num_splits = splits;
    if (p.num_splits > 1) {
        p.softmax_lseaccum_ptr = c.partial_lse; p.oaccum_ptr = c.partial_out;
    }
    return p;
}

/* Explicit row indices avoid the upstream wrapper's singleton grid mapping.
 * Each row uses its explicitly selected partition count and reduction specialization. */
__global__ void decode_group_kernel(const BnCudaFA2DecodeRow *rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const BnCudaFA2DecodeRow &row=rows[blockIdx.z];
    const bn_fa2::Flash_fwd_params p=params_for(row.call,row.splits);
    if((int)blockIdx.x>=p.h||(int)blockIdx.y>=p.num_splits)return;
    if(p.num_splits>1)
        bn_fa2::compute_attn_1rowblock_splitkv<Traits,false,false,false,false,true,false,true,false>(p,0,blockIdx.x,0,blockIdx.y,p.num_splits);
    else
        bn_fa2::compute_attn_1rowblock_splitkv<Traits,false,false,false,false,true,false,false,false>(p,0,blockIdx.x,0,0,1);
#else
    (void)rows;
#endif
}
__global__ void combine_group_kernel(const BnCudaFA2DecodeRow *rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const BnCudaFA2DecodeRow &row=rows[blockIdx.y];
    if(row.splits<=1||(int)blockIdx.x>=(row.call.n_heads+3)/4)return;
    const bn_fa2::Flash_fwd_params p=params_for(row.call,row.splits);
#define BN_GROUP_COMBINE(L) bn_fa2::combine_attn_seqk_parallel<Traits,4,L,true>(p)
    if(p.num_splits<=2){BN_GROUP_COMBINE(1);}
    else if(p.num_splits<=4){BN_GROUP_COMBINE(2);}
    else if(p.num_splits<=8){BN_GROUP_COMBINE(3);}
    else if(p.num_splits<=16){BN_GROUP_COMBINE(4);}
    else if(p.num_splits<=32){BN_GROUP_COMBINE(5);}
    else if(p.num_splits<=64){BN_GROUP_COMBINE(6);}
    else {BN_GROUP_COMBINE(7);}
#undef BN_GROUP_COMBINE
#else
    (void)rows;
#endif
}

template <bool Causal, bool Split>
static int launch(bn_fa2::Flash_fwd_params &p, cudaStream_t stream) {
    dim3 grid((p.seqlen_q + 63) / 64, Split ? p.num_splits : 1, p.h);
    forward_kernel<Causal, Split><<<grid, 128, Traits::kSmemSize, stream>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
}

using namespace bn_cuda_fa2_detail;

extern "C" int bn_cuda_fa2_prepare(void) {
    cudaError_t errors[] = {
        cudaFuncSetAttribute(forward_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize),
        cudaFuncSetAttribute(forward_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize),
        cudaFuncSetAttribute(forward_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize),
        cudaFuncSetAttribute(decode_group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize)
    };
    for (cudaError_t error : errors)
        if (error != cudaSuccess) return -1;
    return 0;
}

extern "C" int bn_cuda_fa2_forward(const BnCudaFA2Call *call) {
    if (!call) return -1;
    const BnCudaFA2Call &c = *call;
    if (!c.q || !c.k || !c.v || !c.out || !c.lse || c.n_tokens <= 0 ||
        c.n_heads <= 0 || c.n_kv_heads <= 0 || c.n_heads % c.n_kv_heads ||
        c.kv_tokens < c.n_tokens || c.cache_capacity < c.kv_tokens || c.multiprocessors <= 0 ||
        c.max_query_tokens<0 || (c.max_query_tokens&&c.max_query_tokens<c.n_tokens))
        return -1;
    int splits=split_count(c);
    if(splits>1&&(!c.partial_lse||!c.partial_out))return -1;
    bn_fa2::Flash_fwd_params p=params_for(c,splits);
    int rc;
    if (p.is_causal)
        rc = launch<true, false>(p,c.stream);
    else
        rc = p.num_splits > 1 ? launch<false, true>(p,c.stream) : launch<false, false>(p,c.stream);
    if (rc || p.num_splits <= 1) return rc;
    unsigned blocks = (p.h * p.seqlen_q + 3) / 4;
#define BN_FA2_COMBINE(L) combine_kernel<L><<<blocks,128,0,c.stream>>>(p)
    if (p.num_splits <= 2) { BN_FA2_COMBINE(1); }
    else if (p.num_splits <= 4) { BN_FA2_COMBINE(2); }
    else if (p.num_splits <= 8) { BN_FA2_COMBINE(3); }
    else if (p.num_splits <= 16) { BN_FA2_COMBINE(4); }
    else if (p.num_splits <= 32) { BN_FA2_COMBINE(5); }
    else if (p.num_splits <= 64) { BN_FA2_COMBINE(6); }
    else { BN_FA2_COMBINE(7); }
#undef BN_FA2_COMBINE
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int bn_cuda_fa2_decode_splits(const BnCudaFA2Call *c) {
    if(!c||c->n_tokens!=1||c->n_heads<=0||c->n_kv_heads<=0||
       c->n_heads%c->n_kv_heads||c->n_heads/c->n_kv_heads>64||
       c->kv_tokens<=0||c->cache_capacity<c->kv_tokens||c->multiprocessors<=0||
       c->split_batch_size<0||c->split_batch_size>256||c->split_kv_tokens<0||
       c->max_query_tokens<0||(c->max_query_tokens&&c->max_query_tokens<c->n_tokens)||
       (c->split_kv_tokens&&c->split_kv_tokens<c->kv_tokens))return -1;
    return split_count(*c);
}
extern "C" int bn_cuda_fa2_decode_group(const BnCudaFA2DecodeRow *rows,int count,
    int heads,int splits,int combine,cudaStream_t stream) {
    if(!rows||count<=0||count>256||heads<=0||heads>65535||
       splits<=0||splits>128||combine<=0)return -1;
    decode_group_kernel<<<dim3(heads,splits,count),128,Traits::kSmemSize,stream>>>(rows);
    if(cudaGetLastError()!=cudaSuccess)return -1;
    if(splits>1)combine_group_kernel<<<dim3(combine,count),128,0,stream>>>(rows);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
