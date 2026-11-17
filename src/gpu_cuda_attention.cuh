#include "gpu_cuda_attention_gate.cuh"
#ifndef BN_GPU_CUDA_ATTENTION_CUH
#define BN_GPU_CUDA_ATTENTION_CUH

namespace bn_cuda_attention {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::set_device;
using bn_cuda_gdn::synchronize;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;

template<bool INDIRECT = false>
static __global__ void qk_prep_kernel(const __nv_bfloat16 *q_gate,
    const __nv_bfloat16 *kin, const __nv_bfloat16 *vin,
    const __nv_bfloat16 *qw, const __nv_bfloat16 *kw, const __nv_bfloat16 *cs,
    __nv_bfloat16 *q, __nv_bfloat16 *k, __nv_bfloat16 *v, __nv_bfloat16 *gate_out,
    int heads, int kv_heads, int rotary, int position, float eps, int offset,
    size_t q_stride, size_t k_stride, size_t v_stride,
    const BnCudaQKRow *rows = NULL, int row_count = 0) {
    int token=blockIdx.x, head=blockIdx.y, tid=threadIdx.x;
    if (INDIRECT) {
        /* Flatten token/head tiles instead of padding every sequence to the
         * longest chunk. All lanes select the same descriptor. */
        int lo=0,hi=row_count;
        while(lo<hi) {int mid=lo+(hi-lo)/2;if((int)blockIdx.x>=rows[mid].block_end)lo=mid+1;else hi=mid;}
        if(lo==row_count)return;
        BnCudaQKRow row=rows[lo];
        int tile=(int)blockIdx.x-row.block_begin;
        token=tile/(row.heads+row.kv_heads);head=tile%(row.heads+row.kv_heads);
        q_gate=row.q_gate;kin=row.kin;vin=row.vin;qw=row.qw;kw=row.kw;cs=row.cs;
        q=row.q;k=row.k;v=row.v;gate_out=row.gate;
        heads=row.heads;kv_heads=row.kv_heads;rotary=row.rotary;
        position=row.position;eps=row.eps;offset=row.offset;
        q_stride=row.q_stride;k_stride=row.k_stride;v_stride=row.v_stride;
    }
    bool is_k=head>=heads;
    int h=is_k?head-heads:head;
    const __nv_bfloat16 *x=is_k?kin+(size_t)token*k_stride+(size_t)h*256:
                                  q_gate+(size_t)token*q_stride+(size_t)h*512;
    const __nv_bfloat16 *w=is_k?kw:qw;
    /* The reference's 128-thread layout reduces adjacent pairs first, then
     * 32 lanes, then four warps. A strided two-element reduction differs. */
    float a=__bfloat162float(x[2*tid]), b=__bfloat162float(x[2*tid+1]);
    float sum=__fmaf_rn(a,a,__fmul_rn(b,b));
    for(int step=16;step;step>>=1)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,step));
    __shared__ float sums[4], total_sum, normalized[256];
    if((tid&31)==0)sums[tid>>5]=sum;
    __syncthreads();
    if(tid<32) {
        float total=sums[tid&3];
        total=__fadd_rn(total,__shfl_xor_sync(0xffffffffu,total,2));
        total=__fadd_rn(total,__shfl_xor_sync(0xffffffffu,total,1));
        if(tid==0)total_sum=total;
    }
    __syncthreads();
    float scale=rsqrtf(__fadd_rn(__fmul_rn(total_sum,1.0f/256),eps));
    for(int j=0;j<2;j++) {
        int i=tid*2+j;
        float weight=__bfloat162float(w[i]);
        if(offset)weight=__fadd_rn(weight,1.0f);
        /* Normalization is stored as BF16 before the F32 rotary arithmetic. */
        normalized[i]=__bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[i]),scale),weight)));
    }
    __syncthreads();
    size_t out_base=((size_t)token*(is_k?kv_heads:heads)+h)*256;
    __nv_bfloat16 *out=(is_k?k:q)+out_base;
    for(int j=0;j<2;j++) {
        int i=tid*2+j, half=rotary/2;
        float y=normalized[i];
        if(i<rotary) {
            int r=i%half;
            const __nv_bfloat16 *cache=cs+(size_t)(position+token)*rotary;
            float c=__bfloat162float(cache[r]),s=__bfloat162float(cache[half+r]);
            float first=normalized[r],second=normalized[half+r];
            /* Preserve the reference's multiply/FMA order in each half. */
            y=i<half?__fmaf_rn(first,c,-__fmul_rn(second,s)):
                         __fmaf_rn(second,c,__fmul_rn(first,s));
        }
        out[i]=__float2bfloat16_rn(y);
        if(is_k)v[out_base+i]=vin[(size_t)token*v_stride+(size_t)h*256+i];
        else gate_out[out_base+i]=x[256+i];
    }
}

static int qk_validate(void *ctx_ptr, void *ptr, const BnGPUQKPrepResidentPlan *p,
    BnCudaQKRow *row, ResidentRequest *activation_owner = NULL) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||
       p->head_size!=256||p->n_tokens<=0||p->n_heads<=0||p->n_heads>32767||
       p->n_kv_heads<=0||p->n_kv_heads>p->n_heads||p->n_heads%p->n_kv_heads||
       p->rotary_dim<=0||p->rotary_dim>256||p->rotary_dim%2||p->position<0||
       p->position>INT_MAX-p->n_tokens||!isfinite(p->eps)||p->eps<=0||
       (p->weight_offset!=0&&p->weight_offset!=1)||set_device(r))return -1;
    const BnCudaBuffer *qw=(const BnCudaBuffer*)p->q_weight,*kw=(const BnCudaBuffer*)p->k_weight,
                       *cs=(const BnCudaBuffer*)p->cos_sin;
    if(!qw||!kw||!cs||!qw->data||!kw->data||!cs->data||
       !bn_quant_format_is_bf16(qw->type)||!bn_quant_format_is_bf16(kw->type)||
       !bn_quant_format_is_bf16(cs->type)||qw->rows!=1||kw->rows!=1||
       qw->cols!=256||kw->cols!=256||qw->size!=512||kw->size!=512||
       cs->rows<p->position+p->n_tokens||cs->cols!=p->rotary_dim||
       cs->size/2/(size_t)p->rotary_dim<(size_t)(p->position+p->n_tokens))return -1;
    size_t nq=(size_t)p->n_tokens*p->n_heads*256,nk=(size_t)p->n_tokens*p->n_kv_heads*256;
    if(nq>SIZE_MAX/4||nk>SIZE_MAX/4||nq+nk>SIZE_MAX/4)return -1;
    size_t widths[]={(size_t)p->n_heads*512,(size_t)p->n_kv_heads*256,
                     (size_t)p->n_kv_heads*256};
    size_t strides[]={p->q_gate_row_stride,p->k_in_row_stride,p->v_in_row_stride};
    size_t counts[7]={0,0,0,nq,nk,nk,nq};
    for(int i=0;i<3;i++) {
        if(!strides[i])strides[i]=widths[i];
        if(strides[i]<widths[i]||(strides[i]&7u)||strides[i]>SIZE_MAX/2||
           (size_t)(p->n_tokens-1)>(SIZE_MAX/2-widths[i])/strides[i])return -1;
        counts[i]=(size_t)(p->n_tokens-1)*strides[i]+widths[i];
    }
    BnGPUResidentView views[]={p->q_gate,p->k_in,p->v_in,p->q,p->k,p->v,p->gate};
    void *data[7]={};
    for(int i=0;i<7;i++) {
        ResidentRequest *owner=r;
        if(activation_owner&&views[i].request==activation_owner)owner=activation_owner;
        data[i]=view_data(owner,views[i]);
        if(!data[i]||views[i].count!=counts[i]||!bn_quant_format_is_bf16(views[i].type)||
           ((uintptr_t)data[i]&15u))return -1;
        for(int j=0;j<i;j++)if((i>=3||j>=3)&&overlap(views[i],views[j]))return -1;
    }
    row->q_gate=(const __nv_bfloat16*)data[0];row->kin=(const __nv_bfloat16*)data[1];
    row->vin=(const __nv_bfloat16*)data[2];row->qw=(const __nv_bfloat16*)qw->data;
    row->kw=(const __nv_bfloat16*)kw->data;row->cs=(const __nv_bfloat16*)cs->data;
    row->q=(__nv_bfloat16*)data[3];row->k=(__nv_bfloat16*)data[4];
    row->v=(__nv_bfloat16*)data[5];row->gate=(__nv_bfloat16*)data[6];
    row->tokens=p->n_tokens;row->heads=p->n_heads;row->kv_heads=p->n_kv_heads;
    row->rotary=p->rotary_dim;row->position=p->position;row->eps=p->eps;row->offset=p->weight_offset;
    row->q_stride=strides[0];row->k_stride=strides[1];row->v_stride=strides[2];
    return 0;
}
static int qk_prep(void *ctx_ptr,void *ptr,const BnGPUQKPrepResidentPlan *p) {
    ResidentRequest *r=(ResidentRequest*)ptr;BnCudaQKRow row;
    if(qk_validate(ctx_ptr,ptr,p,&row))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    qk_prep_kernel<false><<<dim3(p->n_tokens,p->n_heads+p->n_kv_heads),128,0,r->stream>>>(
        row.q_gate,row.kin,row.vin,row.qw,row.kw,row.cs,row.q,row.k,row.v,row.gate,
        row.heads,row.kv_heads,row.rotary,row.position,row.eps,row.offset,
        row.q_stride,row.k_stride,row.v_stride);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static size_t qk_group_workspace_size(void *ctx,int capacity) {
    return bn_cuda_gdn::sequence_group_workspace_size<BnCudaQKRow>(ctx,capacity);
}
static int qk_group_reserve(void *ctx,void *ptr,int capacity,size_t budget) {
    ResidentRequest *r=(ResidentRequest*)ptr;
    return r?bn_cuda_gdn::group_reserve(ctx,ptr,capacity,budget,&r->qk_group):-1;
}
static int qk_group(void *ctx,void *ptr,const BnGPUQKPrepGroupPlan *p) {
    ResidentRequest *coordinator=(ResidentRequest*)ptr;
    if(!coordinator||coordinator->backend_owner!=ctx||coordinator->failed||!p||
       !p->sequences||p->n_sequences<=0||p->n_sequences>256||
       !coordinator->qk_group||p->n_sequences>coordinator->qk_group->capacity)return -1;
    ResidentRequest *participants[256];BnCudaQKRow rows[256];int blocks=0;
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUQKPrepSequence *s=&p->sequences[i];
        ResidentRequest *r=(ResidentRequest*)s->request;
        if(!r||r==coordinator||r->backend_owner!=ctx||r->device!=coordinator->device)return -1;
        for(int j=0;j<i;j++)if(participants[j]==r)return -1;
        if(qk_validate(ctx,r,&s->plan,&rows[i],coordinator))return -1;
        participants[i]=r;
        int heads=rows[i].heads+rows[i].kv_heads;
        if(rows[i].tokens>(INT_MAX-blocks)/heads)return -1;
        rows[i].block_begin=blocks;blocks+=rows[i].tokens*heads;rows[i].block_end=blocks;
    }
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUQKPrepResidentPlan *a=&p->sequences[i].plan;
        BnGPUResidentView writes[]={a->q,a->k,a->v,a->gate};
        for(int j=0;j<p->n_sequences;j++)if(i!=j) {
            const BnGPUQKPrepResidentPlan *b=&p->sequences[j].plan;
            BnGPUResidentView all[]={b->q_gate,b->k_in,b->v_in,b->q,b->k,b->v,b->gate};
            for(int w=0;w<4;w++)for(int v=0;v<7;v++)if(overlap(writes[w],all[v]))return -1;
        }
    }
    bn_cuda_gdn::QKGroupWorkspace *w=coordinator->qk_group;
    int status=bn_cuda_gdn::group_prepare(coordinator,w,participants,rows,p->n_sequences);
    if(status)return status;
    qk_prep_kernel<true><<<blocks,128,0,coordinator->stream>>>(
        NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,0,0,0,0,0,0,0,0,0,w->device,p->n_sequences);
    return bn_cuda_gdn::group_finish(coordinator,w,participants,p->n_sequences);
}

/* Slot uniqueness is guaranteed by the request scheduler. Each valid row
 * copies K and V together, preserving every BF16 bit pattern. */
static __global__ void paged_kv_append_kernel(const uint4 *k,const uint4 *v,
    const int *slots,uint4 *kc,uint4 *vc,int capacity,int vectors) {
    int row=blockIdx.x,slot=slots[row];
    if(slot<0||slot>=capacity)return;
    for(int i=threadIdx.x;i<vectors;i+=blockDim.x) {
        kc[(size_t)slot*vectors+i]=k[(size_t)row*vectors+i];
        vc[(size_t)slot*vectors+i]=v[(size_t)row*vectors+i];
    }
}
static int paged_kv_append(void *ctx_ptr,void *ptr,const BnGPUPagedKVAppendResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       p->n_tokens<=0||p->num_cache_blocks<=0||p->page_size<=0||p->page_size>65536||
       p->num_cache_blocks>INT_MAX/p->page_size||p->row_width<8||p->row_width>65536||
       p->row_width%8)return -1;
    int capacity=p->num_cache_blocks*p->page_size;
    if((size_t)p->n_tokens>SIZE_MAX/(size_t)p->row_width/2||
       (size_t)capacity>SIZE_MAX/(size_t)p->row_width/2)return -1;
    size_t rows=(size_t)p->n_tokens*p->row_width,cache=(size_t)capacity*p->row_width;
    BnGPUResidentView views[]={p->k,p->v,p->slots,p->k_cache,p->v_cache};
    size_t counts[]={rows,rows,(size_t)p->n_tokens,cache,cache};void *data[5]={};
    for(int i=0;i<5;i++) {
        data[i]=view_data(r,views[i]);
        if(!data[i]||views[i].count!=counts[i]||
           views[i].type!=(i==2?BN_GGUF_TENSOR_I32:BN_GGUF_TENSOR_BF16)||
           ((uintptr_t)data[i]&(i==2?3u:15u)))return -1;
        for(int j=0;j<i;j++)if((i>=3||j>=3)&&overlap(views[i],views[j]))return -1;
    }
    if(set_device(r))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    paged_kv_append_kernel<<<p->n_tokens,128,0,r->stream>>>((const uint4*)data[0],
        (const uint4*)data[1],(const int*)data[2],(uint4*)data[3],(uint4*)data[4],
        capacity,p->row_width/8);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static size_t workspace_size(void *ctx_ptr, int tokens, int heads, int context) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    if (!ctx || ctx->compute_capability < 800 || tokens <= 0 || heads <= 0 ||
        heads > 65535 || context <= 0 || context > INT_MAX - 127 || tokens > context)
        return 0;
    int splits = (context + 63) / 64;
    if (splits > 128) splits = 128;
    if (tokens > INT_MAX / (heads * 512) || heads > INT_MAX / (splits * 1024)) return 0;
    return ((size_t)tokens * heads + (size_t)splits * heads * 257) * sizeof(float);
}

static int reserve(void *ctx_ptr, void *ptr, int tokens, int heads, int context) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!r || r->backend_owner != ctx || !workspace_size(ctx, tokens, heads, context) || set_device(r))
        return -1;
    if (tokens <= r->attention_tokens && heads <= r->attention_heads &&
        context <= r->attention_context)
        return 0;
    tokens = tokens > r->attention_tokens ? tokens : r->attention_tokens;
    heads = heads > r->attention_heads ? heads : r->attention_heads;
    context = context > r->attention_context ? context : r->attention_context;
    int splits = (context + 63) / 64;
    if (splits > 128) splits = 128;
    if (tokens > INT_MAX / (heads * 512) || heads > INT_MAX / (splits * 1024) ||
        synchronize(r) || bn_cuda_fa2_prepare())
        return -1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, r->device) != cudaSuccess || sms <= 0)
        return -1;
    float *lse = NULL, *partial_lse = NULL, *partial_out = NULL;
    if (cudaMalloc(&lse, (size_t)tokens * heads * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&partial_lse, (size_t)splits * heads * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&partial_out, (size_t)splits * heads * 256 * sizeof(float)) != cudaSuccess) {
        cudaFree(lse); cudaFree(partial_lse); cudaFree(partial_out);
        return -1;
    }
    cudaFree(r->attention_lse);
    cudaFree(r->attention_partial_lse);
    cudaFree(r->attention_partial_out);
    r->attention_lse = lse; r->attention_partial_lse = partial_lse;
    r->attention_partial_out = partial_out; r->attention_tokens = tokens;
    r->attention_heads = heads; r->attention_context = context; r->attention_sms = sms;
    return 0;
}

static __global__ void append(uint16_t *key, uint16_t *value, const uint16_t *k,
    const uint16_t *v, size_t count, size_t offset) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { key[offset+i] = k[i]; value[offset+i] = v[i]; }
}

static int same(BnGPUResidentView a, BnGPUResidentView b) {
    return a.request == b.request && a.slot == b.slot && a.offset == b.offset &&
           a.count == b.count && a.type == b.type;
}

static int attention_validate(void *ctx_ptr,void *ptr,const BnGPUAttentionResidentPlan *p,
    void **data,BnCudaFA2Call *out_call,ResidentRequest *activation_owner=NULL) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->head_size != 256 || p->n_tokens <= 0 || p->n_tokens > r->attention_tokens ||
        p->n_heads <= 0 || p->n_heads > r->attention_heads || p->n_kv_heads <= 0 ||
        p->n_heads % p->n_kv_heads || p->cache_capacity <= 0 ||
        p->cache_capacity > r->attention_context || p->position < 0 ||
        p->n_tokens > p->cache_capacity || p->position > p->cache_capacity-p->n_tokens ||
        p->n_tokens > INT_MAX / (p->n_heads * 512) ||
        p->cache_capacity > INT_MAX / (p->n_kv_heads * 512) ||
        (p->gated != 0 && p->gated != 1) ||
        (p->gate_profile != BN_GPU_ATTENTION_GATE_BF16 &&
         p->gate_profile != BN_GPU_ATTENTION_GATE_COMPILED_F32) ||
        (p->gate_profile == BN_GPU_ATTENTION_GATE_COMPILED_F32 &&
         (!p->gated || ctx->compute_capability != 1200)) || !isfinite(p->scale) || p->scale <= 0 ||
        !r->attention_lse || !r->attention_partial_lse || !r->attention_partial_out || r->attention_sms <= 0 ||
        set_device(r))
        return -1;
    size_t nq = (size_t)p->n_tokens*p->n_heads*256;
    size_t nkv = (size_t)p->n_tokens*p->n_kv_heads*256;
    size_t cache = (size_t)p->cache_capacity*p->n_kv_heads*256;
    BnGPUResidentView views[] = {p->q,p->k,p->v,p->key_cache,p->value_cache,
                                p->attention_out,p->gate,p->out};
    size_t counts[] = {nq,nkv,nkv,cache,cache,nq,nq,nq};
    if (!p->gated && (p->gate.request || p->gate.count || !same(p->out,p->attention_out)))
        return -1;
    for (unsigned i=0; i<8; i++) {
        if (i==6 && !p->gated) continue;
        ResidentRequest *owner=r;
        if(activation_owner&&i!=3&&i!=4&&views[i].request==activation_owner)owner=activation_owner;
        data[i]=view_data(owner,views[i]);
        if (!data[i] || views[i].count!=counts[i] || !bn_quant_format_is_bf16(views[i].type) ||
            ((uintptr_t)data[i] & 15u))
            return -1;
    }
    const unsigned writes[] = {3,4,5,7};
    for (unsigned j=0;j<4;j++)
        for (unsigned i=0;i<8;i++) {
            unsigned o=writes[j];
            if (i==o || (i==6&&!p->gated) ||
                (!p->gated&&((i==5&&o==7)||(i==7&&o==5)))) continue;
            if (overlap(views[o],views[i])) return -1;
        }
    BnCudaFA2Call call = {};
    call.q=data[0];call.k=data[3];call.v=data[4];call.out=data[5];
    call.lse=r->attention_lse;call.partial_lse=r->attention_partial_lse;
    call.partial_out=r->attention_partial_out;call.n_tokens=p->n_tokens;
    call.n_heads=p->n_heads;call.n_kv_heads=p->n_kv_heads;
    call.kv_tokens=p->position+p->n_tokens;call.cache_capacity=p->cache_capacity;
    call.multiprocessors=r->attention_sms;call.scale=p->scale;call.stream=r->stream;
    *out_call=call;return 0;
}
static int execute_validated(ResidentRequest *r,const BnGPUAttentionResidentPlan *p,
    void **data,const BnCudaFA2Call *call) {
    size_t nq=(size_t)p->n_tokens*p->n_heads*256,nkv=(size_t)p->n_tokens*p->n_kv_heads*256;
    if (cudaGetLastError()!=cudaSuccess) { r->failed=1; return -1; }
    append<<<(unsigned)((nkv+255)/256),256,0,r->stream>>>(
        (uint16_t*)data[3],(uint16_t*)data[4],(const uint16_t*)data[1],
        (const uint16_t*)data[2],nkv,(size_t)p->position*p->n_kv_heads*256);
    if (cudaGetLastError()!=cudaSuccess || bn_cuda_fa2_forward(call)) {
        r->failed=1;return -1;
    }
    if (p->gated) {
        if (p->gate_profile == BN_GPU_ATTENTION_GATE_COMPILED_F32)
            bn_cuda_attention_gate::apply<true><<<(unsigned)((nq+255)/256),256,0,r->stream>>>(
                (__nv_bfloat16*)data[7],(const __nv_bfloat16*)data[5],
                (const __nv_bfloat16*)data[6],nq);
        else
            bn_cuda_attention_gate::apply<false><<<(unsigned)((nq+255)/256),256,0,r->stream>>>(
                (__nv_bfloat16*)data[7],(const __nv_bfloat16*)data[5],
                (const __nv_bfloat16*)data[6],nq);
    }
    if (cudaGetLastError()==cudaSuccess) return 0;
    r->failed=1;return -1;
}
static int execute(void *ctx,void *ptr,const BnGPUAttentionResidentPlan *p) {
    ResidentRequest *r=(ResidentRequest *)ptr;void *data[8]={};BnCudaFA2Call call;
    if(attention_validate(ctx,ptr,p,data,&call))return -1;
    return execute_validated(r,p,data,&call);
}
/* KV append preserves raw BF16 bits at each participant's own position. */
static __global__ void append_group(const BnCudaFA2DecodeRow *rows) {
    const BnCudaFA2DecodeRow &row=rows[blockIdx.y];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t n=(size_t)row.call.n_kv_heads*256,offset=(size_t)(row.call.kv_tokens-1)*n;
    if(i<n) {
        ((uint16_t*)row.call.k)[offset+i]=((const uint16_t*)row.kin)[i];
        ((uint16_t*)row.call.v)[offset+i]=((const uint16_t*)row.vin)[i];
    }
}
static __global__ void gate_group(const BnCudaFA2DecodeRow *rows) {
    const BnCudaFA2DecodeRow &row=rows[blockIdx.y];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(row.gate_profile<0||i>=(size_t)row.call.n_heads*256)return;
    __nv_bfloat16 x=((const __nv_bfloat16*)row.call.out)[i],g=((const __nv_bfloat16*)row.gate)[i];
    ((__nv_bfloat16*)row.gated_out)[i]=row.gate_profile==BN_GPU_ATTENTION_GATE_COMPILED_F32
        ?bn_cuda_attention_gate::element<true>(x,g):bn_cuda_attention_gate::element<false>(x,g);
}
static size_t decode_group_workspace_size(void *ctx,int capacity) {
    return bn_cuda_gdn::sequence_group_workspace_size<BnCudaFA2DecodeRow>(ctx,capacity);
}
static int decode_group_reserve(void *ctx,void *ptr,int capacity,size_t budget) {
    ResidentRequest *r=(ResidentRequest*)ptr;
    return r?bn_cuda_gdn::group_reserve(ctx,ptr,capacity,budget,&r->attention_group):-1;
}
static int decode_group(void *ctx,void *ptr,const BnGPUAttentionDecodeGroupPlan *p) {
    ResidentRequest *coordinator=(ResidentRequest*)ptr;
    if(!coordinator||coordinator->backend_owner!=ctx||coordinator->failed||!p||
       !p->sequences||p->n_sequences<=0||p->n_sequences>256||
       (p->split_policy!=BN_GPU_ATTENTION_SPLIT_SINGLETON&&p->split_policy!=BN_GPU_ATTENTION_SPLIT_BATCH)||
       !coordinator->attention_group||
       p->n_sequences>coordinator->attention_group->capacity)return -1;
    ResidentRequest *participants[256];BnCudaFA2DecodeRow rows[256];
    int heads=0,splits=0,combine=0,max_q=0,max_kv=0,gated=0;
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUAttentionDecodeSequence *s=&p->sequences[i];
        ResidentRequest *r=(ResidentRequest*)s->request;
        if(!r||r==coordinator||r->backend_owner!=ctx||r->device!=coordinator->device||s->plan.n_tokens!=1)return -1;
        for(int j=0;j<i;j++)if(participants[j]==r)return -1;
        void *data[8]={};BnCudaFA2DecodeRow *row=&rows[i];
        if(attention_validate(ctx,r,&s->plan,data,&row->call,coordinator))return -1;
        row->splits=bn_cuda_fa2_decode_splits(&row->call);if(row->splits<1)return -1;
        row->kin=data[1];row->vin=data[2];row->gate=data[6];row->gated_out=data[7];
        row->gate_profile=s->plan.gated?(int)s->plan.gate_profile:-1;
        participants[i]=r;int h=row->call.n_kv_heads;
        if(h>heads)heads=h;if(row->splits>splits)splits=row->splits;
        int blocks=(row->call.n_heads+3)/4;if(blocks>combine)combine=blocks;
        if(row->call.n_heads>max_q)max_q=row->call.n_heads;
        if(row->call.n_kv_heads>max_kv)max_kv=row->call.n_kv_heads;
        if(row->gate_profile>=0)gated=1;
    }
    if(p->split_policy==BN_GPU_ATTENTION_SPLIT_BATCH) {
        if(((BnCudaCtx*)ctx)->compute_capability!=1200)return -1;
        int maximum=0;
        for(int i=0;i<p->n_sequences;i++) {
            if(rows[i].call.n_heads!=rows[0].call.n_heads||
               rows[i].call.n_kv_heads!=rows[0].call.n_kv_heads)return -1;
            if(rows[i].call.kv_tokens>maximum)maximum=rows[i].call.kv_tokens;
        }
        splits=0;
        for(int i=0;i<p->n_sequences;i++) {
            rows[i].call.split_batch_size=p->n_sequences;
            rows[i].call.split_kv_tokens=maximum;
            rows[i].splits=bn_cuda_fa2_decode_splits(&rows[i].call);
            int reserved=(participants[i]->attention_context+63)/64;
            if(reserved>128)reserved=128;
            if(rows[i].splits<1||rows[i].splits>reserved)return -1;
            if(rows[i].splits>splits)splits=rows[i].splits;
        }
    }
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUAttentionResidentPlan *a=&p->sequences[i].plan;
        BnGPUResidentView writes[]={a->key_cache,a->value_cache,a->attention_out,a->out};
        for(int j=0;j<p->n_sequences;j++)if(i!=j) {
            const BnGPUAttentionResidentPlan *b=&p->sequences[j].plan;
            BnGPUResidentView all[]={b->q,b->k,b->v,b->key_cache,b->value_cache,b->attention_out,b->gate,b->out};
            for(int w=0;w<4;w++)for(int v=0;v<8;v++)if(overlap(writes[w],all[v]))return -1;
        }
    }
    bn_cuda_gdn::AttentionGroupWorkspace *w=coordinator->attention_group;
    int status=bn_cuda_gdn::group_prepare(coordinator,w,participants,rows,p->n_sequences);if(status)return status;
    append_group<<<dim3(max_kv,p->n_sequences),256,0,coordinator->stream>>>(w->device);
    if(cudaGetLastError()!=cudaSuccess||bn_cuda_fa2_decode_group(w->device,p->n_sequences,heads,splits,combine,coordinator->stream))
        return bn_cuda_gdn::group_fail(coordinator,participants,p->n_sequences);
    if(gated)gate_group<<<dim3(max_q,p->n_sequences),256,0,coordinator->stream>>>(w->device);
    return bn_cuda_gdn::group_finish(coordinator,w,participants,p->n_sequences);
}

static int prefill_group(void *ctx,void *ptr,const BnGPUAttentionPrefillGroupPlan *p) {
    ResidentRequest *coordinator=(ResidentRequest *)ptr;
    if(!coordinator || coordinator->backend_owner!=ctx || coordinator->failed || !p ||
       !p->sequences || p->n_sequences<1 || p->n_sequences>256 || !coordinator->attention_group ||
       p->n_sequences>coordinator->attention_group->capacity||
       (p->split_policy!=BN_GPU_ATTENTION_SPLIT_SINGLETON&&p->split_policy!=BN_GPU_ATTENTION_SPLIT_BATCH)||
       (p->split_policy==BN_GPU_ATTENTION_SPLIT_BATCH&&((BnCudaCtx*)ctx)->compute_capability!=1200))return -1;
    ResidentRequest *participants[256];BnCudaFA2DecodeRow rows[256]={};void *data[256][8]={};
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUAttentionPrefillSequence *s=&p->sequences[i];
        ResidentRequest *r=(ResidentRequest *)s->request;
        if(!r || r==coordinator || r->failed || r->backend_owner!=ctx || r->device!=coordinator->device)return -1;
        for(int j=0;j<i;j++)if(participants[j]==r)return -1;
        if(attention_validate(ctx,r,&s->plan,data[i],&rows[i].call,coordinator))return -1;
        participants[i]=r;rows[i].call.stream=coordinator->stream;
    }
    if(p->split_policy==BN_GPU_ATTENTION_SPLIT_BATCH) {
        int maximum_q=0,maximum_k=0;
        for(int i=0;i<p->n_sequences;i++) {
            if(rows[i].call.n_heads!=rows[0].call.n_heads||
               rows[i].call.n_kv_heads!=rows[0].call.n_kv_heads)return -1;
            if(rows[i].call.n_tokens>maximum_q)maximum_q=rows[i].call.n_tokens;
            if(rows[i].call.kv_tokens>maximum_k)maximum_k=rows[i].call.kv_tokens;
        }
        for(int i=0;i<p->n_sequences;i++) {
            rows[i].call.max_query_tokens=maximum_q;
            rows[i].call.split_batch_size=p->n_sequences;rows[i].call.split_kv_tokens=maximum_k;
            if(rows[i].call.n_tokens==1) {
                int splits=bn_cuda_fa2_decode_splits(&rows[i].call);
                int reserved=(participants[i]->attention_context+63)/64;if(reserved>128)reserved=128;
                if(splits<1||splits>reserved)return -1;
            }
        }
    }
    for(int i=0;i<p->n_sequences;i++) {
        const BnGPUAttentionResidentPlan *a=&p->sequences[i].plan;
        BnGPUResidentView writes[]={a->key_cache,a->value_cache,a->attention_out,a->out};
        for(int j=0;j<p->n_sequences;j++)if(i!=j) {
            const BnGPUAttentionResidentPlan *b=&p->sequences[j].plan;
            BnGPUResidentView all[]={b->q,b->k,b->v,b->key_cache,b->value_cache,b->attention_out,b->gate,b->out};
            for(int w=0;w<4;w++)for(int v=0;v<8;v++)if(overlap(writes[w],all[v]))return -1;
        }
    }
    bn_cuda_gdn::AttentionGroupWorkspace *workspace=coordinator->attention_group;
    int status=bn_cuda_gdn::group_prepare(coordinator,workspace,participants,rows,p->n_sequences);
    if(status)return status;
    for(int i=0;i<p->n_sequences;i++)
        if(execute_validated(coordinator,&p->sequences[i].plan,data[i],&rows[i].call))
            return bn_cuda_gdn::group_fail(coordinator,participants,p->n_sequences);
    return bn_cuda_gdn::group_finish(coordinator,workspace,participants,p->n_sequences);
}

static size_t group_queue_workspace_size(void *ctx,BnGPUResidentGroupKind kind,int capacity,int slots) {
    switch(kind) {
        case BN_GPU_RESIDENT_GROUP_GDN:case BN_GPU_RESIDENT_GROUP_CONV:
            return bn_cuda_gdn::sequence_group_workspace_size<BnCudaSequenceRow>(ctx,capacity,slots);
        case BN_GPU_RESIDENT_GROUP_QK_PREP:
            return bn_cuda_gdn::sequence_group_workspace_size<BnCudaQKRow>(ctx,capacity,slots);
        case BN_GPU_RESIDENT_GROUP_ATTENTION:
            return bn_cuda_gdn::sequence_group_workspace_size<BnCudaFA2DecodeRow>(ctx,capacity,slots);
        default:return 0;
    }
}
static int group_queue_reserve(void *ctx,void *ptr,BnGPUResidentGroupKind kind,int capacity,int slots,size_t budget) {
    ResidentRequest *r=(ResidentRequest*)ptr;if(!r)return -1;
    switch(kind) {
        case BN_GPU_RESIDENT_GROUP_GDN:return bn_cuda_gdn::group_reserve(ctx,ptr,capacity,budget,&r->decode_group,slots);
        case BN_GPU_RESIDENT_GROUP_CONV:return bn_cuda_gdn::group_reserve(ctx,ptr,capacity,budget,&r->conv_group,slots);
        case BN_GPU_RESIDENT_GROUP_QK_PREP:return bn_cuda_gdn::group_reserve(ctx,ptr,capacity,budget,&r->qk_group,slots);
        case BN_GPU_RESIDENT_GROUP_ATTENTION:return bn_cuda_gdn::group_reserve(ctx,ptr,capacity,budget,&r->attention_group,slots);
        default:return -1;
    }
}

}
#endif
