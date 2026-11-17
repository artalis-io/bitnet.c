#ifndef BN_GPU_CUDA_QSA_CUH
#define BN_GPU_CUDA_QSA_CUH

/* Qualified sequential-text QSA side branch; main attention and selection are
 * separate operators. Preserve pinned Triton reduction/rounding boundaries. */
namespace bn_cuda_qsa {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::set_device;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;

template<bool Compressed>
static __device__ __nv_bfloat16 rotary(const __nv_bfloat16 *x,
    const __nv_bfloat16 *cs, int i) {
    if (i >= 64) return x[i];
    int j = i & 31;
    __nv_bfloat16 a = x[j], b = x[j+32], c = cs[j], s = cs[j+32];
    /* The pinned Q/K register layouts contract opposite products in the
     * second half. Preserve their distinct BF16 multiply/FMA boundaries. */
    return i < 32 ? __hfma(a,c,__hneg(__hmul(b,s))) :
        (Compressed ? __hfma(a,s,__hmul(b,c)) : __hfma(b,c,__hmul(a,s)));
}

static __device__ void query(const __nv_bfloat16 *projected,
    const __nv_bfloat16 *weight, const __nv_bfloat16 *cs,
    __nv_bfloat16 *out, int token, int pair, int position, float eps) {
    int tid = threadIdx.x, lane = tid & 15, head = pair * 2 + tid / 16;
    size_t base = (size_t)token * 640 + head * 128;
    float x[8], square[8];
    for (int j = 0; j < 8; ++j) {
        x[j] = __bfloat162float(projected[base + lane*8+j]);
        square[j] = __fmul_rn(x[j],x[j]);
    }
    float even = __fadd_rn(__fadd_rn(square[0],square[2]),__fadd_rn(square[4],square[6]));
    float odd = __fadd_rn(__fadd_rn(square[1],square[3]),__fadd_rn(square[5],square[7]));
    float sum = __fadd_rn(even,odd);
    for (int step = 8; step; step >>= 1)
        sum = __fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,step));
    float scale = rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/128),eps));
    __shared__ __nv_bfloat16 normalized[256];
    int local = tid / 16 * 128;
    for (int j = 0; j < 8; ++j) {
        int i = lane*8+j;
        float gamma = __fadd_rn(__bfloat162float(weight[i]),1.f);
        normalized[local+i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[j],scale),gamma));
    }
    __syncthreads();
    for (int j = 0; j < 8; ++j) {
        int i = lane*8+j;
        out[((size_t)token*4+head)*128+i] =
            rotary<false>(normalized+local,cs+(size_t)(position+token)*64,i);
    }
}

static __device__ void compress(const __nv_bfloat16 *projected,
    const __nv_bfloat16 *history, const __nv_bfloat16 *weight,
    const __nv_bfloat16 *cs, __nv_bfloat16 *out, int group, int position, float eps) {
    int lane = threadIdx.x, first = (position/4 + group)*4;
    float x[4];
    for (int j = 0; j < 4; ++j) {
        int d = lane*4+j;
        float values[4];
        for (int k = 0; k < 4; ++k) {
            int p = first+k;
            values[k] = __bfloat162float(p < position ? history[(p&3)*128+d] :
                projected[(size_t)(p-position)*640+512+d]);
        }
        float pooled = __fmul_rn(__fadd_rn(__fadd_rn(values[0],values[1]),
            __fadd_rn(values[2],values[3])),.25f);
        x[j] = __bfloat162float(__float2bfloat16_rn(pooled));
    }
    float sum = __fadd_rn(__fmaf_rn(x[0],x[0],__fmul_rn(x[2],x[2])),
        __fmaf_rn(x[1],x[1],__fmul_rn(x[3],x[3])));
    for (int step = 16; step; step >>= 1)
        sum = __fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,step));
    float scale = rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/128),eps));
    __shared__ __nv_bfloat16 normalized[128];
    for (int j = 0; j < 4; ++j) {
        int d = lane*4+j;
        float gamma = __fadd_rn(__bfloat162float(weight[d]),1.f);
        normalized[d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[j],scale),gamma));
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
        int d = lane*4+j;
        out[(size_t)group*128+d] = rotary<true>(normalized,cs+(size_t)first*64,d);
    }
}

/* Every CTA reads the immutable initial ring. CTA zero may therefore write
 * the separate next ring while other CTAs compress historical groups. */
static __global__ void indexer_kernel(const __nv_bfloat16 *projected,
    const __nv_bfloat16 *initial, const __nv_bfloat16 *qw, const __nv_bfloat16 *kw,
    const __nv_bfloat16 *cs, __nv_bfloat16 *q, __nv_bfloat16 *compressed,
    __nv_bfloat16 *next, int position, int tokens, float eps) {
    int block = blockIdx.x;
    if (block < tokens*2)
        query(projected,qw,cs,q,block/2,block%2,position,eps);
    else
        compress(projected,initial,kw,cs,compressed,block-tokens*2,position,eps);
    if (block == 0) {
        int end = position+tokens;
        for (int i = threadIdx.x; i < 512; i += 32) {
            int slot = i/128, latest = end-1-(int)((unsigned)(end-1-slot)&3u);
            next[i] = latest >= position ?
                projected[(size_t)(latest-position)*640+512+i%128] : initial[i];
        }
    }
}

static int indexer(void *ctx_ptr, void *ptr, const BnGPUQSAIndexerResidentPlan *p) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    ResidentRequest *r = (ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx || ctx->compute_capability < 800 ||
        p->n_tokens <= 0 || p->n_tokens > INT_MAX/640 || p->position < 0 ||
        p->position > INT_MAX-p->n_tokens || !isfinite(p->eps) || p->eps <= 0) return -1;
    const BnCudaBuffer *qw = (const BnCudaBuffer *)p->q_weight;
    const BnCudaBuffer *kw = (const BnCudaBuffer *)p->k_weight;
    const BnCudaBuffer *cs = (const BnCudaBuffer *)p->cos_sin;
    if (!qw || !kw || !cs || !qw->data || !kw->data || !cs->data ||
        !bn_quant_format_is_bf16(qw->type) || !bn_quant_format_is_bf16(kw->type) ||
        !bn_quant_format_is_bf16(cs->type) || qw->rows != 1 || kw->rows != 1 ||
        qw->cols != 128 || kw->cols != 128 || qw->size != 256 || kw->size != 256 ||
        cs->rows < p->position+p->n_tokens || cs->cols != 64 ||
        cs->size/128 < (size_t)(p->position+p->n_tokens)) return -1;
    int groups = (p->position+p->n_tokens)/4-p->position/4;
    if (!groups && (p->compressed.request || p->compressed.count)) return -1;
    BnGPUResidentView views[] = {p->projected,p->initial_history,p->next_history,p->q,p->compressed};
    size_t counts[] = {(size_t)p->n_tokens*640,512,512,(size_t)p->n_tokens*512,(size_t)groups*128};
    void *data[5] = {};
    for (int i = 0; i < (groups ? 5 : 4); ++i) {
        data[i] = view_data(r,views[i]);
        if (!data[i] || views[i].count != counts[i] || !bn_quant_format_is_bf16(views[i].type) ||
            ((uintptr_t)data[i]&15u)) return -1;
        for (int j = 0; j < i; ++j) if (overlap(views[i],views[j])) return -1;
    }
    if (set_device(r)) return -1;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    indexer_kernel<<<p->n_tokens*2+groups,32,0,r->stream>>>(
        (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],
        (const __nv_bfloat16 *)qw->data,(const __nv_bfloat16 *)kw->data,
        (const __nv_bfloat16 *)cs->data,(__nv_bfloat16 *)data[3],
        (__nv_bfloat16 *)data[4],(__nv_bfloat16 *)data[2],p->position,p->n_tokens,p->eps);
    if (cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

static __device__ unsigned score_pair(const __nv_bfloat16 *p, bool valid) {
    return valid ? *(const unsigned *)p : 0;
}

/* One warp scores sixteen keys against two queries (four heads each). The
 * same m16n8k16 BF16/F32 MMA reduction is used by the pinned SM120 reference. */
static __global__ void score_kernel(const __nv_bfloat16 *q,
    const __nv_bfloat16 *keys, float *scores, int tokens, int position, int groups) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x*64+warp*16+lane/4, k_lane = (lane&3)*2;
    int query_base = blockIdx.y*2, col = lane/4, qt = query_base+col/4;
    int max_visible = (position+tokens)/4;
    float d[4] = {};
#pragma unroll
    for (int k = 0; k < 128; k += 16) {
        unsigned a0 = score_pair(keys+(size_t)row*128+k+k_lane,row<max_visible);
        unsigned a1 = score_pair(keys+(size_t)(row+8)*128+k+k_lane,row+8<max_visible);
        unsigned a2 = score_pair(keys+(size_t)row*128+k+8+k_lane,row<max_visible);
        unsigned a3 = score_pair(keys+(size_t)(row+8)*128+k+8+k_lane,row+8<max_visible);
        const __nv_bfloat16 *b = q+(size_t)qt*512+(col&3)*128+k+k_lane;
        unsigned b0 = score_pair(b,qt<tokens), b1 = score_pair(b+8,qt<tokens);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        /* Triton reduces each lane's adjacent head pair before the XOR
         * exchange. Summing even/odd heads first changes F32 rounding. */
        float sum = __fadd_rn(fmaxf(d[i],0.f),fmaxf(d[i+1],0.f));
        sum = __fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,1));
        int out_row = row+(i/2)*8, out_query = query_base+(lane&3)/2;
        if (!(lane&1) && out_row<groups && out_query<tokens)
            scores[(size_t)out_query*groups+out_row] =
                out_row<(position+out_query+1)/4 ? sum : -INFINITY;
    }
}

static __device__ unsigned selection_key(float value) {
    unsigned bits=__float_as_uint(value);
    /* Both signed zeros are the same score; preserve positive subnormals. */
    return (bits&0x7fffffffu)==0 ? 0 : bits;
}

/* Stable exact radix threshold, followed by chronological compaction. Integer
 * histograms and prefix counts make ties independent of warp arrival order. */
static __global__ void select_kernel(const float *scores, int32_t *out,
    int position, int groups) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid&31, warp = tid>>5;
    int end = position+row+1, visible = end/4, tail = end&3;
    int32_t *dst = out+(size_t)row*2052;
    for (int i = tid; i < 2052; i += 256) dst[i] = -1;
    __syncthreads();
    if (visible <= 512) {
        for (int i = tid; i < end; i += 256) dst[i] = i;
        if (!tid) dst[2051] = end;
        return;
    }
    const float *src = scores+(size_t)row*groups;
    __shared__ unsigned histogram[256], prefix, mask;
    __shared__ int remaining, done, running_ties, running_selected;
    __shared__ int ties[8], elected[8];
    if (!tid) { prefix=mask=0; remaining=512; done=0; running_ties=running_selected=0; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[tid] = 0;
        __syncthreads();
        for (int base = 0; base < visible; base += 256) {
            int i = base+tid;
            unsigned key = i<visible ? selection_key(src[i]) : 0;
            bool valid = i<visible && (key&mask)==prefix;
            unsigned active = __ballot_sync(0xffffffffu,valid);
            if (valid) {
                unsigned bin = (key>>shift)&255u;
                unsigned peers = __match_any_sync(active,bin);
                if (lane==__ffs(peers)-1) atomicAdd(histogram+bin,(unsigned)__popc(peers));
            }
        }
        __syncthreads();
        if (!tid) {
            int need = remaining;
            for (int bin = 255; bin >= 0; --bin) {
                int n = histogram[bin];
                if (need > n) { need-=n; continue; }
                prefix |= (unsigned)bin<<shift; remaining=need;
                mask |= 255u<<shift; done=(need==n); break;
            }
        }
        __syncthreads();
        if (done) break;
    }
    for (int base = 0; base < visible; base += 256) {
        int i = base+tid;
        unsigned key = i<visible ? selection_key(src[i]) : 0;
        bool tie = i<visible && key==prefix;
        unsigned tie_bits = __ballot_sync(0xffffffffu,tie);
        if (!lane) ties[warp] = __popc(tie_bits);
        __syncthreads();
        int tie_before = running_ties;
        for (int w = 0; w < warp; ++w) tie_before+=ties[w];
        int local = __popc(tie_bits&((1u<<lane)-1));
        bool choose = i<visible && (key>prefix || (tie && tie_before+local<remaining));
        unsigned bits = __ballot_sync(0xffffffffu,choose);
        if (!lane) elected[warp] = __popc(bits);
        __syncthreads();
        int before = running_selected;
        for (int w = 0; w < warp; ++w) before+=elected[w];
        int rank = before+__popc(bits&((1u<<lane)-1));
        if (choose && rank<512) for (int j = 0; j < 4; ++j) dst[rank*4+j]=i*4+j;
        __syncthreads();
        if (!tid) for (int w = 0; w < 8; ++w) { running_ties+=ties[w]; running_selected+=elected[w]; }
        __syncthreads();
        if (running_selected>=512) break;
    }
    for (int j = tid; j < tail; j += 256) dst[2048+j]=visible*4+j;
    if (!tid) dst[2051]=2048+tail;
}

static int selection_geometry(const BnCudaCtx *ctx, const ResidentRequest *r,
    int tokens, int position, int groups) {
    return ctx && r && r->backend_owner==ctx && !r->failed && ctx->compute_capability>=800 &&
        tokens>0 && tokens<=131070 && position>=0 && position<=INT_MAX-tokens &&
        groups>0 && groups<=65536 && (position+tokens)/4<=groups;
}

static int score(void *ctx_ptr, void *ptr, const BnGPUQSAScoreResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx *)ctx_ptr; ResidentRequest *r=(ResidentRequest *)ptr;
    if (!p || !selection_geometry(ctx,r,p->n_tokens,p->position,p->cache_groups)) return -1;
    BnGPUResidentView views[]={p->q,p->compressed,p->scores};
    size_t counts[]={(size_t)p->n_tokens*512,(size_t)p->cache_groups*128,(size_t)p->n_tokens*p->cache_groups};
    void *data[3]={};
    for (int i=0;i<3;++i) {
        data[i]=view_data(r,views[i]);
        if (!data[i] || views[i].count!=counts[i] || ((uintptr_t)data[i]&15u) ||
            (i<2 ? !bn_quant_format_is_bf16(views[i].type) : !bn_quant_format_is_f32(views[i].type))) return -1;
        for (int j=0;j<i;++j) if (overlap(views[i],views[j])) return -1;
    }
    if (set_device(r)) return -1;
    if (cudaGetLastError()!=cudaSuccess) { r->failed=1; return -1; }
    score_kernel<<<dim3((p->cache_groups+63)/64,(p->n_tokens+1)/2),128,0,r->stream>>>(
        (const __nv_bfloat16 *)data[0],(const __nv_bfloat16 *)data[1],(float *)data[2],
        p->n_tokens,p->position,p->cache_groups);
    if (cudaGetLastError()==cudaSuccess) return 0;
    r->failed=1; return -1;
}

static int select(void *ctx_ptr, void *ptr, const BnGPUQSASelectResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx *)ctx_ptr; ResidentRequest *r=(ResidentRequest *)ptr;
    if (!p || !selection_geometry(ctx,r,p->n_tokens,p->position,p->cache_groups)) return -1;
    void *scores=view_data(r,p->scores), *out=view_data(r,p->out);
    if (!scores || !out || !bn_quant_format_is_f32(p->scores.type) || p->out.type!=BN_GGUF_TENSOR_I32 ||
        p->scores.count!=(size_t)p->n_tokens*p->cache_groups ||
        p->out.count!=(size_t)p->n_tokens*2052 || ((uintptr_t)scores&15u) || ((uintptr_t)out&15u) ||
        overlap(p->scores,p->out)) return -1;
    if (set_device(r)) return -1;
    if (cudaGetLastError()!=cudaSuccess) { r->failed=1; return -1; }
    select_kernel<<<p->n_tokens,256,0,r->stream>>>((const float *)scores,(int32_t *)out,
        p->position,p->cache_groups);
    if (cudaGetLastError()==cudaSuccess) return 0;
    r->failed=1; return -1;
}

} // namespace bn_cuda_qsa
#endif
