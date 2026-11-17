#ifndef BN_GPU_CUDA_GLM_INDEXER_CUH
#define BN_GPU_CUDA_GLM_INDEXER_CUH
#include "quant/fp8_e4m3_cuda.cuh"

namespace bn_cuda_glm_indexer {
using bn_cuda_gdn::ResidentRequest;
using bn_cuda_gdn::view_data;
using bn_cuda_gdn::overlap;
using bn_cuda_gdn::set_device;

static __global__ void fwht_kernel(const __nv_bfloat16 *x,unsigned char *out,float *scale) {
    float v[4];int row=blockIdx.x,lane=threadIdx.x;
    for(int j=0;j<4;++j)v[j]=__bfloat162float(x[(size_t)row*128+lane+j*32]);
    bn_cuda_fp8::hadamard128_bf16(v);bn_cuda_fp8::quantize128(v,out,scale,row);
}
static int valid_views(ResidentRequest *r,const BnGPUResidentView *v,
    const size_t *counts,const int *types,int n,void **data) {
    for(int i=0;i<n;++i) {
        if(!counts[i]){if(v[i].request||v[i].count)return -1;data[i]=NULL;continue;}
        data[i]=view_data(r,v[i]);
        if(!data[i]||v[i].count!=counts[i]||v[i].type!=types[i])return -1;
        for(int j=0;j<i;++j)if(counts[j]&&overlap(v[i],v[j]))return -1;
    }
    return 0;
}
/* Ordered finite F32 keys, including negatives and subnormals. */
static __device__ unsigned signed_selection_key(float value) {
    unsigned bits=__float_as_uint(value);
    if(!(bits&0x7fffffffu))bits=0; /* +/-0 are equal scores. */
    return bits&0x80000000u?~bits:bits^0x80000000u;
}
static __global__ void select_kernel(const float *scores, int32_t *pools, int32_t *out, int32_t *status,
    int position, int groups, int short_prefill) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid&31, warp = tid>>5;
    int end = position+row+1, visible = end/4, tail = end&3;
    int32_t *dst = out+(size_t)row*2051;
    int32_t *pool_row=pools+(size_t)row*512;
    for(int i=tid;i<2051;i+=256)dst[i]=-1;
    for(int i=tid;i<512;i+=256)pool_row[i]=-1;
    __shared__ int invalid;
    if(!tid)invalid=0;
    __syncthreads();
    const float *src=!short_prefill&&groups?scores+(size_t)row*groups:NULL;
    if(!short_prefill)for(int i=tid;i<visible;i+=256)if(!isfinite(src[i]))atomicExch(&invalid,1);
    __syncthreads();
    if(invalid){if(!tid)atomicExch(status,1);return;}
    if(visible<=512){
        for(int i=tid;i<visible;i+=256){pool_row[i]=i;if(!short_prefill)for(int j=0;j<4;++j)dst[i*4+j]=i*4+j;}
        if(short_prefill){for(int i=tid;i<end;i+=256)dst[i]=i;}
        else for(int j=tid;j<tail;j+=256)dst[2048+j]=visible*4+j;
        return;
    }
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
            unsigned key = i<visible ? signed_selection_key(src[i]) : 0;
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
        unsigned key = i<visible ? signed_selection_key(src[i]) : 0;
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
        if(choose&&rank<512){pool_row[rank]=i;for(int j=0;j<4;++j)dst[rank*4+j]=i*4+j;}
        __syncthreads();
        if (!tid) for (int w = 0; w < 8; ++w) { running_ties+=ties[w]; running_selected+=elected[w]; }
        __syncthreads();
        if (running_selected>=512) break;
    }
    for (int j = tid; j < tail; j += 256) dst[2048+j]=visible*4+j;
}


static int select(void *ctx_ptr,void *ptr,const BnGPUIndexerSelectResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->n_tokens<1||p->n_tokens>INT_MAX/2051||p->position<0||p->position>INT_MAX-p->n_tokens||
       p->n_keys!=(p->position+p->n_tokens)/4||p->n_keys>INT_MAX/128||
       (p->short_prefill!=0&&p->short_prefill!=1)||(p->short_prefill&&p->position+p->n_tokens>2048)||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens,k=(size_t)p->n_keys;
    if(k&&n>SIZE_MAX/sizeof(float)/k)return -1;
    BnGPUResidentView v[]={p->scores,p->pools,p->out,p->status};
    size_t counts[]={p->short_prefill?0:n*k,n*512,n*2051,1};
    int types[]={BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32};void *data[4];
    if(valid_views(r,v,counts,types,4,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    select_kernel<<<p->n_tokens,256,0,r->stream>>>((const float*)data[0],(int32_t*)data[1],
        (int32_t*)data[2],(int32_t*)data[3],p->position,p->n_keys,p->short_prefill);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

/* One warp computes a 16-key x 32-head tile using four m16n8k32 MMAs
 * per head tile. FP8 register packing follows row/column PTX fragment layout.
 * Keep head-pair FMA order, key scaling before XOR-1/XOR-2 reductions, and
 * signed weights: these are observable in the original SM120 scorer. */
template<bool Paged>
static __global__ void score_kernel(const unsigned char*q,const unsigned char*k,const float*s,const float*w,float*out,int nk,int position,const int32_t *table,int blocks,int32_t *status){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
 int tiles=(nk+63)/64,qi=blockIdx.x/tiles,lane=threadIdx.x&31;
 int g=lane/4,t=lane%4,base=((blockIdx.x%tiles)*4+threadIdx.x/32)*16;
 const unsigned char *key_base=k;const float *key_scales=s;
 if(Paged){
    int logical=blockIdx.x%tiles,page=table[logical];
    if(page<0||page>=blocks){
        if(!threadIdx.x&&logical*64<(position+qi+1)/4)atomicExch(status,1);
        int col=logical*64+threadIdx.x;
        if(threadIdx.x<64&&col<nk)out[(size_t)qi*nk+col]=-INFINITY;
        return;
    }
    key_base=k+(size_t)page*8448;key_scales=(const float*)(key_base+8192);
 }
 float p0=0,p1=0;
 for(int nt=0;nt<4;++nt){float d[4]={0,0,0,0};
 for(int ks=0;ks<128;ks+=32){
 uint32_t a[4]={0,0,0,0},b[2];
 for(int j=0;j<4;++j){int row=base+g+(j%2)*8,off=ks+t*4+(j/2)*16;if(row<nk)a[j]=*(const uint32_t*)(key_base+(size_t)(Paged?row%64:row)*128+off);}
 for(int j=0;j<2;++j)b[j]=*(const uint32_t*)(q+((size_t)qi*32+nt*8+g)*128+ks+t*4+j*16);
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}
 float w0=w[(size_t)qi*32+nt*8+t*2],w1=w[(size_t)qi*32+nt*8+t*2+1];
 p0+=fmaxf(d[0],0.f)*w0+fmaxf(d[1],0.f)*w1;p1+=fmaxf(d[2],0.f)*w0+fmaxf(d[3],0.f)*w1;
 }
 float v0=p0*(base+g<nk?key_scales[Paged?(base+g)%64:base+g]:0),v1=p1*(base+g+8<nk?key_scales[Paged?(base+g+8)%64:base+g+8]:0);
 for(int off=1;off<=2;off*=2){v0+=__shfl_xor_sync(0xffffffff,v0,off);v1+=__shfl_xor_sync(0xffffffff,v1,off);}
 if(!t){for(int j=0;j<2;++j){int row=base+g+j*8;if(row<nk)out[(size_t)qi*nk+row]=row<(position+qi+1)/4?(j?v1:v0):-INFINITY;}}
#endif
}

static int score(void *ctx_ptr,void *ptr,const BnGPUIndexerFP8ScoreResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->n_tokens<1||p->n_tokens>INT_MAX/4096||p->position<0||
       p->position>INT_MAX-p->n_tokens||p->n_keys!=(p->position+p->n_tokens)/4||
       p->n_keys>INT_MAX/128||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens,k=(size_t)p->n_keys;
    if(k&&(n>SIZE_MAX/sizeof(float)/k||n>(size_t)INT_MAX/((k+63)/64)))return -1;
    BnGPUResidentView v[]={p->q,p->keys,p->scales,p->weights,p->scores};
    size_t counts[]={n*4096,k*128,k,n*32,n*k};
    int types[]={BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,
        BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};void *data[5];
    if(valid_views(r,v,counts,types,5,data)||((uintptr_t)data[0]&3u)||
       ((uintptr_t)data[1]&3u))return -1;
    if(!k)return 0;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    score_kernel<false><<<(unsigned)(n*((k+63)/64)),128,0,r->stream>>>(
        (const unsigned char*)data[0],(const unsigned char*)data[1],
        (const float*)data[2],(const float*)data[3],(float*)data[4],p->n_keys,p->position,NULL,0,NULL);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}


static int paged_score(void *ctx_ptr,void *ptr,const BnGPUIndexerFP8PagedScoreResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->n_tokens<1||p->n_tokens>INT_MAX/4096||p->position<0||p->position>INT_MAX-p->n_tokens||
       p->n_keys!=(p->position+p->n_tokens)/4||p->n_keys>INT_MAX/128||
       p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||(size_t)p->num_cache_blocks>SIZE_MAX/8448||
       p->logical_blocks<1||p->logical_blocks<(p->n_keys+63)/64||set_device(r))return -1;
    size_t n=(size_t)p->n_tokens,k=(size_t)p->n_keys;
    if(k&&(n>SIZE_MAX/sizeof(float)/k||n>(size_t)INT_MAX/((k+63)/64)))return -1;
    BnGPUResidentView v[]={p->q,p->weights,p->cache,p->block_table,p->scores,p->status};
    size_t counts[]={n*4096,n*32,(size_t)p->num_cache_blocks*8448,(size_t)p->logical_blocks,n*k,1};
    int types[]={BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32};void *data[6];
    if(valid_views(r,v,counts,types,6,data)||((uintptr_t)data[0]&3u)||((uintptr_t)data[2]&3u))return -1;
    if(!k)return 0;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    score_kernel<true><<<(unsigned)(n*((k+63)/64)),128,0,r->stream>>>((const unsigned char*)data[0],
        (const unsigned char*)data[2],NULL,(const float*)data[1],(float*)data[4],p->n_keys,p->position,
        (const int32_t*)data[3],p->num_cache_blocks,(int32_t*)data[5]);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}

static __global__ void cache_append_kernel(const unsigned char *values,
    const uint32_t *scales,const int32_t *slots,unsigned char *cache,int capacity) {
    int row=blockIdx.x,lane=threadIdx.x,slot=slots[row];
    if(slot<0||slot>=capacity)return;
    size_t base=(size_t)(slot/64)*8448;int offset=slot%64;
    ((uint32_t*)(cache+base+(size_t)offset*128))[lane]=((const uint32_t*)values)[(size_t)row*32+lane];
    if(!lane)((uint32_t*)(cache+base+8192))[offset]=scales[row];
}
static int cache_append(void *ctx_ptr,void *ptr,const BnGPUIndexerFP8AppendResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->n_rows<0||
       p->n_rows>INT_MAX/128||p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||
       (size_t)p->num_cache_blocks>SIZE_MAX/8448||set_device(r))return -1;
    size_t n=(size_t)p->n_rows;BnGPUResidentView v[]={p->values,p->scales,p->slots,p->cache};
    size_t counts[]={n*128,n,n,(size_t)p->num_cache_blocks*8448};
    int types[]={BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8};void *data[4];
    if(valid_views(r,v,counts,types,4,data)||((uintptr_t)data[3]&3u)||((uintptr_t)data[0]&3u))return -1;
    if(!n)return 0;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    cache_append_kernel<<<p->n_rows,32,0,r->stream>>>((const unsigned char*)data[0],
        (const uint32_t*)data[1],(const int32_t*)data[2],(unsigned char*)data[3],p->num_cache_blocks*64);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static __global__ void cache_gather_kernel(const unsigned char *cache,
    const int32_t *table,unsigned char *out,uint32_t *scales,int32_t *status,int blocks) {
    int row=blockIdx.x,lane=threadIdx.x,page=table[row/64],offset=row%64;
    if(page<0||page>=blocks) {
        ((uint32_t*)out)[(size_t)row*32+lane]=0;
        if(!lane){scales[row]=0;atomicExch(status,1);}return;
    }
    size_t base=(size_t)page*8448;
    ((uint32_t*)out)[(size_t)row*32+lane]=((const uint32_t*)(cache+base+(size_t)offset*128))[lane];
    if(!lane)scales[row]=((const uint32_t*)(cache+base+8192))[offset];
}
static int cache_gather(void *ctx_ptr,void *ptr,const BnGPUIndexerFP8GatherResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->n_rows<0||
       p->n_rows>INT_MAX/128||p->num_cache_blocks<1||p->num_cache_blocks>INT_MAX/64||
       (size_t)p->num_cache_blocks>SIZE_MAX/8448||p->logical_blocks<1||
       p->logical_blocks<(p->n_rows+63)/64||set_device(r))return -1;
    size_t n=(size_t)p->n_rows;BnGPUResidentView v[]={p->cache,p->block_table,p->out,p->scales,p->status};
    size_t counts[]={(size_t)p->num_cache_blocks*8448,(size_t)p->logical_blocks,n*128,n,1};
    int types[]={BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32};void *data[5];
    if(valid_views(r,v,counts,types,5,data)||((uintptr_t)data[0]&3u)||((uintptr_t)data[2]&3u))return -1;
    if(!n)return 0;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    cache_gather_kernel<<<p->n_rows,32,0,r->stream>>>((const unsigned char*)data[0],
        (const int32_t*)data[1],(unsigned char*)data[2],(uint32_t*)data[3],(int32_t*)data[4],p->num_cache_blocks);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static int fwht(void *ctx_ptr,void *ptr,const BnGPUFWHT128FP8ResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->rows<1||p->rows>INT_MAX/128||set_device(r))return -1;
    BnGPUResidentView v[]={p->x,p->out,p->scales};size_t n=(size_t)p->rows,counts[]={n*128,n*128,n};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32};void *data[3];
    if(valid_views(r,v,counts,types,3,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    fwht_kernel<<<p->rows,32,0,r->stream>>>((const __nv_bfloat16*)data[0],(unsigned char*)data[1],(float*)data[2]);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
/* Preserve both compiled reduction layouts. Batched input pairs features
 * 64 apart; single-row input pairs adjacent features. Alternating feature
 * groups retain opposite final merge orientations instead of broadcasting
 * one canonical root: the original F32 rounding can distinguish them. */
static __device__ void welford_combine(float &mean,float &m2,
    float other,float other_m2,float weight) {
    float delta=mean==other?0.f:__fsub_rn(other,mean);
    mean=fmaf(delta,.5f,mean);
    m2=fmaf(__fmul_rn(__fmul_rn(delta,delta),weight),.5f,__fadd_rn(m2,other_m2));
}
static __global__ void layernorm_kernel(const __nv_bfloat16 *x,const float *w,
    const float *b,__nv_bfloat16 *out,float eps,int rows) {
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,row=blockIdx.x;
    int i0=rows==1?tid*2:tid,i1=rows==1?tid*2+1:tid+64;
    float x0=__bfloat162float(x[row*128+i0]),x1=__bfloat162float(x[row*128+i1]);
    float mean=x0,m2=0;welford_combine(mean,m2,x1,0,1);float weight=2;
    for(int off=16;off;off>>=1) {
        float om=__shfl_xor_sync(0xffffffff,mean,off),ov=__shfl_xor_sync(0xffffffff,m2,off);
        welford_combine(mean,m2,om,ov,weight);weight*=2;
    }
    __shared__ float sm[2],sv[2];
    if(!lane){sm[warp]=mean;sv[warp]=m2;}
    __syncthreads();
    int root=tid&1;mean=sm[root];m2=sv[root];
    welford_combine(mean,m2,sm[root^1],sv[root^1],64);
    float variance=__fadd_rn(__fmul_rn(m2,1.f/128),eps),rs;
    asm("rsqrt.approx.ftz.f32 %0, %1;":"=f"(rs):"f"(variance));
    out[row*128+i0]=__float2bfloat16_rn(fmaf(__fmul_rn(__fsub_rn(x0,mean),rs),w[i0],b[i0]));
    out[row*128+i1]=__float2bfloat16_rn(fmaf(__fmul_rn(__fsub_rn(x1,mean),rs),w[i1],b[i1]));
}
static int layernorm(void *ctx_ptr,void *ptr,const BnGPULayerNorm128F32BF16ResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||
       p->n_tokens<1||p->n_tokens>INT_MAX/128||!isfinite(p->eps)||p->eps<=0||set_device(r))return -1;
    const BnCudaBuffer *w=(const BnCudaBuffer*)p->weight,*b=(const BnCudaBuffer*)p->bias;
    if(!w||!b||!w->data||!b->data||w->type!=BN_GGUF_TENSOR_F32||b->type!=BN_GGUF_TENSOR_F32||
       w->rows!=1||b->rows!=1||w->cols!=128||b->cols!=128||w->size!=512||b->size!=512)return -1;
    BnGPUResidentView v[]={p->x,p->out};size_t n=(size_t)p->n_tokens*128,counts[]={n,n};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16};void *data[2];
    if(valid_views(r,v,counts,types,2,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    layernorm_kernel<<<p->n_tokens,64,0,r->stream>>>((const __nv_bfloat16*)data[0],
        (const float*)w->data,(const float*)b->data,(__nv_bfloat16*)data[1],p->eps,p->n_tokens);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static __device__ float source_value(const __nv_bfloat16 *x,const __nv_bfloat16 *tail,
    int relative,int slot,int d) {
    return __bfloat162float(relative>=0?x[(size_t)relative*128+d]:tail[slot*128+d]);
}
static __global__ void pool_kernel(const __nv_bfloat16 *key,const __nv_bfloat16 *gate,
    const __nv_bfloat16 *tail,const float *ape,unsigned char *out,float *scale,int position) {
    int row=blockIdx.x,lane=threadIdx.x,start=(position/4+row)*4-position;float v[4];
    for(int j=0;j<4;++j) {
        int d=lane+j*32;float mx=-INFINITY;
        for(int s=0;s<4;++s) {
            float score=__fadd_rn(source_value(gate,tail+512,start+s,s,d),ape[s*128+d]);
            mx=fmaxf(mx,score);
        }
        float acc=0,denom=0;
        for(int s=0;s<4;++s) {
            float score=__fadd_rn(source_value(gate,tail+512,start+s,s,d),ape[s*128+d]);
            float prob=bn_cuda_gdn::ex(__fsub_rn(score,mx));denom=__fadd_rn(denom,prob);
            acc=fmaf(source_value(key,tail,start+s,s,d),prob,acc);
        }
        float q;asm("div.full.f32 %0, %1, %2;":"=f"(q):"f"(acc),"f"(denom));
        v[j]=__bfloat162float(__float2bfloat16_rn(q));
    }
    bn_cuda_fp8::hadamard128_bf16(v);bn_cuda_fp8::quantize128(v,out,scale,row);
}
static __global__ void tail_kernel(const __nv_bfloat16 *key,const __nv_bfloat16 *gate,
    const __nv_bfloat16 *initial,__nv_bfloat16 *next,int position,int nt) {
    int d=threadIdx.x;
    for(int s=0;s<4;++s) {
        int last=position+nt-1,logical=last-((last-s)&3),relative=logical-position;
        next[s*128+d]=relative>=0?key[(size_t)relative*128+d]:initial[s*128+d];
        next[512+s*128+d]=relative>=0?gate[(size_t)relative*128+d]:initial[512+s*128+d];
    }
}
static int pool(void *ctx_ptr,void *ptr,const BnGPUKPoolFP8ResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->n_tokens<1||
       p->n_tokens>INT_MAX/128||p->position<0||p->position>INT_MAX-p->n_tokens||set_device(r))return -1;
    const BnCudaBuffer *ape=(const BnCudaBuffer*)p->ape;
    if(!ape||!ape->data||ape->type!=BN_GGUF_TENSOR_F32||ape->rows!=4||ape->cols!=128||ape->size!=2048)return -1;
    size_t n=(size_t)p->n_tokens,groups=(p->position+p->n_tokens)/4-p->position/4;
    BnGPUResidentView v[]={p->key,p->gate,p->initial_tail,p->next_tail,p->out,p->scales};
    size_t counts[]={n*128,n*128,1024,1024,groups*128,groups};
    int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32};
    void *data[6];if(valid_views(r,v,counts,types,6,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    if(groups)pool_kernel<<<groups,32,0,r->stream>>>((const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],
        (const __nv_bfloat16*)data[2],(const float*)ape->data,(unsigned char*)data[4],(float*)data[5],p->position);
    tail_kernel<<<1,128,0,r->stream>>>((const __nv_bfloat16*)data[0],(const __nv_bfloat16*)data[1],
        (const __nv_bfloat16*)data[2],(__nv_bfloat16*)data[3],p->position,p->n_tokens);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
static __global__ void weight_scale_kernel(const float *weights,const float *scale,float *out,int rows,float factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows)out[i]=__fmul_rn(__fmul_rn(weights[i],scale[i]),factor);
}
static int weight_scale(void *ctx_ptr,void *ptr,const BnGPUIndexerWeightScaleResidentPlan *p) {
    BnCudaCtx *ctx=(BnCudaCtx*)ctx_ptr;ResidentRequest *r=(ResidentRequest*)ptr;
    if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability!=1200||p->rows<1||
       !isfinite(p->factor)||p->factor<=0||set_device(r))return -1;
    BnGPUResidentView v[]={p->weights,p->query_scales,p->out};size_t n=p->rows,counts[]={n,n,n};
    int types[]={BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};void *data[3];
    if(valid_views(r,v,counts,types,3,data))return -1;
    if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
    weight_scale_kernel<<<((size_t)p->rows+255)/256,256,0,r->stream>>>((const float*)data[0],(const float*)data[1],(float*)data[2],p->rows,p->factor);
    if(cudaGetLastError()==cudaSuccess)return 0;
    r->failed=1;return -1;
}
} /* namespace bn_cuda_glm_indexer */
#endif
