#ifndef BN_GPU_CUDA_QSA_ATTENTION_CUH
#define BN_GPU_CUDA_QSA_ATTENTION_CUH

#include <cuda_bf16.h>

/* Private paged QSA math, shared by the resident adapter and stage diagnostics. */
namespace bn_cuda_qsa_attention {
static __device__ float exp2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
static __device__ float exp2_preserve(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
static __device__ float divide(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

/* BF16 tensor-core QK/PV tiles for sequential text and paged main KV.
 * Metadata/ownership validation belongs to the resident API adapter. */
static __device__ unsigned pair_load(const __nv_bfloat16 *p) {
    return *(const unsigned *)p;
}
static __device__ unsigned pair_pack(__nv_bfloat16 a, __nv_bfloat16 b) {
    return (unsigned)__bfloat16_as_ushort(a) | ((unsigned)__bfloat16_as_ushort(b)<<16);
}
static __device__ void mma(float *d, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    #else
    (void)d;(void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;
    asm volatile("trap;"); /* Unsupported kernels are never exposed below SM80. */
    #endif
}
static __device__ float lse_log2(float x) {
    /* The pinned tl.math.log2 lowers to libdevice __nv_log2f, not MUFU lg2. */
    return log2f(x);
}

template<int BlockN> struct PartialShared {
    __nv_bfloat16 keys[BlockN*256], values[BlockN*256], probabilities[16*BlockN];
    float scores[16*BlockN], maximum[16], normalizer[16], alpha[16];
    int valid[BlockN], physical[BlockN], offset[BlockN];
};

template<int BlockN, int Warps>
static __device__ float probability_sum(const float *p) {
    const int Registers = BlockN/(8*Warps);
    float warp_sums[Warps];
    #pragma unroll
    for(int w=0;w<Warps;w++) {
        float lane_sums[4];
        #pragma unroll
        for(int lane=0;lane<4;lane++) {
            float even[Registers],odd[Registers];
            #pragma unroll
            for(int j=0;j<Registers;j++) {
                int c=2*lane+8*w+8*Warps*j;
                even[j]=p[c];odd[j]=p[c+1];
            }
            #pragma unroll
            for(int step=1;step<Registers;step*=2) {
                #pragma unroll
                for(int j=0;j<Registers;j+=2*step) {
                    even[j]=__fadd_rn(even[j],even[j+step]);
                    odd[j]=__fadd_rn(odd[j],odd[j+step]);
                }
            }
            lane_sums[lane]=__fadd_rn(even[0],odd[0]);
        }
        warp_sums[w]=__fadd_rn(__fadd_rn(lane_sums[0],lane_sums[2]),
            __fadd_rn(lane_sums[1],lane_sums[3]));
    }
    #pragma unroll
    for(int step=Warps/2;step;step>>=1) {
        #pragma unroll
        for(int j=0;j<step;j++)warp_sums[j]=__fadd_rn(warp_sums[j],warp_sums[j+step]);
    }
    return warp_sums[0];
}

template<int BlockN, int ReferenceWarps, int Splits>
static __global__ void partial_kernel(const __nv_bfloat16 *q,
    const __nv_bfloat16 *keys, const __nv_bfloat16 *values,
    const int *indices, const int *table, const int *requests,
    float *partial, float *lse, const __nv_bfloat16 *gate, __nv_bfloat16 *out,
    int rows, int table_width, int num_blocks, int num_requests) {
    static_assert((BlockN==32||BlockN==64)&&(ReferenceWarps==1||ReferenceWarps==2||ReferenceWarps==4),"QSA tile");
    /* Execution parallelism is independent of the reference reduction tree. */
    const int Warps=4;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    PartialShared<BlockN> &s=*(PartialShared<BlockN>*)shared_bytes;
    int row=blockIdx.x,kv=blockIdx.y,split=blockIdx.z;
    int tid=threadIdx.x,warp=tid/32,lane=tid&31,h=lane/4,c=(lane&3)*2;
    if(tid<16){s.maximum[tid]=-1e20f;s.normalizer[tid]=0.f;}
    int req=requests[row],safe_req=min(max(req,0),num_requests-1);
    int count=min(max(indices[(size_t)row*2052+2051],0),2051);
    int tiles=(count+BlockN-1)/BlockN;
    float accumulator[32/Warps][4]={};
    unsigned query_fragments[16][4];
    if(split<tiles) {
        #pragma unroll
        for(int k=0;k<256;k+=16) {
            const __nv_bfloat16 *qa=q+((size_t)row*24+kv*12+h)*256+k+c;
            query_fragments[k/16][0]=h<12?pair_load(qa):0;
            query_fragments[k/16][1]=h+8<12?pair_load(qa+8*256):0;
            query_fragments[k/16][2]=h<12?pair_load(qa+8):0;
            query_fragments[k/16][3]=h+8<12?pair_load(qa+8*256+8):0;
        }
    }
    __syncthreads();
    for(int tile=split;tile<tiles;tile+=Splits) {
        for(int t=tid;t<BlockN;t+=32*Warps) {
            int col=tile*BlockN+t;
            int token=col<2051?indices[(size_t)row*2052+col]:-1;
            int page=max(token,0)/16;
            bool valid=req>=0&&req<num_requests&&token>=0&&page<table_width;
            int physical=valid?table[(size_t)safe_req*table_width+page]:-1;
            valid=valid&&physical>=0&&physical<num_blocks;
            s.valid[t]=valid;s.physical[t]=valid?physical:0;s.offset[t]=max(token,0)%16;
        }
        __syncthreads();
        /* Eight BF16 values per lane: aligned 128-bit global/shared copies. */
        for(int chunk=tid;chunk<BlockN*32;chunk+=32*Warps) {
            int t=chunk/32,d=(chunk%32)*8;
            size_t src=(((size_t)s.physical[t]*16+s.offset[t])*2+kv)*256+d;
            uint4 kb=s.valid[t]?*(const uint4*)(keys+src):make_uint4(0,0,0,0);
            uint4 vb=s.valid[t]?*(const uint4*)(values+src):make_uint4(0,0,0,0);
            *(uint4*)(s.keys+t*256+d)=kb;*(uint4*)(s.values+t*256+d)=vb;
        }
        __syncthreads();
        #pragma unroll
        for(int ntile=warp;ntile<BlockN/8;ntile+=Warps) {
            float scores[4]={};
            #pragma unroll
            for(int k=0;k<256;k+=16) {
                unsigned a0=query_fragments[k/16][0],a1=query_fragments[k/16][1];
                unsigned a2=query_fragments[k/16][2],a3=query_fragments[k/16][3];
                const __nv_bfloat16 *kb=s.keys+(ntile*8+h)*256+k+c;
                mma(scores,a0,a1,a2,a3,pair_load(kb),pair_load(kb+8));
            }
            s.scores[(ntile*8+c)*16+h]=__fmul_rn(scores[0],0.09016844005556021f);
            s.scores[(ntile*8+c+1)*16+h]=__fmul_rn(scores[1],0.09016844005556021f);
            s.scores[(ntile*8+c)*16+h+8]=__fmul_rn(scores[2],0.09016844005556021f);
            s.scores[(ntile*8+c+1)*16+h+8]=__fmul_rn(scores[3],0.09016844005556021f);
        }
        __syncthreads();
        if(tid<16) {
            /* Retain logits/probabilities in registers instead of three
             * round trips through shared memory during softmax reduction. */
            float probabilities[BlockN];
            float next=s.maximum[tid];
            #pragma unroll
            for(int t=0;t<BlockN;t++) {
                probabilities[t]=s.valid[t]?s.scores[t*16+tid]:-1e20f;
                next=fmaxf(next,probabilities[t]);
            }
            s.alpha[tid]=exp2_ftz(__fsub_rn(s.maximum[tid],next));
            #pragma unroll
            for(int t=0;t<BlockN;t++) {
                float p=s.valid[t]?exp2_ftz(__fsub_rn(probabilities[t],next)):0.f;
                probabilities[t]=p;s.probabilities[tid*BlockN+(t^((tid&7)*2))]=__float2bfloat16_rn(p);
            }
            s.normalizer[tid]=__fmaf_rn(s.normalizer[tid],s.alpha[tid],probability_sum<BlockN,ReferenceWarps>(probabilities));
            s.maximum[tid]=next;
        }
        __syncthreads();
        #pragma unroll
        for(int n=0;n<32/Warps;n++) {
            int dim=(warp+n*Warps)*8+h;
            float *acc=accumulator[n];
            acc[0]=__fmul_rn(acc[0],s.alpha[h]);acc[1]=__fmul_rn(acc[1],s.alpha[h]);
            acc[2]=__fmul_rn(acc[2],s.alpha[h+8]);acc[3]=__fmul_rn(acc[3],s.alpha[h+8]);
            #pragma unroll
            for(int k=0;k<BlockN;k+=16) {
                const __nv_bfloat16 *pa=s.probabilities+h*BlockN;
                int p0=(k+c)^((h&7)*2),p1=(k+c+8)^((h&7)*2);
                const __nv_bfloat16 *vb=s.values+(k+c)*256+dim;
                mma(acc,pair_load(pa+p0),pair_load(pa+8*BlockN+p0),pair_load(pa+p1),pair_load(pa+8*BlockN+p1),
                    pair_pack(vb[0],vb[256]),pair_pack(vb[8*256],vb[9*256]));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int n=0;n<32/Warps;n++) {
        int dim=(warp+n*Warps)*8+c;
        #pragma unroll
        for(int j=0;j<4;j++) {
            int head=h+(j/2)*8,d=dim+j%2;
            if(head<12) {
                float value=s.normalizer[head]>0?divide(accumulator[n][j],fmaxf(s.normalizer[head],1e-20f)):0.f;
                size_t i=((size_t)row*24+kv*12+head)*256+d;
                if(Splits==1) {
                    value=__bfloat162float(__float2bfloat16_rn(value));
                    float g=__bfloat162float(gate[i]);
                    float sigmoid=divide(1.f,__fadd_rn(1.f,exp2_preserve(__fmul_rn(g,-1.4426950408889634f))));
                    out[i]=__float2bfloat16_rn(__fmul_rn(value,sigmoid));
                } else partial[(((size_t)split*rows+row)*24+kv*12+head)*256+d]=value;
            }
        }
    }
    if(Splits>1&&tid<12) {
        float l=s.normalizer[tid]>0?__fadd_rn(s.maximum[tid],lse_log2(fmaxf(s.normalizer[tid],1e-20f))):-__int_as_float(0x7f800000);
        lse[((size_t)split*rows+row)*24+kv*12+tid]=l;
    }
}

/* Inputs are normalized F32 partial outputs [splits, rows, 24, 256] and
 * base-2 LSE [splits, rows, 24]. Empty splits have zero output and -Inf LSE.
 * Preserve the pinned pair-FMA/sum, BF16 output-rounding and F32
 * sigmoid boundaries. One CTA owns one complete token/head output. */
template<int Splits>
static __global__ void merge_kernel(const float *partial, const float *lse,
    const __nv_bfloat16 *gate, __nv_bfloat16 *out, int rows) {
    static_assert(Splits >= 2 && Splits <= 64 && !(Splits & (Splits-1)),
        "power-of-two QSA splits required");
    int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    __shared__ float weights[Splits], denominator;
    if (d == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int s = 0; s < Splits; s++)
            maximum = fmaxf(maximum, lse[((size_t)s*rows+row)*24+head]);
        float sums[Splits];
        for (int s = 0; s < Splits; s++) {
            float v = lse[((size_t)s*rows+row)*24+head];
            weights[s] = sums[s] = maximum > -__int_as_float(0x7f800000) ?
                exp2_ftz(__fsub_rn(v,maximum)) : 0.f;
        }
        /* LSE layout: butterfly reduction inside each 32-lane warp, then
         * the two warp totals. This differs from the partial-output tree. */
        const int lanes = Splits < 32 ? Splits : 32;
        for (int step = lanes/2; step; step >>= 1)
            for (int base = 0; base < Splits; base += lanes)
                for (int i = 0; i < step; i++)
                    sums[base+i] = __fadd_rn(sums[base+i],sums[base+i+step]);
        denominator = Splits == 64 ? __fadd_rn(sums[0],sums[32]) : sums[0];
    }
    __syncthreads();
    float products[Splits/2];
    /* ptxas contracts the pinned mul + packed pair-add into an FMA with
     * the even split product, preserving an already-rounded odd product. */
    #pragma unroll
    for (int pair = 0; pair < Splits/2; pair++) {
        int s = 2*pair;
        float even = partial[(((size_t)s*rows+row)*24+head)*256+d];
        float odd = partial[(((size_t)(s+1)*rows+row)*24+head)*256+d];
        products[pair] = __fmaf_rn(even,weights[s],__fmul_rn(odd,weights[s+1]));
    }
    #pragma unroll
    for (int stride = 1; stride < Splits/2; stride *= 2) {
        #pragma unroll
        for (int i = 0; i < Splits/2; i += 2*stride)
            products[i] = __fadd_rn(products[i],products[i+stride]);
    }
    float value = denominator > 0.f ? divide(products[0],denominator) : 0.f;
    value = __bfloat162float(__float2bfloat16_rn(value));
    size_t i = ((size_t)row*24+head)*256+d;
    float g = __bfloat162float(gate[i]);
    float sigmoid = divide(1.f,__fadd_rn(1.f,exp2_preserve(__fmul_rn(g,-1.4426950408889634f))));
    out[i] = __float2bfloat16_rn(__fmul_rn(value,sigmoid));
}
} /* namespace bn_cuda_qsa_attention */
#endif
