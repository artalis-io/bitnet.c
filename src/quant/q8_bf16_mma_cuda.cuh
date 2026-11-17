#ifndef BN_QUANT_Q8_BF16_MMA_CUDA_CUH
#define BN_QUANT_Q8_BF16_MMA_CUDA_CUH
/* Signed dot32 MMA only. Keep the F32 scale product and sequential per-block
 * FMA of Q8 MMQ; no FP16 product rounding. Q8 weight blocks may be unaligned.
 * Launch Tokens*4 threads. Include after integer MMA and FP16 helpers. */
template<int Tokens>
static __device__ __forceinline__ void cuda_bf16_q8_mma_core(__nv_bfloat16 *out,
    const BnBlockQ8_0 *weights,const BnCudaBlockQ8_1 *input,int rows,int cols,
    int nt,int row0,int token0) {
    static_assert(Tokens>=8&&Tokens<=64&&Tokens%8==0,"Q8 MMA token tile");
    __shared__ __align__(16) int8_t a[16][272],x[Tokens][272];
    __shared__ float wd[16],xd[Tokens];
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,blocks=cols/32;
    float acc[4]={0};
    for(int b=0;b<blocks;b++) {
        for(int i=tid*4;i<16*32;i+=Tokens*16) {
            int r=i/32,k=i%32,row=row0+r;
            uint32_t q=0; if(row<rows)memcpy(&q,weights[(size_t)row*blocks+b].qs+k,4);
            *(uint32_t*)&a[r][k]=q;
        }
        for(int i=tid*4;i<Tokens*32;i+=Tokens*16) {
            int t=i/32,k=i%32,token=token0+t;
            uint32_t q=token<nt?__ldg((const uint32_t*)(input[(size_t)token*blocks+b].qs+k)):0;
            *(uint32_t*)&x[t][k]=q;
        }
        if(tid<16) {
            int row=row0+tid;const BnBlockQ8_0 *w=row<rows?weights+(size_t)row*blocks+b:NULL;
            wd[tid]=w?cuda_fp16_to_fp32(w->d):0;
        }
        if(tid<Tokens) {
            int token=token0+tid;
            const BnCudaBlockQ8_1 *q=token<nt?input+(size_t)token*blocks+b:NULL;
            xd[tid]=q?cuda_fp16_to_fp32(q->d):0;
        }
        __syncthreads();
        int dot[4];cuda_bf16_mma_dot32(a,x,warp,lane,0,dot);
#pragma unroll
        for(int l=0;l<4;l++) {
            int r=lane/4+(l/2)*8,t=warp*8+(lane%4)*2+(l&1);
            float scale=__fmul_rn(wd[r],xd[t]);
            acc[l]=fmaf(scale,(float)dot[l],acc[l]);
        }
        __syncthreads();
    }
#pragma unroll
    for(int l=0;l<4;l++) {
        int row=row0+lane/4+(l/2)*8,token=token0+warp*8+(lane%4)*2+(l&1);
        if(row<rows&&token<nt)out[(size_t)token*rows+row]=__float2bfloat16_rn(acc[l]);
    }
}

template<int Tokens>
static __global__ void cuda_bf16_q8_mma_kernel(__nv_bfloat16 *out,
    const BnBlockQ8_0 *w,const BnCudaBlockQ8_1 *x,int rows,int cols,int nt) {
    int tiles=(nt-1)/Tokens+1;
    cuda_bf16_q8_mma_core<Tokens>(out,w,x,rows,cols,nt,
        (blockIdx.x/tiles)*16,(blockIdx.x%tiles)*Tokens);
}
/* Stage eight adjacent Q8 blocks per barrier pair. Integer dots, the scale
 * product, and every F32 FMA retain the original ascending block order.
 * The 64-token tile also serves small batches: cooperative loading uses eight
 * warps while inactive token rows stay zero and never write outputs. */
template<int Tokens>
static __device__ __forceinline__ void cuda_bf16_q8_grouped_core(__nv_bfloat16 *out,
    const BnBlockQ8_0 *weights,const BnCudaBlockQ8_1 *input,int rows,int cols,
    int nt,int row0,int token0) {
    static_assert(Tokens>=8&&Tokens<=64&&Tokens%8==0,"Q8 grouped token tile");
    __shared__ __align__(16) int8_t a[16][272],x[Tokens][272];
    __shared__ float wd[16][8],xd[Tokens][8];
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,blocks=cols/32;
    float acc[4]={0};
    for(int b=0;b<blocks;b+=8) {
        for(int i=tid*4;i<16*256;i+=Tokens*16) {
            int r=i/256,k=i%256,g=k/32,row=row0+r;
            uint32_t q=0;
            if(row<rows&&b+g<blocks)memcpy(&q,weights[(size_t)row*blocks+b+g].qs+k%32,4);
            *(uint32_t*)&a[r][k]=q;
        }
        for(int i=tid*4;i<Tokens*256;i+=Tokens*16) {
            int t=i/256,k=i%256,g=k/32,token=token0+t;
            uint32_t q=token<nt&&b+g<blocks?__ldg((const uint32_t*)(input[(size_t)token*blocks+b+g].qs+k%32)):0;
            *(uint32_t*)&x[t][k]=q;
        }
        for(int i=tid;i<16*8;i+=Tokens*4) {
            int r=i/8,g=i%8,row=row0+r;
            wd[r][g]=row<rows&&b+g<blocks?cuda_fp16_to_fp32(weights[(size_t)row*blocks+b+g].d):0;
        }
        for(int i=tid;i<Tokens*8;i+=Tokens*4) {
            int t=i/8,g=i%8,token=token0+t;
            xd[t][g]=token<nt&&b+g<blocks?cuda_fp16_to_fp32(input[(size_t)token*blocks+b+g].d):0;
        }
        __syncthreads();
#pragma unroll
        for(int g=0;g<8;g++)if(b+g<blocks) {
            int dot[4];cuda_bf16_mma_dot32(a,x,warp,lane,g,dot);
#pragma unroll
            for(int l=0;l<4;l++) {
                int r=lane/4+(l/2)*8,t=warp*8+(lane%4)*2+(l&1);
                float scale=__fmul_rn(wd[r][g],xd[t][g]);
                acc[l]=fmaf(scale,(float)dot[l],acc[l]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int l=0;l<4;l++) {
        int row=row0+lane/4+(l/2)*8,token=token0+warp*8+(lane%4)*2+(l&1);
        if(row<rows&&token<nt)out[(size_t)token*rows+row]=__float2bfloat16_rn(acc[l]);
    }
}
template<int T>static __global__ void cuda_bf16_q8_grouped_mma_kernel(__nv_bfloat16*out,const BnBlockQ8_0*w,const BnCudaBlockQ8_1*x,int rows,int cols,int nt){int tiles=(nt+T-1)/T;cuda_bf16_q8_grouped_core<T>(out,w,x,rows,cols,nt,(blockIdx.x/tiles)*16,(blockIdx.x%tiles)*T);}

#endif
