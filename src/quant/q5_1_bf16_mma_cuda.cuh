#ifndef BN_QUANT_Q5_1_BF16_MMA_CUDA_CUH
#define BN_QUANT_Q5_1_BF16_MMA_CUDA_CUH
/* Include after the Q5_1 pack helper and integer MMA helpers. Only integer
 * dots move to tensor cores. FP16-rounded scale/minimum products and the
 * sequential per-32-value F32 accumulation retain the MMQ contract, including
 * expert down widths that are not divisible by 256. Launch Tokens*4 threads. */
template<int Tokens>
static __device__ __forceinline__ void cuda_bf16_q51_mma_core(__nv_bfloat16 *out,
    const BnBlockQ5_1 *weights,const BnCudaBlockQ8_1 *input,int rows,int cols,
    int nt,int row0,int token0) {
    static_assert(Tokens>=8&&Tokens<=64&&Tokens%8==0,"Q5_1 MMA token tile");
    __shared__ __align__(16) int8_t a[16][272],x[Tokens][272];
    __shared__ float wd[16],wm[16],xd[Tokens],xs[Tokens];
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,blocks=cols/32;
    float acc[4]={0};
    for(int b=0;b<blocks;b++) {
        for(int i=tid*4;i<16*32;i+=Tokens*16) {
            int r=i/32,k=i%32,row=row0+r;
            uint32_t q=row<rows?q5_1_bf16_pack4(weights+(size_t)row*blocks+b,k):0;
            *(uint32_t*)&a[r][k]=q;
        }
        for(int i=tid*4;i<Tokens*32;i+=Tokens*16) {
            int t=i/32,k=i%32,token=token0+t;
            uint32_t q=token<nt?__ldg((const uint32_t*)(input[(size_t)token*blocks+b].qs+k)):0;
            *(uint32_t*)&x[t][k]=q;
        }
        if(tid<16) {
            int row=row0+tid;const BnBlockQ5_1 *w=row<rows?weights+(size_t)row*blocks+b:NULL;
            wd[tid]=w?cuda_fp16_to_fp32(w->d):0;wm[tid]=w?cuda_fp16_to_fp32(w->m):0;
        }
        if(tid<Tokens) {
            int token=token0+tid;
            const BnCudaBlockQ8_1 *q=token<nt?input+(size_t)token*blocks+b:NULL;
            xd[tid]=q?cuda_fp16_to_fp32(q->d):0;xs[tid]=q?cuda_fp16_to_fp32((uint16_t)q->qsum):0;
        }
        __syncthreads();
        int dot[4];cuda_bf16_mma_dot32(a,x,warp,lane,0,dot);
#pragma unroll
        for(int l=0;l<4;l++) {
            int r=lane/4+(l/2)*8,t=warp*8+(lane%4)*2+(l&1);
            float scale=cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(__fmul_rn(wd[r],xd[t])));
            float offset=cuda_fp16_to_fp32(cuda_fp32_to_fp16_bits(__fmul_rn(wm[r],xs[t])));
            acc[l]=__fadd_rn(acc[l],fmaf((float)dot[l],scale,offset));
        }
        __syncthreads();
    }
#pragma unroll
    for(int l=0;l<4;l++) {
        int row=row0+lane/4+(l/2)*8,token=token0+warp*8+(lane%4)*2+(l&1);
        if(row<rows&&token<nt)out[(size_t)token*rows+row]=__float2bfloat16_rn(acc[l]);
    }
}
#endif
