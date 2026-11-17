#ifndef BN_QUANT_IQ4_BF16_MMVQ_CUDA_CUH
#define BN_QUANT_IQ4_BF16_MMVQ_CUDA_CUH
/* Integer MMA computes each original MMVQ lane's integer dot. The 32 FP32
 * lane accumulators retain their original block stride and scale/FMA order.
 * A separate coalesced reduction reproduces XOR16/8/4/2/1 without changing
 * precision. Shared tiles belong to one warp; syncwarp orders their reuse.
 * Scratch is supplied by the admitted backend request, never by model weights. */
static __device__ void iq4_bf16_ordered_dot32(const int8_t a[16][48], const int8_t b[8][48],
                                           int warp, int lane, int g, int c[4]) {
#if __CUDA_ARCH__ >= 800
    int av[4], bv[2];
    unsigned ap = (unsigned)__cvta_generic_to_shared(&a[lane % 16][g * 32 + (lane / 16) * 16]);
    unsigned bp =
        (unsigned)__cvta_generic_to_shared(&b[warp * 8 + lane % 8][g * 32 + ((lane / 8) & 1) * 16]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3])
                 : "r"(ap));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                 : "=r"(bv[0]), "=r"(bv[1])
                 : "r"(bp));
    c[0] = c[1] = c[2] = c[3] = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                 "{%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
#else
    /* This branch keeps lower-architecture compilation valid. The host routes
     * pre-SM80 devices through the existing DP4A kernels. */
    for (int l = 0; l < 4; ++l) {
        int row = lane / 4 + (l / 2) * 8, token = warp * 8 + (lane % 4) * 2 + (l & 1);
        c[l] = 0;
        for (int k = 0; k < 32; ++k)
            c[l] += (int)a[row][g * 32 + k] * (int)b[token][g * 32 + k];
    }
#endif
}

static __global__ void iq4_bf16_ordered_mma_kernel(float*parts,const void*weights,const BnCudaBlockQ8_1*input,int rows,int cols,int nt,int kind){
 __shared__ __align__(16) int8_t a[4][16][48],x[4][8][48];
 __shared__ float wd[4][16],xd[4][8];
 int lane=threadIdx.x&31,warp=threadIdx.x/32,part=blockIdx.z*4+warp;
 int row0=blockIdx.x*16,token0=blockIdx.y*8,gl=kind==BN_QUANT_BF16_DECODE_IQ4_NL?2:8;
 int blocks=cols/(gl==2?32:256),group=part%gl;float sums[4]={0};
 for(int b=part/gl;b<blocks;b+=32/gl){
  for(int i=lane;i<64;i+=32){
   int rr=i/4,j=i%4,r=row0+rr;uint32_t packed=0;int2 q=make_int2(0,0);
   if(gl==2){
    if(j<2&&r<rows){const BnBlockIQ4NL*w=(const BnBlockIQ4NL*)weights+(size_t)r*blocks+b;memcpy(&packed,w->qs+group*8+j*4,4);q=iq4xs_expand_codes(packed);}
    if(j<2){*(int*)&a[warp][rr][j*4]=q.x;*(int*)&a[warp][rr][8+j*4]=q.y;}
    else {*(int*)&a[warp][rr][16+(j-2)*4]=0;*(int*)&a[warp][rr][24+(j-2)*4]=0;}
   }else{
    if(r<rows){const BnBlockIQ4XS*w=(const BnBlockIQ4XS*)weights+(size_t)r*blocks+b;memcpy(&packed,w->qs+group*16+j*4,4);q=iq4xs_expand_codes(packed);}
    *(int*)&a[warp][rr][j*4]=q.x;*(int*)&a[warp][rr][16+j*4]=q.y;
   }
  }
  for(int i=lane;i<64;i+=32){
   int t=i/8,k=i%8;uint32_t v=0;
   if(token0+t<nt&&(gl!=2||k<4)){
    const BnCudaBlockQ8_1*q=input+(size_t)(token0+t)*(cols/32)+b*(gl==2?1:8)+(gl==2?0:group);
    int at=gl==2?group*8+(k%2)*4+(k>=2?16:0):k*4;memcpy(&v,q->qs+at,4);
   }
   *(uint32_t*)&x[warp][t][k*4]=v;
  }
  if(lane<16){int r=row0+lane;float d=0;
   if(r<rows){if(gl==2)d=cuda_fp16_to_fp32(((const BnBlockIQ4NL*)weights)[(size_t)r*blocks+b].d);
    else{const BnBlockIQ4XS*w=(const BnBlockIQ4XS*)weights+(size_t)r*blocks+b;int sc=((w->scales_l[group/2]>>(4*(group&1)))&15)|(((w->scales_h>>(2*group))&3)<<4);d=__fmul_rn(cuda_fp16_to_fp32(w->d),(float)(sc-32));}}
   wd[warp][lane]=d;
  }
  if(lane<8){const BnCudaBlockQ8_1*q=token0+lane<nt?input+(size_t)(token0+lane)*(cols/32)+b*(gl==2?1:8)+(gl==2?0:group):NULL;xd[warp][lane]=q?cuda_fp16_to_fp32(q->d):0;}
  __syncwarp();int c[4];iq4_bf16_ordered_dot32(a[warp],x[warp],0,lane,0,c);
#pragma unroll
  for(int j=0;j<4;j++){int r=lane/4+(j/2)*8,t=(lane%4)*2+(j&1);float d=__fmul_rn(wd[warp][r],xd[warp][t]);sums[j]=fmaf(d,(float)c[j],sums[j]);}
  __syncwarp();
 }
#pragma unroll
 for(int j=0;j<4;j++){int r=row0+lane/4+(j/2)*8,t=token0+(lane%4)*2+(j&1);if(r<rows&&t<nt)parts[((size_t)part*nt+t)*rows+r]=sums[j];}
}
static __global__ void iq4_bf16_ordered_reduce_kernel(__nv_bfloat16*out,const float*parts,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float a[32];
#pragma unroll
 for(int j=0;j<32;j++)a[j]=parts[(size_t)j*n+i];
#pragma unroll
 for(int offset=16;offset;offset>>=1){
#pragma unroll
  for(int j=0;j<offset;j++)a[j]=__fadd_rn(a[j],a[j+offset]);
 }
 out[i]=__float2bfloat16_rn(a[0]);
}

#endif
