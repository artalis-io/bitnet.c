#ifndef BN_QUANT_MXFP4_CUDA_CUH
#define BN_QUANT_MXFP4_CUDA_CUH

/* Format-local CUDA arithmetic. The including backend supplies the Q8 input
 * block type and FP16 conversion helper. Kernels borrow all buffers; allocation,
 * device selection, and execution policy remain backend responsibilities. */
#ifdef BN_CUDA_MXFP4_SM120
#include <cuda_fp4.h>

static __global__ void mxfp4_mmvq_kernel(
        float *out, const BnBlockMXFP4 *weights, const BnCudaBlockQ8_1 *input,
        int rows, int cols, const float *bias, size_t out_offset) {
    const int values[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                           0, -1, -2, -3, -4, -6, -8, -12};
    __shared__ float partial[3][32];
    int row = blockIdx.x, token = blockIdx.y, tid = threadIdx.x;
    int lane = tid & 31, warp = tid / 32, group = tid & 1;
    int blocks = cols / 32;
    float sum = 0.0f;
    for (int b = tid / 2; b < blocks; b += blockDim.x / 2) {
        const BnBlockMXFP4 *w = weights + (size_t)row * blocks + b;
        const BnCudaBlockQ8_1 *x = input + (size_t)token * blocks + b;
        int dot = 0;
        for (int j = 0; j < 8; j++) {
            int i = group * 8 + j;
            dot += values[w->qs[i] & 15] * (int)x->qs[i] +
                   values[w->qs[i] >> 4] * (int)x->qs[i + 16];
        }
        /* Integer codebook entries are twice the E2M1 values. */
        unsigned bits = w->e < 2 ? 0x00200000u << w->e
                                 : (unsigned)(w->e - 1) << 23;
        float scale = __uint_as_float(bits) * cuda_fp16_to_fp32(x->d);
        sum = fmaf(scale, (float)dot, sum);
    }
    if (warp) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (!warp) {
        for (int i = 0; i < (int)blockDim.x / 32 - 1; i++) sum += partial[i][lane];
        for (int i = 16; i; i /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, i);
        if (!lane) {
            if (bias) sum += bias[row];
            out[out_offset + (size_t)token * rows + row] = sum;
        }
    }
}

static __global__ void mxfp4_quantize_mmq_kernel(
        BnBlockMXFP4 *out, const float *input, int cols) {
    int block = blockIdx.x, token = blockIdx.y, lane = threadIdx.x;
    float value = input[(size_t)token * cols + block * 32 + lane];
    float amax = fabsf(value);
    for (int i = 16; i; i /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, i));
    /* Match the reference's fast log2 and nearest-even exponent selection. */
    int exponent = amax > 0 ? __float2int_rn(__log2f(amax)) + 125 : 0;
    exponent = max(0, min(254, exponent));
    float scale = exponent == 0 ? __uint_as_float(0x00400000u)
                                : __uint_as_float((unsigned)exponent << 23);
    float scaled = amax > 0 ? value * __frcp_rn(scale) : 0;
    float high = __shfl_sync(0xffffffffu, scaled, (lane & 15) + 16);
    BnBlockMXFP4 *dst = out + (size_t)token * (cols / 32) + block;
    if (lane < 16) {
        __nv_fp4x2_e2m1 packed(make_float2(scaled, high));
        dst->qs[lane] = *(const unsigned char *)&packed;
    }
    if (!lane) dst->e = (uint8_t)exponent;
}

static __device__ unsigned mxfp4_packed_word(const BnBlockMXFP4 *block, int word) {
    const unsigned char *q = block->qs + word * 4;
    return (unsigned)q[0] | ((unsigned)q[1] << 8) |
           ((unsigned)q[2] << 16) | ((unsigned)q[3] << 24);
}

/* One warp computes 16 rows by 8 distinct input tokens. The reference logical
 * tile is 128 rows by jwidth tokens. Preserve its split-K boundaries and fixup
 * order even though the physical launch uses smaller output tiles. */
static __global__ void mxfp4_mmq_kernel(
        float *out, const BnBlockMXFP4 *weights, const BnBlockMXFP4 *input,
        int rows, int cols, int n_tokens, int jwidth, int reference_grid) {
    int lane = threadIdx.x, row0 = blockIdx.x * 16, token0 = blockIdx.y * 8;
    int token = token0 + lane / 4, blocks = cols / 32;
    int token_tiles = (n_tokens + jwidth - 1) / jwidth;
    int64_t tile = (int64_t)(row0 / 128) * token_tiles + token0 / jwidth;
    int64_t tile_begin = tile * blocks;
    int64_t total = (int64_t)((rows + 127) / 128) * token_tiles * blocks;
    int first = (int)(((tile_begin + 1) * reference_grid + total - 1) / total) - 1;
    int last = (int)(((tile_begin + blocks) * reference_grid + total - 1) / total) - 1;
    float tail[4] = {}, prefix[4] = {};
    int have_tail = 0;
    for (int bid = last; bid >= first; bid--) {
        int64_t raw_begin = (int64_t)bid * total / reference_grid;
        int64_t raw_end = (int64_t)(bid + 1) * total / reference_grid;
        /* Partition in actual block32 units, then round within each tile.
         * Padding K before partitioning changes partial sums for tail tiles. */
        raw_begin -= (raw_begin % blocks) % 16;
        raw_end -= (raw_end % blocks) % 16;
        raw_begin -= tile_begin;
        raw_end -= tile_begin;
        int begin = raw_begin < 0 ? 0 : (int)raw_begin;
        int end = raw_end > blocks ? blocks : (int)raw_end;
        if (begin >= end) continue;
        float sum[4] = {};
        for (int b = begin / 16; b < (end + 15) / 16; b++) {
            for (int fragment = 0; fragment < 8; fragment++) {
                int g = b * 16 + fragment * 2;
                unsigned a[4] = {}, x[2] = {}, scale_a = 0, scale_x = 0;
                for (int l = 0; l < 4; l++) {
                    /* ldmatrix.x4 register order, not generic tile order. */
                    int row = row0 + lane / 4 + 8 * (l % 2);
                    int word = lane % 4 + 4 * (l / 2), block = g + word / 4;
                    if (row < rows && block < blocks)
                        a[l] = mxfp4_packed_word(
                            weights + (size_t)row * blocks + block, word % 4);
                }
                for (int l = 0; l < 2; l++) {
                    if (g + l < blocks && token < n_tokens)
                        x[l] = mxfp4_packed_word(
                            input + (size_t)token * blocks + g + l, lane % 4);
                }
                int scale_row = row0 + lane / 4 + (lane % 2) * 8;
                for (int l = 0; l < 2; l++) {
                    if (g + l >= blocks) continue;
                    if (scale_row < rows)
                        scale_a |= (unsigned)weights[(size_t)scale_row * blocks + g + l].e << (8 * l);
                    if (token < n_tokens)
                        scale_x |= (unsigned)input[(size_t)token * blocks + g + l].e << (8 * l);
                }
                /* Native MMA rounding is observably different from adding
                 * two scaled integer dot products in scalar FP32 arithmetic. */
                float d[4] = {};
                asm volatile(
                    "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(x[0]), "r"(x[1]), "r"(scale_a), "r"(scale_x));
                for (int l = 0; l < 4; l++) sum[l] = __fadd_rn(sum[l], d[l]);
            }
        }
        for (int l = 0; l < 4; l++) {
            if (!have_tail) tail[l] = sum[l];
            else prefix[l] = __fadd_rn(prefix[l], sum[l]);
        }
        have_tail = 1;
    }
    for (int l = 0; l < 4; l++) {
        int row = row0 + lane / 4 + 8 * (l / 2);
        int column = token0 + 2 * (lane % 4) + l % 2;
        if (row < rows && column < n_tokens)
            out[(size_t)column * rows + row] = __fadd_rn(tail[l], prefix[l]);
    }
}

/* Routed projections borrow expert IDs, compact ranks and inverse maps. */
static __global__ void mxfp4_routed_mmvq_kernel(float*out,const BnBlockMXFP4*w,const BnCudaBlockQ8_1*x,int rows,int cols,int k,const int*ids){
 const int values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
 __shared__ float partial[3][32];
 int row=blockIdx.x,token=blockIdx.y,slot=blockIdx.z,tid=threadIdx.x,lane=tid&31,warp=tid/32;
 int nb=cols/32,expert=ids?ids[token*k+slot]:0,group=tid&1;float sum=0;
 for(int b=tid/2;b<nb;b+=blockDim.x/2){
  const BnBlockMXFP4*v=w+((size_t)expert*rows+row)*nb+b;const BnCudaBlockQ8_1*q=x+(size_t)token*nb+b;int dot=0;
  for(int j=0;j<8;j++){int i=group*8+j;dot+=values[v->qs[i]&15]*(int)q->qs[i]+values[v->qs[i]>>4]*(int)q->qs[i+16];}
  unsigned bits=v->e<2?0x00200000u<<v->e:(unsigned)(v->e-1)<<23;
  float d=__uint_as_float(bits)*cuda_fp16_to_fp32(q->d);sum=fmaf(d,(float)dot,sum);
 }
 if(warp)partial[warp-1][lane]=sum;__syncthreads();
 if(!warp){for(int i=0;i<(int)blockDim.x/32-1;i++)sum+=partial[i][lane];for(int i=16;i;i/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,i);if(!lane)out[((size_t)token*k+slot)*rows+row]=sum;}
}
static __global__ void mxfp4_routed_mmq_kernel(float*out,const BnBlockMXFP4*w,const BnBlockMXFP4*x,int rows,int cols,int nt,int k,int experts,const int*ids,int width,int grid){
 int lane=threadIdx.x,row0=blockIdx.x*16,rank=blockIdx.y*8,expert=blockIdx.z;
 int input_rank=rank+lane/4;
 int source=input_rank<nt?(ids?ids[2*nt*k+expert*nt+input_rank]:input_rank):-1;
 int token=source<0?-1:source/k;
 int nb=cols/32;
 int blocks=nb,token_tiles=(nt+width-1)/width;
 long long tile=((long long)(row0/128)*experts+expert)*token_tiles+rank/width;
 long long tile_begin=tile*blocks,total=(long long)((rows+127)/128)*experts*token_tiles*blocks;
 int first=(int)(((tile_begin+1)*grid+total-1)/total)-1,last=(int)(((tile_begin+blocks)*grid+total-1)/total)-1;
 float tail[4]={},prefix[4]={};int have=0;
 for(int bid=last;bid>=first;bid--){
  long long raw_begin=(long long)bid*total/grid,raw_end=(long long)(bid+1)*total/grid;
  raw_begin-=(raw_begin%nb)%16;raw_end-=(raw_end%nb)%16;
  raw_begin-=tile_begin;raw_end-=tile_begin;
  int begin=raw_begin<0?0:(int)raw_begin,end=raw_end>blocks?blocks:(int)raw_end;if(begin>=end)continue;
  float acc[4]={};
  for(int b=begin/16;b<(end+15)/16;b++)for(int frag=0;frag<8;frag++){
   int g=b*16+frag*2;unsigned a[4]={},bv[2]={},sa=0,sb=0;
   for(int l=0;l<4;l++){
    int r=row0+lane/4+8*(l%2),word=lane%4+4*(l/2),block=g+word/4;
    if(r<rows&&block<nb)a[l]=mxfp4_packed_word(w+((size_t)expert*rows+r)*nb+block,word%4);
   }
   for(int l=0;l<2;l++){int block=g+l;if(block<nb&&token>=0)bv[l]=mxfp4_packed_word(x+(size_t)token*nb+block,lane%4);}
   int sr=row0+lane/4+(lane%2)*8;
   for(int l=0;l<2;l++)if(g+l<nb){
    if(sr<rows)sa|=(unsigned)w[((size_t)expert*rows+sr)*nb+g+l].e<<(8*l);
    if(token>=0)sb|=(unsigned)x[(size_t)token*nb+g+l].e<<(8*l);
   }
   float d[4]={};
   asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(bv[0]),"r"(bv[1]),"r"(sa),"r"(sb));
   for(int l=0;l<4;l++)acc[l]=__fadd_rn(acc[l],d[l]);
  }
  for(int l=0;l<4;l++){if(!have)tail[l]=acc[l];else prefix[l]=__fadd_rn(prefix[l],acc[l]);}have=1;
 }
 for(int l=0;l<4;l++){
  int row=row0+lane/4+8*(l/2),r=rank+2*(lane%4)+l%2;
  int dest=r<nt?(ids?ids[2*nt*k+expert*nt+r]:r):-1;
  if(row<rows&&dest>=0)out[(size_t)dest*rows+row]=__fadd_rn(tail[l],prefix[l]);
 }
}
#endif /* BN_CUDA_MXFP4_SM120 */
#endif /* BN_QUANT_MXFP4_CUDA_CUH */
