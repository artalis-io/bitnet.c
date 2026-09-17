#ifndef BN_QUANT_Q4_CUDA_CUH
#define BN_QUANT_Q4_CUDA_CUH
/* Format-local arithmetic. Kernels borrow all buffers; the backend owns
 * allocation, device policy and graph dispatch. This original-sum layout is
 * distinct from the backend integer-qsum input layout. */
struct BnQ4CudaInput {half d,s;int8_t qs[32];};
static_assert(sizeof(BnQ4CudaInput) == 36, "Q4 CUDA input layout");
static __global__ void q4_quantize_mmvq(BnQ4CudaInput*out,const float*x,int cols){
 int lane=threadIdx.x,b=blockIdx.x,t=blockIdx.y;
 float v=x[t*cols+b*32+lane],a=fabsf(v),s=v;
 for(int i=16;i;i/=2){a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,i));s+=__shfl_xor_sync(0xffffffff,s,i);}
 float d=__fdividef(a,127.f);out[t*(cols/32)+b].qs[lane]=a==0?0:(int8_t)roundf(__fdividef(v,d));
 if(!lane){out[t*(cols/32)+b].d=__float2half(d);out[t*(cols/32)+b].s=__float2half(s);}
}
static __global__ void q4_mmvq(float*out,const BnBlockQ4_0*w,const BnQ4CudaInput*x,int rows,int cols,int nw,const float *bias,size_t offset){
__shared__ float part[3][32];
 int row=blockIdx.x,t=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32,group=tid%2,nb=cols/32;
 float acc=0;
 for(int b=tid/2;b<nb;b+=nw*16){
  const BnBlockQ4_0&a=w[row*nb+b];const BnQ4CudaInput&v=x[t*nb+b];int dot=0;
  for(int j=group*8;j<group*8+8;j++)dot+=(a.qs[j]&15)*(int)v.qs[j]+(a.qs[j]>>4)*(int)v.qs[j+16];
  float z=fmaf((float)dot,__half2float(v.d),-4.f*__half2float(v.s));
  acc=fmaf(cuda_fp16_to_fp32(a.d),z,acc);
 }
 if(warp)part[warp-1][lane]=acc;
 __syncthreads();
 if(!warp){for(int i=0;i<nw-1;i++)acc+=part[i][lane];for(int i=16;i;i/=2)acc+=__shfl_xor_sync(0xffffffff,acc,i);if(!lane)out[offset+(size_t)t*rows+row]=bias?acc+bias[row]:acc;}
}
static __global__ void q4_routed_mmvq(float*out,const BnBlockQ4_0*w,
 const BnQ4CudaInput*x,const int*experts_for_item,int rows,int cols,int nw,
 int input_stride){
__shared__ float part[3][32];
 int row=blockIdx.x,item=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int group=tid%2,nb=cols/32,expert=experts_for_item[item],xt=item/input_stride;
 float acc=0;
 const BnBlockQ4_0*ew=w+(size_t)expert*rows*nb;
 const BnQ4CudaInput*ex=x+(size_t)xt*nb;
 for(int b=tid/2;b<nb;b+=nw*16){
  const BnBlockQ4_0&a=ew[row*nb+b];const BnQ4CudaInput&v=ex[b];int dot=0;
  for(int j=group*8;j<group*8+8;j++)dot+=(a.qs[j]&15)*(int)v.qs[j]+(a.qs[j]>>4)*(int)v.qs[j+16];
  float z=fmaf((float)dot,__half2float(v.d),-4.f*__half2float(v.s));
  acc=fmaf(cuda_fp16_to_fp32(a.d),z,acc);
 }
 if(warp)part[warp-1][lane]=acc;
 __syncthreads();
 if(!warp){for(int i=0;i<nw-1;i++)acc+=part[i][lane];for(int i=16;i;i/=2)acc+=__shfl_xor_sync(0xffffffff,acc,i);if(!lane)out[(size_t)item*rows+row]=acc;}
}
static __global__ void q4_routed_mmvq_2row(float*out,const BnBlockQ4_0*w,
 const BnQ4CudaInput*x,const int*experts_for_item,int rows,int cols,
 int input_stride){
 int row0=blockIdx.x*2,item=blockIdx.y,lane=threadIdx.x,nb=cols/32;
 int expert=experts_for_item[item],xt=item/input_stride;
 const BnBlockQ4_0*ew=w+(size_t)expert*rows*nb;
 const BnQ4CudaInput*ex=x+(size_t)xt*nb;
 float acc0=0.f,acc1=0.f;
 for(int b=lane/2;b<nb;b+=16){
  const BnQ4CudaInput&v=ex[b];int group=lane%2;
  const BnBlockQ4_0&a0=ew[(size_t)row0*nb+b];int dot0=0;
  for(int j=group*8;j<group*8+8;j++)dot0+=(a0.qs[j]&15)*(int)v.qs[j]+(a0.qs[j]>>4)*(int)v.qs[j+16];
  float z=fmaf((float)dot0,__half2float(v.d),-4.f*__half2float(v.s));
  acc0=fmaf(cuda_fp16_to_fp32(a0.d),z,acc0);
  if(row0+1<rows){
   const BnBlockQ4_0&a1=ew[(size_t)(row0+1)*nb+b];int dot1=0;
   for(int j=group*8;j<group*8+8;j++)dot1+=(a1.qs[j]&15)*(int)v.qs[j]+(a1.qs[j]>>4)*(int)v.qs[j+16];
   float z1=fmaf((float)dot1,__half2float(v.d),-4.f*__half2float(v.s));
   acc1=fmaf(cuda_fp16_to_fp32(a1.d),z1,acc1);
  }
 }
 for(int i=16;i;i/=2){acc0+=__shfl_xor_sync(0xffffffff,acc0,i);acc1+=__shfl_xor_sync(0xffffffff,acc1,i);}
 if(lane==0)out[(size_t)item*rows+row0]=acc0;
 if(lane==1&&row0+1<rows)out[(size_t)item*rows+row0+1]=acc1;
}
static __global__ void q4_quantize_mmq(BnQ4CudaInput*out,const float*x,int cols){
 int lane=threadIdx.x,b=blockIdx.x,t=blockIdx.y;
 float v=x[t*cols+b*32+lane],a=fabsf(v);
 for(int i=16;i;i/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,i));
 float inv=__fdividef(127.f,a),d=__fdividef(1.f,inv);out[t*(cols/32)+b].qs[lane]=a==0?0:(int8_t)roundf(v*inv);
 if(!lane){out[t*(cols/32)+b].d=__float2half(d);out[t*(cols/32)+b].s=__float2half(0.f);}
}
static __global__ void q4_mmq(float*out,const BnBlockQ4_0*w,const BnQ4CudaInput*x,int rows,int cols,int batch_tokens,int jwidth,int grid,int total_tokens,int expert,int experts,int geometry_rows,int row_offset){
 /* Four neighboring lanes process tokens for the same row, sharing weight
  * fetches without changing each token's partitioned FMA sequence. */
 int row=blockIdx.x*32+threadIdx.x/4,t=blockIdx.y*4+threadIdx.x%4,nb=cols/32;
 if(row>=rows||t>=batch_tokens)return;
 int xt=(total_tokens+jwidth-1)/jwidth,tiles=((geometry_rows+127)/128)*xt*experts;
 long long tile=(((row+row_offset)/128)*experts+expert)*xt+t/jwidth,base=tile*nb,total=(long long)tiles*nb;
 float tail=0,prefix=0;bool have=false;
 /* A partition boundary can move left by at most seven blocks below.
  * Only partitions touching this tile can contribute to its sum. */
 int first_bid=grid==tiles?(int)tile:(int)((base*grid)/total)-2;
 int last_bid=grid==tiles?(int)tile:(int)(((base+nb+8)*grid)/total)+2;
 if(first_bid<0)first_bid=0;
 if(last_bid>=grid)last_bid=grid-1;
 for(int bid=last_bid;bid>=first_bid;bid--){
 long long beg=(long long)bid*total/grid,end=(long long)(bid+1)*total/grid;
 beg-=(beg%nb)%8;end-=(end%nb)%8;beg-=base;end-=base;
 int lo=beg<0?0:(int)beg,hi=end>nb?nb:(int)end;if(lo>=hi)continue;
 float acc=0;
 for(int b=lo;b<hi;b++){
  const BnBlockQ4_0&a=w[row*nb+b];const BnQ4CudaInput&v=x[t*nb+b];int dot=0;
  for(int j=0;j<16;j++)dot+=((int)(a.qs[j]&15)-8)*(int)v.qs[j]+((int)(a.qs[j]>>4)-8)*(int)v.qs[j+16];
  acc=fmaf((float)dot*cuda_fp16_to_fp32(a.d),__half2float(v.d),acc);
 }
 if(!have)tail=acc;else prefix=__fadd_rn(prefix,acc);have=true;
 }
 out[t*rows+row]=__fadd_rn(tail,prefix);
}

/* Route-ranked warps share expert weights across four tokens while retaining
 * each token's per-expert partition and accumulation order. */
static __global__ void q4_routed_mmq(float*out,const BnBlockQ4_0*w,
 const BnQ4CudaInput*x,const int*map,int rows,int cols,int total_tokens,
 int experts,int k,int input_stride,int jwidth,int grid,
 int geometry_rows,int row_offset){
 int row=blockIdx.x*32+threadIdx.x/4,rank=blockIdx.y*4+threadIdx.x%4;
 int expert=blockIdx.z,nb=cols/32,items=total_tokens*k;
 if(row>=rows||rank>=total_tokens)return;
 int item=map[2*items+expert*total_tokens+rank];
 if(item<0)return;
 int input_token=item/input_stride;
 const BnBlockQ4_0*ew=w+(size_t)expert*rows*nb;
 const BnQ4CudaInput*ex=x+(size_t)input_token*nb;
 int xt=(total_tokens+jwidth-1)/jwidth;
 int tiles=((geometry_rows+127)/128)*xt*experts;
 long long tile=(((row+row_offset)/128)*experts+expert)*xt+rank/jwidth;
 long long base=tile*nb,total=(long long)tiles*nb;
 int first_bid=grid==tiles?(int)tile:(int)((base*grid)/total)-2;
 int last_bid=grid==tiles?(int)tile:(int)(((base+nb+8)*grid)/total)+2;
 if(first_bid<0)first_bid=0;
 if(last_bid>=grid)last_bid=grid-1;
 float tail=0,prefix=0;bool have=false;
 for(int bid=last_bid;bid>=first_bid;bid--){
  long long beg=(long long)bid*total/grid,end=(long long)(bid+1)*total/grid;
  beg-=(beg%nb)%8;end-=(end%nb)%8;beg-=base;end-=base;
  int lo=beg<0?0:(int)beg,hi=end>nb?nb:(int)end;if(lo>=hi)continue;
  float acc=0;
  for(int b=lo;b<hi;b++){
   const BnBlockQ4_0&a=ew[row*nb+b];const BnQ4CudaInput&v=ex[b];int dot=0;
   for(int j=0;j<16;j++)dot+=((int)(a.qs[j]&15)-8)*(int)v.qs[j]+((int)(a.qs[j]>>4)-8)*(int)v.qs[j+16];
   acc=fmaf((float)dot*cuda_fp16_to_fp32(a.d),__half2float(v.d),acc);
  }
  if(!have)tail=acc;else prefix=__fadd_rn(prefix,acc);have=true;
 }
 out[(size_t)item*rows+row]=__fadd_rn(tail,prefix);
}

/* One warp computes a 16-row by 8-token tile with integer MMA. Each Q4_0
 * block still contributes one FP32 FMA in the reference partition order. */
template <bool ROUTED, int WARPS>
static __global__ void q4_mmq_mma_tile(float *out, const BnBlockQ4_0 *w,
    const BnQ4CudaInput *x, const int *map, int rows, int cols,
    int batch_tokens, int total_tokens, int expert_index, int experts,
    int k, int input_stride, int jwidth, int grid, int geometry_rows,
    int row_offset) {
    __shared__ __align__(16) int8_t sa[2][16][32];
    __shared__ __align__(16) int8_t sb[2][WARPS * 8][32];
    __shared__ uint16_t wd[2][16];
    __shared__ half xd[2][WARPS * 8];
    __shared__ int route[WARPS * 8];
    __shared__ int active;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row0 = (int)blockIdx.x * 16;
    int token0 = (int)blockIdx.y * (WARPS * 8);
    int expert = ROUTED ? (int)blockIdx.z : expert_index;
    int nb = cols / 32;
    int items = total_tokens * k;
    for (int t = tid; t < WARPS * 8; t += blockDim.x) {
        int rank = token0 + t;
        route[t] = rank < (ROUTED ? total_tokens : batch_tokens)
            ? (ROUTED ? map[2 * items + expert * total_tokens + rank]
                      : rank)
            : -1;
    }
    __syncthreads();
    if (ROUTED) {
        if (tid == 0) {
            active = 0;
            for (int t = 0; t < WARPS * 8; t++)
                active |= route[t] >= 0;
        }
        __syncthreads();
        if (!active) return;
    }
    const BnBlockQ4_0 *ew = ROUTED
        ? w + (size_t)expert * rows * nb : w;
    int xt = (total_tokens + jwidth - 1) / jwidth;
    int tiles = ((geometry_rows + 127) / 128) * xt * experts;
    long long tile = (((row0 + row_offset) / 128) * experts + expert) * xt
                   + token0 / jwidth;
    long long base = tile * nb;
    long long total = (long long)tiles * nb;
    int first_bid = grid == tiles ? (int)tile : (int)((base * grid) / total) - 2;
    int last_bid = grid == tiles ? (int)tile
        : (int)(((base + nb + 8) * grid) / total) + 2;
    if (first_bid < 0) first_bid = 0;
    if (last_bid >= grid) last_bid = grid - 1;
    float tail[4] = {0.f, 0.f, 0.f, 0.f};
    float prefix[4] = {0.f, 0.f, 0.f, 0.f};
    bool have = false;
    for (int bid = last_bid; bid >= first_bid; bid--) {
        long long beg = (long long)bid * total / grid;
        long long end = (long long)(bid + 1) * total / grid;
        beg -= (beg % nb) % 8;
        end -= (end % nb) % 8;
        beg -= base;
        end -= base;
        int lo = beg < 0 ? 0 : (int)beg;
        int hi = end > nb ? nb : (int)end;
        if (lo >= hi) continue;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = lo; b < hi; b += 2) {
            int n_stage = hi - b < 2 ? 1 : 2;
            /* Stage adjacent blocks together while retaining the reference
             * FP32 accumulation order within each partition. */
            for (int stage = 0; stage < n_stage; stage++) {
                int block = b + stage;
                for (int i = tid; i < 16 * 32; i += blockDim.x) {
                    int ri = i / 32;
                    int col = i & 31;
                    int row = row0 + ri;
                    int value = 0;
                    if (row < rows) {
                        const BnBlockQ4_0 &blk = ew[(size_t)row * nb + block];
                        int packed = blk.qs[col & 15];
                        value = ((col & 16) ? packed >> 4 : packed & 15) - 8;
                    }
                    sa[stage][ri][col] = (int8_t)value;
                }
                if (tid < 16) {
                    int row = row0 + tid;
                    wd[stage][tid] = row < rows ? ew[(size_t)row * nb + block].d : 0;
                }
                for (int i = tid; i < WARPS * 8 * 32; i += blockDim.x) {
                    int ti = i / 32;
                    int col = i & 31;
                    int item = route[ti];
                    int token = ROUTED ? item / input_stride : item;
                    sb[stage][ti][col] = item >= 0 ? x[(size_t)token * nb + block].qs[col] : 0;
                }
                for (int ti = tid; ti < WARPS * 8; ti += blockDim.x) {
                    int item = route[ti];
                    int token = ROUTED ? item / input_stride : item;
                    xd[stage][ti] = item >= 0 ? x[(size_t)token * nb + block].d
                                          : __float2half(0.f);
                }
            }
            __syncthreads();
            for (int stage = 0; stage < n_stage; stage++) {
                int a[4], bv[2], c[4] = {0, 0, 0, 0};
                unsigned ash = (unsigned)__cvta_generic_to_shared(
                    &sa[stage][lane % 16][(lane / 16) * 16]);
                unsigned bsh = (unsigned)__cvta_generic_to_shared(
                    &sb[stage][warp * 8 + lane % 8][((lane / 8) * 16) % 32]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 "
                    "{%0,%1,%2,%3}, [%4];"
                    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                    : "r"(ash));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 "
                    "{%0,%1}, [%2];"
                    : "=r"(bv[0]), "=r"(bv[1]) : "r"(bsh));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%0,%1,%2,%3};"
                    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(bv[0]), "r"(bv[1]));
                for (int l = 0; l < 4; l++) {
                    int ri = (l / 2) * 8 + lane / 4;
                    int ti = warp * 8 + (lane % 4) * 2 + (l & 1);
                    acc[l] = fmaf((float)c[l] * cuda_fp16_to_fp32(wd[stage][ri]),
                                  __half2float(xd[stage][ti]), acc[l]);
                }
            }
            __syncthreads();
        }
        for (int l = 0; l < 4; l++) {
            if (!have) tail[l] = acc[l];
            else prefix[l] = __fadd_rn(prefix[l], acc[l]);
        }
        have = true;
    }
    for (int l = 0; l < 4; l++) {
        int ri = (l / 2) * 8 + lane / 4;
        int ti = warp * 8 + (lane % 4) * 2 + (l & 1);
        int row = row0 + ri;
        int item = route[ti];
        if (row < rows && item >= 0)
            out[(size_t)item * rows + row] = __fadd_rn(tail[l], prefix[l]);
    }
}

#endif
