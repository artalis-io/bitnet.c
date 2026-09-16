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

#endif
