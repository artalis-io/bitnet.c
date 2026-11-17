#ifndef BN_GPU_CUDA_GDN_CP_CUH
#define BN_GPU_CUDA_GDN_CP_CUH
#include <type_traits>
/* Fixed SM120 CP contract: FP16 inverse compositions, BF16 stage rounding,
 * raw-F32 TF32 fixup, and a separate final recurrent sweep. Fixed chunk states
 * represent chunk END positions; they are not the final output state.
 * Included only by gpu_cuda.cu, after the resident request implementation. */
namespace bn_cuda_gdn_cp {
using namespace nvcuda;
static __global__ void gram(float*out,const __nv_bfloat16*k,int nt){
 int tid=threadIdx.x,warp=tid/32,head=blockIdx.y,chunk=blockIdx.x;
 __shared__ __align__(32) __nv_bfloat16 key[64*128];
 for(int i=tid;i<64*128;i+=128){int row=i/128,col=i%128,t=chunk*64+row;key[i]=t<nt?k[((size_t)t*16+head/3)*128+col]:__float2bfloat16_rn(0);}
 __syncthreads();
 for(int col=0;col<64;col+=16){
  wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0.0f);
  for(int base=0;base<128;base+=16){
   wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;
   wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;
   wmma::load_matrix_sync(a,key+warp*16*128+base,128);wmma::load_matrix_sync(b,key+col*128+base,128);wmma::mma_sync(c,a,b,c);
  }
  wmma::store_matrix_sync(out+((size_t)chunk*48+head)*4096+warp*16*64+col,c,64,wmma::mem_row_major);
 }
}
static __device__ void multiply(const __half*a,const __half*b,float*c){
 wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major>fa;
 wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major>fb;
 wmma::fragment<wmma::accumulator,16,16,16,float>fc;wmma::fill_fragment(fc,0.0f);
 wmma::load_matrix_sync(fa,a,16);wmma::load_matrix_sync(fb,b,16);wmma::mma_sync(fc,fa,fb,fc);wmma::store_matrix_sync(c,fc,16,wmma::mem_row_major);
}
static __device__ float exp2fast(float x);
static __device__ float log2fast(float x);
/* A single valid row has no strictly lower-triangular entries. The inverse
 * is exactly [1], independent of the Gram matrix and decay. Preserve the
 * padded tensor consumed by prefill_weights, including its positive zeros. */
static __global__ void inverse_one(__nv_bfloat16 *out,const float *beta) {
 int head=blockIdx.x;
 for(int i=threadIdx.x;i<4096;i+=blockDim.x)
  out[head*4096+i]=__float2bfloat16_rn(i==0?__fmul_rn(-beta[head],1.f):0.f);
}
template<bool GatedInverse>
static __global__ void inv(__nv_bfloat16*out,const float*gram,const float*beta,const float*alpha,int nt){
 int tid=threadIdx.x,warp=tid/32,lane=tid%32,head=blockIdx.y,chunk=blockIdx.x;size_t block=((size_t)chunk*48+head)*4096;
 __shared__ __align__(32) __half mat[4096],aa[4][256],bb[4][256],dc[4][256],dc32[1024],partials[2][1024];
 __shared__ __align__(32) float cc[4][256];
 /* NONCP applies decay before the inverse's FP16 rounding. CP keeps its
  * original ungated inverse and applies decay after inversion. */
 __shared__ float logs[64];
 if(GatedInverse){
  if(tid<32){int valid=min(64,nt-chunk*64);float a=log2fast(__fadd_rn(tid<valid?alpha[(chunk*64+tid)*48+head]:1.f,1e-10f)),b=log2fast(__fadd_rn(tid+32<valid?alpha[(chunk*64+tid+32)*48+head]:1.f,1e-10f));for(int off=1;off<32;off*=2){float x=__shfl_up_sync(0xffffffffu,a,off),y=__shfl_up_sync(0xffffffffu,b,off);if(tid>=off){a=__fadd_rn(a,x);b=__fadd_rn(b,y);}}b=__fadd_rn(b,__shfl_sync(0xffffffffu,a,31));logs[tid]=a;logs[tid+32]=b;}
  __syncthreads();
 }
 for(int i=tid;i<4096;i+=128){int row=i/64,col=i%64;float value=0;
  if(row>col&&chunk*64+row<nt){value=gram[block+i];if(GatedInverse)value=__fmul_rn(value,exp2fast(__fsub_rn(logs[row],logs[col])));value=__fmul_rn(value,beta[(chunk*64+row)*48+head]);}
  mat[i]=__float2half_rn(value);
 }
 __syncthreads();
 if(tid<64){int b=tid/8,row=tid%8;float frag[8];for(int j=0;j<8;j++)frag[j]=row==j?1.0f:row<j?0.0f:__half2float(mat[(b*8+row)*64+b*8+j]);
  for(int src=0;src<7;src++){float scale=-frag[src];for(int j=0;j<src;j++){float value=__shfl_sync(0xffffffffu,frag[j],src,8);if(row>src)frag[j]=__fmaf_rn(scale,value,frag[j]);}if(row>src)frag[src]=scale;}
  for(int j=0;j<8;j++)mat[(b*8+row)*64+b*8+j]=__float2half_rn(frag[j]);
 }
 __syncthreads();
 for(int size=16;size<=32;size*=2){int h=size/2,groups=64/size;
  if(warp<groups){int start=warp*size;
   for(int i=lane;i<256;i+=32){int row=i/16,col=i%16;aa[warp][i]=row<h&&col<h?mat[(start+h+row)*64+start+h+col]:__float2half_rn(0);bb[warp][i]=row<h&&col<h?mat[(start+h+row)*64+start+col]:__float2half_rn(0);}
   __syncwarp();multiply(aa[warp],bb[warp],cc[warp]);__syncwarp();
   for(int i=lane;i<256;i+=32){int row=i/16,col=i%16;dc[warp][i]=__float2half_rn(-cc[warp][i]);aa[warp][i]=row<h&&col<h?mat[(start+row)*64+start+col]:__float2half_rn(0);}
   __syncwarp();multiply(dc[warp],aa[warp],cc[warp]);__syncwarp();
   for(int i=lane;i<256;i+=32){int row=i/16,col=i%16;if(row<h&&col<h)mat[(start+h+row)*64+start+col]=__float2half_rn(cc[warp][i]);}
  }__syncthreads();
 }
 int x=warp%2,y=warp/2;
 wmma::fragment<wmma::accumulator,16,16,16,float> accum;wmma::fill_fragment(accum,0.0f);
 for(int base=0;base<32;base+=16){
  for(int i=lane;i<256;i+=32){int row=i/16,col=i%16;aa[warp][i]=mat[(32+y*16+row)*64+32+base+col];bb[warp][i]=mat[(32+base+row)*64+x*16+col];}
  __syncwarp();wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major>b;
  wmma::load_matrix_sync(a,aa[warp],16);wmma::load_matrix_sync(b,bb[warp],16);wmma::mma_sync(accum,a,b,accum);
 }
 wmma::store_matrix_sync(cc[warp],accum,16,wmma::mem_row_major);__syncwarp();
 for(int i=lane;i<256;i+=32)dc32[(y*16+i/16)*32+x*16+i%16]=__float2half_rn(-cc[warp][i]);
 __syncthreads();
 for(int coltile=0;coltile<2;coltile++){
  for(int i=lane;i<256;i+=32){int row=i/16,col=i%16;aa[warp][i]=dc32[(y*16+row)*32+x*16+col];bb[warp][i]=mat[(x*16+row)*64+coltile*16+col];}
  __syncwarp();multiply(aa[warp],bb[warp],cc[warp]);__syncwarp();
  for(int i=lane;i<256;i+=32)partials[x][(y*16+i/16)*32+coltile*16+i%16]=__float2half_rn(cc[warp][i]);
 }
 __syncthreads();for(int i=tid;i<1024;i+=128)mat[(32+i/32)*64+i%32]=__hadd(partials[0][i],partials[1][i]);__syncthreads();
 for(int i=tid;i<4096;i+=128){int row=i/64,col=i%64;float value=0;if(chunk*64+row<nt&&chunk*64+col<nt)value=__fmul_rn(-beta[(chunk*64+col)*48+head],__half2float(mat[i]));out[block+col*64+row]=__float2bfloat16_rn(value);}
}
using namespace nvcuda;
static __device__ float exp2fast(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
static __device__ float log2fast(float x){float y;asm("lg2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
static __device__ bool active(int group,int nt,int cp,int bi){return group/48*cp+bi*64<nt;}
static __global__ void init(float*m,float*n,int groups){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups*16384)return;m[i]=(i%16384)/128==(i%128)?1.f:0.f;n[i]=0.f;}
static __global__ void prepare(__nv_bfloat16*kt,__nv_bfloat16*vt,__nv_bfloat16*tb,__nv_bfloat16*z,__nv_bfloat16*m16,__nv_bfloat16*n16,float*coeff,float*neg,const __nv_bfloat16*k,const __nv_bfloat16*v,const __nv_bfloat16*t,const float*alpha,const float*m,const float*n,int nt,int cp,int bi){
 int group=blockIdx.x,tid=threadIdx.x,head=group%48,start=group/48*cp+bi*64;if(start>=nt)return;int valid=min(64,nt-start);size_t a=(size_t)group*8192,b=(size_t)group*4096,c=(size_t)group*16384;
 for(int i=tid;i<8192;i+=128){int d=i/64,tok=i%64;auto zero=__float2bfloat16_rn(0);kt[a+i]=tok<valid?k[((size_t)(start+tok)*16+head/3)*128+d]:zero;vt[a+i]=tok<valid?v[((size_t)(start+tok)*48+head)*128+d]:zero;if(!bi)z[a+i]=kt[a+i];}
 for(int i=tid;i<4096;i+=128)tb[b+i]=t[((size_t)(start/64)*48+head)*4096+i];
 for(int i=tid;i<16384;i+=128){m16[c+i]=__float2bfloat16_rn(m[c+i]);n16[c+i]=__float2bfloat16_rn(n[c+i]);}
 if(tid<32){float a0=log2fast(__fadd_rn(tid<valid?alpha[(start+tid)*48+head]:1.f,1e-10f));float a1=log2fast(__fadd_rn(tid+32<valid?alpha[(start+tid+32)*48+head]:1.f,1e-10f));for(int off=1;off<32;off*=2){float x=__shfl_up_sync(0xffffffffu,a0,off),y=__shfl_up_sync(0xffffffffu,a1,off);if(tid>=off){a0=__fadd_rn(a0,x);a1=__fadd_rn(a1,y);}}a1=__fadd_rn(a1,__shfl_sync(0xffffffffu,a0,31));float end=__shfl_sync(0xffffffffu,a1,31);float last=valid<=32?__shfl_sync(0xffffffffu,a0,valid-1):__shfl_sync(0xffffffffu,a1,valid-33);if(tid==0)coeff[group]=exp2fast(last);neg[group*64+tid]=tid<valid?-exp2fast(__fsub_rn(end,a0)):0.f;neg[group*64+tid+32]=tid+32<valid?-exp2fast(__fsub_rn(end,a1)):0.f;}
}
// Mode0: zero C; Mode1: C then scale; Mode2: scaled C; Mode3: unscaled C.
template<bool ColB,int Mode,bool BFOut>
static __global__ void mm(void*dest,const __nv_bfloat16*a,const __nv_bfloat16*b,const float*coeff,int M,int N,int K,int nt,int cp,int bi){int group=blockIdx.z;if(!active(group,nt,cp,bi))return;int row=blockIdx.y*16,col=blockIdx.x*16,lane=threadIdx.x;size_t abase=(size_t)group*M*K,bbase=(size_t)group*K*N,cbase=(size_t)group*M*N;wmma::fragment<wmma::accumulator,16,16,16,float> c;
 if(Mode){wmma::load_matrix_sync(c,(float*)dest+cbase+row*N+col,N,wmma::mem_row_major);if(Mode==2)for(int j=0;j<c.num_elements;j++)c.x[j]=__fmul_rn(c.x[j],coeff[group]);}else wmma::fill_fragment(c,0.f);
 for(int base=0;base<K;base+=16){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>fa;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,typename std::conditional<ColB,wmma::col_major,wmma::row_major>::type>fb;wmma::load_matrix_sync(fa,a+abase+row*K+base,K);wmma::load_matrix_sync(fb,b+bbase+(ColB?col*K+base:base*N+col),ColB?K:N);wmma::mma_sync(c,fa,fb,c);}
 if(Mode==1)for(int j=0;j<c.num_elements;j++)c.x[j]=__fmul_rn(c.x[j],coeff[group]);
 if(BFOut){__shared__ __align__(32) float tile[256];wmma::store_matrix_sync(tile,c,16,wmma::mem_row_major);__syncwarp();for(int i=lane;i<256;i+=32)((__nv_bfloat16*)dest)[cbase+(row+i/16)*N+col+i%16]=__float2bfloat16_rn(tile[i]);}
 else wmma::store_matrix_sync((float*)dest+cbase+row*N+col,c,N,wmma::mem_row_major);
}
static __global__ void make_y(__nv_bfloat16*y,const float*sk,const __nv_bfloat16*v,const float*coeff,const float*neg,int nt,int cp,int bi){int group=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8192||!active(group,nt,cp,bi))return;size_t off=(size_t)group*8192+i;float val=__fmul_rn(__bfloat162float(v[off]),neg[group*64+i%64]);if(bi)val=__fmaf_rn(coeff[group],sk[off],val);y[off]=__float2bfloat16_rn(val);}

static __global__ void fixup(float*fixed,const float*m,const float*n,const float*initial,int chunks){
 int head=blockIdx.x,rowbase=blockIdx.y*64,tid=threadIdx.x,lane=tid%32,warp=tid/32;
 extern __shared__ float shared[];
 for(int i=tid;i<64*128;i+=128)shared[i]=initial[(size_t)head*16384+rowbase*128+i];
 __syncthreads();
 for(int chunk=0;chunk<chunks;chunk++){
  size_t base=((size_t)chunk*48+head)*16384;
  float d[16][4];
  for(int tile=0;tile<16;tile++)for(int j=0;j<4;j++){int row=rowbase+warp*16+lane/4+j/2*8,col=tile*8+2*(lane%4)+j%2;d[tile][j]=n[base+row*128+col];}
  for(int k=0;k<128;k+=8){int c=k+2*(lane%4),r=warp*16+lane/4;
   unsigned a0=__float_as_uint(shared[r*128+c]),a1=__float_as_uint(shared[(r+8)*128+c]),a2=__float_as_uint(shared[r*128+c+1]),a3=__float_as_uint(shared[(r+8)*128+c+1]);
   for(int tile=0;tile<16;tile++){int col=tile*8+lane/4;unsigned b0=__float_as_uint(m[base+c*128+col]),b1=__float_as_uint(m[base+(c+1)*128+col]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d[tile][0]),"+f"(d[tile][1]),"+f"(d[tile][2]),"+f"(d[tile][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
   }
  }
  for(int tile=0;tile<16;tile++)for(int j=0;j<4;j++){int row=warp*16+lane/4+j/2*8,col=tile*8+2*(lane%4)+j%2;shared[row*128+col]=d[tile][j];fixed[base+(rowbase+row)*128+col]=d[tile][j];}
  __syncthreads();
 }
}

static __global__ void prefill_init(float*s,const float*initial,const float*fixed,int groups){int group=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=16384)return;int chunk=group/48,head=group%48;s[(size_t)group*16384+i]=chunk?fixed[((size_t)(chunk-1)*48+head)*16384+i]:initial[head*16384+i];}
static __global__ void prefill_prepare(__nv_bfloat16*kt,__nv_bfloat16*qt,__nv_bfloat16*qr,__nv_bfloat16*vt,__nv_bfloat16*s16,float*logs,float*gammas,float*coeff,const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,const float*alpha,const float*s,int nt,int cp,int bi){
 int group=blockIdx.x,tid=threadIdx.x,head=group%48,start=group/48*cp+bi*64;if(start>=nt)return;int valid=min(64,nt-start);size_t base=(size_t)group*8192;
 for(int i=tid;i<8192;i+=blockDim.x){int d=i/64,t=i%64;auto zero=__float2bfloat16_rn(0);auto qv=t<valid?q[((size_t)(start+t)*16+head/3)*128+d]:zero;kt[base+i]=t<valid?k[((size_t)(start+t)*16+head/3)*128+d]:zero;vt[base+i]=t<valid?v[((size_t)(start+t)*48+head)*128+d]:zero;qt[base+i]=qv;}
 /* QR is token-major: a separate contiguous pass avoids strided stores. */
 for(int i=tid;i<8192;i+=blockDim.x){int t=i/128,d=i%128;qr[base+i]=t<valid?q[((size_t)(start+t)*16+head/3)*128+d]:__float2bfloat16_rn(0);}
 for(int i=tid;i<16384;i+=blockDim.x)s16[(size_t)group*16384+i]=__float2bfloat16_rn(s[(size_t)group*16384+i]);
 if(tid<32){float a=log2fast(__fadd_rn(tid<valid?alpha[(start+tid)*48+head]:1.f,1e-10f)),b=log2fast(__fadd_rn(tid+32<valid?alpha[(start+tid+32)*48+head]:1.f,1e-10f));for(int off=1;off<32;off*=2){float x=__shfl_up_sync(0xffffffffu,a,off),y=__shfl_up_sync(0xffffffffu,b,off);if(tid>=off){a=__fadd_rn(a,x);b=__fadd_rn(b,y);}}b=__fadd_rn(b,__shfl_sync(0xffffffffu,a,31));float last=valid<=32?__shfl_sync(0xffffffffu,a,valid-1):__shfl_sync(0xffffffffu,b,valid-33);if(tid==0)coeff[group]=exp2fast(last);logs[group*64+tid]=a;logs[group*64+tid+32]=b;gammas[group*64+tid]=exp2fast(a);gammas[group*64+tid+32]=exp2fast(b);}
}

template<bool GatedInverse>
static __global__ void prefill_weights(__nv_bfloat16*qk,__nv_bfloat16*w,const float*dot,const __nv_bfloat16*t,const float*logs,int nt,int cp,int bi,float scale){int group=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4096||!active(group,nt,cp,bi))return;int start=group/48*cp+bi*64,head=group%48,len=min(cp,nt-group/48*cp),valid=min(64,nt-start),r=i/64,c=i%64;bool final=bi==0||(bi+1)*64>=len;bool pred=r>=c&&r<valid&&c<valid;float gamma=pred?exp2fast(__fsub_rn(logs[group*64+r],logs[group*64+c])):0.f;size_t off=(size_t)group*4096+i;float qkv=__fmul_rn(__fmul_rn(dot[off],gamma),scale);float inverse=__bfloat162float(t[((size_t)(start/64)*48+head)*4096+c*64+r]);float wt=GatedInverse?-inverse:__fmul_rn(-gamma,inverse);if(final&&!pred){qkv=0;wt=0;}qk[off]=__float2bfloat16_rn(qkv);w[off]=__float2bfloat16_rn(wt);}
static __global__ void prefill_residual(__nv_bfloat16*a,float*history,const float*sk,const __nv_bfloat16*v,const float*gammas,int nt,int cp,int bi,float scale){int group=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8192||!active(group,nt,cp,bi))return;size_t off=(size_t)group*8192+i;float gamma=gammas[group*64+i%64];history[off]=__fmul_rn(__fmul_rn(gamma,scale),history[off]);float rounded=__bfloat162float(__float2bfloat16_rn(__fmul_rn(sk[off],gamma)));a[off]=__float2bfloat16_rn(__fsub_rn(__bfloat162float(v[off]),rounded));}
static __global__ void prefill_newv(__nv_bfloat16*nb,__nv_bfloat16*decay,const float*newv,const float*logs,int nt,int cp,int bi){int group=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8192||!active(group,nt,cp,bi))return;int start=group/48*cp+bi*64,valid=min(64,nt-start),tok=i%64;size_t off=(size_t)group*8192+i;auto value=__float2bfloat16_rn(newv[off]);nb[off]=value;float factor=tok<valid?exp2fast(__fsub_rn(logs[group*64+valid-1],logs[group*64+tok])):0.f;decay[off]=__float2bfloat16_rn(__fmul_rn(__bfloat162float(value),factor));}
static __global__ void prefill_store(__nv_bfloat16*out,const float*matrix,int nt,int cp,int bi){int group=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=8192||!active(group,nt,cp,bi))return;int tok=i%64,d=i/64,start=group/48*cp+bi*64;if(start+tok<nt)out[((size_t)(start+tok)*48+group%48)*128+d]=__float2bfloat16_rn(matrix[(size_t)group*8192+i]);}

/* One bounded allocation per request. The stage tensors remain device resident;
 * no execution-time allocation, readback, or synchronization is permitted. */
struct Layout {
    float *gram, *m, *n, *sk, *coeff, *neg, *fixed, *s;
    float *logs, *gammas, *dot, *history, *newv, *alpha;
    __nv_bfloat16 *t, *kt, *vt, *tb, *z, *x, *y, *m16, *n16;
    __nv_bfloat16 *qt, *qr, *s16, *qk, *w, *a, *nb, *decay;
};

static size_t layout(void *storage, int cap, int chunk, Layout *out) {
    if (cap <= 0 || cap > 8192 || chunk < 64 || chunk > 4096 || chunk % 64)
        return 0;
    size_t tiles = (size_t)((cap + 63) / 64) * 48;
    size_t groups = (size_t)((cap + chunk - 1) / chunk) * 48;
    size_t offset = 0;
    Layout l = {};
    /* Every matrix base is 256-byte aligned for the WMMA row strides. The
     * supported 8192-token bound also bounds every product and grid dimension. */
#define BN_CP_FIELD(NAME, TYPE, COUNT) do { \
    offset = (offset + 255) & ~(size_t)255; \
    if (storage) l.NAME = (TYPE *)((unsigned char *)storage + offset); \
    offset += (COUNT) * sizeof(TYPE); \
} while (0)
    BN_CP_FIELD(alpha, float, (size_t)cap * 48);
    BN_CP_FIELD(gram, float, tiles * 4096);
    BN_CP_FIELD(t, __nv_bfloat16, tiles * 4096);
    BN_CP_FIELD(m, float, groups * 16384);
    BN_CP_FIELD(n, float, groups * 16384);
    BN_CP_FIELD(fixed, float, groups * 16384);
    BN_CP_FIELD(s, float, groups * 16384);
    BN_CP_FIELD(sk, float, groups * 8192);
    BN_CP_FIELD(coeff, float, groups);
    BN_CP_FIELD(neg, float, groups * 64);
    BN_CP_FIELD(logs, float, groups * 64);
    BN_CP_FIELD(gammas, float, groups * 64);
    BN_CP_FIELD(dot, float, groups * 4096);
    BN_CP_FIELD(history, float, groups * 8192);
    BN_CP_FIELD(newv, float, groups * 8192);
    BN_CP_FIELD(kt, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(vt, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(tb, __nv_bfloat16, groups * 4096);
    BN_CP_FIELD(z, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(x, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(y, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(m16, __nv_bfloat16, groups * 16384);
    BN_CP_FIELD(n16, __nv_bfloat16, groups * 16384);
    BN_CP_FIELD(qt, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(qr, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(s16, __nv_bfloat16, groups * 16384);
    BN_CP_FIELD(qk, __nv_bfloat16, groups * 4096);
    BN_CP_FIELD(w, __nv_bfloat16, groups * 4096);
    BN_CP_FIELD(a, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(nb, __nv_bfloat16, groups * 8192);
    BN_CP_FIELD(decay, __nv_bfloat16, groups * 8192);
#undef BN_CP_FIELD
    if (out) *out = l;
    return offset;
}

static size_t workspace_size(void *ctx_ptr, int cap, int chunk) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    /* Raw-F32 TF32 fixup is qualified on SM120 only. */
    return ctx && ctx->compute_capability == 1200 ? layout(NULL, cap, chunk, NULL) : 0;
}

/* HV48 bypasses the reference SM120 short-head heuristic. Target one wave
 * of MN CTAs, with the reference's 512-token granularity. */
static int chunk_tokens(void *ctx_ptr, int nt) {
    const BnCudaCtx *ctx = (const BnCudaCtx *)ctx_ptr;
    if (!ctx || ctx->compute_capability != 1200 || ctx->multiprocessors <= 0 ||
        nt <= 0 || nt > 8192) return 0;
    int target = ctx->multiprocessors / 48;
    if (target < 1) target = 1;
    int chunk = (((nt + target - 1) / target + 511) / 512) * 512;
    return chunk <= 4096 ? chunk : 0;
}

static __global__ void stage_log_decay(float *out, const float *in, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = expf(in[i]);
}

static int reserve(void *ctx_ptr, void *ptr, int cap, int chunk) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    bn_cuda_gdn::ResidentRequest *r = (bn_cuda_gdn::ResidentRequest *)ptr;
    size_t bytes = workspace_size(ctx, cap, chunk);
    if (!bytes || !r || r->backend_owner != ctx || bn_cuda_gdn::set_device(r))
        return -1;
    if (r->cp_chunk == chunk && r->cp_capacity >= cap) return 0;
    if (bn_cuda_gdn::synchronize(r)) return -1;
    void *next = NULL;
    if (cudaMalloc(&next, bytes) != cudaSuccess) return -1;
    cudaFree(r->cp_workspace);
    r->cp_workspace = next;
    r->cp_capacity = cap;
    r->cp_chunk = chunk;
    return 0;
}

static int validate(void *ctx_ptr, void *ptr, const BnGPUGDNCPPrefillResidentPlan *p,
    void **data, bn_cuda_gdn::ResidentRequest *state_owner = NULL) {
    BnCudaCtx *ctx = (BnCudaCtx *)ctx_ptr;
    bn_cuda_gdn::ResidentRequest *r = (bn_cuda_gdn::ResidentRequest *)ptr;
    if (!ctx || !r || !p || r->backend_owner != ctx ||
        (p->recurrence != BN_GPU_GDN_PREFILL_CP && p->recurrence != BN_GPU_GDN_PREFILL_NONCP) ||
        !workspace_size(ctx, p->n_tokens, p->chunk_tokens) ||
        p->n_tokens > r->cp_capacity || p->chunk_tokens < r->cp_chunk ||
        (p->decay_domain != BN_GPU_GDN_DECAY_MULTIPLIER &&
         p->decay_domain != BN_GPU_GDN_DECAY_LOG) ||
        !r->cp_workspace || p->hk != 16 || p->hv != 48 || p->dk != 128 ||
        p->dv != 128 || p->q_scale != 0x1.6a09e6p-4f || bn_cuda_gdn::set_device(r))
        return -1;
    const int nt = p->n_tokens;
    const size_t nq = (size_t)nt * 2048, nv = (size_t)nt * 6144;
    const size_t ng = (size_t)nt * 48, ns = 48u * 16384;
    BnGPUResidentView views[] = {p->q, p->k, p->v, p->alpha, p->beta,
                                p->initial_state, p->next_state, p->out};
    const size_t counts[] = {nq, nq, nv, ng, ng, ns, ns, nv};
    for (unsigned i = 0; i < 8; i++) {
        data[i] = bn_cuda_gdn::view_data(state_owner && (i==5 || i==6) ? state_owner : r, views[i]);
        const bool bf16 = i < 3 || i == 7;
        if (!data[i] || views[i].count != counts[i] ||
            (bf16 ? !bn_quant_format_is_bf16(views[i].type)
                  : !bn_quant_format_is_f32(views[i].type)) ||
            (i < 3 && ((uintptr_t)data[i] & 15u))) return -1;
    }
    for (unsigned i = 0; i < 8; i++)
        if ((i != 6 && bn_cuda_gdn::overlap(views[6], views[i])) ||
            (i != 7 && bn_cuda_gdn::overlap(views[7], views[i]))) return -1;
    return 0;
}
static int launch(void *ptr, const BnGPUGDNCPPrefillResidentPlan *p, void **data) {
    bn_cuda_gdn::ResidentRequest *r = (bn_cuda_gdn::ResidentRequest *)ptr;
    const bool noncp=p->recurrence==BN_GPU_GDN_PREFILL_NONCP;
    const int nt=p->n_tokens, cp=noncp?nt:p->chunk_tokens, groups=noncp?48:(nt+cp-1)/cp*48;
    const size_t ng=(size_t)nt*48, ns=48u*16384;
    Layout l;
    layout(r->cp_workspace, r->cp_capacity, r->cp_chunk, &l);
    const __nv_bfloat16 *q = (const __nv_bfloat16 *)data[0];
    const __nv_bfloat16 *k = (const __nv_bfloat16 *)data[1];
    const __nv_bfloat16 *v = (const __nv_bfloat16 *)data[2];
    const float *alpha = (const float *)data[3], *beta = (const float *)data[4];
    const float *initial = (const float *)data[5];
    const cudaStream_t stream = r->stream;
    if (cudaGetLastError() != cudaSuccess) { r->failed = 1; return -1; }
    if (p->decay_domain == BN_GPU_GDN_DECAY_LOG) {
        stage_log_decay<<<(ng+255)/256,256,0,stream>>>(l.alpha,alpha,(int)ng);
        alpha = l.alpha;
    }
    dim3 tg((nt + 63) / 64, 48);
    if(noncp && nt==1) {
        inverse_one<<<48,256,0,stream>>>(l.t,beta);
    } else {
        gram<<<tg,128,0,stream>>>(l.gram,k,nt);
        if(noncp)inv<true><<<tg,128,0,stream>>>(l.t,l.gram,beta,alpha,nt);
        else inv<false><<<tg,128,0,stream>>>(l.t,l.gram,beta,alpha,nt);
    }
    /* Chunk zero starts directly from initial_state. With one CP chunk,
     * no consumer reads MN or fixup, including the final-state sweep below. */
    if (!noncp && nt > cp) {
        init<<<(groups*16384+255)/256,256,0,stream>>>(l.m,l.n,groups);
        for (int bi = 0; bi < (min(nt,cp)+63)/64; bi++) {
            prepare<<<groups,128,0,stream>>>(l.kt,l.vt,l.tb,l.z,l.m16,l.n16,
                l.coeff,l.neg,k,v,l.t,alpha,l.m,l.n,nt,cp,bi);
            mm<true,0,true><<<dim3(4,8,groups),32,0,stream>>>(l.x,l.kt,l.tb,l.coeff,128,64,64,nt,cp,bi);
            if (bi) {
                mm<false,0,true><<<dim3(4,8,groups),32,0,stream>>>(l.z,l.m16,l.kt,l.coeff,128,64,128,nt,cp,bi);
                mm<false,0,false><<<dim3(4,8,groups),32,0,stream>>>(l.sk,l.n16,l.kt,l.coeff,128,64,128,nt,cp,bi);
            }
            mm<true,1,false><<<dim3(8,8,groups),32,0,stream>>>(l.m,l.z,l.x,l.coeff,128,128,64,nt,cp,bi);
            make_y<<<dim3(32,groups),256,0,stream>>>(l.y,l.sk,l.vt,l.coeff,l.neg,nt,cp,bi);
            mm<true,2,false><<<dim3(8,8,groups),32,0,stream>>>(l.n,l.y,l.x,l.coeff,128,128,64,nt,cp,bi);
        }
        fixup<<<dim3(48,2),128,64*128*4,stream>>>(l.fixed,l.m,l.n,initial,(nt+cp-1)/cp);
    }
    prefill_init<<<dim3(64,groups),256,0,stream>>>(l.s,initial,l.fixed,groups);
    for (int bi = 0; bi < (min(nt,cp)+63)/64; bi++) {
        /* Memory copies/casts scale across the block; prefix math stays in warp zero. */
        prefill_prepare<<<groups,512,0,stream>>>(l.kt,l.qt,l.qr,l.vt,l.s16,l.logs,l.gammas,l.coeff,q,k,v,alpha,l.s,nt,cp,bi);
        mm<false,0,false><<<dim3(4,4,groups),32,0,stream>>>(l.dot,l.qr,l.kt,l.coeff,64,64,128,nt,cp,bi);
        if(noncp)prefill_weights<true><<<dim3(16,groups),256,0,stream>>>(l.qk,l.w,l.dot,l.t,l.logs,nt,cp,bi,p->q_scale);
        else prefill_weights<false><<<dim3(16,groups),256,0,stream>>>(l.qk,l.w,l.dot,l.t,l.logs,nt,cp,bi,p->q_scale);
        mm<false,0,false><<<dim3(4,8,groups),32,0,stream>>>(l.history,l.s16,l.qt,l.coeff,128,64,128,nt,cp,bi);
        mm<false,0,false><<<dim3(4,8,groups),32,0,stream>>>(l.sk,l.s16,l.kt,l.coeff,128,64,128,nt,cp,bi);
        prefill_residual<<<dim3(32,groups),256,0,stream>>>(l.a,l.history,l.sk,l.vt,l.gammas,nt,cp,bi,p->q_scale);
        mm<true,0,false><<<dim3(4,8,groups),32,0,stream>>>(l.newv,l.a,l.w,l.coeff,128,64,64,nt,cp,bi);
        prefill_newv<<<dim3(32,groups),256,0,stream>>>(l.nb,l.decay,l.newv,l.logs,nt,cp,bi);
        mm<true,3,false><<<dim3(4,8,groups),32,0,stream>>>(l.history,l.nb,l.qk,l.coeff,128,64,64,nt,cp,bi);
        prefill_store<<<dim3(32,groups),256,0,stream>>>((__nv_bfloat16 *)data[7],l.history,nt,cp,bi);
        mm<true,2,false><<<dim3(8,8,groups),32,0,stream>>>(l.s,l.decay,l.kt,l.coeff,128,128,64,nt,cp,bi);
    }
    cudaError_t copied = cudaMemcpyAsync(data[6], l.s+(size_t)(groups-48)*16384,
        ns*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (copied == cudaSuccess && cudaGetLastError() == cudaSuccess) return 0;
    r->failed = 1;
    return -1;
}

static int prefill(void *ctx, void *ptr, const BnGPUGDNCPPrefillResidentPlan *p) {
    void *data[8];
    return validate(ctx,ptr,p,data) ? -1 : launch(ptr,p,data);
}
static size_t group_workspace_size(void *ctx, int capacity) {
    const BnCudaCtx *c=(const BnCudaCtx *)ctx;
    return c && c->compute_capability==1200 && capacity>0 && capacity<=256
        ? sizeof(bn_cuda_gdn::CPGroupWorkspace)+(size_t)capacity*sizeof(cudaEvent_t) : 0;
}
static int group_reserve(void *ctx,void *ptr,int capacity,size_t budget) {
    bn_cuda_gdn::ResidentRequest *r=(bn_cuda_gdn::ResidentRequest *)ptr;
    size_t bytes=group_workspace_size(ctx,capacity);
    if(!r || r->backend_owner!=ctx || !bytes || bytes>budget || bn_cuda_gdn::set_device(r))return -1;
    if(r->cp_group)return r->cp_group->capacity==capacity?0:-1;
    bn_cuda_gdn::CPGroupWorkspace *w=(bn_cuda_gdn::CPGroupWorkspace *)calloc(1,sizeof(*w));
    if(!w)return -1;
    w->capacity=capacity;w->ready=(cudaEvent_t *)calloc((size_t)capacity,sizeof(*w->ready));
    cudaError_t error=w->ready?cudaSuccess:cudaErrorMemoryAllocation;
    for(int i=0;i<capacity&&error==cudaSuccess;i++)error=cudaEventCreateWithFlags(&w->ready[i],cudaEventDisableTiming);
    if(error==cudaSuccess)error=cudaEventCreateWithFlags(&w->done,cudaEventDisableTiming);
    if(error!=cudaSuccess){bn_cuda_gdn::free_cp_group(w);if(error!=cudaErrorMemoryAllocation)r->failed=1;(void)cudaGetLastError();return -1;}
    r->cp_group=w;return 0;
}
static int group(void *ctx,void *ptr,const BnGPUGDNCPPrefillGroupPlan *p) {
    using namespace bn_cuda_gdn;
    ResidentRequest *r=(ResidentRequest *)ptr;
    if(!r || r->backend_owner!=ctx || r->failed || !p || !p->sequences ||
       !r->cp_group || p->n_sequences<1 || p->n_sequences>r->cp_group->capacity)return -1;
    ResidentRequest *owners[256];void *data[256][8];
    for(int i=0;i<p->n_sequences;i++) {
        owners[i]=(ResidentRequest *)p->sequences[i].request;
        if(!owners[i] || owners[i]==r || owners[i]->failed || owners[i]->backend_owner!=ctx || owners[i]->device!=r->device)return -1;
        for(int j=0;j<i;j++)if(owners[j]==owners[i])return -1;
        if(validate(ctx,r,&p->sequences[i].plan,data[i],owners[i]))return -1;
    }
    for(int i=0;i<p->n_sequences;i++)for(int j=0;j<p->n_sequences;j++)if(i!=j) {
        const BnGPUGDNCPPrefillResidentPlan &a=p->sequences[i].plan, &b=p->sequences[j].plan;
        BnGPUResidentView writes[]={a.next_state,a.out};
        BnGPUResidentView all[]={b.q,b.k,b.v,b.alpha,b.beta,b.initial_state,b.next_state,b.out};
        for(unsigned k=0;k<2;k++)for(unsigned l=0;l<8;l++)if(overlap(writes[k],all[l]))return -1;
    }
    CPGroupWorkspace *w=r->cp_group;
    for(int i=0;i<p->n_sequences;i++)if(cudaEventRecord(w->ready[i],owners[i]->stream)!=cudaSuccess || cudaStreamWaitEvent(r->stream,w->ready[i],0)!=cudaSuccess)goto fail;
    for(int i=0;i<p->n_sequences;i++)if(launch(r,&p->sequences[i].plan,data[i]))goto fail;
    if(cudaEventRecord(w->done,r->stream)!=cudaSuccess)goto fail;
    for(int i=0;i<p->n_sequences;i++)if(cudaStreamWaitEvent(owners[i]->stream,w->done,0)!=cudaSuccess)goto fail;
    return 0;
fail:
    return group_fail(r,owners,p->n_sequences);
}

} // namespace bn_cuda_gdn_cp
#endif
