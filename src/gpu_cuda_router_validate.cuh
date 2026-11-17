#ifndef BN_GPU_CUDA_ROUTER_VALIDATE_CUH
#define BN_GPU_CUDA_ROUTER_VALIDATE_CUH
#include "gpu_cuda_router.h"
namespace bn_cuda_router {
static __global__ void transpose(half*out,const half*in,int E,int D,int*invalid){
 __shared__ uint16_t tile[32][33];int e0=blockIdx.x*32,d0=blockIdx.y*32,x=threadIdx.x,y=threadIdx.y;
 for(int j=0;j<32;j+=8){half v=e0+x<E?in[(size_t)(d0+y+j)*E+e0+x]:__float2half_rn(0);if(!isfinite(__half2float(v)))atomicExch(invalid,1);tile[y+j][x]=__half_as_ushort(v);}
 __syncthreads();for(int j=0;j<32;j+=8)if(e0+y+j<E)out[(size_t)(e0+y+j)*D+d0+x]=__ushort_as_half(tile[x][y+j]);
}
/* Torch Reduce.cuh vectorized input mean: last_pow2(E/4) threads, four
 * independent accumulators, shared halving to32, DOWN shuffles, final factor.
 * Default TU math preserves subnormals; explicit rounding prevents fusion. */
static __global__ void center(const float*bias,half*out,float*mean,int*invalid,int E,float factor){
 __shared__ float red[128],m;int t=threadIdx.x,T=blockDim.x;float a=0,b=0,c=0,d=0;
 if(E>=128){
  for(int i=t;i<E/4;i+=T){float4 v=((const float4*)bias)[i];if(!isfinite(v.x)||!isfinite(v.y)||!isfinite(v.z)||!isfinite(v.w))atomicExch(invalid,1);a=__fadd_rn(a,v.x);b=__fadd_rn(b,v.y);c=__fadd_rn(c,v.z);d=__fadd_rn(d,v.w);}
  int tail=E-E%4+t;if(tail<E){float v=bias[tail];if(!isfinite(v))atomicExch(invalid,1);a=__fadd_rn(a,v);}
 }else{
  if(t<E){a=__fadd_rn(a,bias[t]);if(!isfinite(bias[t]))atomicExch(invalid,1);}
  if(t+T<E){b=__fadd_rn(b,bias[t+T]);if(!isfinite(bias[t+T]))atomicExch(invalid,1);}
 }
 float v=__fadd_rn(__fadd_rn(__fadd_rn(a,b),c),d);red[t]=v;
 for(int offset=T/2;offset>=32;offset/=2){__syncthreads();if(t<offset){v=__fadd_rn(v,red[t+offset]);red[t]=v;}}
 __syncthreads();for(int offset=(T<32?T:32)/2;offset;offset/=2)v=__fadd_rn(v,__shfl_down_sync(__activemask(),v,offset));
 if(!t){m=__fmul_rn(v,factor);*mean=m;}__syncthreads();
 for(int i=t;i<E;i+=T){half h=__float2half_rn(__fsub_rn(bias[i],m));if(!isfinite(m)||!isfinite(__half2float(h)))atomicExch(invalid,1);out[i]=h;}
}
static void *create(void*vctx,const BnGPURouterWeightPlan*p){
 BnCudaCtx*ctx=(BnCudaCtx*)vctx;if(!ctx||!p||ctx->compute_capability<800)return NULL;
 size_t bytes=bn_gpu_router_weight_bytes(p->experts,p->dim),n=(size_t)p->experts*p->dim;
 const BnCudaBuffer*g=(const BnCudaBuffer*)p->gate,*b=(const BnCudaBuffer*)p->bias;
 if(!bytes||p->device_budget_bytes<bytes||!g||g->native_f16_owner!=ctx||!g->data||g->type!=BN_GGUF_TENSOR_F16||g->rows!=p->experts||g->cols!=p->dim||g->size!=n*2||
 !b||b->native_raw_owner!=ctx||!b->data||b->type!=BN_GGUF_TENSOR_F32||b->rows!=1||b->cols!=p->experts||b->size!=(size_t)p->experts*4||cuda_ctx_set_device(ctx))return NULL;
 BnCudaBuffer*w=(BnCudaBuffer*)calloc(1,sizeof(*w));if(!w)return NULL;
 w->type=BN_GGUF_TENSOR_I8;w->rows=p->experts;w->cols=p->dim;w->size=bytes;w->native_router_owner=ctx;
 cudaError_t error=cudaMalloc(&w->data,bytes);if(error==cudaSuccess)error=cudaMemsetAsync(w->data,0,bytes,0);
 unsigned char*arena=(unsigned char*)w->data;size_t mean_offset=(n*4+(size_t)p->experts*6+3)&~(size_t)3;int*invalid=arena?(int*)(arena+mean_offset+4):NULL;
 if(error==cudaSuccess){transpose<<<dim3((p->experts+31)/32,p->dim/32),dim3(32,8)>>>((half*)arena,(const half*)g->data,p->experts,p->dim,invalid);error=cudaGetLastError();}
 if(error==cudaSuccess&&bn_cuda_router_prepare_launch(0,arena,arena+n*2,(float*)(arena+n*4),p->experts,p->dim))error=cudaErrorInvalidValue;
 if(error==cudaSuccess){int T=1;while(T*2<=(p->experts>=128?p->experts/4:p->experts))T*=2;center<<<1,T>>>((const float*)b->data,(half*)(arena+n*4+(size_t)p->experts*4),(float*)(arena+mean_offset),invalid,p->experts,1.0f/(float)p->experts);error=cudaGetLastError();}
 cudaError_t drained=cudaStreamSynchronize(0);if(error==cudaSuccess)error=drained;int rejected=0;
 if(error==cudaSuccess)error=cudaMemcpy(&rejected,invalid,4,cudaMemcpyDeviceToHost);
 if(error!=cudaSuccess||rejected){cudaFree(w->data);free(w);if(error==cudaErrorMemoryAllocation)(void)cudaGetLastError();return NULL;}return w;
}
static int run(void*vctx,void*request,const BnGPURouterResidentPlan*p){
 BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;BnGPURouterWorkspace ws;
 if(!ctx||!r||!p||r->backend_owner!=ctx||ctx->compute_capability<800||bn_gpu_router_workspace(p->n_tokens,p->dim,p->experts,&ws)||p->k<1||p->k>32||p->k>p->experts||!isfinite(p->scaling_factor)||p->scaling_factor<=0)return -1;
 const BnCudaBuffer*w=(const BnCudaBuffer*)p->weight;size_t bytes=bn_gpu_router_weight_bytes(p->experts,p->dim),n=(size_t)p->experts*p->dim;
 if(!w||w->native_router_owner!=ctx||!w->data||w->size!=bytes||w->rows!=p->experts||w->cols!=p->dim)return -1;
 BnGPUResidentView v[]={p->x,p->logits,p->selected,p->weights,p->ahi,p->alo,p->ascale,p->partials};
 size_t counts[]={(size_t)p->n_tokens*p->dim,(size_t)p->n_tokens*p->experts,(size_t)p->n_tokens*p->k,(size_t)p->n_tokens*p->k,ws.a_count,ws.a_count,ws.scale_count,ws.part_count};
 int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};void*data[8]={0};
 for(int i=0;i<8;i++){
  if(!counts[i]){if(!bn_cuda_hc4::empty(v[i]))return -1;continue;}
  if(v[i].type!=types[i]||v[i].count!=counts[i])return -1;
  data[i]=bn_cuda_gdn::view_data(r,v[i]);size_t align=(i==0||i==4||i==5)?16:i==2||i>=6?4:2;
  if(!data[i]||((uintptr_t)data[i]&(align-1)))return -1;
  for(int j=0;j<i;j++)if(counts[j]&&bn_cuda_gdn::overlap(v[i],v[j]))return -1;
 }
 if(bn_cuda_gdn::set_device(r))return -1;const unsigned char*a=(const unsigned char*)w->data;
 int status=bn_cuda_router_run_launch(r->stream,data[0],a,a+n*2,(const float*)(a+n*4),a+n*4+(size_t)p->experts*4,data[1],(int32_t*)data[2],data[3],data[4],data[5],(float*)data[6],(float*)data[7],p->n_tokens,p->dim,p->experts,p->k,p->scaling_factor);
 if(status)r->failed=1;return status;
}
}
int bn_gpu_cuda_read_router_weight(const BnGPUBackend*g,void*weight,void*out,size_t cap){
 BnCudaCtx*ctx=g?(BnCudaCtx*)g->ctx:NULL;const BnCudaBuffer*w=(const BnCudaBuffer*)weight;
 if(!g||g->kind!=BN_GPU_BACKEND_CUDA||!ctx||!w||w->native_router_owner!=ctx||!out||cap<w->size||cuda_ctx_set_device(ctx))return -1;
 return cudaMemcpy(out,w->data,w->size,cudaMemcpyDeviceToHost)==cudaSuccess?0:-1;
}
#endif
