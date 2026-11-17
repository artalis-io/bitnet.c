#include "cuda_native_router_reference.h"
static int native_router_half_rows(void*c,int row,int count,uint16_t*out,size_t cap){return native_norm_read_weight(c,row,count,out,cap*2);}
static void native_router_guard_check(BnGPUBackend*g,void*r,BnGPUResidentView storage,size_t n,int type,uint64_t hash){size_t width=type==BN_GGUF_TENSOR_I8?1:type==BN_GGUF_TENSOR_F16?2:4;unsigned char*b=malloc((n+32)*width);assert(b&&!g->resident_read(r,storage,b));uint64_t got=native_norm_hash(b+16*width,n*width);if(got!=hash)fprintf(stderr,"Router hash got%016llx expected%016llx n%zu type%d\n",(unsigned long long)got,(unsigned long long)hash,n,type);assert(got==hash);for(size_t i=0;i<16*width;i++)assert(b[i]==0xa5&&b[(n+16)*width+i]==0xa5);free(b);}
static BnGPUResidentView native_router_guard(BnGPUBackend*g,void*r,int t,size_t n,const void*data,BnGPUResidentView*storage){size_t width=t==BN_GGUF_TENSOR_I8?1:t==BN_GGUF_TENSOR_F16?2:4;unsigned char*b=malloc((n+32)*width);assert(b);memset(b,0xa5,(n+32)*width);if(data)memcpy(b+16*width,data,n*width);*storage=cuda_test_resident_alloc(g,r,t,n+32,b);free(b);return bn_gpu_resident_view_slice(*storage,16,n);}
static void run_native_router_reference(BnGPUBackend*g){assert(g->buffer_create_router_weight&&g->router_sqrtsp_resident);size_t checked=0;
 for(size_t c=0;c<sizeof(native_router_fixtures)/sizeof(native_router_fixtures[0]);c++){
  const NativeRouterFixture*f=&native_router_fixtures[c];int E=f->E,D=f->D,R=f->R,K=f->K;size_t n=(size_t)E*D,bytes=bn_gpu_router_weight_bytes(E,D);uint16_t*gate=malloc(n*2),*x=malloc((size_t)R*D*2);float*bias=malloc((size_t)E*4);assert(gate&&x&&bias);
  for(size_t i=0;i<n;i++)gate[i]=bn_fp32_to_fp16(f->zero?0:(float)((int)((i*17+13)%127)-63)/1024);
  for(int i=0;i<E;i++)bias[i]=f->zero?0:34+(float)((i*7)%31-15)/1024;
  for(size_t i=0;i<(size_t)R*D;i++)x[i]=bn_fp32_to_fp16(i/(size_t)D==1?0:(float)((int)((i*73+19)%1021)-510)/1024);
  assert(native_norm_hash(x,(size_t)R*D*2)==f->x_hash);
  NativeNormWeightSource gs={(const unsigned char*)gate,D,2},bs={(const unsigned char*)bias,E,4};
  BnGPUF16RowsPlan gp={E,D,n*2+(size_t)(E<32?E:32)*D*2,(size_t)(E<32?E:32)*D*2,&gs,native_router_half_rows};
  /* F16 and raw readers have identical byte callback contracts. */
  void*gw=g->buffer_create_f16_rows(g->ctx,&gp);assert(gw);BnGPURawRowsPlan bp={BN_GGUF_TENSOR_F32,1,E,(size_t)E*4,(size_t)E*4,&bs,native_norm_read_weight};void*bw=g->buffer_create_raw_rows(g->ctx,&bp);assert(bw);
  BnGPURouterWeightPlan wp={E,D,gw,bw,bytes};BnGPURouterWeightPlan badw=wp;badw.device_budget_bytes=bytes-1;assert(!g->buffer_create_router_weight(g->ctx,&badw));badw=wp;badw.gate=bw;assert(!g->buffer_create_router_weight(g->ctx,&badw));
  void*w=g->buffer_create_router_weight(g->ctx,&wp);assert(w);unsigned char*arena=malloc(bytes);assert(arena&&!bn_gpu_cuda_read_router_weight(g,w,arena,bytes));uint64_t ah=native_norm_hash(arena,bytes);if(ah!=f->arena_hash)fprintf(stderr,"Router arena fixture%zu E%d D%d got%016llx expected%016llx\n",c,E,D,(unsigned long long)ah,(unsigned long long)f->arena_hash);assert(ah==f->arena_hash);free(arena);
  BnGPURouterWorkspace ws;assert(!bn_gpu_router_workspace(R,D,E,&ws));void*r[2];BnGPURouterResidentPlan p[2];BnGPUResidentView storage[2][8];size_t counts[]={(size_t)R*D,(size_t)R*E,(size_t)R*K,(size_t)R*K,ws.a_count,ws.a_count,ws.scale_count,ws.part_count};int types[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};
  for(int q=0;q<2;q++){
   r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[8]={{0}};for(int j=0;j<8;j++)if(counts[j])v[j]=native_router_guard(g,r[q],types[j],counts[j],j==0?x:NULL,&storage[q][j]);
   p[q]=(BnGPURouterResidentPlan){R,D,E,K,1.5f,w,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]};BnGPURouterResidentPlan bad;
#define ROUTE_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->router_sqrtsp_resident(g->ctx,r[q],&bad)==-1);}while(0)
   ROUTE_BAD(n_tokens,0);ROUTE_BAD(dim,D+1);ROUTE_BAD(experts,E+1);ROUTE_BAD(k,0);ROUTE_BAD(k,33);ROUTE_BAD(scaling_factor,NAN);ROUTE_BAD(weight,gw);ROUTE_BAD(x.count,(size_t)R*D-1);ROUTE_BAD(selected.type,BN_GGUF_TENSOR_F32);ROUTE_BAD(weights,bn_gpu_resident_view_slice(p[q].logits,0,(size_t)R*K));ROUTE_BAD(x.offset,p[q].x.offset+1);ROUTE_BAD(x.slot,p[q].x.slot^UINT32_C(0x10000));
   if(R==1){ROUTE_BAD(ahi,p[q].x);}else{ROUTE_BAD(ahi.count,ws.a_count-1);ROUTE_BAD(alo,p[q].ahi);}
#undef ROUTE_BAD
   int dummy;assert(g->router_sqrtsp_resident(&dummy,r[q],&p[q])==-1);
  }
  BnGPURouterResidentPlan bad=p[0];bad.x=p[1].x;assert(g->router_sqrtsp_resident(g->ctx,r[0],&bad)==-1);
  for(int q=0;q<2;q++)assert(!g->router_sqrtsp_resident(g->ctx,r[q],&p[q]));
  for(int q=0;q<2;q++){for(int j=1;j<8;j++)if(counts[j]){native_router_guard_check(g,r[q],storage[q][j],counts[j],types[j],f->hashes[j-1]);checked+=counts[j];}native_router_guard_check(g,r[q],storage[q][0],counts[0],types[0],f->x_hash);g->resident_request_destroy(r[q]);}
  g->buffer_destroy(g->ctx,w);g->buffer_destroy(g->ctx,gw);g->buffer_destroy(g->ctx,bw);free(gate);free(x);free(bias);
 }
 /* Finite source bias can overflow its centered F16 derivative. Reject it
  * rather than publish infinities that let the source top-k emit index -1. */
 uint16_t zeros[256]={0};NativeNormWeightSource gs={(const unsigned char*)zeros,128,2};BnGPUF16RowsPlan gp={2,128,1024,512,&gs,native_router_half_rows};void*gate=g->buffer_create_f16_rows(g->ctx,&gp);assert(gate);
 float cases[4][2]={{70000,-70000},{3.402823466e38f,3.402823466e38f},{NAN,0},{0,0}};
 for(int i=0;i<4;i++){
  NativeNormWeightSource bs={(const unsigned char*)cases[i],2,4};BnGPURawRowsPlan bp={BN_GGUF_TENSOR_F32,1,2,8,8,&bs,native_norm_read_weight};void*bias=g->buffer_create_raw_rows(g->ctx,&bp);assert(bias);BnGPURouterWeightPlan wp={2,128,gate,bias,bn_gpu_router_weight_bytes(2,128)};void*w=g->buffer_create_router_weight(g->ctx,&wp);assert((w!=NULL)==(i==3));if(w)g->buffer_destroy(g->ctx,w);g->buffer_destroy(g->ctx,bias);
 }
 g->buffer_destroy(g->ctx,gate);
 printf("CUDA deterministic native router source-reference/derived layouts/guards/isolation PASSED: %zu elements\n",checked);
}
