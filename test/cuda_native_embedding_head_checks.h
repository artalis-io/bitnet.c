#include "cuda_native_embedding_head_reference.h"
static void run_native_embedding_head_reference(BnGPUBackend*g){
 assert(g->embedding_hc4_f32_resident&&g->embedding_bf16_reserve&&g->hc4_head_workspace_size&&g->hc4_head_reserve&&g->hc4_head_resident);assert(g->hc4_head_workspace_size(g->ctx)==16u*1024u*1024u);size_t words=0;
 for(size_t c=0;c<sizeof(native_embedding_fixtures)/sizeof(native_embedding_fixtures[0]);c++){
  const NativeEmbeddingFixture*f=&native_embedding_fixtures[c];int R=f->rows,D=f->dim,V=f->vocab;size_t wn=(size_t)V*D,n=(size_t)R*4*D;
  uint16_t*w=malloc(wn*2);int32_t*ids=malloc((size_t)R*4);assert(w&&ids);
  for(size_t i=0;i<wn;i++)w[i]=(uint16_t)(i*257+17);
  assert(native_norm_hash(w,wn*2)==f->weight_hash);
  for(int i=0;i<R;i++){ids[i]=(i*17+R*13)%V;}
  ids[0]=0;if(R>1)ids[1]=V-1;
  NativeNormWeightSource source={(const unsigned char*)w,D,2};BnGPURawRowsPlan upload={BN_GGUF_TENSOR_BF16,V,D,wn*2,wn*2,&source,native_norm_read_weight};void*weight=g->buffer_create_raw_rows(g->ctx,&upload);assert(weight);
  void*r[2];BnGPUBF16EmbeddingHC4ResidentPlan p[2];BnGPUResidentView xs[2],ys[2];
  for(int q=0;q<2;q++){
   r[q]=g->resident_request_create(g->ctx);assert(r[q]);xs[q]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_I32,(size_t)R,ids);ys[q]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F32,n,NULL);
   p[q]=(BnGPUBF16EmbeddingHC4ResidentPlan){R,D,V,weight,bn_gpu_resident_view_slice(xs[q],8,(size_t)R),bn_gpu_resident_view_slice(ys[q],8,n)};
   assert(g->embedding_hc4_f32_resident(g->ctx,r[q],&p[q])==-1);
   assert(!g->embedding_bf16_reserve(g->ctx,r[q])&&!g->embedding_bf16_reserve(g->ctx,r[q]));
   BnGPUBF16EmbeddingHC4ResidentPlan bad;
#define EMB_BAD(F,VV) do{bad=p[q];bad.F=(VV);assert(g->embedding_hc4_f32_resident(g->ctx,r[q],&bad)==-1);}while(0)
   EMB_BAD(n_tokens,0);EMB_BAD(n_tokens,65536);EMB_BAD(dim,D+1);EMB_BAD(vocab_size,V+1);EMB_BAD(weight,NULL);EMB_BAD(indices.count,(size_t)R+1);EMB_BAD(indices.type,BN_GGUF_TENSOR_F32);EMB_BAD(out.type,BN_GGUF_TENSOR_BF16);EMB_BAD(out.count,n-1);EMB_BAD(out.offset,p[q].out.offset+1);EMB_BAD(out,p[q].indices);EMB_BAD(indices.slot,p[q].indices.slot^UINT32_C(0x10000));
#undef EMB_BAD
   int dummy;assert(g->embedding_hc4_f32_resident(&dummy,r[q],&p[q])==-1);
   unsigned char*b=malloc((n+16)*4);assert(b&&!g->resident_read(r[q],ys[q],b));for(size_t i=0;i<(n+16)*4;i++)assert(b[i]==0xa5);free(b);
  }
  BnGPUBF16EmbeddingHC4ResidentPlan bad=p[0];bad.indices=p[1].indices;assert(g->embedding_hc4_f32_resident(g->ctx,r[0],&bad)==-1);
  for(int q=0;q<2;q++)assert(!g->embedding_hc4_f32_resident(g->ctx,r[q],&p[q]));
  for(int q=0;q<2;q++){native_hc4_check(g,r[q],ys[q],n,f->out_hash);native_hc4_check(g,r[q],xs[q],(size_t)R,native_norm_hash(ids,(size_t)R*4));words+=n;g->resident_request_destroy(r[q]);}
  g->buffer_destroy(g->ctx,weight);free(w);free(ids);
 }
 for(size_t c=0;c<sizeof(native_head_fixtures)/sizeof(native_head_fixtures[0]);c++){
  const NativeHeadFixture*f=&native_head_fixtures[c];int R=f->rows,D=f->dim;size_t n=(size_t)R*4*D,yn=(size_t)R*D;float*x=malloc(n*4),*pre=malloc((size_t)R*4*4);assert(x&&pre);
  for(size_t i=0;i<n;i++){size_t row=i/((size_t)4*D),h=i/D%4;float v=(float)((int)((i*73+19)%1021)-510)*.1f;if(row==0)v*=262144;if(row==1)v*=0x1p-120f;if(row==2)v=h==1?-0.f:0.f;if(row==3){if(h==0)v=0x1p30f;if(h==1)v=-0x1p30f;if(h==2)v*=0x1p-10f;}x[i]=v;}
  for(size_t i=0;i<(size_t)R*4;i++)pre[i]=(float)((int)((i*7+3)%31)-15)/16;
  assert(native_norm_hash(x,n*4)==f->x_hash&&native_norm_hash(pre,(size_t)R*4*4)==f->pre_hash);
  void*r[2];BnGPUHC4HeadResidentPlan p[2];BnGPUResidentView xs[2],ps[2],ys[2];
  for(int q=0;q<2;q++){
   r[q]=g->resident_request_create(g->ctx);assert(r[q]);xs[q]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F32,n,x);ps[q]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F32,(size_t)R*4,pre);ys[q]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F32,yn,NULL);
   p[q]=(BnGPUHC4HeadResidentPlan){R,D,bn_gpu_resident_view_slice(xs[q],8,n),bn_gpu_resident_view_slice(ps[q],8,(size_t)R*4),bn_gpu_resident_view_slice(ys[q],8,yn)};
   assert(g->hc4_head_resident(g->ctx,r[q],&p[q])==-1);
   if(q)assert(!g->trellis_projection_f16_reserve(g->ctx,r[q]));
   assert(!g->hc4_head_reserve(g->ctx,r[q])&&!g->hc4_head_reserve(g->ctx,r[q]));
   BnGPUHC4HeadResidentPlan bad;
#define HEAD_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->hc4_head_resident(g->ctx,r[q],&bad)==-1);}while(0)
   HEAD_BAD(n_tokens,0);HEAD_BAD(dim,D+1);HEAD_BAD(pre.count,(size_t)R*4+1);HEAD_BAD(streams.type,BN_GGUF_TENSOR_F16);HEAD_BAD(out.count,yn-1);HEAD_BAD(out,p[q].streams);HEAD_BAD(pre,p[q].out);HEAD_BAD(out.offset,p[q].out.offset+1);HEAD_BAD(streams.slot,p[q].streams.slot^UINT32_C(0x10000));
#undef HEAD_BAD
   int dummy;assert(g->hc4_head_resident(&dummy,r[q],&p[q])==-1);
   unsigned char*b=malloc((yn+16)*4);assert(b&&!g->resident_read(r[q],ys[q],b));for(size_t i=0;i<(yn+16)*4;i++)assert(b[i]==0xa5);free(b);
  }
  BnGPUHC4HeadResidentPlan bad=p[0];bad.pre=p[1].pre;assert(g->hc4_head_resident(g->ctx,r[0],&bad)==-1);
  for(int q=0;q<2;q++)assert(!g->hc4_head_resident(g->ctx,r[q],&p[q]));
  for(int q=0;q<2;q++){native_hc4_check(g,r[q],ys[q],yn,f->out_hash);native_hc4_check(g,r[q],xs[q],n,f->x_hash);native_hc4_check(g,r[q],ps[q],(size_t)R*4,f->pre_hash);words+=yn;g->resident_request_destroy(r[q]);}free(x);free(pre);
 }
 /* Invalid resident IDs poison only their owner, at the explicit fence. */
 uint16_t table[128]={0};NativeNormWeightSource src={(const unsigned char*)table,128,2};BnGPURawRowsPlan upload={BN_GGUF_TENSOR_BF16,1,128,sizeof(table),sizeof(table),&src,native_norm_read_weight};void*weight=g->buffer_create_raw_rows(g->ctx,&upload);assert(weight);const int32_t invalid[]={-1,1,INT32_MIN,INT32_MAX};
 for(size_t k=0;k<sizeof(invalid)/sizeof(invalid[0]);k++){
  void*bad=g->resident_request_create(g->ctx),*peer=g->resident_request_create(g->ctx);assert(bad&&peer);assert(!g->embedding_bf16_reserve(g->ctx,bad)&&!g->embedding_bf16_reserve(g->ctx,peer));
  BnGPUBF16EmbeddingHC4ResidentPlan p={1,128,1,weight,cuda_test_resident_alloc(g,bad,BN_GGUF_TENSOR_I32,1,&invalid[k]),cuda_test_resident_alloc(g,bad,BN_GGUF_TENSOR_F32,512,NULL)};
  int32_t zero=0;BnGPUBF16EmbeddingHC4ResidentPlan good={1,128,1,weight,cuda_test_resident_alloc(g,peer,BN_GGUF_TENSOR_I32,1,&zero),cuda_test_resident_alloc(g,peer,BN_GGUF_TENSOR_F32,512,NULL)};
  assert(!g->embedding_hc4_f32_resident(g->ctx,bad,&p)&&!g->embedding_hc4_f32_resident(g->ctx,peer,&good));assert(!g->resident_synchronize(peer));float out[512];assert(!g->resident_read(peer,good.out,out));for(int i=0;i<512;i++)assert(out[i]==0);
  assert(g->resident_synchronize(bad)==-1&&g->embedding_hc4_f32_resident(g->ctx,bad,&p)==-1&&g->resident_read(bad,p.out,out)==-1);g->resident_request_destroy(bad);g->resident_request_destroy(peer);
 }
 g->buffer_destroy(g->ctx,weight);printf("CUDA native BF16 embedding/HC4 head source-reference/guards/isolation PASSED: %zu words\n",words);
}
