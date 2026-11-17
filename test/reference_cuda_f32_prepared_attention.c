/* Independent pinned GGML df18bee11, CUDA SM120, F32 cache capacity1280. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "cuda_gated_mmf_fixture.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t hash(const float*x,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,x+i,4);h=(h^u)*UINT64_C(1099511628211);}return h;}
static uint64_t thash(struct ggml_tensor*t){size_t n=ggml_nelements(t);float*x=malloc(n*4);assert(x);ggml_backend_tensor_get(t,x,0,n*4);uint64_t h=hash(x,n);free(x);return h;}
int main(void){
 const int cases[][5]={{17,0,24,2,0},{23,0,24,2,23},{33,0,24,2,0},{129,0,24,2,23},{512,0,24,2,0},{1047,0,24,2,23},{4,9,24,2,0},{9,17,24,2,23},{16,33,24,2,0},{23,1024,24,2,23},{128,129,24,2,0},{512,512,24,2,23},{17,0,4,1,0},{23,17,4,1,23},{33,0,12,2,0},{129,129,12,2,23}};
 ggml_backend_t b=ggml_backend_cuda_init(0);assert(b);
 for(size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){
  int nt=cases[ci][0],pos=cases[ci][1],nh=cases[ci][2],nk=cases[ci][3],seed=cases[ci][4],rd=seed?128:64,window=seed?3:0,ctxlen=1280,nkeys=(pos+nt+255)/256*256,stride=2*nh*256+7,kv=nk*256;
  size_t nr=(size_t)nt*stride,nkv=(size_t)nt*kv;float*raw=malloc(nr*4),*key=malloc(nkv*4),*val=malloc(nkv*4),*wq=calloc((size_t)nh*256,4),*wk=calloc((size_t)nk*256,4),*kc=calloc((size_t)ctxlen*kv,4),*vc=calloc((size_t)ctxlen*kv,4),*mask=malloc((size_t)nkeys*nt*4);int32_t*positions=malloc(nt*4);assert(raw&&key&&val&&wq&&wk&&kc&&vc&&mask&&positions);
  gated_mmf_fixture(raw,key,val,wq,wk,nt,nh,nk,seed);
  for(int t=0;t<pos;t++)for(int h=0;h<nk;h++)for(int d=0;d<256;d++){size_t i=((size_t)t*nk+h)*256+d;kc[i]=(float)((int)((i*31+7)%257)-128)/133.f;vc[((size_t)h*256+d)*ctxlen+t]=(float)((int)((i*19+3)%263)-131)/129.f;}
  for(int t=0;t<nt;t++){positions[t]=pos+t;for(int j=0;j<nkeys;j++)mask[(size_t)t*nkeys+j]=j<=pos+t&&(!window||j>pos+t-window)?0.f:-INFINITY;}
  struct ggml_init_params ip={4*1024*1024,NULL,true};struct ggml_context*c=ggml_init(ip);assert(c);
  struct ggml_tensor*tr=ggml_new_tensor_2d(c,GGML_TYPE_F32,stride,nt),*tk=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,nk,nt),*tv=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,nk,nt),*tqw=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,seed?nh:1),*tkw=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,seed?nk:1),*tp=ggml_new_tensor_1d(c,GGML_TYPE_I32,nt),*tkc=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,nk,ctxlen),*tvc=ggml_new_tensor_3d(c,GGML_TYPE_F32,ctxlen,256,nk),*tm=ggml_new_tensor_2d(c,GGML_TYPE_F32,nkeys,nt);
  struct ggml_tensor*q=ggml_view_3d(c,tr,256,nh,nt,512*4,(size_t)stride*4,0),*gate=ggml_view_3d(c,tr,256,nh,nt,512*4,(size_t)stride*4,256*4);
  q=ggml_mul(c,ggml_rms_norm(c,q,1e-6f),tqw);struct ggml_tensor*k=ggml_mul(c,ggml_rms_norm(c,tk,1e-6f),tkw);
  q=ggml_rope_ext(c,q,tp,NULL,rd,2,32768,10000000.f,1.f,0.f,1.f,32.f,1.f);k=ggml_rope_ext(c,k,tp,NULL,rd,2,32768,10000000.f,1.f,0.f,1.f,32.f,1.f);
  struct ggml_tensor*ku=ggml_set(c,tkc,k,tkc->nb[1],tkc->nb[2],tkc->nb[3],(size_t)pos*kv*4);
  struct ggml_tensor*vn=ggml_cont(c,ggml_permute(c,tv,1,2,0,3)); /* [nt,256,nk] */
  struct ggml_tensor*vu=ggml_set(c,tvc,vn,tvc->nb[1],tvc->nb[2],tvc->nb[3],(size_t)pos*4);
  struct ggml_tensor*kh=ggml_permute(c,ggml_view_3d(c,ku,256,nk,nkeys,256*4,(size_t)kv*4,0),0,2,1,3),*qh=ggml_permute(c,q,0,2,1,3),*vv=ggml_view_3d(c,vu,nkeys,256,nk,(size_t)ctxlen*4,(size_t)ctxlen*256*4,0);
  struct ggml_tensor*scores=ggml_mul_mat(c,kh,qh);ggml_prec_set_acc(scores,GGML_PREC_F32);struct ggml_tensor*prob=ggml_soft_max_ext(c,scores,tm,0.0625f,0.f),*pv=ggml_mul_mat(c,vv,prob),*out=ggml_mul(c,ggml_cont(c,ggml_permute(c,pv,0,2,1,3)),ggml_sigmoid(c,gate));
  struct ggml_cgraph*g=ggml_new_graph_custom(c,128,false);ggml_build_forward_expand(g,out);ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,b);assert(buf);
#define SET(t,x,n) ggml_backend_tensor_set(t,x,0,(n)*4)
  SET(tr,raw,nr);SET(tk,key,nkv);SET(tv,val,nkv);SET(tqw,wq,(size_t)256*(seed?nh:1));SET(tkw,wk,(size_t)256*(seed?nk:1));SET(tp,positions,nt);SET(tkc,kc,(size_t)ctxlen*kv);SET(tvc,vc,(size_t)ctxlen*kv);SET(tm,mask,(size_t)nkeys*nt);
  assert(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS);printf("{%d,%d,%d,%d,%d,UINT64_C(0x%016llx),UINT64_C(0x%016llx)},\n",nt,pos,nh,nk,seed,(unsigned long long)thash(out),(unsigned long long)thash(k));
  ggml_backend_buffer_free(buf);ggml_free(c);free(raw);free(key);free(val);free(wq);free(wk);free(kc);free(vc);free(mask);free(positions);
 }
 ggml_backend_free(b);return 0;
}
