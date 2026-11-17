#include "cuda_native_hc4_reference.h"
typedef struct {const float *p;int cols;} NativeHC4Source;
static int native_hc4_rows(void *context,int row,int count,void*out,size_t cap){NativeHC4Source*s=context;size_t n=(size_t)count*s->cols*4;assert(n<=cap);memcpy(out,s->p+(size_t)row*s->cols,n);return 0;}
static void *native_hc4_weight(BnGPUBackend*g,const float*p,int rows,int cols){NativeHC4Source s={p,cols};size_t n=(size_t)rows*cols*4;BnGPURawRowsPlan plan={BN_GGUF_TENSOR_F32,rows,cols,n,n,&s,native_hc4_rows};void*w=g->buffer_create_raw_rows(g->ctx,&plan);assert(w);return w;}
static void native_hc4_check(BnGPUBackend*g,void*r,BnGPUResidentView storage,size_t n,uint64_t expected){unsigned char*b=malloc((n+16)*4);assert(b&&!g->resident_read(r,storage,b));uint64_t got=native_norm_hash(b+32,n*4);if(got!=expected)fprintf(stderr,"HC4 hash got%016llx expected%016llx n%zu\n",(unsigned long long)got,(unsigned long long)expected,n);assert(got==expected);for(size_t i=0;i<32;i++)assert(b[i]==0xa5&&b[(n+8)*4+i]==0xa5);free(b);}
static BnGPUResidentView native_hc4_guard(BnGPUBackend*g,void*r,int type,size_t n,const void*data){size_t width=type==BN_GGUF_TENSOR_F16?2:4;unsigned char*b=malloc((n+16)*width);assert(b);memset(b,0xa5,(n+16)*width);if(data)memcpy(b+8*width,data,n*width);BnGPUResidentView v=cuda_test_resident_alloc(g,r,type,n+16,b);free(b);return v;}
static void run_native_hc4_reference(BnGPUBackend*g){assert(g->hc4_carry_resident&&g->hc4_apply_resident);size_t words=0;
assert(!bn_gpu_hc4_partials_count(0,5120)&&!bn_gpu_hc4_partials_count(65536,5120)&&!bn_gpu_hc4_partials_count(1,127)&&!bn_gpu_hc4_partials_count(1,65540));
for(size_t c=0;c<sizeof(native_hc4_fixtures)/sizeof(native_hc4_fixtures[0]);c++){
 const NativeHC4Fixture*f=&native_hc4_fixtures[c];int D=f->dim,R=f->rows;size_t xn=(size_t)R*4*D,yn=(size_t)R*D,pn=bn_gpu_hc4_partials_count(R,D);assert(pn==(size_t)R*f->chunks*25);
 float*x=malloc(xn*4),*y=malloc(yn*4),*fn=malloc((size_t)24*4*D*4),*pi=malloc((size_t)R*4*4);uint16_t*yh=malloc(yn*2);assert(x&&y&&fn&&pi&&yh);
 for(size_t i=0;i<xn;i++){float v=(float)((int)((i*73+19)%1021)-510)/1024;size_t row=i/((size_t)4*D);if(row==0)v*=262144;if(row==1)v*=0x1p-120f;if(row==2)v=i/((size_t)D)%4==1?-0.0f:0.0f;x[i]=v;}
 assert(native_norm_hash(x,xn*4)==f->input_hash);
 for(size_t i=0;i<yn;i++){y[i]=(float)((int)((i*31+17)%1021)-510)/1024;yh[i]=bn_fp32_to_fp16(y[i]);}
 for(size_t i=0;i<(size_t)24*4*D;i++)fn[i]=(float)((int)((i*17+13)%127)-63)/65536+0x1p-29f;
 for(size_t i=0;i<(size_t)R*4;i++)pi[i]=(float)((int)(i%7)-3)/8;
 float base[24],scale[3]={.5f,.75f,1.25f};for(int i=0;i<24;i++)base[i]=(float)((i*11+5)%31-15)/16;
 void*fw=native_hc4_weight(g,fn,24,4*D),*bw=native_hc4_weight(g,base,1,24),*sw=native_hc4_weight(g,scale,1,3);
 void*r[2];BnGPUHC4CarryResidentPlan plans[2];BnGPUResidentView storage[2][7],yv[2][2];size_t sizes[]={xn,pn,(size_t)R*4,(size_t)R*16,yn,(size_t)R*4,(size_t)R*4};
 for(int q=0;q<2;q++){
  r[q]=g->resident_request_create(g->ctx);assert(r[q]);
  for(int j=0;j<7;j++)storage[q][j]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F32,sizes[j],j==0?x:j==6?pi:NULL);
  yv[q][0]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F16,yn,yh);yv[q][1]=native_hc4_guard(g,r[q],BN_GGUF_TENSOR_F32,yn,y);
  BnGPUResidentView v[7];for(int j=0;j<7;j++)v[j]=bn_gpu_resident_view_slice(storage[q][j],8,sizes[j]);
  plans[q]=(BnGPUHC4CarryResidentPlan){R,D,20,1e-20f,1e-6f,fw,bw,sw,v[0],v[1],v[2],v[3],v[4],f->carry?v[6]:(BnGPUResidentView){0},v[5]};
  BnGPUHC4CarryResidentPlan bad;
#define HC_BAD(F,V) do{bad=plans[q];bad.F=(V);assert(g->hc4_carry_resident(g->ctx,r[q],&bad)==-1);}while(0)
  HC_BAD(dim,D+1);HC_BAD(n_tokens,0);HC_BAD(n_tokens,65536);HC_BAD(sinkhorn_iters,0);HC_BAD(sinkhorn_iters,257);HC_BAD(rms_eps,NAN);HC_BAD(hc_eps,0);HC_BAD(fn,bw);HC_BAD(base,sw);HC_BAD(scale,NULL);HC_BAD(partials.count,pn-1);HC_BAD(post.count,(size_t)R*4+1);HC_BAD(collapsed.type,BN_GGUF_TENSOR_F16);HC_BAD(pre_out,plans[q].post);HC_BAD(streams.slot,plans[q].streams.slot^UINT32_C(0x10000));HC_BAD(pre_in,plans[q].pre_out);HC_BAD(pre_in,(BnGPUResidentView){.count=1});HC_BAD(streams.offset,plans[q].streams.offset+1);
#undef HC_BAD
  int dummy;assert(g->hc4_carry_resident(&dummy,r[q],&plans[q])==-1);
  /* Rejections must leave every output untouched. */
  for(int j=1;j<=5;j++){unsigned char*b=malloc((sizes[j]+16)*4);assert(b&&!g->resident_read(r[q],storage[q][j],b));for(size_t i=0;i<(sizes[j]+16)*4;i++)assert(b[i]==0xa5);free(b);}
 }
 BnGPUHC4CarryResidentPlan bad=plans[0];bad.streams=plans[1].streams;assert(g->hc4_carry_resident(g->ctx,r[0],&bad)==-1);
 for(int q=0;q<2;q++)assert(!g->hc4_carry_resident(g->ctx,r[q],&plans[q]));
 for(int q=0;q<2;q++){
  for(int j=1;j<=5;j++){native_hc4_check(g,r[q],storage[q][j],sizes[j],f->hashes[j-1]);words+=sizes[j];}
  native_hc4_check(g,r[q],storage[q][0],xn,f->input_hash);
  for(int yt=0;yt<2;yt++){
   assert(!g->resident_write(r[q],plans[q].streams,x));BnGPUHC4ApplyResidentPlan a={R,D,plans[q].streams,bn_gpu_resident_view_slice(yv[q][yt],8,yn),plans[q].post,plans[q].comb},b;
#define HA_BAD(F,V) do{b=a;b.F=(V);assert(g->hc4_apply_resident(g->ctx,r[q],&b)==-1);}while(0)
   HA_BAD(dim,D+1);HA_BAD(y.count,yn-1);HA_BAD(post.type,BN_GGUF_TENSOR_F16);HA_BAD(comb,a.streams);HA_BAD(y,plans[1-q].collapsed);HA_BAD(streams.offset,a.streams.offset+1);
#undef HA_BAD
   assert(!g->hc4_apply_resident(g->ctx,r[q],&a));native_hc4_check(g,r[q],storage[q][0],xn,f->hashes[5+yt]);words+=xn;
  }
  assert(!g->resident_write(r[q],plans[q].streams,x));
  /* Restore the fixture's incoming carry, then queue the whole three-site chain. */
  BnGPUResidentView incoming=bn_gpu_resident_view_slice(storage[q][6],8,(size_t)R*4);
  assert(!g->resident_write(r[q],incoming,pi));
  BnGPUHC4CarryResidentPlan chain=plans[q];
  BnGPUHC4ApplyResidentPlan apply={R,D,plans[q].streams,bn_gpu_resident_view_slice(yv[q][1],8,yn),plans[q].post,plans[q].comb};
  for(int step=0;step<3;step++){
   assert(!g->hc4_carry_resident(g->ctx,r[q],&chain)&&!g->hc4_apply_resident(g->ctx,r[q],&apply));
   chain.pre_in=chain.pre_out;
   chain.pre_out=bn_gpu_resident_view_slice(storage[q][step%2?5:6],8,(size_t)R*4);
  }
  for(int j=1;j<=5;j++){native_hc4_check(g,r[q],storage[q][j],sizes[j],f->hashes[6+j]);words+=sizes[j];}
  native_hc4_check(g,r[q],storage[q][0],xn,f->hashes[12]);words+=xn;
  g->resident_request_destroy(r[q]);
 }
 g->buffer_destroy(g->ctx,fw);g->buffer_destroy(g->ctx,bw);g->buffer_destroy(g->ctx,sw);free(x);free(y);free(yh);free(fn);free(pi);
}
printf("CUDA HC4 original carry/apply/guards/isolation PASSED: %zu words\n",words);}
