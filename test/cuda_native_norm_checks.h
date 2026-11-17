#include "cuda_native_norm_reference.h"
typedef struct {const unsigned char *data;int cols;size_t width;} NativeNormWeightSource;
static int native_norm_read_weight(void *context,int row,int count,void *out,size_t cap){NativeNormWeightSource*s=context;size_t n=(size_t)count*s->cols*s->width;assert(cap>=n);memcpy(out,s->data+(size_t)row*s->cols*s->width,n);return 0;}
static uint64_t native_norm_hash(const void *data,size_t bytes){const unsigned char*p=data;uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<bytes;i++)h=(h^p[i])*UINT64_C(1099511628211);return h;}
static void run_native_norm_reference(BnGPUBackend*g){assert(g->rmsnorm_vector4_f16_f32_resident);size_t words=0;
for(size_t c=0;c<sizeof(native_norm_fixtures)/sizeof(native_norm_fixtures[0]);c++){
 const NativeNormFixture*f=&native_norm_fixtures[c];int rows=7*f->groups;size_t n=(size_t)rows*f->dim,xwidth=f->xt==16?2:4,ywidth=f->yt==16?2:4;void*x=malloc(n*xwidth),*out=malloc((n+16)*ywidth),*input=malloc((n+16)*xwidth);uint16_t*gamma=malloc((size_t)f->groups*f->dim*2);assert(x&&out&&input&&gamma);
 for(size_t i=0;i<n;i++){float v=(float)((int)((i*73+19)%1021)-510)/1024.0f;size_t row=i/(size_t)f->dim;if(row==0)v=0;if(row==1)v*=1e-12f;if(row==2)v*=1e5f;if(f->xt==16)((uint16_t*)x)[i]=bn_fp32_to_fp16(v);else((float*)x)[i]=v;}
 if(f->xt==16){((uint16_t*)x)[2*f->dim]=0x7c00;((uint16_t*)x)[2*f->dim+1]=0xfc00;((uint16_t*)x)[2*f->dim+2]=f->nan_bits;}
 assert(native_norm_hash(x,n*xwidth)==f->x_hash);
 for(size_t i=0;i<(size_t)f->groups*f->dim;i++){float v=(float)((int)((i*7+3)%33)-16)/16.0f;if(f->wt==16)gamma[i]=bn_fp32_to_fp16(v);else{uint32_t bits;memcpy(&bits,&v,4);gamma[i]=(uint16_t)(bits>>16);}}
 NativeNormWeightSource source={(const unsigned char*)gamma,f->dim,2};BnGPURawRowsPlan upload={f->wt==16?BN_GGUF_TENSOR_F16:BN_GGUF_TENSOR_BF16,f->groups,f->dim,(size_t)f->groups*f->dim*2,(size_t)f->groups*f->dim*2,&source,native_norm_read_weight};void*weight=f->weighted?g->buffer_create_raw_rows(g->ctx,&upload):NULL;assert(!f->weighted||weight);
 void*r[2];BnGPUVector4RMSNormResidentPlan plan[2];BnGPUResidentView xs[2],ys[2];
 for(int q=0;q<2;q++){memset(input,0xa5,(n+16)*xwidth);memcpy((char*)input+8*xwidth,x,n*xwidth);memset(out,0xa5,(n+16)*ywidth);r[q]=g->resident_request_create(g->ctx);assert(r[q]);xs[q]=cuda_test_resident_alloc(g,r[q],f->xt==16?BN_GGUF_TENSOR_F16:BN_GGUF_TENSOR_F32,n+16,input);ys[q]=cuda_test_resident_alloc(g,r[q],f->yt==16?BN_GGUF_TENSOR_F16:BN_GGUF_TENSOR_F32,n+16,out);
 plan[q]=(BnGPUVector4RMSNormResidentPlan){rows,f->dim,f->groups,f->eps,f->bias,f->scale,weight,bn_gpu_resident_view_slice(xs[q],8,n),bn_gpu_resident_view_slice(ys[q],8,n)};BnGPUVector4RMSNormResidentPlan bad;
#define NORM_BAD(F,V) do{bad=plan[q];bad.F=(V);assert(g->rmsnorm_vector4_f16_f32_resident(g->ctx,r[q],&bad)==-1);}while(0)
 NORM_BAD(dim,f->dim+1);NORM_BAD(n_tokens,0);NORM_BAD(weight_groups,0);NORM_BAD(eps,0);NORM_BAD(eps,NAN);NORM_BAD(constant_scale,INFINITY);NORM_BAD(x.count,n-1);NORM_BAD(out.count,n+1);NORM_BAD(out,bn_gpu_resident_view_slice(xs[q],12,n));NORM_BAD(x.slot,plan[q].x.slot^UINT32_C(0x10000));
 if(f->weighted){NORM_BAD(weight_groups,f->groups+1);}else{NORM_BAD(weight_groups,2);}
#undef NORM_BAD
 int dummy;assert(g->rmsnorm_vector4_f16_f32_resident(&dummy,r[q],&plan[q])==-1);assert(!g->resident_read(r[q],ys[q],out));for(size_t i=0;i<(n+16)*ywidth;i++)assert(((unsigned char*)out)[i]==0xa5);
 }
 BnGPUVector4RMSNormResidentPlan bad=plan[0];bad.x=plan[1].x;assert(g->rmsnorm_vector4_f16_f32_resident(g->ctx,r[0],&bad)==-1);
 for(int q=0;q<2;q++)assert(!g->rmsnorm_vector4_f16_f32_resident(g->ctx,r[q],&plan[q]));
 for(int q=0;q<2;q++){assert(!g->resident_read(r[q],ys[q],out));uint64_t hash=native_norm_hash((char*)out+8*ywidth,n*ywidth);if(hash!=f->out_hash)fprintf(stderr,"Native norm mismatch case%zu x%d y%d w%d got%016llx expected%016llx\n",c,f->xt,f->yt,f->wt,(unsigned long long)hash,(unsigned long long)f->out_hash);assert(hash==f->out_hash);for(size_t i=0;i<8*ywidth;i++)assert(((unsigned char*)out)[i]==0xa5&&((unsigned char*)out)[(n+8)*ywidth+i]==0xa5);words+=n;
  if(f->xt==f->yt){plan[q].out=plan[q].x;assert(!g->rmsnorm_vector4_f16_f32_resident(g->ctx,r[q],&plan[q])&&!g->resident_read(r[q],xs[q],input));assert(native_norm_hash((char*)input+8*xwidth,n*xwidth)==f->out_hash);for(size_t i=0;i<8*xwidth;i++)assert(((unsigned char*)input)[i]==0xa5&&((unsigned char*)input)[(n+8)*xwidth+i]==0xa5);words+=n;}
  g->resident_request_destroy(r[q]);}
 if(weight)g->buffer_destroy(g->ctx,weight);
 free(gamma);free(input);free(out);free(x);
}
printf("CUDA shared vector4 FTZ RMSNorm original-kernel/guards/alias/isolation PASSED: %zu words\n",words);}
