#include "cuda_native_ple_reference.h"
typedef struct {int k,n,dim,query;} NativePLESource;
static int native_ple_read_weight(void *p,int row,int count,uint16_t*out,size_t cap){NativePLESource*s=p;assert(cap>=(size_t)count*s->k);for(int r=0;r<count;r++)for(int c=0;c<s->k;c++){size_t i=(size_t)c*s->n+row+r;out[(size_t)r*s->k+c]=bn_fp32_to_fp16((float)((int)((i*37+13)%257)-128)/4096.f);}return 0;}
static int native_ple_read_norm(void*p,int row,int count,void*output,size_t cap){NativePLESource*s=p;assert(cap>=(size_t)count*s->dim*2);uint16_t*out=output;for(int r=0;r<count;r++)for(int c=0;c<s->dim;c++){int col=s->query?s->dim-1-c:c;size_t i=(size_t)(row+r)*s->dim+col;float v=(float)((int)((i*11+3)%127)-63)/64.f;uint32_t bits;memcpy(&bits,&v,4);out[(size_t)r*s->dim+c]=(uint16_t)(bits>>16);}return 0;}
static uint64_t native_ple_hash(const void*p,size_t n){uint64_t h=UINT64_C(14695981039346656037);const unsigned char*b=p;for(size_t i=0;i<n;i++)h=(h^b[i])*UINT64_C(1099511628211);return h;}
static void run_native_ple_reference(BnGPUBackend*g){assert(g->ple_f16_from_bf16_resident);size_t calls=0,words=0;
 for(size_t fi=0;fi<sizeof(native_ple_fixtures)/sizeof(native_ple_fixtures[0]);fi++){
  const NativePLEFixture*f=native_ple_fixtures+fi;int R=f->rows,H=f->streams,D=f->dim,K=f->k,N=(H+1)*D;size_t wide=(size_t)R*H*D;
  NativePLESource ws={K,N,D,0},ks=ws,qs=ws;qs.query=1;size_t tile=(size_t)(N<32?N:32)*K*2;
  BnGPUF16RowsPlan wp={N,K,(size_t)N*K*2+tile,tile,&ws,native_ple_read_weight};void*w=g->buffer_create_f16_rows(g->ctx,&wp);assert(w);
  BnGPURawRowsPlan kp={BN_GGUF_TENSOR_BF16,H,D,(size_t)H*D*2,(size_t)H*D*2,&ks,native_ple_read_norm},qp=kp;qp.source=&qs;
  void*kw=g->buffer_create_raw_rows(g->ctx,&kp),*qw=g->buffer_create_raw_rows(g->ctx,&qp);assert(kw&&qw);
  size_t counts[]={(size_t)R*K,(size_t)R*K,wide,wide,(size_t)R*D,wide,wide,(size_t)R*H,wide,wide};
  int types[]={BN_GGUF_TENSOR_BF16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_F32};
  unsigned char*host[10];BnGPUResidentView storage[2][10],v[2][10];void*request[2];BnGPUPLEF16FromBF16ResidentPlan plan[2];
  for(int i=0;i<10;i++){size_t width=types[i]==BN_GGUF_TENSOR_F32?4:2;host[i]=malloc((counts[i]+16)*width);assert(host[i]);}
  for(int q=0;q<2;q++){
   request[q]=g->resident_request_create(g->ctx);assert(request[q]);
   for(int i=0;i<10;i++){
    size_t width=types[i]==BN_GGUF_TENSOR_F32?4:2;memset(host[i],0xa5,(counts[i]+16)*width);
    if(i==0)for(size_t j=0;j<counts[i];j++){float x=(float)((int)((j*19+7)%251)-125)/256.f;uint32_t bits;memcpy(&bits,&x,4);((uint16_t*)host[i])[j+8]=(uint16_t)(bits>>16);}
    if(i==2)for(size_t j=0;j<counts[i];j++){float x=(float)((int)((j*73+19)%2047)-1023)/512.f;size_t row=j/((size_t)H*D);if(row==0&&!f->nonzero)x=0;if(row==1)x=ldexpf(x,-70);if(row==2)x=ldexpf(x,20);((float*)host[i])[j+8]=x;}
    storage[q][i]=cuda_test_resident_alloc(g,request[q],types[i],counts[i]+16,host[i]);v[q][i]=bn_gpu_resident_view_slice(storage[q][i],8,counts[i]);
   }
   assert(native_ple_hash(host[2]+32,wide*4)==f->hashes[1]);
   plan[q]=(BnGPUPLEF16FromBF16ResidentPlan){R,K,D,H,1e-20f,(float)(1./sqrt((double)D)),w,kw,qw,v[q][0],v[q][1],v[q][2],v[q][3],v[q][4],v[q][5],v[q][6],v[q][7],v[q][8],v[q][9]};
   assert(g->ple_f16_from_bf16_resident(g->ctx,request[q],&plan[q])==-1);assert(!g->linear_f16_reserve(g->ctx,request[q]));
   BnGPUPLEF16FromBF16ResidentPlan bad;
#define PLE_BAD(F,V) do{bad=plan[q];bad.F=(V);assert(g->ple_f16_from_bf16_resident(g->ctx,request[q],&bad)==-1);}while(0)
   PLE_BAD(n_tokens,0);PLE_BAD(k,0);PLE_BAD(dim,D+1);PLE_BAD(streams,9);PLE_BAD(eps,NAN);PLE_BAD(gate_scale,0);PLE_BAD(norm_query,NULL);PLE_BAD(norm_query,w);PLE_BAD(out,plan[q].stream);PLE_BAD(normalized_query,plan[q].normalized_key);PLE_BAD(dot.count,counts[7]-1);PLE_BAD(lookup.type,BN_GGUF_TENSOR_F16);
#undef PLE_BAD
   for(int i=1;i<10;i++)if(i!=2){size_t width=types[i]==BN_GGUF_TENSOR_F32?4:2;assert(!g->resident_read(request[q],storage[q][i],host[i]));for(size_t j=0;j<(counts[i]+16)*width;j++)assert(host[i][j]==0xa5);}
  }
  BnGPUPLEF16FromBF16ResidentPlan bad=plan[0];bad.value=v[1][4];assert(g->ple_f16_from_bf16_resident(g->ctx,request[0],&bad)==-1);
  for(int q=0;q<2;q++)assert(!g->ple_f16_from_bf16_resident(g->ctx,request[q],&plan[q]));
  for(int q=0;q<2;q++){
   for(int i=0;i<10;i++){size_t width=types[i]==BN_GGUF_TENSOR_F32?4:2;assert(!g->resident_read(request[q],storage[q][i],host[i]));for(size_t j=0;j<8*width;j++)assert(host[i][j]==0xa5&&host[i][(counts[i]+8)*width+j]==0xa5);
    if(i){uint64_t hash=native_ple_hash(host[i]+8*width,counts[i]*width);if(hash!=f->hashes[i-1])fprintf(stderr,"Native PLE fixture%zu boundary%d hash%016llx expected%016llx\n",fi,i,(unsigned long long)hash,(unsigned long long)f->hashes[i-1]);assert(hash==f->hashes[i-1]);words+=counts[i];}
   }g->resident_request_destroy(request[q]);calls++;
  }
  for(int i=0;i<10;i++)free(host[i]);
  g->buffer_destroy(g->ctx,qw);g->buffer_destroy(g->ctx,kw);g->buffer_destroy(g->ctx,w);
 }
 printf("CUDA native BF16->F16 PLE full boundaries/guards/isolation PASSED: %zu calls, %zu words\n",calls,words);
}
