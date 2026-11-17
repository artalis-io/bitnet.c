#include "moe.h"
#include "native_moe_plan_reference.h"
#include "cuda_native_moe_reference.h"
static uint64_t native_moe_unwritten(size_t bytes){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<bytes;i++)h=(h^0xa5u)*UINT64_C(1099511628211);return h;}
static void run_native_moe_reference(BnGPUBackend*g){
 assert(g->native_moe_index_resident&&g->native_moe_index_reserve&&g->native_moe_group_resident&&g->native_moe_reduce_resident);
 for(size_t c=0;c<sizeof(native_moe_gpu_fixtures)/sizeof(native_moe_gpu_fixtures[0]);c++){
  const NativeMoEGPUFixture*h=&native_moe_gpu_fixtures[c];const NativeMoEPlanFixture*f=&native_moe_plan_fixtures[h->case_id];int A=f->R*f->K,E=f->E,D=f->D;size_t nx=(size_t)f->R*D;int*ids=malloc((size_t)A*4);uint16_t*w=malloc((size_t)A*2),*x=malloc(nx*2);float*old=malloc(nx*4);assert(ids&&w&&x&&old);
  for(int a=0;a<A;a++){int t=a/f->K,j=a%f->K;ids[a]=E==1?0:t%3==0?(t/3+j)%f->K:(t*17+j*7)%E;w[a]=bn_fp32_to_fp16((float)((a*11+3)%127+1)/256);}
  for(size_t i=0;i<nx;i++){x[i]=bn_fp32_to_fp16((float)((int)((i*73+19)%1021)-510)/1024);old[i]=i%37==0?0x1p-140f:(float)((int)((i*29+7)%127)-63)/65536;}
  assert(native_norm_hash(ids,(size_t)A*4)==h->ids&&native_norm_hash(w,(size_t)A*2)==h->weights&&native_norm_hash(x,nx*2)==h->x&&native_norm_hash(old,nx*4)==h->old);
  int*order=malloc((size_t)E*4),*starts=malloc((size_t)(E+1)*4),*base=malloc((size_t)E*4),*kind=malloc((size_t)E*4);BnMoEEXL3Group*groups=calloc((size_t)E,sizeof(*groups));assert(order&&starts&&base&&kind&&groups);BnMoEEXL3Plan tier={order,starts,base,kind,groups,(size_t)E,(size_t)E,0,0,0,0};assert(!bn_moe_exl3_plan_groups(&tier,f->counts,E,f->R,f->K,D,f->H,f->fused,256u<<20));
  void*r[2];BnGPUNativeMoEIndexPlan p[2];BnGPUResidentView s[2][10];size_t count[]={(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)E+1,(size_t)E+1};int type[]={BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32};
  for(int q=0;q<2;q++){
   r[q]=g->resident_request_create(g->ctx);assert(r[q]);BnGPUResidentView v[10];for(int j=0;j<10;j++)v[j]=native_router_guard(g,r[q],type[j],count[j],j==0?(const void*)ids:j==1?(const void*)w:NULL,&s[q][j]);p[q]=(BnGPUNativeMoEIndexPlan){f->R,f->K,E,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]};
   assert(g->native_moe_index_resident(g->ctx,r[q],&p[q])==-1);assert(!g->native_moe_index_reserve(g->ctx,r[q],A+q*33));assert(!g->native_moe_index_reserve(g->ctx,r[q],A));assert(g->native_moe_index_reserve(g->ctx,r[q],A+q*33+1)==-1);
   BnGPUNativeMoEIndexPlan bad=p[q];bad.inverse=bad.order;assert(g->native_moe_index_resident(g->ctx,r[q],&bad)==-1);bad=p[q];bad.indices.count--;assert(g->native_moe_index_resident(g->ctx,r[q],&bad)==-1);bad=p[q];bad.order.slot^=UINT32_C(0x10000);assert(g->native_moe_index_resident(g->ctx,r[q],&bad)==-1);
   native_router_guard_check(g,r[q],s[q][5],A,BN_GGUF_TENSOR_I32,native_moe_unwritten((size_t)A*4));
  }
  BnGPUNativeMoEIndexPlan bad=p[0];bad.indices=p[1].indices;assert(g->native_moe_index_resident(g->ctx,r[0],&bad)==-1);
  for(int q=0;q<2;q++)assert(!g->native_moe_index_resident(g->ctx,r[q],&p[q]));
  for(int q=0;q<2;q++){
   const int ji[]={0,1,4,5,6,7,8,9};uint64_t hashes[]={h->ids,h->weights,h->sorted,h->order,h->inv,h->sorted_weights,native_norm_hash(f->counts,(size_t)(E+1)*4),f->starts};for(int j=0;j<8;j++)native_router_guard_check(g,r[q],s[q][ji[j]],count[ji[j]],type[ji[j]],hashes[j]);
   BnGPUResidentView xs;BnGPUResidentView xv=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nx,x,&xs);
   for(int gi=0;gi<tier.group_count;gi++){
    const BnMoEEXL3Group*grp=&groups[gi];size_t n=(size_t)grp->count*grp->rows*D;BnGPUResidentView os;BnGPUNativeMoEGroupPlan gp={0};gp.tokens=f->R;gp.topk=f->K;gp.dim=D;gp.batch=grp->count;gp.rows=grp->rows;gp.x=xv;gp.order=p[q].order;gp.out=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,n,NULL,&os);for(int b=0;b<grp->count;b++){int e=grp->experts[b];gp.starts[b]=starts[e];gp.counts[b]=f->counts[e];}
    BnGPUNativeMoEGroupPlan bg=gp;bg.counts[grp->count-1]=grp->rows+1;assert(g->native_moe_group_resident(g->ctx,r[q],&bg)==-1);native_router_guard_check(g,r[q],os,n,BN_GGUF_TENSOR_F16,native_moe_unwritten(n*2));assert(!g->native_moe_group_resident(g->ctx,r[q],&gp));native_router_guard_check(g,r[q],os,n,BN_GGUF_TENSOR_F16,h->groups[gi]);
   }
   int*asgn=malloc((size_t)A*4);assert(asgn&&!g->resident_read(r[q],p[q].order,asgn));size_t ns=(size_t)tier.slot_count*D;float*slots=calloc(ns,4);assert(slots);
   for(int e=0;e<E;e++)for(int j=0;j<f->counts[e];j++){
    int a=asgn[starts[e]+j];for(int col=0;col<D;col++){float v=ldexpf((float)((a*13+col*17)%257-128)/128,(a%19)-9);if(col%41==0)v=a%3==0?16777216.f:a%3==1?-16777216.f:1.f;if(col%53==0)v=0x1p-135f;if(kind[e]==1)v*=bn_fp16_to_fp32(w[a]);slots[((size_t)base[e]+j)*D+col]=v;}}
   assert(native_norm_hash(slots,ns*4)==h->slots);BnGPUResidentView ss,bs,ks,ys;BnGPUNativeMoEReducePlan rp={f->R,f->K,D,E,tier.slot_count,p[q].indices,p[q].inverse,p[q].starts,native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,E,base,&bs),native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,E,kind,&ks),p[q].sorted_weights,native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,ns,slots,&ss),native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,nx,old,&ys)};
   BnGPUNativeMoEReducePlan br=rp;br.slot_kind=rp.slot_base;assert(g->native_moe_reduce_resident(g->ctx,r[q],&br)==-1);native_router_guard_check(g,r[q],ys,nx,BN_GGUF_TENSOR_F32,h->old);assert(!g->native_moe_reduce_resident(g->ctx,r[q],&rp));native_router_guard_check(g,r[q],ys,nx,BN_GGUF_TENSOR_F32,h->out);native_router_guard_check(g,r[q],ss,ns,BN_GGUF_TENSOR_F32,h->slots);native_router_guard_check(g,r[q],xs,nx,BN_GGUF_TENSOR_F16,h->x);free(asgn);free(slots);g->resident_request_destroy(r[q]);
  }
  free(order);free(starts);free(base);free(kind);free(groups);free(ids);free(w);free(x);free(old);
 }
 puts("CUDA native MoE original-reference assignment sort, group gather, fixed-order slot accumulation and isolation PASSED");
}
