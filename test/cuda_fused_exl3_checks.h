#include "cuda_fused_exl3_reference.h"
static uint64_t fused_exl3_poison_hash(size_t bytes){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<bytes;i++)h=(h^0xa5u)*UINT64_C(1099511628211);return h;}
static void fused_exl3_guard(BnGPUBackend*g,void*r,BnGPUResidentView v,size_t count){unsigned char*b=malloc((count+32)*2);assert(b&&!g->resident_read(r,v,b));for(size_t i=0;i<32;i++)assert(b[i]==0xa5&&b[(count+16)*2+i]==0xa5);free(b);}
#include "cuda_fused_exl3_group_checks.h"
static void run_fused_exl3_reference(BnGPUBackend*g,int small){
 assert(g->trellis_fused_resident&&g->trellis_fused_reserve&&g->trellis_fused_launch_info);assert(g->trellis_fused_workspace_size(g->ctx)==4202760);size_t tested=0;
 for(size_t fi=0;fi<sizeof(fused_exl3_fixtures)/sizeof(fused_exl3_fixtures[0]);fi++){
  const FusedEXL3Fixture*f=&fused_exl3_fixtures[fi];if(small&&(f->width!=8||f->limit!=10||!((f->E<=4&&(f->R==7||f->R==17||f->R==33||f->R==65||f->R==129))||(f->E==40&&f->R==64))))continue;
  int E=f->E,R=f->R,K=f->K,D=f->D,H=f->H,A=R*K;size_t nx=(size_t)R*D,ns=(size_t)A*D;void*weight[3][40]={{0}};assert(E<=40);for(int e=0;e<E;e++){int seed=f->seed_mode?3+14*e:e==3?47:3+14*e;for(int pr=0;pr<3;pr++)weight[pr][e]=folded_test_weight(g,pr==2?H:D,pr==2?D:H,seed+44*pr);}
  int*ids=malloc((size_t)A*4),*counts=calloc((size_t)E+1,4),*base=malloc((size_t)E*4),*kind=malloc((size_t)E*4);uint16_t*w=malloc((size_t)A*2),*x=malloc(nx*2);float*zero=calloc(nx,4);unsigned char*poison=malloc(ns*4);assert(ids&&counts&&base&&kind&&w&&x&&zero&&poison);memset(poison,0xa5,ns*4);
  for(int a=0;a<A;a++){int token=a/K,j=a%K;ids[a]=E==40?(token*17+j*7)%E:j;counts[ids[a]]++;w[a]=bn_fp32_to_fp16((float)((a*11+3)%127+1)/256);}
  int at=0;for(int e=0;e<E;e++){base[e]=at;at+=counts[e];kind[e]=1;}assert(at==A);
  for(size_t i=0;i<nx;i++){float v=(float)((int)((i*73+19)%1021)-510)/1024;size_t row=i/D;if(row==1)v=0;if(row==2)v*=0x1p-14f;x[i]=bn_fp32_to_fp16(v);}assert(native_norm_hash(x,nx*2)==f->x);
  void*r[2];BnGPUNativeMoEIndexPlan index[2];BnGPUFusedEXL3ResidentPlan p[2];BnGPUNativeMoEReducePlan reduce[2];BnGPUResidentView index_st[2][10],state[2][4],st[2][4],xs[2],ss[2],ys[2],bs[2],ks[2];size_t maxstate[2],maxinter[2];
  for(int q=0;q<2;q++){
   r[q]=g->resident_request_create(g->ctx);assert(r[q]);assert(!g->native_moe_index_reserve(g->ctx,r[q],A));BnGPUResidentView v[10];size_t n[]={(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)A,(size_t)E+1,(size_t)E+1};int t[]={BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_I32};for(int j=0;j<10;j++)v[j]=native_router_guard(g,r[q],t[j],n[j],j==0?(void*)ids:j==1?(void*)w:NULL,&index_st[q][j]);index[q]=(BnGPUNativeMoEIndexPlan){R,K,E,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]};assert(!g->native_moe_index_resident(g->ctx,r[q],&index[q]));
   BnGPUFusedEXL3Workspace ws;int win=q?1:32,groups=E<win?E:win;if(groups>188/f->width)groups=188/f->width;assert(!bn_gpu_fused_exl3_workspace(groups,f->capacity,D,H,&ws));maxstate[q]=ws.state_count;maxinter[q]=ws.intermediate_count;for(int j=0;j<4;j++)state[q][j]=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,j<2?ws.state_count:ws.intermediate_count,NULL,&st[q][j]);
   p[q]=(BnGPUFusedEXL3ResidentPlan){0};p[q].format=BN_QUANT_FORMAT_EXL3_MUL1_3;p[q].codebook_marker=UINT32_C(0x83dcd12d);p[q].tokens=R;p[q].topk=K;p[q].experts=E;p[q].dim=D;p[q].hidden=H;p[q].capacity=f->capacity;p[q].group_width=f->width;p[q].tile_m=f->tile_m;p[q].tile_n=f->tile_n;p[q].slot_rows=A;p[q].act_limit=f->limit;p[q].x=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nx,x,&xs[q]);p[q].counts=index[q].counts;p[q].order=index[q].order;p[q].weights=index[q].sorted_weights;p[q].slot_base=native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,E,base,&bs[q]);p[q].slots=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,ns,poison,&ss[q]);
   p[q].groups=groups;p[q].resident_count=E<win?E:win;p[q].state_g=state[q][0];p[q].state_u=state[q][1];p[q].intermediate_g=state[q][2];p[q].intermediate_u=state[q][3];for(int b=0;b<p[q].resident_count;b++){p[q].expert_ids[b]=b;for(int pr=0;pr<3;pr++)p[q].weight[pr][b]=weight[pr][b];}
   assert(g->trellis_fused_resident(g->ctx,r[q],&p[q])==-1);assert(!g->trellis_fused_reserve(g->ctx,r[q])&&!g->trellis_fused_reserve(g->ctx,r[q]));BnGPUFusedEXL3LaunchInfo info;assert(!g->trellis_fused_launch_info(g->ctx,r[q],f->tile_m,f->tile_n,&info)&&info.threads==512&&info.multiprocessors==188&&info.blocks_per_sm==1);
   reduce[q]=(BnGPUNativeMoEReducePlan){R,K,D,E,A,index[q].indices,index[q].inverse,index[q].starts,p[q].slot_base,native_router_guard(g,r[q],BN_GGUF_TENSOR_I32,E,kind,&ks[q]),index[q].sorted_weights,p[q].slots,native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,nx,zero,&ys[q])};
  }
  fused_exl3_group_compare(g,r,p,poison,ns*4);
  if(f->E==4&&f->R==33&&f->width==8&&f->limit==10)fused_exl3_group_wide(g,p);
  for(int q=0;q<2;q++){
   int win=q?1:32,n=E<win?E:win,G=n;if(G>188/f->width)G=188/f->width;BnGPUFusedEXL3Workspace ws;assert(!bn_gpu_fused_exl3_workspace(G,f->capacity,D,H,&ws));p[q].groups=G;p[q].resident_count=n;p[q].state_g=bn_gpu_resident_view_slice(state[q][0],0,ws.state_count);p[q].state_u=bn_gpu_resident_view_slice(state[q][1],0,ws.state_count);p[q].intermediate_g=bn_gpu_resident_view_slice(state[q][2],0,ws.intermediate_count);p[q].intermediate_u=bn_gpu_resident_view_slice(state[q][3],0,ws.intermediate_count);for(int b=0;b<n;b++){p[q].expert_ids[b]=b;for(int pr=0;pr<3;pr++)p[q].weight[pr][b]=weight[pr][b];}
   BnGPUFusedEXL3ResidentPlan bad;
#define FUSED_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->trellis_fused_resident(g->ctx,r[q],&bad)==-1);}while(0)
   FUSED_BAD(weight[2][n-1],NULL);FUSED_BAD(group_width,7);FUSED_BAD(groups,64);FUSED_BAD(capacity,65);FUSED_BAD(resident_count,0);FUSED_BAD(expert_ids[0],E);FUSED_BAD(format,BN_GGUF_TENSOR_Q4_K);FUSED_BAD(codebook_marker,0);FUSED_BAD(act_limit,NAN);FUSED_BAD(state_u,p[q].state_g);FUSED_BAD(order.count,(size_t)A-1);FUSED_BAD(order.slot,p[q].order.slot^UINT32_C(0x10000));if(n>1){FUSED_BAD(expert_ids[1],p[q].expert_ids[0]);}
#undef FUSED_BAD
   bad=p[q];bad.x=p[1-q].x;assert(g->trellis_fused_resident(g->ctx,r[q],&bad)==-1);native_router_guard_check(g,r[q],ss[q],ns,BN_GGUF_TENSOR_F32,fused_exl3_poison_hash(ns*4));
  }
  for(int repeat=0;repeat<2;repeat++){
   for(int q=0;q<2;q++){
    int win=q?1:32;for(int first=0;first<E;first+=win){int n=E-first;if(n>win)n=win;int G=n;if(G>188/f->width)G=188/f->width;BnGPUFusedEXL3Workspace ws;assert(!bn_gpu_fused_exl3_workspace(G,f->capacity,D,H,&ws));p[q].groups=G;p[q].resident_count=n;p[q].state_g=bn_gpu_resident_view_slice(state[q][0],0,ws.state_count);p[q].state_u=bn_gpu_resident_view_slice(state[q][1],0,ws.state_count);p[q].intermediate_g=bn_gpu_resident_view_slice(state[q][2],0,ws.intermediate_count);p[q].intermediate_u=bn_gpu_resident_view_slice(state[q][3],0,ws.intermediate_count);for(int b=0;b<n;b++){p[q].expert_ids[b]=first+b;for(int pr=0;pr<3;pr++)p[q].weight[pr][b]=weight[pr][first+b];}assert(!g->trellis_fused_resident(g->ctx,r[q],&p[q]));}
   }
   for(int q=0;q<2;q++)native_router_guard_check(g,r[q],ss[q],ns,BN_GGUF_TENSOR_F32,f->slots);
  }
  for(int q=0;q<2;q++){assert(!g->native_moe_reduce_resident(g->ctx,r[q],&reduce[q]));native_router_guard_check(g,r[q],ys[q],nx,BN_GGUF_TENSOR_F32,f->final);native_router_guard_check(g,r[q],xs[q],nx,BN_GGUF_TENSOR_F16,f->x);for(int j=0;j<4;j++)fused_exl3_guard(g,r[q],st[q][j],j<2?maxstate[q]:maxinter[q]);g->resident_request_destroy(r[q]);}
  for(int e=0;e<E;e++){for(int pr=0;pr<3;pr++){g->buffer_destroy(g->ctx,weight[pr][e]);}}
  free(ids);free(counts);free(base);free(kind);free(w);free(x);free(zero);free(poison);tested++;
 }
 printf("CUDA fused EXL3 source-reference/index/ordered-reduction/windows/concurrent request state PASSED: %zu fixtures\n",tested);
}
