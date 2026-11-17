/* Independent original-kernel fixtures from exllamav3 commit
 * bd1f519172406a9560bd5c90e61000691bb69a07. Arithmetic and library provenance:
 * docs/deepseek-v41-1017-native-folded-ffn.json. */
#include "cuda_folded_exl3_reference.h"
static void *folded_test_weight(BnGPUBackend*g,int K,int N,int seed){
 size_t packed=(size_t)K*N*3/8,bytes=packed+(size_t)(K+N)*2;
 uint16_t*b=malloc(bytes);assert(b);
 for(size_t i=0;i<packed/2;i++)b[i]=(uint16_t)(i*29789+(size_t)seed*104729+((i*13)^(size_t)seed));
 for(int i=0;i<K;i++)b[packed/2+i]=bn_fp32_to_fp16((float)((i*7+seed)%17-8)/8);
 for(int i=0;i<N;i++)b[packed/2+K+i]=bn_fp32_to_fp16((float)((i*11+seed)%19-9)/8);
 void*w=g->trellis_buffer_create(g->ctx,b,bytes,BN_QUANT_FORMAT_EXL3_MUL1_3,K,N,UINT32_C(0x83dcd12d));free(b);assert(w);return w;
}
static uint16_t*folded_test_input(int B,int M,int K){size_t n=(size_t)B*M*K;uint16_t*x=malloc(n*2);assert(x);for(size_t i=0;i<n;i++){float a=(float)((int)((i*73+19)%1021)-510)/1024;size_t row=i/K%M;if(row==1)a=0;if(row==2)a*=0x1p-14f;x[i]=bn_fp32_to_fp16(a);}return x;}
static BnGPUFoldedEXL3ResidentPlan folded_test_plan(int B,int M,int K,int N){BnGPUFoldedEXL3ResidentPlan p={0};p.format=BN_QUANT_FORMAT_EXL3_MUL1_3;p.k=K;p.n=N;p.n_tokens=M;p.batch=B;p.codebook_marker=UINT32_C(0x83dcd12d);return p;}
static void folded_test_kind(BnGPUBackend*g,void*r,const BnGPUFoldedEXL3ResidentPlan*p,int kind){BnGPUFoldedGemmInfo info;assert(!g->trellis_folded_gemm_info(g->ctx,r,p->batch,p->n_tokens,p->k,p->n,&info));assert(info.fp16_partial32==(kind!=0)&&info.tile_n==(kind==1?64:kind?128:0)&&info.tuned==(kind==1||kind==2));}
static uint64_t folded_test_unwritten(size_t bytes){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<bytes;i++)h=(h^0xa5u)*UINT64_C(1099511628211);return h;}
static void run_folded_exl3_reference(BnGPUBackend*g){
 assert(g->trellis_folded_resident&&g->trellis_folded_ffn_resident&&g->trellis_folded_reserve&&g->trellis_folded_probe);int enabled=-1;float ratio=0;assert(!g->trellis_folded_probe(g->ctx,&enabled,&ratio));assert(isfinite(ratio)&&ratio>0&&enabled==(ratio>=1.5f));
 for(size_t c=0;c<sizeof(folded_projection_fixtures)/sizeof(folded_projection_fixtures[0]);c++){
 const FoldedProjectionFixture*f=&folded_projection_fixtures[c];BnGPUFoldedEXL3Workspace ws;assert(!bn_gpu_folded_exl3_workspace(f->B,f->M,f->K,f->N,&ws));uint16_t*x=folded_test_input(f->B,f->M,f->K);assert(native_norm_hash(x,ws.x_count*2)==f->x_hash);
 BnGPUFoldedEXL3ResidentPlan base=folded_test_plan(f->B,f->M,f->K,f->N);for(int b=0;b<f->B;b++)base.weights[b]=folded_test_weight(g,f->K,f->N,f->seed+14*b);
 void*r[2];BnGPUFoldedEXL3ResidentPlan p[2];BnGPUResidentView s[2][3];
 for(int q=0;q<2;q++){
 r[q]=g->resident_request_create(g->ctx);assert(r[q]);p[q]=base;p[q].x=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,ws.x_count,x,&s[q][0]);p[q].decoded=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,ws.decoded_count,NULL,&s[q][1]);p[q].out=native_router_guard(g,r[q],f->f32?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_F16,ws.out_count,NULL,&s[q][2]);
 assert(g->trellis_folded_resident(g->ctx,r[q],&p[q])==-1);assert(!g->trellis_folded_reserve(g->ctx,r[q],f->en));assert(!g->trellis_folded_reserve(g->ctx,r[q],f->en));assert(g->trellis_folded_reserve(g->ctx,r[q],!f->en)==-1);folded_test_kind(g,r[q],&p[q],f->kind);
 BnGPUFoldedEXL3ResidentPlan bad;
#define FOLD_BAD(F,V) do{bad=p[q];bad.F=(V);assert(g->trellis_folded_resident(g->ctx,r[q],&bad)==-1);}while(0)
 FOLD_BAD(weights[f->B-1],NULL);FOLD_BAD(codebook_marker,0);FOLD_BAD(batch,17);FOLD_BAD(n_tokens,0);FOLD_BAD(k,f->K+1);FOLD_BAD(format,BN_GGUF_TENSOR_Q4_K);FOLD_BAD(x.count,ws.x_count-1);FOLD_BAD(x.offset,p[q].x.offset+1);FOLD_BAD(x.slot,p[q].x.slot^UINT32_C(0x10000));FOLD_BAD(decoded,p[q].x);FOLD_BAD(out,p[q].decoded);
#undef FOLD_BAD
 native_router_guard_check(g,r[q],s[q][1],ws.decoded_count,BN_GGUF_TENSOR_F16,folded_test_unwritten(ws.decoded_count*2));
 native_router_guard_check(g,r[q],s[q][2],ws.out_count,p[q].out.type,folded_test_unwritten(ws.out_count*(f->f32?4:2)));
 }
 BnGPUFoldedEXL3ResidentPlan bad=p[0];bad.x=p[1].x;assert(g->trellis_folded_resident(g->ctx,r[0],&bad)==-1);
 for(int q=0;q<2;q++)for(int repeat=0;repeat<2;repeat++){assert(!g->trellis_folded_resident(g->ctx,r[q],&p[q]));native_router_guard_check(g,r[q],s[q][0],ws.x_count,BN_GGUF_TENSOR_F16,f->x_hash);native_router_guard_check(g,r[q],s[q][1],ws.decoded_count,BN_GGUF_TENSOR_F16,f->decoded_hash);native_router_guard_check(g,r[q],s[q][2],ws.out_count,p[q].out.type,f->out_hash);}
 for(int q=0;q<2;q++){g->resident_request_destroy(r[q]);}
 for(int b=0;b<f->B;b++){g->buffer_destroy(g->ctx,(void*)base.weights[b]);}
 free(x);
 }
 for(size_t c=0;c<sizeof(folded_ffn_fixtures)/sizeof(folded_ffn_fixtures[0]);c++){
 const FoldedFFNFixture*f=&folded_ffn_fixtures[c];size_t nx=(size_t)f->B*f->M*f->K,nh=(size_t)f->B*f->M*f->H,nd=(size_t)f->B*f->K*f->H;uint16_t*x=folded_test_input(f->B,f->M,f->K);assert(native_norm_hash(x,nx*2)==f->x_hash);
 BnGPUFoldedEXL3FFNResidentPlan base={0};base.gate=folded_test_plan(f->B,f->M,f->K,f->H);base.up=base.gate;base.down=folded_test_plan(f->B,f->M,f->H,f->K);base.activation_profile=BN_GPU_TRELLIS_FFN_SILU_HALF;base.act_limit=f->limit;
 for(int b=0;b<f->B;b++){base.gate.weights[b]=folded_test_weight(g,f->K,f->H,3+14*b);base.up.weights[b]=folded_test_weight(g,f->K,f->H,47+14*b);base.down.weights[b]=folded_test_weight(g,f->H,f->K,101+14*b);}
 void*r[2];BnGPUFoldedEXL3FFNResidentPlan p[2];BnGPUResidentView s[2][6];
 for(int q=0;q<2;q++){
 r[q]=g->resident_request_create(g->ctx);assert(r[q]&&!g->trellis_folded_reserve(g->ctx,r[q],f->en));p[q]=base;p[q].gate.x=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nx,x,&s[q][0]);p[q].up.x=p[q].gate.x;p[q].gate.out=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nh,NULL,&s[q][1]);p[q].up.out=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nh,NULL,&s[q][2]);p[q].down.x=q?p[q].up.out:native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nh,NULL,&s[q][3]);p[q].down.out=native_router_guard(g,r[q],BN_GGUF_TENSOR_F32,nx,NULL,&s[q][4]);p[q].gate.decoded=native_router_guard(g,r[q],BN_GGUF_TENSOR_F16,nd,NULL,&s[q][5]);p[q].up.decoded=p[q].down.decoded=p[q].gate.decoded;
 folded_test_kind(g,r[q],&p[q].gate,f->kinds[0]);folded_test_kind(g,r[q],&p[q].up,f->kinds[1]);folded_test_kind(g,r[q],&p[q].down,f->kinds[2]);
 BnGPUFoldedEXL3FFNResidentPlan bad=p[q];bad.down.weights[f->B-1]=NULL;assert(g->trellis_folded_ffn_resident(g->ctx,r[q],&bad)==-1);bad=p[q];bad.act_limit=NAN;assert(g->trellis_folded_ffn_resident(g->ctx,r[q],&bad)==-1);bad=p[q];bad.up.out=p[q].gate.out;assert(g->trellis_folded_ffn_resident(g->ctx,r[q],&bad)==-1);
 native_router_guard_check(g,r[q],s[q][1],nh,BN_GGUF_TENSOR_F16,folded_test_unwritten(nh*2));native_router_guard_check(g,r[q],s[q][2],nh,BN_GGUF_TENSOR_F16,folded_test_unwritten(nh*2));native_router_guard_check(g,r[q],s[q][4],nx,BN_GGUF_TENSOR_F32,folded_test_unwritten(nx*4));native_router_guard_check(g,r[q],s[q][5],nd,BN_GGUF_TENSOR_F16,folded_test_unwritten(nd*2));
 }
 for(int q=0;q<2;q++){
 assert(!g->trellis_folded_ffn_resident(g->ctx,r[q],&p[q]));native_router_guard_check(g,r[q],s[q][0],nx,BN_GGUF_TENSOR_F16,f->x_hash);native_router_guard_check(g,r[q],s[q][1],nh,BN_GGUF_TENSOR_F16,f->hashes[0]);native_router_guard_check(g,r[q],s[q][2],nh,BN_GGUF_TENSOR_F16,f->hashes[q?2:1]);if(!q)native_router_guard_check(g,r[q],s[q][3],nh,BN_GGUF_TENSOR_F16,f->hashes[2]);native_router_guard_check(g,r[q],s[q][4],nx,BN_GGUF_TENSOR_F32,f->hashes[3]);g->resident_request_destroy(r[q]);}
 for(int b=0;b<f->B;b++){g->buffer_destroy(g->ctx,(void*)base.gate.weights[b]);g->buffer_destroy(g->ctx,(void*)base.up.weights[b]);g->buffer_destroy(g->ctx,(void*)base.down.weights[b]);}free(x);
 }
 printf("CUDA folded EXL3 original-reference projection/FFN/partial32/guards/isolation PASSED: 40 + 12 fixtures\n");
}
