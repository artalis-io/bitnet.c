#ifndef BN_GPU_CUDA_EXL3_FOLD_VALIDATE_CUH
#define BN_GPU_CUDA_EXL3_FOLD_VALIDATE_CUH
#include "gpu_cuda_exl3_fold.h"
namespace bn_cuda_fold {
static int probe(void*vctx,int*enabled,float*ratio){BnCudaCtx*ctx=(BnCudaCtx*)vctx;if(!ctx||!enabled||!ratio||ctx->compute_capability<800||cuda_ctx_set_device(ctx))return -1;float r=0;if(bn_cuda_fold_probe(ctx->multiprocessors,&r)||!isfinite(r)||r<=0)return -1;*ratio=r;*enabled=r>=1.5f;return 0;}
static int reserve(void*vctx,void*request,int enabled){BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;
 if(!ctx||!r||r->backend_owner!=ctx||ctx->compute_capability<800||(enabled!=0&&enabled!=1)||bn_cuda_gdn::set_device(r))return -1;
 if(r->folded_ready)return r->folded_f16acc==enabled?0:-1;
 if(bn_cuda_exl3::reserve(vctx,request)||bn_cuda_fold_prepare())return -1;r->folded_f16acc=enabled;r->folded_ready=1;return 0;
}
static int info(void*vctx,void*request,int B,int M,int K,int N,BnGPUFoldedGemmInfo*out){BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;BnGPUFoldedEXL3Workspace ws;
 if(!ctx||!r||!out||r->backend_owner!=ctx||!r->folded_ready||r->failed||bn_gpu_folded_exl3_workspace(B,M,K,N,&ws))return -1;
 int kind=bn_cuda_fold_kind(B,M,K,N,ctx->compute_capability,ctx->multiprocessors,r->folded_f16acc);BnGPUFoldedGemmInfo result={kind!=0,kind==1?64:kind?128:0,kind==1||kind==2};*out=result;return 0;
}
static int validate(void*vctx,void*request,const BnGPUFoldedEXL3ResidentPlan*p,void**data,BnCudaFoldPointers*weights){
 BnCudaCtx*ctx=(BnCudaCtx*)vctx;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;BnGPUFoldedEXL3Workspace ws;
 if(!ctx||!r||!p||!weights||r->backend_owner!=ctx||ctx->compute_capability<800||!r->folded_ready||!r->trellis_cublas||!r->trellis_workspace||r->failed||bn_gpu_folded_exl3_workspace(p->batch,p->n_tokens,p->k,p->n,&ws))return -1;
 const BnQuantTrellisOps*ops=bn_quant_format_trellis_ops(p->format);
 if(!ops||ops->folded!=BN_QUANT_TRELLIS_FOLDED_EXL3_MUL1_3_F16||p->codebook_marker!=ops->codebook_marker)return -1;
 size_t packed=(size_t)p->k*p->n/8*3,bytes=packed+((size_t)p->k+p->n)*2;*weights=BnCudaFoldPointers{};
 for(int i=0;i<p->batch;i++){const BnCudaBuffer*w=(const BnCudaBuffer*)p->weights[i];if(!w||w->trellis_owner!=ctx||!w->data||w->type!=p->format||w->trellis_marker!=p->codebook_marker||w->rows!=p->n||w->cols!=p->k||w->size!=bytes||((uintptr_t)w->data&15u))return -1;
 weights->packed[i]=w->data;weights->pre[i]=(const unsigned char*)w->data+packed;weights->post[i]=(const unsigned char*)weights->pre[i]+(size_t)p->k*2;}
 const BnGPUResidentView views[]={p->x,p->decoded,p->out};size_t counts[]={ws.x_count,ws.decoded_count,ws.out_count};
 if(p->out.type!=BN_GGUF_TENSOR_F16&&p->out.type!=BN_GGUF_TENSOR_F32)return -1;
 for(int i=0;i<3;i++){if(views[i].count!=counts[i]||views[i].type!=(i==2?p->out.type:BN_GGUF_TENSOR_F16))return -1;data[i]=bn_cuda_gdn::view_data(r,views[i]);size_t align=i<2?16:p->out.type==BN_GGUF_TENSOR_F32?8:4;if(!data[i]||((uintptr_t)data[i]&(align-1)))return -1;for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(views[i],views[j]))return -1;}
 return 0;
}
static int issue(BnCudaCtx*ctx,bn_cuda_gdn::ResidentRequest*r,const BnGPUFoldedEXL3ResidentPlan*p,void**data,const BnCudaFoldPointers*weights){
 if(bn_cuda_fold_reconstruct(r->stream,weights,data[1],p->batch,p->k,p->n)){r->failed=1;return -1;}
 int kind=bn_cuda_fold_kind(p->batch,p->n_tokens,p->k,p->n,ctx->compute_capability,ctx->multiprocessors,r->folded_f16acc);
 if(kind){if(bn_cuda_fold_gemm(r->stream,kind,data[0],data[1],data[2],p->batch,p->n_tokens,p->k,p->n,p->out.type==BN_GGUF_TENSOR_F32)){r->failed=1;return -1;}}
 else{const float alpha=1,beta=0;if(cublasGemmStridedBatchedEx(r->trellis_cublas,CUBLAS_OP_N,CUBLAS_OP_N,p->n,p->n_tokens,p->k,&alpha,data[1],CUDA_R_16F,p->n,(long long)p->k*p->n,data[0],CUDA_R_16F,p->k,(long long)p->n_tokens*p->k,&beta,data[2],p->out.type==BN_GGUF_TENSOR_F32?CUDA_R_32F:CUDA_R_16F,p->n,(long long)p->n_tokens*p->n,p->batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS){r->failed=1;return -1;}}
 return 0;
}
static int project(void*vctx,void*request,const BnGPUFoldedEXL3ResidentPlan*p){void*data[3];BnCudaFoldPointers w;bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;if(validate(vctx,request,p,data,&w)||bn_cuda_gdn::set_device(r))return -1;return issue((BnCudaCtx*)vctx,r,p,data,&w);}
static int ffn(void*vctx,void*request,const BnGPUFoldedEXL3FFNResidentPlan*p){
 if(!p||p->activation_profile!=BN_GPU_TRELLIS_FFN_SILU_HALF||!isfinite(p->act_limit)||p->act_limit<0||p->act_limit>65504||p->gate.batch!=p->up.batch||p->gate.batch!=p->down.batch||p->gate.n_tokens!=p->up.n_tokens||p->gate.n_tokens!=p->down.n_tokens||p->gate.k!=p->up.k||p->gate.n!=p->up.n||p->gate.n!=p->down.k||p->gate.k!=p->down.n||!bn_cuda_exl3::same_view(p->gate.x,p->up.x)||p->gate.out.type!=BN_GGUF_TENSOR_F16||p->up.out.type!=BN_GGUF_TENSOR_F16||p->down.out.type!=BN_GGUF_TENSOR_F32)return -1;
 void*data[3][3];BnCudaFoldPointers w[3];const BnGPUFoldedEXL3ResidentPlan*plans[]={&p->gate,&p->up,&p->down};
 for(int i=0;i<3;i++)if(validate(vctx,request,plans[i],data[i],&w[i]))return -1;
 BnGPUResidentView persistent[]={p->gate.x,p->gate.out,p->up.out,p->down.x,p->down.out};
 for(int i=0;i<5;i++)for(int j=0;j<i;j++){if(i==3&&j==2&&bn_cuda_exl3::same_view(persistent[i],persistent[j]))continue;if(bn_cuda_gdn::overlap(persistent[i],persistent[j]))return -1;}
 for(int i=0;i<3;i++)for(int j=0;j<5;j++)if(bn_cuda_gdn::overlap(plans[i]->decoded,persistent[j]))return -1;
 bn_cuda_gdn::ResidentRequest*r=(bn_cuda_gdn::ResidentRequest*)request;BnCudaCtx*ctx=(BnCudaCtx*)vctx;if(bn_cuda_gdn::set_device(r))return -1;
 if(issue(ctx,r,&p->up,data[1],&w[1])||issue(ctx,r,&p->gate,data[0],&w[0]))return -1;
 size_t n=p->down.x.count;bn_cuda_exl3::silu_mul<<<(n/2+255)/256,256,0,r->stream>>>((half*)data[2][0],(const half*)data[0][2],(const half*)data[1][2],n,p->act_limit);if(cudaGetLastError()!=cudaSuccess){r->failed=1;return -1;}
 return issue(ctx,r,&p->down,data[2],&w[2]);
}
}
#endif
