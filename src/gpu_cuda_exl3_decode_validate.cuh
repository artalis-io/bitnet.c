#ifndef BN_GPU_CUDA_EXL3_DECODE_VALIDATE_CUH
#define BN_GPU_CUDA_EXL3_DECODE_VALIDATE_CUH
#include "gpu_cuda_exl3_decode.h"
namespace bn_cuda_decode {
using bn_cuda_gdn::ResidentRequest;
static int run(void*vctx,void*request,const BnGPUDecodeEXL3ResidentPlan*p) {
    BnCudaCtx*ctx=(BnCudaCtx*)vctx;ResidentRequest*r=(ResidentRequest*)request;
    BnGPUDecodeEXL3Workspace ws;
    if(!ctx||!r||!p||r->backend_owner!=ctx||r->failed||ctx->compute_capability<800||
       bn_gpu_decode_exl3_workspace(p->tokens,p->topk,p->experts,p->dim,p->hidden,&ws)||
       p->resident_count<1||p->resident_count>BN_GPU_DECODE_EXL3_MAX_EXPERTS||p->resident_count>p->experts||
       !isfinite(p->act_limit)||p->act_limit<0||p->act_limit>65504)return -1;
    const BnQuantTrellisOps*ops=bn_quant_format_trellis_ops(p->format);
    if(!ops||ops->decode!=BN_QUANT_TRELLIS_DECODE_EXL3_MUL1_3_F32||p->codebook_marker!=ops->codebook_marker)return -1;
    BnCudaDecodeEXL3Pointers pointers={};pointers.count=p->resident_count;
    for(int b=0;b<p->resident_count;b++) {
        int e=p->expert_ids[b];if(e<0||e>=p->experts)return -1;
        for(int j=0;j<b;j++)if(p->expert_ids[j]==e)return -1;
        pointers.experts[b]=e;
        for(int pr=0;pr<3;pr++) {
            int K=pr==2?p->hidden:p->dim,N=pr==2?p->dim:p->hidden;
            size_t packed=(size_t)K*N/8*3,bytes=packed+((size_t)K+N)*2;
            const BnCudaBuffer*w=(const BnCudaBuffer*)p->weight[pr][b];
            if(!w||w->trellis_owner!=ctx||!w->data||w->type!=p->format||w->rows!=N||w->cols!=K||w->trellis_marker!=p->codebook_marker||w->size!=bytes||((uintptr_t)w->data&15u))return -1;
            pointers.packed[pr][b]=w->data;pointers.pre[pr][b]=(const unsigned char*)w->data+packed;
            pointers.post[pr][b]=(const unsigned char*)pointers.pre[pr][b]+(size_t)K*2;
        }
    }
    size_t A=(size_t)p->tokens*p->topk;
    BnGPUResidentView v[]={p->x,p->indices,p->weights,p->tables,p->indices64,p->had_gate,p->had_up,p->gate,p->up,p->act,p->down,p->counters,p->out};
    size_t count[]={(size_t)p->tokens*p->dim,A,A,ws.table_bytes,ws.indices64_bytes,ws.had_count,ws.had_count,ws.intermediate_count,ws.intermediate_count,ws.intermediate_count,ws.down_count,ws.counter_count,(size_t)p->tokens*p->dim};
    int type[]={BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_I8,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F16,BN_GGUF_TENSOR_F32,BN_GGUF_TENSOR_I32,BN_GGUF_TENSOR_F32};
    void*d[13];for(int i=0;i<13;i++) {
        if(v[i].count!=count[i]||v[i].type!=type[i])return -1;
        d[i]=bn_cuda_gdn::view_data(r,v[i]);if(!d[i]||((uintptr_t)d[i]&15u))return -1;
        for(int j=0;j<i;j++)if(bn_cuda_gdn::overlap(v[i],v[j]))return -1;
    }
    if(bn_cuda_gdn::set_device(r))return -1;
    int info[7];if(bn_cuda_exl3_decode_info(p->tokens,p->dim,p->hidden,p->topk,info))return -1;
    BnCudaDecodeEXL3IO io={d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12]};
    int err=bn_cuda_exl3_decode_launch(r->stream,&pointers,&io,p->tokens,p->dim,p->hidden,p->experts,p->topk,p->act_limit);
    if(err)r->failed=1;return err;
}
}
#endif
