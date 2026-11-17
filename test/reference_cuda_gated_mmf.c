/* Independent pinned GGML CUDA df18bee11 gated FP32 prefill oracle.
 * Link GGML only. No runtime kernels or runtime-produced values are used. */
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
static uint64_t hash(const float *x, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) { uint32_t v; memcpy(&v,x+i,4); h=(h^v)*UINT64_C(1099511628211); }
    return h;
}
int main(void) {
    const int counts[]={2,3,4,6,8,9,11,16};
    ggml_backend_t backend=ggml_backend_cuda_init(0);assert(backend);
    for(int c=0;c<8;c++)for(int shape=0;shape<2;shape++)for(int seed=0;seed<=23;seed+=23) {
        int nt=counts[c],nh=shape?24:4,nk=shape?2:1,per=seed!=0,rd=seed?128:64;
        int stride=nh*512+7,dim=nh*256,kv=nk*256;
        size_t nq=(size_t)nt*dim,nkv=(size_t)nt*kv;
        float *raw=malloc((size_t)nt*stride*4),*key=malloc(nkv*4),*value=malloc(nkv*4);
        float *wq=malloc((size_t)nh*256*4),*wk=malloc((size_t)nk*256*4);
        float *q=malloc(nq*4),*gate=malloc(nq*4),*out=malloc(nq*4),*ko=malloc(nkv*4);
        assert(raw&&key&&value&&wq&&wk&&q&&gate&&out&&ko);
        gated_mmf_fixture(raw,key,value,wq,wk,nt,nh,nk,seed);
        for(int t=0;t<nt;t++)for(int h=0;h<nh;h++)for(int d=0;d<256;d++) {
            size_t dst=((size_t)t*nh+h)*256+d,src=(size_t)t*stride+h*512+d;
            q[dst]=raw[src];gate[dst]=raw[src+256];
        }
        int32_t pos[16];for(int t=0;t<nt;t++)pos[t]=t;
        ggml_fp16_t mask[256*32];for(int t=0;t<32;t++)for(int j=0;j<256;j++)
            mask[t*256+j]=ggml_fp32_to_fp16(t<nt&&j<=t&&(!seed||j>t-3)?0.0f:-INFINITY);
        struct ggml_init_params ip={32u*1024u*1024u,NULL,true};struct ggml_context *ctx=ggml_init(ip);assert(ctx);
        struct ggml_tensor *qt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,nh,nt);
        struct ggml_tensor *kt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,nk,nt);
        struct ggml_tensor *vt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,nk,nt);
        struct ggml_tensor *gt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,dim,nt);
        struct ggml_tensor *pt=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,nt);
        struct ggml_tensor *mt=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,256,32);
        struct ggml_tensor *qw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,per?nh:1,1);
        struct ggml_tensor *kw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,per?nk:1,1);
        struct ggml_tensor *qr=qt,*kr=kt;
        if(seed){qr=ggml_mul(ctx,ggml_rms_norm(ctx,qr,1e-6f),qw);kr=ggml_mul(ctx,ggml_rms_norm(ctx,kr,1e-6f),kw);}
        qr=ggml_rope_ext(ctx,qr,pt,NULL,rd,GGML_ROPE_TYPE_NEOX,256,10000000.0f,1.0f,0.0f,1.0f,32.0f,1.0f);
        kr=ggml_rope_ext(ctx,kr,pt,NULL,rd,GGML_ROPE_TYPE_NEOX,256,10000000.0f,1.0f,0.0f,1.0f,32.0f,1.0f);
        struct ggml_tensor *kp=ggml_permute(ctx,ggml_pad(ctx,kr,0,0,256-nt,0),0,2,1,3);
        struct ggml_tensor *vp=ggml_permute(ctx,ggml_pad(ctx,vt,0,0,256-nt,0),0,2,1,3);
        struct ggml_tensor *scores=ggml_mul_mat(ctx,kp,ggml_permute(ctx,qr,0,2,1,3));
        assert(ggml_prec_set_acc(scores,GGML_PREC_F32));
        struct ggml_tensor *prob=ggml_soft_max_ext(ctx,scores,mt,0.0625f,0.0f);
        struct ggml_tensor *pv=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,vp)),prob);
        struct ggml_tensor *out3=ggml_cont(ctx,ggml_permute(ctx,pv,0,2,1,3));
        struct ggml_tensor *result=ggml_mul(ctx,ggml_reshape_2d(ctx,out3,dim,nt),ggml_sigmoid(ctx,gt));
        struct ggml_cgraph *graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,result);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        ggml_backend_tensor_set(qt,q,0,nq*4);ggml_backend_tensor_set(kt,key,0,nkv*4);ggml_backend_tensor_set(vt,value,0,nkv*4);
        ggml_backend_tensor_set(gt,gate,0,nq*4);ggml_backend_tensor_set(pt,pos,0,(size_t)nt*4);ggml_backend_tensor_set(mt,mask,0,sizeof(mask));
        ggml_backend_tensor_set(qw,wq,0,(size_t)(per?nh:1)*256*4);ggml_backend_tensor_set(kw,wk,0,(size_t)(per?nk:1)*256*4);
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(result,out,0,nq*4);ggml_backend_tensor_get(kr,ko,0,nkv*4);
        if (getenv("GGML_REFERENCE_DUMP_DIR")) {
            char path[1024];
            snprintf(path,sizeof(path),"%s/nt%d-nh%d-seed%d.bin",getenv("GGML_REFERENCE_DUMP_DIR"),nt,nh,seed);
            FILE *f=fopen(path,"wb");assert(f&&fwrite(out,4,nq,f)==nq);fclose(f);
            snprintf(path,sizeof(path),"%s/nt%d-nh%d-seed%d-key.bin",getenv("GGML_REFERENCE_DUMP_DIR"),nt,nh,seed);
            f=fopen(path,"wb");assert(f&&fwrite(ko,4,nkv,f)==nkv);fclose(f);
        }
        printf("{%d,%d,%d,%d,UINT64_C(0x%016llx),UINT64_C(0x%016llx)},\n",nt,nh,nk,seed,(unsigned long long)hash(out,nq),(unsigned long long)hash(ko,nkv));
        ggml_backend_buffer_free(buffer);ggml_free(ctx);free(raw);free(key);free(value);free(wq);free(wk);free(q);free(gate);free(out);free(ko);
    }
    ggml_backend_free(backend);return 0;
}
