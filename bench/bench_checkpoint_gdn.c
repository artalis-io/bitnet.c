/* Optional real-checkpoint correctness gate. Controlled Qwen27 layer-0
 * quantized projections -> GDN -> quantized output projection. Optional
 * captures add independent convolution histories, then gated norm/head layout.
 * Supplied inputs exclude outer input norm, residuals and FFN.
 * Reference directory is produced independently; no timing claims. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#ifdef BN_ENABLE_CUDA
static void *read_exact(const char *base,const char *name,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",base,name);
    if(n<0||(size_t)n>=sizeof(path)){fprintf(stderr,"fixture path too long\n");exit(1);}
    FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"cannot open fixture %s\n",path);exit(1);}
    void *data=malloc(bytes);
    if(!data||fread(data,1,bytes,f)!=bytes||fgetc(f)!=EOF){fprintf(stderr,"wrong fixture size or read failure: %s\n",path);fclose(f);free(data);exit(1);}
    fclose(f);return data;
}
static BnGPUResidentView allocate(BnGPUBackend*g,void*r,int type,size_t count,const void*data) {
    BnGPUResidentView v={0};assert(g->resident_alloc(r,type,count,&v)==0);if(data)assert(g->resident_write(r,v,data)==0);return v;
}
static void compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*base,const char*name,size_t width,size_t*checked) {
    void *got=malloc(v.count*width),*want=read_exact(base,name,v.count*width);assert(got);
    assert(g->resident_read(r,v,got)==0);
    if(memcmp(got,want,v.count*width)) {
        size_t diff=0,first=SIZE_MAX;for(size_t i=0;i<v.count;i++)if(memcmp((char*)got+i*width,(char*)want+i*width,width)){if(first==SIZE_MAX)first=i;diff++;}
        fprintf(stderr,"MISMATCH %s values=%zu first=%zu\n",name,diff,first);abort();
    }
    *checked+=v.count;free(got);free(want);
}
#endif
int main(int argc,char**argv) {
#ifndef BN_ENABLE_CUDA
    (void)argc;(void)argv;fprintf(stderr,"requires BN_ENABLE_CUDA=1\n");return 2;
#else
    if(argc!=4&&argc!=5&&argc!=6){fprintf(stderr,"usage: %s REFERENCE LINEAR_WEIGHTS GDN_PREP [CONV_CAPTURE [NORM_CAPTURE]]\n",argv[0]);return 2;}
    BnGPUBackend*g=bn_gpu_cuda_create();assert(g);size_t checked=0;
    int with_conv=argc>=5,with_norm=argc==6;
    const size_t ns=48u*128*128,nh=3u*10240;
    void *wc=NULL,*wn=NULL;
    if(with_norm){void*data=read_exact(argv[5],"real-weight.bin",128*2);wn=g->buffer_create_quant_only(g->ctx,data,128*2,BN_GGUF_TENSOR_BF16,1,128);free(data);assert(wn);}
    if(with_conv){void*data=read_exact(argv[4],"real-weight.bin",10240u*4*2);wc=g->buffer_create_quant_only(g->ctx,data,10240u*4*2,BN_GGUF_TENSOR_BF16,10240,4);free(data);assert(wc);}
    void *qdata=read_exact(argv[2],"in_proj_qkvz-weight.bin",16384u*3520),*bdata=read_exact(argv[2],"in_proj_ba-weight.bin",96u*5440),*odata=read_exact(argv[2],"out_proj-weight.bin",5120u*6528);
    void *adata=read_exact(argv[3],"real-log.bin",48*4),*ddata=read_exact(argv[3],"real-bias.bin",48*2);
    void*wq=g->buffer_create_quant_only(g->ctx,qdata,16384u*3520,BN_GGUF_TENSOR_Q5_K,16384,5120);
    void*wb=g->buffer_create_quant_only(g->ctx,bdata,96u*5440,BN_GGUF_TENSOR_Q8_0,96,5120);
    void*wo=g->buffer_create_quant_only(g->ctx,odata,5120u*6528,BN_GGUF_TENSOR_Q8_0,5120,6144);
    void*al=g->buffer_create_quant_only(g->ctx,adata,48*4,BN_GGUF_TENSOR_F32,1,48),*dt=g->buffer_create_quant_only(g->ctx,ddata,48*2,BN_GGUF_TENSOR_BF16,1,48);assert(wq&&wb&&wo&&al&&dt);
    free(qdata);free(bdata);free(odata);free(adata);free(ddata);
    for(int batch=1;batch<=8;batch*=2)for(int seed=0;seed<2;seed++) {
        char name[128];snprintf(name,sizeof(name),"b%d-s%d-initial.bin",batch,seed);float*initial=read_exact(argv[1],name,(size_t)batch*ns*4);
        BnBackendSession*cs=bn_backend_session_create();assert(cs);void*c=bn_gpu_backend_resident_request(g,cs);assert(c);
        assert(g->linear_bf16_reserve(g->ctx,c,(size_t)batch*6144)==0);
        size_t budget=g->gdn_decode_bf16_group_workspace_size(g->ctx,batch);assert(g->gdn_decode_bf16_group_reserve(g->ctx,c,batch,budget)==0);
        BnGPUResidentView hidden=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*5120,NULL),qkvz=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*16384,NULL),ba=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*96,NULL),core=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*6144,NULL),out=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*5120,NULL);
        BnBackendSession*s[8]={0};void*r[8]={0};BnGPUResidentView states[8][2];BnGPUGDNDecodeSequence rows[8]={{0}};
        for(int i=0;i<batch;i++) {
            s[i]=bn_backend_session_create();r[i]=bn_gpu_backend_resident_request(g,s[i]);assert(r[i]);
            for(int k=0;k<2;k++)states[i][k]=allocate(g,r[i],BN_GGUF_TENSOR_F32,ns,initial+(size_t)i*ns);
            rows[i].request=r[i];rows[i].x=bn_gpu_resident_view_slice(qkvz,(size_t)i*16384,10240);rows[i].ba=bn_gpu_resident_view_slice(ba,(size_t)i*96,96);rows[i].out=bn_gpu_resident_view_slice(core,(size_t)i*6144,6144);
        }
        BnGPUResidentView histories[8][2],post={0};BnGPUCausalConvSequence conv_rows[8]={{0}};
        if(with_conv) {
            size_t bytes=g->causal_conv_bf16_group_workspace_size(g->ctx,batch);assert(g->causal_conv_bf16_group_reserve(g->ctx,c,batch,bytes)==0);
            post=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*10240,NULL);
            snprintf(name,sizeof(name),"b%d-s%d-history-initial.bin",batch,seed);void*history=read_exact(argv[1],name,(size_t)batch*nh*2);
            for(int i=0;i<batch;i++) {
                for(int k=0;k<2;k++)histories[i][k]=allocate(g,r[i],BN_GGUF_TENSOR_BF16,nh,(const char*)history+(size_t)i*nh*2);
                conv_rows[i].request=r[i];conv_rows[i].x=rows[i].x;conv_rows[i].out=bn_gpu_resident_view_slice(post,(size_t)i*10240,10240);rows[i].x=conv_rows[i].out;
            }
            free(history);
        }
        BnGPUResidentView split_qkv={0},z={0},normalized={0},transformed={0};
        if(with_norm) {
            split_qkv=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*10240,NULL);z=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*6144,NULL);
            normalized=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*6144,NULL);transformed=allocate(g,c,BN_GGUF_TENSOR_BF16,(size_t)batch*6144,NULL);
            for(int i=0;i<batch;i++)conv_rows[i].x=bn_gpu_resident_view_slice(split_qkv,(size_t)i*10240,10240);
        }
        free(initial);
        for(int step=0;step<5;step++) {
            snprintf(name,sizeof(name),"b%d-s%d-t%d-input.bin",batch,seed,step);void*input=read_exact(argv[1],name,(size_t)batch*5120*2);assert(g->resident_write(c,hidden,input)==0);free(input);
            BnGPUBF16LinearResidentPlan l={0};l.n_tokens=batch;l.cols=5120;l.rows=16384;l.weight.raw=wq;l.weight.type=BN_GGUF_TENSOR_Q5_K;l.x=hidden;l.out=qkvz;assert(g->linear_bf16_resident(g->ctx,c,&l)==0);
            l.rows=96;l.weight.raw=wb;l.weight.type=BN_GGUF_TENSOR_Q8_0;l.out=ba;assert(g->linear_bf16_resident(g->ctx,c,&l)==0);
            if(with_norm) {
                BnGPUSplitColumnsResidentPlan sp={0};sp.rows=batch;sp.cols=16384;sp.left_cols=10240;sp.x=qkvz;sp.left=split_qkv;sp.right=z;
                assert(g->split_columns_resident(g->ctx,c,&sp)==0);
            }
            for(int i=0;i<batch;i++){rows[i].initial_state=states[i][step%2];rows[i].next_state=states[i][1-step%2];}
            if(with_conv) {
                for(int i=0;i<batch;i++){conv_rows[i].initial_history=histories[i][step%2];conv_rows[i].next_history=histories[i][1-step%2];}
                BnGPUCausalConvGroupPlan cp={0};cp.n_sequences=batch;cp.dim=10240;cp.kernel=4;cp.weight=wc;cp.sequences=conv_rows;
                assert(g->causal_conv_bf16_group_resident(g->ctx,c,&cp)==0);
            }
            BnGPUGDNDecodeGroupPlan p={0};p.hk=16;p.hv=48;p.dk=p.dv=128;p.q_scale=0x1.6a09e6p-4f;p.a_log=al;p.dt_bias=dt;p.n_sequences=batch;p.sequences=rows;
            assert(g->gdn_decode_bf16_group_resident(g->ctx,c,&p)==0);
            if(with_norm) {
                BnGPUGatedRMSNormResidentPlan gn={0};gn.rows=batch*48;gn.dim=128;gn.eps=1e-6f;gn.activation=BN_GPU_GATED_NORM_SILU;gn.weight=wn;gn.x=core;gn.z=z;gn.out=normalized;
                assert(g->gated_rmsnorm_bf16_resident(g->ctx,c,&gn)==0);
                BnGPUTransposeAxesResidentPlan tp={0};tp.n_tokens=batch;tp.outer=16;tp.inner=3;tp.width=128;tp.x=normalized;tp.out=transformed;
                assert(g->transpose_axes_bf16_resident(g->ctx,c,&tp)==0);
            }
            l.cols=6144;l.rows=5120;l.weight.raw=wo;l.weight.type=BN_GGUF_TENSOR_Q8_0;l.x=with_norm?transformed:core;l.out=out;assert(g->linear_bf16_resident(g->ctx,c,&l)==0);
#define CHECK(V,N) do{snprintf(name,sizeof(name),"b%d-s%d-t%d-" N ".bin",batch,seed,step);compare(g,c,V,argv[1],name,2,&checked);}while(0)
            CHECK(qkvz,"qkvz");CHECK(ba,"ba");if(with_conv)CHECK(post,"post");CHECK(core,"core");if(with_norm){CHECK(split_qkv,"qkv");CHECK(z,"z");CHECK(normalized,"norm");CHECK(transformed,"trans");}CHECK(out,"output");
#undef CHECK
            snprintf(name,sizeof(name),"b%d-s%d-t%d-state.bin",batch,seed,step);float*expected=read_exact(argv[1],name,(size_t)batch*ns*4),*got=malloc(ns*4);assert(got);
            for(int i=0;i<batch;i++){assert(g->resident_read(r[i],rows[i].next_state,got)==0);if(memcmp(got,expected+(size_t)i*ns,ns*4)){fprintf(stderr,"STATE mismatch %s request=%d\n",name,i);abort();}checked+=ns;}
            if(with_conv) {
                snprintf(name,sizeof(name),"b%d-s%d-t%d-history.bin",batch,seed,step);void*expected_history=read_exact(argv[1],name,(size_t)batch*nh*2),*got_history=malloc(nh*2);assert(got_history);
                for(int i=0;i<batch;i++){assert(g->resident_read(r[i],conv_rows[i].next_history,got_history)==0);assert(!memcmp(got_history,(const char*)expected_history+(size_t)i*nh*2,nh*2));checked+=nh;}
                free(expected_history);free(got_history);
            }
            free(expected);free(got);printf("b%d s%d step%d exact\n",batch,seed,step);fflush(stdout);
        }
        for(int i=0;i<batch;i++)bn_backend_session_free(s[i]);
        bn_backend_session_free(cs);
    }
    g->buffer_destroy(g->ctx,wq);g->buffer_destroy(g->ctx,wb);g->buffer_destroy(g->ctx,wo);g->buffer_destroy(g->ctx,al);g->buffer_destroy(g->ctx,dt);if(wc)g->buffer_destroy(g->ctx,wc);if(wn)g->buffer_destroy(g->ctx,wn);bn_gpu_cuda_destroy(g);
    printf("checkpoint Q5_K/Q8_0 -> %sGDN -> %sQ8_0 PASSED: %zu values\n",with_conv?"conv -> ":"",with_norm?"gated norm/layout -> ":"",checked);return 0;
#endif
}
