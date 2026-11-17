/* Actual Flash checkpoint merged projection -> native QK norm/RoPE/gate.
 * Usage: probe CASES Q_NORM K_NORM COS_SIN [--profile | --linear | --linear-profile].
 * Inputs/goldens are immutable independently captured BF16 stage values. */
#include "gpu_cuda.h"
#include "gpu_backend.h"
#include "backend_session.h"
#include "gguf.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
static void *load(const char *path,size_t *bytes) {
    FILE *f=fopen(path,"rb");assert(f);assert(!fseek(f,0,SEEK_END));
    long end=ftell(f);assert(end>0&&!fseek(f,0,SEEK_SET));*bytes=(size_t)end;
    void *p=malloc(*bytes);assert(p&&fread(p,1,*bytes,f)==*bytes&&fgetc(f)==EOF);
    fclose(f);return p;
}
static BnGPUResidentView alloc(BnGPUBackend *g,void *r,size_t count) {
    BnGPUResidentView v;assert(!g->resident_alloc(r,BN_GGUF_TENSOR_BF16,count,&v));return v;
}
static void *weight(BnGPUBackend *g,const char *path,int cols) {
    size_t bytes;void *raw=load(path,&bytes);assert(bytes%(2*(size_t)cols)==0);
    void *w=g->buffer_create_quant_only(g->ctx,raw,bytes,BN_GGUF_TENSOR_BF16,(int)(bytes/2/(size_t)cols),cols);
    free(raw);assert(w);return w;
}
static size_t compare(BnGPUBackend *g,void *r,BnGPUResidentView v,const char *path,int ci,int stage) {
    size_t bytes;void *gold=load(path,&bytes);assert(bytes==2*v.count);
    void *out=malloc(bytes);assert(out&&!g->resident_read(r,v,out));
    size_t differences=0;for(size_t i=0;i<v.count;i++)if(memcmp((char*)out+2*i,(char*)gold+2*i,2)) {
        if(differences<2)fprintf(stderr,"case%d stage%d word%zu differs\n",ci,stage,i);
        differences++;
    }
    assert(!differences);free(out);free(gold);return v.count;
}
int main(int argc,char **argv) {
    assert(argc==5||argc==6);
    int profile=argc==6&&(!strcmp(argv[5],"--profile")||!strcmp(argv[5],"--linear-profile"));
    int linear=argc==6&&(!strcmp(argv[5],"--linear")||!strcmp(argv[5],"--linear-profile"));
    assert(argc==5||profile||linear);
    BnGPUBackend *g=bn_gpu_cuda_create();assert(g&&g->qk_prep_bf16_resident);
    void *qw=weight(g,argv[2],256),*kw=weight(g,argv[3],256),*cs=weight(g,argv[4],64);
    void *projection=NULL;
    if(linear) {
        /* Actual decoded checkpoint weights, independently retained by oracle. */
        char directory[4096];assert(strlen(argv[2])<sizeof(directory));strcpy(directory,argv[2]);
        char *slash=strrchr(directory,'/');assert(slash);*slash='\0';
        const char *names[]={"attn_q.weight.bf16","attn_k.weight.bf16","attn_v.weight.bf16"};
        const size_t rows[]={12288,512,512};size_t offset=0;
        unsigned char *stack=malloc(13312u*2560*2);assert(stack);
        for(int i=0;i<3;i++) {
            char path[8192];int n=snprintf(path,sizeof(path),"%s/%s",directory,names[i]);assert(n>0&&(size_t)n<sizeof(path));
            size_t bytes;void *raw=load(path,&bytes);assert(bytes==rows[i]*2560*2);
            memcpy(stack+offset,raw,bytes);offset+=bytes;free(raw);
        }
        projection=g->buffer_create_quant_only(g->ctx,stack,offset,BN_GGUF_TENSOR_BF16,13312,2560);
        assert(projection);free(stack);
    }
    FILE *f=fopen(argv[1],"r");assert(f);int nt,pos,cases=0;size_t words=0,linear_words=0;
    char projected[4096],paths[4][4096],input_path[4096];
    while(1) {
        int fields=linear?fscanf(f,"%d %d %4095s %4095s %4095s %4095s %4095s %4095s",&nt,&pos,input_path,projected,paths[0],paths[1],paths[2],paths[3]):
                          fscanf(f,"%d %d %4095s %4095s %4095s %4095s %4095s",&nt,&pos,projected,paths[0],paths[1],paths[2],paths[3]);
        if(fields==EOF)break;
        assert(fields==(linear?8:7));
        assert(nt>0&&pos>=0);BnBackendSession *s=bn_backend_session_create();assert(s);
        void *r=bn_gpu_backend_resident_request(g,s);assert(r);
        size_t bytes;void *raw=load(projected,&bytes);assert(bytes==(size_t)nt*13312*2);
        BnGPUResidentView input=alloc(g,r,bytes/2);
        if(!linear)assert(!g->resident_write(r,input,raw));
        free(raw);
        BnGPUBF16LinearResidentPlan lp={0};
        if(linear) {
            assert(!g->linear_bf16_reserve(g->ctx,r,0));
            raw=load(input_path,&bytes);assert(bytes==(size_t)nt*2560*2);
            lp.n_tokens=nt;lp.rows=13312;lp.cols=2560;lp.weight.type=BN_GGUF_TENSOR_BF16;
            lp.weight.decoded=projection;lp.x=alloc(g,r,bytes/2);lp.out=input;
            assert(!g->resident_write(r,lp.x,raw));free(raw);
        }
        BnGPUQKPrepResidentPlan p={0};p.n_tokens=nt;p.n_heads=24;p.n_kv_heads=2;p.head_size=256;
        p.rotary_dim=64;p.position=pos;p.eps=1e-6f;p.weight_offset=1;
        p.q_weight=qw;p.k_weight=kw;p.cos_sin=cs;
        p.q_gate_row_stride=p.k_in_row_stride=p.v_in_row_stride=13312;
        p.q_gate=bn_gpu_resident_view_slice(input,0,(size_t)(nt-1)*13312+12288);
        p.k_in=bn_gpu_resident_view_slice(input,12288,(size_t)(nt-1)*13312+512);
        p.v_in=bn_gpu_resident_view_slice(input,12800,(size_t)(nt-1)*13312+512);
        p.q=alloc(g,r,(size_t)nt*6144);p.k=alloc(g,r,(size_t)nt*512);
        p.v=alloc(g,r,(size_t)nt*512);p.gate=alloc(g,r,(size_t)nt*6144);
        if(profile) {
            if(linear)assert(!g->linear_bf16_resident(g->ctx,r,&lp));
            assert(!g->qk_prep_bf16_resident(g->ctx,r,&p));assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
            assert(cudaProfilerStart()==cudaSuccess);
#else
            assert(!"build with BN_CAPTURE_CUDA_PROFILER for profiling");
#endif
        }
        if(linear)assert(!g->linear_bf16_resident(g->ctx,r,&lp));
        assert(!g->qk_prep_bf16_resident(g->ctx,r,&p));assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
        if(profile)assert(cudaProfilerStop()==cudaSuccess);
#endif
        if(linear) {
            linear_words+=compare(g,r,input,projected,cases,4);
            compare(g,r,lp.x,input_path,cases,5);
        }
        BnGPUResidentView views[]={p.q,p.k,p.v,p.gate};
        for(int i=0;i<4;i++)words+=compare(g,r,views[i],paths[i],cases,i);
        raw=load(projected,&bytes);void *after=malloc(bytes);assert(after&&!g->resident_read(r,input,after));
        assert(!memcmp(raw,after,bytes));free(raw);free(after);bn_backend_session_free(s);
        printf("case%d tokens%d position%d exact immutable\n",cases,nt,pos);fflush(stdout);cases++;
    }
    assert(feof(f)&&cases>0);fclose(f);
    if(projection)g->buffer_destroy(g->ctx,projection);
    g->buffer_destroy(g->ctx,qw);g->buffer_destroy(g->ctx,kw);g->buffer_destroy(g->ctx,cs);bn_gpu_cuda_destroy(g);
    printf("QK_MERGED_CHECKPOINT_PASSED cases%d bf16_words%zu linear_words%zu\n",cases,words,linear_words);return 0;
}
