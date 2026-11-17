/* Actual dense FFN weights; original inputs only, native intermediates. */
#include "gpu_cuda.h"
#include "model_arch.h"
#include "backend_layout.h"
#include "backend_session.h"
#include "../src/transformer/bf16_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *load(const char*dir,const char*name,size_t bytes) {
    char path[1024];snprintf(path,sizeof(path),"%s/%s.bin",dir,name);FILE*f=fopen(path,"rb");assert(f);void*p=malloc(bytes);assert(p);
    assert(fread(p,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return p;
}
static BnGPUResidentView allocv(BnGPUBackend*g,void*r,int type,size_t count,const void*data) {
    BnGPUResidentView v={0};assert(!g->resident_alloc(r,type,count,&v));if(data)assert(!g->resident_write(r,v,data));return v;
}
static BnGPUResidentView slice(BnGPUResidentView v,size_t off,size_t n){return bn_gpu_resident_view_slice(v,off,n);}
static size_t check(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,int c,int phase,const char*name) {
    char file[128];snprintf(file,sizeof(file),"%d-%d-%s",c,phase,name);size_t word=v.type==BN_GGUF_TENSOR_F32?4:2;
    void*expected=load(dir,file,v.count*word),*actual=malloc(v.count*word);assert(actual);assert(!g->resident_read(r,v,actual));
    size_t bad=0;for(size_t i=0;i<v.count;++i)if(memcmp((char*)actual+i*word,(char*)expected+i*word,word)) {
        if(bad<3)fprintf(stderr,"case%d phase%d %s word%zu mismatch\n",c,phase,name,i);
        bad++;
    }
    if(bad){fprintf(stderr,"%zu / %zu mismatches\n",bad,v.count);exit(3);}free(actual);free(expected);return v.count;
}
int main(int argc,char**argv) {
    if(argc!=3){fprintf(stderr,"usage: %s first-shard.gguf dense-reference-directory\n",argv[0]);return 2;}
    BnGGUFFile*f=bn_gguf_open_file(argv[1]);assert(f);BnModelAnatomy*a=malloc(sizeof(*a));assert(a);assert(!bn_model_arch_plan_anatomy(f,a));BnGPUBackend*g=bn_gpu_cuda_create();assert(g);
    int steps[9][9]={{9},{2,7},{3,3,3},{7,1,1},{6,3},{1,1,1,1,1,1,1,1,1},{32},{33},{65}},lengths[]={1,2,3,3,2,9,1,1,1};size_t words=0;
    for(int layer=0;layer<3;++layer) {
        BnBackendBF16DenseFFNWeights w={0};assert(!bn_backend_layout_bf16_dense_ffn_init(&w,g,f,a,layer,160432128,106954752));
        assert(w.dim==4096&&w.hidden_dim==12288&&w.clamp_limit==10);
        char dir[1024];snprintf(dir,sizeof(dir),"%s/layer%d",argv[2],layer);
        void*input=load(dir,"input",65u*w.dim*2);
        BnBackendSession*session=bn_backend_session_create();assert(session);void*r=bn_gpu_backend_resident_request(g,session);assert(r);
        assert(!g->linear_a8_bf16_reserve(g->ctx,r,65u*w.hidden_dim));
        BnGPUResidentView x=allocv(g,r,BN_GGUF_TENSOR_BF16,65u*w.dim,input),gu=allocv(g,r,BN_GGUF_TENSOR_BF16,65u*2*w.hidden_dim,NULL),act=allocv(g,r,BN_GGUF_TENSOR_BF16,65u*w.hidden_dim,NULL),out=allocv(g,r,BN_GGUF_TENSOR_BF16,65u*w.dim,NULL);
        for(int c=0;c<9;++c){int off=0;for(int phase=0;phase<lengths[c];++phase) {
            int t=steps[c][phase];BnGPUResidentView xv=slice(x,(size_t)off*w.dim,(size_t)t*w.dim),gv=slice(gu,0,(size_t)t*2*w.hidden_dim),av=slice(act,0,(size_t)t*w.hidden_dim),ov=slice(out,0,(size_t)t*w.dim);
            assert(!bn_transformer_bf16_dense_ffn(&w,r,t,xv,gv,av,ov));assert(!g->resident_synchronize(r));
            words+=check(g,r,gv,dir,c,phase,"gate_up");words+=check(g,r,av,dir,c,phase,"activated");words+=check(g,r,ov,dir,c,phase,"out");off+=t;
        }printf("layer%d case%d dense FFN PASSED\n",layer,c);fflush(stdout);}
        free(input);bn_backend_session_free(session);bn_backend_layout_bf16_dense_ffn_free(&w);
    }
    printf("PASSED %zu exact dense FFN words\n",words);bn_gpu_cuda_destroy(g);bn_gguf_free(f);free(a);return 0;
}
