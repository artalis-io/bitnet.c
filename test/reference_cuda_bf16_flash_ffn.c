/* Complete actual-weight HC FFN composition. No expected neural intermediate is
 * an input. Usage FIRST_SHARD CAPTURES [LAYER NT [--profile]]. Whole-model modes:
 * --model-reference REF DUMP or --model-reference-pread REF DUMP [CACHE_MB].
 * Complete captures also exercise ordinary backend-only sessions, isolated
 * requests, no-logits/restart and readback-failure reset/reinitialization. */
#include "gpu_cuda.h"
#include "backend_session.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "model_internal.h"
#include "moe.h"
#include "../src/transformer/bf16_internal.h"
#include "../src/moe_internal.h"
#include "gguf.h"
#include "quant.h"
#include "session.h"
#include "session_internal.h"
#include "transformer.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#ifdef BN_CAPTURE_CUDA_PROFILER
#include <cuda_profiler_api.h>
#endif
#include "cuda_bf16_expert_bank_replay.h"

static uint16_t bf16(float x) { uint32_t u;memcpy(&u,&x,4);return(uint16_t)((u+0x7fff+((u>>16)&1))>>16); }
static void *load(const char *dir,const char *prefix,const char *name,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,prefix,name);assert(n>0&&(size_t)n<sizeof(path));
    FILE*f=fopen(path,"rb");void*p=malloc(bytes);assert(f&&p&&fread(p,1,bytes,f)==bytes&&fgetc(f)==EOF);fclose(f);return p;
}
static BnGPUResidentView alloc(BnGPUBackend*g,void*r,int type,size_t n) { BnGPUResidentView v;assert(!g->resident_alloc(r,type,n,&v));return v; }
static size_t compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,const char*prefix,const char*name,int bytes) {
    void*expected=load(dir,prefix,name,v.count*(size_t)bytes),*actual=malloc(v.count*(size_t)bytes);assert(actual&&!g->resident_read(r,v,actual));
    size_t differences=0;for(size_t i=0;i<v.count;i++)if(memcmp((char*)actual+i*bytes,(char*)expected+i*bytes,bytes)) {
        if(differences<3){uint32_t a=0,b=0;memcpy(&a,(char*)actual+i*bytes,bytes);memcpy(&b,(char*)expected+i*bytes,bytes);fprintf(stderr,"%s %s word%zu got%08x expected%08x\n",prefix,name,i,a,b);}differences++;
    }
    assert(!differences);free(expected);free(actual);return v.count;
}
static size_t mapped_compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const int32_t*slots,
    const char*dir,const char*prefix,const char*name,int cols,int expected_cols,int expected_offset) {
    size_t rows=v.count/(size_t)cols;
    uint16_t*expected=load(dir,prefix,name,rows*expected_cols*2),*actual=malloc(v.count*2);assert(actual&&!g->resident_read(r,v,actual));
    size_t differences=0;for(size_t row=0;row<rows;row++)for(int col=0;col<cols;col++) {
        assert(slots[row]>=0&&(size_t)slots[row]<rows);
        if(actual[row*cols+col]!=expected[(size_t)slots[row]*expected_cols+expected_offset+col])differences++;
    }
    if(differences)fprintf(stderr,"%s %s mapped differences%zu\n",prefix,name,differences);
    assert(!differences);free(expected);free(actual);return v.count;
}
enum {HC_NORM,HC_DOWN,HC_UP,HC_INJECT,ROUTER,SHARED_ROUTER,SHARED_GATE,SHARED_UP,SHARED_DOWN,N_WEIGHTS};
typedef struct {void *handle[N_WEIGHTS];int expert_index[3];size_t expert_bytes[3];const unsigned char*expert_data[3];} Weights;
static void weights_load(BnGPUBackend*g,BnGGUFFile*f,int layer,Weights*w,
    const BnBackendBF16FlashFFNModel *model) {
    const char*names[]={"hc_ffn_norm","hc_ffn_down","hc_ffn_up","hc_ffn_inject","ffn_gate_inp","ffn_gate_inp_shexp","ffn_gate_shexp","ffn_up_shexp","ffn_down_shexp"};
    const int rows[]={4,320,10240,4,512,1,640,640,2560},cols[]={2560,10240,320,10240,2560,2560,2560,2560,640};
    memset(w,0,sizeof(*w));
    for(int role=0;role<N_WEIGHTS;role++) {
        if (model && role >= ROUTER) continue;
        char name[128];snprintf(name,sizeof(name),"blk.%d.%s.weight",layer,names[role]);int ti=bn_gguf_find_tensor(f,name);assert(ti>=0);
        const BnGGUFTensorInfo*t=&f->tensors[ti];size_t count=1;for(int d=0;d<(int)t->n_dims;d++)count*=t->dims[d];assert(count==(size_t)rows[role]*cols[role]);
        assert((t->n_dims==1&&t->dims[0]==count)||(t->n_dims==2&&t->dims[0]==(uint64_t)cols[role]&&t->dims[1]==(uint64_t)rows[role]));
        const void*data=bn_gguf_tensor_data(f,ti);assert(data);size_t bytes;
        if(role>=SHARED_GATE) {
            assert(t->type==BN_GGUF_TENSOR_Q8_0&&bn_gguf_tensor_size(t->type,count,&bytes));
            w->handle[role]=g->buffer_create_quant_only(g->ctx,data,bytes,t->type,rows[role],cols[role]);
        } else {
            uint16_t*decoded=malloc(count*2);float*row=malloc((size_t)cols[role]*4);assert(decoded&&row);
            for(int i=0;i<rows[role];i++) {
                assert(!bn_quant_dequant_row(t->type,data,i,cols[role],row));
                for(int j=0;j<cols[role];j++)decoded[(size_t)i*cols[role]+j]=bf16(role==HC_NORM?row[j]-1.0f:row[j]);
            }
            w->handle[role]=g->buffer_create_quant_only(g->ctx,decoded,count*2,BN_GGUF_TENSOR_BF16,rows[role],cols[role]);free(decoded);free(row);
        }
        assert(w->handle[role]);
    }
    if (model) {
        assert(model->n_layers > layer && model->shared_activation == BN_GPU_BF16_FFN_SILU_FUSED);
        const BnBackendBF16FlashFFNWeights *m = &model->layers[layer];
        w->handle[ROUTER] = m->router.decoded; w->handle[SHARED_ROUTER] = m->shared_router.decoded;
        w->handle[SHARED_GATE] = m->shared.gate.raw; w->handle[SHARED_UP] = m->shared.up.raw;
        w->handle[SHARED_DOWN] = m->shared.down.raw;
        for (int i=ROUTER;i<N_WEIGHTS;i++) assert(w->handle[i]);
    }
    const char*roles[]={"gate","up","down"};
    for(int i=0;i<3;i++) {
        char name[128];snprintf(name,sizeof(name),"blk.%d.ffn_%s_exps.weight",layer,roles[i]);
        int ti=bn_gguf_find_tensor(f,name);assert(ti>=0);w->expert_index[i]=ti;
        const BnGGUFTensorInfo*t=&f->tensors[ti];assert(t->n_dims==3&&t->dims[2]==512);
        assert(t->dims[0]==(uint64_t)(i==2?640:2560)&&t->dims[1]==(uint64_t)(i==2?2560:640));
        assert(bn_gguf_tensor_size(t->type,2560*640,&w->expert_bytes[i]));w->expert_data[i]=bn_gguf_tensor_data(f,ti);assert(w->expert_data[i]);
    }
}
static void linear(BnGPUBackend*g,void*r,void*w,int nt,int rows,int cols,BnGPUResidentView x,BnGPUResidentView out) {
    BnGPUBF16LinearResidentPlan p={0};p.n_tokens=nt;p.rows=rows;p.cols=cols;p.weight.type=BN_GGUF_TENSOR_BF16;p.weight.decoded=w;p.x=x;p.out=out;
    assert(!g->linear_bf16_resident(g->ctx,r,&p));
}
typedef struct {
    BnGPUResidentView hidden,norm,down,scaled,gate,input,inject,combined,metadata;
    BnGPUMoETopKBF16ResidentPlan router;
    BnGPUMoEGroupBF16ResidentPlan group;
    BnGPUMoEReduceBF16ResidentPlan reduce;
    BnGPUFFNResidentPlan shared_ffn;
    BnGPUMoESharedBF16ResidentPlan combine;
    BnGPUResidentView expert_gate,expert_up;
} Frame;
static void prepare(BnGPUBackend*g,void*r,int nt,Weights*w,Frame*v) {
    memset(v,0,sizeof(*v));size_t n=(size_t)nt*10,narrow=(size_t)nt*2560,wide=narrow*4;
#define BF(field,count) v->field=alloc(g,r,BN_GGUF_TENSOR_BF16,count)
    BF(hidden,wide);BF(norm,wide);BF(down,(size_t)nt*320);BF(scaled,(size_t)nt*320);BF(gate,wide);BF(input,narrow);BF(inject,(size_t)nt*4);BF(combined,wide);
    BF(router.logits,(size_t)nt*512);BF(group.gathered,n*2560);BF(expert_gate,n*640);BF(expert_up,n*640);
    BF(reduce.grouped,n*2560);BF(reduce.weighted,n*2560);BF(reduce.out,narrow);
    BF(shared_ffn.gate,(size_t)nt*640);BF(shared_ffn.up,(size_t)nt*640);BF(shared_ffn.out,narrow);
    BF(combine.logits,nt);BF(combine.sigmoid,nt);BF(combine.shared,narrow);BF(combine.out,narrow);
#undef BF
    v->router.n_tokens=nt;v->router.n_experts=512;v->router.k=10;v->router.renormalize=1;
    v->router.weights=alloc(g,r,BN_GGUF_TENSOR_F32,n);v->router.indices=alloc(g,r,BN_GGUF_TENSOR_I32,n);v->router.source_rows=alloc(g,r,BN_GGUF_TENSOR_I32,n);
    v->metadata=alloc(g,r,BN_GGUF_TENSOR_I32,520);assert(!g->resident_zero(r,v->metadata));
    v->group.n_tokens=nt;v->group.n_experts=512;v->group.k=10;v->group.dim=2560;v->group.x=v->input;v->group.indices=v->router.indices;
    v->group.offsets=bn_gpu_resident_view_slice(v->metadata,0,513);v->group.status=bn_gpu_resident_view_slice(v->metadata,516,1);
    v->group.cursors=alloc(g,r,BN_GGUF_TENSOR_I32,512);v->group.slots=alloc(g,r,BN_GGUF_TENSOR_I32,n);v->group.inverse=alloc(g,r,BN_GGUF_TENSOR_I32,n);
    v->reduce.n_tokens=nt;v->reduce.k=10;v->reduce.dim=2560;v->reduce.inverse=v->group.inverse;v->reduce.weights=v->router.weights;v->reduce.status=v->group.status;
    v->shared_ffn.n_tokens=nt;v->shared_ffn.dim=2560;v->shared_ffn.hidden_dim=640;v->shared_ffn.x=v->input;
    v->shared_ffn.activation_profile=BN_GPU_BF16_FFN_SILU_FUSED;
    BnGPUBF16LinearWeight*sw[]={&v->shared_ffn.weights.gate,&v->shared_ffn.weights.up,&v->shared_ffn.weights.down};
    for(int i=0;i<3;i++){sw[i]->type=BN_GGUF_TENSOR_Q8_0;sw[i]->raw=w->handle[SHARED_GATE+i];}
    v->combine.n_tokens=nt;v->combine.dim=2560;v->combine.routed=v->reduce.out;v->combine.down=v->shared_ffn.out;
    assert(!g->linear_bf16_reserve(g->ctx,r,0)&&!g->linear_a8_bf16_reserve(g->ctx,r,narrow));
    (void)reserve_expert_banks(g,r,nt,8,1,1);
    BnGPUFFNResidentPlan bad=v->shared_ffn;
    bad.activation_profile=(BnGPUBF16FFNActivationProfile)2;
    assert(g->dense_ffn_bf16_resident(g->ctx,r,&bad)==-1);
    bad.activation_profile=(BnGPUBF16FFNActivationProfile)-1;
    assert(g->dense_ffn_bf16_resident(g->ctx,r,&bad)==-1);
}
static void execute(BnGPUBackend*g,void*r,BnGGUFFile*f,Weights*w,Frame*v,int nt) {
    BnGPURMSNormResidentPlan norm={0};norm.n_tokens=nt*4;norm.dim=2560;norm.weight_offset=1;norm.weight_groups=4;
    norm.reduction=BN_GPU_RMSNORM_REDUCTION_CONTIGUOUS_F32;norm.eps=1e-6f;norm.weight=w->handle[HC_NORM];norm.x=v->hidden;norm.out=v->norm;
    assert(!g->rmsnorm_bf16_resident(g->ctx,r,&norm));
    linear(g,r,w->handle[HC_DOWN],nt,320,10240,v->norm,v->down);
    BnGPUScaledSiluResidentPlan silu={0};silu.rows=nt;silu.cols=320;silu.divisor=4;silu.x=v->down;silu.out=v->scaled;
    assert(!g->scaled_silu_bf16_resident(g->ctx,r,&silu));
    linear(g,r,w->handle[HC_UP],nt,10240,320,v->scaled,v->gate);
    BnGPUHyperMixResidentPlan mix={0};mix.n_tokens=nt;mix.dim=2560;mix.streams=4;mix.normalized=v->norm;mix.gate=v->gate;mix.out=v->input;
    assert(!g->hyper_connection_mix_bf16_resident(g->ctx,r,&mix));
    linear(g,r,w->handle[ROUTER],nt,512,2560,v->input,v->router.logits);
    linear(g,r,w->handle[SHARED_ROUTER],nt,1,2560,v->input,v->combine.logits);
    assert(!g->moe_topk_bf16_resident(g->ctx,r,&v->router)&&!g->moe_group_bf16_resident(g->ctx,r,&v->group));
    int32_t metadata[520];assert(!g->resident_read(r,v->metadata,metadata)&&metadata[0]==0&&metadata[512]==nt*10&&!metadata[516]);
    assert(!g->dense_ffn_bf16_resident(g->ctx,r,&v->shared_ffn));
    BnGPUMoEBankStats stats;
    int groups=banked_ffn(g,r,f,w->expert_index,w->expert_data,w->expert_bytes,nt,metadata,
        v->group.gathered,v->expert_gate,v->expert_up,v->reduce.grouped,v->input,v->group.slots,v->group.status,8,0,1,1,&stats);
    /* The staging helper's counters are cumulative when a frame is replayed. */
    (void)groups;
    assert(!g->moe_reduce_bf16_resident(g->ctx,r,&v->reduce));
    assert(!g->moe_shared_bf16_resident(g->ctx,r,&v->combine));
    linear(g,r,w->handle[HC_INJECT],nt,4,10240,v->norm,v->inject);
    BnGPUHyperCombineResidentPlan post={0};post.n_tokens=nt;post.dim=2560;post.streams=4;
    post.residual=v->hidden;post.block_out=v->combine.out;post.inject=v->inject;post.out=v->combined;
    assert(!g->hyper_connection_combine_bf16_resident(g->ctx,r,&post));
}
static int acquire_calls;
static int (*saved_acquire)(void*,unsigned,int,int,BnGPUMoEBankLease*);
static int busy_once(void*r,unsigned bank,int gt,int dt,BnGPUMoEBankLease*l) {
    if(acquire_calls++==1)return BN_GPU_MOE_BANK_BUSY;
    return saved_acquire(r,bank,gt,dt,l);
}
static int (*saved_alloc)(void*,int,size_t,BnGPUResidentView*);
static size_t allocated_bytes;
static int tracked_alloc(void*r,int type,size_t n,BnGPUResidentView*v) {
    int result=saved_alloc(r,type,n,v);if(!result)allocated_bytes+=n*(type==BN_GGUF_TENSOR_BF16?2:4);return result;
}
static int reject_alloc(void*r,int type,size_t n,BnGPUResidentView*v) {
    (void)r;(void)type;(void)n;(void)v;assert(!"FFN submission allocated");return -1;
}
static int reject_write(void*r,BnGPUResidentView v,const void*p) {
    (void)r;(void)v;(void)p;assert(!"FFN submission uploaded activation");return -1;
}
static void request_case(BnModel *model,BnGPUBackend*g,const char*dir,int layer,int nt,int prefill,
    size_t*bfwords,size_t*words32,BnTransformerBF16FlashFFNInfo *result) {
    BnBackendSession*s=bn_backend_session_create();assert(s);
    saved_alloc=g->resident_alloc;g->resident_alloc=tracked_alloc;allocated_bytes=0;
    assert(!bn_transformer_bf16_flash_prepare_context(model,g,s,nt,nt+8,(size_t)4*1024*1024*1024,256u*1024*1024));
    g->resident_alloc=saved_alloc;
    BnTransformerBF16FlashInfo info;BnTransformerBF16FlashFFNInfo fi;
    assert(!bn_transformer_bf16_flash_info(s,&info)&&!info.active&&!info.failed);
    assert(!bn_transformer_bf16_flash_ffn_info(s,&fi)&&!fi.pending&&!fi.submitted_blocks);
    size_t opaque=g->linear_bf16_workspace_size(g->ctx)+g->gdn_prefill_bf16_workspace_size(g->ctx,nt)+
        fi.banks.reserved_device_bytes+fi.banks.reserved_input_bytes+fi.banks.reserved_group_device_bytes+(size_t)nt*2880;
    assert(info.gpu_bytes==allocated_bytes+opaque);
    void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashHCViews hc;
    assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));
    char prefix[64];snprintf(prefix,sizeof(prefix),"layer%d-%d",layer,nt);
    void*hidden=load(dir,prefix,"hc-hidden",hc.hidden.count*2);assert(!g->resident_write(r,hc.hidden,hidden));free(hidden);
    assert(bn_transformer_bf16_flash_ffn_submit(s,layer)==-1);
    assert(!bn_transformer_bf16_flash_begin_blocks(s,nt,0,prefill,0,UINT64_C(1)<<layer,0));
    assert(bn_transformer_bf16_flash_ffn_submit(s,(layer+1)%48)==-1);
    saved_acquire=g->moe_bank_acquire;g->moe_bank_acquire=busy_once;acquire_calls=0;
    int(*write)(void*,BnGPUResidentView,const void*)=g->resident_write;
    g->resident_alloc=reject_alloc;g->resident_write=reject_write;
    int status,busy=0;
    while((status=bn_transformer_bf16_flash_ffn_submit(s,layer))==BN_GPU_MOE_BANK_BUSY) {
        busy++;assert(!bn_transformer_bf16_flash_ffn_info(s,&fi)&&fi.pending&&fi.metadata_read_bytes==2080);
        assert(!bn_transformer_bf16_flash_info(s,&info)&&info.active&&!info.failed&&!info.submitted_ffn&&!info.position);
        assert(bn_transformer_bf16_flash_hc_combine(s)==-1);
        assert(bn_transformer_bf16_flash_complete(s)==-1);
        assert(bn_transformer_bf16_flash_ffn_submit(s,(layer+1)%48)==-1);
        thrd_yield();
    }
    assert(!status&&busy);g->moe_bank_acquire=saved_acquire;
    g->resident_alloc=saved_alloc;g->resident_write=write;
    assert(!bn_transformer_bf16_flash_ffn_info(s,&fi)&&!fi.pending&&fi.submitted_blocks==1&&fi.metadata_read_bytes==2080);
    assert(fi.banks.input_preparations==1&&fi.banks.submitted_experts>=10&&fi.banks.grouped_banks==fi.banks.submitted_banks);
    assert(!bn_transformer_bf16_flash_info(s,&info)&&info.submitted_ffn==(UINT64_C(1)<<layer)&&!info.position&&!info.failed);
    assert(bn_transformer_bf16_flash_ffn_submit(s,layer)==-1);
    assert(!bn_transformer_bf16_flash_complete(s));
    assert(!bn_transformer_bf16_flash_info(s,&info)&&info.position==nt&&!info.active);
    assert(bn_transformer_bf16_flash_begin_blocks(s,1,0,0,0,UINT64_C(1)<<layer,0)==-1);
    assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));BnTransformerBF16FlashFFNViews v;
    assert(!bn_transformer_bf16_flash_ffn_views(s,nt,&v));
    BnGPUResidentView bv[]={hc.residual,hc.mixed,v.logits,v.shared_logits,v.shared_gate,v.shared_down,v.sigmoid,v.shared,v.block,hc.combined,v.weighted,v.routed};
    const char*names[]={"hc-hidden","input","logits","shared-logits","shared-activation","shared-down","shared-sigmoid","shared","block","hc-combined","routed-weighted","routed"};
    for(int i=0;i<12;i++)*bfwords+=compare(g,r,bv[i],dir,prefix,names[i],2);
    *words32+=compare(g,r,v.weights,dir,prefix,"weights",4);*words32+=compare(g,r,v.indices,dir,prefix,"ids",4);*words32+=compare(g,r,v.source_rows,dir,prefix,"source",4);
    int32_t*slots=malloc((size_t)nt*10*4);assert(slots&&!g->resident_read(r,v.slots,slots));
    *bfwords+=mapped_compare(g,r,v.gate,slots,dir,prefix,"routed-activation",640,640,0);
    *bfwords+=mapped_compare(g,r,v.up,slots,dir,prefix,"routed-gateup",640,1280,640);
    *bfwords+=mapped_compare(g,r,v.down,slots,dir,prefix,"routed-down",2560,2560,0);free(slots);
    assert(!bn_transformer_bf16_flash_restart(s));
    assert(!bn_transformer_bf16_flash_begin_blocks(s,1,0,0,0,UINT64_C(1)<<layer,0));
    /* An incomplete transaction never commits on teardown. */
    assert(bn_transformer_bf16_flash_complete(s)==-1);
    printf("%s production FFN exact banks%zu raw_bytes%zu source_bytes%zu busy%d gpu_bytes%zu host_bytes%zu\n",prefix,fi.banks.submitted_banks,fi.banks.upload_bytes,fi.source.io_bytes,busy,info.gpu_bytes,info.host_bytes);fflush(stdout);
    if(result)*result=fi;
    bn_backend_session_free(s);
}

static int fail_group_reserve(void*ctx,void*r,int nt,size_t h,size_t d) {
    (void)ctx;(void)r;(void)nt;(void)h;(void)d;return -1;
}
static int fail_reduce(void*ctx,void*r,const BnGPUMoEReduceBF16ResidentPlan*p) {
    (void)ctx;(void)r;(void)p;return -1;
}
static int fail_sync(void*r) {(void)r;return -1;}
static void request_guards(BnModel*m,BnGPUBackend*g,const char*dir) {
    size_t budget=(size_t)4*1024*1024*1024,host=256u*1024*1024;
    BnBackendSession*s=bn_backend_session_create();assert(s);
    int(*shared)(void*,void*,const BnGPUMoESharedBF16ResidentPlan*)=g->moe_shared_bf16_resident;
    g->moe_shared_bf16_resident=NULL;
    assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,budget,host)==-1);
    assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
    g->moe_shared_bf16_resident=shared;
    assert(!bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,budget,host));
    BnTransformerBF16FlashInfo limits;assert(!bn_transformer_bf16_flash_info(s,&limits));
    bn_backend_session_free(s);
    s=bn_backend_session_create();assert(s);
    assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,limits.gpu_bytes-1,host)==-1);
    assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,budget,limits.preparation_host_peak_bytes-1)==-1);
    assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
    int(*reserve)(void*,void*,int,size_t,size_t)=g->moe_bank_grouped_reserve;
    g->moe_bank_grouped_reserve=fail_group_reserve;
    assert(bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,budget,host)==-1);
    g->moe_bank_grouped_reserve=reserve;
    assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
    assert(!bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,limits.gpu_bytes,limits.preparation_host_peak_bytes));
    bn_backend_session_free(s);
    /* Failure after banks have queued, and failure at outer completion. */
    for(int mode=0;mode<2;mode++) {
        s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,1,9,budget,host));
        void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashHCViews hc;
        assert(!bn_transformer_bf16_flash_hc_views(s,1,&hc));
        void*h=load(dir,"layer3-1","hc-hidden",hc.hidden.count*2);assert(!g->resident_write(r,hc.hidden,h));free(h);
        assert(!bn_transformer_bf16_flash_begin_blocks(s,1,0,1,0,UINT64_C(1)<<3,0));
        int(*reduce)(void*,void*,const BnGPUMoEReduceBF16ResidentPlan*)=g->moe_reduce_bf16_resident;
        if(!mode)g->moe_reduce_bf16_resident=fail_reduce;
        int result;while((result=bn_transformer_bf16_flash_ffn_submit(s,3))==BN_GPU_MOE_BANK_BUSY)thrd_yield();
        g->moe_reduce_bf16_resident=reduce;
        if(!mode)assert(result==-1);
        else {
            assert(!result);int(*sync)(void*)=g->resident_synchronize;g->resident_synchronize=fail_sync;
            assert(bn_transformer_bf16_flash_complete(s)==-1);g->resident_synchronize=sync;
        }
        BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info)&&info.failed&&!info.position&&info.active);
        if(!mode)assert(!info.submitted_ffn);
        assert(bn_transformer_bf16_flash_complete(s)==-1&&bn_transformer_bf16_flash_restart(s)==-1);
        bn_backend_session_free(s);
    }
    /* Reuse one allocation across a prompt chunk, a second chunk and decode.
     * Neural inputs are independent preserved FFN fixtures; positions advance
     * transactionally on the actual production request, without restart. */
    s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,65,80,budget,host));
    int nts[]={5,65,1},position=0;
    for(int phase=0;phase<3;phase++) {
        int nt=nts[phase];char prefix[64];snprintf(prefix,sizeof(prefix),"layer3-%d",nt);
        void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashHCViews hc;
        assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));
        void*h=load(dir,prefix,"hc-hidden",hc.hidden.count*2);assert(!g->resident_write(r,hc.hidden,h));free(h);
        assert(!bn_transformer_bf16_flash_begin_blocks(s,nt,position,phase<2,0,UINT64_C(1)<<3,0));
        int status;while((status=bn_transformer_bf16_flash_ffn_submit(s,3))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status);
        BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info)&&info.position==position);
        assert(!bn_transformer_bf16_flash_complete(s));position+=nt;
        assert(!bn_transformer_bf16_flash_info(s,&info)&&info.position==position);
        assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));
        (void)compare(g,r,hc.combined,dir,prefix,"hc-combined",2);
    }
    BnTransformerBF16FlashFFNInfo reused;
    assert(!bn_transformer_bf16_flash_ffn_info(s,&reused)&&reused.submitted_blocks==3&&reused.metadata_read_bytes==6240&&reused.banks.input_preparations==3);
    bn_backend_session_free(s);puts("FFN_REQUEST_REUSE PASSED: chunks5+65 decode1, position71");
    /* Two requests share immutable weights but retain separate live banks,
     * activation scratch, HC residuals and transaction positions. */
    BnBackendSession *sessions[2]={bn_backend_session_create(),bn_backend_session_create()};
    for(int j=0;j<2;j++) {
        assert(sessions[j]&&!bn_transformer_bf16_flash_prepare_context(m,g,sessions[j],1,9,budget,host));
        void*r=bn_gpu_backend_resident_request(g,sessions[j]);BnTransformerBF16FlashHCViews hc;
        assert(!bn_transformer_bf16_flash_hc_views(sessions[j],1,&hc));
        void*h=load(dir,"layer3-1","hc-hidden",hc.hidden.count*2);assert(!g->resident_write(r,hc.hidden,h));free(h);
        assert(!bn_transformer_bf16_flash_begin_blocks(sessions[j],1,0,1,0,UINT64_C(1)<<3,0));
    }
    saved_acquire=g->moe_bank_acquire;g->moe_bank_acquire=busy_once;acquire_calls=0;
    assert(bn_transformer_bf16_flash_ffn_submit(sessions[0],3)==BN_GPU_MOE_BANK_BUSY);
    g->moe_bank_acquire=saved_acquire;
    for(int j=1;j>=0;j--) {
        int status;while((status=bn_transformer_bf16_flash_ffn_submit(sessions[j],3))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status);
        assert(!bn_transformer_bf16_flash_complete(sessions[j]));
        void*r=bn_gpu_backend_resident_request(g,sessions[j]);BnTransformerBF16FlashHCViews hc;
        assert(!bn_transformer_bf16_flash_hc_views(sessions[j],1,&hc));
        (void)compare(g,r,hc.combined,dir,"layer3-1","hc-combined",2);
    }
    for(int j=0;j<2;j++)bn_backend_session_free(sessions[j]);
    puts("FFN_REQUEST_ISOLATION PASSED: interleaved sessions, separate live banks");
    puts("FFN_REQUEST_GUARDS PASSED: budgets, rollback, pending work, outer completion");fflush(stdout);
}

static void io_case(BnModel*m,BnGPUBackend*g,const char*ffndir,const char*iodir,int nt,
    size_t *bfwords,size_t *words32) {
    size_t budget=(size_t)4*1024*1024*1024,host=256u*1024*1024;
    BnBackendSession*s=bn_backend_session_create();assert(s);
    if(nt==1) {
        int(*embedding)(void*,void*,const BnGPUEmbeddingResidentPlan*)=g->embedding_bf16_resident;
        g->embedding_bf16_resident=NULL;
        assert(bn_transformer_bf16_flash_prepare_context(m,g,s,nt,nt+8,budget,host)==-1);
        assert(!bn_backend_session_has_gpu_resources(s)&&!bn_backend_session_has_execution_resources(s));
        g->embedding_bf16_resident=embedding;
    }
    saved_alloc=g->resident_alloc;g->resident_alloc=tracked_alloc;allocated_bytes=0;
    assert(!bn_transformer_bf16_flash_prepare_context(m,g,s,nt,nt+8,budget,host));g->resident_alloc=saved_alloc;
    BnTransformerBF16FlashInfo info;BnTransformerBF16FlashFFNInfo fi;
    assert(!bn_transformer_bf16_flash_info(s,&info)&&!bn_transformer_bf16_flash_ffn_info(s,&fi));
    size_t opaque=g->linear_bf16_workspace_size(g->ctx)+g->gdn_prefill_bf16_workspace_size(g->ctx,nt)+
        fi.banks.reserved_device_bytes+fi.banks.reserved_input_bytes+fi.banks.reserved_group_device_bytes+(size_t)nt*2880+4;
    assert(info.gpu_bytes==allocated_bytes+opaque);
    void*r=bn_gpu_backend_resident_request(g,s);BnTransformerBF16FlashIOViews v;
    assert(!bn_transformer_bf16_flash_io_views(s,nt,&v));
    char prefix[64];snprintf(prefix,sizeof(prefix),"input-%d",nt);
    int32_t *ids=load(iodir,prefix,"ids",(size_t)nt*4);
    assert(bn_transformer_bf16_flash_tokens_submit(s,ids,nt)==-1);
    assert(!bn_transformer_bf16_flash_begin_blocks(s,nt,0,1,0,UINT64_C(1)<<3,0));
    int32_t first=ids[0];ids[0]=-1;assert(bn_transformer_bf16_flash_tokens_submit(s,ids,nt)==-1);ids[0]=first;
    assert(!bn_transformer_bf16_flash_tokens_submit(s,ids,nt));
    assert(bn_transformer_bf16_flash_tokens_submit(s,ids,nt)==-1);
    *words32+=compare(g,r,v.ids,iodir,prefix,"ids",4);
    *bfwords+=compare(g,r,v.embedding,iodir,prefix,"embedding",2);
    *bfwords+=compare(g,r,v.hidden,iodir,prefix,"wide",2);free(ids);
    assert(bn_transformer_bf16_flash_complete(s)==-1);bn_backend_session_free(s);
    s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,nt,nt+8,budget,host));r=bn_gpu_backend_resident_request(g,s);
    BnTransformerBF16FlashHCViews hc;assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));
    snprintf(prefix,sizeof(prefix),"layer3-%d",nt);void *hidden=load(ffndir,prefix,"hc-hidden",hc.hidden.count*2);
    assert(!g->resident_write(r,hc.hidden,hidden));free(hidden);
    assert(!bn_transformer_bf16_flash_begin_blocks(s,nt,0,1,0,UINT64_C(1)<<3,0));
    assert(bn_transformer_bf16_flash_head_submit(s)==-1);
    int(*alloc_saved)(void*,int,size_t,BnGPUResidentView*)=g->resident_alloc;
    int(*write_saved)(void*,BnGPUResidentView,const void*)=g->resident_write;
    g->resident_alloc=reject_alloc;g->resident_write=reject_write;
    int status;while((status=bn_transformer_bf16_flash_ffn_submit(s,3))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status);
    assert(!bn_transformer_bf16_flash_head_submit(s));assert(bn_transformer_bf16_flash_head_submit(s)==-1);
    assert(!bn_transformer_bf16_flash_complete(s));g->resident_alloc=alloc_saved;g->resident_write=write_saved;
    assert(!bn_transformer_bf16_flash_io_views(s,nt,&v));snprintf(prefix,sizeof(prefix),"head-%d",nt);
    *bfwords+=compare(g,r,v.hidden,iodir,prefix,"input",2);
    *bfwords+=compare(g,r,v.mixed,iodir,prefix,"mixed",2);
    *bfwords+=compare(g,r,v.logits,iodir,prefix,"logits",2);
    bn_backend_session_free(s);printf("FLASH_IO_SESSION nt%d exact\n",nt);fflush(stdout);
}

static void model_smoke(BnModel*m,BnGPUBackend*g) {
    BnBackendSession*s=bn_backend_session_create();assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,1,8,(size_t)4*1024*1024*1024,256u*1024*1024));
    void*r=bn_gpu_backend_resident_request(g,s);uint16_t *logits=malloc((size_t)m->config.vocab_size*2);assert(logits);
    int32_t token=116;
    for(int phase=0;phase<2;phase++) {
        int32_t expected=token;
        assert(!bn_transformer_bf16_flash_model_begin(s,&token,1,phase,phase==0));token=0;
        saved_acquire=g->moe_bank_acquire;g->moe_bank_acquire=busy_once;acquire_calls=0;
        int status,busy=0;
        while((status=bn_transformer_bf16_flash_model_submit(s))==BN_GPU_MOE_BANK_BUSY) {
            busy++;BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info)&&info.active&&!info.failed&&info.position==phase);
            assert(bn_transformer_bf16_flash_complete(s)==-1);thrd_yield();
        }
        g->moe_bank_acquire=saved_acquire;assert(!status&&busy);
        assert(!bn_transformer_bf16_flash_model_submit(s));
        BnTransformerBF16FlashIOViews v;assert(!bn_transformer_bf16_flash_io_views(s,1,&v));
        int32_t actual;assert(!g->resident_read(r,v.ids,&actual)&&actual==expected);
        assert(!g->resident_read(r,v.logits,logits));
        float best=-1e30f;int argmax=-1;
        for(int i=0;i<m->config.vocab_size;i++) {
            assert((logits[i]&0x7f80)!=0x7f80);uint32_t bits=(uint32_t)logits[i]<<16;float value;memcpy(&value,&bits,4);
            if(value>best){best=value;argmax=i;}
        }
        assert(argmax>=0);BnTransformerBF16FlashFFNInfo fi;assert(!bn_transformer_bf16_flash_ffn_info(s,&fi));
        assert(fi.submitted_blocks==(size_t)(phase+1)*48&&fi.metadata_read_bytes==(size_t)(phase+1)*48*2080);
        assert(!bn_transformer_bf16_flash_complete(s));BnTransformerBF16FlashInfo info;
        assert(!bn_transformer_bf16_flash_info(s,&info)&&!info.active&&info.position==phase+1&&!info.failed);
        token=argmax;printf("FLASH_MODEL_SMOKE phase%d input%d argmax%d finite%d position%d blocks%zu busy%d\n",phase,expected,argmax,m->config.vocab_size,info.position,fi.submitted_blocks,busy);fflush(stdout);
    }
    free(logits);bn_backend_session_free(s);
}

static size_t whole_compare(BnGPUBackend*g,void*r,BnGPUResidentView v,const char*dir,
    const char*dump,const char*prefix,const char*name,int width) {
    size_t bytes=v.count*(size_t)width;
    unsigned char *expected=load(dir,prefix,name,bytes),*actual=malloc(bytes);assert(actual&&!g->resident_read(r,v,actual));
    size_t diff=0;
    for(size_t i=0;i<v.count;i++)if(memcmp(actual+i*width,expected+i*width,width)) {
        if(diff<3){uint32_t a=0,b=0;memcpy(&a,actual+i*width,width);memcpy(&b,expected+i*width,width);fprintf(stderr,"WHOLE_DIFF %s %s word%zu native%08x reference%08x\n",prefix,name,i,a,b);}diff++;
    }
    if(diff) {
        char file[4096];int n=snprintf(file,sizeof(file),"%s/%s-%s.actual.bin",dump,prefix,name);assert(n>0&&(size_t)n<sizeof(file));
        FILE*f=fopen(file,"wb");assert(f&&fwrite(actual,1,bytes,f)==bytes);fclose(f);
        fprintf(stderr,"WHOLE_DIFF %s %s differences%zu/%zu saved%s\n",prefix,name,diff,v.count,file);fflush(stderr);
    }
    assert(!diff);free(expected);free(actual);return v.count;
}
static int whole_has(const char*dir,const char*prefix,const char*name) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s-%s.bin",dir,prefix,name);assert(n>0&&(size_t)n<sizeof(path));
    FILE*f=fopen(path,"rb");if(!f)return 0;fclose(f);return 1;
}
static BnGPUBackend *whole_gpu;
static const char *whole_dir,*whole_dump;
static char whole_tag[96];
static int (*whole_prepare)(void*,void*,const BnGPUGDNPrepareResidentPlan*);
static int (*whole_read)(void*,BnGPUResidentView,void*);
static int whole_reject_logits_read(void *request,BnGPUResidentView view,void *host) {
    return view.count==248320 ? -1 : whole_read(request,view,host);
}
static int whole_reject_bank(void *ctx,void *request,const BnGPUMoEBankResidentPlan *plan) {
    (void)ctx;(void)request;(void)plan;return -1;
}
static void *whole_reject_cache(void *ctx,const BnGPUExpertCacheConfig *config) {
    (void)ctx;(void)config;return NULL;
}
static int whole_observe_prepare(void*ctx,void*r,const BnGPUGDNPrepareResidentPlan*p) {
    int status=whole_prepare(ctx,r,p);if(status)return status;
    if(whole_has(whole_dir,whole_tag,"gdn-q")) {
        BnGPUResidentView v[]={p->q,p->k,p->v,p->log_decay,p->beta};
        const char*n[]={"gdn-q","gdn-k","gdn-v","gdn-log_decay","gdn-beta"};
        for(int i=0;i<5;i++)(void)whole_compare(whole_gpu,r,v[i],whole_dir,whole_dump,whole_tag,n[i],i<3?2:4);
    }
    return 0;
}
typedef struct { size_t used; uint64_t keys; } WholeCacheState;

/* Inspect published membership without changing LRU order or retaining leases.
 * This harness serializes requests, as required by the source-cache API. */
static WholeCacheState whole_cache_state(BnModel *m) {
    BnMoECache *cache=bn_model_moe_cache(m);assert(cache);
    int capacity=bn_moe_cache_capacity_internal(cache);assert(capacity>0);
    assert(bn_moe_cache_available_internal(cache)==capacity);
    WholeCacheState state={0,UINT64_C(14695981039346656037)};
    for(int layer=0;layer<m->config.n_layers;layer++) {
        const BnMoEExpertMap *map=&m->weights.layers[layer].moe.expert_map;
        const size_t bytes[]={map->expert_gate_bytes,map->expert_up_bytes,map->expert_down_bytes};
        assert(bn_moe_cache_fits_internal(cache,bytes));
        for(int expert=0;expert<512;expert++) {
            int slot=-1;
            if(!bn_moe_cache_pin_internal(cache,layer,expert,0,&slot))continue;
            assert(slot>=0&&slot<capacity);state.used++;
            state.keys=(state.keys^((uint64_t)layer*512+(unsigned)expert))*UINT64_C(1099511628211);
            state.keys=(state.keys^(unsigned)slot)*UINT64_C(1099511628211);
            bn_moe_cache_unpin_internal(cache,slot);
        }
    }
    assert(state.used<=(size_t)capacity&&bn_moe_cache_available_internal(cache)==capacity);
    return state;
}

static void whole_pread_stats(BnModel *m,BnBackendSession *s,int driver,int phase,int prefill,
    const BnTransformerBF16FlashFFNInfo *before,WholeCacheState cache_before) {
    BnTransformerBF16FlashFFNInfo after;assert(!bn_transformer_bf16_flash_ffn_info(s,&after));
    WholeCacheState cache_after=whole_cache_state(m);
    assert(after.banks.submitted_experts>=before->banks.submitted_experts &&
           after.banks.upload_bytes>=before->banks.upload_bytes &&
           after.source.io_bytes>=before->source.io_bytes &&
           after.source.gpu_host_cache_hits>=before->source.gpu_host_cache_hits &&
           after.source.gpu_host_cache_misses>=before->source.gpu_host_cache_misses);
    size_t experts=after.banks.submitted_experts-before->banks.submitted_experts;
    size_t uploads=after.banks.upload_bytes-before->banks.upload_bytes;
    size_t reads=after.source.io_bytes-before->source.io_bytes;
    size_t hits=after.source.gpu_host_cache_hits-before->source.gpu_host_cache_hits;
    size_t misses=after.source.gpu_host_cache_misses-before->source.gpu_host_cache_misses;
    size_t hit_bytes=after.source.gpu_host_cache_hit_bytes-before->source.gpu_host_cache_hit_bytes;
    size_t miss_bytes=after.source.gpu_host_cache_miss_bytes-before->source.gpu_host_cache_miss_bytes;
    size_t gpu_hits=after.cache.hits-before->cache.hits,gpu_misses=after.cache.misses-before->cache.misses;
    int resident=bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(m))->expert_cache!=NULL;
    assert(experts&&hits+misses==(resident?gpu_misses:experts)&&hit_bytes+miss_bytes==uploads&&miss_bytes==reads);
    if(resident) {
        if(before->cache.resident_experts==24576)assert(gpu_hits==experts&&!gpu_misses&&!uploads&&!reads&&!hits&&!misses);
        assert(gpu_hits+gpu_misses==experts&&after.cache.miss_bytes-before->cache.miss_bytes==uploads);
        assert(after.cache.resident_bytes<=after.cache.reserved_device_bytes);
        printf("FLASH_GPU_CACHE_PHASE driver=%d phase=%d prefill=%d hits=%zu misses=%zu hit_bytes=%zu miss_bytes=%zu reserved_bytes=%zu resident_bytes=%zu evictions=%zu bypasses=%zu fill_copy_bytes=%zu\n",
            driver,phase,prefill,gpu_hits,gpu_misses,after.cache.hit_bytes-before->cache.hit_bytes,
            after.cache.miss_bytes-before->cache.miss_bytes,after.cache.reserved_device_bytes,
            after.cache.resident_bytes,after.cache.evictions-before->cache.evictions,
            after.cache.bypasses-before->cache.bypasses,after.cache.fill_copy_bytes-before->cache.fill_copy_bytes);
    } else assert(!gpu_hits&&!gpu_misses);
    assert((size_t)(after.source.io_count-before->source.io_count)==misses*3);
    size_t evictions=0;
    if(prefill) {
        /* Whole-model prefill borrows but never admits/evicts cached experts. */
        assert(cache_before.used==cache_after.used&&cache_before.keys==cache_after.keys);
    } else {
        /* All map sizes fit, leases are released synchronously, and the caller
         * is serialized: every decode miss admits one expert into this LRU. */
        int capacity=bn_moe_cache_capacity_internal(bn_model_moe_cache(m));
        size_t admitted=cache_before.used+misses;
        assert(cache_after.used==(admitted<(size_t)capacity?admitted:(size_t)capacity));
        evictions=admitted-cache_after.used;
    }
    BnTransformerBF16FlashInfo memory;assert(!bn_transformer_bf16_flash_info(s,&memory));
    printf("FLASH_PREAD_PHASE driver=%d phase=%d prefill=%d experts=%zu hits=%zu misses=%zu hit_bytes=%zu miss_bytes=%zu io_bytes=%zu upload_bytes=%zu cache_used_before=%zu cache_used_after=%zu evictions=%zu metadata_read_bytes=%zu banks=%zu busy_acquires=%zu request_gpu_bytes=%zu request_host_bytes=%zu bank_device_bytes=%zu bank_host_bytes=%zu pread_ms=%.6f source_copy_ms=%.6f bank_retry_ms=%.6f metadata_ms=%.6f\n",
        driver,phase,prefill,experts,hits,misses,hit_bytes,miss_bytes,reads,uploads,
        cache_before.used,cache_after.used,evictions,
        after.metadata_read_bytes-before->metadata_read_bytes,
        after.banks.submitted_banks-before->banks.submitted_banks,
        after.banks.busy_acquires-before->banks.busy_acquires,memory.gpu_bytes,memory.host_bytes,
        after.banks.reserved_device_bytes,after.banks.reserved_host_bytes,
        after.source.io_time_ms-before->source.io_time_ms,
        after.source_ms-before->source_ms,after.bank_retry_ms-before->bank_retry_ms,
        after.metadata_ms-before->metadata_ms);fflush(stdout);
}

static void model_reference(BnModel*m,BnGPUBackend*g,const char*dir,const char*dump) {
    size_t bfwords=0,words32=0;int phases=0,max_tokens=1,capacity=2,np=0;
    struct {int phase,position,nt,prefill;} schedule[1024];
    char index[4096];int length=snprintf(index,sizeof(index),"%s/phases.txt",dir);assert(length>0&&(size_t)length<sizeof(index));
    FILE *meta=fopen(index,"r");
    if(meta) {
        while(np<1024&&fscanf(meta,"%d %d %d %d",&schedule[np].phase,&schedule[np].position,&schedule[np].nt,&schedule[np].prefill)==4) {
            assert(schedule[np].phase==np&&schedule[np].nt>0&&schedule[np].position>=0);
            if(schedule[np].nt>max_tokens)max_tokens=schedule[np].nt;
            if(schedule[np].position+schedule[np].nt>capacity)capacity=schedule[np].position+schedule[np].nt;
            np++;
        }
        assert(feof(meta)&&np>=2);fclose(meta);
    } else {
        np=2;for(int i=0;i<2;i++){schedule[i].phase=i;schedule[i].position=i;schedule[i].nt=1;schedule[i].prefill=i==0;}
    }
    int limit=48;snprintf(index,sizeof(index),"%s/layers.txt",dir);meta=fopen(index,"r");
    if(meta){assert(fscanf(meta,"%d",&limit)==1&&limit>0&&limit<=48);fclose(meta);}
    int pread_mode=bn_moe_io_has_pread(bn_model_moe_io_const(m))&&bn_model_moe_cache(m);
    if(pread_mode)assert(!bn_moe_io_has_mmap(bn_model_moe_io_const(m)));
    whole_gpu=g;whole_dir=dir;whole_dump=dump;whole_prepare=g->gdn_prepare_bf16_resident;
    g->gdn_prepare_bf16_resident=whole_observe_prepare;
    BnSession *normal = NULL, *peer = NULL;
    for(int driver=0;driver<(limit==48?4:1);driver++) {
        BnBackendSession *s;
        if (driver == 2) {
            normal = bn_session_create_backend(m,NULL); assert(normal);
            s = bn_session_backend(normal);
            if(max_tokens<=5) {
                peer=bn_session_create_backend(m,NULL);assert(peer);
                assert(!bn_transformer_bf16_flash_prepare_context(m,g,bn_session_backend(peer),max_tokens,capacity,(size_t)4*1024*1024*1024,256u*1024*1024));
            }
        } else if (driver == 3) {
            assert(normal && !bn_transformer_bf16_session_restart(m,normal));
            s = bn_session_backend(normal);
            assert(bn_transformer_bf16_position(s)==0);
        } else s = bn_backend_session_create();
        if (driver != 3) assert(s&&!bn_transformer_bf16_flash_prepare_context(m,g,s,max_tokens,capacity,(size_t)4*1024*1024*1024,256u*1024*1024));
        if(driver>=2) {
            whole_tag[0]=0;
            assert(bn_transformer_bf16_model_active(m));
            assert(bn_transformer_bf16_max_tokens(s)==max_tokens&&bn_transformer_bf16_capacity(s)==capacity);
            int invalid = m->config.vocab_size;
            assert(!bn_transformer_forward(m,normal,invalid,0));
            assert(!bn_transformer_forward(m,normal,0,1));
            assert(bn_transformer_bf16_position(s)==0);
        }
        void*r=bn_gpu_backend_resident_request(g,s);
        for(int phase=0;phase<np;phase++) {
            int nt=schedule[phase].nt,position=schedule[phase].position,prefill=schedule[phase].prefill;
            char file[4096],prefix[64];snprintf(prefix,sizeof(prefix),"phase%d",phase);
            int n=snprintf(file,sizeof(file),"%s/%s-tokens.bin",dir,prefix);assert(n>0&&(size_t)n<sizeof(file));
            FILE*f=fopen(file,"rb");assert(f);
            int32_t *tokens=malloc((size_t)nt*4);assert(tokens&&fread(tokens,4,(size_t)nt,f)==(size_t)nt&&fgetc(f)==EOF);fclose(f);
            BnTransformerBF16FlashFFNInfo source_before={0};WholeCacheState cache_before={0};
            if(pread_mode) {
                assert(!bn_transformer_bf16_flash_ffn_info(s,&source_before));
                cache_before=whole_cache_state(m);
            }
            int want_logits=driver!=3||!prefill;
            if(driver>=2) {
                if(want_logits) {
                    float *out = prefill ? bn_transformer_prefill(m,normal,(const int*)tokens,nt,position) :
                        bn_transformer_forward(m,normal,tokens[0],position);
                    assert(out==normal->state.logits);
                    uint16_t *expected=(uint16_t*)load(dir,prefix,"logits",(size_t)m->config.vocab_size*2);
                    for(int i=0;i<m->config.vocab_size;i++) {
                        uint32_t bits;memcpy(&bits,&out[i],4);
                        assert(bits==((uint32_t)expected[i]<<16));
                    }
                    free(expected);
                } else {
                    size_t bytes=(size_t)m->config.vocab_size*sizeof(float);
                    void *previous=malloc(bytes);assert(previous);
                    memcpy(previous,normal->state.logits,bytes);
                    assert(prefill&&!bn_transformer_prefill_no_logits(m,normal,(const int*)tokens,nt,position));
                    assert(!memcmp(previous,normal->state.logits,bytes));free(previous);
                }
                assert(bn_transformer_bf16_position(s)==position+nt);
                if(driver==2&&peer) {
                    assert(bn_transformer_bf16_position(bn_session_backend(peer))==position);
                    float *out=prefill?bn_transformer_prefill(m,peer,(const int*)tokens,nt,position):
                        bn_transformer_forward(m,peer,tokens[0],position);
                    assert(out&&!memcmp(out,normal->state.logits,(size_t)m->config.vocab_size*sizeof(float)));
                    assert(bn_transformer_bf16_position(bn_session_backend(peer))==position+nt);
                }
            } else if(driver) {
                assert(!bn_transformer_bf16_flash_model_begin(s,tokens,nt,position,prefill));
                int status;while((status=bn_transformer_bf16_flash_model_submit(s))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status);
            } else {
                uint64_t all=(UINT64_C(1)<<limit)-1;
                assert(!bn_transformer_bf16_flash_begin_blocks(s,nt,position,prefill,all,all,1));
                assert(!bn_transformer_bf16_flash_tokens_submit(s,tokens,nt));
                BnTransformerBF16FlashIOViews io;assert(!bn_transformer_bf16_flash_io_views(s,nt,&io));
                bfwords+=whole_compare(g,r,io.embedding,dir,dump,prefix,"embedding",2);
                for(int layer=0;layer<limit;layer++) {
                    char tag[96];snprintf(tag,sizeof(tag),"phase%d-layer%d",phase,layer);
                    if(layer==1) {
                        assert(!bn_transformer_bf16_flash_ple_apply(s,tokens,nt,position));
                        BnTransformerBF16FlashHCViews hc;BnTransformerBF16PLEViews ple;
                        assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc)&&!bn_transformer_bf16_ple_views(s,nt,&ple));
                        bfwords+=whole_compare(g,r,ple.embedding,dir,dump,prefix,"ple-embedding",2);
                        bfwords+=whole_compare(g,r,hc.hidden,dir,dump,prefix,"ple-out",2);
                    }
                    snprintf(whole_tag,sizeof(whole_tag),"%s",tag);
                    assert(!bn_transformer_bf16_flash_attention_submit(s,layer));
                    if(whole_has(dir,tag,"gdn-input")) {
                        BnTransformerBF16FlashGDNViews v;assert(!bn_transformer_bf16_flash_gdn_views(s,layer,nt,&v));
                        BnGPUResidentView views[]={v.input,v.qkvz,v.ba,v.post,v.core,v.norm,v.out};
                        const char*names[]={"gdn-input","gdn-qkvz","gdn-ba","gdn-post","gdn-core","gdn-norm","gdn-out"};
                        for(int j=0;j<7;j++)bfwords+=whole_compare(g,r,views[j],dir,dump,tag,names[j],2);
                    }
                    if(layer==3) {
                        BnTransformerBF16FlashQSAViews v;assert(!bn_transformer_bf16_flash_qsa_views(s,3,nt,&v));
                        BnGPUResidentView views[]={v.input,v.projected,v.q,v.k,v.v,v.index_q,v.attention,v.out};
                        const char*names[]={"qsa-input","qsa-projected","qsa-q","qsa-k","qsa-v","qsa-index_q","qsa-core","qsa-out"};
                        for(int j=0;j<8;j++)bfwords+=whole_compare(g,r,views[j],dir,dump,tag,names[j],2);
                        words32+=whole_compare(g,r,v.indices,dir,dump,tag,"qsa-indices",4);
                    }
                    BnTransformerBF16FlashHCViews hc;assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));
                    bfwords+=whole_compare(g,r,hc.hidden,dir,dump,tag,"attn-hidden",2);
                    int status;while((status=bn_transformer_bf16_flash_ffn_submit(s,layer))==BN_GPU_MOE_BANK_BUSY)thrd_yield();assert(!status);
                    assert(!bn_transformer_bf16_flash_hc_views(s,nt,&hc));
                    if(layer==0) {
                        BnTransformerBF16FlashFFNViews v;assert(!bn_transformer_bf16_flash_ffn_views(s,nt,&v));
                        BnGPUResidentView views[]={hc.mixed,v.logits,v.routed,v.shared,v.block};
                        const char*names[]={"ffn-input","ffn-logits","ffn-routed","ffn-shared","ffn-out"};
                        for(int j=0;j<5;j++)bfwords+=whole_compare(g,r,views[j],dir,dump,tag,names[j],2);
                        words32+=whole_compare(g,r,v.indices,dir,dump,tag,"ffn-ids",4);
                        words32+=whole_compare(g,r,v.weights,dir,dump,tag,"ffn-weights",4);
                    } else if(whole_has(dir,tag,"ffn-ids")) {
                        BnTransformerBF16FlashFFNViews v;assert(!bn_transformer_bf16_flash_ffn_views(s,nt,&v));
                        words32+=whole_compare(g,r,v.indices,dir,dump,tag,"ffn-ids",4);
                    }
                    bfwords+=whole_compare(g,r,hc.hidden,dir,dump,tag,"ffn-hidden",2);
                    printf("WHOLE_LAYER_MATCH phase%d layer%d\n",phase,layer);fflush(stdout);
                }
                if(limit==48)assert(!bn_transformer_bf16_flash_head_submit(s));
            }
            BnTransformerBF16FlashIOViews io;assert(!bn_transformer_bf16_flash_io_views(s,nt,&io));
            if(limit==48&&want_logits) {
                bfwords+=whole_compare(g,r,io.mixed,dir,dump,prefix,"mixed",2);
                bfwords+=whole_compare(g,r,io.logits,dir,dump,prefix,"logits",2);
            }
            if(driver<2)assert(!bn_transformer_bf16_flash_complete(s));
            if(!driver) {
                BnTransformerBF16PLEViews ple;assert(!bn_transformer_bf16_ple_views(s,nt,&ple));
                for(int layer=0;layer<limit;layer++) {
                    char tag[64];snprintf(tag,sizeof(tag),"phase%d-layer%d",phase,layer);
                    if(whole_has(dir,tag,"gdn-state")) {
                        BnTransformerBF16FlashGDNViews state;assert(!bn_transformer_bf16_flash_gdn_views(s,layer,nt,&state));
                        words32+=whole_compare(g,r,state.state,dir,dump,tag,"gdn-state",4);
                        if(whole_has(dir,tag,"gdn-history"))bfwords+=whole_compare(g,r,state.history,dir,dump,tag,"gdn-history",2);
                    }
                }
                bfwords+=whole_compare(g,r,ple.history,dir,dump,prefix,"ple-history",2);
            }
            if(pread_mode)whole_pread_stats(m,s,driver,phase,prefill,&source_before,cache_before);
            BnTransformerBF16FlashInfo info;assert(!bn_transformer_bf16_flash_info(s,&info)&&!info.failed&&info.position==position+nt);
            printf("WHOLE_PHASE_MATCH mode%d phase%d position%d\n",driver,phase,info.position);fflush(stdout);phases++;free(tokens);
        }
        if(driver<2)bn_backend_session_free(s);
        if(driver==2&&peer){bn_session_free(peer,NULL);peer=NULL;}
    }
    if(normal) {
        assert(!bn_transformer_bf16_session_restart(m,normal));
        size_t bytes=(size_t)m->config.vocab_size*sizeof(float);
        void *previous=malloc(bytes);assert(previous);memcpy(previous,normal->state.logits,bytes);
        whole_read=g->resident_read;g->resident_read=whole_reject_logits_read;
        assert(!bn_transformer_forward(m,normal,0,0));
        g->resident_read=whole_read;
        assert(!memcmp(previous,normal->state.logits,bytes));free(previous);
        assert(bn_transformer_bf16_position(bn_session_backend(normal))==-1);
        assert(!bn_transformer_forward(m,normal,0,0));
        assert(bn_transformer_bf16_session_restart(m,normal)==-1);
        bn_session_reset(normal,m);
        assert(bn_transformer_bf16_position(bn_session_backend(normal))==-1);
        assert(!bn_transformer_forward(m,normal,0,0));
        assert(!bn_transformer_bf16_flash_prepare_context(m,g,bn_session_backend(normal),max_tokens,capacity,(size_t)4*1024*1024*1024,256u*1024*1024));
        int nt=schedule[0].nt;int32_t *tokens=(int32_t*)load(dir,"phase0","tokens",(size_t)nt*4);
        float *out=bn_transformer_prefill(m,normal,(const int*)tokens,nt,0);assert(out);
        uint16_t *expected=(uint16_t*)load(dir,"phase0","logits",(size_t)m->config.vocab_size*2);
        for(int i=0;i<m->config.vocab_size;i++) {
            uint32_t bits;memcpy(&bits,&out[i],4);assert(bits==((uint32_t)expected[i]<<16));
        }
        if(bn_backend_layout_bf16_flash_ffn_model(bn_model_backend(m))->expert_cache) {
            assert(!bn_transformer_bf16_session_restart(m,normal));
            int (*bank)(void*,void*,const BnGPUMoEBankResidentPlan*)=g->moe_bank_submit;
            g->moe_bank_submit=whole_reject_bank;
            assert(!bn_transformer_forward(m,normal,0,0));g->moe_bank_submit=bank;
            assert(bn_transformer_bf16_position(bn_session_backend(normal))==-1);
            BnSession *healthy=bn_session_create_backend(m,NULL);assert(healthy);
            assert(!bn_transformer_bf16_flash_prepare_context(m,g,bn_session_backend(healthy),max_tokens,capacity,(size_t)4*1024*1024*1024,256u*1024*1024));
            out=bn_transformer_prefill(m,healthy,(const int*)tokens,nt,0);assert(out);
            for(int i=0;i<m->config.vocab_size;i++) {
                uint32_t bits;memcpy(&bits,&out[i],4);assert(bits==((uint32_t)expected[i]<<16));
            }
            bn_session_free(healthy,NULL);
            puts("FLASH_CACHE_FAILURE_ISOLATION exact");
        }
        free(expected);free(tokens);
        printf("FLASH_RUNTIME_RECOVERY exact session_isolation%d\n",max_tokens<=5);fflush(stdout);
        bn_session_free(normal,NULL);
    }
    g->gdn_prepare_bf16_resident=whole_prepare;
    assert(phases>=(limit==48?4:2));printf("FLASH_WHOLE_MATCH phases%d bf16_words%zu words32=%zu differences0\n",phases,bfwords,words32);
}

int main(int argc,char**argv) {
    int preload=argc>1&&!strcmp(argv[argc-1],"--preload");
    if(preload)argc--;
    int reference_pread=(argc>=6&&argc<=8)&&!strcmp(argv[3],"--model-reference-pread");
    int reference_mode=reference_pread || ((argc==6||argc==7)&&!strcmp(argv[3],"--model-reference"));
    size_t reference_cache_budget=64u*1024*1024;
    if(reference_pread&&argc>=7) {
        char *end=NULL;unsigned long long mb=strtoull(argv[6],&end,10);
        assert(argv[6][0]&&end&&!*end&&mb&&mb<=SIZE_MAX/(1024u*1024));
        reference_cache_budget=(size_t)mb*1024*1024;
    }
    size_t gpu_cache_budget=0;
    if((reference_pread&&argc==8)||(!reference_pread&&reference_mode&&argc==7)) {
        const char *value=argv[reference_pread?7:6];char *end=NULL;
        unsigned long long mb=strtoull(value,&end,10);
        assert(value[0]&&end&&!*end&&mb<=SIZE_MAX/(1024u*1024));
        gpu_cache_budget=(size_t)mb*1024*1024;
    }
    const char *reference_dir=reference_mode?argv[4]:NULL,*dump_dir=reference_mode?argv[5]:NULL;
    if(reference_mode)argc=4;
    int smoke_mode=argc==4&&!strcmp(argv[3],"--model-smoke");
    int io_mode=argc==5&&!strcmp(argv[3],"--request-io");
    const char *io_dir=io_mode?argv[4]:NULL;
    if(io_mode)argc=4;
    int pread_mode = reference_pread || (argc==4 && !strcmp(argv[3],"--request-pread"));
    int request_mode = reference_mode || smoke_mode || io_mode || pread_mode || (argc==4 && !strcmp(argv[3],"--request"));
    int model_layout = request_mode || (argc==4 && !strcmp(argv[3],"--model-layout"));
    if (model_layout) argc=3;
    assert(argc==3||argc==5||(argc==6&&!strcmp(argv[5],"--profile")));int profile=argc==6;
    int only_layer=argc>=5?atoi(argv[3]):-1,only_nt=argc>=5?atoi(argv[4]):-1;
    BnGGUFFile*f=bn_gguf_open_file(argv[1]);BnGPUBackend*g=bn_gpu_cuda_create();assert(f&&g&&g->moe_shared_bf16_resident);
    BnModel model={0}; const BnBackendBF16FlashFFNModel *native=NULL;
    if (model_layout) {
        assert(!bn_model_load_with_cpu_preparation(&model,f,8192,0,0,0));
        if (request_mode) {
            if(pread_mode) {
                assert(f->owned_maps&&!bn_model_set_moe_shard_files(&model,f->owned_maps,f->n_shards));
                bn_model_set_moe_mmap_base(&model,NULL);
                bn_model_set_moe_mmap_shards(&model,NULL,0);
                size_t bytes[3]={1126400,1126400,1740800};
                if(reference_pread) {
                    memset(bytes,0,sizeof(bytes));
                    for(int layer=0;layer<model.config.n_layers;layer++) {
                        const BnMoEExpertMap *map=&model.weights.layers[layer].moe.expert_map;
                        const size_t actual[]={map->expert_gate_bytes,map->expert_up_bytes,map->expert_down_bytes};
                        assert(!map->repacked);
                        for(int role=0;role<3;role++)if(actual[role]>bytes[role])bytes[role]=actual[role];
                    }
                }
                const size_t budget=reference_pread?reference_cache_budget:64u*1024*1024;
                const size_t entry=bytes[0]+bytes[1]+bytes[2];
                void *cache=bn_moe_cache_create(budget,bytes[0],bytes[1],bytes[2]);assert(cache);
                bn_model_set_moe_cache(&model,cache);
                assert(bn_moe_io_has_pread(bn_model_moe_io_const(&model)) &&
                       !bn_moe_io_has_mmap(bn_model_moe_io_const(&model)));
                assert(entry && (size_t)bn_moe_cache_capacity_internal(cache)==budget/entry);
                printf("FLASH_PREAD_CACHE payload_budget=%zu payload_bytes=%zu slots=%d gate=%zu up=%zu down=%zu\n",
                    budget,(budget/entry)*entry,bn_moe_cache_capacity_internal(cache),bytes[0],bytes[1],bytes[2]);
            } else if (f->n_shards) bn_model_set_moe_mmap_shards(&model,(const uint8_t **)f->shard_raws,f->n_shards);
            else bn_model_set_moe_mmap_base(&model,f->raw);
        }
        BnBackendModel *owner=bn_model_backend(&model);size_t budget=(size_t)12*1024*1024*1024,host=64u*1024*1024;
        if(reference_mode) {
            if(gpu_cache_budget) {
                void *(*create)(void*,const BnGPUExpertCacheConfig*)=g->expert_cache_create;
                g->expert_cache_create=whole_reject_cache;
                assert(bn_backend_layout_prepare_bf16_flash_model_with_cache(owner,g,&model.config,&model.weights,budget,host,gpu_cache_budget)==-1);
                assert(!bn_model_gpu(&model)&&!bn_backend_layout_bf16_flash_model(owner)&&!bn_backend_layout_bf16_hc_model(owner));
                g->expert_cache_create=create;
                puts("FLASH_CACHE_PREPARE_ROLLBACK exact");
            }
            assert(!bn_backend_layout_prepare_bf16_flash_model_with_cache(owner,g,&model.config,&model.weights,budget,host,gpu_cache_budget));
            if(preload) {
                BnTransformerBF16FlashPreloadInfo info;
                assert(!bn_transformer_bf16_flash_preload_experts(&model,4u*1024*1024,&info));
                assert(info.cache.resident_experts==24576&&info.cache.resident_bytes==77017907200ull);
                assert(info.cache.upload_bytes==info.cache.resident_bytes&&info.cache.host_staging_bytes==3993600);
                if(reference_pread)assert(info.source.io_bytes==77017907200ull&&info.source.io_count==73728);
                printf("FLASH_FULL_PRELOAD exact experts=%zu bytes=%zu staging=%zu io_bytes=%zu\n",
                    info.cache.resident_experts,info.cache.resident_bytes,info.cache.host_staging_bytes,info.source.io_bytes);
            }
        } else {
        assert(!bn_backend_layout_prepare_bf16_hc_model(owner,g,&model.config,&model.weights,budget));
        assert(!bn_backend_layout_prepare_bf16_ple_model(owner,g,&model.config,&model.weights,budget,host));
        assert(!bn_backend_layout_prepare_bf16_flash_gdn_model(owner,g,&model.config,&model.weights,budget,host));
        assert(!bn_backend_layout_prepare_bf16_flash_qsa_model(owner,g,&model.config,&model.weights,budget,host));
        assert(!bn_backend_layout_prepare_bf16_flash_ffn_model(owner,g,&model.config,&model.weights,budget));
        }
        native=bn_backend_layout_bf16_flash_ffn_model(owner);assert(native&&!bn_backend_layout_bf16(owner));
        assert(native->n_layers==48&&native->bytes==376750080u&&native->peak_bytes==376765440u);
        if(io_mode||smoke_mode||reference_mode) {
            if(!reference_mode)assert(!bn_backend_layout_prepare_bf16_flash_io_model(owner,g,&model.config,&model.weights,budget));
            const BnBackendBF16FlashIOModel *io=bn_backend_layout_bf16_flash_io_model(owner);assert(io);
            assert(io->bytes==1350860800u&&io->model_bytes==UINT64_C(8483329152)&&io->model_peak_bytes==io->model_bytes);
            printf("FLASH_IO_MODEL bytes%zu aggregate%zu aggregate_peak%zu\n",io->bytes,io->model_bytes,io->model_peak_bytes);fflush(stdout);
        }
        printf("FLASH_FFN_MODEL layers%d bytes%zu peak%zu aggregate%zu aggregate_peak%zu\n",native->n_layers,native->bytes,native->peak_bytes,native->model_bytes,native->model_peak_bytes);fflush(stdout);
    }
    if(reference_mode){model_reference(&model,g,reference_dir,dump_dir);bn_model_free(&model);bn_gpu_cuda_destroy(g);bn_gguf_free(f);return 0;}
    if(smoke_mode){model_smoke(&model,g);bn_model_free(&model);bn_gpu_cuda_destroy(g);bn_gguf_free(f);return 0;}
    const int layers[]={3,2,47},tokens[]={1,5,17,63,64,65,128,129,257};size_t bfwords=0,words32=0;int cases=0;
    for(int li=0;li<3;li++) {
        int layer=layers[li];if(only_layer>=0&&layer!=only_layer)continue;
        if(io_mode) {
            if(li)continue;
            const int nts[]={1,5,17,129,257};
            for(int ti=0;ti<5;ti++){io_case(&model,g,argv[2],io_dir,nts[ti],&bfwords,&words32);cases++;}
            continue;
        }
        if(request_mode) {
            for(int ti=0;ti<9;ti++){
                BnTransformerBF16FlashFFNInfo fi;
                request_case(&model,g,argv[2],layer,tokens[ti],1,&bfwords,&words32,&fi);cases++;
                if(pread_mode)assert(!fi.source.gpu_host_cache_hits&&fi.source.gpu_host_cache_misses==fi.banks.submitted_experts);
            }
            continue;
        }
        Weights w;weights_load(g,f,layer,&w,native);
        for(int ti=0;ti<9;ti++) {
            int nt=tokens[ti];if(only_nt>=0&&nt!=only_nt)continue;char prefix[128];snprintf(prefix,sizeof(prefix),"layer%d-%d",layer,nt);
            BnBackendSession*s=bn_backend_session_create();assert(s);void*r=bn_gpu_backend_resident_request(g,s);Frame v;prepare(g,r,nt,&w,&v);
            void*hidden=load(argv[2],prefix,"hc-hidden",(size_t)nt*10240*2);assert(!g->resident_write(r,v.hidden,hidden));free(hidden);
            if(profile){execute(g,r,f,&w,&v,nt);assert(!g->resident_synchronize(r));}
#ifdef BN_CAPTURE_CUDA_PROFILER
            if(profile)assert(cudaProfilerStart()==cudaSuccess);
#endif
            execute(g,r,f,&w,&v,nt);assert(!g->resident_synchronize(r));
#ifdef BN_CAPTURE_CUDA_PROFILER
            if(profile)assert(cudaProfilerStop()==cudaSuccess);
#endif
            (void)profile;
            BnGPUResidentView bv[]={v.hidden,v.input,v.router.logits,v.combine.logits,v.shared_ffn.gate,v.shared_ffn.out,v.combine.sigmoid,v.combine.shared,v.combine.out,v.combined,v.reduce.weighted,v.reduce.out};
            const char*names[]={"hc-hidden","input","logits","shared-logits","shared-activation","shared-down","shared-sigmoid","shared","block","hc-combined","routed-weighted","routed"};
            for(int j=0;j<12;j++)bfwords+=compare(g,r,bv[j],argv[2],prefix,names[j],2);
            words32+=compare(g,r,v.router.weights,argv[2],prefix,"weights",4);words32+=compare(g,r,v.router.indices,argv[2],prefix,"ids",4);words32+=compare(g,r,v.router.source_rows,argv[2],prefix,"source",4);
            int32_t*slots=malloc((size_t)nt*10*4);assert(slots&&!g->resident_read(r,v.group.slots,slots));
            bfwords+=mapped_compare(g,r,v.expert_gate,slots,argv[2],prefix,"routed-activation",640,640,0);
            bfwords+=mapped_compare(g,r,v.expert_up,slots,argv[2],prefix,"routed-gateup",640,1280,640);
            bfwords+=mapped_compare(g,r,v.reduce.grouped,slots,argv[2],prefix,"routed-down",2560,2560,0);free(slots);
            printf("%s complete FFN exact\n",prefix);fflush(stdout);cases++;bn_backend_session_free(s);
        }
        for(int i=0;i<(native?ROUTER:N_WEIGHTS);i++)g->buffer_destroy(g->ctx,w.handle[i]);
    }
    if(request_mode&&!io_mode)request_guards(&model,g,argv[2]);
    if(pread_mode) {
        bn_moe_cache_free(bn_model_moe_cache(&model));
        void *fresh=bn_moe_cache_create(64u*1024*1024,1126400,1126400,1740800);assert(fresh);bn_model_set_moe_cache(&model,fresh);
        size_t a=0,b=0;BnTransformerBF16FlashFFNInfo cold,warm,borrow;
        request_case(&model,g,argv[2],3,1,0,&a,&b,&cold);
        request_case(&model,g,argv[2],3,1,0,&a,&b,&warm);
        request_case(&model,g,argv[2],3,1,1,&a,&b,&borrow);
        assert(cold.source.gpu_host_cache_misses==10&&!cold.source.gpu_host_cache_hits);
        assert(warm.source.gpu_host_cache_hits==10&&!warm.source.gpu_host_cache_misses&&!warm.source.io_bytes);
        assert(borrow.source.gpu_host_cache_hits==10&&!borrow.source.io_bytes);
        assert(cold.banks.upload_bytes==warm.banks.upload_bytes&&warm.banks.upload_bytes==borrow.banks.upload_bytes);
        printf("FFN_RAM_POLICY cold_misses%zu warm_hits%zu prefill_borrow_hits%zu cold_io_bytes%zu warm_io_bytes%zu gpu_upload_each%zu\n",cold.source.gpu_host_cache_misses,warm.source.gpu_host_cache_hits,borrow.source.gpu_host_cache_hits,cold.source.io_bytes,warm.source.io_bytes,warm.banks.upload_bytes);
        bn_moe_cache_free(bn_model_moe_cache(&model));bn_model_set_moe_cache(&model,NULL);
    }
    if(model_layout)bn_model_free(&model);
    bn_gpu_cuda_destroy(g);bn_gguf_free(f);assert(cases==(io_mode?5:(only_layer>=0?1:27)));
    printf("FLASH_FULL_FFN cases%d bf16_words%zu words32=%zu differences0 request%d\n",cases,bfwords,words32,request_mode);return 0;
}
