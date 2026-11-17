#include "transformer_internal.h"
#include "gpu_moe_bridge.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "backend_session.h"
#include "session_internal.h"
#include "model_internal.h"
#include "moe.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Native request resources stay in the existing session container. The
 * destructor needs only its own allocation and backend request destroy hook. */
enum {
    TR_X,TR_LOGITS,TR_IDS,TR_WEIGHTS,TR_AHI,TR_ALO,TR_ASCALE,TR_PARTIALS,
    TR_KEYS,TR_VALUES,TR_SORTED_KEYS,TR_ORDER,TR_INVERSE,TR_SORTED_WEIGHTS,
    TR_COUNTS,TR_STARTS,TR_BASE,TR_KIND,TR_SLOTS,TR_ROUTED,
    TR_STATE_G,TR_STATE_U,TR_INTER_G,TR_INTER_U,
    TR_GROUP_X,TR_GROUP_G,TR_GROUP_U,TR_DECODED,
    TR_SHARED_G,TR_SHARED_U,TR_SHARED_ACT,TR_SHARED,TR_OUT,
    TR_D_HAD_G,TR_D_HAD_U,TR_D_GATE,TR_D_UP,TR_D_ACT,TR_D_DOWN,
    TR_D_COUNTERS,TR_D_TABLES,TR_D_IDS64,TR_COUNT
};
typedef struct TrellisKV TrellisKV;
typedef struct TrellisSWA TrellisSWA;
typedef struct TrellisIndex TrellisIndex;
typedef struct TrellisAttention TrellisAttention;
typedef struct TrellisProducer TrellisProducer;
typedef struct TrellisEngram TrellisEngram;
typedef struct TrellisHC TrellisHC;
typedef struct TrellisForward TrellisForward;
typedef struct {
    const BnModel *model;
    const BnModelNativeAnatomy *anatomy;
    const BnMoEExpertReader *reader;
    BnGPUMoETrellisStore *store;
    BnGPUBackend *gpu;
    void *request;
    void (*destroy)(void*);
    TrellisKV *kv;
    TrellisSWA *swa;
    TrellisIndex *index;
    TrellisAttention *attention;
    TrellisProducer *producer;
    TrellisEngram *engram;
    TrellisHC *hc;
    TrellisForward *forward;
    int block_layer,block_phase,block_admit;
    uint64_t block_epoch,block_index_generation;
    BnBackendRowStage rows;
    BnTransformerTrellisRowsOptions rows_options;
    int row_layers[4];
    uint64_t generation;
    BnTransformerTrellisOptions options;
    BnTransformerTrellisWorkspace workspace;
    BnGPUResidentView buffers[TR_COUNT];
    int *metadata;
    BnMoEEXL3Plan plan;
    int *counts,*eligible;
    int32_t decode_ids[256];
    uint16_t decode_weights[256];
} TrellisRequest;
static const char native_trellis_session_owner=0;
static int trellis_add_bytes(size_t *total,size_t count,size_t width) {
    if(!width||count>SIZE_MAX/width||*total>SIZE_MAX-count*width)return -1;
    *total+=count*width;return 0;
}
static int trellis_request_shape(const BnModel *model,const BnTransformerTrellisOptions *o,
    BnTransformerTrellisWorkspace *out,size_t *counts,int *types) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(model);
    BnGPUBackend *g=bn_model_gpu(model);
    if(!a||!bn_model_native_numerics(model)||!g||!o||!out||
       o->max_tokens<1||o->max_tokens>16384||o->fused_window<1||o->fused_window>32||
       (o->f16acc_enabled!=0&&o->f16acc_enabled!=1)||
       a->experts.format!=BN_QUANT_FORMAT_EXL3_MUL1_3||
       bn_gpu_native_moe_shape(o->max_tokens,a->experts.active_experts,a->experts.experts)||
       a->experts.dim<128||a->experts.dim>32768||a->experts.dim%128||
       a->experts.hidden<128||a->experts.hidden>65536||a->experts.hidden%128||
       a->experts.dim>INT_MAX/a->experts.hidden||
       (size_t)a->experts.dim>SIZE_MAX/(size_t)a->experts.hidden/4||
       !g->resident_request_create||!g->resident_request_destroy||!g->resident_alloc||
       !g->resident_write||!g->resident_read||!g->resident_zero||!g->resident_synchronize||
       !g->router_sqrtsp_resident||!g->trellis_decode_resident||
       !g->linear_f16_workspace_size||!g->linear_f16_reserve||
       !g->ffn_f16_resident||!g->add_f32_resident)return -1;
    int prefill=o->max_tokens>8;
    if(prefill&&(!g->native_moe_index_workspace_size||!g->native_moe_index_reserve||
       !g->native_moe_index_resident||!g->native_moe_group_resident||!g->native_moe_reduce_resident||
       !g->trellis_fused_workspace_size||!g->trellis_fused_reserve||!g->trellis_fused_launch_info||
       !g->trellis_fused_resident||!g->trellis_folded_workspace_size||!g->trellis_folded_reserve||
       !g->trellis_folded_ffn_resident))return -1;
    size_t R=(size_t)o->max_tokens,D=(size_t)a->experts.dim,H=(size_t)a->experts.hidden;
    size_t E=(size_t)a->experts.experts,A=R*(size_t)a->experts.active_experts;
    size_t cap=(256u*1024u*1024u)/(D*H*4);
    if(cap<1)cap=1;
    if(cap>16)cap=16;
    if(cap>E)cap=E;
    size_t group_rows=R>256?R*cap:0;
    if(group_rows>16384)group_rows=16384;
    size_t slots=prefill?A+A/10+1:0,partial=0,ascale=0,activation=0;
    if(slots>SIZE_MAX/D/4)return -1;
    for(int r=1;r<=o->max_tokens;r++) {
        BnGPURouterWorkspace w;
        if(bn_gpu_router_workspace(r,(int)D,(int)E,&w))return -1;
        if(w.part_count>partial)partial=w.part_count;
        if(w.a_count>activation)activation=w.a_count;
        if(w.scale_count>ascale)ascale=w.scale_count;
    }
    BnGPUFusedEXL3Workspace fused={0};
    BnGPUDecodeEXL3Workspace decode;
    if(bn_gpu_decode_exl3_workspace(o->max_tokens<8?o->max_tokens:8,a->experts.active_experts,(int)E,(int)D,(int)H,&decode))return -1;
    int groups=o->fused_window;
    if(groups>(int)E)groups=(int)E;
    if(prefill&&bn_gpu_fused_exl3_workspace(groups,256,(int)D,(int)H,&fused))return -1;
    size_t n[TR_COUNT]={R*D,R*E,A,A,activation,activation,ascale,partial,
        A,A,A,A,A,A,E+1,E+1,E,E,slots*D,R*D,
        fused.state_count,fused.state_count,fused.intermediate_count,fused.intermediate_count,
        group_rows*D,group_rows*H,group_rows*H,group_rows?cap*D*H:0,
        R*H,R*H,R*H,R*D,R*D,
        decode.had_count,decode.had_count,decode.intermediate_count,decode.intermediate_count,
        decode.intermediate_count,decode.down_count,decode.counter_count,decode.table_bytes,decode.indices64_bytes};
    if(!prefill)for(int i=TR_KEYS;i<=TR_SLOTS;i++)n[i]=0;
    int t[TR_COUNT]={BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,
        BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,
        BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,
        BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_F32,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F16,
        BN_BACKEND_SCALAR_F16,BN_BACKEND_SCALAR_F32,BN_BACKEND_SCALAR_I32,BN_BACKEND_SCALAR_BYTES,BN_BACKEND_SCALAR_BYTES};
    size_t bytes=0;
    for(int i=0;i<TR_COUNT;i++)if(trellis_add_bytes(&bytes,n[i],bn_backend_quant_scalar_bytes((BnBackendScalarKind)t[i])))return -1;
    size_t reserve[]={g->linear_f16_workspace_size(g->ctx),0,0,0};
    if(prefill){reserve[1]=g->trellis_fused_workspace_size(g->ctx);
        reserve[2]=g->trellis_folded_workspace_size(g->ctx);reserve[3]=g->native_moe_index_workspace_size(g->ctx,(int)A);}
    for(int i=0;i<(prefill?4:1);i++)if(!reserve[i]||trellis_add_bytes(&bytes,reserve[i],1))return -1;
    size_t host=sizeof(TrellisRequest);
    if(trellis_add_bytes(&host,6*E+2,sizeof(int))||(prefill&&trellis_add_bytes(&host,E,sizeof(BnMoEEXL3Group))))return -1;
    BnTransformerTrellisWorkspace w={bytes,host,prefill?(int)cap:0,(int)slots,(int)group_rows};
    if(counts)memcpy(counts,n,sizeof(n));
    if(types)for(int i=0;i<TR_COUNT;i++)types[i]=bn_backend_quant_scalar_type((BnBackendScalarKind)t[i]);
    *out=w;return 0;
}
int bn_transformer_trellis_session_plan(const BnModel *model,
    const BnTransformerTrellisOptions *o,BnTransformerTrellisWorkspace *out) {
    if(!out)return -1;
    return trellis_request_shape(model,o,out,NULL,NULL);
}
static void trellis_request_free(void *context) {
    TrellisRequest *r=context;
    if(!r)return;
    if(r->request)r->destroy(r->request);
    bn_backend_session_row_stage_free(&r->rows);
    free(r->plan.groups);free(r->metadata);free(r->kv);free(r->swa);free(r->index);free(r->attention);free(r->producer);free(r->engram);free(r->hc);free(r->forward);free(r);
}
static TrellisRequest *trellis_request_get(const BnModel *model,BnSession *session) {
    if(!session||!session->backend_only||!bn_session_matches_model(session,model))return NULL;
    TrellisRequest *r=bn_backend_session_gpu_resources(bn_session_backend(session),&native_trellis_session_owner);
    return r&&r->request&&r->model==model&&r->anatomy==bn_model_native_anatomy(model)&&
        r->reader==bn_model_trellis_expert_reader(model)&&r->gpu==bn_model_gpu(model)&&
        r->store==bn_gpu_moe_trellis_model_store(model)?r:NULL;
}
#include "trellis_kv.h"
#include "trellis_swa.h"
#include "trellis_index.h"
#include "trellis_attention.h"
#include "trellis_producer.h"
#include "trellis_rows.h"
#include "trellis_engram.h"
#include "trellis_hc.h"
#include "trellis_block.h"
#include "trellis_forward.h"
static int trellis_prepare_session(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *o,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,
    const BnTransformerTrellisProducerOptions *producer,const BnTransformerTrellisRowsOptions *rows,
    const BnTransformerTrellisEngramOptions *engram,const BnTransformerTrellisHCOptions *hc,
    const BnTransformerTrellisForwardOptions *forward) {
    BnTransformerTrellisWorkspace w;size_t n[TR_COUNT];int t[TR_COUNT];
    if(!session||!session->backend_only||!bn_session_matches_model(session,model)||
       trellis_request_shape(model,o,&w,n,t)||o->device_budget<w.device_bytes||o->host_budget<w.host_bytes||
       !bn_gpu_moe_trellis_model_store(model))return -1;
    BnTransformerTrellisKVWorkspace kw;
    if(kv&&(bn_transformer_trellis_kv_session_plan(model,kv,&kw)||
       kv->device_budget<kw.device_bytes||kv->host_budget<kw.host_bytes))return -1;
    BnTransformerTrellisSWAWorkspace sw;
    if(swa&&(!kv||bn_transformer_trellis_swa_session_plan(model,&sw)||
       swa->device_budget<sw.device_bytes||swa->host_budget<sw.host_bytes))return -1;
    BnTransformerTrellisIndexWorkspace iw;
    if(index&&(!kv||bn_transformer_trellis_index_session_plan(model,o,kv,&iw)||
       index->device_budget<iw.device_bytes||index->host_budget<iw.host_bytes))return -1;
    BnTransformerTrellisAttentionWorkspace aw;
    if(attention&&(!kv||!swa||!index||bn_transformer_trellis_attention_session_plan(model,o,kv,&aw)||
       attention->device_budget<aw.device_bytes||attention->host_budget<aw.host_bytes))return -1;
    BnTransformerTrellisProducerWorkspace pw;
    if(producer&&(!attention||bn_transformer_trellis_producer_session_plan(model,o,kv,&pw)||
       producer->device_budget<pw.device_bytes||producer->host_budget<pw.host_bytes))return -1;
    BnTransformerTrellisRowsWorkspace rw;
    if(rows&&(bn_transformer_trellis_rows_session_plan(model,rows,&rw)||rows->host_budget<rw.host_bytes||
        rows->pinned_budget<rw.pinned_bytes||rows->device_budget<rw.device_bytes))return -1;
    BnTransformerTrellisEngramWorkspace ew;BnSessionNativeHistoryInfo eh;
    if(engram&&(!rows||bn_transformer_trellis_engram_session_plan(model,o,rows,&ew)||
       engram->device_budget<ew.device_bytes||engram->host_budget<ew.host_bytes||
       bn_session_native_history_info(session,model,&eh)||eh.max_tokens!=ew.max_tokens||eh.host_bytes!=ew.history_bytes))return -1;
    BnTransformerTrellisHCWorkspace hw;
    if(hc&&(!engram||bn_transformer_trellis_hc_session_plan(model,o,&hw)||
       hc->device_budget<hw.device_bytes||hc->host_budget<hw.host_bytes))return -1;
    BnTransformerTrellisForwardWorkspace vw;
    if(forward&&(!hc||!producer||bn_transformer_trellis_forward_session_plan(model,o,forward,&vw)||
       forward->device_budget<vw.device_bytes||forward->host_budget<vw.host_bytes))return -1;
    BnBackendSession *backend=bn_session_backend(session);
    TrellisRequest *old=trellis_request_get(model,session);
    if(old)return old->options.max_tokens==o->max_tokens&&old->options.fused_window==o->fused_window&&
        old->options.f16acc_enabled==o->f16acc_enabled&&
        (!kv||(old->kv&&old->kv->maxseq==kv->maxseq))&&(!swa||old->swa)&&(!index||old->index)&&(!attention||old->attention)&&(!producer||old->producer)&&
        (!rows||(old->rows.state&&old->rows_options.max_rows==rows->max_rows&&old->rows_options.banks==rows->banks))&&(!engram||old->engram)&&(!hc||old->hc)&&(!forward||(old->forward&&old->forward->all_logits==forward->all_logits))?0:-1;
    if(engram&&(session->pos||eh.committed||eh.staged))return -1;
    if(bn_backend_session_has_gpu_resources(backend)||bn_backend_session_has_execution_resources(backend))return -1;
    if(session->backend_state->native_request_generation==UINT64_MAX)return -1;
    TrellisRequest *r=calloc(1,sizeof(*r));if(!r)return -1;
    r->generation=session->backend_state->native_request_generation+1;
    r->model=model;r->anatomy=bn_model_native_anatomy(model);r->reader=bn_model_trellis_expert_reader(model);
    r->store=bn_gpu_moe_trellis_model_store(model);r->gpu=bn_model_gpu(model);r->destroy=r->gpu->resident_request_destroy;
    r->options=*o;r->workspace=w;size_t E=(size_t)r->anatomy->experts.experts;
    r->metadata=calloc(6*E+2,sizeof(int));
    if(o->max_tokens>8)r->plan.groups=calloc(E,sizeof(*r->plan.groups));
    if(!r->metadata||(o->max_tokens>8&&!r->plan.groups))goto fail;
    r->counts=r->metadata;r->plan.order=r->counts+E+1;r->plan.starts=r->plan.order+E;
    r->plan.slot_base=r->plan.starts+E+1;r->plan.slot_kind=r->plan.slot_base+E;r->eligible=r->plan.slot_kind+E;
    r->plan.expert_capacity=E;r->plan.group_capacity=E;
    r->request=r->gpu->resident_request_create(r->gpu->ctx);if(!r->request)goto fail;
    if(r->gpu->linear_f16_reserve(r->gpu->ctx,r->request))goto fail;
    if(o->max_tokens>8&&(r->gpu->trellis_fused_reserve(r->gpu->ctx,r->request)||
       r->gpu->trellis_folded_reserve(r->gpu->ctx,r->request,o->f16acc_enabled)||
       r->gpu->native_moe_index_reserve(r->gpu->ctx,r->request,o->max_tokens*r->anatomy->experts.active_experts)))goto fail;
    for(int tier=0;o->max_tokens>8&&tier<3;tier++) {
        BnGPUFusedEXL3LaunchInfo info;int tile=tier==0?64:tier==1?32:16;
        int tile_n=tile==16&&!(r->anatomy->experts.dim%256)&&!(r->anatomy->experts.hidden%256)?256:128;
        if(r->gpu->trellis_fused_launch_info(r->gpu->ctx,r->request,tile,tile_n,&info)||
           info.multiprocessors<8||info.blocks_per_sm!=1)goto fail;
    }
    for(int i=0;i<TR_COUNT;i++)if(n[i]&&r->gpu->resident_alloc(r->request,t[i],n[i],&r->buffers[i]))goto fail;
    if(r->gpu->resident_zero(r->request,r->buffers[TR_D_COUNTERS]))goto fail;
    if(kv&&trellis_kv_setup(r,kv,&kw))goto fail;
    if(swa&&trellis_swa_setup(r,&sw))goto fail;
    if(index&&trellis_index_setup(r,&iw))goto fail;
    if(attention&&trellis_attention_setup(r,&aw))goto fail;
    if(producer&&trellis_producer_setup(r,&pw))goto fail;
    if(rows&&trellis_rows_setup(r,rows))goto fail;
    if(engram&&trellis_engram_setup(r,session,&ew))goto fail;
    if(hc&&trellis_hc_setup(r,&hw))goto fail;
    if(forward&&trellis_forward_setup(r,forward,&vw))goto fail;
    if(bn_backend_session_attach_gpu_resources(backend,&native_trellis_session_owner,r,trellis_request_free))goto fail;
    session->backend_state->native_request_generation=r->generation;
    return 0;
fail:trellis_request_free(r);return -1;
}
int bn_transformer_trellis_prepare_session(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *o) {return trellis_prepare_session(model,session,o,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);}
int bn_transformer_trellis_prepare_session_kv(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *o,const BnTransformerTrellisKVOptions *kv) {
    return kv?trellis_prepare_session(model,session,o,kv,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL):-1;
}
int bn_transformer_trellis_prepare_session_attention(const BnModel *model,BnSession *session,
    const BnTransformerTrellisOptions *o,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa) {
    return kv&&swa?trellis_prepare_session(model,session,o,kv,swa,NULL,NULL,NULL,NULL,NULL,NULL,NULL):-1;
}
int bn_transformer_trellis_prepare_session_index(const BnModel*model,BnSession*session,
    const BnTransformerTrellisOptions*o,const BnTransformerTrellisKVOptions*kv,
    const BnTransformerTrellisSWAOptions*swa,const BnTransformerTrellisIndexOptions*index) {
    return kv&&index?trellis_prepare_session(model,session,o,kv,swa,index,NULL,NULL,NULL,NULL,NULL,NULL):-1;
}
int bn_transformer_trellis_prepare_session_projected_attention(const BnModel*model,BnSession*session,
    const BnTransformerTrellisOptions*o,const BnTransformerTrellisKVOptions*kv,
    const BnTransformerTrellisSWAOptions*swa,const BnTransformerTrellisIndexOptions*index,
    const BnTransformerTrellisAttentionOptions*attention) {
    return kv&&swa&&index&&attention?trellis_prepare_session(model,session,o,kv,swa,index,attention,NULL,NULL,NULL,NULL,NULL):-1;
}
int bn_transformer_trellis_prepare_session_producer(const BnModel*m,BnSession*s,
    const BnTransformerTrellisOptions*o,const BnTransformerTrellisKVOptions*kv,
    const BnTransformerTrellisSWAOptions*swa,const BnTransformerTrellisIndexOptions*index,
    const BnTransformerTrellisAttentionOptions*attention,const BnTransformerTrellisProducerOptions*producer) {
    return kv&&swa&&index&&attention&&producer?trellis_prepare_session(m,s,o,kv,swa,index,attention,producer,NULL,NULL,NULL,NULL):-1;
}
static BnGPUResidentView trellis_slice(TrellisRequest *r,int slot,size_t count) {
    return bn_gpu_resident_view_slice(r->buffers[slot],0,count);
}
int bn_transformer_trellis_session_input(const BnModel *model,BnSession *session,
    int tokens,void **request,BnGPUResidentView *input) {
    TrellisRequest *r=trellis_request_get(model,session);
    if(!r||!request||!input||tokens<1||tokens>r->options.max_tokens)return -1;
    BnGPUResidentView x=trellis_slice(r,TR_X,(size_t)tokens*r->anatomy->experts.dim);
    if(!x.request)return -1;
    *request=r->request;*input=x;return 0;
}
/* Drain before releasing even partly acquired batches. A failed completion
 * poisons this request: backend destruction precedes lease reclamation. */
static int trellis_finish_leases(TrellisRequest *r,BnGPUMoETrellisLease *leases,int count) {
    int status=r->gpu->resident_synchronize(r->request);
    if(status){r->destroy(r->request);r->request=NULL;}
    for(int b=0;b<count;b++)if(bn_gpu_moe_trellis_release(r->store,&leases[b]))status=-1;
    return status;
}
/* Only routing metadata crosses to the host. Bind every active expert before
 * either decode stage; a prefill window would change the source reduction. */
static int trellis_decode_plan(TrellisRequest *r,int tokens,
    const BnGPURouterResidentPlan *router,BnGPUResidentView routed,float limit,
    BnGPUDecodeEXL3ResidentPlan *out) {
    int D=r->anatomy->experts.dim,H=r->anatomy->experts.hidden;
    int E=r->anatomy->experts.experts,K=r->anatomy->experts.active_experts;
    if(tokens<1||tokens>8)return -1;
    int A=tokens*K,active=0;
    *out=(BnGPUDecodeEXL3ResidentPlan){0};
    if(A>256||r->gpu->resident_read(r->request,router->selected,r->decode_ids)||
       r->gpu->resident_read(r->request,router->weights,r->decode_weights))return -1;
    for(int a=0;a<A;a++) {
        int e=r->decode_ids[a];float weight=bn_fp16_to_fp32(r->decode_weights[a]);
        if(e<0||e>=E||!isfinite(weight))return -1;
        if(weight==0)continue;
        int b=0;while(b<active&&r->eligible[b]!=e)b++;
        if(b==active){if(active==BN_GPU_DECODE_EXL3_MAX_EXPERTS)return -1;r->eligible[active++]=e;}
    }
    if(!active)return 0;
    BnGPUDecodeEXL3Workspace ws;
    if(bn_gpu_decode_exl3_workspace(tokens,K,E,D,H,&ws))return -1;
    BnGPUDecodeEXL3ResidentPlan p={0};p.format=r->anatomy->experts.format;
    p.tokens=tokens;p.topk=K;p.experts=E;p.dim=D;p.hidden=H;p.resident_count=active;
    p.codebook_marker=UINT32_C(0x83dcd12d);p.act_limit=limit;
    p.x=router->x;p.indices=router->selected;p.weights=router->weights;p.out=routed;
    p.had_gate=r->buffers[TR_D_HAD_G];p.had_up=r->buffers[TR_D_HAD_U];
    p.gate=r->buffers[TR_D_GATE];p.up=r->buffers[TR_D_UP];p.act=r->buffers[TR_D_ACT];
    p.down=r->buffers[TR_D_DOWN];p.counters=r->buffers[TR_D_COUNTERS];p.tables=r->buffers[TR_D_TABLES];
    p.indices64=trellis_slice(r,TR_D_IDS64,ws.indices64_bytes);
    for(int b=0;b<active;b++)p.expert_ids[b]=r->eligible[b];
    *out=p;return 0;
}
static int trellis_decode(TrellisRequest *r,int layer,int tokens,int admit,
    const BnGPURouterResidentPlan *router,BnGPUResidentView routed,float limit,
    BnGPUMoETrellisLease *leases,int *acquired) {
    BnGPUDecodeEXL3ResidentPlan p;int status=-1;*acquired=0;
    if(trellis_decode_plan(r,tokens,router,routed,limit,&p))goto fail;
    if(!p.resident_count) {
        if(r->gpu->resident_zero(r->request,routed))goto fail;
        return 0;
    }
    status=0;
    for(int b=0;b<p.resident_count;b++) {
        status=bn_gpu_moe_trellis_acquire(r->store,layer,p.expert_ids[b],admit,leases+b);
        if(status)break;
        (*acquired)++;p.weight[0][b]=leases[b].buffers.gate;
        p.weight[1][b]=leases[b].buffers.up;p.weight[2][b]=leases[b].buffers.down;
    }
    if(!status)status=r->gpu->trellis_decode_resident(r->gpu->ctx,r->request,&p);
    if(!status)return 0;
fail:
    if(trellis_finish_leases(r,leases,*acquired))status=-1;
    *acquired=0;return status;
}
int bn_transformer_trellis_run_learned(const BnModel *model,BnSession *session,
    int layer,int tokens,int admit,BnTransformerTrellisResult *result) {
    TrellisRequest *r=trellis_request_get(model,session);
    if(!r||!result||tokens<1||tokens>r->options.max_tokens||(admit!=0&&admit!=1)||
       layer<0||layer>=r->anatomy->experts.layers)return -1;
    BnGPUBackend *g=r->gpu;void *ctx=g->ctx,*request=r->request;
    int D=r->anatomy->experts.dim,H=r->anatomy->experts.hidden,E=r->anatomy->experts.experts;
    int K=r->anatomy->experts.active_experts;size_t A=(size_t)tokens*K,nx=(size_t)tokens*D,nh=(size_t)tokens*H;
    const BnModelNativeNumerics *numeric=bn_model_native_numerics(model);
    if(!numeric)return -1;
    void *router_weight=bn_backend_layout_native_router_weight(bn_model_backend(model),layer);
    const int roles[]={BN_NATIVE_SHARED_GATE,BN_NATIVE_SHARED_UP,BN_NATIVE_SHARED_DOWN};void *weights[3];
    for(int p=0;p<3;p++) {
        const BnBackendNativeF16Weight *w=bn_backend_layout_native_core_weight(bn_model_backend(model),layer,roles[p]);
        if(!w||w->rows!=(p==2?D:H)||w->cols!=(p==2?H:D))return -1;
        weights[p]=w->weight;
    }
    BnGPURouterWorkspace rw;if(!router_weight||bn_gpu_router_workspace(tokens,D,E,&rw))return -1;
    BnGPUResidentView x=trellis_slice(r,TR_X,nx);
    BnGPURouterResidentPlan router={tokens,D,E,K,numeric->routed_scale,router_weight,x,
        trellis_slice(r,TR_LOGITS,(size_t)tokens*E),trellis_slice(r,TR_IDS,A),trellis_slice(r,TR_WEIGHTS,A),
        trellis_slice(r,TR_AHI,rw.a_count),trellis_slice(r,TR_ALO,rw.a_count),
        trellis_slice(r,TR_ASCALE,rw.scale_count),trellis_slice(r,TR_PARTIALS,rw.part_count)};
    BnGPUMoETrellisLease decode_leases[BN_GPU_DECODE_EXL3_MAX_EXPERTS]={{0}};int decode_acquired=0;
    BnGPUNativeMoEIndexPlan ip={0};BnGPUResidentView base={0},kind={0},slots={0};
    BnGPUResidentView routed=trellis_slice(r,TR_ROUTED,nx);
    if(g->router_sqrtsp_resident(ctx,request,&router))goto fail;
    if(tokens<=8) {
        int status=trellis_decode(r,layer,tokens,admit,&router,routed,numeric->swiglu_limit,decode_leases,&decode_acquired);
        if(status){if(!r->request)return -1;return status;}
    } else {
        ip=(BnGPUNativeMoEIndexPlan){tokens,K,E,router.selected,router.weights,trellis_slice(r,TR_KEYS,A),
            trellis_slice(r,TR_VALUES,A),trellis_slice(r,TR_SORTED_KEYS,A),trellis_slice(r,TR_ORDER,A),
            trellis_slice(r,TR_INVERSE,A),trellis_slice(r,TR_SORTED_WEIGHTS,A),
            trellis_slice(r,TR_COUNTS,(size_t)E+1),trellis_slice(r,TR_STARTS,(size_t)E+1)};
        if(g->native_moe_index_resident(ctx,request,&ip)||
           g->resident_read(request,ip.counts,r->counts)||
           bn_moe_exl3_plan_groups(&r->plan,r->counts,E,tokens,K,D,H,256,256u*1024u*1024u))goto fail;
        if(r->plan.slot_count>r->workspace.slot_rows)return -1;
        base=trellis_slice(r,TR_BASE,E);kind=trellis_slice(r,TR_KIND,E);
        slots=trellis_slice(r,TR_SLOTS,(size_t)r->plan.slot_count*D);
        if(g->resident_write(request,base,r->plan.slot_base)||g->resident_write(request,kind,r->plan.slot_kind)||
           g->resident_zero(request,routed))goto fail;
        const int tiles[]={64,32,16},low[]={33,17,1},high[]={256,32,16};
        for(int tier=0;tier<3;tier++) {
            int active=0;for(int e=0;e<E;e++)if(r->counts[e]>=low[tier]&&r->counts[e]<=high[tier])r->eligible[active++]=e;
            if(!active)continue;
            BnGPUFusedEXL3LaunchInfo info;int tn=tiles[tier]==16&&!(D%256)&&!(H%256)?256:128;
            if(g->trellis_fused_launch_info(ctx,request,tiles[tier],tn,&info)||
               info.multiprocessors<8||info.blocks_per_sm!=1||info.max_groups<1)goto fail;
            int first_groups=info.max_groups;if(first_groups>active)first_groups=active;
            int width=info.multiprocessors/first_groups;if(width>32)width=32;
            for(int first=0;first<active;first+=r->options.fused_window) {
                int n=active-first;if(n>r->options.fused_window)n=r->options.fused_window;
                int groups=info.multiprocessors/width;if(groups>n)groups=n;
                BnGPUFusedEXL3Workspace ws;if(bn_gpu_fused_exl3_workspace(groups,256,D,H,&ws))goto fail;
                BnGPUMoETrellisLease leases[32]={{0}};int acquired=0,status=0;
                BnGPUFusedEXL3ResidentPlan p={0};p.format=BN_QUANT_FORMAT_EXL3_MUL1_3;p.codebook_marker=UINT32_C(0x83dcd12d);
                p.tokens=tokens;p.topk=K;p.experts=E;p.dim=D;p.hidden=H;p.capacity=256;p.groups=groups;p.group_width=width;
                p.tile_m=tiles[tier];p.tile_n=tn;p.resident_count=n;p.slot_rows=r->plan.slot_count;p.act_limit=numeric->swiglu_limit;
                p.x=x;p.counts=ip.counts;p.order=ip.order;p.weights=ip.sorted_weights;p.slot_base=base;p.slots=slots;
                p.state_g=trellis_slice(r,TR_STATE_G,ws.state_count);p.state_u=trellis_slice(r,TR_STATE_U,ws.state_count);
                p.intermediate_g=trellis_slice(r,TR_INTER_G,ws.intermediate_count);p.intermediate_u=trellis_slice(r,TR_INTER_U,ws.intermediate_count);
                for(int b=0;b<n;b++) {
                    int e=r->eligible[first+b];status=bn_gpu_moe_trellis_acquire(r->store,layer,e,admit,&leases[b]);if(status)break;
                    acquired++;p.expert_ids[b]=e;p.weight[0][b]=leases[b].buffers.gate;p.weight[1][b]=leases[b].buffers.up;p.weight[2][b]=leases[b].buffers.down;
                }
                if(!status)status=g->trellis_fused_resident(ctx,request,&p);
                if(trellis_finish_leases(r,leases,acquired))return -1;
                if(status)return status;
            }
        }
        for(int j=0;j<r->plan.group_count;j++) {
            BnMoEEXL3Group *group=&r->plan.groups[j];int B=group->count,M=group->rows;
            if(B>r->workspace.batch_cap||B*M>r->workspace.group_rows)goto fail;
            BnGPUMoETrellisLease leases[16]={{0}};int acquired=0,status=0;
            BnGPUNativeMoEGroupPlan gp={0};gp.tokens=tokens;gp.topk=K;gp.dim=D;gp.batch=B;gp.rows=M;
            gp.x=x;gp.order=ip.order;gp.out=trellis_slice(r,TR_GROUP_X,(size_t)B*M*D);
            for(int b=0;b<B;b++) {
                int e=group->experts[b];gp.starts[b]=r->plan.starts[e];gp.counts[b]=r->counts[e];
                status=bn_gpu_moe_trellis_acquire(r->store,layer,e,admit,&leases[b]);if(status)break;
                acquired++;
            }
            BnGPUFoldedEXL3FFNResidentPlan p={0};p.activation_profile=BN_GPU_TRELLIS_FFN_SILU_HALF;p.act_limit=numeric->swiglu_limit;
            BnGPUFoldedEXL3ResidentPlan *projections[]={&p.gate,&p.up,&p.down};
            for(int pr=0;pr<3;pr++) {
                BnGPUFoldedEXL3ResidentPlan *v=projections[pr];v->format=BN_QUANT_FORMAT_EXL3_MUL1_3;v->codebook_marker=UINT32_C(0x83dcd12d);
                v->batch=B;v->n_tokens=M;v->k=pr==2?H:D;v->n=pr==2?D:H;v->decoded=trellis_slice(r,TR_DECODED,(size_t)B*D*H);
                for(int b=0;b<acquired;b++)v->weights[b]=pr==0?leases[b].buffers.gate:pr==1?leases[b].buffers.up:leases[b].buffers.down;
            }
            p.gate.x=p.up.x=gp.out;p.gate.out=trellis_slice(r,TR_GROUP_G,(size_t)B*M*H);
            p.up.out=p.down.x=trellis_slice(r,TR_GROUP_U,(size_t)B*M*H);
            p.down.out=bn_gpu_resident_view_slice(slots,(size_t)group->slot*D,(size_t)B*M*D);
            if(!status)status=g->native_moe_group_resident(ctx,request,&gp);
            if(!status)status=g->trellis_folded_ffn_resident(ctx,request,&p);
            if(trellis_finish_leases(r,leases,acquired))return -1;
            if(status)return status;
        }
        BnGPUNativeMoEReducePlan reduce={tokens,K,D,E,r->plan.slot_count,ip.indices,ip.inverse,ip.starts,base,kind,ip.sorted_weights,slots,routed};
        if(g->native_moe_reduce_resident(ctx,request,&reduce))goto fail;
    }
    BnGPUResidentView sg=trellis_slice(r,TR_SHARED_G,nh),su=trellis_slice(r,TR_SHARED_U,nh),sa=trellis_slice(r,TR_SHARED_ACT,nh);
    BnGPUResidentView shared=trellis_slice(r,TR_SHARED,nx),out=trellis_slice(r,TR_OUT,nx);
    BnGPUF16FFNResidentPlan dense={{weights[0],D,H,tokens,x,sg},{weights[1],D,H,tokens,x,su},
        {weights[2],H,D,tokens,sa,shared},BN_GPU_F16_FFN_SILU_HALF,numeric->swiglu_limit};
    BnGPUAddF32ResidentPlan add={nx,routed,shared,out};
    if(g->ffn_f16_resident(ctx,request,&dense)||
       g->add_f32_resident(ctx,request,&add))goto fail;
    if(decode_acquired) {if(trellis_finish_leases(r,decode_leases,decode_acquired))return -1;}
    else if(g->resident_synchronize(request)){r->destroy(request);r->request=NULL;return -1;}
    BnTransformerTrellisResult value={request,tokens<=8?0:r->plan.slot_count,router.logits,router.selected,router.weights,ip.order,ip.inverse,
        ip.sorted_weights,ip.counts,ip.starts,base,kind,slots,routed,sg,su,sa,shared,out,
        tokens<=8?BN_TRANSFORMER_TRELLIS_DECODE:BN_TRANSFORMER_TRELLIS_PREFILL};
    *result=value;return 0;
fail:
    if(decode_acquired){(void)trellis_finish_leases(r,decode_leases,decode_acquired);return -1;}
    if(g->resident_synchronize(request)){r->destroy(request);r->request=NULL;}
    return -1;
}

int bn_transformer_trellis_prepare_session_rows(const BnModel *m,BnSession *s,
    const BnTransformerTrellisOptions *o,const BnTransformerTrellisRowsOptions *rows) {
    return rows?trellis_prepare_session(m,s,o,NULL,NULL,NULL,NULL,NULL,rows,NULL,NULL,NULL):-1;
}
int bn_transformer_trellis_prepare_session_producer_rows(const BnModel *m,BnSession *s,
    const BnTransformerTrellisOptions *o,const BnTransformerTrellisKVOptions *kv,
    const BnTransformerTrellisSWAOptions *swa,const BnTransformerTrellisIndexOptions *index,
    const BnTransformerTrellisAttentionOptions *attention,const BnTransformerTrellisProducerOptions *producer,
    const BnTransformerTrellisRowsOptions *rows) {
    return kv&&swa&&index&&attention&&producer&&rows?trellis_prepare_session(m,s,o,kv,swa,index,attention,producer,rows,NULL,NULL,NULL):-1;
}

int bn_transformer_trellis_prepare_session_engram(const BnModel*m,BnSession*s,
    const BnTransformerTrellisOptions*ffn,const BnTransformerTrellisRowsOptions*rows,
    const BnTransformerTrellisEngramOptions*engram) {
    return rows&&engram?trellis_prepare_session(m,s,ffn,NULL,NULL,NULL,NULL,NULL,rows,engram,NULL,NULL):-1;
}
int bn_transformer_trellis_prepare_session_producer_engram(const BnModel*m,BnSession*s,
    const BnTransformerTrellisOptions*ffn,const BnTransformerTrellisKVOptions*kv,
    const BnTransformerTrellisSWAOptions*swa,const BnTransformerTrellisIndexOptions*index,
    const BnTransformerTrellisAttentionOptions*attention,const BnTransformerTrellisProducerOptions*producer,
    const BnTransformerTrellisRowsOptions*rows,const BnTransformerTrellisEngramOptions*engram) {
    return kv&&swa&&index&&attention&&producer&&rows&&engram?
        trellis_prepare_session(m,s,ffn,kv,swa,index,attention,producer,rows,engram,NULL,NULL):-1;
}

int bn_transformer_trellis_prepare_session_hc(const BnModel*m,BnSession*s,
    const BnTransformerTrellisOptions*ffn,const BnTransformerTrellisRowsOptions*rows,
    const BnTransformerTrellisEngramOptions*engram,const BnTransformerTrellisHCOptions*hc) {
    return rows&&engram&&hc?trellis_prepare_session(m,s,ffn,NULL,NULL,NULL,NULL,NULL,rows,engram,hc,NULL):-1;
}
int bn_transformer_trellis_prepare_session_producer_hc(const BnModel*m,BnSession*s,
    const BnTransformerTrellisOptions*ffn,const BnTransformerTrellisKVOptions*kv,
    const BnTransformerTrellisSWAOptions*swa,const BnTransformerTrellisIndexOptions*index,
    const BnTransformerTrellisAttentionOptions*attention,const BnTransformerTrellisProducerOptions*producer,
    const BnTransformerTrellisRowsOptions*rows,const BnTransformerTrellisEngramOptions*engram,
    const BnTransformerTrellisHCOptions*hc) {
    return kv&&swa&&index&&attention&&producer&&rows&&engram&&hc?
        trellis_prepare_session(m,s,ffn,kv,swa,index,attention,producer,rows,engram,hc,NULL):-1;
}

int bn_transformer_trellis_prepare_session_forward(const BnModel*m,BnSession*s,
    const BnTransformerTrellisOptions*ffn,const BnTransformerTrellisKVOptions*kv,
    const BnTransformerTrellisSWAOptions*swa,const BnTransformerTrellisIndexOptions*index,
    const BnTransformerTrellisAttentionOptions*attention,const BnTransformerTrellisProducerOptions*producer,
    const BnTransformerTrellisRowsOptions*rows,const BnTransformerTrellisEngramOptions*engram,
    const BnTransformerTrellisHCOptions*hc,const BnTransformerTrellisForwardOptions*forward) {
    return kv&&swa&&index&&attention&&producer&&rows&&engram&&hc&&forward?
        trellis_prepare_session(m,s,ffn,kv,swa,index,attention,producer,rows,engram,hc,forward):-1;
}

#include "trellis_group.h"
#include "trellis_reuse.h"
#include "trellis_pool.h"
