/* Inspect actual private group execution with synthetic request-owned views.
 * Existing restart checks also guard unchanged ownership/ticket behavior. */
#include "model_internal.h"
#include "backend_layout.h"
#include "gpu_moe_bridge.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static BnModelNativeNumerics group_numeric;
static BnBackendNativeF16Weight group_weight;
static const BnModelNativeNumerics *group_numerics(const BnModel *m){(void)m;return &group_numeric;}
static const BnBackendNativeF16Weight *group_core(const BnBackendModel *b,int l,int role){(void)b;(void)l;(void)role;return &group_weight;}
static void *group_router_weight(const BnBackendModel *b,int l){(void)b;(void)l;return (void*)(uintptr_t)1;}
static int group_acquire(BnGPUMoETrellisStore*,int,int,int,BnGPUMoETrellisLease*);
static int group_release(BnGPUMoETrellisStore*,BnGPUMoETrellisLease*);
#define bn_model_native_numerics group_numerics
#define bn_backend_layout_native_core_weight group_core
#define bn_backend_layout_native_router_weight group_router_weight
#define bn_gpu_moe_trellis_acquire group_acquire
#define bn_gpu_moe_trellis_release group_release
#define main native_restart_checks_main
#include "test_native_restart.c"
#undef main
#undef bn_model_native_numerics
#undef bn_backend_layout_native_core_weight
#undef bn_backend_layout_native_router_weight
#undef bn_gpu_moe_trellis_acquire
#undef bn_gpu_moe_trellis_release
typedef struct {void *data[TR_COUNT];size_t count[TR_COUNT];int type[TR_COUNT],rows,id,pending,destroyed;int selected[1200],hits[8];} GroupGPU;
static GroupGPU group_gpus[2];
static int group_live[8],group_acquires,group_releases,group_fail_acquire,group_fail_status,group_syncs,group_fail_sync;
static int group_joint_calls;
static int group_decode_calls,group_fail_decode,group_zero_weights;
static int group_capacity,group_peak_live,group_overlap_acquires;
static void *group_data(BnGPUResidentView v) {
    GroupGPU *g=v.request;assert(g&&v.slot<TR_COUNT&&v.offset<=g->count[v.slot]&&v.count<=g->count[v.slot]-v.offset);
    return (unsigned char*)g->data[v.slot]+v.offset*(v.type==BN_GGUF_TENSOR_F16?2:4);
}
static int group_read(void *r,BnGPUResidentView v,void *p){assert(r==v.request);memcpy(p,group_data(v),v.count*(v.type==BN_GGUF_TENSOR_F16?2:4));return 0;}
static int group_write(void *r,BnGPUResidentView v,const void *p){assert(r==v.request);memcpy(group_data(v),p,v.count*(v.type==BN_GGUF_TENSOR_F16?2:4));return 0;}
static int group_zero(void *r,BnGPUResidentView v){assert(r==v.request);memset(group_data(v),0,v.count*(v.type==BN_GGUF_TENSOR_F16?2:4));return 0;}
static int group_sync(void *r){GroupGPU *g=r;if(++group_syncs==group_fail_sync)return -1;g->pending=0;return 0;}
static void group_destroy(void *r){GroupGPU *g=r;g->pending=0;g->destroyed++;}
static int group_acquire(BnGPUMoETrellisStore *s,int layer,int e,int admit,BnGPUMoETrellisLease *out) {
    assert(s==&reuse_store&&!layer&&!admit&&e>=0&&e<8&&!out->owner);
    ++group_acquires;
    if(group_fail_acquire&&group_acquires>=group_fail_acquire)return group_fail_status;
    int live=0;for(int i=0;i<8;i++)live+=group_live[i];
    if(group_capacity&&live>=group_capacity)return BN_GPU_UPLOAD_BUSY;
    assert(!group_live[e]);group_live[e]=1;out->owner=s;
    if(live+1>group_peak_live)group_peak_live=live+1;
    if(group_gpus[0].pending||group_gpus[1].pending)group_overlap_acquires++;
    out->buffers.gate=out->buffers.up=out->buffers.down=(void*)(uintptr_t)(e+1);return 0;
}
static int group_release(BnGPUMoETrellisStore *s,BnGPUMoETrellisLease *lease) {
    assert(s==&reuse_store&&lease->owner==s&&!group_gpus[0].pending&&!group_gpus[1].pending);
    int e=(int)(uintptr_t)lease->buffers.gate-1;assert(e>=0&&e<8&&group_live[e]);group_live[e]=0;
    group_releases++;memset(lease,0,sizeof(*lease));return 0;
}
static int group_route_gpu(void *ctx,void *request,const BnGPURouterResidentPlan *p) {
    (void)ctx;GroupGPU *g=request;assert(p->k==1&&p->n_tokens==g->rows);
    int32_t *ids=group_data(p->selected);uint16_t *weights=group_data(p->weights);
    for(int i=0;i<g->rows;i++){ids[i]=g->selected[i];weights[i]=bn_fp32_to_fp16(group_zero_weights?0:1);}
    return 0;
}
static int group_index_gpu(void *ctx,void *request,const BnGPUNativeMoEIndexPlan *p) {
    (void)ctx;GroupGPU *g=request;int32_t *counts=group_data(p->counts),*starts=group_data(p->starts);
    int32_t *order=group_data(p->order),*inverse=group_data(p->inverse);uint16_t *weights=group_data(p->sorted_weights);
    memset(counts,0,9*sizeof(int));for(int i=0;i<g->rows;i++)counts[g->selected[i]]++;
    starts[0]=0;for(int e=0;e<8;e++)starts[e+1]=starts[e]+counts[e];int cursor[8];memcpy(cursor,starts,sizeof(cursor));
    for(int i=0;i<g->rows;i++){int n=cursor[g->selected[i]]++;order[n]=i;inverse[i]=n;weights[n]=bn_fp32_to_fp16(1);}
    return 0;
}
static int group_launch_info(void *ctx,void *r,int tm,int tn,BnGPUFusedEXL3LaunchInfo *out) {
    (void)ctx;(void)r;(void)tm;(void)tn;*out=(BnGPUFusedEXL3LaunchInfo){0,128,1,32,8};return 0;
}
static int group_fused_gpu(void *ctx,void *request,const BnGPUFusedEXL3ResidentPlan *p) {
    (void)ctx;GroupGPU *g=request;int32_t *counts=group_data(p->counts),*base=group_data(p->slot_base),*order=group_data(p->order);
    uint16_t *x=group_data(p->x);float *out=group_data(p->slots);int starts[8]={0};
    for(int e=1;e<8;e++)starts[e]=starts[e-1]+counts[e-1];
    int active=0;for(int e=0;e<8;e++)if(counts[e]>=(p->tile_m==64?33:p->tile_m==32?17:1)&&counts[e]<=(p->tile_m==64?256:p->tile_m==32?32:16))active++;
    int width=32/(active<8?active:8);if(width>32)width=32;
    assert(p->group_width==width&&p->groups==(p->resident_count<32/width?p->resident_count:32/width));
    for(int b=0;b<p->resident_count;b++) {
        int e=p->expert_ids[b];assert(group_live[e]&&counts[e]<=256&&!g->hits[e]++);
        assert((int)(uintptr_t)p->weight[0][b]==e+1);
        for(int j=0;j<counts[e];j++)for(int d=0;d<128;d++)out[(size_t)(base[e]+j)*128+d]=bn_fp16_to_fp32(x[(size_t)order[starts[e]+j]*128+d])*(e+1);
    }
    g->pending++;return 0;
}
static int group_gather_gpu(void *ctx,void *request,const BnGPUNativeMoEGroupPlan *p) {
    (void)ctx;(void)request;uint16_t *x=group_data(p->x),*out=group_data(p->out);int32_t *order=group_data(p->order);
    memset(out,0,p->out.count*2);
    for(int b=0;b<p->batch;b++)for(int j=0;j<p->counts[b];j++)memcpy(out+((size_t)b*p->rows+j)*128,x+(size_t)order[p->starts[b]+j]*128,256);
    return 0;
}
static int group_fused_joint_gpu(void *ctx,void *owner,const BnGPUFusedEXL3GroupPlan *p,BnGPUFusedEXL3GroupStats *out) {
    assert(p&&p->n_consumers==2&&p->consumers[0].request==owner&&p->consumers[1].request!=owner);
    const BnGPUFusedEXL3ResidentPlan *a=p->consumers[0].plan,*b=p->consumers[1].plan;
    int compatible=a->tile_m==b->tile_m&&a->tile_n==b->tile_n&&a->group_width==b->group_width;
    group_joint_calls++;
    for(int i=0;i<p->n_consumers;i++)assert(!group_fused_gpu(ctx,p->consumers[i].request,p->consumers[i].plan));
    *out=(BnGPUFusedEXL3GroupStats){compatible?1:2,compatible?1:0,2};return 0;
}
static int group_folded_gpu(void *ctx,void *request,const BnGPUFoldedEXL3FFNResidentPlan *p) {
    (void)ctx;GroupGPU *g=request;uint16_t *x=group_data(p->gate.x);float *out=group_data(p->down.out);
    int counts[8]={0},heavy=0;for(int i=0;i<g->rows;i++)counts[g->selected[i]]++;
    for(int e=0;e<8;e++)heavy+=counts[e]>256;
    assert(p->gate.batch==heavy&&p->gate.n_tokens==300);
    for(int b=0;b<p->gate.batch;b++) {
        int e=(int)(uintptr_t)p->gate.weights[b]-1;assert(group_live[e]&&!g->hits[e]++);
        for(int j=0;j<p->gate.n_tokens;j++)for(int d=0;d<128;d++) {
            size_t k=((size_t)b*p->gate.n_tokens+j)*128+d;out[k]=bn_fp16_to_fp32(x[k])*(e+1);
        }
    }
    g->pending++;return 0;
}
static int group_decode_gpu(void *ctx,void *request,const BnGPUDecodeEXL3ResidentPlan *p) {
    (void)ctx;GroupGPU *g=request;
    assert(p->tokens==g->rows&&p->topk==1&&p->experts==8&&p->dim==128&&p->hidden==128);
    assert(p->x.request==request&&p->had_gate.request==request&&p->indices64.request==request);
    if(++group_decode_calls==group_fail_decode)return -1;
    int32_t *ids=group_data(p->indices);uint16_t *x=group_data(p->x),*weights=group_data(p->weights);float *out=group_data(p->out);
    for(int b=0;b<p->resident_count;b++) {
        int e=p->expert_ids[b];assert(e>=0&&e<8&&group_live[e]&&!g->hits[e]++);
        assert(p->weight[0][b]==(void*)(uintptr_t)(e+1)&&p->weight[1][b]==p->weight[0][b]&&p->weight[2][b]==p->weight[0][b]);
    }
    for(int i=0;i<g->rows;i++)for(int d=0;d<128;d++)
        out[(size_t)i*128+d]=bn_fp16_to_fp32(x[(size_t)i*128+d])*bn_fp16_to_fp32(weights[i])*(ids[i]+1);
    g->pending++;return 0;
}
static int group_reduce_gpu(void *ctx,void *request,const BnGPUNativeMoEReducePlan *p) {
    (void)ctx;GroupGPU *g=request;int32_t *base=group_data(p->slot_base),*starts=group_data(p->starts),*inverse=group_data(p->inverse);
    float *slots=group_data(p->slots),*out=group_data(p->out);
    for(int i=0;i<g->rows;i++)for(int d=0;d<128;d++)out[(size_t)i*128+d]=slots[(size_t)(base[g->selected[i]]+inverse[i]-starts[g->selected[i]])*128+d];
    return 0;
}
static int group_dense_gpu(void *ctx,void *request,const BnGPUF16FFNResidentPlan *p){(void)ctx;return group_zero(request,p->down.out);}
static int group_add_gpu(void *ctx,void *request,const BnGPUAddF32ResidentPlan *p) {
    (void)ctx;(void)request;float *a=group_data(p->x),*b=group_data(p->y),*out=group_data(p->out);
    for(size_t i=0;i<p->count;i++)out[i]=a[i]+b[i];
    return 0;
}
static TrellisRequest *group_fixture(int id,int rows,int variant) {
    GroupGPU *g=group_gpus+id;memset(g,0,sizeof(*g));g->id=id;g->rows=rows;
    TrellisRequest *r=calloc(1,sizeof(*r));assert(r);r->model=&reuse_model;r->anatomy=&reuse_anatomy;r->store=&reuse_store;
    r->gpu=&reuse_gpu;r->request=g;r->destroy=group_destroy;r->options=(BnTransformerTrellisOptions){rows,2,0,0,0};
    r->workspace=(BnTransformerTrellisWorkspace){0,0,8,rows+rows/10+1,rows*8};r->hc=calloc(1,sizeof(*r->hc));assert(r->hc);r->hc->tokens=rows;
    int E=8;r->metadata=calloc(6*E+2,sizeof(int));r->plan.groups=calloc(E,sizeof(*r->plan.groups));assert(r->metadata&&r->plan.groups);r->plan.group_capacity=r->plan.expert_capacity=E;
    r->counts=r->metadata;r->plan.order=r->counts+E+1;r->plan.starts=r->plan.order+E;r->plan.slot_base=r->plan.starts+E+1;r->plan.slot_kind=r->plan.slot_base+E;r->eligible=r->plan.slot_kind+E;
    for(int slot=0;slot<TR_COUNT;slot++) {
        size_t n=(size_t)rows*128;int type=BN_GGUF_TENSOR_F16;
        if(slot==TR_IDS||slot==TR_KEYS||slot==TR_VALUES||slot==TR_SORTED_KEYS||slot==TR_ORDER||slot==TR_INVERSE){n=rows;type=BN_GGUF_TENSOR_I32;}
        if(slot==TR_WEIGHTS||slot==TR_SORTED_WEIGHTS)n=rows;
        if(slot==TR_COUNTS||slot==TR_STARTS){n=E+1;type=BN_GGUF_TENSOR_I32;}
        if(slot==TR_BASE||slot==TR_KIND){n=E;type=BN_GGUF_TENSOR_I32;}
        if(slot==TR_SLOTS){n=(size_t)r->workspace.slot_rows*128;type=BN_GGUF_TENSOR_F32;}
        if(slot==TR_ROUTED||slot==TR_SHARED||slot==TR_OUT)type=BN_GGUF_TENSOR_F32;
        if(slot==TR_STATE_G||slot==TR_STATE_U||slot==TR_INTER_G||slot==TR_INTER_U)n=8u*256*128;
        if(slot==TR_GROUP_X||slot==TR_GROUP_G||slot==TR_GROUP_U)n=(size_t)rows*8*128;
        if(slot==TR_DECODED)n=8u*128*128;
        g->count[slot]=n;g->type[slot]=type;g->data[slot]=calloc(n,type==BN_GGUF_TENSOR_F16?2:4);assert(g->data[slot]);
        r->buffers[slot]=(BnGPUResidentView){g,0,n,(unsigned)slot,type};
    }
    for(int i=0;i<rows;i++) {
        g->selected[i]=variant?(i<300?0:i<600?1:2+i%6):(id==0?(i<300?0:1+i%7):(i<300?1:i<500?0:2+i%6));
        for(int d=0;d<128;d++)((uint16_t*)g->data[TR_X])[(size_t)i*128+d]=bn_fp32_to_fp16((float)(i%7+1));
    }
    return r;
}
static void group_fixture_free(TrellisRequest *r){GroupGPU *g=r->buffers[TR_X].request;for(int i=0;i<TR_COUNT;i++)free(g->data[i]);free(r->hc);free(r->plan.groups);free(r->metadata);free(r);}
int main(void) {
    assert(!native_restart_checks_main());reuse_anatomy.experts.dim=reuse_anatomy.experts.hidden=128;reuse_anatomy.experts.experts=8;reuse_anatomy.experts.active_experts=1;
    group_numeric.routed_scale=1;group_weight.rows=group_weight.cols=128;group_weight.weight=(void*)(uintptr_t)1;
    reuse_gpu=(BnGPUBackend){.resident_read=group_read,.resident_write=group_write,.resident_zero=group_zero,.resident_synchronize=group_sync,
        .router_sqrtsp_resident=group_route_gpu,.native_moe_index_resident=group_index_gpu,.trellis_fused_launch_info=group_launch_info,
        .trellis_fused_resident=group_fused_gpu,.native_moe_group_resident=group_gather_gpu,.trellis_folded_ffn_resident=group_folded_gpu,
        .trellis_decode_resident=group_decode_gpu,.native_moe_reduce_resident=group_reduce_gpu,.ffn_f16_resident=group_dense_gpu,.add_f32_resident=group_add_gpu};
    for(int joint=0;joint<2;joint++)for(int variant=0;variant<2;variant++) {
        reuse_gpu.trellis_fused_group_resident=joint?group_fused_joint_gpu:NULL;
        TrellisRequest *r[2]={group_fixture(0,800,variant),group_fixture(1,1000,variant)};BnTransformerTrellisGroupStats stats={0};
        group_acquires=group_releases=group_syncs=group_joint_calls=0;assert(!trellis_group_ffn(r,2,0,0,&stats));
        assert(stats.lookahead_borrows>0);
        assert((group_joint_calls>0)==joint);
        assert(stats.fused_launches==stats.joint_fused_launches+stats.individual_fused_launches);
        if(joint&&variant)assert(stats.joint_fused_launches&&stats.fused_launches_avoided);
        assert(group_acquires==(variant?10:8)&&group_acquires==group_releases&&stats.expert_borrows==(uint64_t)group_acquires);
        assert(stats.fused_launches&&stats.folded_launches);
        for(int id=0;id<2;id++) {
            GroupGPU *g=group_gpus+id;for(int e=0;e<8;e++)assert(g->hits[e]==1&&!group_live[e]);
            for(int j=0;j<g->rows;j++)for(int d=0;d<128;d++)assert(((float*)g->data[TR_OUT])[(size_t)j*128+d]==(float)((j%7+1)*(g->selected[j]+1)));
            group_fixture_free(r[id]);
        }
    }
    /* BUSY after a completed wave retries from unchanged normalized inputs;
     * every shared lease must be released after draining both consumers. */
    for(int failure=0;failure<3;failure++) {
        TrellisRequest *r[2]={group_fixture(0,800,0),group_fixture(1,1000,0)};BnTransformerTrellisGroupStats stats={0};
        group_acquires=group_releases=group_syncs=0;group_fail_acquire=failure<2?3:0;
        group_fail_status=failure==0?BN_GPU_UPLOAD_BUSY:-1;group_fail_sync=failure==2?1:0;
        assert(trellis_group_ffn(r,2,0,0,&stats)==(failure==0?BN_GPU_UPLOAD_BUSY:-1));
        for(int e=0;e<8;e++)assert(!group_live[e]);
        assert(!group_gpus[0].pending&&!group_gpus[1].pending);
        if(failure==0) {
            group_fail_acquire=0;memset(group_gpus[0].hits,0,sizeof(group_gpus[0].hits));memset(group_gpus[1].hits,0,sizeof(group_gpus[1].hits));
            assert(!trellis_group_ffn(r,2,0,0,&stats));
            for(int id=0;id<2;id++)for(int e=0;e<8;e++)assert(group_gpus[id].hits[e]==1&&!group_live[e]);
        }
        if(failure==2)assert(group_gpus[0].destroyed==1&&!r[0]->request&&group_syncs>=2);
        group_fail_acquire=group_fail_sync=0;group_fixture_free(r[0]);group_fixture_free(r[1]);
    }
    /* Two-entry compute waves with zero, partial or full lookahead capacity.
     * Mock copies may run while consumers are pending, but release still
     * requires BOTH consumers to be drained. Numeric outputs stay identical. */
    for(int capacity=2;capacity<=4;capacity++) {
        TrellisRequest *r[2]={group_fixture(0,320,0),group_fixture(1,320,0)};
        for(int id=0;id<2;id++)for(int i=0;i<320;i++)group_gpus[id].selected[i]=i%8;
        BnTransformerTrellisGroupStats stats={0};group_capacity=capacity;
        group_acquires=group_releases=group_syncs=group_peak_live=group_overlap_acquires=0;
        assert(!trellis_group_ffn(r,2,0,0,&stats));
        assert(group_peak_live==capacity&&group_releases==8&&stats.expert_borrows==8);
        assert((stats.lookahead_borrows>0)==(capacity>2));
        assert((group_overlap_acquires>0)==(capacity>2));
        assert((stats.lookahead_busy_stops>0)==(capacity<4));
        for(int id=0;id<2;id++) {
            GroupGPU *g=group_gpus+id;
            for(int e=0;e<8;e++)assert(g->hits[e]==1&&!group_live[e]);
            for(int i=0;i<320;i++)for(int d=0;d<128;d++)
                assert(((float*)g->data[TR_OUT])[(size_t)i*128+d]==(float)((i%7+1)*(i%8+1)));
            group_fixture_free(r[id]);
        }
    }
    /* Fail before/after a partial speculative load and while completing its
     * deferred remainder. Every held lease must be released on each path. */
    for(int failure=3;failure<=5;failure++) {
        TrellisRequest *r[2]={group_fixture(0,320,0),group_fixture(1,320,0)};
        for(int id=0;id<2;id++)for(int i=0;i<320;i++)group_gpus[id].selected[i]=i%8;
        BnTransformerTrellisGroupStats stats={0};group_capacity=3;
        group_acquires=group_releases=group_syncs=group_peak_live=group_overlap_acquires=0;
        group_fail_acquire=failure;group_fail_status=-1;
        assert(trellis_group_ffn(r,2,0,0,&stats)==-1);
        assert(!group_gpus[0].pending&&!group_gpus[1].pending);
        for(int e=0;e<8;e++)assert(!group_live[e]);
        assert(group_releases==(int)stats.expert_borrows&&group_peak_live<=3);
        group_fail_acquire=0;group_fixture_free(r[0]);group_fixture_free(r[1]);
    }
    /* Persistent external pressure after a successful speculative prefix must
     * still yield public BUSY, release that prefix, and allow a clean retry. */
    {
        TrellisRequest *r[2]={group_fixture(0,320,0),group_fixture(1,320,0)};
        for(int id=0;id<2;id++)for(int i=0;i<320;i++)group_gpus[id].selected[i]=i%8;
        BnTransformerTrellisGroupStats stats={0};group_capacity=3;
        group_acquires=group_releases=group_syncs=group_peak_live=group_overlap_acquires=0;
        group_fail_acquire=4;group_fail_status=BN_GPU_UPLOAD_BUSY;
        assert(trellis_group_ffn(r,2,0,0,&stats)==BN_GPU_UPLOAD_BUSY);
        assert(stats.lookahead_borrows==1&&group_releases==3);
        for(int e=0;e<8;e++)assert(!group_live[e]);
        assert(!group_gpus[0].pending&&!group_gpus[1].pending);
        group_fail_acquire=0;
        for(int id=0;id<2;id++)memset(group_gpus[id].hits,0,sizeof(group_gpus[id].hits));
        assert(!trellis_group_ffn(r,2,0,0,&stats));
        for(int id=0;id<2;id++)for(int e=0;e<8;e++)assert(group_gpus[id].hits[e]==1&&!group_live[e]);
        group_fixture_free(r[0]);group_fixture_free(r[1]);
    }
    group_capacity=0;
    /* Source-order expert tables and private decode plans are preserved while
     * common immutable weights are leased once for both consumers. */
    for(int rows=1;rows<=8;rows*=2)for(int zero=0;zero<2;zero++) {
        TrellisRequest *r[2]={group_fixture(0,rows,0),group_fixture(1,rows,0)};
        for(int id=0;id<2;id++)for(int i=0;i<rows;i++)group_gpus[id].selected[i]=i%4;
        BnTransformerTrellisGroupStats stats={0};group_zero_weights=zero;
        group_acquires=group_releases=group_syncs=group_decode_calls=0;
        assert(!trellis_group_decode(r,2,0,0,&stats));
        int unique=zero?0:rows<4?rows:4;
        assert(group_acquires==unique&&group_releases==unique&&stats.expert_borrows==(uint64_t)unique);
        assert(stats.decode_submissions==(uint64_t)(zero?0:2)&&stats.decode_cohorts==1&&stats.decode_expert_reuses==(uint64_t)unique);
        for(int id=0;id<2;id++) {
            GroupGPU *g=group_gpus+id;
            for(int i=0;i<rows;i++)for(int d=0;d<128;d++)assert(((float*)g->data[TR_OUT])[(size_t)i*128+d]==(zero?0:(float)((i%7+1)*(i%4+1))));
            for(int e=0;e<8;e++)assert(!group_live[e]);
            assert(!g->pending);group_fixture_free(r[id]);
        }
    }
    group_zero_weights=0;
    /* Failure while acquiring a prefix or after the first consumer queued
     * work releases only after both consumers are drained. BUSY can retry. */
    for(int failure=0;failure<4;failure++) {
        TrellisRequest *r[2]={group_fixture(0,8,0),group_fixture(1,8,0)};
        for(int id=0;id<2;id++)for(int i=0;i<8;i++)group_gpus[id].selected[i]=i%4;
        BnTransformerTrellisGroupStats stats={0};group_acquires=group_releases=group_syncs=group_decode_calls=0;
        group_fail_acquire=failure<2?3:0;group_fail_status=failure==0?BN_GPU_UPLOAD_BUSY:-1;
        group_fail_decode=failure==2?2:0;group_fail_sync=failure==3?1:0;
        assert(trellis_group_decode(r,2,0,0,&stats)==(failure==0?BN_GPU_UPLOAD_BUSY:-1));
        for(int e=0;e<8;e++)assert(!group_live[e]);
        assert(!group_gpus[0].pending&&!group_gpus[1].pending);
        if(failure==3)assert(group_gpus[0].destroyed&&!r[0]->request);
        group_fail_acquire=group_fail_decode=group_fail_sync=0;
        if(!failure)assert(!trellis_group_decode(r,2,0,0,&stats));
        group_fixture_free(r[0]);group_fixture_free(r[1]);
    }
    /* Cohort overflow partitions WHOLE source plans. A48-entry request is
     * legal on its own; duplicates retain order and reuse the same union. */
    {
        BnGPUDecodeEXL3ResidentPlan plans[4]={{0}};int experts[48],n,refs;
        for(int i=0;i<4;i++){plans[i].resident_count=32;for(int j=0;j<32;j++)plans[i].expert_ids[j]=i*16+j;}
        assert(trellis_group_decode_cohort(plans,4,0,experts,&n,&refs)==2&&n==48&&refs==64);
        for(int j=0;j<48;j++)assert(experts[j]==j);
        assert(trellis_group_decode_cohort(plans,4,2,experts,&n,&refs)==4&&n==48&&refs==64);
        for(int j=0;j<48;j++)assert(experts[j]==j+32);
        plans[0].resident_count=48;for(int j=0;j<48;j++)plans[0].expert_ids[j]=j;
        assert(trellis_group_decode_cohort(plans,4,0,experts,&n,&refs)==2&&n==48&&refs==80);
        plans[1].expert_ids[0]=99;
        assert(trellis_group_decode_cohort(plans,4,0,experts,&n,&refs)==1&&n==48&&refs==48);
    }
    group_capacity=0;
    ReuseGPU *left=malloc(sizeof(*left)),*right=malloc(sizeof(*right));assert(left&&right);
    TrellisRequest *a,*b;BnSession *sessions[2]={reuse_fixture(left,&a),reuse_fixture(right,&b)};
    BnTransformerTrellisForwardResult outputs[2],canary[2];memset(outputs,0xa5,sizeof(outputs));memcpy(canary,outputs,sizeof(outputs));
    BnSession *duplicates[2]={sessions[0],sessions[0]};
    assert(bn_transformer_trellis_forward_group(&reuse_model,duplicates,2,0,outputs,NULL)==-1&&
        !left->zero&&!left->sync&&!memcmp(outputs,canary,sizeof(outputs)));
    a->forward->group_owner=b->forward->group_owner=sessions[0];a->forward->group_count=2;
    a->forward->group_members[0]=sessions[0];a->forward->group_members[1]=sessions[1];
    a->forward->group_generations[0]=a->generation;a->forward->group_generations[1]=b->generation;
    a->forward->phase=b->forward->phase=5;
    BnSession *reversed[2]={sessions[1],sessions[0]};
    assert(bn_transformer_trellis_forward_group_cancel(&reuse_model,reversed,2)==-1&&!left->destroyed&&!right->destroyed);
    assert(!bn_transformer_trellis_forward_group_cancel(&reuse_model,sessions,2)&&left->destroyed==1&&right->destroyed==1);
    assert(sessions[0]->pos==3&&sessions[1]->pos==3&&sessions[0]->state.logits[0]==17&&sessions[1]->state.logits[0]==17);
    reuse_free(sessions[0]);reuse_free(sessions[1]);free(left);free(right);
    puts("Native grouped FFN: shared leases, per-request routes, all-consumer fences, fused peers and original folded groups PASSED");return 0;
}
