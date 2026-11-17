#include "backend_session.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "trellis.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    BnGPUBackend *gpu;
    void *request;
    BnBackendRowStageOptions options;
    BnBackendRowStageInfo info;
    BnGPUResidentView buffers[4][3];
    size_t counts[4];
} RowStage;

int bn_backend_session_row_stage_plan(const BnBackendRowStageOptions *o,
    BnBackendRowStageWorkspace *out) {
    if(!o||!out||o->banks<1||o->banks>4)return -1;
    const BnQuantRowScaledOps *ops=bn_quant_format_row_scaled_ops(o->format);
    BnGPUScaledRowsWorkspace shape;
    if(!ops||!ops->validate_scales||ops->gpu_decode!=BN_QUANT_ROW_GPU_INT4_FP16S32_BF16||
        ops->weight_bits!=4||ops->block_cols!=32||ops->scale_element_bytes!=2||
        bn_gpu_scaled_rows_shape(o->max_rows,o->cols,&shape))return -1;
    size_t wb=shape.packed_i32*4,sb=shape.scales_f16*2,ob=shape.out_bf16*2;
    if(wb>SIZE_MAX-sb||wb+sb>SIZE_MAX-ob||wb+sb+ob>SIZE_MAX/(size_t)o->banks||
        wb>SIZE_MAX/(2*(size_t)o->banks))return -1;
    BnBackendRowStageWorkspace w={sizeof(RowStage),2*(size_t)o->banks*wb,
        (size_t)o->banks*(wb+sb+ob),wb,shape.packed_i32,shape.scales_f16,shape.out_bf16};
    *out=w;return 0;
}
static int overlap(BnGPUResidentView a,BnGPUResidentView b) {
    if(!a.request||a.request!=b.request||a.slot!=b.slot)return 0;
    if(a.type!=b.type)return 1;
    return a.offset<=b.offset?b.offset-a.offset<a.count:a.offset-b.offset<b.count;
}
int bn_backend_session_row_stage_init(BnBackendRowStage *stage,BnGPUBackend *g,
    void *request,const BnBackendRowStageOptions *o) {
    BnBackendRowStageWorkspace w;
    if(!stage||!g||!g->ctx||!request||bn_backend_session_row_stage_plan(o,&w)||
        o->host_budget<w.host_bytes||o->pinned_budget<w.pinned_bytes||o->device_budget<w.device_bytes||
        !g->resident_alloc||!g->resident_upload_reserve||!g->resident_upload_acquire||
        !g->resident_upload_cancel||!g->resident_upload_submit||!g->quant_rows_bf16_resident)return -1;
    RowStage *old=stage->state;
    if(old)return !old->info.failed&&old->gpu==g&&old->request==request&&
        old->options.format==o->format&&old->options.cols==o->cols&&
        old->options.max_rows==o->max_rows&&old->options.banks==o->banks?0:-1;
    RowStage *s=calloc(1,sizeof(*s));if(!s)return -1;
    s->gpu=g;s->request=request;s->options=*o;s->info.workspace=w;
    if(g->resident_upload_reserve(request,(unsigned)(2*o->banks),w.slot_bytes,w.pinned_bytes))goto fail;
    const int types[]={bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32),
        bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_F16),bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_BF16)};
    const size_t counts[]={w.packed_count,w.scale_count,w.out_count};
    for(int bank=0;bank<o->banks;bank++)for(int j=0;j<3;j++) {
        BnGPUResidentView v;
        if(g->resident_alloc(request,types[j],counts[j],&v)||v.request!=request||
            v.type!=types[j]||v.count!=counts[j]||v.offset>SIZE_MAX-v.count)goto fail;
        for(int b=0;b<=bank;b++)for(int k=0;k<3;k++)if(overlap(v,s->buffers[b][k]))goto fail;
        s->buffers[bank][j]=v;
    }
    stage->state=s;return 0;
fail:free(s);return -1;
}
void bn_backend_session_row_stage_free(BnBackendRowStage *stage) {
    if(stage){free(stage->state);stage->state=NULL;}
}
int bn_backend_session_row_stage_info(const BnBackendRowStage *stage,BnBackendRowStageInfo *out) {
    const RowStage *s=stage?stage->state:NULL;if(!s||!out)return -1;
    *out=s->info;return 0;
}
static int disjoint(const void *a,size_t n,const void *b,size_t m) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(n>UINTPTR_MAX-x||m>UINTPTR_MAX-y)return 0;
    return x+n<=y||y+m<=x;
}
static void add(uint64_t *p,uint64_t n) {*p=n>UINT64_MAX-*p?UINT64_MAX:*p+n;}
static int cancel(RowStage *s,const BnGPUHostUploadLease leases[2],unsigned mask) {
    int failed=0;
    for(unsigned i=0;i<2;i++)if((mask&(1u<<i))&&s->gpu->resident_upload_cancel(s->request,leases+i))failed=1;
    if(failed)s->info.failed=1;
    return failed?-1:0;
}
int bn_backend_session_row_stage_upload_i32(BnBackendRowStage *stage,
    const int32_t *values,size_t count,const BnGPUResidentView *destination) {
    RowStage*s=stage?stage->state:NULL;
    if(!s||s->info.failed||!values||!destination||!count||count>s->info.workspace.slot_bytes/4||
       destination->request!=s->request||destination->count!=count||destination->offset>SIZE_MAX-count||
       destination->type!=bn_backend_quant_scalar_type(BN_BACKEND_SCALAR_I32))return -1;
    size_t bytes=count*4;BnGPUResidentView dst=*destination;
    if(!disjoint(values,bytes,s,sizeof(*s))||!disjoint(values,bytes,stage,sizeof(*stage)))return -1;
    for(int b=0;b<s->options.banks;b++)for(int j=0;j<3;j++)if(overlap(dst,s->buffers[b][j]))return -1;
    BnGPUHostUploadLease leases[2]={{0}};
    int status=s->gpu->resident_upload_acquire(s->request,0,leases);
    if(status==BN_GPU_UPLOAD_BUSY){add(&s->info.busy_returns,1);return status;}
    if(status){s->info.failed=1;return -1;}
    if(leases[0].request!=s->request||leases[0].slot||!leases[0].data||!leases[0].generation||
       leases[0].capacity!=s->info.workspace.slot_bytes||!disjoint(leases[0].data,bytes,values,bytes)||
       !disjoint(leases[0].data,bytes,s,sizeof(*s))||!disjoint(leases[0].data,bytes,stage,sizeof(*stage))) {
        s->info.failed=1;cancel(s,leases,1);return -1;
    }
    memcpy(leases[0].data,values,bytes);
    /* Submit may consume the lease before reporting failure. */
    if(s->gpu->resident_upload_submit(s->request,leases,dst)){s->info.failed=1;return -1;}
    add(&s->info.upload_bytes,bytes);return 0;
}
static int row_stage_submit(BnBackendRowStage *stage,BnBackendRowCache *cache,
    unsigned bank,const uint64_t *ids,size_t count,BnGPUResidentView *out,uint64_t *epoch,int finite_f16) {
    RowStage *s=stage?stage->state:NULL;
    if(!s||s->info.failed||bank>=(unsigned)s->options.banks||count>(size_t)s->options.max_rows)return -1;
    if(!count)return 0;
    const BnQuantRowScaledOps *ops=bn_quant_format_row_scaled_ops(s->options.format);
    if(!ops||(finite_f16&&!ops->validate_f16))return -1;
    const BnTrellisRowLayout *layout=bn_backend_model_row_cache_layout(cache);
    BnBackendRowCacheStats cs;
    if(!out||!epoch||!ids||!layout||layout->format!=s->options.format||layout->cols!=s->options.cols||
        bn_backend_model_row_cache_stats(cache,&cs)||count>cs.max_gather_rows||s->info.epochs[bank]==UINT64_MAX)return -1;
    size_t ib=count*sizeof(*ids),wb=count*layout->weight_bytes,sb=count*layout->scale_bytes;
    if(!disjoint(out,sizeof(*out),epoch,sizeof(*epoch))||!disjoint(out,sizeof(*out),ids,ib)||
        !disjoint(epoch,sizeof(*epoch),ids,ib)||!disjoint(out,sizeof(*out),s,sizeof(*s))||
        !disjoint(epoch,sizeof(*epoch),s,sizeof(*s))||!disjoint(ids,ib,s,sizeof(*s))||
        !disjoint(out,sizeof(*out),stage,sizeof(*stage))||!disjoint(epoch,sizeof(*epoch),stage,sizeof(*stage))||
        !disjoint(out,sizeof(*out),cache,sizeof(*cache))||!disjoint(epoch,sizeof(*epoch),cache,sizeof(*cache))||
        !disjoint(out,sizeof(*out),cache->state,cs.allocated_bytes)||
        !disjoint(epoch,sizeof(*epoch),cache->state,cs.allocated_bytes)||
        !disjoint(ids,ib,cache->state,cs.allocated_bytes))return -1;
    for(size_t i=0;i<count;i++)if(ids[i]>=layout->rows)return -1;
    BnGPUHostUploadLease leases[2];unsigned mask=0;
    for(unsigned j=0;j<2;j++) {
        int result=s->gpu->resident_upload_acquire(s->request,bank*2+j,leases+j);
        if(result) {
            if(result!=BN_GPU_UPLOAD_BUSY)s->info.failed=1;
            int cancelled=cancel(s,leases,mask);
            if(result==BN_GPU_UPLOAD_BUSY&&!cancelled){add(&s->info.busy_returns,1);return BN_GPU_UPLOAD_BUSY;}
            return -1;
        }
        mask|=1u<<j;
        if(leases[j].request!=s->request||leases[j].slot!=bank*2+j||!leases[j].data||
            !leases[j].generation||leases[j].capacity!=s->info.workspace.slot_bytes) {
            s->info.failed=1;cancel(s,leases,mask);return -1;
        }
    }
    const size_t bytes[]={wb,sb};
    for(int j=0;j<2;j++)if(!disjoint(leases[j].data,bytes[j],ids,ib)||
        !disjoint(leases[j].data,bytes[j],out,sizeof(*out))||
        !disjoint(leases[j].data,bytes[j],epoch,sizeof(*epoch))||
        !disjoint(leases[j].data,bytes[j],s,sizeof(*s))||
        !disjoint(leases[j].data,bytes[j],cache->state,cs.allocated_bytes)) {
        s->info.failed=1;cancel(s,leases,mask);return -1;
    }
    if(!disjoint(leases[0].data,wb,leases[1].data,sb)) {
        s->info.failed=1;cancel(s,leases,mask);return -1;
    }
    if(bn_backend_model_row_cache_gather(cache,ids,count,leases[0].data,wb,leases[1].data,sb)||
        ops->validate_scales(leases[1].data,sb)||
        (finite_f16&&ops->validate_f16(leases[0].data,wb,leases[1].data,sb,(int)count,s->options.cols))) {
        cancel(s,leases,mask);return -1;
    }
    BnGPUQuantRowsBF16ResidentPlan p={0};p.rows=(int)count;p.cols=s->options.cols;p.type=s->options.format;
    p.row_stride_bytes=layout->weight_bytes;
    p.packed=bn_gpu_resident_view_slice(s->buffers[bank][0],0,wb/4);
    p.scales=bn_gpu_resident_view_slice(s->buffers[bank][1],0,sb/2);
    p.out=bn_gpu_resident_view_slice(s->buffers[bank][2],0,count*(size_t)s->options.cols);
    const BnGPUResidentView destinations[]={p.packed,p.scales};
    for(unsigned j=0;j<2;j++) {
        mask&=~(1u<<j); /* A failed submit may already have consumed this lease. */
        if(s->gpu->resident_upload_submit(s->request,leases+j,destinations[j])) {
            s->info.failed=1;cancel(s,leases,mask);return -1;
        }
        add(&s->info.upload_bytes,bytes[j]);
    }
    if(s->gpu->quant_rows_bf16_resident(s->gpu->ctx,s->request,&p)){s->info.failed=1;return -1;}
    add(&s->info.batches,1);add(&s->info.rows,count);s->info.epochs[bank]++;s->counts[bank]=p.out.count;
    *out=p.out;*epoch=s->info.epochs[bank];return 0;
}
int bn_backend_session_row_stage_submit(BnBackendRowStage *stage,BnBackendRowCache *cache,
    unsigned bank,const uint64_t *ids,size_t count,BnGPUResidentView *out,uint64_t *epoch) {
    return row_stage_submit(stage,cache,bank,ids,count,out,epoch,0);
}
int bn_backend_session_row_stage_submit_f16(BnBackendRowStage *stage,BnBackendRowCache *cache,
    unsigned bank,const uint64_t *ids,size_t count,BnGPUResidentView *out,uint64_t *epoch) {
    return row_stage_submit(stage,cache,bank,ids,count,out,epoch,1);
}
int bn_backend_session_row_stage_current(const BnBackendRowStage *stage,unsigned bank,
    uint64_t epoch,const BnGPUResidentView *view) {
    const RowStage *s=stage?stage->state:NULL;
    if(!s||s->info.failed||bank>=(unsigned)s->options.banks||!epoch||epoch!=s->info.epochs[bank]||!view)return 0;
    BnGPUResidentView root=s->buffers[bank][2];
    return view->request==root.request&&view->slot==root.slot&&view->offset==root.offset&&
        view->type==root.type&&view->count==s->counts[bank];
}
