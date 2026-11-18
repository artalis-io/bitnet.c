#include "backend_layout.h"
#include "backend_model.h"
#include "backend_quant.h"
#include "model_internal.h"
#include "model_anatomy.h"
#include "gguf.h"
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#ifdef BN_ENABLE_MKL_VML
#include <mkl.h>
#endif

typedef struct {
    const BnSafetensorsIndex *index;
    const BnQuantBlockScaledOps *ops;
    size_t weight_id, scale_id, raw_capacity, scale_capacity;
    uint8_t *raw, *scale;
    int rows, cols;
} NativeF16Source;

static int native_f16_rows(void *context,int row,int count,uint16_t *out,size_t capacity) {
    NativeF16Source *s=context;
    if(!s||row<0||row>=s->rows||row%32||count<=0||count>32||
        count>s->rows-row)return -1;
    size_t bytes=(size_t)count*s->cols,sc=((size_t)s->cols+31)/32;
    if(bytes>s->raw_capacity||sc>s->scale_capacity||
        bn_safetensors_index_read(s->index,s->weight_id,(uint64_t)row*s->cols,s->raw,bytes)||
        bn_safetensors_index_read(s->index,s->scale_id,(uint64_t)(row/32)*sc,s->scale,sc))return -1;
    return s->ops->decode_region_fp16(s->raw,bytes,s->scale,sc,count,s->cols,
        0,0,count,s->cols,out,capacity);
}

static int native_f16_spec(const BnSafetensorsIndex *index,int format,
    size_t weight_id,size_t scale_id,BnBackendNativeF16Weight *out) {
    const BnQuantBlockScaledOps *ops=bn_quant_format_block_scaled_ops(format);
    if(!out||!index||!index->state||!ops||
        ops->decode!=BN_QUANT_BLOCK_SCALED_E4M3_UE8M0_32_F16||
        ops->block_rows!=32||ops->block_cols!=32||ops->weight_element_bytes!=1||
        ops->scale_element_bytes!=1||!ops->decode_region_fp16)return -1;
    const BnSafetensorsTensor *w=bn_safetensors_index_tensor(index,weight_id),
        *scale=bn_safetensors_index_tensor(index,scale_id);
    if(!w||!scale||w->dtype!=BN_SAFETENSORS_F8_E4M3||w->n_dims!=2||
        !w->shape[0]||!w->shape[1]||w->shape[0]>INT_MAX||w->shape[1]>INT_MAX||
        scale->dtype!=BN_SAFETENSORS_F8_E8M0||scale->n_dims!=2)return -1;
    size_t rows=(size_t)w->shape[0],cols=(size_t)w->shape[1],tr=rows<32?rows:32;
    size_t sr=(rows+31)/32,sc=(cols+31)/32;
    if(rows>SIZE_MAX/cols||rows*cols>SIZE_MAX/2||sr>SIZE_MAX/sc||
        w->bytes!=rows*cols||scale->bytes!=sr*sc||scale->shape[0]!=sr||
        scale->shape[1]!=sc||tr>SIZE_MAX/cols/3)return -1;
    size_t raw=tr*cols,decoded=raw*2,bytes=rows*cols*2;
    if(raw>SIZE_MAX-sc||raw+sc>SIZE_MAX-decoded||bytes>SIZE_MAX-decoded)return -1;
    BnBackendNativeF16Weight p={NULL,NULL,format,(int)rows,(int)cols,
        bytes,bytes+decoded,raw+sc+decoded};*out=p;return 0;
}
int bn_backend_layout_native_f16_init(BnBackendNativeF16Weight *out,
    BnGPUBackend *gpu,const BnSafetensorsIndex *index,int format,
    size_t weight_id,size_t scale_id,size_t device_budget,size_t host_budget) {
    if(!out||out->gpu||out->weight||!bn_gpu_backend_can_create_f16_rows_buffer(gpu))return -1;
    BnBackendNativeF16Weight next;
    if(native_f16_spec(index,format,weight_id,scale_id,&next)||
        device_budget<next.device_peak_bytes||host_budget<next.host_staging_bytes)return -1;
    const BnQuantBlockScaledOps *ops=bn_quant_format_block_scaled_ops(format);
    size_t raw=(size_t)(next.rows<32?next.rows:32)*next.cols,sc=((size_t)next.cols+31)/32;
    uint8_t *input=malloc(raw+sc);if(!input)return -1;
    NativeF16Source source={index,ops,weight_id,scale_id,raw,sc,input,input+raw,next.rows,next.cols};
    BnGPUF16RowsPlan plan={next.rows,next.cols,device_budget,host_budget-raw-sc,
        &source,native_f16_rows};
    void *handle=bn_gpu_backend_create_f16_rows_buffer(gpu,&plan);
    free(input);if(!handle)return -1;
    next.gpu=gpu;next.weight=handle;*out=next;return 0;
}
void bn_backend_layout_native_f16_free(BnBackendNativeF16Weight *weight) {
    if(!weight)return;
    bn_gpu_backend_destroy_buffer(weight->gpu,weight->weight);
    memset(weight,0,sizeof(*weight));
}

/* A single owner is also the future extension point for other native core
 * parameter layouts. Request-local allocations never enter this container. */
static const char native_core_owner=0;
typedef struct {
    BnBackendNativeParameterStats stats;
    int initialized;
    uint16_t map[BN_MODEL_ANATOMY_MAX_LAYERS][BN_NATIVE_ROLE_COUNT],global[BN_NATIVE_GLOBAL_COUNT];
    BnBackendNativeParameter weights[BN_MODEL_NATIVE_MAX_PARAMETERS];
} NativeParameterResources;
static void native_parameters_release(NativeParameterResources *p,BnGPUBackend *gpu) {
    if(!p)return;
    for(int i=p->initialized;i>0;i--)bn_gpu_backend_destroy_buffer(gpu,p->weights[i-1].handle);
    free(p);
}

typedef struct {
    BnBackendNativeRouterStats stats;
    int initialized;
    void *weights[BN_MODEL_ANATOMY_MAX_LAYERS];
} NativeRouterResources;
static void native_routers_release(NativeRouterResources *r,BnGPUBackend *gpu) {
    if(!r)return;
    for(int i=r->initialized;i>0;i--)bn_gpu_backend_destroy_buffer(gpu,r->weights[i-1]);
    free(r);
}

typedef struct {
    BnBackendNativeCoreStats stats;
    BnGPUBackend *gpu;
    NativeParameterResources *parameters;
    NativeRouterResources *routers;
    const void *execution_owner;
    void *execution;
    void (*execution_release)(void *);
    int layers,initialized;
    uint16_t map[BN_MODEL_ANATOMY_MAX_LAYERS][BN_NATIVE_ROLE_COUNT];
    BnBackendNativeF16Weight weights[BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS];
} NativeCoreResources;

int bn_backend_layout_native_core_plan(const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *a,BnBackendNativeCoreStats *out) {
    if(!out||!index||!index->state||!a||a->experts.layers<=0||
        a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||a->core_projection_count<=0||
        a->core_projection_count>BN_MODEL_NATIVE_MAX_CORE_PROJECTIONS)return -1;
    unsigned char covered[BN_MODEL_ANATOMY_MAX_LAYERS][BN_NATIVE_ROLE_COUNT]={{0}};
    BnBackendNativeCoreStats stats={0};
    for(int i=0;i<a->core_projection_count;i++) {
        const BnModelNativeCoreProjection *p=&a->core_projections[i];
        if(p->layer<0||p->layer>=a->experts.layers||p->weight_role<0||
            p->weight_role>=BN_NATIVE_ROLE_COUNT||p->scale_role<0||
            p->scale_role>=BN_NATIVE_ROLE_COUNT||p->weight_role==p->scale_role||
            p->weight_tensor<0||p->scale_tensor<0||
            a->layers[p->layer].tensor[p->weight_role]!=p->weight_tensor||
            a->layers[p->layer].tensor[p->scale_role]!=p->scale_tensor||
            covered[p->layer][p->weight_role]||covered[p->layer][p->scale_role])return -1;
        for(int j=0;j<i;j++)if(a->core_projections[j].weight_tensor==p->weight_tensor||
            a->core_projections[j].scale_tensor==p->scale_tensor)return -1;
        BnBackendNativeF16Weight spec;
        if(native_f16_spec(index,a->core_format,(size_t)p->weight_tensor,
            (size_t)p->scale_tensor,&spec)||spec.rows!=p->rows||spec.cols!=p->cols||
            stats.resident_bytes>SIZE_MAX-spec.device_peak_bytes)return -1;
        size_t peak=stats.resident_bytes+spec.device_peak_bytes;
        if(peak>stats.device_peak_bytes)stats.device_peak_bytes=peak;
        if(spec.host_staging_bytes>stats.host_staging_bytes)stats.host_staging_bytes=spec.host_staging_bytes;
        stats.resident_bytes+=spec.bytes;stats.count++;
        covered[p->layer][p->weight_role]=covered[p->layer][p->scale_role]=1;
    }
    /* No native block-scaled role may silently disappear from the owner. */
    for(int l=0;l<a->experts.layers;l++)for(int r=0;r<BN_NATIVE_ROLE_COUNT;r++) {
        int id=a->layers[l].tensor[r];if(id<0)continue;
        const BnSafetensorsTensor *t=bn_safetensors_index_tensor(index,(size_t)id);
        if(!t)return -1;
        if((t->dtype==BN_SAFETENSORS_F8_E4M3||t->dtype==BN_SAFETENSORS_F8_E8M0||
            t->dtype==BN_SAFETENSORS_F8_E5M2)&&!covered[l][r])return -1;
    }
    *out=stats;return 0;
}
static void native_core_release(void *context) {
    NativeCoreResources *r=context;if(!r)return;
    if(r->execution) {
        void *state=r->execution;void (*release)(void *)=r->execution_release;
        r->execution=NULL;r->execution_owner=NULL;r->execution_release=NULL;
        release(state);
    }
    native_routers_release(r->routers,r->gpu);
    native_parameters_release(r->parameters,r->gpu);
    for(int i=r->initialized;i>0;i--)bn_backend_layout_native_f16_free(&r->weights[i-1]);
    free(r);
}
int bn_backend_layout_prepare_native_core(BnBackendModel *backend,
    BnGPUBackend *gpu,const BnSafetensorsIndex *index,const BnModelNativeAnatomy *a,
    size_t device_budget,size_t host_budget) {
    BnBackendNativeCoreStats stats;
    if(!backend||bn_backend_model_gpu(backend)||
        !bn_gpu_backend_can_create_f16_rows_buffer(gpu)||
        bn_backend_layout_native_core_plan(index,a,&stats)||
        stats.device_peak_bytes>device_budget||stats.host_staging_bytes>host_budget)return -1;
    NativeCoreResources *r=calloc(1,sizeof(*r));if(!r)return -1;
    r->stats=stats;r->gpu=gpu;r->layers=a->experts.layers;size_t current=0;
    for(int i=0;i<a->core_projection_count;i++) {
        const BnModelNativeCoreProjection *p=&a->core_projections[i];
        if(bn_backend_layout_native_f16_init(&r->weights[i],gpu,index,a->core_format,
            (size_t)p->weight_tensor,(size_t)p->scale_tensor,device_budget-current,host_budget)) {
            native_core_release(r);return -1;
        }
        r->initialized++;current+=r->weights[i].bytes;
        r->map[p->layer][p->weight_role]=(uint16_t)(i+1);
    }
    BnBackendModel *staged=bn_backend_model_create();
    if(!staged){native_core_release(r);return -1;}
    if(bn_backend_model_attach_gpu_resources(staged,&native_core_owner,r,native_core_release)) {
        bn_backend_model_free(staged);native_core_release(r);return -1;
    }
    bn_backend_model_bind_gpu(staged,gpu);
    int status=bn_backend_model_move_gpu_resources(backend,staged);
    bn_backend_model_free(staged);return status;
}
const BnBackendNativeCoreStats *bn_backend_layout_native_core_stats(const BnBackendModel *backend) {
    const NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    return r?&r->stats:NULL;
}
const BnBackendNativeF16Weight *bn_backend_layout_native_core_weight(
    const BnBackendModel *backend,int layer,int role) {
    const NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    if(!r||layer<0||layer>=r->layers||role<0||role>=BN_NATIVE_ROLE_COUNT)return NULL;
    unsigned id=r->map[layer][role];return id?&r->weights[id-1]:NULL;
}

int bn_backend_layout_attach_native_execution(BnBackendModel *backend,
    const void *owner,void *state,void (*release)(void *)) {
    NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    if(!r||!owner||!state||!release||r->execution||
        bn_backend_model_gpu(backend)!=r->gpu||state==(void *)r||
        state==(void *)r->parameters||state==(void *)r->routers)return -1;
    r->execution_owner=owner;r->execution=state;r->execution_release=release;
    return 0;
}
void *bn_backend_layout_native_execution(const BnBackendModel *backend,
    const void *owner) {
    const NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    return r&&owner&&r->execution_owner==owner?r->execution:NULL;
}

typedef struct {
    BnBackendNativeParameter weight;
    size_t peak,host,source_width;
} NativeParameterSpec;
static int native_parameter_spec(const BnSafetensorsIndex *index,
    const BnModelNativeParameter *p,NativeParameterSpec *out) {
    if(!p||!out||p->tensor<0||p->rows<=0||p->cols<=0)return -1;
    const BnSafetensorsTensor *t=bn_safetensors_index_tensor(index,(size_t)p->tensor);
    if(!t||t->n_dims<1||t->n_dims>2||
        t->shape[t->n_dims-1]!=(uint64_t)p->cols||
        (t->n_dims==1?p->rows!=1:t->shape[0]!=(uint64_t)p->rows))return -1;
    size_t width=t->dtype==BN_SAFETENSORS_BF16?2:t->dtype==BN_SAFETENSORS_F32?4:0;
    int source_type=width==2?BN_GGUF_TENSOR_BF16:BN_GGUF_TENSOR_F32;
    const BnQuantFormatOps *source=bn_quant_format_ops(source_type),*dest=bn_quant_format_ops(p->type);
    if(!width||!source||source->block_elems!=1||source->bytes_per_block!=width||
        !dest||dest->block_elems!=1||dest->bytes_per_block!=width||
        (size_t)p->rows>SIZE_MAX/(size_t)p->cols/width||
        t->bytes!=(size_t)p->rows*p->cols*width)return -1;
    if(p->layout==BN_NATIVE_PARAMETER_F16_LINEAR) {
        if(t->dtype!=BN_SAFETENSORS_BF16||t->n_dims!=2||p->type!=BN_GGUF_TENSOR_F16)return -1;
    }else if(p->layout!=BN_NATIVE_PARAMETER_RAW||p->type!=source_type)return -1;
    size_t bytes=t->bytes,tile=(size_t)(p->rows<32?p->rows:32)*p->cols*width;
    if(p->layout==BN_NATIVE_PARAMETER_F16_LINEAR&&(bytes>SIZE_MAX-tile||tile>SIZE_MAX/2))return -1;
    NativeParameterSpec spec={{NULL,p->type,p->rows,p->cols,p->layout,bytes},
        bytes+(p->layout==BN_NATIVE_PARAMETER_F16_LINEAR?tile:0),
        tile*(p->layout==BN_NATIVE_PARAMETER_F16_LINEAR?2:1),width};
    *out=spec;return 0;
}
int bn_backend_layout_native_parameter_plan(const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *a,BnBackendNativeParameterStats *out) {
    BnBackendNativeCoreStats core;
    if(!out||bn_backend_layout_native_core_plan(index,a,&core)||a->parameter_count<=0||
        a->parameter_count>BN_MODEL_NATIVE_MAX_PARAMETERS)return -1;
    BnBackendNativeParameterStats stats={0};stats.device_peak_bytes=core.device_peak_bytes;
    stats.host_staging_bytes=core.host_staging_bytes;size_t current=core.resident_bytes;
    unsigned char covered[BN_MODEL_ANATOMY_MAX_LAYERS][BN_NATIVE_ROLE_COUNT]={{0}},globals[BN_NATIVE_GLOBAL_COUNT]={0};
    const int global_ids[]={a->embedding,a->output,a->output_norm};
    for(int i=0;i<a->parameter_count;i++) {
        const BnModelNativeParameter *p=&a->parameters[i];
        if(p->layer==-1){if(p->role<0||p->role>=BN_NATIVE_GLOBAL_COUNT||globals[p->role]||p->tensor!=global_ids[p->role])return -1;globals[p->role]=1;}
        else {if(p->layer<0||p->layer>=a->experts.layers||p->role<0||p->role>=BN_NATIVE_ROLE_COUNT||covered[p->layer][p->role]||a->layers[p->layer].tensor[p->role]!=p->tensor)return -1;covered[p->layer][p->role]=1;}
        for(int j=0;j<i;j++)if(a->parameters[j].tensor==p->tensor)return -1;
        NativeParameterSpec spec;
        if(native_parameter_spec(index,p,&spec)||current>SIZE_MAX-spec.peak)return -1;
        size_t peak=current+spec.peak;if(peak>stats.device_peak_bytes)stats.device_peak_bytes=peak;
        if(spec.host>stats.host_staging_bytes)stats.host_staging_bytes=spec.host;
        current+=spec.weight.bytes;stats.count++;stats.linear_count+=p->layout==BN_NATIVE_PARAMETER_F16_LINEAR;
    }
    for(int i=0;i<BN_NATIVE_GLOBAL_COUNT;i++)if(!globals[i])return -1;
    for(int l=0;l<a->experts.layers;l++)for(int role=0;role<BN_NATIVE_ROLE_COUNT;role++) {
        int id=a->layers[l].tensor[role];if(id<0)continue;
        const BnSafetensorsTensor *t=bn_safetensors_index_tensor(index,(size_t)id);
        if(!t)return -1;
        if((t->dtype==BN_SAFETENSORS_BF16||t->dtype==BN_SAFETENSORS_F32)&&!covered[l][role])return -1;
    }
    stats.bytes=current-core.resident_bytes;stats.aggregate_resident_bytes=current;*out=stats;return 0;
}
typedef struct {
    const BnSafetensorsIndex *index;
    const BnModelNativeParameter *p;
    size_t width,capacity;
    unsigned char *raw;
} NativeParameterSource;
static int native_parameter_rows(void *context,int row,int count,void *out,size_t capacity) {
    NativeParameterSource *s=context;
    if(!s||!out||row<0||row>=s->p->rows||row%32||count<=0||count>32||count>s->p->rows-row)return -1;
    size_t bytes=(size_t)count*s->p->cols*s->width;
    if(capacity<bytes||bn_safetensors_index_read(s->index,(size_t)s->p->tensor,
        (uint64_t)row*s->p->cols*s->width,out,bytes))return -1;
    const unsigned char *b=out;
    for(size_t i=0;i<bytes;i+=s->width) {
        uint32_t bits=(uint32_t)b[i]|(uint32_t)b[i+1]<<8;
        if(s->width==4)bits|=(uint32_t)b[i+2]<<16|(uint32_t)b[i+3]<<24;
        if(s->width==2?(bits&0x7f80u)==0x7f80u:(bits&0x7f800000u)==0x7f800000u)return -1;
    }
    return 0;
}
static int native_parameter_half_rows(void *context,int row,int count,uint16_t *out,size_t capacity) {
    NativeParameterSource *s=context;
    if(!s||!s->p||!out||count<=0||count>32)return -1;
    size_t n=(size_t)count*s->p->cols;
    if(capacity<n||native_parameter_rows(s,row,count,s->raw,s->capacity))return -1;
    for(size_t i=0;i<n;i++){uint16_t bits=(uint16_t)(s->raw[i*2]|(unsigned)s->raw[i*2+1]<<8);out[i]=bn_fp32_to_fp16(bn_bf16_to_fp32(bits));}
    return 0;
}
static int native_parameter_init(BnBackendNativeParameter *out,BnGPUBackend *gpu,
    const BnSafetensorsIndex *index,const BnModelNativeParameter *p,size_t device,size_t host) {
    NativeParameterSpec spec;if(native_parameter_spec(index,p,&spec)||device<spec.peak||host<spec.host)return -1;
    NativeParameterSource source={index,p,spec.source_width,0,NULL};void *handle=NULL;
    if(p->layout==BN_NATIVE_PARAMETER_RAW) {
        BnGPURawRowsPlan plan={p->type,p->rows,p->cols,device,host,&source,native_parameter_rows};
        handle=bn_gpu_backend_create_raw_rows_buffer(gpu,&plan);
    }else {
        size_t tile=(size_t)(p->rows<32?p->rows:32)*p->cols*2;
        source.raw=malloc(tile);if(!source.raw)return -1;source.capacity=tile;
        BnGPUF16RowsPlan plan={p->rows,p->cols,device,host-tile,&source,native_parameter_half_rows};
        handle=bn_gpu_backend_create_f16_rows_buffer(gpu,&plan);free(source.raw);
    }
    if(!handle)return -1;
    spec.weight.handle=handle;*out=spec.weight;return 0;
}
int bn_backend_layout_prepare_native_parameters(BnBackendModel *backend,
    const BnSafetensorsIndex *index,const BnModelNativeAnatomy *a,size_t device,size_t host) {
    NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    BnBackendNativeParameterStats stats;
    if(!r||r->parameters||!r->gpu||!bn_gpu_backend_can_create_raw_rows_buffer(r->gpu)||
        !bn_gpu_backend_can_create_f16_rows_buffer(r->gpu)||
        bn_backend_layout_native_parameter_plan(index,a,&stats)||
        stats.device_peak_bytes>device||stats.host_staging_bytes>host||
        r->stats.count!=(size_t)a->core_projection_count||r->layers!=a->experts.layers)return -1;
    for(int i=0;i<a->core_projection_count;i++) {
        const BnModelNativeCoreProjection *p=&a->core_projections[i];
        const BnBackendNativeF16Weight *w=bn_backend_layout_native_core_weight(backend,p->layer,p->weight_role);
        if(!w||w->rows!=p->rows||w->cols!=p->cols||w->format!=a->core_format)return -1;
    }
    if(stats.aggregate_resident_bytes-stats.bytes!=r->stats.resident_bytes)return -1;
    NativeParameterResources *next=calloc(1,sizeof(*next));if(!next)return -1;
    next->stats=stats;size_t current=r->stats.resident_bytes;
    for(int i=0;i<a->parameter_count;i++) {
        const BnModelNativeParameter *p=&a->parameters[i];
        if(native_parameter_init(&next->weights[i],r->gpu,index,p,device-current,host)) {
            native_parameters_release(next,r->gpu);return -1;
        }
        next->initialized++;current+=next->weights[i].bytes;
        if(p->layer<0)next->global[p->role]=(uint16_t)(i+1);
        else next->map[p->layer][p->role]=(uint16_t)(i+1);
    }
    r->parameters=next;return 0;
}
const BnBackendNativeParameterStats *bn_backend_layout_native_parameter_stats(const BnBackendModel *backend) {
    const NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    return r&&r->parameters?&r->parameters->stats:NULL;
}
const BnBackendNativeParameter *bn_backend_layout_native_parameter(const BnBackendModel *backend,int layer,int role) {
    const NativeCoreResources *r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    if(!r||!r->parameters||role<0)return NULL;
    unsigned id;
    if(layer==-1){if(role>=BN_NATIVE_GLOBAL_COUNT)return NULL;id=r->parameters->global[role];}
    else {if(layer<0||layer>=r->layers||role>=BN_NATIVE_ROLE_COUNT)return NULL;id=r->parameters->map[layer][role];}
    return id?&r->parameters->weights[id-1]:NULL;
}

int bn_backend_layout_native_router_plan(const BnSafetensorsIndex *index,
    const BnModelNativeAnatomy *a,BnBackendNativeRouterStats *out) {
    BnBackendNativeParameterStats parameters;
    if(!out||bn_backend_layout_native_parameter_plan(index,a,&parameters))return -1;
    size_t bytes=bn_gpu_router_weight_bytes(a->experts.experts,a->experts.dim);
    if(!bytes||a->experts.layers<=0||a->experts.layers>BN_MODEL_ANATOMY_MAX_LAYERS||
        (size_t)a->experts.layers>SIZE_MAX/bytes)return -1;
    for(int layer=0;layer<a->experts.layers;layer++) {
        int wi=a->layers[layer].tensor[BN_NATIVE_ROUTER],bi=a->layers[layer].tensor[BN_NATIVE_ROUTER_BIAS];
        if(wi<0||bi<0)return -1;
        const BnSafetensorsTensor*w=bn_safetensors_index_tensor(index,(size_t)wi),*b=bn_safetensors_index_tensor(index,(size_t)bi);
        if(!w||!b||w->dtype!=BN_SAFETENSORS_BF16||w->n_dims!=2||w->shape[0]!=(uint64_t)a->experts.experts||w->shape[1]!=(uint64_t)a->experts.dim||
            b->dtype!=BN_SAFETENSORS_F32||b->n_dims!=1||b->shape[0]!=(uint64_t)a->experts.experts)return -1;
    }
    size_t total=(size_t)a->experts.layers*bytes;
    if(parameters.aggregate_resident_bytes>SIZE_MAX-total)return -1;
    size_t aggregate=parameters.aggregate_resident_bytes+total;
    BnBackendNativeRouterStats stats={(size_t)a->experts.layers,total,aggregate,
        aggregate>parameters.device_peak_bytes?aggregate:parameters.device_peak_bytes,parameters.host_staging_bytes};
    *out=stats;return 0;
}
int bn_backend_layout_prepare_native_routers(BnBackendModel *backend,
    const BnSafetensorsIndex *index,const BnModelNativeAnatomy *a,size_t device,size_t host) {
    NativeCoreResources*r=bn_backend_model_gpu_resources(backend,&native_core_owner);BnBackendNativeRouterStats stats;
    if(!r||!a||!r->parameters||r->routers||r->layers!=a->experts.layers||
        !bn_gpu_backend_can_create_router_weight(r->gpu)||bn_backend_layout_native_router_plan(index,a,&stats)||
        device<stats.device_peak_bytes||host<stats.host_staging_bytes||
        r->parameters->stats.aggregate_resident_bytes!=stats.aggregate_resident_bytes-stats.bytes)return -1;
    for(int layer=0;layer<r->layers;layer++) {
        const BnBackendNativeParameter*w=bn_backend_layout_native_parameter(backend,layer,BN_NATIVE_ROUTER),*b=bn_backend_layout_native_parameter(backend,layer,BN_NATIVE_ROUTER_BIAS);
        if(!w||!w->handle||w->layout!=BN_NATIVE_PARAMETER_F16_LINEAR||w->type!=BN_GGUF_TENSOR_F16||w->rows!=a->experts.experts||w->cols!=a->experts.dim||
            !b||!b->handle||b->layout!=BN_NATIVE_PARAMETER_RAW||b->type!=BN_GGUF_TENSOR_F32||b->rows!=1||b->cols!=a->experts.experts)return -1;
    }
    NativeRouterResources*next=calloc(1,sizeof(*next));if(!next)return -1;
    next->stats=stats;size_t current=r->parameters->stats.aggregate_resident_bytes;
    for(int layer=0;layer<r->layers;layer++) {
        const BnBackendNativeParameter*w=bn_backend_layout_native_parameter(backend,layer,BN_NATIVE_ROUTER),*b=bn_backend_layout_native_parameter(backend,layer,BN_NATIVE_ROUTER_BIAS);
        BnGPURouterWeightPlan plan={a->experts.experts,a->experts.dim,w->handle,b->handle,device-current};
        next->weights[layer]=bn_gpu_backend_create_router_weight(r->gpu,&plan);
        if(!next->weights[layer]){native_routers_release(next,r->gpu);return -1;}
        next->initialized++;current+=bn_gpu_router_weight_bytes(a->experts.experts,a->experts.dim);
    }
    r->routers=next;return 0;
}
const BnBackendNativeRouterStats *bn_backend_layout_native_router_stats(const BnBackendModel *backend) {
    const NativeCoreResources*r=bn_backend_model_gpu_resources(backend,&native_core_owner);return r&&r->routers?&r->routers->stats:NULL;
}
void *bn_backend_layout_native_router_weight(const BnBackendModel *backend,int layer) {
    const NativeCoreResources*r=bn_backend_model_gpu_resources(backend,&native_core_owner);
    return r&&r->routers&&layer>=0&&layer<r->layers?r->routers->weights[layer]:NULL;
}

static void prepared_stats_clear(BnBackendLayoutPreparedStats *stats) {
    if (stats) memset(stats, 0, sizeof(*stats));
}

static void prepared_stats_add(BnBackendLayoutPreparedStats *dst,
                               const BnBackendLayoutPreparedStats *src) {
    if (!dst || !src) return;
    dst->lowbit_repack_bytes += src->lowbit_repack_bytes;
    dst->q8_repack_bytes += src->q8_repack_bytes;
    dst->kquant_scale_table_bytes += src->kquant_scale_table_bytes;
    dst->expanded_kquant_weight_bytes += src->expanded_kquant_weight_bytes;
    dst->f32_scale_table_bytes += src->f32_scale_table_bytes;
}

static void prepared_stats_add_bytes(BnBackendLayoutPreparedStats *stats,
                                     BnPreparedWeightKind kind,
                                     size_t bytes) {
    if (!stats || bytes == 0) return;
    switch (kind) {
        case BN_PREPARED_WEIGHT_Q4_0_REPACK:
        case BN_PREPARED_WEIGHT_Q4_0_X8:
        case BN_PREPARED_WEIGHT_Q5_K_X8:
        case BN_PREPARED_WEIGHT_Q6_K_X8:
            stats->lowbit_repack_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q8_0_REPACK:
            stats->q8_repack_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q4_K_SCALES:
            stats->kquant_scale_table_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q6_K_EXPANDED:
            stats->expanded_kquant_weight_bytes += bytes;
            break;
        case BN_PREPARED_WEIGHT_Q8_0_F32_SCALES:
            stats->f32_scale_table_bytes += bytes;
            break;
        default:
            break;
    }
}

static void prepared_qweight_size_one(const BnQWeight *w,
                                      BnBackendLayoutPreparedStats *stats) {
    if (!stats) return;
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(w, &kind);
    prepared_stats_add_bytes(stats, kind, bytes);
}

static void prepared_qweight_size_layer(const BnLayerWeights *lw,
                                        BnBackendLayoutPreparedStats *stats) {
    prepared_qweight_size_one(&lw->attn.wq, stats);
    prepared_qweight_size_one(&lw->attn.wk, stats);
    prepared_qweight_size_one(&lw->attn.wv, stats);
    prepared_qweight_size_one(&lw->attn.wo, stats);
    prepared_qweight_size_one(&lw->ssm.wqkv, stats);
    prepared_qweight_size_one(&lw->ssm.wz, stats);
    prepared_qweight_size_one(&lw->ssm.ssm_out, stats);
    prepared_qweight_size_one(&lw->ffn.ffn_gate, stats);
    prepared_qweight_size_one(&lw->ffn.ffn_up, stats);
    prepared_qweight_size_one(&lw->ffn.ffn_down, stats);
}

static void prepared_qweight_prepare_one(BnBackendModel *backend,
                                         const BnQWeight *w,
                                         SHArena *arena) {
    BnPreparedWeight prepared = { 0 };
    if (bn_quant_prepare_qweight(&prepared, w, arena) == 0)
        (void)bn_backend_model_register_prepared_qweight(backend, w,
                                                         &prepared);
}

static void prepared_qweight_prepare_layer(BnBackendModel *backend,
                                           const BnLayerWeights *lw,
                                           SHArena *arena) {
    prepared_qweight_prepare_one(backend, &lw->attn.wq, arena);
    prepared_qweight_prepare_one(backend, &lw->attn.wk, arena);
    prepared_qweight_prepare_one(backend, &lw->attn.wv, arena);
    prepared_qweight_prepare_one(backend, &lw->attn.wo, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm.wqkv, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm.wz, arena);
    prepared_qweight_prepare_one(backend, &lw->ssm.ssm_out, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn.ffn_gate, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn.ffn_up, arena);
    prepared_qweight_prepare_one(backend, &lw->ffn.ffn_down, arena);
}

static void *upload_stacked_quant_layout(BnGPUBackend *gpu,
                                         const void *data, size_t size,
                                         int type, int rows, int cols) {
    if (!gpu || !data || size == 0)
        return NULL;
    if (bn_gpu_backend_can_create_quant_only_buffer(gpu))
        return bn_gpu_backend_create_quant_only_buffer(gpu, data, size,
                                                       type, rows, cols);
    return bn_gpu_backend_create_buffer(gpu, data, size, type, rows, cols);
}

const char *bn_backend_layout_reason_string(BnBackendLayoutReason reason) {
    switch (reason) {
        case BN_BACKEND_LAYOUT_OK: return "ok";
        case BN_BACKEND_LAYOUT_NO_GPU: return "no_gpu";
        case BN_BACKEND_LAYOUT_NO_BUFFER_CREATE: return "no_buffer_create";
        case BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED: return "no_buffer_create_biased";
        case BN_BACKEND_LAYOUT_MISSING_WEIGHT: return "missing_weight";
        case BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE:
            return "embedded_scale_not_stackable";
        case BN_BACKEND_LAYOUT_TYPE_MISMATCH: return "type_mismatch";
        case BN_BACKEND_LAYOUT_COL_MISMATCH: return "col_mismatch";
        case BN_BACKEND_LAYOUT_ZERO_SIZE: return "zero_size";
        case BN_BACKEND_LAYOUT_ALLOC_FAILED: return "alloc_failed";
        case BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED: return "bias_unsupported";
        default: return "unknown";
    }
}

size_t bn_backend_layout_qweight_data_size(const BnQWeight *w) {
    return bn_qweight_data_size(w);
}

BnBackendLayoutReason bn_backend_layout_stackable_reason(const BnQWeight *a,
                                                         const BnQWeight *b) {
    if (!a || !b || !a->data || !b->data) return BN_BACKEND_LAYOUT_MISSING_WEIGHT;
    if (!bn_backend_quant_allows_stacked_layout(a->type) ||
        !bn_backend_quant_allows_stacked_layout(b->type))
        return BN_BACKEND_LAYOUT_EMBEDDED_SCALE_NOT_STACKABLE;
    if (a->type != b->type) return BN_BACKEND_LAYOUT_TYPE_MISMATCH;
    if (a->cols != b->cols) return BN_BACKEND_LAYOUT_COL_MISMATCH;
    return BN_BACKEND_LAYOUT_OK;
}

int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b) {
    return bn_backend_layout_stackable_reason(a, b) == BN_BACKEND_LAYOUT_OK;
}

int bn_backend_layout_stackable3(const BnQWeight *a,
                                 const BnQWeight *b,
                                 const BnQWeight *c) {
    return bn_backend_layout_stackable(a, b) &&
           bn_backend_layout_stackable(a, c);
}

BnBackendLayoutReason bn_backend_layout_stacked2_reason(const BnGPUBackend *gpu,
                                                        const BnQWeight *a,
                                                        const BnQWeight *b) {
    if (!gpu) return BN_BACKEND_LAYOUT_NO_GPU;
    if (!bn_gpu_backend_can_create_buffer(gpu))
        return BN_BACKEND_LAYOUT_NO_BUFFER_CREATE;
    BnBackendLayoutReason reason = bn_backend_layout_stackable_reason(a, b);
    if (reason != BN_BACKEND_LAYOUT_OK) return reason;
    if (bn_qweight_data_size(a) == 0 || bn_qweight_data_size(b) == 0)
        return BN_BACKEND_LAYOUT_ZERO_SIZE;
    return BN_BACKEND_LAYOUT_OK;
}

BnBackendLayoutReason bn_backend_layout_biased_qweight_reason(const BnGPUBackend *gpu,
                                                              const BnQWeight *w,
                                                              const float *bias) {
    if (!gpu) return BN_BACKEND_LAYOUT_NO_GPU;
    if (!bn_gpu_backend_can_create_biased_buffer(gpu))
        return BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED;
    if (!w || !w->data || !bias) return BN_BACKEND_LAYOUT_MISSING_WEIGHT;
    if (bn_qweight_data_size(w) == 0) return BN_BACKEND_LAYOUT_ZERO_SIZE;
    return BN_BACKEND_LAYOUT_OK;
}

BnBackendLayoutReason bn_backend_layout_stacked3_qkv_reason(const BnGPUBackend *gpu,
                                                            const BnQWeight *q,
                                                            const BnQWeight *k,
                                                            const BnQWeight *v,
                                                            const float *q_bias,
                                                            const float *k_bias,
                                                            const float *v_bias,
                                                            int q_bias_fused,
                                                            int k_bias_fused,
                                                            int v_bias_fused) {
    if (!gpu) return BN_BACKEND_LAYOUT_NO_GPU;
    if (!bn_gpu_backend_can_create_buffer(gpu))
        return BN_BACKEND_LAYOUT_NO_BUFFER_CREATE;
    BnBackendLayoutReason reason = bn_backend_layout_stackable_reason(q, k);
    if (reason != BN_BACKEND_LAYOUT_OK) return reason;
    reason = bn_backend_layout_stackable_reason(q, v);
    if (reason != BN_BACKEND_LAYOUT_OK) return reason;
    if (bn_qweight_data_size(q) == 0 ||
        bn_qweight_data_size(k) == 0 ||
        bn_qweight_data_size(v) == 0)
        return BN_BACKEND_LAYOUT_ZERO_SIZE;

    int any_bias = q_bias || k_bias || v_bias;
    int all_bias = q_bias && k_bias && v_bias;
    int all_fused = q_bias_fused && k_bias_fused && v_bias_fused;
    if (any_bias &&
        !(all_bias && all_fused &&
          bn_gpu_backend_can_create_biased_buffer(gpu)))
        return BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED;
    return BN_BACKEND_LAYOUT_OK;
}

void *bn_backend_layout_create_prefill_qk_pair(BnGPUBackend *gpu,
                                               void *q, void *k) {
    return bn_gpu_backend_create_prefill_qk_pair(gpu, q, k);
}

void *bn_backend_layout_upload_stacked2(BnGPUBackend *gpu,
                                        const BnQWeight *a,
                                        const BnQWeight *b) {
    if (bn_backend_layout_stacked2_reason(gpu, a, b) != BN_BACKEND_LAYOUT_OK)
        return NULL;
    size_t a_sz = bn_qweight_data_size(a);
    size_t b_sz = bn_qweight_data_size(b);

    if (bn_gpu_backend_can_create_stacked2_buffer(gpu)) {
        return bn_gpu_backend_create_stacked2_buffer(
            gpu, a->data, a_sz, b->data, b_sz, a->type,
            a->rows + b->rows, a->cols);
    }

    size_t combined_sz = a_sz + b_sz;
    uint8_t *combined = (uint8_t *)malloc(combined_sz);
    if (!combined) return NULL;

    memcpy(combined, a->data, a_sz);
    memcpy(combined + a_sz, b->data, b_sz);

    void *buf = upload_stacked_quant_layout(gpu, combined, combined_sz,
                                            a->type, a->rows + b->rows,
                                            a->cols);
    free(combined);
    return buf;
}

void *bn_backend_layout_upload_biased_qweight(BnGPUBackend *gpu,
                                              const BnQWeight *w,
                                              const float *bias) {
    if (bn_backend_layout_biased_qweight_reason(gpu, w, bias) != BN_BACKEND_LAYOUT_OK)
        return NULL;
    size_t sz = bn_qweight_data_size(w);
    return bn_gpu_backend_create_biased_buffer(
        gpu, w->data, sz, w->type, w->rows, w->cols, bias,
        (size_t)w->rows * sizeof(float));
}

void *bn_backend_layout_upload_stacked3_qkv(BnGPUBackend *gpu,
                                            const BnQWeight *q,
                                            const BnQWeight *k,
                                            const BnQWeight *v,
                                            const float *q_bias,
                                            const float *k_bias,
                                            const float *v_bias,
                                            int q_bias_fused,
                                            int k_bias_fused,
                                            int v_bias_fused) {
    if (bn_backend_layout_stacked3_qkv_reason(gpu, q, k, v,
                                              q_bias, k_bias, v_bias,
                                              q_bias_fused, k_bias_fused,
                                              v_bias_fused) != BN_BACKEND_LAYOUT_OK) {
        return NULL;
    }

    size_t q_sz = bn_qweight_data_size(q);
    size_t k_sz = bn_qweight_data_size(k);
    size_t v_sz = bn_qweight_data_size(v);

    int total_rows = q->rows + k->rows + v->rows;
    size_t combined_sz = q_sz + k_sz + v_sz;
    uint8_t *combined = (uint8_t *)malloc(combined_sz);
    if (!combined) return NULL;

    memcpy(combined, q->data, q_sz);
    memcpy(combined + q_sz, k->data, k_sz);
    memcpy(combined + q_sz + k_sz, v->data, v_sz);

    void *buf = NULL;
    int all_biased = q_bias && k_bias && v_bias &&
                     q_bias_fused && k_bias_fused && v_bias_fused;
    int no_bias = !q_bias && !k_bias && !v_bias;

    if (all_biased && bn_gpu_backend_can_create_biased_buffer(gpu)) {
        float *cbias = (float *)malloc((size_t)total_rows * sizeof(float));
        if (cbias) {
            memcpy(cbias, q_bias, (size_t)q->rows * sizeof(float));
            memcpy(cbias + q->rows, k_bias, (size_t)k->rows * sizeof(float));
            memcpy(cbias + q->rows + k->rows, v_bias, (size_t)v->rows * sizeof(float));
            if (bn_gpu_backend_can_create_stacked3_biased_buffer(gpu)) {
                buf = bn_gpu_backend_create_stacked3_biased_buffer(
                    gpu, q->data, q_sz, k->data, k_sz, v->data, v_sz,
                    q->type, total_rows, q->cols, cbias,
                    (size_t)total_rows * sizeof(float));
            } else {
                buf = bn_gpu_backend_create_biased_buffer(
                    gpu, combined, combined_sz, q->type, total_rows, q->cols,
                    cbias, (size_t)total_rows * sizeof(float));
            }
            free(cbias);
        }
    } else if (no_bias) {
        if (bn_gpu_backend_can_create_stacked3_buffer(gpu)) {
            buf = bn_gpu_backend_create_stacked3_buffer(
                gpu, q->data, q_sz, k->data, k_sz, v->data, v_sz,
                q->type, total_rows, q->cols);
        } else {
            buf = upload_stacked_quant_layout(gpu, combined, combined_sz,
                                              q->type, total_rows, q->cols);
        }
    }

    free(combined);
    return buf;
}

size_t bn_backend_layout_prepared_qweights_size(const BnConfig *config,
                                                const BnWeights *weights,
                                                BnBackendLayoutPreparedStats *stats) {
    BnBackendLayoutPreparedStats local = { 0 };
    if (!stats) stats = &local;
    prepared_stats_clear(stats);
    if (!config || !weights) return 0;

    for (int i = 0; i < config->n_layers; i++)
        prepared_qweight_size_layer(&weights->layers[i], stats);
    prepared_qweight_size_one(&weights->output_weight, stats);
    if (bn_backend_quant_tied_logits_uses_prepared_weight(
            weights->tied_embedding_weight.type))
        prepared_qweight_size_one(&weights->tied_embedding_weight, stats);

    return stats->lowbit_repack_bytes + stats->q8_repack_bytes +
           stats->kquant_scale_table_bytes +
           stats->expanded_kquant_weight_bytes +
           stats->f32_scale_table_bytes;
}

void bn_backend_layout_prepare_qweights(BnBackendModel *backend,
                                        const BnConfig *config,
                                        const BnWeights *weights,
                                        SHArena *arena,
                                        BnBackendLayoutPreparedStats *stats) {
    BnBackendLayoutPreparedStats local = { 0 };
    if (!backend || !config || !weights || !arena) {
        prepared_stats_clear(stats);
        return;
    }

    (void)bn_backend_layout_prepared_qweights_size(config, weights, &local);
    if (stats) {
        prepared_stats_clear(stats);
        prepared_stats_add(stats, &local);
    }

    for (int i = 0; i < config->n_layers; i++)
        prepared_qweight_prepare_layer(backend, &weights->layers[i], arena);
    prepared_qweight_prepare_one(backend, &weights->output_weight, arena);
    if (bn_backend_quant_tied_logits_uses_prepared_weight(
            weights->tied_embedding_weight.type))
        prepared_qweight_prepare_one(backend, &weights->tied_embedding_weight,
                                     arena);
}

/* HC preparation retains decoded weights only; raw uploads are temporary. */
void bn_backend_layout_bf16_hc_free(BnBackendBF16HCWeights *w) {
    if (!w) return;
    if (w->gpu) {
        bn_gpu_backend_destroy_buffer(w->gpu, w->inject.decoded);
        bn_gpu_backend_destroy_buffer(w->gpu, w->up.decoded);
        bn_gpu_backend_destroy_buffer(w->gpu, w->down.decoded);
        bn_gpu_backend_destroy_buffer(w->gpu, w->norm);
    }
    memset(w, 0, sizeof(*w));
}

typedef struct {
    BnQWeight sources[4];
    int count;
    size_t bytes, peak_bytes;
} BackendHCPlan;

static int backend_hc_plan(BnGPUBackend *gpu, const BnHyperConnectionWeights *w,
    int dim, int streams, int rank, int combine, BackendHCPlan *plan) {
    if (!bn_gpu_backend_can_create_quant_only_buffer(gpu) ||
        !bn_gpu_backend_can_destroy_buffer(gpu) || !w || !w->norm ||
        dim < 2048 || dim > 130560 || dim % 4 || streams < 2 || streams > 16 ||
        (streams & (streams - 1)) || rank <= 0 || dim > INT_MAX / streams ||
        (!combine && w->inject.data)) return -1;
    int wide = dim * streams;
    BackendHCPlan p = {0};
    p.sources[0] = (BnQWeight){w->norm, BN_GGUF_TENSOR_F32, 1, wide, 1.0f};
    p.sources[1] = w->down; p.sources[2] = w->up; p.sources[3] = w->inject;
    p.count = combine ? 4 : 3;
    int rows[] = {1, rank, wide, streams}, cols[] = {wide, wide, rank, wide};
    for (int i = 0; i < p.count; i++) {
        const BnQWeight *q = &p.sources[i];
        const BnQuantFormatOps *format = bn_quant_format_ops(q->type);
        if (!q->data || q->rows != rows[i] || q->cols != cols[i] || q->scale != 1.0f ||
            !format || format->block_elems <= 0 || !format->bytes_per_block ||
            q->cols % format->block_elems ||
            (!bn_quant_format_is_bf16(q->type) &&
             (format->bf16_decode == BN_QUANT_BF16_DECODE_NONE || !bn_gpu_backend_can_bf16_decode(gpu))) ||
            (size_t)q->rows > SIZE_MAX / (size_t)q->cols / 2) return -1;
        size_t decoded = (size_t)q->rows * q->cols * 2;
        if (((size_t)q->rows * q->cols / format->block_elems) >
            SIZE_MAX / format->bytes_per_block) return -1;
        size_t raw = bn_qweight_data_size(q);
        size_t extra = bn_quant_format_is_bf16(q->type) ? 0 : raw;
        if (!raw || decoded > SIZE_MAX - p.bytes || extra > SIZE_MAX - p.bytes - decoded)
            return -1;
        size_t peak = p.bytes + decoded + extra;
        if (peak > p.peak_bytes) p.peak_bytes = peak;
        p.bytes += decoded;
    }
    *plan = p;
    return 0;
}

/* Recover checkpoint gamma in F32 before BF16 rounding. The shared GGUF
 * array remains the effective scale expected by CPU execution. */
static void *backend_upload_zero_centered_norm(BnGPUBackend *gpu, const BnQWeight *q,
    float offset) {
    if (!offset)
        return bn_gpu_backend_create_quant_only_buffer(gpu, q->data,
            bn_qweight_data_size(q), q->type, q->rows, q->cols);
    if (q->type != BN_GGUF_TENSOR_F32 || q->rows <= 0 || q->cols <= 0 ||
        (size_t)q->rows > SIZE_MAX / sizeof(float) / (size_t)q->cols) return NULL;
    size_t count = (size_t)q->rows * q->cols;
    float *gamma = malloc(count * sizeof(*gamma));
    if (!gamma) return NULL;
    const float *stored = q->data;
    for (size_t j = 0; j < count; j++) gamma[j] = stored[j] - offset;
    void *raw = bn_gpu_backend_create_quant_only_buffer(gpu, gamma,
        count * sizeof(*gamma), q->type, q->rows, q->cols);
    free(gamma);
    return raw;
}

static int backend_hc_init(BnBackendBF16HCWeights *out, BnGPUBackend *gpu,
    const BnHyperConnectionWeights *w, int dim, int streams, int rank,
    size_t budget, int combine, float norm_offset) {
    if (!out || out->gpu || out->norm || out->down.raw || out->down.decoded ||
        out->up.raw || out->up.decoded || out->inject.raw || out->inject.decoded)
        return -1;
    BackendHCPlan plan;
    if (backend_hc_plan(gpu, w, dim, streams, rank, combine, &plan) ||
        plan.peak_bytes > budget) return -1;
    BnBackendBF16HCWeights prepared = {0};
    prepared.gpu = gpu; prepared.bytes = plan.bytes; prepared.peak_bytes = plan.peak_bytes;
    void **destinations[] = {&prepared.norm, &prepared.down.decoded,
        &prepared.up.decoded, &prepared.inject.decoded};
    for (int i = 0; i < plan.count; i++) {
        const BnQWeight *q = &plan.sources[i];
        void *raw = i == 0 ? backend_upload_zero_centered_norm(gpu, q, norm_offset) :
            bn_gpu_backend_create_quant_only_buffer(gpu, q->data,
                bn_qweight_data_size(q), q->type, q->rows, q->cols);
        if (!raw) goto fail;
        if (bn_quant_format_is_bf16(q->type)) *destinations[i] = raw;
        else {
            *destinations[i] = bn_gpu_backend_buffer_create_bf16_decoded(gpu, raw);
            bn_gpu_backend_destroy_buffer(gpu, raw);
            if (!*destinations[i]) goto fail;
        }
    }
    prepared.down.type = prepared.up.type = BN_GGUF_TENSOR_BF16;
    if (combine) prepared.inject.type = BN_GGUF_TENSOR_BF16;
    *out = prepared;
    return 0;
fail:
    bn_backend_layout_bf16_hc_free(&prepared);
    return -1;
}

int bn_backend_layout_bf16_hc_init(BnBackendBF16HCWeights *out,
    BnGPUBackend *gpu, const BnHyperConnectionWeights *w,
    int dim, int streams, int rank, size_t budget) {
    return backend_hc_init(out, gpu, w, dim, streams, rank, budget, 1, 0.0f);
}

int bn_backend_layout_bf16_hc_mixer_init(BnBackendBF16HCWeights *out,
    BnGPUBackend *gpu, const BnHyperConnectionWeights *w,
    int dim, int streams, int rank, size_t budget) {
    return backend_hc_init(out, gpu, w, dim, streams, rank, budget, 0, 0.0f);
}

/* PLE preparation keeps the merged projection shape and releases raw staging. */
typedef struct {
    BnQWeight sources[5]; /* First entry describes logical merged KV only. */
    size_t bytes, peak_bytes, host_staging_bytes;
} BackendPLEPlan;

static void backend_ple_free(BnBackendBF16PLEWeights *w) {
    if (!w) return;
    if (w->gpu) {
        bn_gpu_backend_destroy_buffer(w->gpu, w->conv);
        bn_gpu_backend_destroy_buffer(w->gpu, w->norm_conv);
        bn_gpu_backend_destroy_buffer(w->gpu, w->norm_query);
        bn_gpu_backend_destroy_buffer(w->gpu, w->norm_key);
        bn_gpu_backend_destroy_buffer(w->gpu, w->kv.decoded);
    }
    memset(w, 0, sizeof(*w));
}

static int backend_ple_plan(BnGPUBackend *gpu, const BnConfig *c,
    const BnPositionalLayerEmbeddingWeights *w, BackendPLEPlan *out) {
    if (!gpu || !c || !w || !bn_gpu_backend_can_create_quant_only_buffer(gpu) ||
        !bn_gpu_backend_can_create_stacked2_buffer(gpu) || !bn_gpu_backend_can_bf16_decode(gpu) ||
        !bn_gpu_backend_can_destroy_buffer(gpu) ||
        !bn_model_transformer_policy_uses_positional_layer_embedding(c) ||
        (c->dim != 2560 && c->dim != 4096) || c->hyper_connection_count != 4 ||
        c->ple_conv_kernel < 2 || c->ple_conv_kernel > 4 ||
        c->ple_ngram_size < 1 || c->ple_ngram_size > 8 ||
        !w->norm_key || !w->norm_query || !w->norm_conv || !w->conv1d ||
        !w->key.data || !w->value.data || w->key.type != BN_GGUF_TENSOR_Q8_0 ||
        w->value.type != w->key.type || w->key.scale != 1.0f || w->value.scale != 1.0f ||
        w->key.rows != c->dim * 4 || w->key.cols != c->dim ||
        w->value.rows != c->dim || w->value.cols != c->dim) return -1;
    BackendPLEPlan p = {0};
    p.sources[0] = w->key; p.sources[0].rows += w->value.rows;
    p.sources[1] = bn_quant_f32_weight(w->norm_key, 4, c->dim);
    p.sources[2] = bn_quant_f32_weight(w->norm_query, 4, c->dim);
    p.sources[3] = bn_quant_f32_weight(w->norm_conv, 4, c->dim);
    p.sources[4] = bn_quant_f32_weight(w->conv1d, c->dim * 4, c->ple_conv_kernel);
    for (int i = 0; i < 5; i++) {
        const BnQWeight *q = &p.sources[i];
        const BnQuantFormatOps *format = bn_quant_format_ops(q->type);
        if (!format || format->bf16_decode == BN_QUANT_BF16_DECODE_NONE ||
            format->block_elems <= 0 || !format->bytes_per_block ||
            q->cols % format->block_elems ||
            (size_t)q->rows > SIZE_MAX / (size_t)q->cols / 2) return -1;
        size_t decoded = (size_t)q->rows * q->cols * 2;
        if ((size_t)q->rows * q->cols / format->block_elems > SIZE_MAX / format->bytes_per_block)
            return -1;
        size_t raw = bn_qweight_data_size(q);
        if (!raw || decoded > SIZE_MAX - p.bytes || raw > SIZE_MAX - p.bytes - decoded)
            return -1;
        size_t peak = p.bytes + decoded + raw;
        if (peak > p.peak_bytes) p.peak_bytes = peak;
        p.bytes += decoded;
        if (!i) p.host_staging_bytes = raw;
    }
    *out = p;
    return 0;
}

typedef struct {
    BnQWeight sources[7]; /* Logical merged QKVZ, out, merged BA, conv/norm/dt/A. */
    size_t bytes, peak_bytes, host_staging_bytes;
} BackendFlashGDNPlan;

static void backend_flash_gdn_free(BnGPUBackend *gpu, BnBackendBF16FlashGDNModel *m) {
    BnBackendBF16FlashGDNWeights *layers = (BnBackendBF16FlashGDNWeights *)m->layers;
    if (layers) for (int l = 0; l < m->n_layers; l++) {
        BnBackendBF16FlashGDNWeights *w = &layers[l];
        void *handles[] = {w->qkvz.decoded,w->out.decoded,w->ba.decoded,w->conv,w->norm,w->dt_bias,w->a_log};
        for (int i = 0; i < 7; i++) bn_gpu_backend_destroy_buffer(gpu, handles[i]);
    }
    free(layers);
    memset(m, 0, sizeof(*m));
}

static int backend_flash_gdn_plan(const BnSSMWeights *w, BackendFlashGDNPlan *out) {
    if (!w || !w->ssm_a || !w->ssm_conv1d || !w->ssm_norm || !w->ssm_dt_bias) return -1;
    const BnQWeight *inputs[] = {&w->wqkv,&w->wz,&w->ssm_out,&w->ssm_beta,&w->ssm_alpha};
    int rows[] = {10240,6144,2560,48,48},cols[] = {2560,2560,6144,2560,2560};
    for (int i = 0; i < 5; i++)
        if (!inputs[i]->data || inputs[i]->rows != rows[i] || inputs[i]->cols != cols[i] ||
            inputs[i]->scale != 1.0f || inputs[i]->type !=
            (i < 3 ? BN_GGUF_TENSOR_Q8_0 : BN_GGUF_TENSOR_F32)) return -1;
    for (int h = 0; h < 48; h++) if (!(w->ssm_a[h] < 0.0f) || !isfinite(w->ssm_a[h])) return -1;
    BackendFlashGDNPlan p = {0};
    p.sources[0] = w->wqkv; p.sources[0].rows += w->wz.rows;
    p.sources[1] = w->ssm_out;
    p.sources[2] = w->ssm_beta; p.sources[2].rows += w->ssm_alpha.rows;
    p.sources[3] = bn_quant_f32_weight(w->ssm_conv1d,10240,4);
    p.sources[4] = bn_quant_f32_weight(w->ssm_norm,1,128);
    p.sources[5] = bn_quant_f32_weight(w->ssm_dt_bias,48,1);
    p.sources[6] = bn_quant_f32_weight(w->ssm_a,48,1);
    for (int i = 0; i < 7; i++) {
        const BnQWeight *q = &p.sources[i];
        const BnQuantFormatOps *format = bn_quant_format_ops(q->type);
        if (!format || format->bf16_decode == BN_QUANT_BF16_DECODE_NONE ||
            format->block_elems <= 0 || q->cols % format->block_elems) return -1;
        size_t raw = bn_qweight_data_size(q);
        size_t decoded = i == 6 ? raw : (size_t)q->rows * q->cols * 2;
        if (!raw || decoded > SIZE_MAX - p.bytes ||
            (i != 6 && raw > SIZE_MAX - p.bytes - decoded)) return -1;
        size_t peak = p.bytes + decoded + (i == 6 ? 0 : raw);
        if (peak > p.peak_bytes) p.peak_bytes = peak;
        p.bytes += decoded;
        if ((i == 0 || i == 2) && raw > p.host_staging_bytes) p.host_staging_bytes = raw;
    }
    *out = p;
    return 0;
}

static int backend_flash_gdn_init(BnGPUBackend *gpu, const BnSSMWeights *w,
    const BackendFlashGDNPlan *plan, BnBackendBF16FlashGDNWeights *out) {
    BnBackendBF16FlashGDNWeights p = {0};
    void **dst[] = {&p.qkvz.decoded,&p.out.decoded,&p.ba.decoded,&p.conv,&p.norm,&p.dt_bias};
    for (int i = 0; i < 6; i++) {
        const BnQWeight *q = &plan->sources[i];
        const BnQWeight *first = i == 0 ? &w->wqkv : &w->ssm_beta;
        const BnQWeight *second = i == 0 ? &w->wz : &w->ssm_alpha;
        void *raw = (i == 0 || i == 2) ? bn_gpu_backend_create_stacked2_buffer(gpu,
            first->data,bn_qweight_data_size(first),second->data,bn_qweight_data_size(second),
            q->type,q->rows,q->cols) : bn_gpu_backend_create_quant_only_buffer(gpu,
                q->data,bn_qweight_data_size(q),q->type,q->rows,q->cols);
        if (!raw) goto fail;
        *dst[i] = bn_gpu_backend_buffer_create_bf16_decoded(gpu,raw);
        bn_gpu_backend_destroy_buffer(gpu,raw);
        if (!*dst[i]) goto fail;
    }
    float values[48];
    for (int h = 0; h < 48; h++) values[h] = -w->ssm_a[h];
#ifdef BN_ENABLE_MKL_VML
    vmsLn(48,values,values,VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE);
#else
    goto fail;
#endif
    p.a_log = bn_gpu_backend_create_quant_only_buffer(gpu,values,sizeof(values),BN_GGUF_TENSOR_F32,48,1);
    if (!p.a_log) goto fail;
    p.qkvz.type = p.ba.type = p.out.type = BN_GGUF_TENSOR_BF16; p.valid = 1;
    *out = p;
    return 0;
fail:;
    void *handles[] = {p.qkvz.decoded,p.out.decoded,p.ba.decoded,p.conv,p.norm,p.dt_bias,p.a_log};
    for (int i = 0; i < 7; i++) bn_gpu_backend_destroy_buffer(gpu,handles[i]);
    return -1;
}

typedef struct {
    BnQWeight sources[7]; /* Merged QKV, merged indexer, output, four norms. */
    size_t bytes, peak_bytes, host_staging_bytes;
} BackendFlashQSAPlan;

static void backend_flash_qsa_free(BnGPUBackend *gpu,BnBackendBF16FlashQSAModel *m) {
    BnBackendBF16FlashQSAWeights *layers=(BnBackendBF16FlashQSAWeights*)m->layers;
    if(layers)for(int l=0;l<m->n_layers;l++) {
        BnBackendBF16FlashQSAWeights *w=&layers[l];
        void *handles[]={w->qkv.decoded,w->index.decoded,w->out.decoded,w->q_norm,
                         w->k_norm,w->index_q_norm,w->index_k_norm};
        for(int i=0;i<7;i++)bn_gpu_backend_destroy_buffer(gpu,handles[i]);
    }
    bn_gpu_backend_destroy_buffer(gpu,m->rotary_cache);
    free(layers);memset(m,0,sizeof(*m));
}
static int backend_flash_qsa_plan(const BnLayerWeights *w,BackendFlashQSAPlan *out) {
    const BnAttentionWeights *a=&w->attn;const BnAttentionIndexerWeights *ix=&w->indexer;
    if(a->head_size!=256||a->n_kv_heads!=2||a->kv_dim!=512||a->kv_mul!=12||
       a->q_dim!=12288||!a->has_kv||a->q_bias||a->k_bias||a->v_bias||
       !a->q_norm||!a->k_norm||!ix->q_norm||!ix->k_norm)return -1;
    const BnQWeight *inputs[]={&a->wq,&a->wk,&a->wv,&a->wo,&ix->q,&ix->k};
    int rows[]={12288,512,512,2560,512,128},cols[]={2560,2560,2560,6144,2560,2560};
    for(int i=0;i<6;i++)if(!inputs[i]->data||inputs[i]->rows!=rows[i]||
        inputs[i]->cols!=cols[i]||inputs[i]->scale!=1.0f||inputs[i]->type!=
        (i<4?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_BF16))return -1;
    BackendFlashQSAPlan p={0};
    p.sources[0]=a->wq;p.sources[0].rows+=a->wk.rows+a->wv.rows;
    p.sources[1]=ix->q;p.sources[1].rows+=ix->k.rows;p.sources[2]=a->wo;
    p.sources[3]=bn_quant_f32_weight(a->q_norm,1,256);
    p.sources[4]=bn_quant_f32_weight(a->k_norm,1,256);
    p.sources[5]=bn_quant_f32_weight(ix->q_norm,1,128);
    p.sources[6]=bn_quant_f32_weight(ix->k_norm,1,128);
    for(int i=0;i<7;i++) {
        const BnQWeight *q=&p.sources[i];const BnQuantFormatOps *fmt=bn_quant_format_ops(q->type);
        if(!fmt||fmt->block_elems<=0||q->cols%fmt->block_elems||
           (!bn_quant_format_is_bf16(q->type)&&fmt->bf16_decode==BN_QUANT_BF16_DECODE_NONE))return -1;
        size_t raw=bn_qweight_data_size(q),decoded=(size_t)q->rows*q->cols*2;
        size_t extra=bn_quant_format_is_bf16(q->type)?0:raw;
        if(!raw||decoded>SIZE_MAX-p.bytes||extra>SIZE_MAX-p.bytes-decoded)return -1;
        size_t peak=p.bytes+decoded+extra;if(peak>p.peak_bytes)p.peak_bytes=peak;
        p.bytes+=decoded;
        size_t host=i<2?raw:(i>=3?(size_t)q->cols*sizeof(float):0);
        if(host>p.host_staging_bytes)p.host_staging_bytes=host;
    }
    *out=p;return 0;
}
static int backend_flash_qsa_init(BnGPUBackend *gpu,const BnLayerWeights *w,
    const BackendFlashQSAPlan *plan,float norm_offset,BnBackendBF16FlashQSAWeights *out) {
    BnBackendBF16FlashQSAWeights p={0};
    void **dst[]={&p.qkv.decoded,&p.index.decoded,&p.out.decoded,&p.q_norm,
                 &p.k_norm,&p.index_q_norm,&p.index_k_norm};
    for(int i=0;i<7;i++) {
        const BnQWeight *q=&plan->sources[i];void *raw;
        if(i==0)raw=bn_gpu_backend_create_stacked3_buffer(gpu,
            w->attn.wq.data,bn_qweight_data_size(&w->attn.wq),
            w->attn.wk.data,bn_qweight_data_size(&w->attn.wk),
            w->attn.wv.data,bn_qweight_data_size(&w->attn.wv),q->type,q->rows,q->cols);
        else if(i==1)raw=bn_gpu_backend_create_stacked2_buffer(gpu,
            w->indexer.q.data,bn_qweight_data_size(&w->indexer.q),
            w->indexer.k.data,bn_qweight_data_size(&w->indexer.k),q->type,q->rows,q->cols);
        else if(i>=3)raw=backend_upload_zero_centered_norm(gpu,q,norm_offset);
        else raw=bn_gpu_backend_create_quant_only_buffer(gpu,q->data,bn_qweight_data_size(q),q->type,q->rows,q->cols);
        if(!raw)goto fail;
        if(bn_quant_format_is_bf16(q->type))*dst[i]=raw;
        else {
            *dst[i]=bn_gpu_backend_buffer_create_bf16_decoded(gpu,raw);
            bn_gpu_backend_destroy_buffer(gpu,raw);if(!*dst[i])goto fail;
        }
    }
    p.qkv.type=p.index.type=p.out.type=BN_GGUF_TENSOR_BF16;p.valid=1;*out=p;return 0;
fail:;
    void *handles[]={p.qkv.decoded,p.index.decoded,p.out.decoded,p.q_norm,p.k_norm,p.index_q_norm,p.index_k_norm};
    for(int i=0;i<7;i++)bn_gpu_backend_destroy_buffer(gpu,handles[i]);
    return -1;
}

/* Explicit BF16 layouts are independent of the legacy F32 weight registry. */
static void backend_flash_ffn_free(BnGPUBackend *gpu, BnBackendBF16FlashFFNModel *m) {
    BnBackendBF16FlashFFNWeights *layers = (BnBackendBF16FlashFFNWeights *)m->layers;
    for (int l = 0; layers && l < m->n_layers; l++) {
        BnBackendBF16FlashFFNWeights *w = &layers[l];
        void *handles[] = {w->router.decoded, w->shared_router.decoded,
            w->shared.gate.raw, w->shared.up.raw, w->shared.down.raw};
        for (int i = 0; i < 5; i++) bn_gpu_backend_destroy_buffer(gpu, handles[i]);
    }
    free(layers);
    memset(m, 0, sizeof(*m));
}

/* Checked source anatomy only: no expert bytes are uploaded or repacked here. */
static int backend_flash_ffn_experts(const BnMoEExpertMap *m) {
    if (m->gate_up_fused || m->gate_rows != 640 || m->up_rows != 640 ||
        m->gate_cols != 2560 || m->up_cols != 2560 ||
        m->down_rows != 2560 || m->down_cols != 640 ||
        (m->gate_type != BN_GGUF_TENSOR_Q4_K && m->gate_type != BN_GGUF_TENSOR_Q5_K) ||
        m->up_type != m->gate_type ||
        (m->down_type != BN_GGUF_TENSOR_Q5_1 && m->down_type != BN_GGUF_TENSOR_Q8_0)) return -1;
    int types[] = {m->gate_type, m->up_type, m->down_type};
    size_t sizes[] = {m->expert_gate_bytes, m->expert_up_bytes, m->expert_down_bytes};
    size_t strides[] = {m->gate_stride, m->up_stride, m->down_stride};
    size_t offsets[] = {m->gate_offset, m->up_offset, m->down_offset};
    size_t total = 0;
    for (int i = 0; i < 3; i++) {
        const BnQuantFormatOps *q = bn_quant_format_ops(types[i]);
        if (!q || q->block_elems <= 0 || !q->bytes_per_block) return -1;
        size_t bytes = (size_t)2560 * 640 / q->block_elems * q->bytes_per_block;
        if (sizes[i] != bytes || strides[i] < bytes ||
            offsets[i] > SIZE_MAX - bytes || strides[i] > (SIZE_MAX - offsets[i] - bytes) / 511 ||
            bytes > SIZE_MAX - total) return -1;
        total += bytes;
    }
    return m->repacked && (m->expert_total_bytes != total || total > SIZE_MAX / 512) ? -1 : 0;
}

static int backend_flash_ffn_plan(const BnLayerWeights *w, size_t *bytes, size_t *peak) {
    if (w->ffn_kind != BN_LAYER_FFN_MOE || !w->moe.router_weight ||
        w->moe.router_scale || w->moe.expert_down_scale ||
        !w->shared.shared_expert_gate || w->shared.shared_expert_gate_type != BN_GGUF_TENSOR_F32 ||
        backend_flash_ffn_experts(&w->moe.expert_map)) return -1;
    const BnQWeight *shared[] = {&w->shared.shared_gate, &w->shared.shared_up, &w->shared.shared_down};
    int rows[] = {640, 640, 2560}, cols[] = {2560, 2560, 640};
    size_t live = (size_t)(512 + 1) * 2560 * 2;
    /* Router raw F32 + decoded BF16 coexist; release raw before next tensor. */
    size_t high = (size_t)512 * 2560 * 6;
    for (int i = 0; i < 3; i++) {
        const BnQWeight *q = shared[i];
        if (!q->data || q->type != BN_GGUF_TENSOR_Q8_0 || q->scale != 1.0f ||
            q->rows != rows[i] || q->cols != cols[i]) return -1;
        size_t raw = bn_qweight_data_size(q);
        if (!raw || raw > SIZE_MAX - live) return -1;
        live += raw;
    }
    *bytes = live; *peak = high > live ? high : live;
    return 0;
}

static int backend_flash_ffn_init(BnGPUBackend *gpu, const BnLayerWeights *source,
    BnBackendBF16FlashFFNWeights *out) {
    const float *router[] = {source->moe.router_weight, source->shared.shared_expert_gate};
    BnGPUBF16LinearWeight *decoded[] = {&out->router, &out->shared_router};
    int rows[] = {512, 1};
    for (int i = 0; i < 2; i++) {
        void *raw = bn_gpu_backend_create_quant_only_buffer(gpu, router[i],
            (size_t)rows[i] * 2560 * sizeof(float), BN_GGUF_TENSOR_F32, rows[i], 2560);
        if (!raw) return -1;
        decoded[i]->type = BN_GGUF_TENSOR_BF16;
        decoded[i]->decoded = bn_gpu_backend_buffer_create_bf16_decoded(gpu, raw);
        bn_gpu_backend_destroy_buffer(gpu, raw);
        if (!decoded[i]->decoded) return -1;
    }
    const BnQWeight *shared[] = {&source->shared.shared_gate, &source->shared.shared_up, &source->shared.shared_down};
    BnGPUBF16LinearWeight *quant[] = {&out->shared.gate, &out->shared.up, &out->shared.down};
    for (int i = 0; i < 3; i++) {
        const BnQWeight *q = shared[i];
        quant[i]->type = q->type;
        quant[i]->raw = bn_gpu_backend_create_quant_only_buffer(gpu, q->data,
            bn_qweight_data_size(q), q->type, q->rows, q->cols);
        if (!quant[i]->raw) return -1;
    }
    out->expert_map = source->moe.expert_map;
    return 0;
}

typedef struct {
    BnBackendBF16Model model;
    BnBackendBF16HCModel hc_model;
    BnBackendBF16HCWeights *hc_layers;
    BnBackendBF16HCWeights hc_output;
    BnBackendBF16PLEWeights ple;
    BnBackendBF16FlashGDNModel flash_gdn;
    BnBackendBF16FlashQSAModel flash_qsa;
    BnBackendBF16FlashFFNModel flash_ffn;
    BnBackendBF16FlashIOModel flash_io;
    int flash_complete;
    const BnWeights *source_weights;
    int hc_prepared;
    BnGPUBackend *gpu;
    BnBackendBF16Layer *layers;
    void **handles;
    size_t count, capacity;
    int planning;
} BackendBF16Resources;

static const char backend_bf16_owner = 0;

int bn_backend_layout_bf16_math_available(BnBackendBF16Math profile) {
    if (profile == BN_BACKEND_BF16_MATH_LIBC) return 1;
#ifdef BN_ENABLE_MKL_VML
    if (profile == BN_BACKEND_BF16_MATH_MKL_VML_HA) return 1;
#endif
    return 0;
}

static void backend_bf16_release(void *ptr) {
    BackendBF16Resources *r = ptr;
    if (!r) return;
    while (r->count)
        bn_gpu_backend_destroy_buffer(r->gpu, r->handles[--r->count]);
    bn_gpu_backend_destroy_buffer(r->gpu,r->flash_io.embedding);
    bn_gpu_backend_destroy_buffer(r->gpu,r->flash_io.output.raw);
    if (r->flash_ffn.expert_cache) r->gpu->expert_cache_destroy(r->flash_ffn.expert_cache);
    backend_flash_ffn_free(r->gpu, &r->flash_ffn);
    backend_flash_qsa_free(r->gpu, &r->flash_qsa);
    backend_flash_gdn_free(r->gpu, &r->flash_gdn);
    backend_ple_free(&r->ple);
    while (r->hc_prepared) bn_backend_layout_bf16_hc_free(&r->hc_layers[--r->hc_prepared]);
    bn_backend_layout_bf16_hc_free(&r->hc_output);
    free(r->hc_layers);
    free(r->handles);
    free(r->layers);
    free(r);
}

const BnBackendBF16Model *bn_backend_layout_bf16(const BnBackendModel *backend) {
    const BackendBF16Resources *r = bn_backend_model_gpu_resources(
        backend, &backend_bf16_owner);
    return r && r->model.layers ? &r->model : NULL;
}

const BnBackendBF16HCModel *bn_backend_layout_bf16_hc_model(const BnBackendModel *backend) {
    const BackendBF16Resources *r = bn_backend_model_gpu_resources(backend, &backend_bf16_owner);
    return r && r->hc_model.n_layers ? &r->hc_model : NULL;
}

int bn_backend_layout_prepare_bf16_hc_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *c, const BnWeights *w, size_t budget) {
    if (!backend || !c || !w || !w->layers || c->n_layers <= 0 ||
        c->n_layers > (INT_MAX - 1) / 2 ||
        !bn_model_transformer_policy_uses_hyper_connections(c) ||
        bn_backend_model_gpu_resources(backend, &backend_bf16_owner) ||
        (bn_backend_model_gpu_for_cpu_operations(backend) &&
         bn_backend_model_gpu_for_cpu_operations(backend) != gpu) ||
        (size_t)c->n_layers > SIZE_MAX / 2 / sizeof(BnBackendBF16HCWeights)) return -1;
    int count = c->n_layers * 2;
    size_t bytes = 0, peak = 0;
    for (int i = 0; i <= count; i++) {
        const BnHyperConnectionWeights *source = i == count ? &w->hc_output :
            (i < c->n_layers ? &w->layers[i].hc_attn : &w->layers[i - c->n_layers].hc_ffn);
        BackendHCPlan plan;
        if (backend_hc_plan(gpu, source, c->dim, c->hyper_connection_count,
                c->hyper_connection_rank, i != count, &plan) ||
            plan.peak_bytes > SIZE_MAX - bytes || plan.bytes > SIZE_MAX - bytes)
            return -1;
        if (bytes + plan.peak_bytes > peak) peak = bytes + plan.peak_bytes;
        bytes += plan.bytes;
    }
    if (peak > budget) return -1;
    BackendBF16Resources *r = calloc(1, sizeof(*r));
    if (!r) return -1;
    r->gpu = gpu;
    r->hc_layers = calloc((size_t)count, sizeof(*r->hc_layers));
    if (!r->hc_layers) goto fail;
    size_t prepared_bytes = 0;
    for (int i = 0; i <= count; i++) {
        const BnHyperConnectionWeights *source = i == count ? &w->hc_output :
            (i < c->n_layers ? &w->layers[i].hc_attn : &w->layers[i - c->n_layers].hc_ffn);
        BnBackendBF16HCWeights *out = i == count ? &r->hc_output : &r->hc_layers[i];
        if (backend_hc_init(out, gpu, source, c->dim, c->hyper_connection_count,
                c->hyper_connection_rank, budget - prepared_bytes, i != count,
                bn_model_backend_policy_zero_centered_norm_storage_offset(c))) goto fail;
        prepared_bytes += out->bytes;
        if (i != count) r->hc_prepared++;
    }
    r->source_weights = w;
    r->hc_model.n_layers = c->n_layers; r->hc_model.dim = c->dim;
    r->hc_model.streams = c->hyper_connection_count; r->hc_model.rank = c->hyper_connection_rank;
    r->hc_model.attention = r->hc_layers; r->hc_model.ffn = r->hc_layers + c->n_layers;
    r->hc_model.output = &r->hc_output;
    r->hc_model.bytes = bytes; r->hc_model.peak_bytes = peak;
    if (bn_backend_model_attach_gpu_resources(backend, &backend_bf16_owner,
            r, backend_bf16_release)) goto fail;
    return 0;
fail:
    backend_bf16_release(r);
    return -1;
}

const BnBackendBF16PLEWeights *bn_backend_layout_bf16_ple_model(const BnBackendModel *backend) {
    const BackendBF16Resources *r = bn_backend_model_gpu_resources(backend, &backend_bf16_owner);
    return r && r->ple.gpu ? &r->ple : NULL;
}

int bn_backend_layout_prepare_bf16_ple_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *c, const BnWeights *w,
    size_t gpu_budget, size_t host_budget) {
    BackendBF16Resources *r = bn_backend_model_gpu_resources(backend, &backend_bf16_owner);
    if (!r || !c || !w || !w->layers || r->gpu != gpu || r->ple.gpu ||
        r->model.layers || r->source_weights != w || r->hc_model.n_layers != c->n_layers ||
        r->hc_model.dim != c->dim || r->hc_model.streams != c->hyper_connection_count ||
        r->hc_model.rank != c->hyper_connection_rank || c->ple_layer < 0 ||
        c->ple_layer >= c->n_layers) return -1;
    const BnPositionalLayerEmbeddingWeights *source = &w->layers[c->ple_layer].ple;
    BackendPLEPlan plan;
    if (backend_ple_plan(gpu, c, source, &plan) || plan.host_staging_bytes > host_budget ||
        plan.peak_bytes > SIZE_MAX - r->hc_model.bytes ||
        r->hc_model.bytes + plan.peak_bytes > gpu_budget || r->hc_model.peak_bytes > gpu_budget)
        return -1;
    BnBackendBF16PLEWeights prepared = {0};
    prepared.gpu = gpu; prepared.layer = c->ple_layer; prepared.dim = c->dim;
    prepared.streams = c->hyper_connection_count; prepared.kernel = c->ple_conv_kernel;
    prepared.dilation = c->ple_ngram_size; prepared.bytes = plan.bytes;
    prepared.peak_bytes = plan.peak_bytes; prepared.host_staging_bytes = plan.host_staging_bytes;
    prepared.model_bytes = r->hc_model.bytes + plan.bytes;
    prepared.model_peak_bytes = r->hc_model.bytes + plan.peak_bytes;
    if (r->hc_model.peak_bytes > prepared.model_peak_bytes) prepared.model_peak_bytes = r->hc_model.peak_bytes;
    void **destinations[] = {&prepared.kv.decoded, &prepared.norm_key,
        &prepared.norm_query, &prepared.norm_conv, &prepared.conv};
    for (int i = 0; i < 5; i++) {
        const BnQWeight *q = &plan.sources[i];
        void *raw = i == 0 ? bn_gpu_backend_create_stacked2_buffer(gpu,
            source->key.data, bn_qweight_data_size(&source->key),
            source->value.data, bn_qweight_data_size(&source->value),
            q->type, q->rows, q->cols) :
            (i >= 1 && i <= 3) ? backend_upload_zero_centered_norm(gpu, q,
                bn_model_backend_policy_zero_centered_norm_storage_offset(c)) :
            bn_gpu_backend_create_quant_only_buffer(gpu, q->data,
                bn_qweight_data_size(q), q->type, q->rows, q->cols);
        if (!raw) goto fail;
        *destinations[i] = bn_gpu_backend_buffer_create_bf16_decoded(gpu, raw);
        bn_gpu_backend_destroy_buffer(gpu, raw);
        if (!*destinations[i]) goto fail;
    }
    prepared.kv.type = BN_GGUF_TENSOR_BF16;
    r->ple = prepared;
    return 0;
fail:
    backend_ple_free(&prepared);
    return -1;
}

const BnBackendBF16FlashGDNModel *bn_backend_layout_bf16_flash_gdn_model(const BnBackendModel *backend) {
    const BackendBF16Resources *r = bn_backend_model_gpu_resources(backend,&backend_bf16_owner);
    return r && r->flash_gdn.layers ? &r->flash_gdn : NULL;
}

int bn_backend_layout_prepare_bf16_flash_gdn_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *c, const BnWeights *w, size_t gpu_budget, size_t host_budget) {
    BackendBF16Resources *r = bn_backend_model_gpu_resources(backend,&backend_bf16_owner);
    if (!r || !c || !w || !w->layers || r->gpu != gpu || r->source_weights != w ||
        !r->ple.gpu || r->flash_gdn.layers || r->model.layers || c->dim != 2560 ||
        r->hc_model.n_layers != c->n_layers || r->hc_model.dim != c->dim ||
        r->hc_model.streams != c->hyper_connection_count ||
        !bn_model_transformer_policy_ssm_uses_sigmoid_gate(c) ||
        bn_model_transformer_policy_norm_epsilon(c) != 1e-6f ||
        c->ssm_state_size != 128 || c->ssm_conv_kernel != 4 || c->ssm_inner_size != 6144 ||
        c->ssm_time_step_rank != 48 || c->ssm_group_count != 16 || c->n_layers <= 0 ||
        (size_t)c->n_layers > SIZE_MAX / sizeof(BnBackendBF16FlashGDNWeights) ||
        !bn_backend_layout_bf16_math_available(BN_BACKEND_BF16_MATH_MKL_VML_HA) ||
        !bn_gpu_backend_can_create_quant_only_buffer(gpu) || !bn_gpu_backend_can_create_stacked2_buffer(gpu) ||
        !bn_gpu_backend_can_bf16_decode(gpu) || !bn_gpu_backend_can_destroy_buffer(gpu)) return -1;
    size_t bytes = 0,peak = 0,host = 0; int count = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (w->layers[l].block_kind != BN_LAYER_BLOCK_SSM) continue;
        BackendFlashGDNPlan p;
        if (backend_flash_gdn_plan(&w->layers[l].ssm,&p) || p.peak_bytes > SIZE_MAX - bytes ||
            p.bytes > SIZE_MAX - bytes) return -1;
        if (bytes + p.peak_bytes > peak) peak = bytes + p.peak_bytes;
        bytes += p.bytes; if (p.host_staging_bytes > host) host = p.host_staging_bytes;
        count++;
    }
    if (!count || peak > SIZE_MAX - r->ple.model_bytes ||
        r->ple.model_bytes + peak > gpu_budget || r->ple.model_peak_bytes > gpu_budget || host > host_budget)
        return -1;
    BnBackendBF16FlashGDNModel prepared = {0};
    prepared.n_layers = c->n_layers; prepared.n_gdn_layers = count; prepared.dim = c->dim;
    prepared.hk = 16; prepared.hv = 48; prepared.dk = prepared.dv = 128; prepared.kernel = 4;
    prepared.norm_eps = 1e-6f; prepared.gate_activation = BN_GPU_GATED_NORM_SIGMOID;
    prepared.math_profile = BN_BACKEND_BF16_MATH_MKL_VML_HA;
    prepared.bytes = bytes; prepared.peak_bytes = peak; prepared.host_staging_bytes = host;
    prepared.model_bytes = r->ple.model_bytes + bytes; prepared.model_peak_bytes = r->ple.model_bytes + peak;
    if (r->ple.model_peak_bytes > prepared.model_peak_bytes) prepared.model_peak_bytes = r->ple.model_peak_bytes;
    BnBackendBF16FlashGDNWeights *layers = calloc((size_t)c->n_layers,sizeof(*layers));
    if (!layers) return -1;
    prepared.layers = layers;
    for (int l = 0; l < c->n_layers; l++) {
        if (w->layers[l].block_kind != BN_LAYER_BLOCK_SSM) continue;
        BackendFlashGDNPlan p;
        if (backend_flash_gdn_plan(&w->layers[l].ssm,&p) ||
            backend_flash_gdn_init(gpu,&w->layers[l].ssm,&p,&layers[l])) {
            backend_flash_gdn_free(gpu,&prepared); return -1;
        }
    }
    r->flash_gdn = prepared;
    return 0;
}

const BnBackendBF16FlashQSAModel *bn_backend_layout_bf16_flash_qsa_model(const BnBackendModel *backend) {
    const BackendBF16Resources *r=bn_backend_model_gpu_resources(backend,&backend_bf16_owner);
    return r&&r->flash_qsa.layers?&r->flash_qsa:NULL;
}
int bn_backend_layout_prepare_bf16_flash_qsa_model(BnBackendModel *backend,
    BnGPUBackend *gpu,const BnConfig *c,const BnWeights *w,size_t gpu_budget,size_t host_budget) {
    BackendBF16Resources *r=bn_backend_model_gpu_resources(backend,&backend_bf16_owner);
    if(!r||!c||!w||!w->layers||r->gpu!=gpu||r->source_weights!=w||!r->flash_gdn.layers||
       r->flash_qsa.layers||r->model.layers||!bn_model_backend_policy_uses_query_sparse_attention(c)||
       c->n_layers<=0||c->n_layers>128||r->flash_gdn.n_layers!=c->n_layers||
       c->dim!=2560||c->n_heads!=24||c->n_kv_heads!=2||c->head_size!=256||c->qk_norm_per_head||
       c->indexer_head_count!=4||c->indexer_head_size!=128||c->indexer_top_k!=2048||
       c->rope_dim_count!=64||c->rope_theta!=10000000.0f||c->seq_len<=0||c->seq_len>262144||
       bn_model_transformer_policy_norm_epsilon(c)!=1e-6f||
       bn_model_transformer_policy_attention_scale(c,256)!=0.0625f||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_create_stacked2_buffer(gpu)||
       !bn_gpu_backend_can_create_stacked3_buffer(gpu)||!bn_gpu_backend_can_bf16_decode(gpu)||
       !bn_gpu_backend_can_create_rotary_bf16(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu))return -1;
    size_t bytes=0,peak=0,host=0;int count=0;
    for(int l=0;l<c->n_layers;l++) {
        if(w->layers[l].block_kind==BN_LAYER_BLOCK_SSM) {
            if(c->attention_compress_ratios[l])return -1;
            continue;
        }
        if(w->layers[l].block_kind!=BN_LAYER_BLOCK_ATTENTION||c->attention_compress_ratios[l]!=4)return -1;
        BackendFlashQSAPlan p;
        if(backend_flash_qsa_plan(&w->layers[l],&p)||p.bytes>SIZE_MAX-bytes||p.peak_bytes>SIZE_MAX-bytes)return -1;
        if(bytes+p.peak_bytes>peak)peak=bytes+p.peak_bytes;
        bytes+=p.bytes;if(p.host_staging_bytes>host)host=p.host_staging_bytes;count++;
    }
    size_t rotary=(size_t)c->seq_len*64*2;
    if(!count||rotary>SIZE_MAX-bytes)return -1;
    bytes+=rotary;if(bytes>peak)peak=bytes;
    size_t base=r->flash_gdn.model_bytes;
    if(peak>SIZE_MAX-base||base+peak>gpu_budget||r->flash_gdn.model_peak_bytes>gpu_budget||host>host_budget)return -1;
    BnBackendBF16FlashQSAModel prepared={0};
    prepared.n_layers=c->n_layers;prepared.n_qsa_layers=count;prepared.dim=2560;
    prepared.heads=24;prepared.kv_heads=2;prepared.head_size=256;
    prepared.index_heads=4;prepared.index_head_size=128;prepared.compress_ratio=4;prepared.rotary_dim=64;
    prepared.rotary_capacity=c->seq_len;prepared.rope_theta=c->rope_theta;prepared.norm_eps=1e-6f;
    prepared.bytes=bytes;prepared.peak_bytes=peak;prepared.host_staging_bytes=host;
    prepared.model_bytes=base+bytes;prepared.model_peak_bytes=base+peak;
    if(r->flash_gdn.model_peak_bytes>prepared.model_peak_bytes)prepared.model_peak_bytes=r->flash_gdn.model_peak_bytes;
    BnBackendBF16FlashQSAWeights *layers=calloc((size_t)c->n_layers,sizeof(*layers));
    if(!layers)return -1;
    prepared.layers=layers;
    for(int l=0;l<c->n_layers;l++) {
        if(w->layers[l].block_kind==BN_LAYER_BLOCK_SSM)continue;
        BackendFlashQSAPlan p;
        if(backend_flash_qsa_plan(&w->layers[l],&p)||backend_flash_qsa_init(gpu,&w->layers[l],&p,
            bn_model_backend_policy_zero_centered_norm_storage_offset(c),&layers[l]))goto fail;
    }
    BnGPURotaryCachePlan rp={c->seq_len,64,c->rope_theta};
    prepared.rotary_cache=bn_gpu_backend_create_rotary_bf16(gpu,&rp);
    if(!prepared.rotary_cache)goto fail;
    r->flash_qsa=prepared;return 0;
fail:
    backend_flash_qsa_free(gpu,&prepared);return -1;
}

const BnBackendBF16FlashFFNModel *bn_backend_layout_bf16_flash_ffn_model(
    const BnBackendModel *backend) {
    const BackendBF16Resources *r = bn_backend_model_gpu_resources(backend, &backend_bf16_owner);
    return r && r->flash_ffn.layers ? &r->flash_ffn : NULL;
}

int bn_backend_layout_prepare_bf16_flash_ffn_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *c, const BnWeights *w, size_t budget) {
    BackendBF16Resources *r = bn_backend_model_gpu_resources(backend, &backend_bf16_owner);
    if (!r || !c || !w || !w->layers || r->gpu != gpu || r->source_weights != w ||
        !r->flash_qsa.layers || r->flash_ffn.layers || r->model.layers ||
        r->flash_qsa.n_layers != c->n_layers || c->n_layers <= 0 || c->n_layers > 64 ||
        c->dim != 2560 || c->hyper_connection_count != 4 || c->n_experts != 512 ||
        c->n_experts_active != 10 || c->moe_intermediate_size != 640 ||
        c->shared_expert_intermediate_size != 640 || !c->has_shared_expert ||
        c->moe_norm_topk_prob != 1 ||
        (c->moe_expert_weights_scale != 0.0f && c->moe_expert_weights_scale != 1.0f) ||
        !bn_gpu_backend_can_create_quant_only_buffer(gpu) || !bn_gpu_backend_can_bf16_decode(gpu) ||
        !bn_gpu_backend_can_destroy_buffer(gpu)) return -1;
    size_t bytes = 0, peak = 0;
    for (int l = 0; l < c->n_layers; l++) {
        size_t layer_bytes, layer_peak;
        if (backend_flash_ffn_plan(&w->layers[l], &layer_bytes, &layer_peak) ||
            layer_peak > SIZE_MAX - bytes || layer_bytes > SIZE_MAX - bytes) return -1;
        if (bytes + layer_peak > peak) peak = bytes + layer_peak;
        bytes += layer_bytes;
    }
    size_t base = r->flash_qsa.model_bytes;
    if (peak > SIZE_MAX - base || base + peak > budget || r->flash_qsa.model_peak_bytes > budget) return -1;
    BnBackendBF16FlashFFNModel prepared = {0};
    prepared.n_layers = c->n_layers; prepared.dim = 2560;
    prepared.hidden_dim = prepared.shared_hidden_dim = 640;
    prepared.n_experts = 512; prepared.k = 10; prepared.renormalize = 1;
    prepared.shared_activation = BN_GPU_BF16_FFN_SILU_FUSED;
    prepared.bytes = bytes; prepared.peak_bytes = peak;
    prepared.model_bytes = base + bytes; prepared.model_peak_bytes = base + peak;
    if (r->flash_qsa.model_peak_bytes > prepared.model_peak_bytes)
        prepared.model_peak_bytes = r->flash_qsa.model_peak_bytes;
    BnBackendBF16FlashFFNWeights *layers = calloc((size_t)c->n_layers, sizeof(*layers));
    if (!layers) return -1;
    prepared.layers = layers;
    for (int l = 0; l < c->n_layers; l++) {
        if (backend_flash_ffn_init(gpu, &w->layers[l], &layers[l])) {
            backend_flash_ffn_free(gpu, &prepared); return -1;
        }
    }
    r->flash_ffn = prepared;
    return 0;
}

const BnBackendBF16FlashIOModel *bn_backend_layout_bf16_flash_io_model(
    const BnBackendModel *backend) {
    const BackendBF16Resources *r=bn_backend_model_gpu_resources(backend,&backend_bf16_owner);
    return r&&r->flash_io.embedding?&r->flash_io:NULL;
}

int bn_backend_layout_prepare_bf16_flash_io_model(BnBackendModel *backend,
    BnGPUBackend *gpu,const BnConfig *c,const BnWeights *w,size_t budget) {
    BackendBF16Resources *r=bn_backend_model_gpu_resources(backend,&backend_bf16_owner);
    const BnQuantFormatOps *format=bn_quant_format_ops(BN_GGUF_TENSOR_Q8_0);
    if(!r||!c||!w||r->gpu!=gpu||r->source_weights!=w||!r->flash_ffn.layers||
       r->flash_io.embedding||r->model.layers||c->dim!=2560||c->vocab_size!=248320||
       r->flash_ffn.n_layers!=c->n_layers||!w->token_embedding||w->emb_type!=BN_GGUF_TENSOR_Q8_0||
       !w->output_weight.data||w->output_weight.type!=BN_GGUF_TENSOR_Q8_0||
       w->output_weight.rows!=c->vocab_size||w->output_weight.cols!=c->dim||
       w->output_weight.scale!=1.0f||w->output_norm||!r->hc_model.output||
       r->hc_model.output->inject.raw||r->hc_model.output->inject.decoded||
       !format||format->bf16_embedding!=BN_QUANT_BF16_EMBEDDING_Q8_0||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu))return -1;
    size_t raw=bn_qweight_data_size(&w->output_weight),base=r->flash_ffn.model_bytes;
    if(!raw||raw>SIZE_MAX/2||raw*2>SIZE_MAX-base||base+raw*2>budget||
       r->flash_ffn.model_peak_bytes>budget)return -1;
    BnBackendBF16FlashIOModel p={0};p.dim=c->dim;p.vocab_size=c->vocab_size;
    p.bytes=p.peak_bytes=raw*2;p.model_bytes=base+p.bytes;p.model_peak_bytes=p.model_bytes;
    if(r->flash_ffn.model_peak_bytes>p.model_peak_bytes)p.model_peak_bytes=r->flash_ffn.model_peak_bytes;
    p.embedding=bn_gpu_backend_create_quant_only_buffer(gpu,w->token_embedding,raw,w->emb_type,c->vocab_size,c->dim);
    if(!p.embedding)return -1;
    p.output.type=BN_GGUF_TENSOR_Q8_0;
    p.output.raw=bn_gpu_backend_create_quant_only_buffer(gpu,w->output_weight.data,raw,w->output_weight.type,c->vocab_size,c->dim);
    if(!p.output.raw){bn_gpu_backend_destroy_buffer(gpu,p.embedding);return -1;}
    r->flash_io=p;return 0;
}

const BnBackendBF16FlashIOModel *bn_backend_layout_bf16_flash_model(
    const BnBackendModel *backend) {
    const BackendBF16Resources *r = bn_backend_model_gpu_resources(backend,
        &backend_bf16_owner);
    return r && r->flash_complete ? &r->flash_io : NULL;
}

int bn_backend_layout_prepare_bf16_flash_model(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *c, const BnWeights *w,
    size_t gpu_budget, size_t host_budget) {
    return bn_backend_layout_prepare_bf16_flash_model_with_cache(backend,gpu,c,w,
        gpu_budget,host_budget,0);
}

int bn_backend_layout_prepare_bf16_flash_model_with_cache(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *c, const BnWeights *w,
    size_t gpu_budget, size_t host_budget, size_t cache_budget) {
    if (!backend || !gpu || !c || !w || !w->layers ||
        c->bf16_layout != BN_MODEL_BF16_LAYOUT_FLASH_HC ||
        c->n_layers != 48 || c->dim != 2560 ||
        c->hyper_connection_count != 4 || c->hyper_connection_rank != 320 ||
        bn_backend_model_gpu_for_cpu_operations(backend) ||
        bn_backend_model_gpu_resources(backend, &backend_bf16_owner) ||
        !bn_backend_layout_bf16_math_available(BN_BACKEND_BF16_MATH_MKL_VML_HA) ||
        host_budget < (size_t)c->dim * c->hyper_connection_count * sizeof(float))
        return -1;
    if (cache_budget && (!gpu->expert_cache_create || !gpu->expert_cache_memory || !gpu->expert_cache_destroy ||
        !gpu->expert_cache_bind || !gpu->expert_cache_begin || !gpu->expert_cache_end ||
        !gpu->expert_cache_cancel || !gpu->expert_cache_stats)) return -1;
    BnBackendModel *staged = bn_backend_model_create();
    if (!staged) return -1;
    int status = -1;
    if (bn_backend_layout_prepare_bf16_hc_model(staged, gpu, c, w, gpu_budget) ||
        bn_backend_layout_prepare_bf16_ple_model(staged, gpu, c, w, gpu_budget, host_budget) ||
        bn_backend_layout_prepare_bf16_flash_gdn_model(staged, gpu, c, w, gpu_budget, host_budget) ||
        bn_backend_layout_prepare_bf16_flash_qsa_model(staged, gpu, c, w, gpu_budget, host_budget) ||
        bn_backend_layout_prepare_bf16_flash_ffn_model(staged, gpu, c, w, gpu_budget) ||
        bn_backend_layout_prepare_bf16_flash_io_model(staged, gpu, c, w, gpu_budget))
        goto done;
    BackendBF16Resources *r = bn_backend_model_gpu_resources(staged, &backend_bf16_owner);
    if (cache_budget) {
        int gate[48], down[48];
        for (int l=0;l<48;l++) {
            gate[l]=r->flash_ffn.layers[l].expert_map.gate_type;
            down[l]=r->flash_ffn.layers[l].expert_map.down_type;
        }
        BnGPUExpertCacheConfig cache={48,512,c->dim,c->moe_intermediate_size,gate,down,cache_budget};
        r->flash_ffn.expert_cache=gpu->expert_cache_create(gpu->ctx,&cache);
        if (!r->flash_ffn.expert_cache) goto done;
        BnGPUExpertCacheMemory memory;
        if(gpu->expert_cache_memory(r->flash_ffn.expert_cache,&memory)||
           memory.host_bytes>host_budget||memory.device_bytes>cache_budget)goto done;
        r->flash_ffn.expert_cache_host_bytes=memory.host_bytes;
    }
    r->flash_complete = 1;
    bn_backend_model_bind_gpu(staged, gpu);
    status = bn_backend_model_move_gpu_resources(backend, staged);
done:
    bn_backend_model_free(staged);
    return status;
}

static void *backend_bf16_own(BackendBF16Resources *r, void *handle) {
    if (!handle) return NULL;
    if (r->count == r->capacity) {
        bn_gpu_backend_destroy_buffer(r->gpu, handle);
        return NULL;
    }
    r->handles[r->count++] = handle;
    return handle;
}

/* Original repetition-major V heads become key-head-major groups. */
static int backend_bf16_source_row(int row, int base, int width) {
    if (row < base) return row;
    int i = row - base, head = i / width;
    return base + ((head % 3) * 16 + head / 3) * width + i % width;
}

static uint16_t backend_bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static const char backend_bf16_plan_handle = 0;
static int backend_bf16_plan_add(BackendBF16Resources *r, size_t device,
    size_t staging, size_t temporary_device) {
    BnBackendBF16MemoryPlan *p = &r->model.memory;
    if (device > SIZE_MAX - p->device_bytes ||
        temporary_device > SIZE_MAX - p->device_bytes - device ||
        staging > SIZE_MAX - p->host_bytes) return -1;
    size_t peak = p->device_bytes + device + temporary_device;
    if (peak > p->device_peak_bytes) p->device_peak_bytes = peak;
    p->device_bytes += device;
    peak = p->host_bytes + staging;
    if (peak > p->host_peak_bytes) p->host_peak_bytes = peak;
    return 0;
}

static void *backend_bf16_floats(BackendBF16Resources *r, const float *src,
    int rows, int columns, int offset, int base, int width, int logarithm) {
    size_t count = (size_t)rows * (size_t)columns;
    size_t bytes = count * (logarithm ? sizeof(float) : sizeof(uint16_t));
    if (r->planning) return backend_bf16_plan_add(r, bytes, bytes, 0)
        ? NULL : (void *)&backend_bf16_plan_handle;
    void *data = malloc(bytes);
    if (!data) return NULL;
    for (int row = 0; row < rows; ++row) {
        int old = base < 0 ? row : backend_bf16_source_row(row, base, width);
        for (int col = 0; col < columns; ++col) {
            float value = src[(size_t)old * columns + col];
            size_t i = (size_t)row * columns + col;
            if (logarithm) ((float *)data)[i] = -value;
            else ((uint16_t *)data)[i] = backend_bf16_round(value - (float)offset);
        }
    }
    if (logarithm) {
#ifdef BN_ENABLE_MKL_VML
        if (r->model.math_profile == BN_BACKEND_BF16_MATH_MKL_VML_HA) {
            /* Per-call mode: never change the process-wide VML settings. */
            vmsLn((MKL_INT)count, data, data,
                  VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);
        } else
#endif
        {
            for (size_t i = 0; i < count; ++i)
                ((float *)data)[i] = logf(((float *)data)[i]);
        }
    }
    void *handle = bn_gpu_backend_create_quant_only_buffer(r->gpu, data, bytes,
        logarithm ? BN_GGUF_TENSOR_F32 : BN_GGUF_TENSOR_BF16, rows, columns);
    free(data);
    return backend_bf16_own(r, handle);
}

static int backend_bf16_weight_valid(const BnQWeight *w, int rows, int cols,
    int max_tokens) {
    BnBackendBF16MatmulPlan plan;
    return w->data && w->rows == rows && w->cols == cols && w->scale == 1.0f &&
        bn_backend_quant_bf16_matmul_plan(w->type, rows, cols, 1, &plan) == 0 &&
        bn_backend_quant_bf16_matmul_plan(w->type, rows, cols, max_tokens, &plan) == 0;
}

static int backend_bf16_linear(BackendBF16Resources *r, const BnQWeight *w,
    int base, int width, BnGPUBF16LinearWeight *out) {
    size_t bytes = bn_qweight_data_size(w);
    if (r->planning) {
        BnBackendBF16MatmulPlan plan;
        if (!bytes || bn_backend_quant_bf16_matmul_plan(w->type, w->rows,
                w->cols, r->model.max_tokens, &plan) ||
            plan.decoded_weight_bytes > SIZE_MAX - bytes ||
            backend_bf16_plan_add(r, bytes + plan.decoded_weight_bytes,
                base >= 0 ? bytes : 0, 0)) return -1;
        out->type = w->type; out->raw = (void *)&backend_bf16_plan_handle;
        out->decoded = plan.decoded_weight_bytes ? out->raw : NULL;
        return 0;
    }
    const void *data = w->data;
    void *copy = NULL;
    if (base >= 0) {
        copy = malloc(bytes);
        if (!copy) return -1;
        size_t stride = bytes / (size_t)w->rows;
        for (int i = 0; i < w->rows; ++i)
            memcpy((char *)copy + (size_t)i * stride,
                (const char *)data + (size_t)backend_bf16_source_row(i, base, width) * stride,
                stride);
        data = copy;
    }
    out->type = w->type;
    out->raw = backend_bf16_own(r, bn_gpu_backend_create_quant_only_buffer(r->gpu,
        data, bytes, w->type, w->rows, w->cols));
    free(copy);
    if (!out->raw) return -1;
    BnBackendBF16MatmulPlan plan;
    if (bn_backend_quant_bf16_matmul_plan(w->type, w->rows, w->cols,
            r->model.max_tokens, &plan) != 0) return -1;
    if (plan.kind == BN_BACKEND_BF16_MATMUL_DECODED_GEMM) {
        out->decoded = backend_bf16_own(r,
            bn_gpu_backend_buffer_create_bf16_decoded(r->gpu, out->raw));
        if (!out->decoded) return -1;
    }
    return 0;
}

static int backend_bf16_stack_ffn(BackendBF16Resources *r,BnBackendBF16Layer *b) {
    if(!(r->gpu->resident_ffn_layouts&(1u<<BN_GPU_BF16_FFN_STACKED_DECODED))||
       !bn_gpu_backend_can_stack_bf16_pair(r->gpu)||b->ffn.gate.type!=b->ffn.up.type||
       !b->ffn.gate.decoded||!b->ffn.up.decoded)return 0;
    if (r->planning) {
        /* The joined parent is allocated before both old decoded allocations
         * are destroyed; its row views borrow the parent without payload. */
        if (backend_bf16_plan_add(r, 0, 0, 2u * 17408 * 5120 * sizeof(uint16_t))) return -1;
        r->model.stacked_ffn_layers++;return 0;
    }
    size_t gate=0,up=0;
    while(gate<r->count&&r->handles[gate]!=b->ffn.gate.decoded)gate++;
    while(up<r->count&&r->handles[up]!=b->ffn.up.decoded)up++;
    if(gate>=up||up>=r->count||r->count==r->capacity)return -1;
    void *parent=NULL,*gv=NULL,*uv=NULL;
    if(bn_gpu_backend_buffer_create_bf16_stacked_pair(r->gpu,b->ffn.gate.decoded,
        b->ffn.up.decoded,&parent,&gv,&uv))return -1;
    /* Preserve reverse-order cleanup: both row views precede their parent.
     * Preparation has fenced the copies before releasing the old allocations. */
    void *old_gate=b->ffn.gate.decoded,*old_up=b->ffn.up.decoded;
    r->handles[gate]=parent;r->handles[up]=gv;r->handles[r->count++]=uv;
    b->ffn_gate_up=parent;b->ffn.gate.decoded=gv;b->ffn.up.decoded=uv;
    r->model.stacked_ffn_layers++;
    bn_gpu_backend_destroy_buffer(r->gpu,old_gate);
    bn_gpu_backend_destroy_buffer(r->gpu,old_up);
    return 0;
}

static int backend_bf16_validate(const BnConfig *c, const BnWeights *w, int nt) {
    if (!c || !w || !w->layers || nt <= 0 ||
        c->bf16_layout != BN_MODEL_BF16_LAYOUT_HYBRID_OFFSET_NORM ||
        c->dim != 5120 || c->hidden_dim != 17408 || c->n_layers != 64 ||
        c->n_experts || c->n_heads != 24 || c->n_kv_heads != 4 ||
        c->head_size != 256 || c->ssm_state_size != 128 ||
        c->ssm_group_count != 16 || c->ssm_time_step_rank != 48 ||
        c->ssm_inner_size != 6144 || c->ssm_conv_kernel != 4 ||
        c->vocab_size <= 0 || c->full_attn_interval != 4 ||
        c->seq_len <= 0 || c->seq_len > 262144 ||
        bn_model_transformer_policy_rope_dims_for_head(c, 256) != 64 ||
        !isfinite(bn_model_transformer_policy_rope_theta_for_head(c, 256)) ||
        bn_model_transformer_policy_rope_theta_for_head(c, 256) <= 1.0f ||
        c->act_type != BN_MODEL_ACTIVATION_SILU || !w->output_norm ||
        !w->token_embedding || w->emb_type != BN_GGUF_TENSOR_Q4_K ||
        !backend_bf16_weight_valid(&w->output_weight, c->vocab_size, 5120, nt))
        return -1;
    for (int l = 0; l < c->n_layers; ++l) {
        const BnLayerWeights *b = &w->layers[l];
        if (!b->norm.attn_norm || !b->norm.ffn_norm || b->ffn_kind != BN_LAYER_FFN_DENSE ||
            !backend_bf16_weight_valid(&b->ffn.ffn_gate, 17408, 5120, nt) ||
            !backend_bf16_weight_valid(&b->ffn.ffn_up, 17408, 5120, nt) ||
            !backend_bf16_weight_valid(&b->ffn.ffn_down, 5120, 17408, nt)) return -1;
        if (b->block_kind == BN_LAYER_BLOCK_SSM) {
            if ((l + 1) % 4 == 0 || !b->ssm.ssm_a || !b->ssm.ssm_conv1d ||
                !b->ssm.ssm_dt_bias || !b->ssm.ssm_norm ||
                !backend_bf16_weight_valid(&b->ssm.wqkv, 10240, 5120, nt) ||
                !backend_bf16_weight_valid(&b->ssm.wz, 6144, 5120, nt) ||
                !backend_bf16_weight_valid(&b->ssm.ssm_out, 5120, 6144, nt) ||
                !backend_bf16_weight_valid(&b->ssm.ssm_beta, 48, 5120, nt) ||
                !backend_bf16_weight_valid(&b->ssm.ssm_alpha, 48, 5120, nt) ||
                b->ssm.ssm_beta.type != BN_GGUF_TENSOR_Q8_0 ||
                b->ssm.ssm_alpha.type != BN_GGUF_TENSOR_Q8_0) return -1;
            for (int h = 0; h < 48; ++h)
                if (!(b->ssm.ssm_a[h] < 0.0f) || !isfinite(b->ssm.ssm_a[h])) return -1;
        } else if (b->block_kind == BN_LAYER_BLOCK_ATTENTION) {
            if ((l + 1) % 4 != 0 || !b->attn.q_norm || !b->attn.k_norm ||
                b->attn.q_bias || b->attn.k_bias || b->attn.v_bias ||
                !backend_bf16_weight_valid(&b->attn.wq, 12288, 5120, nt) ||
                !backend_bf16_weight_valid(&b->attn.wk, 1024, 5120, nt) ||
                !backend_bf16_weight_valid(&b->attn.wv, 1024, 5120, nt) ||
                !backend_bf16_weight_valid(&b->attn.wo, 5120, 6144, nt)) return -1;
        } else return -1;
    }
    return 0;
}

static int backend_bf16_layer(BackendBF16Resources *r, const BnLayerWeights *w,
    BnBackendBF16Layer *b) {
    b->ni = backend_bf16_floats(r, w->norm.attn_norm, 1, 5120, 1, -1, 0, 0);
    if (!b->ni) return -1;
    b->nf = backend_bf16_floats(r, w->norm.ffn_norm, 1, 5120, 1, -1, 0, 0);
    if (!b->nf ||
        backend_bf16_linear(r, &w->ffn.ffn_gate, -1, 0, &b->ffn.gate) ||
        backend_bf16_linear(r, &w->ffn.ffn_up, -1, 0, &b->ffn.up) ||
        backend_bf16_stack_ffn(r,b) ||
        backend_bf16_linear(r, &w->ffn.ffn_down, -1, 0, &b->ffn.down)) return -1;
    if (w->block_kind == BN_LAYER_BLOCK_ATTENTION) {
        if (backend_bf16_linear(r, &w->attn.wq, -1, 0, &b->q) ||
            backend_bf16_linear(r, &w->attn.wk, -1, 0, &b->k) ||
            backend_bf16_linear(r, &w->attn.wv, -1, 0, &b->v) ||
            backend_bf16_linear(r, &w->attn.wo, -1, 0, &b->o)) return -1;
        b->qn = backend_bf16_floats(r, w->attn.q_norm, 1, 256, 1, -1, 0, 0);
        if (!b->qn) return -1;
        b->kn = backend_bf16_floats(r, w->attn.k_norm, 1, 256, 1, -1, 0, 0);
        return b->kn ? 0 : -1;
    }
    b->split_qk_v = w->ssm.wqkv.type != w->ssm.wz.type;
    if (b->split_qk_v) {
        BnQWeight q = w->ssm.wqkv, k = w->ssm.wqkv, v = w->ssm.wqkv;
        size_t stride = bn_qweight_data_size(&q) / (size_t)q.rows;
        q.rows = k.rows = 2048;
        k.data = (const char *)k.data + 2048 * stride;
        v.data = (const char *)v.data + 4096 * stride;
        v.rows = 6144;
        /* Logical Q/K shards also select GEMM geometry, beyond quant cutoffs. */
        if (backend_bf16_linear(r, &q, -1, 0, &b->qkv) ||
            backend_bf16_linear(r, &k, -1, 0, &b->k) ||
            backend_bf16_linear(r, &v, 0, 128, &b->v)) return -1;
    } else if (backend_bf16_linear(r, &w->ssm.wqkv, 4096, 128, &b->qkv)) return -1;
    if (backend_bf16_linear(r, &w->ssm.wz, 0, 128, &b->z) ||
        backend_bf16_linear(r, &w->ssm.ssm_out, -1, 0, &b->o)) return -1;
    size_t bytes = bn_qweight_data_size(&w->ssm.ssm_beta), stride = bytes / 48;
    void *ba = r->planning ? (void *)&backend_bf16_plan_handle : malloc(bytes * 2);
    if (!ba) return -1;
    if (r->planning && backend_bf16_plan_add(r, 0, bytes * 2, 0)) return -1;
    for (int h = 0; !r->planning && h < 48; ++h) {
        int old = backend_bf16_source_row(h, 0, 1);
        memcpy((char *)ba + (size_t)h * stride,
            (const char *)w->ssm.ssm_beta.data + (size_t)old * stride, stride);
        memcpy((char *)ba + bytes + (size_t)h * stride,
            (const char *)w->ssm.ssm_alpha.data + (size_t)old * stride, stride);
    }
    BnQWeight joined = {ba, BN_GGUF_TENSOR_Q8_0, 96, 5120, 1.0f};
    int status = backend_bf16_linear(r, &joined, -1, 0, &b->ba);
    if (!r->planning) free(ba);
    if (status) return -1;
    b->conv = backend_bf16_floats(r, w->ssm.ssm_conv1d, 10240, 4, 0, 4096, 128, 0);
    if (!b->conv) return -1;
    b->an = backend_bf16_floats(r, w->ssm.ssm_a, 48, 1, 0, 0, 1, 1);
    if (!b->an) return -1;
    b->dt = backend_bf16_floats(r, w->ssm.ssm_dt_bias, 48, 1, 0, 0, 1, 0);
    if (!b->dt) return -1;
    b->gn = backend_bf16_floats(r, w->ssm.ssm_norm, 1, 128, 0, -1, 0, 0);
    return b->gn ? 0 : -1;
}

int bn_backend_layout_prepare_bf16(BnBackendModel *backend, BnGPUBackend *gpu,
    const BnConfig *config, const BnWeights *weights, int max_tokens) {
    return bn_backend_layout_prepare_bf16_with_math(backend, gpu, config,
        weights, max_tokens, BN_BACKEND_BF16_MATH_LIBC);
}

int bn_backend_layout_bf16_memory_plan(BnGPUBackend *gpu,
    const BnConfig *config, const BnWeights *weights, int max_tokens,
    BnBackendBF16Math profile, BnBackendBF16MemoryPlan *out) {
    if (!out || !bn_backend_layout_bf16_math_available(profile) ||
        !bn_gpu_backend_can_create_quant_only_buffer(gpu) ||
        !bn_gpu_backend_can_bf16_decode(gpu) || !bn_gpu_backend_can_destroy_buffer(gpu) ||
        !bn_gpu_backend_can_create_rotary_bf16(gpu) ||
        backend_bf16_validate(config, weights, max_tokens)) return -1;
    BackendBF16Resources r = {0};
    r.planning = 1; r.gpu = gpu; r.model.max_tokens = max_tokens;
    r.capacity = (size_t)config->n_layers * 32 + 8;
    r.model.memory.host_bytes = sizeof(r) + r.capacity * sizeof(void *) +
        (size_t)config->n_layers * sizeof(BnBackendBF16Layer);
    r.model.memory.host_peak_bytes = r.model.memory.host_bytes;
    for (int l = 0; l < config->n_layers; l++) {
        BnBackendBF16Layer layer = {0};
        if (backend_bf16_layer(&r, &weights->layers[l], &layer)) return -1;
    }
    BnGPUBF16LinearWeight output = {0};
    BnQWeight embedding = {weights->token_embedding, weights->emb_type,
        config->vocab_size, 5120, 1.0f};
    if (!backend_bf16_floats(&r, weights->output_norm, 1, 5120, 1, -1, 0, 0) ||
        backend_bf16_linear(&r, &weights->output_weight, -1, 0, &output) ||
        backend_bf16_plan_add(&r, bn_qweight_data_size(&embedding), 0, 0) ||
        backend_bf16_plan_add(&r, (size_t)config->seq_len * 64 * sizeof(uint16_t), 0, 0)) return -1;
    *out = r.model.memory;return 0;
}

int bn_backend_layout_prepare_bf16_with_math(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    int max_tokens, BnBackendBF16Math profile) {
    return bn_backend_layout_prepare_bf16_with_limits(backend, gpu, config,
        weights, max_tokens, profile, SIZE_MAX, SIZE_MAX);
}

int bn_backend_layout_prepare_bf16_with_limits(BnBackendModel *backend,
    BnGPUBackend *gpu, const BnConfig *config, const BnWeights *weights,
    int max_tokens, BnBackendBF16Math profile, size_t device_budget,
    size_t host_budget) {
    if (!bn_backend_layout_bf16_math_available(profile) ||
        !backend || !gpu || !bn_gpu_backend_can_create_quant_only_buffer(gpu) ||
        !bn_gpu_backend_can_bf16_decode(gpu) || !bn_gpu_backend_can_destroy_buffer(gpu) ||
        !bn_gpu_backend_can_create_rotary_bf16(gpu) ||
        bn_backend_model_gpu_resources(backend, &backend_bf16_owner) ||
        (bn_backend_model_gpu_for_cpu_operations(backend) &&
         bn_backend_model_gpu_for_cpu_operations(backend) != gpu) ||
        backend_bf16_validate(config, weights, max_tokens)) return -1;
    BnBackendBF16MemoryPlan memory;
    if (bn_backend_layout_bf16_memory_plan(gpu, config, weights, max_tokens,
            profile, &memory) || memory.device_peak_bytes > device_budget ||
        memory.host_peak_bytes > host_budget) return -1;
    BackendBF16Resources *r = calloc(1, sizeof(*r));
    if (!r) return -1;
    r->gpu = gpu;
    r->model.memory = memory;
    r->model.n_layers = config->n_layers;
    r->model.max_tokens = max_tokens;
    r->model.math_profile = profile;
    memcpy(r->model.math_library, "libc logf", sizeof("libc logf"));
#ifdef BN_ENABLE_MKL_VML
    if (profile == BN_BACKEND_BF16_MATH_MKL_VML_HA)
        MKL_Get_Version_String(r->model.math_library,
                              sizeof(r->model.math_library));
#endif
    r->capacity = (size_t)config->n_layers * 32 + 8;
    r->handles = calloc(r->capacity, sizeof(*r->handles));
    r->layers = calloc((size_t)config->n_layers, sizeof(*r->layers));
    r->model.layers = r->layers;
    if (!r->handles || !r->layers) goto fail;
    for (int l = 0; l < config->n_layers; ++l)
        if (backend_bf16_layer(r, &weights->layers[l], &r->layers[l])) goto fail;
    r->model.final_norm = backend_bf16_floats(r, weights->output_norm, 1, 5120, 1, -1, 0, 0);
    if (!r->model.final_norm ||
        backend_bf16_linear(r, &weights->output_weight, -1, 0, &r->model.output)) goto fail;
    BnQWeight embedding = {weights->token_embedding, weights->emb_type,
        config->vocab_size, 5120, 1.0f};
    /* Embedding is a gather: it needs no decoded linear layout. */
    r->model.embedding.type = embedding.type;
    r->model.embedding.raw = backend_bf16_own(r, bn_gpu_backend_create_quant_only_buffer(
        gpu, embedding.data, bn_qweight_data_size(&embedding),
        embedding.type, embedding.rows, embedding.cols));
    if (!r->model.embedding.raw) goto fail;
    BnGPURotaryCachePlan rotary = {0};
    rotary.positions = config->seq_len;
    rotary.rotary_dim = bn_model_transformer_policy_rope_dims_for_head(config, 256);
    rotary.theta = bn_model_transformer_policy_rope_theta_for_head(config, 256);
    r->model.rotary_cache = backend_bf16_own(r,
        bn_gpu_backend_create_rotary_bf16(gpu, &rotary));
    r->model.rotary_capacity = rotary.positions;
    if (!r->model.rotary_cache || bn_backend_model_attach_gpu_resources(backend,
            &backend_bf16_owner, r, backend_bf16_release)) goto fail;
    bn_backend_model_bind_gpu(backend, gpu);
    return 0;
fail:
    backend_bf16_release(r);
    return -1;
}

void bn_backend_layout_bf16_kda_free(BnBackendBF16KDAWeights *w) {
    if (!w) return;
    if (w->gpu) {
        void *handles[] = {w->in.raw, w->f_b.raw, w->g_b.raw, w->out.raw,
            w->conv, w->decay, w->bias, w->norm};
        for (size_t i = 0; i < sizeof(handles)/sizeof(handles[0]); ++i)
            bn_gpu_backend_destroy_buffer(w->gpu, handles[i]);
    }
    memset(w, 0, sizeof(*w));
}

/* Shape validation precedes the GGUF reader's shard-aware bounds validation.
 * This binder deliberately accepts only the qualified quantization profile. */
static const void *backend_anatomy_source(BnGGUFFile *file,
    const BnModelLayerAnatomy *layer, BnModelTensorRole role, int type,
    int nd, uint64_t d0, uint64_t d1, uint64_t d2, size_t *bytes) {
    int i = layer->tensor[role];
    if (i < 0 || (uint64_t)i >= file->n_tensors) return NULL;
    const BnGGUFTensorInfo *t = &file->tensors[i];
    if (t->type != (uint32_t)type || t->n_dims != (uint32_t)nd ||
        t->dims[0] != d0 || (nd > 1 && t->dims[1] != d1) ||
        (nd > 2 && t->dims[2] != d2) || !d0 ||
        (nd > 1 && d1 > UINT64_MAX/d0)) return NULL;
    uint64_t count = d0 * (nd > 1 ? d1 : 1);
    if (nd > 2 && d2 > UINT64_MAX/count) return NULL;
    count *= nd > 2 ? d2 : 1;
    if (!bn_gguf_tensor_size(t->type, count, bytes)) return NULL;
    return bn_gguf_tensor_data(file, i);
}

int bn_backend_layout_bf16_kda_init(BnBackendBF16KDAWeights *out,
    BnGPUBackend *gpu, BnGGUFFile *file, const BnModelAnatomy *a, int layer,
    size_t gpu_budget, size_t host_budget) {
    if (!out || out->gpu || out->in.raw || out->in.decoded || out->f_b.raw ||
        out->f_b.decoded || out->g_b.raw || out->g_b.decoded || out->out.raw ||
        out->out.decoded || out->conv || out->decay || out->bias || out->norm ||
        !bn_gpu_backend_can_create_quant_only_buffer(gpu) ||
        !bn_gpu_backend_can_destroy_buffer(gpu) || !file || !file->tensors ||
        (file->n_shards && (!file->shard_raws || !file->shard_raw_sizes ||
                            !file->shard_data_offsets)) ||
        !a || a->n_layers < 1 || a->n_layers > BN_MODEL_ANATOMY_MAX_LAYERS ||
        layer < 0 || layer >= a->n_layers ||
        a->layers[layer].kind != BN_MODEL_ANATOMY_KDA ||
        a->dim <= 0 || a->dim % 32 || a->kda_heads <= 0 ||
        a->kda_heads > (INT_MAX - 256)/385 || a->kda_head_dim != 128 ||
        a->kda_gate_rank != 128 || a->conv_kernel != 4 ||
        !a->kda_parameter_is_negative_exp || a->kda_qk_eps != 1e-6f ||
        !isfinite(a->kda_lower_bound) || a->kda_lower_bound >= 0 ||
        !isfinite(a->kda_output_eps) || a->kda_output_eps <= 0) return -1;
    int p = a->kda_heads * 128, merged = 3*p + a->kda_heads + 256;
    const BnModelLayerAnatomy *l = &a->layers[layer];
    const BnModelTensorRole roles[] = {BN_MODEL_TENSOR_ATTN_Q,
        BN_MODEL_TENSOR_ATTN_K, BN_MODEL_TENSOR_ATTN_V, BN_MODEL_TENSOR_SSM_BETA,
        BN_MODEL_TENSOR_SSM_F_A, BN_MODEL_TENSOR_SSM_G_A,
        BN_MODEL_TENSOR_SSM_F_B, BN_MODEL_TENSOR_SSM_G_B, BN_MODEL_TENSOR_SSM_OUT,
        BN_MODEL_TENSOR_SSM_CONV1D_Q, BN_MODEL_TENSOR_SSM_CONV1D_K,
        BN_MODEL_TENSOR_SSM_CONV1D_V, BN_MODEL_TENSOR_SSM_A,
        BN_MODEL_TENSOR_SSM_DT_BIAS, BN_MODEL_TENSOR_SSM_NORM};
    int rows[] = {p,p,p,a->kda_heads,128,128,p,p,a->dim,
                  p,p,p,a->kda_heads,p,128};
    int cols[] = {a->dim,a->dim,a->dim,a->dim,a->dim,a->dim,
                  128,128,p,4,4,4,1,1,1};
    const void *sources[15]; size_t sizes[15], total = 0, packed = 0;
    for (int i = 0; i < 15; ++i) {
        int type = i < 9 ? BN_GGUF_TENSOR_Q8_0 : BN_GGUF_TENSOR_F32;
        int nd = i < 9 ? 2 : (i < 12 ? 3 : 1);
        sources[i] = backend_anatomy_source(file, l, roles[i], type, nd,
            i < 12 ? (uint64_t)cols[i] : (uint64_t)rows[i],
            i < 9 ? (uint64_t)rows[i] : 1, i < 12 ? (uint64_t)rows[i] : 1,
            &sizes[i]);
        if (!sources[i] || sizes[i] > SIZE_MAX-total) return -1;
        total += sizes[i];
        if (i < 6) packed += sizes[i];
        if (i >= 9) {
            const unsigned char *src = sources[i];
            for (size_t j = 0; j < sizes[i]/sizeof(float); ++j) {
                float value; memcpy(&value, src + j*sizeof(float), sizeof(value));
                if (!isfinite(value) || (i == 12 && value > 0)) return -1;
            }
        }
    }
    size_t conv_bytes = sizes[9] + sizes[10] + sizes[11];
    size_t staging_bytes = packed > conv_bytes ? packed : conv_bytes;
    if (total > gpu_budget || staging_bytes > host_budget) return -1;
    unsigned char *staging = malloc(staging_bytes);
    if (!staging) return -1;
    BnBackendBF16KDAWeights w = {0};
    w.gpu=gpu; w.dim=a->dim; w.layer=layer; w.heads=a->kda_heads; w.head_dim=128;
    w.rank=128; w.kernel=4; w.lower_bound=a->kda_lower_bound;
    w.output_eps=a->kda_output_eps; w.bytes=total; w.host_staging_bytes=staging_bytes;
    size_t offset = 0;
    for (int i=0; i<6; ++i) {
        memcpy(staging+offset, sources[i], sizes[i]); offset+=sizes[i];
    }
    w.in.type=w.f_b.type=w.g_b.type=w.out.type=BN_GGUF_TENSOR_Q8_0;
    w.in.raw=bn_gpu_backend_create_quant_only_buffer(gpu,staging,packed,
        BN_GGUF_TENSOR_Q8_0,merged,a->dim);
    if (!w.in.raw) goto fail;
    void **dst[] = {&w.f_b.raw,&w.g_b.raw,&w.out.raw};
    for (int i=6;i<9;++i) {
        *dst[i-6]=bn_gpu_backend_create_quant_only_buffer(gpu,sources[i],sizes[i],
            BN_GGUF_TENSOR_Q8_0,rows[i],cols[i]);
        if (!*dst[i-6]) goto fail;
    }
    offset=0;
    for (int i=9;i<12;++i) {
        memcpy(staging+offset,sources[i],sizes[i]); offset+=sizes[i];
    }
    w.conv=bn_gpu_backend_create_quant_only_buffer(gpu,staging,conv_bytes,
        BN_GGUF_TENSOR_F32,3*p,4);
    if (!w.conv) goto fail;
    void **params[] = {&w.decay,&w.bias,&w.norm};
    for (int i=12;i<15;++i) {
        *params[i-12]=bn_gpu_backend_create_quant_only_buffer(gpu,sources[i],sizes[i],
            BN_GGUF_TENSOR_F32,i==12?rows[i]:(i==13?a->kda_heads:1),
            i==12?1:(i==13?128:rows[i]));
        if (!*params[i-12]) goto fail;
    }
    free(staging); *out=w; return 0;
fail:
    free(staging); bn_backend_layout_bf16_kda_free(&w); return -1;
}

void bn_backend_layout_bf16_mhc_free(BnBackendBF16MHCWeights *w) {
    if(!w)return;
    if(w->gpu) {
        bn_gpu_backend_destroy_buffer(w->gpu,w->fn);
        bn_gpu_backend_destroy_buffer(w->gpu,w->scale);
        bn_gpu_backend_destroy_buffer(w->gpu,w->base);
        bn_gpu_backend_destroy_buffer(w->gpu,w->norm);
    }
    memset(w,0,sizeof(*w));
}
int bn_backend_layout_bf16_mhc_init(BnBackendBF16MHCWeights *out,
    BnGPUBackend *gpu,BnGGUFFile *file,const BnModelAnatomy *a,int layer,int ffn,
    size_t gpu_budget,size_t host_budget) {
    if(!out||out->gpu||out->fn||out->scale||out->base||out->norm||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu)||
       !file||!file->tensors||(file->n_shards&&(!file->shard_raws||!file->shard_raw_sizes||!file->shard_data_offsets))||
       !a||a->n_layers<1||a->n_layers>BN_MODEL_ANATOMY_MAX_LAYERS||layer<0||layer>=a->n_layers||
       (ffn!=0&&ffn!=1)||a->layers[layer].kind==BN_MODEL_ANATOMY_NONE||a->mhc_streams!=4||
       a->dim<=0||a->dim%1024||a->dim>INT_MAX/4||a->mhc_iterations<1||a->mhc_iterations>128||
       !isfinite(a->mhc_eps)||a->mhc_eps<=0||!isfinite(a->norm_eps)||a->norm_eps<=0)return -1;
    const BnModelLayerAnatomy *l=&a->layers[layer];int wide=a->dim*4;
    BnModelTensorRole roles[]={ffn?BN_MODEL_TENSOR_MHC_FFN_FN:BN_MODEL_TENSOR_MHC_ATTN_FN,
        ffn?BN_MODEL_TENSOR_MHC_FFN_SCALE:BN_MODEL_TENSOR_MHC_ATTN_SCALE,
        ffn?BN_MODEL_TENSOR_MHC_FFN_BASE:BN_MODEL_TENSOR_MHC_ATTN_BASE,
        ffn?BN_MODEL_TENSOR_FFN_NORM:BN_MODEL_TENSOR_ATTN_NORM};
    int ix=l->tensor[roles[0]];
    if(ix<0||(uint64_t)ix>=file->n_tensors)return -1;
    int fn_type=(int)file->tensors[ix].type;
    if(fn_type!=BN_GGUF_TENSOR_Q8_0&&fn_type!=BN_GGUF_TENSOR_F32)return -1;
    int cols[]={wide,3,24,a->dim};const void *src[4];size_t bytes[4];
    for(int i=0;i<4;++i) {
        src[i]=backend_anatomy_source(file,l,roles[i],i?BN_GGUF_TENSOR_F32:fn_type,
            i?1:2,cols[i],i?1:24,1,&bytes[i]);
        if(!src[i])return -1;
    }
    if((size_t)wide>SIZE_MAX/24/sizeof(float))return -1;
    size_t fn_bytes=(size_t)wide*24*sizeof(float),norm_bytes=(size_t)a->dim*2;
    if(fn_bytes>SIZE_MAX-norm_bytes-108)return -1;
    size_t total=fn_bytes+norm_bytes+108,staging=fn_bytes;
    if(total>gpu_budget||staging>host_budget)return -1;
    /* Decode/validate before GPU ownership begins. The same staging buffer is
     * reused for the final norm upload; raw checkpoint pages stay unchanged. */
    float *decoded=malloc(staging);if(!decoded)return -1;
    for(int row=0;row<24;++row) {
        if(bn_quant_dequant_row(fn_type,src[0],row,wide,decoded+(size_t)row*wide))goto invalid;
    }
    for(size_t i=0;i<fn_bytes/4;++i)if(!isfinite(decoded[i]))goto invalid;
    for(int i=1;i<4;++i)for(int j=0;j<cols[i];++j) {
        float value;memcpy(&value,(const unsigned char*)src[i]+(size_t)j*4,4);
        if(!isfinite(value)||(i==3&&((backend_bf16_round(value)&0x7f80u)==0x7f80u)))goto invalid;
    }
    BnBackendBF16MHCWeights w={0};w.gpu=gpu;w.dim=a->dim;w.streams=4;w.iterations=a->mhc_iterations;
    w.rms_eps=w.norm_eps=a->norm_eps;w.hc_eps=a->mhc_eps;w.bytes=total;w.host_staging_bytes=staging;
    w.fn=bn_gpu_backend_create_quant_only_buffer(gpu,decoded,fn_bytes,BN_GGUF_TENSOR_F32,24,wide);
    if(!w.fn)goto fail;
    w.scale=bn_gpu_backend_create_quant_only_buffer(gpu,src[1],bytes[1],BN_GGUF_TENSOR_F32,1,3);
    if(!w.scale)goto fail;
    w.base=bn_gpu_backend_create_quant_only_buffer(gpu,src[2],bytes[2],BN_GGUF_TENSOR_F32,1,24);
    if(!w.base)goto fail;
    uint16_t *norm=(uint16_t*)decoded;
    for(int j=0;j<a->dim;++j){float value;memcpy(&value,(const unsigned char*)src[3]+(size_t)j*4,4);norm[j]=backend_bf16_round(value);}
    w.norm=bn_gpu_backend_create_quant_only_buffer(gpu,norm,norm_bytes,BN_GGUF_TENSOR_BF16,1,a->dim);
    if(!w.norm)goto fail;
    free(decoded);*out=w;return 0;
fail:
    bn_backend_layout_bf16_mhc_free(&w);
invalid:
    free(decoded);return -1;
}

void bn_backend_layout_bf16_dense_ffn_free(BnBackendBF16DenseFFNWeights*w) {
    if(!w)return;
    if(w->gpu){bn_gpu_backend_destroy_buffer(w->gpu,w->gate_up.raw);bn_gpu_backend_destroy_buffer(w->gpu,w->down.raw);}
    memset(w,0,sizeof(*w));
}
static int backend_bf16_clamped_ffn_init(BnBackendBF16DenseFFNWeights*out,
    BnGPUBackend*gpu,BnGGUFFile*file,const BnModelAnatomy*a,int layer,
    size_t gpu_budget,size_t host_budget,int shared) {
    if(!a||a->n_layers<1||a->n_layers>BN_MODEL_ANATOMY_MAX_LAYERS||layer<0||layer>=a->n_layers)return -1;
    const BnModelLayerAnatomy*l=&a->layers[layer];
    int dim=a->dim,h=shared?a->shared_hidden:a->dense_hidden;
    float limit=shared?l->swiglu_clamp_shexp:l->swiglu_clamp_exp;
    if(!out||out->gpu||out->gate_up.raw||out->gate_up.decoded||out->down.raw||out->down.decoded||
        !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu)||
        !file||!file->tensors||(file->n_shards&&(!file->shard_raws||!file->shard_raw_sizes||!file->shard_data_offsets))||
        l->kind==BN_MODEL_ANATOMY_NONE||(shared?(!l->uses_moe||a->shared_experts!=1):l->uses_moe)||
        dim<=0||dim%32||h<=0||h%32||h>INT_MAX/2||!isfinite(limit)||limit<0)return -1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_FFN_GATE,BN_MODEL_TENSOR_FFN_UP,BN_MODEL_TENSOR_FFN_DOWN};
    if(shared){roles[0]=BN_MODEL_TENSOR_SHARED_FFN_GATE;roles[1]=BN_MODEL_TENSOR_SHARED_FFN_UP;roles[2]=BN_MODEL_TENSOR_SHARED_FFN_DOWN;}
    const void*src[3];size_t sizes[3],total=0;
    for(int i=0;i<3;++i){src[i]=backend_anatomy_source(file,l,roles[i],BN_GGUF_TENSOR_Q8_0,2,i==2?h:dim,i==2?dim:h,1,&sizes[i]);if(!src[i]||sizes[i]>SIZE_MAX-total)return -1;total+=sizes[i];}
    size_t merged=sizes[0]+sizes[1];if(total>gpu_budget||merged>host_budget)return -1;
    unsigned char*staging=malloc(merged);if(!staging)return -1;memcpy(staging,src[0],sizes[0]);memcpy(staging+sizes[0],src[1],sizes[1]);
    BnBackendBF16DenseFFNWeights w={0};w.gpu=gpu;w.dim=dim;w.hidden_dim=h;w.clamp_limit=limit;w.bytes=total;w.host_staging_bytes=merged;w.gate_up.type=w.down.type=BN_GGUF_TENSOR_Q8_0;
    w.gate_up.raw=bn_gpu_backend_create_quant_only_buffer(gpu,staging,merged,BN_GGUF_TENSOR_Q8_0,2*h,dim);free(staging);if(!w.gate_up.raw)goto fail;
    w.down.raw=bn_gpu_backend_create_quant_only_buffer(gpu,src[2],sizes[2],BN_GGUF_TENSOR_Q8_0,dim,h);if(!w.down.raw)goto fail;
    *out=w;return 0;
fail:
    bn_backend_layout_bf16_dense_ffn_free(&w);return -1;
}
int bn_backend_layout_bf16_dense_ffn_init(BnBackendBF16DenseFFNWeights*out,
    BnGPUBackend*gpu,BnGGUFFile*file,const BnModelAnatomy*a,int layer,
    size_t gpu_budget,size_t host_budget) {
    return backend_bf16_clamped_ffn_init(out,gpu,file,a,layer,gpu_budget,host_budget,0);
}
int bn_backend_layout_bf16_shared_ffn_init(BnBackendBF16DenseFFNWeights*out,
    BnGPUBackend*gpu,BnGGUFFile*file,const BnModelAnatomy*a,int layer,
    size_t gpu_budget,size_t host_budget) {
    return backend_bf16_clamped_ffn_init(out,gpu,file,a,layer,gpu_budget,host_budget,1);
}

void bn_backend_layout_bf16_moe_free(BnBackendBF16MoEWeights*w) {
    if(!w)return;
    bn_backend_layout_bf16_dense_ffn_free(&w->shared);
    if(w->gpu){bn_gpu_backend_destroy_buffer(w->gpu,w->router);bn_gpu_backend_destroy_buffer(w->gpu,w->bias);}
    memset(w,0,sizeof(*w));
}
int bn_backend_layout_bf16_moe_init(BnBackendBF16MoEWeights*out,BnGPUBackend*gpu,
    BnGGUFFile*file,const BnModelAnatomy*a,int layer,const BnMoEExpertMap*map,
    size_t gpu_budget,size_t host_budget) {
    if(!out||out->gpu||out->router||out->bias||out->shared.gpu||out->shared.gate_up.raw||
       out->shared.gate_up.decoded||out->shared.down.raw||out->shared.down.decoded||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu)||
       !a||a->n_layers<1||a->n_layers>BN_MODEL_ANATOMY_MAX_LAYERS||layer<0||layer>=a->n_layers||
       !file||!file->tensors||(file->n_shards&&(!file->shard_raws||!file->shard_raw_sizes||!file->shard_data_offsets))||!map||a->dim<1||a->dim%256||a->expert_hidden<1||a->expert_hidden%256||
       a->shared_hidden<1||a->shared_hidden%32||a->shared_hidden>INT_MAX/2||a->shared_experts!=1||
       a->experts!=288||a->active_experts!=8||a->normalize_expert_weights!=1||
       !isfinite(a->expert_weights_scale)||a->expert_weights_scale<=0)return -1;
    const BnModelLayerAnatomy*l=&a->layers[layer];
    if(!l->uses_moe||l->kind==BN_MODEL_ANATOMY_NONE||!isfinite(l->swiglu_clamp_exp)||l->swiglu_clamp_exp<0||
       !isfinite(l->swiglu_clamp_shexp)||l->swiglu_clamp_shexp<0)return -1;
    const BnQuantFormatOps *gate_format=bn_quant_format_ops(map->gate_type);
    const BnQuantFormatOps *down_format=bn_quant_format_ops(map->down_type);
    if(!gate_format||!down_format||map->up_type!=map->gate_type||
       (gate_format->bf16_moe!=BN_QUANT_BF16_MOE_Q4_K&&gate_format->bf16_moe!=BN_QUANT_BF16_MOE_Q5_K)||
       (down_format->bf16_moe!=BN_QUANT_BF16_MOE_Q5_K&&down_format->bf16_moe!=BN_QUANT_BF16_MOE_Q6_K))return -1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_MOE_ROUTER,BN_MODEL_TENSOR_MOE_ROUTER_BIAS,
        BN_MODEL_TENSOR_SHARED_FFN_GATE,BN_MODEL_TENSOR_SHARED_FFN_UP,BN_MODEL_TENSOR_SHARED_FFN_DOWN,
        BN_MODEL_TENSOR_MOE_GATE_EXPS,BN_MODEL_TENSOR_MOE_UP_EXPS,BN_MODEL_TENSOR_MOE_DOWN_EXPS};
    const void*src[8];size_t sizes[8],total=0;
    for(int i=0;i<8;++i){int type=i<2?BN_GGUF_TENSOR_F32:i<5?BN_GGUF_TENSOR_Q8_0:i<7?map->gate_type:map->down_type;
        int cols=i==1?288:(i==4?a->shared_hidden:i==7?a->expert_hidden:a->dim);
        int rows=i==0?288:i==1?1:i==4||i==7?a->dim:i<5?a->shared_hidden:a->expert_hidden;
        src[i]=backend_anatomy_source(file,l,roles[i],type,i==1?1:i>=5?3:2,cols,rows,i>=5?288:1,&sizes[i]);
        if(!src[i])return -1;
        if(i<5){if(sizes[i]>SIZE_MAX-total)return -1;total+=sizes[i];}
    }
    size_t gb=sizes[5]/288,db=sizes[7]/288;
    if(map->gate_up_fused||map->expert_gate_bytes!=gb||map->expert_up_bytes!=gb||map->expert_down_bytes!=db||
       map->gate_stride!=gb||map->up_stride!=gb||map->down_stride!=db||
       map->gate_rows!=a->expert_hidden||map->up_rows!=a->expert_hidden||map->down_rows!=a->dim||
       map->gate_cols!=a->dim||map->up_cols!=a->dim||map->down_cols!=a->expert_hidden||
       (map->repacked&&map->expert_total_bytes!=gb*2+db))return -1;
    size_t offsets[]={map->gate_offset,map->up_offset,map->down_offset};
    uint32_t shards[]={map->gate_shard_idx,map->up_shard_idx,map->down_shard_idx};
    for(int i=0;i<3;++i){const BnGGUFTensorInfo*t=&file->tensors[l->tensor[roles[5+i]]];
        size_t prefix=file->n_shards?file->shard_data_offsets[t->shard_idx]:file->data_offset;
        if(shards[i]!=t->shard_idx||offsets[i]!=prefix+t->offset)return -1;}
    size_t staging=sizes[2]+sizes[3];if(total>gpu_budget||staging>host_budget)return -1;
    for(int i=0;i<2;++i)for(size_t j=0;j<sizes[i]/4;++j){float value;memcpy(&value,(const unsigned char*)src[i]+j*4,4);if(!isfinite(value))return -1;}
    BnBackendBF16MoEWeights w={0};w.gpu=gpu;w.source_map=map;w.dim=a->dim;w.hidden_dim=a->expert_hidden;
    w.n_experts=288;w.k=8;w.layer=layer;w.gate_type=map->gate_type;w.down_type=map->down_type;
    w.gate_bytes=gb;w.down_bytes=db;w.clamp_limit=l->swiglu_clamp_exp;w.route_scale=a->expert_weights_scale;
    w.bytes=total;w.host_staging_bytes=staging;
    if(bn_backend_layout_bf16_shared_ffn_init(&w.shared,gpu,file,a,layer,total-sizes[0]-sizes[1],host_budget))goto fail;
    w.router=bn_gpu_backend_create_quant_only_buffer(gpu,src[0],sizes[0],BN_GGUF_TENSOR_F32,288,a->dim);if(!w.router)goto fail;
    w.bias=bn_gpu_backend_create_quant_only_buffer(gpu,src[1],sizes[1],BN_GGUF_TENSOR_F32,1,288);if(!w.bias)goto fail;
    *out=w;return 0;
fail:
    bn_backend_layout_bf16_moe_free(&w);return -1;
}

void bn_backend_layout_bf16_indexer_free(BnBackendBF16IndexerWeights *w) {
    if(!w)return;
    if(w->gpu){void *p[]={w->query.raw,w->key.raw,w->gate.raw,w->head_weights,w->norm,w->bias,w->ape};
        for(size_t i=0;i<sizeof(p)/sizeof(p[0]);++i)if(p[i])bn_gpu_backend_destroy_buffer(w->gpu,p[i]);}
    memset(w,0,sizeof(*w));
}
int bn_backend_layout_bf16_indexer_init(BnBackendBF16IndexerWeights *out,
    BnGPUBackend *gpu,BnGGUFFile *file,const BnModelAnatomy *a,int layer,size_t budget) {
    if(!out||out->gpu||out->query.raw||out->query.decoded||out->key.raw||out->key.decoded||
       out->gate.raw||out->gate.decoded||out->head_weights||out->norm||out->bias||out->ape||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu)||
       !file||!file->tensors||(file->n_shards&&(!file->shard_raws||!file->shard_raw_sizes||!file->shard_data_offsets))||
       !a||a->n_layers<1||a->n_layers>BN_MODEL_ANATOMY_MAX_LAYERS||layer<0||layer>=a->n_layers||
       a->layers[layer].kind!=BN_MODEL_ANATOMY_INDEXED_MLA||a->dim<1||a->dim%32||
       a->q_rank<1||a->q_rank%32||a->indexer_heads!=32||a->indexer_dim!=128||a->indexer_pool!=4||
       a->rope_dim||a->indexer_norm_eps!=1e-6f)return -1;
    const BnModelTensorRole roles[]={BN_MODEL_TENSOR_INDEXER_Q,BN_MODEL_TENSOR_INDEXER_K,
        BN_MODEL_TENSOR_INDEXER_POOL_GATE,BN_MODEL_TENSOR_INDEXER_HEAD_WEIGHTS,
        BN_MODEL_TENSOR_INDEXER_K_NORM,BN_MODEL_TENSOR_INDEXER_K_BIAS,BN_MODEL_TENSOR_INDEXER_POOL_APE};
    int rows[]={a->indexer_heads*128,128,128,a->indexer_heads,1,1,4};
    int cols[]={a->q_rank,a->dim,a->dim,a->dim,128,128,128};
    const void *src[7];size_t sizes[7],total=0;
    for(int i=0;i<7;++i){int type=i<3?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_F32;
        src[i]=backend_anatomy_source(file,&a->layers[layer],roles[i],type,i==4||i==5?1:2,
            (uint64_t)cols[i],(uint64_t)rows[i],1,&sizes[i]);
        if(!src[i]||sizes[i]>SIZE_MAX-total)return -1;
        total+=sizes[i];}
    if(total>budget)return -1;
    for(int i=3;i<7;++i)for(size_t j=0;j<sizes[i]/sizeof(float);++j){float v;memcpy(&v,(const unsigned char*)src[i]+j*sizeof(float),sizeof(v));if(!isfinite(v))return -1;}
    BnBackendBF16IndexerWeights w={0};w.gpu=gpu;w.dim=a->dim;w.q_rank=a->q_rank;w.heads=a->indexer_heads;
    w.head_dim=a->indexer_dim;w.pool_size=a->indexer_pool;w.layer=layer;w.norm_eps=a->indexer_norm_eps;
    w.weight_scale=1.f/sqrtf((float)w.heads*w.head_dim);w.bytes=total;
    w.query.type=w.key.type=w.gate.type=BN_GGUF_TENSOR_Q8_0;
    void **dst[]={&w.query.raw,&w.key.raw,&w.gate.raw,&w.head_weights,&w.norm,&w.bias,&w.ape};
    for(int i=0;i<7;++i){*dst[i]=bn_gpu_backend_create_quant_only_buffer(gpu,src[i],sizes[i],
        i<3?BN_GGUF_TENSOR_Q8_0:BN_GGUF_TENSOR_F32,rows[i],cols[i]);if(!*dst[i])goto fail;}
    *out=w;return 0;
fail:
    bn_backend_layout_bf16_indexer_free(&w);return -1;
}

void bn_backend_layout_bf16_mla_free(BnBackendBF16MLAWeights *w) {
    if(!w)return;
    if(w->gpu){void *p[]={w->q_a.raw,w->q_b.raw,w->kv_a.raw,w->output.raw,w->q_norm,w->kv_norm,w->key,w->value};
        for(size_t i=0;i<sizeof(p)/sizeof(p[0]);++i)if(p[i])bn_gpu_backend_destroy_buffer(w->gpu,p[i]);}
    memset(w,0,sizeof(*w));
}
int bn_backend_layout_bf16_mla_init(BnBackendBF16MLAWeights *out,BnGPUBackend *gpu,
    BnGGUFFile *file,const BnModelAnatomy *a,int layer,size_t budget) {
    if(!out||out->gpu||out->q_a.raw||out->q_a.decoded||out->q_b.raw||out->q_b.decoded||out->kv_a.raw||out->kv_a.decoded||
       out->output.raw||out->output.decoded||out->q_norm||out->kv_norm||out->key||out->value||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu)||!bn_gpu_backend_can_bf16_decode(gpu)||!bn_gpu_backend_can_bf16_identity_materialize(gpu)||
       !file||!file->tensors||(file->n_shards&&(!file->shard_raws||!file->shard_raw_sizes||!file->shard_data_offsets))||
       !a||a->n_layers<1||a->n_layers>BN_MODEL_ANATOMY_MAX_LAYERS||layer<0||layer>=a->n_layers||
       a->layers[layer].kind!=BN_MODEL_ANATOMY_INDEXED_MLA||a->rope_dim||a->dim<32||a->dim%32||a->q_rank<32||a->q_rank%32||
       a->kv_rank<32||a->kv_rank%32||a->attention_heads<1||a->attention_heads>128||a->mla_key_dim<32||a->mla_key_dim%32||
       a->mla_value_dim<32||a->mla_value_dim%32||a->mla_key_dim>INT_MAX/a->attention_heads||
       a->mla_value_dim>INT_MAX/a->attention_heads||a->kv_rank>INT_MAX/a->attention_heads||!isfinite(a->norm_eps)||a->norm_eps<=0)return -1;
    BnModelTensorRole roles[]={BN_MODEL_TENSOR_MLA_Q_A,BN_MODEL_TENSOR_MLA_Q_B,BN_MODEL_TENSOR_MLA_KV_A,BN_MODEL_TENSOR_ATTN_OUTPUT,
        BN_MODEL_TENSOR_MLA_Q_A_NORM,BN_MODEL_TENSOR_MLA_KV_A_NORM,BN_MODEL_TENSOR_MLA_K_B,BN_MODEL_TENSOR_MLA_V_B};
    int rows[]={a->q_rank,a->attention_heads*a->mla_key_dim,a->kv_rank,a->dim,1,1,a->attention_heads*a->kv_rank,a->attention_heads*a->mla_value_dim};
    int cols[]={a->dim,a->q_rank,a->dim,a->attention_heads*a->mla_value_dim,a->q_rank,a->kv_rank,a->mla_key_dim,a->kv_rank};
    const void *src[8];size_t raw[8],retained[8],total=0,peak=0;
    for(int j=0;j<8;++j){int type=j==4||j==5?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_Q8_0;
        int dims=j==4||j==5?1:j>=6?3:2;
        src[j]=backend_anatomy_source(file,&a->layers[layer],roles[j],type,dims,(uint64_t)cols[j],
            j>=6?(uint64_t)rows[j]/a->attention_heads:(uint64_t)rows[j],j>=6?(uint64_t)a->attention_heads:1,&raw[j]);
        if(!src[j]||(size_t)rows[j]>SIZE_MAX/2/(size_t)cols[j])return -1;
        retained[j]=j<4?raw[j]:(size_t)rows[j]*cols[j]*2;
        if(retained[j]>SIZE_MAX-total||raw[j]>SIZE_MAX-total-retained[j])return -1;
        size_t next=total+retained[j],transient=j<4?next:next+raw[j];if(transient>peak)peak=transient;total=next;
    }
    if(peak>budget)return -1;
    for(int j=4;j<6;++j)for(size_t k=0;k<raw[j]/4;++k){float v;memcpy(&v,(const unsigned char*)src[j]+k*4,4);if(!isfinite(v))return -1;}
    BnBackendBF16MLAWeights w={0};w.gpu=gpu;w.dim=a->dim;w.q_rank=a->q_rank;w.kv_rank=a->kv_rank;w.heads=a->attention_heads;
    w.key_dim=a->mla_key_dim;w.value_dim=a->mla_value_dim;w.layer=layer;w.norm_eps=a->norm_eps;w.scale=1.f/sqrtf((float)w.key_dim);w.bytes=total;w.peak_bytes=peak;
    w.q_a.type=w.q_b.type=w.kv_a.type=w.output.type=BN_GGUF_TENSOR_Q8_0;
    void **dst[]={&w.q_a.raw,&w.q_b.raw,&w.kv_a.raw,&w.output.raw,&w.q_norm,&w.kv_norm,&w.key,&w.value};
    for(int j=0;j<8;++j){void *uploaded=bn_gpu_backend_create_quant_only_buffer(gpu,src[j],raw[j],j==4||j==5?BN_GGUF_TENSOR_F32:BN_GGUF_TENSOR_Q8_0,rows[j],cols[j]);if(!uploaded)goto fail;
        if(j<4)*dst[j]=uploaded;
        else {*dst[j]=j>=6?bn_gpu_backend_buffer_create_bf16_identity_materialized(gpu,uploaded):bn_gpu_backend_buffer_create_bf16_decoded(gpu,uploaded);bn_gpu_backend_destroy_buffer(gpu,uploaded);if(!*dst[j])goto fail;}}
    *out=w;return 0;
fail:
    bn_backend_layout_bf16_mla_free(&w);return -1;
}

/* A separate immutable owner keeps GLM layouts out of CPU model weights and
 * prevents partial preparation from becoming visible to sessions. */
static const char backend_glm_owner=0;
typedef struct {
    BnBackendBF16GLMModel model;
    BnModelAnatomy anatomy;
    BnBackendBF16GLMLayer *layers;
} BackendGLMResources;
static void backend_glm_release(void *ptr) {
    BackendGLMResources *r=ptr;if(!r)return;
    if(r->model.expert_cache)r->model.gpu->expert_cache_destroy(r->model.expert_cache);
    if(r->layers)for(int i=0;i<r->model.n_layers;++i){BnBackendBF16GLMLayer *l=&r->layers[i];
        bn_backend_layout_bf16_moe_free(&l->moe);
        bn_backend_layout_bf16_dense_ffn_free(&l->dense);
        bn_backend_layout_bf16_mhc_free(&l->ffn_mhc);
        bn_backend_layout_bf16_mhc_free(&l->attention_mhc);
        bn_backend_layout_bf16_indexer_free(&l->indexer);
        bn_backend_layout_bf16_mla_free(&l->mla);
        bn_backend_layout_bf16_kda_free(&l->kda);
    }
    if(r->model.gpu){
        bn_gpu_backend_destroy_buffer(r->model.gpu,r->model.embedding.raw);
        bn_gpu_backend_destroy_buffer(r->model.gpu,r->model.output.raw);
        bn_gpu_backend_destroy_buffer(r->model.gpu,r->model.output_norm);
    }
    free(r->layers);free(r);
}
const BnBackendBF16GLMModel *bn_backend_layout_bf16_glm_model(const BnBackendModel *backend) {
    const BackendGLMResources *r=bn_backend_model_gpu_resources(backend,&backend_glm_owner);
    return r&&bn_backend_model_gpu(backend)==r->model.gpu?&r->model:NULL;
}
static int backend_glm_account(BackendGLMResources *r,size_t bytes,size_t peak,size_t host,size_t budget) {
    if(bytes>peak||r->model.bytes>budget||peak>budget-r->model.bytes)return -1;
    if(r->model.bytes+peak>r->model.peak_bytes)r->model.peak_bytes=r->model.bytes+peak;
    r->model.bytes+=bytes;
    if(host>r->model.host_staging_bytes)r->model.host_staging_bytes=host;
    return 0;
}
static const void *backend_glm_global(BnGGUFFile *f,int index,int type,int nd,
    int dim,int rows,size_t *bytes) {
    BnModelLayerAnatomy global={0};global.tensor[BN_MODEL_TENSOR_ATTN_Q]=index;
    return backend_anatomy_source(f,&global,BN_MODEL_TENSOR_ATTN_Q,type,nd,dim,rows,1,bytes);
}
static int backend_glm_cache_init(BackendGLMResources *r,size_t budget,size_t host_budget) {
    int count=0;
    for(int l=0;l<r->model.n_layers;l++)count+=r->anatomy.layers[l].uses_moe!=0;
    if(!count)return -1;
    int *types=calloc((size_t)count*2,sizeof(*types));if(!types)return -1;
    BnGPUExpertCacheConfig cfg={.n_layers=count,.gate_types=types,.down_types=types+count,.device_budget_bytes=budget};
    int index=0;
    for(int l=0;l<r->model.n_layers;l++)if(r->anatomy.layers[l].uses_moe) {
        BnBackendBF16GLMLayer *layer=&r->layers[l];const BnBackendBF16MoEWeights *w=&layer->moe;
        if(!index){cfg.n_experts=w->n_experts;cfg.dim=w->dim;cfg.hidden_dim=w->hidden_dim;}
        if(w->n_experts!=cfg.n_experts||w->dim!=cfg.dim||w->hidden_dim!=cfg.hidden_dim){free(types);return -1;}
        types[index]=w->gate_type;types[count+index]=w->down_type;layer->expert_cache_layer=index++;
    }
    r->model.expert_cache=r->model.gpu->expert_cache_create(r->model.gpu->ctx,&cfg);
    free(types);
    if(!r->model.expert_cache)return -1;
    BnGPUExpertCacheMemory memory;
    if(r->model.gpu->expert_cache_memory(r->model.expert_cache,&memory)||
       memory.host_bytes>host_budget||memory.device_bytes>budget)return -1;
    r->model.expert_cache_layers=count;r->model.expert_cache_host_bytes=memory.host_bytes;return 0;
}
int bn_backend_layout_prepare_bf16_glm_model(BnBackendModel *backend,BnGPUBackend *gpu,
    BnGGUFFile *file,const BnModelAnatomy *a,const BnMoEExpertMap *maps,size_t n_maps,
    size_t gpu_budget,size_t host_budget) {
    return bn_backend_layout_prepare_bf16_glm_model_with_cache(backend,gpu,file,a,maps,n_maps,gpu_budget,host_budget,0);
}
int bn_backend_layout_prepare_bf16_glm_model_with_cache(BnBackendModel *backend,BnGPUBackend *gpu,
    BnGGUFFile *file,const BnModelAnatomy *a,const BnMoEExpertMap *maps,size_t n_maps,
    size_t gpu_budget,size_t host_budget,size_t cache_budget) {
    if(!backend||!a||a->n_layers<1||a->n_layers>BN_MODEL_ANATOMY_MAX_LAYERS||a->dim!=4096||
       a->vocab_size<1||a->mhc_streams!=4||!file||!file->tensors||
       (file->n_shards&&(!file->shard_raws||!file->shard_raw_sizes||!file->shard_data_offsets))||
       !bn_gpu_backend_can_create_quant_only_buffer(gpu)||!bn_gpu_backend_can_destroy_buffer(gpu)||
       !bn_gpu_backend_can_bf16_decode(gpu)||bn_backend_model_gpu_for_cpu_operations(backend)||
       bn_backend_model_gpu_resources(backend,&backend_glm_owner))return -1;
    if(cache_budget&&(!gpu->expert_cache_create||!gpu->expert_cache_destroy||!gpu->expert_cache_bind||
       !gpu->expert_cache_memory||!gpu->expert_cache_begin||!gpu->expert_cache_begin_group||!gpu->expert_cache_end||
       !gpu->expert_cache_cancel||!gpu->expert_cache_stats))return -1;
    for(int i=0;i<a->n_layers;++i){const BnModelLayerAnatomy *l=&a->layers[i];
        if((l->kind!=BN_MODEL_ANATOMY_KDA&&l->kind!=BN_MODEL_ANATOMY_INDEXED_MLA)||
           (l->uses_moe!=0&&l->uses_moe!=1)||(l->uses_moe&&(!maps||n_maps<(size_t)a->n_layers)))return -1;
    }
    size_t emb_bytes,out_bytes,norm_bytes;
    const void *emb=backend_glm_global(file,a->embedding_tensor,BN_GGUF_TENSOR_Q8_0,2,a->dim,a->vocab_size,&emb_bytes);
    const void *out=backend_glm_global(file,a->output_tensor,BN_GGUF_TENSOR_Q8_0,2,a->dim,a->vocab_size,&out_bytes);
    const void *norm=backend_glm_global(file,a->output_norm_tensor,BN_GGUF_TENSOR_F32,1,a->dim,1,&norm_bytes);
    if(!emb||!out||!norm||norm_bytes!=(size_t)a->dim*4)return -1;
    for(int i=0;i<a->dim;++i){float v;memcpy(&v,(const char*)norm+(size_t)i*4,4);if(!isfinite(v))return -1;}
    BackendGLMResources *r=calloc(1,sizeof(*r));if(!r)return -1;
    r->anatomy=*a;r->model.gpu=gpu;r->model.n_layers=a->n_layers;r->model.dim=a->dim;r->model.vocab_size=a->vocab_size;r->model.anatomy=&r->anatomy;
    r->layers=calloc((size_t)a->n_layers,sizeof(*r->layers));if(!r->layers)goto fail;r->model.layers=r->layers;
    if(backend_glm_account(r,emb_bytes,emb_bytes,0,gpu_budget))goto fail;
    r->model.embedding=(BnGPUBF16LinearWeight){.type=BN_GGUF_TENSOR_Q8_0};
    r->model.embedding.raw=bn_gpu_backend_create_quant_only_buffer(gpu,emb,emb_bytes,BN_GGUF_TENSOR_Q8_0,a->vocab_size,a->dim);if(!r->model.embedding.raw)goto fail;
    if(backend_glm_account(r,out_bytes,out_bytes,0,gpu_budget))goto fail;
    r->model.output=(BnGPUBF16LinearWeight){.type=BN_GGUF_TENSOR_Q8_0};
    r->model.output.raw=bn_gpu_backend_create_quant_only_buffer(gpu,out,out_bytes,BN_GGUF_TENSOR_Q8_0,a->vocab_size,a->dim);if(!r->model.output.raw)goto fail;
    size_t decoded=(size_t)a->dim*2;
    if(backend_glm_account(r,decoded,norm_bytes+decoded,0,gpu_budget))goto fail;
    void *temporary=bn_gpu_backend_create_quant_only_buffer(gpu,norm,norm_bytes,BN_GGUF_TENSOR_F32,1,a->dim);if(!temporary)goto fail;
    r->model.output_norm=bn_gpu_backend_buffer_create_bf16_decoded(gpu,temporary);bn_gpu_backend_destroy_buffer(gpu,temporary);if(!r->model.output_norm)goto fail;
    for(int i=0;i<a->n_layers;++i){BnBackendBF16GLMLayer *l=&r->layers[i];const BnModelLayerAnatomy *shape=&r->anatomy.layers[i];
        l->expert_cache_layer=-1;
        if(shape->kind==BN_MODEL_ANATOMY_KDA){
            if(bn_backend_layout_bf16_kda_init(&l->kda,gpu,file,&r->anatomy,i,gpu_budget-r->model.bytes,host_budget)||
               backend_glm_account(r,l->kda.bytes,l->kda.bytes,l->kda.host_staging_bytes,gpu_budget))goto fail;
        }else{
            if(bn_backend_layout_bf16_mla_init(&l->mla,gpu,file,&r->anatomy,i,gpu_budget-r->model.bytes)||
               backend_glm_account(r,l->mla.bytes,l->mla.peak_bytes,0,gpu_budget))goto fail;
            if(bn_backend_layout_bf16_indexer_init(&l->indexer,gpu,file,&r->anatomy,i,gpu_budget-r->model.bytes)||
               backend_glm_account(r,l->indexer.bytes,l->indexer.bytes,0,gpu_budget))goto fail;
        }
        if(bn_backend_layout_bf16_mhc_init(&l->attention_mhc,gpu,file,&r->anatomy,i,0,gpu_budget-r->model.bytes,host_budget)||
           backend_glm_account(r,l->attention_mhc.bytes,l->attention_mhc.bytes,l->attention_mhc.host_staging_bytes,gpu_budget))goto fail;
        if(bn_backend_layout_bf16_mhc_init(&l->ffn_mhc,gpu,file,&r->anatomy,i,1,gpu_budget-r->model.bytes,host_budget)||
           backend_glm_account(r,l->ffn_mhc.bytes,l->ffn_mhc.bytes,l->ffn_mhc.host_staging_bytes,gpu_budget))goto fail;
        if(shape->uses_moe){
            l->expert_map=maps[i];
            if(bn_backend_layout_bf16_moe_init(&l->moe,gpu,file,&r->anatomy,i,&l->expert_map,gpu_budget-r->model.bytes,host_budget)||
               backend_glm_account(r,l->moe.bytes,l->moe.bytes,l->moe.host_staging_bytes,gpu_budget))goto fail;
        }else{
            if(bn_backend_layout_bf16_dense_ffn_init(&l->dense,gpu,file,&r->anatomy,i,gpu_budget-r->model.bytes,host_budget)||
               backend_glm_account(r,l->dense.bytes,l->dense.bytes,l->dense.host_staging_bytes,gpu_budget))goto fail;
        }
    }
    if(cache_budget&&backend_glm_cache_init(r,cache_budget,host_budget))goto fail;
    if(bn_backend_model_attach_gpu_resources(backend,&backend_glm_owner,r,backend_glm_release))goto fail;
    bn_backend_model_bind_gpu(backend,gpu);return 0;
fail:
    backend_glm_release(r);return -1;
}
