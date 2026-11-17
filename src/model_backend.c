#include "model_internal.h"
#include "backend_layout.h"
#include "backend_model.h"
#include "sh_log.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const BnTrellisRowSource *sources;
    const BnModelNativeAnatomy *anatomy;
    BnModelNativeRowsInfo info;
    BnBackendRowCache caches[BN_MODEL_ANATOMY_MAX_LAYERS];
} NativeRowsCache;
static const char native_rows_cache_owner=0;
static void native_rows_cache_free(void *ptr) {
    NativeRowsCache *r=ptr;
    for(int i=0;i<BN_MODEL_ANATOMY_MAX_LAYERS;i++)bn_backend_model_row_cache_free(r->caches+i);
    free(r);
}
static NativeRowsCache *native_rows_cache_get(const BnModel *m) {
    NativeRowsCache *r=bn_backend_model_cpu_resources(bn_model_backend(m),&native_rows_cache_owner);
    return r&&m->runtime&&r->sources==m->runtime->native_rows&&
        r->anatomy==bn_model_native_anatomy(m)?r:NULL;
}
int bn_model_prepare_native_row_caches(BnModel *m,size_t per,int max_rows,size_t budget) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(m);BnBackendModel *b=bn_model_backend(m);
    if(!a||!b||!m->runtime->native_rows||!per||max_rows<1)return -1;
    int tables=0;for(int i=0;i<a->experts.layers;i++)if(bn_model_native_row_source(m,i))tables++;
    if(!tables||per>(SIZE_MAX-sizeof(NativeRowsCache))/(size_t)tables)return -1;
    size_t bound=sizeof(NativeRowsCache)+(size_t)tables*per;
    if(budget<bound)return -1;
    NativeRowsCache *old=native_rows_cache_get(m);
    if(old)return old->info.max_rows==max_rows&&old->info.per_table_budget==per?0:-1;
    NativeRowsCache *r=calloc(1,sizeof(*r));if(!r)return -1;
    r->sources=m->runtime->native_rows;r->anatomy=a;
    r->info=(BnModelNativeRowsInfo){tables,max_rows,per,bound,sizeof(*r)};
    for(int i=0;i<a->experts.layers;i++) {
        const BnTrellisRowSource *source=bn_model_native_row_source(m,i);if(!source)continue;
        BnBackendRowCacheStats stats;
        if(bn_backend_model_row_cache_init(r->caches+i,source,per,(size_t)max_rows)||
            bn_backend_model_row_cache_stats(r->caches+i,&stats))goto fail;
        r->info.host_allocated+=stats.allocated_bytes;
    }
    if(bn_backend_model_attach_cpu_resources(b,&native_rows_cache_owner,r,native_rows_cache_free))goto fail;
    return 0;
fail:native_rows_cache_free(r);return -1;
}
BnBackendRowCache *bn_model_native_row_cache(const BnModel *m,int layer) {
    NativeRowsCache *r=native_rows_cache_get(m);
    return r&&layer>=0&&layer<r->anatomy->experts.layers&&r->caches[layer].state?r->caches+layer:NULL;
}
int bn_model_native_row_cache_info(const BnModel *m,BnModelNativeRowsInfo *out) {
    NativeRowsCache *r=native_rows_cache_get(m);if(!r||!out)return -1;*out=r->info;return 0;
}

BnBackendModel *bn_model_backend(const BnModel *model) {
    return (model && model->backend_state) ? model->backend_state->backend : NULL;
}

int bn_model_ensure_backend(BnModel *model) {
    if (!model) return -1;
    if (!model->backend_state) {
        model->backend_state = calloc(1, sizeof(*model->backend_state));
        if (!model->backend_state) return -1;
    }
    if (!model->backend_state->backend)
        model->backend_state->backend = bn_backend_model_create();
    return model->backend_state->backend ? 0 : -1;
}

size_t bn_model_backend_prepared_size(const BnConfig *config,
                                      const BnWeights *weights) {
    return bn_backend_layout_prepared_qweights_size(config, weights, NULL);
}

static void log_prepared_bytes(const char *message, size_t bytes) {
    if (!bytes) return;
    char mb[16];
    snprintf(mb, sizeof(mb), "%.0f", (double)bytes / (1024 * 1024));
    SH_LOG_INFO(message, "MB", mb);
}

void bn_model_backend_prepare(BnModel *model, SHArena *arena) {
    BnBackendLayoutPreparedStats stats = {0};
    bn_backend_layout_prepare_qweights(bn_model_backend(model), &model->config,
                                       &model->weights, arena, &stats);
    log_prepared_bytes("Q4_0 weights repacked", stats.lowbit_repack_bytes);
    log_prepared_bytes("Q8_0 weights repacked", stats.q8_repack_bytes);
    log_prepared_bytes("Q4_K scales prepared", stats.kquant_scale_table_bytes);
    log_prepared_bytes("Q6_K weights expanded", stats.expanded_kquant_weight_bytes);
    log_prepared_bytes("Q8_0 FP32 scales ready", stats.f32_scale_table_bytes);
}

void bn_model_set_cpu_prepared_cache_budget(BnModel *model,
                                            size_t budget_bytes) {
    bn_backend_model_set_cpu_prepared_cache_budget(bn_model_backend(model),
                                                   budget_bytes);
}

void bn_model_backend_free(BnModel *model) {
    if (!model || !model->backend_state) return;
    bn_backend_model_free(model->backend_state->backend);
}

int bn_model_prepare_bf16_topology(BnModel *model, BnGPUBackend *gpu,
                                  size_t device_budget, size_t host_budget) {
    const BnModelAnatomy *a = bn_model_anatomy(model);
    size_t count = 0;
    const BnMoEExpertMap *maps = bn_model_topology_experts(model, &count);
    BnGGUFFile *file = bn_model_topology_file(model);
    if (!a || !file || !maps || !gpu) return -1;
    return bn_backend_layout_prepare_bf16_glm_model(bn_model_backend(model),
        gpu, file, a, maps, count, device_budget, host_budget);
}

int bn_model_prepare_native_core(BnModel *model, BnGPUBackend *gpu,
    size_t device_budget, size_t host_budget) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(model);
    const BnSafetensorsIndex *index=bn_model_native_index(model);
    if(!a||!index||!bn_model_native_numerics(model)||!gpu||model->backend_state)return -1;
    struct BnModelBackendState *state=calloc(1,sizeof(*state));
    if(!state)return -1;
    state->backend=bn_backend_model_create();
    if(!state->backend||bn_backend_layout_prepare_native_core(state->backend,
        gpu,index,a,device_budget,host_budget)) {
        bn_backend_model_free(state->backend);free(state);return -1;
    }
    model->backend_state=state;return 0;
}

int bn_model_prepare_native_parameters(BnModel *model,size_t device_budget,size_t host_budget) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(model);
    const BnSafetensorsIndex *index=bn_model_native_index(model);
    if(!a||!index||!bn_model_native_numerics(model))return -1;
    return bn_backend_layout_prepare_native_parameters(bn_model_backend(model),
        index,a,device_budget,host_budget);
}

int bn_model_prepare_native_routers(BnModel *model,size_t device,size_t host) {
    const BnModelNativeAnatomy *a=bn_model_native_anatomy(model);
    const BnSafetensorsIndex *index=bn_model_native_index(model);
    if(!a||!index||!bn_model_native_numerics(model))return -1;
    return bn_backend_layout_prepare_native_routers(bn_model_backend(model),index,a,device,host);
}
