#ifndef BN_MOE_INTERNAL_H
#define BN_MOE_INTERNAL_H

#include "model.h"
#include "moe.h"
#include "session.h"
#include "platform.h"
#include "quant.h"
#include "sh_log.h"
#include "bn_alloc.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#ifndef __EMSCRIPTEN__
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#endif

typedef struct BnMoECache BnMoECache;
typedef struct BnMoEPrefetch BnMoEPrefetch;

typedef struct {
    float *hb;
    const float *gate;
    const float *up;
    int uses_reference_silu;
} BnSwiGLUCtx;

typedef struct {
    int uses_scaled_router_input;
    int uses_dense_residual_branch;
    int uses_reference_silu;
} BnMoEExecutionPolicy;

typedef struct {
    int requires_matvec_prefill;
    int uses_grouped_expert_route;
} BnMoEPrefillPolicy;

typedef struct {
    int total_experts;
    int active_experts;
    int expert_hidden_dim;
    int norm_topk_prob;
    float expert_weights_scale;
} BnMoERoutePolicy;

typedef struct {
    const BnQWeight *gate;
    const BnQWeight *up;
    const BnQWeight *down;
} BnMoESharedExpertWeights;

typedef struct {
    int has_loaded_path;
    int hidden_dim;
} BnMoELoadedSharedExpertPolicy;

int bn_moe_checked_mul_size(size_t a, size_t b, size_t *out);
int bn_moe_proj_info(const BnMoEExpertMap *map, int expert_idx, int proj,
                     size_t *offset, size_t *proj_bytes);
int bn_moe_io_has_mmap(const BnMoEIO *io);
const uint8_t *bn_moe_mmap_base_for_proj(const BnMoEIO *io,
                                         const BnMoEExpertMap *map,
                                         int proj);
#if !defined(__EMSCRIPTEN__)
void bn_moe_madvise_experts(const BnMoEIO *io, const BnMoEExpertMap *map,
                            const int *indices, int n, int advice, int proj_mask);
#endif
const void *bn_moe_load_expert_proj_into(const BnMoEIO *io, BnMoEStats *stats,
                                         const BnMoEExpertMap *map,
                                         int expert_idx, int proj,
                                         uint8_t *buf, size_t buf_size);
const void *bn_moe_load_expert_proj(const BnMoEIO *io, BnMoEState *ms,
                                    const BnMoEExpertMap *map,
                                    int expert_idx, int proj);
BnQWeight bn_moe_make_qweight(const void *data, int type, int rows, int cols);
uint32_t bn_moe_float_kquant_gateup_fallback_task_flags(const BnConfig *c);
BnMoEExecutionPolicy bn_moe_execution_policy(const BnConfig *c);
int bn_moe_policy_uses_reference_silu(const BnConfig *c);
BnMoEPrefillPolicy bn_moe_prefill_policy(const BnConfig *c);
BnMoERoutePolicy bn_moe_route_policy(const BnConfig *c);
int bn_moe_policy_uses_expert_weights(const BnConfig *c);
int bn_moe_policy_uses_all_active_two_expert_set(const BnConfig *c);
int bn_moe_policy_uses_all_active_two_expert_route(const BnConfig *c,
                                                   int dim);
int bn_moe_policy_uses_grouped_expert_route(const BnConfig *c);
int bn_moe_policy_normalizes_topk_route_weights(const BnConfig *c);
int bn_moe_policy_layer_has_router(const BnLayerWeights *lw);
int bn_moe_policy_has_shared_expert(const BnConfig *c,
                                    const BnLayerWeights *lw);
int bn_moe_policy_has_shared_expert_gate_vector(
    const BnLayerWeights *lw);
const float *bn_moe_shared_expert_gate_vector(const BnLayerWeights *lw);
int bn_moe_policy_has_loaded_shared_gate_projection(
    const BnLayerWeights *lw);
int bn_moe_policy_has_loaded_shared_expert_path(const BnConfig *c,
                                                const BnLayerWeights *lw);
int bn_moe_policy_has_loaded_shared_expert(const BnConfig *c,
                                           const BnLayerWeights *lw);
int bn_moe_policy_shared_expert_hidden_dim(const BnConfig *c);
BnMoELoadedSharedExpertPolicy
bn_moe_loaded_shared_expert_policy(const BnConfig *c,
                                   const BnLayerWeights *lw);
int bn_moe_policy_supports_resident_routed_ffn_shape(
    int dim,
    int expert_hidden_dim,
    const BnMoEExpertMap *em);
int bn_moe_policy_supports_resident_routed_ffn_layout(
    const BnConfig *c,
    const BnMoEExpertMap *em);
int bn_moe_policy_supports_gateup_split_layout(const BnMoEExpertMap *em);
int bn_moe_policy_supports_shared_gateup_batch_type(int shared_gate_type,
                                                    int shared_up_type,
                                                    int batch_type);
int bn_moe_policy_supports_shared_gateup_batch_type_on_cpu(
    int shared_gate_type,
    int shared_up_type,
    int batch_type,
    int mixed_shared_gateup_supported);
int bn_moe_policy_can_batch_loaded_shared_gateup(
    const BnMatvecTask *tasks,
    int n_tasks,
    const BnLayerWeights *lw);
int bn_moe_shared_expert_gateup_tasks(BnMatvecTask *tasks,
                                      float *gate_out,
                                      float *up_out,
                                      const BnLayerWeights *lw,
                                      uint32_t flags);
const BnQWeight *bn_moe_shared_expert_down_weight(const BnLayerWeights *lw);
int bn_moe_shared_expert_projection_weights(
    BnMoESharedExpertWeights *out,
    const BnLayerWeights *lw);
int bn_moe_quant_uses_embedded_tensor_scale(int type);
size_t bn_moe_quant_embedded_tensor_scale_offset(int type, int rows, int cols);
void bn_moe_quant_matvec(float *out,
                         const BnQWeight *W,
                         const float *x,
                         int8_t *quantized_buf,
                         BnThreadPool *pool);
void bn_moe_quant_matvec_batch(const BnMatvecTask *tasks,
                               int n_tasks,
                               const float *x,
                               int8_t *quantized_buf,
                               BnThreadPool *pool);
void bn_moe_quant_matvec_multi(const BnMatvecMultiTask *tasks,
                               int n_tasks,
                               int8_t *quantized_bufs,
                               BnThreadPool *pool);
void bn_moe_quant_matmul(float *out,
                         const BnQWeight *W,
                         const float *x,
                         int n_tokens,
                         int8_t *quantized_buf,
                         BnThreadPool *pool);
void bn_moe_quant_matvec_gateup_gpu_buffers(BnMatvecTask *tasks,
                                            const void **buffers,
                                            int n_tasks,
                                            const float *x,
                                            int8_t *quantized_buf,
                                            BnThreadPool *pool,
                                            BnGPUBackend *gpu);
void bn_moe_quant_matvec_down_gpu_buffer(float *out,
                                         const BnQWeight *W,
                                         void *W_buf,
                                         const float *x,
                                         int8_t *quantized_buf,
                                         BnThreadPool *pool,
                                         BnGPUBackend *gpu);
void bn_moe_swiglu_range(void *ctx, int start, int end);
void bn_moe_swiglu(float *hb, const float *gate, const float *up, int n,
                   int uses_reference_silu);
double bn_moe_time_ms(void);
void bn_moe_rmsnorm(float *out, const float *x, const float *w,
                    int size, float eps);
float bn_moe_dot_row(const float *row, const float *x, int dim);
float bn_moe_shared_expert_gate_weight(const BnLayerWeights *lw,
                                       const float *x,
                                       int dim);
int bn_moe_dot4_rows(float *out, const float *router_w, const float *x,
                     int dim, int start_expert);
void bn_moe_swiglu_silu(float *hb, const float *gate, const float *up,
                        int n, int uses_reference_silu);
int bn_moe_can_batch_shared_gateup(const BnMatvecTask *tasks, int n_tasks,
                                   int shared_gate_type, int shared_up_type);
void bn_moe_weighted_add(float *dst, const float *src, float weight, int n);
void bn_moe_residual_add(float *x, const float *r, int n);

const uint8_t *bn_moe_cache_lookup_internal(BnMoECache *c, int layer, int expert_idx);
uint8_t *bn_moe_cache_insert_internal(BnMoECache *c, int layer, int expert_idx);
size_t bn_moe_cache_gate_bytes(const BnMoECache *c);
size_t bn_moe_cache_up_bytes(const BnMoECache *c);

#if !defined(__EMSCRIPTEN__)
BnMoEPrefetch *bn_moe_prefetch_init_internal(int fd);
void bn_moe_prefetch_free_internal(BnMoEPrefetch *pf);
void bn_moe_prefetch_start2_internal(BnMoEPrefetch *pf,
                                     uint8_t *buf1, size_t size1, off_t off1,
                                     uint8_t *buf2, size_t size2, off_t off2);
void bn_moe_prefetch_start1_internal(BnMoEPrefetch *pf,
                                     uint8_t *buf1, size_t size1, off_t off1);
int bn_moe_prefetch_wait_internal(BnMoEPrefetch *pf);
void bn_moe_prefetch_collect_stats(BnMoEPrefetch *pf, BnMoEStats *stats);
#endif

#endif // BN_MOE_INTERNAL_H
