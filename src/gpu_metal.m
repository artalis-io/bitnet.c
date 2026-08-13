/*
 * gpu_metal.m — Native Metal compute backend for BnGPUBackend
 *
 * Implements the BnGPUBackend vtable using Apple Metal.
 * Unified memory (storageModeShared) — no staging buffers.
 * setBytes for uniforms — no ring buffer.
 * Runtime shader compilation from .metal source files.
 * precise transcendentals for SSM IEEE compliance.
 */

#ifdef BN_ENABLE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "gpu_metal.h"
#include "gpu_backend.h"
#include "gpu_policy.h"
#include "gpu_shader.h"
#include "backend_quant.h"
#include "quant.h"
#include "gguf.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>

/* Max tensor type enum value we index into (I2_S = 36, plus margin) */
#define BN_METAL_MAX_TYPES 40
#define BN_METAL_MAX_MOE_ROUTE_EXPERTS 256
#define BN_METAL_MAX_MOE_RESIDENT_LAYERS 256
#define BN_METAL_MOE_COPY_WORKERS 4
#define BN_METAL_MAX_MMAP_BUFFERS 8
#define BN_METAL_MMAP_BUFFER_OVERLAP (1ull * 1024ull * 1024ull * 1024ull)
#define BN_METAL_MMAP_RESIDENT_RESERVE_BYTES (4ull * 1024ull * 1024ull * 1024ull)

typedef struct BnMetalMoEResident BnMetalMoEResident;

/* ── Internal context ──────────────────────────────────────────────── */

typedef struct {
    const BnBackendRuntimePolicy *runtime_policy;
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pipelines[BN_METAL_MAX_TYPES];  /* matvec per quant type */
    id<MTLComputePipelineState> fwd_pipelines[BN_GPU_SHADER_COUNT]; /* forward-pass shaders */
    id<MTLComputePipelineState> moe_routed_q4_0_gateup_pipeline;
    id<MTLComputePipelineState> moe_routed_q4_0_down_pipeline;
    id<MTLComputePipelineState> moe_routed_q4k_down_pipeline;
    id<MTLComputePipelineState> moe_routed_q5k_down_pipeline;
    id<MTLComputePipelineState> moe_routed_q6k_down_pipeline;
    id<MTLComputePipelineState> moe_routed_q8k_quant_pipeline;
    id<MTLComputePipelineState> moe_route_logits_pipeline;
    id<MTLComputePipelineState> moe_route_capture_pipeline;
    id<MTLComputePipelineState> native_quant_pipeline;
    id<MTLComputePipelineState> q8_native_quant_matvec_pipeline;
    id<MTLComputePipelineState> specialized_native_quant_pipeline;
    id<MTLComputePipelineState> small_dense_native_quant_matvec_pipeline;
    id<MTLComputePipelineState> prepared_small_dense_native_quant_matvec_pipeline;
    id<MTLComputePipelineState> prepared_reference_native_quant_matvec_pipeline;
    id<MTLComputePipelineState> reference_native_quant_matvec_pipeline;
    id<MTLComputePipelineState> prepared_f32_matvec_pipeline;
    id<MTLComputePipelineState> prepared_f32_split_pipeline;
    id<MTLComputePipelineState> prepared_small_dense_native_quant_split_pipeline;
    id<MTLComputePipelineState> prepared_small_dense_native_quant_gateup_pipeline;
    id<MTLComputePipelineState> small_dense_native_quant_split_pipeline;
    id<MTLComputePipelineState> small_dense_native_quant_gateup_pipeline;
    id<MTLComputePipelineState> cpu_order_rmsnorm_pipeline;
    id<MTLComputePipelineState> reference_gqa_scores_pipeline;
    id<MTLComputePipelineState> reference_softmax_pipeline;
    id<MTLComputePipelineState> reference_gqa_combine_pipeline;
    id<MTLComputePipelineState> specialized_native_matvec_pipeline;
    id<MTLComputePipelineState> specialized_native_four_row_matvec_pipeline;
    id<MTLComputePipelineState> specialized_q4k_native_matvec_pipeline;
    id<MTLComputePipelineState> specialized_q5k_native_matvec_pipeline;
    id<MTLComputePipelineState> specialized_q4k_native_split_pipeline;
    id<MTLComputePipelineState> reference_q6k_matvec_pipeline;
    id<MTLComputePipelineState> reference_q5k_matvec_pipeline;
    id<MTLComputePipelineState> reference_q4k_matvec_pipeline;
    id<MTLComputePipelineState> borrowed_native_q4_matvec_pipeline;
    id<MTLComputePipelineState> argmax_pipeline;
    id<MTLComputePipelineState> argmax_reduce_pipeline;
    int small_dense_native_quant_enabled;
    int prepared_native_quant_enabled;
    int native_quant_barriers_enabled;
    int cpu_order_rmsnorm_enabled;
    int full_barriers_enabled;
    int barriers_disabled;
    int route_history_enabled;
    uint32_t reference_attention_stage_mask;

    /* GPU-resident activation buffers (storageModeShared) */
    id<MTLBuffer> act_bufs[BN_GPU_BUF_COUNT];
    size_t        act_sizes[BN_GPU_BUF_COUNT];

    /* Persistent scratch buffers for standalone matvec */
    id<MTLBuffer> x_buf;
    size_t        x_buf_size;
    id<MTLBuffer> out_buf;
    size_t        out_buf_size;
    id<MTLBuffer> native_quant_buf;
    size_t        native_quant_buf_size;
    id<MTLBuffer> native_quant_scales_buf;
    size_t        native_quant_scales_buf_size;
    id<MTLBuffer> native_quant_block_sums_buf;
    size_t        native_quant_block_sums_buf_size;
    id<MTLBuffer> argmax_result_buf;
    id<MTLBuffer> argmax_partials_buf;
    size_t        argmax_partials_buf_size;
    id<MTLBuffer> argmax_penalty_buf;
    size_t        argmax_penalty_buf_size;
    id<MTLCommandBuffer> pending_logits_cmd;
    /* Shader directory path */
    char shader_dir[256];

    /* Profiling */
    int gpu_frame;
    int gpu_profile;
    int native_quant_dispatches;
    int specialized_native_quant_dispatches;
    int small_dense_native_quant_matvec_dispatches;
    int small_dense_native_quant_split_dispatches;
    int routed_profile_types_printed;
    int ssm_profile_shape_printed;
    int small_dense_native_quant_gateup_dispatches;
    int argmax_calls;
    id<MTLBuffer> route_history_buf;
    size_t route_history_capacity;
    size_t route_history_count;
    int route_history_stride;
    int route_history_layers;
    int route_history_shape_printed;

    /* Slab allocator for MoE weight suballocation */
    id<MTLBuffer> slab_buf;
    size_t        slab_size;
    struct { size_t offset, size; } *slab_free;
    int           slab_free_count;
    int           slab_free_cap;
    BnMetalMoEResident *moe_resident;
    int           moe_resident_count;
    uint64_t      moe_resident_clock;
    size_t        moe_resident_hits;
    size_t        moe_resident_misses;
    int           moe_resident_layout_printed;
    size_t        moe_resident_budget;
    size_t        moe_resident_bytes;
    int           moe_resident_layer_count;

    /* Zero-copy mmap range (Phase 5) */
    const void   *mmap_base;
    size_t        mmap_size;
    id<MTLBuffer> mmap_bufs[BN_METAL_MAX_MMAP_BUFFERS];
    size_t        mmap_buf_starts[BN_METAL_MAX_MMAP_BUFFERS];
    size_t        mmap_buf_sizes[BN_METAL_MAX_MMAP_BUFFERS];
    int           mmap_buf_count;
    size_t        mmap_buf_offset;
    id            mmap_residency_set;
    int           mmap_fits_working_set;
    int           mmap_prefaulted;
    uint8_t       mmap_prefault_checksum;
} BnMetalCtx;

static int metal_reset_activations(void *vctx);

static int metal_mmap_fits_resident_budget(BnMetalCtx *ctx, size_t size)
{
    if (!ctx || size == 0) return 0;
    size_t working_set =
        (size_t)ctx->device.recommendedMaxWorkingSetSize;
    return working_set > BN_METAL_MMAP_RESIDENT_RESERVE_BYTES &&
        size <= working_set - BN_METAL_MMAP_RESIDENT_RESERVE_BYTES;
}

static int metal_memory_info(void *vctx,
                             size_t *free_bytes,
                             size_t *total_bytes)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ctx->device || !free_bytes || !total_bytes)
        return -1;
    size_t total = (size_t)ctx->device.recommendedMaxWorkingSetSize;
    size_t allocated = (size_t)ctx->device.currentAllocatedSize;
    if (total == 0)
        return -1;
    *total_bytes = total;
    *free_bytes = allocated < total ? total - allocated : 0;
    return 0;
}

static void metal_prefault_all_mmap(BnMetalCtx *ctx);

static void metal_prefault_routed_mmap(BnMetalCtx *ctx,
                                       const BnGPUOp *ops,
                                       int n_ops)
{
    if (!ctx || !ops || n_ops <= 0 || !ctx->mmap_base ||
        ctx->mmap_size == 0 || !ctx->mmap_fits_working_set ||
        ctx->mmap_prefaulted)
        return;
    int uses_routed_moe = 0;
    for (int i = 0; i < n_ops; i++) {
        if (ops[i].op_code == BN_GPU_CODE_MOE_ROUTED_FFN) {
            uses_routed_moe = 1;
            break;
        }
    }
    if (!uses_routed_moe) return;

    metal_prefault_all_mmap(ctx);
}

static void metal_prefault_all_mmap(BnMetalCtx *ctx)
{
    if (!ctx || !ctx->mmap_base || ctx->mmap_size == 0 ||
        !ctx->mmap_fits_working_set || ctx->mmap_prefaulted)
        return;

    double t0 = bn_platform_time_ms();
    const volatile uint8_t *bytes =
        (const volatile uint8_t *)ctx->mmap_base;
    size_t page = (size_t)getpagesize();
    uint8_t checksum = 0;
    for (size_t offset = 0; offset < ctx->mmap_size; offset += page)
        checksum ^= bytes[offset];
    checksum ^= bytes[ctx->mmap_size - 1];
    ctx->mmap_prefault_checksum = checksum;
    ctx->mmap_prefaulted = 1;
    fprintf(stderr,
            "[bn:gpu:metal] routed mmap resident: %.1f MB in %.0f ms\n",
            (double)ctx->mmap_size / (1024.0 * 1024.0),
            bn_platform_time_ms() - t0);
}

static void metal_request_mmap_residency(BnMetalCtx *ctx)
{
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (ctx && ctx->mmap_residency_set)
            [ctx->mmap_residency_set requestResidency];
    }
#else
    (void)ctx;
#endif
}

static int metal_prepare_cpu_operations(void *vctx)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx) return -1;
    metal_request_mmap_residency(ctx);
    metal_prefault_all_mmap(ctx);
    return 0;
}

static void metal_release_mmap_residency(BnMetalCtx *ctx)
{
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (ctx && ctx->mmap_residency_set) {
            [ctx->mmap_residency_set endResidency];
            [ctx->mmap_residency_set removeAllAllocations];
            ctx->mmap_residency_set = nil;
        }
    }
#else
    (void)ctx;
#endif
}

static void metal_create_mmap_residency_set(BnMetalCtx *ctx)
{
    if (!ctx || ctx->mmap_buf_count <= 0 || ctx->mmap_residency_set)
        return;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (!metal_mmap_fits_resident_budget(ctx, ctx->mmap_size))
            return;
        ctx->mmap_fits_working_set = 1;
        MTLResidencySetDescriptor *desc =
            [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"bitnet-model";
        desc.initialCapacity = (NSUInteger)ctx->mmap_buf_count;
        NSError *error = nil;
        id rset = [ctx->device newResidencySetWithDescriptor:desc error:&error];
        if (!rset || error)
            return;
        for (int i = 0; i < ctx->mmap_buf_count; i++)
            [rset addAllocation:ctx->mmap_bufs[i]];
        [rset commit];
        ctx->mmap_residency_set = rset;
        metal_request_mmap_residency(ctx);
    }
#endif
}

static int metal_flush_pending_logits(BnMetalCtx *ctx)
{
    if (!ctx || !ctx->pending_logits_cmd) return 0;
    id<MTLCommandBuffer> cmd = ctx->pending_logits_cmd;
    [cmd waitUntilCompleted];
    ctx->pending_logits_cmd = nil;
    return [cmd status] == MTLCommandBufferStatusCompleted ? 0 : -1;
}

typedef struct {
    int shader;
    int type;
    uint32_t rows;
    uint32_t cols;
    uint32_t aux;
    int count;
    double gpu_ms;
    double wall_ms;
} BnMetalProfileShape;

static const char *metal_shader_profile_name(int shader)
{
    static const char *names[] = {
        "matvec","rmsnorm","rope","gqa_scores","softmax","gqa_combine",
        "silu_gate","relu2_gate","resid_add","copy","bias_add","resid_rmsnorm",
        "weighted_add","ssm_conv","ssm_l2norm","ssm_ab","ssm_delta","ssm_gate",
        "per_head_norm","deinterleave_q","sigmoid_gate","flash_attn",
        "matvec_split","rope_qk","fused_gateup","ssm_ab_split","q4k_split",
        "q8_split","q5k_split","silu_act","relu2_act","gelu_gate",
        "weighted_add_sigmoid","moe_route_topk","moe_routed_ffn"
    };
    if (shader >= 0 && shader < (int)(sizeof(names) / sizeof(names[0])))
        return names[shader];
    return "?";
}

static void metal_profile_add_shape(BnMetalProfileShape *shapes,
                                    int *n_shapes,
                                    int max_shapes,
                                    int shader,
                                    int type,
                                    uint32_t rows,
                                    uint32_t cols,
                                    uint32_t aux)
{
    for (int i = 0; i < *n_shapes; i++) {
        if (shapes[i].shader == shader && shapes[i].type == type &&
            shapes[i].rows == rows &&
            shapes[i].cols == cols && shapes[i].aux == aux) {
            shapes[i].count++;
            return;
        }
    }
    if (*n_shapes >= max_shapes) return;
    shapes[*n_shapes].shader = shader;
    shapes[*n_shapes].type = type;
    shapes[*n_shapes].rows = rows;
    shapes[*n_shapes].cols = cols;
    shapes[*n_shapes].aux = aux;
    shapes[*n_shapes].count = 1;
    shapes[*n_shapes].gpu_ms = 0.0;
    shapes[*n_shapes].wall_ms = 0.0;
    (*n_shapes)++;
}

static void metal_profile_add_shape_time(BnMetalProfileShape *shapes,
                                         int *n_shapes,
                                         int max_shapes,
                                         int shader,
                                         int type,
                                         uint32_t rows,
                                         uint32_t cols,
                                         uint32_t aux,
                                         double gpu_ms,
                                         double wall_ms)
{
    for (int i = 0; i < *n_shapes; i++) {
        if (shapes[i].shader == shader && shapes[i].type == type &&
            shapes[i].rows == rows &&
            shapes[i].cols == cols && shapes[i].aux == aux) {
            shapes[i].count++;
            shapes[i].gpu_ms += gpu_ms;
            shapes[i].wall_ms += wall_ms;
            return;
        }
    }
    if (*n_shapes >= max_shapes) return;
    shapes[*n_shapes].shader = shader;
    shapes[*n_shapes].type = type;
    shapes[*n_shapes].rows = rows;
    shapes[*n_shapes].cols = cols;
    shapes[*n_shapes].aux = aux;
    shapes[*n_shapes].count = 1;
    shapes[*n_shapes].gpu_ms = gpu_ms;
    shapes[*n_shapes].wall_ms = wall_ms;
    (*n_shapes)++;
}

/* ── GPU buffer handle ─────────────────────────────────────────────── */

typedef struct {
    id<MTLBuffer> buf;
    size_t        size;
    size_t        offset;       /* byte offset into slab (0 for standalone) */
    int           type;
    int           rows;
    int           cols;
    uint32_t      bias_offset;  /* u32 offset for fused bias, 0 = none */
    int           is_slab;
    int           is_borrowed;
    int           native_quant_prepared;
    int           native_matvec_layout;
} BnMetalBuf;

static size_t slab_alloc(BnMetalCtx *ctx, size_t size);
static void slab_free_range(BnMetalCtx *ctx, size_t offset, size_t size);

struct BnMetalMoEResident {
    BnMetalBuf *source_gate;
    BnMetalBuf *source_up;
    BnMetalBuf *source_down;
    BnMetalBuf gate;
    BnMetalBuf up;
    BnMetalBuf down;
    int *experts;
    uint64_t *ages;
    int slots;
};

static BnMetalMoEResident *metal_find_moe_resident(
    BnMetalCtx *ctx, BnMetalBuf *gate, BnMetalBuf *up, BnMetalBuf *down)
{
    for (int i = 0; i < ctx->moe_resident_count; i++) {
        BnMetalMoEResident *entry = &ctx->moe_resident[i];
        if (entry->source_gate == gate && entry->source_up == up &&
            entry->source_down == down)
            return entry;
    }
    return NULL;
}

static int metal_init_moe_resident_buffer(BnMetalCtx *ctx,
                                          BnMetalBuf *out,
                                          const BnMetalBuf *source,
                                          size_t size)
{
    if (!ctx || !out || !source || size == 0)
        return -1;
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:size
                                          options:MTLResourceStorageModeShared];
    if (!buffer) return -1;
    memset((void *)out, 0, sizeof(*out));
    out->buf = buffer;
    out->size = size;
    out->type = source->type;
    out->rows = source->rows;
    out->cols = source->cols;
    return 0;
}

static BnMetalMoEResident *metal_create_moe_resident(
    BnMetalCtx *ctx, BnMetalBuf *gate, BnMetalBuf *up, BnMetalBuf *down,
    int slots, uint32_t gate_up_stride, uint32_t down_stride)
{
    if (!ctx || !gate || !up || !down || slots <= 0 ||
        gate_up_stride == 0 || down_stride == 0 ||
        (size_t)slots > SIZE_MAX / gate_up_stride ||
        (size_t)slots > SIZE_MAX / down_stride)
        return NULL;
    size_t gate_up_size = (size_t)slots * gate_up_stride;
    size_t down_size = (size_t)slots * down_stride;
    if (gate_up_size > (SIZE_MAX - down_size) / 2)
        return NULL;
    size_t total = gate_up_size * 2 + down_size;
    if (ctx->moe_resident_budget == 0 ||
        ctx->moe_resident_bytes > ctx->moe_resident_budget ||
        total > ctx->moe_resident_budget - ctx->moe_resident_bytes)
        return NULL;
    if (!ctx->moe_resident) {
        ctx->moe_resident = (BnMetalMoEResident *)calloc(
            BN_METAL_MAX_MOE_RESIDENT_LAYERS, sizeof(*ctx->moe_resident));
        if (!ctx->moe_resident) return NULL;
    }
    if (ctx->moe_resident_count >= BN_METAL_MAX_MOE_RESIDENT_LAYERS)
        return NULL;
    BnMetalMoEResident *entry =
        &ctx->moe_resident[ctx->moe_resident_count];
    memset((void *)entry, 0, sizeof(*entry));
    entry->experts = (int *)malloc((size_t)slots * sizeof(int));
    entry->ages = (uint64_t *)calloc((size_t)slots, sizeof(uint64_t));
    if (!entry->experts || !entry->ages) goto fail;
    for (int i = 0; i < slots; i++) entry->experts[i] = -1;
    if (metal_init_moe_resident_buffer(
            ctx, &entry->gate, gate, gate_up_size) != 0 ||
        metal_init_moe_resident_buffer(
            ctx, &entry->up, up, gate_up_size) != 0 ||
        metal_init_moe_resident_buffer(
            ctx, &entry->down, down, down_size) != 0)
        goto fail;
    entry->source_gate = gate;
    entry->source_up = up;
    entry->source_down = down;
    entry->slots = slots;
    ctx->moe_resident_bytes += total;
    ctx->moe_resident_count++;
    return entry;

fail:
    entry->gate.buf = nil;
    entry->up.buf = nil;
    entry->down.buf = nil;
    free(entry->experts);
    free(entry->ages);
    memset((void *)entry, 0, sizeof(*entry));
    return NULL;
}

static int metal_moe_resident_slot(BnMetalCtx *ctx,
                                   BnMetalMoEResident *entry,
                                   int expert,
                                   int *miss)
{
    if (miss) *miss = 0;
    int slot = -1;
    for (int i = 0; i < entry->slots; i++) {
        if (entry->experts[i] == expert) {
            entry->ages[i] = ++ctx->moe_resident_clock;
            ctx->moe_resident_hits++;
            return i;
        }
        if (slot < 0 ||
            (entry->experts[slot] >= 0 && entry->experts[i] < 0) ||
            (entry->experts[i] >= 0 && entry->ages[i] < entry->ages[slot]))
            slot = i;
    }
    if (slot < 0) return -1;
    entry->experts[slot] = expert;
    entry->ages[slot] = ++ctx->moe_resident_clock;
    ctx->moe_resident_misses++;
    if (miss) *miss = 1;
    return slot;
}

static BnMetalMoEResident *metal_prepare_moe_resident(
    BnMetalCtx *ctx, BnMetalBuf *gate, BnMetalBuf *up, BnMetalBuf *down,
    float *route, int n_experts, int k, uint32_t gate_up_stride,
    uint32_t down_stride, int hidden, int dim)
{
    BnQWeight gate_weight = {
        .data = (void *)1, .type = gate ? gate->type : -1,
        .rows = hidden, .cols = dim, .scale = 1.0f,
    };
    BnQWeight up_weight = {
        .data = (void *)1, .type = up ? up->type : -1,
        .rows = hidden, .cols = dim, .scale = 1.0f,
    };
    BnQWeight down_weight = {
        .data = (void *)1, .type = down ? down->type : -1,
        .rows = dim, .cols = hidden, .scale = 1.0f,
    };
    size_t gate_bytes = bn_qweight_data_size(&gate_weight);
    size_t up_bytes = bn_qweight_data_size(&up_weight);
    size_t down_bytes = bn_qweight_data_size(&down_weight);
    size_t slot_bytes = (size_t)gate_up_stride * 2 + down_stride;
    int layer_count = ctx && ctx->moe_resident_layer_count > 0
        ? ctx->moe_resident_layer_count : 1;
    size_t per_layer_budget = ctx && layer_count > 0
        ? ctx->moe_resident_budget / (size_t)layer_count : 0;
    size_t slot_budget = slot_bytes > 0 ? per_layer_budget / slot_bytes : 0;
    if (slot_budget > (size_t)n_experts) slot_budget = (size_t)n_experts;
    int resident_slots = slot_budget > (size_t)INT_MAX
        ? INT_MAX : (int)slot_budget;
    resident_slots -= k > 0 ? resident_slots % k : resident_slots;
    if (ctx && ctx->gpu_profile && !ctx->moe_resident_layout_printed) {
        fprintf(stderr,
                "[gpu:metal:moe-cache-layout] gate=%zu/%zu up=%zu/%zu "
                "down=%zu/%zu n=%d stride=%u/%u shape=%dx%d mmap=%d\n",
                gate ? gate->size : 0, gate_bytes,
                up ? up->size : 0, up_bytes,
                down ? down->size : 0, down_bytes,
                n_experts, gate_up_stride, down_stride, hidden, dim,
                gate && up && down && gate->is_borrowed &&
                    up->is_borrowed && down->is_borrowed);
        ctx->moe_resident_layout_printed = 1;
    }
    if (!ctx || !gate || !up || !down || !route || n_experts <= 0 ||
        k <= 0 || resident_slots < k ||
        hidden <= 0 || dim <= 0 ||
        ctx->moe_resident_budget == 0 ||
        !gate->is_borrowed || !up->is_borrowed || !down->is_borrowed ||
        gate_up_stride == 0 ||
        down_stride == 0 || gate_bytes == 0 || up_bytes == 0 ||
        down_bytes == 0 || gate_bytes > gate_up_stride ||
        up_bytes > gate_up_stride || down_bytes > down_stride ||
        (size_t)(n_experts - 1) >
            (SIZE_MAX - gate_bytes) / gate_up_stride ||
        (size_t)(n_experts - 1) >
            (SIZE_MAX - up_bytes) / gate_up_stride ||
        (size_t)(n_experts - 1) >
            (SIZE_MAX - down_bytes) / down_stride ||
        gate->size < (size_t)(n_experts - 1) * gate_up_stride + gate_bytes ||
        up->size < (size_t)(n_experts - 1) * gate_up_stride + up_bytes ||
        down->size < (size_t)(n_experts - 1) * down_stride + down_bytes)
        return NULL;
    for (int i = 0; i < k; i++) {
        int expert = (int)(route[k + i] + 0.5f);
        if (expert < 0 || expert >= n_experts) return NULL;
    }
    BnMetalMoEResident *entry =
        metal_find_moe_resident(ctx, gate, up, down);
    if (!entry)
        entry = metal_create_moe_resident(
            ctx, gate, up, down,
            resident_slots,
            gate_up_stride, down_stride);
    if (!entry) return NULL;
    int experts[BN_MAX_MOE_K];
    int slots[BN_MAX_MOE_K];
    int misses[BN_MAX_MOE_K];
    for (int i = 0; i < k; i++) {
        int expert = (int)(route[k + i] + 0.5f);
        int slot = metal_moe_resident_slot(
            ctx, entry, expert, &misses[i]);
        if (slot < 0) return NULL;
        experts[i] = expert;
        slots[i] = slot;
        route[k + i] = (float)slot;
    }
    const int *expert_list = experts;
    const int *slot_list = slots;
    const int *miss_list = misses;
    int miss_count = 0;
    for (int i = 0; i < k; i++) miss_count += misses[i] != 0;
    int copy_workers = miss_count < BN_METAL_MOE_COPY_WORKERS
        ? miss_count : BN_METAL_MOE_COPY_WORKERS;
    dispatch_apply((size_t)copy_workers,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t worker) {
        int ordinal = 0;
        for (int i = 0; i < k; i++) {
            if (!miss_list[i]) continue;
            if (ordinal++ % copy_workers != (int)worker) continue;
            size_t gate_offset = (size_t)expert_list[i] * gate_up_stride;
            size_t down_offset = (size_t)expert_list[i] * down_stride;
            size_t gate_dst = (size_t)slot_list[i] * gate_up_stride;
            size_t down_dst = (size_t)slot_list[i] * down_stride;
            memcpy((uint8_t *)[entry->gate.buf contents] + gate_dst,
                   (const uint8_t *)[entry->source_gate->buf contents] +
                       entry->source_gate->offset + gate_offset,
                   gate_bytes);
            memcpy((uint8_t *)[entry->up.buf contents] + gate_dst,
                   (const uint8_t *)[entry->source_up->buf contents] +
                       entry->source_up->offset + gate_offset,
                   up_bytes);
            memcpy((uint8_t *)[entry->down.buf contents] + down_dst,
                   (const uint8_t *)[entry->source_down->buf contents] +
                       entry->source_down->offset + down_offset,
                   down_bytes);
        }
    });
    return entry;
}

/* ── Shader compilation ────────────────────────────────────────────── */

static id<MTLComputePipelineState> compile_shader_with_math(
    BnMetalCtx *ctx,
    const char *dir,
    const char *filename,
    const char *fn_name,
    int precise_math)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);

    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:nsPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
    if (!source) return nil;

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = precise_math ? MTLMathModeSafe : MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = precise_math ? NO : YES;
#pragma clang diagnostic pop
    }
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:source
                                                   options:opts
                                                     error:&err];
    if (!lib) {
        fprintf(stderr, "[bn:gpu:metal] shader compile error (%s): %s\n",
                filename, [[err localizedDescription] UTF8String]);
        return nil;
    }

    NSString *fnName = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [lib newFunctionWithName:fnName];
    if (!fn) {
        fprintf(stderr, "[bn:gpu:metal] function '%s' not found in %s\n",
                fn_name, filename);
        return nil;
    }

    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
    if (!pso) {
        fprintf(stderr, "[bn:gpu:metal] pipeline error (%s): %s\n",
                filename, [[err localizedDescription] UTF8String]);
    }
    return pso;
}

static id<MTLComputePipelineState> compile_shader(BnMetalCtx *ctx,
                                                   const char *dir,
                                                   const char *filename,
                                                   const char *fn_name)
{
    return compile_shader_with_math(ctx, dir, filename, fn_name, 0);
}

static int compile_matvec_pipeline(BnMetalCtx *ctx, int type, const char *dir)
{
    const char *name = bn_quant_format_gpu_shader_name(type);
    if (!name) return -1;
    if (type < 0 || type >= BN_METAL_MAX_TYPES) return -1;

    char filename[64], fn_name[64];
    snprintf(filename, sizeof(filename), "%s_matvec.metal", name);
    snprintf(fn_name, sizeof(fn_name), "%s_matvec", name);

    id<MTLComputePipelineState> pso = compile_shader(ctx, dir, filename, fn_name);
    if (!pso) return -1;

    ctx->pipelines[type] = pso;
    return 0;
}

static id<MTLBuffer> metal_new_weight_buffer(BnMetalCtx *ctx,
                                             const void *data,
                                             size_t size)
{
    if (!ctx || !data || size == 0) return nil;
    if (bn_gpu_policy_metal_shared_weights_enabled(ctx->runtime_policy)) {
        return [ctx->device newBufferWithBytes:data
                                        length:size
                                       options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> dst = [ctx->device newBufferWithLength:size
                                                  options:MTLResourceStorageModePrivate];
    id<MTLBuffer> staging = [ctx->device newBufferWithBytes:data
                                                     length:size
                                                    options:MTLResourceStorageModeShared];
    if (!dst || !staging) return nil;

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] == MTLCommandBufferStatusError) return nil;
    return dst;
}

static void *metal_buffer_create_borrowed(void *vctx, const void *data,
                                          size_t size, int type,
                                          int rows, int cols);
static void *metal_buffer_create_quant_only(void *vctx, const void *data,
                                             size_t size, int type,
                                             int rows, int cols);

static int metal_native_matvec_borrowed_supported(
    void *vctx, size_t size, int type, int rows, int cols)
{
    (void)vctx;
    if (type != BN_GGUF_TENSOR_Q4_0 || rows <= 0 || cols <= 0 ||
        (cols % 32) != 0)
        return 0;
    size_t blocks_per_row = (size_t)cols / 32;
    return (size_t)rows <= SIZE_MAX / blocks_per_row / 18 &&
           size == (size_t)rows * blocks_per_row * 18;
}

static size_t metal_buffer_cache_charge(
    void *vctx, size_t size, int type, int rows, int cols)
{
    if (metal_native_matvec_borrowed_supported(
            vctx, size, type, rows, cols))
        return sizeof(BnMetalBuf);
    return size;
}

/* ── Slab allocator ────────────────────────────────────────────────── */

static int slab_init(BnMetalCtx *ctx, size_t size)
{
    if (ctx->slab_buf || size == 0) return -1;
    ctx->slab_buf = [ctx->device newBufferWithLength:size
                                             options:MTLResourceStorageModeShared];
    if (!ctx->slab_buf) return -1;
    ctx->slab_size = size;
    ctx->slab_free_cap = 256;
    ctx->slab_free = calloc((size_t)ctx->slab_free_cap,
                            sizeof(ctx->slab_free[0]));
    if (!ctx->slab_free) return -1;
    ctx->slab_free[0].offset = 0;
    ctx->slab_free[0].size = size;
    ctx->slab_free_count = 1;
    return 0;
}

static void *metal_buffer_create_native_matvec_borrowed(
    void *vctx, const void *data, size_t size, int type, int rows, int cols)
{
    if (!metal_native_matvec_borrowed_supported(
            vctx, size, type, rows, cols))
        return NULL;
    BnMetalBuf *buf = (BnMetalBuf *)metal_buffer_create_borrowed(
        vctx, data, size, type, rows, cols);
    if (buf)
        buf->native_matvec_layout = 1;
    return buf;
}

static size_t slab_alloc(BnMetalCtx *ctx, size_t size)
{
    size = (size + 255) & ~(size_t)255;  /* 256-byte align */
    for (int i = 0; i < ctx->slab_free_count; i++) {
        if (ctx->slab_free[i].size >= size) {
            size_t offset = ctx->slab_free[i].offset;
            ctx->slab_free[i].offset += size;
            ctx->slab_free[i].size -= size;
            if (ctx->slab_free[i].size == 0) {
                ctx->slab_free[i] = ctx->slab_free[--ctx->slab_free_count];
            }
            return offset;
        }
    }
    return (size_t)-1;
}

static void slab_free_range(BnMetalCtx *ctx, size_t offset, size_t size)
{
    if (ctx->slab_free_count >= ctx->slab_free_cap) {
        ctx->slab_free_cap *= 2;
        ctx->slab_free = realloc(ctx->slab_free,
                          (size_t)ctx->slab_free_cap * sizeof(ctx->slab_free[0]));
    }
    ctx->slab_free[ctx->slab_free_count].offset = offset;
    ctx->slab_free[ctx->slab_free_count].size = size;
    ctx->slab_free_count++;
}

/* ── Vtable: buffer_create ─────────────────────────────────────────── */

static BnMetalBuf *metal_repack_native_quant_for_gpu(BnMetalCtx *ctx,
                                             const void *data,
                                             size_t size,
                                             int type,
                                             int rows,
                                             int cols,
                                             const float *bias,
                                             int bias_len,
                                             int allow_prepared)
{
    (void)size;
    if (!ctx || !data || rows <= 0 || cols <= 0 || (cols % 32) != 0)
        return NULL;

    int blocks_per_row = cols / 32;
    int n_blocks = rows * blocks_per_row;
    if (allow_prepared &&
        (ctx->prepared_native_quant_enabled ||
         bn_gpu_policy_metal_native_quant_prepared_upload_enabled(
             ctx->runtime_policy)) &&
        (rows % 4) == 0) {
        int n_groups = rows / 4;
        size_t n_group_blocks = (size_t)n_groups * (size_t)blocks_per_row;
        size_t scale_bytes = n_group_blocks * 4 * sizeof(uint16_t);
        size_t qs_offset = (scale_bytes + 3) & ~(size_t)3;
        size_t qs_bytes = n_group_blocks * 64;
        size_t bias_bytes = (bias && bias_len > 0)
            ? (size_t)bias_len * sizeof(float) : 0;
        size_t bias_offset_bytes = qs_offset + qs_bytes;
        size_t prepared_size = (bias_offset_bytes + bias_bytes + 3) & ~(size_t)3;
        uint8_t *prepared_data = (uint8_t *)calloc(1, prepared_size);
        if (!prepared_data) return NULL;

        uint16_t *scales = (uint16_t *)prepared_data;
        uint8_t *qs_out = prepared_data + qs_offset;
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)data;
        for (int g = 0; g < n_groups; g++) {
            for (int b = 0; b < blocks_per_row; b++) {
                size_t gb = (size_t)g * (size_t)blocks_per_row + (size_t)b;
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) *
                                 (size_t)blocks_per_row + (size_t)b;
                    scales[gb * 4 + (size_t)r] = blocks[src].d;
                }
                uint8_t *dst = qs_out + gb * 64;
                for (int ng = 0; ng < 4; ng++) {
                    for (int r = 0; r < 4; r++) {
                        size_t src = (size_t)(g * 4 + r) *
                                     (size_t)blocks_per_row + (size_t)b;
                        const uint8_t *qs = blocks[src].qs + ng * 4;
                        uint8_t *dp = dst + ng * 16 + r * 4;
                        for (int j = 0; j < 4; j++)
                            dp[j] = qs[j];
                    }
                }
            }
        }
        uint32_t bias_offset = 0;
        if (bias && bias_len > 0) {
            bias_offset = (uint32_t)(bias_offset_bytes / sizeof(uint32_t));
            memcpy(prepared_data + bias_offset_bytes, bias,
                   (size_t)bias_len * sizeof(float));
        }

        BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
        if (!buf) {
            free(prepared_data);
            return NULL;
        }
        buf->buf = metal_new_weight_buffer(ctx, prepared_data, prepared_size);
        free(prepared_data);
        if (!buf->buf) {
            free(buf);
            return NULL;
        }
        buf->size = prepared_size;
        buf->offset = 0;
        buf->type = bn_gpu_policy_metal_repacked_buffer_type(type);
        buf->rows = rows;
        buf->cols = cols;
        buf->bias_offset = bias_offset;
        buf->native_quant_prepared = 1;
        return buf;
    }

    size_t scale_bytes = (size_t)n_blocks * sizeof(uint16_t);
    size_t nibble_offset = (scale_bytes + 3) & ~(size_t)3;
    size_t base_size = nibble_offset +
                       (size_t)n_blocks * 4 * sizeof(uint32_t);
    size_t bias_bytes = (bias && bias_len > 0) ?
                        (size_t)bias_len * sizeof(float) : 0;
    size_t repacked_size = (base_size + bias_bytes + 3) & ~(size_t)3;

    uint8_t *repacked = (uint8_t *)calloc(1, repacked_size);
    if (!repacked) return NULL;

    uint16_t *scales = (uint16_t *)repacked;
    uint8_t *nibbles = repacked + nibble_offset;
    const uint8_t *src = (const uint8_t *)data;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + (size_t)b * 18;
        uint16_t d_bits = (uint16_t)(block[0] | (block[1] << 8));
        scales[b] = d_bits;

        uint8_t *dst_nib = nibbles + (size_t)b * 16;
        const uint8_t *qs = block + 2;
        dst_nib[0]  = (qs[0] & 0x0F) | ((qs[1] & 0x0F) << 4);
        dst_nib[1]  = (qs[2] & 0x0F) | ((qs[3] & 0x0F) << 4);
        dst_nib[2]  = (qs[4] & 0x0F) | ((qs[5] & 0x0F) << 4);
        dst_nib[3]  = (qs[6] & 0x0F) | ((qs[7] & 0x0F) << 4);
        dst_nib[4]  = (qs[8] & 0x0F) | ((qs[9] & 0x0F) << 4);
        dst_nib[5]  = (qs[10] & 0x0F) | ((qs[11] & 0x0F) << 4);
        dst_nib[6]  = (qs[12] & 0x0F) | ((qs[13] & 0x0F) << 4);
        dst_nib[7]  = (qs[14] & 0x0F) | ((qs[15] & 0x0F) << 4);
        dst_nib[8]  = (qs[0] >> 4) | ((qs[1] >> 4) << 4);
        dst_nib[9]  = (qs[2] >> 4) | ((qs[3] >> 4) << 4);
        dst_nib[10] = (qs[4] >> 4) | ((qs[5] >> 4) << 4);
        dst_nib[11] = (qs[6] >> 4) | ((qs[7] >> 4) << 4);
        dst_nib[12] = (qs[8] >> 4) | ((qs[9] >> 4) << 4);
        dst_nib[13] = (qs[10] >> 4) | ((qs[11] >> 4) << 4);
        dst_nib[14] = (qs[12] >> 4) | ((qs[13] >> 4) << 4);
        dst_nib[15] = (qs[14] >> 4) | ((qs[15] >> 4) << 4);
    }

    uint32_t bias_offset = 0;
    if (bias && bias_len > 0) {
        bias_offset = (uint32_t)(base_size / sizeof(uint32_t));
        memcpy(repacked + base_size, bias, (size_t)bias_len * sizeof(float));
    }

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) {
        free(repacked);
        return NULL;
    }
    buf->buf = metal_new_weight_buffer(ctx, repacked, repacked_size);
    free(repacked);
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = repacked_size;
    buf->offset = 0;
    buf->type = bn_gpu_policy_metal_repacked_buffer_type(type);
    buf->rows = rows;
    buf->cols = cols;
    buf->bias_offset = bias_offset;
    return buf;
}

static void *metal_buffer_create(void *vctx, const void *data, size_t size,
                                  int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0) return NULL;

    if (bn_gpu_policy_metal_repacked_buffer_supported(type))
        return metal_repack_native_quant_for_gpu(ctx, data, size, type, rows, cols,
                                         NULL, 0, 1);

    /* Preserve immutable mmap-backed model storage when the native tensor
     * layout is already consumable by Metal. */
    void *borrowed = metal_buffer_create_borrowed(
        vctx, data, size, type, rows, cols);
    if (borrowed) return borrowed;

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) return NULL;

    /* Try slab allocation first */
    if (ctx->slab_buf) {
        size_t aligned = (size + 255) & ~(size_t)255;
        size_t offset = slab_alloc(ctx, aligned);
        if (offset != (size_t)-1) {
            memcpy((uint8_t *)[ctx->slab_buf contents] + offset, data, size);
            buf->buf = ctx->slab_buf;
            buf->size = size;
            buf->offset = offset;
            buf->type = type;
            buf->rows = rows;
            buf->cols = cols;
            buf->is_slab = 1;
            return buf;
        }
    }

    buf->buf = metal_new_weight_buffer(ctx, data, size);
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = size;
    buf->offset = 0;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    buf->is_slab = 0;
    return buf;
}

static void *metal_buffer_create_quant_only(void *vctx, const void *data,
                                             size_t size, int type,
                                             int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0) return NULL;

    if (bn_gpu_policy_metal_repacked_buffer_supported(type))
        return metal_repack_native_quant_for_gpu(
            ctx, data, size, type, rows, cols, NULL, 0, 0);

    void *borrowed = metal_buffer_create_borrowed(
        vctx, data, size, type, rows, cols);
    if (borrowed) return borrowed;

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) return NULL;
    buf->buf = metal_new_weight_buffer(ctx, data, size);
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = size;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    return buf;
}

static void *metal_buffer_create_borrowed(void *vctx, const void *data,
                                          size_t size, int type,
                                          int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0 || !ctx->mmap_base || ctx->mmap_size == 0)
        return NULL;
    const uint8_t *base = (const uint8_t *)ctx->mmap_base;
    const uint8_t *ptr = (const uint8_t *)data;
    if (ptr < base || size > ctx->mmap_size ||
        (size_t)(ptr - base) > ctx->mmap_size - size)
        return NULL;

    size_t page = (size_t)getpagesize();
    uintptr_t aligned_start = (uintptr_t)ptr & ~(page - 1);
    size_t prefix = (uintptr_t)ptr - aligned_start;
    if (prefix > SIZE_MAX - size || prefix + size > SIZE_MAX - (page - 1))
        return NULL;
    size_t aligned_size = (prefix + size + page - 1) & ~(page - 1);
    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) return NULL;
    if (ctx->mmap_buf_count > 0) {
        size_t logical_offset = (size_t)(ptr - base);
        if (ctx->mmap_buf_offset <= SIZE_MAX - logical_offset) {
            size_t absolute_offset = ctx->mmap_buf_offset + logical_offset;
            for (int i = 0; i < ctx->mmap_buf_count; i++) {
                size_t start = ctx->mmap_buf_starts[i];
                size_t view_size = ctx->mmap_buf_sizes[i];
                if (absolute_offset < start ||
                    absolute_offset - start > view_size ||
                    size > view_size - (absolute_offset - start))
                    continue;
                buf->buf = ctx->mmap_bufs[i];
                buf->size = size;
                buf->offset = absolute_offset - start;
                buf->type = type;
                buf->rows = rows;
                buf->cols = cols;
                buf->is_borrowed = 1;
                return buf;
            }
        }
    }
    buf->buf = [ctx->device newBufferWithBytesNoCopy:(void *)aligned_start
                                              length:aligned_size
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = size;
    buf->offset = prefix;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    buf->is_borrowed = 1;
    return buf;
}

static void *metal_buffer_create_biased(void *vctx, const void *data, size_t size,
                                         int type, int rows, int cols,
                                         const void *bias, size_t bias_size)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0 || !bias || bias_size == 0) return NULL;

    if (bn_gpu_policy_metal_repacked_buffer_supported(type)) {
        int bias_len = (int)(bias_size / sizeof(float));
        return metal_repack_native_quant_for_gpu(ctx, data, size, type, rows, cols,
                                         (const float *)bias, bias_len, 1);
    }

    /* Other types: combine weight data + bias into one buffer */
    size_t total = size + bias_size;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data, size);
    memcpy(combined + size, bias, bias_size);

    BnMetalBuf *buf = (BnMetalBuf *)metal_buffer_create(vctx, combined, total,
                                                          type, rows, cols);
    free(combined);
    if (!buf) return NULL;

    buf->bias_offset = (uint32_t)(size / sizeof(uint32_t));
    return buf;
}

static void *metal_buffer_create_stacked2(void *vctx,
                                          const void *data0, size_t size0,
                                          const void *data1, size_t size1,
                                          int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data0 || !data1 || size0 == 0 || size1 == 0) return NULL;

    size_t total = size0 + size1;
    if (bn_gpu_policy_metal_repacked_buffer_supported(type)) {
        uint8_t *combined = (uint8_t *)malloc(total);
        if (!combined) return NULL;
        memcpy(combined, data0, size0);
        memcpy(combined + size0, data1, size1);
        BnMetalBuf *buf = metal_repack_native_quant_for_gpu(
            ctx, combined, total, type, rows, cols, NULL, 0, 0);
        free(combined);
        return buf;
    }

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) return NULL;

    if (ctx->slab_buf) {
        size_t aligned = (total + 255) & ~(size_t)255;
        size_t offset = slab_alloc(ctx, aligned);
        if (offset != (size_t)-1) {
            uint8_t *dst = (uint8_t *)[ctx->slab_buf contents] + offset;
            memcpy(dst, data0, size0);
            memcpy(dst + size0, data1, size1);
            buf->buf = ctx->slab_buf;
            buf->size = total;
            buf->offset = offset;
            buf->type = type;
            buf->rows = rows;
            buf->cols = cols;
            buf->is_slab = 1;
            return buf;
        }
    }

    buf->buf = [ctx->device newBufferWithLength:total
                                        options:MTLResourceStorageModeShared];
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    uint8_t *dst = (uint8_t *)[buf->buf contents];
    memcpy(dst, data0, size0);
    memcpy(dst + size0, data1, size1);
    buf->size = total;
    buf->offset = 0;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    return buf;
}

static void *metal_buffer_create_stacked3(void *vctx,
                                          const void *data0, size_t size0,
                                          const void *data1, size_t size1,
                                          const void *data2, size_t size2,
                                          int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data0 || !data1 || !data2 ||
        size0 == 0 || size1 == 0 || size2 == 0)
        return NULL;
    if (bn_gpu_policy_metal_prepared_stacked_upload_blocked(
            ctx->runtime_policy, type))
        return NULL;

    size_t total = size0 + size1 + size2;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    memcpy(combined + size0 + size1, data2, size2);

    BnMetalBuf *buf = NULL;
    if (bn_gpu_policy_metal_repacked_buffer_supported(type)) {
        buf = metal_repack_native_quant_for_gpu(
            ctx, combined, total, type, rows, cols, NULL, 0, 0);
    } else {
        buf = (BnMetalBuf *)metal_buffer_create(
            vctx, combined, total, type, rows, cols);
    }
    free(combined);
    return buf;
}

static void *metal_buffer_create_stacked3_biased(void *vctx,
                                                 const void *data0,
                                                 size_t size0,
                                                 const void *data1,
                                                 size_t size1,
                                                 const void *data2,
                                                 size_t size2,
                                                 int type,
                                                 int rows,
                                                 int cols,
                                                 const void *bias,
                                                 size_t bias_size)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data0 || !data1 || !data2 || !bias ||
        size0 == 0 || size1 == 0 || size2 == 0 || bias_size == 0)
        return NULL;
    if (bn_gpu_policy_metal_prepared_stacked_upload_blocked(
            ctx->runtime_policy, type))
        return NULL;

    size_t total = size0 + size1 + size2;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    memcpy(combined + size0 + size1, data2, size2);

    BnMetalBuf *buf = NULL;
    if (bn_gpu_policy_metal_repacked_buffer_supported(type)) {
        int bias_len = (int)(bias_size / sizeof(float));
        buf = metal_repack_native_quant_for_gpu(
            ctx, combined, total, type, rows, cols, (const float *)bias,
            bias_len, 0);
    } else {
        size_t combined_biased_size = total + bias_size;
        uint8_t *combined_biased = (uint8_t *)malloc(combined_biased_size);
        if (combined_biased) {
            memcpy(combined_biased, combined, total);
            memcpy(combined_biased + total, bias, bias_size);
            buf = (BnMetalBuf *)metal_buffer_create(
                vctx, combined_biased, combined_biased_size, type, rows, cols);
            if (buf)
                buf->bias_offset = (uint32_t)(total / sizeof(uint32_t));
            free(combined_biased);
        }
    }
    free(combined);
    return buf;
}

static void metal_buffer_destroy(void *vctx, void *buffer)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *buf = (BnMetalBuf *)buffer;
    if (!buf) return;

    if (buf->is_slab && ctx) {
        slab_free_range(ctx, buf->offset, (buf->size + 255) & ~(size_t)255);
    }
    buf->buf = nil;
    free(buf);
}

/* ── Vtable: init_activations ──────────────────────────────────────── */

static void metal_free_activations(void *vctx);  /* forward decl */

static void metal_report_route_history(BnMetalCtx *ctx)
{
    if (!ctx || !ctx->route_history_buf)
        return;
    fprintf(stderr, "[gpu:metal:route-history] records=%zu capacity=%zu\n",
            ctx->route_history_count, ctx->route_history_capacity);
    if (ctx->route_history_count == 0 || ctx->route_history_stride <= 1 ||
        ctx->route_history_layers <= 0)
        return;
    const uint32_t *history =
        (const uint32_t *)[ctx->route_history_buf contents];
    uint64_t position_hash = UINT64_C(1469598103934665603);
    size_t position = 0;
    int position_records = 0;
    for (size_t record = 0; record < ctx->route_history_count; record++) {
        const uint32_t *entry =
            history + record * (size_t)ctx->route_history_stride;
        for (int field = 0; field < ctx->route_history_stride; field++) {
            position_hash ^= entry[field];
            position_hash *= UINT64_C(1099511628211);
        }
        position_records++;
        if (position_records == ctx->route_history_layers ||
            record + 1 == ctx->route_history_count) {
            fprintf(stderr,
                    "[gpu:metal:route-history] pos=%zu records=%d hash=%016llx\n",
                    position, position_records,
                    (unsigned long long)position_hash);
            position++;
            position_records = 0;
            position_hash = UINT64_C(1469598103934665603);
        }
    }
    static const int cache_sizes[] = { 1, 2, 4, 8, 16, 32 };
    for (size_t ci = 0; ci < sizeof(cache_sizes) / sizeof(cache_sizes[0]); ci++) {
        int slots = cache_sizes[ci];
        int tags[BN_METAL_MAX_MOE_RESIDENT_LAYERS][32];
        uint64_t ages[BN_METAL_MAX_MOE_RESIDENT_LAYERS][32];
        memset(tags, 0xff, sizeof(tags));
        memset(ages, 0, sizeof(ages));
        uint64_t clock = 0, hits = 0, accesses = 0;
        for (size_t record = 0; record < ctx->route_history_count; record++) {
            const uint32_t *entry =
                history + record * (size_t)ctx->route_history_stride;
            uint32_t layer = entry[0];
            if (layer >= (uint32_t)ctx->route_history_layers ||
                layer >= BN_METAL_MAX_MOE_RESIDENT_LAYERS)
                continue;
            for (int selected = 1; selected < ctx->route_history_stride;
                 selected++) {
                int expert = (int)entry[selected];
                int found = -1, victim = 0;
                for (int slot = 0; slot < slots; slot++) {
                    if (tags[layer][slot] == expert) {
                        found = slot;
                        break;
                    }
                    if (tags[layer][victim] >= 0 &&
                        (tags[layer][slot] < 0 ||
                         ages[layer][slot] < ages[layer][victim]))
                        victim = slot;
                }
                accesses++;
                if (found >= 0) {
                    hits++;
                    ages[layer][found] = ++clock;
                } else {
                    tags[layer][victim] = expert;
                    ages[layer][victim] = ++clock;
                }
            }
        }
        fprintf(stderr,
                "[gpu:metal:route-cache] slots/layer=%d hits=%llu/%llu "
                "rate=%.3f\n",
                slots, (unsigned long long)hits,
                (unsigned long long)accesses,
                accesses ? (double)hits / (double)accesses : 0.0);
    }
}

static int metal_init_activations(void *vctx,
                                  const BnGPUActivationPlan *plan)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !plan) return -1;

    /* Compute buffer sizes (same logic as wgpu) */
    int n_attn = plan->attention_layer_count;
    if (ctx->gpu_profile < 0)
        ctx->gpu_profile = bn_gpu_policy_profile_level(ctx->runtime_policy);
    ctx->moe_resident_layer_count =
        plan->moe_expert_hidden_dim > 0 ? plan->n_layers : 0;
    int q_dim = plan->n_heads * plan->head_size;
    int xb_size = q_dim > plan->dim ? q_dim : plan->dim;

    size_t sizes[BN_GPU_BUF_COUNT] = {0};
    sizes[BN_GPU_BUF_X]           = (size_t)plan->dim * sizeof(float);
    sizes[BN_GPU_BUF_XB]          = (size_t)xb_size * sizeof(float);
    sizes[BN_GPU_BUF_XB2]         =
        (size_t)plan->xb2_elements * sizeof(float);
    sizes[BN_GPU_BUF_Q]           = (size_t)q_dim * sizeof(float);
    sizes[BN_GPU_BUF_HB] =
        (size_t)plan->hb_elements * sizeof(float);
    sizes[BN_GPU_BUF_HB2] =
        (size_t)plan->hb_elements * sizeof(float);
    sizes[BN_GPU_BUF_KEY_CACHE]   = (size_t)n_attn * plan->seq_len * plan->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_VALUE_CACHE] = (size_t)n_attn * plan->seq_len * plan->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_ATT]         = (size_t)plan->n_heads * plan->seq_len * sizeof(float);
    sizes[BN_GPU_BUF_LOGITS]      = (size_t)plan->vocab_size * sizeof(float);
    sizes[BN_GPU_BUF_ROPE_FREQ]   = (size_t)plan->n_layers *
                                    (size_t)(plan->head_size / 2) * sizeof(float);
    sizes[BN_GPU_BUF_SCRATCH]     = (size_t)xb_size * sizeof(float);
    if (plan->per_layer_input_dim > 0)
        sizes[BN_GPU_BUF_PER_LAYER_INPUT] =
            (size_t)plan->n_layers * (size_t)plan->per_layer_input_dim *
            sizeof(float);
    {
        size_t qkv_size = (size_t)(q_dim + 2 * plan->kv_dim) * sizeof(float);
        size_t gated_q_size = (size_t)(2 * q_dim) * sizeof(float);
        sizes[BN_GPU_BUF_QKV] = qkv_size > gated_q_size ? qkv_size : gated_q_size;
    }

    if (plan->moe_expert_hidden_dim > 0) {
        sizes[BN_GPU_BUF_MOE_HB] =
            (size_t)plan->moe_active_experts *
            (size_t)plan->moe_expert_hidden_dim * sizeof(float);
        sizes[BN_GPU_BUF_MOE_HB2] =
            (size_t)plan->moe_active_experts *
            (size_t)plan->moe_expert_hidden_dim * sizeof(float);
        sizes[BN_GPU_BUF_MOE_OUT] = (size_t)plan->dim * sizeof(float);
        if (ctx->route_history_enabled &&
            plan->moe_active_experts > 0 && plan->n_layers > 0 &&
            plan->seq_len > 0) {
            ctx->route_history_stride = plan->moe_active_experts + 1;
            ctx->route_history_layers = plan->n_layers;
            ctx->route_history_capacity =
                (size_t)plan->seq_len * (size_t)plan->n_layers;
            size_t history_bytes = ctx->route_history_capacity *
                (size_t)ctx->route_history_stride * sizeof(uint32_t);
            ctx->route_history_buf =
                [ctx->device newBufferWithLength:history_bytes
                                         options:MTLResourceStorageModeShared];
        }
    }

    if (plan->uses_hybrid_ssm) {
        int n_ssm = plan->ssm_layer_count;
        int num_v_heads = plan->ssm_time_step_rank;
        int head_k_dim  = plan->ssm_state_size;
        int head_v_dim  = plan->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
        int key_dim     = plan->ssm_group_count * head_k_dim;
        int value_dim   = plan->ssm_inner_size;
        int qkv_dim     = key_dim * 2 + value_dim;
        int kern        = plan->ssm_conv_kernel > 0 ? plan->ssm_conv_kernel : 4;

        sizes[BN_GPU_BUF_SSM_STATE]      = (size_t)n_ssm * num_v_heads * head_k_dim * head_v_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_CONV_STATE] = (size_t)n_ssm * (kern - 1) * qkv_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_QKV]        = (size_t)qkv_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_Z]          = (size_t)value_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_ALPHA]      = (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_BUF_SSM_BETA]       = (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_BUF_SSM_V]          = (size_t)value_dim * sizeof(float);
    }

    /* Create activation buffers (storageModeShared — unified memory) */
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        if (sizes[i] == 0) continue;
        size_t aligned = (sizes[i] + 15) & ~(size_t)15;
        ctx->act_bufs[i] = [ctx->device newBufferWithLength:aligned
                                                    options:MTLResourceStorageModeShared];
        if (!ctx->act_bufs[i]) {
            metal_free_activations(ctx);
            return -1;
        }
        ctx->act_sizes[i] = aligned;
    }

    /* Upload precomputed RoPE frequencies */
    {
        if (plan->rope_frequency_count > 0 && !plan->rope_frequencies)
            return -1;
        memcpy([ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] contents],
               plan->rope_frequencies,
               (size_t)plan->rope_frequency_count * sizeof(float));
    }

    /* Compile forward-pass shaders */
    static const struct { int id; const char *file; const char *fn; } fwd_shaders[] = {
        { BN_GPU_SHADER_RMSNORM,          "rmsnorm.metal",          "rmsnorm"          },
        { BN_GPU_SHADER_ROPE,             "rope.metal",             "rope"             },
        { BN_GPU_SHADER_GQA_SCORES,       "gqa_scores.metal",       "gqa_scores"       },
        { BN_GPU_SHADER_SOFTMAX,          "softmax.metal",          "softmax"          },
        { BN_GPU_SHADER_GQA_COMBINE,      "gqa_combine.metal",      "gqa_combine"      },
        { BN_GPU_SHADER_SILU_GATE,        "silu_gate.metal",        "silu_gate"        },
        { BN_GPU_SHADER_RELU2_GATE,       "relu2_gate.metal",       "relu2_gate"       },
        { BN_GPU_SHADER_GELU_GATE,        "gelu_gate.metal",        "gelu_gate"        },
        { BN_GPU_SHADER_RESIDUAL_ADD,     "residual_add.metal",     "residual_add"     },
        { BN_GPU_SHADER_BIAS_ADD,         "bias_add.metal",         "bias_add"         },
        { BN_GPU_SHADER_RESIDUAL_RMSNORM, "residual_rmsnorm.metal", "residual_rmsnorm" },
        { BN_GPU_SHADER_WEIGHTED_ADD,     "weighted_add.metal",     "weighted_add"     },
        { BN_GPU_SHADER_WEIGHTED_ADD_SIGMOID,
          "weighted_add_sigmoid.metal", "weighted_add_sigmoid" },
        { BN_GPU_SHADER_MOE_ROUTE_TOPK,
          "moe_route_topk.metal", "moe_route_topk" },
        { BN_GPU_SHADER_MOE_ROUTED_FFN,
          "moe_q4k_q6k_routed.metal", "moe_q4k_gateup_routed" },
        { BN_GPU_SHADER_SSM_CONV_SILU,    "ssm_conv_silu.metal",    "ssm_conv_silu"    },
        { BN_GPU_SHADER_SSM_L2NORM,       "ssm_l2norm.metal",       "ssm_l2norm"       },
        { BN_GPU_SHADER_SSM_ALPHA_BETA,   "ssm_alpha_beta.metal",   "ssm_alpha_beta"   },
        { BN_GPU_SHADER_SSM_DELTA,        "ssm_delta.metal",        "ssm_delta"        },
        { BN_GPU_SHADER_SSM_GATE,         "ssm_gate.metal",         "ssm_gate"         },
        { BN_GPU_SHADER_PER_HEAD_RMSNORM, "per_head_rmsnorm.metal", "per_head_rmsnorm" },
        { BN_GPU_SHADER_DEINTERLEAVE_Q,   "deinterleave_q.metal",   "deinterleave_q"   },
        { BN_GPU_SHADER_SIGMOID_GATE,     "sigmoid_gate.metal",     "sigmoid_gate"     },
        { BN_GPU_SHADER_FLASH_ATTN,       "flash_attn.metal",       "flash_attn"       },
        { BN_GPU_SHADER_COPY,             "buf_copy.metal",         "buf_copy"         },
        { BN_GPU_SHADER_MATVEC_SPLIT,     "q4_matvec_split.metal",  "q4_matvec_split"  },
        { BN_GPU_SHADER_ROPE_QK,          "rope_qk.metal",          "rope_qk"          },
        { BN_GPU_SHADER_FUSED_GATEUP_SILU,"q4_fused_gateup_silu.metal","q4_fused_gateup_silu"},
        { BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT, "ssm_alpha_beta_split.metal", "ssm_alpha_beta_split" },
        { BN_GPU_SHADER_Q4K_MATVEC_SPLIT, "q4k_matvec_split.metal", "q4k_matvec_split" },
    };
    int n_fwd = (int)(sizeof(fwd_shaders) / sizeof(fwd_shaders[0]));
    int compiled = 0;
    for (int i = 0; i < n_fwd; i++) {
        int precise_math =
            fwd_shaders[i].id == BN_GPU_SHADER_SSM_CONV_SILU ||
            fwd_shaders[i].id == BN_GPU_SHADER_SSM_L2NORM ||
            fwd_shaders[i].id == BN_GPU_SHADER_SSM_ALPHA_BETA ||
            fwd_shaders[i].id == BN_GPU_SHADER_SSM_GATE ||
            fwd_shaders[i].id == BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT ||
            fwd_shaders[i].id == BN_GPU_SHADER_PER_HEAD_RMSNORM ||
            fwd_shaders[i].id == BN_GPU_SHADER_GELU_GATE;
        id<MTLComputePipelineState> pso = compile_shader_with_math(
            ctx, ctx->shader_dir, fwd_shaders[i].file, fwd_shaders[i].fn,
            precise_math);
        if (pso) {
            ctx->fwd_pipelines[fwd_shaders[i].id] = pso;
            compiled++;
        }
    }
    ctx->reference_gqa_scores_pipeline = compile_shader_with_math(
        ctx, ctx->shader_dir, "gqa_reference.metal",
        "gqa_scores_reference", 1);
    ctx->reference_softmax_pipeline = compile_shader_with_math(
        ctx, ctx->shader_dir, "gqa_reference.metal",
        "softmax_reference", 1);
    ctx->reference_gqa_combine_pipeline = compile_shader_with_math(
        ctx, ctx->shader_dir, "gqa_reference.metal",
        "gqa_combine_reference", 1);
    ctx->borrowed_native_q4_matvec_pipeline = compile_shader_with_math(
        ctx, ctx->shader_dir, "q4_native_matvec.metal",
        "q4_native_matvec", 1);
    if (!ctx->reference_gqa_scores_pipeline ||
        !ctx->reference_softmax_pipeline ||
        !ctx->reference_gqa_combine_pipeline ||
        !ctx->borrowed_native_q4_matvec_pipeline)
        return -1;
    ctx->moe_routed_q4k_down_pipeline = compile_shader(
        ctx, ctx->shader_dir, "moe_q4k_q6k_routed.metal",
        "moe_q4k_down_routed");
    ctx->moe_routed_q4_0_gateup_pipeline = compile_shader(
        ctx, ctx->shader_dir, "moe_q4k_q6k_routed.metal",
        "moe_q4_0_gateup_routed");
    ctx->moe_routed_q4_0_down_pipeline = compile_shader(
        ctx, ctx->shader_dir, "moe_q4k_q6k_routed.metal",
        "moe_q4_0_down_routed");
    ctx->moe_routed_q5k_down_pipeline = compile_shader(
        ctx, ctx->shader_dir, "moe_q4k_q6k_routed.metal",
        "moe_q5k_down_routed");
    ctx->moe_routed_q6k_down_pipeline = compile_shader(
        ctx, ctx->shader_dir, "moe_q4k_q6k_routed.metal",
        "moe_q6k_down_routed");
    ctx->moe_routed_q8k_quant_pipeline = compile_shader(
        ctx, ctx->shader_dir, "q8k_quantize.metal", "q8k_quantize");
    ctx->moe_route_logits_pipeline = compile_shader(
        ctx, ctx->shader_dir, "moe_route_topk.metal", "moe_route_logits");
    if (ctx->route_history_buf)
        ctx->moe_route_capture_pipeline = compile_shader(
            ctx, ctx->shader_dir, "moe_route_topk.metal", "moe_route_capture");
    if (!ctx->moe_routed_q4_0_gateup_pipeline ||
        !ctx->moe_routed_q4_0_down_pipeline ||
        !ctx->moe_routed_q4k_down_pipeline ||
        !ctx->moe_routed_q5k_down_pipeline ||
        !ctx->moe_routed_q6k_down_pipeline ||
        !ctx->moe_routed_q8k_quant_pipeline ||
        !ctx->moe_route_logits_pipeline ||
        (ctx->route_history_buf && !ctx->moe_route_capture_pipeline))
        return -1;
    fprintf(stderr, "[bn:gpu:metal] compiled %d/%d forward-pass shaders\n",
            compiled, n_fwd);
    ctx->cpu_order_rmsnorm_pipeline = compile_shader_with_math(
        ctx, ctx->shader_dir, "rmsnorm_cpu_order.metal",
        "rmsnorm_cpu_order", 1);

    /* Backend-resident KV and recurrent state are session state.  Give a
     * freshly initialized backend the same state as a freshly created CPU
     * session instead of relying on allocator-provided buffer contents. */
    return metal_reset_activations(ctx);
}

static void metal_configure_prepared_native_quant(void *vctx, int enabled)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx) return;
    ctx->prepared_native_quant_enabled = enabled != 0;
    ctx->native_quant_barriers_enabled =
        ctx->prepared_native_quant_enabled ||
        bn_gpu_policy_metal_native_quant_barriers_enabled(
            ctx->runtime_policy);
}

static void metal_free_activations(void *vctx)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx) return;
    (void)metal_flush_pending_logits(ctx);
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        ctx->act_bufs[i] = nil;
        ctx->act_sizes[i] = 0;
    }
    for (int i = 0; i < BN_GPU_SHADER_COUNT; i++)
        ctx->fwd_pipelines[i] = nil;
}

static int metal_reset_activations(void *vctx)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx) return -1;
    if (metal_flush_pending_logits(ctx) != 0) return -1;
    static const int mutable_buffers[] = {
        BN_GPU_BUF_KEY_CACHE,
        BN_GPU_BUF_VALUE_CACHE,
        BN_GPU_BUF_SSM_STATE,
        BN_GPU_BUF_SSM_CONV_STATE,
    };
    for (size_t i = 0;
         i < sizeof(mutable_buffers) / sizeof(mutable_buffers[0]); i++) {
        int index = mutable_buffers[i];
        if (ctx->act_bufs[index] && ctx->act_sizes[index] > 0)
            memset([ctx->act_bufs[index] contents], 0, ctx->act_sizes[index]);
    }
    if (ctx->route_history_buf)
        memset([ctx->route_history_buf contents], 0,
               [ctx->route_history_buf length]);
    return 0;
}

/* ── Vtable: write/read activation ─────────────────────────────────── */

static int metal_write_activation(void *vctx, int buf_idx, const void *data,
                                   size_t size, size_t offset)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (metal_flush_pending_logits(ctx) != 0) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    /* Unified memory: direct memcpy */
    memcpy((uint8_t *)[ctx->act_bufs[buf_idx] contents] + offset, data, size);
    return 0;
}

static int metal_read_buffer(BnMetalCtx *ctx, id<MTLBuffer> buffer,
                             size_t buffer_size, void *out,
                             size_t size, size_t offset)
{
    if (!ctx || !buffer || !out || offset + size > buffer_size) return -1;
    if ([buffer storageMode] != MTLStorageModePrivate) {
        memcpy(out, (uint8_t *)[buffer contents] + offset, size);
        return 0;
    }

    id<MTLBuffer> staging = [ctx->device newBufferWithLength:size
                                                    options:MTLResourceStorageModeShared];
    if (!staging) return -1;
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:buffer sourceOffset:offset
                toBuffer:staging destinationOffset:0 size:size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) return -1;
    memcpy(out, [staging contents], size);
    return 0;
}

static int metal_read_activation(void *vctx, int buf_idx, void *out,
                                  size_t size, size_t offset)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !out) return -1;
    if (metal_flush_pending_logits(ctx) != 0) return -1;
    if (buf_idx == BN_GPU_DEBUG_BUF_NATIVE_QUANT_ACT) {
        return metal_read_buffer(ctx, ctx->native_quant_buf,
                                 ctx->native_quant_buf_size,
                                 out, size, offset);
    }
    if (buf_idx == BN_GPU_DEBUG_BUF_NATIVE_QUANT_SCALE) {
        return metal_read_buffer(ctx, ctx->native_quant_scales_buf,
                                 ctx->native_quant_scales_buf_size,
                                 out, size, offset);
    }
    if (buf_idx == BN_GPU_DEBUG_BUF_NATIVE_QUANT_BLOCK_SUM) {
        return metal_read_buffer(ctx, ctx->native_quant_block_sums_buf,
                                 ctx->native_quant_block_sums_buf_size,
                                 out, size, offset);
    }
    if (buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    memcpy(out, (uint8_t *)[ctx->act_bufs[buf_idx] contents] + offset, size);
    return 0;
}

static int metal_argmax_activation(void *vctx, int buf_idx, int n,
                                   const int *penalty_tokens,
                                   int n_penalty_tokens,
                                   float repeat_penalty,
                                   int *out_token)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !out_token || !ctx->argmax_pipeline ||
        !ctx->argmax_reduce_pipeline || n <= 0 ||
        buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT ||
        !ctx->act_bufs[buf_idx] || (size_t)n * sizeof(float) > ctx->act_sizes[buf_idx])
        return -1;
    if (repeat_penalty == 1.0f)
        n_penalty_tokens = 0;
    if (n_penalty_tokens < 0 || (n_penalty_tokens > 0 && !penalty_tokens))
        return -1;

    if (!ctx->argmax_result_buf) {
        ctx->argmax_result_buf = [ctx->device newBufferWithLength:sizeof(int)
            options:MTLResourceStorageModeShared];
        if (!ctx->argmax_result_buf) return -1;
    }
    uint32_t n_groups = ((uint32_t)n + 1023u) / 1024u;
    if (n_groups < 1u) n_groups = 1u;
    if (n_groups > 256u) n_groups = 256u;
    size_t partial_bytes = (size_t)n_groups * 8u;
    if (!ctx->argmax_partials_buf ||
        ctx->argmax_partials_buf_size < partial_bytes) {
        ctx->argmax_partials_buf =
            [ctx->device newBufferWithLength:partial_bytes
                                     options:MTLResourceStorageModePrivate];
        if (!ctx->argmax_partials_buf) return -1;
        ctx->argmax_partials_buf_size = partial_bytes;
    }
    size_t penalty_bytes = n_penalty_tokens > 0
        ? (size_t)n_penalty_tokens * sizeof(int) : sizeof(int);
    if (!ctx->argmax_penalty_buf || ctx->argmax_penalty_buf_size < penalty_bytes) {
        ctx->argmax_penalty_buf = [ctx->device newBufferWithLength:penalty_bytes
            options:MTLResourceStorageModeShared];
        if (!ctx->argmax_penalty_buf) return -1;
        ctx->argmax_penalty_buf_size = penalty_bytes;
    }
    if (n_penalty_tokens > 0)
        memcpy([ctx->argmax_penalty_buf contents], penalty_tokens, penalty_bytes);

    struct {
        uint32_t n;
        uint32_t n_penalty_tokens;
        float repeat_penalty;
    } params = {(uint32_t)n, (uint32_t)n_penalty_tokens, repeat_penalty};

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) {
        (void)metal_flush_pending_logits(ctx);
        return -1;
    }
    [enc setComputePipelineState:ctx->argmax_pipeline];
    [enc setBuffer:ctx->act_bufs[buf_idx] offset:0 atIndex:0];
    [enc setBuffer:ctx->argmax_penalty_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->argmax_partials_buf offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc setComputePipelineState:ctx->argmax_reduce_pipeline];
    [enc setBuffer:ctx->argmax_partials_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->argmax_result_buf offset:0 atIndex:1];
    [enc setBytes:&n_groups length:sizeof(n_groups) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (ctx->gpu_profile >= 1 &&
        (ctx->argmax_calls < 5 || (ctx->argmax_calls % 50) == 0)) {
        double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
        fprintf(stderr,
                "[gpu:metal:argmax] call=%d groups=%u gpu=%.3fms\n",
                ctx->argmax_calls, n_groups, gpu_ms);
    }
    ctx->argmax_calls++;
    int graph_status = 0;
    if (ctx->pending_logits_cmd) {
        graph_status = [ctx->pending_logits_cmd status] ==
            MTLCommandBufferStatusCompleted ? 0 : -1;
        ctx->pending_logits_cmd = nil;
    }
    if (graph_status != 0 || [cmd status] != MTLCommandBufferStatusCompleted)
        return -1;
    *out_token = *(const int *)[ctx->argmax_result_buf contents];
    return 0;
}

/* ── Vtable: matvec (standalone, not forward-pass) ─────────────────── */

static int ensure_scratch(BnMetalCtx *ctx, size_t x_need, size_t out_need)
{
    if (!ctx->x_buf || ctx->x_buf_size < x_need) {
        ctx->x_buf = [ctx->device newBufferWithLength:x_need
                                              options:MTLResourceStorageModeShared];
        if (!ctx->x_buf) return -1;
        ctx->x_buf_size = x_need;
    }
    if (!ctx->out_buf || ctx->out_buf_size < out_need) {
        ctx->out_buf = [ctx->device newBufferWithLength:out_need
                                                options:MTLResourceStorageModeShared];
        if (!ctx->out_buf) return -1;
        ctx->out_buf_size = out_need;
    }
    return 0;
}

static int ensure_native_quant_scratch(BnMetalCtx *ctx, int cols, int n_tokens)
{
    size_t quantized_need = (size_t)cols * (size_t)n_tokens * sizeof(int8_t);
    size_t scales_need = (size_t)(cols >> 5) * (size_t)n_tokens * sizeof(float);
    if (!ctx->native_quant_buf || ctx->native_quant_buf_size < quantized_need) {
        ctx->native_quant_buf = [ctx->device newBufferWithLength:quantized_need
                                                options:MTLResourceStorageModePrivate];
        if (!ctx->native_quant_buf) return -1;
        ctx->native_quant_buf_size = quantized_need;
    }
    if (!ctx->native_quant_scales_buf || ctx->native_quant_scales_buf_size < scales_need) {
        ctx->native_quant_scales_buf = [ctx->device newBufferWithLength:scales_need
                                                      options:MTLResourceStorageModePrivate];
        if (!ctx->native_quant_scales_buf) return -1;
        ctx->native_quant_scales_buf_size = scales_need;
    }
    return 0;
}

static int ensure_specialized_native_quant_scratch(BnMetalCtx *ctx, int cols, int n_tokens)
{
    size_t quantized_need = (size_t)cols * (size_t)n_tokens * sizeof(int8_t);
    size_t n_blocks = (size_t)(cols >> 8) * (size_t)n_tokens;
    size_t scales_need = n_blocks * sizeof(float);
    size_t block_sums_need = n_blocks * 16 * sizeof(int16_t);
    if (!ctx->native_quant_buf || ctx->native_quant_buf_size < quantized_need) {
        ctx->native_quant_buf = [ctx->device newBufferWithLength:quantized_need
                                                options:MTLResourceStorageModePrivate];
        if (!ctx->native_quant_buf) return -1;
        ctx->native_quant_buf_size = quantized_need;
    }
    if (!ctx->native_quant_scales_buf || ctx->native_quant_scales_buf_size < scales_need) {
        ctx->native_quant_scales_buf = [ctx->device newBufferWithLength:scales_need
                                                      options:MTLResourceStorageModePrivate];
        if (!ctx->native_quant_scales_buf) return -1;
        ctx->native_quant_scales_buf_size = scales_need;
    }
    if (!ctx->native_quant_block_sums_buf ||
        ctx->native_quant_block_sums_buf_size < block_sums_need) {
        ctx->native_quant_block_sums_buf = [ctx->device newBufferWithLength:block_sums_need
                                                     options:MTLResourceStorageModePrivate];
        if (!ctx->native_quant_block_sums_buf) return -1;
        ctx->native_quant_block_sums_buf_size = block_sums_need;
    }
    return 0;
}

static void metal_encode_native_quant(id<MTLComputeCommandEncoder> enc,
                                      BnMetalCtx *ctx,
                                      id<MTLBuffer> x_buf,
                                      uint32_t cols,
                                      uint32_t n_tokens)
{
    ctx->native_quant_dispatches++;
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    id<MTLBuffer> bufs[2] = {
        ctx->native_quant_buf, ctx->native_quant_scales_buf
    };
    [enc memoryBarrierWithResources:bufs count:2];
    [enc setComputePipelineState:ctx->native_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];
    MTLSize tpg = MTLSizeMake(32, 1, 1);
    MTLSize grid = MTLSizeMake((cols + 31) / 32, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    if (ctx->native_quant_barriers_enabled) {
        [enc memoryBarrierWithResources:bufs count:2];
    }
}

static void metal_encode_specialized_native_quant(
    id<MTLComputeCommandEncoder> enc,
    BnMetalCtx *ctx,
    id<MTLBuffer> x_buf,
    uint32_t cols,
    uint32_t n_tokens)
{
    ctx->specialized_native_quant_dispatches++;
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    id<MTLBuffer> bufs[3] = {
        ctx->native_quant_buf,
        ctx->native_quant_scales_buf,
        ctx->native_quant_block_sums_buf
    };
    [enc memoryBarrierWithResources:bufs count:3];
    [enc setComputePipelineState:ctx->specialized_native_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(params) atIndex:4];
    MTLSize tpg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(cols / 256, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc memoryBarrierWithResources:bufs count:3];
}

static void metal_encode_routed_q8k_quant(
    id<MTLComputeCommandEncoder> enc,
    BnMetalCtx *ctx,
    id<MTLBuffer> x_buf,
    uint32_t cols,
    uint32_t n_tokens)
{
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    id<MTLBuffer> bufs[3] = {
        ctx->native_quant_buf,
        ctx->native_quant_scales_buf,
        ctx->native_quant_block_sums_buf
    };
    [enc memoryBarrierWithResources:bufs count:3];
    [enc setComputePipelineState:ctx->moe_routed_q8k_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(params) atIndex:4];
    MTLSize tpg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(cols / 256, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static id<MTLComputePipelineState>
metal_specialized_native_matvec_pipeline(BnMetalCtx *ctx, int tensor_type) {
    switch (tensor_type) {
        case BN_GGUF_TENSOR_Q4_K:
            return ctx->specialized_q4k_native_matvec_pipeline;
        case BN_GGUF_TENSOR_Q5_K:
            return ctx->specialized_q5k_native_matvec_pipeline;
        default:
            return ctx->specialized_native_matvec_pipeline;
    }
}

static id<MTLComputePipelineState>
metal_specialized_native_matvec_pipeline_for_shape(BnMetalCtx *ctx,
                                                   int tensor_type,
                                                   int rows,
                                                   int cols) {
    /* Four-row Q6 amortizes best for wide inputs or tall output matrices. */
    if (tensor_type == BN_GGUF_TENSOR_Q6_K &&
        (cols >= 4096 || rows >= 65536) &&
        ctx->specialized_native_four_row_matvec_pipeline)
        return ctx->specialized_native_four_row_matvec_pipeline;
    return metal_specialized_native_matvec_pipeline(ctx, tensor_type);
}

/* Workgroup geometry belongs to the backend-private shader contract. */
static uint32_t metal_float_matvec_tile_rows(int tensor_type)
{
    if (tensor_type == BN_GGUF_TENSOR_Q4_K) return 16u;
    return 32u;
}

static id<MTLComputePipelineState>
metal_reference_kquant_matvec_pipeline(BnMetalCtx *ctx, int tensor_type,
                                       int rows, int cols)
{
    (void)rows;
    (void)cols;
    if (tensor_type == BN_GGUF_TENSOR_Q4_K)
        return ctx->reference_q4k_matvec_pipeline;
    if (tensor_type == BN_GGUF_TENSOR_Q5_K)
        return ctx->reference_q5k_matvec_pipeline;
    if (bn_backend_quant_uses_down_kquant(tensor_type))
        return ctx->reference_q6k_matvec_pipeline;
    if (tensor_type >= 0 && tensor_type < BN_METAL_MAX_TYPES)
        return ctx->pipelines[tensor_type];
    return nil;
}

static int metal_small_dense_native_quant_graph_path_supported(BnMetalCtx *ctx,
                                            int tensor_type,
                                            int native_quant_prepared,
                                            int native_quant_prepared_path,
                                            id<MTLComputePipelineState> pipeline)
{
    if (bn_gpu_policy_metal_specialized_native_quant_enabled(
            ctx->runtime_policy) &&
        bn_quant_format_supports_specialized_native_quant_matvec(tensor_type) &&
        metal_specialized_native_matvec_pipeline(ctx, tensor_type))
        return 0;
    return bn_gpu_policy_metal_small_dense_native_quant_graph_path_supported(
        tensor_type, ctx->small_dense_native_quant_enabled, native_quant_prepared,
        native_quant_prepared_path,
        ctx->native_quant_pipeline != nil, pipeline != nil);
}

static int metal_block_q8_activation_graph_path_supported(BnMetalCtx *ctx,
                                                           int tensor_type,
                                                           int enabled)
{
    return bn_gpu_policy_metal_block_q8_activation_graph_path_supported(
        tensor_type, enabled,
        ctx->native_quant_pipeline != nil,
        ctx->q8_native_quant_matvec_pipeline != nil);
}

static int metal_matvec(void *vctx, float *out, void *W_buf, const float *x,
                         int rows, int cols, int type)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *wbuf = (BnMetalBuf *)W_buf;
    if (!ctx || !wbuf || !x || !out) return -1;
    if (type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type]) return -1;

    size_t x_size = (size_t)cols * sizeof(float);
    size_t out_size = (size_t)rows * sizeof(float);
    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;
    int use_prepared_small_dense_native_quant = bn_gpu_policy_metal_small_dense_native_quant_matvec_supported(
        type, ctx->small_dense_native_quant_enabled, wbuf->native_quant_prepared,
        ctx->native_quant_pipeline != nil, 0,
        ctx->prepared_small_dense_native_quant_matvec_pipeline != nil);
    int use_small_dense_native_quant = bn_gpu_policy_metal_small_dense_native_quant_matvec_supported(
        type, ctx->small_dense_native_quant_enabled, wbuf->native_quant_prepared,
        ctx->native_quant_pipeline != nil, ctx->small_dense_native_quant_matvec_pipeline != nil, 0);
    int use_specialized_native_quant =
        bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
            ctx->runtime_policy,
            type, cols, ctx->specialized_native_quant_pipeline != nil,
            metal_specialized_native_matvec_pipeline(ctx, type) != nil) ||
        (bn_gpu_policy_metal_specialized_native_quant_shape_default_enabled(
             ctx->runtime_policy,
             type, rows, cols) &&
         ctx->specialized_native_quant_pipeline != nil &&
         metal_specialized_native_matvec_pipeline(ctx, type) != nil);
    int use_prepared_f32 = bn_gpu_policy_metal_prepared_f32_enabled(
                               ctx->runtime_policy) &&
                           wbuf->native_quant_prepared &&
                           ctx->prepared_f32_matvec_pipeline != nil &&
                           !use_prepared_small_dense_native_quant;
    int use_q8_native_quant = 0;
    if (wbuf->native_quant_prepared &&
        !use_prepared_small_dense_native_quant && !use_prepared_f32)
        return -1;
    if ((use_q8_native_quant || use_prepared_small_dense_native_quant ||
         use_small_dense_native_quant) &&
        ensure_native_quant_scratch(ctx, cols, 1) != 0) return -1;
    if (use_specialized_native_quant &&
        ensure_specialized_native_quant_scratch(ctx, cols, 1) != 0)
        return -1;

    memcpy([ctx->x_buf contents], x, x_size);

    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols, 1, 0, 0, 0, 0, 0 };
    if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;

    uint32_t tile_rows =
        (use_small_dense_native_quant && !use_prepared_small_dense_native_quant)
            ? 16 : metal_float_matvec_tile_rows(type);
    uint32_t wg_x = ((uint32_t)rows + tile_rows - 1) / tile_rows;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        if (use_q8_native_quant) {
            metal_encode_native_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->q8_native_quant_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
        } else if (use_prepared_small_dense_native_quant) {
            metal_encode_native_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->prepared_small_dense_native_quant_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
        } else if (use_prepared_f32) {
            [enc setComputePipelineState:ctx->prepared_f32_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
        } else if (use_small_dense_native_quant) {
            metal_encode_native_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->small_dense_native_quant_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
        } else if (use_specialized_native_quant) {
            id<MTLComputePipelineState> specialized_pipeline =
                metal_specialized_native_matvec_pipeline_for_shape(
                    ctx, type, rows, cols);
            metal_encode_specialized_native_quant(enc, ctx, ctx->x_buf,
                                                  (uint32_t)cols, 1);
            [enc setComputePipelineState:specialized_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:4];
            [enc setBytes:params length:sizeof(params) atIndex:5];
        } else {
            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
        }

        uint32_t threads_per_tg =
            (use_small_dense_native_quant &&
             !use_prepared_small_dense_native_quant) ? 128u : 256u;
        MTLSize tpg = MTLSizeMake(threads_per_tg, 1, 1);
        MTLSize grid = MTLSizeMake(wg_x, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "[bn:gpu:metal] matvec command failed: %s\n",
                    [[[cmd error] localizedDescription] UTF8String]);
            return -1;
        }
    }

    memcpy(out, [ctx->out_buf contents], out_size);
    return 0;
}

static int metal_matmul(void *vctx, float *out, void *W_buf, const float *X,
                         int rows, int cols, int n_tokens, int type)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *wbuf = (BnMetalBuf *)W_buf;
    if (!ctx || !wbuf || !X || !out) return -1;
    if (type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type]) return -1;
    if (wbuf->native_quant_prepared) return -1;

    size_t x_size = (size_t)n_tokens * cols * sizeof(float);
    size_t out_size = (size_t)n_tokens * rows * sizeof(float);
    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;

    memcpy([ctx->x_buf contents], X, x_size);

    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols, (uint32_t)n_tokens, 0, 0, 0, 0, 0 };

    uint32_t tile_rows = metal_float_matvec_tile_rows(type);
    uint32_t wg_x = ((uint32_t)rows + tile_rows - 1) / tile_rows;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx->pipelines[type]];
        [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
        [enc setBytes:params length:sizeof(params) atIndex:3];

        MTLSize tpg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(wg_x, n_tokens, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, [ctx->out_buf contents], out_size);
    return 0;
}

static int metal_matvec_batch(void *vctx, const BnGPUMatvecOp *ops, int n_ops,
                               const float *x, int x_cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0 || !x) return -1;

    size_t x_size = (size_t)x_cols * sizeof(float);
    int max_rows = 0;
    for (int i = 0; i < n_ops; i++) {
        BnMetalBuf *wbuf = (BnMetalBuf *)ops[i].W_buf;
        if (wbuf && wbuf->native_quant_prepared)
            return -1;
        if (ops[i].rows > max_rows) max_rows = ops[i].rows;
    }
    size_t out_size = (size_t)max_rows * sizeof(float);

    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;
    memcpy([ctx->x_buf contents], x, x_size);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        for (int i = 0; i < n_ops; i++) {
            BnMetalBuf *wbuf = (BnMetalBuf *)ops[i].W_buf;
            int type = ops[i].type;
            if (!wbuf || type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type])
                continue;

            uint32_t params[8] = { (uint32_t)ops[i].rows, (uint32_t)ops[i].cols, 1, 0, 0, 0, 0, 0 };
            if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;

            uint32_t tile_rows = metal_float_matvec_tile_rows(type);
            uint32_t wg_x = ((uint32_t)ops[i].rows + tile_rows - 1) / tile_rows;

            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];

            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];

            /* Memory barrier between dispatches sharing out_buf */
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* Copy results to host pointers (need per-op dispatch for different out ptrs) */
    /* Re-dispatch individually since each op has a different host out pointer */
    /* TODO: optimize with a single submission + per-op output buffers */
    for (int i = 0; i < n_ops; i++) {
        BnMetalBuf *wbuf = (BnMetalBuf *)ops[i].W_buf;
        if (!wbuf) continue;
        int type = ops[i].type;
        if (type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type]) continue;

        uint32_t params[8] = { (uint32_t)ops[i].rows, (uint32_t)ops[i].cols, 1, 0, 0, 0, 0, 0 };
        if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
        uint32_t tile_rows = metal_float_matvec_tile_rows(type);
        uint32_t wg_x = ((uint32_t)ops[i].rows + tile_rows - 1) / tile_rows;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];

            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }

        memcpy(ops[i].out, [ctx->out_buf contents],
               (size_t)ops[i].rows * sizeof(float));
    }

    return 0;
}

/* ── Vtable: execute (forward-pass) ────────────────────────────────── */

static int metal_execute(void *vctx, const void *ops_raw, int n_ops,
                         int readback_buf, float *out_host, int out_len)
{
    const BnGPUOp *ops = (const BnGPUOp *)ops_raw;
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0) return -1;
    if (metal_flush_pending_logits(ctx) != 0) return -1;
    metal_prefault_routed_mmap(ctx, ops, n_ops);
    metal_request_mmap_residency(ctx);
    if (ctx->gpu_profile < 0)
        ctx->gpu_profile = bn_gpu_policy_profile_level(ctx->runtime_policy);
    double t0 = bn_platform_time_ms();
    double t_encode = 0, t_gpu = 0;
    double routed_wait_ms = 0.0, routed_copy_ms = 0.0;
    int n_barriers = 0;
    BnMetalProfileShape matvec_shapes[32], native_quant_shapes[16], small_dense_native_quant_weight_shapes[16];
    int n_matvec_shapes = 0, n_native_quant_shapes = 0, n_small_dense_native_quant_weight_shapes = 0;
    BnMetalProfileShape timed_shapes[64];
    int n_timed_shapes = 0;
    int profile_each_op = ctx->gpu_profile >= 4;
    double shader_gpu_ms[BN_GPU_SHADER_COUNT];
    double shader_wall_ms[BN_GPU_SHADER_COUNT];
    int shader_profile_counts[BN_GPU_SHADER_COUNT];
    double moe_gateup_gpu_ms = 0.0;
    double moe_gateup_wall_ms = 0.0;
    double moe_down_gpu_ms = 0.0;
    double moe_down_wall_ms = 0.0;
    int moe_profile_count = 0;
    memset(matvec_shapes, 0, sizeof(matvec_shapes));
    memset(native_quant_shapes, 0, sizeof(native_quant_shapes));
    memset(small_dense_native_quant_weight_shapes, 0, sizeof(small_dense_native_quant_weight_shapes));
    memset(timed_shapes, 0, sizeof(timed_shapes));
    memset(shader_gpu_ms, 0, sizeof(shader_gpu_ms));
    memset(shader_wall_ms, 0, sizeof(shader_wall_ms));
    memset(shader_profile_counts, 0, sizeof(shader_profile_counts));
    ctx->native_quant_dispatches = 0;
    ctx->specialized_native_quant_dispatches = 0;
    ctx->small_dense_native_quant_matvec_dispatches = 0;
    ctx->small_dense_native_quant_split_dispatches = 0;
    ctx->small_dense_native_quant_gateup_dispatches = 0;
    int full_barriers = ctx->prepared_native_quant_enabled ||
                        ctx->full_barriers_enabled;
    int disable_barriers = ctx->barriers_disabled;
    int produces_unread_logits = 0;
    if (!out_host && readback_buf < 0 && ctx->gpu_profile == 0) {
        for (int i = 0; i < n_ops; i++) {
            if (ops[i].buf_out == BN_GPU_VALUE_LOGITS) {
                produces_unread_logits = 1;
                break;
            }
        }
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;
#define METAL_EXEC_FAIL() do {                                                \
    if (ctx->gpu_profile >= 1) {                                             \
        BnMetalBuf *failed_wbuf = (BnMetalBuf *)op->W_buf;                   \
        fprintf(stderr,                                                      \
                "[gpu:metal:reject] op=%d code=%d shader=%s type=%d "       \
                "prepared=%d flags=%u native=%u enabled=%d pipeline=%d "   \
                "rows=%d cols=%d\n",                                     \
                i, op->op_code, metal_shader_profile_name(shader),           \
                op->type, failed_wbuf ? failed_wbuf->native_quant_prepared   \
                                      : 0,                                   \
                op->flags, op->p[6], ctx->small_dense_native_quant_enabled,  \
                ctx->prepared_small_dense_native_quant_matvec_pipeline != nil,\
                op->rows, op->cols);                                         \
    }                                                                        \
    if (enc) [enc endEncoding];                                              \
    return -1;                                                               \
} while (0)

        /* Dependency tracking: only insert barriers on actual RAW/WAR/WAW conflicts.
         * Same logic as wgpu execute — track read/write buffer masks since last barrier. */
        uint32_t since_barrier_writes = 0;
        id<MTLComputePipelineState> current_pso = nil;
        int route_pending = 0;

        for (int i = 0; i < n_ops; i++) {
            const BnGPUOp *op = &ops[i];
            int shader = bn_gpu_shader_from_op_code(op->op_code);

            /* COPY as compute shader — stays in compute encoder, no blit transitions */

            /* Determine pipeline */
            id<MTLComputePipelineState> pipeline = nil;
            if (shader == BN_GPU_SHADER_MATVEC) {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
            if (wbuf && wbuf->native_matvec_layout) {
                pipeline = ctx->borrowed_native_q4_matvec_pipeline;
            } else if ((op->flags & BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT) != 0) {
                pipeline = metal_reference_kquant_matvec_pipeline(
                    ctx, op->type, op->rows, op->cols);
            } else if (bn_gpu_policy_metal_prepared_f32_enabled(
                           ctx->runtime_policy) && wbuf &&
                    wbuf->native_quant_prepared &&
                    ctx->prepared_f32_matvec_pipeline) {
                    pipeline = ctx->prepared_f32_matvec_pipeline;
                } else if (metal_block_q8_activation_graph_path_supported(
                        ctx, op->type, op->p[6] != 0)) {
                    pipeline = ctx->q8_native_quant_matvec_pipeline;
                } else if (op->p[6] && wbuf &&
                    (op->flags &
                     BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0 &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 1,
                        ctx->prepared_reference_native_quant_matvec_pipeline)) {
                    pipeline = ctx->prepared_reference_native_quant_matvec_pipeline;
                } else if (op->p[6] && wbuf &&
                    (op->flags &
                     BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0 &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 0,
                        ctx->reference_native_quant_matvec_pipeline)) {
                    pipeline = ctx->reference_native_quant_matvec_pipeline;
                } else if (op->p[6] && wbuf &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 1,
                        ctx->prepared_small_dense_native_quant_matvec_pipeline)) {
                    pipeline = ctx->prepared_small_dense_native_quant_matvec_pipeline;
                } else if (op->p[6] && wbuf &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 0,
                        ctx->small_dense_native_quant_matvec_pipeline)) {
                    pipeline = ctx->small_dense_native_quant_matvec_pipeline;
                } else if (op->type >= 0 && op->type < BN_METAL_MAX_TYPES) {
                    pipeline = ctx->pipelines[op->type];
                }
            } else if (shader == BN_GPU_SHADER_RMSNORM &&
                       ctx->cpu_order_rmsnorm_enabled &&
                       ctx->cpu_order_rmsnorm_pipeline) {
                pipeline = ctx->cpu_order_rmsnorm_pipeline;
            } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                       op->p[6] && op->W_buf &&
                       metal_small_dense_native_quant_graph_path_supported(
                           ctx, op->type,
                           ((BnMetalBuf *)op->W_buf)->native_quant_prepared, 1,
                           ctx->prepared_small_dense_native_quant_gateup_pipeline)) {
                pipeline = ctx->prepared_small_dense_native_quant_gateup_pipeline;
            } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                       op->p[6] && op->W_buf &&
                       metal_small_dense_native_quant_graph_path_supported(
                           ctx, op->type,
                           ((BnMetalBuf *)op->W_buf)->native_quant_prepared, 0,
                           ctx->small_dense_native_quant_gateup_pipeline)) {
                pipeline = ctx->small_dense_native_quant_gateup_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       bn_quant_format_supports_native_quant_split(op->type) &&
                       op->cols > 0 && (op->cols % 256) == 0 &&
                       ctx->specialized_q4k_native_split_pipeline) {
                pipeline = ctx->specialized_q4k_native_split_pipeline;
            } else if (shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT &&
                       bn_quant_format_supports_native_quant_split(op->type) &&
                       bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
                           ctx->runtime_policy,
                           op->type, op->cols,
                           ctx->specialized_native_quant_pipeline != nil,
                           ctx->specialized_q4k_native_split_pipeline != nil)) {
                pipeline = ctx->specialized_q4k_native_split_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       bn_gpu_policy_metal_prepared_f32_enabled(
                           ctx->runtime_policy) &&
                       op->W_buf &&
                       ((BnMetalBuf *)op->W_buf)->native_quant_prepared &&
                       ctx->prepared_f32_split_pipeline) {
                pipeline = ctx->prepared_f32_split_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       (op->flags & 1u) && op->W_buf &&
                       metal_small_dense_native_quant_graph_path_supported(
                           ctx, op->type,
                           ((BnMetalBuf *)op->W_buf)->native_quant_prepared, 1,
                           ctx->prepared_small_dense_native_quant_split_pipeline)) {
                pipeline = ctx->prepared_small_dense_native_quant_split_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       (op->flags & 1u) && op->W_buf &&
                       metal_small_dense_native_quant_graph_path_supported(
                           ctx, op->type,
                           ((BnMetalBuf *)op->W_buf)->native_quant_prepared, 0,
                           ctx->small_dense_native_quant_split_pipeline)) {
                pipeline = ctx->small_dense_native_quant_split_pipeline;
            } else if ((op->flags &
                        BN_GPU_OP_FLAG_REFERENCE_ATTENTION_ORDER) &&
                       (ctx->reference_attention_stage_mask & 1u) &&
                       shader == BN_GPU_SHADER_GQA_SCORES) {
                pipeline = ctx->reference_gqa_scores_pipeline;
            } else if ((op->flags &
                        BN_GPU_OP_FLAG_REFERENCE_ATTENTION_ORDER) &&
                       (ctx->reference_attention_stage_mask & 2u) &&
                       shader == BN_GPU_SHADER_SOFTMAX) {
                pipeline = ctx->reference_softmax_pipeline;
            } else if ((op->flags &
                        BN_GPU_OP_FLAG_REFERENCE_ATTENTION_ORDER) &&
                       (ctx->reference_attention_stage_mask & 4u) &&
                       shader == BN_GPU_SHADER_GQA_COMBINE) {
                pipeline = ctx->reference_gqa_combine_pipeline;
            } else if (shader > 0 && shader < BN_GPU_SHADER_COUNT) {
                pipeline = ctx->fwd_pipelines[shader];
            }
            if (!pipeline)
                METAL_EXEC_FAIL();

            if (ctx->gpu_profile >= 2 &&
                (shader == BN_GPU_SHADER_MATVEC ||
                 shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                 shader == BN_GPU_SHADER_FUSED_GATEUP_SILU)) {
                uint32_t n_tokens = shader == BN_GPU_SHADER_MATVEC && op->p[2]
                    ? op->p[2]
                    : 1;
                uint32_t rows = shader == BN_GPU_SHADER_MATVEC_SPLIT
                    ? op->p[0]
                    : (uint32_t)op->rows;
                if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU && op->p[0])
                    rows = op->p[0];
                metal_profile_add_shape(matvec_shapes, &n_matvec_shapes, 32,
                                        shader, op->type, rows,
                                        (uint32_t)op->cols,
                                        n_tokens);
            }

            /* Compute this op's read/write buffer masks. */
            uint32_t op_reads = 0, op_writes = 0;
            if (bn_gpu_shader_access_masks(op, shader, &op_reads,
                                           &op_writes) != 0)
                continue;

            /* Insert barrier only on RAW conflict (read-after-write).
             * WAR and WAW don't need barriers — Metal dispatches execute in
             * submission order within a compute command encoder, so reads
             * always complete before subsequent writes to the same buffer. */
            int conflict = disable_barriers ? 0
                : (full_barriers ? (since_barrier_writes != 0)
                                 : (op_reads & since_barrier_writes));
            if (conflict && enc) {
                /* Use resource-specific barriers for less stalling */
                uint32_t barrier_mask = full_barriers
                    ? since_barrier_writes
                    : (op_reads & since_barrier_writes);
                uint32_t resource_barrier_mask = 0;
                id<MTLBuffer> barrier_bufs[BN_GPU_BUF_COUNT];
                int n_bbuf = 0;
                for (int b = 0; b < BN_GPU_BUF_COUNT; b++) {
                    if ((barrier_mask & (1u << b)) && ctx->act_bufs[b]) {
                        barrier_bufs[n_bbuf++] = ctx->act_bufs[b];
                        resource_barrier_mask |= 1u << b;
                    }
                }
                if (n_bbuf > 0) {
                    [enc memoryBarrierWithResources:barrier_bufs count:(NSUInteger)n_bbuf];
                    since_barrier_writes &= ~resource_barrier_mask;
                } else {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    since_barrier_writes = 0;
                }
                n_barriers++;
            }

            since_barrier_writes |= op_writes;

            /* Start compute encoder if needed */
            if (!cmd)
                cmd = [ctx->queue commandBuffer];
            if (!enc) {
                enc = [cmd computeCommandEncoder];
                current_pso = nil;
            }
            BnMetalBuf *pre_wbuf = (BnMetalBuf *)op->W_buf;
            int native_quant_deferred_pso =
                pre_wbuf &&
                !pre_wbuf->native_matvec_layout &&
                (op->flags & BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT) == 0 &&
                ((shader == BN_GPU_SHADER_MATVEC && op->p[6] &&
                  (metal_small_dense_native_quant_graph_path_supported(
                       ctx, op->type, pre_wbuf->native_quant_prepared, 1,
                       ctx->prepared_small_dense_native_quant_matvec_pipeline) ||
                   metal_small_dense_native_quant_graph_path_supported(
                       ctx, op->type, pre_wbuf->native_quant_prepared, 0,
                       ctx->small_dense_native_quant_matvec_pipeline))) ||
                 (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                  (op->flags & 1u) &&
                  (metal_small_dense_native_quant_graph_path_supported(
                       ctx, op->type, pre_wbuf->native_quant_prepared, 1,
                       ctx->prepared_small_dense_native_quant_split_pipeline) ||
                   metal_small_dense_native_quant_graph_path_supported(
                       ctx, op->type, pre_wbuf->native_quant_prepared, 0,
                       ctx->small_dense_native_quant_split_pipeline))) ||
                 (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                  op->p[6] &&
                  (metal_small_dense_native_quant_graph_path_supported(
                       ctx, op->type, pre_wbuf->native_quant_prepared, 1,
                       ctx->prepared_small_dense_native_quant_gateup_pipeline) ||
                   metal_small_dense_native_quant_graph_path_supported(
                       ctx, op->type, pre_wbuf->native_quant_prepared, 0,
                       ctx->small_dense_native_quant_gateup_pipeline))));
            /* Skip redundant PSO switch — avoids GPU instruction cache flush */
            if (!native_quant_deferred_pso && pipeline != current_pso) {
                [enc setComputePipelineState:pipeline];
                current_pso = pipeline;
            }

            /* Set buffers per shader type + setBytes for uniforms */
            uint32_t params[BN_GPU_OP_PARAMS];
            memcpy(params, op->p, sizeof(params));

            /* Inject fused bias for matvec */
            if (shader == BN_GPU_SHADER_MATVEC && op->W_buf) {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
            }

            BnMetalBuf *routed_down = NULL;
            switch (shader) {
            case BN_GPU_SHADER_MATVEC: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                if (wbuf->native_matvec_layout) {
                    size_t blocks_per_row = op->cols > 0
                        ? (size_t)op->cols / 32 : 0;
                    if (op->rows <= 0 || op->cols <= 0 ||
                        (op->cols % 32) != 0 || blocks_per_row == 0 ||
                        (size_t)op->rows > SIZE_MAX / blocks_per_row / 18 ||
                        (size_t)op->rows * blocks_per_row * 18 > wbuf->size)
                        METAL_EXEC_FAIL();
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in]
                           offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out]
                           offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                } else if ((op->flags &
                     BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT) != 0) {
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in]
                           offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out]
                           offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                } else if (bn_gpu_policy_metal_prepared_f32_enabled(
                               ctx->runtime_policy) &&
                    wbuf->native_quant_prepared &&
                    ctx->prepared_f32_matvec_pipeline) {
                    [enc setComputePipelineState:ctx->prepared_f32_matvec_pipeline];
                    current_pso = ctx->prepared_f32_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                } else if (metal_block_q8_activation_graph_path_supported(
                        ctx, op->type, op->p[6] != 0)) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_native_quant_scratch(
                            ctx, op->cols, (int)n_tokens) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, n_tokens);
                    [enc setComputePipelineState:ctx->q8_native_quant_matvec_pipeline];
                    current_pso = ctx->q8_native_quant_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->p[6] &&
                    (op->flags &
                     BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0 &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 1,
                        ctx->prepared_reference_native_quant_matvec_pipeline)) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_native_quant_scratch(
                            ctx, op->cols, (int)n_tokens) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, n_tokens);
                    ctx->small_dense_native_quant_matvec_dispatches++;
                    [enc setComputePipelineState:
                        ctx->prepared_reference_native_quant_matvec_pipeline];
                    current_pso =
                        ctx->prepared_reference_native_quant_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->p[6] &&
                    (op->flags &
                     BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0 &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 0,
                        ctx->reference_native_quant_matvec_pipeline)) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_native_quant_scratch(
                            ctx, op->cols, (int)n_tokens) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, n_tokens);
                    [enc setComputePipelineState:
                        ctx->reference_native_quant_matvec_pipeline];
                    current_pso = ctx->reference_native_quant_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->p[6] &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 1,
                        ctx->prepared_small_dense_native_quant_matvec_pipeline)) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_native_quant_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        METAL_EXEC_FAIL();
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(native_quant_shapes, &n_native_quant_shapes, 16,
                                                shader, op->type, 0,
                                                (uint32_t)op->cols,
                                                n_tokens);
                        metal_profile_add_shape(small_dense_native_quant_weight_shapes, &n_small_dense_native_quant_weight_shapes, 16,
                                                shader, op->type,
                                                (uint32_t)op->rows,
                                                (uint32_t)op->cols, n_tokens);
                    }
                    metal_encode_native_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, n_tokens);
                    ctx->small_dense_native_quant_matvec_dispatches++;
                    [enc setComputePipelineState:ctx->prepared_small_dense_native_quant_matvec_pipeline];
                    current_pso = ctx->prepared_small_dense_native_quant_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->p[6] &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 0,
                        ctx->small_dense_native_quant_matvec_pipeline)) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_native_quant_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        METAL_EXEC_FAIL();
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(native_quant_shapes, &n_native_quant_shapes, 16,
                                                shader, op->type, 0,
                                                (uint32_t)op->cols,
                                                n_tokens);
                        metal_profile_add_shape(small_dense_native_quant_weight_shapes, &n_small_dense_native_quant_weight_shapes, 16,
                                                shader, op->type,
                                                (uint32_t)op->rows,
                                                (uint32_t)op->cols, n_tokens);
                    }
                    metal_encode_native_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, n_tokens);
                    ctx->small_dense_native_quant_matvec_dispatches++;
                    [enc setComputePipelineState:ctx->small_dense_native_quant_matvec_pipeline];
                    current_pso = ctx->small_dense_native_quant_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if ((bn_gpu_policy_metal_specialized_native_quant_matvec_supported(
                                ctx->runtime_policy,
                                op->type, op->cols,
                                ctx->specialized_native_quant_pipeline != nil,
                                metal_specialized_native_matvec_pipeline(
                                    ctx, op->type) != nil) ||
                            (bn_gpu_policy_metal_specialized_native_quant_shape_default_enabled(
                                 ctx->runtime_policy,
                                 op->type, op->rows, op->cols) &&
                             ctx->specialized_native_quant_pipeline != nil &&
                             metal_specialized_native_matvec_pipeline(
                                 ctx, op->type) != nil))) {
                    int reference_accumulation =
                        (op->flags &
                         BN_GPU_OP_FLAG_REFERENCE_BLOCK_ACCUMULATION) != 0;
                    id<MTLComputePipelineState> specialized_pipeline =
                        reference_accumulation &&
                        bn_backend_quant_supports_reference_prepared_accumulation(
                            op->type)
                            ? ctx->specialized_native_matvec_pipeline
                            : metal_specialized_native_matvec_pipeline_for_shape(
                                  ctx, op->type, op->rows, op->cols);
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_specialized_native_quant_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_specialized_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, n_tokens);
                    [enc setComputePipelineState:specialized_pipeline];
                    current_pso = specialized_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:4];
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                } else {
                    if (wbuf->native_quant_prepared)
                        METAL_EXEC_FAIL();
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                }
                break;
            }
            case BN_GPU_SHADER_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (ctx->cpu_order_rmsnorm_enabled &&
                    ctx->cpu_order_rmsnorm_pipeline) {
                    [enc setComputePipelineState:ctx->cpu_order_rmsnorm_pipeline];
                    current_pso = ctx->cpu_order_rmsnorm_pipeline;
                }
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                if (wbuf)
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:1];
                else
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_ROPE: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_GQA_SCORES: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_KEY_CACHE] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ATT] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_SOFTMAX: {
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ATT] offset:0 atIndex:0];
                [enc setBytes:params length:sizeof(params) atIndex:1];
                break;
            }
            case BN_GPU_SHADER_GQA_COMBINE: {
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ATT] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_VALUE_CACHE] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_SILU_GATE:
            case BN_GPU_SHADER_RELU2_GATE:
            case BN_GPU_SHADER_GELU_GATE: {
                if (shader == BN_GPU_SHADER_SILU_GATE)
                    params[1] = op->flags;
                params[2] = op->flags;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_RESIDUAL_ADD: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_BIAS_ADD: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_RESIDUAL_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_WEIGHTED_ADD: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_WEIGHTED_ADD_SIGMOID: {
                BnMetalBuf *gate = (BnMetalBuf *)op->W_buf;
                if (!gate || gate->type != BN_GGUF_TENSOR_F32)
                    continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBuffer:gate->buf offset:gate->offset atIndex:2];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_XB] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_MOE_ROUTE_TOPK: {
                BnMetalBuf *router = (BnMetalBuf *)op->W_buf;
                BnMetalBuf *expert_down_scale = (BnMetalBuf *)op->W_buf2;
                if (!router || router->type != BN_GGUF_TENSOR_F32 ||
                    op->p[0] > BN_METAL_MAX_MOE_ROUTE_EXPERTS) {
                    fprintf(stderr,
                            "[bn:gpu:metal] invalid routed MoE router buffer "
                            "type=%d\n", router ? router->type : -1);
                    [enc endEncoding];
                    enc = nil;
                    return -1;
                }
                params[3] = op->flags;
                params[5] = (uint32_t)op->cols;
                [enc setComputePipelineState:ctx->moe_route_logits_pipeline];
                [enc setBuffer:router->buf offset:router->offset atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                MTLSize route_tpg = MTLSizeMake(256, 1, 1);
                MTLSize route_grid = MTLSizeMake((uint32_t)op->p[0], 1, 1);
                [enc dispatchThreadgroups:route_grid
                     threadsPerThreadgroup:route_tpg];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:pipeline];
                current_pso = pipeline;
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBuffer:expert_down_scale
                                   ? expert_down_scale->buf
                                   : ctx->act_bufs[op->buf_aux]
                         offset:expert_down_scale ? expert_down_scale->offset : 0
                        atIndex:2];
                params[4] = expert_down_scale ? 1u : 0u;
                [enc setBytes:params length:sizeof(params) atIndex:3];
                route_pending = 1;
                break;
            }
            case BN_GPU_SHADER_MOE_ROUTED_FFN: {
                BnMetalBuf *gate = (BnMetalBuf *)op->W_buf;
                BnMetalBuf *up = (BnMetalBuf *)op->W_buf2;
                BnMetalBuf *down = (BnMetalBuf *)op->W_buf3;
                if (route_pending && !ctx->mmap_fits_working_set &&
                    op->p[2] > 0 &&
                    op->p[1] > op->p[2] &&
                    ctx->moe_resident_budget > 0 &&
                    gate && up && down && gate->is_borrowed &&
                    up->is_borrowed && down->is_borrowed) {
                    double routed_wait_start = bn_platform_time_ms();
                    [enc endEncoding];
                    enc = nil;
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    if ([cmd status] != MTLCommandBufferStatusCompleted)
                        METAL_EXEC_FAIL();
                    routed_wait_ms +=
                        bn_platform_time_ms() - routed_wait_start;
                    float *route = (float *)
                        [ctx->act_bufs[op->buf_aux] contents];
                    double routed_copy_start = bn_platform_time_ms();
                    BnMetalMoEResident *resident =
                        metal_prepare_moe_resident(
                            ctx, gate, up, down, route, (int)op->p[1],
                            (int)op->p[2], op->p[7], op->p[6],
                            (int)op->p[0], op->cols);
                    routed_copy_ms +=
                        bn_platform_time_ms() - routed_copy_start;
                    if (resident) {
                        gate = &resident->gate;
                        up = &resident->up;
                        down = &resident->down;
                        params[1] = (uint32_t)resident->slots;
                    }
                    cmd = [ctx->queue commandBuffer];
                    enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pipeline];
                    current_pso = pipeline;
                    since_barrier_writes = 0;
                }
                route_pending = 0;
                if (ctx->route_history_buf &&
                    !ctx->route_history_shape_printed) {
                    fprintf(stderr,
                            "[gpu:metal:route-history] stride=%d op_k=%u "
                            "layer=%u pipeline=%d\n",
                            ctx->route_history_stride, op->p[2], op->p[5],
                            ctx->moe_route_capture_pipeline != nil);
                    ctx->route_history_shape_printed = 1;
                }
                if (ctx->route_history_buf &&
                    ctx->route_history_count < ctx->route_history_capacity &&
                    ctx->route_history_stride == (int)op->p[2] + 1) {
                    uint32_t capture_params[3] = {
                        (uint32_t)(ctx->route_history_count *
                                   (size_t)ctx->route_history_stride),
                        op->p[2], op->p[5]
                    };
                    [enc setComputePipelineState:ctx->moe_route_capture_pipeline];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux]
                           offset:0 atIndex:0];
                    [enc setBuffer:ctx->route_history_buf offset:0 atIndex:1];
                    [enc setBytes:capture_params
                           length:sizeof(capture_params) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    ctx->route_history_count++;
                }
                int mid_buf = (int)op->p[4];
                if (ctx->gpu_profile && !ctx->routed_profile_types_printed) {
                    fprintf(stderr,
                            "[gpu:metal:profile] routed_types gate=%d up=%d down=%d\n",
                            gate ? gate->type : -1, up ? up->type : -1,
                            down ? down->type : -1);
                    ctx->routed_profile_types_printed = 1;
                }
                int routed_q4_0 = gate && up && down &&
                    bn_backend_quant_moe_routed_lowbit_block32(
                        gate->type, up->type, down->type);
                int routed_kquant = gate && up && down &&
                    bn_backend_quant_moe_routed_kquant_gateup(
                        gate->type, up->type) &&
                    bn_backend_quant_moe_direct_routed_down(down->type);
                if (!gate || !up || !down ||
                    (!routed_q4_0 && !routed_kquant) ||
                    mid_buf < 0 || mid_buf >= BN_GPU_BUF_COUNT) {
                    fprintf(stderr,
                            "[bn:gpu:metal] invalid routed MoE buffers "
                            "gate=%d up=%d down=%d mid=%d\n",
                            gate ? gate->type : -1, up ? up->type : -1,
                            down ? down->type : -1, mid_buf);
                    [enc endEncoding];
                    enc = nil;
                    return -1;
                }
                int scratch_ok = routed_q4_0
                    ? ensure_native_quant_scratch(
                          ctx, (int)op->p[0], (int)op->p[2])
                    : ensure_specialized_native_quant_scratch(
                          ctx, (int)op->p[0], (int)op->p[2]);
                if (scratch_ok != 0) {
                    [enc endEncoding];
                    enc = nil;
                    return -1;
                }
                /* The routed down dispatch from an earlier op reads the same
                 * backend-private quant scratch that this dispatch rewrites.
                 * Graph access masks cannot describe these internal buffers. */
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if (routed_q4_0)
                    metal_encode_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, 1);
                else
                    metal_encode_routed_q8k_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, 1);
                id<MTLComputePipelineState> gateup_pipeline = routed_q4_0
                    ? ctx->moe_routed_q4_0_gateup_pipeline : pipeline;
                [enc setComputePipelineState:gateup_pipeline];
                current_pso = gateup_pipeline;
                [enc setBuffer:gate->buf offset:gate->offset atIndex:0];
                [enc setBuffer:up->buf offset:up->offset atIndex:1];
                [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:2];
                [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:3];
                int route_index = routed_q4_0 ? 4 : 5;
                int mid_index = routed_q4_0 ? 5 : 6;
                int params_index = routed_q4_0 ? 6 : 7;
                if (!routed_q4_0)
                    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:4];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:route_index];
                [enc setBuffer:ctx->act_bufs[mid_buf] offset:0 atIndex:mid_index];
                params[5] = (uint32_t)op->cols;
                [enc setBytes:params length:sizeof(params) atIndex:params_index];
                routed_down = down;
                break;
            }
            case BN_GPU_SHADER_SSM_CONV_SILU: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_CONV_STATE] offset:0 atIndex:1];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_SSM_L2NORM: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_SSM_ALPHA_BETA: {
                BnMetalBuf *dt_buf = (BnMetalBuf *)op->W_buf;
                if (!dt_buf) continue;
                void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
                BnMetalBuf *a_wbuf = (BnMetalBuf *)a_ptr;
                if (!a_wbuf) continue;
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_BETA] offset:0 atIndex:1];
                [enc setBuffer:dt_buf->buf offset:dt_buf->offset atIndex:2];
                [enc setBuffer:a_wbuf->buf offset:a_wbuf->offset atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT: {
                BnMetalBuf *dt_buf = (BnMetalBuf *)op->W_buf;
                if (!dt_buf) continue;
                void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
                BnMetalBuf *a_wbuf = (BnMetalBuf *)a_ptr;
                if (!a_wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_BETA] offset:0 atIndex:2];
                [enc setBuffer:dt_buf->buf offset:dt_buf->offset atIndex:3];
                [enc setBuffer:a_wbuf->buf offset:a_wbuf->offset atIndex:4];
                [enc setBytes:params length:sizeof(params) atIndex:5];
                break;
            }
            case BN_GPU_SHADER_SSM_DELTA: {
                int v_buf = op->p[7] ? op->buf_in : BN_GPU_BUF_SSM_V;
                if (ctx->gpu_profile >= 2 &&
                    !ctx->ssm_profile_shape_printed) {
                    fprintf(stderr,
                            "[gpu:metal:profile] ssm_delta_shape hk=%u hv=%u "
                            "k_heads=%u v_heads=%d\n",
                            op->p[0], op->p[1], op->p[2], op->rows);
                    ctx->ssm_profile_shape_printed = 1;
                }
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_STATE] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                [enc setBuffer:ctx->act_bufs[v_buf] offset:0 atIndex:4];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA] offset:0 atIndex:5];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_BETA] offset:0 atIndex:6];
                [enc setBytes:params length:sizeof(params) atIndex:7];
                break;
            }
            case BN_GPU_SHADER_SSM_GATE: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_PER_HEAD_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_DEINTERLEAVE_Q: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_SIGMOID_GATE: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_FLASH_ATTN: {
                /* Fused: Q(buf_in) + key_cache + value_cache → xb(buf_out) */
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_KEY_CACHE] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_VALUE_CACHE] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_COPY: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_MATVEC_SPLIT: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                int out2_idx = (op->rows >= 0 && op->rows < BN_GPU_BUF_COUNT)
                    ? op->rows
                    : op->buf_aux;
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
                if (pipeline == ctx->specialized_q4k_native_split_pipeline) {
                    if (ensure_specialized_native_quant_scratch(
                            ctx, op->cols, 1) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_specialized_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, 1);
                    [enc setComputePipelineState:pipeline];
                    current_pso = pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:5];
                    [enc setBuffer:ctx->act_bufs[out2_idx] offset:0 atIndex:6];
                    [enc setBytes:params length:sizeof(params) atIndex:7];
                } else if (bn_gpu_policy_metal_prepared_f32_enabled(
                               ctx->runtime_policy) &&
                    wbuf->native_quant_prepared &&
                    ctx->prepared_f32_split_pipeline) {
                    [enc setComputePipelineState:ctx->prepared_f32_split_pipeline];
                    current_pso = ctx->prepared_f32_split_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[out2_idx] offset:0 atIndex:4];
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                } else if ((op->flags & 1u) &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 1,
                        ctx->prepared_small_dense_native_quant_split_pipeline)) {
                    if (ensure_native_quant_scratch(ctx, op->cols, 1) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_native_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->small_dense_native_quant_split_dispatches++;
                    [enc setComputePipelineState:ctx->prepared_small_dense_native_quant_split_pipeline];
                    current_pso = ctx->prepared_small_dense_native_quant_split_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[out2_idx] offset:0 atIndex:5];
                    [enc setBytes:params length:sizeof(params) atIndex:6];
                } else if ((op->flags & 1u) &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 0,
                        ctx->small_dense_native_quant_split_pipeline)) {
                    if (ensure_native_quant_scratch(ctx, op->cols, 1) != 0)
                        METAL_EXEC_FAIL();
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(native_quant_shapes, &n_native_quant_shapes, 16,
                                                shader, op->type, 0,
                                                (uint32_t)op->cols,
                                                1);
                        metal_profile_add_shape(small_dense_native_quant_weight_shapes, &n_small_dense_native_quant_weight_shapes, 16,
                                                shader, op->type, op->p[0],
                                                (uint32_t)op->cols, 1);
                    }
                    metal_encode_native_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->small_dense_native_quant_split_dispatches++;
                    [enc setComputePipelineState:ctx->small_dense_native_quant_split_pipeline];
                    current_pso = ctx->small_dense_native_quant_split_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[out2_idx] offset:0 atIndex:5];
                    [enc setBytes:params length:sizeof(params) atIndex:6];
                } else {
                    if (wbuf->native_quant_prepared)
                        METAL_EXEC_FAIL();
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];  // out0
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];  // out1
                    [enc setBuffer:ctx->act_bufs[out2_idx] offset:0 atIndex:4];     // out2
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                }
                break;
            }
            case BN_GPU_SHADER_ROPE_QK: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];   // Q
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];  // K (KEY_CACHE)
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_FUSED_GATEUP_SILU: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                params[3] = op->flags;
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
                if (op->p[6] &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 1,
                        ctx->prepared_small_dense_native_quant_gateup_pipeline)) {
                    if (ensure_native_quant_scratch(ctx, op->cols, 1) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_native_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->small_dense_native_quant_gateup_dispatches++;
                    [enc setComputePipelineState:ctx->prepared_small_dense_native_quant_gateup_pipeline];
                    current_pso = ctx->prepared_small_dense_native_quant_gateup_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->p[6] &&
                    metal_small_dense_native_quant_graph_path_supported(
                        ctx, op->type, wbuf->native_quant_prepared, 0,
                        ctx->small_dense_native_quant_gateup_pipeline)) {
                    if (ensure_native_quant_scratch(ctx, op->cols, 1) != 0)
                        METAL_EXEC_FAIL();
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(native_quant_shapes, &n_native_quant_shapes, 16,
                                                shader, op->type, 0,
                                                (uint32_t)op->cols,
                                                1);
                        metal_profile_add_shape(small_dense_native_quant_weight_shapes, &n_small_dense_native_quant_weight_shapes, 16,
                                                shader, op->type, op->p[2],
                                                (uint32_t)op->cols, 1);
                    }
                    metal_encode_native_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->small_dense_native_quant_gateup_dispatches++;
                    [enc setComputePipelineState:ctx->small_dense_native_quant_gateup_pipeline];
                    current_pso = ctx->small_dense_native_quant_gateup_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else {
                    if (wbuf->native_quant_prepared)
                        METAL_EXEC_FAIL();
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                }
                break;
            }
            case BN_GPU_SHADER_Q4K_MATVEC_SPLIT: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                if (pipeline == ctx->specialized_q4k_native_split_pipeline) {
                    int out2_idx = (op->rows >= 0 &&
                                    op->rows < BN_GPU_BUF_COUNT)
                        ? op->rows : op->buf_aux;
                    if (ensure_specialized_native_quant_scratch(
                            ctx, op->cols, 1) != 0)
                        METAL_EXEC_FAIL();
                    metal_encode_specialized_native_quant(
                        enc, ctx, ctx->act_bufs[op->buf_in],
                        (uint32_t)op->cols, 1);
                    [enc setComputePipelineState:pipeline];
                    current_pso = pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:5];
                    [enc setBuffer:ctx->act_bufs[out2_idx] offset:0 atIndex:6];
                    [enc setBytes:params length:sizeof(params) atIndex:7];
                } else {
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                }
                break;
            }
            default: continue;
            }

            /* Compute workgroup count (same logic as wgpu) */
            uint32_t wg_x = 1, wg_y = 1;
            uint32_t tile_rows = metal_float_matvec_tile_rows(op->type);
            uint32_t threads_per_tg = 256;
            if (shader == BN_GPU_SHADER_MATVEC &&
                (op->flags & BN_GPU_OP_FLAG_MATVEC_REFERENCE_KQUANT) != 0)
                tile_rows = 256;
            if (pipeline == ctx->specialized_q4k_native_split_pipeline) {
                tile_rows = 128;
            }
            if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                pipeline == ctx->fwd_pipelines[BN_GPU_SHADER_FUSED_GATEUP_SILU])
                tile_rows = 64;
            BnMetalBuf *grid_wbuf = (BnMetalBuf *)op->W_buf;
            int native_quant_tile =
                grid_wbuf &&
                ((shader == BN_GPU_SHADER_MATVEC &&
                  !metal_block_q8_activation_graph_path_supported(
                      ctx, op->type, op->p[6] != 0) &&
                  op->p[6] &&
                  metal_small_dense_native_quant_graph_path_supported(
                      ctx, op->type, grid_wbuf->native_quant_prepared, 0,
                      ctx->small_dense_native_quant_matvec_pipeline)) ||
                 (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                  (op->flags & 1u) &&
                  metal_small_dense_native_quant_graph_path_supported(
                      ctx, op->type, grid_wbuf->native_quant_prepared, 0,
                      ctx->small_dense_native_quant_split_pipeline)) ||
                 (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                  op->p[6] &&
                  metal_small_dense_native_quant_graph_path_supported(
                      ctx, op->type, grid_wbuf->native_quant_prepared, 0,
                      ctx->small_dense_native_quant_gateup_pipeline)));
            if (native_quant_tile) {
                tile_rows = 16;
                threads_per_tg = 128;
            }
            if (pipeline == ctx->borrowed_native_q4_matvec_pipeline) {
                tile_rows = 32;
                threads_per_tg = 256;
            }
            if (pipeline == ctx->reference_native_quant_matvec_pipeline) {
                tile_rows = 32;
                threads_per_tg = 128;
            }
            if (pipeline ==
                ctx->prepared_small_dense_native_quant_gateup_pipeline) {
                tile_rows = 16;
                threads_per_tg = 128;
            }
            switch (shader) {
            case BN_GPU_SHADER_MATVEC: {
                if (op->p[3] > 0) {
                    uint32_t tiled_rows = ((uint32_t)op->rows + tile_rows - 1) / tile_rows;
                    wg_x = op->p[3];
                    wg_y = (tiled_rows + op->p[3] - 1) / op->p[3];
                } else {
                    wg_x = ((uint32_t)op->rows + tile_rows - 1) / tile_rows;
                    wg_y = op->p[2];
                    if (wg_y == 0) wg_y = 1;
                }
                break;
            }
            case BN_GPU_SHADER_RMSNORM:
            case BN_GPU_SHADER_RESIDUAL_RMSNORM:
            case BN_GPU_SHADER_SSM_ALPHA_BETA:
            case BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT:
                wg_x = 1;
                break;
            case BN_GPU_SHADER_ROPE:
                wg_x = op->p[0];
                break;
            case BN_GPU_SHADER_SOFTMAX:
                if (pipeline == ctx->reference_softmax_pipeline) {
                    threads_per_tg = 64;
                    wg_x = (op->p[0] + threads_per_tg - 1) / threads_per_tg;
                } else {
                    wg_x = op->p[0];
                }
                break;
            case BN_GPU_SHADER_GQA_COMBINE:
                if (pipeline == ctx->reference_gqa_combine_pipeline) {
                    threads_per_tg = 64;
                    uint32_t elements = op->p[0] * op->p[1];
                    wg_x = (elements + threads_per_tg - 1) / threads_per_tg;
                } else {
                    wg_x = op->p[0];
                }
                break;
            case BN_GPU_SHADER_GQA_SCORES:
                if (pipeline == ctx->reference_gqa_scores_pipeline) {
                    threads_per_tg = 64;
                    uint32_t elements = op->p[0] * op->p[2];
                    wg_x = (elements + threads_per_tg - 1) / threads_per_tg;
                } else {
                    wg_x = op->p[0];
                    wg_y = (op->p[2] + 7) / 8;
                }
                break;
            case BN_GPU_SHADER_SILU_GATE:
            case BN_GPU_SHADER_RELU2_GATE:
            case BN_GPU_SHADER_GELU_GATE:
            case BN_GPU_SHADER_RESIDUAL_ADD:
            case BN_GPU_SHADER_BIAS_ADD:
            case BN_GPU_SHADER_WEIGHTED_ADD:
            case BN_GPU_SHADER_SSM_CONV_SILU:
                wg_x = (op->p[0] + 255) / 256;
                break;
            case BN_GPU_SHADER_WEIGHTED_ADD_SIGMOID:
            case BN_GPU_SHADER_MOE_ROUTE_TOPK:
                wg_x = 1;
                break;
            case BN_GPU_SHADER_MOE_ROUTED_FFN:
                wg_x = ((op->p[0] * op->p[2]) + 31) / 32;
                break;
            case BN_GPU_SHADER_SSM_L2NORM:
                wg_x = (uint32_t)op->rows;
                break;
            case BN_GPU_SHADER_SSM_DELTA:
            case BN_GPU_SHADER_SSM_GATE:
                wg_x = (uint32_t)op->rows;
                break;
            case BN_GPU_SHADER_PER_HEAD_RMSNORM:
                wg_x = (uint32_t)op->rows;
                break;
            case BN_GPU_SHADER_DEINTERLEAVE_Q:
            case BN_GPU_SHADER_SIGMOID_GATE:
                wg_x = (op->p[0] + 255) / 256;
                break;
            case BN_GPU_SHADER_FLASH_ATTN:
                wg_x = op->p[0];  /* one head per threadgroup */
                break;
            case BN_GPU_SHADER_COPY:
                wg_x = (op->p[2] + 255) / 256;
                break;
            case BN_GPU_SHADER_MATVEC_SPLIT:
                wg_x = (op->p[0] + tile_rows - 1) / tile_rows;
                break;
            case BN_GPU_SHADER_ROPE_QK:
                wg_x = op->p[0] + op->p[4];   // n_q_heads + n_kv_heads
                break;
            case BN_GPU_SHADER_FUSED_GATEUP_SILU:
                wg_x = (op->p[2] + tile_rows - 1) / tile_rows;
                break;
            case BN_GPU_SHADER_Q4K_MATVEC_SPLIT:
                wg_x = (op->p[0] + 31) / 32;
                break;
            }

            if (wg_x == 0) wg_x = 1;
            MTLSize tpg = MTLSizeMake(threads_per_tg, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, wg_y, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];

            if (shader == BN_GPU_SHADER_MOE_ROUTED_FFN) {
                BnMetalBuf *down = routed_down
                    ? routed_down : (BnMetalBuf *)op->W_buf3;
                int mid_buf = (int)op->p[4];
                if (profile_each_op) {
                    double phase_wall0 = bn_platform_time_ms();
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    double phase_wall1 = bn_platform_time_ms();
                    double gpu_start = cmd.GPUStartTime;
                    double gpu_end = cmd.GPUEndTime;
                    moe_gateup_gpu_ms += gpu_end > gpu_start
                        ? (gpu_end - gpu_start) * 1000.0
                        : phase_wall1 - phase_wall0;
                    moe_gateup_wall_ms += phase_wall1 - phase_wall0;
                    cmd = [ctx->queue commandBuffer];
                    enc = [cmd computeCommandEncoder];
                    current_pso = nil;
                }
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if (down && down->type == BN_GGUF_TENSOR_Q4_0)
                    metal_encode_native_quant(
                        enc, ctx, ctx->act_bufs[mid_buf], op->p[0], op->p[2]);
                else if (down && (down->type == BN_GGUF_TENSOR_Q4_K ||
                             down->type == BN_GGUF_TENSOR_Q5_K))
                    metal_encode_routed_q8k_quant(
                        enc, ctx, ctx->act_bufs[mid_buf], op->p[0], op->p[2]);
                id<MTLComputePipelineState> down_pipeline =
                    down && down->type == BN_GGUF_TENSOR_Q4_0
                        ? ctx->moe_routed_q4_0_down_pipeline
                    : down && down->type == BN_GGUF_TENSOR_Q6_K
                        ? ctx->moe_routed_q6k_down_pipeline
                    : down && down->type == BN_GGUF_TENSOR_Q5_K
                        ? ctx->moe_routed_q5k_down_pipeline
                            : ctx->moe_routed_q4k_down_pipeline;
                [enc setComputePipelineState:down_pipeline];
                current_pso = down_pipeline;
                [enc setBuffer:down->buf offset:down->offset atIndex:0];
                if (down && down->type == BN_GGUF_TENSOR_Q4_0) {
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:4];
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                } else if (down && down->type == BN_GGUF_TENSOR_Q5_K) {
                    [enc setBuffer:ctx->native_quant_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->native_quant_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->native_quant_block_sums_buf offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:5];
                    [enc setBytes:params length:sizeof(params) atIndex:6];
                } else {
                    [enc setBuffer:ctx->act_bufs[mid_buf] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                }
                uint32_t down_rows_per_group = down &&
                    bn_backend_quant_uses_down_kquant(down->type) ? 16u : 32u;
                MTLSize down_grid = MTLSizeMake(
                    ((uint32_t)op->cols + down_rows_per_group - 1) /
                        down_rows_per_group,
                    1, 1);
                [enc dispatchThreadgroups:down_grid threadsPerThreadgroup:tpg];
            }

            if (profile_each_op && shader >= 0 && shader < BN_GPU_SHADER_COUNT) {
                double op_wall0 = bn_platform_time_ms();
                [enc endEncoding];
                enc = nil;
                id<MTLCommandBuffer> done_cmd = cmd;
                [done_cmd commit];
                [done_cmd waitUntilCompleted];
                double op_wall1 = bn_platform_time_ms();
                double gpu_ms = 0.0;
                double gpu_start = done_cmd.GPUStartTime;
                double gpu_end = done_cmd.GPUEndTime;
                if (gpu_end > gpu_start)
                    gpu_ms = (gpu_end - gpu_start) * 1000.0;
                shader_gpu_ms[shader] += gpu_ms;
                shader_wall_ms[shader] += op_wall1 - op_wall0;
                shader_profile_counts[shader]++;
                if (shader == BN_GPU_SHADER_MOE_ROUTED_FFN) {
                    moe_down_gpu_ms += gpu_ms > 0.0
                        ? gpu_ms : op_wall1 - op_wall0;
                    moe_down_wall_ms += op_wall1 - op_wall0;
                    moe_profile_count++;
                }
                if (shader == BN_GPU_SHADER_MATVEC ||
                    shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                    shader == BN_GPU_SHADER_FUSED_GATEUP_SILU ||
                    shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT) {
                    uint32_t rows = (uint32_t)op->rows;
                    uint32_t cols = (uint32_t)op->cols;
                    uint32_t aux = 1;
                    if (shader == BN_GPU_SHADER_MATVEC) {
                        aux = op->p[2] ? op->p[2] : 1;
                    } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                               shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT) {
                        rows = op->p[0];
                    } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU) {
                        rows = op->p[2];
                    }
                    metal_profile_add_shape_time(
                        timed_shapes, &n_timed_shapes, 64, shader, op->type,
                        rows,
                        cols, aux, gpu_ms, op_wall1 - op_wall0);
                }
                cmd = nil;
                current_pso = nil;
            }
        }

        if (enc) [enc endEncoding];
#undef METAL_EXEC_FAIL

        t_encode = bn_platform_time_ms();

        if (cmd) {
            [cmd commit];
            if (produces_unread_logits)
                ctx->pending_logits_cmd = cmd;
            else
                [cmd waitUntilCompleted];
        }

        t_gpu = bn_platform_time_ms();
    }

    /* Readback: unified memory — just memcpy */
    if (out_host && out_len > 0 && readback_buf >= 0
        && readback_buf < BN_GPU_BUF_COUNT && ctx->act_bufs[readback_buf]) {
        size_t readback_size = (size_t)out_len * sizeof(float);
        memcpy(out_host, [ctx->act_bufs[readback_buf] contents], readback_size);
    }


    double t1 = bn_platform_time_ms();

    /* GPU profiling */
    if (ctx->gpu_profile >= 1 &&
        (ctx->gpu_profile >= 3 || ctx->gpu_frame < 5 ||
         (ctx->gpu_frame % 50 == 0))) {
        fprintf(stderr, "[gpu:metal:profile] frame=%d ops=%d native_quant=%d small_dense_native_quant_matvec=%d small_dense_native_quant_split=%d small_dense_native_quant_gateup=%d specialized_native_quant=%d barriers=%d encode=%.1fms route_wait=%.1fms route_copy=%.1fms gpu=%.1fms readback=%.1fms total=%.1fms\n",
                ctx->gpu_frame, n_ops, ctx->native_quant_dispatches,
                ctx->small_dense_native_quant_matvec_dispatches,
                ctx->small_dense_native_quant_split_dispatches,
                ctx->small_dense_native_quant_gateup_dispatches,
                ctx->specialized_native_quant_dispatches,
                n_barriers,
                t_encode - t0, routed_wait_ms, routed_copy_ms,
                t_gpu - t_encode, t1 - t_gpu, t1 - t0);
        if (ctx->moe_resident_count > 0)
            fprintf(stderr,
                    "[gpu:metal:moe-cache] entries=%d hits=%zu misses=%zu "
                    "resident_MB=%zu\n",
                    ctx->moe_resident_count, ctx->moe_resident_hits,
                    ctx->moe_resident_misses,
                    ctx->moe_resident_bytes / (1024u * 1024u));
    }
    /* Per-op-type breakdown (GPU profile level >= 2, first graph only) */
    if (ctx->gpu_profile >= 2 && ctx->gpu_frame == 0) {
        int cat_count[BN_GPU_SHADER_COUNT]; memset(cat_count, 0, sizeof(cat_count));
        for (int i = 0; i < n_ops; i++) {
            int s = bn_gpu_shader_from_op_code(ops[i].op_code);
            if (s >= 0 && s < BN_GPU_SHADER_COUNT) cat_count[s]++;
        }
        fprintf(stderr, "[gpu:metal:breakdown] --- op counts ---\n");
        for (int s = 0; s < BN_GPU_SHADER_COUNT; s++) {
            if (cat_count[s] > 0)
                fprintf(stderr, "  %-16s: %3d ops\n",
                        metal_shader_profile_name(s), cat_count[s]);
        }
        if (n_native_quant_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- native quant activation shapes ---\n");
            for (int i = 0; i < n_native_quant_shapes; i++) {
                fprintf(stderr, "  %-16s: %3d dispatches type=%s cols=%u tokens=%u\n",
                        metal_shader_profile_name(native_quant_shapes[i].shader),
                        native_quant_shapes[i].count,
                        bn_quant_format_gpu_shader_name(native_quant_shapes[i].type),
                        native_quant_shapes[i].cols,
                        native_quant_shapes[i].aux);
            }
        }
        if (n_small_dense_native_quant_weight_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- small-dense native-quant weight matvec shapes ---\n");
            for (int i = 0; i < n_small_dense_native_quant_weight_shapes; i++) {
                fprintf(stderr, "  %-16s: %3d dispatches type=%s rows=%u cols=%u tokens=%u\n",
                        metal_shader_profile_name(small_dense_native_quant_weight_shapes[i].shader),
                        small_dense_native_quant_weight_shapes[i].count,
                        bn_quant_format_gpu_shader_name(small_dense_native_quant_weight_shapes[i].type),
                        small_dense_native_quant_weight_shapes[i].rows,
                        small_dense_native_quant_weight_shapes[i].cols, small_dense_native_quant_weight_shapes[i].aux);
            }
        }
        if (n_matvec_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- matvec shapes ---\n");
            for (int i = 0; i < n_matvec_shapes; i++) {
                fprintf(stderr, "  %-16s: %3d ops type=%s rows=%u cols=%u tokens=%u\n",
                        metal_shader_profile_name(matvec_shapes[i].shader),
                        matvec_shapes[i].count,
                        bn_quant_format_gpu_shader_name(matvec_shapes[i].type),
                        matvec_shapes[i].rows,
                        matvec_shapes[i].cols, matvec_shapes[i].aux);
            }
        }
        int adjacent_quant_candidates = 0;
        for (int i = 1; i < n_ops; i++) {
            int prev_shader =
                bn_gpu_shader_from_op_code(ops[i - 1].op_code);
            int shader = bn_gpu_shader_from_op_code(ops[i].op_code);
            int prev_matvec = prev_shader == BN_GPU_SHADER_MATVEC ||
                prev_shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                prev_shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT;
            int matvec = shader == BN_GPU_SHADER_MATVEC ||
                shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT;
            if (prev_matvec && matvec &&
                ops[i - 1].buf_in == ops[i].buf_in &&
                ops[i - 1].cols == ops[i].cols &&
                bn_quant_format_supports_specialized_native_quant_matvec(
                    ops[i - 1].type) &&
                bn_quant_format_supports_specialized_native_quant_matvec(
                    ops[i].type))
                adjacent_quant_candidates++;
        }
        fprintf(stderr,
                "[gpu:metal:breakdown] adjacent quant reuse candidates: %d\n",
                adjacent_quant_candidates);
    }
    /* Per-op command-buffer timing (GPU profile level >= 4, first graph only).
     * This intentionally changes submission granularity and is diagnostic-only. */
    if (ctx->gpu_profile >= 4 && ctx->gpu_frame == 0) {
        fprintf(stderr, "[gpu:metal:breakdown] --- per-op shader timing ---\n");
        for (int s = 0; s < BN_GPU_SHADER_COUNT; s++) {
            if (shader_profile_counts[s] > 0) {
                double shown_gpu = shader_gpu_ms[s] > 0.0
                    ? shader_gpu_ms[s]
                    : shader_wall_ms[s];
                fprintf(stderr, "  %-16s: %3d ops gpu=%.3fms wall=%.3fms avg=%.3fms\n",
                        metal_shader_profile_name(s),
                        shader_profile_counts[s],
                        shown_gpu,
                        shader_wall_ms[s],
                        shown_gpu / (double)shader_profile_counts[s]);
            }
        }
        if (moe_profile_count > 0) {
            fprintf(stderr,
                    "[gpu:metal:breakdown] routed phases: %d layers "
                    "gateup_gpu=%.3fms gateup_wall=%.3fms "
                    "down_gpu=%.3fms down_wall=%.3fms\n",
                    moe_profile_count, moe_gateup_gpu_ms,
                    moe_gateup_wall_ms, moe_down_gpu_ms,
                    moe_down_wall_ms);
        }
        if (n_timed_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- per-shape timing ---\n");
            for (int i = 0; i < n_timed_shapes; i++) {
                double shown_gpu = timed_shapes[i].gpu_ms > 0.0
                    ? timed_shapes[i].gpu_ms
                    : timed_shapes[i].wall_ms;
                fprintf(stderr, "  %-16s: %3d ops type=%s rows=%u cols=%u tokens=%u gpu=%.3fms avg=%.3fms\n",
                        metal_shader_profile_name(timed_shapes[i].shader),
                        timed_shapes[i].count,
                        bn_quant_format_gpu_shader_name(timed_shapes[i].type),
                        timed_shapes[i].rows,
                        timed_shapes[i].cols,
                        timed_shapes[i].aux,
                        shown_gpu,
                        shown_gpu / (double)timed_shapes[i].count);
            }
        }
    }
    ctx->gpu_frame++;

    return 0;
}

static uint32_t metal_routed_expert_stride(const BnMetalBuf *buf,
                                           int n_experts,
                                           int rows,
                                           int cols) {
    if (!buf || n_experts <= 0 || rows <= 0 || cols <= 0)
        return 0;
    BnQWeight weight = {
        .data = (void *)1,
        .type = buf->type,
        .rows = rows,
        .cols = cols,
        .scale = 1.0f,
    };
    size_t expert_bytes = bn_qweight_data_size(&weight);
    if (expert_bytes == 0 || expert_bytes > UINT32_MAX ||
        buf->size < expert_bytes)
        return 0;
    if (n_experts == 1)
        return (uint32_t)expert_bytes;
    size_t gaps = buf->size - expert_bytes;
    if (gaps % (size_t)(n_experts - 1) != 0)
        return 0;
    size_t stride = gaps / (size_t)(n_experts - 1);
    return stride > 0 && stride <= UINT32_MAX ? (uint32_t)stride : 0;
}

static int metal_moe_routed_ffn_batch(
    void *vctx, float *out,
    void *gate_all_buf, void *up_all_buf, void *down_all_buf,
    const int *indices, const float *weights, const float *X,
    int n_tokens, int dim, int hidden_dim, int n_experts, int k,
    int gate_type, int up_type, int down_type, int act_type) {
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *gate = (BnMetalBuf *)gate_all_buf;
    BnMetalBuf *up = (BnMetalBuf *)up_all_buf;
    BnMetalBuf *down = (BnMetalBuf *)down_all_buf;
    if (!ctx || !out || !gate || !up || !down || !indices || !weights ||
        !X || n_tokens <= 0 || dim <= 0 || hidden_dim <= 0 ||
        n_experts <= 0 || k <= 0 ||
        (act_type != BN_MODEL_ACTIVATION_SILU &&
         act_type != BN_MODEL_ACTIVATION_RELU2 &&
         act_type != BN_MODEL_ACTIVATION_GELU) ||
        gate->type != gate_type || up->type != up_type ||
        down->type != down_type ||
        (!bn_backend_quant_moe_routed_lowbit_block32(
             gate_type, up_type, down_type) &&
         !(bn_backend_quant_moe_routed_kquant_gateup(gate_type, up_type) &&
           bn_backend_quant_moe_direct_routed_down(down_type))) ||
        !ctx->act_bufs[BN_GPU_BUF_XB] ||
        !ctx->act_bufs[BN_GPU_BUF_MOE_HB] ||
        !ctx->act_bufs[BN_GPU_BUF_MOE_HB2] ||
        !ctx->act_bufs[BN_GPU_BUF_MOE_OUT])
        return -1;

    uint32_t gate_stride = metal_routed_expert_stride(
        gate, n_experts, hidden_dim, dim);
    uint32_t up_stride = metal_routed_expert_stride(
        up, n_experts, hidden_dim, dim);
    uint32_t down_stride = metal_routed_expert_stride(
        down, n_experts, dim, hidden_dim);
    if (gate_stride == 0 || gate_stride != up_stride || down_stride == 0)
        return -1;
    size_t input_bytes = (size_t)dim * sizeof(float);
    size_t route_bytes = (size_t)(2 * k) * sizeof(float);
    if (ctx->act_sizes[BN_GPU_BUF_XB] < input_bytes ||
        ctx->act_sizes[BN_GPU_BUF_MOE_HB2] < route_bytes)
        return -1;

    float route[BN_MAX_MOE_K * 2];
    if (k > BN_MAX_MOE_K)
        return -1;
    for (int token = 0; token < n_tokens; token++) {
        if (n_tokens == 1 && ctx->route_history_buf &&
            ctx->route_history_count < ctx->route_history_capacity &&
            ctx->route_history_stride == k + 1 &&
            ctx->route_history_layers > 0) {
            uint32_t *entry = (uint32_t *)[ctx->route_history_buf contents] +
                ctx->route_history_count * (size_t)ctx->route_history_stride;
            entry[0] = (uint32_t)(ctx->route_history_count %
                                  (size_t)ctx->route_history_layers);
            for (int slot = 0; slot < k; slot++)
                entry[slot + 1] = (uint32_t)indices[slot];
            ctx->route_history_count++;
        }
        for (int slot = 0; slot < k; slot++) {
            route[slot] = weights[(size_t)token * k + slot];
            route[k + slot] = (float)indices[(size_t)token * k + slot];
        }
        memcpy([ctx->act_bufs[BN_GPU_BUF_XB] contents],
               X + (size_t)token * dim, input_bytes);
        memcpy([ctx->act_bufs[BN_GPU_BUF_MOE_HB2] contents], route,
               route_bytes);

        BnGPUOp op = {0};
        op.op_kind = BN_GPU_OP_FFN;
        op.op_code = BN_GPU_CODE_MOE_ROUTED_FFN;
        op.type = gate_type;
        op.W_buf = gate;
        op.W_buf2 = up;
        op.W_buf3 = down;
        op.buf_in = BN_GPU_BUF_XB;
        op.buf_out = BN_GPU_BUF_MOE_OUT;
        op.buf_aux = BN_GPU_BUF_MOE_HB2;
        op.rows = hidden_dim;
        op.cols = dim;
        op.p[0] = (uint32_t)hidden_dim;
        op.p[1] = (uint32_t)n_experts;
        op.p[2] = (uint32_t)k;
        op.p[3] = (uint32_t)act_type;
        op.p[4] = BN_GPU_BUF_MOE_HB;
        op.p[6] = down_stride;
        op.p[7] = gate_stride;
        if (metal_execute(ctx, &op, 1, BN_GPU_BUF_MOE_OUT,
                          out + (size_t)token * dim, dim) != 0)
            return -1;
    }
    return 0;
}

/* ── Public API ────────────────────────────────────────────────────── */

BnGPUBackend *bn_gpu_metal_create(const char *shader_dir) {
    BnBackendRuntimePolicy policy;
    if (bn_gpu_backend_runtime_policy_init(&policy) != 0) return NULL;
    if (bn_gpu_policy_metal_apply_small_dense_native_quant_default(
            &policy) != 0) {
        bn_backend_runtime_policy_free(&policy);
        return NULL;
    }
    BnGPUBackend *gpu = bn_gpu_metal_create_with_policy(shader_dir, &policy);
    bn_backend_runtime_policy_free(&policy);
    return gpu;
}

BnGPUBackend *bn_gpu_metal_create_with_policy(
    const char *shader_dir, const BnBackendRuntimePolicy *runtime_policy)
{
    BnMetalCtx *ctx = (BnMetalCtx *)calloc(1, sizeof(BnMetalCtx));
    if (!ctx) return NULL;
    ctx->runtime_policy = runtime_policy;
    ctx->gpu_profile = -1;
    ctx->cpu_order_rmsnorm_enabled =
        bn_gpu_policy_metal_cpu_order_rmsnorm_enabled(runtime_policy);
    ctx->full_barriers_enabled =
        bn_gpu_policy_metal_full_barriers_enabled(runtime_policy);
    ctx->barriers_disabled =
        bn_gpu_policy_metal_barriers_disabled(runtime_policy);
    ctx->route_history_enabled =
        bn_gpu_policy_metal_route_history_enabled(runtime_policy);
    ctx->reference_attention_stage_mask =
        bn_gpu_policy_metal_reference_attention_stage_mask(runtime_policy);

    @autoreleasepool {
        /* Get default Metal device */
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            if ([devices count] > 0)
                ctx->device = devices[0];
        }
        if (!ctx->device) {
            fprintf(stderr, "[bn:gpu:metal] no Metal device found\n");
            free(ctx);
            return NULL;
        }

        fprintf(stderr, "[bn:gpu:metal] device: %s\n",
                [[ctx->device name] UTF8String]);

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "[bn:gpu:metal] failed to create command queue\n");
            free(ctx);
            return NULL;
        }

        /* Store shader directory */
        const char *dir = shader_dir ? shader_dir : "shaders/metal/";
        snprintf(ctx->shader_dir, sizeof(ctx->shader_dir), "%s", dir);

        /* Compile matvec pipelines for all supported quant types */
        int compiled = 0;
        int n_supported_types = bn_quant_format_gpu_shader_type_count(1);
        for (int i = 0; i < n_supported_types; i++) {
            int type = bn_quant_format_gpu_shader_type_at(i, 1);
            if (compile_matvec_pipeline(ctx, type, dir) == 0)
                compiled++;
        }
        fprintf(stderr, "[bn:gpu:metal] compiled %d/%d matvec pipelines\n",
                compiled, n_supported_types);

        ctx->small_dense_native_quant_enabled =
            bn_gpu_policy_metal_small_dense_native_quant_enabled(
                ctx->runtime_policy);
        ctx->native_quant_barriers_enabled =
            bn_gpu_policy_metal_native_quant_barriers_enabled(
                ctx->runtime_policy);

        ctx->specialized_native_quant_pipeline = compile_shader_with_math(
            ctx, dir, "q8k_quantize.metal", "q8k_quantize", 1);
        ctx->specialized_native_matvec_pipeline = compile_shader_with_math(
            ctx, dir, "q6k_q8k_matvec_reference.metal",
            "q6k_q8k_matvec", 1);
        ctx->specialized_native_four_row_matvec_pipeline = compile_shader(
            ctx, dir, "q6k_q8k_matvec.metal", "q6k_q8k_matvec");
        ctx->specialized_q4k_native_matvec_pipeline = compile_shader(
            ctx, dir, "q4k_q8k_matvec.metal", "q4k_q8k_matvec");
        ctx->specialized_q5k_native_matvec_pipeline = compile_shader(
            ctx, dir, "q5k_q8k_matvec.metal", "q5k_q8k_matvec");
        ctx->specialized_q4k_native_split_pipeline = compile_shader(
            ctx, dir, "q4k_q8k_matvec_split.metal",
            "q4k_q8k_matvec_split");
        ctx->reference_q6k_matvec_pipeline = compile_shader(
            ctx, dir, "kquant_matvec_reference.metal",
            "q6k_matvec_reference");
        ctx->reference_q5k_matvec_pipeline = compile_shader(
            ctx, dir, "kquant_matvec_reference.metal",
            "q5k_matvec_reference");
        ctx->reference_q4k_matvec_pipeline = compile_shader_with_math(
            ctx, dir, "q4k_matvec_reference.metal",
            "q4k_matvec_reference", 1);
        if (!ctx->specialized_native_quant_pipeline ||
            !ctx->specialized_native_matvec_pipeline ||
            !ctx->specialized_native_four_row_matvec_pipeline ||
            !ctx->specialized_q4k_native_matvec_pipeline ||
            !ctx->specialized_q5k_native_matvec_pipeline ||
            !ctx->specialized_q4k_native_split_pipeline ||
            !ctx->reference_q6k_matvec_pipeline ||
            !ctx->reference_q5k_matvec_pipeline ||
            !ctx->reference_q4k_matvec_pipeline) {
            fprintf(stderr, "[bn:gpu:metal] required native-quant pipelines "
                    "failed to compile\n");
            free(ctx);
            return NULL;
        }
        ctx->native_quant_pipeline = compile_shader_with_math(
            ctx, dir, "q8_quantize.metal", "q8_quantize", 1);
        ctx->argmax_pipeline = compile_shader(ctx, dir,
            "argmax.metal", "argmax_stage1");
        ctx->argmax_reduce_pipeline = compile_shader(ctx, dir,
            "argmax.metal", "argmax_stage2");
        ctx->q8_native_quant_matvec_pipeline = compile_shader(ctx, dir,
            "q8_prepared_q8_matvec.metal", "q8_prepared_q8_matvec");
        if (!ctx->native_quant_pipeline || !ctx->argmax_pipeline ||
            !ctx->argmax_reduce_pipeline ||
            !ctx->q8_native_quant_matvec_pipeline) {
            fprintf(stderr, "[bn:gpu:metal] required utility pipelines failed to compile\n");
            free(ctx);
            return NULL;
        }
        if (ctx->small_dense_native_quant_enabled) {
            ctx->small_dense_native_quant_matvec_pipeline = compile_shader(ctx, dir,
                "q4_native_prepared_q8_matvec.metal",
                "q4_native_prepared_q8_matvec");
            ctx->prepared_small_dense_native_quant_matvec_pipeline = compile_shader(ctx, dir,
                "q4_prepared_q8_matvec.metal",
                "q4_prepared_q8_matvec");
            ctx->prepared_reference_native_quant_matvec_pipeline =
                compile_shader_with_math(ctx, dir,
                    "q4_prepared_q8_matvec.metal",
                    "q4_prepared_q8_matvec_reference", 1);
            ctx->reference_native_quant_matvec_pipeline =
                compile_shader_with_math(ctx, dir,
                    "q4_native_prepared_q8_matvec.metal",
                    "q4_native_prepared_q8_matvec_reference", 1);
            ctx->prepared_f32_matvec_pipeline = compile_shader(
                ctx, dir, "q4_prepared_f32_matvec.metal",
                "q4_prepared_f32_matvec");
            ctx->prepared_f32_split_pipeline = compile_shader(
                ctx, dir, "q4_prepared_f32_split.metal",
                "q4_prepared_f32_split");
            ctx->prepared_small_dense_native_quant_split_pipeline = compile_shader(ctx, dir,
                "q4_prepared_q8_split.metal",
                "q4_prepared_q8_split");
            ctx->prepared_small_dense_native_quant_gateup_pipeline = compile_shader(ctx, dir,
                "q4_prepared_q8_gateup.metal",
                "q4_prepared_q8_gateup");
            ctx->small_dense_native_quant_split_pipeline = compile_shader(ctx, dir,
                "q4_matvec_split_prepared_q8.metal",
                "q4_matvec_split_prepared_q8");
            ctx->small_dense_native_quant_gateup_pipeline = compile_shader(ctx, dir,
                "q4_fused_gateup_silu_prepared_q8.metal",
                "q4_fused_gateup_silu_prepared_q8");
        }

        /* Build vtable */
        BnGPUBackend *gpu = (BnGPUBackend *)calloc(1, sizeof(BnGPUBackend));
        if (!gpu) {
            free(ctx);
            return NULL;
        }
        if (bn_gpu_backend_capture_runtime_policy_from(gpu,
                                                       runtime_policy) != 0) {
            free(ctx);
            free(gpu);
            return NULL;
        }
        ctx->runtime_policy = &gpu->runtime_policy;
        gpu->buffer_create        = metal_buffer_create;
        gpu->buffer_create_quant_only = metal_buffer_create_quant_only;
        gpu->buffer_create_borrowed = metal_buffer_create_borrowed;
        gpu->buffer_create_native_matvec_borrowed =
            metal_buffer_create_native_matvec_borrowed;
        gpu->native_matvec_borrowed_supported =
            metal_native_matvec_borrowed_supported;
        gpu->buffer_cache_charge = metal_buffer_cache_charge;
        gpu->buffer_create_biased = metal_buffer_create_biased;
        gpu->buffer_create_stacked2 = metal_buffer_create_stacked2;
        gpu->buffer_create_stacked3 = metal_buffer_create_stacked3;
        gpu->buffer_create_stacked3_biased = metal_buffer_create_stacked3_biased;
        gpu->buffer_destroy       = metal_buffer_destroy;
        gpu->matvec               = metal_matvec;
        gpu->matmul               = metal_matmul;
        gpu->matvec_batch         = metal_matvec_batch;
        gpu->execute              = metal_execute;
        gpu->init_activations     = metal_init_activations;
        gpu->reset_activations    = metal_reset_activations;
        gpu->free_activations     = metal_free_activations;
        gpu->write_activation     = metal_write_activation;
        gpu->read_activation      = metal_read_activation;
        gpu->argmax_activation    = metal_argmax_activation;
        gpu->memory_info          = metal_memory_info;
        gpu->prepare_cpu_operations = metal_prepare_cpu_operations;
        if (bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
                ctx->runtime_policy))
            gpu->moe_routed_ffn_batch = metal_moe_routed_ffn_batch;
        gpu->configure_prepared_native_quant =
            metal_configure_prepared_native_quant;
        gpu->ctx                  = ctx;
        gpu->max_storage_binding_size = (size_t)[ctx->device maxBufferLength];
        gpu->caps                 = BN_GPU_CAP_FLASH_ATTN |
                                    BN_GPU_CAP_LAYERWISE_ROPE |
                                    BN_GPU_CAP_PER_LAYER_INPUT_GRAPH |
                                    BN_GPU_CAP_LARGE_GRAPH_NATIVE |
                                    BN_GPU_CAP_SSM_GRAPH |
                                    BN_GPU_CAP_HYBRID_SSM_MOE_GRAPH |
                                    BN_GPU_CAP_REFERENCE_RECURRENT |
                                    BN_GPU_CAP_REFERENCE_ATTENTION |
                                    BN_GPU_CAP_REFERENCE_ATTENTION_FALLBACK |
                                    BN_GPU_CAP_PREPARED_NATIVE_QUANT |
                                    BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN |
                                    BN_GPU_CAP_PREPARED_NATIVE_QUANT_PER_LAYER_FFN_DOWN |
                                    BN_GPU_CAP_DENSE_RESIDUAL_LOWBIT_BLOCK32 |
                                    BN_GPU_CAP_MOE_ROUTED_LOWBIT_BLOCK32 |
                                    BN_GPU_CAP_LOWBIT_BLOCK32_MATVEC_SPLIT |
                                    BN_GPU_CAP_ASYMMETRIC_KQUANT_MATVEC_SPLIT |
                                    BN_GPU_CAP_LOWBIT_BLOCK32_FUSED_GATEUP_SILU;
        if (bn_gpu_policy_metal_reference_attention_enabled(
                ctx->runtime_policy))
            gpu->caps |= BN_GPU_CAP_REFERENCE_ATTENTION_NATIVE_GRAPH;
        if (bn_gpu_policy_metal_prepared_native_quant_attention_enabled(
                ctx->runtime_policy))
            gpu->caps |= BN_GPU_CAP_PREPARED_NATIVE_QUANT_ATTENTION;
        if (bn_gpu_policy_metal_routed_moe_decode_enabled(
                ctx->runtime_policy) ||
            bn_gpu_policy_metal_cpu_route_resident_moe_enabled(
                ctx->runtime_policy))
            gpu->caps |= BN_GPU_CAP_MOE_ROUTED_FFN;
        if (bn_gpu_policy_metal_moe_expert_graph_enabled(
                ctx->runtime_policy))
            gpu->caps |= BN_GPU_CAP_MOE_EXPERT_GRAPH;
        gpu->kind                 = BN_GPU_BACKEND_METAL;
        gpu->max_moe_route_experts = BN_METAL_MAX_MOE_ROUTE_EXPERTS;

        return gpu;
    }
}

void bn_gpu_metal_destroy(BnGPUBackend *gpu)
{
    if (!gpu) return;

    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    if (ctx) {
        metal_report_route_history(ctx);
        metal_free_activations(ctx);

        /* Release matvec pipelines */
        for (int i = 0; i < BN_METAL_MAX_TYPES; i++)
            ctx->pipelines[i] = nil;

        ctx->moe_routed_q4_0_gateup_pipeline = nil;
        ctx->moe_routed_q4_0_down_pipeline = nil;
        ctx->moe_routed_q4k_down_pipeline = nil;
        ctx->moe_routed_q5k_down_pipeline = nil;
        ctx->moe_routed_q6k_down_pipeline = nil;
        ctx->moe_routed_q8k_quant_pipeline = nil;
        ctx->moe_route_logits_pipeline = nil;
        ctx->moe_route_capture_pipeline = nil;
        ctx->native_quant_pipeline = nil;
        ctx->q8_native_quant_matvec_pipeline = nil;
        ctx->specialized_native_quant_pipeline = nil;
        ctx->reference_q6k_matvec_pipeline = nil;
        ctx->reference_q5k_matvec_pipeline = nil;
        ctx->reference_q4k_matvec_pipeline = nil;
        ctx->small_dense_native_quant_matvec_pipeline = nil;
        ctx->prepared_small_dense_native_quant_matvec_pipeline = nil;
        ctx->prepared_reference_native_quant_matvec_pipeline = nil;
        ctx->reference_native_quant_matvec_pipeline = nil;
        ctx->prepared_f32_matvec_pipeline = nil;
        ctx->prepared_f32_split_pipeline = nil;
        ctx->prepared_small_dense_native_quant_split_pipeline = nil;
        ctx->prepared_small_dense_native_quant_gateup_pipeline = nil;
        ctx->small_dense_native_quant_split_pipeline = nil;
        ctx->small_dense_native_quant_gateup_pipeline = nil;
        ctx->cpu_order_rmsnorm_pipeline = nil;
        ctx->specialized_native_matvec_pipeline = nil;
        ctx->specialized_native_four_row_matvec_pipeline = nil;
        ctx->specialized_q4k_native_matvec_pipeline = nil;
        ctx->specialized_q5k_native_matvec_pipeline = nil;
        ctx->specialized_q4k_native_split_pipeline = nil;
        ctx->argmax_pipeline = nil;
        ctx->argmax_reduce_pipeline = nil;

        ctx->x_buf = nil;
        ctx->out_buf = nil;
        ctx->native_quant_buf = nil;
        ctx->native_quant_scales_buf = nil;
        ctx->native_quant_block_sums_buf = nil;
        ctx->argmax_result_buf = nil;
        ctx->argmax_partials_buf = nil;
        ctx->argmax_penalty_buf = nil;
        ctx->route_history_buf = nil;

        /* Release slab */
        for (int i = 0; i < ctx->moe_resident_count; i++) {
            ctx->moe_resident[i].gate.buf = nil;
            ctx->moe_resident[i].up.buf = nil;
            ctx->moe_resident[i].down.buf = nil;
            free(ctx->moe_resident[i].experts);
            free(ctx->moe_resident[i].ages);
        }
        free(ctx->moe_resident);
        ctx->slab_buf = nil;
        metal_release_mmap_residency(ctx);
        for (int i = 0; i < ctx->mmap_buf_count; i++)
            ctx->mmap_bufs[i] = nil;
        ctx->mmap_buf_count = 0;
        free(ctx->slab_free);

        ctx->queue = nil;
        ctx->device = nil;

        free(ctx);
    }
    bn_gpu_backend_release_runtime_policy(gpu);
    free(gpu);
}

int bn_gpu_metal_init_slab(BnGPUBackend *gpu, size_t size_mb)
{
    if (!gpu || !gpu->ctx || size_mb == 0) return -1;
    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    if (size_mb > SIZE_MAX / (1024u * 1024u)) return -1;
    ctx->moe_resident_budget = size_mb * 1024u * 1024u;
    if (ctx->mmap_buf_count > 0)
        return 0;
    return slab_init(ctx, ctx->moe_resident_budget);
}

size_t bn_gpu_metal_recommended_slab_mb(const BnGPUBackend *gpu)
{
    if (!gpu || !gpu->ctx) return 0;
    const BnMetalCtx *ctx = (const BnMetalCtx *)gpu->ctx;
    size_t working_set =
        (size_t)ctx->device.recommendedMaxWorkingSetSize;
    size_t size_mb = working_set / (8u * 1024u * 1024u);
    if (size_mb < 512u) size_mb = 512u;
    if (size_mb > 4096u) size_mb = 4096u;
    return size_mb;
}

void bn_gpu_metal_set_mmap_range(BnGPUBackend *gpu, const void *base, size_t size)
{
    if (!gpu || !gpu->ctx) return;
    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    metal_release_mmap_residency(ctx);
    for (int i = 0; i < ctx->mmap_buf_count; i++)
        ctx->mmap_bufs[i] = nil;
    ctx->mmap_buf_count = 0;
    ctx->mmap_buf_offset = 0;
    ctx->mmap_fits_working_set = 0;
    ctx->mmap_prefaulted = 0;
    ctx->mmap_base = base;
    ctx->mmap_size = size;
    if (base && size > 0) {
        gpu->caps |= BN_GPU_CAP_BORROWED_WEIGHT_BUFFERS;
        if (bn_gpu_policy_metal_shared_mmap_buffer_enabled(
                ctx->runtime_policy)) {
            size_t page = (size_t)getpagesize();
            uintptr_t aligned_start = (uintptr_t)base & ~(page - 1);
            size_t prefix = (uintptr_t)base - aligned_start;
            if (prefix <= SIZE_MAX - size &&
                prefix + size <= SIZE_MAX - (page - 1)) {
                size_t aligned_size =
                    (prefix + size + page - 1) & ~(page - 1);
                size_t max_view = (size_t)[ctx->device maxBufferLength];
                size_t overlap = BN_METAL_MMAP_BUFFER_OVERLAP;
                if (overlap >= max_view) overlap = max_view / 8;
                size_t step = max_view - overlap;
                for (size_t start = 0;
                     start < aligned_size &&
                         ctx->mmap_buf_count < BN_METAL_MAX_MMAP_BUFFERS;
                     start += step) {
                    size_t view_size = aligned_size - start;
                    if (view_size > max_view) view_size = max_view;
            id<MTLBuffer> view = [ctx->device
                        newBufferWithBytesNoCopy:(void *)(aligned_start + start)
                                         length:view_size
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
                    if (!view) break;
                    int index = ctx->mmap_buf_count++;
                    ctx->mmap_bufs[index] = view;
                    ctx->mmap_buf_starts[index] = start;
                    ctx->mmap_buf_sizes[index] = view_size;
                }
                if (ctx->mmap_buf_count > 0 &&
                    ctx->mmap_buf_starts[ctx->mmap_buf_count - 1] +
                        ctx->mmap_buf_sizes[ctx->mmap_buf_count - 1] >=
                        aligned_size) {
                    ctx->mmap_buf_offset = prefix;
                    fprintf(stderr,
                            "[bn:gpu:metal] shared model views: count=%d "
                            "max=%.0f MB overlap=%.0f MB\n",
                            ctx->mmap_buf_count,
                            (double)max_view / (1024.0 * 1024.0),
                            (double)overlap / (1024.0 * 1024.0));
                    metal_create_mmap_residency_set(ctx);
                } else {
                    for (int i = 0; i < ctx->mmap_buf_count; i++)
                        ctx->mmap_bufs[i] = nil;
                    ctx->mmap_buf_count = 0;
                }
            }
        }
    }
}

#endif /* BN_ENABLE_METAL */
