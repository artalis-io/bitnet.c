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

#include "gpu_metal.h"
#include "gpu_backend.h"
#include "gpu_shader.h"
#include "model.h"
#include "quant.h"
#include "gguf.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* Max tensor type enum value we index into (I2_S = 36, plus margin) */
#define BN_METAL_MAX_TYPES 40

/* ── Internal context ──────────────────────────────────────────────── */

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pipelines[BN_METAL_MAX_TYPES];  /* matvec per quant type */
    id<MTLComputePipelineState> fwd_pipelines[BN_GPU_SHADER_COUNT]; /* forward-pass shaders */
    id<MTLComputePipelineState> q8_quant_pipeline;
    id<MTLComputePipelineState> q8k_quant_pipeline;
    id<MTLComputePipelineState> q4_q8_matvec_pipeline;
    id<MTLComputePipelineState> q4_q8_split_pipeline;
    id<MTLComputePipelineState> q4_q8_gateup_pipeline;
    id<MTLComputePipelineState> q6_q8k_matvec_pipeline;
    int q4_q8_enabled;

    /* GPU-resident activation buffers (storageModeShared) */
    id<MTLBuffer> act_bufs[BN_GPU_BUF_COUNT];
    size_t        act_sizes[BN_GPU_BUF_COUNT];

    /* Persistent scratch buffers for standalone matvec */
    id<MTLBuffer> x_buf;
    size_t        x_buf_size;
    id<MTLBuffer> out_buf;
    size_t        out_buf_size;
    id<MTLBuffer> q8_buf;
    size_t        q8_buf_size;
    id<MTLBuffer> q8_scales_buf;
    size_t        q8_scales_buf_size;
    id<MTLBuffer> q8_bsums_buf;
    size_t        q8_bsums_buf_size;

    /* Shader directory path */
    char shader_dir[256];

    /* Profiling */
    int gpu_frame;
    int gpu_profile;

    /* Slab allocator for MoE weight suballocation */
    id<MTLBuffer> slab_buf;
    size_t        slab_size;
    struct { size_t offset, size; } *slab_free;
    int           slab_free_count;
    int           slab_free_cap;

    /* Zero-copy mmap range (Phase 5) */
    const void   *mmap_base;
    size_t        mmap_size;
} BnMetalCtx;

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
    int           q4_repacked;
} BnMetalBuf;

/* ── Shader type name mapping (same as wgpu) ──────────────────────── */

static const char *shader_name_for_type(int type)
{
    switch (type) {
        case BN_GGUF_TENSOR_F32:     return "f32";
        case BN_GGUF_TENSOR_F16:     return "f16";
        case BN_GGUF_TENSOR_I2_S:    return "i2s";
        case BN_GGUF_TENSOR_TQ1_0:   return "tq1";
        case BN_GGUF_TENSOR_TQ2_0:   return "tq2";
        case BN_GGUF_TENSOR_Q4_0:    return "q4";
        case BN_GGUF_TENSOR_Q4_1:    return "q4_1";
        case BN_GGUF_TENSOR_Q8_0:    return "q8";
        case BN_GGUF_TENSOR_BF16:    return "bf16";
        case BN_GGUF_TENSOR_Q2_K:    return "q2k";
        case BN_GGUF_TENSOR_Q3_K:    return "q3k";
        case BN_GGUF_TENSOR_Q4_K:    return "q4k";
        case BN_GGUF_TENSOR_Q5_K:    return "q5k";
        case BN_GGUF_TENSOR_Q6_K:    return "q6k";
        case BN_GGUF_TENSOR_Q8_K:    return "q8k";
        case BN_GGUF_TENSOR_IQ4_NL:  return "iq4nl";
        case BN_GGUF_TENSOR_IQ4_XS:  return "iq4xs";
        case BN_GGUF_TENSOR_IQ3_XXS: return "iq3xxs";
        case BN_GGUF_TENSOR_IQ3_S:   return "iq3s";
        case BN_GGUF_TENSOR_IQ2_XXS: return "iq2xxs";
        case BN_GGUF_TENSOR_IQ2_XS:  return "iq2xs";
        case BN_GGUF_TENSOR_IQ2_S:   return "iq2s";
        default:                      return NULL;
    }
}

static const int supported_types[] = {
    BN_GGUF_TENSOR_I2_S, BN_GGUF_TENSOR_TQ1_0, BN_GGUF_TENSOR_TQ2_0,
    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_1, BN_GGUF_TENSOR_Q8_0,
    BN_GGUF_TENSOR_F16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_Q2_K, BN_GGUF_TENSOR_Q3_K,
    BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K,
    BN_GGUF_TENSOR_Q8_K, BN_GGUF_TENSOR_IQ4_NL, BN_GGUF_TENSOR_IQ4_XS,
    BN_GGUF_TENSOR_IQ3_XXS, BN_GGUF_TENSOR_IQ3_S, BN_GGUF_TENSOR_IQ2_XXS,
    BN_GGUF_TENSOR_IQ2_XS, BN_GGUF_TENSOR_IQ2_S,
};
#define N_SUPPORTED_TYPES ((int)(sizeof(supported_types) / sizeof(supported_types[0])))

/* ── Shader compilation ────────────────────────────────────────────── */

static id<MTLComputePipelineState> compile_shader(BnMetalCtx *ctx,
                                                   const char *dir,
                                                   const char *filename,
                                                   const char *fn_name)
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
        opts.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
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

static int compile_matvec_pipeline(BnMetalCtx *ctx, int type, const char *dir)
{
    const char *name = shader_name_for_type(type);
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

static BnMetalBuf *metal_repack_q4_0_for_gpu(BnMetalCtx *ctx,
                                             const void *data,
                                             size_t size,
                                             int rows,
                                             int cols,
                                             const float *bias,
                                             int bias_len)
{
    (void)size;
    if (!ctx || !data || rows <= 0 || cols <= 0 || (cols % 32) != 0)
        return NULL;

    int blocks_per_row = cols / 32;
    int n_blocks = rows * blocks_per_row;
    size_t base_size = (size_t)n_blocks * sizeof(float) +
                       (size_t)n_blocks * 4 * sizeof(uint32_t);
    size_t bias_bytes = (bias && bias_len > 0) ?
                        (size_t)bias_len * sizeof(float) : 0;
    size_t repacked_size = (base_size + bias_bytes + 3) & ~(size_t)3;

    uint8_t *repacked = (uint8_t *)calloc(1, repacked_size);
    if (!repacked) return NULL;

    float *scales = (float *)repacked;
    uint8_t *nibbles = repacked + (size_t)n_blocks * sizeof(float);
    const uint8_t *src = (const uint8_t *)data;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + (size_t)b * 18;
        uint16_t d_bits = (uint16_t)(block[0] | (block[1] << 8));
        scales[b] = bn_fp16_to_fp32(d_bits);

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
    buf->buf = [ctx->device newBufferWithBytes:repacked
                                        length:repacked_size
                                       options:MTLResourceStorageModeShared];
    free(repacked);
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = repacked_size;
    buf->offset = 0;
    buf->type = BN_GGUF_TENSOR_Q4_0;
    buf->rows = rows;
    buf->cols = cols;
    buf->bias_offset = bias_offset;
    buf->q4_repacked = 1;
    return buf;
}

static void *metal_buffer_create(void *vctx, const void *data, size_t size,
                                  int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0) return NULL;

    if (type == BN_GGUF_TENSOR_Q4_0)
        return metal_repack_q4_0_for_gpu(ctx, data, size, rows, cols, NULL, 0);

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

    /* Zero-copy: if data is within mmap range, wrap it without copying.
     * Requires page-aligned pointer. The mmap'd file stays alive for the
     * lifetime of the model, so the buffer is valid. */
    if (ctx->mmap_base && ctx->mmap_size > 0) {
        const uint8_t *base = (const uint8_t *)ctx->mmap_base;
        const uint8_t *ptr = (const uint8_t *)data;
        if (ptr >= base && ptr + size <= base + ctx->mmap_size) {
            /* Page-align: extend range to page boundaries */
            size_t page = (size_t)getpagesize();
            uintptr_t aligned_start = (uintptr_t)ptr & ~(page - 1);
            size_t prefix = (uintptr_t)ptr - aligned_start;
            size_t aligned_size = (prefix + size + page - 1) & ~(page - 1);
            buf->buf = [ctx->device newBufferWithBytesNoCopy:(void *)aligned_start
                                                      length:aligned_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
            if (buf->buf) {
                buf->offset = prefix;
                buf->size = size;
                buf->type = type;
                buf->rows = rows;
                buf->cols = cols;
                buf->is_slab = 0;
                return buf;
            }
            /* Fall through to copy path if NoCopy fails */
        }
    }

    /* Standalone buffer — storageModeShared for zero-copy on Apple Silicon */
    buf->buf = [ctx->device newBufferWithBytes:data
                                        length:size
                                       options:MTLResourceStorageModeShared];
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

static void *metal_buffer_create_biased(void *vctx, const void *data, size_t size,
                                         int type, int rows, int cols,
                                         const void *bias, size_t bias_size)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0 || !bias || bias_size == 0) return NULL;

    if (type == BN_GGUF_TENSOR_Q4_0) {
        int bias_len = (int)(bias_size / sizeof(float));
        return metal_repack_q4_0_for_gpu(ctx, data, size, rows, cols,
                                         (const float *)bias, bias_len);
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

static void metal_buffer_destroy(void *vctx, void *buffer)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *buf = (BnMetalBuf *)buffer;
    if (!buf) return;

    if (buf->is_slab && ctx) {
        slab_free_range(ctx, buf->offset, (buf->size + 255) & ~(size_t)255);
    }
    /* Standalone buffers: ARC releases when buf->buf goes out of scope */
    /* (Under ARC, setting to nil or letting it deallocate handles release) */
    free(buf);
}

/* ── Vtable: init_activations ──────────────────────────────────────── */

static void metal_free_activations(void *vctx);  /* forward decl */

static int metal_init_activations(void *vctx, const void *config_ptr)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    const BnConfig *c = (const BnConfig *)config_ptr;
    if (!ctx || !c) return -1;

    /* Compute buffer sizes (same logic as wgpu) */
    int n_attn = (c->full_attn_interval > 0)
                     ? c->n_layers / c->full_attn_interval
                     : c->n_layers;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;

    size_t sizes[BN_GPU_BUF_COUNT] = {0};
    sizes[BN_GPU_BUF_X]           = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_BUF_XB]          = (size_t)xb_size * sizeof(float);
    sizes[BN_GPU_BUF_XB2]         = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_BUF_Q]           = (size_t)q_dim * sizeof(float);
    {
        int hb_dim = c->hidden_dim;
        if (c->moe_intermediate_size > hb_dim) hb_dim = c->moe_intermediate_size;
        sizes[BN_GPU_BUF_HB]  = (size_t)hb_dim * sizeof(float);
        sizes[BN_GPU_BUF_HB2] = (size_t)hb_dim * sizeof(float);
    }
    sizes[BN_GPU_BUF_KEY_CACHE]   = (size_t)n_attn * c->seq_len * c->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_VALUE_CACHE] = (size_t)n_attn * c->seq_len * c->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_ATT]         = (size_t)c->n_heads * c->seq_len * sizeof(float);
    sizes[BN_GPU_BUF_LOGITS]      = (size_t)c->vocab_size * sizeof(float);
    sizes[BN_GPU_BUF_ROPE_FREQ]   = (size_t)(c->head_size / 2) * sizeof(float);
    sizes[BN_GPU_BUF_SCRATCH]     = (size_t)xb_size * sizeof(float);
    {
        size_t qkv_size = (size_t)(q_dim + 2 * c->kv_dim) * sizeof(float);
        size_t gated_q_size = (size_t)(2 * q_dim) * sizeof(float);
        sizes[BN_GPU_BUF_QKV] = qkv_size > gated_q_size ? qkv_size : gated_q_size;
    }

    if (c->moe_intermediate_size > 0) {
        sizes[BN_GPU_BUF_MOE_HB]  = (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_BUF_MOE_HB2] = (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_BUF_MOE_OUT] = (size_t)c->dim * sizeof(float);
    }

    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
        int n_ssm = c->n_layers - n_attn;
        int num_v_heads = c->ssm_time_step_rank;
        int head_k_dim  = c->ssm_state_size;
        int head_v_dim  = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
        int key_dim     = c->ssm_group_count * head_k_dim;
        int value_dim   = c->ssm_inner_size;
        int qkv_dim     = key_dim * 2 + value_dim;
        int kern        = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;

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
        int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
        int half = rope_dims / 2;
        float *freq = (float *)malloc((size_t)half * sizeof(float));
        if (!freq) return -1;
        for (int i = 0; i < half; i++)
            freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)rope_dims);
        memcpy([ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] contents], freq,
               (size_t)half * sizeof(float));
        free(freq);
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
        { BN_GPU_SHADER_RESIDUAL_ADD,     "residual_add.metal",     "residual_add"     },
        { BN_GPU_SHADER_BIAS_ADD,         "bias_add.metal",         "bias_add"         },
        { BN_GPU_SHADER_RESIDUAL_RMSNORM, "residual_rmsnorm.metal", "residual_rmsnorm" },
        { BN_GPU_SHADER_WEIGHTED_ADD,     "weighted_add.metal",     "weighted_add"     },
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
        id<MTLComputePipelineState> pso = compile_shader(ctx, ctx->shader_dir,
                                                          fwd_shaders[i].file,
                                                          fwd_shaders[i].fn);
        if (pso) {
            ctx->fwd_pipelines[fwd_shaders[i].id] = pso;
            compiled++;
        }
    }
    fprintf(stderr, "[bn:gpu:metal] compiled %d/%d forward-pass shaders\n",
            compiled, n_fwd);

    return 0;
}

static void metal_free_activations(void *vctx)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx) return;
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        ctx->act_bufs[i] = nil;
        ctx->act_sizes[i] = 0;
    }
    for (int i = 0; i < BN_GPU_SHADER_COUNT; i++)
        ctx->fwd_pipelines[i] = nil;
}

/* ── Vtable: write/read activation ─────────────────────────────────── */

static int metal_write_activation(void *vctx, int buf_idx, const void *data,
                                   size_t size, size_t offset)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    /* Unified memory: direct memcpy */
    memcpy((uint8_t *)[ctx->act_bufs[buf_idx] contents] + offset, data, size);
    return 0;
}

static int metal_read_activation(void *vctx, int buf_idx, void *out,
                                  size_t size, size_t offset)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !out || buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    memcpy(out, (uint8_t *)[ctx->act_bufs[buf_idx] contents] + offset, size);
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

static int ensure_q8_scratch(BnMetalCtx *ctx, int cols, int n_tokens)
{
    size_t q8_need = (size_t)cols * (size_t)n_tokens * sizeof(int8_t);
    size_t scales_need = (size_t)(cols >> 5) * (size_t)n_tokens * sizeof(float);
    if (!ctx->q8_buf || ctx->q8_buf_size < q8_need) {
        ctx->q8_buf = [ctx->device newBufferWithLength:q8_need
                                                options:MTLResourceStorageModeShared];
        if (!ctx->q8_buf) return -1;
        ctx->q8_buf_size = q8_need;
    }
    if (!ctx->q8_scales_buf || ctx->q8_scales_buf_size < scales_need) {
        ctx->q8_scales_buf = [ctx->device newBufferWithLength:scales_need
                                                      options:MTLResourceStorageModeShared];
        if (!ctx->q8_scales_buf) return -1;
        ctx->q8_scales_buf_size = scales_need;
    }
    return 0;
}

static int ensure_q8k_scratch(BnMetalCtx *ctx, int cols, int n_tokens)
{
    size_t q8_need = (size_t)cols * (size_t)n_tokens * sizeof(int8_t);
    size_t n_blocks = (size_t)(cols >> 8) * (size_t)n_tokens;
    size_t scales_need = n_blocks * sizeof(float);
    size_t bsums_need = n_blocks * 16 * sizeof(int16_t);
    if (!ctx->q8_buf || ctx->q8_buf_size < q8_need) {
        ctx->q8_buf = [ctx->device newBufferWithLength:q8_need
                                                options:MTLResourceStorageModeShared];
        if (!ctx->q8_buf) return -1;
        ctx->q8_buf_size = q8_need;
    }
    if (!ctx->q8_scales_buf || ctx->q8_scales_buf_size < scales_need) {
        ctx->q8_scales_buf = [ctx->device newBufferWithLength:scales_need
                                                      options:MTLResourceStorageModeShared];
        if (!ctx->q8_scales_buf) return -1;
        ctx->q8_scales_buf_size = scales_need;
    }
    if (!ctx->q8_bsums_buf || ctx->q8_bsums_buf_size < bsums_need) {
        ctx->q8_bsums_buf = [ctx->device newBufferWithLength:bsums_need
                                                     options:MTLResourceStorageModeShared];
        if (!ctx->q8_bsums_buf) return -1;
        ctx->q8_bsums_buf_size = bsums_need;
    }
    return 0;
}

static void metal_encode_q8_quant(id<MTLComputeCommandEncoder> enc,
                                  BnMetalCtx *ctx,
                                  id<MTLBuffer> x_buf,
                                  uint32_t cols,
                                  uint32_t n_tokens)
{
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    [enc setComputePipelineState:ctx->q8_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];
    MTLSize tpg = MTLSizeMake(1, 1, 1);
    MTLSize grid = MTLSizeMake((cols + 31) / 32, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc memoryBarrierWithResources:&ctx->q8_buf count:1];
    id<MTLBuffer> scale_buf = ctx->q8_scales_buf;
    [enc memoryBarrierWithResources:&scale_buf count:1];
}

static void metal_encode_q8k_quant(id<MTLComputeCommandEncoder> enc,
                                   BnMetalCtx *ctx,
                                   id<MTLBuffer> x_buf,
                                   uint32_t cols,
                                   uint32_t n_tokens)
{
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    [enc setComputePipelineState:ctx->q8k_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->q8_bsums_buf offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(params) atIndex:4];
    MTLSize tpg = MTLSizeMake(1, 1, 1);
    MTLSize grid = MTLSizeMake(cols / 256, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc memoryBarrierWithResources:&ctx->q8_buf count:1];
    id<MTLBuffer> bufs[2] = { ctx->q8_scales_buf, ctx->q8_bsums_buf };
    [enc memoryBarrierWithResources:bufs count:2];
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
    int use_q4_q8 = ctx->q4_q8_enabled && type == BN_GGUF_TENSOR_Q4_0 &&
                    ctx->q8_quant_pipeline && ctx->q4_q8_matvec_pipeline;
    int use_q6_q8k = type == BN_GGUF_TENSOR_Q6_K &&
                     ctx->q8k_quant_pipeline && ctx->q6_q8k_matvec_pipeline &&
                     (cols % 256) == 0;
    if (use_q4_q8 && ensure_q8_scratch(ctx, cols, 1) != 0) return -1;
    if (use_q6_q8k && ensure_q8k_scratch(ctx, cols, 1) != 0) return -1;

    memcpy([ctx->x_buf contents], x, x_size);

    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols, 1, 0, 0, 0, 0, 0 };
    if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;

    uint32_t tile_rows = 32;
    uint32_t wg_x = ((uint32_t)rows + tile_rows - 1) / tile_rows;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        if (use_q4_q8) {
            metal_encode_q8_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->q4_q8_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
        } else if (use_q6_q8k) {
            metal_encode_q8k_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->q6_q8k_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->q8_bsums_buf offset:0 atIndex:3];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:4];
            [enc setBytes:params length:sizeof(params) atIndex:5];
        } else {
            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
        }

        MTLSize tpg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(wg_x, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
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

    size_t x_size = (size_t)n_tokens * cols * sizeof(float);
    size_t out_size = (size_t)n_tokens * rows * sizeof(float);
    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;

    memcpy([ctx->x_buf contents], X, x_size);

    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols, (uint32_t)n_tokens, 0, 0, 0, 0, 0 };

    uint32_t tile_rows = 32;
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
    for (int i = 0; i < n_ops; i++)
        if (ops[i].rows > max_rows) max_rows = ops[i].rows;
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

            uint32_t tile_rows = 32;
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
        uint32_t tile_rows = 32;
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

    double t0 = bn_platform_time_ms();
    double t_encode = 0, t_gpu = 0;
    int n_barriers = 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;

        /* Dependency tracking: only insert barriers on actual RAW/WAR/WAW conflicts.
         * Same logic as wgpu execute — track read/write buffer masks since last barrier. */
        uint32_t since_barrier_writes = 0;
        id<MTLComputePipelineState> current_pso = nil;

        for (int i = 0; i < n_ops; i++) {
            const BnGPUOp *op = &ops[i];
            int shader = bn_gpu_shader_from_op_code(op->op_code);

            /* COPY as compute shader — stays in compute encoder, no blit transitions */

            /* Determine pipeline */
            id<MTLComputePipelineState> pipeline = nil;
            if (shader == BN_GPU_SHADER_MATVEC) {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (ctx->q4_q8_enabled && op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf &&
                    ctx->q8_quant_pipeline && ctx->q4_q8_matvec_pipeline) {
                    pipeline = ctx->q4_q8_matvec_pipeline;
                } else if (op->type >= 0 && op->type < BN_METAL_MAX_TYPES) {
                    pipeline = ctx->pipelines[op->type];
                }
            } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                       ctx->q4_q8_enabled &&
                       op->p[6] &&
                       op->type == BN_GGUF_TENSOR_Q4_0 &&
                       op->W_buf &&
                       ctx->q8_quant_pipeline &&
                       ctx->q4_q8_gateup_pipeline) {
                pipeline = ctx->q4_q8_gateup_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       ctx->q4_q8_enabled &&
                       op->type == BN_GGUF_TENSOR_Q4_0 &&
                       ctx->q8_quant_pipeline &&
                       ctx->q4_q8_split_pipeline) {
                pipeline = ctx->q4_q8_split_pipeline;
            } else if (shader > 0 && shader < BN_GPU_SHADER_COUNT) {
                pipeline = ctx->fwd_pipelines[shader];
            }
            if (!pipeline) continue;

            /* Compute this op's read/write buffer masks. */
            uint32_t op_reads = 0, op_writes = 0;
            if (bn_gpu_shader_access_masks(op, shader, &op_reads,
                                           &op_writes) != 0)
                continue;

            /* Insert barrier only on RAW conflict (read-after-write).
             * WAR and WAW don't need barriers — Metal dispatches execute in
             * submission order within a compute command encoder, so reads
             * always complete before subsequent writes to the same buffer. */
            int conflict = (op_reads & since_barrier_writes);
            if (conflict && enc) {
                /* Use resource-specific barriers for less stalling */
                /* Collect MTLBuffer pointers for written buffers that this op reads */
                id<MTLBuffer> barrier_bufs[BN_GPU_BUF_COUNT];
                int n_bbuf = 0;
                for (int b = 0; b < BN_GPU_BUF_COUNT && b < 23; b++) {
                    if ((since_barrier_writes & (1u << b)) &&
                        ((op_reads | op_writes) & (1u << b)) &&
                        ctx->act_bufs[b]) {
                        barrier_bufs[n_bbuf++] = ctx->act_bufs[b];
                    }
                }
                if (n_bbuf > 0)
                    [enc memoryBarrierWithResources:barrier_bufs count:(NSUInteger)n_bbuf];
                else
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                since_barrier_writes = 0;
                n_barriers++;
            }

            since_barrier_writes |= op_writes;

            /* Start compute encoder if needed */
            if (!enc) {
                enc = [cmd computeCommandEncoder];
                current_pso = nil;
            }
            /* Skip redundant PSO switch — avoids GPU instruction cache flush */
            if (pipeline != current_pso) {
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

            switch (shader) {
            case BN_GPU_SHADER_MATVEC: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                if (ctx->q4_q8_enabled && op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    ctx->q8_quant_pipeline && ctx->q4_q8_matvec_pipeline) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_q8_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        return -1;
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, n_tokens);
                    [enc setComputePipelineState:ctx->q4_q8_matvec_pipeline];
                    current_pso = ctx->q4_q8_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->type == BN_GGUF_TENSOR_Q6_K &&
                           ctx->q8k_quant_pipeline &&
                           ctx->q6_q8k_matvec_pipeline &&
                           (op->cols % 256) == 0) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_q8k_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        return -1;
                    metal_encode_q8k_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                           (uint32_t)op->cols, n_tokens);
                    [enc setComputePipelineState:ctx->q6_q8k_matvec_pipeline];
                    current_pso = ctx->q6_q8k_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->q8_bsums_buf offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:4];
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                } else {
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                }
                break;
            }
            case BN_GPU_SHADER_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
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
            case BN_GPU_SHADER_RELU2_GATE: {
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
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
                if (ctx->q4_q8_enabled &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_q8_split_pipeline) {
                    if (ensure_q8_scratch(ctx, op->cols, 1) != 0)
                        return -1;
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    [enc setComputePipelineState:ctx->q4_q8_split_pipeline];
                    current_pso = ctx->q4_q8_split_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[op->rows] offset:0 atIndex:5];
                    [enc setBytes:params length:sizeof(params) atIndex:6];
                } else {
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];  // out0
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];  // out1
                    [enc setBuffer:ctx->act_bufs[op->rows] offset:0 atIndex:4];     // out2
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
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
                if (ctx->q4_q8_enabled &&
                    op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_q8_gateup_pipeline) {
                    if (ensure_q8_scratch(ctx, op->cols, 1) != 0)
                        return -1;
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    [enc setComputePipelineState:ctx->q4_q8_gateup_pipeline];
                    current_pso = ctx->q4_q8_gateup_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else {
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
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            default: continue;
            }

            /* Compute workgroup count (same logic as wgpu) */
            uint32_t wg_x = 1, wg_y = 1;
            switch (shader) {
            case BN_GPU_SHADER_MATVEC: {
                if (op->p[3] > 0) {
                    uint32_t tiled_rows = ((uint32_t)op->rows + 31) / 32;
                    wg_x = op->p[3];
                    wg_y = (tiled_rows + op->p[3] - 1) / op->p[3];
                } else {
                    wg_x = ((uint32_t)op->rows + 31) / 32;
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
            case BN_GPU_SHADER_SOFTMAX:
            case BN_GPU_SHADER_GQA_COMBINE:
                wg_x = op->p[0];
                break;
            case BN_GPU_SHADER_GQA_SCORES:
                wg_x = op->p[0];
                wg_y = (op->p[2] + 7) / 8;
                break;
            case BN_GPU_SHADER_SILU_GATE:
            case BN_GPU_SHADER_RELU2_GATE:
            case BN_GPU_SHADER_RESIDUAL_ADD:
            case BN_GPU_SHADER_BIAS_ADD:
            case BN_GPU_SHADER_WEIGHTED_ADD:
            case BN_GPU_SHADER_SSM_CONV_SILU:
                wg_x = (op->p[0] + 255) / 256;
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
                wg_x = (op->p[0] + 7) / 8;  /* 8 heads per threadgroup */
                break;
            case BN_GPU_SHADER_COPY:
                wg_x = (op->p[2] + 255) / 256;
                break;
            case BN_GPU_SHADER_MATVEC_SPLIT:
                wg_x = (op->p[0] + 31) / 32;  // total_rows / 32
                break;
            case BN_GPU_SHADER_ROPE_QK:
                wg_x = op->p[0] + op->p[4];   // n_q_heads + n_kv_heads
                break;
            case BN_GPU_SHADER_FUSED_GATEUP_SILU:
                wg_x = (op->p[2] + 31) / 32;  // gate_rows / 32
                break;
            case BN_GPU_SHADER_Q4K_MATVEC_SPLIT:
                wg_x = (op->p[0] + 31) / 32;  // total_rows / 32
                break;
            }

            if (wg_x == 0) wg_x = 1;
            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, wg_y, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        }

        if (enc) [enc endEncoding];

        t_encode = bn_platform_time_ms();

        [cmd commit];
        [cmd waitUntilCompleted];

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
    if (ctx->gpu_profile < 0) {
        const char *env = getenv("BN_GPU_PROFILE");
        ctx->gpu_profile = env ? atoi(env) : 0;
    }
    if (ctx->gpu_profile >= 1 && (ctx->gpu_frame < 5 || (ctx->gpu_frame % 50 == 0))) {
        fprintf(stderr, "[gpu:metal:profile] frame=%d ops=%d barriers=%d encode=%.1fms gpu=%.1fms readback=%.1fms total=%.1fms\n",
                ctx->gpu_frame, n_ops, n_barriers,
                t_encode - t0, t_gpu - t_encode, t1 - t_gpu, t1 - t0);
    }
    /* Per-op-type breakdown (BN_GPU_PROFILE>=2, frame 1 only) */
    if (ctx->gpu_profile >= 2 && ctx->gpu_frame == 1) {
        /* Re-execute ops by category, timing each category separately */
        static const char *cat_names[] = {
            "matvec","rmsnorm","rope","gqa_scores","softmax","gqa_combine",
            "silu_gate","relu2_gate","resid_add","copy","bias_add","resid_rmsnorm",
            "weighted_add","ssm_conv","ssm_l2norm","ssm_ab","ssm_delta","ssm_gate",
            "per_head_norm","deinterleave_q","sigmoid_gate","flash_attn",
            "matvec_split","rope_qk","fused_gateup","ssm_ab_split","q4k_split"
        };
        int cat_count[BN_GPU_SHADER_COUNT]; memset(cat_count, 0, sizeof(cat_count));
        for (int i = 0; i < n_ops; i++) {
            int s = bn_gpu_shader_from_op_code(ops[i].op_code);
            if (s >= 0 && s < BN_GPU_SHADER_COUNT) cat_count[s]++;
        }
        fprintf(stderr, "[gpu:metal:breakdown] --- op counts ---\n");
        for (int s = 0; s < BN_GPU_SHADER_COUNT; s++) {
            if (cat_count[s] > 0)
                fprintf(stderr, "  %-16s: %3d ops\n",
                        s < (int)(sizeof(cat_names)/sizeof(cat_names[0])) ? cat_names[s] : "?", cat_count[s]);
        }
    }
    ctx->gpu_frame++;

    return 0;
}

/* ── Public API ────────────────────────────────────────────────────── */

BnGPUBackend *bn_gpu_metal_create(const char *shader_dir)
{
    BnMetalCtx *ctx = (BnMetalCtx *)calloc(1, sizeof(BnMetalCtx));
    if (!ctx) return NULL;
    ctx->gpu_profile = -1;

    @autoreleasepool {
        /* Get default Metal device */
        ctx->device = MTLCreateSystemDefaultDevice();
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
        for (int i = 0; i < N_SUPPORTED_TYPES; i++) {
            int type = supported_types[i];
            if (compile_matvec_pipeline(ctx, type, dir) == 0)
                compiled++;
        }
        fprintf(stderr, "[bn:gpu:metal] compiled %d/%d matvec pipelines\n",
                compiled, N_SUPPORTED_TYPES);

	        ctx->q4_q8_enabled = getenv("BN_GPU_Q4_Q8") ? 1 : 0;
	        ctx->q8k_quant_pipeline = compile_shader(ctx, dir,
	            "q8k_quantize.metal", "q8k_quantize");
	        ctx->q6_q8k_matvec_pipeline = compile_shader(ctx, dir,
	            "q6k_q8k_matvec.metal", "q6k_q8k_matvec");
	        if (ctx->q4_q8_enabled) {
            ctx->q8_quant_pipeline = compile_shader(ctx, dir,
                "q8_quantize.metal", "q8_quantize");
            ctx->q4_q8_matvec_pipeline = compile_shader(ctx, dir,
                "q4_native_q8_prequant_matvec.metal",
                "q4_native_q8_prequant_matvec");
            ctx->q4_q8_split_pipeline = compile_shader(ctx, dir,
                "q4_matvec_split_q8_prequant.metal",
                "q4_matvec_split_q8_prequant");
            ctx->q4_q8_gateup_pipeline = compile_shader(ctx, dir,
                "q4_fused_gateup_silu_q8_prequant.metal",
                "q4_fused_gateup_silu_q8_prequant");
        }

        /* Build vtable */
        BnGPUBackend *gpu = (BnGPUBackend *)calloc(1, sizeof(BnGPUBackend));
        if (!gpu) {
            free(ctx);
            return NULL;
        }
        gpu->buffer_create        = metal_buffer_create;
        gpu->buffer_create_biased = metal_buffer_create_biased;
        gpu->buffer_create_stacked2 = metal_buffer_create_stacked2;
        gpu->buffer_destroy       = metal_buffer_destroy;
        gpu->matvec               = metal_matvec;
        gpu->matmul               = metal_matmul;
        gpu->matvec_batch         = metal_matvec_batch;
        gpu->execute              = metal_execute;
        gpu->init_activations     = metal_init_activations;
        gpu->free_activations     = metal_free_activations;
        gpu->write_activation     = metal_write_activation;
        gpu->read_activation      = metal_read_activation;
        gpu->ctx                  = ctx;
        gpu->caps                 = BN_GPU_CAP_FLASH_ATTN |
                                    BN_GPU_CAP_Q4_MATVEC_SPLIT |
                                    BN_GPU_CAP_Q4K_MATVEC_SPLIT |
                                    BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        gpu->kind                 = BN_GPU_BACKEND_METAL;

        return gpu;
    }
}

void bn_gpu_metal_destroy(BnGPUBackend *gpu)
{
    if (!gpu) return;

    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    if (ctx) {
        metal_free_activations(ctx);

        /* Release matvec pipelines */
        for (int i = 0; i < BN_METAL_MAX_TYPES; i++)
            ctx->pipelines[i] = nil;

        ctx->x_buf = nil;
        ctx->out_buf = nil;

        /* Release slab */
        ctx->slab_buf = nil;
        free(ctx->slab_free);

        ctx->queue = nil;
        ctx->device = nil;

        free(ctx);
    }
    free(gpu);
}

int bn_gpu_metal_init_slab(BnGPUBackend *gpu, size_t size_mb)
{
    if (!gpu || !gpu->ctx || size_mb == 0) return -1;
    return slab_init((BnMetalCtx *)gpu->ctx, size_mb * 1024 * 1024);
}

void bn_gpu_metal_set_mmap_range(BnGPUBackend *gpu, const void *base, size_t size)
{
    if (!gpu || !gpu->ctx) return;
    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    ctx->mmap_base = base;
    ctx->mmap_size = size;
}

#endif /* BN_ENABLE_METAL */
