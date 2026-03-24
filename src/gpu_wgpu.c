/*
 * gpu_wgpu.c — wgpu-native backend for BnGPUBackend
 *
 * Implements the BnGPUBackend vtable using wgpu-native v27.
 * Compiles WGSL compute shaders (one per quant type), dispatches
 * matvec/matmul on GPU with staging buffer readback.
 */

#ifdef BN_ENABLE_GPU

#include "gpu_wgpu.h"
#include "gpu_backend.h"
#include "quant.h"
#include "gguf.h"
#include "webgpu.h"
#include "wgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Max tensor type enum value we index into (I2_S = 36, plus margin) */
#define BN_WGPU_MAX_TYPES 40

/* ── Helper: C string -> WGPUStringView ────────────────────────────── */

static WGPUStringView sv(const char *s)
{
    return (WGPUStringView){ .data = s, .length = s ? strlen(s) : 0 };
}

/* ── Internal context ──────────────────────────────────────────────── */

typedef struct {
    WGPUInstance        instance;
    WGPUAdapter         adapter;
    WGPUDevice          device;
    WGPUQueue           queue;
    WGPUComputePipeline pipelines[BN_WGPU_MAX_TYPES];
    WGPUBindGroupLayout layouts[BN_WGPU_MAX_TYPES];
} BnWgpuCtx;

/* ── GPU buffer handle ─────────────────────────────────────────────── */

typedef struct {
    WGPUBuffer buf;
    size_t     size;
    int        type;
    int        rows;
    int        cols;
} BnWgpuBuf;

/* ── Uniform block for compute shaders ─────────────────────────────── */

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t n_tokens;
    uint32_t extra;  /* reserved / padding */
} BnWgpuUniforms;

/* ── Error tracking ────────────────────────────────────────────────── */

static _Thread_local int wgpu_last_error;

static void on_uncaptured_error(WGPUDevice const *device,
                                 WGPUErrorType type,
                                 WGPUStringView message,
                                 void *userdata1, void *userdata2)
{
    (void)device; (void)type; (void)userdata1; (void)userdata2;
    wgpu_last_error = 1;
    if (message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] uncaptured error: %.*s\n",
                (int)message.length, message.data);
}

static void on_device_lost(WGPUDevice const *device,
                            WGPUDeviceLostReason reason,
                            WGPUStringView message,
                            void *userdata1, void *userdata2)
{
    (void)device; (void)reason; (void)userdata1; (void)userdata2;
    if (message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] device lost: %.*s\n",
                (int)message.length, message.data);
}

/* ── Sync callback helpers (v27: two userdata pointers) ────────────── */

typedef struct { WGPUAdapter adapter; int ok; } AdapterReq;

static void on_adapter_request(WGPURequestAdapterStatus status,
                                WGPUAdapter adapter,
                                WGPUStringView message,
                                void *userdata1, void *userdata2)
{
    (void)userdata2;
    AdapterReq *r = (AdapterReq *)userdata1;
    r->adapter = adapter;
    r->ok = (status == WGPURequestAdapterStatus_Success);
    if (!r->ok && message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] adapter request failed: %.*s\n",
                (int)message.length, message.data);
}

typedef struct { WGPUDevice device; int ok; } DeviceReq;

static void on_device_request(WGPURequestDeviceStatus status,
                               WGPUDevice device,
                               WGPUStringView message,
                               void *userdata1, void *userdata2)
{
    (void)userdata2;
    DeviceReq *r = (DeviceReq *)userdata1;
    r->device = device;
    r->ok = (status == WGPURequestDeviceStatus_Success);
    if (!r->ok && message.data && message.length > 0)
        fprintf(stderr, "[bn:gpu:wgpu] device request failed: %.*s\n",
                (int)message.length, message.data);
}

typedef struct { int done; WGPUMapAsyncStatus status; } MapReq;

static void on_buffer_map(WGPUMapAsyncStatus status,
                            WGPUStringView message,
                            void *userdata1, void *userdata2)
{
    (void)message; (void)userdata2;
    MapReq *r = (MapReq *)userdata1;
    r->status = status;
    r->done = 1;
}

/* ── Shader type name mapping ──────────────────────────────────────── */

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

/* All supported quant types for pipeline compilation */
static const int supported_types[] = {
    BN_GGUF_TENSOR_I2_S, BN_GGUF_TENSOR_TQ1_0, BN_GGUF_TENSOR_TQ2_0,
    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_1, BN_GGUF_TENSOR_Q8_0,
    BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_Q2_K, BN_GGUF_TENSOR_Q3_K,
    BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K,
    BN_GGUF_TENSOR_Q8_K, BN_GGUF_TENSOR_IQ4_NL, BN_GGUF_TENSOR_IQ4_XS,
    BN_GGUF_TENSOR_IQ3_XXS, BN_GGUF_TENSOR_IQ3_S, BN_GGUF_TENSOR_IQ2_XXS,
    BN_GGUF_TENSOR_IQ2_XS, BN_GGUF_TENSOR_IQ2_S,
};
#define N_SUPPORTED_TYPES ((int)(sizeof(supported_types) / sizeof(supported_types[0])))

/* ── Shader loading ────────────────────────────────────────────────── */

static char *load_shader_file(const char *shader_dir, const char *type_name)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_matvec.wgsl", shader_dir, type_name);

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t read = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if ((long)read != len) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

/* ── Pipeline compilation ──────────────────────────────────────────── */

static int compile_pipeline(BnWgpuCtx *ctx, int type, const char *shader_dir)
{
    if (type < 0 || type >= BN_WGPU_MAX_TYPES) return -1;

    const char *name = shader_name_for_type(type);
    if (!name) return -1;

    /* Load shader source from file */
    char *wgsl = NULL;
    if (shader_dir) {
        wgsl = load_shader_file(shader_dir, name);
    }
    if (!wgsl) return -1;  /* No embedded shaders yet */

    size_t wgsl_len = strlen(wgsl);

    /* Clear error flag */
    wgpu_last_error = 0;

    /* Create shader module */
    WGPUShaderSourceWGSL wgsl_desc = {
        .chain = { .sType = WGPUSType_ShaderSourceWGSL },
        .code = { .data = wgsl, .length = wgsl_len },
    };
    WGPUShaderModuleDescriptor sm_desc = {
        .nextInChain = &wgsl_desc.chain,
        .label = sv("bn_shader"),
    };

    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(ctx->device, &sm_desc);
    free(wgsl);

    if (!shader || wgpu_last_error) {
        if (shader) wgpuShaderModuleRelease(shader);
        fprintf(stderr, "[bn:gpu:wgpu] shader compilation failed for %s\n", name);
        return -1;
    }

    /* Create compute pipeline with auto-layout */
    WGPUComputePipelineDescriptor pipe_desc = {
        .label = sv("bn_pipeline"),
        .layout = NULL,  /* auto-layout */
        .compute = {
            .module = shader,
            .entryPoint = sv("main"),
        },
    };

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(
        ctx->device, &pipe_desc);
    wgpuShaderModuleRelease(shader);

    if (!pipeline || wgpu_last_error) {
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        fprintf(stderr, "[bn:gpu:wgpu] pipeline creation failed for %s\n", name);
        return -1;
    }

    /* Get bind group layout from auto-layout pipeline */
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    if (!layout) {
        wgpuComputePipelineRelease(pipeline);
        return -1;
    }

    ctx->pipelines[type] = pipeline;
    ctx->layouts[type] = layout;
    return 0;
}

/* ── Staging buffer readback ───────────────────────────────────────── */

static int readback_buffer(BnWgpuCtx *ctx, WGPUBuffer src,
                            size_t size, void *out_host)
{
    /* Create staging buffer (MAP_READ | COPY_DST) */
    WGPUBufferDescriptor staging_desc = {
        .label = sv("bn_staging"),
        .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
        .size  = size,
    };
    WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx->device, &staging_desc);
    if (!staging) return -1;

    /* Encode copy command */
    WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_readback_enc") };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
    if (!encoder) {
        wgpuBufferDestroy(staging);
        wgpuBufferRelease(staging);
        return -1;
    }
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, staging, 0, size);

    WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_readback_cmd") };
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuCommandEncoderRelease(encoder);
    if (!cmd) {
        wgpuBufferDestroy(staging);
        wgpuBufferRelease(staging);
        return -1;
    }

    /* Submit and poll */
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);
    wgpuCommandBufferRelease(cmd);

    /* Map staging buffer */
    MapReq map_req = {0};
    WGPUBufferMapCallbackInfo map_cb = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = on_buffer_map,
        .userdata1 = &map_req,
    };
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, map_cb);
    wgpuDevicePoll(ctx->device, 1, NULL);

    if (!map_req.done || map_req.status != WGPUMapAsyncStatus_Success) {
        wgpuBufferDestroy(staging);
        wgpuBufferRelease(staging);
        return -1;
    }

    const void *mapped = wgpuBufferGetConstMappedRange(staging, 0, size);
    if (!mapped) {
        wgpuBufferUnmap(staging);
        wgpuBufferDestroy(staging);
        wgpuBufferRelease(staging);
        return -1;
    }

    memcpy(out_host, mapped, size);

    wgpuBufferUnmap(staging);
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
    return 0;
}

/* ── Vtable: buffer_create ─────────────────────────────────────────── */

static void *wgpu_buffer_create(void *vctx, const void *data, size_t size,
                                 int type, int rows, int cols)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    if (!ctx || !ctx->device || !data || size == 0)
        return NULL;

    /* Align size to 4 bytes (WebGPU requirement) */
    size_t aligned = (size + 3) & ~(size_t)3;

    WGPUBufferDescriptor desc = {
        .label = sv("bn_weight"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size  = aligned,
    };
    WGPUBuffer buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
    if (!buf) return NULL;

    /* Upload weight data */
    wgpuQueueWriteBuffer(ctx->queue, buf, 0, data, size);

    /* Wrap in handle struct */
    BnWgpuBuf *handle = malloc(sizeof(BnWgpuBuf));
    if (!handle) {
        wgpuBufferDestroy(buf);
        wgpuBufferRelease(buf);
        return NULL;
    }
    handle->buf = buf;
    handle->size = aligned;
    handle->type = type;
    handle->rows = rows;
    handle->cols = cols;
    return handle;
}

/* ── Vtable: buffer_destroy ────────────────────────────────────────── */

static void wgpu_buffer_destroy(void *vctx, void *buffer)
{
    (void)vctx;
    if (!buffer) return;
    BnWgpuBuf *h = (BnWgpuBuf *)buffer;
    if (h->buf) {
        wgpuBufferDestroy(h->buf);
        wgpuBufferRelease(h->buf);
    }
    free(h);
}

/* ── Vtable: matvec ────────────────────────────────────────────────── */

static int wgpu_matvec(void *vctx, float *out, void *W_buf, const float *x,
                        int rows, int cols, int type)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    BnWgpuBuf *wbuf = (BnWgpuBuf *)W_buf;
    if (!ctx || !wbuf || !x || !out) return -1;
    if (type < 0 || type >= BN_WGPU_MAX_TYPES) return -1;
    if (!ctx->pipelines[type]) return -1;  /* no pipeline -> CPU fallback */

    int rc = -1;
    WGPUBuffer x_buf = NULL, out_buf = NULL, uniform_buf = NULL;
    WGPUBindGroup bind_group = NULL;
    WGPUCommandEncoder encoder = NULL;
    WGPUComputePassEncoder pass = NULL;
    WGPUCommandBuffer cmd = NULL;

    size_t x_size = (size_t)cols * sizeof(float);
    size_t x_aligned = (x_size + 3) & ~(size_t)3;
    size_t out_size = (size_t)rows * sizeof(float);
    size_t out_aligned = (out_size + 3) & ~(size_t)3;

    /* Create x buffer (input) */
    {
        WGPUBufferDescriptor desc = {
            .label = sv("bn_x"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size  = x_aligned,
        };
        x_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!x_buf) goto cleanup;
        wgpuQueueWriteBuffer(ctx->queue, x_buf, 0, x, x_size);
    }

    /* Create output buffer */
    {
        WGPUBufferDescriptor desc = {
            .label = sv("bn_out"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
                     | WGPUBufferUsage_CopyDst,
            .size  = out_aligned,
        };
        out_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!out_buf) goto cleanup;
    }

    /* Create uniform buffer */
    {
        BnWgpuUniforms uniforms = {
            .rows = (uint32_t)rows,
            .cols = (uint32_t)cols,
            .n_tokens = 1,
            .extra = 0,
        };
        size_t uni_size = (sizeof(uniforms) + 15) & ~(size_t)15;
        WGPUBufferDescriptor desc = {
            .label = sv("bn_uniforms"),
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size  = uni_size,
        };
        uniform_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!uniform_buf) goto cleanup;
        wgpuQueueWriteBuffer(ctx->queue, uniform_buf, 0,
                              &uniforms, sizeof(uniforms));
    }

    /* Create bind group: 0=W, 1=x, 2=out, 3=uniforms */
    {
        WGPUBindGroupEntry entries[4] = {
            { .binding = 0, .buffer = wbuf->buf, .offset = 0, .size = wbuf->size },
            { .binding = 1, .buffer = x_buf,     .offset = 0, .size = x_aligned },
            { .binding = 2, .buffer = out_buf,   .offset = 0, .size = out_aligned },
            { .binding = 3, .buffer = uniform_buf, .offset = 0,
              .size = (sizeof(BnWgpuUniforms) + 15) & ~(size_t)15 },
        };
        WGPUBindGroupDescriptor bg_desc = {
            .label = sv("bn_matvec_bg"),
            .layout = ctx->layouts[type],
            .entryCount = 4,
            .entries = entries,
        };
        bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
        if (!bind_group) goto cleanup;
    }

    /* Encode compute pass */
    {
        WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_matvec_enc") };
        encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
        if (!encoder) goto cleanup;

        WGPUComputePassDescriptor pass_desc = { .label = sv("bn_matvec_pass") };
        pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        if (!pass) goto cleanup;

        wgpuComputePassEncoderSetPipeline(pass, ctx->pipelines[type]);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);

        /* One workgroup per row */
        uint32_t wg_x = (uint32_t)rows;
        wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, 1, 1);
        wgpuComputePassEncoderEnd(pass);

        WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_matvec_cmd") };
        cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        if (!cmd) goto cleanup;
    }

    /* Submit and poll */
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);

    /* Readback output */
    rc = readback_buffer(ctx, out_buf, out_size, out);

cleanup:
    if (cmd) wgpuCommandBufferRelease(cmd);
    if (pass) wgpuComputePassEncoderRelease(pass);
    if (encoder) wgpuCommandEncoderRelease(encoder);
    if (bind_group) wgpuBindGroupRelease(bind_group);
    if (uniform_buf) { wgpuBufferDestroy(uniform_buf); wgpuBufferRelease(uniform_buf); }
    if (out_buf) { wgpuBufferDestroy(out_buf); wgpuBufferRelease(out_buf); }
    if (x_buf) { wgpuBufferDestroy(x_buf); wgpuBufferRelease(x_buf); }
    return rc;
}

/* ── Vtable: matmul ────────────────────────────────────────────────── */

static int wgpu_matmul(void *vctx, float *out, void *W_buf, const float *X,
                        int rows, int cols, int n_tokens, int type)
{
    BnWgpuCtx *ctx = (BnWgpuCtx *)vctx;
    BnWgpuBuf *wbuf = (BnWgpuBuf *)W_buf;
    if (!ctx || !wbuf || !X || !out) return -1;
    if (type < 0 || type >= BN_WGPU_MAX_TYPES) return -1;
    if (!ctx->pipelines[type]) return -1;

    int rc = -1;
    WGPUBuffer x_buf = NULL, out_buf = NULL, uniform_buf = NULL;
    WGPUBindGroup bind_group = NULL;
    WGPUCommandEncoder encoder = NULL;
    WGPUComputePassEncoder pass = NULL;
    WGPUCommandBuffer cmd = NULL;

    size_t x_size = (size_t)n_tokens * (size_t)cols * sizeof(float);
    size_t x_aligned = (x_size + 3) & ~(size_t)3;
    size_t out_size = (size_t)n_tokens * (size_t)rows * sizeof(float);
    size_t out_aligned = (out_size + 3) & ~(size_t)3;

    /* Create x buffer */
    {
        WGPUBufferDescriptor desc = {
            .label = sv("bn_x_batch"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size  = x_aligned,
        };
        x_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!x_buf) goto cleanup;
        wgpuQueueWriteBuffer(ctx->queue, x_buf, 0, X, x_size);
    }

    /* Create output buffer */
    {
        WGPUBufferDescriptor desc = {
            .label = sv("bn_out_batch"),
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
                     | WGPUBufferUsage_CopyDst,
            .size  = out_aligned,
        };
        out_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!out_buf) goto cleanup;
    }

    /* Create uniform buffer */
    {
        BnWgpuUniforms uniforms = {
            .rows = (uint32_t)rows,
            .cols = (uint32_t)cols,
            .n_tokens = (uint32_t)n_tokens,
            .extra = 0,
        };
        size_t uni_size = (sizeof(uniforms) + 15) & ~(size_t)15;
        WGPUBufferDescriptor desc = {
            .label = sv("bn_uniforms_batch"),
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size  = uni_size,
        };
        uniform_buf = wgpuDeviceCreateBuffer(ctx->device, &desc);
        if (!uniform_buf) goto cleanup;
        wgpuQueueWriteBuffer(ctx->queue, uniform_buf, 0,
                              &uniforms, sizeof(uniforms));
    }

    /* Create bind group */
    {
        WGPUBindGroupEntry entries[4] = {
            { .binding = 0, .buffer = wbuf->buf, .offset = 0, .size = wbuf->size },
            { .binding = 1, .buffer = x_buf,     .offset = 0, .size = x_aligned },
            { .binding = 2, .buffer = out_buf,   .offset = 0, .size = out_aligned },
            { .binding = 3, .buffer = uniform_buf, .offset = 0,
              .size = (sizeof(BnWgpuUniforms) + 15) & ~(size_t)15 },
        };
        WGPUBindGroupDescriptor bg_desc = {
            .label = sv("bn_matmul_bg"),
            .layout = ctx->layouts[type],
            .entryCount = 4,
            .entries = entries,
        };
        bind_group = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
        if (!bind_group) goto cleanup;
    }

    /* Encode compute pass: dispatch(rows, n_tokens, 1) */
    {
        WGPUCommandEncoderDescriptor enc_desc = { .label = sv("bn_matmul_enc") };
        encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &enc_desc);
        if (!encoder) goto cleanup;

        WGPUComputePassDescriptor pass_desc = { .label = sv("bn_matmul_pass") };
        pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        if (!pass) goto cleanup;

        wgpuComputePassEncoderSetPipeline(pass, ctx->pipelines[type]);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
            pass, (uint32_t)rows, (uint32_t)n_tokens, 1);
        wgpuComputePassEncoderEnd(pass);

        WGPUCommandBufferDescriptor cmd_desc = { .label = sv("bn_matmul_cmd") };
        cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        if (!cmd) goto cleanup;
    }

    /* Submit and poll */
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuDevicePoll(ctx->device, 1, NULL);

    /* Readback output */
    rc = readback_buffer(ctx, out_buf, out_size, out);

cleanup:
    if (cmd) wgpuCommandBufferRelease(cmd);
    if (pass) wgpuComputePassEncoderRelease(pass);
    if (encoder) wgpuCommandEncoderRelease(encoder);
    if (bind_group) wgpuBindGroupRelease(bind_group);
    if (uniform_buf) { wgpuBufferDestroy(uniform_buf); wgpuBufferRelease(uniform_buf); }
    if (out_buf) { wgpuBufferDestroy(out_buf); wgpuBufferRelease(out_buf); }
    if (x_buf) { wgpuBufferDestroy(x_buf); wgpuBufferRelease(x_buf); }
    return rc;
}

/* ── Public API: create ────────────────────────────────────────────── */

BnGPUBackend *bn_gpu_wgpu_create(const char *shader_dir)
{
    BnWgpuCtx *ctx = calloc(1, sizeof(BnWgpuCtx));
    if (!ctx) return NULL;

    /* Create instance with primary backends */
    WGPUInstanceExtras extras = {
        .chain = { .sType = (WGPUSType)WGPUSType_InstanceExtras },
        .backends = WGPUInstanceBackend_Primary,
    };
    WGPUInstanceDescriptor inst_desc = {
        .nextInChain = &extras.chain,
    };
    ctx->instance = wgpuCreateInstance(&inst_desc);
    if (!ctx->instance) {
        fprintf(stderr, "[bn:gpu:wgpu] failed to create instance\n");
        free(ctx);
        return NULL;
    }

    /* Request high-performance adapter */
    AdapterReq areq = {0};
    WGPURequestAdapterOptions adapter_opts = {
        .powerPreference = WGPUPowerPreference_HighPerformance,
    };
    WGPURequestAdapterCallbackInfo adapter_cb = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = on_adapter_request,
        .userdata1 = &areq,
    };
    wgpuInstanceRequestAdapter(ctx->instance, &adapter_opts, adapter_cb);
    if (!areq.ok || !areq.adapter) {
        fprintf(stderr, "[bn:gpu:wgpu] no GPU adapter found\n");
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }
    ctx->adapter = areq.adapter;

    /* Request device (default limits — wgpu-native v27 provides large buffers by default) */
    DeviceReq dreq = {0};
    WGPUDeviceDescriptor dev_desc = {
        .label = sv("bn_device"),
        .uncapturedErrorCallbackInfo = { .callback = on_uncaptured_error },
        .deviceLostCallbackInfo = { .callback = on_device_lost },
    };
    WGPURequestDeviceCallbackInfo device_cb = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = on_device_request,
        .userdata1 = &dreq,
    };
    wgpuAdapterRequestDevice(ctx->adapter, &dev_desc, device_cb);
    if (!dreq.ok || !dreq.device) {
        fprintf(stderr, "[bn:gpu:wgpu] device request failed\n");
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }
    ctx->device = dreq.device;
    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    if (!ctx->queue) {
        fprintf(stderr, "[bn:gpu:wgpu] failed to get device queue\n");
        wgpuDeviceRelease(ctx->device);
        wgpuAdapterRelease(ctx->adapter);
        wgpuInstanceRelease(ctx->instance);
        free(ctx);
        return NULL;
    }

    /* Compile pipelines for all supported quant types */
    int compiled = 0;
    for (int i = 0; i < N_SUPPORTED_TYPES; i++) {
        int type = supported_types[i];
        if (compile_pipeline(ctx, type, shader_dir) == 0)
            compiled++;
        /* Non-fatal: types without shaders just fall back to CPU */
    }

    fprintf(stderr, "[bn:gpu:wgpu] compiled %d/%d shader pipelines\n",
            compiled, N_SUPPORTED_TYPES);

    /* Build vtable */
    BnGPUBackend *gpu = calloc(1, sizeof(BnGPUBackend));
    if (!gpu) {
        bn_gpu_wgpu_destroy(NULL);  /* ctx leaked — but OOM anyway */
        return NULL;
    }
    gpu->buffer_create  = wgpu_buffer_create;
    gpu->buffer_destroy = wgpu_buffer_destroy;
    gpu->matvec         = wgpu_matvec;
    gpu->matmul         = wgpu_matmul;
    gpu->ctx            = ctx;

    return gpu;
}

/* ── Public API: destroy ───────────────────────────────────────────── */

void bn_gpu_wgpu_destroy(BnGPUBackend *gpu)
{
    if (!gpu) return;

    BnWgpuCtx *ctx = (BnWgpuCtx *)gpu->ctx;
    if (ctx) {
        /* Release pipelines and layouts */
        for (int i = 0; i < BN_WGPU_MAX_TYPES; i++) {
            if (ctx->layouts[i])
                wgpuBindGroupLayoutRelease(ctx->layouts[i]);
            if (ctx->pipelines[i])
                wgpuComputePipelineRelease(ctx->pipelines[i]);
        }
        if (ctx->queue) wgpuQueueRelease(ctx->queue);
        if (ctx->device) wgpuDeviceRelease(ctx->device);
        if (ctx->adapter) wgpuAdapterRelease(ctx->adapter);
        if (ctx->instance) wgpuInstanceRelease(ctx->instance);
        free(ctx);
    }
    free(gpu);
}

#endif /* BN_ENABLE_GPU */
