#include "quant.h"
#include "quant_ctx.h"
#include "quant_kernels_scalar.h"
#include "sh_arena.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { rows = 8, cols = 64, n_blocks = cols / 32 };
    BnBlockQ4_0 blocks[rows * n_blocks];
    float x[cols];
    int8_t x_q[cols];
    float x_scales[n_blocks];
    float native[rows];
    float prepared_out[rows];

    memset(native, 0, sizeof(native));
    memset(prepared_out, 0, sizeof(prepared_out));
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            BnBlockQ4_0 *blk = &blocks[r * n_blocks + b];
            blk->d = bn_fp32_to_fp16(0.03125f * (float)(r + b + 1));
            for (int i = 0; i < 16; i++)
                blk->qs[i] = (uint8_t)(r * 31 + b * 19 + i * 7);
        }
    }
    for (int i = 0; i < cols; i++) {
        x[i] = 0.0625f * (float)((i * 17 + 3) % 31) - 0.75f;
    }

    BnQWeight weight = {
        .data = blocks, .type = BN_GGUF_TENSOR_Q4_0,
        .rows = rows, .cols = cols, .scale = 1.0f
    };
    BnPreparedWeightKind kind = BN_PREPARED_WEIGHT_NONE;
    size_t bytes = bn_quant_prepared_qweight_size(&weight, &kind);
    assert(kind == BN_PREPARED_WEIGHT_Q4_0_REPACK && bytes > 0);
    SHArena *arena = sh_arena_create(bytes + SH_ARENA_ALIGN);
    assert(arena != NULL);
    BnPreparedWeight prepared = {0};
    assert(bn_quant_prepare_qweight(&prepared, &weight, arena) == 0);

    for (int b = 0; b < n_blocks; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float value = x[b * 32 + i];
            float magnitude = value < 0.0f ? -value : value;
            if (magnitude > amax) amax = magnitude;
        }
        float scale = amax / 127.0f;
        float inverse = 1.0f / scale;
        x_scales[b] = bn_fp16_to_fp32(bn_fp32_to_fp16(scale));
        for (int i = 0; i < 32; i++) {
            float scaled = x[b * 32 + i] * inverse;
            int value = (int)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
            if (value < -127) value = -127;
            if (value > 127) value = 127;
            x_q[b * 32 + i] = (int8_t)value;
        }
    }

    BnQ4SdotCtx native_ctx = {
        native, &weight, x_q, x_scales, NULL
    };
    BnQ4SdotCtx prepared_ctx = {
        prepared_out, &weight, x_q, x_scales, &prepared
    };
    bn_quant_q4_scalar_sdot_range(&native_ctx, 0, rows);
    bn_quant_q4_repacked_scalar_sdot_range(&prepared_ctx, 0, rows);
    for (int r = 0; r < rows; r++)
        assert(prepared_out[r] == native[r]);

    sh_arena_free(arena);
    printf("Scalar Q4 prepared-layout test PASSED\n");
    return 0;
}
