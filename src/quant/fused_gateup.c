#include "quant_ctx.h"
#include "quant_kernels_neon.h"
#include "quant_kernels_wasm.h"
#include "threadpool.h"
#include "gguf.h"
#include <stdlib.h>

#ifdef BN_FORCE_SCALAR
#undef __ARM_NEON
#undef __ARM_FEATURE_DOTPROD
#undef __wasm_relaxed_simd__
#undef __wasm_simd128__
#endif

#define BN_MAX_SCALE_BLOCKS 8192

int bn_quant_q4_gate_up_silu(float *out,
                             const BnQWeight *gate,
                             const BnPreparedWeight *gate_prepared,
                             const BnQWeight *up,
                             const BnPreparedWeight *up_prepared,
                             const float *x,
                             int8_t *x_q_buf,
                             BnThreadPool *pool) {
    if (!out || !gate || !up || !x || !x_q_buf) return -1;
    if (gate->type != BN_GGUF_TENSOR_Q4_0 ||
        up->type != BN_GGUF_TENSOR_Q4_0 ||
        gate->cols != up->cols ||
        gate->rows != up->rows ||
        gate->cols % 32 != 0)
        return -1;
    int n_blocks = gate->cols / 32;
    if (n_blocks <= 0 || n_blocks > BN_MAX_SCALE_BLOCKS) return -1;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (!gate_prepared || !gate_prepared->scales ||
        !up_prepared || !up_prepared->scales)
        return -1;
    float x_scales[n_blocks];
    bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, gate->cols);
    BnQ4GateUpCtx ctx = {
        out, gate, up, x_q_buf, x_scales, gate_prepared, up_prepared
    };
    BnTPTask task = {
        bn_quant_q4_repacked_gate_up_silu_neon_range,
        &ctx,
        gate->rows
    };
    bn_tp_dispatch(pool, &task, 1);
    return 0;
#elif defined(__wasm_relaxed_simd__)
    int use_canonical4 = getenv("BN_WASM_Q4_CANONICAL4") != NULL;
    if (!use_canonical4 &&
        (!gate_prepared || !gate_prepared->qs ||
         !up_prepared || !up_prepared->qs))
        return -1;
    float x_scales[n_blocks];
    bn_quant_x_to_q8_blocks(x, x_q_buf, x_scales, gate->cols);
    BnQ4GateUpCtx ctx = {
        out, gate, up, x_q_buf, x_scales, gate_prepared, up_prepared
    };
    BnTPTask task = use_canonical4
        ? (BnTPTask){ bn_quant_q4_wasm_gate_up_silu_4row_range, &ctx, (gate->rows + 3) / 4 }
        : (BnTPTask){ bn_quant_q4_repacked_gate_up_silu_wasm_range, &ctx, gate->rows };
    bn_tp_dispatch(pool, &task, 1);
    return 0;
#else
    (void)gate_prepared;
    (void)up_prepared;
    (void)pool;
    return -1;
#endif
}
