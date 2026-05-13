#include "transformer_rmsnorm_internal.h"
#include "transformer_simd_internal.h"
#include <math.h>

#ifdef __wasm_simd128__

void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps) {
    v128_t sum_sq = wasm_f32x4_splat(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        v128_t xv = wasm_v128_load(x + i);
        sum_sq = wasm_f32x4_add(sum_sq, wasm_f32x4_mul(xv, xv));
    }
    float ss = bn_wasm_hsum_f32x4(sum_sq);
    for (; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    v128_t ss_v = wasm_f32x4_splat(ss);
    for (i = 0; i + 3 < size; i += 4) {
        v128_t xv = wasm_v128_load(x + i);
        v128_t wv = wasm_v128_load(w + i);
        wasm_v128_store(out + i, wasm_f32x4_mul(wasm_f32x4_mul(xv, ss_v), wv));
    }
    for (; i < size; i++) out[i] = x[i] * ss * w[i];
}

#endif // __wasm_simd128__
