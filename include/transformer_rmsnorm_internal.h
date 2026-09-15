#ifndef BN_TRANSFORMER_RMSNORM_INTERNAL_H
#define BN_TRANSFORMER_RMSNORM_INTERNAL_H

// F32 squares accumulated in double, then (x * scale) * weight.
// Supports unaligned inputs and out == x; size zero is a no-op.
void bn_transformer_rmsnorm_reference(float *out, const float *x,
    const float *w, int size, float eps);

void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_default(float *out, const float *x, const float *w,
                                    int size, float eps);

#endif // BN_TRANSFORMER_RMSNORM_INTERNAL_H
