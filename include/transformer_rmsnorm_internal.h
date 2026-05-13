#ifndef BN_TRANSFORMER_RMSNORM_INTERNAL_H
#define BN_TRANSFORMER_RMSNORM_INTERNAL_H

void bn_transformer_rmsnorm_neon(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_avx2(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_wasm(float *out, const float *x, const float *w, int size, float eps);
void bn_transformer_rmsnorm_scalar(float *out, const float *x, const float *w, int size, float eps);

#endif // BN_TRANSFORMER_RMSNORM_INTERNAL_H
