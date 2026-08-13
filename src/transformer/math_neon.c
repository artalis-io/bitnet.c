#include "transformer_math_internal.h"
#include "transformer_cpu_features_internal.h"
#include "simd_helpers.h"

#if BN_TRANSFORMER_CPU_HAS_NEON
#include <arm_neon.h>
#include <math.h>

void bn_transformer_softmax_neon(float *x, int size) {
    if (size <= 0) return;
    float max_val = x[0];
    int idx = 0;
    float32x4_t vmaxv = vdupq_n_f32(x[0]);
    for (; idx + 3 < size; idx += 4)
        vmaxv = vmaxq_f32(vmaxv, vld1q_f32(x + idx));
    max_val = vmaxvq_f32(vmaxv);
    for (; idx < size; idx++) {
        if (x[idx] > max_val) max_val = x[idx];
    }
    double sum = 0.0;
    int exp_idx = 0;
    const float32x4_t maxv = vdupq_n_f32(max_val);
    for (; exp_idx + 3 < size; exp_idx += 4) {
        float32x4_t value = bn_neon_fast_exp_f32(
            vsubq_f32(vld1q_f32(x + exp_idx), maxv));
        vst1q_f32(x + exp_idx, value);
        sum += (double)vaddvq_f32(value);
    }
    if (exp_idx < size) {
        float tail[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        int tail_size = size - exp_idx;
        for (int i = 0; i < tail_size; i++)
            tail[i] = x[exp_idx + i];
        float32x4_t value = bn_neon_fast_exp_f32(
            vsubq_f32(vld1q_f32(tail), maxv));
        vst1q_f32(tail, value);
        sum += (double)vaddvq_f32(value);
        for (int i = 0; i < tail_size; i++)
            x[exp_idx + i] = tail[i];
    }
    float inv = (float)(1.0 / sum);
    float32x4_t inv_sum = vdupq_n_f32(inv);
    int norm_idx = 0;
    for (; norm_idx + 3 < size; norm_idx += 4)
        vst1q_f32(x + norm_idx,
                  vmulq_f32(vld1q_f32(x + norm_idx), inv_sum));
    for (; norm_idx < size; norm_idx++) x[norm_idx] *= inv;
}
#endif
