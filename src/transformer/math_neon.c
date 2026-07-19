#include "transformer_math_internal.h"
#include "transformer_cpu_features_internal.h"

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
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float32x4_t inv_sum = vdupq_n_f32(1.0f / sum);
    int norm_idx = 0;
    for (; norm_idx + 3 < size; norm_idx += 4)
        vst1q_f32(x + norm_idx,
                  vmulq_f32(vld1q_f32(x + norm_idx), inv_sum));
    for (; norm_idx < size; norm_idx++) x[norm_idx] /= sum;
}
#endif
