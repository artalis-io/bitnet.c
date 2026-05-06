#include <metal_stdlib>
using namespace metal;

// SSM decay/update rate computation from stacked [alpha | beta] matvec output.
// Dispatch: (1, 1, 1)
kernel void ssm_alpha_beta_split(device const float *src     [[buffer(0)]],
                                 device float       *alpha   [[buffer(1)]],
                                 device float       *beta    [[buffer(2)]],
                                 device const float *dt_bias [[buffer(3)]],
                                 device const float *a_log   [[buffer(4)]],
                                 constant uint      *p       [[buffer(5)]],
                                 uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = lid.x;
    uint n = p[0];
    uint beta_off = p[1];
    if (h >= n) return;

    float dt = src[h] + dt_bias[h];
    float dt_sp = (dt > 20.0f) ? dt : log(1.0f + exp(dt));
    alpha[h] = exp(dt_sp * a_log[h]);
    float b = src[beta_off + h];
    beta[h] = 1.0f / (1.0f + exp(-b));
}
