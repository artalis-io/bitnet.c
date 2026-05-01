#include <metal_stdlib>
using namespace metal;

// SSM decay/update rate computation (per V-head)
// IEEE-compliant transcendentals via mathMode=safe (set at compile time)
// Dispatch: (1, 1, 1)
kernel void ssm_alpha_beta(device float       *alpha   [[buffer(0)]],
                           device float       *beta    [[buffer(1)]],
                           device const float *dt_bias [[buffer(2)]],
                           device const float *a_log   [[buffer(3)]],
                           constant uint      *p       [[buffer(4)]],
                           uint3 lid [[thread_position_in_threadgroup]]) {
    uint h = lid.x;
    uint n = p[0];
    if (h >= n) return;

    float dt = alpha[h] + dt_bias[h];
    // Softplus: log(1 + exp(x)) for x <= 20, else x
    // mathMode=safe ensures IEEE-compliant exp/log (no fast-math)
    float dt_sp = (dt > 20.0f) ? dt : log(1.0f + exp(dt));
    alpha[h] = exp(dt_sp * a_log[h]);
    beta[h] = 1.0f / (1.0f + exp(-beta[h]));
}
