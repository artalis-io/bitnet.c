#include <metal_stdlib>
using namespace metal;

// Delta rule recurrence (one workgroup per V-head)
// Dispatch: (num_v_heads, 1, 1)
kernel void ssm_delta(device float       *state [[buffer(0)]],
                      device float       *out   [[buffer(1)]],
                      device const float *q     [[buffer(2)]],
                      device const float *k     [[buffer(3)]],
                      device const float *v     [[buffer(4)]],
                      device const float *alpha [[buffer(5)]],
                      device const float *beta  [[buffer(6)]],
                      constant uint      *p     [[buffer(7)]],
                      uint3 wid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float sk[512];
    uint hv_idx = wid.x;
    uint tid = lid.x;
    uint hk = p[0], hv = p[1], num_k_heads = p[2];
    float q_scale = as_type<float>(p[3]);
    uint state_layer_off = p[4] / 4;
    uint q_off = p[6];
    uint k_off = p[7];
    uint v_off = 2 * num_k_heads * hk;

    uint hk_idx = hv_idx % num_k_heads;
    uint state_base = state_layer_off + hv_idx * hk * hv;
    float decay = alpha[hv_idx];
    float b = beta[hv_idx];

    // Step 1: Decay state
    uint total = hk * hv;
    for (uint i = tid; i < total; i += 256)
        state[state_base + i] *= decay;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: sk[v] = sum_k S[k,v] * k[k] (Kahan compensated)
    for (uint vi = tid; vi < hv; vi += 256) {
        float sum = 0.0f, comp = 0.0f;
        for (uint ki = 0; ki < hk; ki++) {
            float y = state[state_base + ki * hv + vi] * k[k_off + hk_idx * hk + ki] - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sk[vi] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: S += k outer (beta * (v - sk))
    for (uint i = tid; i < total; i += 256) {
        uint ki = i / hv;
        uint vi2 = i % hv;
        float kk = k[k_off + hk_idx * hk + ki];
        state[state_base + i] += kk * b * (v[v_off + hv_idx * hv + vi2] - sk[vi2]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: out[v] = sum_k S[k,v] * q[k] * q_scale (Kahan compensated)
    for (uint vi = tid; vi < hv; vi += 256) {
        float sum = 0.0f, comp = 0.0f;
        for (uint ki = 0; ki < hk; ki++) {
            float y = state[state_base + ki * hv + vi] * q[q_off + hk_idx * hk + ki] - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        out[hv_idx * hv + vi] = sum * q_scale;
    }
}
