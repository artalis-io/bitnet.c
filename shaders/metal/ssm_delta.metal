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
    uint hv_idx = wid.x;
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

    // State is stored transposed as S[v][k]. One thread owns a row so the
    // float4 FMA and pairwise reduction order matches the ARM reference.
    for (uint vi = lid.x; vi < hv; vi += 256) {
        uint row = state_base + vi * hk;
        uint k_base = k_off + hk_idx * hk;
        uint q_base = q_off + hk_idx * hk;
        float4 s_k4 = 0.0f;
        uint ki = 0;
        for (; ki + 4 <= hk; ki += 4) {
            device float4 *state4 = (device float4 *)(state + row + ki);
            device const float4 *k4 =
                (device const float4 *)(k + k_base + ki);
            float4 ls = *state4 * decay;
            *state4 = ls;
            s_k4 = fma(ls, *k4, s_k4);
        }
        float2 s_k2 = s_k4.xy + s_k4.zw;
        float s_k = s_k2.x + s_k2.y;
        for (; ki < hk; ki++) {
            state[row + ki] *= decay;
            s_k += state[row + ki] * k[k_base + ki];
        }
        float delta = (v[v_off + hv_idx * hv + vi] - s_k) * b;

        float4 vdelta = delta;
        float4 y4 = 0.0f;
        for (ki = 0; ki + 4 <= hk; ki += 4) {
            device float4 *state4 = (device float4 *)(state + row + ki);
            device const float4 *k4 =
                (device const float4 *)(k + k_base + ki);
            device const float4 *q4 =
                (device const float4 *)(q + q_base + ki);
            float4 ls = fma(*k4, vdelta, *state4);
            *state4 = ls;
            y4 = fma(ls, *q4, y4);
        }
        float2 y2 = y4.xy + y4.zw;
        float y = y2.x + y2.y;
        for (; ki < hk; ki++) {
            state[row + ki] += k[k_base + ki] * delta;
            y += state[row + ki] * q[q_base + ki];
        }
        out[hv_idx * hv + vi] = y * q_scale;
    }
}
