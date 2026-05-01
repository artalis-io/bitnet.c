#include <metal_stdlib>
using namespace metal;

// Conv1d (kernel=4) + SiLU activation + conv_state shift
// Dispatch: (ceil(qkv_dim/256), 1, 1)
kernel void ssm_conv_silu(device float       *qkv        [[buffer(0)]],
                          device float       *conv_state [[buffer(1)]],
                          device const float *conv1d_w   [[buffer(2)]],
                          constant uint      *p          [[buffer(3)]],
                          uint3 wid [[threadgroup_position_in_grid]],
                          uint3 lid [[thread_position_in_threadgroup]]) {
    uint ch = wid.x * 256 + lid.x;
    uint qkv_dim = p[0];
    uint kern = p[1];
    uint cs_off = p[2];

    if (ch >= qkv_dim) return;

    float sum = 0.0f;
    for (uint k = 0; k < kern - 1; k++)
        sum += conv_state[cs_off + k * qkv_dim + ch] * conv1d_w[ch * kern + k];
    float cur = qkv[ch];
    sum += cur * conv1d_w[ch * kern + kern - 1];

    for (uint k = 0; k < kern - 2; k++)
        conv_state[cs_off + k * qkv_dim + ch] = conv_state[cs_off + (k + 1) * qkv_dim + ch];
    conv_state[cs_off + (kern - 2) * qkv_dim + ch] = cur;

    qkv[ch] = sum / (1.0f + exp(-sum));
}
