#include <metal_stdlib>
using namespace metal;

kernel void q8_quantize(
    device const float *x      [[buffer(0)]],
    device char        *x_q    [[buffer(1)]],
    device float       *scales [[buffer(2)]],
    constant uint      *p      [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]])
{
    uint cols = p[0];
    uint token = gid.y;
    uint block = gid.x;
    uint base = token * cols + block * 32;

    device const float4 *xp = (device const float4 *)(x + base);
    float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    float4 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
    float4 m0 = max(max(abs(x0), abs(x1)), max(abs(x2), abs(x3)));
    float4 m1 = max(max(abs(x4), abs(x5)), max(abs(x6), abs(x7)));
    float4 mv = max(m0, m1);
    float amax = max(max(mv.x, mv.y), max(mv.z, mv.w));

    uint scale_idx = token * (cols >> 5) + block;
    if (amax == 0.0f) {
        scales[scale_idx] = 0.0f;
        for (uint i = 0; i < 32; i++)
            x_q[base + i] = 0;
        return;
    }

    float inv = 127.0f / amax;
    scales[scale_idx] = amax / 127.0f;
    for (uint i = 0; i < 32; i++) {
        float q = clamp(round(x[base + i] * inv), -127.0f, 127.0f);
        x_q[base + i] = char(q);
    }
}
