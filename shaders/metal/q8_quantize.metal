#include <metal_stdlib>
using namespace metal;

kernel void q8_quantize(
    device const float *x      [[buffer(0)]],
    device char        *x_q    [[buffer(1)]],
    device float       *scales [[buffer(2)]],
    constant uint      *p      [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    uint cols = p[0];
    uint token = gid.y;
    uint block = gid.x;
    uint base = token * cols + block * 32;
    uint lane = lid.x & 31;

    float xv = x[base + lane];
    float amax = simd_max(abs(xv));

    uint scale_idx = token * (cols >> 5) + block;
    if (lane == 0)
        scales[scale_idx] = amax == 0.0f ? 0.0f : amax / 127.0f;

    if (amax == 0.0f) {
        x_q[base + lane] = 0;
        return;
    }

    float inv = 127.0f / amax;
    float q = clamp(round(xv * inv), -127.0f, 127.0f);
    x_q[base + lane] = char(q);
}
