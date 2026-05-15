#include <metal_stdlib>
using namespace metal;

kernel void q8k_quantize(device const float *x      [[buffer(0)]],
                         device char        *xq     [[buffer(1)]],
                         device float       *xd     [[buffer(2)]],
                         device short       *bsums  [[buffer(3)]],
                         constant uint      *p      [[buffer(4)]],
                         uint3 gid [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]]) {
    uint tid = lid.x;
    uint cols = p[0];
    uint token = gid.y;
    uint block = gid.x;
    uint base = token * cols + block * 256u;

    threadgroup float abs_vals[256];

    float xv = x[base + tid];
    abs_vals[tid] = fabs(xv);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            abs_vals[tid] = max(abs_vals[tid], abs_vals[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sb = token * (cols / 256u) + block;
    float amax = abs_vals[0];
    float id = (amax == 0.0f) ? 0.0f : 127.0f / amax;
    if (tid == 0) {
        xd[sb] = (amax == 0.0f) ? 0.0f : amax / 127.0f;
    }

    int q = int(rint(xv * id));
    q = min(127, max(-128, q));
    xq[base + tid] = char(q);

    (void)bsums;
}
