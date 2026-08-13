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
    threadgroup float signed_vals[256];
    threadgroup short quant_vals[256];

    float xv = x[base + tid];
    abs_vals[tid] = fabs(xv);
    signed_vals[tid] = xv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (abs_vals[tid + stride] > abs_vals[tid]) {
                abs_vals[tid] = abs_vals[tid + stride];
                signed_vals[tid] = signed_vals[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sb = token * (cols / 256u) + block;
    float amax = abs_vals[0];
    float max_value = signed_vals[0];
    float id = (amax == 0.0f) ? 0.0f : -127.0f / max_value;
    if (tid == 0) {
        xd[sb] = (amax == 0.0f) ? 0.0f : 1.0f / id;
    }

    int q = int(rint(xv * id));
    q = min(127, max(-128, q));
    xq[base + tid] = char(q);
    quant_vals[tid] = short(q);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {
        short sum = 0;
        for (uint i = 0; i < 16u; i++)
            sum += quant_vals[tid * 16u + i];
        bsums[sb * 16u + tid] = sum;
    }
}
