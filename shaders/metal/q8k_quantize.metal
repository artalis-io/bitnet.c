#include <metal_stdlib>
using namespace metal;

kernel void q8k_quantize(device const float *x      [[buffer(0)]],
                         device char        *xq     [[buffer(1)]],
                         device float       *xd     [[buffer(2)]],
                         device short       *bsums  [[buffer(3)]],
                         constant uint      *p      [[buffer(4)]],
                         uint3 gid [[threadgroup_position_in_grid]]) {
    uint cols = p[0];
    uint token = gid.y;
    uint block = gid.x;
    uint base = token * cols + block * 256u;

    float amax = 0.0f;
    for (uint i = 0; i < 256u; i++) {
        amax = max(amax, fabs(x[base + i]));
    }

    uint sb = token * (cols / 256u) + block;
    if (amax == 0.0f) {
        xd[sb] = 0.0f;
        for (uint i = 0; i < 256u; i++) xq[base + i] = 0;
        for (uint g = 0; g < 16u; g++) bsums[sb * 16u + g] = 0;
        return;
    }

    float d = amax / 127.0f;
    float id = 127.0f / amax;
    xd[sb] = d;
    for (uint g = 0; g < 16u; g++) {
        int sum = 0;
        for (uint i = 0; i < 16u; i++) {
            uint idx = base + g * 16u + i;
            int q = int(rint(x[idx] * id));
            q = min(127, max(-128, q));
            xq[idx] = char(q);
            sum += q;
        }
        bsums[sb * 16u + g] = short(sum);
    }
}
