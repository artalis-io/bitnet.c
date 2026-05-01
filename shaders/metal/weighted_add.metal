#include <metal_stdlib>
using namespace metal;

// Weighted addition: x[i] += weight * r[i]
// Dispatch: (ceil(dim/256), 1, 1)
kernel void weighted_add(device float       *x [[buffer(0)]],
                         device const float *r [[buffer(1)]],
                         constant uint      *p [[buffer(2)]],
                         uint3 wid [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]]) {
    uint gid = wid.x * 256 + lid.x;
    if (gid >= p[0]) return;
    float w = as_type<float>(p[1]);
    x[gid] += w * r[gid];
}
