#include <metal_stdlib>
using namespace metal;

// ReLU² gated activation: gate[i] = max(0, gate[i])² * up[i]
// Dispatch: (ceil(dim/256), 1, 1)
kernel void relu2_gate(device float       *gate [[buffer(0)]],
                       device const float *up   [[buffer(1)]],
                       constant uint      *p    [[buffer(2)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint gid = wid.x * 256 + lid.x;
    uint dim = p[0];
    if (gid >= dim) return;
    float g = max(0.0f, gate[gid]);
    gate[gid] = g * g * up[gid];
}
