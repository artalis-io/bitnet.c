#include <metal_stdlib>
using namespace metal;

kernel void gelu_gate(device float *gate [[buffer(0)]],
                      const device float *aux [[buffer(1)]],
                      constant uint *p [[buffer(2)]],
                      uint3 wid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    uint gid = wid.x * 256 + lid.x;
    if (gid >= p[0]) return;
    bool reference = (p[2] & 32u) != 0u;
    float g = reference ? float(half(gate[gid])) : gate[gid];
    float inner = 0.7978845608028654f * g *
                  (1.0f + 0.044715f * g * g);
    float gelu = g <= -10.0f ? 0.0f
               : g >= 10.0f ? g
               : 0.5f * g * (1.0f + precise::tanh(inner));
    if (reference)
        gelu = float(half(gelu));
    gate[gid] = gelu * aux[p[1] + gid];
}
