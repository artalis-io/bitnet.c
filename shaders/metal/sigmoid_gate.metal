#include <metal_stdlib>
using namespace metal;

// Sigmoid gate: out[i] *= sigmoid(gate[offset + i])
// Dispatch: (ceil(q_dim/256), 1, 1)
kernel void sigmoid_gate(device float       *out  [[buffer(0)]],
                         device const float *gate [[buffer(1)]],
                         constant uint      *p    [[buffer(2)]],
                         uint3 wid [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]]) {
    uint gid = wid.x * 256 + lid.x;
    uint q_dim = p[0], hs = p[1];
    if (gid >= q_dim) return;
    uint head = gid / hs;
    uint d = gid % hs;
    uint gate_idx = head * 2 * hs + hs + d;
    float g = gate[gate_idx];
    out[gid] *= 1.0f / (1.0f + exp(-g));
}
