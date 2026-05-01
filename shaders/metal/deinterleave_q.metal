#include <metal_stdlib>
using namespace metal;

// Extract Q from interleaved [Q0, Gate0, Q1, Gate1, ...] layout
// Dispatch: (ceil(q_dim/256), 1, 1)
kernel void deinterleave_q(device const float *src [[buffer(0)]],
                           device float       *dst [[buffer(1)]],
                           constant uint      *p   [[buffer(2)]],
                           uint3 wid [[threadgroup_position_in_grid]],
                           uint3 lid [[thread_position_in_threadgroup]]) {
    uint gid = wid.x * 256 + lid.x;
    uint q_dim = p[0], hs = p[1];
    if (gid >= q_dim) return;
    uint head = gid / hs;
    uint d = gid % hs;
    dst[gid] = src[head * 2 * hs + d];
}
