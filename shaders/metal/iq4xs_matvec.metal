#include <metal_stdlib>
using namespace metal;

// IQ type stub — falls back to CPU SIMD (returns without writing)
// TODO: implement codebook lookup for this type
kernel void iq4xs_matvec(device const uchar *weights [[buffer(0)]],
                         device const float *x       [[buffer(1)]],
                         device float       *out     [[buffer(2)]],
                         constant uint      *p       [[buffer(3)]],
                         uint3 wid [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]]) {
    // Stub: zero output so caller detects fallback needed
    uint rows = p[0];
    uint gid = wid.x * 256 + lid.x;
    if (gid < rows) out[gid] = 0.0f;
}
