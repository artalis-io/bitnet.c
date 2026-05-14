#include <metal_stdlib>
using namespace metal;

static inline float bn_fast_exp(float x) {
    const float log2e = 1.4426950409f;
    const float ln2 = 0.6931471806f;
    x = clamp(x, -87.3f, 88.7f);
    float n = floor(fma(x, log2e, 0.5f));
    float r = fma(-n, ln2, x);
    float poly = fma(0.04166664f, r, 0.16666667f);
    poly = fma(poly, r, 0.49999994f);
    poly = fma(poly, r, 1.0f);
    poly = fma(poly, r, 1.0f);
    int e = (int(n) + 127) << 23;
    return poly * as_type<float>(e);
}

static inline float bn_fast_silu(float x) {
    return x / (1.0f + bn_fast_exp(-x));
}

// SiLU gated activation: gate[i] = silu(gate[i]) * up[i]
// Dispatch: (ceil(dim/256), 1, 1)
kernel void silu_gate(device float       *gate [[buffer(0)]],
                      device const float *up   [[buffer(1)]],
                      constant uint      *p    [[buffer(2)]],
                      uint3 wid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    uint gid = wid.x * 256 + lid.x;
    uint dim = p[0];
    if (gid >= dim) return;
    float g = gate[gid];
    gate[gid] = bn_fast_silu(g) * up[gid];
}
