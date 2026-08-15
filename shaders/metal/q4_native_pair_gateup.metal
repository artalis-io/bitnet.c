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

kernel void q4_native_pair_gateup(
    device const uchar *gate_weights [[buffer(0)]],
    device const uchar *up_weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1];
    uint lane = lid.x & 31;
    uint block_lane = lane >> 1;
    uint half_off = (lane & 1) * 8;
    uint tile_start = wid.x * 4;
    uint blocks_per_row = cols >> 5;
    uint row_bytes = blocks_per_row * 18;
    float gate_acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float up_acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint b = block_lane; b < blocks_per_row; b += 16) {
        device const float *xb = x + b * 32 + half_off;
        float4 x0 = *(device const float4 *)(xb + 0);
        float4 x1 = *(device const float4 *)(xb + 4);
        float4 x2 = *(device const float4 *)(xb + 16);
        float4 x3 = *(device const float4 *)(xb + 20);
        float sumx = (x0.x + x0.y) + (x0.z + x0.w) +
                     (x1.x + x1.y) + (x1.z + x1.w) +
                     (x2.x + x2.y) + (x2.z + x2.w) +
                     (x3.x + x3.y) + (x3.z + x3.w);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; r++) {
            uint row = tile_start + r;
            if (row >= rows) continue;
            device const uchar *gate = gate_weights + row * row_bytes + b * 18;
            device const uchar *up = up_weights + row * row_bytes + b * 18;
            device const uchar *gq = gate + 2 + half_off;
            device const uchar *uq = up + 2 + half_off;
            uchar4 g0 = *(device const packed_uchar4 *)(gq + 0);
            uchar4 g1 = *(device const packed_uchar4 *)(gq + 4);
            uchar4 u0 = *(device const packed_uchar4 *)(uq + 0);
            uchar4 u1 = *(device const packed_uchar4 *)(uq + 4);
            float gdot = dot(float4(g0 & uchar4(15)), x0) +
                         dot(float4(g1 & uchar4(15)), x1) +
                         dot(float4(g0 >> uchar4(4)), x2) +
                         dot(float4(g1 >> uchar4(4)), x3);
            float udot = dot(float4(u0 & uchar4(15)), x0) +
                         dot(float4(u1 & uchar4(15)), x1) +
                         dot(float4(u0 >> uchar4(4)), x2) +
                         dot(float4(u1 >> uchar4(4)), x3);
            gate_acc[r] += float(*(device const half *)gate) *
                           (gdot - 8.0f * sumx);
            up_acc[r] += float(*(device const half *)up) *
                         (udot - 8.0f * sumx);
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; r++) {
        float gate = simd_sum(gate_acc[r]);
        float up = simd_sum(up_acc[r]);
        uint row = tile_start + r;
        if (lane == 0 && row < rows) {
            float activated = (p[3] & 4u) != 0u
                ? gate / (1.0f + exp(-gate))
                : gate / (1.0f + bn_fast_exp(-gate));
            out[row] = activated * up;
        }
    }
}
