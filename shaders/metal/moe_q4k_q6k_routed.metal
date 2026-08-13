#include <metal_stdlib>
using namespace metal;

constant uint BN_QK_K = 256;
constant uint BN_QK_BLOCK32 = 32;

static inline float moe_activation(float x, uint act_type) {
    if (act_type == 1)
        return max(x, 0.0f) * max(x, 0.0f);
    if (act_type == 2) {
        float inner = 0.7978845608028654f * x *
                      (1.0f + 0.044715f * x * x);
        return x <= -10.0f ? 0.0f
             : x >= 10.0f ? x
             : 0.5f * x * (1.0f + precise::tanh(inner));
    }
    return x / (1.0f + exp(-x));
}

static inline float q4_0_q8_0_lane_dot(device const uchar *row,
                                       device const char *xq,
                                       device const float *xd,
                                       uint cols, uint lane) {
    float acc = 0.0f;
    uint n_blocks = cols / BN_QK_BLOCK32;
    for (uint b = lane; b < n_blocks; b += 8) {
        device const uchar *block = row + b * 18;
        device const uchar *q = block + 2;
        device const char *xb = xq + b * BN_QK_BLOCK32;
        int qx = 0;
        for (uint i = 0; i < 16; i++) {
            uchar packed = q[i];
            qx += (int(packed & 15) - 8) * int(xb[i]);
            qx += (int(packed >> 4) - 8) * int(xb[i + 16]);
        }
        acc += float(*(device const half *)block) * xd[b] * float(qx);
    }
    return acc;
}

kernel void moe_q4_0_gateup_routed(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up [[buffer(1)]],
    device const char *xq [[buffer(2)]],
    device const float *xd [[buffer(3)]],
    device const float *route [[buffer(4)]],
    device float *mid [[buffer(5)]],
    constant uint *p [[buffer(6)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint lid = lid3.x;
    uint hidden = p[0], n_experts = p[1], k = p[2], dim = p[5];
    uint expert_stride = p[7];
    uint local_row = lid >> 3, lane = lid & 7;
    uint task = wid.x * 32 + local_row;
    uint slot = task / hidden, row = task - slot * hidden;
    if (slot >= k || row >= hidden) return;
    uint expert = min(uint(route[k + slot] + 0.5f), n_experts - 1);
    uint row_bytes = (dim / BN_QK_BLOCK32) * 18;
    float g = q4_0_q8_0_lane_dot(
        gate + expert * expert_stride + row * row_bytes,
        xq, xd, dim, lane);
    float u = q4_0_q8_0_lane_dot(
        up + expert * expert_stride + row * row_bytes,
        xq, xd, dim, lane);
    g += simd_shuffle_xor(g, 4);
    g += simd_shuffle_xor(g, 2);
    g += simd_shuffle_xor(g, 1);
    u += simd_shuffle_xor(u, 4);
    u += simd_shuffle_xor(u, 2);
    u += simd_shuffle_xor(u, 1);
    if (lane == 0)
        mid[slot * hidden + row] = moe_activation(g, p[3]) * u;
}

kernel void moe_q4_0_down_routed(
    device const uchar *down [[buffer(0)]],
    device const char *midq [[buffer(1)]],
    device const float *midd [[buffer(2)]],
    device const float *route [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint *p [[buffer(5)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint lid = lid3.x;
    uint hidden = p[0], n_experts = p[1], k = p[2], dim = p[5];
    uint expert_stride = p[6];
    uint local_row = lid >> 3, lane = lid & 7;
    uint row = wid.x * 32 + local_row;
    if (row >= dim) return;
    uint row_bytes = (hidden / BN_QK_BLOCK32) * 18;
    uint blocks_per_slot = hidden / BN_QK_BLOCK32;
    float acc = 0.0f;
    for (uint slot = 0; slot < k; slot++) {
        uint expert = min(uint(route[k + slot] + 0.5f), n_experts - 1);
        acc += route[slot] * q4_0_q8_0_lane_dot(
            down + expert * expert_stride + row * row_bytes,
            midq + slot * hidden,
            midd + slot * blocks_per_slot, hidden, lane);
    }
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) out[row] = acc;
}

static inline uint2 q4k_scale_min(uint j, device const uchar *scales) {
    if (j < 4)
        return uint2(scales[j] & 63, scales[j + 4] & 63);
    return uint2((scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                 (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
}

static inline float q4k_lane_dot(device const uchar *row,
                                 device const float *x,
                                 uint cols, uint lane) {
    float acc = 0.0f;
    uint group = lane >> 1, high = lane & 1, sub = group * 2 + high;
    uint qoff = group * 32;
    for (uint b = 0; b < cols / BN_QK_K; b++) {
        device const uchar *block = row + b * 144;
        float d = float(*(device const half *)block);
        float dmin = float(*(device const half *)(block + 2));
        uint2 sm = q4k_scale_min(sub, block + 4);
        device const uchar *q = block + 16 + qoff;
        device const float *xb = x + b * BN_QK_K + lane * 32;
        float qx = 0.0f, sx = 0.0f;
        for (uint i = 0; i < 32; i++) {
            float xv = xb[i];
            qx += float(high ? q[i] >> 4 : q[i] & 15) * xv;
            sx += xv;
        }
        acc += d * float(sm.x) * qx - dmin * float(sm.y) * sx;
    }
    return acc;
}

static inline float q4k_q8k_lane_dot(device const uchar *row,
                                     device const char *xq,
                                     device const float *xd,
                                     device const short *bsums,
                                     uint cols, uint lane) {
    float acc = 0.0f;
    uint group = lane >> 1, high = lane & 1, sub = lane;
    uint qoff = group * 32;
    for (uint b = 0; b < cols / BN_QK_K; b++) {
        device const uchar *block = row + b * 144;
        float d = float(*(device const half *)block);
        float dmin = float(*(device const half *)(block + 2));
        uint2 sm = q4k_scale_min(sub, block + 4);
        device const uchar *q = block + 16 + qoff;
        device const char *xb = xq + b * BN_QK_K + lane * 32;
        int qx = 0;
        for (uint i = 0; i < 32; i += 4) {
            uchar4 raw = *(device const uchar4 *)(q + i);
            char4 qv = char4(high ? raw >> 4 : raw & uchar4(15));
            char4 xv = *(device const char4 *)(xb + i);
            qx += int(dot(float4(qv), float4(xv)));
        }
        int sx = int(bsums[b * 16 + lane * 2]) +
                 int(bsums[b * 16 + lane * 2 + 1]);
        acc += xd[b] * (d * float(sm.x) * float(qx) -
                        dmin * float(sm.y) * float(sx));
    }
    return acc;
}

static inline void q4k_q8k_lane_dot_pair(device const uchar *gate_row,
                                         device const uchar *up_row,
                                         device const char *xq,
                                         device const float *xd,
                                         device const short *bsums,
                                         uint cols, uint lane,
                                         thread float &gate_acc,
                                         thread float &up_acc) {
    gate_acc = 0.0f;
    up_acc = 0.0f;
    uint group = lane >> 1, high = lane & 1;
    uint sub = lane;
    uint qoff = group * 32;
    for (uint b = 0; b < cols / BN_QK_K; b++) {
        device const uchar *gate_block = gate_row + b * 144;
        device const uchar *up_block = up_row + b * 144;
        uint2 gate_sm = q4k_scale_min(sub, gate_block + 4);
        uint2 up_sm = q4k_scale_min(sub, up_block + 4);
        device const uchar *gate_q = gate_block + 16 + qoff;
        device const uchar *up_q = up_block + 16 + qoff;
        device const char *xb = xq + b * BN_QK_K + lane * 32;
        int gate_qx = 0, up_qx = 0;
        for (uint i = 0; i < 32; i++) {
            int xv = int(xb[i]);
            gate_qx += int(high ? gate_q[i] >> 4 : gate_q[i] & 15) * xv;
            up_qx += int(high ? up_q[i] >> 4 : up_q[i] & 15) * xv;
        }
        int sx = int(bsums[b * 16 + lane * 2]) +
                 int(bsums[b * 16 + lane * 2 + 1]);
        float dx = xd[b];
        float gate_d = float(*(device const half *)gate_block);
        float gate_dmin = float(*(device const half *)(gate_block + 2));
        float up_d = float(*(device const half *)up_block);
        float up_dmin = float(*(device const half *)(up_block + 2));
        gate_acc += dx * (gate_d * float(gate_sm.x) * float(gate_qx) -
                          gate_dmin * float(gate_sm.y) * float(sx));
        up_acc += dx * (up_d * float(up_sm.x) * float(up_qx) -
                        up_dmin * float(up_sm.y) * float(sx));
    }
}

kernel void moe_q4k_gateup_routed(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up [[buffer(1)]],
    device const char *xq [[buffer(2)]],
    device const float *xd [[buffer(3)]],
    device const short *bsums [[buffer(4)]],
    device const float *route [[buffer(5)]],
    device float *mid [[buffer(6)]],
    constant uint *p [[buffer(7)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint lid = lid3.x;
    uint hidden = p[0], n_experts = p[1], k = p[2], dim = p[5];
    uint expert_stride = p[7];
    uint local_row = lid >> 3, lane = lid & 7;
    uint task = wid.x * 32 + local_row;
    uint slot = 0, row = 0, expert = 0;
    if (lane == 0) {
        slot = task / hidden;
        row = task - slot * hidden;
        if (slot < k)
            expert = min(uint(route[k + slot] + 0.5f), n_experts - 1);
    }
    ushort source_lane = ushort((lid & 31u) & ~7u);
    slot = simd_shuffle(slot, source_lane);
    row = simd_shuffle(row, source_lane);
    expert = simd_shuffle(expert, source_lane);
    if (slot >= k || row >= hidden) return;
    uint row_bytes = (dim / BN_QK_K) * 144;
    device const uchar *gate_row = gate + expert * expert_stride + row * row_bytes;
    device const uchar *up_row = up + expert * expert_stride + row * row_bytes;
    float g, u;
    q4k_q8k_lane_dot_pair(gate_row, up_row, xq, xd, bsums, dim, lane,
                          g, u);
    g += simd_shuffle_xor(g, 4);
    g += simd_shuffle_xor(g, 2);
    g += simd_shuffle_xor(g, 1);
    u += simd_shuffle_xor(u, 4);
    u += simd_shuffle_xor(u, 2);
    u += simd_shuffle_xor(u, 1);
    if (lane == 0)
        mid[slot * hidden + row] = g / (1.0f + exp(-g)) * u;
}

static inline float2 q6k_lane_dot32_pair(device const uchar *row0,
                                         device const uchar *row1,
                                         device const float *x,
                                         uint cols, uint lane) {
    float2 acc = 0.0f;
    uint logical_lane = lane >> 2;
    uint sublane = lane & 3;
    uint half_idx = logical_lane >> 2;
    uint quarter = logical_lane & 3;
    uint ql_off = half_idx * 64 + ((quarter & 1) ? 32 : 0);
    uint qh_off = half_idx * 32;
    uint qh_shift = quarter * 2;
    bool ql_high = quarter >= 2;
    uint sc_off = half_idx * 8 + quarter * 2;
    for (uint b = 0; b < cols / BN_QK_K; b++) {
        device const uchar *blocks[2] = {
            row0 + b * 210, row1 + b * 210
        };
        uint elem = sublane * 8;
        device const float *xb =
            x + b * BN_QK_K + logical_lane * 32 + elem;
        float2 sum = 0.0f;
        for (uint j = 0; j < 8; j++) {
            float xv = xb[j];
            for (uint r = 0; r < 2; r++) {
                device const uchar *ql = blocks[r] + ql_off;
                device const uchar *qh = blocks[r] + 128 + qh_off;
                uint i = elem + j;
                uint low = ql_high ? ql[i] >> 4 : ql[i] & 15;
                float q = float(low |
                    (((qh[i] >> qh_shift) & 3) << 4)) - 32.0f;
                sum[r] += q * xv;
            }
        }
        for (uint r = 0; r < 2; r++) {
            device const char *sc =
                (device const char *)(blocks[r] + 192);
            float d = float(*(device const half *)(blocks[r] + 208));
            acc[r] += d * float(sc[sc_off + (sublane >> 1)]) * sum[r];
        }
    }
    return acc;
}

static inline float q5k_lane_dot(device const uchar *row,
                                 device const float *x,
                                 uint cols, uint lane) {
    float acc = 0.0f;
    uint group = lane >> 1, high = lane & 1;
    uint bit = group * 2 + high;
    uint qoff = group * 32;
    for (uint b = 0; b < cols / BN_QK_K; b++) {
        device const uchar *block = row + b * 176;
        float d = float(*(device const half *)block);
        float dmin = float(*(device const half *)(block + 2));
        uint2 sm = q4k_scale_min(bit, block + 4);
        device const uchar *qh = block + 16;
        device const uchar *q = block + 48 + qoff;
        device const float *xb = x + b * BN_QK_K + lane * 32;
        for (uint i = 0; i < 32; i++) {
            uint nibble = high ? q[i] >> 4 : q[i] & 15;
            uint q5 = nibble | (((qh[i] >> bit) & 1) << 4);
            acc += (d * float(sm.x) * float(q5) -
                    dmin * float(sm.y)) * xb[i];
        }
    }
    return acc;
}

static inline float q5k_q8k_lane_dot(device const uchar *row,
                                     device const char *xq,
                                     device const float *xd,
                                     device const short *bsums,
                                     uint cols, uint lane) {
    float acc = 0.0f;
    uint group = lane >> 1, high = lane & 1;
    uint bit = group * 2 + high;
    uint qoff = group * 32;
    for (uint b = 0; b < cols / BN_QK_K; b++) {
        device const uchar *block = row + b * 176;
        float d = float(*(device const half *)block);
        float dmin = float(*(device const half *)(block + 2));
        uint2 sm = q4k_scale_min(bit, block + 4);
        device const uchar *qh = block + 16;
        device const uchar *q = block + 48 + qoff;
        device const char *xb = xq + b * BN_QK_K + lane * 32;
        int qx = 0;
        for (uint i = 0; i < 32; i++) {
            uint nibble = high ? q[i] >> 4 : q[i] & 15;
            uint q5 = nibble | (((qh[i] >> bit) & 1) << 4);
            qx += int(q5) * int(xb[i]);
        }
        int sx = int(bsums[b * 16 + lane * 2]) +
                 int(bsums[b * 16 + lane * 2 + 1]);
        acc += xd[b] * (d * float(sm.x) * float(qx) -
                        dmin * float(sm.y) * float(sx));
    }
    return acc;
}

kernel void moe_q4k_down_routed(
    device const uchar *down [[buffer(0)]],
    device const char *midq [[buffer(1)]],
    device const float *midd [[buffer(2)]],
    device const short *bsums [[buffer(3)]],
    device const float *route [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint *p [[buffer(6)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint hidden = p[0], n_experts = p[1], k = p[2], dim = p[5];
    uint expert_stride = p[6];
    uint lid = lid3.x;
    uint local_row = lid >> 3, lane = lid & 7;
    uint row = wid.x * 32 + local_row;
    if (row >= dim) return;
    uint row_bytes = (hidden / BN_QK_K) * 144;
    float acc = 0.0f;
    for (uint slot = 0; slot < k; slot++) {
        uint expert = min(uint(route[k + slot] + 0.5f), n_experts - 1);
        uint blocks_per_slot = hidden / BN_QK_K;
        acc += route[slot] * q4k_q8k_lane_dot(
            down + expert * expert_stride + row * row_bytes,
            midq + slot * hidden,
            midd + slot * blocks_per_slot,
            bsums + slot * blocks_per_slot * 16u, hidden, lane);
    }
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) out[row] = acc;
}

kernel void moe_q5k_down_routed(
    device const uchar *down [[buffer(0)]],
    device const char *midq [[buffer(1)]],
    device const float *midd [[buffer(2)]],
    device const short *bsums [[buffer(3)]],
    device const float *route [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint *p [[buffer(6)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint hidden = p[0], n_experts = p[1], k = p[2], dim = p[5];
    uint expert_stride = p[6];
    uint lid = lid3.x;
    uint local_row = lid >> 3, lane = lid & 7;
    uint row = wid.x * 32 + local_row;
    if (row >= dim) return;
    uint row_bytes = (hidden / BN_QK_K) * 176;
    float acc = 0.0f;
    for (uint slot = 0; slot < k; slot++) {
        uint expert = min(uint(route[k + slot] + 0.5f), n_experts - 1);
        uint blocks_per_slot = hidden / BN_QK_K;
        acc += route[slot] * q5k_q8k_lane_dot(
            down + expert * expert_stride + row * row_bytes,
            midq + slot * hidden,
            midd + slot * blocks_per_slot,
            bsums + slot * blocks_per_slot * 16u, hidden, lane);
    }
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) out[row] = acc;
}

kernel void moe_q6k_down_routed(
    device const uchar *down [[buffer(0)]],
    device const float *mid [[buffer(1)]],
    device const float *route [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
    uint lid = lid3.x;
    uint hidden = p[0], n_experts = p[1], k = p[2], dim = p[5];
    uint expert_stride = p[6];
    uint local_pair = lid >> 5, lane = lid & 31;
    uint row = wid.x * 16 + local_pair * 2;
    if (row >= dim) return;
    uint row_bytes = (hidden / BN_QK_K) * 210;
    float2 acc = 0.0f;
    for (uint slot = 0; slot < k; slot++) {
        uint expert = min(uint(route[k + slot] + 0.5f), n_experts - 1);
        device const uchar *row0 =
            down + expert * expert_stride + row * row_bytes;
        device const uchar *row1 = row + 1 < dim ? row0 + row_bytes : row0;
        acc += route[slot] * q6k_lane_dot32_pair(
            row0, row1,
            mid + slot * hidden,
            hidden, lane);
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    if (lane == 0) {
        out[row] = acc.x;
        if (row + 1 < dim) out[row + 1] = acc.y;
    }
}
