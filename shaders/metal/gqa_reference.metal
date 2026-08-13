#include <metal_stdlib>
using namespace metal;

static inline float reference_exp(float x) {
    const float r = 0x1.8p23f;
    float z = fma(x, 0x1.715476p+0f, r);
    float n = z - r;
    float b = fma(-n, 0x1.7f7d1cp-20f,
                  fma(-n, 0x1.62e4p-1f, x));
    uint e = as_type<uint>(z) << 23;
    float k = as_type<float>(e + 0x3f800000u);
    float u = b * b;
    float j = fma(0x1.573e2ep-5f, 1.0f, 0x1.0e4020p-7f * b);
    j = fma(j, u, fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f));
    j = fma(j, u, 0x1.ffffecp-1f * b);
    if (abs(n) <= 126.0f)
        return fma(k, j, k);
    uint d = n <= 0.0f ? 0x82000000u : 0u;
    float s1 = as_type<float>(d + 0x7f000000u);
    float s2 = as_type<float>(e - d);
    return abs(n) > 192.0f ? s1 * s1 : fma(s2, j, s2) * s1;
}

kernel void gqa_scores_reference(
    device const float *q [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device float *att [[buffer(2)]], constant uint *p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint score = gid.x;
    if (score >= n_heads * n_kv) return;
    uint h = score / n_kv, i = score - h * n_kv;
    uint q_base = h * head_size;
    uint k_base = p[6] + i * p[4] + (h / kv_mul) * head_size;
    float sums[4][4];
    for (uint group = 0; group < 4; group++)
        for (uint lane = 0; lane < 4; lane++)
            sums[group][lane] = 0.0f;
    uint d = 0, np = head_size & ~15u;
    for (; d < np; d += 16)
        for (uint group = 0; group < 4; group++)
            for (uint lane = 0; lane < 4; lane++) {
                uint j = d + group * 4 + lane;
                sums[group][lane] = fma(q[q_base + j], key_cache[k_base + j],
                                        sums[group][lane]);
            }
    float lanes[4];
    for (uint lane = 0; lane < 4; lane++)
        lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                      (sums[1][lane] + sums[3][lane]);
    float dot = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; d < head_size; d++)
        dot = fma(q[q_base + d], key_cache[k_base + d], dot);
    att[h * p[5] + i] = dot * as_type<float>(p[7]);
}

kernel void softmax_reference(
    device float *att [[buffer(0)]], constant uint *p [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint h = gid.x, n_heads = p[0], n_kv = p[1], base = h * p[2];
    if (h >= n_heads || n_kv == 0) return;
    float max_val = att[base];
    for (uint i = 1; i < n_kv; i++) max_val = max(max_val, att[base + i]);
    float sum = 0.0f;
    uint i = 0;
    for (; i + 3 < n_kv; i += 4) {
        float e0 = reference_exp(att[base + i] - max_val);
        float e1 = reference_exp(att[base + i + 1] - max_val);
        float e2 = reference_exp(att[base + i + 2] - max_val);
        float e3 = reference_exp(att[base + i + 3] - max_val);
        att[base + i] = e0;
        att[base + i + 1] = e1;
        att[base + i + 2] = e2;
        att[base + i + 3] = e3;
        sum += (e0 + e1) + (e2 + e3);
    }
    if (i < n_kv) {
        float tail[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint count = n_kv - i;
        for (uint j = 0; j < count; j++) {
            tail[j] = reference_exp(att[base + i + j] - max_val);
            att[base + i + j] = tail[j];
        }
        sum += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    }
    float inv = 1.0f / sum;
    for (uint i = 0; i < n_kv; i++) att[base + i] *= inv;
}

kernel void gqa_combine_reference(
    device const float *att [[buffer(0)]],
    device const float *value_cache [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint *p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint elem = gid.x;
    if (elem >= n_heads * head_size) return;
    uint h = elem / head_size, d = elem - h * head_size;
    uint kv_head = h / kv_mul;
    if (n_kv <= 16) {
        float sums[4][4];
        for (uint group = 0; group < 4; group++)
            for (uint lane = 0; lane < 4; lane++)
                sums[group][lane] = 0.0f;
        for (uint group = 0; group < 4; group++)
            for (uint lane = 0; lane < 4; lane++) {
                uint i = group * 4 + lane;
                if (i >= n_kv) break;
                uint v = p[6] + i * p[4] + kv_head * head_size + d;
                sums[group][lane] = fma(
                    att[h * p[5] + i], value_cache[v], 0.0f);
            }
        float lanes[4];
        for (uint lane = 0; lane < 4; lane++)
            lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                          (sums[1][lane] + sums[3][lane]);
        out[elem] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        return;
    }
    float acc = 0.0f;
    for (uint i = 0; i < n_kv; i++) {
        uint v = p[6] + i * p[4] + kv_head * head_size + d;
        acc = fma(att[h * p[5] + i], value_cache[v], acc);
    }
    out[elem] = acc;
}
