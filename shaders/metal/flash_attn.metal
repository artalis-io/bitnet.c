#include <metal_stdlib>
using namespace metal;

// Fused short-context attention: Q·K scores + softmax + weighted V combine.
//
// One threadgroup handles one attention head. Eight simdgroups compute score
// tiles in parallel, then the full threadgroup normalizes scores and combines V.
// This keeps short decode parallelism closer to the scores/softmax/combine path
// while avoiding the intermediate global attention buffer and two extra
// dispatches.
//
// Dispatch: (n_heads, 1, 1)
//
// p0 = n_heads, p1 = head_size, p2 = n_kv, p3 = kv_mul
// p4 = kv_dim, p5 = seq_len, p6 = loff, p7 = inv_sqrt_hs (bitcast)

constant uint MAX_FLASH_KV = 1024;

kernel void flash_attn(device const float *q           [[buffer(0)]],
                       device const float *key_cache   [[buffer(1)]],
                       device const float *value_cache [[buffer(2)]],
                       device float       *xb          [[buffer(3)]],
                       constant uint      *p           [[buffer(4)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint n_heads = p[0], head_size = p[1], n_kv = p[2], kv_mul = p[3];
    uint kv_dim = p[4], loff = p[6];
    float scale = as_type<float>(p[7]);

    threadgroup float scores[MAX_FLASH_KV];
    threadgroup float simd_scratch[8];

    uint tid = lid.x;
    uint simd_id = tid >> 5;
    uint lane = tid & 31;
    uint h = wid.x;

    if (h >= n_heads || n_kv > MAX_FLASH_KV) return;

    uint kv_h = h / kv_mul;
    uint q_base = h * head_size;

    for (uint t = simd_id; t < n_kv; t += 8) {
        uint kv_base = loff + t * kv_dim + kv_h * head_size;
        float partial = 0.0f;
        for (uint d = lane; d < head_size; d += 32)
            partial += q[q_base + d] * key_cache[kv_base + d];
        float score = simd_sum(partial) * scale;
        if (lane == 0)
            scores[t] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -3.402823e+38f;
    for (uint t = tid; t < n_kv; t += 256)
        local_max = max(local_max, scores[t]);

    float partial_max = simd_max(local_max);
    if (lane == 0) simd_scratch[simd_id] = partial_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_scratch[tid];
        v = max(v, simd_shuffle_xor(v, 4));
        v = max(v, simd_shuffle_xor(v, 2));
        v = max(v, simd_shuffle_xor(v, 1));
        if (tid == 0) simd_scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_score = simd_scratch[0];

    float local_sum = 0.0f;
    for (uint t = tid; t < n_kv; t += 256) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }

    float partial_sum = simd_sum(local_sum);
    if (lane == 0) simd_scratch[simd_id] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_scratch[tid];
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (tid == 0) simd_scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / simd_scratch[0];

    uint out_base = h * head_size;
    for (uint d = tid; d < head_size; d += 256) {
        float acc = 0.0f;
        for (uint t = 0; t < n_kv; t++) {
            uint v_base = loff + t * kv_dim + kv_h * head_size;
            acc += scores[t] * value_cache[v_base + d];
        }
        xb[out_base + d] = acc * inv_sum;
    }
}
