#include <metal_stdlib>
using namespace metal;

// One threadgroup cooperatively computes one expert logit so adjacent lanes
// read adjacent weights. A second dispatch performs top-k selection.
kernel void moe_route_logits(
    device const float *router [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *logits [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint n_experts = p[0], dim = p[5];
    if (expert >= n_experts) return;
    device const float *row = router + expert * dim;
    float score = 0.0f;
    for (uint d = lid; d < dim; d += 256)
        score = fma(row[d], x[d], score);
    score = simd_sum(score);
    threadgroup float partial[8];
    if (simd_lane == 0) partial[simd_group] = score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        score = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        score = simd_sum(score);
        if (simd_lane == 0) logits[expert] = score;
    }
}

kernel void moe_route_topk(
    device const float *logits [[buffer(0)]],
    device float *route [[buffer(1)]],
    device const float *expert_down_scale [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[256];
    threadgroup float group_scores[8];
    threadgroup uint group_indices[8];
    threadgroup uint selected[16];
    threadgroup float selected_scores[16];
    uint n_experts = p[0], k = p[1];
    scores[lid] = lid < n_experts ? logits[lid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    k = min(k, 16u);
    for (uint i = 0; i < k; i++) {
        bool used = false;
        for (uint j = 0; j < i; j++) used |= selected[j] == lid;
        float best_score = used ? -INFINITY : scores[lid];
        uint best = lid;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_score = simd_shuffle_down(best_score, offset);
            uint other = simd_shuffle_down(best, offset);
            if (simd_lane + offset < 32 &&
                (other_score > best_score ||
                 (other_score == best_score && other < best))) {
                best_score = other_score;
                best = other;
            }
        }
        if (simd_lane == 0) {
            group_scores[simd_group] = best_score;
            group_indices[simd_group] = best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            best_score = simd_lane < 8 ? group_scores[simd_lane] : -INFINITY;
            best = simd_lane < 8 ? group_indices[simd_lane] : simd_lane;
            for (uint offset = 16; offset > 0; offset >>= 1) {
                float other_score = simd_shuffle_down(best_score, offset);
                uint other = simd_shuffle_down(best, offset);
                if (simd_lane + offset < 32 &&
                    (other_score > best_score ||
                     (other_score == best_score && other < best))) {
                    best_score = other_score;
                    best = other;
                }
            }
            if (simd_lane == 0) {
                selected[i] = best;
                selected_scores[i] = best_score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid != 0) return;

    bool normalize_topk = (p[3] & 2u) == 0;
    float max_score = normalize_topk ? selected_scores[0] : scores[0];
    uint count = normalize_topk ? k : n_experts;
    for (uint i = 1; i < count; i++) {
        float v = normalize_topk ? selected_scores[i] : scores[i];
        max_score = max(max_score, v);
    }
    float denom = 0.0f;
    for (uint i = 0; i < count; i++) {
        float v = normalize_topk ? selected_scores[i] : scores[i];
        denom += exp(v - max_score);
    }
    float scale = as_type<float>(p[2]);
    if (scale == 0.0f) scale = 1.0f;
    for (uint i = 0; i < k; i++) {
        float expert_scale = p[4] != 0u
            ? expert_down_scale[selected[i]] : 1.0f;
        if (expert_scale == 0.0f) expert_scale = 1.0f;
        route[i] = exp(selected_scores[i] - max_score) / denom * scale *
                   expert_scale;
        route[k + i] = float(selected[i]);
    }
}

kernel void moe_route_capture(
    device const float *route [[buffer(0)]],
    device uint *history [[buffer(1)]],
    constant uint *p [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]) {
    uint offset = p[0], k = p[1];
    if (lid == 0) history[offset] = p[2];
    if (lid < k)
        history[offset + 1 + lid] = uint(route[k + lid] + 0.5f);
}
