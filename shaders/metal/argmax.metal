#include <metal_stdlib>
using namespace metal;

struct ArgmaxParams {
    uint n;
    uint n_penalty_tokens;
    float repeat_penalty;
};

struct ArgmaxPair {
    float value;
    uint index;
};

static inline bool argmax_better(float value, uint index,
                                 float best_value, uint best_index) {
    return value > best_value ||
           (value == best_value && index < best_index);
}

static inline bool penalty_contains(device const int *tokens, uint n,
                                    uint token) {
    for (uint i = 0; i < n; ++i) {
        if (uint(tokens[i]) == token) return true;
    }
    return false;
}

kernel void argmax_stage1(
    device const float *values [[buffer(0)]],
    device const int *penalty_tokens [[buffer(1)]],
    device ArgmaxPair *partials [[buffer(2)]],
    constant ArgmaxParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]],
    uint threads [[threads_per_threadgroup]])
{
    uint global_tid = group * threads + tid;
    uint grid_size = groups * threads;
    float best_value = -INFINITY;
    uint best_index = 0;
    for (uint i = global_tid; i < p.n; i += grid_size) {
        float value = values[i];
        if (p.repeat_penalty != 1.0f &&
            penalty_contains(penalty_tokens, p.n_penalty_tokens, i)) {
            value = value > 0.0f ? value / p.repeat_penalty
                                 : value * p.repeat_penalty;
        }
        if (argmax_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    threadgroup ArgmaxPair shared[256];
    shared[tid] = { best_value, best_index };
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ArgmaxPair other = shared[tid + stride];
            if (argmax_better(other.value, other.index,
                              shared[tid].value, shared[tid].index))
                shared[tid] = other;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = shared[0];
}

kernel void argmax_stage2(device const ArgmaxPair *partials [[buffer(0)]],
                          device int *result [[buffer(1)]],
                          constant uint &n [[buffer(2)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]])
{
    ArgmaxPair best = { -INFINITY, 0 };
    for (uint i = tid; i < n; i += threads) {
        ArgmaxPair candidate = partials[i];
        if (argmax_better(candidate.value, candidate.index,
                          best.value, best.index))
            best = candidate;
    }
    threadgroup ArgmaxPair shared[256];
    shared[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ArgmaxPair other = shared[tid + stride];
            if (argmax_better(other.value, other.index,
                              shared[tid].value, shared[tid].index))
                shared[tid] = other;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[0] = int(shared[0].index);
}
