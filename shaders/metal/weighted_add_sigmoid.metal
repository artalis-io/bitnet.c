#include <metal_stdlib>
using namespace metal;

// Compute the shared-expert scalar gate, then apply its sigmoid to the output.
kernel void weighted_add_sigmoid(
    device float *out [[buffer(0)]],
    device const float *shared [[buffer(1)]],
    device const float *gate_weight [[buffer(2)]],
    device const float *gate_input [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float sums[256];
    uint dim = p[3];
    float sum = 0.0f;
    for (uint i = lid; i < dim; i += threads)
        sum += gate_weight[i] * gate_input[i];
    sums[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            sums[lid] += sums[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = 1.0f / (1.0f + exp(-sums[0]));
    if (p[4] != 0)
        scale = 1.0f - scale;
    scale *= as_type<float>(p[1]);
    for (uint i = lid; i < p[0]; i += threads) {
        float weighted = scale * shared[i];
        out[i] = p[2] != 0 ? weighted : out[i] + weighted;
    }
}
