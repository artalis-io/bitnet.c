#!/bin/sh
set -eu

fail=0

require_file() {
    if [ ! -f "$1" ]; then
        echo "MISSING $1"
        fail=1
    fi
}

quant_formats="i2s tq2 tq1 q8 q4 q4_1 f32 f16 bf16 q6k q8k q4k q5k q3k q2k iq4nl iq4xs iq3xxs iq3s iq2xxs iq2xs iq2s"
cpu_backends="scalar avx2 neon wasm"

for fmt in $quant_formats; do
    for backend in $cpu_backends; do
        require_file "src/quant/${fmt}_${backend}.c"
    done
done

transformer_kernels="rmsnorm gqa logits ssm"
for kernel in $transformer_kernels; do
    for backend in $cpu_backends; do
        require_file "src/transformer/${kernel}_${backend}.c"
    done
done

require_file "src/transformer/gqa_tq_scalar.c"
require_file "src/transformer/gqa_tq_neon.c"

gpu_formats="i2s tq2 tq1 q8 q4 q4_1 bf16 f16 f32 q6k q8k q4k q5k q3k q2k iq4nl iq4xs iq3xxs iq3s iq2xxs iq2xs iq2s"
for fmt in $gpu_formats; do
    require_file "shaders/${fmt}_matvec.wgsl"
    require_file "shaders/metal/${fmt}_matvec.metal"
done

gpu_ops="rmsnorm rope rope_qk gqa_scores gqa_combine silu_gate relu2_gate residual_add residual_rmsnorm softmax bias_add weighted_add sigmoid_gate per_head_rmsnorm deinterleave_q ssm_conv_silu ssm_l2norm ssm_alpha_beta ssm_alpha_beta_split ssm_delta ssm_gate buf_copy"
for op in $gpu_ops; do
    require_file "shaders/${op}.wgsl"
    require_file "shaders/metal/${op}.metal"
done

require_file "include/gpu_backend.h"
require_file "include/gpu_wgpu.h"
require_file "include/gpu_metal.h"
require_file "src/gpu_wgpu.c"
require_file "src/gpu_metal.m"

if grep -n 'BN_GPU_SHADER_' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "GPU emit must use BN_GPU_CODE_* and backend lowering, not BN_GPU_SHADER_*"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "Backend matrix FAILED"
    exit 1
fi

echo "Backend matrix PASSED"
