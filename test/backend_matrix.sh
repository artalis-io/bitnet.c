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
avx512_quant_formats="q8 q4 q4k q5k q6k"

for fmt in $quant_formats; do
    for backend in $cpu_backends; do
        require_file "src/quant/${fmt}_${backend}.c"
    done
done

for fmt in $avx512_quant_formats; do
    case "$fmt" in
        q5k) require_file "src/quant/${fmt}_avx512.c" ;;
        *) require_file "src/quant/${fmt}_avx512_vnni.c" ;;
    esac
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

if ! grep -n '"avx512"' src/transformer/cpu_backend.c >/dev/null 2>&1 ||
   ! grep -n 'BN_CPU_BACKEND_AVX512' src/transformer/plan.c >/dev/null 2>&1 ||
   ! grep -n '"avx512"' src/transformer/prefill_backend.c >/dev/null 2>&1 ||
   ! grep -n '"avx512"' src/transformer/kv_backend.c >/dev/null 2>&1; then
    echo "CPU backend matrix must expose AVX512 as an explicit backend"
    fail=1
fi

if ! grep -n 'BnPrefillCPUOps' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill must dispatch CPU kernels through an explicit backend ops table"
    fail=1
fi

if ! grep -n 'ffn_activation' src/transformer/prefill.c >/dev/null 2>&1 ||
   grep -n 'prefill_ffn_activation_range' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill FFN activation must be selected through the backend ops table"
    fail=1
fi

if ! grep -n 'BnKVCPUOps' src/transformer/kv.c >/dev/null 2>&1; then
    echo "KV helpers must dispatch CPU kernels through an explicit backend ops table"
    fail=1
fi

if ! grep -n 'prepare_f16_x' src/transformer/logits.c >/dev/null 2>&1; then
    echo "Logits FP16 input conversion must be selected through backend ops"
    fail=1
fi

for file in \
    src/transformer/cpu.c \
    src/transformer/gpu.c \
    src/transformer/gpu_emit.c \
    src/transformer/gpu_fallback.c \
    src/transformer/gpu_policy.c \
    src/transformer/logits.c \
    src/transformer/plan.c \
    src/transformer/prefill.c \
    src/transformer/kv.c \
    src/model.c \
    src/main.c \
    src/model_gpu.c \
    src/model_session.c \
    src/model_embed.c \
    src/gpu_moe_bridge.c \
    src/moe_execute.c \
    src/transformer.c \
    src/generate.c
do
    if grep -n 'BN_MODEL_ARCH_POLICY_\|policy_flags' "$file" >/dev/null 2>&1; then
        echo "$file must use model_arch policy helpers, not direct architecture policy flags"
        fail=1
    fi
done

for file in \
    src/transformer.c \
    src/transformer/cpu.c \
    src/transformer/gpu.c \
    src/transformer/gpu_emit.c \
    src/transformer/gpu_fallback.c \
    src/transformer/gpu_policy.c \
    src/transformer/logits.c \
    src/transformer/plan.c \
    src/transformer/prefill.c \
    src/transformer/kv.c \
    src/model.c \
    src/model_session.c \
    src/model_embed.c \
    src/moe_execute.c \
    src/generate.c
do
    if grep -n 'BN_GGUF_TENSOR_' "$file" >/dev/null 2>&1; then
        echo "$file must use quant/backend_quant policy helpers, not direct tensor-format checks"
        fail=1
    fi
done

for file in \
    src/transformer.c \
    src/transformer/cpu.c \
    src/transformer/logits.c \
    src/model_session.c
do
    if grep -n 'bn_model_arch_gemma4_divides_rope_freqs\|static .*qwen\|static .*gemma' "$file" >/dev/null 2>&1; then
        echo "$file must use model-neutral architecture policy helpers"
        fail=1
    fi
done

for file in include/model_config.h include/model_weights.h include/model_run_state.h
do
    if grep -n 'Gemma4\|Qwen\|gemma4_\|qwen2_moe\|qwen2moe_\|BN_MODEL_ARCH_POLICY_.*GEMMA\|BN_MODEL_ARCH_POLICY_.*QWEN\|BN_MODEL_ARCH_POLICY_.*BITNET' "$file" >/dev/null 2>&1; then
        echo "$file must expose behavior-named shared model state, not family-prefixed fields or comments"
        fail=1
    fi
done

for file in \
    src/moe_route.c \
    src/moe_math.c \
    src/moe_prefill.c \
    src/moe_execute.c \
    src/moe_internal.h
do
    if grep -n '__AVX\|__ARM_NEON\|float32x\|__m256\|_mm[0-9]*_\|vfmaq\|vld1q\|vaddvq' "$file" >/dev/null 2>&1; then
        echo "$file must dispatch MoE CPU kernels through src/moe_cpu_kernels.c"
        fail=1
    fi
done

for file in \
    src/model.c \
    src/model_session.c \
    src/model_embed.c
do
    if grep -n '__AVX\|__ARM_NEON\|__wasm_simd128__\|__wasm_relaxed_simd__' "$file" >/dev/null 2>&1; then
        echo "$file must use backend/quant helpers, not direct CPU ISA checks"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "Backend matrix FAILED"
    exit 1
fi

echo "Backend matrix PASSED"
