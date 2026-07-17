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
    src/transformer.c \
    src/transformer/gpu.c \
    src/transformer/logits.c \
    src/transformer/prefill.c
do
    if grep -n 'bn_backend_quant_cuda_small_dense\|bn_backend_quant_is_kquant_float_fallback_candidate\|bn_backend_quant_supports_q[68]k*_logits_refine' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format capability helpers for quant-format policy"
        fail=1
    fi
    if grep -n 'bn_backend_quant_moe_route_q4_down\|bn_backend_quant_moe_gateup_q4\|bn_backend_quant_cpu_fused_q4_gateup_silu\|bn_backend_quant_stacked_pair_same_format\|bn_backend_quant_moe_route_q8' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format tuple helpers for quant-format policy"
        fail=1
    fi
done

if grep -n 'bn_quant_format_' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use backend_quant helpers for quant-format policy"
    fail=1
fi

if grep -n 'bn_quant_format_supports_q[68]_logits_refine\|bn_backend_quant_supports_q[68]k*_logits_refine' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for logits refine capability policy"
    fail=1
fi

if grep -n 'bn_quant_format_pair_same_format\|bn_backend_quant_stacked_pair_same_format' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for stacked Q/K quant compatibility"
    fail=1
fi

if grep -n 'bn_quant_format_allows_stacked_layout' src/backend_layout.c >/dev/null 2>&1; then
    echo "src/backend_layout.c must use backend_quant helpers for stacked layout quant-format policy"
    fail=1
fi

if grep -n 'bn_quant_format_supports_q[68]_logits_refine\|bn_backend_quant_supports_q[68]k*_logits_refine' src/transformer/logits.c >/dev/null 2>&1; then
    echo "src/transformer/logits.c must use logits policy helpers for logits refine capability policy"
    fail=1
fi

if grep -n '#include "backend_quant.h"\|bn_backend_quant_matvec.*gpu_buf' src/transformer/logits.c >/dev/null 2>&1; then
    echo "src/transformer/logits.c must use logits policy helpers for GPU-resident quant matvec dispatch"
    fail=1
fi

for file in \
    src/model.c \
    src/transformer/plan.c
do
    if grep -n 'bn_backend_quant_logits_uses_f16_path\|bn_backend_quant_tied_logits_uses_quant_path\|bn_backend_quant_logits_i8_cache_supported\|bn_backend_quant_tied_logits_uses_f16_path\|bn_backend_quant_tied_logits_i[0-9]_weight_type\|bn_backend_quant_tied_logits_f[0-9][0-9]_weight_type' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format logits helpers for quant-format policy"
        fail=1
    fi
done

if grep -n 'bn_quant_format_uses_f16_logits_path\|bn_quant_format_tied_logits_uses_quant_path\|bn_quant_format_tied_logits_uses_f16_path\|bn_quant_format_tied_logits_i[0-9]_weight_type\|bn_quant_format_tied_logits_f[0-9][0-9]_weight_type' src/transformer/plan.c >/dev/null 2>&1; then
    echo "src/transformer/plan.c must use logits policy helpers for logits quant-format policy"
    fail=1
fi

if grep -n 'bn_quant_format_tied_logits_uses_quant_path\|bn_quant_format_supports_logits_i8_cache' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model policy helpers for logits quant-format policy"
    fail=1
fi

if grep -n 'bn_quant_format_is_float_kquant_fallback_candidate' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use prefill policy helpers for float K-quant fallback policy"
    fail=1
fi

for file in \
    src/transformer/gpu_emit.c \
    src/transformer/plan.c
do
    if grep -n 'bn_quant_format_gpu_requires_exact_silu\|bn_backend_quant_gpu_requires_exact_silu\|bn_quant_format_gpu_prefers_gateup_split\|bn_backend_quant_gpu_prefers_gateup_split\|bn_backend_quant_gpu_fused_gateup_requires_cuda_opt_in\|bn_backend_quant_can_gpu_gateup_split_activation' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format GPU behavior helpers for quant-format policy"
        fail=1
    fi
done

if grep -n 'bn_quant_format_pair_same_format\|bn_backend_quant_stacked_pair_same_format\|bn_quant_format_supports_moe_q4_gateup\|bn_backend_quant_moe_gateup_q4' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for stacked pair and shared gate-up quant-format policy"
    fail=1
fi

for file in \
    src/transformer/gpu_emit.c \
    src/transformer/plan.c \
    src/gpu_moe_bridge.c
do
    if grep -n 'bn_quant_format_gpu_split_cap\|bn_backend_quant_gpu_split_cap\|bn_backend_quant_gpu_fused_gateup_silu_cap\|bn_quant_format_gpu_matvec_q8k_dot_flag\|bn_backend_quant_gpu_matvec_q8k_dot_flag\|bn_quant_format_gpu_matvec_exact_q6k_flag\|bn_backend_quant_gpu_matvec_exact_q6k_flag\|bn_quant_format_gpu_float_buffer_type\|bn_backend_quant_gpu_float_buffer_type\|BN_BACKEND_QUANT_GPU_MATVEC_FLAG_' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format GPU cap/flag helpers for quant-format policy"
        fail=1
    fi
done

for file in \
    src/gpu_moe_bridge.c
do
    if grep -n '#include "backend_quant.h"' "$file" >/dev/null 2>&1; then
        echo "$file must use quant/backend capability helpers directly, not backend_quant facade policy"
        fail=1
    fi
done

if grep -n 'BN_GPU_CODE_Q4K_MATVEC_SPLIT' src/gpu_moe_bridge.c >/dev/null 2>&1; then
    echo "src/gpu_moe_bridge.c must use GPU policy helpers for MoE gate-up split op policy"
    fail=1
fi

if grep -n 'BN_GPU_CODE_Q4K_MATVEC_SPLIT' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for dense gate-up split op policy"
    fail=1
fi

if grep -n 'BN_GPU_CODE_MATVEC_SPLIT\|BN_GPU_CODE_Q8_MATVEC_SPLIT\|BN_GPU_CODE_Q5K_MATVEC_SPLIT\|BN_GPU_CODE_UNKNOWN' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for QKV/SSM split op policy"
    fail=1
fi

if ! grep -n '#include "backend_quant.h"' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose quant-format policy through backend_quant helpers"
    fail=1
fi

if grep -n 'BN_CUDA_QWEN2MOE_GPU_ROUTE_FROM_LAYER\|BN_CUDA_QWEN2MOE_GPU_ROUTE_TO_LAYER' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for Qwen2MoE route range compatibility env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_[^"]*QWEN\|cuda_env_enabled("BN_CUDA_[^"]*QWEN\|cuda_env_value("BN_CUDA_[^"]*QWEN\|cuda_env_int("BN_CUDA_[^"]*QWEN\|cuda_env_float("BN_CUDA_[^"]*QWEN\|cuda_env_layer_selected("BN_CUDA_[^"]*QWEN' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must expose model-family CUDA env vars only as compatibility fallbacks for neutral policy helpers"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_[^"]*QWEN\|gpu_env_enabled("BN_CUDA_[^"]*QWEN\|gpu_env_value("BN_CUDA_[^"]*QWEN\|gpu_policy_env_int("BN_CUDA_[^"]*QWEN' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must expose model-family CUDA env vars only as compatibility fallbacks for neutral policy helpers"
    fail=1
fi

if grep -n 'cpu_attn_safe_default' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU-aware CPU attention fallback policy helpers"
    fail=1
fi

for file in \
    src/generate.c \
    src/transformer/prefill.c
do
    if grep -n 'BN_CUDA_DISABLE_PREFILL_SSM_LAYER' "$file" >/dev/null 2>&1; then
        echo "$file must use GPU policy helpers for prefill SSM layer compatibility env vars"
        fail=1
    fi
done

if grep -n 'BN_CUDA_DISABLE_SSM_FFN_FUSE\|BN_CUDA_ENABLE_MOE_PREFILL\|BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU policy helpers for prefill compatibility env vars"
    fail=1
fi

if grep -n 'bn_transformer_gpu_backend_is_cuda(gpu)\|bn_transformer_gpu_all2_q4_moe_requires_opt_in\|bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU-aware prefill backend policy helpers"
    fail=1
fi

if grep -n '!gpu || !gpu->prefill_ssm_layer' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU policy helpers for SSM prefill backend availability"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN\|BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN\|BN_CUDA_DISABLE_PREFILL_ATTN\|BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU policy helpers for prefill chain/attention env vars"
    fail=1
fi

if grep -n 'BN_GPU_DISABLE_FUSED_GATEUP\|BN_CUDA_ENABLE_Q5K_FUSED_GATEUP' src/transformer/plan.c >/dev/null 2>&1; then
    echo "src/transformer/plan.c must use GPU policy helpers for fused gate-up compatibility env vars"
    fail=1
fi

if grep -n 'bn_quant_format_can_gpu_native\|bn_quant_format_gpu_fused_gateup_silu_cap\|bn_quant_format_gpu_allows_gateup_split_activation\|bn_quant_format_gpu_split_cap' src/transformer/plan.c >/dev/null 2>&1; then
    echo "src/transformer/plan.c must use GPU policy helpers for quant-format GPU capability policy"
    fail=1
fi

if grep -n 'BN_GPU_BACKEND_' src/transformer/plan.c >/dev/null 2>&1; then
    echo "src/transformer/plan.c must use GPU policy helpers for backend placement"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_MOE_DECODE_CACHE' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for MoE decode cache compatibility env vars"
    fail=1
fi

if grep -n 'BN_GPU_Q8_REFINE_TOP' src/transformer/logits.c >/dev/null 2>&1; then
    echo "src/transformer/logits.c must use GPU policy helpers for Q8 logits refine top compatibility env vars"
    fail=1
fi

if grep -n 'BnBlockQ8_0\|BnBlockQ6K\|bn_fp16_to_fp32' src/transformer/logits.c >/dev/null 2>&1; then
    echo "src/transformer/logits.c must use quant logits-refine helpers, not direct quant block layout"
    fail=1
fi

if grep -n 'BnBlockQ8_0\|BnBlockQ6K\|bn_fp16_to_fp32\|gpu_exact_q[68]' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use quant logits-refine helpers, not direct quant block layout"
    fail=1
fi

if grep -n 'BN_CPU_TIED_Q6K_REFINE_TOP\|BN_CPU_TIED_Q6K_HYBRID_TOP\|BN_CPU_NATIVE_TIED_LOGITS' src/transformer/logits.c src/transformer/logits_backend.c >/dev/null 2>&1; then
    echo "Logits orchestration/backend code must use logits policy helpers for CPU tied-logits env vars"
    fail=1
fi

if ! grep -n 'BN_CPU_TIED_Q6K_REFINE_TOP\|BN_CPU_TIED_Q6K_HYBRID_TOP\|BN_CPU_NATIVE_TIED_LOGITS' src/transformer/logits_policy.c >/dev/null 2>&1; then
    echo "CPU tied-logits env compatibility policy must live in src/transformer/logits_policy.c"
    fail=1
fi

if grep -n 'BN_CPU_DISABLE_PREPARED_QWEIGHTS' src/transformer/cpu.c src/transformer/gpu_fallback.c src/transformer/cpu_backend.c >/dev/null 2>&1; then
    echo "CPU execution/backend and GPU fallback must use CPU backend policy helpers for prepared qweight env vars"
    fail=1
fi

if grep -n 'BN_DUMP_LAYER_INP\|BN_DUMP_LAYER_POS\|BN_DUMP_ALL_HEADS\|BN_CPU_REFERENCE_DOT\|BN_CPU_REFERENCE_Q4_DOT\|BN_CPU_LLAMA_DOT\|BN_CPU_LLAMA_Q4_DOT' src/transformer/cpu.c src/transformer/cpu_backend.c >/dev/null 2>&1; then
    echo "CPU execution/backend code must use CPU backend policy helpers for debug and fused gate-up env vars"
    fail=1
fi

if grep -n 'bn_quant_format_supports_cpu_fused_q4_gateup_silu\|bn_backend_quant_cpu_fused_q4_gateup_silu' src/transformer/cpu.c >/dev/null 2>&1; then
    echo "CPU execution code must use CPU backend policy helpers for fused gate-up quant capability"
    fail=1
fi

if grep -n '#include "backend_quant.h"\|bn_backend_quant_matvec.*gpu_buf' src/transformer/cpu.c >/dev/null 2>&1; then
    echo "CPU execution code must use CPU backend policy helpers for GPU-resident quant matvec dispatch"
    fail=1
fi

if grep -n 'bn_quant_format_can_preq8k\|bn_backend_quant_can_preq8k' src/transformer/cpu.c >/dev/null 2>&1; then
    echo "CPU execution code must use CPU backend policy helpers for preq8k quant capability"
    fail=1
fi

if grep -n 'bn_quant_format_can_preq8k\|bn_backend_quant_can_preq8k' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill execution code must use prefill policy helpers for preq8k quant capability"
    fail=1
fi

if grep -n 'bn_quant_format_can_preq8k\|bn_backend_quant_can_preq8k' src/moe_cpu_kernels.c >/dev/null 2>&1; then
    echo "MoE CPU kernels must use MoE policy helpers for preq8k quant capability"
    fail=1
fi

if grep -n 'bn_quant_format_has_embedded_tensor_scale\|bn_quant_embedded_tensor_scale_offset' src/moe_math.c >/dev/null 2>&1; then
    echo "MoE math must use MoE policy helpers for embedded tensor scale policy"
    fail=1
fi

if grep -n '#include "backend_quant.h"\|bn_backend_quant_matvec.*gpu_buf' src/moe_execute.c >/dev/null 2>&1; then
    echo "MoE execution must use MoE policy helpers for GPU-resident quant matvec dispatch"
    fail=1
fi

if grep -n 'bn_quant_format_pair_same_format\|bn_backend_quant_stacked_pair_same_format' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill execution code must use prefill policy helpers for stacked pair quant compatibility"
    fail=1
fi

if grep -n '#include "backend_quant.h"\|bn_backend_quant_matmul.*gpu_buf' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill execution code must use prefill policy helpers for GPU-resident quant matmul dispatch"
    fail=1
fi

if ! grep -n 'BN_CPU_DISABLE_PREPARED_QWEIGHTS\|BN_DUMP_LAYER_INP\|BN_DUMP_LAYER_POS\|BN_DUMP_ALL_HEADS\|BN_CPU_REFERENCE_DOT\|BN_CPU_REFERENCE_Q4_DOT\|BN_CPU_LLAMA_DOT\|BN_CPU_LLAMA_Q4_DOT' src/transformer/cpu_policy.c >/dev/null 2>&1; then
    echo "CPU env compatibility policy must live in src/transformer/cpu_policy.c"
    fail=1
fi

if grep -n 'BN_AVX512_Q5K_VNNI\|BN_AVX2_KQUANT_FLOAT\|BN_CPU_REFERENCE_DOT\|BN_CPU_REFERENCE_Q4_DOT\|BN_CPU_REFERENCE_Q6_DOT\|BN_CPU_LLAMA_DOT\|BN_CPU_LLAMA_Q4_DOT\|BN_CPU_LLAMA_Q6_DOT\|BN_WASM_Q4_CANONICAL4\|BN_DISABLE_Q8_0_MATMUL_BATCH' src/quant/dispatch.c src/quant/matvec_batch.c src/quant/matvec_multi.c src/quant/matmul.c src/quant/fused_gateup.c >/dev/null 2>&1; then
    echo "quant dispatch/kernels must use quant policy helpers for compatibility env vars"
    fail=1
fi

if ! grep -n 'BN_AVX512_Q5K_VNNI\|BN_AVX2_KQUANT_FLOAT\|BN_CPU_REFERENCE_DOT\|BN_CPU_REFERENCE_Q4_DOT\|BN_CPU_REFERENCE_Q6_DOT\|BN_CPU_LLAMA_DOT\|BN_CPU_LLAMA_Q4_DOT\|BN_CPU_LLAMA_Q6_DOT\|BN_WASM_Q4_CANONICAL4\|BN_DISABLE_Q8_0_MATMUL_BATCH' src/quant/policy.c >/dev/null 2>&1; then
    echo "quant env compatibility policy must live in src/quant/policy.c"
    fail=1
fi

if grep -n 'BN_GPU_Q4_Q8_DISABLE_GATEUP' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for Q4/Q8 fused gate-up compatibility env vars"
    fail=1
fi

if grep -n 'BN_GPU_CPU_FALLBACK_LAYER\|BN_GPU_CPU_FALLBACK_FROM_LAYER\|BN_GPU_CPU_ATTN_LAYER\|BN_GPU_CPU_ATTN_FROM_LAYER\|BN_GPU_CPU_FFN_LAYER\|BN_GPU_CPU_FFN_FROM_LAYER\|BN_GPU_CPU_FFN_DOWN_FROM_LAYER\|BN_GPU_Q4_Q8_FROM_LAYER\|BN_GPU_Q4_Q8_TO_LAYER\|BN_GPU_Q4_Q8_TAIL_NATIVE\|BN_GPU_Q4_Q8_ATTN_ONLY\|BN_GPU_Q4_Q8_FFN_ONLY\|BN_METAL_Q4_PREPARED' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for CPU fallback and Q4/Q8 layer env vars"
    fail=1
fi

if grep -n 'BN_GPU_MOE_ROUTE_PROFILE\|BN_GPU_MOE_ROUTE_PROFILE_EVERY' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for MoE route profile env vars"
    fail=1
fi

if grep -n 'BN_GPU_COMPARE_ATTENTION_LAYER\|BN_GPU_COMPARE_ATTENTION_POS\|BN_GPU_COMPARE_GQA_LAYER\|BN_GPU_COMPARE_GQA_POS\|BN_GPU_COMPARE_QKV_LAYER\|BN_GPU_COMPARE_QKV_POS\|BN_GPU_COMPARE_FFN_DOWN_LAYER\|BN_GPU_COMPARE_FFN_DOWN_POS\|BN_GPU_COMPARE_FFN_STATE_LAYER\|BN_GPU_COMPARE_FFN_STATE_POS' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for compare layer env vars"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_MOE_FFN' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for MoE FFN disable env vars"
    fail=1
fi

if grep -n 'bn_transformer_gpu_all2_q4_moe_requires_opt_in\|bn_transformer_gpu_cuda_all2_q4q6_moe_layer(c, lw' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU-aware MoE FFN fallback and route policy helpers"
    fail=1
fi

if grep -n 'BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK\|BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for shared MoE CPU fallback env vars"
    fail=1
fi

if grep -n 'BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL\|BN_GPU_COMPARE_MOE_LAYER\|BN_GPU_COMPARE_MOE_POS' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for MoE override and compare selector env vars"
    fail=1
fi

if grep -n 'BN_GPU_COMPARE_MOE_INPUT_NORM\|BN_GPU_COMPARE_MOE_ACTUAL\|BN_GPU_COMPARE_MOE_ROUTE\|BN_GPU_COMPARE_MOE_RAW\|BN_GPU_COMPARE_MOE_MID\|BN_GPU_COMPARE_MOE_PARTS\|BN_GPU_COMPARE_MOE_SHARED_MID\|BN_GPU_COMPARE_MOE_SHARED_DOWN\|BN_GPU_COMPARE_MOE_NORM\|BN_GPU_CPU_LOGITS\|BN_GPU_DEBUG_ARGMAX_COMPARE\|BN_GPU_COMPARE_LOGITS' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for MoE/logits debug env vars"
    fail=1
fi

if grep -n 'BN_GPU_PROFILE' src/transformer/gpu_fallback.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_fallback.c must use GPU policy helpers for GPU profile env vars"
    fail=1
fi

if grep -n 'BN_GPU_PROFILE' src/gpu_wgpu.c src/gpu_metal.m >/dev/null 2>&1; then
    echo "GPU backends must use GPU policy helpers for GPU profile env vars"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_CUBLAS_MATMUL\|BN_CUDA_DISABLE_Q6K_CUBLAS_F16\|BN_CUDA_CUBLAS_CACHE_MAX_MB\|BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for cublas aux-cache env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MATVEC")\|getenv("BN_CUDA_DISABLE_Q8_0")\|getenv("BN_CUDA_DISABLE_Q5_0")\|getenv("BN_CUDA_DISABLE_Q4_K")\|getenv("BN_CUDA_DISABLE_Q5_K")\|getenv("BN_CUDA_DISABLE_Q6_K")\|getenv("BN_CUDA_DISABLE_Q8_K")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for matvec type disable env vars"
    fail=1
fi

if grep -n 'BN_GGUF_TENSOR_' src/gpu_policy.c >/dev/null 2>&1; then
    echo "src/gpu_policy.c must use backend quant helpers for tensor-format policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_Q8_0_QUANT_MATMUL")\|getenv("BN_CUDA_DISABLE_Q8_0_QUANT_MATMUL")\|getenv("BN_CUDA_DISABLE_F16_Q8_0_MATMUL")\|getenv("BN_CUDA_ENABLE_Q8_0_PREQ_SPLIT")\|getenv("BN_CUDA_DISABLE_Q8_0_PREQ_SPLIT")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for Q8_0 matmul env vars"
    fail=1
fi

if grep -n 'BN_GPU_DEBUG_ARGMAX' src/generate.c >/dev/null 2>&1; then
    echo "src/generate.c must use GPU policy helpers for argmax debug env vars"
    fail=1
fi

if grep -n 'BN_CUDA_MOE_PREFILL_MIN_TOKENS\|BN_CUDA_DISABLE_MOE_CACHE_PREFILL\|BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE' src/moe_prefill.c >/dev/null 2>&1; then
    echo "src/moe_prefill.c must use GPU policy helpers for MoE prefill env vars"
    fail=1
fi

if grep -n 'BN_CUDA_DEBUG_MOE_ROUTE_BATCH' src/moe_prefill.c >/dev/null 2>&1; then
    echo "src/moe_prefill.c must use GPU policy helpers for MoE route debug env vars"
    fail=1
fi

if grep -n 'n_experts == 2 && K == 2\|n_experts > 2' src/moe_prefill.c >/dev/null 2>&1; then
    echo "src/moe_prefill.c must use model_arch helpers for MoE shape policy"
    fail=1
fi

if grep -n 'BN_GPU_BACKEND_CUDA\|kind == .*CUDA\|bn_transformer_gpu_cuda_moe_prefill_min_tokens' src/moe_prefill.c >/dev/null 2>&1; then
    echo "src/moe_prefill.c must use GPU policy helpers for MoE prefill backend policy"
    fail=1
fi

if grep -n 'bn_transformer_gpu_all2_q4_moe_requires_opt_in\|bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed' src/moe_prefill.c >/dev/null 2>&1; then
    echo "src/moe_prefill.c must use behavior-named GPU policy helpers for MoE prefill eligibility"
    fail=1
fi

if grep -n 'BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE\|BN_GPU_BACKEND_CUDA\|kind == .*CUDA\|bn_quant_format_cuda_lazy_moe_aux_cache_candidate' src/gpu_moe_bridge.c >/dev/null 2>&1; then
    echo "src/gpu_moe_bridge.c must use GPU policy helpers for MoE lazy aux cache policy"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_MOE_ROUTED_FFN\|BN_CUDA_ENABLE_MOE_ALL_F16_CACHE\|BN_CUDA_DISABLE_MOE_ALL_F16_CACHE\|BN_CUDA_ENABLE_MOE_GATEUP_F16_CACHE\|BN_CUDA_DISABLE_MOE_GATEUP_F16_CACHE\|BN_CUDA_ENABLE_PARTIAL_MOE_F16_CACHE\|BN_CUDA_DEBUG_MOE_FIT\|BN_CUDA_KEEP_INDIVIDUAL_F16_CACHE\|BN_CUDA_MOE_FULL_RESERVE_MB' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use GPU policy helpers for CUDA MoE residency env vars"
    fail=1
fi

if grep -n 'BN_CUDA_ENABLE_Q6K_LOGITS_F32_CACHE\|BN_CUDA_DISABLE_Q6K_LOGITS_F32_CACHE\|BN_CUDA_ENABLE_LOGITS_F16_CACHE\|BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_CACHE\|BN_CUDA_ENABLE_Q4K_MOE_DOWN_F32_CACHE\|BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE\|BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE\|BN_CUDA_CUBLAS_CACHE_MAX_MB\|BN_CUDA_DISABLE_CUBLAS_MATMUL\|BN_CUDA_DISABLE_Q6K_CUBLAS_F16\|BN_CUDA_LAYOUT_RESERVE_MB' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use GPU policy helpers for CUDA cache env vars"
    fail=1
fi

if grep -n 'BN_GPU_BACKEND_CUDA\|kind == .*CUDA' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use GPU policy helpers for CUDA backend policy"
    fail=1
fi

if grep -n 'n_experts == 2 && c->n_experts_active == 2\|c->n_experts > 2' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use model_arch helpers for MoE shape policy"
    fail=1
fi

if grep -n 'c->n_experts > 2' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for MoE shape policy"
    fail=1
fi

if grep -n 'c->n_experts > 0 || c->full_attn_interval > 0 ||' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for small dense CUDA shape policy"
    fail=1
fi

if grep -n 'c && c->n_experts > 0 && c->full_attn_interval <= 0' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for non-hybrid MoE policy"
    fail=1
fi

if grep -n 'c->n_experts > 0 || c->full_attn_interval > 0' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for dense attention-only policy"
    fail=1
fi

if grep -n 'c->n_experts > 0 || c->dim < 4096' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for large dense shape policy"
    fail=1
fi

if grep -n 'c->dim >= 4096' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for large graph fallback shape policy"
    fail=1
fi

if grep -n 'c->dim <= 2560 &&' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for small dense CUDA shape policy"
    fail=1
fi

if grep -n 'c->dim > 2560 || c->dim <= 1024' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for small dense Q8 native shape policy"
    fail=1
fi

if grep -n 'c && c->n_experts > 0 && c->full_attn_interval > 0' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must compose model_arch helpers for hybrid MoE policy"
    fail=1
fi

if grep -n 'bn_quant_format_gpu_float_buffer_type\|bn_quant_format_supports_moe_q4_down_route\|bn_quant_format_supports_moe_q8_route' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use GPU policy helpers for GPU upload quant-format policy"
    fail=1
fi

if grep -n 'bn_quant_format_supported\|bn_quant_format_uses_embedded_scale\|bn_quant_format_has_embedded_tensor_scale\|bn_quant_embedded_tensor_scale_offset\|bn_quant_format_is_f32\|bn_quant_format_can_convert_dense_to_f32\|bn_quant_format_convert_dense_to_f32\|bn_quant_format_dense_f32_type' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model policy helpers for load-time quant-format policy"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_CUBLAS_MATMUL\|BN_CUDA_DISABLE_Q6K_CUBLAS_F16\|BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE\|BN_CUDA_DISABLE_MOE_ROUTED_FFN\|BN_GPU_MOE_DISABLE_AUTO_RESIDENT\|BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE\|BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE\|BN_METAL_ENABLE_MMAP_ZERO_COPY\|BN_GPU_BACKEND_CUDA\|bn_quant_format_cuda_moe_down_cublas_cache_supported\|bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use GPU policy helpers for GPU cache/env policy"
    fail=1
fi

if grep -n 'BN_CUDA_DEBUG_PREFILL_MOE_CHAIN\|BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU policy helpers for CUDA prefill chain debug env vars"
    fail=1
fi

if grep -n 'BN_PREFILL_PROFILE\|BN_PREFILL_ALLOW_HYBRID_BATCH\|BN_PREFILL_FORCE_TOKEN_ATTN' src/transformer/prefill.c src/transformer/prefill_backend.c >/dev/null 2>&1; then
    echo "Prefill orchestration/backend code must use prefill policy helpers for prefill env vars"
    fail=1
fi

if ! grep -n 'BN_PREFILL_PROFILE\|BN_PREFILL_ALLOW_HYBRID_BATCH\|BN_PREFILL_FORCE_TOKEN_ATTN' src/transformer/prefill_policy.c >/dev/null 2>&1; then
    echo "Prefill env compatibility policy must live in src/transformer/prefill_policy.c"
    fail=1
fi

if grep -n 'BN_GPU_DISABLE_FUSED_GATEUP\|BN_GPU_DISABLE_GATEUP_SPLIT\|BN_GPU_Q4_Q8_DISABLE_FFN_DOWN' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for fused gate-up and split compatibility env vars"
    fail=1
fi

if grep -n 'BN_GPU_DISABLE_QKV_SPLIT\|BN_GPU_DISABLE_SSM_QKVZ_SPLIT\|BN_GPU_DISABLE_SSM_AB_STACK\|BN_GPU_DEBUG_QKV_SPLIT\|BN_GPU_SPLIT_RESIDUAL_RMSNORM\|BN_GPU_DEBUG_FALLBACK' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for QKV/SSM split and debug compatibility env vars"
    fail=1
fi

if grep -n 'BN_GPU_DEBUG_FALLBACK' src/gpu_graph_lowering_internal.h >/dev/null 2>&1; then
    echo "src/gpu_graph_lowering_internal.h must receive GPU debug policy as an argument"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT\|BN_CUDA_DISABLE_SHARED_EXPERT_GATE' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for shared MoE compatibility env vars"
    fail=1
fi

for file in \
    src/model.c \
    src/model_gpu.c \
    src/transformer/gpu_emit.c \
    src/transformer/plan.c
do
    if grep -n 'bn_backend_quant_can_gpu_native\|bn_backend_quant_can_gpu_repack\|bn_backend_quant_gpu_float_buffer_type\|bn_backend_quant_dense_f32_type\|bn_backend_quant_already_f32\|bn_backend_quant_can_convert_dense_to_f32\|bn_backend_quant_convert_dense_to_f32' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format dense/GPU type helpers for quant-format policy"
        fail=1
    fi
done

for file in \
    include/backend_quant.h \
    src/transformer/gpu.c \
    src/transformer/gpu_fallback.c \
    test/test_gpu_backend.c
do
    if grep -n 'BN_BACKEND_QUANT_HAS_NATIVE_Q8X_QUANT\|bn_backend_quant_has_native_q8x_quant' "$file" >/dev/null 2>&1; then
        echo "$file must use transformer CPU feature helpers for native Q8x ISA policy"
        fail=1
    fi
done

for file in \
    src/model_gpu.c \
    src/gpu_moe_bridge.c \
    src/main.c
do
    if grep -n 'bn_backend_quant_cuda_logits_q6_f32_cache_supported\|bn_backend_quant_cuda_moe_all_f16_cache_supported\|bn_backend_quant_cuda_moe_down_q6_f32_cache_supported\|bn_backend_quant_cuda_moe_down_cublas_cache_supported\|bn_backend_quant_cuda_moe_down_cublas_cache_elem_bytes\|bn_backend_quant_cuda_moe_down_q4_f32_cache_supported\|bn_backend_quant_cuda_moe_quant_only_after_cache\|bn_backend_quant_cuda_lazy_moe_aux_cache_candidate\|bn_backend_quant_cuda_moe_prefers_quant_only\|bn_backend_quant_cuda_aux_cache_supported\|bn_backend_quant_cuda_aux_cache_can_use_f16\|bn_backend_quant_cuda_aux_cache_uses_f32\|bn_backend_quant_cuda_aux_cache_prefers_large_budget\|bn_quant_format_cuda_moe_down_q6_f32_cache_supported\|bn_quant_format_cuda_moe_quant_only_after_cache\|bn_quant_format_cuda_moe_prefers_quant_only\|bn_quant_format_cuda_aux_cache_supported\|bn_quant_format_cuda_aux_cache_can_use_f16\|bn_quant_format_cuda_aux_cache_uses_f32\|bn_quant_format_cuda_aux_cache_prefers_large_budget' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format CUDA cache helpers for quant-format policy"
        fail=1
    fi
done

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

if grep -n 'BN_GPU_BACKEND_CUDA\|BN_CUDA_\|bn_quant_format_supports_gpu_small_dense_q8\|bn_model_arch_cpu_force_float_kquant' src/transformer.c >/dev/null 2>&1; then
    echo "src/transformer.c must use GPU policy helpers for CUDA matvec fallback policy"
    fail=1
fi

if grep -n 'bn_transformer_gpu_backend_is_cuda(prefill_gpu)\|bn_transformer_gpu_backend_is_cuda(gpu_ffn)\|int cuda_hybrid_prefill\|int cuda_moe_prefill' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU policy helpers for prefill chain policy"
    fail=1
fi

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

for file in include/model_config.h include/model_weights.h include/model_run_state.h include/transformer_plan_internal.h
do
    if grep -n 'Gemma4\|Qwen\|gemma4_\|qwen2_moe\|qwen2moe_\|BN_MODEL_ARCH_POLICY_\|RMSNORM_LLAMA\|requires_llama_scalar' "$file" >/dev/null 2>&1; then
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
    src/transformer.c \
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
