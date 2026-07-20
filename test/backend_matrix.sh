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

if grep -n 'getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FLASH_FUSE")\|getenv("BN_CUDA_DISABLE_QK_NORM_ROPE_FUSE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA QK norm rope fuse env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_PROFILE_SHAPES")\|getenv("BN_CUDA_DEVICE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA profile/device env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_RMSNORM_FUSE")\|getenv("BN_CUDA_DISABLE_WEIGHTED_ADD_SIGMOID_RESIDUAL_FUSE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA weighted-add sigmoid fuse env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_CUBLAS_LOGITS")\|getenv("BN_CUDA_ENABLE_F32_LOGITS_MATVEC")\|getenv("BN_CUDA_DISABLE_F32_LOGITS_MATVEC")\|getenv("BN_CUDA_ENABLE_F16_LOGITS_MATVEC")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA logits matvec env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_Q5K_DEINT_PAIR_MATVEC")\|getenv("BN_CUDA_ENABLE_Q6K_Q4K_PAIR_MATVEC")\|getenv("BN_CUDA_DISABLE_Q6K_Q4K_PAIR_MATVEC")\|getenv("BN_CUDA_DISABLE_Q6K_MATVEC4")\|getenv("BN_CUDA_ENABLE_F16_Q6K_MATVEC")\|getenv("BN_CUDA_DISABLE_F16_Q6K_MATVEC")\|getenv("BN_CUDA_DISABLE_Q8_0_SSM_MATVEC")\|getenv("BN_CUDA_ENABLE_F16_Q8_0_SSM_MATVEC")\|getenv("BN_CUDA_DISABLE_F16_Q8_0_SSM_MATVEC")\|getenv("BN_CUDA_ENABLE_F16_Q8_0_MATVEC")\|getenv("BN_CUDA_DISABLE_F16_Q8_0_MATVEC")\|getenv("BN_CUDA_ENABLE_F16_Q5K_MATVEC")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA f16/pair matvec env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q4K_PAIR_MATVEC")\|getenv("BN_CUDA_ENABLE_Q6K_Q8_1_DOT")\|getenv("BN_CUDA_DISABLE_Q6K_Q8_1_DOT")\|getenv("BN_CUDA_ENABLE_Q6K_Q8_1_ALL")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ_512_2048")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ_1536_8960")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2560_9728")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ_LOGITS_1536")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_LOGITS_SMALL")\|getenv("BN_CUDA_DISABLE_Q6K_MMVQ_2WARP_1536")\|getenv("BN_CUDA_ENABLE_Q6K_DOWN_RESID_RMSNORM_FUSE")\|getenv("BN_CUDA_DISABLE_Q4K_Q8K_MATVEC4")\|getenv("BN_CUDA_DISABLE_Q4K_4WARP_1536_8960")\|getenv("BN_CUDA_DISABLE_Q4K_4WARP_2560_9728")\|getenv("BN_CUDA_ENABLE_Q4K_OUT_RESID_RMSNORM_FUSE")\|getenv("BN_CUDA_DISABLE_Q5K_4WARP")\|getenv("BN_CUDA_ENABLE_Q5K_SPLIT_4WARP")\|getenv("BN_CUDA_DISABLE_Q8_0_SSM_PREQ")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA Q4/Q6/Q8 matvec variant env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_Q4K_QKV_MIXED_FUSE")\|getenv("BN_CUDA_ENABLE_Q8_MIXED_PREQ")\|getenv("BN_CUDA_ENABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE")\|getenv("BN_CUDA_DISABLE_Q4K_SPLIT_K_ROPE_CACHE_FUSE")\|getenv("BN_CUDA_DISABLE_Q4K_SPLIT_QK_ROPE_CACHE_FUSE")\|getenv("BN_CUDA_DISABLE_Q4K_SPLIT_4WARP_2048")\|getenv("BN_CUDA_DISABLE_Q4K_SPLIT_5WARP_2560")\|getenv("BN_CUDA_ENABLE_Q4K_SPLIT_VALUE_FUSE_1792")\|getenv("BN_CUDA_DISABLE_Q4K_SPLIT_VALUE_FUSE")\|getenv("BN_CUDA_DISABLE_Q4K_GATEUP_Q8_1_FAST")\|getenv("BN_CUDA_DISABLE_Q4K_GATEUP_QWARP4")\|getenv("BN_CUDA_DISABLE_Q4K_GATEUP_5WARP_2560")\|getenv("BN_CUDA_DISABLE_Q4K_GATEUP_2WARP")\|getenv("BN_CUDA_DISABLE_Q5K_GATEUP_2WARP")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA QKV split/gate-up env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8K_PREQUANT")\|getenv("BN_CUDA_ENABLE_MOE_ROUTE_Q8_1_PREQUANT")\|getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT")\|getenv("BN_CUDA_ENABLE_MOE_Q4K_Q8K_DOT_ALL2")\|getenv("BN_CUDA_DISABLE_MOE_ROUTE_Q8_1_PREQUANT")\|getenv("BN_CUDA_ENABLE_MOE_ROUTER_FUSED_TOPK")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_FUSED_TOPK")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_4WARP")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_2WARP")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_WARP_TOPK")\|getenv("BN_CUDA_DISABLE_Q8_MOE_Q8X")\|getenv("BN_CUDA_DISABLE_MOE_ALL2_FAST")\|getenv("BN_CUDA_DISABLE_MOE_Q4K_Q8K_DOT")\|getenv("BN_CUDA_PROFILE_MOE_INTERNAL")\|getenv("BN_CUDA_DISABLE_MOE_Q4K_ALL2_FIXED")\|getenv("BN_CUDA_DISABLE_MOE_Q4K_GATEUP_4ROW")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA MoE route/gate-up env policy"
    fail=1
fi

if grep -n 'int routed_q4 = gate_type == BN_GGUF_TENSOR_Q4_K\|int routed_q8 = gate_type == BN_GGUF_TENSOR_Q8_0\|int routed_q4 = op->type == BN_GGUF_TENSOR_Q4_K\|int routed_q8 = op->type == BN_GGUF_TENSOR_Q8_0' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for routed MoE quant groups"
    fail=1
fi

if sed -n '/^static int cuda_ops_have_native_quant_moe_routed_ffn/,/^static int cuda_ops_have_moe_all_active_two_kquant_routed_ffn/p' src/gpu_cuda.cu | grep -n 'op->type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_ops_have_moe_all_active_two_kquant_routed_ffn/,/^static int cuda_ops_have_logits/p' src/gpu_cuda.cu | grep -n 'op->type == BN_GGUF_TENSOR_Q4_K\|(int)op->p\[3\] == BN_GGUF_TENSOR_Q6_K' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_ops_have_moe_cublas_decode/,/^static int cuda_ops_moe_max_experts/p' src/gpu_cuda.cu | grep -n 'op->type == BN_GGUF_TENSOR_Q4_K\|(int)op->p\[3\] == BN_GGUF_TENSOR_Q6_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for routed MoE graph quant predicates"
    fail=1
fi

if sed -n '/int reserve_q8_1_cols = 0;/,/cuda_ensure_q8_1/p' src/gpu_cuda.cu | grep -n 'rop->type == BN_GGUF_TENSOR_Q4_K\|rop->type == BN_GGUF_TENSOR_Q5_K\|rop->type == BN_GGUF_TENSOR_Q6_K\|rop->type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for graph scratch quant predicates"
    fail=1
fi

if sed -n '/bn_backend_quant_deinterleaved_kquant_pair_matvec/,/bn_gpu_policy_cuda_deinterleaved_kquant_pair_matvec_enabled/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_Q5_K' >/dev/null 2>&1 ||
   sed -n '/bn_backend_quant_asymmetric_kquant_pair_matvec/,/bn_gpu_policy_cuda_asymmetric_kquant_pair_matvec_enabled/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_Q6_K\|BN_GGUF_TENSOR_Q4_K' >/dev/null 2>&1 ||
   sed -n '/bn_backend_quant_symmetric_kquant_pair_matvec/,/bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_Q4_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA pair matvec quant predicates"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q5k_deint_pair_matvec_enabled\|bn_gpu_policy_cuda_q6k_q4k_pair_matvec_enabled\|bn_gpu_policy_cuda_q4k_pair_matvec_enabled' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA K-quant pair matvec policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q4k_dot_enabled\|bn_gpu_policy_cuda_q5k_dot_enabled\|bn_gpu_policy_cuda_q6k_dot_enabled\|bn_gpu_policy_cuda_q6k_dot_forced\|bn_gpu_policy_cuda_q6k_warp_enabled' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA K-quant dot policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q6k_4warp_long_enabled\|bn_gpu_policy_cuda_q6k_5warp_exact_enabled\|bn_gpu_policy_cuda_q6k_3warp_exact_enabled\|bn_gpu_policy_cuda_q6k_2warp_long_enabled\|bn_gpu_policy_cuda_q6k_matvec4_shape_disabled\|cuda_use_q6k_4warp_long\|cuda_use_q6k_5warp_exact\|cuda_use_q6k_3warp_exact\|cuda_use_q6k_2warp_long\|cuda_disable_q6k_matvec4_shape' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA down-K-quant shape policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q6k_cublas_f16_cache_enabled\|bn_gpu_policy_cuda_q6k_q8_1_dot_enabled\|bn_gpu_policy_cuda_q6k_mmvq_enabled\|bn_gpu_policy_cuda_q6k_mmvq_2warp_logits_enabled\|bn_gpu_policy_cuda_q6k_down_residual_rmsnorm_fuse_enabled\|bn_gpu_policy_cuda_f16_q6k_matvec_enabled\|bn_gpu_policy_cuda_q6k_matmul8_enabled\|bn_gpu_policy_cuda_q6k_matmul4_enabled\|bn_gpu_policy_cuda_q6k_matvec4_enabled\|bn_gpu_policy_cuda_q6k_batch_warp_enabled\|bn_gpu_policy_cuda_q6k_f16_cache_adds_f32_down_cache\|use_f16_q6k_matvec\|use_q6k_q8_1\|use_q6k_mmvq' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA down-K-quant execution policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q4k_4warp_enabled\|bn_gpu_policy_cuda_q4k_4warp_shape_enabled\|bn_gpu_policy_cuda_q4k_out_residual_rmsnorm_fuse_enabled\|bn_gpu_policy_cuda_q4k_qkv_mixed_fuse_enabled\|bn_gpu_policy_cuda_q4k_split_k_rope_cache_fuse_enabled\|bn_gpu_policy_cuda_q4k_split_qk_rope_cache_fuse_enabled\|bn_gpu_policy_cuda_q4k_split_4warp_enabled\|bn_gpu_policy_cuda_q4k_split_5warp_enabled\|bn_gpu_policy_cuda_q4k_split_value_rows\|bn_gpu_policy_cuda_q4k_split_value_fuse_enabled\|bn_gpu_policy_cuda_q4k_gateup_qwarp4_enabled\|bn_gpu_policy_cuda_q4k_gateup_5warp_enabled\|bn_gpu_policy_cuda_q4k_gateup_2warp_enabled\|bn_gpu_policy_cuda_q4k_gateup_4warp_enabled\|bn_gpu_policy_cuda_q4k_matmul8_enabled\|bn_gpu_policy_cuda_q4k_sharedx_enabled\|bn_gpu_policy_cuda_q4k_batch_sharedx_enabled\|bn_gpu_policy_cuda_prefill_fused_q4k_gateup_batch_enabled\|bn_gpu_policy_cuda_prefill_ssm_fused_q4k_gateup_batch_enabled\|enable_q4k_4warp' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA asymmetric K-quant policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q5_matvec4_enabled\|bn_gpu_policy_cuda_q5_warp_enabled\|bn_gpu_policy_cuda_q5k_4warp_enabled\|bn_gpu_policy_cuda_q5k_split_4warp_enabled\|bn_gpu_policy_cuda_q5k_gateup_2warp_enabled\|bn_gpu_policy_cuda_q5_gateup_warp_disabled\|enable_q5_matvec4\|enable_q5_warp\|disable_q5_gateup_warp' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA Q5 matvec/gate-up policy helpers must use behavior names"
    fail=1
fi

if sed -n '/int small_state_native_matvec =/,/bn_gpu_policy_cuda_symmetric_kquant_pair_matvec_enabled/p' src/gpu_cuda.cu | grep -n 'op->type == BN_GGUF_TENSOR_Q8_0\|op->type == BN_GGUF_TENSOR_Q3_K\|op->type == BN_GGUF_TENSOR_IQ3_XXS\|op->type == BN_GGUF_TENSOR_IQ4_XS\|op->type == BN_GGUF_TENSOR_Q5_K\|op->type == BN_GGUF_TENSOR_Q6_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA f16/logits matvec quant predicates"
    fail=1
fi

if sed -n '/^static int cuda_buffer_create_f16_cache/,/size_t bytes = n/p' src/gpu_cuda.cu | grep -n 'buf->type == BN_GGUF_TENSOR_Q4_K\|buf->type == BN_GGUF_TENSOR_Q6_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA aux cache quant policy"
    fail=1
fi

if sed -n '/bn_backend_quant_cuda_aux_cache_dequant_route/,/err = cudaGetLastError()/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA aux cache dequant routing"
    fail=1
fi

if sed -n '/^static int cuda_buffer_create_iq_f16_cache/,/bn_gpu_policy_cuda_cublas_matmul_enabled/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_IQ3_XXS\|BN_GGUF_TENSOR_IQ4_XS' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA IQ aux-cache eligibility"
    fail=1
fi

if sed -n '/^static int cuda_buffer_create_iq_f16_cache/,/^static void \*cuda_buffer_create_impl/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_IQ3_XXS\|BN_GGUF_TENSOR_IQ4_XS\|bn_quant_dequant_iq3xxs\|bn_quant_dequant_iq4xs' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA IQ aux-cache dequant routing"
    fail=1
fi

if sed -n '/^static BnCudaBuffer \*cuda_buffer_create_impl/,/^static void \*cuda_buffer_create(/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_IQ3_XXS\|BN_GGUF_TENSOR_IQ4_XS' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA IQ aux-cache routing"
    fail=1
fi

if sed -n '/^static int cuda_moe_ffn_batch/,/^static int cuda_moe_route_batch/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_F32' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_moe_route_batch/,/^static int cuda_moe_route_routed_ffn_batch/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_F32' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_moe_route_routed_ffn_batch_impl/,/size_t full_values/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_F32' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA MoE dense-F32 host validation"
    fail=1
fi

if sed -n '/^static int cuda_prefill_ssm(/,/int ab_preactivated = 0/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_F32' >/dev/null 2>&1 ||
   sed -n '/int ab_preactivated = 0;/,/BN_CUDA_SSM_PROFILE_STEP(BN_CUDA_SSM_PROF_AB)/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_F32' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA SSM dense-F32 alpha/beta policy"
    fail=1
fi

if sed -n '/bn_backend_quant_legacy_block_matvec_candidate/,/BN_CUDA_LAUNCH_STATIC(ctx, matvec_kernel/p' src/gpu_cuda.cu | grep -n 'op->type == BN_GGUF_TENSOR_Q5_0\|op->type == BN_GGUF_TENSOR_Q6_K\|op->type == BN_GGUF_TENSOR_Q4_K\|op->type == BN_GGUF_TENSOR_Q5_K\|op->type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA matvec dispatch quant predicates"
    fail=1
fi

if sed -n '/case BN_GPU_CODE_MATVEC:/,/case BN_GPU_CODE_MATVEC_SPLIT:/p' src/gpu_cuda.cu | grep -n 'op->type != BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA graph matvec bias quant predicates"
    fail=1
fi

if sed -n '/^static int cuda_matmul_device_out(/,/^static int cuda_matmul_batch(/p' src/gpu_cuda.cu | grep -n 'type == BN_GGUF_TENSOR_Q3_K\|type == BN_GGUF_TENSOR_IQ3_XXS\|type == BN_GGUF_TENSOR_IQ4_XS\|type == BN_GGUF_TENSOR_Q5_0\|type == BN_GGUF_TENSOR_Q6_K\|type == BN_GGUF_TENSOR_Q4_K\|type == BN_GGUF_TENSOR_Q5_K\|type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_matmul_batch(/,/^static int cuda_argmax_activation(/p' src/gpu_cuda.cu | grep -n 'type == BN_GGUF_TENSOR_Q5_0\|type == BN_GGUF_TENSOR_Q6_K\|type == BN_GGUF_TENSOR_Q4_K\|type == BN_GGUF_TENSOR_Q5_K\|type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for direct CUDA matvec/matmul dispatch quant predicates"
    fail=1
fi

if sed -n '/^static int cuda_matvec_argmax_activation(/,/^static int cuda_dense_ffn(/p' src/gpu_cuda.cu | grep -n 'type != BN_GGUF_TENSOR_Q6_K\|type == BN_GGUF_TENSOR_Q6_K' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_dense_ffn_batch_impl(/,/^static int cuda_dense_ffn_batch(/p' src/gpu_cuda.cu | grep -n 'gate_type == BN_GGUF_TENSOR_Q5_0\|gate_type == BN_GGUF_TENSOR_Q6_K\|gate_type == BN_GGUF_TENSOR_Q4_K\|gate_type == BN_GGUF_TENSOR_Q5_K\|gate_type == BN_GGUF_TENSOR_Q8_0\|up_type == BN_GGUF_TENSOR_Q5_0\|up_type == BN_GGUF_TENSOR_Q4_K\|up_type == BN_GGUF_TENSOR_Q5_K\|up_type == BN_GGUF_TENSOR_Q8_0\|down_type == BN_GGUF_TENSOR_Q5_0\|down_type == BN_GGUF_TENSOR_Q4_K\|down_type == BN_GGUF_TENSOR_Q5_K\|down_type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1 ||
   sed -n '/^static int cuda_prefill_attention_wo(/,/^static int cuda_prefill_qkv_attention_wo_impl(/p' src/gpu_cuda.cu | grep -n 'wo_type == BN_GGUF_TENSOR_Q5_0\|wo_type == BN_GGUF_TENSOR_Q6_K\|wo_type == BN_GGUF_TENSOR_Q4_K\|wo_type == BN_GGUF_TENSOR_Q5_K\|wo_type == BN_GGUF_TENSOR_Q8_0' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA FFN/prefill output dispatch quant predicates"
    fail=1
fi

if sed -n '/bn_gpu_policy_cuda_prefill_fused_asymmetric_kquant_gateup_batch_enabled/,/ffn_act_ready = 1/p' src/gpu_cuda.cu | grep -n 'gate_type == BN_GGUF_TENSOR_Q4_K' >/dev/null 2>&1 ||
   sed -n '/bn_gpu_policy_cuda_prefill_ssm_fused_asymmetric_kquant_gateup_batch_enabled/,/ffn_act_ready = 1/p' src/gpu_cuda.cu | grep -n 'ffn_gate_type == BN_GGUF_TENSOR_Q4_K' >/dev/null 2>&1 ||
   sed -n '/case BN_GPU_CODE_FUSED_GATEUP_SILU:/,/case BN_GPU_CODE_MOE_ROUTE_TOPK:/p' src/gpu_cuda.cu | grep -n 'op->type == BN_GGUF_TENSOR_Q5_0\|op->type == BN_GGUF_TENSOR_Q8_0\|op->type == BN_GGUF_TENSOR_Q4_K\|op->type == BN_GGUF_TENSOR_Q5_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA fused gate/up quant predicates"
    fail=1
fi

if sed -n '/case BN_GPU_CODE_MATVEC_SPLIT:/,/case BN_GPU_CODE_FUSED_GATEUP_SILU:/p' src/gpu_cuda.cu | grep -n 'op->type != BN_GGUF_TENSOR_Q8_0\|op->type == BN_GGUF_TENSOR_Q4_K\|op->type == BN_GGUF_TENSOR_Q5_K\|op->type == BN_GGUF_TENSOR_Q8_0\|scan->type == BN_GGUF_TENSOR_Q4_K\|scan->type == BN_GGUF_TENSOR_Q6_K\|vop->type == BN_GGUF_TENSOR_Q4_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA graph split matvec quant predicates"
    fail=1
fi

if sed -n '/case BN_GPU_CODE_MOE_ROUTE_TOPK:/,/case BN_GPU_CODE_MOE_ROUTED_FFN:/p' src/gpu_cuda.cu | grep -n 'next->type == BN_GGUF_TENSOR_Q4_K\|(int)next->p\[3\] == BN_GGUF_TENSOR_Q6_K' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA route-topk next-op quant predicates"
    fail=1
fi

if sed -n '/case BN_GPU_CODE_MOE_ROUTE_TOPK:/,/int next_moe_all_active_two_kquant =/p' src/gpu_cuda.cu | grep -n 'BN_GGUF_TENSOR_F32' >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use backend quant helpers for CUDA route-topk dense-F32 policy"
    fail=1
fi

if grep -n 'bn_backend_quant_cuda_q5k_deint_pair_matvec\|bn_backend_quant_cuda_q6q4_pair_matvec\|bn_backend_quant_cuda_q4_pair_matvec\|bn_backend_quant_cuda_q8_small_ssm_matvec_candidate\|bn_backend_quant_cuda_f16_q8_matvec_candidate\|bn_backend_quant_cuda_f16_float_cache_matvec_candidate\|bn_backend_quant_cuda_f16_q5k_matvec_candidate\|bn_backend_quant_cuda_f16_q6k_matvec_candidate\|bn_backend_quant_cuda_logits_q6_matvec_candidate\|bn_backend_quant_cuda_q5_0_matvec_candidate\|bn_backend_quant_cuda_q6k_q8k_matvec_candidate\|bn_backend_quant_cuda_q6k_warp_matvec_candidate\|bn_backend_quant_cuda_q4k_q8k_matvec_candidate\|bn_backend_quant_cuda_q4k_q8_1_matvec_candidate\|bn_backend_quant_cuda_q5k_q8_1_matvec_candidate\|bn_backend_quant_cuda_q8_0_preq_matvec_candidate\|bn_backend_quant_cuda_q8_0_warp_matvec_candidate\|bn_backend_quant_cuda_q4k_q8k_matmul_candidate\|bn_backend_quant_cuda_q4k_q8_1_matmul_candidate\|bn_backend_quant_cuda_q5k_q8_1_matmul_candidate\|bn_backend_quant_cuda_q6k_q8k_matmul_candidate\|bn_backend_quant_cuda_q8_0_matmul_candidate\|bn_backend_quant_cuda_q5_0_matmul_candidate\|bn_backend_quant_cuda_q5_0_fused_gateup_candidate\|bn_backend_quant_cuda_q8_0_fused_gateup_candidate\|bn_backend_quant_cuda_q4k_fused_gateup_q8k_candidate\|bn_backend_quant_cuda_q4k_fused_gateup_q8_1_candidate\|bn_backend_quant_cuda_q5k_fused_gateup_q8_1_candidate\|bn_backend_quant_cuda_matvec_allows_fused_bias\|bn_backend_quant_cuda_split_allows_fused_bias\|bn_backend_quant_cuda_q4k_split_q8k_candidate\|bn_backend_quant_cuda_q4k_split_q8_1_candidate\|bn_backend_quant_cuda_q5k_split_q8_1_candidate\|bn_backend_quant_cuda_q8_0_split_candidate\|bn_backend_quant_cuda_q4k_split_value_fuse_candidate\|bn_backend_quant_cuda_q6k_split_value_fuse_candidate\|bn_backend_quant_cuda_split_value_fuse_candidate\|bn_backend_quant_cuda_q5_0_pair_matmul\|bn_backend_quant_cuda_q8_0_pair_matmul\|bn_backend_quant_cuda_q4k_pair_matmul\|bn_backend_quant_cuda_q5k_pair_matmul\|bn_backend_quant_cuda_q6_logits_argmax_candidate' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use neutral backend quant candidate helpers"
    fail=1
fi

if ! grep -n '"avx512"' src/transformer/cpu_backend.c >/dev/null 2>&1 ||
   ! grep -n 'bn_transformer_cpu_backend_supports_float_kquant_prefill' src/transformer/plan.c >/dev/null 2>&1 ||
   ! grep -n '"avx512"' src/transformer/prefill_backend.c >/dev/null 2>&1 ||
   ! grep -n '"avx512"' src/transformer/kv_backend.c >/dev/null 2>&1; then
    echo "CPU backend matrix must expose AVX512 as an explicit backend"
    fail=1
fi

if grep -n 'BN_CPU_BACKEND_AVX2\|BN_CPU_BACKEND_AVX512' src/transformer/plan.c >/dev/null 2>&1; then
    echo "src/transformer/plan.c must use CPU backend helpers for AVX-family ISA policy"
    fail=1
fi

if sed -n '/^int bn_transformer_cpu_backend_supports_float_kquant_prefill/,/^}/p' \
    src/transformer/cpu_backend.c | grep -n 'BN_CPU_BACKEND_AVX2\|BN_CPU_BACKEND_AVX512' >/dev/null 2>&1; then
    echo "CPU float K-quant prefill support must use backend ops capability, not AVX placement checks"
    fail=1
fi

if grep -n 'BN_TRANSFORMER_CPU_HAS_' src/moe_policy.c >/dev/null 2>&1; then
    echo "src/moe_policy.c must use CPU backend helpers for ISA capability policy"
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

if grep -n 'activation == [012]\|act_type == [012]\|activation != [12]\|act_type != [12]' \
    src/transformer/cpu_policy.c \
    src/transformer/cpu_backend.c \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_emit.c \
    src/transformer/gpu_fallback.c \
    src/transformer/prefill_backend.c \
    src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "Transformer/CUDA activation policy must use named activation helpers"
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

if grep -n 'bn_backend_quant_moe_route_q4_down\|bn_backend_quant_moe_routed_q4' \
    include/backend_quant.h \
    src/gpu_policy.c \
    src/transformer/gpu_policy.c \
    src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "Backend quant MoE route helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_backend_quant_is_kquant_float_fallback_candidate' src/transformer/prefill_policy.c >/dev/null 2>&1; then
    echo "src/transformer/prefill_policy.c must use behavior-named float K-quant fallback capability helpers"
    fail=1
fi

if grep -n 'bn_quant_format_' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use backend_quant helpers for quant-format policy"
    fail=1
fi

if grep -n 'bn_quant_format_supports_q[68]_logits_refine\|bn_backend_quant_supports_q[68]k*_logits_refine' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for logits refine capability policy"
    fail=1
fi

if grep -n 'bn_backend_quant_supports_q8_logits_refine\|bn_backend_quant_supports_q6k_logits_refine\|bn_backend_quant_cpu_tied_q6k_\(refine\|hybrid\)_top' include/backend_quant.h src/backend_quant.c src/transformer/gpu_policy.c src/transformer/logits_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Backend logits refine capability helpers must use behavior names, not quant-format names"
    fail=1
fi

if grep -n 'bn_quant_format_pair_same_format\|bn_backend_quant_stacked_pair_same_format' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for stacked Q/K quant compatibility"
    fail=1
fi

if grep -n 'map->up_type == map->gate_type\|map->gate_type == map->up_type' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "GPU policy must use stacked gate/up quant compatibility helpers for MoE split policy"
    fail=1
fi

if grep -n 'attn\.wq\.rows ==\|attn\.wk\.rows ==\|attn\.wq\.cols == .*attn\.wk\.cols' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for stacked Q/K shape compatibility"
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

if grep -n 'supports_q8_refine\|bn_transformer_logits_q[68]_refine_supported\|bn_transformer_logits_small_backend\|logits_refine_q8\|logits_refine_tied_q6k\|logits_hybrid_tied_q6k\|logits_refine_small_backend' src/transformer/logits.c src/transformer/logits_policy.c include/transformer_logits_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer logits refine policy must expose behavior-named helpers, not quant-format helper names"
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

if awk '
    /int bn_quant_format_can_gpu_native\(/ { in_fn=1 }
    /int bn_quant_format_can_gpu_repack\(/ { in_fn=1 }
    /int bn_quant_format_gpu_requires_exact_silu\(/ { in_fn=1 }
    /int bn_quant_format_gpu_prefers_gateup_split\(/ { in_fn=1 }
    /int bn_quant_format_gpu_fused_gateup_requires_backend_opt_in\(/ { in_fn=1 }
    /int bn_quant_format_logits_q6_f32_cache_supported\(/ { in_fn=1 }
    /int bn_quant_format_moe_all_f16_cache_supported\(/ { in_fn=1 }
    /int bn_quant_format_moe_down_q6_f32_cache_supported\(/ { in_fn=1 }
    /int bn_quant_format_moe_down_cublas_cache_supported\(/ { in_fn=1 }
    /int bn_quant_format_moe_down_q4_f32_cache_supported\(/ { in_fn=1 }
    /int bn_quant_format_moe_quant_only_after_cache\(/ { in_fn=1 }
    /int bn_quant_format_lazy_moe_aux_cache_candidate\(/ { in_fn=1 }
    /int bn_quant_format_moe_prefers_quant_only\(/ { in_fn=1 }
    /int bn_quant_format_aux_cache_supported\(/ { in_fn=1 }
    /int bn_quant_format_aux_cache_can_use_f16\(/ { in_fn=1 }
    /int bn_quant_format_aux_cache_uses_f32\(/ { in_fn=1 }
    /int bn_quant_format_aux_cache_prefers_large_budget\(/ { in_fn=1 }
    /int bn_quant_format_uses_f16_logits_path\(/ { in_fn=1 }
    /int bn_quant_format_tied_logits_uses_quant_path\(/ { in_fn=1 }
    /int bn_quant_format_supports_logits_i8_cache\(/ { in_fn=1 }
    /int bn_quant_format_tied_logits_uses_f16_path\(/ { in_fn=1 }
    /int bn_quant_format_supports_moe_q4_down_route\(/ { in_fn=1 }
    /int bn_quant_format_supports_moe_q4_gateup\(/ { in_fn=1 }
    /int bn_quant_format_supports_cpu_fused_q4_gateup_silu\(/ { in_fn=1 }
    /int bn_quant_format_supports_moe_q8_route\(/ { in_fn=1 }
    in_fn && /BN_GGUF_TENSOR_/ { found=1 }
    in_fn && /^}/ { in_fn=0 }
    END { exit found ? 0 : 1 }
' src/quant/registry.c >/dev/null 2>&1; then
    echo "src/quant/registry.c must use quant capability flags for GPU behavior helpers"
    fail=1
fi

if grep -n 'bn_quant_format_pair_same_format\|bn_backend_quant_stacked_pair_same_format\|bn_quant_format_supports_moe_q4_gateup' src/transformer/gpu_emit.c >/dev/null 2>&1 ||
   grep -n 'bn_backend_quant_moe_gateup_q4' src/transformer/gpu_emit.c src/transformer/gpu_policy.c include/backend_quant.h >/dev/null 2>&1; then
    echo "transformer GPU code must use GPU policy helpers for stacked pair and shared gate-up quant-format policy"
    fail=1
fi

for file in \
    src/transformer/gpu.c \
    src/transformer/gpu_emit.c \
    src/transformer/plan.c
do
    if grep -n 'ffn_gate\.rows == .*ffn_up\.rows\|ffn_gate\.cols == .*ffn_up\.cols\|shared_gate\.rows == .*shared_up\.rows\|shared_gate\.cols == .*shared_up\.cols' "$file" >/dev/null 2>&1; then
        echo "$file must use GPU policy helpers for gate-up stackability policy"
        fail=1
    fi
done

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

if grep -n 'gpu_quant_lowering_internal\|bn_gpu_quant_split_op_code' src/gpu_moe_bridge.c >/dev/null 2>&1; then
    echo "src/gpu_moe_bridge.c must use GPU policy helpers for MoE gate-up split op policy"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_moe_gateup_split_enabled' src/gpu_moe_bridge.c >/dev/null 2>&1; then
    echo "src/gpu_moe_bridge.c must use behavior-named GPU policy helpers for MoE gate-up split eligibility"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_prefers_quant_only' src/gpu_moe_bridge.c >/dev/null 2>&1; then
    echo "src/gpu_moe_bridge.c must use behavior-named GPU policy helpers for quant-only expert upload policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_prefers_quant_only' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "MoE quant-only preference policy must expose/test behavior-named helpers, not CUDA implementation aliases"
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

if grep -n 'bn_gpu_quant_split_op_is_q4k\|bn_gpu_quant_split_op_is_q5k\|bn_gpu_quant_split_op_is_q8' src/gpu_quant_lowering_internal.h src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "GPU split op predicates must use behavior names, not exact tensor format names"
    fail=1
fi

if grep -n 'bn_transformer_gpu_moe_routed_q[48]\|bn_transformer_gpu_moe_routed_byte_quant\|allow_q4_down\|moe_routed_q8\|moe_routed_byte_quant\|routed_q[48]\|routed_byte_quant' \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_internal.h \
    src/transformer/prefill_policy.c \
    include/transformer_prefill_internal.h \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer MoE route policy must use behavior names, not routed Q4/Q8 helper names"
    fail=1
fi

if grep -n '\brouted_q[48]\b\|moe_routed_q8\|routed_byte_quant' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA/GPU MoE route policy must use behavior names, not routed Q4/Q8 helper names"
    fail=1
fi

if grep -n 'bn_transformer_gpu_small_dense_q8_cpu_attn\|bn_transformer_gpu_small_dense_byte_quant_cpu_attn' \
    src/transformer/gpu_policy.c \
    src/transformer/gpu_internal.h \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer small-dense CPU attention fallback policy must use behavior names, not quant-format helper names"
    fail=1
fi

if grep -n 'gpu_quant_lowering_internal\|bn_gpu_quant_split_op_code' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for split op policy"
    fail=1
fi

if grep -n 'bn_model_arch_activation_' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for activation classification"
    fail=1
fi

if grep -n 'BN_GPU_CODE_MATVEC_SPLIT\|BN_GPU_CODE_Q4K_MATVEC_SPLIT\|BN_GPU_CODE_Q8_MATVEC_SPLIT\|BN_GPU_CODE_Q5K_MATVEC_SPLIT\|BN_GPU_CODE_UNKNOWN' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must classify split op codes through GPU quant lowering helpers"
    fail=1
fi

if grep -n 'BN_GGUF_TENSOR_' src/gpu_quant_lowering_internal.h >/dev/null 2>&1; then
    echo "src/gpu_quant_lowering_internal.h must map quant split capabilities, not raw tensor identities"
    fail=1
fi

if grep -n 'BN_GPU_CODE_' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must classify backend shader op codes through GPU IR helpers"
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

if grep -n 'BN_CUDA_[^"]*QWEN' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must expose model-family CUDA env vars only as compatibility fallbacks for neutral policy helpers"
    fail=1
fi

if grep -n 'BN_CUDA_[^"]*\(QWEN\|GEMMA\)' \
    include/gpu_policy.h \
    src/gpu_cuda.cu \
    src/gpu_metal.m \
    src/transformer.c \
    src/transformer/*.c \
    shaders/metal/* >/dev/null 2>&1; then
    echo "model-family CUDA compatibility env vars must stay in backend GPU policy/tests, not execution or shared interfaces"
    fail=1
fi

if awk '/^int bn_gpu_policy_small_dense_native_quant_cpu_attention_safe_disabled/{flag=1} flag{print}' src/gpu_policy.c | grep -n 'BN_CUDA_[^"]*\(QWEN\|GEMMA\)' >/dev/null 2>&1; then
    echo "src/gpu_policy.c public helpers must keep model-family CUDA env aliases behind private compatibility helpers"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_BATCHED_GEMM")\|getenv("BN_CUDA_DEBUG_PREFILL_GEMM")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for prefill GEMM env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_CUBLAS_GEMM_ALGO")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for cuBLAS GEMM algorithm env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DEBUG_READBACK")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for readback debug env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DEBUG_CUBLAS_CACHE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for cublas cache debug env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP_F16_OUT")\|getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_VARIABLE")\|getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED_VARIABLE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE cuBLAS grouped gate/up env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GROUPED")\|getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GROUPED")\|getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GROUPED_SMALL")\|getenv("BN_CUDA_DISABLE_Q8_MOE_CUBLAS_GATEUP")\|getenv("BN_CUDA_ENABLE_MOE_CUBLAS_GATEUP")\|getenv("BN_CUDA_DISABLE_MOE_CUBLAS_GATEUP")\|getenv("BN_CUDA_DISABLE_MOE_CUBLAS_ALL2_FIXED")\|getenv("BN_CUDA_ENABLE_MOE_ROUTE_SORT")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE cuBLAS route/gate-up env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_PROFILE_MOE_PREFILL_INTERNAL")\|getenv("BN_CUDA_DISABLE_MOE_PREFILL_DIRECT_RESID_OUT")\|getenv("BN_CUDA_ENABLE_MOE_BATCH_FUSED_ROUTE_TOPK")\|getenv("BN_CUDA_DISABLE_MOE_BATCH_FUSED_ROUTE_TOPK")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE routed prefill env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_PROFILE_MOE_ROUTE_DIST")\|cuda_env_int("BN_CUDA_PROFILE_MOE_ROUTE_DIST_EVERY"\|getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GROUPED")\|getenv("BN_CUDA_DEBUG_MOE_CUBLAS_GATEUP")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE routed prefill profile/debug env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_CUBLAS_DECODE")\|getenv("BN_CUDA_DEBUG_MOE_CUBLAS_DECODE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE cuBLAS decode env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_FFN_BATCH")\|getenv("BN_CUDA_PROFILE_MOE_FFN_BATCH_INTERNAL")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE FFN batch env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_ROUTE_BATCH")\|getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN_BATCH")\|getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH")\|getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for MoE route batch env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MATMUL_BATCH")\|getenv("BN_CUDA_DISABLE_MATVEC_BATCH")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA batch execution env policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_matmul_batch_enabled\|bn_gpu_policy_cuda_matvec_batch_enabled' include/gpu_policy.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU batch execution policy must use neutral behavior helper names"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_DENSE_FFN")\|getenv("BN_CUDA_DISABLE_DENSE_FFN_BATCH")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA dense FFN env policy"
    fail=1
fi

if grep -n 'act_type != 0' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use activation policy helpers instead of raw SiLU activation sentinel checks"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_ARGMAX_FAST")\|getenv("BN_CUDA_ENABLE_OPTIMISTIC_ARGMAX_PENALTY")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA argmax env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_Q5_MATVEC4")\|getenv("BN_CUDA_ENABLE_Q5_WARP")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA Q5 matvec env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q4K_DOT")\|getenv("BN_CUDA_DISABLE_Q5K_DOT")\|getenv("BN_CUDA_DISABLE_Q4K_4WARP")\|getenv("BN_CUDA_DISABLE_Q8_WARP")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA Q4/Q5/Q8 matvec env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q4K_Q8K_DOT")\|getenv("BN_CUDA_ENABLE_Q4K_Q8K_DOT")\|getenv("BN_CUDA_ENABLE_Q4K_MATMUL8")\|getenv("BN_CUDA_DISABLE_Q4K_SHAREDX_BATCH")\|getenv("BN_CUDA_ENABLE_Q4K_SHAREDX_BATCH")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA Q4K shared matmul env policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q8k_input_cache_enabled\|bn_gpu_policy_cuda_q4k_gateup_q8k_path_enabled\|bn_gpu_policy_cuda_q4k_q8k_dot_enabled\|bn_gpu_policy_cuda_q4k_q8k_dot_forced\|bn_gpu_policy_cuda_q4k_q8k_matvec4_enabled' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "K-quant GPU policy helpers must use backend-neutral behavior names"
    fail=1
fi

if grep -n 'BN_CUDA_Q8K_INPUT_CACHE\|q8k_input_cache\|enable_q8k_input_cache\|reuse_q8k_input' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA prepared K-quant input cache locals must use behavior names"
    fail=1
fi

if ! grep -n 'bn_gpu_policy_prepared_kquant_input_cache_enabled' src/gpu_cuda.cu >/dev/null 2>&1 ||
   ! grep -n 'bn_gpu_policy_kquant_dot_enabled' src/gpu_cuda.cu >/dev/null 2>&1 ||
   ! grep -n 'bn_gpu_policy_kquant_gateup_prepared_path_enabled' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must compose K-quant GPU policy through behavior-named helpers"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q6K_DOT")\|getenv("BN_CUDA_ENABLE_Q6K_DOT")\|getenv("BN_CUDA_ENABLE_Q6K_WARP")\|getenv("BN_CUDA_ENABLE_Q6K_MATMUL8")\|getenv("BN_CUDA_DISABLE_Q6K_MATMUL4")\|getenv("BN_CUDA_ENABLE_Q6K_BATCH_WARP")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA Q6 matvec/matmul env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_FUSE_BIAS")\|getenv("BN_CUDA_DISABLE_ROPE_FLASH_FUSE")\|getenv("BN_CUDA_ENABLE_BIAS_ROPE_FLASH_FUSE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA bias/rope fuse env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_ATTN")\|getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA prefill attention admission env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_GEMM_ATTN")\|getenv("BN_CUDA_ENABLE_PREFILL_GEMM_ATTN")\|getenv("BN_CUDA_PREFILL_GEMM_ATTN_MIN_TOKENS")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA prefill GEMM attention env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_ATTN_WO")\|getenv("BN_CUDA_DISABLE_PREFILL_QKV_ATTN_WO")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA prefill attention-WO env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_MOE_LAYER")\|getenv("BN_CUDA_DISABLE_PREFILL_DENSE_LAYER")\|getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA prefill layer admission env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DEBUG_PREFILL_DENSE_LAYER")\|getenv("BN_CUDA_PREFILL_DENSE_PROFILE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA dense prefill env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_FUSED_Q4K_GATEUP_BATCH")\|getenv("BN_CUDA_ENABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH")\|getenv("BN_CUDA_DISABLE_PREFILL_SSM_FUSED_Q4K_GATEUP_BATCH")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA prefill fused Q4K gate/up env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_SSM_PROFILE")\|getenv("BN_CUDA_DISABLE_SSM_STACKED_PREFILL")\|getenv("BN_CUDA_DISABLE_SSM_STREAM_PREFILL")\|getenv("BN_CUDA_DISABLE_SSM_PREFILL_INPUT_ALIAS")\|getenv("BN_CUDA_DISABLE_SSM_F32_AB_PREFILL")\|getenv("BN_CUDA_DISABLE_SSM_PREFILL_SCAN")\|getenv("BN_CUDA_DISABLE_SSM_DELTA_128_WARP")\|getenv("BN_CUDA_SSM_FFN_PROFILE")\|getenv("BN_CUDA_ENABLE_SSM_FFN_GATEUP_F16_OUT")\|getenv("BN_CUDA_DISABLE_SSM_FFN_GATEUP_F16_OUT")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA SSM prefill env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX")\|getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX")\|getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536")\|getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_1WARP16_1536")\|getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_1WARP8_1536_UNROLL")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA MMVQ argmax variant env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DEBUG_NAN_VERBOSE")\|getenv("BN_CUDA_DISABLE_STREAM_EXEC")\|getenv("BN_CUDA_PROFILE")\|getenv("BN_CUDA_PROFILE_WALL")\|getenv("BN_CUDA_DEBUG_EXEC_FAIL")\|getenv("BN_CUDA_DEBUG_SYNC_EACH_OP")\|getenv("BN_CUDA_DEBUG_NAN")\|getenv("BN_CUDA_DUMP_OPS")\|getenv("BN_CUDA_DUMP_OPS_EVERY")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA execution debug/profile env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_QKV_MIXED_FUSE")\|getenv("BN_CUDA_DISABLE_QKV_KCACHE_FUSE")\|getenv("BN_CUDA_ENABLE_QKV_KPAIR_OPT")\|getenv("BN_CUDA_DISABLE_Q5_GATEUP_WARP")\|getenv("BN_CUDA_DISABLE_Q8_GATEUP_WARP")\|getenv("BN_CUDA_ENABLE_GRAPH_EXEC")\|getenv("BN_CUDA_ENABLE_UNSAFE_MOE_FFN")\|getenv("BN_CUDA_ENABLE_Q8_PREQ")\|getenv("BN_CUDA_DISABLE_Q8_PREQ")\|getenv("BN_CUDA_DISABLE_Q8_PREQ_LOGITS")\|getenv("BN_CUDA_DISABLE_Q8K_INPUT_CACHE")\|cuda_env_int("BN_CUDA_MOE_GRAPH_MAX_EXPERTS"\|getenv("BN_CUDA_DISABLE_GRAPH_EXEC")\|getenv("BN_CUDA_ENABLE_MOE_FFN")\|getenv("BN_CUDA_ENABLE_Q8_PREQ_LOGITS")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use GPU policy helpers for CUDA graph/QKV/Q8 prequant env policy"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_[^"]*QWEN\|gpu_env_enabled("BN_CUDA_[^"]*QWEN\|gpu_env_value("BN_CUDA_[^"]*QWEN\|gpu_policy_env_int("BN_CUDA_[^"]*QWEN' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must expose model-family CUDA env vars only as compatibility fallbacks for neutral policy helpers"
    fail=1
fi

if grep -n 'BN_CUDA_[^"]*ALL2_Q4Q6\|BN_CUDA_[^"]*QWEN2MOE' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use backend GPU policy helpers for all2 MoE compatibility env vars"
    fail=1
fi

if grep -n 'BN_CUDA_[^"]*SMALL_DENSE\|BN_CUDA_[^"]*SMALL_QWEN' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use backend GPU policy helpers for small-dense compatibility env vars"
    fail=1
fi

if grep -n 'gpu_policy_env_' src/gpu_policy.c >/dev/null 2>&1; then
    echo "src/gpu_policy.c must label compatibility env fallback helpers explicitly"
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

if grep -n 'bn_transformer_gpu_backend_is_cuda(gpu)\|bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in\|bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed' src/transformer/prefill.c >/dev/null 2>&1; then
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

if grep -n 'BN_GPU_MAX_STORAGE_BINDING_MB' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use backend GPU policy helpers for max storage binding env vars"
    fail=1
fi

if grep -n 'BN_GPU_ENABLE_Q6_LOGITS_REFINE\|BN_GPU_DISABLE_Q6_LOGITS_REFINE\|BN_GPU_Q6_Q8K_REFINE_TOP\|BN_GPU_ENABLE_Q8_LOGITS_REFINE\|BN_GPU_DISABLE_Q8_LOGITS_REFINE\|BN_GPU_Q8_REFINE_TOP' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use backend GPU policy helpers for logits refine env vars"
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

if grep -n 'BN_CPU_TIED_Q6K_REFINE_TOP\|BN_CPU_TIED_Q6K_HYBRID_TOP' src/transformer/logits_policy.c >/dev/null 2>&1; then
    echo "Q6K tied-logits top-N policy must live in backend quant policy"
    fail=1
fi

if ! grep -n 'BN_CPU_TIED_Q6K_REFINE_TOP\|BN_CPU_TIED_Q6K_HYBRID_TOP' src/backend_quant.c >/dev/null 2>&1; then
    echo "Q6K tied-logits top-N policy must live in src/backend_quant.c"
    fail=1
fi

if ! grep -n 'BN_CPU_NATIVE_TIED_LOGITS' src/transformer/logits_policy.c >/dev/null 2>&1; then
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

if grep -n 'bn_quant_format_supports_cpu_fused_q4_gateup_silu' src/transformer/cpu.c src/transformer/cpu_policy.c >/dev/null 2>&1 ||
   grep -n 'bn_backend_quant_cpu_fused_q4_gateup_silu' include/backend_quant.h src/transformer/cpu.c src/transformer/cpu_policy.c >/dev/null 2>&1; then
    echo "CPU execution code must use CPU backend policy helpers for fused gate-up quant capability"
    fail=1
fi

if grep -n 'bn_transformer_cpu_.*fused_q4_gateup_silu' \
    src/transformer/cpu.c \
    src/transformer/cpu_policy.c \
    include/transformer_cpu_backend_internal.h \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer CPU fused gate-up policy must use behavior names, not Q4 helper names"
    fail=1
fi

if grep -n '#include "backend_quant.h"\|bn_backend_quant_matvec.*gpu_buf' src/transformer/cpu.c >/dev/null 2>&1; then
    echo "CPU execution code must use CPU backend policy helpers for GPU-resident quant matvec dispatch"
    fail=1
fi

if grep -n 'bn_quant_format_supports_prepared_kquant\|bn_backend_quant_supports_prepared_kquant' src/transformer/cpu.c >/dev/null 2>&1; then
    echo "CPU execution code must use CPU backend policy helpers for preq8k quant capability"
    fail=1
fi

if grep -n 'bn_transformer_cpu_can_preq8k\|bn_transformer_cpu_route_preq8k\|attn_preq8k\|ssm_preq8k' \
    include/transformer_cpu_backend_internal.h \
    src/transformer/cpu_policy.c \
    src/transformer/cpu.c \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer CPU prepared K-quant route policy must use behavior names, not preq8k helper names"
    fail=1
fi

if grep -n 'supports_preq8k\|rmsnorm_q8k' \
    include/transformer_cpu_backend_internal.h \
    src/transformer/cpu_policy.c \
    src/transformer/cpu.c \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer CPU backend ops must use prepared K-quant behavior names, not preq8k/Q8K field names"
    fail=1
fi

if grep -n 'cpu_quant_matvec_batch_preq8k' src/transformer/cpu.c >/dev/null 2>&1; then
    echo "Transformer CPU prepared K-quant execution helpers must use behavior names, not preq8k helper names"
    fail=1
fi

if grep -n 'bn_quant_format_supports_prepared_kquant\|bn_backend_quant_supports_prepared_kquant' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill execution code must use prefill policy helpers for preq8k quant capability"
    fail=1
fi

if grep -n 'bn_transformer_prefill_can_preq8k\|bn_transformer_prefill_route_preq8k\|used_preq8k\|ssm_preq8k' \
    include/transformer_prefill_internal.h \
    src/transformer/prefill_policy.c \
    src/transformer/prefill.c \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer prefill prepared K-quant route policy must use behavior names, not preq8k helper names"
    fail=1
fi

if grep -n 'supports_preq8k\|prepare_preq8k\|prefill_prepare_preq8k' \
    include/transformer_prefill_internal.h \
    src/transformer/prefill_policy.c \
    src/transformer/prefill_backend.c \
    src/transformer/prefill.c \
    test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer prefill backend ops must use prepared K-quant behavior names, not preq8k helper names"
    fail=1
fi

if grep -n 'BN_QUANT_CAP_CPU_PREQ8K\|BN_QUANT_CAP_LOADABLE_CPU_EMBEDDED_PREQ8K\|bn_quant_format_can_preq8k\|bn_backend_quant_can_preq8k' \
    include/quant.h \
    include/backend_quant.h \
    src/quant/registry.c \
    src/transformer/cpu_policy.c \
    src/transformer/prefill_policy.c \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Prepared K-quant capability helpers must not use preq8k implementation names"
    fail=1
fi

if grep -n 'bn_quant_matvec_batch_preq8k\|bn_quant_matmul_preq8k' \
    include/quant.h \
    src/quant/batch_prepared_kquant_input.c \
    src/quant/matmul.c \
    src/transformer/cpu.c \
    src/transformer/prefill.c \
    test/test_quant.c >/dev/null 2>&1; then
    echo "Prepared K-quant public quant APIs must not use preq8k implementation names"
    fail=1
fi

if [ -e src/quant/batch_preq8k.c ]; then
    echo "Prepared K-quant source file must use behavior names, not preq8k implementation names"
    fail=1
fi

if grep -n 'BnPrefillPreQ8KBuffers\|prefill_quant_matmul_preq8k_multi\|prefill_preq8k_arena_bytes\|prefill_alloc_preq8k_buffers\|prefill_prepare_preq8k\|prefill_try_preq8k' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill execution prepared K-quant helpers must use behavior names, not preq8k helper names"
    fail=1
fi

if grep -n 'bn_quant_format_supports_prepared_kquant\|bn_backend_quant_supports_prepared_kquant' src/moe_cpu_kernels.c >/dev/null 2>&1; then
    echo "MoE CPU kernels must use MoE policy helpers for preq8k quant capability"
    fail=1
fi

if grep -n 'shared_gate_type != batch_type\|shared_up_type != batch_type' src/moe_cpu_kernels.c >/dev/null 2>&1; then
    echo "MoE CPU kernels must use MoE policy helpers for ISA-specific shared gate/up batch compatibility"
    fail=1
fi

if grep -n 'bn_quant_format_supports_prepared_kquant\|bn_backend_quant_supports_prepared_kquant' src/moe_policy.c >/dev/null 2>&1; then
    echo "MoE policy must use shared gate/up batch quant policy helpers for preq8k compatibility"
    fail=1
fi

if grep -n 'bn_quant_format_has_embedded_tensor_scale\|bn_quant_embedded_tensor_scale_offset' src/moe_math.c >/dev/null 2>&1; then
    echo "MoE math must use MoE policy helpers for embedded tensor scale policy"
    fail=1
fi

if grep -n 'bn_quant_format_has_embedded_tensor_scale\|bn_quant_embedded_tensor_scale_offset' src/moe_policy.c >/dev/null 2>&1; then
    echo "MoE policy must use backend quant helpers for embedded tensor scale policy"
    fail=1
fi

if awk '
    /bn_quant_format_has_embedded_tensor_scale\(/ { in_fn=1 }
    in_fn && /BN_GGUF_TENSOR_/ { bad=1 }
    in_fn && /^}/ { in_fn=0 }
    END { exit bad ? 0 : 1 }
' src/quant/registry.c >/dev/null 2>&1; then
    echo "Quant embedded tensor-scale policy must use registry metadata, not tensor-specific checks"
    fail=1
fi

if awk '
    /static int cuda_type_supported\(/ { in_fn=1 }
    in_fn && /BN_GGUF_TENSOR_/ { bad=1 }
    in_fn && /^}/ { in_fn=0 }
    END { exit bad ? 0 : 1 }
' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA type support must use GPU policy helpers, not tensor-specific checks"
    fail=1
fi

if awk '
    /static int cuda_buffer_create_f16_cache\(/ { in_fn=1 }
    in_fn && /buf->type != BN_GGUF_TENSOR_/ { bad=1 }
    in_fn && /bn_gpu_policy_cuda_cublas_aux_cache_supported/ { saw_policy=1 }
    in_fn && /^}/ { in_fn=0 }
    END { exit (bad || !saw_policy) ? 0 : 1 }
' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA aux-cache eligibility must use GPU policy helpers, not tensor-specific whitelists"
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

if grep -n 'attn\.wq\.cols != dim\|attn\.wk\.cols != dim\|attn\.wv\.cols != dim\|q_stride < .*attn\.wq\.rows' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Prefill execution code must use prefill policy helpers for stacked Q/K shape compatibility"
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

if grep -n 'BN_MATVEC_TASK_LLAMA_DOT' include/quant.h src test/test_*.c >/dev/null 2>&1; then
    echo "Matvec task flags must use behavior names, not model-family aliases"
    fail=1
fi

if ! grep -n 'BN_AVX512_Q5K_VNNI\|BN_AVX2_KQUANT_FLOAT\|BN_CPU_REFERENCE_DOT\|BN_CPU_REFERENCE_Q4_DOT\|BN_CPU_REFERENCE_Q6_DOT\|BN_CPU_LLAMA_DOT\|BN_CPU_LLAMA_Q4_DOT\|BN_CPU_LLAMA_Q6_DOT\|BN_WASM_Q4_CANONICAL4\|BN_DISABLE_Q8_0_MATMUL_BATCH' src/quant/policy.c >/dev/null 2>&1; then
    echo "quant env compatibility policy must live in src/quant/policy.c"
    fail=1
fi

if grep -n '== BN_GGUF_TENSOR_\|!= BN_GGUF_TENSOR_' include/backend_quant.h >/dev/null 2>&1; then
    echo "backend quant wrappers must use quant-format helpers, not raw tensor identity checks"
    fail=1
fi

if grep -n 'bn_backend_quant_cuda_\|BN_BACKEND_QUANT_CUDA_AUX_CACHE_DEQUANT_\|BnBackendQuantCudaAuxCacheDequant' include/backend_quant.h >/dev/null 2>&1; then
    echo "backend quant wrappers must expose backend-neutral helper names"
    fail=1
fi

if grep -n 'bn_backend_quant_q5k_deint_pair_matvec\|bn_backend_quant_q6q4_pair_matvec\|bn_backend_quant_q4_pair_matvec\|bn_backend_quant_q4k_split_value_fuse_candidate\|bn_backend_quant_q6k_split_value_fuse_candidate' include/backend_quant.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "backend quant K-quant pair/split-value helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_backend_quant_q8_small_ssm_matvec_candidate\|bn_backend_quant_f16_q8_matvec_candidate\|bn_backend_quant_f16_q5k_matvec_candidate\|bn_backend_quant_f16_q6k_matvec_candidate\|bn_backend_quant_logits_q6_matvec_candidate\|bn_backend_quant_q6_logits_argmax_candidate\|bn_backend_quant_q8_0_prepared_input_matvec_candidate\|bn_backend_quant_q8_0_warp_matvec_candidate\|bn_backend_quant_q8_0_split_candidate\|bn_backend_quant_q6k_q8k_matvec_candidate\|bn_backend_quant_q6k_warp_matvec_candidate\|bn_backend_quant_q4k_q8k_matvec_candidate\|bn_backend_quant_q4k_q8_1_matvec_candidate\|bn_backend_quant_q5k_q8_1_matvec_candidate\|bn_backend_quant_q5_0_matvec_candidate' include/backend_quant.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "backend quant matvec candidate helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_backend_quant_q4k_q8k_matmul_candidate\|bn_backend_quant_q4k_q8_1_matmul_candidate\|bn_backend_quant_q5k_q8_1_matmul_candidate\|bn_backend_quant_q6k_q8k_matmul_candidate\|bn_backend_quant_q8_0_matmul_candidate\|bn_backend_quant_q5_0_matmul_candidate\|bn_backend_quant_q5_0_fused_gateup_candidate\|bn_backend_quant_q8_0_fused_gateup_candidate\|bn_backend_quant_q4k_fused_gateup_q8k_candidate\|bn_backend_quant_q4k_fused_gateup_q8_1_candidate\|bn_backend_quant_q5k_fused_gateup_q8_1_candidate\|bn_backend_quant_q4k_split_q8k_candidate\|bn_backend_quant_q4k_split_q8_1_candidate\|bn_backend_quant_q5k_split_q8_1_candidate\|bn_backend_quant_q5_0_pair_matmul\|bn_backend_quant_q8_0_pair_matmul\|bn_backend_quant_q4k_pair_matmul\|bn_backend_quant_q5k_pair_matmul' include/backend_quant.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "backend quant matmul/fused/split helper names must use behavior names"
    fail=1
fi

if grep -n 'bn_backend_quant_gpu_graph_gateup_needs_q8_1_scratch\|bn_backend_quant_gpu_graph_matvec_needs_q8_1_scratch\|bn_backend_quant_gpu_graph_matvec_q6_needs_q8k_scratch\|bn_backend_quant_gpu_graph_matvec_q4_needs_q8k_scratch' include/backend_quant.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "backend quant GPU graph scratch helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_backend_quant_moe_route_q8\|bn_backend_quant_moe_routed_q8\|bn_backend_quant_moe_routed_op_is_q8\|bn_backend_quant_moe_down_is_q6k\|bn_backend_quant_moe_down_is_q4k\|bn_backend_quant_moe_down_is_q4k_or_q6k' include/backend_quant.h src/gpu_cuda.cu src/gpu_policy.c src/transformer/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "backend quant MoE helper names must describe behavior, not exact tensor formats"
    fail=1
fi

if grep -n 'bn_quant_format_cuda_matvec_supported\|bn_quant_policy_cuda_matvec_type_disabled\|bn_quant_format_cuda_cublas_aux_cache_supported\|bn_quant_format_cuda_q8_quant_matmul_on_f16_disable\|bn_quant_format_cuda_force_q[46]k_quant_matmul_candidate\|bn_quant_format_cuda_\(logits_q6_f32_cache_supported\|moe_all_f16_cache_supported\|moe_down_q6_f32_cache_supported\|moe_down_cublas_cache_supported\|moe_down_cublas_cache_elem_bytes\|moe_down_q4_f32_cache_supported\|moe_quant_only_after_cache\|lazy_moe_aux_cache_candidate\|moe_prefers_quant_only\|aux_cache_supported\|aux_cache_can_use_f16\|aux_cache_uses_f32\|aux_cache_prefers_large_budget\)' include/quant.h src/quant test/test_quant.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "quant format helpers must expose GPU matvec capability without CUDA env policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_matvec_disabled\|bn_gpu_policy_cuda_matvec_type_disabled\|bn_gpu_policy_cuda_matvec_type_supported' include/gpu_policy.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU matvec policy must use neutral behavior helper names"
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

if grep -n 'bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in\|bn_transformer_gpu_cuda_all_active_two_kquant_moe_layer(c, lw' src/transformer/gpu.c >/dev/null 2>&1; then
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

if grep -n 'getenv("BN_METAL_SHARED_WEIGHTS")\|getenv("BN_METAL_Q4_PREPARED")\|getenv("BN_METAL_ENABLE_Q6_Q8K")\|getenv("BN_METAL_FULL_BARRIERS")\|getenv("BN_METAL_ENABLE_BARRIERS")\|getenv("BN_METAL_DISABLE_BARRIERS")\|getenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT")\|getenv("BN_METAL_Q8_BARRIERS")\|getenv("BN_METAL_CPU_ORDER_RMSNORM")\|getenv("BN_GPU_Q4_Q8")\|getenv("BN_GPU_Q4_Q8_FROM_LAYER")\|getenv("BN_GPU_Q4_Q8_ATTN_ONLY")\|getenv("BN_GPU_Q4_Q8_FFN_ONLY")' src/gpu_metal.m >/dev/null 2>&1; then
    echo "Metal backend must use GPU policy helpers for Metal/Q4-Q8 compatibility env vars"
    fail=1
fi

if grep -n 'getenv("BN_METAL_' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for Metal env vars"
    fail=1
fi

if grep -n 'return gpu && gpu->kind == BN_GPU_BACKEND_CUDA' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA backend selection"
    fail=1
fi

if ! grep -n 'bn_gpu_policy_metal_native_quant_prepared_enabled\|bn_gpu_policy_metal_native_quant_prepared_upload_enabled\|bn_gpu_policy_metal_repacked_buffer_supported\|bn_gpu_policy_metal_repacked_buffer_type\|bn_gpu_policy_metal_specialized_native_quant_enabled\|bn_gpu_policy_metal_barriers_disabled' src/gpu_policy.c >/dev/null 2>&1; then
    echo "src/gpu_policy.c must own Metal feature-policy env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_apply_q4_q8_prepared_override\|bn_gpu_policy_metal_q4_prepared_enabled\|bn_gpu_policy_metal_q4_prepared_upload_enabled\|metal_q4_prepared\|->q4_prepared[),;[:space:]]\|\.q4_prepared[),;[:space:]]\|int[[:space:]]\+q4_prepared[),;[:space:]]' include/gpu_policy.h src/gpu_policy.c src/main.c src/gpu_metal.m test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Metal prepared native-quant policy must use behavior names, not Q4-prepared helper names"
    fail=1
fi

if grep -n 'bn_gpu_policy_apply_metal_q4_q8_default_disable_override\|bn_gpu_policy_metal_apply_q4_q8_default\|bn_gpu_policy_metal_q4_q8_enabled' include/gpu_policy.h src/gpu_policy.c src/main.c src/gpu_metal.m test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Metal small-dense exact-native default policy must use behavior names, not Q4/Q8 default helper names"
    fail=1
fi

if grep -n 'bn_gpu_policy_apply_specialized_q6_q8k_override\|bn_gpu_policy_metal_q6_q8k_enabled\|bn_gpu_policy_metal_q6_q8k_matvec_supported' include/gpu_policy.h src/gpu_policy.c src/main.c src/gpu_metal.m test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Metal specialized native-quant policy must use behavior names, not Q6/Q8K helper names"
    fail=1
fi

for fn in \
    metal_buffer_create \
    metal_buffer_create_biased \
    metal_buffer_create_stacked2 \
    metal_buffer_create_stacked3 \
    metal_buffer_create_stacked3_biased
do
    if awk -v fn="$fn" '
        $0 ~ "static void \\*" fn "\\(" { in_fn=1 }
        in_fn && /type == BN_GGUF_TENSOR_/ { bad=1 }
        in_fn && /bn_gpu_policy_metal_repacked_buffer_supported|bn_gpu_policy_metal_prepared_stacked_upload_blocked/ { saw_policy=1 }
        in_fn && /^}/ { in_fn=0 }
        END { exit (bad || !saw_policy) ? 0 : 1 }
    ' src/gpu_metal.m >/dev/null 2>&1; then
        echo "Metal buffer upload eligibility in $fn must use GPU policy helpers, not tensor-specific checks"
        fail=1
    fi
done

if awk '
    /static int metal_matvec\(/ { in_fn=1 }
    in_fn && /type == BN_GGUF_TENSOR_/ { bad=1 }
    in_fn && /bn_gpu_policy_metal_q4_q8_matvec_supported/ { saw_q4=1 }
    in_fn && /bn_gpu_policy_metal_specialized_native_quant_matvec_supported/ { saw_q6=1 }
    in_fn && /^}/ { in_fn=0 }
    END { exit (bad || !saw_q4 || !saw_q6) ? 0 : 1 }
' src/gpu_metal.m >/dev/null 2>&1; then
    echo "Metal matvec optimized-path eligibility must use GPU policy helpers, not tensor-specific checks"
    fail=1
fi

if awk '
    /static int metal_execute\(/ { in_fn=1 }
    in_fn && /^static / && !/static int metal_execute\(/ { in_fn=0 }
    in_fn && /op->type == BN_GGUF_TENSOR_/ { bad=1 }
    in_fn && /metal_q4_q8_graph_path_supported/ { saw_q4=1 }
    in_fn && /bn_gpu_policy_metal_specialized_native_quant_matvec_supported/ { saw_q6=1 }
    END { exit (bad || !saw_q4 || !saw_q6) ? 0 : 1 }
' src/gpu_metal.m >/dev/null 2>&1; then
    echo "Metal graph optimized-path eligibility must use GPU policy helpers, not tensor-specific checks"
    fail=1
fi

if awk '
    /static int metal_q4_q8_graph_path_supported\(/ { in_fn=1 }
    in_fn && /^static / && !/static int metal_q4_q8_graph_path_supported\(/ { in_fn=0 }
    in_fn && /bn_gpu_policy_metal_q4_q8_graph_path_supported/ { saw_policy=1 }
    END { exit saw_policy ? 1 : 0 }
' src/gpu_metal.m >/dev/null 2>&1; then
    echo "Metal graph Q4/Q8 wrapper must delegate to GPU policy"
    fail=1
fi

if grep -n 'static const char \*shader_name_for_type\|static const int supported_types' src/gpu_wgpu.c src/gpu_metal.m >/dev/null 2>&1; then
    echo "GPU backends must use quant format helpers for shader type-name and supported-type policy"
    fail=1
fi

if grep -n 'bn_quant_format_gpu_uses_repacked_layout\|bn_quant_format_gpu_supports_repacked_bias' src/gpu_wgpu.c >/dev/null 2>&1; then
    echo "WebGPU repacked upload eligibility must use GPU policy helpers, not direct quant-format checks"
    fail=1
fi

if grep -n 'handle->type = BN_GGUF_TENSOR_' src/gpu_wgpu.c >/dev/null 2>&1; then
    echo "WebGPU repacked upload buffers must preserve the selected tensor type instead of hard-coding a format"
    fail=1
fi

if grep -n 'buf->type = BN_GGUF_TENSOR_' src/gpu_metal.m >/dev/null 2>&1; then
    echo "Metal repacked upload buffers must use GPU policy for selected tensor type"
    fail=1
fi

if ! grep -n 'bn_gpu_policy_webgpu_repacked_buffer_supported\|bn_gpu_policy_webgpu_repacked_bias_supported' src/gpu_policy.c >/dev/null 2>&1; then
    echo "src/gpu_policy.c must own WebGPU repacked upload policy helpers"
    fail=1
fi

if ! grep -n 'bn_quant_format_gpu_shader_name\|bn_quant_format_gpu_shader_type_count' src/quant/registry.c >/dev/null 2>&1; then
    echo "quant registry must own GPU shader type-name and supported-type policy"
    fail=1
fi

if grep -n 'BN_CUDA_DISABLE_CUBLAS_MATMUL\|BN_CUDA_DISABLE_Q6K_CUBLAS_F16\|BN_CUDA_CUBLAS_CACHE_MAX_MB\|BN_CUDA_DISABLE_MOE_F16_Q6K_F32_DOWN_CACHE' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for cublas aux-cache env vars"
    fail=1
fi

if grep -n '\[bn:gpu:argmax:cmp\] cuda=' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "Generic transformer GPU argmax debug logs must not label the backend as CUDA"
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

if grep -n 'bn_gpu_policy_cuda_q8_0_preq_split_enabled\|bn_gpu_policy_cuda_q8_preq_all_enabled\|bn_gpu_policy_cuda_q8_preq_logits_disabled\|bn_gpu_policy_cuda_q8_preq_logits_default_enabled\|bn_gpu_policy_cuda_moe_route_q8k_prequant_enabled\|bn_gpu_policy_cuda_moe_route_q8_1_prequant_enabled\|bn_gpu_policy_cuda_q8_0_ssm_preq_enabled\|bn_gpu_policy_cuda_q8_mixed_preq_enabled\|bn_gpu_policy_all_active_two_kquant_route_q8_1_prequant_enabled' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU policy prepared-input helpers must use behavior names, not preq/prequant helper names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q8_0_quant_matmul_enabled\|bn_gpu_policy_cuda_f16_q8_0_matmul_enabled\|bn_gpu_policy_cuda_q8_0_prepared_input_split_enabled\|bn_gpu_policy_cuda_q8_prepared_input_all_enabled\|bn_gpu_policy_cuda_q8_prepared_input_logits_disabled\|bn_gpu_policy_cuda_q8_prepared_input_logits_default_enabled' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA native-quant prepared-input policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q8_warp_disabled\|bn_gpu_policy_cuda_q8_0_ssm_matvec_enabled\|bn_gpu_policy_cuda_q8_0_ssm_prepared_input_enabled\|bn_gpu_policy_cuda_q8_mixed_prepared_input_enabled\|bn_gpu_policy_cuda_f16_q8_0_ssm_matvec_enabled\|bn_gpu_policy_cuda_f16_q8_0_matvec_enabled\|bn_gpu_policy_cuda_f16_q5k_matvec_enabled\|bn_gpu_policy_cuda_q8_gateup_warp_disabled\|cuda_q8_0_prepared_input\|cuda_q8_prepared_input\|cuda_q8_0_ssm_prepared_input\|cuda_q8_mixed_prepared_input\|cuda_use_q8_0_quant_matmul\|cuda_use_q8_0_prepared_input_split\|cuda_use_f16_q8_0_matmul\|f16_q8_0_matmul_enabled\|q8_prepared_input_logits_default\|disable_q8_warp\|disable_q8_gateup_warp\|enable_q8_prepared_input_all\|disable_q8_prepared_input_logits' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA native-quant policy helpers must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_route_q8k_prepared_input_enabled\|bn_gpu_policy_cuda_moe_route_q8_1_prepared_input_enabled\|bn_gpu_policy_all_active_two_kquant_moe_q8k_default_disabled\|bn_gpu_policy_all_active_two_kquant_route_q8k_default_disabled\|bn_gpu_policy_all_active_two_kquant_route_q8_1_prepared_input_enabled\|cuda_moe_route_q8k_prepared_input\|cuda_moe_route_q8_1_prepared_input\|all_active_two_route_q8_1_prepared_input\|all_active_two_kquant_route_q8k_default' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "MoE route prepared-input policy helpers must use behavior names, not exact quant-prep names"
    fail=1
fi

if grep -n 'bn_backend_quant_q8_0_preq_matvec_candidate' \
    include/backend_quant.h \
    src/gpu_cuda.cu \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Backend quant prepared-input helpers must use behavior names, not preq helper names"
    fail=1
fi

if grep -n 'q8_0_matvec_preq\|cuda_dot_row_q8_0_preq\|prequant scratch\|preq scratch' \
    src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA prepared-input execution helpers must use behavior names, not preq/prequant shorthand"
    fail=1
fi

if grep -n 'q4_native_q8_prequant_matvec\|q4_matvec_split_q8_prequant\|q4_fused_gateup_silu_q8_prequant\|prequantized Q8 activation' \
    src/gpu_metal.m shaders/metal >/dev/null 2>&1; then
    echo "Metal prepared-input shader names must use behavior names, not prequant shorthand"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_FORCE_Q4K_QUANT_MATMUL")\|getenv("BN_CUDA_FORCE_Q6K_QUANT_MATMUL")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for forced quant matmul env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q6K_4WARP_LONG")\|getenv("BN_CUDA_DISABLE_Q6K_4WARP_1536_8960")\|getenv("BN_CUDA_ENABLE_Q6K_4WARP_5120")\|getenv("BN_CUDA_DISABLE_Q6K_5WARP_1536_8960")\|getenv("BN_CUDA_DISABLE_Q6K_5WARP_2560_9728")\|getenv("BN_CUDA_DISABLE_Q6K_3WARP_1536_8960")\|getenv("BN_CUDA_DISABLE_Q6K_3WARP_2560_9728")\|getenv("BN_CUDA_ENABLE_Q6K_2WARP_LONG")\|getenv("BN_CUDA_DISABLE_Q6K_2WARP_LONG")\|getenv("BN_CUDA_ENABLE_Q6K_MATVEC4_1024_2560")\|getenv("BN_CUDA_ENABLE_Q6K_MATVEC4_512_2048")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for Q6K matvec shape env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q6K_FLOAT_MOE_DOWN")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN")\|getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_PAIR2")\|getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F32_4ROW")\|getenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_ACCUM")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR4_SUM")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_K8_4ROW_SUM")\|getenv("BN_CUDA_ENABLE_MOE_Q6K_K8_8ROW_SUM")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_ALL2_FIXED")\|getenv("BN_CUDA_DISABLE_MOE_DOWN_RESID_RMSNORM_FUSE")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_K8_EXACT_2048_768")\|getenv("BN_CUDA_ENABLE_MOE_Q6K_ALL2_ACCUM_4ROW")\|getenv("BN_CUDA_DISABLE_MOE_Q6K_PAIR_DOWN_4ROW")\|getenv("BN_CUDA_ENABLE_Q6K_MOE_DOWN_F16_CACHE")\|getenv("BN_CUDA_DISABLE_Q6K_MOE_DOWN_F16_CACHE")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for Q6K MoE down variant env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q6k_moe_quant_down_preferred\|bn_gpu_policy_cuda_q6k_moe_down_f32_cache_path_enabled\|bn_gpu_policy_cuda_q6k_moe_down_halfwarp_enabled\|bn_gpu_policy_cuda_q6k_moe_down_split4_enabled\|bn_gpu_policy_cuda_q6k_moe_down_scatter_enabled\|bn_gpu_policy_cuda_q6k_moe_down_scatter_16row_enabled\|bn_gpu_policy_cuda_q6k_moe_float_down_enabled\|bn_gpu_policy_cuda_q6k_moe_pair_down_enabled\|bn_gpu_policy_cuda_q6k_moe_prefer_f32_down\|bn_gpu_policy_cuda_q6k_moe_down_f32_pair2_enabled\|bn_gpu_policy_cuda_q6k_moe_down_f32_pair2_4row_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_4row_sum_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_8row_sum_enabled\|bn_gpu_policy_cuda_q6k_moe_down_resid_rmsnorm_fuse_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_k8_exact_2048_768_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_pair_4row_enabled\|bn_gpu_policy_cuda_q6k_moe_down_f32_cache_enabled\|bn_gpu_policy_cuda_q6k_moe_down_f16_cache_enabled' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA MoE down policy helpers must use behavior names, not Q6K facade names"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q4K_MOE_DOWN_F32_CACHE")\|getenv("BN_CUDA_ENABLE_MOE_Q4K_PAIR_DOWN")\|getenv("BN_CUDA_DISABLE_MOE_Q4K_PAIR_DOWN")\|getenv("BN_CUDA_ENABLE_MOE_Q4K_DOWN_8ROW")\|getenv("BN_CUDA_DISABLE_MOE_Q4K_DOWN_8ROW")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for Q4K MoE down variant env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_Q8_MOE_BATCH_Q8_1")\|getenv("BN_CUDA_DISABLE_Q8_MOE_GATEUP_2ROW")\|getenv("BN_CUDA_ENABLE_Q8_MOE_DOWN_4ROW")\|getenv("BN_CUDA_DISABLE_Q8_MOE_DOWN_2ROW")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for Q8 MoE native env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_Q4K_Q8K_MOE_GATEUP")\|getenv("BN_CUDA_DISABLE_MOE_GATEUP_8ROW")\|getenv("BN_CUDA_ENABLE_MOE_GATEUP_SPLIT")\|getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT")' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "CUDA backend must use GPU policy helpers for Q4K/Q8K MoE gate-up env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q4k_moe_down_f32_cache_enabled\|bn_gpu_policy_cuda_q4k_moe_pair_down_enabled\|bn_gpu_policy_cuda_q4k_moe_down_8row_enabled\|bn_gpu_policy_cuda_q4k_q8k_moe_gateup_enabled\|bn_gpu_policy_cuda_q4k_moe_gateup_8row_enabled\|bn_gpu_policy_cuda_q4k_moe_gateup_split_enabled\|bn_gpu_policy_cuda_q8_moe_q8_1_batch_enabled\|bn_gpu_policy_cuda_q8_moe_q8x_enabled\|bn_gpu_policy_cuda_q8_moe_gateup_2row_enabled\|bn_gpu_policy_cuda_q8_moe_down_4row_enabled\|bn_gpu_policy_cuda_q8_moe_down_2row_enabled\|bn_gpu_policy_cuda_moe_q4k_q8k_dot_enabled\|bn_gpu_policy_cuda_moe_q4k_all_active_two_fixed_4row_enabled\|bn_gpu_policy_cuda_moe_q4k_gateup_4row_disabled' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "CUDA MoE prepared/block policy helpers must use behavior names, not Q4/Q8 facade names"
    fail=1
fi

if ! awk '
    /static int cuda_force_quant_matmul_for_type/ { in_fn=1 }
    in_fn && /bn_gpu_policy_cuda_force_quant_matmul_for_type/ { found=1 }
    in_fn && /BN_GGUF_TENSOR_/ { bad=1 }
    in_fn && /^}/ { in_fn=0 }
    END { exit(found && !bad ? 0 : 1) }
' src/gpu_cuda.cu; then
    echo "CUDA forced quant matmul selection must delegate tensor/env policy"
    fail=1
fi

for fn in \
    cuda_use_down_kquant_4warp_long \
    cuda_use_down_kquant_5warp_exact \
    cuda_use_down_kquant_3warp_exact \
    cuda_use_down_kquant_2warp_long \
    cuda_disable_down_kquant_matvec4_shape \
    cuda_prefer_moe_down_quant_path \
    cuda_use_moe_down_f32_cache_path \
    cuda_use_moe_down_4row \
    cuda_use_moe_down_8row \
    cuda_use_moe_down_halfwarp \
    cuda_use_moe_down_split4 \
    cuda_use_moe_down_scatter \
    cuda_use_moe_down_scatter_16row \
    cuda_use_moe_down_float_path \
    cuda_use_moe_down_pair_path \
    cuda_prefer_moe_down_f32_cache \
    cuda_use_moe_down_f32_pair2 \
    cuda_use_moe_down_f32_pair2_4row \
    cuda_use_moe_down_prepared_k8_4row_sum \
    cuda_use_moe_down_prepared_k8_8row_sum \
    cuda_use_moe_down_resid_rmsnorm_fuse \
    cuda_use_moe_down_prepared_k8_exact_2048_768 \
    cuda_use_moe_down_prepared_pair_4row \
    cuda_use_moe_down_f32_cache \
    cuda_use_moe_down_f16_cache \
    cuda_use_moe_down_aux_f32_cache \
    cuda_use_moe_down_prepared_pair8 \
    cuda_use_moe_down_prepared_8row
do
    if ! awk -v fn="$fn" '
        $0 ~ "static int " fn "\\(" { in_fn=1 }
        in_fn && /bn_gpu_policy_cuda_/ { found=1 }
        in_fn && /getenv\(/ { bad=1 }
        in_fn && /BN_GGUF_TENSOR_/ { bad=1 }
        in_fn && /^}/ { in_fn=0 }
        END { exit(found && !bad ? 0 : 1) }
    ' src/gpu_cuda.cu; then
        echo "CUDA policy selection in $fn must delegate to GPU policy"
        fail=1
    fi
done

for mapping in \
    "cuda_use_all_active_two_kquant_moe_down_accum bn_gpu_policy_all_active_two_kquant_moe_down_accum_enabled" \
    "cuda_use_all_active_two_kquant_moe_down_pair4_sum bn_gpu_policy_all_active_two_kquant_moe_down_pair4_sum_enabled" \
    "cuda_use_all_active_two_kquant_moe_down_fixed bn_gpu_policy_all_active_two_kquant_moe_down_fixed_enabled" \
    "cuda_use_all_active_two_kquant_moe_down_accum_4row bn_gpu_policy_all_active_two_kquant_moe_down_accum_4row_enabled"
do
    set -- $mapping
    fn="$1"
    policy="$2"
    if ! awk -v fn="$fn" -v policy="$policy" '
        $0 ~ "static int " fn "\\(" { in_fn=1 }
        in_fn && index($0, policy) { found=1 }
        in_fn && /getenv\(/ { bad=1 }
        in_fn && /BN_GGUF_TENSOR_/ { bad=1 }
        in_fn && /^}/ { in_fn=0 }
        END { exit(found && !bad ? 0 : 1) }
    ' src/gpu_cuda.cu; then
        echo "CUDA all-active-two K-quant MoE down selection in $fn must delegate to $policy"
        fail=1
    fi
done

if grep -n 'getenv("BN_CUDA_ENABLE_LOGITS_CACHE")\|getenv("BN_CUDA_ENABLE_MOE_DECODE_CACHE")\|getenv("BN_CUDA_DISABLE_MOE_DECODE_CACHE")\|getenv("BN_CUDA_DISABLE_DECODE_CACHE")\|getenv("BN_CUDA_DISABLE_Q4_Q8_DECODE_CACHE")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA decode-cache env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_decode_logits_cache_enabled\|bn_gpu_policy_cuda_moe_decode_cache_enabled\|bn_gpu_policy_cuda_moe_decode_cache_disabled\|bn_gpu_policy_cuda_decode_cache_disabled\|bn_gpu_policy_cuda_q4_q8_decode_cache_disabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use behavior-named GPU policy helpers for decode-cache compatibility policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_decode_logits_cache_enabled\|bn_gpu_policy_cuda_moe_decode_cache_enabled\|bn_gpu_policy_cuda_moe_decode_cache_disabled\|bn_gpu_policy_cuda_decode_cache_disabled\|bn_gpu_policy_cuda_q4_q8_decode_cache_disabled\|bn_gpu_policy_q4_q8_decode_cache_disabled' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Decode cache policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_LOGITS_ARGMAX")\|getenv("BN_CUDA_ENABLE_DENSE_LOGITS_ARGMAX")\|getenv("BN_CUDA_ENABLE_MOE_LOGITS_MMVQ_ARGMAX")\|getenv("BN_CUDA_DISABLE_MOE_LOGITS_MMVQ_ARGMAX")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA logits argmax env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_logits_argmax_disabled\|bn_gpu_policy_cuda_dense_logits_argmax_enabled\|bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled\|bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use behavior-named GPU policy helpers for logits argmax compatibility policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_logits_argmax_disabled\|bn_gpu_policy_cuda_dense_logits_argmax_enabled\|bn_gpu_policy_cuda_moe_logits_mmvq_argmax_enabled\|bn_gpu_policy_cuda_moe_logits_mmvq_argmax_disabled' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Logits argmax policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_PREFILL_SSM_LAYER")\|getenv("BN_CUDA_ENABLE_Q5K_FUSED_GATEUP")\|getenv("BN_CUDA_DISABLE_SHARED_Q4K_Q8K_DOT")\|getenv("BN_CUDA_DISABLE_SHARED_EXPERT_GATE")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA prefill/dispatch env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_explicit_q5k_fused_gateup_enabled' include/gpu_policy.h src/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU fused gate/up opt-in policy must expose behavior-named helpers"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_PREFILL_ATTN_MIN_TOKENS")\|getenv("BN_CUDA_DISABLE_PREFILL_DENSE_CHAIN")\|getenv("BN_CUDA_DISABLE_PREFILL_HYBRID_CHAIN")\|getenv("BN_CUDA_DISABLE_PREFILL_ATTN")\|getenv("BN_CUDA_DISABLE_PREFILL_SSM_RUN_CHAIN")\|getenv("BN_CUDA_DISABLE_SSM_FFN_FUSE")\|getenv("BN_CUDA_DEBUG_PREFILL_MOE_CHAIN")\|getenv("BN_CUDA_DEBUG_PREFILL_HYBRID_CHAIN")\|getenv("BN_CUDA_ENABLE_MOE_PREFILL")\|getenv("BN_CUDA_MOE_PREFILL_MIN_TOKENS")\|getenv("BN_CUDA_DISABLE_MOE_CACHE_PREFILL")\|getenv("BN_CUDA_DISABLE_MOE_PREFILL_SHARED_FUSE")\|getenv("BN_CUDA_DEBUG_MOE_ROUTE_BATCH")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA prefill env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ATTN")\|getenv("BN_CUDA_ENABLE_LARGE_HYBRID_CPU_ATTN_SAFE")\|getenv("BN_CUDA_DISABLE_LARGE_HYBRID_CPU_ATTN_SAFE")\|getenv("BN_CUDA_FORCE_LARGE_HYBRID_CPU_ATTN_SAFE")\|getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL")\|getenv("BN_CUDA_ENABLE_LARGE_HYBRID_PREFILL_CHAIN")\|getenv("BN_CUDA_DISABLE_LARGE_HYBRID_PREFILL")\|getenv("BN_CUDA_ENABLE_LARGE_HYBRID_ARGMAX")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA large-hybrid env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_ENABLE_SMALL_KQUANT_NATIVE")\|getenv("BN_CUDA_DISABLE_SMALL_KQUANT_NATIVE")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA small K-quant native env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_small_kquant_native_enabled\|bn_gpu_policy_cuda_small_kquant_native_disabled' include/gpu_policy.h test/test_gpu_backend.c src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Public GPU policy, tests, and transformer GPU policy must use neutral helpers for small K-quant native policy"
    fail=1
fi

if grep -n 'getenv("BN_GPU_DISABLE_PREFILL_MATMUL")\|getenv("BN_GPU_PREFILL_MATMUL")\|getenv("BN_CUDA_DISABLE_PREFILL_DIRECT_KV")\|getenv("BN_CUDA_ENABLE_PREFILL_DIRECT_KV_WITH_CPU_FALLBACK")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for prefill matmul/direct-KV env vars"
    fail=1
fi

if grep -n 'getenv("BN_GPU_CPU_FALLBACK_LAYER")\|getenv("BN_GPU_CPU_FALLBACK_FROM_LAYER")\|getenv("BN_GPU_CPU_ATTN_LAYER")\|getenv("BN_GPU_CPU_ATTN_FROM_LAYER")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CPU decode fallback selector env vars"
    fail=1
fi

if grep -n 'gpu_policy_env_int\|BN_GPU_CPU_FFN_LAYER\|BN_GPU_CPU_FFN_FROM_LAYER\|BN_GPU_CPU_FFN_DOWN_FROM_LAYER\|BN_GPU_COMPARE_ATTENTION_LAYER\|BN_GPU_COMPARE_ATTENTION_POS\|BN_GPU_COMPARE_GQA_LAYER\|BN_GPU_COMPARE_GQA_POS\|BN_GPU_COMPARE_QKV_LAYER\|BN_GPU_COMPARE_QKV_POS\|BN_GPU_COMPARE_FFN_DOWN_LAYER\|BN_GPU_COMPARE_FFN_DOWN_POS\|BN_GPU_COMPARE_FFN_STATE_LAYER\|BN_GPU_COMPARE_FFN_STATE_POS' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CPU fallback and compare selector env vars"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_SSM_GRAPH")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA SSM graph env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_ssm_graph_disabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use neutral helpers for SSM graph policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_prefill_ssm_layer_disabled\|bn_gpu_policy_cuda_shared_q4_q8_dot_enabled\|bn_gpu_policy_cuda_shared_expert_gate_enabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use behavior-named GPU policy helpers for SSM/shared-expert compatibility policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_large_hybrid_attention\|bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe\|bn_gpu_policy_cuda_large_hybrid_prefill\|bn_gpu_policy_cuda_large_hybrid_argmax\|bn_gpu_policy_cuda_prefill_direct_kv\|bn_gpu_policy_cuda_prefill_attention_min_tokens\|bn_gpu_policy_cuda_prefill_dense_chain\|bn_gpu_policy_cuda_prefill_hybrid_chain\|bn_gpu_policy_cuda_prefill_attention_enabled\|bn_gpu_policy_cuda_prefill_ssm_run_chain\|bn_gpu_policy_cuda_prefill_ssm_ffn_fuse\|bn_gpu_policy_cuda_prefill_moe_chain_debug\|bn_gpu_policy_cuda_prefill_hybrid_chain_debug\|bn_gpu_policy_cuda_moe_prefill\|bn_gpu_policy_cuda_moe_cache_prefill\|bn_gpu_policy_cuda_moe_route_batch_debug' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use behavior-named GPU policy helpers for prefill/hybrid compatibility policy"
    fail=1
fi

if grep -n 'getenv("BN_GPU_CPU_LOGITS")\|getenv("BN_GPU_COMPARE_LOGITS")\|getenv("BN_GPU_DEBUG_ARGMAX_COMPARE")\|getenv("BN_CUDA_DISABLE_MOE_FFN")\|getenv("BN_CUDA_OVERRIDE_MOE_WITH_CPU_ACTUAL")\|getenv("BN_GPU_COMPARE_MOE_LAYER")\|getenv("BN_GPU_COMPARE_MOE_POS")\|getenv("BN_GPU_COMPARE_MOE_INPUT_NORM")\|getenv("BN_GPU_COMPARE_MOE_ACTUAL")\|getenv("BN_GPU_COMPARE_MOE_ROUTE")\|getenv("BN_GPU_COMPARE_MOE_RAW")\|getenv("BN_GPU_COMPARE_MOE_MID")\|getenv("BN_GPU_COMPARE_MOE_PARTS")\|getenv("BN_GPU_COMPARE_MOE_SHARED_MID")\|getenv("BN_GPU_COMPARE_MOE_SHARED_DOWN")\|getenv("BN_GPU_COMPARE_MOE_NORM")\|getenv("BN_CUDA_ENABLE_MOE_SHARED_CPU_FALLBACK")\|getenv("BN_CUDA_DISABLE_MOE_SHARED_CPU_FALLBACK")\|getenv("BN_CUDA_DISABLE_MOE_GATEUP_SPLIT")\|getenv("BN_CUDA_ENABLE_MOE_LAZY_AUX_CACHE")\|getenv("BN_GPU_MOE_ROUTE_PROFILE")\|getenv("BN_GPU_MOE_ROUTE_PROFILE_EVERY")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for GPU debug/MoE env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_lazy_aux_cache_enabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use neutral helpers for MoE lazy aux cache policy"
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

if grep -n 'bn_transformer_gpu_all_active_two_kquant_moe_requires_opt_in\|bn_transformer_gpu_cuda_moe_routed_ffn_batch_allowed\|bn_transformer_gpu_cuda_moe_route_batch_debug_enabled' src/moe_prefill.c >/dev/null 2>&1; then
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

if grep -n 'bn_gpu_policy_cuda_layout_reserve_bytes\|optional CUDA layout\|estimate_cuda_base_model_bytes\|full CUDA MoE residency' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use behavior-named GPU policy helpers for optional layout and residency diagnostics"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_layout_reserve_bytes\|bn_gpu_policy_cuda_moe_full_reserve_bytes' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU reserve policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled\|bn_gpu_policy_cuda_logits_f16_cache_enabled' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use behavior-named GPU policy helpers for logits upload cache policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q6k_logits_f32_cache_enabled\|bn_gpu_policy_cuda_logits_f16_cache_enabled' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Logits upload cache policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_individual_upload_quant_only_enabled' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use behavior-named GPU policy helpers for individual quant-only upload policy"
    fail=1
fi

if grep -n 'can_use_cuda_moe_routed_ffn\|bn_gpu_policy_cuda_moe_routed_ffn_enabled\|bn_gpu_policy_cuda_moe_resident_routed_ffn_quant_eligible' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use behavior-named GPU policy helpers for resident routed-FFN upload policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_all_f16_cache\|bn_gpu_policy_cuda_moe_gateup_f16_cache\|bn_gpu_policy_cuda_partial_moe_f16_cache\|bn_gpu_policy_cuda_moe_f16_aux_cache_auto\|bn_gpu_policy_cuda_moe_full_reserve\|bn_gpu_policy_cuda_moe_down_q6_f32_cache\|bn_gpu_policy_cuda_moe_down_q4_f32_cache\|bn_gpu_policy_cuda_moe_quant_only_after_cache\|bn_gpu_policy_cuda_aux_cache_bytes\|bn_gpu_policy_cuda_moe_fit_debug' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use behavior-named GPU policy helpers for MoE residency and aux-cache policy"
    fail=1
fi

if grep -n '\bq8_f16_cache\b\|upload_moe_all_q8_f16_cache' \
    include/gpu_policy.h \
    src/gpu_policy.c \
    src/model_gpu.c \
    include/backend_quant.h \
    include/quant.h \
    src/quant/registry.c \
    test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "MoE F16 aux-cache policy flags must use behavior names, not Q8 cache names"
    fail=1
fi

if grep -n 'bn_gpu_policy_logits_q6_f32_cache_enabled\|bn_gpu_policy_cuda_moe_down_q6_f32_cache_enabled\|bn_gpu_policy_cuda_moe_down_q6_f32_cache_forced\|bn_gpu_policy_cuda_moe_down_q6_f32_cache_default_for_cols\|bn_gpu_policy_moe_down_q6_f32_cache_preferred\|bn_gpu_policy_moe_down_q6_f32_cache_bytes\|bn_gpu_policy_moe_down_q6_f32_cache_requires_full_buffer\|bn_gpu_policy_moe_down_q4_f32_cache_enabled\|gpu_policy_logits_q6_f32_cache_enabled\|gpu_policy_moe_down_q6_f32_cache_preferred\|gpu_policy_moe_down_q6_f32_cache_bytes\|gpu_policy_moe_down_q6_f32_cache_requires_full_buffer\|gpu_policy_moe_down_q4_f32_cache_enabled\|moe_down_q6_f32_cache_bytes\|prefer_q6_f32_cache\|prefer_q4_f32_cache\|\<logits_q6_f32_cache\>' include/gpu_policy.h src/gpu_policy.c src/model_gpu.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "MoE down-cache policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_\(f16_aux_cache_auto_enabled\|resident_routed_ffn_quant_eligible\|all_f16_cache_forced\|all_f16_cache_enabled_for_type\|gateup_f16_cache_enabled\|partial_moe_f16_cache_enabled\|fit_debug_enabled\|lazy_aux_cache_enabled\)\|bn_gpu_policy_cuda_\(keep_individual_f16_cache_enabled\|individual_upload_quant_only_enabled\)' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU MoE residency policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'n_experts == 2 && c->n_experts_active == 2\|c->n_experts > 2' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use model_arch helpers for MoE shape policy"
    fail=1
fi

if grep -n 'gate_rows == c->moe_intermediate_size\|up_rows == c->moe_intermediate_size\|down_cols == c->moe_intermediate_size' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use MoE policy helpers for resident routed-FFN layout eligibility"
    fail=1
fi

if grep -n 'c->n_experts <= 0' src/model_gpu.c >/dev/null 2>&1; then
    echo "src/model_gpu.c must use model_arch helpers for loaded-model MoE presence policy"
    fail=1
fi

if grep -n 'c->n_experts > 2' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for MoE shape policy"
    fail=1
fi

if grep -n 'n_experts > 2' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for large MoE route policy"
    fail=1
fi

if grep -n 'n_experts <= 2' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for two-expert MoE route policy"
    fail=1
fi

if grep -n 'c->n_experts <= 0' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for MoE presence policy"
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

if grep -n 'bn_model_arch_.*cuda' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose behavior-named model_arch helpers, not CUDA-named compatibility aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_small_dense_.*\|bn_transformer_gpu_cuda_small_backend_q8_logits_refine' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "small-dense transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_prefill_\(dense_chain_enabled\|hybrid_chain_enabled\|attention_enabled\|ssm_run_chain_enabled\|ssm_ffn_fuse_allowed\|moe_chain_debug_enabled\|hybrid_chain_debug_enabled\|ssm_layer_disabled\|attention_min_tokens\|dense_chain_min_tokens\|moe_chain_min_tokens\|moe_ffn_batch_available\|ssm_moe_chain_available\|ssm_dense_chain_available\)' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "prefill transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_backend_quant_gpu_fused_gateup_requires_cuda_opt_in\|bn_gpu_policy_cuda_q5k_fused_gateup_enabled' src/gpu_policy.c src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "GPU policy must compose fused gate/up policy through behavior-named helpers"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_large_hybrid_prefill_disabled' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "large-hybrid prefill transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_large_hybrid_\(cpu_attn_safe_default\|cpu_attn_safe_fallback_enabled\|prefill_decode_fallback_default\|prefill_chain_disabled_default\)\|bn_transformer_gpu_cuda_matvec_fallback_kept\|bn_transformer_gpu_cuda_prefill_direct_kv_allowed' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "large-hybrid/direct-KV transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_\(moe_decode_cacheable\|decode_cacheable\|large_hybrid_argmax_blocked\)' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "decode cache transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_\(all_active_two_kquant_moe_q6_logits_refine_default\|all_active_two_kquant_moe_cpu_moe_safe_default\|all_active_two_kquant_moe_model\|all_active_two_kquant_moe_layer\|all_active_two_kquant_moe_layer_enabled\|all_active_two_kquant_moe_cpu_attn_safe_default\|all_active_two_kquant_moe_cpu_attn_fallback_enabled\|moe_exact_attention_enabled\)' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "MoE refine/exact-attention transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_\(all2_moe_direct_route_enabled\|all_active_two_kquant_moe_route_layer_selected\|all_active_two_kquant_moe_route_layer_range\|all_active_two_kquant_moe_exact_gpu_route_enabled\|all_active_two_kquant_moe_router\)' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "MoE route transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_transformer_gpu_cuda_moe_\(prefill_enabled\|prefill_min_tokens\|cache_prefill_enabled\|prefill_shared_fuse_enabled\|route_batch_debug_enabled\|lazy_aux_cache_enabled\|routed_ffn_batch_allowed\|ffn_disabled\|cpu_actual_override_enabled\|shared_cpu_fallback_enabled\|gateup_split_enabled\|cpu_route_resident_ffn_enabled\|route_topk_enabled\|routed_ffn_enabled\)' src/transformer/gpu_policy.c src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "MoE prefill transformer GPU policy must expose backend-neutral helper names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_route_batch_enabled' include/gpu_policy.h src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU MoE route batch policy must use neutral behavior helper names"
    fail=1
fi

if grep -n 'BN_MODEL_ARCH_POLICY_.*CUDA\|bn_model_arch_.*cuda' include/model_arch.h src/model_arch.c >/dev/null 2>&1; then
    echo "model_arch must expose backend-neutral policy names, not CUDA-named aliases"
    fail=1
fi

if grep -n 'bn_quant_format_is_cuda_f16_float_cache_matvec_candidate' include/quant.h src/quant test/test_quant.c >/dev/null 2>&1; then
    echo "quant format helpers must name F16 float-cache matvec capability without CUDA"
    fail=1
fi

if awk '
    /int bn_transformer_gpu_batch_prefill_enabled\(/ { in_func = 1 }
    in_func && /return c->dim <= (8192|2560)/ { found = 1 }
    in_func && /^}/ { in_func = 0 }
    END { exit found ? 0 : 1 }
' src/transformer/gpu_policy.c; then
    echo "src/transformer/gpu_policy.c must compose GPU policy helpers for dense batch prefill shape policy"
    fail=1
fi

if grep -n 'c->dim <= 8192\|c->dim <= 2560' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for dense batch prefill shape limits"
    fail=1
fi

if grep -n 'logits->rows > 262144\|logits->cols == 1536' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose model_arch helpers for logits argmax shape policy"
    fail=1
fi

if grep -n 'em->gate_rows != c->moe_intermediate_size\|map->gate_rows == moe_hidden' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose MoE policy helpers for resident routed FFN layout"
    fail=1
fi

if grep -n 'map->gate_rows == map->up_rows\|map->gate_cols == map->up_cols' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must compose MoE policy helpers for MoE gate/up split layout"
    fail=1
fi

if grep -n 'full_attn_interval\|c && c->n_experts > 0 && c->full_attn_interval > 0' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must compose model_arch helpers for hybrid layout policy"
    fail=1
fi

if grep -n 'full_attn_interval\|n_ssm = c->n_layers - n_attn' src/gpu_wgpu.c src/gpu_metal.m src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "GPU backends must compose model_arch helpers for hybrid activation sizing policy"
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

if grep -n 'BN_CUDA_DISABLE_CUBLAS_MATMUL\|BN_CUDA_DISABLE_Q6K_CUBLAS_F16\|BN_CUDA_ENABLE_Q6K_MOE_DOWN_F32_CACHE\|BN_CUDA_DISABLE_MOE_ROUTED_FFN\|BN_GPU_MOE_DISABLE_AUTO_RESIDENT\|BN_GPU_MOE_CACHE_RESERVE_MB\|BN_CUDA_ENABLE_DUPLICATE_MOE_CACHE\|BN_CUDA_DISABLE_DUPLICATE_MOE_CACHE\|BN_METAL_ENABLE_MMAP_ZERO_COPY\|BN_GPU_BACKEND_CUDA\|bn_quant_format_cuda_moe_down_cublas_cache_supported\|bn_quant_format_cuda_moe_down_cublas_cache_elem_bytes' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use GPU policy helpers for GPU cache/env policy"
    fail=1
fi

if grep -n 'setenv("BN_METAL_' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use GPU policy helpers for Metal compatibility env overrides"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_down_cublas_cache_bytes\|bn_gpu_policy_cuda_moe_routed_ffn_enabled\|bn_gpu_policy_cuda_duplicate_moe_cache_enabled\|count_cuda_routed_moe' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use behavior-named GPU policy helpers for MoE cache policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_duplicate_moe_cache_enabled' include/gpu_policy.h src/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Duplicate MoE cache policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'snprintf([^,]*, *sizeof([^)]*), *"%zu", *reserve_mb)' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must format GPU MoE reserve policy from reserve bytes, not stale reserve_mb locals"
    fail=1
fi

if grep -n 'expert_count\|n_experts > 0' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use GPU policy helpers for Metal MoE sequence auto-cap policy"
    fail=1
fi

if grep -n 'bn_model_arch_gguf_uses_moe' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use GPU policy helpers for GGUF MoE sequence auto-cap policy"
    fail=1
fi

if grep -n 'general\.architecture\|context_length\|bn_gguf_get_u32' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use model_arch/GPU policy helpers for arch-prefixed GGUF sequence metadata"
    fail=1
fi

if grep -n 'model\(\.config\|->config\)\.n_experts [<>!=]=\? 0' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use model_arch helpers for loaded-model MoE presence policy"
    fail=1
fi

if grep -n '#include "model_arch.h"\|bn_model_arch_' src/main.c >/dev/null 2>&1; then
    echo "src/main.c must use loaded-model policy helpers instead of reaching into model_arch"
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

if grep -n 'getenv("BN_GPU_DISABLE_FUSED_GATEUP")\|getenv("BN_GPU_Q4_Q8_DISABLE_GATEUP")\|getenv("BN_GPU_DISABLE_GATEUP_SPLIT")\|getenv("BN_GPU_Q4_Q8_DISABLE_FFN_DOWN")\|getenv("BN_GPU_DISABLE_QKV_SPLIT")\|getenv("BN_GPU_DEBUG_QKV_SPLIT")\|getenv("BN_GPU_DISABLE_SSM_QKVZ_SPLIT")\|getenv("BN_GPU_DISABLE_SSM_AB_STACK")\|getenv("BN_GPU_SPLIT_RESIDUAL_RMSNORM")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for generic GPU feature env vars"
    fail=1
fi

if grep -n 'BN_GPU_DISABLE_QKV_SPLIT\|BN_GPU_DISABLE_SSM_QKVZ_SPLIT\|BN_GPU_DISABLE_SSM_AB_STACK\|BN_GPU_DEBUG_QKV_SPLIT\|BN_GPU_SPLIT_RESIDUAL_RMSNORM\|BN_GPU_DEBUG_FALLBACK' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for QKV/SSM split and debug compatibility env vars"
    fail=1
fi

if grep -n 'ssm_alpha\.rows == .*ssm_beta\.rows\|ssm_alpha\.cols == .*ssm_beta\.cols' src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_emit.c must use GPU policy helpers for SSM alpha/beta stackability"
    fail=1
fi

if grep -n 'BN_GPU_DEBUG_FALLBACK' src/gpu_graph_lowering_internal.h >/dev/null 2>&1; then
    echo "src/gpu_graph_lowering_internal.h must receive GPU debug policy as an argument"
    fail=1
fi

if grep -n 'getenv("BN_GPU_DEBUG_FALLBACK")\|getenv("BN_GPU_FORCE_GRAPH")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for graph/debug env vars"
    fail=1
fi

if grep -n 'getenv("BN_GPU_Q4_Q8[^"]*")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for Q4/Q8 layer env vars"
    fail=1
fi

if grep -n 'getenv("BN_GPU_FLASH_MIN_KV")\|getenv("BN_GPU_FLASH_MAX_KV")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for flash attention env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_flash_max_kv_or_default\|bn_gpu_policy_kquant_logits_refine_enabled\|bn_gpu_policy_native_quant_logits_refine_enabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-aware helpers for flash/logits refine policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_flash_max_kv_or_default\|bn_gpu_policy_kquant_logits_refine_enabled\|bn_gpu_policy_native_quant_logits_refine_enabled' include/gpu_policy.h src/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU policy API/tests must use backend-aware helpers instead of raw backend booleans"
    fail=1
fi

if grep -n 'bn_gpu_policy_backend_small_dense_q8_logits_refine\|bn_gpu_policy_backend_all2_q4q6_moe_q6_logits_refine\|bn_gpu_policy_backend_q[68]_logits_refine\|bn_gpu_policy_q[68]_logits_refine\|bn_gpu_policy_small_dense_q8_logits_refine\|bn_gpu_policy_all2_q4q6_moe_q6_logits_refine' include/gpu_policy.h src/gpu_policy.c src/transformer/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU policy logits refine helpers must use behavior names, not quant-format helper names"
    fail=1
fi

if grep -n 'transformer_gpu_backend_is_cuda\|bn_gpu_policy_backend_is_cuda(gpu)' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must compose named backend capability helpers instead of raw backend kind checks"
    fail=1
fi

if grep -n 'bn_gpu_policy_backend_is_cuda' include/gpu_policy.h test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "GPU policy public API/tests must not expose raw CUDA backend predicates"
    fail=1
fi

if grep -n 'BN_GPU_BACKEND_CUDA\|BN_GPU_BACKEND_METAL\|BN_GPU_BACKEND_WEBGPU\|BN_GPU_BACKEND_UNKNOWN' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use GPU policy helpers for backend placement mapping"
    fail=1
fi

if grep -n 'bn_gpu_policy_metal_specialized_native_quant_enabled\|bn_gpu_policy_metal_native_quant_prepared_enabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-neutral behavior helpers for Metal native-quant policy"
    fail=1
fi

if sed -n '/^int bn_gpu_policy_backend_flash_default_enabled/,/^int bn_gpu_policy_argmax_debug_enabled/p' \
    src/gpu_policy.c | grep -n 'bn_gpu_policy_backend_is_cuda' >/dev/null 2>&1; then
    echo "GPU backend capability helpers must use the backend capability table, not repeated CUDA predicates"
    fail=1
fi

if grep -n 'bn_gpu_policy_backend_is_cuda' src/gpu_policy.c >/dev/null 2>&1; then
    echo "GPU policy must use named backend capabilities instead of raw CUDA predicates"
    fail=1
fi

if grep -n 'bn_gpu_policy_specialized_q6_q8k_path_enabled' include/gpu_policy.h src/gpu_policy.c src/transformer/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Specialized native-quant decode policy must use behavior names, not Q6/Q8K helper names"
    fail=1
fi

if awk '/^int bn_transformer_gpu_moe_quant_only_without_aux_cache/{flag=1} /^int bn_transformer_gpu_large_hybrid_prefill_disabled/{flag=0} flag{print}' src/transformer/gpu_policy.c | grep -n 'transformer_gpu_backend_is_cuda(gpu)' >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-aware helpers for lazy MoE aux cache support"
    fail=1
fi

if awk '/^int bn_transformer_gpu_native_quant_logits_refine_enabled/{flag=1} /^int bn_transformer_gpu_kquant_logits_refine_enabled/{flag=0} flag{print}' src/transformer/gpu_policy.c | grep -n 'transformer_gpu_backend_is_cuda(gpu)' >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-aware helpers for logits refine defaults"
    fail=1
fi

if grep -n 'bn_transformer_gpu_small_backend\|small_backend_.*logits_refine\|\.small_backend_\|bn_transformer_gpu_all2_q4q6_moe_q6_logits_refine\|bn_transformer_gpu_all2_moe_logits_refine_default\|bn_transformer_gpu_q[68]_logits_refine\|\.q[68]_\(default\|enabled\|captures_xb\|refine_top\)\|gpu_refine_q6k_logits_top\|gpu_refine_q8_logits_top' src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu.c test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer GPU logits refine policy must expose behavior-named helpers, not quant-format helper names"
    fail=1
fi

if grep -n 'BnTransformerGPUQ4Q8\|bn_transformer_gpu_q4_q8\|bn_transformer_gpu_all2_q4q6_moe\|bn_transformer_gpu_all2_q4_moe\|bn_transformer_gpu_all2_moe_direct_route_enabled\|bn_gpu_policy_backend_all2_q4q6_moe_enabled\|bn_gpu_policy_backend_all2_kquant_moe_logits_refine_default_supported\|bn_gpu_policy_backend_all2_moe_direct_route_supported\|bn_gpu_policy_all2_kquant_moe_logits_refine_disabled\|\.all2_q4q6_moe\|use_q4_q8' include/gpu_policy.h include/transformer_plan_internal.h src/gpu_policy.c src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu.c src/transformer/gpu_emit.c test/test_transformer.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Transformer GPU small-dense and all-active-two MoE policy must expose behavior-named helpers"
    fail=1
fi

if grep -n 'BnTransformerGPUMoEAll2ResourcePolicy\|bn_transformer_gpu_moe_all2_resource_policy\|gpu_resolve_moe_all2_resources\|BnTransformerPrefillSharedAll2DecodeFallbackPolicy\|bn_transformer_prefill_shared_all2_decode_fallback_policy\|prefill_use_shared_all2_decode_fallback' include/transformer_prefill_internal.h src/transformer/prefill_policy.c src/transformer/prefill.c src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu.c test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer all-active-two MoE policies must not use all2 shorthand in policy names"
    fail=1
fi

if grep -n 'bn_gpu_policy_all2_q4q6_' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "All-active-two K-quant MoE policy must use behavior names, not all2_q4q6 shorthand"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_[a-z0-9_]*all2\|cuda_use_[a-z0-9_]*all2\|cuda_moe_cublas_all2\|moe_all2_\|use_moe_all2_\|use_q6k_all2_\|use_cublas_all2_\|all2_disable\|all2_f32\|use_all2_\|all2_fast_enabled\|all2_blocks' include/gpu_policy.h src/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1 ||
   grep -n 'bn_gpu_policy_cuda_[a-z0-9_]*all2\|cuda_use_[a-z0-9_]*all2\|cuda_moe_cublas_all2\|moe_all2_\|use_moe_all2_\|use_q6k_all2_\|use_cublas_all2_\|all2_disable\|all2_f32\|use_all2_\|all2_fast_enabled\|all2_blocks' src/gpu_cuda.cu | grep -v '_kernel' >/dev/null 2>&1; then
    echo "CUDA all-active-two MoE policy and helper names must not use all2 shorthand"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_q6k_moe_down_q8k_all_active_two_accum_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_pair4_sum_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_all_active_two_fixed_enabled\|bn_gpu_policy_cuda_q6k_moe_down_q8k_all_active_two_accum_4row_enabled' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "All-active-two K-quant MoE down policy must use behavior names"
    fail=1
fi

if grep -n 'bn_gpu_policy_all_active_two_kquant_q6k_pair_down_enabled\|bn_gpu_policy_all_active_two_kquant_q6k_pair_down_f32_layers_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_pair_down_f32_layer_selected\|bn_gpu_policy_all_active_two_kquant_q6k_ordered_down_enabled\|bn_gpu_policy_all_active_two_kquant_q6k_ordered_down_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_f32_down_default_enabled\|bn_gpu_policy_all_active_two_kquant_q6k_f32_down_default_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_f32_all_active_down_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_f32_cache_enabled\|bn_gpu_policy_all_active_two_kquant_q6k_float_4row_down_default_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_float_4row_down_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_f32_exact_4row_down_layer_selected\|bn_gpu_policy_all_active_two_kquant_q6k_f32_exact_4row_down_default_disabled\|bn_gpu_policy_all_active_two_kquant_q6k_f32_exact_4row_down_disabled' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "All-active-two K-quant MoE down policy must not use Q6K facade names"
    fail=1
fi

if grep -n 'bn_backend_quant_moe_all2_q4q6\|bn_backend_quant_moe_all2_q4_or_q6\|bn_backend_quant_moe_all_active_two_q4_or_q6_shape\|moe_all_active_two_q4_or_q6\|all_active_two_q4_or_q6' include/backend_quant.h include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Backend quant all-active-two MoE predicates must use behavior names, not Q4/Q6 shorthand"
    fail=1
fi

if grep -n 'BN_MODEL_ARCH_POLICY_SMALL_DENSE_EXACT_Q4_Q8\|BN_MODEL_ARCH_POLICY_SMALL_DENSE_Q8_LOGIT_REFINE\|bn_model_arch_allows_small_dense_exact_q4_q8\|bn_model_arch_small_dense_exact_q4_q8_to_layer\|bn_model_arch_allows_small_dense_q8_logit_refine\|bn_model_arch_uses_small_dense_q8_native_shape\|bn_gpu_policy_backend_small_dense_exact_q4_q8_supported\|bn_gpu_policy_small_dense_exact_q4_q8_disabled\|bn_gpu_policy_q4_q8_fused_gateup_enabled\|bn_gpu_policy_q4_q8_attn_only_enabled\|bn_gpu_policy_q4_q8_ffn_only_enabled\|bn_gpu_policy_q4_q8_from_layer_or_default\|bn_gpu_policy_q4_q8_to_layer_or_default\|bn_gpu_policy_q4_q8_ffn_down_enabled\|bn_gpu_policy_q4_q8_prepared_layer_default_enabled' include/model_arch.h include/gpu_policy.h src/model_arch.c src/gpu_policy.c src/transformer/gpu_policy.c test/test_transformer.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Small-dense exact-native policy must use behavior names, not Q4/Q8 internal helper names"
    fail=1
fi

if grep -n 'args\.q4_q8_\|int[[:space:]]\+q4_q8_' src/main.c >/dev/null 2>&1; then
    echo "CLI internal state for small-dense exact-native diagnostics must use behavior names"
    fail=1
fi

if grep -n 'bn_backend_quant_small_dense_q8_supported\|bn_backend_quant_cuda_small_dense_q8_supported\|bn_gpu_policy_small_dense_q8_cpu_attention_safe_disabled' include/backend_quant.h include/gpu_policy.h src/gpu_policy.c src/transformer/gpu_policy.c test/test_transformer.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Small-dense native-quant policy must use behavior names, not Q8 internal helper names"
    fail=1
fi

if grep -n 'bn_transformer_gpu_shared_q4_q8\|bn_gpu_policy_shared_q4_q8_dot_enabled\|shared_q4_q8_eligible\|use_shared_q4_q8' include/gpu_policy.h include/transformer_plan_internal.h src/gpu_policy.c src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu_emit.c test/test_transformer.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Transformer GPU shared MoE dot policy must expose behavior-named helpers"
    fail=1
fi

if grep -n 'bn_transformer_gpu_matvec_q8k_dot_flags\|bn_transformer_gpu_matvec_exact_q6k_flags' include/transformer_plan_internal.h src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu_emit.c test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer GPU matvec flag policy must use behavior-named K-quant helpers"
    fail=1
fi

if grep -n 'bn_backend_quant_gpu_matvec_q8k_dot_flag\|bn_backend_quant_gpu_matvec_exact_q6k_flag' src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu_emit.c test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer GPU matvec flag policy must use behavior-named backend K-quant helpers"
    fail=1
fi

if grep -n 'bn_transformer_gpu_qkv_split_q[58]_supported\|bn_transformer_gpu_qkv_split_byte_quant_supported\|use_q[58]_split\|use_byte_quant_split\|use_packed_q5_split' src/transformer/gpu_policy.c src/transformer/gpu_internal.h src/transformer/gpu_emit.c test/test_transformer.c >/dev/null 2>&1; then
    echo "Transformer GPU QKV split policy must use behavior-named helpers"
    fail=1
fi

if sed -n '/bn_transformer_gpu_validate_forward/,/if (c->dim > BN_TRANSFORMER_GPU_MAX_VLA_ELEMS)/p' src/transformer/gpu_policy.c | grep -n 'backend_large_native = transformer_gpu_backend_is_cuda\|transformer_gpu_backend_is_cuda(gpu)' >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-aware helpers for native graph policy"
    fail=1
fi

if awk '/^int bn_transformer_gpu_all_active_two_kquant_moe_layer_enabled/{flag=1} /^int bn_transformer_gpu_batch_prefill_enabled/{flag=0} flag{print}' src/transformer/gpu_policy.c | grep -n 'transformer_gpu_backend_is_cuda(gpu)' >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-aware helpers for fallback/exact/prefill support"
    fail=1
fi

if awk '/^int bn_transformer_gpu_batch_prefill_enabled/{flag=1} /^int bn_transformer_gpu_moe_cache_prefill_enabled/{flag=0} flag{print}' src/transformer/gpu_policy.c | grep -n 'transformer_gpu_backend_is_cuda(gpu)' >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend-aware helpers for prefill scheduling support"
    fail=1
fi

if grep -n 'getenv("BN_CUDA_DISABLE_MOE_ROUTER_TOPK")\|getenv("BN_CUDA_DISABLE_Q8_MOE_CPU_ROUTE_RESIDENT")\|getenv("BN_CUDA_DISABLE_MOE_ROUTED_FFN")\|getenv("BN_CUDA_ENABLE_MOE_ROUTER_GPU")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_GPU")\|getenv("BN_CUDA_DISABLE_MOE_ROUTER_DIFF2")\|getenv("BN_CUDA_DISABLE_MOE_ROUTE_ROUTED_FFN_BATCH")\|getenv("BN_CUDA_ENABLE_MOE_ROUTE_ROUTED_FFN_BATCH_LARGE")' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend GPU policy helpers for CUDA MoE route env vars"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_router_topk_enabled\|bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled\|bn_gpu_policy_cuda_moe_routed_ffn_enabled\|bn_gpu_policy_cuda_moe_router_gpu_enabled\|bn_gpu_policy_cuda_moe_router_diff2_enabled\|bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed\|bn_gpu_policy_cuda_moe_ffn_disabled\|bn_gpu_policy_cuda_moe_cpu_actual_override_enabled\|bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled\|bn_gpu_policy_cuda_moe_gateup_split_enabled' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use neutral helpers for MoE route policy"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_moe_router_topk_enabled\|bn_gpu_policy_cuda_q8_moe_cpu_route_resident_enabled\|bn_gpu_policy_cuda_moe_router_gpu_enabled\|bn_gpu_policy_cuda_moe_router_diff2_enabled\|bn_gpu_policy_cuda_moe_routed_ffn_batch_enabled\|bn_gpu_policy_cuda_moe_routed_ffn_batch_allowed\|bn_gpu_policy_cuda_moe_ffn_disabled\|bn_gpu_policy_cuda_moe_cpu_actual_override_enabled\|bn_gpu_policy_cuda_moe_shared_cpu_fallback_enabled\|bn_gpu_policy_cuda_moe_gateup_split_enabled\|bn_gpu_policy_q8_moe_cpu_route_resident_enabled\|bn_gpu_policy_byte_quant_moe_cpu_route_resident_enabled' include/gpu_policy.h src/gpu_policy.c src/gpu_cuda.cu test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "MoE route policy must expose/test behavior-named helpers, not CUDA implementation aliases"
    fail=1
fi

if grep -n 'bn_gpu_policy_cuda_large_hybrid_attention\|bn_gpu_policy_cuda_large_hybrid_cpu_attention_safe\|bn_gpu_policy_cuda_large_hybrid_prefill\|bn_gpu_policy_cuda_large_hybrid_argmax\|bn_gpu_policy_cuda_prefill_dense_chain\|bn_gpu_policy_cuda_prefill_hybrid_chain\|bn_gpu_policy_cuda_prefill_ssm_run_chain\|bn_gpu_policy_cuda_prefill_ssm_ffn_fuse\|bn_gpu_policy_cuda_prefill_moe_chain_debug\|bn_gpu_policy_cuda_prefill_hybrid_chain_debug\|bn_gpu_policy_cuda_prefill_direct_kv\|bn_gpu_policy_cuda_moe_prefill_enabled\|bn_gpu_policy_cuda_moe_prefill_min_tokens\|bn_gpu_policy_cuda_moe_cache_prefill\|bn_gpu_policy_cuda_moe_prefill_shared_fuse\|bn_gpu_policy_cuda_moe_route_batch_debug\|bn_gpu_policy_cuda_shared_q4_q8_dot_enabled\|bn_gpu_policy_cuda_shared_expert_gate_enabled\|bn_gpu_policy_cuda_ssm_graph_disabled' include/gpu_policy.h src/gpu_policy.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Prefill and large-hybrid policy must expose/test behavior-named helpers, not CUDA implementation aliases"
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

if grep -n 'BN_TRANSFORMER_CPU_HAS_NATIVE_Q8X_QUANT\|transformer_cpu_features_internal.h' src/transformer/gpu.c src/transformer/gpu_fallback.c >/dev/null 2>&1; then
    echo "GPU execution/fallback code must use CPU backend helpers for native Q8x ISA policy"
    fail=1
fi

if grep -n '__ARM_NEON\|arm_neon.h\|vdupq_n_f32\|vmaxq_f32\|vst1q_f32' include/transformer_math_internal.h >/dev/null 2>&1; then
    echo "include/transformer_math_internal.h must keep ISA-specific softmax policy in transformer backend sources"
    fail=1
fi

for file in \
    include/transformer_gqa_internal.h \
    include/transformer_logits_internal.h \
    include/transformer_ssm_internal.h
do
    if grep -n 'transformer_simd_internal.h\|simd_helpers.h\|arm_neon.h\|immintrin.h\|wasm_simd128.h' "$file" >/dev/null 2>&1; then
        echo "$file must keep SIMD helper policy in ISA-specific transformer sources"
        fail=1
    fi
done

for file in \
    src/gpu_policy.c \
    src/model_gpu.c \
    src/gpu_moe_bridge.c \
    src/main.c
do
    if grep -n 'bn_backend_quant_cuda_logits_q6_f32_cache_supported\|bn_backend_quant_cuda_moe_all_f16_cache_supported\|bn_backend_quant_cuda_moe_down_q6_f32_cache_supported\|bn_backend_quant_cuda_moe_down_cublas_cache_supported\|bn_backend_quant_cuda_moe_down_cublas_cache_elem_bytes\|bn_backend_quant_cuda_moe_down_q4_f32_cache_supported\|bn_backend_quant_cuda_moe_quant_only_after_cache\|bn_backend_quant_cuda_lazy_moe_aux_cache_candidate\|bn_backend_quant_cuda_moe_prefers_quant_only\|bn_backend_quant_cuda_aux_cache_supported\|bn_backend_quant_cuda_aux_cache_can_use_f16\|bn_backend_quant_cuda_aux_cache_uses_f32\|bn_backend_quant_cuda_aux_cache_prefers_large_budget\|bn_quant_format_cuda_\(logits_q6_f32_cache_supported\|moe_all_f16_cache_supported\|moe_down_q6_f32_cache_supported\|moe_down_cublas_cache_supported\|moe_down_cublas_cache_elem_bytes\|moe_down_q4_f32_cache_supported\|moe_quant_only_after_cache\|lazy_moe_aux_cache_candidate\|moe_prefers_quant_only\|aux_cache_supported\|aux_cache_can_use_f16\|aux_cache_uses_f32\|aux_cache_prefers_large_budget\)' "$file" >/dev/null 2>&1; then
        echo "$file must use quant format CUDA cache helpers for quant-format policy"
        fail=1
    fi
done

if grep -n 'bn_backend_quant_cuda_lazy_moe_aux_cache_candidate\|bn_backend_quant_cuda_lazy_moe_aux_cache_dequant_block\|bn_backend_quant_cuda_aux_cache_force_q4_f32\|bn_backend_quant_cuda_aux_cache_q6_can_use_f16\|bn_backend_quant_cuda_aux_cache_add_q6_f32\|bn_backend_quant_cuda_aux_cache_f32_storage\|bn_backend_quant_cuda_aux_cache_dequant_route\|BnBackendQuantCudaAuxCacheDequant\|BN_BACKEND_QUANT_CUDA_AUX_CACHE_DEQUANT_' src/gpu_cuda.cu >/dev/null 2>&1; then
    echo "src/gpu_cuda.cu must use neutral backend quant aux-cache helpers for quant-format policy"
    fail=1
fi

if grep -n 'bn_backend_quant_cuda_q5k_deint_pair_matvec\|bn_backend_quant_cuda_q6q4_pair_matvec\|bn_backend_quant_cuda_q4_pair_matvec\|bn_backend_quant_cuda_q8_small_ssm_matvec_candidate\|bn_backend_quant_cuda_f16_q8_matvec_candidate\|bn_backend_quant_cuda_f16_float_cache_matvec_candidate\|bn_backend_quant_cuda_f16_q5k_matvec_candidate\|bn_backend_quant_cuda_f16_q6k_matvec_candidate\|bn_backend_quant_cuda_logits_q6_matvec_candidate\|bn_backend_quant_cuda_q5_0_matvec_candidate\|bn_backend_quant_cuda_q6k_q8k_matvec_candidate\|bn_backend_quant_cuda_q6k_warp_matvec_candidate\|bn_backend_quant_cuda_q4k_q8k_matvec_candidate\|bn_backend_quant_cuda_q4k_q8_1_matvec_candidate\|bn_backend_quant_cuda_q5k_q8_1_matvec_candidate\|bn_backend_quant_cuda_q8_0_preq_matvec_candidate\|bn_backend_quant_cuda_q8_0_warp_matvec_candidate\|bn_backend_quant_cuda_q4k_q8k_matmul_candidate\|bn_backend_quant_cuda_q4k_q8_1_matmul_candidate\|bn_backend_quant_cuda_q5k_q8_1_matmul_candidate\|bn_backend_quant_cuda_q6k_q8k_matmul_candidate\|bn_backend_quant_cuda_q8_0_matmul_candidate\|bn_backend_quant_cuda_q5_0_matmul_candidate\|bn_backend_quant_cuda_q5_0_fused_gateup_candidate\|bn_backend_quant_cuda_q8_0_fused_gateup_candidate\|bn_backend_quant_cuda_q4k_fused_gateup_q8k_candidate\|bn_backend_quant_cuda_q4k_fused_gateup_q8_1_candidate\|bn_backend_quant_cuda_q5k_fused_gateup_q8_1_candidate\|bn_backend_quant_cuda_matvec_allows_fused_bias\|bn_backend_quant_cuda_split_allows_fused_bias\|bn_backend_quant_cuda_q4k_split_q8k_candidate\|bn_backend_quant_cuda_q4k_split_q8_1_candidate\|bn_backend_quant_cuda_q5k_split_q8_1_candidate\|bn_backend_quant_cuda_q8_0_split_candidate\|bn_backend_quant_cuda_q4k_split_value_fuse_candidate\|bn_backend_quant_cuda_q6k_split_value_fuse_candidate\|bn_backend_quant_cuda_split_value_fuse_candidate\|bn_backend_quant_cuda_q5_0_pair_matmul\|bn_backend_quant_cuda_q8_0_pair_matmul\|bn_backend_quant_cuda_q4k_pair_matmul\|bn_backend_quant_cuda_q5k_pair_matmul\|bn_backend_quant_cuda_q6_logits_argmax_candidate\|bn_backend_quant_cuda_lazy_moe_aux_cache_candidate\|bn_backend_quant_cuda_lazy_moe_aux_cache_dequant_block\|bn_backend_quant_cuda_aux_cache_force_q4_f32\|bn_backend_quant_cuda_aux_cache_q6_can_use_f16\|bn_backend_quant_cuda_aux_cache_add_q6_f32\|bn_backend_quant_cuda_aux_cache_f32_storage\|bn_backend_quant_cuda_aux_cache_dequant_route\|BN_BACKEND_QUANT_CUDA_AUX_CACHE_DEQUANT_' test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "test/test_gpu_backend.c must cover neutral backend quant helper names"
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

if grep -n 'm->config\.n_experts <= 0' src/moe_cache.c >/dev/null 2>&1; then
    echo "src/moe_cache.c must use model_arch helpers for loaded-model MoE presence policy"
    fail=1
fi

if grep -n 'c->n_experts > 0' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model_arch helpers for loaded-model MoE presence policy"
    fail=1
fi

if grep -n 'bn_gguf_get_u32(f, key)' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model_arch helpers for arch-prefixed scalar GGUF metadata"
    fail=1
fi

if grep -n 'bn_gguf_get_f32(f, key)' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model_arch helpers for arch-prefixed float GGUF metadata"
    fail=1
fi

if grep -n 'gguf_get_u32_or_i32_array' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model_arch helpers for arch-prefixed array GGUF metadata"
    fail=1
fi

if grep -n 'snprintf(key, sizeof(key), "%s\\.' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must not build arch-prefixed GGUF metadata keys directly"
    fail=1
fi

if grep -n 'bn_model_arch_uses_full_rope_text_dims(arch)' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use config policy helpers for full RoPE text dims"
    fail=1
fi

if grep -n 'bn_model_arch_uses_full_rope_text_dims' include/model_arch.h src/model_arch.c >/dev/null 2>&1; then
    echo "model_arch must expose full RoPE text dims through config policy helpers"
    fail=1
fi

if grep -n 'if (c->full_attn_interval > 0)' src/model.c >/dev/null 2>&1; then
    echo "src/model.c must use model_arch helpers for hybrid layout predicates"
    fail=1
fi

if grep -n 'c->n_experts > 0' src/model_session.c >/dev/null 2>&1; then
    echo "src/model_session.c must use model_arch helpers for loaded-session MoE presence policy"
    fail=1
fi

if grep -n 'full_attn_interval\|n_attn_layers = (c->full_attn_interval > 0)\|n_ssm_layers = c->n_layers - n_attn_layers' src/model_session.c >/dev/null 2>&1; then
    echo "src/model_session.c must use model_arch helpers for hybrid layer layout/count policy"
    fail=1
fi

if grep -n 'BN_GPU_BACKEND_CUDA\|BN_CUDA_\|bn_quant_format_supports_gpu_dense_graph_q8\|bn_model_arch_cpu_force_float_kquant' src/transformer.c >/dev/null 2>&1; then
    echo "src/transformer.c must use GPU policy helpers for CUDA matvec fallback policy"
    fail=1
fi

if grep -n 'bn_quant_format_supports_gpu_dense_graph_q8' include/quant.h include/backend_quant.h src/quant/registry.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Small-dense native-quant format policy must use behavior names, not Q8 internal helper names"
    fail=1
fi

if grep -n 'BN_QUANT_CAP_GPU_SMALL_DENSE\|bn_quant_format_supports_gpu_small_dense\|bn_backend_quant_small_dense_' include/quant.h include/backend_quant.h src/quant/registry.c test/test_gpu_backend.c >/dev/null 2>&1; then
    echo "Dense graph quant capability helpers must not use small-dense model-shape names"
    fail=1
fi

if grep -n 'BN_QUANT_CAP_GPU_DENSE_GRAPH_Q8\|BN_QUANT_CAP_GPU_DENSE_GRAPH_Q8_FORMAT' include/quant.h src/quant/registry.c >/dev/null 2>&1; then
    echo "Small-dense native-quant capability must use behavior names, not Q8 internal capability names"
    fail=1
fi

if grep -n '#include "model_arch.h"\|bn_model_arch_' src/transformer.c >/dev/null 2>&1; then
    echo "src/transformer.c must use transformer planning helpers for model-family policy"
    fail=1
fi

for file in \
    src/transformer/cpu.c \
    src/transformer/prefill.c
do
    if grep -n 'bn_transformer_cpu_uses_scalar_hybrid_ssm\|bn_transformer_ssm_.*_scalar_range' "$file" >/dev/null 2>&1; then
        echo "$file must use backend SSM op selectors for scalar-hybrid SSM CPU policy"
        fail=1
    fi
done

if grep -n 'bn_transformer_gpu_backend_is_cuda(prefill_gpu)\|bn_transformer_gpu_backend_is_cuda(gpu_ffn)\|int cuda_hybrid_prefill\|int cuda_moe_prefill' src/transformer/prefill.c >/dev/null 2>&1; then
    echo "src/transformer/prefill.c must use GPU policy helpers for prefill chain policy"
    fail=1
fi

if grep -n 'c->n_layers - 33 - 1\|n_layers - 33 - 1' src/transformer/gpu.c >/dev/null 2>&1; then
    echo "src/transformer/gpu.c must use GPU policy helpers for small-dense Q4/Q8 layer cutoff policy"
    fail=1
fi

if grep -n 'c->n_layers - 33 - 1\|n_layers - 33 - 1' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "src/transformer/gpu_policy.c must use model_arch helpers for small-dense Q4/Q8 layer cutoff policy"
    fail=1
fi

if grep -n 'full_attn_interval' src/transformer/prefill.c test/test_prefill.c >/dev/null 2>&1; then
    echo "prefill code and tests must use model_arch helpers for hybrid layer layout policy"
    fail=1
fi

if grep -n 'ssm\.wqkv\.cols ==\|ssm\.wqkv\.rows == .*2 \*' src/transformer/gpu_emit.c src/transformer/prefill.c >/dev/null 2>&1; then
    echo "Transformer GPU/prefill code must use packed QKV shape policy helpers"
    fail=1
fi

if grep -n 'p->q_wide = .*attn\.wq\.rows\|!p->q_gated && lw->attn\.wq\.data' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer planning must use wide-Q projection policy helpers"
    fail=1
fi

if grep -n 'attn\.wq\.data && .*attn\.wq\.rows >\|attn\.wq\.rows > .*q_dim' src/transformer/plan.c src/transformer/gpu_emit.c >/dev/null 2>&1; then
    echo "Transformer planning/emit code must use gated-Q projection policy helpers"
    fail=1
fi

if grep -n 'p->head_size = .*attn\.head_size > 0\|p->kv_dim = .*attn\.kv_dim > 0\|p->n_kv_heads = .*attn\.n_kv_heads > 0\|p->kv_mul = .*attn\.kv_mul > 0' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer attention planning must use attention shape default helpers"
    fail=1
fi

if grep -n 'p->qk_stride = .*qk_norm_per_head\|p->has_qk_norm = .*attn\.q_norm\|p->has_bias = .*attn\.q_bias' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer attention planning must use attention feature helpers"
    fail=1
fi

if grep -n 'p->kind = .*BN_LAYER_ATTN_\|p->kind = .*BN_LAYER_SSM\|p->kind = p->is_attn' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer layer-shape planning must use layer-kind policy helpers"
    fail=1
fi

if grep -n 'p->needs_cpu_fallback = p->placement == BN_EXEC_GPU\|p->use_flash = .*flash_attn\|p->use_packed_qkv = .*qkv_stacked\|p->use_qkv_split = .*qkv_stacked\|p->placement == BN_EXEC_GPU && !k_bias' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer attention execution planning must use attention policy helpers"
    fail=1
fi

if grep -n 'p->hidden_dim = .*ffn\.ffn_up\.rows' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer FFN planning must use hidden-dim policy helpers"
    fail=1
fi

if grep -n 'p->kind = .*ffn_kind == BN_LAYER_FFN_MOE\|p->kind = .*has_ffn_gate' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer FFN planning must use FFN-kind policy helpers"
    fail=1
fi

if grep -n 'p->has_gate = .*has_ffn_gate\|p->has_sub_norm = .*ffn_sub_norm\|p->use_fused_gateup_silu = .*p->placement\|p->use_gateup_split = .*p->placement\|p->placement == BN_EXEC_GPU.*BN_FUSION_RESIDUAL_RMSNORM\|p->kind == BN_FFN_MOE && p->placement == BN_EXEC_GPU' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer FFN execution planning must use FFN policy helpers"
    fail=1
fi

if grep -n 'p->has_shared_expert = .*has_shared_expert.*shared_expert_gate' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer MoE planning must use shared-expert policy helpers"
    fail=1
fi

if grep -n 'p->placement == BN_EXEC_GPU && .*moe\.router_weight' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer MoE planning must use CPU-fallback policy helpers"
    fail=1
fi

if grep -n 'p->use_qkvz_stack = .*p->placement\|p->use_alpha_beta_stack = .*p->placement' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer SSM execution planning must use SSM policy helpers"
    fail=1
fi

if grep -n 'p->use_i8_output = .*emb_out_i8\|p->kind = .*output_weight\.type\|p->weight_type = .*output_weight\.type\|} else if (w->emb_out_i8)\|} else if (bn_transformer_logits_tied_uses_.*w->emb_type)' src/transformer/plan.c >/dev/null 2>&1; then
    echo "Transformer logits planning must use logits policy helpers"
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

if grep -n 'bn_transformer_gpu_backend_is_cuda' include/transformer_plan_internal.h src/transformer/gpu_internal.h test/test_transformer.c >/dev/null 2>&1; then
    echo "shared transformer plan/test APIs must not expose CUDA backend predicates"
    fail=1
fi

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

if grep -n 'full_attn_interval' src/transformer/plan.c >/dev/null 2>&1; then
    echo "src/transformer/plan.c must use model_arch helpers for hybrid layer layout policy"
    fail=1
fi

if grep -n 'full_attn_interval' src/prompt_cache.c >/dev/null 2>&1; then
    echo "src/prompt_cache.c must use model_arch helpers for hybrid layer layout policy"
    fail=1
fi

if grep -n 'full_attn_interval\|c->n_layers - n_attn' src/session.c >/dev/null 2>&1; then
    echo "src/session.c must use model_arch helpers for hybrid layer layout policy"
    fail=1
fi

for file in include/model_config.h include/model_weights.h include/model_run_state.h include/transformer_plan_internal.h
do
    if grep -n 'Gemma4\|Qwen\|gemma4_\|qwen2_moe\|qwen2moe_\|BN_MODEL_ARCH_POLICY_\|RMSNORM_LLAMA\|requires_llama_scalar' "$file" >/dev/null 2>&1; then
        echo "$file must expose behavior-named shared model state, not family-prefixed fields or comments"
        fail=1
    fi
done

if awk '/^static int small_dense_backend_native_by_default/{flag=1} /^int bn_transformer_gpu_all_active_two_kquant_moe_cpu_attn_safe_default/{flag=0} flag{print}' src/transformer/gpu_policy.c | grep -n 'bn_backend_quant_small_dense' >/dev/null 2>&1; then
    echo "Small-dense backend default policy must compose behavior-named quant helpers"
    fail=1
fi

if awk '/^int bn_transformer_gpu_backend_matvec_fallback_kept/{flag=1} /^BnTransformerGPUMatvecFallbackPolicy/{flag=0} flag{print}' src/transformer/gpu_policy.c | grep -n 'bn_backend_quant_small_dense' >/dev/null 2>&1; then
    echo "GPU matvec fallback policy must compose behavior-named small-dense quant helpers"
    fail=1
fi

if grep -n 'small_dense_tensor_quant_supported\|small_dense_weight_quant_supported\|small_dense_model_quant_supported' src/transformer/gpu_policy.c >/dev/null 2>&1; then
    echo "Transformer GPU policy must use backend quant model helpers for dense quant capability aggregation"
    fail=1
fi

for file in \
    src/transformer/cpu_backend.c \
    src/transformer/prefill_backend.c \
    src/transformer/gpu_fallback.c \
    src/transformer/gpu_emit.c
do
    if grep -n '#include "model_arch.h"\|bn_model_arch_activation_' "$file" >/dev/null 2>&1; then
        echo "$file must use transformer policy helpers for activation classification"
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

if grep -n 'bn_model_arch_moe_forces_float_kquant_gateup\|BN_MATVEC_TASK_FORCE_FLOAT_KQUANT' src/moe_execute.c >/dev/null 2>&1; then
    echo "src/moe_execute.c must use MoE policy helpers for float K-quant gate/up task flags"
    fail=1
fi

if ! grep -n 'bn_model_arch_moe_forces_float_kquant_gateup\|BN_MATVEC_TASK_FORCE_FLOAT_KQUANT' src/moe_policy.c >/dev/null 2>&1; then
    echo "MoE float K-quant gate/up task flag policy must live in src/moe_policy.c"
    fail=1
fi

for file in \
    src/transformer.c \
    src/model.c \
    src/model_session.c \
    src/model_embed.c \
    src/transformer/gqa_tq_scalar.c
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
