#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export EM_CACHE="${EM_CACHE:-/tmp/bitnet-emscripten-cache}"
mkdir -p "$EM_CACHE"

WASM_SIMD_FLAGS=(-msimd128)
if [ "${BN_WASM_RELAXED:-1}" != "0" ]; then
    WASM_SIMD_FLAGS+=(-mrelaxed-simd)
fi
WASM_THREAD_FLAGS=()
if [ "${BN_WASM_THREADS:-0}" != "0" ]; then
    WASM_THREAD_FLAGS+=(-pthread "-sPTHREAD_POOL_SIZE=${BN_WASM_PTHREAD_POOL_SIZE:-7}")
fi

echo "Building WASM benchmark..."

emcc \
    "$PROJECT_DIR/bench/bench_kernels.c" \
    "$PROJECT_DIR/src/gguf.c" \
    "$PROJECT_DIR/src/quant/fp16.c" \
    "$PROJECT_DIR/src/quant/dequant.c" \
    "$PROJECT_DIR/src/quant/registry.c" \
    "$PROJECT_DIR/src/quant/dispatch.c" \
    "$PROJECT_DIR/src/quant/x_quant_wasm.c" \
    "$PROJECT_DIR/src/quant/i2s_wasm.c" \
    "$PROJECT_DIR/src/quant/i2s_scalar.c" \
    "$PROJECT_DIR/src/quant/tq2_scalar.c" \
    "$PROJECT_DIR/src/quant/tq2_wasm.c" \
    "$PROJECT_DIR/src/quant/tq1_scalar.c" \
    "$PROJECT_DIR/src/quant/tq1_wasm.c" \
    "$PROJECT_DIR/src/quant/q8_wasm.c" \
    "$PROJECT_DIR/src/quant/q8_scalar.c" \
    "$PROJECT_DIR/src/quant/q4_wasm.c" \
    "$PROJECT_DIR/src/quant/q4_scalar.c" \
    "$PROJECT_DIR/src/quant/q6k_wasm.c" \
    "$PROJECT_DIR/src/quant/q6k_scalar.c" \
    "$PROJECT_DIR/src/quant/q8k_wasm.c" \
    "$PROJECT_DIR/src/quant/q8k_scalar.c" \
    "$PROJECT_DIR/src/quant/q4k_wasm.c" \
    "$PROJECT_DIR/src/quant/q4k_scalar.c" \
    "$PROJECT_DIR/src/quant/q5k_wasm.c" \
    "$PROJECT_DIR/src/quant/q5k_scalar.c" \
    "$PROJECT_DIR/src/quant/q3k_wasm.c" \
    "$PROJECT_DIR/src/quant/q3k_scalar.c" \
    "$PROJECT_DIR/src/quant/q2k_wasm.c" \
    "$PROJECT_DIR/src/quant/q2k_scalar.c" \
    "$PROJECT_DIR/src/quant/q4_1_wasm.c" \
    "$PROJECT_DIR/src/quant/q4_1_scalar.c" \
    "$PROJECT_DIR/src/quant/f32_wasm.c" \
    "$PROJECT_DIR/src/quant/f32_scalar.c" \
    "$PROJECT_DIR/src/quant/f16_wasm.c" \
    "$PROJECT_DIR/src/quant/f16_scalar.c" \
    "$PROJECT_DIR/src/quant/bf16_wasm.c" \
    "$PROJECT_DIR/src/quant/bf16_scalar.c" \
    "$PROJECT_DIR/src/quant/iq4nl_wasm.c" \
    "$PROJECT_DIR/src/quant/iq4nl_scalar.c" \
    "$PROJECT_DIR/src/quant/iq4xs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq4xs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq3xxs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq3xxs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq3s_wasm.c" \
    "$PROJECT_DIR/src/quant/iq3s_scalar.c" \
    "$PROJECT_DIR/src/quant/iq2xxs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq2xxs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq2xs_wasm.c" \
    "$PROJECT_DIR/src/quant/iq2xs_scalar.c" \
    "$PROJECT_DIR/src/quant/iq2s_wasm.c" \
    "$PROJECT_DIR/src/quant/iq2s_scalar.c" \
    "$PROJECT_DIR/src/turboquant.c" \
    "$PROJECT_DIR/src/model.c" \
    "$PROJECT_DIR/src/backend_layout.c" \
    "$PROJECT_DIR/src/backend_model.c" \
    "$PROJECT_DIR/src/moe.c" \
    "$PROJECT_DIR/src/gpu_moe_cache.c" \
    "$PROJECT_DIR/src/transformer.c" \
    "$PROJECT_DIR/src/transformer/cpu.c" \
    "$PROJECT_DIR/src/transformer/gpu.c" \
    "$PROJECT_DIR/src/transformer/gpu_emit.c" \
    "$PROJECT_DIR/src/transformer/kv.c" \
    "$PROJECT_DIR/src/transformer/logits.c" \
    "$PROJECT_DIR/src/transformer/plan.c" \
    "$PROJECT_DIR/src/transformer/prefill.c" \
    "$PROJECT_DIR/src/transformer/rmsnorm_wasm.c" \
    "$PROJECT_DIR/src/transformer/rmsnorm_scalar.c" \
    "$PROJECT_DIR/src/transformer/gqa_wasm.c" \
    "$PROJECT_DIR/src/transformer/gqa_scalar.c" \
    "$PROJECT_DIR/src/transformer/gqa_tq_scalar.c" \
    "$PROJECT_DIR/src/transformer/logits_wasm.c" \
    "$PROJECT_DIR/src/transformer/logits_scalar.c" \
    "$PROJECT_DIR/src/transformer/ssm_wasm.c" \
    "$PROJECT_DIR/src/transformer/ssm_scalar.c" \
    "$PROJECT_DIR/src/tokenizer.c" \
    "$PROJECT_DIR/src/sampler.c" \
    "$PROJECT_DIR/src/platform.c" \
    "$PROJECT_DIR/src/threadpool.c" \
    "$PROJECT_DIR/src/sh_arena.c" \
    "$PROJECT_DIR/src/sh_log.c" \
    "$PROJECT_DIR/src/bn_alloc.c" \
    "$PROJECT_DIR/src/session.c" \
    -I"$PROJECT_DIR/include" \
    -std=c11 -D_GNU_SOURCE -O3 -flto "${WASM_SIMD_FLAGS[@]}" "${WASM_THREAD_FLAGS[@]}" \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4294967296 \
    -sSTACK_SIZE=1048576 \
    --minify 0 \
    -sNODERAWFS=1 \
    -sENVIRONMENT=node \
    -sEXIT_RUNTIME=1 \
    -o "$SCRIPT_DIR/bench_wasm.js"

echo "WASM benchmark built: bench/bench_wasm.js"
echo "Usage: node bench/bench_wasm.js model.gguf [--iters N]"
echo "Requires a runtime with WASM relaxed SIMD when built with the default BN_WASM_RELAXED=1."
echo "On Node.js, use a recent runtime such as Node 22; older Node 18 builds may reject relaxed SIMD opcodes."
echo "Use BN_WASM_RELAXED=0 ./bench/bench_wasm.sh for older SIMD128-only runtimes."
echo "Use BN_WASM_THREADS=1 BN_WASM_PTHREAD_POOL_SIZE=N ./bench/bench_wasm.sh to enable pthreads."
echo "Use BN_WASM_Q4_CANONICAL4=1 node bench/bench_wasm.js ... to probe the canonical 4-row Q4_0 kernel."
echo "Pass --q4-expand to benchmark the experimental expanded-int8 Q4_0 probe."
