#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building bitnet.c WASM module..."

emcc \
    "$PROJECT_DIR/src/gguf.c" \
    "$PROJECT_DIR/src/quant/fp16.c" \
    "$PROJECT_DIR/src/quant/dequant.c" \
    "$PROJECT_DIR/src/quant/dispatch.c" \
    "$PROJECT_DIR/src/quant/x_quant_wasm.c" \
    "$PROJECT_DIR/src/quant/i2s_wasm.c" \
    "$PROJECT_DIR/src/quant/i2s_scalar.c" \
    "$PROJECT_DIR/src/quant/tq2_wasm.c" \
    "$PROJECT_DIR/src/quant/tq2_scalar.c" \
    "$PROJECT_DIR/src/quant/tq1_wasm.c" \
    "$PROJECT_DIR/src/quant/tq1_scalar.c" \
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
    "$PROJECT_DIR/src/model.c" \
    "$PROJECT_DIR/src/moe.c" \
    "$PROJECT_DIR/src/transformer.c" \
    "$PROJECT_DIR/src/transformer/rmsnorm_wasm.c" \
    "$PROJECT_DIR/src/transformer/rmsnorm_scalar.c" \
    "$PROJECT_DIR/src/transformer/gqa_wasm.c" \
    "$PROJECT_DIR/src/transformer/gqa_scalar.c" \
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
    "$PROJECT_DIR/wasm/api.c" \
    -I"$PROJECT_DIR/include" \
    -std=c11 -D_GNU_SOURCE -Wall -Wextra \
    -O3 -flto -msimd128 -mrelaxed-simd \
    -sWASM=1 \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4294967296 \
    -sFILESYSTEM=0 \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=BitNet \
    -sENVIRONMENT=worker \
    '-sEXPORTED_FUNCTIONS=[
        "_bitnet_init",
        "_bitnet_forward_token",
        "_bitnet_get_logits",
        "_bitnet_sample",
        "_bitnet_encode",
        "_bitnet_decode",
        "_bitnet_vocab_size",
        "_bitnet_bos_id",
        "_bitnet_eos_id",
        "_bitnet_free",
        "_bitnet_chat_init",
        "_bitnet_chat_reset",
        "_bitnet_chat_submit",
        "_bitnet_chat_next",
        "_bitnet_chat_end_turn",
        "_malloc",
        "_free"
    ]' \
    '-sEXPORTED_RUNTIME_METHODS=["cwrap","UTF8ToString","HEAPU8","HEAPF32"]' \
    -o "$SCRIPT_DIR/bitnet.js"

echo "WASM build complete: wasm/bitnet.js + wasm/bitnet.wasm"
