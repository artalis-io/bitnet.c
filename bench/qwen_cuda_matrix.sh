#!/usr/bin/env bash
# Qwen CUDA parity matrix for real GGUF model coverage.
#
# Defaults to /data/models/gguf. Set REQUIRE_MODELS=1 to make missing Qwen
# dense/MoE families fail instead of skip. Set RUN_LLAMA_COMPARE=1 for a short
# generation comparison against llama.cpp CUDA, and RUN_BENCH=1 for throughput
# comparison via bench/cuda_compare.sh.

set -uo pipefail

ROOT="${BN_MODEL_ROOT:-/data/models/gguf}"
COHERENCE="${COHERENCE:-./test_coherence}"
COMPARE_LLAMA="${COMPARE_LLAMA:-./test/compare_llama.sh}"
CUDA_COMPARE="${CUDA_COMPARE:-./bench/cuda_compare.sh}"
BITNET="${BITNET:-./bitnet}"
REQUIRE_MODELS="${REQUIRE_MODELS:-0}"
RUN_COHERENCE="${RUN_COHERENCE:-1}"
RUN_SHARDED_MOE_SMOKE="${RUN_SHARDED_MOE_SMOKE:-1}"
RUN_SHARDED_MOE_COHERENCE="${RUN_SHARDED_MOE_COHERENCE:-1}"
RUN_LLAMA_COMPARE="${RUN_LLAMA_COMPARE:-0}"
RUN_BENCH="${RUN_BENCH:-0}"
N_TOKENS="${N_TOKENS:-8}"
THREADS="${THREADS:-8}"
MAXSEQ="${MAXSEQ:-512}"

fail=0
ran=0
missing=0
bench_models=""

find_model() {
    env_name=$1
    rel_dir=$2
    shift 2
    search_root=$ROOT
    if [ -n "$rel_dir" ] && [ -d "$ROOT/$rel_dir" ]; then
        search_root="$ROOT/$rel_dir"
    fi
    value=$(eval "printf '%s' \"\${$env_name:-}\"")
    if [ -n "$value" ]; then
        printf '%s\n' "$value"
        return 0
    fi
    if [ -n "$search_root" ] && [ -d "$search_root" ]; then
        for pattern in "$@"; do
            found=$(find "$search_root" -type f -iname "$pattern" | sort | head -n 1)
            if [ -n "$found" ]; then
                printf '%s\n' "$found"
                return 0
            fi
        done
    fi
    return 1
}

run_case() {
    name=$1
    env_name=$2
    rel_dir=$3
    shift 3

    if path=$(find_model "$env_name" "$rel_dir" "$@"); then
        if [[ "$path" == *-of-*.gguf ]]; then
            echo "RUN $name sharded mmap: $path"
            ran=$((ran + 1))
            if [ "$RUN_BENCH" = "1" ]; then
                bench_models="${bench_models}${bench_models:+ }$path"
            fi
            if [ "$RUN_COHERENCE" = "1" ] && [ "$RUN_SHARDED_MOE_COHERENCE" = "1" ]; then
                "$COHERENCE" "$path" --cuda --require-all-tokens || fail=1
            elif [ "$RUN_SHARDED_MOE_SMOKE" = "1" ]; then
                "$BITNET" "$path" --cuda -n 0 --maxseq 32 --quiet || fail=1
            fi
            return
        fi
        echo "RUN $name: $path"
        ran=$((ran + 1))
        bench_models="${bench_models}${bench_models:+ }$path"
        if [ "$name" = "Qwen 2.5 dense" ] || [ "$name" = "Qwen 3 dense" ]; then
            echo "  using default small dense Q4_K/Q6_K CUDA graph correctness fallback"
        fi
        if [ "$RUN_COHERENCE" = "1" ]; then
            "$COHERENCE" "$path" --cuda --require-all-tokens || fail=1
        fi
        if [ "$RUN_LLAMA_COMPARE" = "1" ]; then
            "$COMPARE_LLAMA" "$path" --cuda --llama-cuda -n "$N_TOKENS" -t "$THREADS" --maxseq "$MAXSEQ" || fail=1
        fi
    else
        echo "SKIP $name: set $env_name or BN_MODEL_ROOT=$ROOT"
        missing=$((missing + 1))
    fi
}

run_case "Qwen 2.5 dense" "BN_MODEL_QWEN25" \
    "qwen2_5/3b" \
    "Qwen2.5*.gguf" "qwen2.5*.gguf"
run_case "Qwen 2.5 sparse MoE" "BN_MODEL_QWEN25_MOE" \
    "qwen2_5/moe_2x1_5b" \
    "Qwen2.5*MOE*.gguf" "qwen2.5*moe*.gguf"
run_case "Qwen 3 dense" "BN_MODEL_QWEN3_DENSE" \
    "qwen3/4b" \
    "Qwen3-*B*.gguf" "qwen3-*b*.gguf"
run_case "Qwen 3 sparse MoE" "BN_MODEL_QWEN3_MOE" \
    "qwen3/30b_a3b" \
    "Qwen3-*A*B*.gguf" "qwen3-*a*b*.gguf" "Qwen3*MOE*00001-of-*.gguf" "qwen3*moe*00001-of-*.gguf"
run_case "Qwen 3.5 dense" "BN_MODEL_QWEN35_DENSE" \
    "qwen3_5" \
    "Qwen3.5-27B*.gguf" "qwen3.5-27b*.gguf"
run_case "Qwen 3.5 sparse MoE" "BN_MODEL_QWEN35_MOE" \
    "qwen3_5" \
    "Qwen3.5-*-UD-Q3_K_XL-00001-of-*.gguf" "qwen3.5-*-ud-q3_k_xl-00001-of-*.gguf" \
    "Qwen3.5-*A*B*00001-of-*.gguf" "qwen3.5-*a*b*00001-of-*.gguf" "Qwen3.5*MOE*00001-of-*.gguf" "qwen3.5*moe*00001-of-*.gguf"
run_case "Qwen 3.6 dense" "BN_MODEL_QWEN36_DENSE" \
    "qwen3_6" \
    "Qwen3.6-27B*.gguf" "qwen3.6-27b*.gguf"
run_case "Qwen 3.6 sparse MoE" "BN_MODEL_QWEN36_MOE" \
    "qwen3_6" \
    "Qwen3.6-*A*B*.gguf" "qwen3.6*a*b*.gguf"

if [ "$RUN_BENCH" = "1" ] && [ -n "$bench_models" ]; then
    MODELS="$bench_models" "$CUDA_COMPARE" || fail=1
fi

if [ "$REQUIRE_MODELS" = "1" ] && [ "$missing" -ne 0 ]; then
    echo "Qwen CUDA matrix FAILED: $missing required model case(s) missing"
    exit 1
fi

if [ "$fail" -ne 0 ]; then
    echo "Qwen CUDA matrix FAILED"
    exit 1
fi

echo "Qwen CUDA matrix PASSED: ran=$ran skipped=$missing"
