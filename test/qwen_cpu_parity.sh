#!/usr/bin/env bash
set -euo pipefail

ROOT=${BN_MODEL_ROOT:-models}
LEVEL=${QWEN_CPU_PARITY_LEVEL:-standard}
REQUIRE_MODELS=${REQUIRE_MODELS:-0}
COMPARE=${COMPARE_LLAMA:-./test/compare_llama.sh}
NEON_BIN=${BITNET_NEON:-./bitnet}
SCALAR_BIN=${BITNET_SCALAR:-./bitnet_scalar}
AVX2_BIN=${BITNET_AVX2:-./bitnet_avx2}
AVX512_BIN=${BITNET_AVX512:-./bitnet_avx512}
BACKENDS=${QWEN_CPU_PARITY_BACKENDS:-${CPU_PARITY_BACKENDS:-neon,scalar}}
CASES=${QWEN_CPU_PARITY_CASES:-all}

fail=0
ran=0
missing=0

find_model() {
    local env_name=$1
    local pattern=$2
    local value
    value=$(eval "printf '%s' \"\${$env_name:-}\"")
    if [[ -n "$value" ]]; then
        printf '%s\n' "$value"
        return 0
    fi
    if [[ -n "$ROOT" && -d "$ROOT" ]]; then
        local found
        found=$(find "$ROOT" -type f -iname "$pattern" 2>/dev/null | head -n 1 || true)
        if [[ -n "$found" ]]; then
            printf '%s\n' "$found"
            return 0
        fi
    fi
    return 1
}

run_backend() {
    local backend=$1
    local bitnet=$2
    local name=$3
    local model=$4
    local tokens=$5
    shift 5

    if [[ ! -x "$bitnet" ]]; then
        echo "FAIL $backend $name: missing executable $bitnet"
        fail=1
        return
    fi

    echo "RUN $backend $name: $model (-n $tokens)"
    ran=$((ran + 1))
    if ! BITNET="$bitnet" "$COMPARE" "$model" -n "$tokens" --strict -t 1 \
        --llama-cache-k f32 --llama-cache-v f32 "$@"; then
        fail=1
    fi
}

validate_backends() {
    local backend
    for backend in ${BACKENDS//,/ }; do
        case "$backend" in
            neon|native|scalar|avx2|avx512) ;;
            "")
                echo "ERROR: QWEN_CPU_PARITY_BACKENDS contains an empty backend" >&2
                exit 1
                ;;
            *)
                echo "ERROR: unknown Qwen CPU parity backend '$backend'" >&2
                echo "Valid backends: neon,native,scalar,avx2,avx512" >&2
                exit 1
                ;;
        esac
    done
}

run_selected_backends() {
    local name=$1
    local model=$2
    local tokens=$3
    shift 3

    local backend
    for backend in ${BACKENDS//,/ }; do
        case "$backend" in
            neon) run_backend "NEON" "$NEON_BIN" "$name" "$model" "$tokens" "$@" ;;
            native) run_backend "native" "$NEON_BIN" "$name" "$model" "$tokens" "$@" ;;
            scalar) run_backend "scalar" "$SCALAR_BIN" "$name" "$model" "$tokens" "$@" ;;
            avx2) run_backend "AVX2" "$AVX2_BIN" "$name" "$model" "$tokens" "$@" ;;
            avx512) run_backend "AVX512" "$AVX512_BIN" "$name" "$model" "$tokens" "$@" ;;
        esac
    done
}

run_case() {
    local case_id=$1
    local name=$2
    local env_name=$3
    local pattern=$4
    local standard_tokens=$5
    local full_tokens=$6
    shift 6

    if [[ "$CASES" != "all" ]]; then
        case ",$CASES," in
            *",$case_id,"*) ;;
            *) return ;;
        esac
    fi

    local tokens=$standard_tokens
    if [[ "$LEVEL" == "full" ]]; then
        tokens=$full_tokens
    fi

    local model
    if ! model=$(find_model "$env_name" "$pattern"); then
        echo "SKIP $name: set $env_name or BN_MODEL_ROOT"
        missing=$((missing + 1))
        return
    fi

    run_selected_backends "$name" "$model" "$tokens" "$@"
}

case "$LEVEL" in
    standard|full) ;;
    *)
        echo "ERROR: QWEN_CPU_PARITY_LEVEL must be standard or full" >&2
        exit 1
        ;;
esac

validate_backends

run_case "qwen25" "Qwen 2.5 dense" "BN_MODEL_QWEN25" \
    "*qwen2.5*.gguf" 1 16 --flash
run_case "qwen3_dense" "Qwen 3 dense" "BN_MODEL_QWEN3_DENSE" \
    "*qwen3-[0-9.]*b-q*.gguf" 5 16 --llama-flash-off
run_case "qwen3_moe" "Qwen 3 sparse MoE" "BN_MODEL_QWEN3_MOE" \
    "*qwen3-*a3b*.gguf" 1 5
run_case "qwen35_dense" "Qwen 3.5 dense" "BN_MODEL_QWEN35_DENSE" \
    "*qwen3*5*9b-q*.gguf" 5 16
run_case "qwen35_moe" "Qwen 3.5 sparse MoE" "BN_MODEL_QWEN35_MOE" \
    "*qwen3.5*35b*a3b*.gguf" 1 5
run_case "qwen36_dense" "Qwen 3.6 dense" "BN_MODEL_QWEN36_DENSE" \
    "*qwen3.6*27b*.gguf" 1 5
run_case "qwen36_moe" "Qwen 3.6 sparse MoE" "BN_MODEL_QWEN36_MOE" \
    "*qwen3.6*35b*a3b*.gguf" 1 5

if [[ "$REQUIRE_MODELS" == "1" && "$missing" -ne 0 ]]; then
    echo "Qwen CPU parity FAILED: $missing required model case(s) missing"
    exit 1
fi

if [[ "$fail" -ne 0 ]]; then
    echo "Qwen CPU parity FAILED"
    exit 1
fi

echo "Qwen CPU parity PASSED: ran=$ran skipped=$missing level=$LEVEL backends=$BACKENDS"
