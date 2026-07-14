#!/usr/bin/env bash
set -euo pipefail

ROOT=${BN_MODEL_ROOT:-models}
LEVEL=${GEMMA4_CPU_PARITY_LEVEL:-standard}
REQUIRE_MODELS=${REQUIRE_MODELS:-0}
COMPARE=${COMPARE_LLAMA:-./test/compare_llama.sh}
NEON_BIN=${BITNET_NEON:-./bitnet}
SCALAR_BIN=${BITNET_SCALAR:-./bitnet_scalar}
CASES=${GEMMA4_CPU_PARITY_CASES:-all}
MAXSEQ=${GEMMA4_CPU_PARITY_MAXSEQ:-512}

fail=0
ran=0
missing=0

find_model() {
    local env_name=$1
    shift
    local value
    value=$(eval "printf '%s' \"\${$env_name:-}\"")
    if [[ -n "$value" ]]; then
        printf '%s\n' "$value"
        return 0
    fi
    if [[ -n "$ROOT" && -d "$ROOT" ]]; then
        local pattern
        local found
        for pattern in "$@"; do
            found=$(find "$ROOT" -type f -iname "$pattern" ! -iname "*mmproj*" 2>/dev/null | head -n 1 || true)
            if [[ -n "$found" ]]; then
                printf '%s\n' "$found"
                return 0
            fi
        done
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

    echo "RUN $backend $name: $model (-n $tokens --maxseq $MAXSEQ)"
    ran=$((ran + 1))
    if ! BITNET="$bitnet" "$COMPARE" "$model" -n "$tokens" --strict -t 1 --maxseq "$MAXSEQ" "$@"; then
        fail=1
    fi
}

run_case() {
    local case_id=$1
    local name=$2
    local env_name=$3
    local standard_tokens=$4
    local full_tokens=$5
    shift 5

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
    if ! model=$(find_model "$env_name" "$@"); then
        echo "SKIP $name: set $env_name or BN_MODEL_ROOT"
        missing=$((missing + 1))
        return
    fi

    run_backend "NEON" "$NEON_BIN" "$name" "$model" "$tokens"
    run_backend "scalar" "$SCALAR_BIN" "$name" "$model" "$tokens"
}

case "$LEVEL" in
    standard|full) ;;
    *)
        echo "ERROR: GEMMA4_CPU_PARITY_LEVEL must be standard or full" >&2
        exit 1
        ;;
esac

run_case "gemma4_dense" "Gemma4 dense" "BN_MODEL_GEMMA4_DENSE" 1 5 \
    "*gemma*4*e*b*q*.gguf" \
    "*gemma*4*31b*q*.gguf"
run_case "gemma4_moe" "Gemma4 sparse MoE" "BN_MODEL_GEMMA4_MOE" 1 5 \
    "*gemma*4*26b*q*.gguf" \
    "*gemma*4*a4b*q*.gguf" \
    "*gemma*4*a4b*mxfp4*.gguf"

if [[ "$REQUIRE_MODELS" == "1" && "$missing" -ne 0 ]]; then
    echo "Gemma4 CPU parity FAILED: $missing required model case(s) missing"
    exit 1
fi

if [[ "$fail" -ne 0 ]]; then
    echo "Gemma4 CPU parity FAILED"
    exit 1
fi

echo "Gemma4 CPU parity PASSED: ran=$ran skipped=$missing level=$LEVEL"
