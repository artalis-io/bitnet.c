#!/usr/bin/env bash
# Gemma4 CUDA correctness, token-parity, and throughput matrix.

set -uo pipefail

ROOT="${BN_MODEL_ROOT:-/data/models/gguf}"
COHERENCE="${COHERENCE:-./test_coherence}"
COMPARE_LLAMA="${COMPARE_LLAMA:-./test/compare_llama.sh}"
CUDA_COMPARE="${CUDA_COMPARE:-./bench/cuda_compare.sh}"
BITNET="${BITNET:-./bitnet}"
REQUIRE_MODELS="${REQUIRE_MODELS:-0}"
RUN_COHERENCE="${RUN_COHERENCE:-1}"
RUN_LLAMA_COMPARE="${RUN_LLAMA_COMPARE:-0}"
RUN_BENCH="${RUN_BENCH:-0}"
N_TOKENS="${N_TOKENS:-5}"
THREADS="${THREADS:-8}"
MAXSEQ="${MAXSEQ:-512}"
CASE_FILTERS=("$@")

PREFILL_MODES=${CUDA_PARITY_PREFILL_MODES:-default,tokenwise}
if [[ ! "$PREFILL_MODES" =~ ^(default|tokenwise)(,(default|tokenwise))*$ ]]; then
    echo "ERROR: CUDA_PARITY_PREFILL_MODES must contain default and/or tokenwise" >&2
    exit 1
fi

run_llama_case() {
    local model=$1
    shift
    local mode
    local mode_args=()
    for mode in ${PREFILL_MODES//,/ }; do
        mode_args=()
        [[ "$mode" != tokenwise ]] || mode_args+=(--no-prefill)
        echo "  prefill=$mode"
        "$COMPARE_LLAMA" "$model" --cuda --llama-cuda --strict \
            -n "$N_TOKENS" -t "$THREADS" --maxseq "$MAXSEQ" \
            "${mode_args[@]}" "$@" || fail=1
    done
}

fail=0
ran=0
missing=0
bench_models=""
gate_phases=$((RUN_COHERENCE + RUN_LLAMA_COMPARE + RUN_BENCH))

case_selected() {
    [ "${#CASE_FILTERS[@]}" -eq 0 ] && return 0
    local name=${1,,}
    local filter
    for filter in "${CASE_FILTERS[@]}"; do
        filter=${filter,,}
        [[ "$name" == *"$filter"* ]] && return 0
    done
    return 1
}

find_model() {
    local env_name=$1
    local rel_dir=$2
    shift 2
    local value
    value=$(eval "printf '%s' \"\${$env_name:-}\"")
    if [[ -n "$value" ]]; then
        printf '%s\n' "$value"
        return 0
    fi
    local search_root=$ROOT
    if [[ -d "$ROOT/$rel_dir" ]]; then
        search_root="$ROOT/$rel_dir"
    fi
    local pattern found
    for pattern in "$@"; do
        found=$(find "$search_root" -type f -iname "$pattern" \
            ! -iname '*mmproj*' | sort | head -n 1)
        if [[ -n "$found" ]]; then
            printf '%s\n' "$found"
            return 0
        fi
    done
    return 1
}

run_case() {
    local name=$1
    local env_name=$2
    local rel_dir=$3
    shift 3
    case_selected "$name" || return

    local path
    if ! path=$(find_model "$env_name" "$rel_dir" "$@"); then
        echo "SKIP $name: set $env_name or BN_MODEL_ROOT=$ROOT"
        missing=$((missing + 1))
        return
    fi

    echo "RUN $name: $path"
    ran=$((ran + 1))
    bench_models="${bench_models}${bench_models:+ }$path"
    if [[ "$RUN_COHERENCE" == 1 ]]; then
        "$COHERENCE" "$path" --cuda --require-all-tokens || fail=1
    fi
    if [[ "$RUN_LLAMA_COMPARE" == 1 ]]; then
        run_llama_case "$path"
    fi
}

run_case "Gemma4 dense" "BN_MODEL_GEMMA4_DENSE" "gemma4/31b" \
    "gemma-4-31B*.gguf" "gemma-4-31b*.gguf"
run_case "Gemma4 sparse MoE" "BN_MODEL_GEMMA4_MOE" "gemma4" \
    "gemma*4*a4b*00001-of-*.gguf" "gemma*4*26b*00001-of-*.gguf" \
    "gemma*4*a4b*.gguf" "gemma*4*26b*.gguf"

if [[ "$RUN_BENCH" == 1 && -n "$bench_models" ]]; then
    MODELS="$bench_models" "$CUDA_COMPARE" || fail=1
fi

if [[ "$REQUIRE_MODELS" == 1 && "$missing" -ne 0 ]]; then
    echo "Gemma4 CUDA matrix FAILED: $missing required model case(s) missing"
    exit 1
fi
if [[ "$fail" -ne 0 ]]; then
    echo "Gemma4 CUDA matrix FAILED"
    exit 1
fi

if [[ "$gate_phases" -eq 0 ]]; then
    echo "Gemma4 CUDA matrix DISCOVERY PASSED: found=$ran missing=$missing"
else
    echo "Gemma4 CUDA matrix PASSED: ran=$ran skipped=$missing"
fi
