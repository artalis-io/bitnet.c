#!/bin/bash
# Compare Gemma4 decode throughput against llama.cpp on local GGUF files.
#
# Defaults are intentionally short so this can run as a smoke benchmark on
# large 26B/31B files. Increase TOKS/REPS for release-quality numbers.

set -uo pipefail

ROOT="${GEMMA4_ROOT:-/data/models/gguf/gemma-4}"
DENSE_MODEL="${BN_MODEL_GEMMA4_DENSE:-$ROOT/31b/gemma-4-31B-it-Q4_K_M.gguf}"
MOE_MODEL="${BN_MODEL_GEMMA4_MOE:-$ROOT/26b-a4b-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
BITNET_BENCH="${BITNET_BENCH:-./bench_avx2}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
THREADS="${THREADS:-4}"
TOKS="${TOKS:-8}"
ITERS="${ITERS:-3}"
REPS="${REPS:-1}"
BITNET_ARGS="${BITNET_ARGS:-}"
LLAMA_ARGS="${LLAMA_ARGS:--ngl 0}"

run_case() {
    local name="$1"
    local model="$2"

    if [ ! -f "$model" ]; then
        echo -e "$name\tSKIP\tmodel not found: $model"
        return 0
    fi

    local b_out l_out b_tps l_tps ratio status
    b_out=$("$BITNET_BENCH" "$model" --iters "$ITERS" --threads "$THREADS" --toks "$TOKS" $BITNET_ARGS 2>&1)
    b_tps=$(printf '%s\n' "$b_out" | awk '/Throughput:/ { print $2; found=1 } END { if (!found) print "0" }')

    l_out=$("$LLAMA_BENCH" -m "$model" -p 0 -n "$TOKS" -t "$THREADS" -r "$REPS" -o csv $LLAMA_ARGS 2>&1)
    l_tps=$(printf '%s\n' "$l_out" | awk -F, 'NR > 1 && $1 ~ /^"/ { gsub(/"/, "", $(NF-1)); v=$(NF-1) } END { if (v == "") v="0"; print v }')

    ratio=$(awk -v b="$b_tps" -v l="$l_tps" 'BEGIN { if (l > 0) printf "%.3f", b / l; else printf "0" }')
    status=$(awk -v r="$ratio" 'BEGIN { if (r >= 1.20) print "PASS_20PCT"; else if (r >= 1.0) print "PASS_PARITY"; else print "FAIL" }')
    echo -e "$name\t$b_tps\t$l_tps\t$ratio\t$status"
}

if [ ! -x "$BITNET_BENCH" ]; then
    echo "ERROR: $BITNET_BENCH not found or not executable" >&2
    exit 1
fi
if ! command -v "$LLAMA_BENCH" >/dev/null 2>&1; then
    echo "ERROR: $LLAMA_BENCH not found" >&2
    exit 1
fi

echo -e "model\tbitnet_tok_s\tllama_tok_s\tratio\tstatus"
if [ "${RUN_DENSE:-1}" != "0" ]; then
    run_case "Gemma4 dense" "$DENSE_MODEL"
fi
if [ "${RUN_MOE:-1}" != "0" ]; then
    run_case "Gemma4 sparse MoE" "$MOE_MODEL"
fi
