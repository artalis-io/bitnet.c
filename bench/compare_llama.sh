#!/bin/bash
# compare_llama.sh — CPU generation benchmark against llama.cpp llama-bench.
#
# Usage:
#   ./bench/compare_llama.sh model.gguf [tokens] [threads] [iters] [trials]
#
# Environment:
#   LLAMA_BENCH=/path/to/llama-bench
#
# Output: TSV summary on stdout; full command output paths on stderr.

set -euo pipefail

MODEL="${1:-models/qwen2.5-3b-instruct-q4_0.gguf}"
TOKENS="${2:-32}"
THREADS="${3:-8}"
ITERS="${4:-5}"
TRIALS="${5:-${BENCH_TRIALS:-3}}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

if ! command -v "$LLAMA_BENCH" >/dev/null 2>&1; then
    echo "ERROR: llama-bench not found; set LLAMA_BENCH=/path/to/llama-bench" >&2
    exit 1
fi

case "$TRIALS" in
    ''|*[!0-9]*)
        echo "ERROR: trials must be a positive integer" >&2
        exit 1
        ;;
esac
if [ "$TRIALS" -le 0 ]; then
    echo "ERROR: trials must be a positive integer" >&2
    exit 1
fi

make -s bench_avx2

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

BITNET_VALS="$OUT_DIR/bitnet.tsv"
LLAMA_VALS="$OUT_DIR/llama.tsv"
: > "$BITNET_VALS"
: > "$LLAMA_VALS"

for trial in $(seq 1 "$TRIALS"); do
    BITNET_OUT="$OUT_DIR/bitnet.$trial.out"
    LLAMA_OUT="$OUT_DIR/llama.$trial.out"

    ./bench_avx2 "$MODEL" --iters "$ITERS" --threads "$THREADS" --toks "$TOKENS" \
        > "$BITNET_OUT" 2>&1
    "$LLAMA_BENCH" -m "$MODEL" -p 0 -n "$TOKENS" -t "$THREADS" -r 1 -o csv -ngl 0 -dev none \
        > "$LLAMA_OUT" 2>&1

    BITNET_TOKS="$(grep -oE 'Throughput: +[0-9.]+' "$BITNET_OUT" | awk '{print $2}' | tail -1)"
    LLAMA_TOKS="$(awk -F, 'END { v = $(NF - 1); gsub(/"/, "", v); print v }' "$LLAMA_OUT")"

    if [ -z "$BITNET_TOKS" ] || [ -z "$LLAMA_TOKS" ]; then
        echo "ERROR: failed to parse benchmark output for trial $trial" >&2
        echo "bitnet output: $BITNET_OUT" >&2
        echo "llama output:  $LLAMA_OUT" >&2
        exit 1
    fi

    printf "%s\t%s\n" "$trial" "$BITNET_TOKS" >> "$BITNET_VALS"
    printf "%s\t%s\n" "$trial" "$LLAMA_TOKS" >> "$LLAMA_VALS"
done

MODEL_NAME="$(basename "$MODEL")"
BITNET_SORTED="$OUT_DIR/bitnet.sorted"
LLAMA_SORTED="$OUT_DIR/llama.sorted"
cut -f2 "$BITNET_VALS" | sort -n > "$BITNET_SORTED"
cut -f2 "$LLAMA_VALS" | sort -n > "$LLAMA_SORTED"

echo "model	tokens	threads	trials	bitnet_median_tok_s	llama_median_tok_s	ratio_median	bitnet_mean_tok_s	llama_mean_tok_s	ratio_mean"
awk -v model="$MODEL_NAME" \
    -v tokens="$TOKENS" \
    -v threads="$THREADS" \
    -v trials="$TRIALS" \
    -v bitnet_vals="$BITNET_SORTED" \
    -v llama_vals="$LLAMA_SORTED" \
    'BEGIN {
        while ((getline line < bitnet_vals) > 0) {
            b[++nb] = line + 0;
            bsum += b[nb];
        }
        close(bitnet_vals);
        while ((getline line < llama_vals) > 0) {
            l[++nl] = line + 0;
            lsum += l[nl];
        }
        close(llama_vals);
        if (nb != trials || nl != trials) exit 1;
        mid = int((trials + 1) / 2);
        if (trials % 2) {
            bmed = b[mid];
            lmed = l[mid];
        } else {
            bmed = (b[mid] + b[mid + 1]) / 2.0;
            lmed = (l[mid] + l[mid + 1]) / 2.0;
        }
        bmean = bsum / trials;
        lmean = lsum / trials;
        rmed = (lmed > 0) ? bmed / lmed : 0;
        rmean = (lmean > 0) ? bmean / lmean : 0;
        printf "%s\t%s\t%s\t%s\t%.6f\t%.6f\t%.4f\t%.6f\t%.6f\t%.4f\n",
               model, tokens, threads, trials, bmed, lmed, rmed, bmean, lmean, rmean;
    }'

if [ "${KEEP_BENCH_OUTPUT:-0}" = "1" ]; then
    trap - EXIT
    echo "benchmark output captured in $OUT_DIR" >&2
fi
