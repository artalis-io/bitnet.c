#!/bin/bash
# Compare bitnet.c CUDA decode smoke throughput against llama.cpp CUDA tg16.
#
# This is a short local parity gate for the Qwen2.5 GGUFs commonly present in
# ./models. The workloads are not identical: bitnet.c uses bench_kernels'
# random next-token loop, while llama.cpp uses llama-bench tg16. Treat the ratio
# as a directional CUDA backend regression signal, not a formal benchmark.

set -uo pipefail

BITNET_BENCH="${BITNET_BENCH:-./bench_kernels}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/mark/artalis.io/tools/llama.cpp/build/bin/llama-bench}"
LLAMA_LIB_DIR="${LLAMA_LIB_DIR:-$(dirname "$LLAMA_BENCH")}"
THREADS="${THREADS:-8}"
LLAMA_TOKS="${LLAMA_TOKS:-16}"
TOKS="${TOKS:-$LLAMA_TOKS}"
PREFILL_TOKS="${PREFILL_TOKS:-16}"
ITERS="${ITERS:-10}"
CUDA_DEVICE="${BN_CUDA_DEVICE:-auto}"
MODELS="${MODELS:-models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/qwen2.5-0.5b-instruct-q8_0.gguf}"

if [ ! -x "$BITNET_BENCH" ]; then
    echo "ERROR: $BITNET_BENCH not found or not executable" >&2
    exit 1
fi

if [ ! -x "$LLAMA_BENCH" ]; then
    echo "ERROR: $LLAMA_BENCH not found or not executable" >&2
    exit 1
fi

echo -e "model\tbitnet_tok_s\tllama_tg16_tok_s\tratio\tstatus"

for model in $MODELS; do
    if [ ! -f "$model" ]; then
        echo -e "$model\tSKIP\tmodel not found\t0\tSKIP"
        continue
    fi

    bitnet_out=$(BN_CUDA_DEVICE="$CUDA_DEVICE" "$BITNET_BENCH" "$model" \
        --cuda --iters "$ITERS" --toks "$TOKS" --prefill-toks "$PREFILL_TOKS" \
        --prefill-iters 1 --threads "$THREADS" --random-gen 2>&1)
    bitnet_tps=$(printf '%s\n' "$bitnet_out" |
        awk '/Throughput:/ { v=$2 } END { if (v == "") v="0"; print v }')

    llama_out=$(LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA_BENCH" -m "$model" -n "$LLAMA_TOKS" -t "$THREADS" -ngl 99 2>&1)
    llama_tps=$(printf '%s\n' "$llama_out" |
        awk '$0 ~ /tg[0-9]+/ { for (i = 1; i <= NF; i++) if ($i == "±") v=$(i - 1) } END { if (v == "") v="0"; print v }')

    ratio=$(awk -v b="$bitnet_tps" -v l="$llama_tps" \
        'BEGIN { if (l > 0) printf "%.3f", b / l; else print "0" }')
    status=$(awk -v r="$ratio" \
        'BEGIN { if (r >= 1.0) print "PASS_PARITY"; else if (r >= 0.8) print "WARN_CLOSE"; else print "FAIL" }')

    echo -e "$(basename "$model")\t$bitnet_tps\t$llama_tps\t$ratio\t$status"
done
