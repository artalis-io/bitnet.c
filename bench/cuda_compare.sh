#!/bin/bash
# Compare bitnet.c CUDA prompt/decode throughput against llama.cpp CUDA.
#
# This is a short parity gate for Qwen GGUFs. The workloads are not identical:
# bitnet.c prompt processing is measured with bench_kernels. Decode defaults to
# the real bitnet CLI generation path so logits/readback are included like
# llama-bench tg. Set BITNET_TG_MODE=bench to use bench_kernels' random
# next-token loop instead. Treat the ratio as a directional CUDA backend
# regression signal, not a formal benchmark.

set -uo pipefail

BITNET_BENCH="${BITNET_BENCH:-./bench_kernels}"
BITNET_CLI="${BITNET_CLI:-./bitnet}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/mark/artalis.io/tools/llama.cpp/build/bin/llama-bench}"
LLAMA_LIB_DIR="${LLAMA_LIB_DIR:-$(dirname "$LLAMA_BENCH")}"
THREADS="${THREADS:-8}"
LLAMA_TOKS="${LLAMA_TOKS:-16}"
TOKS="${TOKS:-$LLAMA_TOKS}"
PREFILL_TOKS="${PREFILL_TOKS:-16}"
ITERS="${ITERS:-10}"
CUDA_DEVICE="${BN_CUDA_DEVICE:-auto}"
MODEL_ROOT="${BN_MODEL_ROOT:-${MODEL_ROOT:-/data/models/gguf}}"
MAXSEQ="${MAXSEQ:-512}"
BITNET_TG_MODE="${BITNET_TG_MODE:-generate}"

LLAMA_CUDA_ENV=()
if [ "$CUDA_DEVICE" != "auto" ] && [ -n "$CUDA_DEVICE" ]; then
    LLAMA_CUDA_ENV=(CUDA_VISIBLE_DEVICES="$CUDA_DEVICE")
fi

find_first_model() {
    pattern=$1
    if [ -n "$MODEL_ROOT" ] && [ -d "$MODEL_ROOT" ]; then
        find "$MODEL_ROOT" -type f -iname "$pattern" | sort | head -n 1
    fi
}

if [ -z "${MODELS:-}" ]; then
    MODELS=""
    for pattern in \
        "Qwen2.5*Q4_K_M.gguf" \
        "Qwen2.5*MOE*Q4*k*m.gguf" \
        "Qwen3-4B*Q4_K_M.gguf" \
        "Qwen3-30B-A3B-Q4_K_M.gguf" \
        "Qwen3.5-27B*Q5_K_M.gguf" \
        "Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf" \
        "Qwen3.6-27B-Q4_K_M.gguf" \
        "qwen3.6*35b*a3b*Q8_0.gguf" \
        "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
        "qwen2.5-0.5b-instruct-q8_0.gguf"; do
        found=$(find_first_model "$pattern")
        if [ -n "$found" ]; then
            MODELS="${MODELS}${MODELS:+ }$found"
        fi
    done
fi

if [ -z "$MODELS" ]; then
    echo "ERROR: no models found; set MODELS or BN_MODEL_ROOT" >&2
    exit 1
fi

if [ ! -x "$BITNET_BENCH" ]; then
    echo "ERROR: $BITNET_BENCH not found or not executable" >&2
    exit 1
fi

if [ "$BITNET_TG_MODE" = "generate" ] && [ ! -x "$BITNET_CLI" ]; then
    echo "ERROR: $BITNET_CLI not found or not executable" >&2
    exit 1
fi

if [ ! -x "$LLAMA_BENCH" ]; then
    echo "ERROR: $LLAMA_BENCH not found or not executable" >&2
    exit 1
fi

echo -e "model\tbitnet_pp_tok_s\tllama_pp_tok_s\tpp_ratio\tbitnet_tg_tok_s\tllama_tg_tok_s\ttg_ratio\tstatus"

for model in $MODELS; do
    if [ ! -f "$model" ]; then
        echo -e "$model\tSKIP\tmodel not found\t0\tSKIP"
        continue
    fi

    if ! bitnet_out=$(BN_CUDA_DEVICE="$CUDA_DEVICE" "$BITNET_BENCH" "$model" \
        --cuda --iters "$ITERS" --toks "$TOKS" --prefill-toks "$PREFILL_TOKS" \
        --prefill-iters 1 --threads "$THREADS" --random-gen 2>&1); then
        echo -e "$(basename "$model")\tERROR\tbitnet bench failed\t0\tFAIL"
        printf '%s\n' "$bitnet_out" >&2
        continue
    fi
    bitnet_pp=$(printf '%s\n' "$bitnet_out" |
        awk '/^Prefill:/ { v=$2 } END { if (v == "") v="0"; print v }')

    if [ "$BITNET_TG_MODE" = "generate" ]; then
        if ! bitnet_tg_out=$(BN_CUDA_DEVICE="$CUDA_DEVICE" "$BITNET_CLI" \
            "$model" --cuda -n "$TOKS" --maxseq "$MAXSEQ" --quiet 2>&1); then
            echo -e "$(basename "$model")\t$bitnet_pp\tSKIP\t0\tERROR\tbitnet generate failed\t0\tFAIL"
            printf '%s\n' "$bitnet_tg_out" >&2
            continue
        fi
        bitnet_tps=$(printf '%s\n' "$bitnet_tg_out" |
            sed -n 's/.*tok\/s=\([0-9.][0-9.]*\).*/\1/p' |
            tail -n 1)
        if [ -z "$bitnet_tps" ]; then bitnet_tps="0"; fi
        bitnet_generated=$(printf '%s\n' "$bitnet_tg_out" |
            sed -n 's/.*Generation complete | tokens=\(-\{0,1\}[0-9][0-9]*\).*/\1/p' |
            tail -n 1)
        if [ -z "$bitnet_generated" ] || [ "$bitnet_generated" -le 0 ] || [ "$bitnet_tps" = "0" ]; then
            echo -e "$(basename "$model")\t$bitnet_pp\tSKIP\t0\tERROR\tbitnet generate invalid\t0\tFAIL"
            printf '%s\n' "$bitnet_tg_out" >&2
            continue
        fi
    else
        bitnet_tps=$(printf '%s\n' "$bitnet_out" |
            awk '/Throughput:/ { v=$2 } END { if (v == "") v="0"; print v }')
    fi

    if ! llama_out=$(env "${LLAMA_CUDA_ENV[@]}" \
        LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA_BENCH" -m "$model" -p "$PREFILL_TOKS" -n "$LLAMA_TOKS" -t "$THREADS" -ngl 99 2>&1); then
        echo -e "$(basename "$model")\t$bitnet_pp\tllama-bench failed\t0\t$bitnet_tps\tllama-bench failed\t0\tFAIL"
        printf '%s\n' "$llama_out" >&2
        continue
    fi
    if printf '%s\n' "$llama_out" | grep -qiE 'cuda init failed|failed to initialize CUDA|no CUDA-capable device'; then
        echo -e "$(basename "$model")\t$bitnet_pp\tllama CUDA unavailable\t0\t$bitnet_tps\tllama CUDA unavailable\t0\tFAIL"
        printf '%s\n' "$llama_out" >&2
        continue
    fi
    llama_tps=$(printf '%s\n' "$llama_out" |
        awk '$0 ~ /tg[0-9]+/ { for (i = 1; i <= NF; i++) if ($i == "±") v=$(i - 1) } END { if (v == "") v="0"; print v }')
    llama_pp=$(printf '%s\n' "$llama_out" |
        awk '$0 ~ /pp[0-9]+/ { for (i = 1; i <= NF; i++) if ($i == "±") v=$(i - 1) } END { if (v == "") v="0"; print v }')

    tg_ratio=$(awk -v b="$bitnet_tps" -v l="$llama_tps" \
        'BEGIN { if (l > 0) printf "%.3f", b / l; else print "0" }')
    pp_ratio=$(awk -v b="$bitnet_pp" -v l="$llama_pp" \
        'BEGIN { if (l > 0) printf "%.3f", b / l; else print "0" }')
    status=$(awk -v p="$pp_ratio" -v t="$tg_ratio" \
        'BEGIN {
            if (p >= 1.0 && t >= 1.0) print "PASS_PARITY";
            else if (p >= 0.8 && t >= 0.8) print "WARN_CLOSE";
            else print "FAIL";
        }')

    echo -e "$(basename "$model")\t$bitnet_pp\t$llama_pp\t$pp_ratio\t$bitnet_tps\t$llama_tps\t$tg_ratio\t$status"
done
