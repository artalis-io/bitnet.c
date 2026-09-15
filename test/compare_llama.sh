#!/usr/bin/env bash
# Compare bitnet.c output against llama.cpp for numerical equivalence.
#
# Usage:
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf -n 50
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --prompt "The sum of 2 + 2 ="
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --metal
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --cuda --llama-cuda
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --metal --llama-metal --flash
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --metal --llama-gpu-layers 12
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --metal --llama-metal --llama-flash-off
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --llama-cache-k f32 --llama-cache-v f32
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --llama-batch 1 --llama-ubatch 1
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf -v    # verbose
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --strict
#
# Requires: llama-completion (brew install llama.cpp)
# Strict mode compares sampled token IDs using LLAMA_TOKEN_TRACE.

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [-n tokens] [-v]}"
shift
N_TOKENS=30
VERBOSE=0
STRICT=0
TOP_LOGITS=0
# Keep both runtimes on the same greedy sampling policy.  bitnet's CLI default
# applies a 1.1 repetition penalty, while the llama.cpp invocation below uses
# --repeat-penalty 1.
BITNET_ARGS=(--repeat-penalty 1)
BITNET_BACKEND="cpu"
LLAMA_BACKEND_ARGS=(-ngl 0 -dev none)
LLAMA_ARGS=()
LLAMA_CACHE_ARGS=(-ctk f32 -ctv f32)
LLAMA_GPU_LAYERS=0
LLAMA_FLASH=(-fa off)
LLAMA_THREADS=1
LLAMA_CTX=0
LLAMA_VERBOSITY=1
LLAMA_THREADS_BATCH=()
CUSTOM_PROMPTS=()
LLAMA_BIN_DIR_EXPLICIT=${LLAMA_BIN_DIR+x}
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/mark/artalis.io/tools/llama.cpp/build/bin}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) N_TOKENS="$2"; shift 2 ;;
        --prompt) CUSTOM_PROMPTS+=("$2"); shift 2 ;;
        --metal) BITNET_ARGS+=(--metal); BITNET_BACKEND="metal"; shift ;;
        --llama-metal) LLAMA_BACKEND_ARGS=(-ngl 99); LLAMA_GPU_LAYERS=99; LLAMA_VERBOSITY=4; shift ;;
        --cuda) BITNET_ARGS+=(--cuda); BITNET_BACKEND="cuda"; shift ;;
        --kv16) BITNET_ARGS+=(--kv16); LLAMA_CACHE_ARGS+=(-ctk f16 -ctv f16); shift ;;
        --llama-cuda) LLAMA_BACKEND_ARGS=(-ngl 99); LLAMA_GPU_LAYERS=99; LLAMA_VERBOSITY=4; shift ;;
        --llama-gpu-layers) LLAMA_BACKEND_ARGS=(-ngl "$2"); LLAMA_GPU_LAYERS="$2"; LLAMA_VERBOSITY=4; shift 2 ;;
        --llama-cache-k) LLAMA_CACHE_ARGS+=(-ctk "$2"); shift 2 ;;
        --llama-cache-v) LLAMA_CACHE_ARGS+=(-ctv "$2"); shift 2 ;;
        --llama-batch) LLAMA_ARGS+=(-b "$2"); shift 2 ;;
        --llama-ubatch) LLAMA_ARGS+=(-ub "$2"); shift 2 ;;
        --llama-threads-batch) LLAMA_THREADS_BATCH=(-tb "$2"); shift 2 ;;
        --llama-no-repack) LLAMA_ARGS+=(--no-repack); shift ;;
        --webgpu|--gpu) BITNET_ARGS+=(--webgpu); BITNET_BACKEND="webgpu"; shift ;;
        --no-prefill) BITNET_ARGS+=(--no-prefill); LLAMA_ARGS+=(-b 1 -ub 1); shift ;;
        --pread) BITNET_ARGS+=(--pread); shift ;;
        --cache-mb) BITNET_ARGS+=(--cache-mb "$2"); shift 2 ;;
        --madvise) BITNET_ARGS+=(--madvise); shift ;;
        --flash) BITNET_ARGS+=(--flash); LLAMA_FLASH=(-fa on); shift ;;
        --llama-flash-off) LLAMA_FLASH=(-fa off); shift ;;
        --metal-disable-small-dense-native-quant|--metal-disable-small-dense-exact-native|--metal-disable-q4-q8) BITNET_ARGS+=(--metal-disable-small-dense-native-quant); shift ;;
        --metal-specialized-native-quant) BITNET_ARGS+=(--metal-specialized-native-quant); shift ;;
        --metal-enable-q6-q8k) BITNET_ARGS+=(--metal-specialized-native-quant); shift ;;
        --metal-disable-specialized-native-quant|--metal-disable-q6-q8k) BITNET_ARGS+=(--metal-disable-specialized-native-quant); shift ;;
        --gpu-cpu-fallback-layer) BITNET_ARGS+=(--gpu-cpu-fallback-layer "$2"); shift 2 ;;
        --gpu-cpu-fallback-from-layer) BITNET_ARGS+=(--gpu-cpu-fallback-from-layer "$2"); shift 2 ;;
        --gpu-max-storage-binding-mb) BITNET_ARGS+=(--gpu-max-storage-binding-mb "$2"); shift 2 ;;
        --small-dense-native-quant-to-layer|--small-dense-exact-native-to-layer|--q4-q8-to-layer) BITNET_ARGS+=(--small-dense-native-quant-to-layer "$2"); shift 2 ;;
        --small-dense-native-quant-tail|--small-dense-exact-native-tail|--q4-q8-tail-native) BITNET_ARGS+=(--small-dense-native-quant-tail "$2"); shift 2 ;;
        --small-dense-native-quant-attn-only|--small-dense-exact-native-attn-only|--q4-q8-attn-only) BITNET_ARGS+=(--small-dense-native-quant-attn-only); shift ;;
        --small-dense-native-quant-ffn-only|--small-dense-exact-native-ffn-only|--q4-q8-ffn-only) BITNET_ARGS+=(--small-dense-native-quant-ffn-only); shift ;;
        --gpu-flash-min-kv) BITNET_ARGS+=(--gpu-flash-min-kv "$2"); shift 2 ;;
        --metal-private-weights) BITNET_ARGS+=(--metal-private-weights); shift ;;
        --metal-cpu-route-resident-moe) BITNET_ARGS+=(--metal-cpu-route-resident-moe); shift ;;
        --maxseq) BITNET_ARGS+=(--maxseq "$2"); LLAMA_ARGS+=(-c "$2"); LLAMA_CTX="$2"; shift 2 ;;
        -t) BITNET_ARGS+=(-t "$2"); LLAMA_THREADS="$2"; shift 2 ;;
        -v) VERBOSE=1; shift ;;
        --strict) STRICT=1; shift ;;
        --top-logits) TOP_LOGITS="$2"; shift 2 ;;
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Check dependencies
BITNET="${BITNET:-./bitnet}"
LLAMA_ARGS=("${LLAMA_BACKEND_ARGS[@]}" "${LLAMA_ARGS[@]}")
LLAMA="${LLAMA:-llama-completion}"
LLAMA_TOKEN_TRACE="${LLAMA_TOKEN_TRACE:-./test/libllama_token_trace.so}"
if [[ "$LLAMA" == "llama-completion" ]]; then
    if [[ -x "$LLAMA_BIN_DIR/llama-completion" ]]; then
        LLAMA="$LLAMA_BIN_DIR/llama-completion"
    elif [[ -n "$LLAMA_BIN_DIR_EXPLICIT" ]]; then
        echo "ERROR: $LLAMA_BIN_DIR/llama-completion not found" >&2
        exit 1
    fi
fi
LLAMA_LIB_DIR="${LLAMA_LIB_DIR:-$(dirname "$LLAMA")}"
if [[ ! -x "$BITNET" ]]; then
    echo "ERROR: $BITNET not found. Run 'make' first." >&2; exit 1
fi
if [[ ! -x "$LLAMA" ]] && ! command -v "$LLAMA" &>/dev/null; then
    echo "ERROR: $LLAMA not found. Run 'brew install llama.cpp'." >&2; exit 1
fi
if (( STRICT )) && [[ ! -f "$LLAMA_TOKEN_TRACE" ]]; then
    echo "ERROR: $LLAMA_TOKEN_TRACE not found. Run 'make test_llama_layer_probe' or set LLAMA_TOKEN_TRACE." >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL" >&2; exit 1
fi

validate_bitnet_backend() {
    local stderr_file="$1"
    local marker=""
    case "$BITNET_BACKEND" in
        cpu) return 0 ;;
        cuda) marker="CUDA weights uploaded" ;;
        metal) marker="[bn:gpu:metal] device:" ;;
        webgpu) marker="WebGPU weights uploaded" ;;
    esac
    if grep -Fq "falling back to CPU" "$stderr_file" ||
       grep -Fq "[gpu:fallback]" "$stderr_file" ||
       ! grep -Fq "$marker" "$stderr_file"; then
        echo "ERROR: requested $BITNET_BACKEND backend was not activated" >&2
        sed 's/^/  /' "$stderr_file" | head -40 >&2
        return 1
    fi
}

validate_llama_backend() {
    local stderr_file="$1"
    if (( LLAMA_GPU_LAYERS > 0 )) &&
       ! grep -Eq 'offloaded [1-9][0-9]*/[1-9][0-9]* layers to GPU' "$stderr_file"; then
        echo "ERROR: llama.cpp did not activate the requested GPU offload" >&2
        sed 's/^/  /' "$stderr_file" | head -40 >&2
        return 1
    fi
}

# Prompts: factual completions with strong first-token predictions
PROMPTS=(
    "The capital of France is"
    "In the year 2020, the world"
    "The quick brown fox jumps over the lazy"
    "Once upon a time, there was a"
    "The sum of 2 + 2 ="
    "HTTP status code 404 means"
    "The color of the sky is"
    "Python is a programming language created by"
)
if (( ${#CUSTOM_PROMPTS[@]} > 0 )); then
    PROMPTS=("${CUSTOM_PROMPTS[@]}")
fi

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

total_prompts=${#PROMPTS[@]}
total_words_matched=0
total_words_compared=0
first_word_matches=0
exact_first_output_word_matches=0
first_token_matches=0
total_token_ids_matched=0
total_token_ids_compared=0
total_bitnet_token_ids=0
total_llama_token_ids=0
token_count_mismatches=0
tmp_files=()
cleanup() {
    if (( ${#tmp_files[@]} > 0 )); then
        rm -f "${tmp_files[@]}"
    fi
}
trap cleanup EXIT

echo -e "${BOLD}Output comparison: bitnet.c vs llama.cpp${RESET}"
echo "Model:  $MODEL"
echo "Tokens: $N_TOKENS per prompt"
if (( ${#BITNET_ARGS[@]} > 0 )); then
    echo "bitnet args: ${BITNET_ARGS[*]}"
fi
if (( ${#LLAMA_FLASH[@]} > 0 )); then
    echo "llama args:  ${LLAMA_ARGS[*]} ${LLAMA_CACHE_ARGS[*]} ${LLAMA_FLASH[*]} -t $LLAMA_THREADS ${LLAMA_THREADS_BATCH[*]-}"
else
    echo "llama args:  ${LLAMA_ARGS[*]} ${LLAMA_CACHE_ARGS[*]} -t $LLAMA_THREADS ${LLAMA_THREADS_BATCH[*]-}"
fi
echo "---"

for prompt in "${PROMPTS[@]}"; do
    # Run bitnet.c (raw completion, temp=0, no repeat penalty)
    bitnet_stderr="/dev/null"
    bitnet_run_args=()
    if (( ${#BITNET_ARGS[@]} > 0 )); then
        bitnet_run_args=("${BITNET_ARGS[@]}")
    fi
    if (( STRICT )) || [[ "$BITNET_BACKEND" != "cpu" ]]; then
        bitnet_stderr=$(mktemp)
        tmp_files+=("$bitnet_stderr")
    fi
    if (( STRICT )); then
        bitnet_run_args+=(--token-ids)
    fi
    if (( TOP_LOGITS > 0 )); then
        bitnet_run_args+=(--top-logits "$TOP_LOGITS")
    fi
    # llama.cpp has no repeated-ngram abort. Disable bitnet's interactive
    # loop guard so both sides run to EOG or the requested token count.
    if bitnet_out=$(BN_GPU_DEBUG_FALLBACK=1 BN_DISABLE_LOOP_ABORT=1 \
        "$BITNET" "$MODEL" \
        ${bitnet_run_args[@]+"${bitnet_run_args[@]}"} \
        -p "$prompt" -n "$N_TOKENS" \
        --temp 0 --repeat-penalty 1 2>"$bitnet_stderr"); then
        :
    else
        status=$?
        echo "ERROR: bitnet inference failed (exit $status) for prompt: $prompt" >&2
        sed -n '1,40p' "$bitnet_stderr" >&2
        exit 1
    fi
    validate_bitnet_backend "$bitnet_stderr" || exit 1

    # Run llama.cpp (raw completion, no chat template, temp=0)
    llama_stderr=$(mktemp)
    tmp_files+=("$llama_stderr")
    llama_run_args=("${LLAMA_ARGS[@]}" "${LLAMA_CACHE_ARGS[@]}")
    if (( ${#LLAMA_FLASH[@]} > 0 )); then
        llama_run_args+=("${LLAMA_FLASH[@]}")
    fi
    if llama_out=$(LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        LLAMA_TOKEN_TRACE_TOP="$TOP_LOGITS" \
        LD_PRELOAD="$([[ $STRICT -eq 1 ]] && printf '%s' "$LLAMA_TOKEN_TRACE")${LD_PRELOAD:+:$LD_PRELOAD}" \
        "$LLAMA" -m "$MODEL" "${llama_run_args[@]}" -p "$prompt" -n "$N_TOKENS" \
        --temp 0 --repeat-penalty 1 --reasoning-format none \
        --no-display-prompt -no-cnv \
        --simple-io --verbosity "$LLAMA_VERBOSITY" \
        -t "$LLAMA_THREADS" ${LLAMA_THREADS_BATCH[@]+"${LLAMA_THREADS_BATCH[@]}"} \
        2>"$llama_stderr" | sed 's/> EOF by user$//'); then
        :
    else
        status=$?
        echo "ERROR: llama.cpp inference failed (exit $status) for prompt: $prompt" >&2
        sed -n '1,40p' "$llama_stderr" >&2
        exit 1
    fi
    validate_llama_backend "$llama_stderr" || exit 1

    # Strict parity is defined by sampled IDs, including immediate stop tokens
    # that render no text. Word-only comparisons still require completions.
    if (( ! STRICT )) && [[ -z "$bitnet_out" || -z "$llama_out" ]]; then
        echo -e "${RED}ERROR${RESET}   \"$prompt\""
        [[ -n "$bitnet_out" ]] || echo "  bitnet produced no completion"
        [[ -n "$llama_out" ]] || echo "  llama.cpp produced no completion"
        if [[ -s "$llama_stderr" ]]; then
            echo "  llama.cpp stderr:"
            sed 's/^/    /' "$llama_stderr" | head -20
        fi
        exit 1
    fi

    # Collapse whitespace for word-level comparison
    read -ra bwords <<< "${bitnet_out//$'\n'/ }" || bwords=()
    read -ra lwords <<< "${llama_out//$'\n'/ }"  || lwords=()

    # Find longest matching word prefix
    max_cmp=${#bwords[@]}
    if (( ${#lwords[@]} < max_cmp )); then max_cmp=${#lwords[@]}; fi

    match=0
    for (( i=0; i<max_cmp; i++ )); do
        if [[ "${bwords[$i]}" == "${lwords[$i]}" ]]; then
            (( match++ )) || true
        else
            break
        fi
    done

    total_words_compared=$((total_words_compared + max_cmp))
    total_words_matched=$((total_words_matched + match))

    # Check first-word match (most important signal for kernel correctness)
    # Strip trailing punctuation for comparison (e.g. "blue." vs "blue,")
    if (( ${#bwords[@]} > 0 && ${#lwords[@]} > 0 )); then
        if [[ "${bwords[0]}" == "${lwords[0]}" ]]; then
            (( exact_first_output_word_matches++ )) || true
        fi
        bw0="${bwords[0]%%[.,;:!?]}"
        lw0="${lwords[0]%%[.,;:!?]}"
        if [[ "$bw0" == "$lw0" ]]; then
            (( first_word_matches++ )) || true
        fi
    fi

    bitnet_first_token=""
    llama_first_token=""
    token_id_match=0
    token_id_cmp=0
    if (( STRICT )); then
        bitnet_first_token=$(sed -n 's/^token_id=//p' "$bitnet_stderr" | head -n 1) || bitnet_first_token=""
        bitnet_ids_csv=$(sed -n 's/^token_id=//p' "$bitnet_stderr" | paste -sd, -) || bitnet_ids_csv=""
        llama_ids_csv=$(sed -n 's/^llama_token_id=//p' "$llama_stderr" |
            paste -sd, -) || llama_ids_csv=""
        IFS=, read -ra bitnet_ids <<< "$bitnet_ids_csv"
        IFS=, read -ra llama_ids <<< "$llama_ids_csv"
        if (( ${#bitnet_ids[@]} == 0 || ${#llama_ids[@]} == 0 )); then
            echo "ERROR: missing sampled token IDs for prompt: $prompt" >&2
            echo "  token ID counts: bitnet=${#bitnet_ids[@]} llama=${#llama_ids[@]}" >&2
            exit 1
        fi
        llama_first_token="${llama_ids[0]:-}"
        if [[ -n "$bitnet_first_token" && "$bitnet_first_token" == "$llama_first_token" ]]; then
            (( first_token_matches++ )) || true
        fi
        total_bitnet_token_ids=$((total_bitnet_token_ids + ${#bitnet_ids[@]}))
        total_llama_token_ids=$((total_llama_token_ids + ${#llama_ids[@]}))
        if (( ${#bitnet_ids[@]} != ${#llama_ids[@]} )); then
            token_count_mismatches=$((token_count_mismatches + 1))
        fi
        token_id_cmp=${#bitnet_ids[@]}
        if (( ${#llama_ids[@]} < token_id_cmp )); then token_id_cmp=${#llama_ids[@]}; fi
        for (( i=0; i<token_id_cmp; i++ )); do
            if [[ "${bitnet_ids[$i]}" == "${llama_ids[$i]}" ]]; then
                (( token_id_match++ )) || true
            else
                break
            fi
        done
        total_token_ids_matched=$((total_token_ids_matched + token_id_match))
        total_token_ids_compared=$((total_token_ids_compared + token_id_cmp))
    fi

    # Report
    prompt_short="${prompt:0:45}"
    if (( STRICT )) && (( ${#bitnet_ids[@]} != ${#llama_ids[@]} )); then
        echo -e "${YELLOW}PARTIAL${RESET} ${DIM}[unequal token counts]${RESET}  \"$prompt_short\""
    elif (( match == max_cmp && max_cmp > 0 )); then
        echo -e "${GREEN}MATCH${RESET}   ${DIM}[$match/$max_cmp words]${RESET}  \"$prompt_short\""
    elif (( STRICT && max_cmp == 0 && token_id_cmp > 0 && token_id_match == token_id_cmp )); then
        echo -e "${GREEN}MATCH${RESET}   ${DIM}[$token_id_match/$token_id_cmp token IDs]${RESET}  \"$prompt_short\""
    elif (( match >= 1 )); then
        echo -e "${YELLOW}PARTIAL${RESET} ${DIM}[$match/$max_cmp words]${RESET}  \"$prompt_short\""
        # Show context around divergence point
        ctx_end=$((match + 3))
        echo -e "  ${DIM}agree:${RESET}  ${bwords[*]:0:$match}"
        if (( match < ${#bwords[@]} )); then
            echo -e "  ${RED}bitnet:${RESET} ...${bwords[*]:$match:3}"
        fi
        if (( match < ${#lwords[@]} )); then
            echo -e "  ${RED}llama:${RESET}  ...${lwords[*]:$match:3}"
        fi
    else
        echo -e "${RED}DIVERGE${RESET} ${DIM}[$match/$max_cmp words]${RESET}  \"$prompt_short\""
        echo -e "  bitnet: ${bwords[*]:0:5}"
        echo -e "  llama:  ${lwords[*]:0:5}"
    fi

    if (( STRICT )) && (( ${#bitnet_ids[@]} != ${#llama_ids[@]} )); then
        echo "  token ID counts: bitnet=${#bitnet_ids[@]} llama=${#llama_ids[@]}"
    fi

    if (( VERBOSE )); then
        echo -e "  ${DIM}[full bitnet] $bitnet_out${RESET}"
        echo -e "  ${DIM}[full llama]  $llama_out${RESET}"
        if (( STRICT )); then
            echo -e "  ${DIM}[first token IDs] bitnet=$bitnet_first_token llama=$llama_first_token${RESET}"
            echo -e "  ${DIM}[token ID prefix] $token_id_match/$token_id_cmp${RESET}"
        fi
    elif (( STRICT )) && [[ "$bitnet_first_token" != "$llama_first_token" ]]; then
        echo -e "  ${DIM}first token IDs:${RESET} bitnet=$bitnet_first_token llama=$llama_first_token"
    elif (( STRICT )) && (( token_id_match < token_id_cmp )); then
        echo -e "  ${DIM}token ID prefix:${RESET} $token_id_match/$token_id_cmp"
    fi
    if (( STRICT && TOP_LOGITS > 0 && token_id_match < token_id_cmp )); then
        echo -e "  ${DIM}top logits at mismatch step $token_id_match:${RESET}"
        sed -n "/^top_logit step=$token_id_match /s/^/    bitnet /p" "$bitnet_stderr"
        sed -n "/^llama_top_logit step=$token_id_match /s/^/    llama  /p" "$llama_stderr"
    fi
done

echo "---"
if (( STRICT )); then
    echo "First output-token ID matches: $first_token_matches / $total_prompts prompts"
    echo "Generated token-ID prefix matches: $total_token_ids_matched / $total_token_ids_compared tokens"
    echo "Token-count mismatches: $token_count_mismatches / $total_prompts prompts (bitnet=$total_bitnet_token_ids llama=$total_llama_token_ids)"
fi
if (( STRICT )) && [[ -f "$LLAMA_TOKEN_TRACE" ]]; then
    echo "strict token source: llama-completion sampled IDs ($LLAMA_TOKEN_TRACE)"
elif (( STRICT )); then
    echo "strict token source: retokenized requested-backend llama.cpp text" >&2
fi
echo "Exact first output-word matches: $exact_first_output_word_matches / $total_prompts prompts"
echo "Punctuation-normalized first-word matches: $first_word_matches / $total_prompts prompts"
echo "Word prefix matches: $total_words_matched / $total_words_compared total words"
echo ""

# Strict mode uses IDs sampled by the same llama-completion process.
# Display text can include frontend formatting that was not sampled by the
# model, and retokenized output is not necessarily invertible.
sampled_ids_required=0
if (( STRICT )) && [[ -f "$LLAMA_TOKEN_TRACE" ]]; then
    sampled_ids_required=1
fi
if (( STRICT && sampled_ids_required &&
      token_count_mismatches == 0 &&
      total_bitnet_token_ids == total_llama_token_ids &&
      total_token_ids_compared == total_bitnet_token_ids &&
      total_token_ids_matched == total_bitnet_token_ids &&
      total_bitnet_token_ids > 0 )); then
    echo -e "${GREEN}${BOLD}PASS${RESET} — sampled token ID parity with llama.cpp"
    exit 0
elif (( STRICT && !sampled_ids_required &&
        total_token_ids_matched == total_token_ids_compared &&
        total_token_ids_compared > 0 &&
        total_words_matched == total_words_compared &&
        total_words_compared > 0 )); then
    echo -e "${GREEN}${BOLD}PASS${RESET} — requested-backend token and text parity with llama.cpp"
    exit 0
elif (( STRICT )); then
    echo -e "${RED}${BOLD}FAIL${RESET} — complete sampled token ID parity required by --strict"
    exit 1
elif (( exact_first_output_word_matches == total_prompts )); then
    echo -e "${GREEN}${BOLD}PASS${RESET} — exact first output-word parity with llama.cpp"
    exit 0
elif (( first_word_matches >= (total_prompts + 1) / 2 )); then
    echo -e "${YELLOW}${BOLD}SMOKE PASS${RESET} — majority normalized first-word parity only; use --strict for coherence"
    exit 0
else
    echo -e "${RED}${BOLD}FAIL${RESET} — first-word divergence on most prompts, investigate kernel"
    exit 1
fi
