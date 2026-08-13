#!/usr/bin/env bash
# Compare bitnet.c Q4_0 output against llama.cpp for numerical equivalence.
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
# Strict mode also uses llama-tokenize to compare retokenized first output IDs.

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [-n tokens] [-v]}"
shift
N_TOKENS=30
VERBOSE=0
STRICT=0
BITNET_ARGS=()
LLAMA_ARGS=(-ngl 0 -dev none)
LLAMA_GPU_LAYERS=0
LLAMA_FLASH=()
LLAMA_THREADS=1
LLAMA_CTX=0
LLAMA_PROBE_KV_F16=0
LLAMA_PROBE_FLASH=0
LLAMA_THREADS_BATCH=()
CUSTOM_PROMPTS=()
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/mark/artalis.io/tools/llama.cpp/build/bin}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) N_TOKENS="$2"; shift 2 ;;
        --prompt) CUSTOM_PROMPTS+=("$2"); shift 2 ;;
        --metal) BITNET_ARGS+=(--metal); shift ;;
        --llama-metal) LLAMA_ARGS=(-ngl 99); LLAMA_GPU_LAYERS=99; shift ;;
        --cuda) BITNET_ARGS+=(--cuda); shift ;;
        --kv16) BITNET_ARGS+=(--kv16); LLAMA_PROBE_KV_F16=1; shift ;;
        --llama-cuda) LLAMA_ARGS=(-ngl 99); LLAMA_GPU_LAYERS=99; shift ;;
        --llama-gpu-layers) LLAMA_ARGS=(-ngl "$2"); LLAMA_GPU_LAYERS="$2"; shift 2 ;;
        --llama-cache-k) LLAMA_ARGS+=(-ctk "$2"); shift 2 ;;
        --llama-cache-v) LLAMA_ARGS+=(-ctv "$2"); shift 2 ;;
        --llama-batch) LLAMA_ARGS+=(-b "$2"); shift 2 ;;
        --llama-ubatch) LLAMA_ARGS+=(-ub "$2"); shift 2 ;;
        --llama-threads-batch) LLAMA_THREADS_BATCH=(-tb "$2"); shift 2 ;;
        --webgpu|--gpu) BITNET_ARGS+=(--webgpu); shift ;;
        --no-prefill) BITNET_ARGS+=(--no-prefill); shift ;;
        --pread) BITNET_ARGS+=(--pread); shift ;;
        --cache-mb) BITNET_ARGS+=(--cache-mb "$2"); shift 2 ;;
        --madvise) BITNET_ARGS+=(--madvise); shift ;;
        --flash) BITNET_ARGS+=(--flash); LLAMA_FLASH=(-fa on); LLAMA_PROBE_FLASH=1; shift ;;
        --llama-flash-off) LLAMA_FLASH=(-fa off); LLAMA_PROBE_FLASH=0; shift ;;
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
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Check dependencies
BITNET="${BITNET:-./bitnet}"
LLAMA="${LLAMA:-llama-completion}"
LLAMA_TOKENIZE="${LLAMA_TOKENIZE:-llama-tokenize}"
LLAMA_TOKEN_PROBE="${LLAMA_TOKEN_PROBE:-./test/llama_layer_probe}"
if [[ "$LLAMA" == "llama-completion" && -x "$LLAMA_BIN_DIR/llama-completion" ]]; then
    LLAMA="$LLAMA_BIN_DIR/llama-completion"
fi
if [[ "$LLAMA_TOKENIZE" == "llama-tokenize" && -x "$LLAMA_BIN_DIR/llama-tokenize" ]]; then
    LLAMA_TOKENIZE="$LLAMA_BIN_DIR/llama-tokenize"
fi
LLAMA_LIB_DIR="${LLAMA_LIB_DIR:-$(dirname "$LLAMA")}"
if [[ ! -x "$BITNET" ]]; then
    echo "ERROR: $BITNET not found. Run 'make' first." >&2; exit 1
fi
if [[ ! -x "$LLAMA" ]] && ! command -v "$LLAMA" &>/dev/null; then
    echo "ERROR: $LLAMA not found. Run 'brew install llama.cpp'." >&2; exit 1
fi
if (( STRICT )) && [[ ! -x "$LLAMA_TOKENIZE" ]] && ! command -v "$LLAMA_TOKENIZE" &>/dev/null; then
    echo "ERROR: $LLAMA_TOKENIZE not found. Run 'brew install llama.cpp'." >&2; exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL" >&2; exit 1
fi

first_token_id() {
    local text="$1"
    local ids
    ids=$(LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA_TOKENIZE" -m "$MODEL" --ids --no-bos -p "$text" --log-disable 2>/dev/null) || return 1
    ids="${ids#[}"
    ids="${ids%]}"
    ids="${ids//[[:space:]]/}"
    if [[ -z "$ids" ]]; then
        return 1
    fi
    printf '%s\n' "${ids%%,*}"
}

token_ids_csv() {
    local text="$1"
    local ids
    ids=$(LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA_TOKENIZE" -m "$MODEL" --ids --no-bos -p "$text" --log-disable 2>/dev/null) || return 1
    ids="${ids#[}"
    ids="${ids%]}"
    ids="${ids//[[:space:]]/}"
    printf '%s\n' "$ids"
}

llama_generated_ids_csv() {
    local prompt="$1"
    local probe_args=()
    if (( LLAMA_PROBE_KV_F16 )); then
        probe_args+=(--kv-f16)
    fi
    if (( LLAMA_PROBE_FLASH )); then
        probe_args+=(--flash)
    fi
    "$LLAMA_TOKEN_PROBE" -m "$MODEL" -p "$prompt" --generate "$N_TOKENS" \
        --no-observer \
        ${probe_args[@]+"${probe_args[@]}"} --ctx "$LLAMA_CTX" \
        2>/dev/null | sed -n 's/^llama_token_id=//p' | paste -sd, -
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

echo -e "${BOLD}Q4_0 output comparison: bitnet.c vs llama.cpp${RESET}"
echo "Model:  $MODEL"
echo "Tokens: $N_TOKENS per prompt"
if (( ${#BITNET_ARGS[@]} > 0 )); then
    echo "bitnet args: ${BITNET_ARGS[*]}"
fi
if (( ${#LLAMA_FLASH[@]} > 0 )); then
    echo "llama args:  ${LLAMA_ARGS[*]} ${LLAMA_FLASH[*]} -t $LLAMA_THREADS ${LLAMA_THREADS_BATCH[*]-}"
else
    echo "llama args:  ${LLAMA_ARGS[*]} -t $LLAMA_THREADS ${LLAMA_THREADS_BATCH[*]-}"
fi
echo "---"

for prompt in "${PROMPTS[@]}"; do
    # Run bitnet.c (raw completion, temp=0, no repeat penalty)
    bitnet_stderr="/dev/null"
    bitnet_run_args=()
    if (( ${#BITNET_ARGS[@]} > 0 )); then
        bitnet_run_args=("${BITNET_ARGS[@]}")
    fi
    if (( STRICT )); then
        bitnet_stderr=$(mktemp)
        tmp_files+=("$bitnet_stderr")
        bitnet_run_args+=(--token-ids)
    fi
    bitnet_out=$("$BITNET" "$MODEL" \
        ${bitnet_run_args[@]+"${bitnet_run_args[@]}"} \
        -p "$prompt" -n "$N_TOKENS" \
        --temp 0 --repeat-penalty 1 2>"$bitnet_stderr") || true

    # Run llama.cpp (raw completion, no chat template, temp=0)
    llama_stderr=$(mktemp)
    tmp_files+=("$llama_stderr")
    llama_run_args=("${LLAMA_ARGS[@]}")
    if (( ${#LLAMA_FLASH[@]} > 0 )); then
        llama_run_args+=("${LLAMA_FLASH[@]}")
    fi
    llama_out=$(LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA" -m "$MODEL" "${llama_run_args[@]}" -p "$prompt" -n "$N_TOKENS" \
        --temp 0 --repeat-penalty 1 --reasoning-format none \
        --no-display-prompt -no-cnv \
        --simple-io --verbosity 1 \
        -t "$LLAMA_THREADS" ${LLAMA_THREADS_BATCH[@]+"${LLAMA_THREADS_BATCH[@]}"} \
        2>"$llama_stderr" | sed 's/> EOF by user$//') || true

    if [[ -z "$bitnet_out" || -z "$llama_out" ]]; then
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
    read -ra bwords <<< "$bitnet_out" || bwords=()
    read -ra lwords <<< "$llama_out"  || lwords=()

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
        llama_first_token=$(first_token_id "$llama_out") || llama_first_token=""
        if [[ -n "$bitnet_first_token" && "$bitnet_first_token" == "$llama_first_token" ]]; then
            (( first_token_matches++ )) || true
        fi
        bitnet_ids_csv=$(sed -n 's/^token_id=//p' "$bitnet_stderr" | paste -sd, -) || bitnet_ids_csv=""
        if [[ -x "$LLAMA_TOKEN_PROBE" && "$LLAMA_GPU_LAYERS" -eq 0 ]]; then
            llama_ids_csv=$(llama_generated_ids_csv "$prompt") || llama_ids_csv=""
        else
            llama_ids_csv=$(token_ids_csv "$llama_out") || llama_ids_csv=""
        fi
        IFS=, read -ra bitnet_ids <<< "$bitnet_ids_csv"
        IFS=, read -ra llama_ids <<< "$llama_ids_csv"
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
    if (( match == max_cmp && max_cmp > 0 )); then
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
done

echo "---"
if (( STRICT )); then
    echo "First output-token ID matches: $first_token_matches / $total_prompts prompts"
    echo "Generated token-ID prefix matches: $total_token_ids_matched / $total_token_ids_compared tokens"
fi
if (( STRICT )) && [[ -x "$LLAMA_TOKEN_PROBE" && "$LLAMA_GPU_LAYERS" -eq 0 ]]; then
    echo "strict token source: llama.cpp sampled IDs ($LLAMA_TOKEN_PROBE)"
elif (( STRICT )); then
    echo "strict token source: retokenized requested-backend llama.cpp text" >&2
fi
echo "Exact first output-word matches: $exact_first_output_word_matches / $total_prompts prompts"
echo "Punctuation-normalized first-word matches: $first_word_matches / $total_prompts prompts"
echo "Word prefix matches: $total_words_matched / $total_words_compared total words"
echo ""

# Strict mode uses actual llama.cpp sampled IDs when the probe is available.
# Display text can include frontend formatting that was not sampled by the
# model, and retokenized output is not necessarily invertible.
sampled_ids_required=0
if (( STRICT )) && [[ -x "$LLAMA_TOKEN_PROBE" && "$LLAMA_GPU_LAYERS" -eq 0 ]]; then
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
