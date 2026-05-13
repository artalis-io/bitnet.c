#!/usr/bin/env bash
# Compare bitnet.c Q4_0 output against llama.cpp for numerical equivalence.
#
# Usage:
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf -n 50
#   ./test/compare_llama.sh models/qwen2.5-3b-instruct-q4_0.gguf --metal
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
LLAMA_THREADS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) N_TOKENS="$2"; shift 2 ;;
        --metal) BITNET_ARGS+=(--metal); shift ;;
        --webgpu|--gpu) BITNET_ARGS+=(--webgpu); shift ;;
        --no-prefill) BITNET_ARGS+=(--no-prefill); shift ;;
        --maxseq) BITNET_ARGS+=(--maxseq "$2"); LLAMA_ARGS+=(-c "$2"); shift 2 ;;
        -t) BITNET_ARGS+=(-t "$2"); LLAMA_THREADS="$2"; shift 2 ;;
        -v) VERBOSE=1; shift ;;
        --strict) STRICT=1; shift ;;
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Check dependencies
BITNET="./bitnet"
LLAMA="llama-completion"
LLAMA_TOKENIZE="llama-tokenize"
if [[ ! -x "$BITNET" ]]; then
    echo "ERROR: $BITNET not found. Run 'make' first." >&2; exit 1
fi
if ! command -v "$LLAMA" &>/dev/null; then
    echo "ERROR: $LLAMA not found. Run 'brew install llama.cpp'." >&2; exit 1
fi
if (( STRICT )) && ! command -v "$LLAMA_TOKENIZE" &>/dev/null; then
    echo "ERROR: $LLAMA_TOKENIZE not found. Run 'brew install llama.cpp'." >&2; exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL" >&2; exit 1
fi

first_token_id() {
    local text="$1"
    local ids
    ids=$("$LLAMA_TOKENIZE" -m "$MODEL" --ids --no-bos -p "$text" --log-disable 2>/dev/null) || return 1
    ids="${ids#[}"
    ids="${ids%]}"
    ids="${ids//[[:space:]]/}"
    if [[ -z "$ids" ]]; then
        return 1
    fi
    printf '%s\n' "${ids%%,*}"
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
echo "llama args:  ${LLAMA_ARGS[*]} -t $LLAMA_THREADS"
echo "---"

for prompt in "${PROMPTS[@]}"; do
    # Run bitnet.c (raw completion, temp=0, no repeat penalty)
    bitnet_stderr="/dev/null"
    bitnet_run_args=("${BITNET_ARGS[@]}")
    if (( STRICT )); then
        bitnet_stderr=$(mktemp)
        tmp_files+=("$bitnet_stderr")
        bitnet_run_args+=(--token-ids)
    fi
    bitnet_out=$("$BITNET" "$MODEL" "${bitnet_run_args[@]}" -p "$prompt" -n "$N_TOKENS" \
        --temp 0 --repeat-penalty 1 2>"$bitnet_stderr") || true

    # Run llama.cpp (raw completion, no chat template, temp=0)
    llama_out=$("$LLAMA" -m "$MODEL" "${LLAMA_ARGS[@]}" -p "$prompt" -n "$N_TOKENS" \
        --temp 0 --no-display-prompt -no-cnv --simple-io --verbosity 1 \
        -t "$LLAMA_THREADS" 2>/dev/null | sed 's/> EOF by user$//') || true

    if [[ -z "${bitnet_out//[[:space:]]/}" || -z "${llama_out//[[:space:]]/}" ]]; then
        echo -e "${RED}ERROR${RESET}   \"$prompt\""
        [[ -n "${bitnet_out//[[:space:]]/}" ]] || echo "  bitnet produced no completion"
        [[ -n "${llama_out//[[:space:]]/}" ]] || echo "  llama.cpp produced no completion"
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
    if (( STRICT )); then
        bitnet_first_token=$(sed -n 's/^token_id=//p' "$bitnet_stderr" | head -n 1) || bitnet_first_token=""
        llama_first_token=$(first_token_id "$llama_out") || llama_first_token=""
        if [[ -n "$bitnet_first_token" && "$bitnet_first_token" == "$llama_first_token" ]]; then
            (( first_token_matches++ )) || true
        fi
    fi

    # Report
    prompt_short="${prompt:0:45}"
    if (( match == max_cmp && max_cmp > 0 )); then
        echo -e "${GREEN}MATCH${RESET}   ${DIM}[$match/$max_cmp words]${RESET}  \"$prompt_short\""
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
        fi
    elif (( STRICT )) && [[ "$bitnet_first_token" != "$llama_first_token" ]]; then
        echo -e "  ${DIM}first token IDs:${RESET} bitnet=$bitnet_first_token llama=$llama_first_token"
    fi
done

echo "---"
if (( STRICT )); then
    echo "First output-token ID matches: $first_token_matches / $total_prompts prompts"
fi
echo "Exact first output-word matches: $exact_first_output_word_matches / $total_prompts prompts"
echo "Punctuation-normalized first-word matches: $first_word_matches / $total_prompts prompts"
echo "Word prefix matches: $total_words_matched / $total_words_compared total words"
echo ""

# The strict correctness signal is first generated token ID parity across all
# prompts. bitnet IDs come from the generation callback; llama.cpp IDs are
# retokenized from decoded output until a llama token trace is available.
if (( STRICT && first_token_matches == total_prompts )); then
    echo -e "${GREEN}${BOLD}PASS${RESET} — first output-token ID parity with llama.cpp"
    exit 0
elif (( STRICT )); then
    echo -e "${RED}${BOLD}FAIL${RESET} — first output-token ID parity required by --strict"
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
