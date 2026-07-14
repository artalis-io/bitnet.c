#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/mark/artalis.io/tools/llama.cpp/build/bin}"
LLAMA="${LLAMA:-llama-completion}"
LLAMA_TOKENIZE="${LLAMA_TOKENIZE:-llama-tokenize}"

if [[ "$LLAMA" == "llama-completion" && -x "$LLAMA_BIN_DIR/llama-completion" ]]; then
    LLAMA="$LLAMA_BIN_DIR/llama-completion"
fi
if [[ "$LLAMA_TOKENIZE" == "llama-tokenize" && -x "$LLAMA_BIN_DIR/llama-tokenize" ]]; then
    LLAMA_TOKENIZE="$LLAMA_BIN_DIR/llama-tokenize"
fi

fail=0

check_tool() {
    local label=$1
    local tool=$2
    if [[ -x "$tool" ]]; then
        echo "OK $label: $tool"
        return
    fi
    if command -v "$tool" >/dev/null 2>&1; then
        echo "OK $label: $(command -v "$tool")"
        return
    fi
    echo "MISSING $label: $tool"
    fail=1
}

check_tool "llama-completion" "$LLAMA"
check_tool "llama-tokenize" "$LLAMA_TOKENIZE"

if [[ "$fail" -ne 0 ]]; then
    echo "CPU parity tool check FAILED"
    exit 1
fi

echo "CPU parity tool check PASSED"
