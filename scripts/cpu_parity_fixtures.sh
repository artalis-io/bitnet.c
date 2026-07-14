#!/usr/bin/env bash

CPU_PARITY_QWEN36_SPARSE_NAME="Qwen3.6 sparse MoE"
CPU_PARITY_QWEN36_SPARSE_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
CPU_PARITY_QWEN36_SPARSE_FILE="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
CPU_PARITY_QWEN36_SPARSE_BYTES="22134528992"
CPU_PARITY_QWEN36_SPARSE_ARCH="qwen35moe"

CPU_PARITY_GEMMA4_DENSE_NAME="Gemma4 dense"
CPU_PARITY_GEMMA4_DENSE_REPO="google/gemma-4-E4B-it-qat-q4_0-gguf"
CPU_PARITY_GEMMA4_DENSE_FILE="gemma-4-E4B_q4_0-it.gguf"
CPU_PARITY_GEMMA4_DENSE_BYTES="5154939136"
CPU_PARITY_GEMMA4_DENSE_ARCH="gemma4"

CPU_PARITY_GEMMA4_MOE_NAME="Gemma4 sparse MoE"
CPU_PARITY_GEMMA4_MOE_REPO="google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
CPU_PARITY_GEMMA4_MOE_FILE="gemma-4-26B_q4_0-it.gguf"
CPU_PARITY_GEMMA4_MOE_BYTES="14439361440"
CPU_PARITY_GEMMA4_MOE_ARCH="gemma4"

cpu_parity_human_gb() {
    awk -v b="$1" 'BEGIN { printf "%.2f GB", b / 1000000000 }'
}

cpu_parity_available_bytes() {
    local dir=$1
    mkdir -p "$dir"
    df -Pk "$dir" | awk 'NR == 2 { print $4 * 1024 }'
}

cpu_parity_require_space() {
    local dst=$1
    local expected_bytes=$2
    local dir
    local have=0
    local needed
    local available
    dir=$(dirname "$dst")
    if [[ -f "$dst" ]]; then
        have=$(wc -c < "$dst" | tr -d ' ')
    fi
    needed=$((expected_bytes - have))
    if (( needed <= 0 )); then
        return 0
    fi
    available=$(cpu_parity_available_bytes "$dir")
    if (( available < needed )); then
        echo "ERROR: not enough free space for $dst" >&2
        echo "  need: $(cpu_parity_human_gb "$needed") remaining" >&2
        echo "  free: $(cpu_parity_human_gb "$available") in $dir" >&2
        return 1
    fi
}

cpu_parity_each_requested_pattern_case() {
    "$@" "Qwen 2.5 dense" "BN_MODEL_QWEN25" \
        "*qwen2.5*.gguf"
    "$@" "Qwen 3 dense" "BN_MODEL_QWEN3_DENSE" \
        "*qwen3-[0-9.]*b-q*.gguf"
    "$@" "Qwen 3 sparse MoE" "BN_MODEL_QWEN3_MOE" \
        "*qwen3-*a3b*.gguf"
    "$@" "Qwen 3.5 dense" "BN_MODEL_QWEN35_DENSE" \
        "*qwen3*5*9b-q*.gguf"
    "$@" "Qwen 3.5 sparse MoE" "BN_MODEL_QWEN35_MOE" \
        "*qwen3.5*35b*a3b*.gguf"
    "$@" "Qwen 3.6 dense" "BN_MODEL_QWEN36_DENSE" \
        "*qwen3.6*27b*.gguf"
}

cpu_parity_each_download_fixture() {
    "$@" "$CPU_PARITY_QWEN36_SPARSE_NAME" \
        "$CPU_PARITY_QWEN36_SPARSE_REPO" \
        "$CPU_PARITY_QWEN36_SPARSE_FILE" \
        "$CPU_PARITY_QWEN36_SPARSE_BYTES" \
        "$CPU_PARITY_QWEN36_SPARSE_ARCH"
    "$@" "$CPU_PARITY_GEMMA4_DENSE_NAME" \
        "$CPU_PARITY_GEMMA4_DENSE_REPO" \
        "$CPU_PARITY_GEMMA4_DENSE_FILE" \
        "$CPU_PARITY_GEMMA4_DENSE_BYTES" \
        "$CPU_PARITY_GEMMA4_DENSE_ARCH"
    "$@" "$CPU_PARITY_GEMMA4_MOE_NAME" \
        "$CPU_PARITY_GEMMA4_MOE_REPO" \
        "$CPU_PARITY_GEMMA4_MOE_FILE" \
        "$CPU_PARITY_GEMMA4_MOE_BYTES" \
        "$CPU_PARITY_GEMMA4_MOE_ARCH"
}
