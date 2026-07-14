#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models}
CASE=${1:-all}
DRY_RUN=${DRY_RUN:-0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/cpu_parity_fixtures.sh"

usage() {
    cat <<EOF
Usage: $0 [dense|moe|all]

Downloads official Gemma4 GGUF fixtures for CPU parity checks into MODEL_DIR.

Environment:
  MODEL_DIR   destination directory, default: models
  DRY_RUN=1   print planned downloads without fetching bytes

After download:
  REQUIRE_MODELS=1 make test_gemma4_cpu_parity
EOF
}

download_one() {
    local name=$1
    local repo=$2
    local file=$3
    local bytes=$4
    local dst="$MODEL_DIR/$file"
    local url="https://huggingface.co/$repo/resolve/main/$file"

    echo "$name: $file ($(cpu_parity_human_gb "$bytes"))"
    echo "  repo: https://huggingface.co/$repo"
    echo "  dest: $dst"

    if [[ -f "$dst" ]]; then
        local have
        have=$(wc -c < "$dst" | tr -d ' ')
        if [[ "$have" == "$bytes" ]]; then
            echo "  present: byte size matches"
            return 0
        fi
        echo "  present: $have bytes, expected $bytes; curl will resume or replace as needed"
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  dry-run: curl -L --fail -C - -o \"$dst\" \"$url\""
        return 0
    fi

    cpu_parity_require_space "$dst" "$bytes"
    curl -L --fail -C - -o "$dst" "$url"

    local actual
    actual=$(wc -c < "$dst" | tr -d ' ')
    if [[ "$actual" != "$bytes" ]]; then
        echo "ERROR: $dst has $actual bytes, expected $bytes" >&2
        exit 1
    fi
}

case "$CASE" in
    -h|--help)
        usage
        exit 0
        ;;
    dense)
        download_one "$CPU_PARITY_GEMMA4_DENSE_NAME" \
            "$CPU_PARITY_GEMMA4_DENSE_REPO" \
            "$CPU_PARITY_GEMMA4_DENSE_FILE" \
            "$CPU_PARITY_GEMMA4_DENSE_BYTES"
        ;;
    moe|sparse)
        download_one "$CPU_PARITY_GEMMA4_MOE_NAME" \
            "$CPU_PARITY_GEMMA4_MOE_REPO" \
            "$CPU_PARITY_GEMMA4_MOE_FILE" \
            "$CPU_PARITY_GEMMA4_MOE_BYTES"
        ;;
    all)
        download_one "$CPU_PARITY_GEMMA4_DENSE_NAME" \
            "$CPU_PARITY_GEMMA4_DENSE_REPO" \
            "$CPU_PARITY_GEMMA4_DENSE_FILE" \
            "$CPU_PARITY_GEMMA4_DENSE_BYTES"
        download_one "$CPU_PARITY_GEMMA4_MOE_NAME" \
            "$CPU_PARITY_GEMMA4_MOE_REPO" \
            "$CPU_PARITY_GEMMA4_MOE_FILE" \
            "$CPU_PARITY_GEMMA4_MOE_BYTES"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
