#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models}
DRY_RUN=${DRY_RUN:-0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/cpu_parity_fixtures.sh"

usage() {
    cat <<EOF
Usage: $0

Downloads the Qwen3.6 sparse MoE GGUF fixture for CPU parity checks into
MODEL_DIR.

Environment:
  MODEL_DIR   destination directory, default: models
  DRY_RUN=1   print planned download without fetching bytes

After download:
  REQUIRE_MODELS=1 make test_qwen_cpu_parity
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 ]]; then
    usage >&2
    exit 2
fi

name=$CPU_PARITY_QWEN36_SPARSE_NAME
repo=$CPU_PARITY_QWEN36_SPARSE_REPO
file=$CPU_PARITY_QWEN36_SPARSE_FILE
bytes=$CPU_PARITY_QWEN36_SPARSE_BYTES
dst="$MODEL_DIR/$file"
url="https://huggingface.co/$repo/resolve/main/$file"

echo "$name: $file ($(cpu_parity_human_gb "$bytes"))"
echo "  repo: https://huggingface.co/$repo"
echo "  dest: $dst"

if [[ -f "$dst" ]]; then
    have=$(wc -c < "$dst" | tr -d ' ')
    if [[ "$have" == "$bytes" ]]; then
        echo "  present: byte size matches"
        exit 0
    fi
    echo "  present: $have bytes, expected $bytes; curl will resume or replace as needed"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "  dry-run: curl -L --fail -C - -o \"$dst\" \"$url\""
    exit 0
fi

cpu_parity_require_space "$dst" "$bytes"
curl -L --fail -C - -o "$dst" "$url"

actual=$(wc -c < "$dst" | tr -d ' ')
if [[ "$actual" != "$bytes" ]]; then
    echo "ERROR: $dst has $actual bytes, expected $bytes" >&2
    exit 1
fi
