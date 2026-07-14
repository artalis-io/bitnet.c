#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/cpu_parity_fixtures.sh"

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not found" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found" >&2
    exit 1
fi

fail=0

check_remote_fixture() {
    local name=$1
    local repo=$2
    local file=$3
    local bytes=$4
    local arch=$5
    local api="https://huggingface.co/api/models/$repo"
    local tree_api="https://huggingface.co/api/models/$repo/tree/main?recursive=false"
    local json
    local tree_json
    local found_file
    local remote_arch
    local remote_size

    if ! json=$(curl -fsSL "$api"); then
        echo "REMOTE FAIL $name: cannot fetch $api"
        fail=1
        return
    fi
    if ! tree_json=$(curl -fsSL "$tree_api"); then
        echo "REMOTE FAIL $name: cannot fetch $tree_api"
        fail=1
        return
    fi

    found_file=$(printf '%s\n' "$json" | jq -r --arg file "$file" '
        any(.siblings[]?; .rfilename == $file)
    ')
    remote_arch=$(printf '%s\n' "$json" | jq -r '.gguf.architecture // ""')
    remote_size=$(printf '%s\n' "$tree_json" | jq -r --arg file "$file" '
        map(select(.path == $file)) | .[0].size // ""
    ')

    if [[ "$found_file" != "true" ]]; then
        echo "REMOTE MISSING $name: $file not listed in $repo"
        fail=1
        return
    fi
    if [[ "$remote_arch" != "$arch" ]]; then
        echo "REMOTE BADARCH $name: $repo reports '$remote_arch', expected '$arch'"
        fail=1
        return
    fi
    if [[ "$remote_size" != "$bytes" ]]; then
        echo "REMOTE BADSIZE $name: $repo/$file reports $remote_size bytes, expected $bytes"
        fail=1
        return
    fi

    echo "REMOTE OK $name: $repo/$file arch=$remote_arch size=$(cpu_parity_human_gb "$bytes")"
}

cpu_parity_each_download_fixture check_remote_fixture

if [[ "$fail" -ne 0 ]]; then
    echo "CPU parity remote fixture check FAILED"
    exit 1
fi

echo "CPU parity remote fixture check PASSED"
