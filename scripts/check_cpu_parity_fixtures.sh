#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models}
REQUIRE_MODELS=${REQUIRE_MODELS:-0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/cpu_parity_fixtures.sh"

missing=0
bad=0

check_file() {
    local name=$1
    local repo=$2
    local file=$3
    local bytes=$4
    local path="$MODEL_DIR/$file"

    if [[ ! -f "$path" ]]; then
        echo "MISSING $name: $path ($(cpu_parity_human_gb "$bytes"))"
        echo "  source: https://huggingface.co/$repo"
        missing=$((missing + 1))
        return
    fi

    local actual
    actual=$(wc -c < "$path" | tr -d ' ')
    if [[ "$actual" != "$bytes" ]]; then
        echo "BADSIZE $name: $path has $actual bytes, expected $bytes"
        bad=$((bad + 1))
        return
    fi

    echo "OK $name: $path ($(cpu_parity_human_gb "$bytes"))"
}

check_pattern() {
    local name=$1
    local env_name=$2
    shift 2
    local value
    value=$(eval "printf '%s' \"\${$env_name:-}\"")
    if [[ -n "$value" ]]; then
        if [[ -f "$value" ]]; then
            echo "OK $name: $value"
            return
        fi
        echo "MISSING $name: $value from $env_name"
        missing=$((missing + 1))
        return
    fi

    local pattern
    local found
    for pattern in "$@"; do
        found=$(find "$MODEL_DIR" -type f -iname "$pattern" ! -iname "*mmproj*" 2>/dev/null | head -n 1 || true)
        if [[ -n "$found" ]]; then
            echo "OK $name: $found"
            return
        fi
    done

    echo "MISSING $name: set $env_name or MODEL_DIR"
    missing=$((missing + 1))
}

cpu_parity_each_requested_pattern_case check_pattern
cpu_parity_each_download_fixture check_file

if [[ "$bad" -ne 0 ]]; then
    echo "CPU parity fixture check FAILED: $bad size-mismatched fixture(s)"
    exit 1
fi

if [[ "$REQUIRE_MODELS" == "1" && "$missing" -ne 0 ]]; then
    echo "CPU parity fixture check FAILED: $missing required fixture(s) missing"
    exit 1
fi

echo "CPU parity fixture check PASSED: missing=$missing bad=$bad"
