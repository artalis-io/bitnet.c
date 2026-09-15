#!/usr/bin/env bash
set -euo pipefail

fail=0
echo "=== Qwen CPU parity ==="
if ! ./test/qwen_cpu_parity.sh; then
    fail=1
fi

echo "=== Gemma4 CPU parity ==="
if ! ./test/gemma4_cpu_parity.sh; then
    fail=1
fi
exit "$fail"
