#!/usr/bin/env bash
set -euo pipefail

echo "=== Qwen CPU parity ==="
./test/qwen_cpu_parity.sh

echo "=== Gemma4 CPU parity ==="
./test/gemma4_cpu_parity.sh
