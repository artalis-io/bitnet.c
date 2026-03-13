#!/bin/bash
# Benchmark bitnet.c inference speed
set -e

MODEL="${1:-models/bitnet-b1.58-2B-4T.gguf}"
PROMPT="The capital of France is"
N_TOKENS=64
RUNS=3

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo "Usage: $0 [model.gguf]"
    exit 1
fi

# Build optimized binary
echo "Building..."
make -s clean && make -s bitnet

echo ""
echo "=== Benchmark ==="
echo "Model: $MODEL"
echo "Prompt: \"$PROMPT\""
echo "Tokens: $N_TOKENS"
echo "Runs: $RUNS"
echo ""

for i in $(seq 1 $RUNS); do
    echo "--- Run $i ---"
    ./bitnet "$MODEL" -p "$PROMPT" -n $N_TOKENS 2>&1 | tee /dev/stderr | grep -E "Speed:|Thread pool:" > /dev/null
    echo ""
done
