#!/bin/bash
# bench_avx2_vs_scalar.sh — A/B benchmark: AVX2 vs scalar on same model
# Usage: ./bench/bench_avx2_vs_scalar.sh model.gguf [iters] [toks]
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 model.gguf [iters] [toks]"
    echo ""
    echo "Builds AVX2 and scalar bench binaries, then runs each model"
    echo "at thread counts 1,2,4,8,16 and prints side-by-side comparison."
    exit 1
fi

MODEL="$1"
ITERS="${2:-100}"
TOKS="${3:-64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THREADS="1 2 4 8 16"

cd "$PROJECT_DIR"

# Build both binaries
echo "=== Building AVX2 binary ==="
make bench_avx2 2>&1 | tail -1
echo "=== Building scalar binary ==="
make bench_scalar 2>&1 | tail -1
echo ""

# Extract model name
FNAME="$(basename "$MODEL")"

echo "================================================================"
echo "A/B Benchmark: AVX2 vs Scalar"
echo "Model: $FNAME | Iters: $ITERS | Toks: $TOKS"
echo "================================================================"
echo ""

for T in $THREADS; do
    echo "--- Threads: $T ---"

    AVX2_OUT=$(mktemp)
    SCALAR_OUT=$(mktemp)

    ./bench_avx2 "$MODEL" --iters "$ITERS" --threads "$T" --toks "$TOKS" > "$AVX2_OUT" 2>&1
    ./bench_scalar "$MODEL" --iters "$ITERS" --threads "$T" --toks "$TOKS" > "$SCALAR_OUT" 2>&1

    # Print AVX2 results
    echo "  AVX2:"
    while IFS= read -r line; do echo "    $line"; done < "$AVX2_OUT"
    echo ""

    # Print scalar results
    echo "  Scalar:"
    while IFS= read -r line; do echo "    $line"; done < "$SCALAR_OUT"
    echo ""

    # Compute speedup from tok/s lines
    AVX2_TOKS=$(grep -o '[0-9.]\+ tok/s' "$AVX2_OUT" | head -1 | awk '{print $1}')
    SCALAR_TOKS=$(grep -o '[0-9.]\+ tok/s' "$SCALAR_OUT" | head -1 | awk '{print $1}')

    if [ -n "$AVX2_TOKS" ] && [ -n "$SCALAR_TOKS" ]; then
        SPEEDUP=$(echo "$AVX2_TOKS $SCALAR_TOKS" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')
        echo "  >> Speedup: $SPEEDUP ($AVX2_TOKS vs $SCALAR_TOKS tok/s)"
    fi

    echo ""
    rm -f "$AVX2_OUT" "$SCALAR_OUT"
done

echo "Done."
