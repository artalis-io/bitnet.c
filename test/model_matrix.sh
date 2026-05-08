#!/bin/sh
set -eu

ROOT=${BN_MODEL_ROOT:-}
REQUIRE_MODELS=${REQUIRE_MODELS:-0}
RUN_WEBGPU=${RUN_WEBGPU:-0}
RUN_METAL=${RUN_METAL:-0}
COHERENCE=${COHERENCE:-./test_coherence}

fail=0
ran=0
missing=0

find_model() {
    env_name=$1
    pattern=$2
    value=$(eval "printf '%s' \"\${$env_name:-}\"")
    if [ -n "$value" ]; then
        printf '%s\n' "$value"
        return 0
    fi
    if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then
        found=$(find "$ROOT" -type f -iname "$pattern" 2>/dev/null | head -n 1 || true)
        if [ -n "$found" ]; then
            printf '%s\n' "$found"
            return 0
        fi
    fi
    return 1
}

run_case() {
    name=$1
    env_name=$2
    pattern=$3

    if path=$(find_model "$env_name" "$pattern"); then
        echo "RUN $name: $path"
        ran=$((ran + 1))
        args=$path
        if [ "$RUN_WEBGPU" = "1" ]; then
            "$COHERENCE" "$args" --webgpu || fail=1
        elif [ "$RUN_METAL" = "1" ]; then
            "$COHERENCE" "$args" --metal || fail=1
        else
            "$COHERENCE" "$args" || fail=1
        fi
    else
        echo "SKIP $name: set $env_name or BN_MODEL_ROOT"
        missing=$((missing + 1))
    fi
}

run_case "Llama 2 dense" "BN_MODEL_LLAMA2" "*llama*2*.gguf"
run_case "Llama 3 dense" "BN_MODEL_LLAMA3" "*llama*3*.gguf"
run_case "Microsoft BitNet 1.58" "BN_MODEL_BITNET158" "*bitnet*b1.58*.gguf"
run_case "Qwen 2.5 dense" "BN_MODEL_QWEN25" "*qwen2.5*.gguf"
run_case "Qwen 3 dense" "BN_MODEL_QWEN3_DENSE" "*qwen3-*0.6b*.gguf"
run_case "Qwen 3 sparse MoE" "BN_MODEL_QWEN3_MOE" "*qwen3-*a3b*.gguf"
run_case "Qwen 3.5 dense" "BN_MODEL_QWEN35_DENSE" "*qwen3.5*9b*.gguf"
run_case "Qwen 3.5 sparse MoE" "BN_MODEL_QWEN35_MOE" "*qwen3.5*35b*a3b*.gguf"

if [ "$REQUIRE_MODELS" = "1" ] && [ "$missing" -ne 0 ]; then
    echo "Model matrix FAILED: $missing required model case(s) missing"
    exit 1
fi

if [ "$fail" -ne 0 ]; then
    echo "Model matrix FAILED"
    exit 1
fi

echo "Model matrix PASSED: ran=$ran skipped=$missing"
