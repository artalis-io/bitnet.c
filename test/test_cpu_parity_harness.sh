#!/usr/bin/env bash
# Synthetic parity matrix wiring tests: no model load or inference needed.
set -euo pipefail

if [[ "${CPU_PARITY_ARGS_FIXTURE:-0}" == 1 ]]; then
    if [[ "${1:-}" != -m && "${BN_DISABLE_LOOP_ABORT:-}" != 1 ]]; then
        echo 'Parity comparison did not disable the interactive loop guard' >&2
        exit 9
    fi
    if [[ "${1:-}" == -m ]]; then
        args=" $* "
        for pair in "-c 512" "-b ${CPU_PARITY_EXPECT_BATCH:-17}" \
                    "-ub ${CPU_PARITY_EXPECT_UBATCH:-9}" \
                    "-ngl ${CPU_PARITY_EXPECT_LAYERS:-99}"; do
            if [[ "$args" != *" $pair "* ]]; then
                echo "Missing reference arguments: $pair" >&2
                exit 8
            fi
        done
        echo 'offloaded 40/40 layers to GPU' >&2
    else
        args=" $* "
        if [[ "$args" != *" --repeat-penalty 1 "* ]]; then
            echo 'Parity comparison did not disable bitnet repetition penalty' >&2
            exit 8
        fi
        echo 'CUDA weights uploaded' >&2
    fi
    printf 'Paris.\n'
    exit 0
fi

if [[ -n "${CPU_PARITY_FAILED_PROCESS_FIXTURE:-}" ]]; then
    printf 'Paris.\n'
    backend=bitnet
    prefix=token_id
    if [[ "${1:-}" == -m ]]; then
        backend=llama
        prefix=llama_token_id
    fi
    printf '%s=1\n' "$prefix" >&2
    if [[ "$CPU_PARITY_FAILED_PROCESS_FIXTURE" == "$backend" ]]; then
        exit 7
    fi
    exit 0
fi

if [[ -n "${CPU_PARITY_EMPTY_FIXTURE:-}" ]]; then
    backend=bitnet
    prefix=token_id
    if [[ "${1:-}" == -m ]]; then
        backend=llama
        prefix=llama_token_id
    fi
    prompt=""
    while (( $# )); do
        if [[ "$1" == -p ]]; then prompt=$2; break; fi
        shift
    done
    case "$CPU_PARITY_EMPTY_FIXTURE" in
        missing-both) exit 0 ;;
        missing-bitnet|missing-llama)
            [[ "$CPU_PARITY_EMPTY_FIXTURE" != "missing-$backend" ]] || exit 0 ;;
        missing-one-prompt) [[ "$prompt" != missing ]] || exit 0 ;;
    esac
    token=248046
    if [[ "$CPU_PARITY_EMPTY_FIXTURE" == mismatch && "$backend" == llama ]]; then
        token=248047
    fi
    printf '%s=%s\n' "$prefix" "$token" >&2
    exit 0
fi

if [[ "${CPU_PARITY_TOKEN_COUNT_FIXTURE:-0}" == 1 ]]; then
    printf 'Paris.\n'
    if [[ "${1:-}" == -m ]]; then
        printf 'llama_token_id=1\nllama_token_id=2\nllama_token_id=3\n' >&2
    else
        printf 'token_id=1\ntoken_id=2\n' >&2
    fi
    exit 0
fi

if [[ "${CPU_PARITY_MULTILINE_FIXTURE:-0}" == 1 ]]; then
    if [[ "${1:-}" == -m ]]; then
        printf 'Paris.\nDifferent ending.\n'
    else
        printf 'Paris.\nSecond line.\n'
    fi
    exit 0
fi

export COMPARE_LLAMA
COMPARE_LLAMA=$(type -P echo)
export BITNET_AVX2 BITNET_AVX512
BITNET_AVX2=$(type -P true)
BITNET_AVX512=$BITNET_AVX2
failing_compare=$(type -P false)
export CPU_PARITY_BACKENDS=avx2,avx512 CPU_PARITY_THREADS=8
export QWEN_CPU_PARITY_BACKENDS=avx2,avx512
export GEMMA4_CPU_PARITY_BACKENDS=avx2,avx512
export QWEN_CPU_PARITY_CASES=all GEMMA4_CPU_PARITY_CASES=all
export QWEN_CPU_PARITY_LEVEL=full GEMMA4_CPU_PARITY_LEVEL=full
export REQUIRE_MODELS=1
unset QWEN_CPU_PARITY_THREADS GEMMA4_CPU_PARITY_THREADS
unset CPU_PARITY_PREFILL_MODES QWEN_CPU_PARITY_PREFILL_MODES GEMMA4_CPU_PARITY_PREFILL_MODES
for name in BN_MODEL_QWEN25 BN_MODEL_QWEN3_DENSE BN_MODEL_QWEN3_MOE \
    BN_MODEL_QWEN35_DENSE BN_MODEL_QWEN35_MOE BN_MODEL_QWEN36_DENSE \
    BN_MODEL_QWEN36_MOE BN_MODEL_QWEN38_DENSE BN_MODEL_QWEN38_MOE \
    BN_MODEL_GEMMA4_DENSE BN_MODEL_GEMMA4_MOE; do
    export "$name=$0"
done

expect_count() {
    local output=$1 needle=$2 expected=$3 count=0 line
    while IFS= read -r line; do
        if [[ "$line" == *"$needle"* ]]; then
            count=$((count + 1))
        fi
    done <<< "$output"
    if [[ "$count" != "$expected" ]]; then
        echo "Expected $expected occurrences of '$needle', got $count" >&2
        echo "$output" >&2
        exit 1
    fi
}

output=$(./test/qwen_cpu_parity.sh)
expect_count "$output" '-n 16 --strict -t 8' 36
output=$(./test/gemma4_cpu_parity.sh)
expect_count "$output" '-n 16 --strict -t 8' 8

output=$(QWEN_CPU_PARITY_THREADS=3 ./test/qwen_cpu_parity.sh)
expect_count "$output" '-n 16 --strict -t 3' 36
output=$(GEMMA4_CPU_PARITY_THREADS=4 ./test/gemma4_cpu_parity.sh)
expect_count "$output" '-n 16 --strict -t 4' 8

# Preserve the quicker standard smoke level and the historical one-thread
# default; full acceptance is explicitly selected by the caller.
output=$(CPU_PARITY_THREADS=1 QWEN_CPU_PARITY_LEVEL=standard \
    ./test/qwen_cpu_parity.sh)
expect_count "$output" '-n 3 --strict -t 1' 28
expect_count "$output" '-n 5 --strict -t 1' 8
output=$(CPU_PARITY_THREADS=1 GEMMA4_CPU_PARITY_LEVEL=standard \
    ./test/gemma4_cpu_parity.sh)
expect_count "$output" '-n 3 --strict -t 1' 8

# Both modes must be wired separately, and a focused selection must not
# accidentally run the other prefill configuration.
for family in qwen gemma4; do
    output=$("./test/${family}_cpu_parity.sh")
    count=18
    [[ "$family" != gemma4 ]] || count=4
    expect_count "$output" '  prefill=default' "$count"
    expect_count "$output" '--no-prefill' "$count"
    output=$(CPU_PARITY_PREFILL_MODES=default "./test/${family}_cpu_parity.sh")
    expect_count "$output" '--no-prefill' 0
    output=$(CPU_PARITY_PREFILL_MODES=tokenwise "./test/${family}_cpu_parity.sh")
    expect_count "$output" '  prefill=default' 0
    expect_count "$output" '--no-prefill' "$count"
    for bad in invalid default, ,tokenwise default,,tokenwise; do
        if CPU_PARITY_PREFILL_MODES="$bad" "./test/${family}_cpu_parity.sh" >/dev/null 2>&1; then
            echo "Invalid prefill modes '$bad' accepted by $family" >&2
            exit 1
        fi
    done
done

for bad in 0 -1 abc 1.5; do
    for family in qwen gemma4; do
        if CPU_PARITY_THREADS="$bad" "./test/${family}_cpu_parity.sh" \
            >/dev/null 2>&1; then
            echo "Invalid thread count '$bad' accepted by $family" >&2
            exit 1
        fi
    done
done

# A failed Qwen comparison must not skip Gemma4, and the overall command
# must fail even after collecting the later family's results.
if output=$(COMPARE_LLAMA="$failing_compare" ./test/cpu_parity.sh 2>&1); then
    echo "CPU matrix incorrectly passed failed comparisons" >&2
    exit 1
fi
expect_count "$output" 'RUN AVX2 Qwen' 9
expect_count "$output" 'RUN AVX512 Qwen' 9
expect_count "$output" 'RUN AVX2 Gemma4' 2
expect_count "$output" 'RUN AVX512 Gemma4' 2
expect_count "$output" 'CPU parity FAILED' 2

output=$(CPU_PARITY_MULTILINE_FIXTURE=1 BITNET="$0" LLAMA="$0" \
    ./test/compare_llama.sh "$0" --prompt fixture)
expect_count "$output" 'Word prefix matches: 1 / 3 total words' 1

# The fake programs emit sampled IDs directly; the existing script is only
# a regular-file placeholder for the trace-library presence check.
if output=$(CPU_PARITY_TOKEN_COUNT_FIXTURE=1 BITNET="$0" LLAMA="$0" \
    LLAMA_TOKEN_TRACE="$0" ./test/compare_llama.sh "$0" --prompt fixture --strict); then
    echo "Unequal sampled token counts incorrectly passed" >&2
    exit 1
fi
expect_count "$output" 'token ID counts: bitnet=2 llama=3' 1
expect_count "$output" 'Token-count mismatches: 1 / 1 prompts (bitnet=2 llama=3)' 1

# Matching sampled stop tokens are valid parity even when neither frontend
# prints text. Missing traces must fail per prompt, not only in aggregate.
output=$(CPU_PARITY_EMPTY_FIXTURE=match BITNET="$0" LLAMA="$0" \
    LLAMA_TOKEN_TRACE="$0" ./test/compare_llama.sh "$0" --prompt fixture --strict)
expect_count "$output" 'Generated token-ID prefix matches: 1 / 1 tokens' 1
expect_count "$output" 'sampled token ID parity with llama.cpp' 1
for fixture in mismatch missing-bitnet missing-llama missing-both missing-one-prompt; do
    if output=$(CPU_PARITY_EMPTY_FIXTURE="$fixture" BITNET="$0" LLAMA="$0" \
        LLAMA_TOKEN_TRACE="$0" ./test/compare_llama.sh "$0" \
        --prompt fixture --prompt missing --strict 2>&1); then
        echo "Empty-output $fixture incorrectly passed strict comparison" >&2
        exit 1
    fi
    if [[ "$fixture" == mismatch ]]; then
        expect_count "$output" 'complete sampled token ID parity required' 1
    else
        expect_count "$output" 'missing sampled token IDs' 1
    fi
done
if output=$(CPU_PARITY_EMPTY_FIXTURE=match BITNET="$0" LLAMA="$0" \
    ./test/compare_llama.sh "$0" --prompt fixture 2>&1); then
    echo "Empty text incorrectly passed word-only comparison" >&2
    exit 1
fi
expect_count "$output" 'produced no completion' 2

# Matching partial output must not hide a failed inference process, including
# llama-completion's failure upstream of the output-cleanup pipeline.
for backend in bitnet llama; do
    for mode in smoke strict; do
        args=()
        [[ "$mode" != strict ]] || args+=(--strict)
        if output=$(CPU_PARITY_FAILED_PROCESS_FIXTURE="$backend" \
            BITNET="$0" LLAMA="$0" LLAMA_TOKEN_TRACE="$0" \
            ./test/compare_llama.sh "$0" --prompt fixture "${args[@]}" 2>&1); then
            echo "Failed $backend inference incorrectly passed $mode comparison" >&2
            exit 1
        fi
        expect_count "$output" 'inference failed (exit 7)' 1
    done
done

for family in qwen gemma4; do
    filter=dense
    [[ "$family" != qwen ]] || filter=qwen3_dense
    output=$(RUN_COHERENCE=0 RUN_SHARDED_MOE_SMOKE=0 RUN_BENCH=0 \
        RUN_LLAMA_COMPARE=1 CUDA_PARITY_PREFILL_MODES=default,tokenwise \
        "./bench/${family}_cuda_matrix.sh" "$filter")
    expect_count "$output" '--cuda --llama-cuda --strict' 2
    expect_count "$output" '--no-prefill' 1
    if output=$(RUN_COHERENCE=0 RUN_SHARDED_MOE_SMOKE=0 RUN_BENCH=0 \
        RUN_LLAMA_COMPARE=1 CUDA_PARITY_PREFILL_MODES=default,tokenwise \
        COMPARE_LLAMA="$failing_compare" \
        "./bench/${family}_cuda_matrix.sh" "$filter" 2>&1); then
        echo "CUDA matrix incorrectly passed failed comparisons" >&2
        exit 1
    fi
    expect_count "$output" '  prefill=default' 1
    expect_count "$output" '  prefill=tokenwise' 1
done

# Backend selection must not erase context or batch options, regardless of
# ordering. Exercise the actual child argv with synthetic executables.
for selector in cuda metal layers; do
    backend_args=(--llama-cuda)
    layers=99
    [[ "$selector" != metal ]] || backend_args=(--llama-metal)
    if [[ "$selector" == layers ]]; then
        backend_args=(--llama-gpu-layers 7)
        layers=7
    fi
    for order in before after; do
        for batch_mode in explicit tokenwise; do
            options=(--maxseq 512 --llama-batch 17 --llama-ubatch 9)
            batch=17
            ubatch=9
            if [[ "$batch_mode" == tokenwise ]]; then
                options=(--maxseq 512 --no-prefill)
                batch=1
                ubatch=1
            fi
            args=("${options[@]}" "${backend_args[@]}")
            [[ "$order" != after ]] || args=("${backend_args[@]}" "${options[@]}")
            CPU_PARITY_ARGS_FIXTURE=1 CPU_PARITY_EXPECT_LAYERS="$layers" \
                CPU_PARITY_EXPECT_BATCH="$batch" CPU_PARITY_EXPECT_UBATCH="$ubatch" \
                BITNET="$0" LLAMA="$0" ./test/compare_llama.sh "$0" \
                --cuda --prompt fixture "${args[@]}" >/dev/null
        done
    done
done

echo "CPU parity harness tests PASSED"
