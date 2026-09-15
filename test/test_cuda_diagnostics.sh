#!/usr/bin/env bash
set -euo pipefail
binary=${1:-./test_cuda_backend}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$binary" --diagnostic-isolation >"$work/out" 2>"$work/err"
if grep -q 'skipped:' "$work/out"; then
    cat "$work/out"
    exit 0
fi
expect_count() {
    local pattern=$1 expected=$2 actual
    actual=$(grep -Ec "$pattern" "$work/err" || true)
    if [[ "$actual" != "$expected" ]]; then
        echo "CUDA diagnostic isolation: expected $expected matches for $pattern, got $actual" >&2
        cat "$work/err" >&2
        exit 1
    fi
}
expect_count '^\[bn:gpu:cuda:ops\] ' 2
for call in 1 2; do
    expect_count "^\\[bn:gpu:cuda:profile\\] calls=$call$" 2
    expect_count "^\\[bn:gpu:cuda:wall\\] calls=$call ops=$call " 2
    expect_count "type=0 rows=4 cols=4 ops=$call total_ms=" 2
done
echo 'CUDA diagnostic isolation PASSED'
