#!/bin/sh
set -eu

fa_root=${1:-.deps/flash-attention}
fa_pin=506341a143fcabd4bb79052a7605ada727d6b3f5
cutlass_pin=62750a2b75c802660e4894434dc55e839f322277
if [ ! -d "$fa_root/.git" ]; then
    if [ -e "$fa_root" ] && [ -n "$(ls -A "$fa_root")" ]; then
        echo "CUDA attention dependency directory is not an empty checkout: $fa_root" >&2
        exit 1
    fi
    git init --quiet "$fa_root"
    git -C "$fa_root" remote add origin https://github.com/vllm-project/flash-attention.git
    git -C "$fa_root" fetch --depth 1 origin "$fa_pin"
    git -C "$fa_root" checkout --detach "$fa_pin"
fi
test "$(git -C "$fa_root" rev-parse HEAD)" = "$fa_pin"
git -C "$fa_root" diff --quiet HEAD --ignore-submodules
git -C "$fa_root" submodule update --init --depth 1 csrc/cutlass
test "$(git -C "$fa_root/csrc/cutlass" rev-parse HEAD)" = "$cutlass_pin"
git -C "$fa_root/csrc/cutlass" diff --quiet HEAD
printf '%s %s\n' "$fa_pin" "$cutlass_pin" > "$fa_root/.bn-fa2-ready"
