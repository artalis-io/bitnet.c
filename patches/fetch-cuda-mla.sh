#!/bin/sh
set -eu
mla_root=${1:-.deps/flashinfer}
mla_pin=beabf5004585b03da60e927bbb803acbaa62fe36
cutlass_pin=b46b16d003484063bca4ed365e44095c4c6ed633
cccl_pin=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
if [ ! -d "$mla_root/.git" ]; then
    if [ -e "$mla_root" ] && [ -n "$(ls -A "$mla_root")" ]; then
        echo "CUDA MLA dependency directory is not an empty checkout: $mla_root" >&2
        exit 1
    fi
    git init --quiet "$mla_root"
    git -C "$mla_root" remote add origin https://github.com/flashinfer-ai/flashinfer.git
    git -C "$mla_root" fetch --depth 1 origin "$mla_pin"
    git -C "$mla_root" checkout --detach "$mla_pin"
fi
test "$(git -C "$mla_root" rev-parse HEAD)" = "$mla_pin"
git -C "$mla_root" diff --quiet HEAD --ignore-submodules
git -C "$mla_root" submodule update --init --depth 1 3rdparty/cutlass 3rdparty/cccl
test "$(git -C "$mla_root/3rdparty/cutlass" rev-parse HEAD)" = "$cutlass_pin"
test "$(git -C "$mla_root/3rdparty/cccl" rev-parse HEAD)" = "$cccl_pin"
git -C "$mla_root/3rdparty/cutlass" diff --quiet HEAD
git -C "$mla_root/3rdparty/cccl" diff --quiet HEAD
printf '%s %s %s\n' "$mla_pin" "$cutlass_pin" "$cccl_pin" > "$mla_root/.bn-mla-ready"
