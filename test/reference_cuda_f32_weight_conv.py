"""Capture unchanged vLLM convolution with F32 weights and BF16 activations.

Run in the qualified local vLLM environment, on the designated CUDA device.
Expected values come exclusively from the original operator. Both full prefill
and repeated decode must agree before a fixture is emitted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update,
)

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--header", type=Path, required=True)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)


def save(name, tensor):
    flat = torch.empty(tensor.numel(), dtype=tensor.dtype)
    flat.copy_(tensor.detach().cpu().reshape(-1))
    data = flat.view(torch.uint8).numpy().tobytes()
    (args.directory / name).write_bytes(data)
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


cases = []
for dim, tokens in [(1, 1), (31, 5), (257, 17), (768, 65), (24576, 17)]:
    for kernel in [2, 3, 4]:
        for seed in [0, 1]:
            i = torch.arange(tokens * dim)
            x = (((i * 37 + i // 13 + seed * 11) % 509 - 254).float() / 128)
            x = x.to(torch.bfloat16).reshape(tokens, dim).cuda()
            i = torch.arange(dim * kernel)
            # The low bits must survive; these are deliberately not BF16 weights.
            w = (((i * 11 + i // 7 + seed * 3) % 127 - 63).float() / 128
                 + 2 ** -17).reshape(dim, kernel).cuda()
            i = torch.arange((kernel - 1) * dim)
            history = (((i * 19 + seed * 5) % 251 - 125).float() / 256)
            history = history.to(torch.bfloat16).reshape(kernel - 1, dim).cuda()
            cache = torch.stack([torch.zeros_like(history.T), history.T]).contiguous()
            initial = cache.clone()
            indices = torch.tensor([1], device="cuda", dtype=torch.int32)
            loc = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
            y = causal_conv1d_fn(
                x.T, w, None, cache, loc, cache_indices=indices,
                has_initial_state=torch.tensor([True], device="cuda"),
            ).T.contiguous()
            decode = initial.clone()
            outputs = []
            for t in range(tokens):
                outputs.append(causal_conv1d_update(
                    x[t:t + 1].clone(), decode, w, activation="silu",
                    conv_state_indices=indices,
                ).clone())
            assert torch.equal(torch.cat(outputs), y)
            assert torch.equal(decode, cache) and torch.isfinite(y).all()
            name = f"conv-{dim}-{tokens}-{kernel}-{seed}"
            cases.append(dict(dim=dim, tokens=tokens, kernel=kernel, seed=seed,
                              out=save(name + ".bin", y),
                              history=save(name + "-history.bin", cache[1].T)))
            print(name, "captured", flush=True)

lines = ["/* Independent unchanged vLLM F32-weight/BF16 causal convolution. */",
         "#ifndef BN_CUDA_F32_WEIGHT_CONV_REFERENCE_H",
         "#define BN_CUDA_F32_WEIGHT_CONV_REFERENCE_H",
         "typedef struct { int dim,tokens,kernel,seed; uint64_t out,history; } BnF32WeightConvReference;",
         "static const BnF32WeightConvReference cuda_f32_weight_conv_cases[] = {"]
for case in cases:
    lines.append(" {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)}," % tuple(
        case[key] for key in ["dim", "tokens", "kernel", "seed", "out", "history"]))
args.header.write_text("\n".join(lines + ["};", "#endif"]) + "\n")
manifest = dict(cases=cases, torch=torch.__version__,
                operator_sha256=hashlib.sha256(
                    Path(__import__(causal_conv1d_fn.__module__, fromlist=["__file__"]).__file__).read_bytes()
                ).hexdigest())
(args.directory / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
