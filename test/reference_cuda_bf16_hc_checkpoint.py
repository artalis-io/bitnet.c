"""Capture complete common HC blocks using actual Flash Next GGUF tensors.

Run in image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
with llama.cpp gguf-py on PYTHONPATH. This is an explicit decoded-BF16
checkpoint profile, not a claim that the GGUF plugin supports this model.
No normalization, projection or elementwise result is replaced. Hooks only
observe intermediate results. Existing captures must remain byte-identical.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common.hyperconnection import (
    GatedResidual, GroupedGemmaRMSNorm, HyperConnectionConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("shards", type=Path)
    parser.add_argument("--output-mixer", action="store_true",
                        help="Capture the final HC mixer without injection/combine")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    readers = [gguf.GGUFReader(str(path)) for path in sorted(args.shards.glob("*.gguf"))]
    tensors = {}
    for reader in readers:
        for tensor in reader.tensors:
            assert tensor.name not in tensors, tensor.name
            tensors[tensor.name] = tensor
    assert tensors, "No GGUF tensors"
    identities, cases = {}, []

    def save(name, value):
        data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        path = args.directory / name
        if path.exists():
            assert path.read_bytes() == data, name
        else:
            path.write_bytes(data)
        return dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    shape=list(value.shape), dtype=str(value.dtype))

    def weight(name):
        tensor = tensors[name]
        raw = np.asarray(tensor.data)
        value = gguf.dequantize(raw, tensor.tensor_type)
        offset = 1.0 if name.endswith("norm.weight") else 0.0
        decoded = (torch.from_numpy(np.array(value, copy=True)).float() - offset).to(torch.bfloat16)
        identities[name] = dict(gemma_storage_offset=offset, type=tensor.tensor_type.name,
            shape=list(map(int, tensor.shape)), raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            decoded=save(name + ".bf16", decoded))
        return decoded.cuda()

    original_linear = F.linear
    with torch.inference_mode():
        for layer in ([-1] if args.output_mixer else [0, 2, 47]):
            for role in (["output"] if args.output_mixer else ["attn", "ffn"]):
                prefix = "output_hc_" if args.output_mixer else f"blk.{layer}.hc_{role}_"
                model = GatedResidual(HyperConnectionConfig(hc_count=4,
                    hidden_size=2560, hc_lowrank=320, hc_per_branch_norm=True,
                    params_dtype=torch.bfloat16), use_combine=not args.output_mixer).cuda()
                parameters = [
                    ("norm", model.hc_norm.weight),
                    ("down", model.input_mix_weight_down.weight),
                    ("up", model.input_mix_weight_up.weight),
                ]
                if not args.output_mixer:
                    parameters.append(("inject", model.block_inject_weight.weight))
                for suffix, parameter in parameters:
                    decoded = weight(prefix + suffix + ".weight")
                    assert parameter.shape == decoded.shape, (suffix, parameter.shape, decoded.shape)
                    parameter.copy_(decoded)
                for tokens in [1, 5, 17, 129]:
                    key = f"{layer}-{role}-{tokens}"
                    i = torch.arange(tokens * 10240, dtype=torch.int64)
                    x = (((i * 37 + layer * 11) % 509 - 254).float() / 128).to(
                        torch.bfloat16).reshape(tokens, 10240).cuda()
                    j = torch.arange(tokens * 2560, dtype=torch.int64)
                    block = (((j * 13 + layer * 7) % 251 - 125).float() / 64).to(
                        torch.bfloat16).reshape(tokens, 2560).cuda()
                    stages = dict(input=save(key + "-input.bin", x),
                                  block=save(key + "-block.bin", block))

                    def observe_linear(value, parameter, bias=None):
                        result = original_linear(value, parameter, bias)
                        names = [(model.input_mix_weight_down.weight, "down"),
                                 (model.input_mix_weight_up.weight, "up")]
                        if not args.output_mixer:
                            names.append((model.block_inject_weight.weight, "inject"))
                        name = next(name for candidate, name in names if candidate is parameter)
                        stages[name] = save(key + "-" + name + ".bin", result)
                        if name == "up":
                            stages["scaled"] = save(key + "-scaled.bin", value)
                        return result

                    def observe_norm(module, inputs, output):
                        stages["norm"] = save(key + "-norm.bin", output)

                    hook = model.hc_norm.register_forward_hook(observe_norm)
                    try:
                        with patch.object(F, "linear", side_effect=observe_linear):
                            mixed, residuals = model.mix(x)
                            combined = None if args.output_mixer else model.combine(block, residuals)
                    finally:
                        hook.remove()
                    stages["mixed"] = save(key + "-mixed.bin", mixed)
                    if combined is not None:
                        stages["combined"] = save(key + "-combined.bin", combined)
                    cases.append(dict(layer=layer, role=role, tokens=tokens, stages=stages))
                    print("captured", key, flush=True)
    provenance = dict(profile="common HC, GGUF CPU dequantize -> BF16 parameters, F32 GEMM accumulation",
        scope="Complete isolated HC blocks; synthetic hidden states, actual checkpoint weights; no whole-model claim",
        output_mixer=args.output_mixer,
        torch=torch.__version__, cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),
        source_sha256=hashlib.sha256((inspect.getsource(GatedResidual) +
            inspect.getsource(GroupedGemmaRMSNorm)).encode()).hexdigest(),
        weights=identities, cases=cases)
    (args.directory / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Plain records consumed by test_cuda_hc_checkpoint: weight/input/output
    # paths followed by output width, input width and token count.
    plan = []
    for case in cases:
        for operation, input_stage in ([("down", "norm"), ("up", "scaled")] +
                                       ([] if args.output_mixer else [("inject", "norm")])):
            name = ("output_hc_" if args.output_mixer else
                    f"blk.{case['layer']}.hc_{case['role']}_") + operation + ".weight"
            weight_info = identities[name]["decoded"]
            x = case["stages"][input_stage]
            y = case["stages"][operation]
            plan.append(f"{weight_info['file']} {x['file']} {y['file']} "
                        f"{y['shape'][1]} {x['shape'][1]} {case['tokens']}")
    (args.directory / "plan.txt").write_text("\n".join(plan) + "\n")


if __name__ == "__main__":
    main()
