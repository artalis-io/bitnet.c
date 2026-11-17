"""Pinned vLLM HC elementwise references, with projections isolated.

Normalization and linear maps supply deterministic BF16 activations; the
actual GatedResidual.mix/combine code runs unchanged. This is not a full
model or projection qualification. Use the image pinned in wide_mean.py.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
directory = parser.parse_args().directory
directory.mkdir(parents=True, exist_ok=True)


def pattern(count, seed, role):
    i = torch.arange(count, dtype=torch.int64)
    value = torch.ldexp(((i * (13 + role * 6) + seed * 11 + role * 19) % 509 - 254).float() / 128,
                        (i % 9 - 4).to(torch.int32))
    return value.to(torch.bfloat16).cuda()


def save(name, tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    path = directory / name
    if path.exists():
        assert path.read_bytes() == data, name
    else:
        path.write_bytes(data)
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value, hashlib.sha256(data).hexdigest()


cases = []
for streams in [2, 4, 8, 16]:
    for dim in [1, 3, 33, 127, 2560, 4096]:
        rank = 320 if dim >= 127 else 17
        for tokens in [1, 5, 17, 129]:
            seed = tokens + streams * 7 + dim
            full = tokens * streams * dim
            down = pattern(tokens * rank, seed, 0).reshape(tokens, rank)
            normed = pattern(full, seed, 1).reshape(tokens, streams * dim)
            gate = pattern(full, seed, 2).reshape_as(normed)
            residual = pattern(full, seed, 3).reshape_as(normed)
            block = pattern(tokens * dim, seed, 4).reshape(tokens, dim)
            inject = pattern(tokens * streams, seed, 5).reshape(tokens, streams)
            with torch.device("meta"):
                model = GatedResidual(HyperConnectionConfig(
                    hc_count=streams, hidden_size=dim, hc_lowrank=rank,
                    hc_per_branch_norm=True, params_dtype=torch.bfloat16))
            scaled = []

            def linear(x, weight, bias=None):
                assert bias is None
                if weight is model.input_mix_weight_down.weight:
                    assert x is normed
                    return down
                if weight is model.input_mix_weight_up.weight:
                    scaled.append(x)
                    return gate
                assert weight is model.block_inject_weight.weight and x is normed
                return inject

            with patch.object(model, "_normalize", return_value=normed), patch.object(F, "linear", side_effect=linear):
                mixed, saved = model.mix(residual)
                assert saved[0] is residual and saved[1] is normed
                combined = model.combine(block, saved)
            assert len(scaled) == 1
            key = f"{tokens}-{dim}-{streams}"
            hashes = {}
            for name, value in [("scaled", scaled[0]), ("unscaled", F.silu(down)), ("mixed", mixed), ("combined", combined)]:
                h, sha = save(key + "-" + name + ".bin", value)
                hashes[name] = h
                hashes[name + "_sha256"] = sha
            cases.append(dict(tokens=tokens, dim=dim, streams=streams, rank=rank, seed=seed, **hashes))
        print("captured HC", streams, dim, flush=True)

(directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
(directory / "provenance.json").write_text(json.dumps(dict(
    torch=torch.__version__, source_sha256=hashlib.sha256(inspect.getsource(GatedResidual).encode()).hexdigest(),
    isolation="Only normalization and linear projections replaced by synthetic BF16 inputs; actual elementwise module code unchanged."), indent=2) + "\n")
lines = ["#ifndef BN_CUDA_BF16_HC_REFERENCE_H", "#define BN_CUDA_BF16_HC_REFERENCE_H",
         "typedef struct {int tokens,dim,streams,rank,seed;uint64_t scaled,unscaled,mixed,combined;} BnBF16HCReference;",
         "static const BnBF16HCReference cuda_bf16_hc_cases[] = {"]
for case in cases:
    lines.append(" {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)}," %
                 tuple(case[k] for k in ["tokens", "dim", "streams", "rank", "seed", "scaled", "unscaled", "mixed", "combined"]))
lines += ["};", "#endif"]
(directory / "reference.h").write_text("\n".join(lines) + "\n")
