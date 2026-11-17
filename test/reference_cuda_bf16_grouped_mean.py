"""Independent per-stream affine normalization references for Flash Next.

Use the pinned image documented in reference_cuda_bf16_wide_mean.py.
Offset, unfused cases are additionally checked against vLLM's actual
GroupedGemmaRMSNorm module with separate weights for every HC stream.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm.models.qwen4_exp.common.hyperconnection import GroupedGemmaRMSNorm

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
directory = parser.parse_args().directory
directory.mkdir(parents=True, exist_ok=True)


def save(name, tensor):
    data = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
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
for dim in [2052, 2560, 4096, 10240]:
    for groups in [3, 4]:
        for tokens in [1, 2, 5, 17, 129]:
            rows = tokens * groups
            for offset in [0, 1]:
                seed = rows + offset * 17
                i = torch.arange(rows * dim, dtype=torch.int64)
                x = torch.ldexp(((i * 37 + seed * 11) % 509 - 254).float() / 128,
                                (i % 13 - 6).to(torch.int32))
                r = torch.ldexp(((i * 13 + i // 17 + seed * 3) % 251 - 125).float() / 64,
                                (i % 11 - 5).to(torch.int32))
                x = x.to(torch.bfloat16).reshape(tokens, groups, dim).cuda()
                r = r.to(torch.bfloat16).reshape(tokens, groups, dim).cuda()
                j = torch.arange(groups * dim, dtype=torch.int64)
                w = (((j * 7 + seed) % 127 - 63).float() / 64).to(torch.bfloat16).cuda()
                gamma = (w.float() + offset).reshape(groups, dim)
                for add in [0, 1]:
                    value = x.float() + r.float() if add else x.float()
                    mean = value.square().mean(dim=-1, keepdim=True)
                    y = (value * torch.rsqrt(mean + 1e-6) * gamma).to(torch.bfloat16)
                    if offset and not add:
                        norm = GroupedGemmaRMSNorm(groups * dim, 1e-6, dim, torch.bfloat16).cuda()
                        with torch.no_grad():
                            norm.weight.copy_(w)
                        expected = norm(x.reshape(tokens, groups * dim)).reshape_as(x)
                        assert torch.equal(expected, y), (dim, groups, tokens)
                    key = f"{dim}-{groups}-{tokens}-{offset}-{add}"
                    h, digest = save(key + ".bin", y)
                    rh, rdigest = save(key + "-res.bin", value.to(torch.bfloat16)) if add else (0, None)
                    cases.append(dict(dim=dim, tokens=rows, groups=groups, offset=offset,
                                      add=add, seed=seed, hash=h, residual_hash=rh,
                                      sha256=digest, residual_sha256=rdigest))
            print("captured grouped", dim, groups, tokens, flush=True)

(directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
lines = ["#ifndef BN_CUDA_BF16_GROUPED_MEAN_REFERENCE_H",
         "#define BN_CUDA_BF16_GROUPED_MEAN_REFERENCE_H",
         "typedef struct {int dim,tokens,groups,offset,add,seed;uint64_t hash,residual_hash;} BnBF16GroupedMeanReference;",
         "static const BnBF16GroupedMeanReference cuda_bf16_grouped_mean_cases[] = {"]
for case in cases:
    lines.append(" {%d,%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)}," %
                 tuple(case[k] for k in ["dim", "tokens", "groups", "offset", "add", "seed", "hash", "residual_hash"]))
lines += ["};", "#endif"]
(directory / "reference.h").write_text("\n".join(lines) + "\n")
