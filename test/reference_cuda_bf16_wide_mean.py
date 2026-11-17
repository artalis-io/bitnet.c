"""Independent wide normalization fixtures from pinned vLLM/Torch.

Run in image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Preserves the existing 5120-wide reference files.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import torch
import vllm.ir as ir
import vllm._custom_ops
from vllm.config import VllmConfig, set_current_vllm_config
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
with set_current_vllm_config(VllmConfig()):
    for dim in [2048, 2052, 2560, 4096, 8192, 8196, 10240, 130560]:
        row_counts = [1, 2, 17] if dim == 130560 else [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 20, 31, 32, 33, 65, 129, 516]
        for rows in row_counts:
            for offset in [0, 1]:
                seed = rows + offset * 17
                i = torch.arange(rows * dim, dtype=torch.int64)
                x = torch.ldexp(((i * 37 + seed * 11) % 509 - 254).float() / 128,
                                (i % 13 - 6).to(torch.int32))
                r = torch.ldexp(((i * 13 + i // 17 + seed * 3) % 251 - 125).float() / 64,
                                (i % 11 - 5).to(torch.int32))
                x = x.to(torch.bfloat16).reshape(rows, dim).cuda()
                r = r.to(torch.bfloat16).reshape(rows, dim).cuda()
                j = torch.arange(dim, dtype=torch.int64)
                w = (((j * 7 + seed) % 127 - 63).float() / 64).to(torch.bfloat16).cuda()
                gamma = w.float() + offset
                for add in [0, 1]:
                    if add:
                        y, residual = ir.ops.fused_add_rms_norm(x, r, gamma, 1e-6)
                    else:
                        y, residual = ir.ops.rms_norm(x, gamma, 1e-6), None
                        if offset:
                            norm = GroupedGemmaRMSNorm(dim, 1e-6, None, torch.bfloat16).cuda()
                            with torch.no_grad():
                                norm.weight.copy_(w)
                            assert torch.equal(norm(x), y), (dim, rows, "Flash HC norm")
                    key = f"{dim}-{rows}-{offset}-{add}"
                    value, digest = save(key + ".bin", y)
                    rh, rsha = save(key + "-res.bin", residual) if add else (0, None)
                    cases.append(dict(dim=dim, tokens=rows, offset=offset, add=add,
                                      seed=seed, hash=value, residual_hash=rh,
                                      sha256=digest, residual_sha256=rsha))
            print("captured dim/rows", dim, rows, flush=True)

(directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
lines = ["#ifndef BN_CUDA_BF16_WIDE_MEAN_REFERENCE_H",
         "#define BN_CUDA_BF16_WIDE_MEAN_REFERENCE_H",
         "typedef struct {int dim,tokens,offset,add,seed;uint64_t hash,residual_hash;} BnBF16WideMeanReference;",
         "static const BnBF16WideMeanReference cuda_bf16_wide_mean_cases[] = {"]
for case in cases:
    lines.append(" {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)}," %
                 tuple(case[k] for k in ["dim", "tokens", "offset", "add", "seed", "hash", "residual_hash"]))
lines += ["};", "#endif"]
(directory / "reference.h").write_text("\n".join(lines) + "\n")
(directory / "identity.json").write_text(json.dumps(dict(
    torch=torch.__version__, vllm=metadata.version("vllm"), cuda=torch.version.cuda,
    device=str(torch.cuda.get_device_properties(0))), indent=2) + "\n")
