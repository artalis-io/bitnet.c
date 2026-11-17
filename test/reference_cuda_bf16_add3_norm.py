"""Independent eager Torch reference for retained F32 three-term residuals.

Run in pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
The normalization uses the existing contiguous eager F32 reduction contract,
not an arbitrary compiled/autotuned reduction. No BitNet code supplies outputs.
Existing reference files are compared and never overwritten.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--header', type=Path)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
assert torch.cuda.device_count() == 1

def save(path, data):
    if path.exists():
        assert path.read_bytes() == data, path
    else:
        path.write_bytes(data)

def capture(name, value):
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    save(args.directory / (name + '.bin'), raw)
    result = 14695981039346656037
    for byte in raw:
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result

cases = []
shapes = [(dim, nt, 1) for dim in [2048, 2560, 5120, 8192]
          for nt in [1, 5, 17, 129]] + [(5120, 18, 3), (5120, 128, 4), (130560, 1, 1)]
for dim, nt, groups in shapes:
    for seed in [0, 1]:
        i = torch.arange(nt * dim, dtype=torch.int64)
        x = (((i * 37 + i // 13 + seed) % 509 - 254).float() / 64).bfloat16()
        r = (((i * 23 + i // 17 + seed * 7) % 503 - 251).float() / 128).bfloat16()
        z = (((i * 11 + i // 19 + seed * 3) % 499 - 249).float() / 4096).bfloat16()
        if seed:
            mask = i % 16 == 0
            x[mask] = 2 ** 24
            r[mask] = -(2 ** 24)
            z[mask] = ((i[mask] % 31 + 1).float() / 256).bfloat16()
        x, r, z = [v.reshape(nt, dim).cuda() for v in [x, r, z]]
        j = torch.arange(groups * dim, dtype=torch.int64)
        offset = seed
        w = ((-1 if offset else .25) + ((j * 7 + seed) % 31 + 1).float() / 64).bfloat16().reshape(groups, dim).cuda()
        # Eager materialization is intentional: no FP32 square/sum contraction
        # or BF16 store between the first and second residual additions.
        summed = (x.float() + r.float()) + z.float()
        mean = summed.square().mean(-1, keepdim=True)
        gamma = (w.float() + offset)[torch.arange(nt, device='cuda') % groups]
        norm = ((summed * torch.rsqrt(mean + 1e-6)) * gamma).bfloat16()
        residual = summed.bfloat16()
        assert torch.isfinite(norm).all() and torch.isfinite(residual).all()
        name = f'add3-{dim}-{nt}-{groups}-{seed}'
        for role, value in [('x', x), ('r', r), ('z', z), ('weight', w)]:
            capture(name + '-' + role, value)
        record = dict(dim=dim, tokens=nt, groups=groups, seed=seed, offset=offset,
                      norm=capture(name + '-norm', norm), residual=capture(name + '-residual', residual))
        cases.append(record)
        print(name, 'captured', flush=True)
save(args.directory / 'cases.json', (json.dumps(cases, indent=2) + '\n').encode())
if args.header:
    rows = ['/* Independent eager Torch F32 three-term sum/norm; see reference_cuda_bf16_add3_norm.py. */',
            '#ifndef BN_CUDA_BF16_ADD3_NORM_REFERENCE_H', '#define BN_CUDA_BF16_ADD3_NORM_REFERENCE_H',
            'typedef struct { int dim,tokens,groups,seed,offset; uint64_t norm,residual; } BnBF16Add3NormReference;',
            'static const BnBF16Add3NormReference cuda_bf16_add3_norm_cases[] = {']
    for c in cases:
        rows.append(' {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},' %
                    (c['dim'], c['tokens'], c['groups'], c['seed'], c['offset'], c['norm'], c['residual']))
    save(args.header, ('\n'.join(rows + ['};', '#endif']) + '\n').encode())
