"""Independent pinned Torch GPU recipe used by vLLM RotaryEmbeddingBase.

Run in the pinned reference image with its standard NVIDIA entrypoint:
  python reference_cuda_bf16_rope.py --output cuda_bf16_rope_reference.h
No native engine output is used to generate expected values.
"""
import argparse
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
torch.set_default_device('cuda')
positions = [0, 1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 511, 512, 513,
             1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193,
             65535, 65536, 65537, 131071, 131072, 131073, 262142, 262143]
lines = ['#ifndef BN_CUDA_BF16_ROPE_REFERENCE_H',
         '#define BN_CUDA_BF16_ROPE_REFERENCE_H',
         '#include <stdint.h>',
         '/* Pinned Torch GPU F32 pow/reciprocal -> sin/cos -> BF16. */',
         'typedef struct {float theta; int position; uint16_t words[64];} BnBF16RopeReference;',
         'static const BnBF16RopeReference cuda_bf16_rope_cases[] = {']
for base in [10000.0, 1000000.0, 10000000.0]:
    inv_freq = 1.0 / (base ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))
    t = torch.tensor(positions, dtype=torch.float)
    freqs = torch.einsum('i,j -> ij', t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    rows = cache.cpu().view(torch.uint16).tolist()
    for position, row in zip(positions, rows):
        words = ','.join(f'0x{x:04x}' for x in row)
        lines.append(f' {{{base:.1f}f,{position},{{{words}}}}},')
lines += ['};', '#endif', '']
Path(args.output).write_text('\n'.join(lines))
print(f'Generated {len(rows)*3} cases, {len(rows)*3*64} reference words', flush=True)
