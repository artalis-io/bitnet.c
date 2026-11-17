"""Independent GGUF Q5_1 -> FP32 -> BF16 layout and dense GEMM oracle.

Run in pinned image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
with gguf-py on PYTHONPATH. Writes fresh captures; never replaces an oracle.
Arguments: output-directory Flash-Next-inventory-json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('inventory', type=Path)
a = p.parse_args()
a.directory.mkdir(exist_ok=False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
manifest = []


def save(name, data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    (a.directory / name).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def capture(name, raw, rows, cols):
    raw = np.asarray(raw, dtype=np.uint8).reshape(rows, cols // 32 * 24)
    decoded = torch.from_numpy(gguf.dequantize(raw, gguf.GGMLQuantizationType.Q5_1).copy()).bfloat16()
    assert decoded.shape == (rows, cols)
    item = dict(name=name, rows=rows, cols=cols, files={})
    item['files']['raw'] = save(name + '-raw.bin', raw.tobytes())
    item['files']['decoded'] = save(name + '-decoded.bin', decoded)
    w = decoded.cuda()
    for nt in [1, 5, 17, 129]:
        i = torch.arange(nt * cols, dtype=torch.int64)
        x = (((i * 17 + i // 37) % 509 - 254).float() / 128).bfloat16().reshape(nt, cols)
        y = x.cuda() @ w.T
        item['files'][f'x-{nt}'] = save(f'{name}-x-{nt}.bin', x)
        item['files'][f'y-{nt}'] = save(f'{name}-y-{nt}.bin', y)
    manifest.append(item)


for rows, cols in [(1, 32), (33, 640), (129, 672)]:
    nb = rows * cols // 32
    j = np.arange(nb, dtype=np.uint64)
    raw = np.empty((nb, 24), dtype=np.uint8)
    # Finite signed FP16 scales, minima, and all high/low bits.
    for offset, bits in [(0, ((j * 61 + 0x1800) % 0x7800) | ((j & 1) << 15)),
                         (2, ((j * 97 + 0x0400) % 0x7800) | (((j >> 1) & 1) << 15))]:
        raw[:, offset:offset + 2] = bits.astype('<u2').view(np.uint8).reshape(nb, 2)
    for k in range(4, 24):
        raw[:, k] = ((j * 73 + k * 37 + j // 11) % 256).astype(np.uint8)
    capture(f'synthetic-{rows}-{cols}', raw, rows, cols)

inventory = json.loads(a.inventory.read_text())
readers = [gguf.GGUFReader(s['path']) for s in inventory['shards']]
t = next(t for r in readers for t in r.tensors if t.name == 'blk.3.ffn_down_exps.weight')
assert t.tensor_type == gguf.GGMLQuantizationType.Q5_1
raw = np.asarray(t.data).reshape(512, 2560, 640 // 32 * 24)
for expert in [0, 511]:
    capture(f'checkpoint-layer3-expert{expert}', raw[expert], 2560, 640)
(a.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(a.directory / 'cases.txt').write_text(''.join(f"{c['name']} {c['rows']} {c['cols']}\n" for c in manifest))
print('Q51_LAYOUT_REFERENCE_CAPTURED', len(manifest), flush=True)
