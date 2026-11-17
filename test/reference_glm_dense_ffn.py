"""Independent original GGUF linear and CUDA clamped-SwiGLU captures.

This qualifies dense FFNs, not a whole GLM model or performance parity.
"""
from pathlib import Path
import argparse, hashlib, inspect, json
import numpy as np
import torch
import gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm.model_executor.layers.activation import SiluAndMulWithClamp

ap = argparse.ArgumentParser()
ap.add_argument('--model', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
readers = [gguf.GGUFReader(str(a.model).replace('00001-of', f'{i:05d}-of')) for i in range(1, 7)]
table = {t.name: t for r in readers for t in r.tensors}
torch.set_num_threads(1)
records = []
for layer in range(3):
    out = a.out / f'layer{layer}'
    out.mkdir(exist_ok=True)
    weights, hashes = {}, {}
    for name in ['gate', 'up', 'down']:
        t = table[f'blk.{layer}.ffn_{name}.weight']
        assert int(t.tensor_type) == 8
        raw = np.array(t.data, copy=True)
        hashes[name] = hashlib.sha256(raw.tobytes()).hexdigest()
        cols = 12288 if name == 'down' else 4096
        weights[name] = torch.from_numpy(raw).reshape(-1, cols // 32 * 34).cuda()
    merged = torch.cat([weights['gate'], weights['up']], 0).contiguous()
    gen = torch.Generator().manual_seed(17789 + layer)
    x = torch.randn((65, 4096), generator=gen).mul_(4).bfloat16().cuda()
    def save(name, value):
        data = value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        (out / f'{name}.bin').write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    save('input', x)
    for ci, steps in enumerate([[9], [2, 7], [3, 3, 3], [7, 1, 1], [6, 3], [1]*9, [32], [33], [65]]):
        def run():
            values, off = [], 0
            for n in steps:
                gu = _fused_mul_mat_gguf(x[off:off+n].contiguous(), merged, 8)
                act = torch.empty((n, 12288), device='cuda', dtype=torch.bfloat16)
                # Exact original forward_cuda call, without unrelated config constructor.
                torch.ops._C.silu_and_mul_with_clamp(act, gu, 10., 1., 0.)
                result = _fused_mul_mat_gguf(act, weights['down'], 8)
                values.append(dict(gate_up=gu, activated=act, out=result))
                off += n
            torch.cuda.synchronize()
            return values
        first, repeat = run(), run()
        assert all(torch.equal(x[k], y[k]) for x, y in zip(first, repeat) for k in x)
        records.append(dict(layer=layer, case=ci, steps=steps, weights=hashes.copy(),
            hashes=[{k: save(f'{ci}-{phase}-{k}', v) for k, v in row.items()} for phase, row in enumerate(first)]))
        print(layer, ci, steps, 'captured twice', flush=True)
sources = [Path(inspect.getfile(f)) for f in [_fused_mul_mat_gguf, SiluAndMulWithClamp]]
(a.out / 'reference.json').write_text(json.dumps(dict(records=records, sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2)+'\n')
