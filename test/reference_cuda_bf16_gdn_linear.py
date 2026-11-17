"""Replay Qwen27 BF16 GDN projections against independent model captures.

Run inside the pinned vLLM image/plugin documented in
reference_cuda_bf16_residual.py, with gguf-py available. Arguments: model GGUF,
capture directory (including manifest.json), output directory. No model weights
or reference captures are changed. This tests the reference topology and layout;
it does not qualify native BitNet BF16 linear kernels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('captures', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(args.model)
    weights = {t.name: t for t in reader.tensors}
    manifest = json.loads((args.captures / 'manifest.json').read_text())
    # Qwen27: 48 value heads / 16 key heads, 128 values per head.
    layout = GGUFHeadTilingLayout(3, 128)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def capture(module, phase, role):
        entry = next(c for c in manifest if c['module'] == module
                     and c['call'] == phase and c['role'] == role)
        assert entry['dtype'] == 'torch.bfloat16'
        data = np.fromfile(args.captures / Path(entry['file']).name, np.uint16)
        return torch.from_numpy(data).view(torch.bfloat16).reshape(entry['shape']).cuda()

    def weight(name):
        tensor = weights['blk.0.' + name + '.weight']
        data = np.array(tensor.data, copy=True).reshape(int(tensor.shape[1]), -1)
        value = torch.from_numpy(data).cuda()
        if name == 'attn_qkv':
            value = torch.cat([value[:4096], layout.weight_to_vllm(value[4096:], dim=0)])
        elif name == 'attn_gate':
            value = layout.weight_to_vllm(value, dim=0)
        elif name in ('ssm_beta', 'ssm_alpha'):
            value = layout.weight_to_vllm(value, dim=0, head_dim=1)
        return value, int(tensor.tensor_type)

    results = []
    for phase in (0, 1):
        for suffix, names in (
            ('in_proj_qkvz', ('attn_qkv', 'attn_gate')),
            ('in_proj_ba', ('ssm_beta', 'ssm_alpha')),
            ('out_proj', ('ssm_out',)),
        ):
            module = 'model.layers.0.linear_attn.' + suffix
            x = capture(module, phase, 'input')
            expected = capture(module, phase, 'output')
            if suffix == 'out_proj':
                x = layout.input_to_gguf(x)
            shards = [weight(name) for name in names]
            # Same-type merged modules dispatch once on the full row count.
            # Dispatching shards separately can change the MMVQ/MMQ threshold.
            assert len({typ for _, typ in shards}) == 1
            packed = torch.cat([w for w, _ in shards], dim=0)
            packed_data = packed.cpu().contiguous().numpy().tobytes()
            weight_file = f'{suffix}-weight.bin'
            (args.output / weight_file).write_bytes(packed_data)
            input_data = x.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            input_file = f'{suffix}-{phase}-input.bin'
            (args.output / input_file).write_bytes(input_data)
            y = _fused_mul_mat_gguf(x, packed, shards[0][1])
            exact = int((y == expected).sum())
            data = y.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            filename = f'{suffix}-{phase}.bin'
            (args.output / filename).write_bytes(data)
            result = dict(module=module, phase=phase, shape=list(y.shape),
                          type=shards[0][1], exact=exact, words=y.numel(),
                          file=filename, sha256=hashlib.sha256(data).hexdigest(),
                          weight_file=weight_file,
                          weight_sha256=hashlib.sha256(packed_data).hexdigest(),
                          packed_weight_shape=list(packed.shape),
                          input_file=input_file,
                          input_sha256=hashlib.sha256(input_data).hexdigest(),
                          input_shape=list(x.shape))
            results.append(result)
            print(result, flush=True)
    (args.output / 'capture-replay.json').write_text(json.dumps(results, indent=2))
    assert all(c['exact'] == c['words'] for c in results)
    print('Independent BF16 GDN linear replay PASSED:',
          sum(c['words'] for c in results), 'exact words', flush=True)


if __name__ == '__main__':
    main()
