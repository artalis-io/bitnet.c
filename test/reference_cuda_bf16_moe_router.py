"""Pinned Flash Next inherited router, actual parameter loading and top-k.

Run in image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
with gguf-py. Arguments: fresh output directory, checkpoint inventory JSON.
Only single-rank parameter topology is mocked; load/linear/top-k are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk, _sorted_softmax_topk_kernel

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('inventory', type=Path)
a = p.parse_args()
a.directory.mkdir(exist_ok=False)
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
readers = [gguf.GGUFReader(s['path']) for s in json.loads(a.inventory.read_text())['shards']]
tensors = {t.name: t for r in readers for t in r.tensors}
files = {}
cases = []


def save(name, x):
    data = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    (a.directory / name).write_bytes(data)
    files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), dtype=str(x.dtype), shape=list(x.shape))


compiled = {}
original_run = _sorted_softmax_topk_kernel.run


def observe(*args, **kwargs):
    binary = original_run(*args, **kwargs)
    ptx = binary.asm['ptx']
    digest = hashlib.sha256(ptx.encode()).hexdigest()
    (a.directory / ('sorted-' + digest + '.ptx')).write_text(ptx)
    compiled[digest] = dict(n_regs=binary.n_regs, metadata=str(binary.metadata))
    return binary


_sorted_softmax_topk_kernel.run = observe
config = VllmConfig()
with set_current_vllm_config(config), torch.device('cuda'), patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank', return_value=0), patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size', return_value=1):
    for layer in [0, 2, 3, 47]:
        gate = ReplicatedLinear(2560, 512, bias=False, disable_tp=True)
        shared = ReplicatedLinear(2560, 1, bias=False, disable_tp=True)
        for module, name in [(gate, 'ffn_gate_inp.weight'), (shared, 'ffn_gate_inp_shexp.weight')]:
            t = tensors[f'blk.{layer}.' + name]
            assert t.tensor_type == gguf.GGMLQuantizationType.F32
            value = torch.from_numpy(np.array(t.data, copy=True)).reshape(module.weight.shape)
            module.weight_loader(module.weight, value)
            assert module.weight.dtype == torch.bfloat16
        save(f'layer{layer}-gate.bf16', gate.weight)
        save(f'layer{layer}-shared.bf16', shared.weight)
        for nt in [1, 3, 5, 17, 128, 129, 257]:
            name = f'layer{layer}-{nt}'
            i = torch.arange(nt * 2560, dtype=torch.int64)
            x = (((i * 17 + i // 37 + layer * 11) % 509 - 254).float() / 128).bfloat16().reshape(nt, 2560)
            with torch.inference_mode():
                logits = gate(x)[0]
                shared_logits = shared(x)[0]
                weights, ids, source = fused_topk(x, logits, 10, True)
            for stage, value in [('input', x), ('logits', logits), ('shared-logits', shared_logits),
                                 ('weights', weights), ('ids', ids), ('source', source)]:
                save(name + '-' + stage + '.bin', value)
            cases.append(dict(name=name, layer=layer, tokens=nt, renormalize=1, checkpoint=True))
    # Tie, underflow and nonfinite rows independently exercise probability ranking.
    for nt in [1, 17, 128, 129, 257]:
        for renormalize in [False, True]:
            name = f'edge-{nt}-{int(renormalize)}'
            i = torch.arange(nt * 512, dtype=torch.int64).reshape(nt, 512)
            logits = ((i * 7 % 31 - 15).float() / 8).bfloat16()
            logits[0] = 0
            if nt > 1:
                logits[1] = -1000
                logits[1, 400] = 1000
                logits[2] = float('nan')
                logits[3] = float('inf')
            weights, ids, source = fused_topk(torch.zeros(nt, 2560), logits, 10, renormalize)
            for stage, value in [('logits', logits), ('weights', weights), ('ids', ids), ('source', source)]:
                save(name + '-' + stage + '.bin', value)
            cases.append(dict(name=name, tokens=nt, renormalize=int(renormalize), checkpoint=False))
(a.directory / 'manifest.json').write_text(json.dumps(dict(parameter_dtype='torch.bfloat16', cases=cases, files=files, compiled=compiled), indent=2))
(a.directory / 'cases.txt').write_text(''.join(f"{c['name']} {c['tokens']} {c['renormalize']} {int(c['checkpoint'])} {c.get('layer', -1)}\n" for c in cases))
print('MOE_ROUTER_CAPTURED', len(cases), flush=True)
