"""Audit pinned GGUF MoE kernels and expert-count corrections at IDs 255/256/511.

Identical raw weights at all 512 experts make equal outputs mandatory within
each execution path. This detects capacity errors independently of dot-product
arithmetic. Run with the pinned plugin wheel installed; writes fresh evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm_gguf_plugin import ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--expect-corrected', action='store_true')
a = p.parse_args()
a.directory.mkdir(exist_ok=False)
torch.set_default_dtype(torch.bfloat16)
w = torch.full((512, 32, 144), 17, dtype=torch.uint8, device='cuda')
w[:, :, :16] = 1
w[:, :, :4] = 0
w[:, :, 1] = 60  # FP16 d=1, dmin=0, nonzero quant values/scales.
results = []
for nt in [1, 64, 65, 129]:
    x = torch.ones(nt, 256, dtype=torch.bfloat16, device='cuda')
    for mode in ['vec', 'mmq']:
        outputs = {}
        for expert in [0, 255, 256, 511]:
            ids = torch.full((nt, 1), expert, dtype=torch.int32, device='cuda')
            if mode == 'vec':
                y = ops.ggml_moe_a8_vec(x, w, ids, 1, 12, 32, nt)
            else:
                block = ops.ggml_moe_get_block_size(12)
                sorted_ids, expert_ids, padded = moe_align_block_size(ids, block, 512)
                y = ops.ggml_moe_a8(x, w, sorted_ids, expert_ids, padded, 12, 32, 1, nt)
            data = y.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            name = f'{mode}-{nt}-{expert}.bin'
            (a.directory / name).write_bytes(data)
            outputs[expert] = data
            results.append(dict(mode=mode, tokens=nt, expert=expert, file=name,
                                nonzero=int(torch.count_nonzero(y).item()),
                                matches_expert0=data == outputs[0],
                                sha256=hashlib.sha256(data).hexdigest()))
assert all(r['nonzero'] > 0 and r['matches_expert0'] for r in results if r['mode'] == 'vec')
assert all(r['nonzero'] > 0 and r['matches_expert0'] for r in results if r['expert'] <= 255)
if a.expect_corrected:
    assert all(r['nonzero'] > 0 and r['matches_expert0'] for r in results)
# Construct valid sorted token slots, then poison only the expert ID table.
# Invalid experts must be rejected without reading outside the weight tensor.
x = torch.ones(65, 256, dtype=torch.bfloat16, device='cuda')
ids = torch.zeros((65, 1), dtype=torch.int32, device='cuda')
sorted_ids, expert_ids, padded = moe_align_block_size(ids, ops.ggml_moe_get_block_size(12), 512)
for invalid in [-1, 512]:
    y = ops.ggml_moe_a8(x, w, sorted_ids, torch.full_like(expert_ids, invalid), padded, 12, 32, 1, 65)
    assert not torch.count_nonzero(y)
(a.directory / 'audit.json').write_text(json.dumps(dict(cuda_enabled=ops._CUDA_ENABLED, results=results, invalid_ids_rejected=[-1, 512]), indent=2))
print('MOE_CAPACITY_AUDIT_CAPTURED', json.dumps(results), flush=True)
