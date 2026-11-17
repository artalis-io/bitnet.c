"""Audit expert-group placement invariance in the pinned GGUF MoE projection.

Every expert has identical raw weights. Reassigning a token to an identical
expert must preserve output bytes within the same vector/matrix path. Varying
input scale exposes incorrect per-column activation metadata indexing.
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
p.add_argument('--types', type=int, nargs='+', default=[7, 8, 12, 13])
p.add_argument('--expect-corrected', action='store_true')
a = p.parse_args()
a.directory.mkdir(exist_ok=False)
results = []
print('reference_extension_sha256', hashlib.sha256(Path(ops._C_gguf.__file__).read_bytes()).hexdigest(), flush=True)
for typ in a.types:
    cols, block, size = (640, 32, 24 if typ == 7 else 34) if typ in [7, 8] else (2560, 256, 144 if typ == 12 else 176)
    raw = torch.zeros(128, 32, cols // block, size, dtype=torch.uint8, device='cuda')
    raw[..., 1] = 60  # FP16 d=1.
    if typ == 7:
        raw[..., 8:] = 17
    elif typ == 8:
        raw[..., 2:] = 1
    else:
        raw[..., 4:16] = 1
        raw[..., 16 if typ == 12 else 48:] = 17
    w = raw.reshape(128, 32, -1)
    nt = 65
    scale = (2.0 ** (torch.arange(nt, device='cuda') % 5).float()) / 16
    x = scale[:, None].expand(nt, cols).to(torch.bfloat16).contiguous()
    for mode in ['vec', 'mmq']:
        outputs = []
        for assignment in ['one', 'spread']:
            ids = torch.zeros(nt, 1, dtype=torch.int32, device='cuda') if assignment == 'one' else torch.arange(nt, dtype=torch.int32, device='cuda').reshape(nt, 1)
            if mode == 'vec':
                y = ops.ggml_moe_a8_vec(x, w, ids, 1, typ, 32, nt)
            else:
                slots, experts, padded = moe_align_block_size(ids, ops.ggml_moe_get_block_size(typ), 128)
                y = ops.ggml_moe_a8(x, w, slots, experts, padded, typ, 32, 1, nt)
            data = y.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            name = f'{typ}-{mode}-{assignment}.bin'
            (a.directory / name).write_bytes(data)
            outputs.append(data)
            results.append(dict(type=typ, mode=mode, assignment=assignment, file=name, sha256=hashlib.sha256(data).hexdigest()))
        equal = outputs[0] == outputs[1]
        print(typ, mode, 'grouping_invariant', equal, flush=True)
        if mode == 'vec' or a.expect_corrected:
            assert equal, (typ, mode)
        results[-1]['grouping_invariant'] = equal
        (a.directory / 'audit.json').write_text(json.dumps(results, indent=2))
print('MOE_GROUPING_AUDIT_CAPTURED', len(results), flush=True)
