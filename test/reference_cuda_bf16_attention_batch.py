"""Independent unchanged paged FA2 batch-split oracle; no model required.

Run in the pinned vLLM image on the qualified GPU. This writes a new fixture,
never replaces existing singleton references, and never runs in unit tests.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

ap = argparse.ArgumentParser()
ap.add_argument('directory', type=Path)
p = ap.parse_args().directory
p.mkdir(parents=True, exist_ok=True)
assert torch.cuda.device_count() == 1


def raw(t):
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h


def values(count, role, seed, start=0):
    i = torch.arange(start, start + count, dtype=torch.int64)
    if role == 0:
        x = ((i * 17 + seed * 11) % 509 - 254).float() / 128
    elif role == 1:
        x = ((i * 7 + seed * 3) % 251 - 125).float() / 64
    else:
        x = ((i * 13 + seed * 5) % 257 - 128).float() / 32
    return x.to(torch.bfloat16).cuda()


specs = [[2049], [2049] * 2, [129] * 4, [1025] * 4,
         [2049] * 4, [4097] * 4, [65, 257, 1025, 2049],
         [65, 129, 257, 513, 1025, 2049, 4097, 2113]]
rows = []
for case, lengths in enumerate(specs):
    batch = len(lengths)
    seed = 3 + case
    heads, kv_heads, dim, page = 24, 4, 256, 256
    maximum = max(lengths)
    pages = (maximum + page - 1) // page
    k = torch.empty((batch * pages, page, kv_heads, dim), dtype=torch.bfloat16, device='cuda')
    v = torch.empty_like(k)
    k.view(torch.int16).fill_(0x5a5a)
    v.view(torch.int16).fill_(0x5a5a)
    q = torch.empty((batch, heads, dim), dtype=torch.bfloat16, device='cuda')
    table = torch.arange(batch * pages, dtype=torch.int32, device='cuda').reshape(batch, pages)
    for row, length in enumerate(lengths):
        rs = seed + row * 7
        q[row].copy_(values(heads * dim, 0, rs, (length - 1) * heads * dim).reshape(heads, dim))
        k[row * pages:(row + 1) * pages].view(-1, kv_heads, dim)[:length].copy_(values(length * kv_heads * dim, 1, rs).reshape(length, kv_heads, dim))
        v[row * pages:(row + 1) * pages].view(-1, kv_heads, dim)[:length].copy_(values(length * kv_heads * dim, 2, rs).reshape(length, kv_heads, dim))
    cu = torch.arange(batch + 1, dtype=torch.int32, device='cuda')
    used = torch.tensor(lengths, dtype=torch.int32, device='cuda')

    def execute():
        return flash_attn_varlen_func(q, k, v, 1, cu, max_seqlen_k=maximum,
                                     seqused_k=used, block_table=table,
                                     softmax_scale=0.0625, causal=True,
                                     fa_version=2, num_splits=0)

    out = execute()
    data = raw(out)
    assert data == raw(execute()) and torch.isfinite(out).all()
    path = p / f'case-{case}.bin'
    if path.exists():
        assert path.read_bytes() == data
    else:
        path.write_bytes(data)
    row_bytes = heads * dim * 2
    hashes = [fnv(data[i * row_bytes:(i + 1) * row_bytes]) for i in range(batch)]
    rows.append(dict(batch=batch, seed=seed, lengths=lengths, hashes=hashes,
                     sha256=hashlib.sha256(data).hexdigest()))
    (p / 'reference.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(case, batch, lengths, flush=True)
header = ['/* Independent pinned vLLM FA2 batch automatic split-KV hashes. */',
          '#include <stdint.h>',
          'typedef struct { int batch,seed,lengths[8]; uint64_t hashes[8]; } BnBF16BatchAttentionReference;',
          'static const BnBF16BatchAttentionReference bf16_batch_attention_cases[] = {']
for row in rows:
    header.append('{%d,%d,{%s},{%s}},' % (row['batch'], row['seed'],
                  ','.join(map(str, row['lengths'])),
                  ','.join('UINT64_C(0x%016x)' % h for h in row['hashes'])))
header.append('};')
(p / 'reference.h').write_text('\n'.join(header) + '\n')
props = torch.cuda.get_device_properties(0)
(p / 'provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    image='sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',
    device=props.name, multiprocessors=props.multi_processor_count,
    torch=torch.__version__, cuda=torch.version.cuda,
    source='Unchanged public flash_attn_varlen_func,fa_version=2,num_splits=0'), indent=2) + '\n')
