"""Independent paged FA2 + BF16 gate oracle for the request-owned CUDA path.

Pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Uses the unchanged reference FlashAttention interface, including its decode GQA
packing and automatic split-KV policy. Existing expected files are immutable.
Run with the image's standard NVIDIA entrypoint (which enables its CUDA
compatibility libraries): python this.py <output-directory>. No models required.
Do not override the entrypoint: doing so can select an older host PTX JIT.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
args = ap.parse_args()
p = args.directory
p.mkdir(parents=True, exist_ok=True)


def save(name, data):
    f = p / name
    if f.exists():
        assert f.read_bytes() == data, name
    else:
        f.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def blob(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def digest(t):
    h = 14695981039346656037
    for b in blob(t):
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h


def values(count, role, seed, start=0):
    i = torch.arange(start, start + count, dtype=torch.int64)
    if role == 0: x = ((i * 17 + seed * 11) % 509 - 254).float() / 128
    elif role == 1: x = ((i * 7 + seed * 3) % 251 - 125).float() / 64
    elif role == 2: x = ((i * 13 + seed * 5) % 257 - 128).float() / 32
    else:
        x = ((i * 19 + seed * 7) % 509 - 254).float() / 16
        special = [0., -0., -88., -89., -90., 88., 89., 90.]
        x[:min(len(x), len(special))] = torch.tensor(special[:min(len(x), len(special))])
    return x.to(torch.bfloat16).cuda()


# Chunks within each chain retain the SAME paged KV buffers. This covers empty,
# short, long, resumed/chunked prefill and subsequent dependent decode steps.
specs = [
    (24, 4, 1, [5, 1, 1]),
    (24, 4, 1, [129, 1, 17, 1]),
    (24, 4, 1, [129, 128, 1, 1]),
    (24, 4, 1, [512, 512, 23, 1]),
    (24, 4, 1, [1047, 1]),
    (24, 4, 1, [4095, 1, 1, 1]),
    (4, 4, 0, [17, 1]),
    (3, 1, 1, [33, 1]),
    (32, 4, 0, [129, 1]),
    (24, 8, 1, [65, 1]),
    (1, 1, 1, [1, 1]),
]
cases = []
for chain, (heads, kv_heads, gated, chunks) in enumerate(specs):
    seed = chain + 1
    capacity = sum(chunks) + 3
    page = 784
    pages = (capacity + page - 1) // page
    initial = torch.empty((pages, page, kv_heads, 256), dtype=torch.bfloat16)
    initial.view(torch.int16).fill_(0x5a5a)
    k_cache, v_cache = initial.cuda(), initial.cuda()
    table = torch.arange(pages, dtype=torch.int32).reshape(1, -1).cuda()
    position = 0
    for step, nt in enumerate(chunks):
        q = values(nt * heads * 256, 0, seed, position * heads * 256).reshape(nt, heads, 256)
        k = values(nt * kv_heads * 256, 1, seed, position * kv_heads * 256).reshape(nt, kv_heads, 256)
        v = values(nt * kv_heads * 256, 2, seed, position * kv_heads * 256).reshape(nt, kv_heads, 256)
        gate = values(nt * heads * 256, 3, seed, position * heads * 256).reshape_as(q)
        k_cache.view(-1, kv_heads, 256)[position:position+nt].copy_(k)
        v_cache.view(-1, kv_heads, 256)[position:position+nt].copy_(v)
        actual = position + nt
        cu_q = torch.tensor([0, nt], dtype=torch.int32, device="cuda")
        used_k = torch.tensor([actual], dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        out = flash_attn_varlen_func(q, k_cache, v_cache, nt, cu_q,
                                    max_seqlen_k=actual, seqused_k=used_k,
                                    block_table=table, softmax_scale=0.0625,
                                    causal=True, fa_version=2, num_splits=0)
        final = out * torch.sigmoid(gate) if gated else out
        assert torch.isfinite(out).all() and torch.isfinite(final).all()
        prefix = f"chain{chain}-{step}"
        oh, gh = digest(out), digest(final)
        sh = save(prefix + "-attention.bin", blob(out))
        fh = save(prefix + "-out.bin", blob(final))
        cases.append(dict(chain=chain, step=step, seed=seed, tokens=nt,
                          heads=heads, kv_heads=kv_heads, gated=gated,
                          position=position, capacity=capacity, attention_hash=oh,
                          hash=gh, attention_file=prefix+"-attention.bin",
                          file=prefix+"-out.bin", attention_sha256=sh, sha256=fh))
        position = actual
    print("Captured dependent chain", chain, chunks, flush=True)
save("cases.json", (json.dumps(cases, indent=2)+"\n").encode())
lines = ["/* Pinned vLLM FA2 paged cache + BF16 sigmoid gate, unchanged reference. */",
         "#ifndef BN_CUDA_BF16_ATTENTION_REFERENCE_H",
         "#define BN_CUDA_BF16_ATTENTION_REFERENCE_H",
         "typedef struct { int chain,seed,tokens,heads,kv_heads,gated,position,capacity; uint64_t attention_hash,hash; } BnBF16AttentionReference;",
         "static const BnBF16AttentionReference cuda_bf16_attention_cases[] = {"]
for c in cases:
    lines.append(" {%d,%d,%d,%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)}," %
                 tuple(c[k] for k in ["chain", "seed", "tokens", "heads", "kv_heads",
                                      "gated", "position", "capacity", "attention_hash", "hash"]))
lines += ["};", "#endif"]
save("reference.h", ("\n".join(lines)+"\n").encode())
