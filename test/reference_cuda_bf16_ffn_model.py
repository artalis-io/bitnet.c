"""Independent complete FFNs from actual Qwen27 GGUF checkpoint weights.

Run inside pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
with unchanged GGUF plugin e454affe0ca23c9dc8754ec3033270cb30622ed5.
Inputs are synthetic BF16 hidden rows; these references qualify complete FFNs,
not full-model execution. Existing reference files are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import vllm._custom_ops
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
p = args.directory
p.mkdir(parents=True, exist_ok=True)
tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def save(name, data):
    target = p / name
    if target.exists():
        assert target.read_bytes() == data, name
    else:
        target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


cases = []
for layer in [0, 3, 14]:
    weights = []
    geometry = []
    for role in ["gate", "up", "down"]:
        name = f"blk.{layer}.ffn_{role}.weight"
        t = tensors[name]
        cols, rows = map(int, t.shape)
        raw = np.array(t.data, copy=True).reshape(rows, -1)
        filename = f"layer{layer}-{role}-weight.bin"
        digest = save(filename, raw.tobytes())
        weights.append((torch.from_numpy(raw).cuda(), int(t.tensor_type)))
        geometry.append(dict(tensor=name, rows=rows, cols=cols,
                             type=int(t.tensor_type), file=filename, sha256=digest))
    dim, hidden = geometry[0]["cols"], geometry[0]["rows"]
    assert geometry[1]["cols"] == dim and geometry[1]["rows"] == hidden
    assert geometry[2]["cols"] == hidden and geometry[2]["rows"] == dim
    for nt in [1, 5, 17, 129]:
        i = torch.arange(nt * dim)
        x = (((i * 17 + i // 37 + layer * 11) % 509 - 254).float() / 8192)
        x = x.to(torch.bfloat16).reshape(nt, dim).cuda()
        g = _fused_mul_mat_gguf(x, weights[0][0], weights[0][1])
        u = _fused_mul_mat_gguf(x, weights[1][0], weights[1][1])
        mid = torch.empty_like(g)
        torch.ops._C.silu_and_mul(mid, torch.cat([g, u], -1))
        y = _fused_mul_mat_gguf(mid, weights[2][0], weights[2][1])
        assert torch.isfinite(y).all() and torch.isfinite(mid).all()
        prefix = f"layer{layer}-{nt}"
        xf, yf = prefix + "-input.bin", prefix + "-out.bin"
        xh, yh = save(xf, tensor_bytes(x)), save(yf, tensor_bytes(y))
        cases.append(dict(layer=layer, dim=dim, hidden=hidden, tokens=nt,
                          weights=geometry, input_file=xf, input_sha256=xh,
                          file=yf, sha256=yh))
        print(prefix, "captured", flush=True)
    del weights, g, u, mid, y, x

manifest = json.dumps(cases, indent=2) + "\n"
save("cases.json", manifest.encode())
