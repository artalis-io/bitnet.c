"""Independent selected-row BF16 embedding references from actual Qwen27.

Pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,
unchanged GGUF plugin e454affe0ca23c9dc8754ec3033270cb30622ed5. Uses the plugin's
embedding implementation, which selects raw rows before dequantization.
No native output supplies expected values. Existing files are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
p = args.directory
p.mkdir(parents=True, exist_ok=True)
reader = gguf.GGUFReader(args.model)
t = next(t for t in reader.tensors if t.name == "token_embd.weight")
dim, vocab = map(int, t.shape)
kind = int(t.tensor_type)
assert ops._cuda_kernel_available("ggml_dequantize", kind), "Requires pinned CUDA profile"
raw = np.array(t.data, copy=True).reshape(vocab, -1)
weight_sha = hashlib.sha256(raw.tobytes()).hexdigest()
weight = torch.from_numpy(raw).cuda()


def save(name, data):
    target = p / name
    if target.exists():
        assert target.read_bytes() == data, name
    else:
        target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


cases = []
for nt in [1, 5, 17, 129, 1047]:
    i = np.arange(nt, dtype=np.int64)
    ids = ((i * 104729 + i // 7) % vocab).astype(np.int32)
    boundary = [0, vocab - 1, 1, vocab - 2, 0, vocab // 2, vocab - 1]
    ids[:min(nt, len(boundary))] = boundary[:min(nt, len(boundary))]
    indices = torch.from_numpy(ids.astype(np.int64)).cuda()
    y = _apply_gguf_embedding(indices, weight, kind, dim, torch.bfloat16)
    assert y.shape == (nt, dim) and torch.isfinite(y).all()
    name = f"embedding-{nt}"
    index_file, output_file = name + "-ids.bin", name + "-out.bin"
    ih = save(index_file, ids.tobytes())
    yh = save(output_file, y.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    cases.append(dict(tensor=t.name, type=kind, dim=dim, vocab=vocab, tokens=nt,
                      weight_sha256=weight_sha, index_dtype="int32",
                      index_file=index_file, index_sha256=ih,
                      file=output_file, sha256=yh))
    print(name, "captured", flush=True)
save("cases.json", (json.dumps(cases, indent=2) + "\n").encode())
