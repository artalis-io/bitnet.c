"""Selected-row BF16 embedding oracle, pinned vLLM image 745feb916ee8/plugin e454affe.

Raw synthetic weights come from reference_cuda_bf16_embedding.c. Only the
unchanged plugin embedding implementation generates expected output values.
Existing outputs cannot be overwritten with different bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding

assert ops._cuda_kernel_available("ggml_dequantize", 12), "Requires pinned CUDA HALF profile"

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
ap.add_argument("weights", type=Path)
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


cases = []
for vocab in [1, 7, 33]:
    for dim in [256, 512]:
        for seed in [0, 1]:
            raw = np.fromfile(args.weights / f"w-{vocab}-{dim}-{seed}.bin", np.uint8)
            w = torch.from_numpy(raw.reshape(vocab, -1)).cuda()
            tokens = [1, 3, 17, 129]
            if (vocab, dim, seed) == (1, 256, 0):
                tokens += [65537]
            for nt in tokens:
                i = np.arange(nt, dtype=np.int64)
                ids = ((i * 17 + i // 7 + seed * 3) % vocab).astype(np.int32)
                boundary = [0, vocab - 1, 0, 1 % vocab]
                ids[:min(nt, 4)] = boundary[:min(nt, 4)]
                y = _apply_gguf_embedding(torch.from_numpy(ids.astype(np.int64)).cuda(),
                                          w, 12, dim, torch.bfloat16)
                data = y.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                h = 14695981039346656037
                for b in data:
                    h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
                name = f"embedding-{vocab}-{dim}-{seed}-{nt}"
                sha = save(name + ".bin", data)
                cases.append(dict(vocab=vocab, dim=dim, seed=seed, tokens=nt,
                                  hash=h, sha256=sha, file=name + ".bin"))
save("cases.json", (json.dumps(cases, indent=2) + "\n").encode())
lines = ["/* Unchanged pinned GGUF plugin selected-row BF16 embedding. */",
         "#ifndef BN_CUDA_BF16_EMBEDDING_REFERENCE_H",
         "#define BN_CUDA_BF16_EMBEDDING_REFERENCE_H",
         "typedef struct { int vocab,dim,seed,tokens; uint64_t hash; } BnBF16EmbeddingReference;",
         "static const BnBF16EmbeddingReference cuda_bf16_embedding_cases[] = {"]
for c in cases:
    lines.append(" {%d,%d,%d,%d,UINT64_C(0x%016x)}," %
                 (c["vocab"], c["dim"], c["seed"], c["tokens"], c["hash"]))
lines += ["};", "#endif"]
save("reference.h", ("\n".join(lines) + "\n").encode())
print("Generated", len(cases), "embedding cases", flush=True)
