"""Unchanged original GLM RMS/sigmoid output norm, with full F32 gamma."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--header", type=Path, required=True)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def digest(tensor):
    flat = torch.empty(tensor.numel(), dtype=tensor.dtype)
    flat.copy_(tensor.detach().cpu().reshape(-1))
    data = flat.view(torch.uint8).numpy().tobytes()
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value, data


cases = []
for rows in [1, 3, 15, 16, 17, 64, 130, 377, 1088, 6144, 65537]:
    for seed in [0, 1]:
        i = torch.arange(rows * 128)
        x = (((i * 37 + i // 13 + seed * 11) % 509 - 254).float() / 128)
        x = x.reshape(rows, 128)
        x *= torch.tensor([0., 2 ** -20, 1., 256.])[(torch.arange(rows) + seed) % 4, None]
        x = x.bfloat16().cuda()
        z = (((i * 13 + i // 17 + seed * 3) % 251 - 125).float() / 8)
        z = z.reshape(rows, 128)
        z *= torch.tensor([1., 8., .125, 4.])[(torch.arange(rows) + seed) % 4, None]
        z = z.bfloat16().cuda()
        i = torch.arange(128)
        w = (((i * 7 + seed) % 31 - 15).float() / 32 + 1 + 2 ** -17).cuda()
        eps = 1e-6 if seed == 0 else 1e-5
        # The original helper writes x in place: preserve the independent input.
        out = rms_norm_gated(x.clone(), z, w, None, activation="sigmoid", eps=eps)
        again = rms_norm_gated(x.clone(), z, w, None, activation="sigmoid", eps=eps)
        rounded = rms_norm_gated(x.clone(), z, w.bfloat16(), None,
                                 activation="sigmoid", eps=eps)
        assert torch.equal(out, again) and torch.isfinite(out).all()
        value, data = digest(out)
        (args.directory / f"norm-{rows}-{seed}.bin").write_bytes(data)
        cases.append(dict(rows=rows, seed=seed, eps=eps, hash=value,
                          rounded_weight_mismatches=int(torch.count_nonzero(out != rounded))))
        print(rows, seed, "captured and repeated", flush=True)
assert sum(c["rounded_weight_mismatches"] for c in cases) > 0
lines = ["/* Unchanged original GLM RMS/sigmoid norm with non-BF16 F32 gamma. */",
         "#ifndef BN_CUDA_GLM_OUTPUT_NORM_REFERENCE_H",
         "#define BN_CUDA_GLM_OUTPUT_NORM_REFERENCE_H",
         "typedef struct { int rows,seed; float eps; uint64_t hash; } BnCudaGLMOutputNormReference;",
         "static const BnCudaGLMOutputNormReference cuda_glm_output_norm_cases[] = {"]
for c in cases:
    lines.append(" {%d,%d,%.9ef,UINT64_C(0x%016x)}," %
                 (c["rows"], c["seed"], c["eps"], c["hash"]))
args.header.write_text("\n".join(lines + ["};", "#endif"]) + "\n")
source = Path(inspect.getfile(rms_norm_gated))
(args.directory / "reference.json").write_text(json.dumps(dict(
    cases=cases, torch=torch.__version__, source_sha256={
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()}), indent=2) + "\n")
