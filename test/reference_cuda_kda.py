"""Independent resident KDA fixtures from the unchanged original GLM operator.

Raw-log amplitudes use the original fused gate. Stored negative-exponential
amplitudes are preserved directly: eager F32 add/multiply prepares scaled gate
logits, then the original fused gate runs with A_log=0 and bias=0. This avoids
an inaccurate log/exp round trip and retains the same F32 operation boundaries.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--header", type=Path, required=True)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def fnv(values):
    result = 14695981039346656037
    for value in values:
        result = ((result ^ int(value)) * 1099511628211) & ((1 << 64) - 1)
    return result


cases = []
for tokens, heads in [(1, 1), (5, 2), (17, 64), (65, 2)]:
    for storage in [0, 1]:
        for seed in [0, 1]:
            i = torch.arange(tokens * heads * 384)
            x = (((i * 37 + i // 13 + seed * 11) % 509 - 254).float() / 128)
            x = x.reshape(tokens, 3 * heads * 128)
            for t in range(tokens):
                x[t] *= [0., 2 ** -20, 1., 256.][t % 4]
            x = x.bfloat16().cuda()
            i = torch.arange(tokens * heads * 128)
            g = (((i * 17 + seed * 7) % 509 - 254).float() / 32)
            g = g.reshape(1, tokens, heads, 128).bfloat16().cuda()
            i = torch.arange(tokens * heads)
            beta = (((i * 13 + seed * 11) % 251 - 125).float() / 16)
            beta = beta.reshape(1, tokens, heads).bfloat16().cuda()
            i = torch.arange(heads)
            a = (((i * 11 + seed * 5) % 29 - 14).float() / 8 + 2 ** -17)
            if storage:
                a = -(((i * 7 + seed) % 31 + 1).float() / 16 + 2 ** -19)
            a = a.cuda()
            i = torch.arange(heads * 128)
            bias = (((i * 7 + seed * 3) % 127 - 63).float() / 256 + 2 ** -19)
            bias = bias.reshape(heads, 128).cuda()
            i = torch.arange(heads * 128 * 128)
            initial = (((i * 17 + i // 37 + seed * 19) % 251 - 125).float()
                       / 4096 + 2 ** -27).reshape(1, heads, 128, 128).cuda()
            q, k, v = [part.reshape(1, tokens, heads, 128)
                       for part in x.split(heads * 128, dim=-1)]
            gate = g if not storage else (-a[None, None, :, None]) * (g.float() + bias)
            al = a if not storage else torch.zeros_like(a)
            gb = bias if not storage else torch.zeros_like(bias)
            options = dict(scale=float(np.float32(128 ** -.5)),
                           initial_state=initial, inplace_final_state=False,
                           use_qk_l2norm_in_kernel=True, sigmoid_beta=True,
                           a_log=al, g_bias=gb, compute_gate=True, lower_bound=-5.)
            out, states = fused_recurrent_kda(q, k, v, gate, beta, **options)
            again, again_states = fused_recurrent_kda(q, k, v, gate, beta, **options)
            assert torch.equal(out, again) and torch.equal(states, again_states)
            assert torch.isfinite(out).all() and torch.isfinite(states).all()
            raw = out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            state_words = states.cpu().numpy().view(np.uint32).reshape(tokens, -1)
            name = f"kda-{tokens}-{heads}-{storage}-{seed}"
            (args.directory / (name + "-out.bin")).write_bytes(raw)
            cases.append(dict(tokens=tokens, heads=heads, storage=storage, seed=seed,
                              out=fnv(raw), states=[fnv(row) for row in state_words]))
            print(name, "captured and repeated", flush=True)

lines = ["/* Unchanged original GLM fused KDA: BF16 core and every F32 prefix state. */",
         "#ifndef BN_CUDA_KDA_REFERENCE_H", "#define BN_CUDA_KDA_REFERENCE_H",
         "typedef struct { int tokens,heads,storage,seed; uint64_t out,states[65]; } BnCudaKDAReference;",
         "static const BnCudaKDAReference cuda_kda_cases[] = {"]
for c in cases:
    hs = ",".join(f"UINT64_C(0x{x:016x})" for x in c["states"])
    lines.append(" {%d,%d,%d,%d,UINT64_C(0x%016x),{%s}}," %
                 (c["tokens"], c["heads"], c["storage"], c["seed"], c["out"], hs))
args.header.write_text("\n".join(lines + ["};", "#endif"]) + "\n")
source = Path(inspect.getfile(fused_recurrent_kda))
manifest = dict(cases=cases, torch=torch.__version__, source_sha256={
    str(f): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [source, source.with_name("fused_recurrent.py")]})
(args.directory / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
