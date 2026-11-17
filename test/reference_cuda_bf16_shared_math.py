"""Pinned Torch sigmoid, BF16 shared product and routed add; fresh files only."""
from pathlib import Path
import argparse, json, hashlib
import torch

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
a.directory.mkdir()
i = torch.arange(65536, dtype=torch.int32, device='cuda')
logits = i.to(torch.int16).view(torch.bfloat16)
sigmoid = torch.sigmoid(logits)
dim = 5
j = torch.arange(65536 * dim, dtype=torch.int64, device='cuda')
down = (((j * 37 + 19) % 509 - 254).float() / 128).bfloat16().reshape(-1, dim)
routed = (((j * 23 + 71) % 251 - 125).float() / 64).bfloat16().reshape(-1, dim)
shared = sigmoid[:, None] * down
out = routed + shared
hn, hd, hs = 7, 37, 4
hi = torch.arange(hn * hs * hd, dtype=torch.int64, device='cuda')
normalized = torch.ldexp(torch.where(hi % 2 == 0, 1.0, -1.0), (hi % 12 + 110).int()).bfloat16().reshape(hn, hs, hd)
levels = torch.tensor([-87.5, -88, -88.5, 0, 1, float('inf'), -float('inf')], device='cuda').bfloat16()
hc_gate = levels[:, None, None].expand(hn, hs, hd).contiguous()
mixed = (normalized * torch.sigmoid(hc_gate)).mean(dim=1)
bi = torch.arange(hn * hd, dtype=torch.int64, device='cuda')
block = torch.ldexp(torch.where(bi % 2 == 0, 1.0, -1.0), (bi % 12 + 110).int()).bfloat16().reshape(hn, hd)
residual = torch.zeros_like(normalized)
inject = (levels[:, None].expand(hn, hs) * 4).contiguous()
combined = residual + block[:, None, :] * (2 * torch.sigmoid(inject / hs))[:, :, None]
files = {}
for name, x in [('logits', logits), ('sigmoid', sigmoid), ('down', down), ('routed', routed), ('shared', shared), ('out', out),
                ('hc-normalized', normalized), ('hc-gate', hc_gate), ('hc-mixed', mixed), ('hc-block', block),
                ('hc-residual', residual), ('hc-inject', inject), ('hc-combined', combined)]:
    raw = x.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    (a.directory / (name + '.bin')).write_bytes(raw)
    values = x.cpu().contiguous().view(torch.uint16).numpy().reshape(-1)
    h = 14695981039346656037
    for v in values:
        v = int(v)
        if v & 0x7f80 == 0x7f80 and v & 0x7f:
            v = 0x7fc0  # NaN payload is not part of the sigmoid contract.
        h = ((h ^ v) * 1099511628211) & ((1 << 64) - 1)
    files[name] = dict(shape=list(x.shape), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), canonical_nan_fnv64=f'{h:016x}')
manifest = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), dim=dim,
                profile='sigmoid BF16, product BF16, routed add BF16; NaN payload unspecified', files=files)
(a.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
