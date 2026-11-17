"""Pinned QSA pre-indexer with actual layer-three Flash GGUF weights.
Explicit decoded-BF16, sequential text, BF16 side cache, no speculative tokens.
No expected neural intermediates are injected. Existing captures are immutable.
"""
import argparse, hashlib, importlib, inspect, json
from pathlib import Path
import gguf
import numpy as np
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.config import VllmConfig, set_current_vllm_config
ops = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer')
p = argparse.ArgumentParser(); p.add_argument('directory', type=Path); p.add_argument('inventory', type=Path)
a = p.parse_args(); a.directory.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
m = json.loads(a.inventory.read_text()); readers = [gguf.GGUFReader(x['path']) for x in m['shards']]
ts = {t.name: t for r in readers for t in r.tensors}; identities = {}
def save(name, x):
    data = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    f = a.directory / name
    if f.exists(): assert f.read_bytes() == data, name
    else: f.write_bytes(data)
    return dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), shape=list(x.shape), dtype=str(x.dtype))
def weight(suffix):
    name = 'blk.3.indexer.' + suffix + '.weight'; t = ts[name]; raw = np.asarray(t.data)
    offset = 1.0 if suffix in ('q_norm', 'k_norm') else 0.0
    v = (torch.from_numpy(np.array(gguf.dequantize(raw, t.tensor_type), copy=True)).float() - offset).bfloat16()
    identities[name] = dict(gemma_storage_offset=offset, type=t.tensor_type.name, raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), decoded=save(suffix+'.bf16', v))
    return v.cuda()
wq, wk, nq, nk = [weight(x) for x in ['q_proj', 'k_proj', 'q_norm', 'k_norm']]
w = torch.cat([wq, wk]); nq=nq.reshape(128); nk=nk.reshape(128)
with set_current_vllm_config(VllmConfig()), torch.device('cuda'):
    rope = get_rope(head_size=256, max_position=8192, rope_parameters=dict(rope_type='default', rope_theta=10000000., partial_rotary_factor=.25), dtype=torch.bfloat16)
cs = rope.cos_sin_cache.cuda(); assert cs.shape == (8192,64); save('cos-sin.bf16',cs)
original = ops._qsa_pre_indexer_kernel.run
def observed(*args, **kwargs):
    compiled = original(*args, **kwargs)
    text = compiled.asm['ptx']; key=hashlib.sha256(text.encode()).hexdigest()
    (a.directory/('kernel-'+key+'.ptx')).write_text(text)
    return compiled
ops._qsa_pre_indexer_kernel.run = observed
cases=[]
for nt, chunk in [(1,1),(3,3),(4,4),(5,5),(17,7),(129,129),(129,64),(129,7),(129,1),(2053,2053),(2053,257),(4097,4097)]:
    name=f'{nt}-{chunk}'; ring=torch.zeros(2,4,128,device='cuda',dtype=torch.bfloat16)
    comp=torch.zeros(1+(nt+8+63)//64,16,128,device='cuda',dtype=torch.bfloat16)
    phases=[]; steps=[(pos,min(chunk,nt-pos)) for pos in range(0,nt,chunk)]+[(nt+i,1) for i in range(8)]
    for phase,(pos,take) in enumerate(steps):
        idx=torch.arange(pos*2560,(pos+take)*2560,dtype=torch.int64)
        x=(((idx*17+idx//37)%509-254).float()/128).reshape(take,2560).bfloat16().cuda()
        merged=F.linear(x,w); q=merged[:,:512]; k=merged[:,512:]
        logical=torch.arange(pos,pos+take,device='cuda',dtype=torch.int64)
        slots=(4+logical%4).long(); comp_slots=torch.where((logical+1)%4==0,16+logical//4,-1).long()
        groups=(pos+take)//4-pos//4
        work=torch.tensor([[0,i] for i in range(max(groups,1))],device='cuda',dtype=torch.int32)
        out=torch.empty(take,4,128,device='cuda',dtype=torch.bfloat16)
        ops.qsa_pre_indexer(q,k,logical,cs,nq,nk,1e-6,out,ring,slots,
            torch.tensor([[1]],device='cuda',dtype=torch.int32),torch.tensor([0,take],device='cuda',dtype=torch.int32),logical,
            comp,comp_slots,work,compress_ratio=4,mrope_section=(11,11,10),rope_pos_offset=None)
        visible=(pos+take)//4
        stages={s:save(f'{name}-{phase}-{s}.bin',v) for s,v in [('input',x),('projected',merged),('q',out),('history',ring[1]),('compressed',comp.reshape(-1,128)[16:16+visible])]}
        phases.append(dict(position=pos,take=take,phase=phase,visible=visible,stages=stages)); print(name,phase,pos,take,visible,flush=True)
    cases.append(dict(name=name,tokens=nt,chunk=chunk,phases=phases))
manifest=dict(rope_device=str(cs.device), profile='actual layer3 merged BF16 index QK, fused pinned pre-indexer, ratio4, sequential text, BF16 cache, no speculation',weights=identities,source_sha256=hashlib.sha256(inspect.getsource(ops).encode()).hexdigest(),cases=cases)
f=a.directory/'manifest.json'; data=(json.dumps(manifest,indent=2)+'\n').encode()
if f.exists(): assert f.read_bytes()==data
else: f.write_bytes(data)
print('QSA_INDEXER_CAPTURED',len(cases),flush=True)
