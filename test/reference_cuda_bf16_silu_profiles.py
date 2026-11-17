"""Independent original vLLM fused SiLU and eager Torch BF16 profiles."""
from pathlib import Path
import argparse, json, hashlib
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir()
i=torch.arange(65536,dtype=torch.int32,device='cuda')
gate=i.to(torch.int16).view(torch.bfloat16)[:,None].expand(-1,8).contiguous()
up=torch.tensor([-2,-1,-0.5,0,0.5,1,2,3.140625],device='cuda').bfloat16()[None,:].expand(65536,-1).contiguous()
with set_current_vllm_config(VllmConfig()):fused=SiluAndMul()(torch.cat([gate,up],dim=1))
two=torch.nn.functional.silu(gate)*up
scaled=torch.nn.functional.silu(gate/4)
files={}
for name,x in [('gate',gate),('up',up),('fused',fused),('two',two),('scaled',scaled)]:
 raw=x.cpu().contiguous().view(torch.uint8).numpy().tobytes();(a.directory/(name+'.bin')).write_bytes(raw)
 h=14695981039346656037
 for v in x.cpu().contiguous().view(torch.uint16).numpy().reshape(-1):
  v=int(v)
  if v&0x7f80==0x7f80 and v&127:v=0x7fc0
  h=((h^v)*1099511628211)&((1<<64)-1)
 files[name]=dict(bytes=len(raw),shape=list(x.shape),sha256=hashlib.sha256(raw).hexdigest(),canonical_nan_fnv64=f'{h:016x}')
(a.directory/'manifest.json').write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,files=files),indent=2)+'\n')
print('captured independent SiLU profiles')
