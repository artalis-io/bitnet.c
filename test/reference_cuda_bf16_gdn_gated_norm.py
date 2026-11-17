"""Independent vLLM RMSNormGated (norm-before-gate) BF16 references.
Pinned image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
No native implementation supplies expected values. Captures use original Qwen27
layer0 core output, projected Z and GGUF normalization parameter.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn,layer_norm_fwd_kernel
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--header',type=Path);ap.add_argument('--model',type=Path);ap.add_argument('--captures',type=Path);args=ap.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True);cases=[]
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();f=p/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 h=14695981039346656037
 for v in data:h=((h^int(v))*1099511628211)&((1<<64)-1)
 return h
for rows in [1,3,48,240,377,753,6144]:
 for seed in [0,1]:
  for activation in ['silu','sigmoid']:
   eps=1e-6 if seed==0 else 1e-5;i=torch.arange(rows*128);x=(((i*37+i//13+seed*11)%509-254).float()/128).reshape(rows,128);x=x*torch.tensor([0.,2**-20,1.,256.])[(torch.arange(rows)+seed)%4,None];x=x.to(torch.bfloat16).cuda()
   z=((((i*13+i//17+seed*3)%251-125).float()/8).reshape(rows,128)*torch.tensor([1.,8.,.125,4.])[(torch.arange(rows)+seed)%4,None]).to(torch.bfloat16).cuda();i=torch.arange(128);w=(((i*7+seed)%31-15).float()/32+1).to(torch.bfloat16).cuda()
   out=rmsnorm_fn(x,w,None,z=z,eps=eps,norm_before_gate=True,activation=activation);assert torch.isfinite(out).all();name=f'norm-{rows}-{seed}-{activation}'
   for key,t in [('x',x),('z',z),('weight',w)]:save(name+'-'+key+'.bin',t)
   cases.append(dict(rows=rows,seed=seed,activation=activation,eps=eps,hash=save(name+'-out.bin',out)));print(name,'captured',flush=True)
obj=layer_norm_fwd_kernel
while not hasattr(obj,"device_caches") and hasattr(obj,"fn"):obj=obj.fn
for dev,cache in obj.device_caches.items():
 for c in cache:
  if isinstance(c,dict):
   for key,kernel in c.items():
    if hasattr(kernel,'asm'):
     name=kernel.metadata.hash;(p/(name+'.ptx')).write_text(kernel.asm['ptx']);(p/(name+'.ttgir')).write_text(kernel.asm['ttgir'])
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
if args.header:
 lines=['/* Independent pinned vLLM BF16 RMSNormGated; see generator. */','#ifndef BN_CUDA_BF16_GDN_GATED_NORM_REFERENCE_H','#define BN_CUDA_BF16_GDN_GATED_NORM_REFERENCE_H','typedef struct { int rows,seed,activation; float eps; uint64_t hash; } BnBF16GDNGatedNormReference;','static const BnBF16GDNGatedNormReference cuda_bf16_gdn_gated_norm_cases[] = {']
 for c in cases:lines.append(' {%d,%d,%d,%.9ef,UINT64_C(0x%016x)},'%(c['rows'],c['seed'],0 if c['activation']=='silu' else 1,c['eps'],c['hash']))
 args.header.write_text('\n'.join(lines+['};','#endif'])+'\n')
if args.model:
 import gguf
 reader=gguf.GGUFReader(args.model);tensor=next(t for t in reader.tensors if t.name=='blk.0.ssm_norm.weight');weight=torch.from_numpy(np.array(gguf.quants.dequantize(tensor.data,tensor.tensor_type),copy=True)).reshape(128).to(torch.bfloat16).cuda();save('real-weight.bin',weight)
 results=[]
 for phase in [0,1]:
  bp=lambda name:torch.from_numpy(np.fromfile(args.captures/name,np.uint16)).view(torch.bfloat16).cuda()
  x=bp(f'model.layers.0.linear_attn.norm.{phase}.input.bin').reshape(-1,128);qkvz=bp(f'model.layers.0.linear_attn.in_proj_qkvz.{phase}.output.bin').reshape(-1,16384);z=qkvz[:,10240:].contiguous().reshape(-1,128)
  ref=bp(f'model.layers.0.linear_attn.norm.{phase}.output.bin').reshape_as(x)
  out=rmsnorm_fn(x,weight,None,z=z,eps=1e-6,norm_before_gate=True,activation='silu');assert torch.equal(out.view(torch.int16),ref.view(torch.int16))
  for key,t in [('x',x),('z',z),('out',out)]:save(f'real-{phase}-{key}.bin',t)
  results.append(dict(phase=phase,rows=x.shape[0],words=out.numel(),exact=True));print('real capture',phase,'exact',out.numel(),flush=True)
 (p/'real.json').write_text(json.dumps(results,indent=2)+'\n')
