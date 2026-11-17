"""Capture installed vLLM MLA writer twice; never replace existing oracle bytes."""
from pathlib import Path
import argparse, hashlib, json, torch, vllm
from vllm import _custom_ops as ops
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--header',type=Path,required=True);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
records=[]
for n in [0,1,2,63,64,65,129,1025,4105]:
 for seed in [0,1]:
  blocks=(n+63)//64+2;capacity=blocks*64;i=torch.arange(n*512,dtype=torch.int64);row=i//512
  exponents=torch.tensor([-30,-16,-4,0,4,12,24,0])[((row+seed)%8)]
  x=torch.ldexp(((i*37+i//13+seed*11)%509-254).float()/128,exponents).bfloat16().reshape(n,512)
  if n:x[0]=0
  if n>1:x[1]=-0.
  if n>2:x[2]=1
  slots=(torch.arange(n,dtype=torch.int64)*37+seed*11)%capacity
  slots[torch.arange(n)%11==3]=-1;slots[torch.arange(n)%17==5]=-1
  x=x.cuda();slots=slots.cuda()
  def run():
   cache=torch.full((blocks,64,656),0xa5,device='cuda',dtype=torch.uint8)
   if n:ops.concat_and_cache_mla(x,torch.empty((n,0),device='cuda',dtype=torch.bfloat16),cache,slots,'fp8_ds_mla',torch.ones(1,device='cuda'))
   torch.cuda.synchronize();return cache.cpu().numpy().tobytes()
  b=run();assert b==run();f=a.directory/f'mla-cache-{n}-{seed}.bin'
  if f.exists():assert f.read_bytes()==b
  else:f.write_bytes(b)
  h=14695981039346656037
  for v in b:h=((h^v)*1099511628211)&((1<<64)-1)
  records.append(dict(tokens=n,seed=seed,blocks=blocks,digest=h,sha256=hashlib.sha256(b).hexdigest()));print(n,seed,flush=True)
lines=['/* Installed vLLM concat_and_cache_mla, NoPE fp8_ds_mla, captured twice. */','#ifndef BN_CUDA_GLM_MLA_CACHE_REFERENCE_H','#define BN_CUDA_GLM_MLA_CACHE_REFERENCE_H','typedef struct {int tokens,seed,blocks;uint64_t digest;} BnGLMMLACacheReference;','static const BnGLMMLACacheReference glm_mla_cache_reference[] = {']
lines += [' {%d,%d,%d,UINT64_C(0x%016x)},'%(r['tokens'],r['seed'],r['blocks'],r['digest'])for r in records]
a.header.write_text('\n'.join(lines+['};','#endif'])+'\n');ext=Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so'
(a.directory/'reference.json').write_text(json.dumps(dict(records=records,torch=torch.__version__,vllm=vllm.__version__,extension_sha256=hashlib.sha256(ext.read_bytes()).hexdigest()),indent=2)+'\n')
