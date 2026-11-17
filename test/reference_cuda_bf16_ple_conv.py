"""Pinned NVIDIA PLE dilated-convolution references; no native result injection.
Image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Chronological independent initial/next history snapshots, BF16 eager boundaries.
"""
import argparse,json,hashlib,inspect
from pathlib import Path
import torch
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
assert NULL_BLOCK_ID != 1
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def pattern(n,seed,role):
 i=torch.arange(n,dtype=torch.int64)
 return torch.ldexp(((i*(13+role*6)+seed*11+role*19)%509-254).float()/128,(i%7-3).int()).to(torch.bfloat16).cuda()
def save(name,x):
 data=x.cpu().view(torch.int16).numpy().astype("<i2").tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 h=14695981039346656037
 for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
 return h,hashlib.sha256(data).hexdigest()
cases=[]
for dim in [1,33,10240]:
 for kernel in [2,3,4]:
  for dilation in [1,3,8]:
   hist=(kernel-1)*dilation
   for nt in [1,5,17,129]:
    for seed in [0,1]:
     x=pattern(nt*dim,seed,0).reshape(nt,dim);w=pattern(dim*kernel,seed,1).reshape(dim,kernel)
     residual=pattern(nt*dim,seed,2).reshape(nt,dim);outer=pattern(nt*dim,seed,3).reshape(nt,dim)
     initial=pattern(hist*dim,seed,4).reshape(hist,dim)
     state=initial.T.contiguous().unsqueeze(0).repeat(2,1,1);out=residual.clone()
     idx=torch.tensor([1],device='cuda',dtype=torch.int32);qsl=torch.tensor([0,nt],device='cuda',dtype=torch.int32);has=torch.tensor([True],device='cuda')
     ple_conv(x,out,state,w,idx,outer,mode='prefill',dilation=dilation,query_start_loc=qsl,has_initial_states=has)
     if nt==1:
      ds=initial.T.contiguous().unsqueeze(0).repeat(2,1,1);dy=residual.clone()
      ple_conv(x,dy,ds,w,idx,outer,mode='decode',dilation=dilation,has_initial_states=has)
      assert torch.equal(dy,out) and torch.equal(ds,state),'decode vs prefill'
     key=f'{dim}-{kernel}-{dilation}-{nt}-{seed}'
     oh,os=save(key+'-out.bin',out);sh,ss=save(key+'-history.bin',state[1].T)
     cases.append(dict(dim=dim,kernel=kernel,dilation=dilation,nt=nt,seed=seed,out=oh,history=sh,out_sha256=os,history_sha256=ss))
   print('captured',dim,kernel,dilation,flush=True)
(a.directory/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(a.directory/'provenance.json').write_text(json.dumps(dict(torch=torch.__version__,source_sha256=hashlib.sha256(Path(inspect.getfile(ple_conv)).read_bytes()).hexdigest()),indent=2)+'\n')
lines=['#ifndef BN_CUDA_BF16_PLE_CONV_REFERENCE_H','#define BN_CUDA_BF16_PLE_CONV_REFERENCE_H','typedef struct {int dim,kernel,dilation,nt,seed;uint64_t out,history;} BnPLEConvReference;','static const BnPLEConvReference ple_conv_reference[] = {']
for c in cases:lines.append(' {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(c[k]for k in ['dim','kernel','dilation','nt','seed','out','history']))
lines+=['};','#endif'];(a.directory/'reference.h').write_text('\n'.join(lines)+'\n');print('CAPTURED',len(cases),flush=True)
