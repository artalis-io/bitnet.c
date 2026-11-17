"""Independent BF16 causal-convolution/SiLU outputs and carried history.

Pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,
Torch 2.14.0a0+4fdf77b940. Uses unchanged native Triton prefill and update ops.
BF16 products round before F32 accumulation; SiLU consumes that F32 sum.
Arguments: existing output directory, optional --header destination. No native
BitNet arithmetic generates expected values. State files use [width-1,dim].
"""
import argparse,json
from pathlib import Path
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);parser.add_argument('--model',type=Path);parser.add_argument('--captures',type=Path);args=parser.parse_args();p=args.directory;cases=[]
def save(name,t):
 flat=torch.empty(t.numel(),dtype=t.dtype);flat.copy_(t.detach().cpu().reshape(-1));data=flat.view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
 for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
 return h
shapes=[(d,n,k,s) for d in [1,31,32,33,255,256,257,10240] for n in [1,5,17] for k in [2,3,4] for s in [0,1]]+[(1,65537,4,s) for s in [0,1]]
for dim,nt,kernel,seed in shapes:
 i=torch.arange(nt*dim);x=(((i*37+i//13+seed*11)%509-254).float()/128).to(torch.bfloat16).reshape(nt,dim).cuda();i=torch.arange(dim*kernel);w=(((i*11+i//7+seed*3)%127-63).float()/128).to(torch.bfloat16).reshape(dim,kernel).cuda();i=torch.arange((kernel-1)*dim);state=((((i*19+seed*5)%251-125).float()/256) if seed else torch.zeros_like(i,dtype=torch.float32)).to(torch.bfloat16).reshape(kernel-1,dim).cuda();cache=torch.cat([torch.zeros_like(state.T.unsqueeze(0)),state.T.unsqueeze(0)],0).contiguous();idx=torch.tensor([1],dtype=torch.int32,device='cuda');loc=torch.tensor([0,nt],dtype=torch.int32,device='cuda');initial=torch.tensor([True],device='cuda')
 y=causal_conv1d_fn(x.T,w,None,cache,loc,cache_indices=idx,has_initial_state=initial,out=torch.empty_like(x.T)).T
 final=cache[1].T.contiguous();assert torch.isfinite(y).all();expected=torch.cat([state,x],0)[-(kernel-1):];assert torch.equal(final,expected)
 # Update and prefill must retain identical output/history. Cap the long test
 # to prefill; sequential decode of 65K tokens would test launch overhead.
 if nt<=17:
  decode=torch.cat([torch.zeros_like(state.T.unsqueeze(0)),state.T.unsqueeze(0)],0).contiguous();ys=[]
  for t in range(nt):ys.append(causal_conv1d_update(x[t:t+1].clone(),decode,w,activation='silu',conv_state_indices=idx).clone())
  assert torch.equal(torch.cat(ys),y) and torch.equal(decode[1].T,final)
 name=f'conv-{dim}-{nt}-{kernel}-{seed}';h=save(name+'.bin',y);hs=save(name+'-state.bin',final);cases.append(dict(dim=dim,tokens=nt,kernel=kernel,seed=seed,hash=h,state_hash=hs,file=name+'.bin'))
(p/'cases.json').write_text(json.dumps(cases,indent=2));lines=['/* Unchanged pinned vLLM Triton BF16 causal convolution + SiLU. */','#ifndef BN_CUDA_BF16_CONV_REFERENCE_H','#define BN_CUDA_BF16_CONV_REFERENCE_H','typedef struct { int dim,tokens,kernel,seed; uint64_t hash,state_hash; } BnBF16ConvReference;','static const BnBF16ConvReference cuda_bf16_conv_cases[] = {']
for c in cases:lines.append(' {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['dim'],c['tokens'],c['kernel'],c['seed'],c['hash'],c['state_hash']))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_conv_reference.h').write_text('\n'.join(lines)+'\n');print('Conv references',len(cases),'output words',sum(c['dim']*c['tokens'] for c in cases),'state words',sum(c['dim']*(c['kernel']-1) for c in cases),flush=True)

if args.model:
 assert args.captures is not None
 import gguf,numpy as np
 from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
 reader=gguf.GGUFReader(args.model);tensor=next(t for t in reader.tensors if t.name=='blk.0.ssm_conv1d.weight');dim=10240;kernel=4
 weight=torch.from_numpy(np.array(tensor.data,copy=True).reshape(dim,kernel)).cuda().to(torch.bfloat16);layout=GGUFHeadTilingLayout(3,128);weight=torch.cat([weight[:4096],layout.weight_to_vllm(weight[4096:],dim=0)],0)
 save('real-weight.bin',weight);cache=torch.zeros(2,dim,kernel-1,device='cuda',dtype=torch.bfloat16);idx=torch.tensor([1],dtype=torch.int32,device='cuda');real=[]
 for phase,nt in [(0,5),(1,1)]:
  filename=args.captures/f'model.layers.0.linear_attn.in_proj_qkvz.{phase}.output.bin';x=torch.from_numpy(np.fromfile(filename,np.uint16)).view(torch.bfloat16).reshape(nt,16384)[:,:dim].contiguous().cuda();save(f'real-input-{phase}.bin',x)
  if phase==0:y=causal_conv1d_fn(x.T,weight,None,cache,torch.tensor([0,nt],dtype=torch.int32,device='cuda'),cache_indices=idx,has_initial_state=torch.tensor([False],device='cuda'),out=torch.empty_like(x.T)).T
  else:y=causal_conv1d_update(x.clone(),cache,weight,activation='silu',conv_state_indices=idx)
  assert torch.isfinite(y).all();h=save(f'real-output-{phase}.bin',y);hs=save(f'real-history-{phase}.bin',cache[1].T);real.append(dict(phase=phase,tokens=nt,dim=dim,kernel=kernel,hash=h,state_hash=hs))
 (p/'real-cases.json').write_text(json.dumps(real,indent=2));print('Real Qwen BF16 conv references',real,flush=True)
