"""Independent packed BF16 GDN decode/state references from pinned vLLM.

Image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Uses the unmodified packed recurrent kernel, logical repeated K heads and F32
[V,K] state. Eight dependent steps plus exported states permit fresh resumes.
BF16 output hashes consume bytes; F32 state hashes consume uint32 bit patterns.
No BitNet code generates expected values. Geometry is K=V=128.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True)
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
 for v in (data if t.dtype==torch.bfloat16 else np.frombuffer(data,dtype='<u4')):h=((h^int(v))*1099511628211)&((1<<64)-1)
 return h
cases=[]
for hk,hv in [(1,1),(2,6),(16,48)]:
 for seed in [0,1,2]:
  name=f'decode-{hk}-{hv}-{seed}';dim=2*hk*128+hv*128;i=torch.arange(hv)
  log=(((i*11+seed*5)%29-14).float()/8+2**-17).cuda();bias=(((i*7+seed)%31-15).float()/32).to(torch.bfloat16).cuda()
  save(name+'-log.bin',log);save(name+'-bias.bin',bias)
  cache=torch.zeros(2,hv,128,128,dtype=torch.float32,device='cuda');i=torch.arange(hv*128*128)
  if seed:cache[1]=(((i*17+i//37+seed*19)%251-125).float()/4096).reshape(hv,128,128).cuda()
  save(name+'-initial.bin',cache[1]);oh=[];sh=[]
  for step in range(8):
   i=torch.arange(dim);scales=[0.,2**-20,1.,256.];x=(((i*37+i//13+seed*11+step*7)%509-254).float()/128*scales[(step+seed)%4]).to(torch.bfloat16).reshape(1,dim).cuda()
   i=torch.arange(hv);b=(((i*13+seed*11+step*5)%251-125).float()/16).to(torch.bfloat16).reshape(1,hv).cuda();a=(((i*17+i//19+seed*3+step*11)%509-254).float()/8).to(torch.bfloat16).reshape(1,hv).cuda();ba=torch.cat([b,a],-1)
   save(f'{name}-{step}-x.bin',x);save(f'{name}-{step}-ba.bin',ba)
   out=torch.empty(1,1,hv,128,dtype=torch.bfloat16,device='cuda')
   fused_recurrent_gated_delta_rule_packed_decode(x,ba[:,hv:],ba[:,:hv],log,bias,128**-0.5,cache,out,torch.tensor([1],device='cuda',dtype=torch.int32),True)
   assert torch.isfinite(out).all() and torch.isfinite(cache).all()
   oh.append(save(f'{name}-{step}-out.bin',out));sh.append(save(f'{name}-{step}-state.bin',cache[1]))
  cases.append(dict(hk=hk,hv=hv,seed=seed,out=oh,state=sh));print(name,flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
lines=['/* Pinned unmodified vLLM packed BF16 GDN decode, eight dependent steps. */','#ifndef BN_CUDA_BF16_GDN_DECODE_REFERENCE_H','#define BN_CUDA_BF16_GDN_DECODE_REFERENCE_H','typedef struct { int hk,hv,seed; uint64_t out[8],state[8]; } BnBF16GDNDecodeReference;','static const BnBF16GDNDecodeReference cuda_bf16_gdn_decode_cases[] = {']
for c in cases:lines.append(' {%d,%d,%d,{%s},{%s}},'%(c['hk'],c['hv'],c['seed'],','.join('UINT64_C(0x%016x)'%h for h in c['out']),','.join('UINT64_C(0x%016x)'%h for h in c['state'])))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_gdn_decode_reference.h').write_text('\n'.join(lines)+'\n')
