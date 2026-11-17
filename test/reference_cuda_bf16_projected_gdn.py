"""Independent selector-GEMM to packed BF16 GDN references from pinned vLLM.

Image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Uses the unmodified packed recurrent kernel, logical repeated K heads and F32
[V,K] state. Eight dependent steps plus exported states permit fresh resumes.
BF16 output hashes consume bytes; F32 state hashes consume uint32 bit patterns.
No BitNet code generates expected values. Geometry is K=V=128.
BF16 selector GEMMs precede the original recurrent kernel; zero signs follow
actual GEMM output. Original direct-input references are retained separately.
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
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
cases=[]
for hk,hv in [(1,1),(2,6),(16,48)]:
 for seed in [0,1,2]:
  name=f'projected-decode-{hk}-{hv}-{seed}';dim=2*hk*128+hv*128;i=torch.arange(hv)
  log=(((i*11+seed*5)%29-14).float()/8+2**-17).cuda();bias=(((i*7+seed)%31-15).float()/32).to(torch.bfloat16).cuda()
  save(name+'-log.bin',log);save(name+'-bias.bin',bias)
  cache=torch.zeros(2,hv,128,128,dtype=torch.float32,device='cuda');i=torch.arange(hv*128*128)
  if seed:cache[1]=(((i*17+i//37+seed*19)%251-125).float()/4096).reshape(hv,128,128).cuda()
  save(name+'-initial.bin',cache[1]);oh=[];sh=[]
  inputs=[];bas=[]
  for step in range(8):
   i=torch.arange(dim);scales=[0.,2**-20,1.,256.];x=(((i*37+i//13+seed*11+step*7)%509-254).float()/128*scales[(step+seed)%4]).to(torch.bfloat16).reshape(1,dim).cuda()
   i=torch.arange(hv);b=(((i*13+seed*11+step*5)%251-125).float()/16).to(torch.bfloat16).reshape(1,hv).cuda();a=(((i*17+i//19+seed*3+step*11)%509-254).float()/8).to(torch.bfloat16).reshape(1,hv).cuda();ba=torch.cat([b,a],-1)
   inputs.append(x);bas.append(ba)
  guard=torch.tensor([0x5a5a],dtype=torch.int16,device='cuda').view(torch.bfloat16)
  wx=torch.stack([torch.cat([guard,v[0],guard]) for v in inputs],dim=1).contiguous()
  wb=torch.stack([torch.cat([guard,v[0],guard]) for v in bas],dim=1).contiguous()
  save(name+'-wx.bin',wx);save(name+'-wb.bin',wb)
  def project(steps):
   onehot=torch.zeros(len(steps),8,dtype=torch.bfloat16,device='cuda')
   onehot[torch.arange(len(steps),device='cuda'),torch.tensor(steps,device='cuda')]=1
   px=onehot@wx.T;pb=onehot@wb.T
   assert (px[:,[0,-1]].view(torch.int16)==0x5a5a).all() and (pb[:,[0,-1]].view(torch.int16)==0x5a5a).all()
   for row,step in enumerate(steps):
    for result,original in [(px[row,1:-1],inputs[step][0]),(pb[row,1:-1],bas[step][0])]:
     bits=original.view(torch.int16);canonical=torch.where((bits&0x7fff)==0,torch.zeros_like(bits),bits)
     assert torch.equal(result.contiguous().view(torch.int16),canonical)
   return px[:,1:-1].contiguous(),pb[:,1:-1].contiguous()
  for batch in [1,2,3,8,16]:
   for round in range(5):project([q%4+round for q in range(batch)])
  for step in range(8):
   x,ba=project([step])
   save(f'{name}-{step}-x.bin',x);save(f'{name}-{step}-ba.bin',ba)
   out=torch.empty(1,1,hv,128,dtype=torch.bfloat16,device='cuda')
   fused_recurrent_gated_delta_rule_packed_decode(x,ba[:,hv:],ba[:,:hv],log,bias,128**-0.5,cache,out,torch.tensor([1],device='cuda',dtype=torch.int32),True)
   assert torch.isfinite(out).all() and torch.isfinite(cache).all()
   oh.append(save(f'{name}-{step}-out.bin',out));sh.append(save(f'{name}-{step}-state.bin',cache[1]))
  cases.append(dict(hk=hk,hv=hv,seed=seed,out=oh,state=sh));print(name,flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
lines=['/* Pinned Torch BF16 selector GEMMs -> unmodified vLLM GDN, eight dependent steps. */','#ifndef BN_CUDA_BF16_PROJECTED_GDN_REFERENCE_H','#define BN_CUDA_BF16_PROJECTED_GDN_REFERENCE_H','#include "cuda_bf16_gdn_decode_reference.h"','static const BnBF16GDNDecodeReference cuda_bf16_projected_gdn_decode_cases[] = {']
for c in cases:lines.append(' {%d,%d,%d,{%s},{%s}},'%(c['hk'],c['hv'],c['seed'],','.join('UINT64_C(0x%016x)'%h for h in c['out']),','.join('UINT64_C(0x%016x)'%h for h in c['state'])))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_projected_gdn_reference.h').write_text('\n'.join(lines)+'\n')

import hashlib,inspect
source=Path(inspect.getfile(inspect.unwrap(fused_recurrent_gated_delta_rule_packed_decode)))
(p/'provenance.json').write_text(json.dumps({'image':'sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29','torch':torch.__version__,'kernel_source':str(source),'kernel_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selector_batches':[1,2,3,8,16],'selector_projection_bits_match_canonical_inputs':True},indent=2))
