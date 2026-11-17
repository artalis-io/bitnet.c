"""Frozen independent convolution -> GDN prefill -> recurrent decode chain.

Pinned image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Upstream prefill convolution outputs/history come from the original independent
vLLM causal-convolution fixtures. Unmodified vLLM preparation/chunk/update/decode
operators produce new expected values; no native implementation supplies them.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
ap=argparse.ArgumentParser();ap.add_argument('upstream',type=Path);ap.add_argument('directory',type=Path);ap.add_argument('--header',type=Path);args=ap.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True);upstream={};rows=[]
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
 for x in (data if t.dtype==torch.bfloat16 else np.frombuffer(data,dtype='<u4')):h=((h^int(x))*1099511628211)&((1<<64)-1)
 return h
for nt,kernel in [(5,4),(17,2),(17,3),(17,4)]:
 for seed in [0,1]:
  name=f'chain-{nt}-{kernel}-{seed}';dim=10240
  def load(n,shape):
   f=args.upstream/n;upstream[n]=hashlib.sha256(f.read_bytes()).hexdigest();return torch.from_numpy(np.fromfile(f,np.uint16)).view(torch.bfloat16).reshape(shape).cuda()
  post=load(f'conv-{dim}-{nt}-{kernel}-{seed}.bin',(nt,dim));hist=load(f'conv-{dim}-{nt}-{kernel}-{seed}-state.bin',(kernel-1,dim))
  i=torch.arange(nt*48);alpha=(((i*17+i//19+seed*3)%509-254).float()/8).to(torch.bfloat16).reshape(nt,48).cuda();beta=(((i*13+seed*11)%251-125).float()/16).to(torch.bfloat16).reshape(nt,48).cuda()
  i=torch.arange(48);log=(((i*11+seed*5)%29-14).float()/8+2**-17).cuda();bias=(((i*7+seed)%31-15).float()/32).to(torch.bfloat16).cuda()
  i=torch.arange(48*128*128);initial=(((i*17+i//37+seed*19)%251-125).float()/4096).reshape(1,48,128,128).cuda()
  q,k,v,g,b=fused_post_conv_prep(post,alpha,beta,log,bias,16,128,128)
  out,state=chunk_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),b.unsqueeze(0),initial_state=initial,output_final_state=True,cu_seqlens=torch.tensor([0,nt],device='cuda',dtype=torch.int32),use_qk_l2norm_in_kernel=False)
  ph=save(name+'-prefill-out.bin',out);ps=save(name+'-prefill-state.bin',state);hh=[save(name+'-prefill-history.bin',hist)];oh=[];sh=[]
  cache=torch.zeros(2,48,128,128,device='cuda',dtype=torch.float32);cache[1].copy_(state[0]);conv=torch.stack([torch.zeros_like(hist.T),hist.T]).contiguous();idx=torch.tensor([1],device='cuda',dtype=torch.int32)
  i=torch.arange(dim*kernel);w=(((i*11+i//7+seed*3)%127-63).float()/128).to(torch.bfloat16).reshape(dim,kernel).cuda()
  for step in range(4):
   i=torch.arange(dim);raw=(((i*37+i//13+seed*11+step*7)%509-254).float()/128*[0.,2**-20,1.,256.][(step+seed)%4]).to(torch.bfloat16).reshape(1,dim).cuda()
   post=causal_conv1d_update(raw.clone(),conv,w,activation='silu',conv_state_indices=idx)
   i=torch.arange(48);b=(((i*13+seed*11+step*5)%251-125).float()/16).to(torch.bfloat16).reshape(1,48).cuda();a=(((i*17+i//19+seed*3+step*11)%509-254).float()/8).to(torch.bfloat16).reshape(1,48).cuda()
   out=torch.empty(1,1,48,128,device='cuda',dtype=torch.bfloat16)
   fused_recurrent_gated_delta_rule_packed_decode(post,a,b,log,bias,128**-0.5,cache,out,idx,True)
   assert torch.isfinite(out).all() and torch.isfinite(cache).all()
   oh.append(save(f'{name}-{step}-out.bin',out));sh.append(save(f'{name}-{step}-state.bin',cache[1]));hh.append(save(f'{name}-{step}-history.bin',conv[1].T))
  rows.append(dict(tokens=nt,kernel=kernel,seed=seed,prefill_out=ph,prefill_state=ps,history=hh,out=oh,state=sh));print(name,'captured',flush=True)
(p/'cases.json').write_text(json.dumps(rows,indent=2)+'\n');(p/'upstream-sha256.json').write_text(json.dumps(upstream,indent=2)+'\n')
if args.header:
 lines=['/* Independent pinned vLLM conv/GDN chain; original upstream conv files retained. */','#ifndef BN_CUDA_BF16_GDN_CONV_CHAIN_REFERENCE_H','#define BN_CUDA_BF16_GDN_CONV_CHAIN_REFERENCE_H','typedef struct { int tokens,kernel,seed; uint64_t prefill_out,prefill_state,history[5],out[4],state[4]; } BnBF16GDNConvChainReference;','static const BnBF16GDNConvChainReference cuda_bf16_gdn_conv_chain_cases[] = {']
 h=lambda v:','.join(f'UINT64_C(0x{x:016x})' for x in v)
 for r in rows:lines.append(' {%d,%d,%d,%s,%s,{%s},{%s},{%s}},'%(r['tokens'],r['kernel'],r['seed'],h([r['prefill_out']]),h([r['prefill_state']]),h(r['history']),h(r['out']),h(r['state'])))
 args.header.write_text('\n'.join(lines+['};','#endif'])+'\n')
