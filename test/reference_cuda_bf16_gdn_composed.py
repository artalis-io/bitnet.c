"""Independent resident post-conv preparation -> chunk GDN composition.

Pinned image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Unmodified fused_post_conv_prep and chunk_gated_delta_rule generate all expected
values. Raw packed BF16 QKV/BA use the existing preparation fixture formula;
F32 nonzero [V,K] state uses the prepared-prefill fixture formula. No native
BitNet arithmetic generates expected results. Each split is a dependent call.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True)
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
 for v in (data if t.dtype==torch.bfloat16 else np.frombuffer(data,dtype='<u4')):h=((h^int(v))*1099511628211)&((1<<64)-1)
 return h
cases=[]
for nt,splits in [(5,[5]),(65,[33,32]),(129,[64,65]),(257,[129,128]),(1047,[512,535])]:
 for seed in [0,1]:
  dim=10240;hv=48;i=torch.arange(nt*dim);x=(((i*37+i//13+seed*11)%509-254).float()/128).reshape(nt,dim)
  x=x*torch.tensor([0.,2**-20,1.,256.])[(torch.arange(nt)+seed)%4,None]
  if seed:x=x*torch.pow(2.,((torch.arange(dim)%7)-3).float())[None,:]
  x=x.to(torch.bfloat16).cuda();i=torch.arange(nt*hv);alpha=(((i*17+i//19+seed*3)%509-254).float()/8).to(torch.bfloat16).reshape(nt,hv).cuda();beta=(((i*13+seed*11)%251-125).float()/16).to(torch.bfloat16).reshape(nt,hv).cuda();ba=torch.cat([beta,alpha],-1)
  i=torch.arange(hv);log=(((i*11+seed*5)%29-14).float()/8+2**-17).cuda();bias=(((i*7+seed)%31-15).float()/32).to(torch.bfloat16).cuda()
  i=torch.arange(hv*128*128);initial=(((i*17+i//37+seed*19)%251-125).float()/4096).reshape(1,hv,128,128).cuda()
  name=f'composed-{nt}-{seed}'
  for key,t in [('x',x),('ba',ba),('log',log),('bias',bias),('initial',initial)]:save(name+'-'+key+'.bin',t)
  offset=0;state=initial.clone();oh=[];sh=[]
  for phase,take in enumerate(splits):
   q,k,v,g,b=fused_post_conv_prep(x[offset:offset+take].contiguous(),ba[offset:offset+take,hv:],ba[offset:offset+take,:hv],log,bias,16,128,128)
   cu=torch.tensor([0,take],device='cuda',dtype=torch.int32)
   out,state=chunk_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),b.unsqueeze(0),initial_state=state,output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
   assert torch.isfinite(out).all() and torch.isfinite(state).all()
   oh.append(save(f'{name}-{phase}-out.bin',out));sh.append(save(f'{name}-{phase}-state.bin',state));offset+=take
  cases.append(dict(tokens=nt,seed=seed,splits=splits,out=oh,state=sh));print(name,'captured',flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
if args.header:
 rows=['/* Independent pinned vLLM prep -> chunk GDN; see reference_cuda_bf16_gdn_composed.py. */','#ifndef BN_CUDA_BF16_GDN_COMPOSED_REFERENCE_H','#define BN_CUDA_BF16_GDN_COMPOSED_REFERENCE_H','typedef struct { int tokens,seed,n_phases,splits[2]; uint64_t out[2],state[2]; } BnBF16GDNComposedReference;','static const BnBF16GDNComposedReference cuda_bf16_gdn_composed_cases[] = {']
 for c in cases:
  h=lambda v:','.join(f'UINT64_C(0x{x:016x})' for x in v)
  rows.append(' {%d,%d,%d,{%s},{%s},{%s}},'%(c['tokens'],c['seed'],len(c['splits']),','.join(map(str,c['splits'])),h(c['out']),h(c['state'])))
 args.header.write_text('\n'.join(rows+['};','#endif'])+'\n')
