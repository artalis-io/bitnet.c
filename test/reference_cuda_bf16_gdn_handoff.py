"""Independent resident prefill->decode references from frozen vLLM prefill states.

Image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
The upstream states are original unmodified vLLM preparation/chunk outputs;
expected decode values come only from its unmodified packed recurrent kernel.
No native implementation supplies expected data. Upstream file hashes retained.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
parser=argparse.ArgumentParser();parser.add_argument('upstream',type=Path);parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True)
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
 for x in (data if t.dtype==torch.bfloat16 else np.frombuffer(data,dtype='<u4')):h=((h^int(x))*1099511628211)&((1<<64)-1)
 return h
rows=[];upstream={}
for c in json.loads((args.upstream/'cases.json').read_text()):
 nt,seed=c['tokens'],c['seed'];name=f'composed-{nt}-{seed}';phase=len(c['splits'])-1
 def load(n,dt,shape):
  f=args.upstream/n;upstream[n]=hashlib.sha256(f.read_bytes()).hexdigest()
  return torch.from_numpy(np.fromfile(f,np.uint16 if dt==torch.bfloat16 else np.float32)).view(dt).reshape(shape).cuda()
 state=load(f'{name}-{phase}-state.bin',torch.float32,(48,128,128));log=load(name+'-log.bin',torch.float32,(48,));bias=load(name+'-bias.bin',torch.bfloat16,(48,))
 cache=torch.zeros(2,48,128,128,device='cuda',dtype=torch.float32);cache[1].copy_(state);oh=[];sh=[]
 for step in range(4):
  i=torch.arange(10240);x=(((i*37+i//13+seed*11+step*7)%509-254).float()/128*[0.,2**-20,1.,256.][(step+seed)%4]).to(torch.bfloat16).reshape(1,10240).cuda()
  i=torch.arange(48);b=(((i*13+seed*11+step*5)%251-125).float()/16).to(torch.bfloat16).reshape(1,48).cuda();a=(((i*17+i//19+seed*3+step*11)%509-254).float()/8).to(torch.bfloat16).reshape(1,48).cuda();ba=torch.cat([b,a],-1)
  save(f'{name}-{step}-x.bin',x);save(f'{name}-{step}-ba.bin',ba);out=torch.empty(1,1,48,128,device='cuda',dtype=torch.bfloat16)
  fused_recurrent_gated_delta_rule_packed_decode(x,ba[:,48:],ba[:,:48],log,bias,128**-0.5,cache,out,torch.tensor([1],device='cuda',dtype=torch.int32),True)
  assert torch.isfinite(out).all() and torch.isfinite(cache).all()
  oh.append(save(f'{name}-{step}-out.bin',out));sh.append(save(f'{name}-{step}-state.bin',cache[1]))
 rows.append(dict(tokens=nt,seed=seed,out=oh,state=sh));print(name,'handoff captured',flush=True)
(p/'cases.json').write_text(json.dumps(rows,indent=2)+'\n');(p/'upstream-sha256.json').write_text(json.dumps(upstream,indent=2)+'\n')
if args.header:
 lines=['/* Independent pinned vLLM packed decode after original frozen prefill. */','#ifndef BN_CUDA_BF16_GDN_HANDOFF_REFERENCE_H','#define BN_CUDA_BF16_GDN_HANDOFF_REFERENCE_H','typedef struct { int tokens,seed; uint64_t out[4],state[4]; } BnBF16GDNHandoffReference;','static const BnBF16GDNHandoffReference cuda_bf16_gdn_handoff_cases[] = {']
 for r in rows:
  h=lambda v:','.join(f'UINT64_C(0x{x:016x})' for x in v)
  lines.append(' {%d,%d,{%s},{%s}},'%(r['tokens'],r['seed'],h(r['out']),h(r['state'])))
 args.header.write_text('\n'.join(lines+['};','#endif'])+'\n')
