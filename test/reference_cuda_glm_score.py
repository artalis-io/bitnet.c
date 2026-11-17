"""Independent original SM120 GLM FP8 scorer captures, signed head weights.
Only masked F32 outputs are fixtures: original compressed-logit padding is
uninitialized and must be masked before use. No candidate code is imported.
"""
from pathlib import Path
import argparse, hashlib, inspect, json
import numpy as np
import torch
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
from vllm.models.glm5next.nvidia.ops.kpool_compress import mqa_logits_use_mx_scales
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--header',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);assert not mqa_logits_use_mx_scales()
cases=[(1,0),(2,1),(1,3),(2,62),(17,0),(65,0),(129,3),(17,2047),(65,4098),(129,8191)]
def raw(n,seed):
 i=np.arange(n,dtype=np.uint64);m=(i*37+i//13+seed*11)%127
 return (m+(((i*17+seed)%3==0).astype(np.uint64))*128).astype(np.uint8)
def fnv(b):
 h=14695981039346656037
 for v in b:h=((h^v)*1099511628211)&((1<<64)-1)
 return h
records=[];lines=['/* Original DeepGEMM SM120, masked causal scores. */','#ifndef BN_CUDA_GLM_SCORE_REFERENCE_H','#define BN_CUDA_GLM_SCORE_REFERENCE_H','typedef struct {int tokens,position,seed;uint64_t hash;} BnGLMScoreReference;','static const BnGLMScoreReference glm_score_reference[]={']
for ci,(nt,pos) in enumerate(cases):
 nk=(pos+nt)//4;seed=ci+1;q=raw(nt*4096,seed).reshape(nt,32,128);k=raw(nk*128,seed+17).reshape(nk,128);sc=np.ldexp(np.ones(nk,dtype=np.float32),(np.arange(nk)%13-6).astype(np.int32)).astype(np.float32);i=np.arange(nt*32);w=(((i*17+seed*13)%251-125).astype(np.float32)/128+np.float32(2**-17)).reshape(nt,32)
 qt=torch.from_numpy(q).view(torch.float8_e4m3fn).cuda();kt=torch.from_numpy(k).view(torch.float8_e4m3fn).cuda();st=torch.from_numpy(sc).cuda();wt=torch.from_numpy(w).cuda();end=((torch.arange(nt,device='cuda',dtype=torch.int32)+pos+1)//4).contiguous();start=torch.zeros_like(end);mask=torch.arange(nk,device='cuda')[None,:]<end[:,None]
 def run():
  if not nk:return torch.empty((nt,0),device='cuda')
  out=fp8_fp4_mqa_logits((qt,None),(kt,st),wt,start,end,nk)
  return torch.where(mask,out[:,:nk],-torch.inf)
 x,y=run(),run();assert torch.equal(x.view(torch.uint8),y.view(torch.uint8));b=x.contiguous().view(torch.uint8).cpu().numpy().tobytes();(a.out/f'{ci}.bin').write_bytes(b);h=fnv(b);records.append(dict(tokens=nt,position=pos,seed=seed,keys=nk,sha256=hashlib.sha256(b).hexdigest(),fnv=f'{h:016x}'));lines.append(f'{{{nt},{pos},{seed},UINT64_C(0x{h:016x})}},');print(ci,nt,pos,nk,'original repeat exact',flush=True)
lines+=['};','#endif'];a.header.write_text('\n'.join(lines)+'\n');source=Path(inspect.getfile(fp8_fp4_mqa_logits));(a.out/'reference.json').write_text(json.dumps(dict(cases=records,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profile='SM120 FP8, signed folded weights, F32 key scales, per-query complete-pool mask; synthetic scorer inputs, no whole-model or top-k claim'),indent=2)+'\n')
