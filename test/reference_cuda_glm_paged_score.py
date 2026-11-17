"""Independent original SM120 paged and nonpaged FP8 score captures.
Input cache is planar64 FP8 values then F32 scales; reversed physical pages.
No candidate implementation is imported. Original padding is masked.
"""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np
import torch,vllm
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits,fp8_fp4_mqa_logits,get_paged_mqa_logits_metadata
from vllm.models.glm5next.nvidia.ops.kpool_compress import mqa_logits_use_mx_scales
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--header',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);assert not mqa_logits_use_mx_scales()
def raw(n,seed):
 i=np.arange(n,dtype=np.uint64);return (((i*37+i//13+seed*11)%127)+((i*17+seed)%3==0).astype(np.uint64)*128).astype(np.uint8)
def fnv(b):
 h=14695981039346656037
 for v in b:h=((h^v)*1099511628211)&((1<<64)-1)
 return h
cases=[(1,0),(2,1),(1,3),(2,254),(3,2047),(4,4098),(7,8191),(17,262143)];records=[];lines=['/* Original SM120 paged/nonpaged score captures, raw future mask. */','#ifndef BN_CUDA_GLM_PAGED_SCORE_REFERENCE_H','#define BN_CUDA_GLM_PAGED_SCORE_REFERENCE_H','typedef struct {int tokens,position,seed;uint64_t hash;} BnGLMPagedScoreReference;','static const BnGLMPagedScoreReference glm_paged_score_reference[]={']
for ci,(nt,pos)in enumerate(cases):
 seed=ci+1;nk=(pos+nt)//4;blocks=(nk+63)//64+2;table=np.arange(blocks-1,-1,-1,dtype=np.int32);cache=np.full((blocks,8448),0xa5,dtype=np.uint8);q=raw(nt*4096,seed).reshape(1,nt,32,128);k=raw(nk*128,seed+17).reshape(nk,128);sc=np.ldexp(np.ones(nk,dtype=np.float32),(np.arange(nk)%13-6).astype(np.int32)).astype(np.float32);ix=np.arange(nt*32);weights=(((ix*17+seed*13)%251-125).astype(np.float32)/128+np.float32(2**-17)).reshape(nt,32)
 for ki in range(nk):
  page=table[ki//64];off=ki%64;cache[page,off*128:(off+1)*128]=k[ki];cache[page,8192+off*4:8192+(off+1)*4]=sc[ki:ki+1].view(np.uint8)
 qt=torch.from_numpy(q).view(torch.float8_e4m3fn).cuda();kt=torch.from_numpy(k).view(torch.float8_e4m3fn).cuda();st=torch.from_numpy(sc).cuda();wt=torch.from_numpy(weights).cuda();ct=torch.from_numpy(cache).reshape(blocks,64,1,132).cuda();bt=torch.from_numpy(table).reshape(1,blocks).cuda();ends=((torch.arange(nt,device='cuda',dtype=torch.int32)+pos+1)//4).reshape(1,nt).contiguous();mask=torch.arange(nk,device='cuda')[None,:]<ends.reshape(nt,1)
 def run():
  if not nk:return torch.empty((nt,0),device='cuda')
  schedule=get_paged_mqa_logits_metadata(ends,64,torch.cuda.get_device_properties(0).multi_processor_count);x=fp8_fp4_paged_mqa_logits((qt,None),ct,wt,ends,bt,schedule,((nk+63)//64)*64)
  return torch.where(mask,x[:,:nk],-torch.inf).contiguous()
 x,y=run(),run();assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
 if nk:
  z=fp8_fp4_mqa_logits((qt.reshape(nt,32,128),None),(kt,st),wt,torch.zeros(nt,device='cuda',dtype=torch.int32),ends.flatten(),nk);z=torch.where(mask,z[:,:nk],-torch.inf);assert torch.equal(x.view(torch.uint8),z.contiguous().view(torch.uint8))
 b=x.view(torch.uint8).cpu().numpy().tobytes();(a.out/f'{ci}.bin').write_bytes(b);h=fnv(b);records.append(dict(tokens=nt,position=pos,seed=seed,keys=nk,blocks=blocks,sha256=hashlib.sha256(b).hexdigest(),fnv=f'{h:016x}'));lines.append(f'{{{nt},{pos},{seed},UINT64_C(0x{h:016x})}},');print(ci,nt,pos,nk,'original paged repeat/nonpaged exact',flush=True)
lines+=['};','#endif'];a.header.write_text('\n'.join(lines)+'\n');sources={}
for f in [fp8_fp4_paged_mqa_logits,fp8_fp4_mqa_logits,get_paged_mqa_logits_metadata]:
 s=Path(inspect.getfile(f));sources[str(s)]=hashlib.sha256(s.read_bytes()).hexdigest()
s=Path(vllm.__file__).parent/'third_party/deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh';sources[str(s)]=hashlib.sha256(s.read_bytes()).hexdigest();(a.out/'reference.json').write_text(json.dumps(dict(cases=records,sources=sources,profile='Original SM120 signed/F32-scale paged and contiguous logits, masked causal prefix, planar64 shuffled cache. Synthetic scorer inputs, no MLA/model claim.'),indent=2)+'\n')
