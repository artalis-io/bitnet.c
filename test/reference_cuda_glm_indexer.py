"""Unchanged GLM Hadamard/FP8, gated-pool and head-weight reference operators.

Synthetic values expose F32 APE/head-weight precision and pool boundaries;
outputs are never computed by a candidate implementation. Existing captures
must match byte-for-byte. The original paged decode operator supplies next tail
state and independently verifies the original whole-pool compress operator.
"""
import argparse,hashlib,inspect,json
from pathlib import Path
import numpy as np
import torch
from vllm.models.glm5next.nvidia.ops import kpool_compress as ops
from vllm.models.glm5next.common.attention import _fused_indexer_weight_scale
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--header',type=Path,required=True);a=ap.parse_args();a.directory.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
def save(name,x):
 b=x.contiguous().view(torch.uint8).cpu().numpy().tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==b,name
 else:f.write_bytes(b)
 h=14695981039346656037
 for v in b:h=((h^v)*1099511628211)&((1<<64)-1)
 return h
qcases=[];pcases=[]
for rows in [1,2,31,32,33,129,1025,65537]:
 for seed in [0,1]:
  i=torch.arange(rows*128,dtype=torch.int64);x=(((i*37+i//13+seed*11)%509-254).float()/128).reshape(rows,128)
  exponents=torch.tensor([-30,-16,-4,0,4,12,24,0]);x*=torch.exp2(exponents[(torch.arange(rows)+seed)%8].float())[:,None]
  x[0]=0
  if rows>1:x[1]=1
  if rows>2:x[2]=0;x[2,1]=1
  if rows>3:x[3]=-0.
  x=x.bfloat16().cuda();out,scale=ops.fwht128_quant_fp8(x);again,as_=ops.fwht128_quant_fp8(x);assert torch.equal(out.view(torch.uint8),again.view(torch.uint8))and torch.equal(scale,as_)
  ri=torch.arange(rows,dtype=torch.int64);w=(((ri*17+seed*13)%251-125).float()/128+2**-17).cuda().reshape(rows,1);factor=.015625 if seed==0 else .01234567
  weighted=_fused_indexer_weight_scale(w,scale.reshape(rows,1,1),factor).reshape(-1);repeat=_fused_indexer_weight_scale(w,scale.reshape(rows,1,1),factor).reshape(-1);assert torch.equal(weighted,repeat)
  prefix=f'q-{rows}-{seed}';qcases.append(dict(rows=rows,seed=seed,factor=factor,out=save(prefix+'.bin',out),scale=save(prefix+'-scale.bin',scale),weighted=save(prefix+'-weighted.bin',weighted)))
  print(prefix,'captured twice',flush=True)
for pos,nt in [(0,1),(0,3),(0,4),(0,5),(0,17),(0,129),(0,4097),(1,1),(1,2),(1,3),(2,1),(2,2),(3,1),(3,5),(3,129),(31,1),(31,2),(31,65),(1023,1),(1023,5),(1025,129)]:
 for seed in [0,1]:
  def values(start,count,role):
   i=torch.arange(start*128,(start+count)*128,dtype=torch.int64)
   if role==0:v=((i*37+i//13+seed*11)%509-254).float()/128
   else:v=((i*13+i//17+seed*3)%251-125).float()/(16 if seed==0 else 2)
   return v.reshape(count,128).bfloat16().cuda()
  key=values(pos,nt,0);gate=values(pos,nt,1);tail=torch.zeros((1,2,4,128),device='cuda',dtype=torch.bfloat16)
  for logical in range(max(0,pos-4),pos):tail[0,0,logical%4]=values(logical,1,0)[0];tail[0,1,logical%4]=values(logical,1,1)[0]
  initial=tail.clone();ai=torch.arange(512,dtype=torch.int64);ape=(((ai*19+seed*13)%127-63).float()/32+2**-17).reshape(4,128).cuda()
  groups=(pos+nt)//4-pos//4;ks=[];gs=[]
  for gi in range(groups):
   start=(pos//4+gi)*4
   ks.append(torch.stack([key[t-pos]if t>=pos else initial[0,0,t%4]for t in range(start,start+4)]))
   gs.append(torch.stack([gate[t-pos]if t>=pos else initial[0,1,t%4]for t in range(start,start+4)]))
  cache=torch.zeros((max(1,(groups+63)//64),64,132),dtype=torch.uint8,device='cuda');loc=torch.arange(groups,device='cuda',dtype=torch.int64)
  if groups:
   out,scale=ops.kpool_compress_and_write_cache(cache,torch.stack(ks),torch.stack(gs),ape,loc,4,return_compressed=True,write_cache=False)
   again,as_=ops.kpool_compress_and_write_cache(cache,torch.stack(ks),torch.stack(gs),ape,loc,4,return_compressed=True,write_cache=False);assert torch.equal(out.view(torch.uint8),again.view(torch.uint8))and torch.equal(scale,as_)
  else:out=torch.empty((0,128),device='cuda',dtype=torch.float8_e4m3fn);scale=torch.empty(0,device='cuda',dtype=torch.float32)
  positions=torch.arange(pos,pos+nt,device='cuda',dtype=torch.int32).reshape(1,nt);slots=torch.where((positions+1)%4==0,positions//4-pos//4,-1).int();tail_slots=(positions%4).int()
  ops.kpool_decode_update_and_maybe_write_cache_batched(cache,tail,tail_slots,key[None],gate[None],ape,slots,positions,4)
  view=cache.reshape(cache.shape[0],-1)
  for gi in range(groups):
   page,offset=divmod(gi,64);assert torch.equal(view[page,offset*128:(offset+1)*128],out[gi].view(torch.uint8))
   # SM120 stores packed UE8M0 scale bits when DeepGEMM MX scales are enabled.
   bits=view[page,8192+offset*4:8192+offset*4+4].view(torch.int32)[0]
   expected=(torch.log2(scale[gi]).int()+127)*0x01010101 if ops.mqa_logits_use_mx_scales()else scale[gi].view(torch.int32)
   assert bits==expected
  prefix=f'pool-{pos}-{nt}-{seed}';pcases.append(dict(position=pos,tokens=nt,seed=seed,groups=groups,out=save(prefix+'.bin',out),scale=save(prefix+'-scale.bin',scale),tail=save(prefix+'-tail.bin',tail)))
  print(prefix,'whole pool/decode exact',flush=True)
lines=['/* Unchanged original GLM FP8/indexer operators; synthetic model-free fixtures. */','#ifndef BN_CUDA_GLM_INDEXER_REFERENCE_H','#define BN_CUDA_GLM_INDEXER_REFERENCE_H','typedef struct { int rows,seed; float factor; uint64_t out,scale,weighted; } BnGLMQueryReference;','static const BnGLMQueryReference glm_query_reference[] = {']
for c in qcases:lines.append(' {%d,%d,%.9ef,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(c[k]for k in ['rows','seed','factor','out','scale','weighted']))
lines+=['};','typedef struct { int position,tokens,seed,groups; uint64_t out,scale,tail; } BnGLMPoolReference;','static const BnGLMPoolReference glm_pool_reference[] = {']
for c in pcases:lines.append(' {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(c[k]for k in ['position','tokens','seed','groups','out','scale','tail']))
lines+=['};','#endif'];a.header.write_text('\n'.join(lines)+'\n');sources={str(Path(inspect.getfile(inspect.unwrap(f)))):hashlib.sha256(Path(inspect.getfile(inspect.unwrap(f))).read_bytes()).hexdigest()for f in [ops.fwht128_quant_fp8,_fused_indexer_weight_scale]};(a.directory/'reference.json').write_text(json.dumps(dict(query_cases=qcases,pool_cases=pcases,sources=sources,header_sha256=hashlib.sha256(a.header.read_bytes()).hexdigest(),torch=torch.__version__,mx_scales=ops.mqa_logits_use_mx_scales()),indent=2)+'\n')
