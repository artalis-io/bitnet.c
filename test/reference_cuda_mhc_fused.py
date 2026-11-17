"""Procedural default-fused mHC regression hashes from original operators.

Hashes cover every output byte; actual-checkpoint replay additionally compares
all values directly. Original operators are called twice before saving values.
"""
from pathlib import Path
import argparse,hashlib,json,inspect,torch
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang,mhc_fused_post_pre_tilelang,_hc_prenorm_gemm_outputs
from vllm.model_executor.kernels.mhc.tilelang_kernels import _MHC_FUSED_TILELANG_KERNEL,mhc_pre_big_fuse_tilelang,mhc_fused_post_pre_split_config
parser=argparse.ArgumentParser();parser.add_argument('--header',type=Path,required=True);parser.add_argument('--metadata',type=Path,required=True);args=parser.parse_args();d=4096
row=torch.arange(24,dtype=torch.int64)[:,None];col=torch.arange(4*d,dtype=torch.int64)[None,:]
fn=(((row*65537+col*31337+19)%65521-32760).float()*2**-20).cuda();scale=torch.tensor([.3,-.7,1.2],device='cuda');base=(((torch.arange(24,dtype=torch.int64)*7)%31-15).float()*2**-4).cuda();j=torch.arange(d,dtype=torch.int64);norm=(1+((j*11)%33-16).float()*2**-7).bfloat16().cuda();records=[]
for nt in [1,4,5,9,15,16,32,33,128]:
 i=torch.arange(nt*4*d,dtype=torch.int64);j=torch.arange(nt*d,dtype=torch.int64)
 r=(((i*17+13)%257-128).float()*2**-5).bfloat16().reshape(nt,4,d).cuda();branch=(((j*29+3)%257-128).float()*2**-6).bfloat16().reshape(nt,d).cuda();op,oc,_=mhc_pre_tilelang(r,fn,scale,base,1e-5,1e-6,1e-6,2.,20)
 def run():
  res,post,comb,x=mhc_fused_post_pre_tilelang(branch,r,op,oc,fn,scale,base,1e-5,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-5)
  _,_,_,raw=mhc_fused_post_pre_tilelang(branch,r,op,oc,fn,scale,base,1e-5,1e-6,1e-6,2.,20)
  config=mhc_fused_post_pre_split_config(nt,d,4)
  if config is None:dot,sq=_hc_prenorm_gemm_outputs(res.flatten(1),fn,hidden_size=d,hc_mult=4)
  else:dot,sq,res2=_MHC_FUSED_TILELANG_KERNEL(oc.reshape(nt,4,4),r,op.reshape(nt,4),branch,fn.reshape(24,4,d),4,d,24);assert torch.equal(res,res2)
  pre=torch.empty((nt,4),device='cuda');po=torch.empty_like(post);co=torch.empty_like(comb)
  mhc_pre_big_fuse_tilelang(dot,sq,scale,base,res,po.reshape(nt,4),co.reshape(nt,16),torch.empty_like(x),torch.zeros_like(pre),pre,torch.empty_like(x),hidden_size=d,rms_eps=1e-5,hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=2.,sinkhorn_repeat=20,n_splits=dot.shape[0],hc_mult=4,save_pre_mix=True)
  assert torch.equal(po,post)and torch.equal(co,comb)
  return dict(residual=res,dot=dot,sq=sq,pre=pre,post=post,comb=comb,collapsed=raw,normalized=x)
 a=run();b=run();assert all(torch.equal(a[k],b[k])for k in a);hashes=[];sha=[];counts=[]
 for t in a.values():
  data=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();h=14695981039346656037
  for v in data:h=((h^v)*1099511628211)&((1<<64)-1)
  hashes.append(h);sha.append(hashlib.sha256(data).hexdigest());counts.append(t.numel())
 sp=a['dot'].shape[0];records.append(dict(tokens=nt,splits=sp,hashes=hashes,sha256=sha,counts=counts));print(nt,sp,'captured twice',flush=True)
text='/* Original default SM120 fused mHC procedural reference; no native outputs. */\n#ifndef BN_CUDA_MHC_FUSED_REFERENCE_H\n#define BN_CUDA_MHC_FUSED_REFERENCE_H\ntypedef struct { int tokens, splits; uint64_t hash[8]; } BnCudaMHCFusedReference;\nstatic const BnCudaMHCFusedReference cuda_mhc_fused_refs[] = {\n'
for r in records:text+='    {%d,%d,{%s}},\n'%(r['tokens'],r['splits'],','.join('UINT64_C(%d)'%h for h in r['hashes']))
text+='};\n#endif\n';args.header.write_text(text);args.metadata.write_text(json.dumps(dict(records=records,repeated_exact=True,header_sha256=hashlib.sha256(text.encode()).hexdigest(),sources={str(q):hashlib.sha256(q.read_bytes()).hexdigest()for q in [Path(inspect.getfile(mhc_pre_tilelang)),Path(inspect.getfile(mhc_pre_tilelang)).with_name('tilelang_kernels.py')]}),indent=2)+'\n')
