#!/usr/bin/env python3
"""Offline F32 mHC semantic captures using original CPU helpers.
No model, quantized projection or BF16-output bit parity is inferred.
"""
from pathlib import Path
import argparse,importlib.util,importlib.metadata,json,hashlib
import torch,numpy as np
parser=argparse.ArgumentParser(description='Capture unchanged original CPU torch mHC coefficient and mix operations.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--fixture-header',type=Path)
options=parser.parse_args();p=options.output;p.mkdir(parents=True,exist_ok=True)
source=Path(importlib.metadata.distribution('vllm').locate_file('vllm/model_executor/kernels/mhc/torch.py'));spec=importlib.util.spec_from_file_location('primary_mhc',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);torch.set_num_threads(1);cases=[]
for i,(n,h,dim,iters) in enumerate([(3,4,3,20),(1,1,1,1),(2,2,17,3),(17,4,4096,20),(3,8,31,20),(2,16,3,128),(5,4,5120,20)]):
 gen=torch.Generator().manual_seed(8172+i)
 def random(shape):return torch.randn(shape,generator=gen)
 residual=random((n,h,dim)).bfloat16().float();branch=random((n,dim)).bfloat16().float();fn=random((h*h+2*h,h*dim))*.1;scale=torch.tensor([.3,-.7,1.2]);base=random((h*h+2*h,))*.5;eps=1e-6;pre_eps=1e-6;mult=2.
 x=residual.flatten(1);mixes=(x@fn.t())*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)
 def run():
  post,comb,_,pre=ref.mhc_pre_delayed_torch(residual,fn,scale,base,eps,pre_eps,eps,mult,iters)
  _,_,collapsed,_=ref.mhc_pre_delayed_torch(residual,fn,scale,base,eps,pre_eps,eps,mult,iters,pre_mix=pre)
  out=ref.mhc_post_torch(branch,residual,post,comb)
  return [v.numpy().copy()for v in [pre,post.squeeze(-1),comb,collapsed,out]]
 values=run();again=run();assert all(np.array_equal(a,b)for a,b in zip(values,again))
 pre,post,comb,collapsed,out=values
 # Independent F64 projection-free coefficient and mix check.
 m=mixes.double();s=scale.double();b=base.double();epre=torch.sigmoid(m[:,:h]*s[0]+b[:h])+pre_eps;epost=torch.sigmoid(m[:,h:2*h]*s[1]+b[h:2*h])*mult;c=torch.softmax(m[:,2*h:].reshape(n,h,h)*s[2]+b[2*h:].reshape(h,h),-1)+eps;c=c/(c.sum(-2,keepdim=True)+eps)
 for _ in range(iters-1):c=c/(c.sum(-1,keepdim=True)+eps);c=c/(c.sum(-2,keepdim=True)+eps)
 err=max(np.max(np.abs(epre.numpy()-pre)),np.max(np.abs(epost.numpy()-post)),np.max(np.abs(c.numpy()-comb)));assert err<5e-6,err
 f=p/f'case{i}.npz';np.savez(f,mixes=mixes.numpy(),scale=scale.numpy(),base=base.numpy(),residual=residual.numpy(),branch=branch.numpy(),pre=pre,post=post,comb=comb,collapsed=collapsed,out=out)
 cases.append(dict(case=i,tokens=n,streams=h,dim=dim,iterations=iters,pre_eps=pre_eps,sinkhorn_eps=eps,post_mult=mult,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),double_max_abs=float(err)));print(cases[-1],flush=True)
(p/'reference.json').write_text(json.dumps(dict(cases=cases,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')

if options.fixture_header:
 d=np.load(p/'case0.npz');text='/* Independent original vLLM torch mHC capture; seed8172 H4 D3 N3. */\n#ifndef BN_MHC_REFERENCE_H\n#define BN_MHC_REFERENCE_H\n'
 for name in d.files:
  a=d[name].flatten();text+=f'static const float mhc_ref_{name}[{len(a)}] = {{\n'+'\n'.join('    '+', '.join(float(x).hex()+'f'for x in a[i:i+6])+','for i in range(0,len(a),6))+'\n};\n'
 text+='#endif\n';options.fixture_header.write_text(text)
