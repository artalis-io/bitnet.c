#!/usr/bin/env python3
"""Offline independent direct L2/KDA normalization captures; select CUDA device explicitly."""
from pathlib import Path
import argparse,inspect,json,hashlib
import numpy as np,torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
parser=argparse.ArgumentParser(description='Capture original F32 L2 normalization and fused normalized KDA recurrence.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--fixture-header',type=Path)
options=parser.parse_args();p=options.output;p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);cases=[]
for i,(t,h,dim)in enumerate([(5,2,3),(9,1,1),(17,64,128),(65,2,128)]):
 gen=torch.Generator().manual_seed(3327+i)
 def rand(shape):return torch.randn(shape,generator=gen)
 q=rand((1,t,h,dim));k=rand(q.shape)
 for token in range(t):q[:,token]*=[0,1e-5,1e-3,1,100][token%5];k[:,token]*=[100,1,1e-3,1e-5,0][token%5]
 q=q.bfloat16().float();k=k.bfloat16().float();v=rand(q.shape).bfloat16().float();g=-5*torch.rand(q.shape,generator=gen);beta=torch.rand((1,t,h),generator=gen);beta[:,::3]=0;initial=rand((1,h,dim,dim))*.05;scale=float(np.float32(dim**-.5))
 cuda_args=[a.cuda()for a in [q,k,v,g,beta]]
 def run():
  nq=l2norm_fwd(cuda_args[0],eps=1e-6,output_dtype=torch.float32);nk=l2norm_fwd(cuda_args[1],eps=1e-6,output_dtype=torch.float32)
  out,states=fused_recurrent_kda(*cuda_args,scale=scale,initial_state=initial.cuda(),inplace_final_state=False,use_qk_l2norm_in_kernel=True);torch.cuda.synchronize();return [a.cpu().numpy().copy()for a in [nq,nk,out,states]]
 vals=run();again=run();assert all(np.array_equal(a,b)for a,b in zip(vals,again));nq,nk,out,states=vals
 for raw,norm in [(q,nq),(k,nk)]:
  d=raw.double();expected=d/torch.sqrt(d.square().sum(-1,keepdim=True)+1e-6);assert np.max(np.abs(expected.numpy()-norm))<2e-6
 f=p/f'case{i}.npz';np.savez(f,q=q.numpy()[0],k=k.numpy()[0],v=v.numpy()[0],g=g.numpy()[0],beta=beta.numpy()[0],initial=initial.numpy()[0],nq=nq[0],nk=nk[0],out=out[0],states=states,scale=np.array(scale,np.float32))
 cases.append(dict(case=i,tokens=t,heads=h,dim=dim,eps=1e-6,sha256=hashlib.sha256(f.read_bytes()).hexdigest()));print(cases[-1],flush=True)
(p/'reference.json').write_text(json.dumps(dict(cases=cases,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(inspect.getfile(l2norm_fwd)),Path(inspect.getfile(fused_recurrent_kda)),Path(inspect.getfile(fused_recurrent_kda)).with_name('fused_recurrent.py')]}),indent=2)+'\n')

if options.fixture_header:
 d=np.load(p/'case0.npz');text='/* Original F32 GPU L2 normalization, seed3327 H2 D3 T5. */\n#ifndef BN_KDA_NORM_REFERENCE_H\n#define BN_KDA_NORM_REFERENCE_H\n'
 for name in ['q','k','nq','nk']:
  a=d[name].flatten();text+=f'static const float kda_norm_{name}[{len(a)}] = {{\n'+'\n'.join('    '+', '.join(float(x).hex()+'f'for x in a[i:i+6])+','for i in range(0,len(a),6))+'\n};\n'
 text+='#endif\n';options.fixture_header.write_text(text)
