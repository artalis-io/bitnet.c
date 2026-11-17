#!/usr/bin/env python3
"""Offline independent reference capture; does not overwrite checked-in fixtures.
Run with the qualified vLLM environment and explicitly selected CUDA device.
"""
from pathlib import Path
import argparse,inspect,json,hashlib

import torch,numpy as np
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
parser=argparse.ArgumentParser(description='Capture unchanged original GLM KDA outputs/states and cross-check F64 recurrence.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--fixture-header',type=Path,help='explicitly write the small case0 C fixture')
options=parser.parse_args();p=options.output;p.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);cases=[]
for index,(heads,dim,tokens) in enumerate([(2,3,4),(1,1,5),(2,16,9),(64,128,17),(2,128,65)]):
 gen=torch.Generator().manual_seed(817+index)
 def random(shape):return torch.randn(shape,generator=gen,dtype=torch.float32)
 q=random((1,tokens,heads,dim));q=(q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)).to(torch.bfloat16).float()
 k=random(q.shape);k=(k/torch.linalg.vector_norm(k,dim=-1,keepdim=True)).to(torch.bfloat16).float()
 v=random(q.shape).to(torch.bfloat16).float();g=-5*torch.rand(q.shape,generator=gen);g[...,::3]=0;g[...,1::3]=-5
 beta=torch.rand((1,tokens,heads),generator=gen);beta[:,::3]=0;beta[:,1::3]=1
 state0=random((1,heads,dim,dim))*.05
 if index%2:state0.zero_()
 scale=float(np.float32(dim**-.5));args=[x.cuda()for x in [q,k,v,g,beta]]
 def run():
  out,states=fused_recurrent_kda(*args,scale=scale,initial_state=state0.cuda(),inplace_final_state=False,use_qk_l2norm_in_kernel=False)
  torch.cuda.synchronize();return out.cpu().numpy().copy(),states.cpu().numpy().copy()
 out,states=run();out2,states2=run();assert np.array_equal(out,out2)and np.array_equal(states,states2)
 s=state0[0].double();max_state=max_out=0
 for t in range(tokens):
  s=s*torch.exp(g[0,t].double()).unsqueeze(-2)
  error=v[0,t].double()-torch.einsum('hvk,hk->hv',s,k[0,t].double())
  s=s+(beta[0,t].double().unsqueeze(-1)*error).unsqueeze(-1)*k[0,t].double().unsqueeze(-2)
  y=torch.einsum('hvk,hk->hv',s,q[0,t].double()*scale)
  max_state=max(max_state,float(np.max(np.abs(s.numpy()-states[t]))));max_out=max(max_out,float(np.max(np.abs(y.numpy()-out[0,t]))))
 assert max_state<2e-5 and max_out<2e-5,(index,max_state,max_out)
 arrays=dict(q=q.numpy()[0],k=k.numpy()[0],v=v.numpy()[0],g=g.numpy()[0],beta=beta.numpy()[0],initial=state0.numpy()[0],out=out[0],states=states,scale=np.array(scale,dtype=np.float32));f=p/f'case{index}.npz';np.savez(f,**arrays)
 cases.append(dict(case=index,heads=heads,dim=dim,tokens=tokens,file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),reference_repeat_exact=True,double_state_max_abs=max_state,double_out_max_abs=max_out));print(cases[-1],flush=True)
(p/'reference.json').write_text(json.dumps(dict(cases=cases,source={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(inspect.getfile(fused_recurrent_kda)),Path(inspect.getfile(fused_recurrent_kda)).with_name('fused_recurrent.py')]}),indent=2)+'\n')

if options.fixture_header:
 d=np.load(p/'case0.npz')
 text='/* Independent original GLM KDA Triton capture; seed817, H2 K3 V3 T4.\n * Inputs are lifted BF16, state/output F32. See reference_ssm_kda.py. */\n#ifndef BN_KDA_REFERENCE_H\n#define BN_KDA_REFERENCE_H\n'
 for name in ['q','k','v','g','beta','initial','out','states']:
  a=d[name].flatten();text+=f'static const float kda_ref_{name}[{len(a)}] = {{\n'
  text+='\n'.join('    '+', '.join(float(x).hex()+'f' for x in a[i:i+6])+','for i in range(0,len(a),6))+'\n};\n'
 text+=f'static const float kda_ref_scale = {float(d["scale"]).hex()}f;\n#endif\n'
 options.fixture_header.write_text(text)
