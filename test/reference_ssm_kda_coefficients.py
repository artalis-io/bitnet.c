#!/usr/bin/env python3
"""Offline independent captures. Select the CUDA device explicitly.
The checked-in fixture is written only with --fixture-header.
"""
from pathlib import Path
import argparse,inspect,json,hashlib
import torch,numpy as np
from vllm.models.glm5next.nvidia.ops.third_party.kda.kernels import fused_kda_gate, fused_recurrent_kda
parser=argparse.ArgumentParser(description='Capture unchanged original GLM bounded KDA gate, beta and fused recurrence.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--fixture-header',type=Path,help='explicitly write the small case1 C fixture')
options=parser.parse_args();p=options.output;p.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);cases=[]
for index,(heads,dim,tokens,lb) in enumerate([(1,1,9,-5.),(2,3,7,-5.),(64,128,17,-5.),(3,17,65,-1.),(2,256,3,-12.)]):
 rng=torch.Generator().manual_seed(2301+index)
 def rand(shape):return torch.randn(shape,generator=rng)
 gate=(rand((tokens,heads,dim))*3).bfloat16().float();bias=rand((heads,dim))*.25;a=rand((heads,))*2
 if heads>2:a[0]=-100;a[-1]=10
 gate.flatten()[::17]=-1000;gate.flatten()[1::17]=1000;gate.flatten()[2::17]=0
 raw=rand((tokens,heads)).bfloat16().float();raw.flatten()[::7]=-1000;raw.flatten()[1::7]=1000;raw.flatten()[2::7]=0
 def run():
  g=fused_kda_gate(gate.cuda().reshape(tokens,heads*dim),a.cuda(),dim,bias.cuda().flatten(),safe_gate=True,lower_bound=lb)
  b=raw.cuda().sigmoid();torch.cuda.synchronize();return g.cpu().numpy(),b.cpu().numpy()
 decay,beta=run();g2,b2=run();assert np.array_equal(decay,g2)and np.array_equal(beta,b2)
 x=(gate+bias).double();oracle=lb*torch.sigmoid(torch.exp(a.double())[None,:,None]*x)
 err=float(np.max(np.abs(decay-oracle.numpy())));berr=float(np.max(np.abs(beta-torch.sigmoid(raw.double()).numpy())));assert err<5e-6 and berr<2e-7,(err,berr)
 data=dict(gate=gate.numpy(),bias=bias.numpy(),a_log=a.numpy(),beta_raw=raw.numpy(),log_decay=decay,beta=beta,lower_bound=np.array(lb,dtype=np.float32))
 if index==2:
  q=rand((1,tokens,heads,dim));q=(q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)).bfloat16().float()
  k=rand(q.shape);k=(k/torch.linalg.vector_norm(k,dim=-1,keepdim=True)).bfloat16().float();v=rand(q.shape).bfloat16().float();initial=rand((1,heads,dim,dim))*.05;scale=float(np.float32(dim**-.5))
  cuda_args=[t.cuda()for t in [q,k,v,gate[None],raw[None]]]
  def coupled():
   y,s=fused_recurrent_kda(*cuda_args,scale=scale,initial_state=initial.cuda(),inplace_final_state=False,use_qk_l2norm_in_kernel=False,sigmoid_beta=True,a_log=a.cuda(),g_bias=bias.cuda(),compute_gate=True,lower_bound=lb)
   torch.cuda.synchronize();return y.cpu().numpy()[0].copy(),s.cpu().numpy().copy()
  out,states=coupled();o2,s2=coupled();assert np.array_equal(out,o2)and np.array_equal(states,s2)
  data.update(q=q.numpy()[0],k=k.numpy()[0],v=v.numpy()[0],initial=initial.numpy()[0],out=out,states=states,scale=np.array(scale,dtype=np.float32))
 f=p/f'case{index}.npz';np.savez(f,**data);cases.append(dict(case=index,heads=heads,dim=dim,tokens=tokens,lower_bound=lb,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),double_decay_max_abs=err,double_beta_max_abs=berr,coupled=index==2));print(cases[-1],flush=True)
source=Path(inspect.getfile(fused_kda_gate)).parent
(p/'reference.json').write_text(json.dumps(dict(cases=cases,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [source/'kernels.py',source/'fused_recurrent.py']}),indent=2)+'\n')

if options.fixture_header:
 d=np.load(p/'case1.npz')
 text='/* Original GLM safe-gate Triton + F32 torch sigmoid, seed2302 H2 D3 T7. */\n#ifndef BN_KDA_COEFFICIENTS_REFERENCE_H\n#define BN_KDA_COEFFICIENTS_REFERENCE_H\n'
 for name in ['gate','bias','a_log','beta_raw','log_decay','beta']:
  a=d[name].flatten();text+=f'static const float kda_coeff_{name}[{len(a)}] = {{\n'
  text+='\n'.join('    '+', '.join(float(x).hex()+'f'for x in a[i:i+6])+','for i in range(0,len(a),6))+'\n};\n'
 text+='#endif\n';options.fixture_header.write_text(text)
