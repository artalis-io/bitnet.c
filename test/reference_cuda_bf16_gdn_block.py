"""Independent complete Qwen35 GDN block with carried BF16/F32 state.

Run in the pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
with the unchanged e454affe GGUF plugin. Inputs are synthetic normalized hidden
states; immutable parameters come from the original independently replayed
Qwen27 artifacts, never from native output. No expected file is overwritten.
This covers the linear-attention block, excluding outer residual/FFN layers.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
ap=argparse.ArgumentParser();ap.add_argument('linear',type=Path);ap.add_argument('conv',type=Path);ap.add_argument('prep',type=Path);ap.add_argument('norm',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--tokens',type=int,nargs='+',default=[257,1047]);ap.add_argument('--decode',type=int,default=8);args=ap.parse_args();p=args.output;p.mkdir(parents=True,exist_ok=True);provenance={}
def load(base,name,dtype,shape):
 f=base/name;provenance[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();raw=np.fromfile(f,np.float32 if dtype==torch.float32 else np.uint16 if dtype==torch.bfloat16 else np.uint8);return torch.from_numpy(raw).view(dtype).reshape(shape).cuda()
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();f=p/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return hashlib.sha256(data).hexdigest()
wq=load(args.linear,'in_proj_qkvz-weight.bin',torch.uint8,(16384,3520));wb=load(args.linear,'in_proj_ba-weight.bin',torch.uint8,(96,5440));wo=load(args.linear,'out_proj-weight.bin',torch.uint8,(5120,6528));wc=load(args.conv,'real-weight.bin',torch.bfloat16,(10240,4));al=load(args.prep,'real-log.bin',torch.float32,(48,));dt=load(args.prep,'real-bias.bin',torch.bfloat16,(48,));wn=load(args.norm,'real-weight.bin',torch.bfloat16,(128,));layout=GGUFHeadTilingLayout(3,128);cases=[]
for nt in args.tokens:
 cache=torch.zeros(2,48,128,128,device='cuda',dtype=torch.float32);history=torch.zeros(2,10240,3,device='cuda',dtype=torch.bfloat16);idx=torch.tensor([1],device='cuda',dtype=torch.int32)
 for phase in range(args.decode+1):
  take=nt if phase==0 else 1;i=torch.arange(take*5120);x=(((i*17+i//37+phase*11)%509-254).float()/128).reshape(take,5120).to(torch.bfloat16).cuda();prefix=f'block-{nt}-{phase}';save(prefix+'-input.bin',x)
  qkvz=_fused_mul_mat_gguf(x,wq,13);ba=_fused_mul_mat_gguf(x,wb,8);qkv=qkvz[:,:10240].contiguous();z=qkvz[:,10240:].contiguous();save(prefix+'-qkvz.bin',qkvz);save(prefix+'-ba.bin',ba)
  if phase==0:
   post=causal_conv1d_fn(qkv.T,wc,None,history,torch.tensor([0,take],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([False],device='cuda'),out=torch.empty_like(qkv.T)).T
   q,k,v,g,b=fused_post_conv_prep(post,ba[:,48:],ba[:,:48],al,dt,16,128,128)
   core,final=chunk_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),b.unsqueeze(0),initial_state=cache[1:2].clone(),output_final_state=True,cu_seqlens=torch.tensor([0,take],device='cuda',dtype=torch.int32),use_qk_l2norm_in_kernel=False);cache[1].copy_(final[0])
  else:
   post=causal_conv1d_update(qkv.clone(),history,wc,activation='silu',conv_state_indices=idx);core=torch.empty(1,1,48,128,device='cuda',dtype=torch.bfloat16);fused_recurrent_gated_delta_rule_packed_decode(post,ba[:,48:],ba[:,:48],al,dt,128**-0.5,cache,core,idx,True)
  normalized=rmsnorm_fn(core.reshape(-1,128),wn,None,z=z.reshape(-1,128),eps=1e-6,norm_before_gate=True,activation='silu').reshape(take,6144);trans=layout.input_to_gguf(normalized);out=_fused_mul_mat_gguf(trans,wo,8)
  assert all(torch.isfinite(t).all() for t in [post,core,normalized,out,cache])
  hashes={}
  for name,t in [('post',post),('history',history[1].T),('core',core),('state',cache[1]),('norm',normalized),('trans',trans),('out',out)]:hashes[name]=save(prefix+'-'+name+'.bin',t)
  cases.append(dict(tokens=nt,phase=phase,take=take,hashes=hashes));print(prefix,'captured',flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');(p/'upstream-sha256.json').write_text(json.dumps(provenance,indent=2)+'\n')
