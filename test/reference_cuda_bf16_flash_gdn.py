"""Independent actual Flash GDN block: natural heads, merged projections,
sigmoid gated norm. Explicit decoded-BF16 checkpoint profile, not plugin loading.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch,gguf
import torch.nn.functional as F
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
m=json.loads(a.inventory.read_text());readers=[gguf.GGUFReader(x['path'])for x in m['shards']];ts={t.name:t for r in readers for t in r.tensors};identities={}
def save(name,x):
 if x.dtype==torch.bfloat16:data=x.detach().cpu().view(torch.int16).numpy().astype('<i2').tobytes()
 else:data=x.detach().cpu().contiguous().numpy().astype('<f4').tobytes()
 assert torch.isfinite(x).all(),name
 f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),dtype=str(x.dtype),shape=list(x.shape))
def raw(name):
 t=ts['blk.0.'+name];v=np.asarray(t.data);identities[name]=dict(type=t.tensor_type.name,raw_sha256=hashlib.sha256(v.tobytes()).hexdigest())
 return torch.from_numpy(np.array(gguf.dequantize(v,t.tensor_type),copy=True))
def weight(name):
 w=raw(name).bfloat16();identities[name]['decoded']=save(name+'.bf16',w);return w.cuda()
wq=weight('attn_qkv.weight');wz=weight('attn_gate.weight');wb=weight('ssm_beta.weight');wa=weight('ssm_alpha.weight');wo=weight('ssm_out.weight');wc=weight('ssm_conv1d.weight').reshape(10240,4);wn=weight('ssm_norm.weight').reshape(128)
coef=raw('ssm_a').reshape(48);assert(coef<0).all();al=torch.log(-coef);dt=raw('ssm_dt.bias').reshape(48).bfloat16();save('a-log.bin',al);save('dt-bias.bin',dt);al=al.cuda();dt=dt.cuda()
wqz=torch.cat([wq,wz]);wba=torch.cat([wb,wa]);cases=[]
for nt,chunk in [(1,1),(5,5),(129,129),(129,64),(1025,1025),(1025,512)]:
 name=f'{nt}-{chunk}';cache=torch.zeros(2,48,128,128,device='cuda',dtype=torch.float32);history=torch.zeros(2,10240,3,device='cuda',dtype=torch.bfloat16);idx=torch.tensor([1],device='cuda',dtype=torch.int32)
 phases=[];steps=[(pos,min(chunk,nt-pos),True)for pos in range(0,nt,chunk)]+[(nt+i,1,False)for i in range(8)]
 for phase,(pos,take,prefill)in enumerate(steps):
  i=torch.arange(take*2560);x=(((i*17+i//37+phase*11)%509-254).float()/128).reshape(take,2560).bfloat16().cuda();prefix=f'{name}-{phase}'
  qkvz=F.linear(x,wqz);ba=F.linear(x,wba);qkv=qkvz[:,:10240].contiguous();z=qkvz[:,10240:].contiguous()
  if prefill:
   post=causal_conv1d_fn(qkv.T,wc,None,history,torch.tensor([0,take],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([phase>0],device='cuda'),activation='silu',out=torch.empty_like(qkv.T)).T
   q,k,v,g,b=fused_post_conv_prep(post,ba[:,48:],ba[:,:48],al,dt,16,128,128)
   core,final=chunk_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),b.unsqueeze(0),initial_state=cache[1:2].clone(),output_final_state=True,cu_seqlens=torch.tensor([0,take],device='cuda',dtype=torch.int32),use_qk_l2norm_in_kernel=False);cache[1].copy_(final[0])
  else:
   post=causal_conv1d_update(qkv.clone(),history,wc,activation='silu',conv_state_indices=idx)
   core=torch.empty(1,1,48,128,device='cuda',dtype=torch.bfloat16)
   fused_recurrent_gated_delta_rule_packed_decode(post,ba[:,48:],ba[:,:48],al,dt,128**-.5,cache,core,idx,True)
  norm=rmsnorm_fn(core.reshape(-1,128),wn,None,z=z.reshape(-1,128),eps=1e-6,norm_before_gate=True,activation='sigmoid').reshape(take,6144)
  out=F.linear(norm,wo) # Natural Flash heads; NO Qwen27 head transpose.
  stages={k:save(prefix+'-'+k+'.bin',v)for k,v in [('input',x),('qkvz',qkvz),('ba',ba),('post',post),('core',core),('norm',norm),('out',out),('state',cache[1]),('history',history[1].T)]}
  phases.append(dict(phase=phase,pos=pos,take=take,prefill=prefill,stages=stages));print(prefix,'captured',flush=True)
 cases.append(dict(name=name,nt=nt,chunk=chunk,phases=phases))
(a.directory/'manifest.json').write_text(json.dumps(dict(profile='natural heads, decodedBF16 merged projections, sigmoid norm',weights=identities,cases=cases),indent=2)+'\n');print('FLASH_GDN_CAPTURED',len(cases),flush=True)
