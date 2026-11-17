from pathlib import Path
import json,hashlib,functools
import torch
from triton.runtime.jit import JITFunction
from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule_packed_decode,fused_sigmoid_gating_delta_rule_update
p=Path('/evidence');(p/'kernels').mkdir(exist_ok=True);assert torch.cuda.device_count()==1
original=JITFunction.run
@functools.wraps(original)
def observe(self,*args,**kwargs):
 result=original(self,*args,**kwargs);name=getattr(self,'__name__',getattr(getattr(self,'fn',None),'__name__',''))
 if ('fused_sigmoid_gating_delta_rule_update_kernel'in name or 'fused_recurrent_gated_delta_rule_packed_decode_kernel'in name)and result is not None:
  for kind in ['ptx','ttgir']:
   if kind in result.asm:(p/'kernels'/f'{name}.{kind}').write_text(result.asm[kind])
 return result
JITFunction.run=observe

def raw(t):return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
def fnv(data):
 h=14695981039346656037
 for x in data:h=((h^x)*1099511628211)&((1<<64)-1)
 return h

def values(n,mult,offset,mod,denom):return (((torch.arange(n,dtype=torch.int64)*mult+offset)%mod)-(mod//2)).float()/denom
records=[]
try:
 for case,(batch,seed)in enumerate([(1,7),(2,3),(2,11),(4,13),(1,17),(2,19),(3,23),(4,29)]):
  x=values(batch*10240,7,seed*17,251,128).reshape(batch,10240).to(torch.bfloat16).cuda();ba=values(batch*96,11,seed*3,127,32).reshape(batch,96).to(torch.bfloat16).cuda()
  alog=values(48,3,seed,127,64).cuda();bias=values(48,5,seed*7,127,64).to(torch.bfloat16).cuda()
  initial=values((batch+1)*48*128*128,11,seed*13,511,8192).reshape(batch+1,48,128,128).cuda();indices=torch.arange(1,batch+1,dtype=torch.int32,device='cuda');cu=torch.arange(batch+1,dtype=torch.int32,device='cuda')
  mode=int(case>=4)
  if mode:
   ix=torch.arange(batch*10240,dtype=torch.int64,device='cuda');x=torch.ldexp(x.float().flatten(),((ix//17+seed)%15-7).int()).reshape(batch,10240).to(torch.bfloat16)
   ih=torch.arange(initial.numel(),dtype=torch.int64,device='cuda');initial=(torch.ldexp(initial.flatten(),((ih//29+seed)%9-4).int())+((ih%7)-3).float()*2**-23).reshape(batch+1,48,128,128)
  results=[]
  for profile in [0,1]:
   state=initial.clone()
   if profile==0:
    out=torch.empty((batch,1,48,128),dtype=torch.bfloat16,device='cuda')
    fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv=x,a=ba[:,48:],b=ba[:,:48],A_log=alog,dt_bias=bias,scale=128**-0.5,initial_state=state,out=out,ssm_state_indices=indices,use_qk_l2norm_in_kernel=True)
   else:
    q=x[:,:2048].reshape(1,batch,16,128);k=x[:,2048:4096].reshape(1,batch,16,128);v=x[:,4096:].reshape(1,batch,48,128)
    out,_=fused_sigmoid_gating_delta_rule_update(A_log=alog,a=ba[:,48:].unsqueeze(0),b=ba[:,:48].unsqueeze(0),dt_bias=bias,q=q,k=k,v=v,initial_state=state,inplace_final_state=True,cu_seqlens=cu,ssm_state_indices=indices,use_qk_l2norm_in_kernel=True)
   assert torch.isfinite(out).all()and torch.isfinite(state).all();assert raw(state[0])==raw(initial[0]);ob=raw(out);sb=raw(state[1:]);
   for suffix,data in [('out',ob),('state',sb)]:
    path=p/f'case-{case}-profile-{profile}-{suffix}.bin'
    if path.exists():assert path.read_bytes()==data,path
    else:path.write_bytes(data)
   results.append(dict(profile=profile,out_hash=fnv(ob),state_hash=fnv(sb),out_sha256=hashlib.sha256(ob).hexdigest(),state_sha256=hashlib.sha256(sb).hexdigest()))
  records.append(dict(batch=batch,seed=seed,mode=mode,profiles=results));(p/'reference.json').write_text(json.dumps(records,indent=2)+'\n');print('GDN_PROFILES',case,batch,seed,flush=True)
finally:JITFunction.run=original
props=torch.cuda.get_device_properties(0);(p/'provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),image='sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',device=props.name,sms=props.multi_processor_count,operators=['Unmodified fused_recurrent_gated_delta_rule_packed_decode','Unmodified fused_sigmoid_gating_delta_rule_update'],state_dtype='float32'),indent=2)+'\n')
