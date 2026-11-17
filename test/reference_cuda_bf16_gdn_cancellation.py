"""Reduce the independent real-prompt GDN input to one active value column.
Run with the pinned vLLM image; output is an immutable model-free fixture.
"""
from pathlib import Path
import torch,numpy as np,importlib
p=Path(__import__('sys').argv[1]);out=p/'cancellation-profile-reference';out.mkdir(exist_ok=False)
def load(name,dtype,shape):
 a=np.fromfile(p/'diagnostic-reference'/('phase1-layer1-gdn-'+name+'.bin'),np.uint16 if dtype==torch.bfloat16 else np.float32)
 return torch.from_numpy(a.copy()).view(dtype).reshape(shape).cuda()
q0=load('q',torch.bfloat16,(1,2,16,128));k0=load('k',torch.bfloat16,(1,2,16,128));v0=load('v',torch.bfloat16,(1,2,48,128));g0=load('log_decay',torch.float32,(1,2,48));b0=load('beta',torch.float32,(1,2,48));h0=load('state-in',torch.float32,(1,48,128,128))
q=q0[:,:,2:3,:].expand(-1,-1,16,-1).contiguous();k=k0[:,:,2:3,:].expand(-1,-1,16,-1).contiguous()
g=g0[:,:,8:9].expand(-1,-1,48).contiguous();b=b0[:,:,8:9].expand(-1,-1,48).contiguous()
v=torch.zeros_like(v0);v[:,:,:,22]=v0[:,:,8:9,22];h=torch.zeros_like(h0);h[:,:,22,:]=h0[:,8:9,22,:]
m=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk')
co=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk_o')
tuner=co.chunk_fwd_kernel_o
while not hasattr(tuner,'configs'):tuner=tuner.fn
configs=list(tuner.configs)
def select(warps):
 tuner.configs=[c for c in configs if c.kwargs['BK']==64 and c.kwargs['BV']==32 and c.num_warps==warps and c.num_stages==3]
 assert len(tuner.configs)==1
 tuner.cache.clear()
select(2)
with torch.inference_mode():
 y,state=m.chunk_gated_delta_rule(q,k,v,g,b,initial_state=h,output_final_state=True,cu_seqlens=torch.tensor([0,2],dtype=torch.int32,device='cuda'),use_qk_l2norm_in_kernel=False)
select(4)
with torch.inference_mode():
 local_y,local_state=m.chunk_gated_delta_rule(q,k,v,g,b,initial_state=h,output_final_state=True,cu_seqlens=torch.tensor([0,2],dtype=torch.int32,device='cuda'),use_qk_l2norm_in_kernel=False)
assert torch.equal(local_state,state)
original_y=load('core',torch.bfloat16,(1,2,48,128));original_s=load('state',torch.float32,(1,48,128,128))
assert torch.equal(y[:,:,:,22],original_y[:,:,8:9,22].expand(-1,-1,48))
assert torch.equal(state[:,:,22,:],original_s[:,8:9,22,:].expand(-1,48,-1))
def raw(x):return x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def hash64(x,state=False):
 h=14695981039346656037
 for b in (np.frombuffer(raw(x),np.uint32) if state else raw(x)):h=((h^int(b))*1099511628211)&((1<<64)-1)
 return h
with (out/'cuda_bf16_gdn_cancellation_reference.h').open('w')as f:
 f.write('/* Independent pinned vLLM GDN cancellation fixture. See reference_cuda_bf16_gdn_cancellation.py. */\n')
 for name,x,width in [('q',q0[:,:,2,:],16),('k',k0[:,:,2,:],16),('v',v0[:,:,8,22],16),('g',g0[:,:,8],32),('beta',b0[:,:,8],32),('state',h0[:,8,22,:],32)]:
  a=np.frombuffer(raw(x),np.uint16 if width==16 else np.uint32)
  f.write('static const uint%d_t cuda_gdn_cancel_%s[%d] = {\n'%(width,name,len(a)))
  for i in range(0,len(a),8):f.write('    '+','.join('0x%0*x'%(width//4,int(n))for n in a[i:i+8])+',\n')
  f.write('};\n')
 f.write('static const uint64_t cuda_gdn_cancel_out_hash = UINT64_C(0x%x);\n'%hash64(y))
 f.write('static const uint64_t cuda_gdn_cancel_local_hash = UINT64_C(0x%x);\n'%hash64(local_y))
 f.write('static const uint64_t cuda_gdn_cancel_state_hash = UINT64_C(0x%x);\n'%hash64(state,True))
for n,x in [('out',y),('local-out',local_y),('state',state)]: (out/(n+'.bin')).write_bytes(raw(x))
print('independent reduced fixture matches original active column; hashes',hex(hash64(y)),hex(hash64(state,True)),flush=True)
