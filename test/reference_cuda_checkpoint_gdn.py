"""Independent controlled Qwen27 layer-0 Q5_K/Q8_0 -> GDN -> Q8_0 gate.

Run in the pinned 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
image with the e454affe GGUF plugin/qualified compatible extension. Parameters
come from independently replayed checkpoint tensor captures. This controlled
subgraph optionally includes convolution and gated norm/head layout. Supplied
inputs exclude outer input norm, residuals and FFN; this is not a complete
transformer layer or a serving benchmark.
"""
from pathlib import Path
import argparse,hashlib,json,inspect
import numpy as np,torch
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as gdn
ap=argparse.ArgumentParser();ap.add_argument('linear',type=Path);ap.add_argument('prep',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--conv',type=Path);ap.add_argument('--norm',type=Path);args=ap.parse_args()
if args.norm and not args.conv:ap.error('--norm requires --conv')
p=args.output;p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);provenance={};cases=[]
def load(base,name,dtype,shape):
 f=Path(base)/name;data=f.read_bytes();provenance[str(f)]=hashlib.sha256(data).hexdigest();return torch.from_numpy(np.frombuffer(data,dtype=np.uint8).copy()).view(dtype).reshape(shape).cuda()
def save(name,t):
 data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);return hashlib.sha256(data).hexdigest()
wq=load(args.linear,'in_proj_qkvz-weight.bin',torch.uint8,(16384,3520));wb=load(args.linear,'in_proj_ba-weight.bin',torch.uint8,(96,5440));wo=load(args.linear,'out_proj-weight.bin',torch.uint8,(5120,6528));al=load(args.prep,'real-log.bin',torch.float32,(48,));dt=load(args.prep,'real-bias.bin',torch.bfloat16,(48,))
wc=load(args.conv,'real-weight.bin',torch.bfloat16,(10240,4)) if args.conv else None
wn=load(args.norm,'real-weight.bin',torch.bfloat16,(128,)) if args.norm else None
layout=GGUFHeadTilingLayout(3,128) if args.norm else None
for batch in [1,2,4,8]:
 for seed in [0,1]:
  stem=f'b{batch}-s{seed}';cache=torch.zeros(batch+1,48,128,128,device='cuda',dtype=torch.float32);indices=torch.arange(1,batch+1,device='cuda',dtype=torch.int32)
  i=torch.arange(48*128*128)
  if seed:
   for q in range(batch):cache[q+1]=(((i*17+i//37+(q+1)*19)%251-125).float()/4096).reshape(48,128,128).cuda()
  save(stem+'-initial.bin',cache[1:])
  history=torch.zeros(batch+1,10240,3,device='cuda',dtype=torch.bfloat16) if args.conv else None
  if args.conv:
   i=torch.arange(3*10240)
   if seed:
    for q in range(batch):history[q+1]=(((i*19+(q+1)*5)%251-125).float()/256).to(torch.bfloat16).reshape(3,10240).T.cuda()
   save(stem+'-history-initial.bin',history[1:].transpose(1,2))
  for step in range(5):
   i=torch.arange(5120);x=torch.stack([(((i*17+i//37+q*11+step*23+seed*7)%509-254).float()/128).to(torch.bfloat16)for q in range(batch)]).cuda()
   qkvz=_fused_mul_mat_gguf(x,wq,13);ba=_fused_mul_mat_gguf(x,wb,8)
   core=torch.empty(batch,1,48,128,device='cuda',dtype=torch.bfloat16)
   post=causal_conv1d_update(qkvz[:,:10240].clone(),history,wc,activation='silu',conv_state_indices=indices) if args.conv else qkvz[:,:10240]
   gdn(post,ba[:,48:],ba[:,:48],al,dt,128**-0.5,cache,core,indices,True)
   if args.norm:
    normalized=rmsnorm_fn(core.reshape(-1,128),wn,None,z=qkvz[:,10240:].contiguous().reshape(-1,128),eps=1e-6,norm_before_gate=True,activation='silu').reshape(batch,6144)
    transformed=layout.input_to_gguf(normalized)
   else:transformed=core.reshape(batch,6144)
   output=_fused_mul_mat_gguf(transformed,wo,8)
   assert all(torch.isfinite(t).all()for t in [x,qkvz,ba,core,cache,output])
   hashes={name:save(f'{stem}-t{step}-{name}.bin',value)for name,value in [('input',x),('qkvz',qkvz),('ba',ba),('core',core),('state',cache[1:]),('output',output)]}
   if args.conv:
    hashes['post']=save(f'{stem}-t{step}-post.bin',post);hashes['history']=save(f'{stem}-t{step}-history.bin',history[1:].transpose(1,2))
   if args.norm:
    for name,tensor in [('qkv',qkvz[:,:10240]),('z',qkvz[:,10240:]),('norm',normalized),('trans',transformed)]:hashes[name]=save(f'{stem}-t{step}-{name}.bin',tensor)
   cases.append(dict(batch=batch,seed=seed,step=step,sha256=hashes));print(stem,step,flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2));provenance['extension_sha256']=hashlib.sha256((Path(inspect.getfile(__import__('vllm_gguf_plugin'))).parent/'_C_gguf.abi3.so').read_bytes()).hexdigest();(p/'provenance.json').write_text(json.dumps(provenance,indent=2))

provenance['torch_version']=torch.__version__
provenance['gdn_source_sha256']=hashlib.sha256(Path(inspect.getfile(inspect.unwrap(gdn))).read_bytes()).hexdigest()
(p/'provenance.json').write_text(json.dumps(provenance,indent=2))
