"""Independent captured Qwen27 layer3 projections -> QK/RoPE -> FA2/gate -> WO.

Pinned vLLM image745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,
GGUF plugin e454affe and qualified compatible extension. Uses unchanged original
operations. Synthetic supplied BF16 hidden states and independent analytical KV
prefixes isolate this subblock; outer input norm/residual/FFN and model generation
are excluded. BF16 sigmoid gate profile only. Existing output files are immutable.
"""
from pathlib import Path
import argparse,hashlib,json,inspect
import numpy as np,torch,gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
ap=argparse.ArgumentParser();ap.add_argument('model',type=Path);ap.add_argument('capture',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
p=args.output;p.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);provenance={};cases=[]
def blob(t):return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
def save(name,t):
 data=blob(t);f=p/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return hashlib.sha256(data).hexdigest()
def cap(name,shape):
 f=args.capture/(name+'.bin');data=f.read_bytes();provenance[str(f)]=hashlib.sha256(data).hexdigest()
 return torch.from_numpy(np.frombuffer(data,dtype=np.uint16).copy()).view(torch.bfloat16).reshape(shape).cuda()
qw=cap('q-weight',(256,));kw=cap('k-weight',(256,));cs=cap('cos-sin',(-1,64))
reader=gguf.GGUFReader(str(args.model));tensors={t.name:t for t in reader.tensors};weights=[]
for role in ['q','k','v','output']:
 t=tensors['blk.3.attn_'+role+'.weight'];cols,rows=map(int,t.shape);data=np.array(t.data,copy=True).reshape(rows,-1)
 provenance[t.name]=dict(type=int(t.tensor_type),cols=cols,rows=rows,bytes=data.nbytes,sha256=hashlib.sha256(data.tobytes()).hexdigest())
 weights.append((torch.from_numpy(data).cuda(),int(t.tensor_type)))
starts=[0,1,5,33,65,129,257,1047]
for batch in [1,2,4,8]:
 for seed in [0,1]:
  stem=f'b{batch}-s{seed}';positions=[starts[q]+seed*4095 for q in range(batch)];capacities=[pos+8 for pos in positions];caches=[]
  for q,(pos,capacity) in enumerate(zip(positions,capacities)):
   pages=(capacity+783)//784;kc=torch.empty(pages,784,4,256,device='cuda',dtype=torch.bfloat16);kc.view(torch.int16).fill_(0x5a5a);vc=kc.clone()
   i=torch.arange(pos*1024,dtype=torch.int64)
   kc.view(-1)[:pos*1024].copy_((((i*7+q*11+seed*13)%251-125).float()/128).to(torch.bfloat16).cuda())
   vc.view(-1)[:pos*1024].copy_((((i*13+q*17+seed*7)%257-128).float()/64).to(torch.bfloat16).cuda())
   caches.append((kc,vc,torch.arange(pages,dtype=torch.int32,device='cuda').reshape(1,-1)))
   save(f'{stem}-r{q}-key-initial.bin',kc.view(-1)[:capacity*1024]);save(f'{stem}-r{q}-value-initial.bin',vc.view(-1)[:capacity*1024])
  for step in range(5):
   i=torch.arange(5120,dtype=torch.int64)
   x=torch.stack([(((i*17+i//37+q*11+step*23+seed*7)%509-254).float()/128).to(torch.bfloat16)for q in range(batch)]).cuda()
   qg,ki,vi=[_fused_mul_mat_gguf(x,w,t)for w,t in weights[:3]]
   q,k,gate=fused_qk_rmsnorm_rope_gate(qg,ki,qw,kw,cs,torch.tensor(positions,dtype=torch.int64,device='cuda'),1e-6,24,4,256,64,norm_beta=1)
   v=vi.reshape(batch,4,256);q=q.reshape(batch,24,256);k=k.reshape(batch,4,256);attention=[]
   for row,pos in enumerate(positions):
    kc,vc,table=caches[row];kc.view(-1,4,256)[pos].copy_(k[row]);vc.view(-1,4,256)[pos].copy_(v[row]);actual=pos+1
    a=flash_attn_varlen_func(q[row:row+1],kc,vc,1,torch.tensor([0,1],dtype=torch.int32,device='cuda'),max_seqlen_k=actual,seqused_k=torch.tensor([actual],dtype=torch.int32,device='cuda'),block_table=table,softmax_scale=.0625,causal=True,fa_version=2,num_splits=0)
    attention.append(a.reshape(1,6144))
   a=torch.cat(attention);gated=a*torch.sigmoid(gate);out=_fused_mul_mat_gguf(gated,*weights[3]);hashes={}
   for name,t in [('input',x),('qg',qg),('ki',ki),('vi',vi),('q',q),('k',k),('v',v),('gate',gate),('attention',a),('gated',gated),('output',out)]:hashes[name]=save(f'{stem}-t{step}-{name}.bin',t)
   for row,capacity in enumerate(capacities):
    for role,t in zip(['key','value'],caches[row][:2]):hashes[f'r{row}-{role}']=save(f'{stem}-t{step}-r{row}-{role}.bin',t.view(-1)[:capacity*1024])
   cases.append(dict(batch=batch,seed=seed,step=step,positions=positions.copy(),capacities=capacities,sha256=hashes));positions=[pos+1 for pos in positions];print(stem,step,flush=True)
provenance['model_path']=str(args.model);provenance['torch_version']=torch.__version__
for name,fn in [('linear',_fused_mul_mat_gguf),('qk',fused_qk_rmsnorm_rope_gate),('attention',flash_attn_varlen_func)]:
 provenance[name+'_source_sha256']=hashlib.sha256(Path(inspect.getfile(inspect.unwrap(fn))).read_bytes()).hexdigest()
provenance['extension_sha256']=hashlib.sha256((Path(inspect.getfile(__import__('vllm_gguf_plugin'))).parent/'_C_gguf.abi3.so').read_bytes()).hexdigest()
(p/'cases.json').write_text(json.dumps(cases,indent=2));(p/'provenance.json').write_text(json.dumps(provenance,indent=2))
