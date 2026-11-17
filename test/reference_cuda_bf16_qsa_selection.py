"""Observe pinned QSA scores, top-k sets/order and packed causal expansion.
Reference inputs are immutable independent checkpoint pre-indexer captures.
Native qualification must recompute those inputs, not inject these artifacts.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import vllm.models.qwen4_exp.nvidia.ops.qsa_indexer as ops
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('pre_indexer',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def save(name,x):
 b=x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==b,name
 else:f.write_bytes(b)
 return dict(file=name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),shape=list(x.shape),dtype=str(x.dtype))
def load(file,shape):return torch.from_numpy(np.fromfile(a.pre_indexer/file,dtype=np.uint16).copy()).view(torch.bfloat16).reshape(shape).cuda()
original=ops._topk;rows=[];name='';observed={}
def hook(logits,visible,token_topk,ratio,out,workspace):
 global observed
 mask=torch.arange(logits.shape[1],device='cuda')[None,:]<visible[:,None]
 observed['scores']=save(name+'-scores.bin',torch.where(mask,logits,-torch.inf))
 original(logits,visible,token_topk,ratio,out,workspace)
 first=out.clone();order_changes=set();set_changes=set();invalid=set();suboptimal=set()
 k=token_topk//ratio
 # CPU exact ranking of OBSERVED logits checks selection without replacing it.
 raw=logits.cpu().numpy();lengths=visible.cpu().numpy()
 first_cpu=first.cpu().numpy()
 for repeat in range(12):
  alternate=torch.empty_like(out);original(logits,visible,token_topk,ratio,alternate,workspace)
  x=alternate.cpu().numpy()
  for r,n in enumerate(lengths):
   count=min(int(n),k);v=x[r,:count];base=first_cpu[r,:count]
   if not np.array_equal(v,base):order_changes.add(r)
   if not np.array_equal(np.sort(v),np.sort(base)):set_changes.add(r)
   if len(set(v.tolist()))!=count or (v<0).any() or (v>=n).any():invalid.add(r);continue
   if count and not np.array_equal(np.sort(raw[r,v]),np.sort(raw[r,:n])[-count:]):suboptimal.add(r)
 observed.update(order_changed_rows=len(order_changes),set_changed_rows=len(set_changes),invalid_rows=sorted(invalid),suboptimal_rows=sorted(suboptimal),original=save(name+'-groups.bin',first),replays=12)
ops._topk=hook
m=json.loads((a.pre_indexer/'manifest.json').read_text())
for case in m['cases']:
 if case['name'] not in ['1-1','129-129','2053-2053','2053-257','4097-4097']:continue
 for phase in case['phases']:
  if phase['phase'] not in [0,len(case['phases'])-1]:continue
  pos,take=phase['position'],phase['take'];visible_end=(pos+take)//4;name=f"{case['name']}-{phase['phase']}"
  q=load(phase['stages']['q']['file'],(take,4,128));flat=load(phase['stages']['compressed']['file'],(visible_end,128))
  pages=max(1,(visible_end+15)//16);cache=torch.zeros(pages+1,16,1,128,device='cuda',dtype=torch.bfloat16);cache.reshape(-1,128)[16:16+visible_end]=flat
  table=torch.arange(1,pages+1,device='cuda',dtype=torch.int32).reshape(1,-1)
  positions=torch.arange(pos,pos+take,device='cuda',dtype=torch.int64);visible=((positions+1)//4).int();groups=torch.empty(take,512,device='cuda',dtype=torch.int32)
  observed={}
  if take==1:ops.qsa_select_paged_decode(q,cache,table,visible,2048,4,1,groups)
  else:ops.qsa_select_paged_prefill(q,cache,table,torch.tensor([0,take],device='cuda',dtype=torch.int32),visible,2048,4,take,groups,pos+take)
  packed=torch.empty(take,2052,device='cuda',dtype=torch.int32);ops.expand_qsa_block_indices(groups,positions,visible,4,2048,packed)
  observed.update(name=name,position=pos,tokens=take,packed=save(name+'-packed.bin',packed))
  rows.append(observed);print(name,'order_changed',observed['order_changed_rows'],'set_changed',observed['set_changed_rows'],'invalid',len(observed['invalid_rows']),'suboptimal',len(observed['suboptimal_rows']),flush=True)
# Fully initialized tied logits isolate tie policy from unused-logit padding.
ops._topk=original
for n in [0,1,511,512,513,1024,1025,8192,8193]:
 name=f'tie-{n}';width=max(64,(n+63)//64*64);logits=torch.full((1,width),-torch.inf,device='cuda');logits[:,:n]=0
 out=torch.empty(1,512,device='cuda',dtype=torch.int32);observed={};hook(logits,torch.tensor([n],device='cuda',dtype=torch.int32),2048,4,out,torch.empty(1<<20,device='cuda',dtype=torch.uint8))
 observed.update(name=name,tokens=1,visible=n);rows.append(observed);print(name,'order_changed',observed['order_changed_rows'],'set_changed',observed['set_changed_rows'],'invalid',observed['invalid_rows'],flush=True)
save('manifest.json',torch.tensor(list((json.dumps(rows,indent=2)+'\n').encode()),dtype=torch.uint8))
