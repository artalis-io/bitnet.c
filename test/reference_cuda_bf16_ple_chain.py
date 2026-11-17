"""Actual-checkpoint decoded-BF16 PLE composition; no replaced intermediates.
Pinned NVIDIA image and gguf-py required. Eager gate contract, pinned conv.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch,gguf
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids,ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
assert NULL_BLOCK_ID!=1
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
m=json.loads(a.inventory.read_text());meta=m['metadata'];readers=[gguf.GGUFReader(x['path'])for x in m['shards']];tensors={t.name:t for r in readers for t in r.tensors}
def save(name,x):
 data=x.detach().cpu().view(torch.int16).numpy().astype('<i2').tobytes() if isinstance(x,torch.Tensor) else x
 f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
identities={}
def weight(role):
 t=tensors['blk.1.ple_'+role+'.weight'];raw=np.asarray(t.data)
 offset=1.0 if role.startswith('norm_') else 0.0
 w=(torch.from_numpy(np.array(gguf.dequantize(raw,t.tensor_type),copy=True)).float()-offset).bfloat16().cuda()
 identities[role]=dict(gemma_storage_offset=offset,type=t.tensor_type.name,shape=list(w.shape),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),decoded=save(role+'-weight.bin',w))
 return w
wk=weight('key');wv=weight('value');nk=weight('norm_key').reshape(4,2560);nq=weight('norm_query').reshape(4,2560);nc=weight('norm_conv').reshape(4,2560);wc=weight('conv1d').reshape(10240,4)
merged=torch.cat([wk,wv]);table=tensors['per_layer_token_embd.weight'];assert table.tensor_type==gguf.GGMLQuantizationType.IQ4_NL
rawtable=np.asarray(table.data).reshape(-1,90)
def pattern(n,role):
 i=torch.arange(n,dtype=torch.int64)
 return torch.ldexp(((i*(13+role*6)+role*19)%509-254).float()/128,(i%7-3).int()).bfloat16().cuda()
def norm(x,w):
 f=x.float();return (f*torch.rsqrt((f*f).mean(-1,keepdim=True)+1e-6)*(1+w.float())).bfloat16()
def gate(k,v,q):
 k=k.reshape(-1,4,2560);q=q.reshape_as(k)
 products=(norm(k,nk)*norm(q,nq)).bfloat16()
 d=(products.float().sum(-1).bfloat16()/(2560**.5)).bfloat16()
 g=(d.sign()*d.abs().clamp(min=1e-6).sqrt().bfloat16()).sigmoid().bfloat16()
 gated=(g[:,:,None]*v[:,None,:]).bfloat16()
 return gated.flatten(1),norm(gated,nc).flatten(1)
cases=[];dev=lambda x:torch.tensor(x,device='cuda',dtype=torch.int64)
for nt,chunk in [(1,1),(5,5),(17,17),(129,129),(129,64),(129,1)]:
 name=f'{nt}-{chunk}';tokens=[248044 if i%11==5 else(i*37+97)%248000 for i in range(nt)]
 ids=ple_ngram_ids(dev(tokens).int(),dev([0,nt]).int(),dev([248044,116]).int().reshape(1,2),dev(meta['qwen4exp.ple.layer_multipliers']),dev(meta['qwen4exp.ple.head_vocab_sizes']),dev(meta['qwen4exp.ple.head_offsets']),248044,8).cpu().numpy().astype('<i4')
 raw=rawtable[ids.reshape(-1)];e=torch.from_numpy(gguf.dequantize(raw,table.tensor_type)).bfloat16().cuda().reshape(nt,2560)
 hidden=pattern(nt*10240,2).reshape(nt,10240);initial=pattern(9*10240,4).reshape(9,10240)
 state=initial.T.contiguous().unsqueeze(0).repeat(2,1,1);idx=dev([1]).int();has=dev([1]).bool()
 stages={k:[]for k in ['key','value','gated','normed','out']};split_differences=0
 for start in range(0,nt,chunk):
  stop=min(nt,start+chunk);emb=e[start:stop];kv=F.linear(emb,merged);key,value=kv.split([10240,2560],-1)
  split_differences+=int((key!=F.linear(emb,wk)).sum())+int((value!=F.linear(emb,wv)).sum())
  gated,normed=gate(key,value,hidden[start:stop]);out=gated.clone()
  ple_conv(normed,out,state,wc,idx,hidden[start:stop],mode='decode'if stop-start==1 else'prefill',dilation=3,query_start_loc=dev([0,stop-start]).int(),has_initial_states=has)
  for k,x in [('key',key),('value',value),('gated',gated),('normed',normed),('out',out)]:stages[k].append(x)
 record=dict(name=name,nt=nt,chunk=chunk,split_projection_differences=split_differences,stages={})
 record['stages']['ids']=save(name+'-ids.bin',ids.tobytes())
 for k,x in [('embedding',e),('hidden',hidden),('initial',initial),('history',state[1].T)]+[(k,torch.cat(x))for k,x in stages.items()]:record['stages'][k]=save(name+'-'+k+'.bin',x)
 cases.append(record);print(name,'captured; split projection differences',split_differences,flush=True)
(a.directory/'manifest.json').write_text(json.dumps(dict(profile='decoded-BF16 weights, merged F.linear, eager gate, NVIDIA conv',weights=identities,cases=cases),indent=2)+'\n')
print('PLE_CHAIN_CAPTURED',len(cases),flush=True)
