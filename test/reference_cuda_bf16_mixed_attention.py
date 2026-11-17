"""Independent unchanged paged FA2 mixed-query oracle; no model required.

Run in the pinned vLLM image on the qualified GPU. This writes a new fixture,
never replaces existing singleton references, and never runs in unit tests.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

ap = argparse.ArgumentParser()
ap.add_argument('directory', type=Path)
p = ap.parse_args().directory
p.mkdir(parents=True, exist_ok=True)
assert torch.cuda.device_count() == 1


def raw(t):
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h


def values(count, role, seed, start=0):
    i = torch.arange(start, start + count, dtype=torch.int64)
    if role == 0:
        x = ((i * 17 + seed * 11) % 509 - 254).float() / 128
    elif role == 1:
        x = ((i * 7 + seed * 3) % 251 - 125).float() / 64
    else:
        x = ((i * 13 + seed * 5) % 257 - 128).float() / 32
    return x.to(torch.bfloat16).cuda()


specs = [([1,5],[135,5]),
 ([1]*4+[5]*4,[135,136,137,138]+[5]*4),
 ([1]*4+[63]*4,[65,66,67,68]+[63]*4),
 ([1,1,1,3],[65,129,257,3]),
 ([1]*4,[65,129,257,513]),
 ([3,5,2],[129,257,65]),
 ([1,7,1,3],[1025,1025,257,513])]
rows=[]
for case,(queries,lengths) in enumerate(specs):
 batch=len(lengths);seed=31+case
 heads,kv_heads,dim,page=24,4,256,256
 maximum=max(lengths);pages=(maximum+page-1)//page
 k=torch.empty((batch*pages,page,kv_heads,dim),dtype=torch.bfloat16,device='cuda')
 v=torch.empty_like(k);k.view(torch.int16).fill_(0x5a5a);v.view(torch.int16).fill_(0x5a5a)
 q=torch.empty((sum(queries),heads,dim),dtype=torch.bfloat16,device='cuda')
 table=torch.arange(batch*pages,dtype=torch.int32,device='cuda').reshape(batch,pages)
 offsets=[0]
 for row,(n,length) in enumerate(zip(queries,lengths)):
  rs=seed+row*7
  q[offsets[-1]:offsets[-1]+n].copy_(values(n*heads*dim,0,rs,(length-n)*heads*dim).reshape(n,heads,dim))
  offsets.append(offsets[-1]+n)
  for role,cache in [(1,k),(2,v)]:
   cache[row*pages:(row+1)*pages].view(-1,kv_heads,dim)[:length].copy_(values(length*kv_heads*dim,role,rs).reshape(length,kv_heads,dim))
 cu=torch.tensor(offsets,dtype=torch.int32,device='cuda')
 used=torch.tensor(lengths,dtype=torch.int32,device='cuda')
 def execute(q,cu,used,table,mq,mk):
  return flash_attn_varlen_func(q,k,v,mq,cu,max_seqlen_k=mk,seqused_k=used,block_table=table,softmax_scale=.0625,causal=True,fa_version=2,num_splits=0)
 out=execute(q,cu,used,table,max(queries),maximum)
 data=raw(out)
 assert data==raw(execute(q,cu,used,table,max(queries),maximum)) and torch.isfinite(out).all()
 batch_hashes=[fnv(data[offsets[i]*heads*dim*2:offsets[i+1]*heads*dim*2]) for i in range(batch)]
 singleton=[]
 for i,n in enumerate(queries):
  owncu=torch.tensor([0,n],dtype=torch.int32,device='cuda')
  own=execute(q[offsets[i]:offsets[i+1]],owncu,used[i:i+1],table[i:i+1],n,lengths[i])
  assert torch.isfinite(own).all()
  singleton.append(raw(own))
 singledata=b''.join(singleton)
 for name,blob in [('batch',data),('singleton',singledata)]:
  path=p/f'case-{case}-{name}.bin'
  if path.exists():assert path.read_bytes()==blob
  else:path.write_bytes(blob)
 rows.append(dict(batch=batch,seed=seed,queries=queries,lengths=lengths,batch_hashes=batch_hashes,singleton_hashes=[fnv(x) for x in singleton],batch_sha256=hashlib.sha256(data).hexdigest(),singleton_sha256=hashlib.sha256(singledata).hexdigest()))
 print(case,queries,lengths,flush=True)
(p/'reference.json').write_text(json.dumps(rows,indent=2)+'\n')
header=['/* Independent pinned vLLM FA2 ragged query cohort and singleton modes. */','#include <stdint.h>',
 'typedef struct { int batch,seed,queries[8],lengths[8]; uint64_t singleton_hashes[8],batch_hashes[8]; } BnBF16MixedAttentionReference;',
 'static const BnBF16MixedAttentionReference bf16_mixed_attention_cases[] = {']
for r in rows:
 fields=[str(r['batch']),str(r['seed'])]
 for name in ['queries','lengths']:fields.append('{'+','.join(map(str,r[name]))+'}')
 for name in ['singleton_hashes','batch_hashes']:fields.append('{'+','.join('UINT64_C(0x%016x)'%h for h in r[name])+'}')
 header.append('{'+','.join(fields)+'},')
header.append('};')
(p/'reference.h').write_text('\n'.join(header)+'\n')
props=torch.cuda.get_device_properties(0)
(p/'provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),image='sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',device=props.name,multiprocessors=props.multi_processor_count,torch=torch.__version__,cuda=torch.version.cuda,source='Unchanged public flash_attn_varlen_func,fa_version=2,num_splits=0'),indent=2)+'\n')
