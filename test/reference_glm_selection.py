"""Preserve raw original GLM top-k and expansion (including arrival order).
Actual untied pool membership is compared exactly; raw order is never rewritten.
Short-mode output uses the original causal-index fill rather than sparse packing.
"""
from pathlib import Path
import argparse,hashlib,inspect,json
import torch,numpy as np
import vllm
from vllm.models.glm5next.common.sparse_indexer import _fill_causal_indices
from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail
ap=argparse.ArgumentParser();ap.add_argument('--scores',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);meta=json.loads((a.scores/'reference.json').read_text());records=[]
for c in meta['records']:
 phases=[]
 for pi,v in enumerate(c['phases']):
  nt,pos,nk=v['tokens'],v['position'],v['keys'];d=a.scores/str(c['case'])/str(pi);raw=np.fromfile(d/'scores.bin',np.float32).copy().reshape(nt,nk);assert hashlib.sha256(raw.tobytes()).hexdigest()==v['sha256'];scores=torch.from_numpy(raw).cuda();start=torch.zeros(nt,device='cuda',dtype=torch.int32);end=(torch.arange(nt,device='cuda',dtype=torch.int32)+pos+1)//4;positions=torch.arange(pos,pos+nt,device='cuda',dtype=torch.int64);short=int(pos+nt<=2048);results=[]
  for rep in range(2):
   if short:
    pools=torch.empty((nt,0),device='cuda',dtype=torch.int32);out=torch.full((nt,2051),-1,device='cuda',dtype=torch.int32);_fill_causal_indices(out,positions)
   else:
    pools=torch.empty((nt,512),device='cuda',dtype=torch.int32)
    if nk:torch.ops._C.top_k_per_row_prefill(scores,start,end,pools,nt,scores.stride(0),scores.stride(1),512)
    else:pools.fill_(-1)
    out=expand_pools_and_append_tail(pools.to(torch.int64),positions.int()+1,4)
   torch.cuda.synchronize();results.append((pools.cpu().numpy(),out.cpu().numpy()))
  assert np.array_equal(np.sort(results[0][0],axis=1),np.sort(results[1][0],axis=1))
  assert np.array_equal(np.sort(results[0][1],axis=1),np.sort(results[1][1],axis=1));dest=a.out/str(c['case'])/str(pi);dest.mkdir(parents=True,exist_ok=True);hashes={}
  for rep,(pools,out)in enumerate(results):
   for n,x in [('pools',pools),('expanded',out)]:
    name=n+('-repeat'if rep else'');b=x.tobytes();(dest/(name+'.bin')).write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
  phases.append(dict(position=pos,tokens=nt,short_prefill=short,hashes=hashes,raw_repeat_exact=all(np.array_equal(x,y)for x,y in zip(*results)),token_membership_repeat_exact=True))
 records.append(dict(case=c['case'],phases=phases));print(c['case'],'original selection/expansion captured twice',flush=True)
sources={}
for f in [_fill_causal_indices,expand_pools_and_append_tail]:
 s=Path(inspect.getfile(inspect.unwrap(f)));sources[str(s)]=hashlib.sha256(s.read_bytes()).hexdigest()
(a.out/'reference.json').write_text(json.dumps(dict(records=records,sources=sources,vllm_extension_sha256=hashlib.sha256((Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so').read_bytes()).hexdigest(),score_manifest_sha256=hashlib.sha256((a.scores/'reference.json').read_bytes()).hexdigest(),profile='Raw original short/sparse index lists, preserved arrival order. Untied membership is exact; native deterministic ordering/ties require downstream MLA qualification.'),indent=2)+'\n')
