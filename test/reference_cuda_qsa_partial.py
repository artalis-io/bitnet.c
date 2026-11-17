"""Pinned paged QSA stage reference: all dispatch profiles and masking.
Native tests consume explicit synthetic inputs, independently compare partials,
base-2 LSE and final gated BF16 output; no full-model qualification.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from vllm.models.qwen4_exp.nvidia.ops import qsa
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--timing',action='store_true');a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def save(name,x):
 data=x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),dtype=str(x.dtype),shape=list(x.shape))
def pattern(shape,role):
 n=1
 for v in shape:n*=v
 i=torch.arange(n,device='cuda',dtype=torch.int64)
 return torch.ldexp(((i*(17+role*4)+i//37+role*13)%509-254).float()/128,(i%5-2).int()).bfloat16().reshape(shape)
observed={};prefix='';compiled={};original=qsa._qsa_sparse_paged_gqa_splitk_kernel.run
def hook(*args,**kw):
 binary=original(*args,**kw)
 if kw['NUM_SPLITS']>1:
  observed['partial']=save(prefix+'-partial.bin',args[6]);observed['lse']=save(prefix+'-lse.bin',args[7])
 key=f"{kw['BLOCK_N']}-{kw['num_warps']}-{kw['NUM_SPLITS']}"
 compiled[key]=dict(metadata=str(binary.metadata),ptx_sha256=hashlib.sha256(binary.asm['ptx'].encode()).hexdigest())
 file=a.directory/f'{key}.ptx'
 if file.exists():assert file.read_text()==binary.asm['ptx']
 else:file.write_text(binary.asm['ptx'])
 return binary
qsa._qsa_sparse_paged_gqa_splitk_kernel.run=hook
pages=140;blocks=2*pages+3;table=torch.empty(2,pages,device='cuda',dtype=torch.int32)
for r in range(2):
 for logical in range(pages):table[r,logical]=1+r*pages+(logical*37+19)%pages
 table[r,17]=-1;table[r,31]=blocks+3;table[r,9]=0
keys=pattern((blocks,16,2,256),1);values=pattern(keys.shape,2)
cases=[];timings=[]
for rows,prefill in [(1,False),(13,True),(17,True),(64,True),(128,True),(129,True),(257,True),(1025,True),(1025,False)]:
 prefix=f'{rows}-{int(prefill)}';observed={}
 query=pattern((rows,24,256),3);gate=pattern(query.shape,4)
 requests=(torch.arange(rows,device='cuda',dtype=torch.int32)%2)
 if rows>2:requests[2]=-1
 if rows>3:requests[3]=2
 indices=torch.full((rows,2052),-1,device='cuda',dtype=torch.int32)
 counts=[0,1,31,32,33,63,64,65,127,128,129,2048,2051]
 for row in range(rows):
  count=2051 if rows==1 else counts[row%len(counts)]
  indices[row,:count]=torch.arange(count,device='cuda',dtype=torch.int32)
  if count>80:
   indices[row,79]=-2147483648;indices[row,80]=pages*16+17
  indices[row,-1]=count
 output=qsa.qsa_sparse_paged_attention(query,keys,values,indices,table,requests,prefill,output_gate=gate)
 assert torch.isfinite(output).all()
 stages={name:save(prefix+'-'+name+'.bin',x)for name,x in [('q',query),('keys',keys),('values',values),('indices',indices),('gate',gate),('table',table),('requests',requests),('attention',output)]}
 if a.timing:
  qsa._qsa_sparse_paged_gqa_splitk_kernel.run=original
  for _ in range(3):qsa.qsa_sparse_paged_attention(query,keys,values,indices,table,requests,prefill,output_gate=gate)
  torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):timed_output=qsa.qsa_sparse_paged_attention(query,keys,values,indices,table,requests,prefill,output_gate=gate)
  for _ in range(3):graph.replay()
  start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True);start.record()
  for _ in range(100):graph.replay()
  stop.record();stop.synchronize();us=start.elapsed_time(stop)*10
  timings.append(dict(name=prefix,warm_graph_us=us,iterations=100));print('PINNED_TIMING',prefix,us,flush=True)
  assert torch.equal(timed_output.view(torch.int16),output.view(torch.int16))
  qsa._qsa_sparse_paged_gqa_splitk_kernel.run=hook
 stages.update(observed);record=dict(name=prefix,rows=rows,prefill=prefill,config=qsa._select_config(rows,2,prefill,2051),pages=pages,blocks=blocks,num_requests=2,stages=stages);cases.append(record);print(prefix,record['config'],flush=True)
manifest=dict(profile='synthetic paged QSA with two requests, permuted/shared pages, invalid requests/pages/indices and boundary counts',cases=cases,kernels=compiled)
f=a.directory/'manifest.json';data=json.dumps(manifest,indent=2)+'\n'
if f.exists():assert f.read_text()==data
else:f.write_text(data)
if a.timing:(a.directory/'timings.json').write_text(json.dumps(timings,indent=2)+'\n')
print('QSA_PAGED_REFERENCE',len(cases),flush=True)
