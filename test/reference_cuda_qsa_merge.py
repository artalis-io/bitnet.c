"""Pinned QSA merge replay and independent synthetic inputs; no replaced math.
Run with pinned NVIDIA image and corrected main-attention captures.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_merge_splitk_kernel
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('checkpoint',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def run(partial,lse,gate):
 splits,rows,heads,dim=partial.shape;assert heads==24 and dim==256
 out=torch.empty_like(gate)
 binary=_qsa_merge_splitk_kernel[(rows,24)](partial,lse,out,gate,out.stride(0),out.stride(1),gate.stride(0),gate.stride(1),rows,HEAD_DIM=256,NUM_QUERY_HEADS=24,NUM_SPLITS=splits,BLOCK_SPLITS=splits,num_warps=2,num_stages=1)
 cubin=a.directory/f'merge-{splits}.cubin'
 if cubin.exists():assert cubin.read_bytes()==binary.asm['cubin']
 else:cubin.write_bytes(binary.asm['cubin'])
 (a.directory/f'merge-{splits}-metadata.txt').write_text(str(binary.metadata))
 return out
manifest=json.loads((a.checkpoint/'manifest.json').read_text());replays=[]
for case in manifest['cases']:
 for phase in case['phases']:
  st=phase['stages']
  if 'partial'not in st:continue
  def load(k,dtype):
   e=st[k];b=(a.checkpoint/e['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==e['sha256'];t=torch.from_numpy(np.frombuffer(b,dtype=dtype).copy())
   if dtype==np.uint16:t=t.view(torch.bfloat16)
   return t.reshape(e['shape']).cuda()
  out=run(load('partial',np.float32),load('lse',np.float32),load('gate',np.uint16));gold=load('attention',np.uint16)
  n=int((out.view(torch.int16)!=gold.view(torch.int16)).sum());assert not n,(case['name'],phase['phase'],n)
  replays.append(dict(case=case['name'],phase=phase['phase'],words=out.numel()))
records=[]
for mode,splits in [(mode,splits)for mode in ['ordinary','extreme']for splits in [4,8,16,64]]:
 rows=3;i=torch.arange(splits*rows*24*256,device='cuda',dtype=torch.int64)
 partial=torch.ldexp(((i*37+i//29)%509-254).float()/128,(i%21-10).int()).reshape(splits,rows,24,256)
 j=torch.arange(splits*rows*24,device='cuda',dtype=torch.int64)
 lse=(((j*17)%127-63).float()/8).reshape(splits,rows,24)
 lse[:,0,:]=-torch.inf;partial[:,0,:,:]=0
 lse[1:,1,:]=-torch.inf;partial[1:,1,:,:]=0
 k=torch.arange(rows*24*256,device='cuda',dtype=torch.int64);gate=(((k*13)%513-256).float()/8).reshape(rows,24,256).bfloat16()
 if mode=='extreme':
  partial=torch.ldexp(((i*37+i//29)%509-254).float()/128,(i%201-100).int()).reshape(splits,rows,24,256)
  partial[:,0,:,:]=0;partial[1:,1,:,:]=0
  exponents=torch.tensor([0.,-1.,-125.,-126.,-127.,-148.,-149.,-2000.],device='cuda')
  lse[:,2,:]=exponents[(torch.arange(splits,device='cuda')[:,None]+torch.arange(24,device='cuda')[None,:])%8]
  gates=torch.tensor([-128.,128.,-0.,0.,88.,-88.,100.,-100.],device='cuda')
  gate=gates[k%8].reshape(rows,24,256).bfloat16()
 out=run(partial,lse,gate);assert torch.isfinite(out).all()
 record=dict(mode=mode,rows=rows,splits=splits,stages={})
 for name,t in [('partial',partial),('lse',lse),('gate',gate),('attention',out)]:
  data=t.cpu().contiguous().view(torch.uint8).numpy().tobytes();f=a.directory/f'{mode}-{splits}-{name}.bin'
  if f.exists():assert f.read_bytes()==data
  else:f.write_bytes(data)
  record['stages'][name]=dict(file=f.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 records.append(record)
manifest_file=a.directory/'manifest.json';data=json.dumps(dict(checkpoint_replays=replays,cases=records),indent=2)+'\n'
if manifest_file.exists():assert manifest_file.read_text()==data
else:manifest_file.write_text(data)
print('QSA_MERGE_REFERENCE',len(replays),'checkpoint replays',len(records),'synthetic cases',flush=True)
