"""Independent pinned Triton scatter oracle, preserving every BF16 bit pattern."""
import argparse,hashlib,inspect,json
from pathlib import Path
import numpy as np
import torch
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_store_cache_rows,_store_qsa_rows_kernel
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def blob(t):return t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
def save(name,data):
 f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return hashlib.sha256(data).hexdigest()
def bf(data):return torch.from_numpy(data.astype(np.uint16)).view(torch.bfloat16).cuda()
def hashed(data):
 h=14695981039346656037
 for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
 return h
shapes=[(1,1,8),(17,16,512),(33,32,128),(129,7,2048),(4097,16,512),(257,16,512),(1,65536,8),(3,1,65536),(32,16,8),(65,32,512)]
rows=[]
for ci,(nt,page,width)in enumerate(shapes):
 blocks=(nt+8+page-1)//page+1;capacity=blocks*page
 i=np.arange(nt*width,dtype=np.uint64);k=bf(i*17+ci*31).reshape(nt,1,width);v=bf(i*37+ci*13+0x8000).reshape(nt,1,width)
 j=np.arange(capacity*width,dtype=np.uint64);kc=bf(j*5+7).reshape(blocks,page,1,width);vc=bf(j*11+19).reshape(blocks,page,1,width)
 slots=[]
 for r in range(nt):
  slot=capacity-r if r else 0
  if r and r%11==0:slot=-1
  elif r and r%13==0:slot=-(1<<31)
  elif r and r%17==0:slot=capacity
  elif r and r%19==0:slot=(1<<31)-1
  slots.append(slot)
 slots=torch.tensor(slots,device='cuda',dtype=torch.int32)
 qsa_store_cache_rows(kc,slots,k);qsa_store_cache_rows(vc,slots,v)
 kb,vb=blob(kc),blob(vc)
 row=dict(tokens=nt,page=page,width=width,blocks=blocks,k_hash=hashed(kb),v_hash=hashed(vb),k_sha=save(f'{ci}-k.bin',kb),v_sha=save(f'{ci}-v.bin',vb))
 rows.append(row);print('case',ci,nt,page,width,flush=True)
source=inspect.getsource(qsa_store_cache_rows)+_store_qsa_rows_kernel.src
save('manifest.json',(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),cases=rows),indent=2)+'\n').encode())
lines=['#ifndef BN_CUDA_PAGED_KV_APPEND_REFERENCE_H','#define BN_CUDA_PAGED_KV_APPEND_REFERENCE_H','/* Pinned Triton cache scatter; raw BF16 bit-pattern hashes. */','static const uint64_t cuda_paged_kv_append_hashes[][2] = {']
for r in rows:lines.append(' {UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(r['k_hash'],r['v_hash']))
lines+=['};','#endif'];save('reference.h',('\n'.join(lines)+'\n').encode())
