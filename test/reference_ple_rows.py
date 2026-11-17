"""Actual IQ4_NL PLE row references using pinned Triton IDs and GGUF decode.
Run in the pinned reference_ple_ids.py image with gguf-py on PYTHONPATH.
Only selected rows are copied; the complete table is never decoded/uploaded.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch,gguf
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
m=json.loads(a.inventory.read_text());meta=m['metadata'];k='qwen4exp.ple.'
reader=None;table=None
for shard in m['shards']:
 r=gguf.GGUFReader(shard['path'])
 for t in r.tensors:
  if t.name=='per_layer_token_embd.weight':reader=r;table=t;break
 if table is not None:break
assert table is not None and table.tensor_type==gguf.GGMLQuantizationType.IQ4_NL
assert list(map(int,table.shape))==[160,320001536]
def save(name,data):
 path=a.directory/name
 if path.exists():assert path.read_bytes()==data,name
 else:path.write_bytes(data)
 return hashlib.sha256(data).hexdigest()
cases=[]
for nt in [1,5,17,129]:
 nr=4 if nt>=5 else 1;seed=97
 tokens=[248044 if i%11==5 else (i*37+seed)%248000 for i in range(nt)]
 starts=[0,0,nt//3,nt//2,nt] if nr==4 else [0,nt]
 ctx=[248044 if i%3==0 else (i*19+seed)%251 for i in range(nr*2)]
 dev=lambda x:torch.tensor(x,dtype=torch.int64,device='cuda')
 ids=ple_ngram_ids(dev(tokens).int(),dev(starts).int(),dev(ctx).int().reshape(nr,2),
     dev(meta[k+'layer_multipliers']),dev(meta[k+'head_vocab_sizes']),dev(meta[k+'head_offsets']),248044,8).cpu().numpy().astype(np.int32)
 raw=np.asarray(table.data).reshape(-1,90)[ids.reshape(-1)]
 padded=np.zeros((raw.shape[0],96),dtype=np.uint8);padded[:,:90]=raw
 bf16=torch.from_numpy(gguf.dequantize(raw,table.tensor_type)).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
 cases.append(dict(tokens=nt,ids_sha256=save(f'{nt}-ids.bin',ids.tobytes()),
    packed_sha256=save(f'{nt}-packed.bin',padded.tobytes()),decoded_sha256=save(f'{nt}-bf16.bin',bf16),packed_bytes=padded.nbytes))
(a.directory/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');print('ACTUAL_ROWS_CAPTURED',cases,flush=True)
