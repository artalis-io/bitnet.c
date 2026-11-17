"""Independent natural-order CPU coefficient roundtrip for every Flash GDN layer."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import torch,gguf
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
m=json.loads(a.inventory.read_text());cases=[]
for shard in m['shards']:
 reader=gguf.GGUFReader(shard['path'])
 for t in reader.tensors:
  match=re.fullmatch(r'blk\.(\d+)\.ssm_a',t.name)
  if not match:continue
  x=torch.from_numpy(np.array(gguf.dequantize(np.asarray(t.data),t.tensor_type),copy=True)).reshape(48);assert(x<0).all()
  y=torch.log(-x);assert torch.isfinite(y).all();data=y.numpy().astype('<f4').tobytes();name=f"a-log-{int(match[1])}.bin";f=a.directory/name
  if f.exists():assert f.read_bytes()==data,name
  else:f.write_bytes(data)
  cases.append(dict(layer=int(match[1]),file=name,sha256=hashlib.sha256(data).hexdigest()))
assert len(cases)==36
(a.directory/'manifest.json').write_text(json.dumps(cases,indent=2)+'\n');print('FLASH_COEFFICIENTS_CAPTURED',len(cases),flush=True)
