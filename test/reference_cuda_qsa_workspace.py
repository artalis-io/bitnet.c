"""Independent workspace oracle from the pinned dispatch function, no GPU work."""
import argparse,hashlib,inspect,json
from pathlib import Path
from vllm.models.qwen4_exp.nvidia.ops.qsa import _select_config
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
caps=set(range(1,301))|{1025,4097,131070};peak=0;rows=[]
for n in range(1,max(caps)+1):
 for prefill in [False,True]:
  bn,warps,tiles,splits=_select_config(n,2,prefill,2051)
  count=n*splits*24*(256+1)*4 if splits>1 else 0
  peak=max(peak,count)
 if n in caps:rows.append((n,peak))
outputs={'workspace.txt':''.join(f'{n} {b}\n'for n,b in rows),'manifest.json':json.dumps(dict(source_sha256=hashlib.sha256(inspect.getsource(_select_config).encode()).hexdigest(),caps=rows),indent=2)+'\n'}
for name,data in outputs.items():
 f=a.directory/name
 if f.exists():assert f.read_text()==data
 else:f.write_text(data)
print('QSA_WORKSPACE_ORACLE',len(rows),'peak',peak,flush=True)
