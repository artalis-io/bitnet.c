"""Apply original MLA cache writer to preserved original projection outputs."""
from pathlib import Path
import argparse,torch,json,hashlib
from vllm import _custom_ops as ops
p=argparse.ArgumentParser();p.add_argument('projection_reference',type=Path);p.add_argument('directory',type=Path);a=p.parse_args();src=a.projection_reference;out=a.directory;out.mkdir(parents=True,exist_ok=True)
records=[]
for ci,nt in enumerate([1,2,3,7,17,65,129,1025]):
 b=(src/str(ci)/'kv.bin').read_bytes();x=torch.frombuffer(bytearray(b),dtype=torch.bfloat16).reshape(nt,512).cuda();blocks=(nt+63)//64+2;slots=(torch.arange(nt,device='cuda',dtype=torch.int64)*37+11)%(blocks*64)
 def run():
  c=torch.full((blocks,64,656),0xa5,device='cuda',dtype=torch.uint8);ops.concat_and_cache_mla(x,torch.empty((nt,0),device='cuda',dtype=torch.bfloat16),c,slots,'fp8_ds_mla',torch.ones(1,device='cuda'));torch.cuda.synchronize();return c.cpu().numpy().tobytes()
 b=run();assert b==run();f=out/f'{ci}.bin'
 if f.exists():assert f.read_bytes()==b
 else:f.write_bytes(b)
 records.append(dict(tokens=nt,blocks=blocks,sha256=hashlib.sha256(b).hexdigest()));print(ci,nt,flush=True)
(out/'reference.json').write_text(json.dumps(records,indent=2)+'\n')
