"""Independent Torch BF16 merged-FFN oracle; deterministic synthetic tensors."""
from pathlib import Path
import json,hashlib
import torch

def main():
 p=Path('/evidence');assert torch.cuda.device_count()==1
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 d,h=32,64
 i=torch.arange(h,dtype=torch.int64)[:,None];j=torch.arange(d,dtype=torch.int64)[None,:]
 g=(((i*3+j*5)%17-8).float()/16).to(torch.bfloat16).cuda()
 u=(((i*7+j*11)%19-9).float()/16).to(torch.bfloat16).cuda()
 i=torch.arange(d,dtype=torch.int64)[:,None];j=torch.arange(h,dtype=torch.int64)[None,:]
 down=(((i*13+j*3)%23-11).float()/32).to(torch.bfloat16).cuda();weight=torch.cat([g,u],dim=0)
 @torch.compile(fullgraph=True)
 def activation(merged):return torch.nn.functional.silu(merged[:,:h])*merged[:,h:]
 rows=[]
 for nt in [3,24,32]:
  t=torch.arange(nt,dtype=torch.int64)[:,None];j=torch.arange(d,dtype=torch.int64)[None,:]
  x=(((t*5+j*7)%13-6).float()/8).to(torch.bfloat16).cuda()
  out=activation(x@weight.T)@down.T
  data=out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/f'output-{nt}.bin').write_bytes(data)
  value=14695981039346656037
  for b in data:value=((value^b)*1099511628211)&((1<<64)-1)
  rows.append(dict(tokens=nt,hash=value,shape=list(out.shape),sha256=hashlib.sha256(data).hexdigest()))
 props=torch.cuda.get_device_properties(0)
 (p/'reference.json').write_text(json.dumps(dict(method='Unmodified Torch BF16 combined GEMM, Inductor-compiled SiLU/product, BF16 down GEMM',dim=d,hidden=h,allow_bf16_reduced_precision_reduction=False,hardware=dict(name=props.name,sm=props.major*10+props.minor),torch_cuda=torch.version.cuda,cases=rows),indent=2)+'\n')
 print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
