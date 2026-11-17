"""Unchanged PyTorch hc_contract and original vLLM RMSNorm references."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch
from vllm.model_executor.layers.mhc import hc_contract
from vllm import _custom_ops as ops
p=argparse.ArgumentParser();p.add_argument('--header',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def fnv(b):
 h=14695981039346656037
 for x in b:h=((h^x)*1099511628211)&((1<<64)-1)
 return h
def save(f,b):
 if f.exists():assert f.read_bytes()==b,f
 else:f.write_bytes(b)
def data(nt,mode):
 i=np.arange(nt*4*4096,dtype=np.uint32);z=i+np.uint32(7193);z^=z<<np.uint32(13);z^=z>>np.uint32(17);z^=z<<np.uint32(5)
 if mode==0:return((z&0x807f)|(((z>>8)%50+100)<<7)).astype(np.uint16).reshape(nt,4,4096)
 if mode==2:return np.array([0,0x8000,1,0x8001,0x7f,0x807f,0x80,0x8080],dtype=np.uint16)[z%8].reshape(nt,4,4096)
 patterns=np.array([[0x6400,0x3f80,0xe400,0x3f80],[0x6400,0xe400,0x3f80,0x3f80],[0x3f80,0x6400,0x3f80,0xe400],[0x8000]*4,[0]*4,[0x0001]*4],dtype=np.uint16)
 row=i//16384;stream=i//4096%4;col=i%4096
 return patterns[(row+col)%6,stream].reshape(nt,4,4096)
j=np.arange(4096,dtype=np.uint32);gamma=((0x3f00+(j*13)%256)|np.where(j%5==0,0x8000,0)).astype(np.uint16)
w=torch.from_numpy(gamma).view(torch.bfloat16).cuda();save(a.directory/'gamma.bin',gamma.tobytes());records=[];lines=['/* Independent original PyTorch/vLLM captures; no native outputs. */','#ifndef BN_CUDA_GLM_CONTRACT_REFERENCE_H','#define BN_CUDA_GLM_CONTRACT_REFERENCE_H','typedef struct { int tokens,mode; uint64_t input,mean,norm; } BnCudaGLMContractReference;','static const BnCudaGLMContractReference cuda_glm_contract_refs[] = {']
for nt in [1,2,9,31,32,33,255,256,257]:
 for mode in range(3):
  raw=data(nt,mode);x=torch.from_numpy(raw).view(torch.bfloat16).cuda()
  def run():
   mean=hc_contract(x,4);norm=torch.empty_like(mean);ops.rms_norm(norm,mean,w,1e-5);return[mean.contiguous().view(torch.uint8).cpu().numpy().tobytes(),norm.contiguous().view(torch.uint8).cpu().numpy().tobytes()]
  first=run();assert first==run();values=[raw.tobytes(),*first];hashes=[]
  for name,b in zip(['input','mean','norm'],values):save(a.directory/f'{nt}-{mode}-{name}.bin',b);hashes.append(fnv(b))
  lines.append(' {%d,%d,UINT64_C(%d),UINT64_C(%d),UINT64_C(%d)},'%(nt,mode,*hashes))
  records.append(dict(tokens=nt,mode=mode,fnv=hashes,sha256=[hashlib.sha256(b).hexdigest()for b in values]));print(nt,mode,'captured twice',flush=True)
lines+=['};','#endif'];save(a.header,('\n'.join(lines)+'\n').encode());save(a.metadata,(json.dumps(dict(records=records,torch=torch.__version__,profile='hc_contract BF16 mean along4; original custom RMSNorm4096 eps1e-5, BF16 shared gamma, default non-batch-invariant launch'),indent=2)+'\n').encode())
