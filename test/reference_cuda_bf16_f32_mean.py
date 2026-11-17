"""Unchanged pinned vLLM IR mixed-gamma F32 mean references."""
from pathlib import Path
import torch,json,hashlib,argparse
import vllm.ir as ir
from vllm.config import VllmConfig,set_current_vllm_config
import vllm._custom_ops
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True);cases=[]
def save(name,t):
 b=t.cpu().contiguous().view(torch.uint8).numpy().tobytes();f=p/name
 if f.exists():assert f.read_bytes()==b,name
 else:f.write_bytes(b)
 h=14695981039346656037
 for c in b:h=((h^c)*1099511628211)&((1<<64)-1)
 return h,hashlib.sha256(b).hexdigest()
with set_current_vllm_config(VllmConfig()):
 for nt in [1,2,3,4,5,7,8,9,15,16,17,129,1047]:
  for offset in [0,1]:
   seed=nt+offset*17;i=torch.arange(nt*5120,dtype=torch.int64)
   x=torch.ldexp(((i*37+seed*11)%509-254).float()/128,(i%13-6).to(torch.int32)).to(torch.bfloat16).reshape(nt,5120).cuda()
   r=torch.ldexp(((i*13+i//17+seed*3)%251-125).float()/64,(i%11-5).to(torch.int32)).to(torch.bfloat16).reshape(nt,5120).cuda()
   j=torch.arange(5120,dtype=torch.int64);w=(((j*7+seed)%127-63).float()/64).to(torch.bfloat16).cuda();gamma=w.float()+offset
   for add in [0,1]:
    y,rr=ir.ops.fused_add_rms_norm(x,r,gamma,1e-6) if add else (ir.ops.rms_norm(x,gamma,1e-6),None)
    key=f'{nt}-{offset}-{add}';h,sh=save(key+'.bin',y)
    rh,rsh=save(key+'-res.bin',rr) if add else (0,None)
    cases.append(dict(tokens=nt,offset=offset,add=add,seed=seed,hash=h,residual_hash=rh,file=key+'.bin',sha256=sh,residual_file=key+'-res.bin'if add else None,residual_sha256=rsh))
  print('captured nt',nt,flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
lines=['#ifndef BN_CUDA_BF16_F32_MEAN_REFERENCE_H','#define BN_CUDA_BF16_F32_MEAN_REFERENCE_H','typedef struct {int tokens,offset,add,seed;uint64_t hash,residual_hash;} BnBF16F32MeanReference;','static const BnBF16F32MeanReference cuda_bf16_f32_mean_cases[] = {']
for c in cases:lines.append(' {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(c[k]for k in ['tokens','offset','add','seed','hash','residual_hash']))
lines+=['};','#endif'];(p/'reference.h').write_text('\n'.join(lines)+'\n')
