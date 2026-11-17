"""Independent pinned NVIDIA PLE gate and eager-materialization comparison."""
import argparse,hashlib,inspect,json
from pathlib import Path
import torch
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_gate
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
def pattern(n,seed,role,denom=128):
 i=torch.arange(n,dtype=torch.int64)
 return torch.ldexp(((i*(13+role*6)+seed*11+role*19)%509-254).float()/denom,(i%7-3).int()).to(torch.bfloat16).cuda()
def save(name,x):
 data=x.cpu().view(torch.int16).numpy().astype('<i2').tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 h=14695981039346656037
 for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
 return h,hashlib.sha256(data).hexdigest()
def norm(x,w):
 f=x.float();return (f*torch.rsqrt((f*f).mean(-1,keepdim=True)+1e-6)*(1+w.float())).bfloat16()
cases=[]
for h in [2560,4096]:
 for hc in [1,4]:
  for nt in [1,5,129]:
   for seed in [0,1,2]:
    n=nt*hc*h;k=pattern(n,seed,0).reshape(nt,hc,h);v=pattern(nt*h,seed,1).reshape(nt,h)
    q=pattern(n,seed,2).reshape(nt,hc,h)
    if seed==1:q=-k
    if seed==2:q.zero_()
    weights=[pattern(hc*h,seed,r,4096).reshape(hc,h)for r in [3,4,5]]
    gated,normed=ple_gate(k.reshape(nt,hc*h),v,q.reshape(nt,hc*h),*[w.flatten()for w in weights],1e-6)
    kn=norm(k,weights[0]);qn=norm(q,weights[1]);products=(kn*qn).bfloat16()
    d=(products.float().sum(-1).bfloat16()/(h**0.5)).bfloat16()
    mag=d.abs().clamp(min=1e-6).sqrt().bfloat16();g=(d.sign()*mag).sigmoid().bfloat16()
    eg=(g[:,:,None]*v[:,None,:]).bfloat16();en=norm(eg,weights[2])
    name=f'{h}-{hc}-{nt}-{seed}'
    gh,gs=save(name+'-gated.bin',gated);nh,ns=save(name+'-normed.bin',normed)
    eh,es=save(name+'-eager-gated.bin',eg);fh,fs=save(name+'-eager-normed.bin',en)
    diff_g=int((eg.reshape_as(gated)!=gated).sum());diff_n=int((en.reshape_as(normed)!=normed).sum())
    cases.append(dict(h=h,hc=hc,nt=nt,seed=seed,gated=gh,normed=nh,eager_gated=eh,eager_normed=fh,gated_sha256=gs,normed_sha256=ns,eager_gated_sha256=es,eager_normed_sha256=fs,diff_g=diff_g,diff_n=diff_n))
    print(name,'eager differences',diff_g,diff_n,flush=True)
(a.directory/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(a.directory/'provenance.json').write_text(json.dumps(dict(torch=torch.__version__,source_sha256=hashlib.sha256(Path(inspect.getfile(ple_gate)).read_bytes()).hexdigest()),indent=2)+'\n')
lines=['#ifndef BN_CUDA_BF16_PLE_GATE_REFERENCE_H','#define BN_CUDA_BF16_PLE_GATE_REFERENCE_H','typedef struct {int h,hc,nt,seed;uint64_t gated,normed,eager_gated,eager_normed;} BnPLEGateReference;','static const BnPLEGateReference ple_gate_reference[] = {']
for c in cases:lines.append(' {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(c[k]for k in ['h','hc','nt','seed','gated','normed','eager_gated','eager_normed']))
lines+=['};','#endif'];(a.directory/'reference.h').write_text('\n'.join(lines)+'\n');print('CAPTURED',len(cases),flush=True)
