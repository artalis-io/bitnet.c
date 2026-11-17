"""Unchanged compiled GLM F32-gamma/bias LayerNorm D128, BF16 input/output."""
from pathlib import Path
import argparse,hashlib,inspect,json
import torch
from vllm.models.glm5next.common.attention import _fused_indexer_k_norm
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--header',type=Path,required=True);a=ap.parse_args();a.directory.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
def save(name,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==b,name
 else:f.write_bytes(b)
 h=14695981039346656037
 for v in b:h=((h^v)*1099511628211)&((1<<64)-1)
 return h
cases=[]
for nr in [1,2,3,17,32,33,129,1025,65537]:
 for seed in [0,1]:
  i=torch.arange(nr*128,dtype=torch.int64);row=i//128;x=((i*37+i//13+seed*11)%509-254).float()/128
  mode=(row+seed)%8;x=torch.where(mode==0,0.,x);x=torch.where(mode==1,1.,x);x=torch.where(mode==2,4096.+x*32,x);x=torch.where(mode==3,x*2**-20,x);x=torch.where(mode==4,x*2**16,x);x=torch.where(mode==5,-0.,x);x=x.reshape(nr,128).bfloat16().cuda()
  j=torch.arange(128,dtype=torch.int64);w=(((j*7+seed*3)%31-15).float()/16+1+2**-17).cuda();b=(((j*11+seed*5)%61-30).float()/128+2**-19).cuda()
  for ei,eps in enumerate([1e-6,1e-5]):
   out=_fused_indexer_k_norm(x,w,b,128,eps);again=_fused_indexer_k_norm(x,w,b,128,eps);assert torch.equal(out,again)and torch.isfinite(out).all()
   rounded=_fused_indexer_k_norm(x,w.bfloat16().float(),b.bfloat16().float(),128,eps)
   name=f'norm-{nr}-{seed}-{ei}.bin';cases.append(dict(rows=nr,seed=seed,eps_index=ei,eps=eps,hash=save(name,out),rounded_mismatches=int((out!=rounded).sum())))
   print(nr,seed,ei,'captured twice',flush=True)
assert sum(c['rounded_mismatches']for c in cases)>0
lines=['/* Unchanged compiled GLM F32 LayerNorm; preserve both row layouts. */','#ifndef BN_CUDA_GLM_LAYERNORM_REFERENCE_H','#define BN_CUDA_GLM_LAYERNORM_REFERENCE_H','typedef struct {int rows,seed,eps_index;float eps;uint64_t hash;} BnGLMLayerNormReference;','static const BnGLMLayerNormReference glm_layernorm_reference[] = {']
for c in cases:lines.append(' {%d,%d,%d,%.9ef,UINT64_C(0x%016x)},'%tuple(c[k]for k in ['rows','seed','eps_index','eps','hash']))
a.header.write_text('\n'.join(lines+['};','#endif'])+'\n');source=Path(inspect.getfile(inspect.unwrap(_fused_indexer_k_norm)));(a.directory/'reference.json').write_text(json.dumps(dict(cases=cases,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,header_sha256=hashlib.sha256(a.header.read_bytes()).hexdigest()),indent=2)+'\n')
