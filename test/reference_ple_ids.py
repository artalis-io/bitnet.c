"""Pinned NVIDIA Triton PLE hash references (no native implementation).
Image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Includes empty requests, EOS boundaries, chunk contexts and signed overflow.
"""
import argparse,json,hashlib,inspect
from pathlib import Path
import torch
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
cases=[]
for ng in [2,3,8]:
 for hp in [1,8]:
  for nt in [1,5,17,129]:
   for profile in [0,1,2]:
    seed=ng*11+hp*7+nt;nr=4 if nt>=5 else 1;heads=(ng-1)*hp
    ids=[248044 if i%11==5 else (i*37+seed)%248000 for i in range(nt)]
    starts=[0,0,nt//3,nt//2,nt] if nr==4 else [0,nt]
    ctx=[248044 if i%3==0 else (i*19+seed)%251 for i in range(nr*(ng-1))]
    mul=[(23703573157769+i*1000003 if profile==0 else
          ((1<<64)-1-i*2 if profile==1 else (1<<63)+i*1000003))for i in range(ng)]
    signed=[v if v<(1<<63) else v-(1<<64) for v in mul]
    out=ple_ngram_ids(torch.tensor(ids,dtype=torch.int32,device='cuda'),
        torch.tensor(starts,dtype=torch.int32,device='cuda'),
        torch.tensor(ctx,dtype=torch.int32,device='cuda').reshape(nr,ng-1),
        torch.tensor(signed,dtype=torch.int64,device='cuda'),
        torch.tensor([101+2*h for h in range(heads)],dtype=torch.int64,device='cuda'),
        torch.tensor([1000*h for h in range(heads)],dtype=torch.int64,device='cuda'),248044,hp).to(torch.int32)
    data=out.cpu().contiguous().view(torch.uint8).numpy().tobytes();name=f'{ng}-{hp}-{nt}-{profile}.bin';path=a.directory/name
    if path.exists():assert path.read_bytes()==data,name
    else:path.write_bytes(data)
    h=14695981039346656037
    for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
    cases.append(dict(ng=ng,hp=hp,nt=nt,nr=nr,seed=seed,profile=profile,hash=h,sha256=hashlib.sha256(data).hexdigest()))
(a.directory/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
lines=['#ifndef BN_PLE_IDS_REFERENCE_H','#define BN_PLE_IDS_REFERENCE_H','typedef struct {int ng,hp,nt,nr,seed,profile;uint64_t hash;} BnPLEIdsReference;','static const BnPLEIdsReference ple_ids_reference[] = {']
for c in cases:lines.append(' {%d,%d,%d,%d,%d,%d,UINT64_C(0x%016x)},'%tuple(c[k]for k in ['ng','hp','nt','nr','seed','profile','hash']))
lines+=['};','#endif'];(a.directory/'reference.h').write_text('\n'.join(lines)+'\n');print('CAPTURED',len(cases),flush=True)

(a.directory/"provenance.json").write_text(json.dumps(dict(torch=torch.__version__, source_sha256=hashlib.sha256(Path(inspect.getfile(ple_ngram_ids)).read_bytes()).hexdigest()),indent=2)+"\n")
