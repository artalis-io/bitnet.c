"""Original SM120 scores from independently captured GLM pre-indexer/cache.
No candidate implementation is imported. Uninitialized compressed-logit padding
is masked per query before serializing. Every score capture is repeated.
"""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np
import torch,vllm
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
from vllm.models.glm5next.nvidia.ops.kpool_compress import mqa_logits_use_mx_scales
ap=argparse.ArgumentParser();ap.add_argument('--preindexer',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);assert not mqa_logits_use_mx_scales()
pre=json.loads((a.preindexer/'reference.json').read_text());cache=json.loads((a.cache/'reference.json').read_text());assert cache['preindexer_manifest_sha256']==hashlib.sha256((a.preindexer/'reference.json').read_bytes()).hexdigest()
def load(d,n,typ,shape):return torch.from_numpy(np.fromfile(d/(n+'.bin'),typ).copy()).reshape(shape).cuda()
records=[];schedules=[str(len(pre['records']))]
for case,cc in zip(pre['records'],cache['records'],strict=True):
 assert case['case']==cc['case'];phases=[];schedules.append(' '.join(map(str,[case['case'],len(case['schedule']),*case['schedule']])))
 for pi,(phase,cp) in enumerate(zip(case['phases'],cc['phases'],strict=True)):
  nt,pos=phase['tokens'],phase['position'];assert nt==cp['tokens'] and pos==cp['position'];nk=(pos+nt)//4;d=a.preindexer/str(case['case'])/str(pi);cd=a.cache/str(case['case'])/str(pi)
  q=load(d,'q_fp8',np.uint8,(nt,32,128)).view(torch.float8_e4m3fn);k=load(cd,'gather',np.uint8,(nk,128)).view(torch.float8_e4m3fn);sc=load(cd,'scales',np.float32,(nk,));w=load(d,'head_weights',np.float32,(nt,32));end=((torch.arange(nt,device='cuda',dtype=torch.int32)+pos+1)//4).contiguous();start=torch.zeros_like(end);mask=torch.arange(nk,device='cuda')[None,:]<end[:,None]
  def run():
   if not nk:return torch.empty((nt,0),device='cuda',dtype=torch.float32)
   result=fp8_fp4_mqa_logits((q,None),(k,sc),w,start,end,nk)
   return torch.where(mask,result[:,:nk],-torch.inf).contiguous()
  x,y=run(),run();assert torch.equal(x.view(torch.uint8),y.view(torch.uint8));b=x.view(torch.uint8).cpu().numpy().tobytes();dest=a.out/str(case['case'])/str(pi);dest.mkdir(parents=True,exist_ok=True);f=dest/'scores.bin'
  if f.exists():assert f.read_bytes()==b
  else:f.write_bytes(b)
  phases.append(dict(position=pos,tokens=nt,keys=nk,score_elements=nt*nk,sha256=hashlib.sha256(b).hexdigest()))
 records.append(dict(case=case['case'],schedule=case['schedule'],phases=phases));print(case['case'],case['schedule'],'original scores captured twice',flush=True)
(a.out/'schedules.txt').write_text('\n'.join(schedules)+'\n');sources={}
for f in [Path(inspect.getfile(fp8_fp4_mqa_logits)),Path(vllm.__file__).parent/'third_party/deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh']:
 sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(a.out/'reference.json').write_text(json.dumps(dict(records=records,preindexer_manifest_sha256=hashlib.sha256((a.preindexer/'reference.json').read_bytes()).hexdigest(),cache_manifest_sha256=hashlib.sha256((a.cache/'reference.json').read_bytes()).hexdigest(),sources=sources,profile='Original SM120 signed/F32-scale causal scores on original actual-GGUF pre-indexer and persistent paged-cache history; no candidate intermediate injection.'),indent=2)+'\n')
