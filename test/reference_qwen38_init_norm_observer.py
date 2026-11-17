import os,json,hashlib,time
from pathlib import Path
import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
_original_bn_run=CachingAutotuner.run
_bn_trace=Path('/evidence')/('init-norm-'+str(os.getpid())+'.jsonl')
_bn_sources=Path('/evidence/init-kernel-sources');_bn_sources.mkdir(exist_ok=True)
def _bn_observe_norm(self,*args,**kwargs):
 result=_original_bn_run(self,*args,**kwargs)
 if 'rms_norm' in self.fn.__name__:
  source=self.fn.src;key=hashlib.sha256(source.encode()).hexdigest();dest=_bn_sources/(key+'.py')
  if not dest.exists():dest.write_text(source)
  launcher=self.launchers[0];tensors={str(i):dict(pointer=t.data_ptr(),shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))for i,t in enumerate(args)if isinstance(t,torch.Tensor)}
  record=dict(pid=os.getpid(),timestamp_ns=time.perf_counter_ns(),capturing=torch.cuda.is_current_stream_capturing(),autotuner_id=id(self),filename=str(getattr(self,'filename',None)),kernel=self.fn.__name__,source=key,config=launcher.config.kwargs,num_warps=launcher.config.num_warps,cache_hash=launcher.cache_hash,tensors=tensors,scalar_args={str(i):v for i,v in enumerate(args)if isinstance(v,(int,float,bool))})
  with _bn_trace.open('a')as f:f.write(json.dumps(record)+'\n')
 return result
CachingAutotuner.run=_bn_observe_norm
