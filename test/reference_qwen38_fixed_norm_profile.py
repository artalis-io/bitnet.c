"""External diagnostic: retain independently observed original profile C choices."""
import sys, importlib.abc, importlib.machinery, json, hashlib, os
from pathlib import Path
import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
import triton
ROOT=Path('/evidence')
PROFILE=json.loads((ROOT/'profile.json').read_text())
POINTERS={}
LOCKED={}
RETAINED=[]
SEEN=set()
ORIGINAL=CachingAutotuner.run
TARGET='vllm.v1.worker.gpu.model_runner'
def event(kind,**data):
 with (ROOT/('profile-lock-'+str(os.getpid())+'.jsonl')).open('a')as f:
  f.write(json.dumps(dict(kind=kind,**data))+'\n')
def run(self,*args,**kwargs):
 if 'rms_norm' in self.fn.__name__ and POINTERS:
  names={POINTERS[int(t.data_ptr())]for t in args if isinstance(t,torch.Tensor)and int(t.data_ptr())in POINTERS}
  if names:
   assert len(names)==1,names
   name=next(iter(names));row=PROFILE[name]
   source=hashlib.sha256(self.fn.src.encode()).hexdigest()
   assert source==row['source'],(name,source,row['source'])
   signature=(tuple(sorted(row['config'].items())),row['num_warps'])
   if id(self)in LOCKED:assert LOCKED[id(self)]==signature,('shared-instance-conflict',name)
   else:
    assert not torch.cuda.is_current_stream_capturing(),('first-lock-during-capture',name)
    candidates=[r for r in self.compile_results if r.config.kwargs==row['config']and r.config.num_warps==row['num_warps']]
    if not candidates:
     from triton.compiler.compiler import CompiledKernel
     root=Path(row['filename']).parent.parent/'triton/0'/row['cache_hash']
     group=json.loads((root/('__grp__'+row['kernel']+'.json')).read_text())['child_paths']
     group={k:str(root/Path(v).name)for k,v in group.items()}
     meta=json.loads((root/(row['kernel']+'.json')).read_text())
     assert meta['num_warps']==row['num_warps']and meta['num_stages']==1
     cfg=triton.Config(row['config'],num_warps=row['num_warps'],num_stages=meta['num_stages'])
     compile_meta=self._create_compile_meta(cfg)
     from triton.compiler.compiler import ASTSource
     ast_source=ASTSource(self.fn,compile_meta['signature'],compile_meta['constants'],compile_meta['configs'][0])
     binary=CompiledKernel(ast_source,group,meta['hash'])
     from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult,TritonCompileResult
     static=StaticTritonCompileResult.can_statically_launch(binary,self.inductor_meta,self.triton_meta,self.heuristic_type)
     candidates=[StaticTritonCompileResult(static,cfg,self._create_compile_meta(cfg),self.inductor_meta)if static is not None else TritonCompileResult(binary,cfg,self._create_compile_meta(cfg),self.inductor_meta)]
    assert len(candidates)==1
    self.inductor_meta=dict(self.inductor_meta)
    self.inductor_meta['coordinate_descent_tuning']=False
    self.inductor_meta['combo_tuning_groups']=None
    self._cached_launcher=None
    self.compile_results=candidates
    self.launchers=[]
    self._make_launchers()
    assert len(self.launchers)==1
    launcher=self.launchers[0]
    assert launcher.config.kwargs==row['config']and launcher.config.num_warps==row['num_warps']
    assert launcher.cache_hash==row['cache_hash'],('compiled-binary-identity',name,launcher.cache_hash,row['cache_hash'])
    LOCKED[id(self)]=signature
    RETAINED.append(self)
    event('locked',weight=name,instance=id(self),config=row['config'],num_warps=row['num_warps'],source=source,cache_hash=launcher.cache_hash)
   SEEN.add(name)
 return ORIGINAL(self,*args,**kwargs)
CachingAutotuner.run=run
def stop(model):
 assert len(SEEN)==129,(len(SEEN),sorted(set(PROFILE)-SEEN))
 assert CachingAutotuner.run is run
 CachingAutotuner.run=ORIGINAL
 event('restored',weights=len(SEEN),instances=len(LOCKED))
 return dict(restored=True,weights=len(SEEN),instances=len(LOCKED),profile='previously-observed-original-C',universal_default_equivalence=False)
class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)if hasattr(self.original,'create_module')else None
 def exec_module(self,module):
  self.original.exec_module(module)
  cls=module.GPUModelRunner;original=cls.load_model
  def load(runner,*args,**kwargs):
   result=original(runner,*args,**kwargs)
   POINTERS.update({int(t.data_ptr()):name for name,t in runner.model.named_parameters()if name in PROFILE})
   assert len(POINTERS)==129,len(POINTERS)
   event('weights-mapped',count=len(POINTERS))
   cls.load_model=original
   return result
  cls.load_model=load
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=TARGET:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  assert spec and spec.loader
  spec.loader=Loader(spec.loader);sys.meta_path.remove(self)
  return spec
sys.meta_path.insert(0,Finder())
