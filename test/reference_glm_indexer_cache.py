from pathlib import Path
import argparse,hashlib,json,torch,numpy as np,gguf
import vllm
from vllm import _custom_ops
from vllm.models.glm5next.nvidia.ops import kpool_compress as ops
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--preindexer',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);assert not ops.mqa_logits_use_mx_scales();manifest=json.loads((a.preindexer/'reference.json').read_text())
r=gguf.GGUFReader(str(a.model).replace('00001-of','00002-of'));t=next(t for t in r.tensors if t.name=='blk.3.indexer_compressor_ape.weight');raw=np.array(t.data,copy=True);assert hashlib.sha256(raw.tobytes()).hexdigest()==manifest['weights']['indexer_compressor_ape.weight'];ape=torch.from_numpy(raw).reshape(4,128).cuda()
def load(d,n,nt):return torch.from_numpy(np.fromfile(d/(n+'.bin'),dtype=np.uint16).copy()).view(torch.bfloat16).reshape(nt,128).cuda()
def save(f,x):
 b=x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b,str(f)
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
records=[]
for case in manifest['records']:
 total=sum(case['schedule']);blocks=((total//4)+63)//64+2;table=torch.arange(blocks-1,-1,-1,dtype=torch.int32).reshape(1,blocks).cuda()
 def run():
  cache=torch.full((blocks,64,132),0xa5,dtype=torch.uint8,device='cuda');values=[]
  for pi,phase in enumerate(case['phases']):
   pos,nt=phase['position'],phase['tokens'];d=a.preindexer/str(case['case'])/str(pi);key=load(d,'key',nt);gate=load(d,'gate',nt);tail=torch.from_numpy(np.fromfile(d/'initial_tail.bin',np.uint16).copy()).view(torch.bfloat16).reshape(2,4,128).cuda();visible=(pos+nt)//4;groups=visible-pos//4;ks=[];gs=[]
   for gi in range(groups):
    start=(pos//4+gi)*4;ks.append(torch.stack([key[t-pos]if t>=pos else tail[0,t%4]for t in range(start,start+4)]));gs.append(torch.stack([gate[t-pos]if t>=pos else tail[1,t%4]for t in range(start,start+4)]))
   if groups:
    logical=torch.arange(pos//4,visible,device='cuda',dtype=torch.int64);slots=(table[0,logical//64].long()*64+logical%64).contiguous();ops.kpool_compress_and_write_cache(cache,torch.stack(ks),torch.stack(gs),ape,slots,4)
   out=torch.empty((visible,128),device='cuda',dtype=torch.float8_e4m3fn);scales=torch.empty((visible,4),device='cuda',dtype=torch.uint8)
   if visible:_custom_ops.cp_gather_indexer_k_quant_cache(cache,out,scales,table,torch.tensor([0,visible],device='cuda',dtype=torch.int32))
   torch.cuda.synchronize();values.append(dict(cache=cache.clone(),gather=out,scales=scales))
  return values
 first,second=run(),run();phases=[]
 for pi,(x,y)in enumerate(zip(first,second)):
  assert all(torch.equal(v.view(torch.uint8),y[n].view(torch.uint8))for n,v in x.items());d=a.out/str(case['case'])/str(pi);d.mkdir(parents=True,exist_ok=True);phases.append(dict(**{k:case['phases'][pi][k]for k in ['position','tokens']},hashes={n:save(d/(n+'.bin'),v)for n,v in x.items()}))
 records.append(dict(case=case['case'],blocks=blocks,phases=phases));print(case['case'],'original cache chain captured twice',flush=True)
(a.out/'reference.json').write_text(json.dumps(dict(records=records,preindexer_manifest_sha256=hashlib.sha256((a.preindexer/'reference.json').read_bytes()).hexdigest(),pool_source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),vllm_extension_sha256=hashlib.sha256((Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so').read_bytes()).hexdigest()),indent=2)+'\n')
