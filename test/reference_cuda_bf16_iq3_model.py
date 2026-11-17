"""Independent IQ3_S projection and decoded BF16 weights from the actual Qwen27 checkpoint.

Pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,
unchanged GGUF plugin e454affe. Uses raw checkpoint rows and synthetic BF16
inputs in GGUF order; this does not test the surrounding FFN topology.
No native output supplies expected values. Existing expected bytes are immutable.
"""
import argparse,hashlib,json
from pathlib import Path
import gguf
import numpy as np
import torch
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
ap=argparse.ArgumentParser();ap.add_argument('model',type=Path);ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True);reader=gguf.GGUFReader(args.model);tensors={t.name:t for t in reader.tensors};cases=[]
def save(name,data):
 f=p/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return hashlib.sha256(data).hexdigest()
for key,name,typ in [('layer14-down','blk.14.ffn_down.weight',21)]:
 t=tensors[name];assert int(t.tensor_type)==typ;cols,rows=map(int,t.shape);data=np.array(t.data,copy=True).reshape(rows,-1);wf=key+'-weight.bin';wh=save(wf,data.tobytes());w=torch.from_numpy(data).cuda()
 for nt in [1,8,16,17,129]:
  i=torch.arange(nt*cols);x=(((i*17+i//37)%509-254).float()/128).to(torch.bfloat16).reshape(nt,cols).cuda();y=_fused_mul_mat_gguf(x,w,typ);assert torch.isfinite(y).all();prefix=f'{key}-{nt}';xh=save(prefix+'-input.bin',x.cpu().contiguous().view(torch.uint8).numpy().tobytes());yh=save(prefix+'-out.bin',y.cpu().contiguous().view(torch.uint8).numpy().tobytes());cases.append(dict(key=key,tensor=name,type=typ,rows=rows,cols=cols,tokens=nt,weight_file=wf,weight_sha256=wh,input_file=prefix+'-input.bin',input_sha256=xh,file=prefix+'-out.bin',sha256=yh));print(prefix,rows,cols,'captured',flush=True)
 from vllm_gguf_plugin import ops
 decoded=ops.ggml_dequantize(w,typ,rows,cols,torch.bfloat16);save(key+'-decoded.bin',decoded.cpu().contiguous().view(torch.uint8).numpy().tobytes());del decoded,w,data
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
