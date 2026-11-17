"""Extract original full-model FFN norm edge rows; no native expected output.
Usage: extract_looped_add_norm_reference.py CAPTURE_ROOT OPERAND_ROOT OUTPUT
CAPTURE_ROOT is the immutable original FlashInfer/Triton full-model capture.
OPERAND_ROOT contains independently replayed GGUF embedding and adapter gamma.
"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
q=Path(sys.argv[1]);p=Path(sys.argv[2]);out=Path(sys.argv[3]);assert json.loads((q/'reference-check.json').read_text())['all_raw_logits_identical'];assert json.loads((p/'embedding-check.json').read_text())['different_words']==0;assert json.loads((p/'gamma-check.json').read_text())['exact']
rs=json.loads((q/'prefill-dispatch-records.json').read_text())
def tensor(index,label):
 r=next(t for t in rs[index]['tensors']if t['label']==label);b=(q/'capture'/r['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256'];assert r['dtype']=='torch.bfloat16';return np.frombuffer(b,np.uint16).reshape(r['shape'])
x=tensor(3,'output');y=tensor(4,'args.0');r=np.fromfile(p/'embedding-reference.bin',np.uint16).reshape(1025,5120);w=np.fromfile(p/'checkpoint-gamma.bin',np.uint16);assert w.shape==(5120,)
f=lambda a:(a.astype(np.uint32)<<16).view(np.float32);s=(f(x)+f(r)).astype(np.float32).view(np.uint32);s=((s+0x7fff+((s>>16)&1))>>16).astype(np.uint16)
rows=[49,63,0,1024];lines=['/* Original compiled Qwen FFN0 norm, independent model/embedding/gamma.',' * Rows49/63 exercise reduction and mean rounding, original model1025tokens.',' * Selected reference R0_BLOCK1024,num_warps8; see extractor. */','#ifndef BN_CUDA_LOOPED_ADD_NORM_REFERENCE_H','#define BN_CUDA_LOOPED_ADD_NORM_REFERENCE_H']
for name,a in [('weight',w),('x',x[rows]),('residual',r[rows]),('out',y[rows]),('sum',s[rows])]:
 lines.append('static const uint16_t cuda_looped_add_norm_'+name+''.join('['+str(v)+']'for v in a.shape)+' = {')
 for row in ([a]if a.ndim==1 else a):
  if a.ndim>1:lines.append(' {')
  for j in range(0,len(row),16):lines.append(' '+','.join(f'0x{v:04x}'for v in row[j:j+16])+',')
  if a.ndim>1:lines.append(' },')
 lines.append('};')
# Add a natural layer-7 edge row that distinguishes the reversed first FMA.
manifest = {v['file']: v for v in json.loads((p/'stage-manifest-v3.json').read_text())}
def stage(layer, role):
 name = f'l{layer}-{role}.bin'
 b = (p/'reference-stages'/name).read_bytes()
 assert hashlib.sha256(b).hexdigest() == manifest[name]['sha256']
 return np.frombuffer(b,np.uint16).reshape(1025,5120)
def rounded(a):
 b = a.astype(np.float32).view(np.uint32)
 return ((b+0x7fff+((b>>16)&1))>>16).astype(np.uint16)
r7 = r.copy()
for layer in range(7):
 r7 = rounded((f(stage(layer,'block-output'))+f(r7)).astype(np.float32)+f(stage(layer,'ffn-output')))
x7 = stage(7,'block-output'); y7 = stage(7,'ffn-norm')
w7 = np.fromfile(p/'all-post-gamma.bin',np.uint16).reshape(64,5120)[7]
s7 = rounded(f(x7)+f(r7)); rows7 = [323,0,49,1024]
for name,a in [('weight',w7),('x',x7[rows7]),('residual',r7[rows7]),('out',y7[rows7]),('sum',s7[rows7])]:
 lines.append('static const uint16_t cuda_looped_add_norm_edge_'+name+''.join('['+str(v)+']'for v in a.shape)+' = {')
 for row in ([a]if a.ndim==1 else a):
  if a.ndim>1:lines.append(' {')
  for j in range(0,len(row),16):lines.append(' '+','.join(f'0x{v:04x}'for v in row[j:j+16])+',')
  if a.ndim>1:lines.append(' },')
 lines.append('};')
lines.append('#endif');out.write_text('\n'.join(lines)+'\n');print(out,hashlib.sha256(out.read_bytes()).hexdigest())
