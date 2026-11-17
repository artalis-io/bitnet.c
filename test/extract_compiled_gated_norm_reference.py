"""Extract immutable gated-norm rows from the independent full-model capture.
Usage: python3 extract_compiled_gated_norm_reference.py CAPTURE_ROOT OUTPUT_HEADER
CAPTURE_ROOT is /tmp/bitnet-goal-batch-reference from the full-model audit.
No native runtime/kernel output is used as an expected value.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
p=Path(sys.argv[1]);out=Path(sys.argv[2]);q=p/'core-reference';d=p/'dispatch-reference'
for root in [q,d]:
 assert json.loads((root/'reference-check.json').read_text())['all_raw_logits_identical']
def tensor(root,row):
 data=(root/'capture'/row['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'];assert row['dtype']=='torch.bfloat16';return np.frombuffer(data,np.uint16).reshape(row['shape'])
records=json.loads((q/'dispatch-records.json').read_text());core=[r for r in records if 'qwen_gdn_attention_core'in r['op']];assert len(core)==48
x=tensor(q,next(r for r in core[-1]['tensors']if r['label']=='args.3')).reshape(192,128)
records=json.loads((d/'dispatch-records.json').read_text());assert records[0]['weights'][0][1]=='model.layers.62.linear_attn.in_proj_qkvz.weight';assert records[2]['weights'][0][1]=='model.layers.62.linear_attn.out_proj.weight'
z=tensor(d,next(r for r in records[0]['tensors']if r['label']=='output'))[:,10240:].reshape(192,128)
y=tensor(d,next(r for r in records[2]['tensors']if r['label']=='args.0')).reshape(4,3,16,128).transpose(0,2,1,3).copy().reshape(192,128)
w=np.fromfile(p/'norm-weight.bin',np.uint16);assert w.shape==(128,)
rows=[0,1,47,48,82,96,184,191]
lines=['/* Frozen original compiled Qwen layer62 gated norm. See extractor.',' * SM120, max batch128, Inductor XBLOCK8/2warps; rows82/184 exercise',' * exponential and reduction rounding. Inputs/outputs captured from original',' * full model with all raw logits unchanged by observation. */','#ifndef BN_CUDA_COMPILED_GATED_NORM_REFERENCE_H','#define BN_CUDA_COMPILED_GATED_NORM_REFERENCE_H']
def emit(name,a):
 a=np.ascontiguousarray(a);dims=''.join(f'[{v}]'for v in a.shape);lines.append(f'static const uint16_t {name}{dims} = {{');flat=a.reshape(-1)
 if a.ndim==1:
  for start in range(0,len(flat),16):lines.append(' '+','.join(f'0x{v:04x}'for v in flat[start:start+16])+',')
 else:
  for row in a:
   lines.append(' {')
   for start in range(0,len(row),16):lines.append('  '+','.join(f'0x{v:04x}'for v in row[start:start+16])+',')
   lines.append(' },')
 lines.append('};')
emit('cuda_compiled_gated_weight',w)
for name,a in [('x',x),('z',z),('out',y)]:emit('cuda_compiled_gated_'+name,a[rows])
lines.append('#endif');out.write_text('\n'.join(lines)+'\n')
print('wrote',out,hashlib.sha256(out.read_bytes()).hexdigest())
