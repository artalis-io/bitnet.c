"""Generate exact BF16 linear-dispatch references in the pinned vLLM image.

Image: sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
Plugin: e454affe0ca23c9dc8754ec3033270cb30622ed5, wheel SHA256
 d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef
Torch: 2.14.0a0+4fdf77b940.nv26.8.63802676
First generate raw weights using reference_cuda_bf16_iq3.c. Run this script
inside the image with the same wheel installed and gguf-py available. Reference
arithmetic and dispatch are unmodified. Shapes exercise both token and output-
width thresholds, partial rows and input widths not divisible by the plugin's
512-element activation padding. Output includes full BF16 data and byte hashes.
Run once with each pinned cuBLAS library version into separate directories; run
the primary version again with --alternate-cases <other-directory>/cases.json
for the combined header. This retains complete independently generated outputs
for both algorithm selections, without a floating-point tolerance or a runtime-
generated golden value.
"""
import torch,numpy as np,json,argparse
from pathlib import Path
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);parser.add_argument('--alternate-cases',type=Path);args=parser.parse_args();p=args.directory;cases=[]
def save(name,data):
 f=p/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
for typ in [21]:
 shapes=[(r,256,s,nt) for r in [33,129] for s in [0,1] for nt in [1,4,5,8,9,16,17,33,129]]
 shapes += [(r,256,0,nt) for r in [5120,5121] for nt in [8,9,16,17]]
 shapes += [(33,k,0,nt) for k in [768,5120] for nt in [1,5,16,17]]
 for rows,cols,seed,nt in shapes:
  w=torch.from_numpy(np.fromfile(p/f'w-{typ}-{rows}-{cols}-{seed}.bin',np.uint8).reshape(rows,-1)).cuda()
  i=torch.arange(nt*cols);x=(((i*17+i//37+seed*11)%509-254).float()/128).to(torch.bfloat16).reshape(nt,cols).cuda()
  y=_fused_mul_mat_gguf(x,w,typ).cpu().contiguous();data=y.view(torch.uint8).numpy().tobytes();name=f'ref-{typ}-{rows}-{cols}-{seed}-{nt}.bin';save(name,data)
  h=14695981039346656037
  for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
  cases.append(dict(type=typ,rows=rows,cols=cols,seed=seed,tokens=nt,hash=h,file=name))
save('cases.json',json.dumps(cases,indent=2).encode())
lines=['/* Pinned vLLM GGUF plugin e454affe, Torch 2.14.0a0+4fdf77b940,', ' * image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.', ' * Exact whole-output hashes for independent Torch runs on cuBLAS 13.7/13.10.', ' * Reference linear dispatch and precision are unmodified. */', '#ifndef BN_CUDA_BF16_IQ3_REFERENCE_H', '#define BN_CUDA_BF16_IQ3_REFERENCE_H',  'static const BnBF16DispatchReference cuda_bf16_iq3_cases[] = {']
alternate=json.loads(args.alternate_cases.read_text()) if args.alternate_cases else cases
assert len(alternate)==len(cases)
for c,a in zip(cases,alternate):
 assert {k:v for k,v in c.items() if k!='hash'}=={k:v for k,v in a.items() if k!='hash'}
 lines.append('    {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['type'],c['rows'],c['cols'],c['seed'],c['tokens'],c['hash'],a['hash']))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_iq3_reference.h').write_text('\n'.join(lines)+'\n');print('reference cases',len(cases),'words',sum(c['rows']*c['tokens'] for c in cases),flush=True)
