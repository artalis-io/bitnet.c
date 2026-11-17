"""Generate Q3_K BF16 A8 references using the unchanged pinned plugin.

Image: sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
Plugin: e454affe0ca23c9dc8754ec3033270cb30622ed5, wheel SHA256
 d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef
Torch: 2.14.0a0+4fdf77b940.nv26.8.63802676
Generate raw weights with reference_cuda_bf16_q3.c first. Reference inputs are
synthetic and use the shared signed-MMQ fixture formula. Q3_K dispatch uses
MMVQ up to 6 tokens (2 above output width 5120), and MMQ for larger batches.
The 65537-token case concatenates two MMQ calls because the reference activation
quantizer has a grid-Y limit; each token uses the unchanged MMQ arithmetic.
Both digest fields use the same independent reference: Q3_K A8 uses no cuBLAS.
"""
import torch,numpy as np,json,argparse
from pathlib import Path
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;cases=[]
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
for typ in [11]:
 shapes=[(r,256,s,nt) for r in [33,129] for s in [0,1] for nt in [1,2,3,5,6,7,9,17,33,129]]
 shapes += [(r,256,0,nt) for r in [5120,5121] for nt in [2,3,6,7]]
 shapes += [(33,k,0,nt) for k in [768,5120] for nt in [1,3,6,7,17,129]]
 shapes += [(1,256,0,65537)]
 for rows,cols,seed,nt in shapes:
  w=torch.from_numpy(np.fromfile(p/f'w-{typ}-{rows}-{cols}-{seed}.bin',np.uint8).reshape(rows,-1)).cuda()
  i=torch.arange(nt*cols);x=(((i*17+i//37+seed*11)%509-254).float()/128).to(torch.bfloat16).reshape(nt,cols).cuda()
  # The reference quantizer has a 65535-row grid-Y limit. Both chunks
  # retain MMQ dispatch and exactly the same per-token arithmetic.
  y=(_fused_mul_mat_gguf(x,w,typ) if nt<=65535 else torch.cat([_fused_mul_mat_gguf(x[:32768],w,typ),_fused_mul_mat_gguf(x[32768:],w,typ)])).cpu().contiguous();data=y.view(torch.uint8).numpy().tobytes();name=f'ref-{typ}-{rows}-{cols}-{seed}-{nt}.bin';(p/name).write_bytes(data)
  h=14695981039346656037
  for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
  cases.append(dict(type=typ,rows=rows,cols=cols,seed=seed,tokens=nt,hash=h,file=name))
(p/'cases.json').write_text(json.dumps(cases,indent=2))
lines=['/* Pinned vLLM GGUF plugin e454affe, Torch 2.14.0a0+4fdf77b940,', ' * image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.', ' * Whole-output hashes from the unchanged Q3_K A8 reference; no cuBLAS.', ' * Reference linear dispatch and precision are unmodified. */', '#ifndef BN_CUDA_BF16_Q3_REFERENCE_H', '#define BN_CUDA_BF16_Q3_REFERENCE_H', 'static const BnBF16DispatchReference cuda_bf16_q3_cases[] = {']
alternate=cases
assert len(alternate)==len(cases)
for c,a in zip(cases,alternate):
 assert {k:v for k,v in c.items() if k!='hash'}=={k:v for k,v in a.items() if k!='hash'}
 lines.append('    {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['type'],c['rows'],c['cols'],c['seed'],c['tokens'],c['hash'],a['hash']))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_dispatch_reference.h').write_text('\n'.join(lines)+'\n');print('reference cases',len(cases),'words',sum(c['rows']*c['tokens'] for c in cases),flush=True)
