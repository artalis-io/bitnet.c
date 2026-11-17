"""Independent Q4_K/Q6_K BF16 A8 reference preparation for the next format milestone.
Pinned vLLM/plugin dispatch is unchanged. Raw weights use reference_cuda_bf16_q4q6.c; inputs use
the common signed-MMQ fixture formula. No native Q4_K/Q6_K BF16 result is used.
"""
import torch,numpy as np,json,argparse
from pathlib import Path
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;cases=[]
def save(name,data):
 f=p/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
for typ in [12,14]:
 shapes=[(r,256,s,nt) for r in [33,129] for s in [0,1] for nt in [1,2,3,5,6,7,9,17,33,129]]
 shapes += [(r,256,0,nt) for r in [5120,5121] for nt in [2,3,6,7]]
 shapes += [(33,k,0,nt) for k in [768,5120,6144] for nt in [1,3,6,7,17,129]]
 shapes += [(1,256,0,65537)]

 for rows,cols,seed,nt in shapes:
  w=torch.from_numpy(np.fromfile(p/f'w-{typ}-{rows}-{cols}-{seed}.bin',np.uint8).reshape(rows,-1)).cuda()
  i=torch.arange(nt*cols);x=(((i*17+i//37+seed*11)%509-254).float()/128).to(torch.bfloat16).reshape(nt,cols).cuda()
  # The reference quantizer has a 65535-row grid-Y limit. Both chunks
  # retain MMQ dispatch and exactly the same per-token arithmetic.
  y=(_fused_mul_mat_gguf(x,w,typ) if nt<=65535 else torch.cat([_fused_mul_mat_gguf(x[:32768],w,typ),_fused_mul_mat_gguf(x[32768:],w,typ)])).cpu().contiguous();data=y.view(torch.uint8).numpy().tobytes();name=f'ref-{typ}-{rows}-{cols}-{seed}-{nt}.bin';save(name,data)
  h=14695981039346656037
  for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
  cases.append(dict(type=typ,rows=rows,cols=cols,seed=seed,tokens=nt,hash=h,file=name))
save('cases.json',json.dumps(cases,indent=2).encode())
lines=['/* Pinned vLLM image 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29,', ' * GGUF plugin e454affe, unchanged Q4_K/Q6_K A8 precision and dispatch.', ' * Whole-output independent hashes; no cuBLAS. */', '#ifndef BN_CUDA_BF16_Q4Q6_REFERENCE_H', '#define BN_CUDA_BF16_Q4Q6_REFERENCE_H', 'static const BnBF16DispatchReference cuda_bf16_q4q6_cases[] = {']
for c in cases:lines.append('    {%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['type'],c['rows'],c['cols'],c['seed'],c['tokens'],c['hash'],c['hash']))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_q4q6_reference.h').write_text('\n'.join(lines)+'\n');print('Q4_K/Q6_K references',len(cases),flush=True)
