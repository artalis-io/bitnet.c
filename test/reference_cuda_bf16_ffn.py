"""Generate independent resident-FFN reference outputs using pinned vLLM ops.

Image: sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
Plugin: e454affe0ca23c9dc8754ec3033270cb30622ed5, wheel SHA256
 d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef
Torch: 2.14.0a0+4fdf77b940.nv26.8.63802676
Generate raw weights with reference_cuda_bf16_ffn.c first. Uses the unchanged
plugin linear dispatcher and torch.ops._C.silu_and_mul, which rounds SiLU to
BF16 before multiplying. Outputs full FFN and activation-intermediate bytes,
shape manifests and an exact whole-output hash header. Covered shapes produce
identical references with both the image's cuBLAS 13.7 and host cuBLAS 13.10.
No runtime implementation output is used as a golden value.
"""
import torch,numpy as np,json,argparse
from pathlib import Path
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
import vllm._custom_ops
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
cases=[]
for dim,hid in [(256,256),(256,512),(768,512),(256,5376)]:
 for types in [(23,11,23),(11,23,20),(20,20,11),(23,23,23)]:
  for seed in [0,1]:
   for nt in [1,5,9,17,33]:
    def weight(typ,rows,cols,s):return torch.from_numpy(np.fromfile(p/f'w-{typ}-{rows}-{cols}-{s}.bin',np.uint8).reshape(rows,-1)).cuda()
    g=weight(types[0],hid,dim,seed);u=weight(types[1],hid,dim,1-seed);d=weight(types[2],dim,hid,seed)
    i=torch.arange(nt*dim);x=(((i*17+i//37+seed*11)%509-254).float()/4096).to(torch.bfloat16).reshape(nt,dim).cuda()
    gate=_fused_mul_mat_gguf(x,g,types[0]);up=_fused_mul_mat_gguf(x,u,types[1]);mid=torch.empty_like(gate);torch.ops._C.silu_and_mul(mid,torch.cat([gate,up],-1));y=_fused_mul_mat_gguf(mid,d,types[2]);assert torch.isfinite(y).all()
    name=f'ffn-{dim}-{hid}-{types[0]}-{types[1]}-{types[2]}-{seed}-{nt}';data=y.cpu().contiguous().view(torch.uint8).numpy().tobytes();(p/(name+'.bin')).write_bytes(data);(p/(name+'-mid.bin')).write_bytes(mid.cpu().contiguous().view(torch.uint8).numpy().tobytes());h=14695981039346656037
    for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
    cases.append(dict(dim=dim,hidden=hid,gate=types[0],up=types[1],down=types[2],seed=seed,tokens=nt,hash=h,file=name+'.bin'))
(p/'cases.json').write_text(json.dumps(cases,indent=2));lines=['/* Pinned vLLM image 745feb916ee8, plugin e454affe, Torch 2.14.', ' * BF16 gate/up outputs, BF16 SiLU before BF16 multiply, then down.', ' * Whole FFN output hashes from the unchanged linear and activation ops. */','#ifndef BN_CUDA_BF16_FFN_REFERENCE_H','#define BN_CUDA_BF16_FFN_REFERENCE_H','typedef struct { int dim, hidden, gate, up, down, seed, tokens; uint64_t hash; } BnBF16FFNReference;','static const BnBF16FFNReference cuda_bf16_ffn_cases[] = {']
for c in cases:lines.append('    {%d,%d,%d,%d,%d,%d,%d,UINT64_C(0x%016x)},'%(c['dim'],c['hidden'],c['gate'],c['up'],c['down'],c['seed'],c['tokens'],c['hash']))
lines+=['};','#endif'];(args.header or p/'cuda_bf16_ffn_reference.h').write_text('\n'.join(lines)+'\n');print('FFN references',len(cases),'words',sum(c['dim']*c['tokens'] for c in cases),flush=True)
