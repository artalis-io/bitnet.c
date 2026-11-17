"""Independent BF16 residual/RMSNorm and composed FFN references.

Image: sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
Plugin: e454affe0ca23c9dc8754ec3033270cb30622ed5; wheel SHA256
 d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef
Torch: 2.14.0a0+4fdf77b940.nv26.8.63802676
Generate raw weights using reference_cuda_bf16_ffn.c. This generator uses the
unchanged native vLLM IR fused_add_rms_norm, unchanged plugin linear dispatch
and BF16 silu_and_mul. The F32 residual sum drives normalization; a separately
rounded BF16 residual is returned. Gamma is formed in F32 from BF16 parameters.
Output contains both complete vectors and whole-byte hashes; no implementation-
generated golden values or elementwise tolerance is used.
"""
import torch,numpy as np,json,argparse
from pathlib import Path
import vllm.ir as ir
from vllm.config import VllmConfig,set_current_vllm_config
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
import vllm._custom_ops
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('weights',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;weight_dir=args.weights;context=set_current_vllm_config(VllmConfig());context.__enter__();torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
hash_bytes=lambda b: __import__('functools').reduce(lambda h,x:((h^x)*1099511628211)&((1<<64)-1),b,14695981039346656037)
def save(name,t):
 data=t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);return hash_bytes(data)
def inputs(dim,nt,seed):
 i=torch.arange(nt*dim);x=(((i*37+seed*11)%509-254).float()/256).to(torch.bfloat16).reshape(nt,dim).cuda();r=(((i*13+i//17+seed*3)%251-125).float()/512).to(torch.bfloat16).reshape(nt,dim).cuda();return x,r
norm=[]
for dim,nt in [(d,n) for d in [1,31,32,33,255,256,257,768,5120] for n in [1,3]]+[(1,65537)]:
 for offset in [0,1]:
  seed=dim+nt*7+offset;x,r=inputs(dim,nt,seed);j=torch.arange(dim);w=(((j*7+seed)%31-15).float()/(4096 if offset else 32)+(0 if offset else 1)).to(torch.bfloat16).cuda();gamma=w.float()+offset
  y,rr=ir.ops.fused_add_rms_norm(x,r,gamma,1e-6);name=f'norm-{dim}-{nt}-{offset}';h=save(name+'.bin',y);hr=save(name+'-res.bin',rr);norm.append(dict(dim=dim,tokens=nt,offset=offset,seed=seed,hash=h,residual_hash=hr,file=name+'.bin'))
(p/'norm-cases.json').write_text(json.dumps(norm,indent=2));ffn=[]
for dim,hid in [(256,256),(768,512),(256,5376)]:
 for types in [(23,11,23),(23,23,23)]:
  for offset in [0,1]:
   for nt in [1,5,17,33]:
    seed=offset;x,r=inputs(dim,nt,seed);j=torch.arange(dim);w=((-1+(1+(j*7+seed)%5).float()/128) if offset else (1/64+(((j*7+seed)%9)-4).float()/1024)).to(torch.bfloat16).cuda();n,rr=ir.ops.fused_add_rms_norm(x,r,w.float()+offset,1e-6)
    def weight(typ,rows,cols,s):return torch.from_numpy(np.fromfile(weight_dir/f'w-{typ}-{rows}-{cols}-{s}.bin',np.uint8).reshape(rows,-1)).cuda()
    g=_fused_mul_mat_gguf(n,weight(types[0],hid,dim,seed),types[0]);u=_fused_mul_mat_gguf(n,weight(types[1],hid,dim,1-seed),types[1]);mid=torch.empty_like(g);torch.ops._C.silu_and_mul(mid,torch.cat([g,u],-1));y=_fused_mul_mat_gguf(mid,weight(types[2],dim,hid,seed),types[2]);assert torch.isfinite(n).all() and torch.isfinite(mid).all() and torch.isfinite(y).all();name=f'ffn-{dim}-{hid}-{types[1]}-{offset}-{nt}';h=save(name+'.bin',y);hr=save(name+'-res.bin',rr);ffn.append(dict(dim=dim,hidden=hid,gate=types[0],up=types[1],down=types[2],offset=offset,seed=seed,tokens=nt,hash=h,residual_hash=hr,file=name+'.bin'))
(p/'ffn-cases.json').write_text(json.dumps(ffn,indent=2));lines=['/* Unchanged pinned vLLM IR add/RMSNorm + plugin linear + BF16 SwiGLU. */','#ifndef BN_CUDA_BF16_RESIDUAL_REFERENCE_H','#define BN_CUDA_BF16_RESIDUAL_REFERENCE_H','typedef struct { int dim,tokens,offset,seed; uint64_t hash,residual_hash; } BnBF16AddNormReference;','static const BnBF16AddNormReference cuda_bf16_add_norm_cases[] = {']
for c in norm:lines.append(' {%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['dim'],c['tokens'],c['offset'],c['seed'],c['hash'],c['residual_hash']))
lines+=['};','typedef struct { int dim,hidden,gate,up,down,offset,seed,tokens; uint64_t hash,residual_hash; } BnBF16ResidualFFNReference;','static const BnBF16ResidualFFNReference cuda_bf16_residual_ffn_cases[] = {']
for c in ffn:lines.append(' {%d,%d,%d,%d,%d,%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x)},'%(c['dim'],c['hidden'],c['gate'],c['up'],c['down'],c['offset'],c['seed'],c['tokens'],c['hash'],c['residual_hash']))
lines+=['};','#endif'];(args.header or p/'reference.h').write_text('\n'.join(lines)+'\n');print('generated norm',len(norm),'ffn',len(ffn),flush=True)
