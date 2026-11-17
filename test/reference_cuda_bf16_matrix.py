"""Generate independent BF16 references inside the pinned vLLM image.

Image digest: 745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
Plugin wheel: 0.0.5+e454affe, SHA256
 d6dd44400cc02ca7f3a0452f32589d2eed6588da10ff04858d78b551f9f828ef
Torch: 2.14.0a0+4fdf77b940.nv26.8.63802676
Input directory contains w-{type}-{rows}-{seed}.bin generated using
signed_mmq_fixture_weights() from cuda_signed_mmq_fixture.h. Outputs include
all decoded BF16 weight bytes and GEMM output bytes, with JSON shape manifests.
The plugin identity projection independently validates decoded weight values;
signed-zero bits are retained in the explicit decoded-weight files.
"""
import argparse
import torch,numpy as np,json
from pathlib import Path
from gguf.quants import IQ4_NL
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;cases=[];table=torch.tensor(IQ4_NL.kvalues,dtype=torch.float32)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
for typ in [20,23]:
 for rows in [33,129]:
  for seed in [0,1]:
   raw=np.fromfile(p/f'w-{typ}-{rows}-{seed}.bin',dtype=np.uint8).reshape(-1,18 if typ==20 else 136)
   d=torch.from_numpy(raw[:,:2].copy().view('<f2').astype(np.float32))
   if typ==20:
    q=torch.from_numpy(raw[:,2:].copy()).to(torch.int64);idx=torch.cat([q&15,q>>4],dim=1);decoded=(table[idx]*d).to(torch.bfloat16)
   else:
    h=raw[:,2:4].copy().view('<u2').reshape(-1);lo=raw[:,4:8];sc=np.stack([((lo[:,g//2]>>(4*(g%2)))&15)|(((h>>(2*g))&3)<<4) for g in range(8)],axis=1).astype(np.int16)-32
    scales=torch.from_numpy(sc).to(torch.float32);ds=d*scales
    q=torch.from_numpy(raw[:,8:].copy()).to(torch.int64).reshape(-1,8,16);idx=torch.cat([q&15,q>>4],dim=2);decoded=(table[idx]*ds[:,:,None]).to(torch.bfloat16)
   decoded=decoded.reshape(rows,256).contiguous();f=f'decode-{typ}-{rows}-{seed}.bin';(p/f).write_bytes(decoded.view(torch.uint8).numpy().tobytes())
   # Independent production CUDA dequantization plus BF16 GEMM identity projection.
   w=torch.from_numpy(raw.copy().reshape(rows,-1)).cuda();ident=torch.eye(256,dtype=torch.bfloat16,device='cuda');projected=_fused_mul_mat_gguf(ident,w,typ).T.contiguous().cpu();assert torch.equal(projected,decoded),(typ,rows,seed)
   cases.append({'type':typ,'rows':rows,'cols':256,'seed':seed,'file':f})
(p/'decode-cases.json').write_text(json.dumps(cases,indent=2));gemm=[]
for rows,cols,nt in [(1,1,1),(7,32,5),(33,256,17),(129,256,33),(33,512,129),(17,257,9)]:
 i=torch.arange(rows*cols);w=(((i*13)%31-15).float()/16).to(torch.bfloat16).reshape(rows,cols)
 j=torch.arange(nt*cols);x=(((j*7)%31-15).float()/8).to(torch.bfloat16).reshape(nt,cols)
 y=torch.matmul(x.cuda(),w.cuda().T).cpu().contiguous();f=f'gemm-{rows}-{cols}-{nt}.bin';(p/f).write_bytes(y.view(torch.uint8).numpy().tobytes());gemm.append({'rows':rows,'cols':cols,'tokens':nt,'file':f})
(p/'gemm-cases.json').write_text(json.dumps(gemm,indent=2));print('native references',len(cases),'decode layouts,',len(gemm),'GEMMs',flush=True)

if args.header:
 out=['/* Independent Torch BF16 GEMM references: pinned vLLM image 745feb916ee8,', ' * Torch 2.14.0a0+4fdf77b940, FP32 accumulation, reduced-precision reduction off.', ' * Dyadic inputs make the FP32 sums exact. Generator:', ' * test/reference_cuda_bf16_matrix.py; odd K, partial rows, and batching. */', '#ifndef BN_CUDA_BF16_MATRIX_REFERENCE_H', '#define BN_CUDA_BF16_MATRIX_REFERENCE_H', 'typedef struct { int rows, cols, tokens; size_t start; } BnBF16MatrixReference;', 'static const uint16_t cuda_bf16_matrix_expected[] = {']
 cases=json.loads((p/'gemm-cases.json').read_text());start=0;items=[]
 for c in cases:
  data=(p/c['file']).read_bytes();words=__import__("struct").unpack('<'+'H'*(len(data)//2),data)
  for i in range(0,len(words),12):out.append('    '+','.join('0x%04x'%x for x in words[i:i+12])+',')
  items.append('    {%d,%d,%d,%d},'%(c['rows'],c['cols'],c['tokens'],start));start+=len(words)
 out+=['};','static const BnBF16MatrixReference cuda_bf16_matrix_cases[] = {',*items,'};',
  '/* Complete decoded-weight byte hashes from independent Torch FP32 decoding,',
  ' * BF16 RNE conversion, validated by the pinned plugin CUDA identity GEMM. */',
  'typedef struct { int type, rows, cols, seed; uint64_t hash; } BnBF16DecodeReference;',
  'static const BnBF16DecodeReference cuda_bf16_decode_cases[] = {']
 for c in json.loads((p/'decode-cases.json').read_text()):
  h=14695981039346656037
  for b in (p/c['file']).read_bytes(): h=((h^b)*1099511628211)&((1<<64)-1)
  out.append('    {%d,%d,%d,%d,UINT64_C(0x%016x)},'%(c['type'],c['rows'],c['cols'],c['seed'],h))
 out+=['};','#endif']
 args.header.write_text('\n'.join(out)+'\n')
