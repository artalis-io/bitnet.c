"""Independent resident prepared GDN references from unmodified pinned vLLM.

Image sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29.
Only the original chunk_gated_delta_rule produces expected output/state hashes.
Inputs are deterministic CPU arithmetic; initial state is nonzero F32 [V,K].
No BitNet code supplies expected values. BF16 hashes consume bytes; F32 hashes
consume uint32 patterns. Exact schedule is part of each reference case.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--header',type=Path);args=parser.parse_args();p=args.directory;p.mkdir(parents=True,exist_ok=True)
def save(name,t):
    data=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();(p/name).write_bytes(data);h=14695981039346656037
    for x in (data if t.dtype==torch.bfloat16 else np.frombuffer(data,dtype='<u4')):h=((h^int(x))*1099511628211)&((1<<64)-1)
    return h
cases=[]
for nt,splits in [(1,[1]),(5,[5]),(64,[64]),(65,[33,32]),(129,[129]),(257,[129,128]),(1047,[129]*8+[15])]:
    for seed in [0,1]:
        name=f'prefill-{nt}-{seed}';nq=nt*16*128;nv=nt*48*128;ng=nt*48;ns=48*128*128
        def bp(n,m,d,shift):
            i=torch.arange(n,dtype=torch.int64);return (((i*m+i//13+seed*11+shift)%509-254).float()/d).to(torch.bfloat16).cuda()
        q=bp(nq,37,8192,0).reshape(1,nt,16,128);k=bp(nq,29,8192,17).reshape_as(q);v=bp(nv,23,256,31).reshape(1,nt,48,128)
        i=torch.arange(ng,dtype=torch.int64);g=(-((i*7+seed)%17+1).float()/1024).reshape(1,nt,48).cuda();beta=(((i*13+seed)%31+1).float()/64).reshape_as(g).cuda()
        i=torch.arange(ns,dtype=torch.int64);initial=(((i*17+i//37+seed*19)%251-125).float()/4096).reshape(1,48,128,128).cuda()
        for key,value in [('q',q),('k',k),('v',v),('g',g),('beta',beta),('initial',initial)]:save(name+'-'+key+'.bin',value)
        offset=0;state=initial.clone();outputs=[];states=[]
        for phase,take in enumerate(splits):
            cu=torch.tensor([0,take],dtype=torch.int32,device='cuda')
            out,state=chunk_gated_delta_rule(q[:,offset:offset+take].contiguous(),k[:,offset:offset+take].contiguous(),v[:,offset:offset+take].contiguous(),g[:,offset:offset+take].contiguous(),beta[:,offset:offset+take].contiguous(),initial_state=state,output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
            assert torch.isfinite(out).all() and torch.isfinite(state).all()
            outputs.append(save(f'{name}-{phase}-out.bin',out));states.append(save(f'{name}-{phase}-state.bin',state));offset+=take
        cases.append(dict(tokens=nt,seed=seed,splits=splits,out=outputs,state=states));print(name,splits,'captured',flush=True)
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
if args.header:
    rows=['/* Independent pinned vLLM prepared GDN; see reference_cuda_bf16_gdn_prefill.py. */','#ifndef BN_CUDA_BF16_GDN_PREFILL_REFERENCE_H','#define BN_CUDA_BF16_GDN_PREFILL_REFERENCE_H','typedef struct { int tokens,seed,n_phases,splits[9]; uint64_t out[9],state[9]; } BnBF16GDNPrefillReference;','static const BnBF16GDNPrefillReference cuda_bf16_gdn_prefill_cases[] = {']
    for c in cases:
        h=lambda values:','.join(f'UINT64_C(0x{x:016x})' for x in values)
        rows.append(' {%d,%d,%d,{%s},{%s},{%s}},'%(c['tokens'],c['seed'],len(c['splits']),','.join(map(str,c['splits'])),h(c['out']),h(c['state'])))
    args.header.write_text('\n'.join(rows+['};','#endif'])+'\n')
