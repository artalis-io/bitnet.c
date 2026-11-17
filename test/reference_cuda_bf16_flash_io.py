"""Pinned original GGUF embedding + output HC/LM-head references, fresh files.

The HC input is the preserved independent FFN oracle's output, never native output.
Synthetic Q8 rows enumerate every FP16 scale bit pattern and every signed quant.
"""
import argparse, hashlib, json
from pathlib import Path
import gguf
import numpy as np
import torch
from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path);p.add_argument('ffn_captures',type=Path);a=p.parse_args()
a.directory.mkdir(exist_ok=False)
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
files={};sources={}
def save(name,x):
    b=x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();f=a.directory/name
    assert not f.exists();f.write_bytes(b)
    files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),shape=list(x.shape),dtype=str(x.dtype))
    return b
raw=np.zeros((65536,8,34),np.uint8);i=np.arange(65536,dtype=np.uint32)
raw[:,:,0]=(i&255).astype(np.uint8)[:,None];raw[:,:,1]=(i>>8).astype(np.uint8)[:,None]
raw[:,:,2:]=np.arange(-128,128,dtype=np.int16).astype(np.int8).view(np.uint8).reshape(8,32)[None,:,:]
w=torch.from_numpy(raw.reshape(65536,-1)).cuda();synthetic=[]
with torch.inference_mode():
    for nt in [1,5,17,129,65537]:
        ids=(torch.arange(nt,device='cuda',dtype=torch.int64)*8191)%65536
        out=_apply_gguf_embedding(ids,w,8,256,torch.bfloat16)
        data=save(f'synthetic-{nt}.bin',out)
        bits=np.frombuffer(data,dtype='<u2').copy();nan=((bits&0x7f80)==0x7f80)&((bits&0x7f)!=0);bits[nan]=0x7fc0
        h=14695981039346656037
        for b in bits.tobytes():h=((h^b)*1099511628211)&((1<<64)-1)
        synthetic.append(dict(tokens=nt,hash=h,nan_words=int(nan.sum())))
del w,raw,out;torch.cuda.empty_cache()
readers=[gguf.GGUFReader(s['path']) for s in json.loads(a.inventory.read_text())['shards']]
tensors={t.name:t for r in readers for t in r.tensors}
def raw_tensor(name):
    t=tensors[name];sources[name]=dict(type=t.tensor_type.name,shape=[int(v)for v in t.shape],bytes=int(t.n_bytes),sha256=hashlib.sha256(memoryview(t.data).cast('B')).hexdigest())
    return torch.from_numpy(np.array(t.data,copy=True)).cuda()
def decoded(name):
    t=tensors[name];v=torch.from_numpy(np.array(gguf.dequantize(t.data,t.tensor_type),copy=True)).float()
    if name=='output_hc_norm.weight':v=v-1.0
    sources[name]=dict(type=t.tensor_type.name,sha256=hashlib.sha256(memoryview(t.data).cast('B')).hexdigest())
    return v.bfloat16().cuda()
emb=raw_tensor('token_embd.weight');head=raw_tensor('output.weight')
assert all(tensors[n].tensor_type==gguf.GGMLQuantizationType.Q8_0 for n in ['token_embd.weight','output.weight'])
hc=GatedResidual(HyperConnectionConfig(hc_count=4,hidden_size=2560,hc_lowrank=320,hc_per_branch_norm=True,params_dtype=torch.bfloat16),use_combine=False).cuda()
with torch.no_grad():
    hc.hc_norm.weight.copy_(decoded('output_hc_norm.weight'))
    hc.input_mix_weight_down.weight.copy_(decoded('output_hc_down.weight'))
    hc.input_mix_weight_up.weight.copy_(decoded('output_hc_up.weight'))
cases=[]
with torch.inference_mode():
    for nt in [1,5,17,129,257]:
        ids=((torch.arange(nt,device='cuda',dtype=torch.int64)*15485863+116)%248320)
        if nt>=4:ids[:4]=torch.tensor([0,248319,248044,116],device='cuda')
        embedding=_apply_gguf_embedding(ids,emb,8,2560,torch.bfloat16)
        save(f'input-{nt}-ids.bin',ids.to(torch.int32));save(f'input-{nt}-embedding.bin',embedding);save(f'input-{nt}-wide.bin',embedding.repeat(1,4))
        source=a.ffn_captures/f'layer3-{nt}-hc-combined.bin';b=source.read_bytes();assert len(b)==nt*10240*2
        hidden=torch.from_numpy(np.frombuffer(b,dtype=np.uint16).copy()).view(torch.bfloat16).reshape(nt,10240).cuda()
        mixed,_=hc.mix(hidden);logits=_fused_mul_mat_gguf(mixed[-1:].contiguous(),head,8)
        save(f'head-{nt}-input.bin',hidden);save(f'head-{nt}-mixed.bin',mixed);save(f'head-{nt}-logits.bin',logits)
        cases.append(dict(tokens=nt,ffn_source=str(source),ffn_source_sha256=hashlib.sha256(b).hexdigest(),argmax=int(logits.float().argmax())))
header=['/* Original pinned GGUF CUDA Q8 embedding; NaNs canonicalized for hashing. */','#ifndef BN_CUDA_BF16_Q8_EMBEDDING_REFERENCE_H','#define BN_CUDA_BF16_Q8_EMBEDDING_REFERENCE_H','typedef struct { int tokens; uint64_t hash; } BnQ8EmbeddingReference;','static const BnQ8EmbeddingReference cuda_q8_embedding_cases[] = {']
for c in synthetic:header.append(' {%d,UINT64_C(0x%016x)},'%(c['tokens'],c['hash']))
header+=['};','#endif'];(a.directory/'reference.h').write_text('\n'.join(header)+'\n')
(a.directory/'manifest.json').write_text(json.dumps(dict(files=files,sources=sources,synthetic=synthetic,cases=cases),indent=2)+'\n')
print('FLASH_IO_REFERENCE cases',len(cases),'synthetic_words',sum(c['tokens']*256 for c in synthetic),flush=True)
