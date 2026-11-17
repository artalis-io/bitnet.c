"""Actual raw GGUF GLM pre-indexer from hidden state and MLA-normalized q rank.
QA/RMS compute the caller boundary independently; native helper starts at x/qr.
The explicit head profile preserves GGUF F32, unlike HF merged BF16 loading.
"""
from pathlib import Path
import argparse,json,inspect,hashlib
import numpy as np,torch,gguf
import vllm
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin import ops as ggops
from vllm.models.glm5next.common.attention import _fused_indexer_k_norm,_fused_indexer_weight_scale
from vllm.models.glm5next.nvidia.ops import kpool_compress as ops
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--schedules',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
assert ggops._CUDA_ENABLED, 'requires the original CUDA GGUF operators'
readers=[gguf.GGUFReader(str(a.model).replace('00001-of',f'{i:05d}-of'))for i in range(1,7)];table={t.name:t for r in readers for t in r.tensors};hashes={}
def source(name,typ,shape):
 t=table['blk.3.'+name];assert int(t.tensor_type)==typ;raw=np.array(t.data,copy=True).reshape(shape);hashes[name]=hashlib.sha256(raw.tobytes()).hexdigest();return torch.from_numpy(raw).cuda()
qa=source('attn_q_a.weight',8,(1536,4352));qn=source('attn_q_a_norm.weight',0,(1536,)).bfloat16()
q=source('indexer.attn_q_b.weight',8,(4096,1632));k=source('indexer.attn_k.weight',8,(128,4352));gate=source('indexer_compressor_gate.weight',8,(128,4352));head=source('indexer.proj.weight',0,(32,4096));norm=source('indexer.k_norm.weight',0,(128,));bias=source('indexer.k_norm.bias',0,(128,));ape=source('indexer_compressor_ape.weight',0,(4,128))
def save(path,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if path.exists():assert path.read_bytes()==b,str(path)
 else:path.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
steps=[[1],[2],[6],[7],[9],[65],[129],[1025],[2,7],[3,3,3],[7,1,1],[1]*9];records=[]
if a.schedules:
 steps=json.loads(a.schedules.read_text());assert isinstance(steps,list) and steps and all(isinstance(seq,list) and seq and all(isinstance(n,int) and 0<n<=16384 for n in seq)for seq in steps)
for warm in ([0]if a.schedules else[0,1]):
 for ci,sequence in enumerate(steps):
  schedule=([3]if warm else[])+sequence;case=warm*len(steps)+ci
  def run():
   tail=torch.zeros((1,2,4,128),device='cuda',dtype=torch.bfloat16);position=0;values=[]
   for nt in schedule:
    i=torch.arange(position*4096,(position+nt)*4096,dtype=torch.int64);x=(((i*17+i//31+case*7)%509-254).float()/128).reshape(nt,4096).bfloat16().cuda();pre=_fused_mul_mat_gguf(x,qa,8);qr=torch.empty_like(pre);torch.ops._C.rms_norm(qr,pre,qn,1e-5)
    q_bf16=_fused_mul_mat_gguf(qr,q,8);q_fp8,q_scale=ops.fwht128_quant_fp8(q_bf16.reshape(-1,128));key_raw=_fused_mul_mat_gguf(x,k,8);key=_fused_indexer_k_norm(key_raw,norm,bias,128,1e-6);raw_head=x.float()@head.T;weights=_fused_indexer_weight_scale(raw_head,q_scale.reshape(nt,32,1),.015625);gate_out=_fused_mul_mat_gguf(x,gate,8)
    initial=tail.clone();groups=(position+nt)//4-position//4;ks=[];gs=[]
    for gi in range(groups):
     start=(position//4+gi)*4;ks.append(torch.stack([key[t-position]if t>=position else initial[0,0,t%4]for t in range(start,start+4)]));gs.append(torch.stack([gate_out[t-position]if t>=position else initial[0,1,t%4]for t in range(start,start+4)]))
    cache=torch.zeros((max(1,(groups+63)//64),64,132),device='cuda',dtype=torch.uint8)
    if groups:compressed,scales=ops.kpool_compress_and_write_cache(cache,torch.stack(ks),torch.stack(gs),ape,torch.arange(groups,device='cuda',dtype=torch.int64),4,return_compressed=True,write_cache=False)
    else:compressed=torch.empty((0,128),device='cuda',dtype=torch.float8_e4m3fn);scales=torch.empty(0,device='cuda')
    pos=torch.arange(position,position+nt,device='cuda',dtype=torch.int32).reshape(1,nt);slot=torch.where((pos+1)%4==0,pos//4-position//4,-1).int();ops.kpool_decode_update_and_maybe_write_cache_batched(cache,tail,(pos%4).int(),key[None],gate_out[None],ape,slot,pos,4)
    cv=cache.reshape(cache.shape[0],-1)
    for gi in range(groups):
     pg,off=divmod(gi,64);assert torch.equal(cv[pg,off*128:(off+1)*128],compressed[gi].view(torch.uint8))
    values.append(dict(x=x,qr=qr,q=q_bf16,key_raw=key_raw,key=key,gate=gate_out,head_raw=raw_head,q_fp8=q_fp8,q_scales=q_scale,head_weights=weights,compressed=compressed,compressed_scales=scales,initial_tail=initial,next_tail=tail.clone()));position+=nt
   torch.cuda.synchronize();return values
  first,repeat=run(),run();phases=[];pos=0
  for pi,(x,y)in enumerate(zip(first,repeat)):
   assert all(torch.equal(v.view(torch.uint8),y[n].view(torch.uint8))for n,v in x.items());directory=a.out/str(case)/str(pi);directory.mkdir(parents=True,exist_ok=True);phases.append(dict(position=pos,tokens=schedule[pi],hashes={n:save(directory/(n+'.bin'),v)for n,v in x.items()}));pos+=schedule[pi]
  records.append(dict(case=case,warm=warm,schedule=schedule,phases=phases));print(case,schedule,'all stages captured twice',flush=True)
sources={}
for f in [_fused_mul_mat_gguf,_fused_indexer_k_norm,_fused_indexer_weight_scale,ops.fwht128_quant_fp8]:
 path=Path(inspect.getfile(inspect.unwrap(f)));sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(a.out/'reference.json').write_text(json.dumps(dict(records=records,weights=hashes,sources=sources,extension_sha256=hashlib.sha256(Path(ggops._C_gguf.__file__).read_bytes()).hexdigest(),vllm_extension_sha256=hashlib.sha256((Path(vllm.__file__).parent/"_C_stable_libtorch.abi3.so").read_bytes()).hexdigest(),cuda_backend=bool(ggops._CUDA_ENABLED),profile='Actual block3 raw Q8 indexer projections and raw F32 head/norm/bias/APE; TF32 off. Caller QR derived by original Q8 QA + BF16-gamma RMSNorm; native pre-indexer boundary accepts x/qr. Whole indexer scores/selection/MLA not included.',mx_scales=ops.mqa_logits_use_mx_scales()),indent=2)+'\n')
