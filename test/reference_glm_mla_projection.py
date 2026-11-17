"""Raw-GGUF MLA projection and original identity-materialized K/V reference.
Uses original GGUF CUDA operators, original RMSNorm and torch.bmm. Captures
input preparation and output projection separately; latent attention is absent.
"""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np,torch,gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin import ops as ggops
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert ggops._CUDA_ENABLED
r=gguf.GGUFReader(str(a.model).replace('00001-of','00002-of'));table={t.name:t for t in r.tensors};hashes={}
def source(name,shape):
 t=table['blk.3.'+name];raw=np.array(t.data,copy=True).reshape(shape);hashes[name]=hashlib.sha256(raw.tobytes()).hexdigest();return torch.from_numpy(raw).cuda()
def save(f,x):
 b=x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
qa=source('attn_q_a.weight',(1536,4352));qb=source('attn_q_b.weight',(16384,1632));kva=source('attn_kv_a_mqa.weight',(512,4352));outw=source('attn_output.weight',(4096,17408));qn=source('attn_q_a_norm.weight',(1536,)).bfloat16();kn=source('attn_kv_a_norm.weight',(512,)).bfloat16();kb=source('attn_k_b.weight',(32768,272));vb=source('attn_v_b.weight',(16384,544))
# This is the generic original quantized-weight getter's identity path,
# applied to the split tensors in their actual GGUF storage axes.
key=_fused_mul_mat_gguf(torch.eye(256,device='cuda',dtype=torch.bfloat16),kb,8).T.contiguous().reshape(64,512,256);value=_fused_mul_mat_gguf(torch.eye(512,device='cuda',dtype=torch.bfloat16),vb,8).T.contiguous().reshape(64,256,512)
key2=_fused_mul_mat_gguf(torch.eye(256,device='cuda',dtype=torch.bfloat16),kb,8).T.contiguous().reshape(64,512,256);value2=_fused_mul_mat_gguf(torch.eye(512,device='cuda',dtype=torch.bfloat16),vb,8).T.contiguous().reshape(64,256,512);assert torch.equal(key.view(torch.uint8),key2.view(torch.uint8));assert torch.equal(value.view(torch.uint8),value2.view(torch.uint8));save(a.out/'key-basis.bin',key.permute(2,0,1));save(a.out/'value-basis.bin',value.permute(2,0,1))
records=[]
for ci,nt in enumerate([1,2,3,7,17,65,129,1025]):
 i=torch.arange(nt*4096,dtype=torch.int64);x=(((i*17+i//31+ci*7)%509-254).float()/128).reshape(nt,4096).bfloat16().cuda()
 def run():
  qraw=_fused_mul_mat_gguf(x,qa,8);qr=torch.empty_like(qraw);torch.ops._C.rms_norm(qr,qraw,qn,1e-5);kraw=_fused_mul_mat_gguf(x,kva,8);kv=torch.empty_like(kraw);torch.ops._C.rms_norm(kv,kraw,kn,1e-5);q=_fused_mul_mat_gguf(qr,qb,8).reshape(nt,64,256);query=torch.empty((nt,64,512),device='cuda',dtype=torch.bfloat16);torch.bmm(q.transpose(0,1),key.transpose(1,2),out=query.transpose(0,1));context=(query.float()*.125+kv[:,None,:].float()).bfloat16();values=torch.empty((nt,64,256),device='cuda',dtype=torch.bfloat16);torch.bmm(context.transpose(0,1),value.transpose(1,2),out=values.transpose(0,1));out=_fused_mul_mat_gguf(values.reshape(nt,16384),outw,8);torch.cuda.synchronize();return dict(x=x,q_raw=qraw,qr=qr,kv_raw=kraw,kv=kv,q=q,query=query,context=context,values=values,out=out)
 first,second=run(),run();assert all(torch.equal(t.contiguous().view(torch.uint8),second[n].contiguous().view(torch.uint8))for n,t in first.items());d=a.out/str(ci);d.mkdir(parents=True,exist_ok=True);records.append(dict(tokens=nt,hashes={n:save(d/(n+'.bin'),t)for n,t in first.items()}));print(ci,nt,'original projections/identity BMM captured twice',flush=True)
sources={}
for f in [_fused_mul_mat_gguf]:
 s=Path(inspect.getfile(inspect.unwrap(f)));sources[str(s)]=hashlib.sha256(s.read_bytes()).hexdigest()
(a.out/'reference.json').write_text(json.dumps(dict(records=records,weights=hashes,sources=sources,extension_sha256=hashlib.sha256(Path(ggops._C_gguf.__file__).read_bytes()).hexdigest(),profile='Separate raw Q8 QA/KVA/QB/O; BF16-gamma original RMSNorm; original identity-materialized split Q8 K/V in GGUF axes; original token-major torch.bmm; output projection starts at supplied latent context. No MLA attention/cache or default HF merged-loader claim.'),indent=2)+'\n')
