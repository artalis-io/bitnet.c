"""Original final post/contract/RMS/raw Q8 head from a frozen FFN tuple."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch,gguf
from vllm import _custom_ops as custom
from vllm.model_executor.layers.mhc import hc_contract
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf as linear
from vllm_gguf_plugin import ops
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--tuple-reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
readers=[gguf.GGUFReader(str(a.model).replace('00001-of',f'{i:05d}-of'))for i in range(1,7)];table={t.name:t for r in readers for t in r.tensors};tw=table['output.weight'];tn=table['output_norm.weight'];assert int(tw.tensor_type)==8 and int(tn.tensor_type)==0
raw=np.array(tw.data,copy=True).reshape(154880,4352);ng=np.array(tn.data,copy=True).reshape(4096);w=torch.from_numpy(raw).cuda();gamma=torch.from_numpy(ng).bfloat16().cuda()
def read(name,typ,shape):return torch.frombuffer(bytearray((a.tuple_reference/(name+'.bin')).read_bytes()),dtype=typ).reshape(shape).cuda()
def save(f,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b,f
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
records=[]
for ci,nt in enumerate([1,9,255,256,257]):
 base='5-1-'if nt==1 else'0-0-';n=1 if nt==1 else 9;idx=torch.arange(nt,device='cuda')%n
 branch=read(base+'moe_out',torch.bfloat16,(n,4096))[idx].contiguous();residual=read(base+'ffn_residual',torch.bfloat16,(n,4,4096))[idx].contiguous();post=read(base+'ffn_post',torch.float32,(n,4))[idx].contiguous();comb=read(base+'ffn_comb',torch.float32,(n,4,4))[idx].contiguous()
 def run():
  completed=mhc_post_tilelang(branch,residual,post,comb);collapsed=hc_contract(completed,4);norm=torch.empty_like(collapsed);custom.rms_norm(norm,collapsed,gamma,1e-5)
  return dict(completed=completed,collapsed=collapsed,normalized=norm,logits_all=linear(norm,w,8),logits_last=linear(norm[-1:],w,8))
 first=run();second=run();assert all(torch.equal(t.contiguous().view(torch.uint8),second[k].contiguous().view(torch.uint8))for k,t in first.items());d=a.out/str(ci);d.mkdir(exist_ok=True)
 values=dict(branch=branch,residual=residual,post=post,comb=comb,**first);records.append(dict(tokens=nt,source=base,hashes={k:save(d/(k+'.bin'),t)for k,t in values.items()}));print(ci,nt,'final output captured twice',flush=True)
manifest=dict(records=records,head_weight_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),norm_weight_sha256=hashlib.sha256(ng.tobytes()).hexdigest(),original_extension_sha256=hashlib.sha256(Path(ops._C_gguf.__file__).read_bytes()).hexdigest(),profile='Independent original layer44 deferred FFN tuples; repeated rows for255/256/257 launch-boundary tests, not long model histories. Original Tilelang post, PyTorch hc_contract, custom RMS4096 gammaBF16 eps1e-5 (default non-batch-invariant launch), rawQ8 head for all or final sampled row. No whole-model generation claim.')
b=(json.dumps(manifest,indent=2)+'\n').encode();dest=a.out/'reference.json'
if dest.exists():assert dest.read_bytes()==b,dest
else:dest.write_bytes(b)
