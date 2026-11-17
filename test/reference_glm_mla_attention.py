"""Installed original MLA cache/FlashInfer attention/value/output operators.
Input projection captures and identity-materialized weights remain unchanged.
"""
from pathlib import Path
import argparse,json,hashlib,inspect
import torch,numpy as np,gguf,vllm,flashinfer
from vllm import _custom_ops as ops
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as attention
from flashinfer.mla._sparse_mla_sm120 import _prepared
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--projection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
reader=gguf.GGUFReader(str(a.model).replace('00001-of','00002-of'));t=next(t for t in reader.tensors if t.name=='blk.3.attn_output.weight');raw=np.array(t.data,copy=True).reshape(4096,17408);outw=torch.from_numpy(raw).cuda()
def load(f,dtype):return torch.frombuffer(bytearray(f.read_bytes()),dtype=dtype).cuda()
def save(f,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b,f
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
value=load(a.projection/'value-basis.bin',torch.bfloat16).reshape(512,64,256).permute(1,2,0).contiguous();plans=[];resolve=_prepared.resolve_execution
# Observe policy decisions; do not change reference dispatch or arithmetic.
def capture(*args,**kw):
 plan=resolve(*args,**kw);plans.append({str(k):str(v)if isinstance(v,str)else int(v)for k,v in plan.inspect().items()});return plan
_prepared.resolve_execution=capture
workspace=torch.empty(394*1024*1024,device='cuda',dtype=torch.uint8);records=[]
for ci,nt in enumerate([1,2,3,7,17,65,129,1025]):
 src=a.projection/str(ci);query=load(src/'query.bin',torch.bfloat16).reshape(nt,1,64,512);kv=load(src/'kv.bin',torch.bfloat16).reshape(nt,512);blocks=(nt+63)//64+2;table=torch.arange(blocks-1,-1,-1,device='cuda',dtype=torch.int32);logical=torch.arange(nt,device='cuda',dtype=torch.int32);slots=table[logical//64]*64+logical%64
 selection=torch.full((nt,2051),-1,device='cuda',dtype=torch.int32);physical=torch.full((nt,1,2176),-1,device='cuda',dtype=torch.int32)
 for row in range(nt):selection[row,:row+1]=logical[:row+1];physical[row,0,:row+1]=slots[:row+1]
 def run():
  cache=torch.full((blocks,64,656),0xa5,device='cuda',dtype=torch.uint8);ops.concat_and_cache_mla(kv,torch.empty((nt,0),device='cuda',dtype=torch.bfloat16),cache,slots.long(),'fp8_ds_mla',torch.ones(1,device='cuda'))
  context,lse=attention(query=query,kv_cache=cache.unsqueeze(1),workspace_buffer=workspace,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=physical,seq_lens=None,max_seq_len=2176,sparse_mla_top_k=2176,bmm1_scale=.0625,bmm2_scale=1.,kv_scale_format='arbitrary_fp32',return_lse=True)
  context=context.reshape(nt,64,512);values=torch.empty((nt,64,256),device='cuda',dtype=torch.bfloat16);torch.bmm(context.transpose(0,1),value.transpose(1,2),out=values.transpose(0,1));out=_fused_mul_mat_gguf(values.reshape(nt,16384),outw,8);torch.cuda.synchronize();return dict(cache=cache,context=context,lse=lse,values=values,out=out)
 first,second=run(),run();assert all(torch.equal(t.contiguous().view(torch.uint8),second[n].contiguous().view(torch.uint8))for n,t in first.items());directory=a.out/str(ci);directory.mkdir(exist_ok=True);fields=dict(first,selection=selection,physical=physical,table=table,slots=slots);records.append(dict(tokens=nt,blocks=blocks,plan=plans[-1],hashes={name:save(directory/(name+'.bin'),t)for name,t in fields.items()}));print(ci,nt,'original attention/cache/value/output captured twice',flush=True)
(a.out/'reference.json').write_text(json.dumps(dict(records=records,torch=torch.__version__,vllm=vllm.__version__,flashinfer=flashinfer.__version__,flashinfer_commit=flashinfer.__git_commit__,output_weight_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),original_extension_sha256=hashlib.sha256((Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so').read_bytes()).hexdigest(),profile='Raw GGUF projected inputs and identity-materialized V; original cache writer; original FlashInfer default SM120 policy, padded2176; causal logical selection, reverse page table; original token-major BF16 BMM and raw Q8 O projection. No full model or overbudget selection-order claim.'),indent=2)+'\n')
