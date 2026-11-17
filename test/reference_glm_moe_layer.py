"""Original raw-F32 GLM routing plus shared branch; optional complete MoE.
The profile explicitly preserves GGUF F32 router weights and disables TF32.
For matrix MoE the installed original extension rejects expert IDs above 255.
Partition only its expert domain into supported local IDs, preserving token
geometry and bit-selecting results. This does not qualify the whole plugin
method, which also omits the checkpoint clamp. See docs/glm53-cuda-moe-layer.md.
"""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np,torch,gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin import ops
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--experts',action='store_true');ap.add_argument('--layer',type=int,default=3);ap.add_argument('--corrected-expert-domain',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
extension_sha=hashlib.sha256(Path(ops._C_gguf.__file__).read_bytes()).hexdigest()
if a.corrected_expert_domain:
 assert a.experts and extension_sha=='f61f8846d7fb50fd2d03254b03a0801db62de9dbb6b1a102fb85233c53c3dd5e', 'requires the separately qualified corrected GGUF extension'
readers=[gguf.GGUFReader(str(a.model).replace('00001-of',f'{i:05d}-of'))for i in range(1,7)];table={t.name:t for r in readers for t in r.tensors};hashes={}
def raw(name,typ,shape):
 t=table[f'blk.{a.layer}.{name}'];assert int(t.tensor_type)==typ;data=np.array(t.data,copy=True).reshape(shape);hashes[name]=hashlib.sha256(data.tobytes()).hexdigest();return torch.from_numpy(data).cuda()
router=raw('ffn_gate_inp.weight',0,(288,4096));bias=raw('exp_probs_b.bias',0,(288,));gu=torch.cat([raw(f'ffn_{r}_shexp.weight',8,(2048,4352))for r in ['gate','up']],0).contiguous();down=raw('ffn_down_shexp.weight',8,(4096,2176))
if a.experts:
 gate_type=int(table[f'blk.{a.layer}.ffn_gate_exps.weight'].tensor_type);down_type=int(table[f'blk.{a.layer}.ffn_down_exps.weight'].tensor_type);assert gate_type in [12,13] and down_type in [13,14]
 gb=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(gate_type)][1];db=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(down_type)][1]
 routed_gu=torch.cat([raw(f'ffn_{r}_exps.weight',gate_type,(288,2048,16*gb))for r in ['gate','up']],1).contiguous();routed_down=raw('ffn_down_exps.weight',down_type,(288,4096,8*db))
def save(path,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if path.exists():assert path.read_bytes()==b,path
 else:path.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
records=[]
for ci,nt in enumerate([1,2,6,7,9,64,65,129,1024,1025,2048]if not a.experts else[1,9,65,129,1024,1025]):
 gen=torch.Generator().manual_seed(21473+ci);x=torch.randn((nt,4096),generator=gen).bfloat16().cuda()
 def routed_matrix(inp,weight,typ,rows,top_k,total):
  # Original CUDA matrix entry rejects expert IDs >255. Partition only the
  # expert dimension; preserve original token/slot geometry and operator math.
  result=torch.zeros((total*top_k,rows),device='cuda',dtype=inp.dtype)
  domain=288 if a.corrected_expert_domain else 256
  for lo in range(0,288,domain):
   hi=min(lo+domain,288);mask=(ids>=lo)&(ids<hi)
   local=torch.where(mask,ids-lo,-torch.ones_like(ids)).contiguous()
   block=ops.ggml_moe_get_block_size(typ)
   si,ei,padded=moe_align_block_size(local,block,hi-lo,ignore_invalid_experts=True)
   part=ops.ggml_moe_a8(inp,weight[lo:hi].contiguous(),si,ei,padded,typ,rows,top_k,total)
   result=torch.where(mask.reshape(-1,1),part,result)
  return result
 def run():
  global ids
  logits=x.float()@router.T;weights,ids=fused_grouped_topk(x,logits,8,True,bias,1,1,'sigmoid',2.5)
  sgu=_fused_mul_mat_gguf(x,gu,8);sact=torch.empty((nt,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(sact,sgu,10.,1.,0.);shared=_fused_mul_mat_gguf(sact,down,8);values=dict(logits=logits,weights=weights,ids=ids,shared_gate_up=sgu,shared_activated=sact,shared_out=shared)
  if a.experts:
   if nt<=64:rgu=ops.ggml_moe_a8_vec(x,routed_gu,ids,8,gate_type,4096,nt)
   else:
    rgu=routed_matrix(x,routed_gu,gate_type,4096,8,nt)
   ract=torch.empty((nt*8,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(ract,rgu,10.,1.,0.)
   rd=ops.ggml_moe_a8_vec(ract,routed_down,ids,1,down_type,4096,nt*8)if nt<=64 else routed_matrix(ract,routed_down,down_type,4096,1,nt*8)
   weighted=rd.reshape(nt,8,4096).clone().mul_(weights[:,:,None]);routed=torch.empty_like(x);ops.moe_sum(weighted,routed);result=routed+shared
   values.update(routed_gate_up=rgu,routed_activated=ract,routed_down=rd,weighted=weighted,routed_out=routed,out=result)
  torch.cuda.synchronize();return values
 first,repeat=run(),run();assert all(torch.equal(first[k],repeat[k])for k in first);o=a.out/str(ci);o.mkdir(exist_ok=True);values=dict(x=x,**first);records.append(dict(case=ci,tokens=nt,hashes={k:save(o/(k+'.bin'),v)for k,v in values.items()}));print(ci,nt,'captured twice',flush=True)
(a.out/'reference.json').write_text(json.dumps(dict(layer=a.layer,records=records,weights=hashes,complete_moe=a.experts,matrix_reference_profile='Separately qualified corrected GGUF extension with full 288-expert domain' if a.corrected_expert_domain else 'Unchanged CUDA kernel, expert domain partitioned into [0,256) and [256,288) with local IDs; original token/slot geometry preserved; bit-select outputs without arithmetic merging',router_profile='raw GGUF F32 weights, BF16 input expanded to F32, TF32 disabled; grouped single-group sigmoid top8 normalized scale2.5',sources={str(Path(inspect.getfile(f))):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest()for f in [_fused_mul_mat_gguf,fused_grouped_topk,SiluAndMulWithClamp]},gguf_extension_sha256=hashlib.sha256(Path(ops._C_gguf.__file__).read_bytes()).hexdigest()),indent=2)+'\n')
