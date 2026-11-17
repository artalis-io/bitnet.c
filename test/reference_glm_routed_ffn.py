"""Actual GLM routed Q4_K/Q5_K execution with explicit model clamp.
Routing is an input to this execution reference; it does not qualify the router.
"""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np,torch,gguf
from vllm_gguf_plugin import ops
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
rs=[gguf.GGUFReader(str(a.model).replace('00001-of',f'{i:05d}-of'))for i in range(1,7)];table={t.name:t for r in rs for t in r.tensors}
assert all(ops._cuda_kernel_available('ggml_moe_a8_vec',q) and ops._cuda_gemm_kernel_available('ggml_moe_a8',q)for q in [12,13])
experts=[0,1,17,71,128,191,211,233,256,271,283,287];raw={};hashes={}
for name,typ,rows,cols,size in [('gate',12,2048,4096,144),('up',12,2048,4096,144),('down',13,4096,2048,176)]:
 t=table[f'blk.3.ffn_{name}_exps.weight'];assert int(t.tensor_type)==typ
 data=np.array(t.data.reshape(288,rows,cols//256*size)[experts],copy=True);hashes[name]=hashlib.sha256(data.tobytes()).hexdigest();raw[name]=torch.from_numpy(data).cuda()
w13=torch.cat([raw['gate'],raw['up']],1).contiguous();records=[]
def save(path,t):
 data=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();path.write_bytes(data);return hashlib.sha256(data).hexdigest()
for ci,nt in enumerate([1,2,9,64,65,129]):
 out=a.out/str(ci);out.mkdir(exist_ok=True);gen=torch.Generator().manual_seed(87619+ci);x=torch.randn((nt,4096),generator=gen).mul_(4).bfloat16().cuda()
 local=((torch.arange(nt)[:,None]*7+torch.arange(8)[None,:])%12).int().cuda();actual=torch.tensor(experts,device='cuda',dtype=torch.int32)[local.long()]
 weights=((torch.arange(nt*8).reshape(nt,8)%31+1).float());weights=weights/weights.sum(-1,keepdim=True)*2.5;weights=weights.cuda()
 def run():
  if nt<=64:
   gu=ops.ggml_moe_a8_vec(x,w13,local,8,12,4096,nt)
   act=torch.empty((nt*8,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(act,gu,10.,1.,0.)
   down=ops.ggml_moe_a8_vec(act,raw['down'],local,1,13,4096,nt*8)
  else:
   block=ops.ggml_moe_get_block_size(12);sorted_ids,expert_ids,padded=moe_align_block_size(local,block,12)
   gu=ops.ggml_moe_a8(x,w13,sorted_ids,expert_ids,padded,12,4096,8,nt)
   act=torch.empty((nt*8,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(act,gu,10.,1.,0.)
   down=ops.ggml_moe_a8(act,raw['down'],sorted_ids,expert_ids,padded,13,4096,1,nt*8)
  weighted=down.reshape(nt,8,4096).clone().mul_(weights[:,:,None]);result=torch.empty_like(x);ops.moe_sum(weighted,result)
  torch.cuda.synchronize();return dict(gate_up=gu,activated=act,down=down,weighted=weighted,out=result)
 first,repeat=run(),run();assert all(torch.equal(first[k],repeat[k])for k in first)
 record=dict(case=ci,tokens=nt,hashes={k:save(out/(k+'.bin'),v)for k,v in dict(x=x,ids=actual,weights=weights,**first).items()});records.append(record);print(ci,nt,'captured twice',flush=True)
(a.out/'reference.json').write_text(json.dumps(dict(experts=experts,weights=hashes,records=records,profile='Explicit GLM clamp with original GGUF routed operators; router inputs provided',cuda_backend=True,gguf_extension_sha256=hashlib.sha256(Path(ops._C_gguf.__file__).read_bytes()).hexdigest(),activation_source_sha256=hashlib.sha256(Path(inspect.getfile(SiluAndMulWithClamp)).read_bytes()).hexdigest()),indent=2)+'\n')
