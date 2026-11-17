"""Model-free original vLLM writer/FlashInfer MLA attention oracles."""
from pathlib import Path
import argparse,torch,json,hashlib
from vllm import _custom_ops as ops
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as attention
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--header',type=Path,required=True);a=ap.parse_args();a.directory.mkdir(parents=True,exist_ok=True);records=[]
def save(f,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b
 else:f.write_bytes(b)
 h=14695981039346656037
 for x in b:h=((h^x)*1099511628211)&((1<<64)-1)
 return h
workspace=torch.empty(394*1024*1024,device='cuda',dtype=torch.uint8)
for position in [0,4096]:
 for nt in [1,2,3,64,65,129]:
  end=position+nt;blocks=(end+63)//64+2;i=torch.arange(end*512,dtype=torch.int64);kv=(((i*37+i//13)%509-254).float()/128).bfloat16().reshape(end,512).cuda();j=torch.arange(nt*64*512,dtype=torch.int64);q=(((j*17+j//11)%251-125).float()/64).bfloat16().reshape(nt,1,64,512).cuda();table=torch.arange(blocks-1,-1,-1,device='cuda',dtype=torch.int32);logical=torch.arange(end,device='cuda',dtype=torch.int32);slots=table[logical//64]*64+logical%64;selection=torch.full((nt,2051),-1,device='cuda',dtype=torch.int32)
  for row in range(nt):
   visible=position+row+1
   if visible<=2051:selection[row,:visible]=logical[:visible]
   else:selection[row,:2048]=logical[(torch.arange(2048,device='cuda')*37)%2048];selection[row,2048:]=logical[visible-3:visible]
  physical=torch.full((nt,1,2176),-1,device='cuda',dtype=torch.int32);valid=selection>=0;physical[:,0,:2051]=torch.where(valid,table[selection.clamp_min(0)//64]*64+selection.clamp_min(0)%64,-1)
  def run():
   cache=torch.full((blocks,64,656),0xa5,device='cuda',dtype=torch.uint8);ops.concat_and_cache_mla(kv,torch.empty((end,0),device='cuda',dtype=torch.bfloat16),cache,slots.long(),'fp8_ds_mla',torch.ones(1,device='cuda'))
   out,lse=attention(query=q,kv_cache=cache.unsqueeze(1),workspace_buffer=workspace,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=physical,seq_lens=None,max_seq_len=2176,sparse_mla_top_k=2176,bmm1_scale=.0625,bmm2_scale=1.,kv_scale_format='arbitrary_fp32',return_lse=True);torch.cuda.synchronize();return cache,out,lse
  first,repeat=run(),run();assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8))for x,y in zip(first,repeat));d=a.directory/f'{position}-{nt}';d.mkdir(exist_ok=True);record=dict(position=position,tokens=nt,blocks=blocks)
  for name,t in zip(['cache','out','lse','physical'],[*first,physical]):record[name]=save(d/(name+'.bin'),t)
  records.append(record);print(position,nt,'captured twice',flush=True)
lines=['/* Original model-free NoPE FP8 writer and FlashInfer sparse attention. */','#ifndef BN_CUDA_GLM_MLA_ATTENTION_REFERENCE_H','#define BN_CUDA_GLM_MLA_ATTENTION_REFERENCE_H','typedef struct {int position,tokens,blocks;uint64_t cache,out,lse,physical;} BnGLMMLAAttentionReference;','static const BnGLMMLAAttentionReference glm_mla_attention_reference[] = {']
lines+=[' {%d,%d,%d,UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x),UINT64_C(0x%016x)},'%tuple(r[k]for k in ['position','tokens','blocks','cache','out','lse','physical'])for r in records];a.header.write_text('\n'.join(lines+['};','#endif'])+'\n');(a.directory/'reference.json').write_text(json.dumps(records,indent=2)+'\n')
