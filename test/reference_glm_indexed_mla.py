"""Original indexed MLA with explicit separate chronological diagnostic.
Raw original selections/results are preserved separately and never rewritten.
Canonical captures qualify composition only; they do not establish full-model
parity with the original atomic selection order.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch,gguf,vllm
from vllm import _custom_ops as ops
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf as linear
from vllm.models.glm5next.common.sparse_indexer import _fill_causal_indices
from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.v1.worker.workspace import init_workspace_manager
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as attention
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--preindexer',type=Path,required=True);ap.add_argument('--scores',type=Path,required=True);ap.add_argument('--projection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;init_workspace_manager(torch.device('cuda'),num_ubatches=1);decode_topk=get_indexer_topk('auto')
reader=gguf.GGUFReader(str(a.model).replace('00001-of','00002-of'));table={t.name:t for t in reader.tensors}
def weight(name,shape):return torch.from_numpy(np.array(table['blk.3.'+name].data,copy=True).reshape(shape)).cuda()
def load(f,dtype,shape):
 b=f.read_bytes()
 return torch.frombuffer(bytearray(b),dtype=dtype).reshape(shape).cuda()if b else torch.empty(shape,dtype=dtype,device="cuda")
def save(f,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b,f
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
def metric(x,y):
 return dict(words=x.numel(),bit_mismatch=int((x.contiguous().view(torch.uint8)!=y.contiguous().view(torch.uint8)).reshape(-1,x.element_size()).any(dim=1).sum()),relative_l2=float(torch.linalg.vector_norm(x.float()-y.float())/torch.linalg.vector_norm(x.float()).clamp_min(1e-20)),max_abs=float((x.float()-y.float()).abs().max()))
qb=weight('attn_q_b.weight',(16384,1632));kva=weight('attn_kv_a_mqa.weight',(512,4352));kn=weight('attn_kv_a_norm.weight',(512,)).bfloat16();outw=weight('attn_output.weight',(4096,17408));key=load(a.projection/'key-basis.bin',torch.bfloat16,(256,64,512)).permute(1,2,0).contiguous();value=load(a.projection/'value-basis.bin',torch.bfloat16,(512,64,256)).permute(1,2,0).contiguous();work=torch.empty(394*1024*1024,device='cuda',dtype=torch.uint8)
meta=json.loads((a.scores/'reference.json').read_text());records=[];raw_records=[];schedule=['3']
for case in [c for c in meta['records']if c['case']in[0,5,8]]:
 cid=case['case'];steps=case['schedule'];schedule.append(' '.join(map(str,[cid,len(steps),*steps])));blocks=(sum(steps)+63)//64+2;pages=torch.arange(blocks-1,-1,-1,device='cuda',dtype=torch.int32);cache=torch.full((blocks,64,656),0xa5,device='cuda',dtype=torch.uint8);phases=[]
 for pi,phase in enumerate(case['phases']):
  nt,pos,nk=phase['tokens'],phase['position'],phase['keys'];source=a.preindexer/str(cid)/str(pi);x=load(source/'x.bin',torch.bfloat16,(nt,4096));qr=load(source/'qr.bin',torch.bfloat16,(nt,1536));q=linear(qr,qb,8).reshape(nt,64,256);query=torch.empty((nt,64,512),device='cuda',dtype=torch.bfloat16);torch.bmm(q.transpose(0,1),key.transpose(1,2),out=query.transpose(0,1));kr=linear(x,kva,8);kv=torch.empty_like(kr);torch.ops._C.rms_norm(kv,kr,kn,1e-5);positions=torch.arange(pos,pos+nt,device='cuda',dtype=torch.int32);slots=pages[positions//64]*64+positions%64;ops.concat_and_cache_mla(kv,torch.empty((nt,0),device='cuda',dtype=torch.bfloat16),cache,slots.long(),'fp8_ds_mla',torch.ones(1,device='cuda'))
  rawfile=a.scores/str(cid)/str(pi)/'scores.bin';assert hashlib.sha256(rawfile.read_bytes()).hexdigest()==phase['sha256'];scores=load(rawfile,torch.float32,(nt,nk));ends=(positions+1)//4;short=pos+nt<=2048
  pools=torch.full((nt,512),-1,device='cuda',dtype=torch.int32);raw=torch.full((nt,2051),-1,device='cuda',dtype=torch.int32);backend='short'
  if short:_fill_causal_indices(raw,positions.long());canonical=raw.clone()
  else:
   if nt==1:
    backend=decode_topk.resolve_backend(scores,512,nt);decode_topk(scores,ends[:,None],1,pools,512,nk)
   else:
    backend='prefill-per-row';torch.ops._C.top_k_per_row_prefill(scores,torch.zeros_like(ends),ends,pools,nt,nk,1,512)
   raw=expand_pools_and_append_tail(pools.long(),positions+1,4);ordered=torch.where(pools<0,2147483647,pools).sort(dim=-1).values;ordered=torch.where(ordered==2147483647,-1,ordered);canonical=expand_pools_and_append_tail(ordered.long(),positions+1,4)
   # Independently require exact signed-score top-k membership when untied.
   expected=scores.topk(min(512,nk),dim=-1).indices;expected=torch.where(torch.arange(expected.shape[1],device='cuda')[None,:]<ends[:,None],expected,-1);expected=torch.nn.functional.pad(expected,(0,512-expected.shape[1]),value=-1)
   assert torch.equal(pools.sort(dim=-1).values,expected.int().sort(dim=-1).values),(cid,pi,'membership mismatch')
  def run(indices):
   physical=torch.full((nt,1,2176),-1,device='cuda',dtype=torch.int32);safe=indices.clamp_min(0);physical[:,0,:2051]=torch.where(indices>=0,pages[safe//64]*64+safe%64,-1)
   context,lse=attention(query=query[:,None],kv_cache=cache[:,None],workspace_buffer=work,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=physical,seq_lens=None,max_seq_len=2176,sparse_mla_top_k=2176,bmm1_scale=.0625,bmm2_scale=1.,kv_scale_format='arbitrary_fp32',return_lse=True);context=context[:,0];values=torch.empty((nt,64,256),device='cuda',dtype=torch.bfloat16);torch.bmm(context.transpose(0,1),value.transpose(1,2),out=values.transpose(0,1));out=linear(values.reshape(nt,16384),outw,8);torch.cuda.synchronize();return dict(context=context,lse=lse,values=values,out=out,physical=physical)
  first,repeat=run(canonical),run(canonical);assert all(torch.equal(t.contiguous().view(torch.uint8),repeat[name].contiguous().view(torch.uint8))for name,t in first.items());directory=a.out/'canonical'/str(cid)/str(pi);directory.mkdir(parents=True,exist_ok=True);fields=dict(first,query=query,kv=kv,q=q,cache=cache,selection=canonical,table=pages,slots=slots);phases.append(dict(position=pos,tokens=nt,blocks=blocks,short=short,decode_topk_backend=backend,hashes={name:save(directory/(name+'.bin'),t)for name,t in fields.items()}))
  raw_result=run(raw);rawdir=a.out/'raw'/str(cid)/str(pi);rawdir.mkdir(parents=True,exist_ok=True);raw_records.append(dict(case=cid,phase=pi,backend=backend,indices_sha256=save(rawdir/'selection.bin',raw),output_sha256=save(rawdir/'out.bin',raw_result['out']),differences={name:metric(raw_result[name],first[name])for name in ['context','values','out']}));print(cid,pi,pos,nt,backend,'canonical capture repeated; raw result separately preserved',flush=True)
 records.append(dict(case=cid,schedule=steps,phases=phases))
(a.out/'schedules.txt').write_text('\n'.join(schedule)+'\n');(a.out/'reference.json').write_text(json.dumps(dict(records=records,preindexer_manifest_sha256=hashlib.sha256((a.preindexer/'reference.json').read_bytes()).hexdigest(),score_manifest_sha256=hashlib.sha256((a.scores/'reference.json').read_bytes()).hexdigest(),scope='Original operators on explicit chronological selected-set diagnostic. Default raw selections/outputs preserved separately. Does not qualify original-order parity, ties, logits or generation.'),indent=2)+'\n');(a.out/'raw-order.json').write_text(json.dumps(raw_records,indent=2)+'\n')
