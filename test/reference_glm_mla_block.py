"""Original complete block3 from the independently captured dense-prefix boundary.
Default deferred mHC, short causal MLA and unchanged full-domain MoE vector
operators with checkpoint clamp. Prefix inputs are never produced by native code.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch,gguf
from vllm import _custom_ops as custom
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf as linear
from vllm_gguf_plugin import ops as ggops
from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang as fused_mhc
from vllm.models.glm5next.common.attention import _fused_indexer_k_norm,_fused_indexer_weight_scale
from vllm.models.glm5next.common.sparse_indexer import _fill_causal_indices
from vllm.models.glm5next.nvidia.ops import kpool_compress as poolops
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as attention
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--prefix',type=Path,required=True);ap.add_argument('--projection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
readers=[gguf.GGUFReader(str(a.model).replace('00001-of',f'{i:05d}-of'))for i in range(1,7)];table={t.name:t for r in readers for t in r.tensors};hashes={}
def tensor(name,shape,typ=None):
 t=table['blk.3.'+name];raw=np.array(t.data,copy=True);hashes[name]=hashlib.sha256(raw.tobytes()).hexdigest()
 if typ is not None:assert int(t.tensor_type)==typ
 return torch.from_numpy(raw.reshape(shape)).cuda()
def load(f,dtype,shape):return torch.frombuffer(bytearray(f.read_bytes()),dtype=dtype).reshape(shape).cuda()
def save(f,t):
 b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 if f.exists():assert f.read_bytes()==b,f
 else:f.write_bytes(b)
 return hashlib.sha256(b).hexdigest()
mhc=[]
for kind in ['attn','ffn']:
 t=table[f'blk.3.hc_{kind}_fn.weight'];raw=np.array(t.data,copy=True);hashes[f'hc_{kind}_fn.weight']=hashlib.sha256(raw.tobytes()).hexdigest();fn=torch.from_numpy(gguf.quants.dequantize(raw,int(t.tensor_type)).reshape(24,16384)).cuda();mhc.append([fn,tensor(f'hc_{kind}_scale.weight',(3,)),tensor(f'hc_{kind}_base.weight',(24,)),tensor(('attn_norm'if kind=='attn'else'ffn_norm')+'.weight',(4096,)).bfloat16()])
qa=tensor('attn_q_a.weight',(1536,4352),8);qb=tensor('attn_q_b.weight',(16384,1632),8);kva=tensor('attn_kv_a_mqa.weight',(512,4352),8);qn=tensor('attn_q_a_norm.weight',(1536,)).bfloat16();kn=tensor('attn_kv_a_norm.weight',(512,)).bfloat16();ow=tensor('attn_output.weight',(4096,17408),8)
key=load(a.projection/'key-basis.bin',torch.bfloat16,(256,64,512)).permute(1,2,0).contiguous();value=load(a.projection/'value-basis.bin',torch.bfloat16,(512,64,256)).permute(1,2,0).contiguous()
iq=tensor('indexer.attn_q_b.weight',(4096,1632),8);ik=tensor('indexer.attn_k.weight',(128,4352),8);ig=tensor('indexer_compressor_gate.weight',(128,4352),8);ih=tensor('indexer.proj.weight',(32,4096),0);inn=tensor('indexer.k_norm.weight',(128,),0);ibias=tensor('indexer.k_norm.bias',(128,),0);ape=tensor('indexer_compressor_ape.weight',(4,128),0)
router=tensor('ffn_gate_inp.weight',(288,4096),0);bias=tensor('exp_probs_b.bias',(288,),0);sgu=torch.cat([tensor(f'ffn_{r}_shexp.weight',(2048,4352),8)for r in ['gate','up']],0).contiguous();sd=tensor('ffn_down_shexp.weight',(4096,2176),8);rgu=torch.cat([tensor(f'ffn_{r}_exps.weight',(288,2048,2304),12)for r in ['gate','up']],1).contiguous();rd=tensor('ffn_down_exps.weight',(288,4096,1408),13)
workspace=torch.empty(394*1024*1024,device='cuda',dtype=torch.uint8);meta=json.loads((a.prefix/'reference.json').read_text());records=[]
for case in meta['records']:
 resumed,cid,steps=case['resumed'],case['case'],case['steps'];source=a.prefix/f'{resumed}-{cid}';inputs=[]
 for phase,nt in enumerate(steps):
  inputs.append([load(source/f'{phase}-layer2-{name}.bin',dtype,shape)for name,dtype,shape in [('out',torch.bfloat16,(nt,4096)),('ffn_residual',torch.bfloat16,(nt,4,4096)),('ffn_post',torch.float32,(nt,4)),('ffn_comb',torch.float32,(nt,4,4))]])
 def run():
  latent=torch.full((3,64,656),0xa5,device='cuda',dtype=torch.uint8);index_cache=torch.full((3,64,132),0xa5,device='cuda',dtype=torch.uint8);tail=torch.zeros((1,2,4,128),device='cuda',dtype=torch.bfloat16);pages=torch.tensor([2,1,0],device='cuda',dtype=torch.int32);position=0;results=[]
  for nt,incoming in zip(steps,inputs):
   branch,residual,post,comb=incoming;ar,ap,ac,x=fused_mhc(branch,residual,post,comb,*mhc[0][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=mhc[0][3],norm_eps=1e-5);qr0=linear(x,qa,8);qr=torch.empty_like(qr0);custom.rms_norm(qr,qr0,qn,1e-5);q=linear(qr,qb,8).reshape(nt,64,256);query=torch.empty((nt,64,512),device='cuda',dtype=torch.bfloat16);torch.bmm(q.transpose(0,1),key.transpose(1,2),out=query.transpose(0,1));kr=linear(x,kva,8);kv=torch.empty_like(kr);custom.rms_norm(kv,kr,kn,1e-5);pos=torch.arange(position,position+nt,device='cuda',dtype=torch.int32);slots=pages[pos//64]*64+pos%64;custom.concat_and_cache_mla(kv,torch.empty((nt,0),device='cuda',dtype=torch.bfloat16),latent,slots.long(),'fp8_ds_mla',torch.ones(1,device='cuda'))
   ibq=linear(qr,iq,8);fp8,scales=poolops.fwht128_quant_fp8(ibq.reshape(-1,128));keyraw=linear(x,ik,8);ikn=_fused_indexer_k_norm(keyraw,inn,ibias,128,1e-6);gate=linear(x,ig,8);head=x.float()@ih.T;head_weights=_fused_indexer_weight_scale(head,scales.reshape(nt,32,1),.015625);initial_tail=tail.clone();pool_ids=pos//4;loc=pages[pool_ids//64]*64+pool_ids%64;loc=torch.where((pos+1)%4==0,loc,-1);poolops.kpool_decode_update_and_maybe_write_cache_batched(index_cache,tail,(pos%4)[None],ikn[None],gate[None],ape,loc[None],pos[None],4)
   selection=torch.full((nt,2051),-1,device='cuda',dtype=torch.int32);_fill_causal_indices(selection,pos.long());safe=selection.clamp_min(0);physical=torch.full((nt,1,2176),-1,device='cuda',dtype=torch.int32);physical[:,0,:2051]=torch.where(selection>=0,pages[safe//64]*64+safe%64,-1)
   context,lse=attention(query=query[:,None],kv_cache=latent[:,None],workspace_buffer=workspace,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=physical,seq_lens=None,max_seq_len=2176,sparse_mla_top_k=2176,bmm1_scale=.0625,bmm2_scale=1.,kv_scale_format='arbitrary_fp32',return_lse=True);context=context[:,0];values=torch.empty((nt,64,256),device='cuda',dtype=torch.bfloat16);torch.bmm(context.transpose(0,1),value.transpose(1,2),out=values.transpose(0,1));attn_out=linear(values.reshape(nt,16384),ow,8);fr,fp,fc,fx=fused_mhc(attn_out,ar,ap,ac,*mhc[1][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=mhc[1][3],norm_eps=1e-5)
   logits=fx.float()@router.T;weights,ids=fused_grouped_topk(fx,logits,8,True,bias,1,1,'sigmoid',2.5);shared_gu=linear(fx,sgu,8);shared_act=torch.empty((nt,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(shared_act,shared_gu,10.,1.,0.);shared_out=linear(shared_act,sd,8);routed_gu=ggops.ggml_moe_a8_vec(fx,rgu,ids,8,12,4096,nt);routed_act=torch.empty((nt*8,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(routed_act,routed_gu,10.,1.,0.);routed_down=ggops.ggml_moe_a8_vec(routed_act,rd,ids,1,13,4096,nt*8);weighted=routed_down.reshape(nt,8,4096).clone().mul_(weights[:,:,None]);routed=torch.empty_like(fx);ggops.moe_sum(weighted,routed);out=routed+shared_out
   results.append({name:t.clone()for name,t in dict(attn_residual=ar,attn_post=ap,attn_comb=ac,input=x,qr=qr,query=query,kv=kv,latent_cache=latent,index_cache=index_cache,index_query=ibq,index_key_raw=keyraw,index_key=ikn,index_gate=gate,index_head=head,index_q_fp8=fp8,index_q_scales=scales,index_weights=head_weights,initial_tail=initial_tail,next_tail=tail,selection=selection,physical=physical,context=context,lse=lse,values=values,attn_out=attn_out,ffn_residual=fr,ffn_post=fp,ffn_comb=fc,ffn_input=fx,logits=logits,route_weights=weights,ids=ids,shared_gate_up=shared_gu,shared_activated=shared_act,shared_out=shared_out,routed_gate_up=routed_gu,routed_activated=routed_act,routed_down=routed_down,weighted=weighted,routed_out=routed,out=out).items()});position+=nt
  torch.cuda.synchronize();return results
 first,repeat=run(),run();assert all(torch.equal(t.contiguous().view(torch.uint8),repeat[phase][name].contiguous().view(torch.uint8))for phase,row in enumerate(first)for name,t in row.items());dest=a.out/f'{resumed}-{cid}';dest.mkdir(exist_ok=True);hashes2=[]
 for phase,row in enumerate(first):hashes2.append({name:save(dest/f'{phase}-{name}.bin',t)for name,t in row.items()})
 records.append(dict(resumed=resumed,case=cid,steps=steps,hashes=hashes2));print(resumed,cid,steps,'complete block captured twice',flush=True)
(a.out/'reference.json').write_text(json.dumps(dict(records=records,weights=hashes,prefix_manifest_sha256=hashlib.sha256((a.prefix/'reference.json').read_bytes()).hexdigest(),original_extension_sha256=hashlib.sha256(Path(ggops._C_gguf.__file__).read_bytes()).hexdigest(),profile='Actual dense-prefix deferred boundary -> default fused mHC -> short causal NoPE MLA -> default fused FFN pre -> raw F32 grouped router/shared+routed vectors, checkpoint clamp10. Original vector operators support full288 expert domain. No long-order/tie, whole-plugin, default HF merged-loader or whole-model generation claim.'),indent=2)+'\n')
