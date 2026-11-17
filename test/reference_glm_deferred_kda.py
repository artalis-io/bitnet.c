"""Original default FFN-post -> attention-pre -> KDA -> FFN-pre capture."""
from pathlib import Path
import argparse,json,hashlib,inspect,torch,numpy as np,gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--incoming-reference',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--moe',action='store_true');args=parser.parse_args();p=args.out;p.mkdir(parents=True,exist_ok=True);table={t.name:t for i in range(1,7)for t in gguf.GGUFReader(str(args.model).replace('00001-of',f'{i:05d}-of')).tensors};records=[];torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
for layer in ([44]if args.moe else[1,44]):
 out=p/'reference'/f'layer{layer}';out.mkdir(parents=True,exist_ok=True);raw={};weight_sha={}
 def tensor(suffix):
  t=table[f'blk.{layer}.{suffix}'];a=np.array(t.data,copy=True);weight_sha[suffix]=hashlib.sha256(a.tobytes()).hexdigest();return torch.from_numpy(a).cuda()
 for k,suffix in dict(q='attn_q.weight',k='attn_k.weight',v='attn_v.weight',beta='ssm_beta.weight',fa='ssm_f_a.weight',ga='ssm_g_a.weight',fb='ssm_f_b.weight',gb='ssm_g_b.weight',out='attn_output.weight',cq='ssm_conv1d_q.weight',ck='ssm_conv1d_k.weight',cv='ssm_conv1d_v.weight',a='ssm_a',bias='ssm_dt.bias',norm='ssm_norm.weight').items():raw[k]=tensor(suffix)
 assert all(int(table[f'blk.{layer}.{n}'].tensor_type)==8 for n in ['attn_q.weight','attn_k.weight','attn_v.weight','ssm_beta.weight','ssm_f_a.weight','ssm_g_a.weight','ssm_f_b.weight','ssm_g_b.weight','attn_output.weight'])
 merged=torch.cat([raw[k].reshape(-1,4352)for k in ['q','k','v','beta','fa','ga']],0).contiguous()
 for k,c in [('fb',128),('gb',128),('out',8192)]:raw[k]=raw[k].reshape(-1,c//32*34)
 cw=torch.cat([raw[k].reshape(8192,4)for k in ['cq','ck','cv']],0).float();decay=raw['a'].reshape(64).float();bias=raw['bias'].reshape(64,128).float();gamma=raw['norm'].reshape(128).float()
 def mhc(kind):
  name=f'hc_{kind}_fn.weight';a=np.array(table[f'blk.{layer}.{name}'].data,copy=True);weight_sha[name]=hashlib.sha256(a.tobytes()).hexdigest();fn=torch.from_numpy(gguf.quants.dequantize(a,8).reshape(24,16384)).cuda()
  return [fn,tensor(f'hc_{kind}_scale.weight').reshape(3),tensor(f'hc_{kind}_base.weight').reshape(24),tensor(('attn_norm'if kind=='attn'else'ffn_norm')+'.weight').reshape(4096).bfloat16()]
 attn,ffn=mhc('attn'),mhc('ffn');d=args.incoming_reference/f'{layer}-attn-9'
 if args.moe:
  from vllm_gguf_plugin import ops
  from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
  router=tensor('ffn_gate_inp.weight').reshape(288,4096);rbias=tensor('exp_probs_b.bias').reshape(288)
  sgu=torch.cat([tensor(f'ffn_{r}_shexp.weight').reshape(2048,4352)for r in ['gate','up']],0).contiguous();sd=tensor('ffn_down_shexp.weight').reshape(4096,2176)
  rgu=torch.cat([tensor(f'ffn_{r}_exps.weight').reshape(288,2048,2304)for r in ['gate','up']],1).contiguous();rd=tensor('ffn_down_exps.weight').reshape(288,4096,1680)
  assert all(int(table[f'blk.{layer}.ffn_{r}_exps.weight'].tensor_type)==12 for r in ['gate','up'])and int(table[f'blk.{layer}.ffn_down_exps.weight'].tensor_type)==14

 def read(name,shape,bf=False):
  a=torch.from_numpy(np.fromfile(d/(name+'.bin'),dtype=np.uint16 if bf else np.float32).copy());return(a.view(torch.bfloat16)if bf else a).reshape(shape).cuda()
 oldres=read('old_residual',(9,4,4096),True);oldbranch=read('branch',(9,4096),True);oldpost=read('old_post',(9,4));oldcomb=read('old_comb',(9,4,4));gen=torch.Generator().manual_seed(83019+layer);history=torch.randn((3,24576),generator=gen).mul_(.01).bfloat16().cuda();initial=torch.randn((1,64,128,128),generator=gen).mul_(.001).cuda()
 def save(name,t):
  b=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();dest=out/(name+'.bin')
  if dest.exists():assert dest.read_bytes()==b,dest
  else:dest.write_bytes(b)
  return hashlib.sha256(b).hexdigest()
 for k,t in dict(old_residual=oldres,old_branch=oldbranch,old_post=oldpost,old_comb=oldcomb,history0=history,initial=initial).items():save(k,t)
 for ci,steps in enumerate([[9],[2,7],[3,3,3],[7,1,1],[6,3],[1]*9]):
  def run():
   hist=torch.stack([torch.zeros_like(history.T),history.T]).contiguous();idx=torch.tensor([1],device='cuda',dtype=torch.int32);state=initial.clone();off=0;outputs=[]
   for t in steps:
    r,post,comb,x=mhc_fused_post_pre_tilelang(oldbranch[off:off+t],oldres[off:off+t],oldpost[off:off+t],oldcomb[off:off+t],*attn[:3],1e-5,1e-6,1e-6,2.,20,norm_weight=attn[3],norm_eps=1e-5)
    projected=_fused_mul_mat_gguf(x.contiguous(),merged,8);qkv,b,fa,ga=projected.split([24576,64,128,128],-1);g1=_fused_mul_mat_gguf(fa.contiguous(),raw['fb'],8);g2=_fused_mul_mat_gguf(ga.contiguous(),raw['gb'],8)
    if t==1:conv=causal_conv1d_update(qkv.contiguous().clone(),hist,cw,activation='silu',conv_state_indices=idx)
    else:conv=causal_conv1d_fn(qkv.T,cw,None,hist,torch.tensor([0,t],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([True],device='cuda')).T.contiguous()
    q,k,v=[a.reshape(1,t,64,128)for a in conv.split(8192,-1)];gate=(-decay[None,None,:,None])*(g1.reshape(1,t,64,128).float()+bias)
    core,states=fused_recurrent_kda(q,k,v,gate,b.contiguous().reshape(1,t,64),scale=float(np.float32(128**-.5)),initial_state=state,inplace_final_state=False,use_qk_l2norm_in_kernel=True,sigmoid_beta=True,a_log=torch.zeros_like(decay),g_bias=torch.zeros_like(bias),compute_gate=True,lower_bound=-5.);state=states[-1:].clone();gated=rms_norm_gated(core.clone(),g2.reshape(t,64,128),gamma,None,activation='sigmoid',eps=1e-5);result=_fused_mul_mat_gguf(gated.reshape(t,8192).contiguous(),raw['out'],8)
    fr,fp,fc,fx=mhc_fused_post_pre_tilelang(result,r,post,comb,*ffn[:3],1e-5,1e-6,1e-6,2.,20,norm_weight=ffn[3],norm_eps=1e-5)
    values=dict(input=x,entry_residual=r,entry_post=post,entry_comb=comb,projected=projected,g1=g1,g2=g2,conv=conv,core=core,gated=gated,out=result,state=state,history=hist[1].T.contiguous(),ffn_residual=fr,ffn_post=fp,ffn_comb=fc,ffn_input=fx);
    if args.moe:
     logits=fx.float()@router.T;rw,ids=fused_grouped_topk(fx,logits,8,True,rbias,1,1,'sigmoid',2.5)
     sg=_fused_mul_mat_gguf(fx,sgu,8);sa=torch.empty((t,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(sa,sg,10.,1.,0.);so=_fused_mul_mat_gguf(sa,sd,8)
     rg=ops.ggml_moe_a8_vec(fx,rgu,ids,8,12,4096,t);ra=torch.empty((t*8,2048),device='cuda',dtype=torch.bfloat16);torch.ops._C.silu_and_mul_with_clamp(ra,rg,10.,1.,0.);rdown=ops.ggml_moe_a8_vec(ra,rd,ids,1,14,4096,t*8)
     weighted=rdown.reshape(t,8,4096).clone().mul_(rw[:,:,None]);rout=torch.empty_like(fx);ops.moe_sum(weighted,rout)
     values.update(moe_logits=logits,moe_weights=rw,moe_ids=ids,moe_shared_gate_up=sg,moe_shared_activated=sa,moe_shared_out=so,moe_routed_gate_up=rg,moe_routed_activated=ra,moe_routed_down=rdown,moe_weighted=weighted,moe_routed_out=rout,moe_out=rout+so)
    outputs.append({k:v.clone()for k,v in values.items()});off+=t
   return outputs
  a=run();b=run();assert all(torch.equal(x[k],y[k])for x,y in zip(a,b)for k in x);hashes=[]
  for phase,values in enumerate(a):hashes.append({k:save(f'{ci}-{phase}-{k}',v)for k,v in values.items()})
  records.append(dict(layer=layer,case=ci,steps=steps,weights=weight_sha.copy(),hashes=hashes));print(layer,ci,steps,'captured twice',flush=True)
(p/'reference.json').write_text(json.dumps(dict(records=records,sources={str(Path(inspect.getfile(f))):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest()for f in [_fused_mul_mat_gguf,mhc_fused_post_pre_tilelang,causal_conv1d_fn,fused_recurrent_kda,rms_norm_gated]}),indent=2)+'\n')
