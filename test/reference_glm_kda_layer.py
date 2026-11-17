"""Capture actual GLM KDA sublayers with original GGUF/vLLM primitives.

Requires the GLM-enabled vLLM and GGUF plugin environments. This captures the
recurrent precision profile, not optimized chunk_kda or a whole vLLM model.
Example: python test/reference_glm_kda_layer.py --model MODEL.gguf --out /tmp/ref
"""
from pathlib import Path
import json,hashlib,inspect,os,argparse
import numpy as np,torch,gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--mhc-reference',type=Path,help='Original mHC capture reference directory for a materialized attention transition');args=parser.parse_args();p=args.out;p.mkdir(parents=True,exist_ok=True);p.joinpath('reference').mkdir(exist_ok=True);torch.set_num_threads(1)
model=args.model;readers=[gguf.GGUFReader(str(model).replace('00001-of','%05d-of'%i))for i in range(1,7)];table={t.name:t for r in readers for t in r.tensors};records=[]
for layer in [0,44]:
 o=p/'reference'/f'layer{layer}';o.mkdir(exist_ok=True);names={k:f'blk.{layer}.{v}'for k,v in dict(q='attn_q.weight',k='attn_k.weight',v='attn_v.weight',beta='ssm_beta.weight',fa='ssm_f_a.weight',ga='ssm_g_a.weight',fb='ssm_f_b.weight',gb='ssm_g_b.weight',out='attn_output.weight',cq='ssm_conv1d_q.weight',ck='ssm_conv1d_k.weight',cv='ssm_conv1d_v.weight',a='ssm_a',bias='ssm_dt.bias',norm='ssm_norm.weight').items()};weight_sha={};raw={}
 for key,name in names.items():
  t=table[name];a=np.array(t.data,copy=True);weight_sha[name]=hashlib.sha256(a.tobytes()).hexdigest();raw[key]=torch.from_numpy(a).cuda()
 for k in ['q','k','v','beta','fa','ga','fb','gb','out']:assert int(table[names[k]].tensor_type)==8
 merged=torch.cat([raw[k].reshape(-1,4352)for k in ['q','k','v','beta','fa','ga']],0).contiguous();assert merged.shape==(24896,4352)
 for key,cols in [('fb',128),('gb',128),('out',8192)]:raw[key]=raw[key].reshape(-1,cols//32*34)
 convw=torch.cat([raw[k].reshape(8192,4)for k in ['cq','ck','cv']],0).float();a=raw['a'].float().reshape(64);bias=raw['bias'].float().reshape(64,128);gamma=raw['norm'].float().reshape(128);assert (a<=0).all()
 gen=torch.Generator().manual_seed(27183+layer);x=torch.randn((9,4096),generator=gen).bfloat16().cuda();history=torch.randn((3,24576),generator=gen).mul_(.01).bfloat16().cuda();initial=torch.randn((1,64,128,128),generator=gen).mul_(.001).cuda()
 if args.mhc_reference:
  from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
  mhc_dir=args.mhc_reference/f'{layer}-attn-9'
  x=torch.from_numpy(np.fromfile(mhc_dir/'normed.bin',dtype=np.uint16).copy()).view(torch.bfloat16).reshape(9,4096).cuda()
  residual=torch.from_numpy(np.fromfile(mhc_dir/'residual.bin',dtype=np.uint16).copy()).view(torch.bfloat16).reshape(9,4,4096).cuda()
  post=torch.from_numpy(np.fromfile(mhc_dir/'post.bin',dtype=np.float32).copy()).reshape(9,4,1).cuda()
  comb=torch.from_numpy(np.fromfile(mhc_dir/'comb.bin',dtype=np.float32).copy()).reshape(9,4,4).cuda()
 def save(name,t):
  a=t.detach().contiguous().cpu();flat=torch.empty(a.numel(),dtype=a.dtype);flat.copy_(a.reshape(-1));data=flat.view(torch.uint8).numpy().tobytes();(o/name).write_bytes(data);return hashlib.sha256(data).hexdigest()
 save('input.bin',x);save('history0.bin',history);save('initial.bin',initial)
 schedules=[[9],[2,7],[3,3,3],[7,1,1],[6,3],[1]*9]
 for ci,steps in enumerate(schedules):
  def run():
   hist=torch.stack([torch.zeros_like(history.T),history.T]).contiguous();idx=torch.tensor([1],device='cuda',dtype=torch.int32);state=initial.clone();offset=0;out=[]
   for take in steps:
    inp=x[offset:offset+take];projected=_fused_mul_mat_gguf(inp,merged,8);qkv,b,fa,ga=projected.split([24576,64,128,128],dim=-1)
    # GGUF A8 primitives consume contiguous activation rows. Materialize these
    # logical views explicitly; no stride-oblivious input is used as an oracle.
    # update writes its input: clone even when a one-row view is contiguous,
    # preserving the original merged projection for independent comparison.
    g1=_fused_mul_mat_gguf(fa.contiguous(),raw['fb'],8);g2=_fused_mul_mat_gguf(ga.contiguous(),raw['gb'],8)
    if take==1:conv=causal_conv1d_update(qkv.contiguous().clone(),hist,convw,activation='silu',conv_state_indices=idx)
    else:conv=causal_conv1d_fn(qkv.T,convw,None,hist,torch.tensor([0,take],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([True],device='cuda')).T.contiguous()
    q,k,v=[s.reshape(1,take,64,128)for s in conv.split(8192,dim=-1)];scaled=(-a[None,None,:,None])*(g1.reshape(1,take,64,128).float()+bias)
    core,states=fused_recurrent_kda(q,k,v,scaled,b.contiguous().reshape(1,take,64),scale=float(np.float32(128**-.5)),initial_state=state,inplace_final_state=False,use_qk_l2norm_in_kernel=True,sigmoid_beta=True,a_log=torch.zeros_like(a),g_bias=torch.zeros_like(bias),compute_gate=True,lower_bound=-5.)
    state=states[-1:].clone();gated=rms_norm_gated(core.clone(),g2.reshape(take,64,128),gamma,None,activation='sigmoid',eps=1e-5);result=_fused_mul_mat_gguf(gated.reshape(take,8192).contiguous(),raw['out'],8)
    values=dict(projected=projected,g1=g1,g2=g2,conv=conv,core=core,gated=gated,out=result,state=state,history=hist[1].T.contiguous())
    if args.mhc_reference:values['transition']=mhc_post_tilelang(result,residual[offset:offset+take],post[offset:offset+take],comb[offset:offset+take])
    out.append({k:v.clone()for k,v in values.items()});offset+=take
   torch.cuda.synchronize();return out
  result=run();repeat=run();assert all(torch.equal(a[k],b[k])for a,b in zip(result,repeat)for k in a);hashes=[]
  for phase,row in enumerate(result):hashes.append({key:save(f'{ci}-{phase}-{key}.bin',value)for key,value in row.items()})
  records.append(dict(layer=layer,case=ci,steps=steps,hashes=hashes,weights=weight_sha));print('layer',layer,'case',ci,steps,'captured and repeated',flush=True)
 del raw,merged,convw,x,history,initial
(p/'reference.json').write_text(json.dumps(dict(records=records,torch=torch.__version__,sources={str(Path(inspect.getfile(fn))):hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()for fn in [_fused_mul_mat_gguf,causal_conv1d_fn,fused_recurrent_kda,rms_norm_gated]}),indent=2)+'\n')
