"""Capture original default fused mHC profile, including all batch regimes.

Uses actual checkpoint next-layer parameters and original incoming mHC
coefficients. The small-batch reference retains the unrounded F32 residual for
projection/statistics; the large-batch reference uses its default materialized
TF32 path. No native outputs produce oracle values.
"""
from pathlib import Path
import torch,numpy as np,gguf,json,hashlib,inspect,argparse
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang,mhc_post_tilelang,mhc_fused_post_pre_tilelang,_hc_prenorm_gemm_outputs
from vllm.model_executor.kernels.mhc.tilelang_kernels import _MHC_FUSED_TILELANG_KERNEL,mhc_pre_big_fuse_tilelang,mhc_fused_post_pre_split_config
parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();p=args.out;p.mkdir(parents=True,exist_ok=True);table={t.name:t for i in range(1,7)for t in gguf.GGUFReader(str(args.model).replace('00001-of',f'{i:05d}-of')).tensors};records=[];torch.set_num_threads(1)
def weights(layer,kind):
 raw={k:np.array(table[f'blk.{layer}.hc_{kind}_{k}.weight'].data,copy=True)for k in ['fn','scale','base']};raw['norm']=np.array(table[f'blk.{layer}.'+('attn_norm'if kind=='attn'else'ffn_norm')+'.weight'].data,copy=True)
 return [torch.from_numpy(gguf.quants.dequantize(raw['fn'],8).reshape(24,16384)).cuda(),torch.from_numpy(raw['scale'].reshape(3)).cuda(),torch.from_numpy(raw['base'].reshape(24)).cuda(),torch.from_numpy(raw['norm'].reshape(4096)).cuda().bfloat16()],{k:hashlib.sha256(a.tobytes()).hexdigest()for k,a in raw.items()}
for layer,kind,pl,pk in [(0,'ffn',0,'attn'),(1,'attn',0,'ffn'),(44,'ffn',44,'attn'),(44,'attn',43,'ffn')]:
 (fn,scale,base,norm),hashes=weights(layer,kind);(pf,ps,pb,pn),phashes=weights(pl,pk)
 for nt in [1,4,5,9,15,16,32,33,128]:
  o=p/'reference'/f'{layer}-{kind}-{nt}';o.mkdir(parents=True,exist_ok=True);gen=torch.Generator().manual_seed(71930+layer*3+(kind=='ffn'));r=torch.randn((nt,4,4096),generator=gen).bfloat16().cuda();branch=torch.randn((nt,4096),generator=gen).mul_(2).bfloat16().cuda();oldpost,oldcomb,_=mhc_pre_tilelang(r,pf,ps,pb,1e-5,1e-6,1e-6,2.,20)
  def run():
   new,post,comb,x=mhc_fused_post_pre_tilelang(branch,r,oldpost,oldcomb,fn,scale,base,1e-5,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-5)
   _,po,co,raw=mhc_fused_post_pre_tilelang(branch,r,oldpost,oldcomb,fn,scale,base,1e-5,1e-6,1e-6,2.,20)
   assert torch.equal(post,po)and torch.equal(comb,co)
   config=mhc_fused_post_pre_split_config(nt,4096,4)
   if config is not None:dot,sq,res=_MHC_FUSED_TILELANG_KERNEL(oldcomb.reshape(nt,4,4),r,oldpost.reshape(nt,4),branch,fn.reshape(24,4,4096),4,4096,24);assert torch.equal(new,res)
   else:dot,sq=_hc_prenorm_gemm_outputs(new.flatten(1),fn,hidden_size=4096,hc_mult=4)
   pre=torch.empty((nt,4),device='cuda');po=torch.empty_like(post);co=torch.empty_like(comb)
   mhc_pre_big_fuse_tilelang(dot,sq,scale,base,new,po.reshape(nt,4),co.reshape(nt,16),torch.empty_like(raw),torch.zeros_like(pre),pre,torch.empty_like(raw),hidden_size=4096,rms_eps=1e-5,hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=2.,sinkhorn_repeat=20,n_splits=dot.shape[0],hc_mult=4,save_pre_mix=True)
   assert torch.equal(post,po)and torch.equal(comb,co)
   materialized=mhc_post_tilelang(branch,r,oldpost,oldcomb);_,_,materialized_x=mhc_pre_tilelang(materialized,fn,scale,base,1e-5,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-5)
   return dict(residual=new,dot=dot,sq=sq,pre=pre,post=post,comb=comb,collapsed=raw,normalized=x),int((materialized_x!=x).sum())
  data,diff=run();again,diff2=run();assert diff==diff2 and all(torch.equal(data[k],again[k])for k in data)
  sp=data['dot'].shape[0];data.update(old_residual=r,branch=branch,old_post=oldpost,old_comb=oldcomb,fn=fn,scale=scale,base=base,norm=norm);out_hash={}
  for k,t in data.items():buf=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();(o/(k+'.bin')).write_bytes(buf);out_hash[k]=hashlib.sha256(buf).hexdigest()
  row=dict(layer=layer,kind=kind,previous_layer=pl,previous_kind=pk,tokens=nt,splits=sp,config=mhc_fused_post_pre_split_config(nt,4096,4),materialized_mismatches=diff,hashes=out_hash,weights=hashes,previous_weights=phashes);records.append(row);print(layer,kind,nt,sp,row['config'],'materialized differs',diff,'captured twice',flush=True)
(p/'reference.json').write_text(json.dumps(dict(records=records,sources={str(Path(inspect.getfile(f))):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest()for f in [mhc_pre_tilelang,mhc_fused_post_pre_tilelang,mhc_post_tilelang]}),indent=2)+'\n')
