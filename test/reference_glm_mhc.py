"""Capture checkpoint mHC with original vLLM SM120 DeepGEMM/TileLang ops.

CUDA_HOME and PATH must select a supported compiler (>=12.9). This generator
preserves the operator's default split estimator. It does not use native outputs.
"""
from pathlib import Path
import torch,numpy as np,gguf,json,hashlib,inspect,argparse
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang,mhc_post_tilelang,_hc_prenorm_gemm_outputs
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_pre_big_fuse_tilelang
parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();p=args.out;p.mkdir(parents=True,exist_ok=True);table={t.name:t for i in range(1,7)for t in gguf.GGUFReader(str(args.model).replace('00001-of',f'{i:05d}-of')).tensors};records=[];torch.set_num_threads(1)
print('deep_gemm_supported',is_deep_gemm_supported(),flush=True)
for layer,kind in [(0,'attn'),(0,'ffn'),(44,'attn'),(44,'ffn')]:
 prefix=f'blk.{layer}.hc_{kind}_';raw={k:np.array(table[prefix+k+'.weight'].data,copy=True)for k in ['fn','base','scale']};assert int(table[prefix+'fn.weight'].tensor_type)==8
 fn=torch.from_numpy(gguf.quants.dequantize(raw['fn'],8).reshape(24,16384)).cuda();base=torch.from_numpy(raw['base'].reshape(24)).cuda();scale=torch.from_numpy(raw['scale'].reshape(3)).cuda();normraw=np.array(table[f'blk.{layer}.'+('attn_norm'if kind=='attn'else'ffn_norm')+'.weight'].data,copy=True);norm=torch.from_numpy(normraw.reshape(4096)).cuda().bfloat16()
 for tokens in [1,9,127,128,1024]:
  o=p/'reference'/f'{layer}-{kind}-{tokens}';o.mkdir(parents=True,exist_ok=True);gen=torch.Generator().manual_seed(57000+layer*2+(kind=='ffn'));r=torch.randn((tokens,4,4096),generator=gen).bfloat16().cuda();branch=torch.randn((tokens,4096),generator=gen).bfloat16().cuda()
  def run():
   dot,sq=_hc_prenorm_gemm_outputs(r.flatten(1),fn,hidden_size=4096,hc_mult=4)
   post,comb,x=mhc_pre_tilelang(r,fn,scale,base,1e-5,1e-6,1e-6,2.,20)
   postn,combn,normed=mhc_pre_tilelang(r,fn,scale,base,1e-5,1e-6,1e-6,2.,20,norm_weight=norm,norm_eps=1e-5)
   assert torch.equal(post,postn)and torch.equal(comb,combn)
   # Capture the original coefficient kernel with the SAME projection splits.
   # The delayed wrapper selects different splits and is a different profile.
   pre=torch.empty((tokens,4),device='cuda');temp_post=torch.empty_like(post);temp_comb=torch.empty_like(comb)
   mhc_pre_big_fuse_tilelang(dot,sq,scale,base,r,temp_post.reshape(tokens,4),temp_comb.reshape(tokens,16),torch.empty_like(x),torch.zeros_like(pre),pre,torch.empty_like(x),hidden_size=4096,rms_eps=1e-5,hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=2.,sinkhorn_repeat=20,n_splits=dot.shape[0],hc_mult=4,save_pre_mix=True)
   assert torch.equal(temp_post,post)and torch.equal(temp_comb,comb)
   y=mhc_post_tilelang(branch,r,post,comb)
   return dict(dot=dot,sq=sq,pre=pre,post=post,comb=comb,x=x,normed=normed,out=y)
  data=run();again=run();assert all(torch.equal(data[k],again[k])for k in data)
  data.update(residual=r,branch=branch,fn=fn,base=base,scale=scale,norm=norm)
  hashes={}
  for k,t in data.items():
   buf=t.contiguous().view(torch.uint8).cpu().numpy().tobytes();(o/(k+'.bin')).write_bytes(buf);hashes[k]=hashlib.sha256(buf).hexdigest()
  records.append(dict(layer=layer,kind=kind,tokens=tokens,n_splits=(o/'dot.bin').stat().st_size//tokens//24//4,hashes=hashes,weight_hashes={k:hashlib.sha256(a.tobytes()).hexdigest()for k,a in raw.items()},norm_sha256=hashlib.sha256(normraw.tobytes()).hexdigest()));print(layer,kind,tokens,'captured and repeated',flush=True)
(p/'reference.json').write_text(json.dumps(dict(records=records,deep_gemm_supported=is_deep_gemm_supported(),sources={str(Path(inspect.getfile(fn))):hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()for fn in [mhc_pre_tilelang,mhc_post_tilelang]}),indent=2)+'\n')
