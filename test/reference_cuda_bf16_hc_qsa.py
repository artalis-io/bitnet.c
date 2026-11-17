"""Pinned common HC -> actual QSA attention -> common HC combine capture.
Derived from the immutable QSA oracle; operators are unmodified, hooks observe.
"""
"""Actual Flash layer3 full QSA block, explicit decoded-BF16 text profile.
Preserves original pinned selection and an independently ranked deterministic
selection separately. No expected neural intermediates are injected. Outputs
for stochastic original selections are retained in a fresh capture directory.
"""
import argparse, hashlib, importlib, json
from pathlib import Path
import gguf
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import patch
from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
pre = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer')
sel = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_indexer')
att = importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa')
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
inventory=json.loads(a.inventory.read_text());readers=[gguf.GGUFReader(x['path'])for x in inventory['shards']];tensors={t.name:t for r in readers for t in r.tensors};weights={}
def save(name,x):
 data=x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();f=a.directory/name
 if f.exists():assert f.read_bytes()==data,name
 else:f.write_bytes(data)
 return dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),dtype=str(x.dtype),shape=list(x.shape))
def weight(name):
 t=tensors['blk.3.'+name];raw=np.asarray(t.data);offset=1.0 if 'norm.weight' in name else 0.0
 v=(torch.from_numpy(np.array(gguf.dequantize(raw,t.tensor_type),copy=True)).float()-offset).bfloat16()
 weights[name]=dict(gemma_storage_offset=offset,type=t.tensor_type.name,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),decoded=save(name+'.bf16',v));return v.cuda()
wmain=torch.cat([weight(n)for n in ['attn_q.weight','attn_k.weight','attn_v.weight']]);wo=weight('attn_output.weight');nq=weight('attn_q_norm.weight').reshape(256);nk=weight('attn_k_norm.weight').reshape(256)
widx=torch.cat([weight('indexer.q_proj.weight'),weight('indexer.k_proj.weight')]);iqn=weight('indexer.q_norm.weight').reshape(128);ikn=weight('indexer.k_norm.weight').reshape(128)
hc=GatedResidual(HyperConnectionConfig(hc_count=4,hidden_size=2560,hc_lowrank=320,hc_per_branch_norm=True,params_dtype=torch.bfloat16),use_combine=True).cuda()
with torch.no_grad():
 for suffix,param in [('norm',hc.hc_norm.weight),('down',hc.input_mix_weight_down.weight),('up',hc.input_mix_weight_up.weight),('inject',hc.block_inject_weight.weight)]:param.copy_(weight('hc_attn_'+suffix+'.weight'))
original_linear=F.linear
hc_stages={}
def hc_linear(value,param,bias=None):
 result=original_linear(value,param,bias)
 name=next(n for n,p in [('down',hc.input_mix_weight_down.weight),('up',hc.input_mix_weight_up.weight),('inject',hc.block_inject_weight.weight)]if p is param)
 hc_stages['hc-'+name]=save(prefix+'-hc-'+name+'.bin',result)
 if name=='up':hc_stages['hc-scaled']=save(prefix+'-hc-scaled.bin',value)
 return result
def hc_norm(module,inputs,out):hc_stages['hc-norm']=save(prefix+'-hc-norm.bin',out)
hc.hc_norm.register_forward_hook(hc_norm)

with set_current_vllm_config(VllmConfig()),torch.device('cuda'):
 rope=get_rope(head_size=256,max_position=8192,rope_parameters=dict(rope_type='default',rope_theta=10000000.,partial_rotary_factor=.25),dtype=torch.bfloat16)
cs=rope.cos_sin_cache;save('cos-sin.bf16',cs)
observed_scores=None;original_topk=sel._topk
# Observation only; the original selection operation receives unchanged inputs.
def topk(logits,visible,budget,ratio,out,workspace):
 global observed_scores
 original_topk(logits,visible,budget,ratio,out,workspace);observed_scores=logits.cpu().numpy().copy()
sel._topk=topk
prefix='';tag='';partials={};compiled={}
for opname in ['_qsa_sparse_paged_gqa_splitk_kernel','_qsa_merge_splitk_kernel']:
 op=getattr(att,opname);original=op.run
 def hook(*args,_original=original,_name=opname,**kwargs):
  binary=_original(*args,**kwargs);ptx=binary.asm['ptx'];h=hashlib.sha256(ptx.encode()).hexdigest();file='kernel-'+h+'.ptx';f=a.directory/file
  if not f.exists():f.write_text(ptx)
  compiled[file]=dict(op=_name,num_splits=kwargs.get('NUM_SPLITS'),block_n=kwargs.get('BLOCK_N'),num_warps=kwargs.get('num_warps'))
  if _name=='_qsa_sparse_paged_gqa_splitk_kernel' and kwargs['NUM_SPLITS']>1 and tag=='canonical':
   partials['partial']=save(prefix+'-partial.bin',args[6]);partials['lse']=save(prefix+'-lse.bin',args[7])
  return binary
 op.run=hook
cases=[];audits=[]
for nt,chunk in [(1,1),(3,3),(4,4),(5,5),(17,7),(129,129),(129,64),(129,7),(129,1),(2053,2053),(2053,257),(4097,4097)]:
 name=f'{nt}-{chunk}';ring=torch.zeros(2,4,128,device='cuda',dtype=torch.bfloat16);cp=torch.zeros(1+(nt+8+63)//64,16,128,device='cuda',dtype=torch.bfloat16)
 pages=(nt+8+15)//16;kc=torch.zeros(pages+1,16,2,256,device='cuda',dtype=torch.bfloat16);vc=torch.zeros_like(kc);table=torch.arange(1,pages+1,device='cuda',dtype=torch.int32).reshape(1,-1)
 phases=[];steps=[(pos,min(chunk,nt-pos),True)for pos in range(0,nt,chunk)]+[(nt+i,1,False)for i in range(8)]
 for phase,(pos,take,prefill)in enumerate(steps):
  prefix=f'{name}-{phase}';partials={};idx=torch.arange(pos*2560,(pos+take)*2560,dtype=torch.int64);x=(((idx*17+idx//37)%509-254).float()/128).reshape(take,2560).bfloat16().cuda()
  hc_stages={};j=torch.arange(pos*10240,(pos+take)*10240,dtype=torch.int64)
  hidden=(((j*37+33)%509-254).float()/128).bfloat16().reshape(take,10240).cuda()
  hc_stages['hc-hidden']=save(prefix+'-hc-hidden.bin',hidden)
  with torch.inference_mode(),patch.object(F,'linear',side_effect=hc_linear):x,residual=hc.mix(hidden)
  positions=torch.arange(pos,pos+take,device='cuda',dtype=torch.int64);projected=F.linear(x,wmain)
  q,k,gate=fused_qk_rmsnorm_rope_gate(projected[:,:12288],projected[:,12288:12800],nq,nk,cs,positions,1e-6,24,2,256,64,norm_beta=1)
  q=q.reshape(take,24,256);k=k.reshape(take,2,256);gate=gate.reshape(take,24,256);v=projected[:,12800:].contiguous().reshape(take,2,256)
  ip=F.linear(x,widx);iq=torch.empty(take,4,128,device='cuda',dtype=torch.bfloat16);slots=4+positions%4;cp_slots=torch.where((positions+1)%4==0,16+positions//4,-1)
  groups=(pos+take)//4-pos//4;work=torch.tensor([[0,i]for i in range(max(groups,1))],device='cuda',dtype=torch.int32)
  pre.qsa_pre_indexer(ip[:,:512],ip[:,512:],positions,cs,iqn,ikn,1e-6,iq,ring,slots,torch.tensor([[1]],device='cuda',dtype=torch.int32),torch.tensor([0,take],device='cuda',dtype=torch.int32),positions,cp,cp_slots,work,compress_ratio=4,mrope_section=(11,11,10),rope_pos_offset=None)
  cp_pages=max(1,((pos+take)//4+15)//16);cp_table=torch.arange(1,cp_pages+1,device='cuda',dtype=torch.int32).reshape(1,-1);visible=((positions+1)//4).int();original_groups=torch.empty(take,512,device='cuda',dtype=torch.int32)
  if take==1:sel.qsa_select_paged_decode(iq,cp.unsqueeze(2),cp_table,visible,2048,4,1,original_groups)
  else:sel.qsa_select_paged_prefill(iq,cp.unsqueeze(2),cp_table,torch.tensor([0,take],device='cuda',dtype=torch.int32),visible,2048,4,take,original_groups,pos+take)
  canonical=np.full((take,512),-1,dtype=np.int32)
  for row,n in enumerate(visible.cpu().numpy()):
   ids=np.arange(n,dtype=np.int32);canonical[row,:min(n,512)]=np.sort(ids[np.lexsort((ids,-observed_scores[row,:n]))[:512]])
  canonical_groups=torch.from_numpy(canonical).cuda();ci=torch.empty(take,2052,device='cuda',dtype=torch.int32);oi=torch.empty_like(ci)
  sel.expand_qsa_block_indices(canonical_groups,positions,visible,4,2048,ci);sel.expand_qsa_block_indices(original_groups,positions,visible,4,2048,oi)
  att.qsa_store_cache_rows(kc.reshape(pages+1,16,1,512),16+positions,k.reshape(take,1,512));att.qsa_store_cache_rows(vc.reshape(pages+1,16,1,512),16+positions,v.reshape(take,1,512))
  to_req=torch.zeros(take,device='cuda',dtype=torch.int32)
  tag='canonical';ca=att.qsa_sparse_paged_attention(q,kc,vc,ci,table,to_req,prefill,output_gate=gate)
  tag='original';oa=att.qsa_sparse_paged_attention(q,kc,vc,oi,table,to_req,prefill,output_gate=gate)
  co=F.linear(ca.reshape(take,6144),wo);oo=F.linear(oa.reshape(take,6144),wo)
  with torch.inference_mode(),patch.object(F,'linear',side_effect=hc_linear):combined=hc.combine(co,residual)
  hc_stages['hc-combined']=save(prefix+'-hc-combined.bin',combined)
  audit=dict(name=prefix,attention_changed=int((ca!=oa).sum()),attention_max_abs=float((ca.float()-oa.float()).abs().max()),output_changed=int((co!=oo).sum()),output_max_abs=float((co.float()-oo.float()).abs().max()))
  audits.append(audit)
  stages={s:save(prefix+'-'+s+'.bin',t)for s,t in [('input',x),('projected',projected),('q',q),('k',k),('v',v),('gate',gate),('index_q',iq),('indices',ci),('original_indices',oi),('attention',ca),('original_attention',oa),('out',co),('original_out',oo),('key_cache',kc.reshape(-1,2,256)[16:16+pos+take]),('value_cache',vc.reshape(-1,2,256)[16:16+pos+take])]}
  stages.update(hc_stages);stages.update(partials);phases.append(dict(phase=phase,position=pos,take=take,prefill=prefill,config=att._select_config(take,2,prefill,2051),stages=stages));print(prefix,'config',att._select_config(take,2,prefill,2051),'attention_changed',audit['attention_changed'],'output_max',audit['output_max_abs'],flush=True)
 cases.append(dict(name=name,tokens=nt,chunk=chunk,phases=phases))
manifest=dict(profile='actual common HC layer3 attention mix -> actual QSA -> common HC combine; canonical selection; original outputs retained separately',weights=weights,cases=cases,kernels=compiled,selection_effect=audits)
f=a.directory/'manifest.json';data=(json.dumps(manifest,indent=2)+'\n').encode()
if f.exists():assert f.read_bytes()==data
else:f.write_bytes(data)
print('QSA_ATTENTION_CAPTURED',len(cases),'profiles',sum(len(c['phases'])for c in cases),'phases',flush=True)
