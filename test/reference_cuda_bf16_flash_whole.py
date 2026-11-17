"""Independent Flash model oracle with explicit numerical/selection contracts.

Every neural input comes from actual token lookup and preceding layers. Source
weights use the already qualified decoded-BF16 core / raw GGUF expert profile.
This is a correctness oracle, not a vLLM throughput benchmark or GGUF adapter.
The default selector remains the original arrival-ordered operation. Optional
canonical selection validates its raw score-optimal choices, retains them, then
independently ranks original F32 scores with lower-ID ties and chronological
emission before calling original attention. This does not claim literal parity
with arrival-ordered outputs. Inversion warps can also be fixed explicitly:
original autotune configurations use different F32 reduction trees. Captures
record these choices rather than silently replacing an existing reference.
"""
import argparse, hashlib, importlib, json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import gguf
import numpy as np
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids, ple_conv
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf
# Freeze a supported original output configuration so autotune timing cannot
# silently change FMA contraction between reference captures. Other GDN math
# remains in the original wrapper. BK64/BV32, two warps uses history FMA.
gdn_output=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk_o')
gdn_tuner=gdn_output.chunk_fwd_kernel_o
while not hasattr(gdn_tuner,'configs'):gdn_tuner=gdn_tuner.fn
gdn_tuner.configs=[c for c in gdn_tuner.configs if c.kwargs['BK']==64 and c.kwargs['BV']==32 and c.num_warps==2 and c.num_stages==3]
assert len(gdn_tuner.configs)==1
gdn_tuner.cache.clear()
pre=importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer')
sel=importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa_indexer')
att=importlib.import_module('vllm.models.qwen4_exp.nvidia.ops.qsa')
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('inventory',type=Path)
p.add_argument('--initial-token',type=int,default=116);p.add_argument('--steps',type=int,default=2)
p.add_argument('--layer-limit',type=int,default=48);p.add_argument('--trace-gdn',type=int,nargs='+',default=[0])
p.add_argument('--prompt-tokens',type=int,nargs='+');p.add_argument('--chunk',type=int,default=0);p.add_argument('--decode',type=int,default=1)
p.add_argument('--gdn-inverse-warps',type=int,choices=[2,4,8])
p.add_argument('--qsa-selection',choices=['original','canonical'],default='original')
p.add_argument('--trace-routing',action='store_true',help='retain top-k IDs for every FFN layer')
a=p.parse_args();assert 1<=a.steps<=8 and 0<=a.initial_token<248320
if a.gdn_inverse_warps:
    inverse=importlib.import_module('vllm.third_party.flash_linear_attention.ops.solve_tril')
    inverse_tuner=inverse.merge_16x16_to_64x64_inverse_kernel
    while not hasattr(inverse_tuner,'configs'):inverse_tuner=inverse_tuner.fn
    inverse_tuner.configs=[c for c in inverse_tuner.configs if c.num_warps==a.gdn_inverse_warps and c.num_stages==3]
    assert len(inverse_tuner.configs)==1
    inverse_tuner.cache.clear()
prompt=a.prompt_tokens or [a.initial_token];chunk=a.chunk or len(prompt);decode=a.decode if a.prompt_tokens else a.steps-1
assert chunk>0 and 0<=decode<=8 and all(0<=t<248320 for t in prompt)
capacity=len(prompt)+decode;assert 0<capacity<=8192 and 0<a.layer_limit<=48
assert a.layer_limit==48 or decode==0
schedule=[(pos,prompt[pos:pos+chunk],True)for pos in range(0,len(prompt),chunk)]+[(len(prompt)+i,None,False)for i in range(decode)]
a.directory.mkdir(exist_ok=False)
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
inventory=json.loads(a.inventory.read_text());meta=inventory['metadata']
readers=[gguf.GGUFReader(s['path'])for s in inventory['shards']]
tensors={t.name:t for r in readers for t in r.tensors};files={};weights={};states={};qsa_states={}
selection_checks=[];selection_effects=[]
def save(name,x,allow_nonfinite=False):
    assert allow_nonfinite or torch.isfinite(x).all(),name
    b=x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();f=a.directory/name
    assert not f.exists(),name;f.write_bytes(b)
    files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),dtype=str(x.dtype),shape=list(x.shape))
def source(name):
    t=tensors[name]
    if name not in weights:weights[name]=dict(type=t.tensor_type.name,shape=[int(v)for v in t.shape],raw_sha256=hashlib.sha256(memoryview(t.data).cast('B')).hexdigest())
    return t
def raw(name):return torch.from_numpy(np.array(source(name).data,copy=True)).cuda()
def decoded(name,offset=0):
    t=source(name);v=torch.from_numpy(np.array(gguf.dequantize(t.data,t.tensor_type),copy=True)).float()
    return (v-offset).bfloat16().cuda()
def hc_module(layer,role):
    root=f'blk.{layer}.hc_{role}_' if layer>=0 else 'output_hc_'
    hc=GatedResidual(HyperConnectionConfig(hc_count=4,hidden_size=2560,hc_lowrank=320,hc_per_branch_norm=True,params_dtype=torch.bfloat16),use_combine=layer>=0).cuda()
    with torch.no_grad():
        hc.hc_norm.weight.copy_(decoded(root+'norm.weight',1))
        hc.input_mix_weight_down.weight.copy_(decoded(root+'down.weight'))
        hc.input_mix_weight_up.weight.copy_(decoded(root+'up.weight'))
        if layer>=0:hc.block_inject_weight.weight.copy_(decoded(root+'inject.weight'))
    return hc
idx=torch.tensor([1],device='cuda',dtype=torch.int32)
with set_current_vllm_config(VllmConfig()),torch.device('cuda'):
    rope=get_rope(head_size=256,max_position=8192,rope_parameters=dict(rope_type='default',rope_theta=10000000.,partial_rotary_factor=.25),dtype=torch.bfloat16)
cs=rope.cos_sin_cache

def gdn(x,layer,phase,prefill,prefix):
    base=f'blk.{layer}.'
    wqz=torch.cat([decoded(base+'attn_qkv.weight'),decoded(base+'attn_gate.weight')])
    wba=torch.cat([decoded(base+'ssm_beta.weight'),decoded(base+'ssm_alpha.weight')])
    wc=decoded(base+'ssm_conv1d.weight').reshape(10240,4)
    wn=decoded(base+'ssm_norm.weight').reshape(128);wo=decoded(base+'ssm_out.weight')
    t=source(base+'ssm_a');coef=torch.from_numpy(np.array(t.data,copy=True)).float().reshape(48);assert(coef<0).all()
    al=torch.log(-coef).cuda();dt=decoded(base+'ssm_dt.bias').reshape(48)
    if layer not in states:states[layer]=(torch.zeros(2,48,128,128,device='cuda',dtype=torch.float32),torch.zeros(2,10240,3,device='cuda',dtype=torch.bfloat16))
    cache,history=states[layer];nt=x.shape[0]
    if layer in a.trace_gdn:
        save(prefix+'-gdn-state-in.bin',cache[1]);save(prefix+'-gdn-history-in.bin',history[1].T)
        save(prefix+'-gdn-a-log.bin',al);save(prefix+'-gdn-dt-bias.bin',dt)
    qkvz=F.linear(x,wqz);ba=F.linear(x,wba);qkv=qkvz[:,:10240].contiguous();z=qkvz[:,10240:].contiguous()
    if prefill:
        post=causal_conv1d_fn(qkv.T,wc,None,history,torch.tensor([0,nt],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([phase>0],device='cuda'),activation='silu',out=torch.empty_like(qkv.T)).T
        q,k,v,g,b=fused_post_conv_prep(post,ba[:,48:],ba[:,:48],al,dt,16,128,128)
        if layer in a.trace_gdn:
            for name,value in [('q',q),('k',k),('v',v),('log_decay',g),('beta',b)]:save(prefix+'-gdn-'+name+'.bin',value)
        core,final=chunk_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),b.unsqueeze(0),initial_state=cache[1:2].clone(),output_final_state=True,cu_seqlens=torch.tensor([0,nt],device='cuda',dtype=torch.int32),use_qk_l2norm_in_kernel=False);cache[1].copy_(final[0])
    else:
        post=causal_conv1d_update(qkv.clone(),history,wc,activation='silu',conv_state_indices=idx)
        core=torch.empty(1,nt,48,128,device='cuda',dtype=torch.bfloat16)
        fused_recurrent_gated_delta_rule_packed_decode(post,ba[:,48:],ba[:,:48],al,dt,128**-.5,cache,core,idx,True)
    norm=rmsnorm_fn(core.reshape(-1,128),wn,None,z=z.reshape(-1,128),eps=1e-6,norm_before_gate=True,activation='sigmoid').reshape(nt,6144)
    out=F.linear(norm,wo)
    if layer in a.trace_gdn:
        save(prefix+'-gdn-history.bin',history[1].T)
        for n,v in [('input',x),('qkvz',qkvz),('ba',ba),('post',post),('core',core),('norm',norm),('out',out),('state',cache[1])]:save(prefix+'-gdn-'+n+'.bin',v)
    return out

def canonical_topk_observer(prefix,originals):
    """Use original scores and independently rank them, retaining raw choices.

    Native's declared policy is lower-ID score ties and chronological emission.
    Validate the original operation's score-optimal membership before applying
    that deterministic policy. This is not literal arrival-order parity.
    """
    original=sel._topk
    def observed(logits,visible_blocks,token_topk,compress_ratio,block_indices,workspace):
        assert token_topk==2048 and compress_ratio==4
        scores=logits.detach().cpu().contiguous()
        visible=visible_blocks.detach().cpu().numpy().reshape(-1)
        original(logits,visible_blocks,token_topk,compress_ratio,block_indices,workspace)
        raw_groups=block_indices.detach().clone();raw=raw_groups.cpu().numpy()
        chosen=np.full(raw.shape,-1,dtype=np.int32)
        report=dict(prefix=prefix,call=len(originals),rows=len(visible),pruned_rows=0,
                    ambiguous_rows=0,changed_memberships=0,changed_order=0)
        for row,n in enumerate(visible):
            n=int(n);values=scores[row,:n].numpy()
            assert 0<=n<=scores.shape[1] and np.isfinite(values).all() and (values>=0).all()
            if n<=512:
                chosen[row,:n]=np.arange(n,dtype=np.int32)
                continue
            report['pruned_rows']+=1
            raw_ids=raw[row]
            assert ((raw_ids>=0)&(raw_ids<n)).all() and len(np.unique(raw_ids))==512
            cutoff=np.partition(values,n-512)[n-512]
            above=np.flatnonzero(values>cutoff)
            assert (values[raw_ids]>=cutoff).all() and np.isin(above,raw_ids).all()
            ambiguous=np.count_nonzero(values==cutoff)>512-len(above)
            report['ambiguous_rows']+=int(ambiguous)
            ranked=np.lexsort((np.arange(n,dtype=np.int32),-values))[:512]
            chosen[row]=np.sort(ranked)
            different=not np.array_equal(np.sort(raw_ids),chosen[row])
            assert not different or ambiguous
            report['changed_memberships']+=int(different)
            report['changed_order']+=int(not np.array_equal(raw_ids,chosen[row]))
        call=len(originals)
        save(f'{prefix}-qsa-original-scores-{call}.bin',scores,allow_nonfinite=True)
        save(f'{prefix}-qsa-original-groups-{call}.bin',raw_groups)
        replacement=torch.from_numpy(chosen).to(device=block_indices.device)
        save(f'{prefix}-qsa-canonical-groups-{call}.bin',replacement)
        block_indices.copy_(replacement);originals.append(raw_groups)
        selection_checks.append(report)
    return observed

def qsa(x,layer,pos,prefill,prefix):
    nt=x.shape[0];base=f'blk.{layer}.'
    wmain=torch.cat([decoded(base+n)for n in ['attn_q.weight','attn_k.weight','attn_v.weight']]);wo=decoded(base+'attn_output.weight')
    nq=decoded(base+'attn_q_norm.weight',1).reshape(256);nk=decoded(base+'attn_k_norm.weight',1).reshape(256)
    wi=torch.cat([decoded(base+'indexer.q_proj.weight'),decoded(base+'indexer.k_proj.weight')])
    niq=decoded(base+'indexer.q_norm.weight',1).reshape(128);nik=decoded(base+'indexer.k_norm.weight',1).reshape(128)
    if layer not in qsa_states:
        qsa_states[layer]=(torch.zeros(2,4,128,device='cuda',dtype=torch.bfloat16),torch.zeros(1+(capacity+63)//64,16,128,device='cuda',dtype=torch.bfloat16),torch.zeros(1+(capacity+15)//16,16,2,256,device='cuda',dtype=torch.bfloat16),torch.zeros(1+(capacity+15)//16,16,2,256,device='cuda',dtype=torch.bfloat16))
    ring,cp,kc,vc=qsa_states[layer];positions=torch.arange(pos,pos+nt,device='cuda',dtype=torch.int64)
    projected=F.linear(x,wmain)
    q,k,gate=fused_qk_rmsnorm_rope_gate(projected[:,:12288],projected[:,12288:12800],nq,nk,cs,positions,1e-6,24,2,256,64,norm_beta=1)
    q=q.reshape(nt,24,256);k=k.reshape(nt,2,256);gate=gate.reshape(nt,24,256);v=projected[:,12800:].contiguous().reshape(nt,2,256)
    ip=F.linear(x,wi);iq=torch.empty(nt,4,128,device='cuda',dtype=torch.bfloat16)
    slots=4+positions%4;cp_slots=torch.where((positions+1)%4==0,16+positions//4,-1)
    groups=(pos+nt)//4-pos//4;work=torch.tensor([[0,i]for i in range(max(groups,1))],device='cuda',dtype=torch.int32)
    pre.qsa_pre_indexer(ip[:,:512],ip[:,512:],positions,cs,niq,nik,1e-6,iq,ring,slots,torch.tensor([[1]],device='cuda',dtype=torch.int32),torch.tensor([0,nt],device='cuda',dtype=torch.int32),positions,cp,cp_slots,work,compress_ratio=4,mrope_section=(11,11,10),rope_pos_offset=None)
    cp_pages=max(1,((pos+nt)//4+15)//16);cp_table=torch.arange(1,cp_pages+1,device='cuda',dtype=torch.int32).reshape(1,-1)
    table=torch.arange(1,(capacity+15)//16+1,device='cuda',dtype=torch.int32).reshape(1,-1);visible=((positions+1)//4).int();selected=torch.empty(nt,512,device='cuda',dtype=torch.int32)
    original_selections=[]
    with patch.object(sel,'_topk',side_effect=canonical_topk_observer(prefix,original_selections)) if a.qsa_selection=='canonical' else nullcontext():
        if nt==1:sel.qsa_select_paged_decode(iq,cp.unsqueeze(2),cp_table,visible,2048,4,1,selected)
        else:sel.qsa_select_paged_prefill(iq,cp.unsqueeze(2),cp_table,torch.tensor([0,nt],device='cuda',dtype=torch.int32),visible,2048,4,nt,selected,pos+nt)
    indices=torch.empty(nt,2052,device='cuda',dtype=torch.int32);sel.expand_qsa_block_indices(selected,positions,visible,4,2048,indices)
    att.qsa_store_cache_rows(kc.reshape(-1,16,1,512),16+positions,k.reshape(nt,1,512));att.qsa_store_cache_rows(vc.reshape(-1,16,1,512),16+positions,v.reshape(nt,1,512))
    core=att.qsa_sparse_paged_attention(q,kc,vc,indices,table,torch.zeros(nt,device='cuda',dtype=torch.int32),prefill,output_gate=gate)
    out=F.linear(core.reshape(nt,6144),wo)
    if layer==3:
        for n,v in [('input',x),('projected',projected),('q',q),('k',k),('v',v),('index_q',iq),('indices',indices),('core',core),('out',out)]:save(prefix+'-qsa-'+n+'.bin',v)
        if original_selections:
            raw_indices=torch.empty_like(indices)
            sel.expand_qsa_block_indices(torch.cat(original_selections),positions,visible,4,2048,raw_indices)
            raw_core=att.qsa_sparse_paged_attention(q,kc,vc,raw_indices,table,torch.zeros(nt,device='cuda',dtype=torch.int32),prefill,output_gate=gate)
            raw_out=F.linear(raw_core.reshape(nt,6144),wo)
            for name,value in [('indices',raw_indices),('core',raw_core),('out',raw_out)]:save(prefix+'-qsa-original-'+name+'.bin',value)
            selection_effects.append(dict(prefix=prefix,indices_different=int(torch.count_nonzero(raw_indices!=indices)),
                core_words_different=int(torch.count_nonzero(raw_core.view(torch.int16)!=core.view(torch.int16))),
                output_words_different=int(torch.count_nonzero(raw_out.view(torch.int16)!=out.view(torch.int16)))))
    return out

ple_history=torch.zeros(2,10240,9,device='cuda',dtype=torch.bfloat16);context=[248044,248044]
def ple(hidden,tokens,pos,prefix):
    global context
    base='blk.1.ple_'
    wk=decoded(base+'key.weight');wv=decoded(base+'value.weight');nk=decoded(base+'norm_key.weight',1).reshape(4,2560);nq=decoded(base+'norm_query.weight',1).reshape(4,2560);nc=decoded(base+'norm_conv.weight',1).reshape(4,2560);wc=decoded(base+'conv1d.weight').reshape(10240,4)
    dev=lambda x:torch.tensor(x,device='cuda',dtype=torch.int64)
    nt=len(tokens);ids=ple_ngram_ids(dev(tokens).int(),dev([0,nt]).int(),dev(context).int().reshape(1,2),dev(meta['qwen4exp.ple.layer_multipliers']),dev(meta['qwen4exp.ple.head_vocab_sizes']),dev(meta['qwen4exp.ple.head_offsets']),248044,8)
    table=source('per_layer_token_embd.weight');rows=np.asarray(table.data).reshape(-1,90)[ids.cpu().numpy().reshape(-1)]
    e=torch.from_numpy(gguf.dequantize(rows,table.tensor_type)).bfloat16().cuda().reshape(nt,2560)
    key,value=F.linear(e,torch.cat([wk,wv])).split([10240,2560],-1)
    def norm(x,w):
        f=x.float();return(f*torch.rsqrt((f*f).mean(-1,keepdim=True)+1e-6)*(1+w.float())).bfloat16()
    key=key.reshape(nt,4,2560);query=hidden.reshape_as(key)
    products=(norm(key,nk)*norm(query,nq)).bfloat16()
    d=(products.float().sum(-1).bfloat16()/(2560**.5)).bfloat16()
    g=(d.sign()*d.abs().clamp(min=1e-6).sqrt().bfloat16()).sigmoid().bfloat16()
    gated=(g[:,:,None]*value[:,None,:]).bfloat16().flatten(1);normalized=norm(gated.reshape(nt,4,2560),nc).flatten(1)
    out=gated.clone();ple_conv(normalized,out,ple_history,wc,idx,hidden,mode='decode'if nt==1 else'prefill',dilation=3,query_start_loc=dev([0,nt]).int(),has_initial_states=dev([pos>0]).bool())
    context=(context+tokens)[-2:]
    for n,v in [('ids',ids),('embedding',e),('out',out),('history',ple_history[1].T)]:save(prefix+'-ple-'+n+'.bin',v)
    return out

def ffn(x,layer,prefix):
    base=f'blk.{layer}.'
    with torch.device('cuda'):
        router=ReplicatedLinear(2560,512,bias=False,disable_tp=True);shared_router=ReplicatedLinear(2560,1,bias=False,disable_tp=True)
    for module,name in [(router,'ffn_gate_inp.weight'),(shared_router,'ffn_gate_inp_shexp.weight')]:
        t=source(base+name);module.weight_loader(module.weight,torch.from_numpy(np.array(t.data,copy=True)).reshape(module.weight.shape))
    logits=router(x)[0];shared_logits=shared_router(x)[0];weights,ids,source_rows=fused_topk(x,logits,10,True)
    w13=torch.cat([raw(base+'ffn_gate_exps.weight'),raw(base+'ffn_up_exps.weight')],dim=1);w2=raw(base+'ffn_down_exps.weight')
    typ1=int(source(base+'ffn_gate_exps.weight').tensor_type);typ2=int(source(base+'ffn_down_exps.weight').tensor_type)
    routed=_fused_moe_gguf(x,w13,w2,weights,ids,typ1,typ2,'silu')
    shared13=torch.cat([raw(base+'ffn_gate_shexp.weight'),raw(base+'ffn_up_shexp.weight')],dim=0);shared2=raw(base+'ffn_down_shexp.weight')
    activation=SiluAndMul();shared=(_fused_mul_mat_gguf(activation(_fused_mul_mat_gguf(x,shared13,8)),shared2,8)*torch.sigmoid(shared_logits)).bfloat16()
    out=(routed+shared).bfloat16()
    if layer==0:
        for n,v in [('input',x),('logits',logits),('ids',ids),('weights',weights),('routed',routed),('shared',shared),('out',out)]:save(prefix+'-ffn-'+n+'.bin',v)
    elif a.trace_routing:
        save(prefix+'-ffn-ids.bin',ids)
    return out

phases=[];token=a.initial_token
with set_current_vllm_config(VllmConfig()),patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank',return_value=0),patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size',return_value=1),torch.inference_mode():
    for phase,(position,tokens,prefill)in enumerate(schedule):
        if tokens is None:tokens=[token]
        prefix=f'phase{phase}';save(prefix+'-tokens.bin',torch.tensor(tokens,dtype=torch.int32))
        embedding=_apply_gguf_embedding(torch.tensor(tokens,device='cuda',dtype=torch.int64),raw('token_embd.weight'),8,2560,torch.bfloat16)
        hidden=embedding.repeat(1,4);save(prefix+'-embedding.bin',embedding)
        for layer in range(a.layer_limit):
            tag=f'{prefix}-layer{layer}'
            if layer==1:hidden=ple(hidden,tokens,position,prefix)
            hc=hc_module(layer,'attn');x,residual=hc.mix(hidden)
            out=qsa(x,layer,position,prefill,tag)if(layer+1)%4==0 else gdn(x,layer,phase,prefill,tag)
            hidden=hc.combine(out,residual);save(tag+'-attn-hidden.bin',hidden)
            del hc,x,residual,out
            hc=hc_module(layer,'ffn');x,residual=hc.mix(hidden);out=ffn(x,layer,tag)
            hidden=hc.combine(out,residual);save(tag+'-ffn-hidden.bin',hidden)
            del hc,x,residual,out
            torch.cuda.empty_cache();print(tag,'captured',flush=True)
        if a.layer_limit<48:
            phases.append(dict(phase=phase,tokens=tokens,position=position,prefill=prefill,argmax=None))
            del embedding,hidden;torch.cuda.empty_cache();continue
        hc=hc_module(-1,'');mixed,_=hc.mix(hidden);logits=_fused_mul_mat_gguf(mixed[-1:].contiguous(),raw('output.weight'),8)
        save(prefix+'-mixed.bin',mixed);save(prefix+'-logits.bin',logits)
        token=int(logits.float().argmax());phases.append(dict(phase=phase,tokens=tokens,position=position,prefill=prefill,argmax=token))
        print('WHOLE_PHASE',phase,'argmax',token,flush=True);del hc,mixed,logits,embedding,hidden;torch.cuda.empty_cache()
(a.directory/'layers.txt').write_text(str(a.layer_limit)+'\n')
(a.directory/'phases.txt').write_text(''.join(f"{p['phase']} {p['position']} {len(p['tokens'])} {int(p['prefill'])}\n"for p in phases))
(a.directory/'manifest.json').write_text(json.dumps(dict(layer_limit=a.layer_limit,gdn_inverse_warps=a.gdn_inverse_warps,qsa_selection=a.qsa_selection,trace_routing=a.trace_routing,selection_checks=selection_checks,selection_effects=selection_effects,gdn_output_profile='BK64/BV32/warps2/stages3: history FMA',profile='actual tokens through pinned original operators, decoded BF16 core and raw GGUF experts, per-layer expert placement',files=files,weights=weights,phases=phases),indent=2)+'\n')
print('FLASH_WHOLE_REFERENCE',len(phases),'phases',flush=True)
