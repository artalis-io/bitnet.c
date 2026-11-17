"""Original token embedding -> first three dense KDA decoder blocks.

Each schedule starts from cold state, or computes its own three-token warmup.
The output stays deferred, matching a non-final decoder block's contract.
"""
from pathlib import Path
import argparse, hashlib, inspect, json
import numpy as np
import torch
import gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.mhc import hc_expand
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang, mhc_fused_post_pre_tilelang
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
readers=[gguf.GGUFReader(str(args.model).replace('00001-of',f'{i:05d}-of')) for i in range(1,7)]
table={t.name:t for r in readers for t in r.tensors};torch.set_num_threads(1);hashes={};layers=[]
def tensor(name):
    t=table[name];raw=np.array(t.data,copy=True);hashes[name]=hashlib.sha256(raw.tobytes()).hexdigest();return torch.from_numpy(raw).cuda()
assert int(table['token_embd.weight'].tensor_type)==8
embedding=tensor('token_embd.weight').reshape(154880,4352)
for layer in range(3):
    prefix=f'blk.{layer}.'
    names=dict(q='attn_q.weight',k='attn_k.weight',v='attn_v.weight',beta='ssm_beta.weight',fa='ssm_f_a.weight',ga='ssm_g_a.weight',fb='ssm_f_b.weight',gb='ssm_g_b.weight',out='attn_output.weight',cq='ssm_conv1d_q.weight',ck='ssm_conv1d_k.weight',cv='ssm_conv1d_v.weight',a='ssm_a',bias='ssm_dt.bias',norm='ssm_norm.weight')
    w={k:tensor(prefix+name) for k,name in names.items()}
    assert all(int(table[prefix+names[k]].tensor_type)==8 for k in ['q','k','v','beta','fa','ga','fb','gb','out'])
    w['merged']=torch.cat([w[k].reshape(-1,4352) for k in ['q','k','v','beta','fa','ga']],0).contiguous()
    for k,cols in [('fb',128),('gb',128),('out',8192)]:w[k]=w[k].reshape(-1,cols//32*34)
    w['cw']=torch.cat([w[k].reshape(8192,4) for k in ['cq','ck','cv']],0).float()
    w['a']=w['a'].reshape(64).float();w['bias']=w['bias'].reshape(64,128).float();w['norm']=w['norm'].reshape(128).float()
    for kind in ['attn','ffn']:
        name=prefix+f'hc_{kind}_fn.weight';raw=np.array(table[name].data,copy=True);hashes[name]=hashlib.sha256(raw.tobytes()).hexdigest();assert int(table[name].tensor_type)==8
        w[kind]=[torch.from_numpy(gguf.quants.dequantize(raw,8).reshape(24,16384)).cuda(),tensor(prefix+f'hc_{kind}_scale.weight').reshape(3),tensor(prefix+f'hc_{kind}_base.weight').reshape(24),tensor(prefix+('attn_norm' if kind=='attn' else 'ffn_norm')+'.weight').reshape(4096).bfloat16()]
    raw=[]
    for name in ['gate','up','down']:
        full=prefix+f'ffn_{name}.weight';assert int(table[full].tensor_type)==8
        value=tensor(full).reshape(-1,(12288 if name=='down' else 4096)//32*34)
        if name=='down':w['down']=value
        else:raw.append(value)
    w['gu']=torch.cat(raw,0).contiguous();layers.append(w)
ids=torch.tensor([0,154879,1,713,32767,100000,42,8192,12345,17,154878,256],device='cuda',dtype=torch.int32)
def save(path,value):
    data=value.contiguous().view(torch.uint8).cpu().numpy().tobytes();path.write_bytes(data);return hashlib.sha256(data).hexdigest()
save(args.out/'ids.bin',ids);records=[]
for resumed in [0,1]:
    for ci,steps in enumerate([[9],[2,7],[3,3,3],[7,1,1],[6,3],[1]*9]):
        schedule=([3] if resumed else [])+steps
        def run():
            histories=[torch.zeros((2,24576,3),device='cuda',dtype=torch.bfloat16) for _ in layers]
            states=[torch.zeros((1,64,128,128),device='cuda',dtype=torch.float32) for _ in layers]
            outputs=[];offset=0
            for nt in schedule:
                emb=_apply_gguf_embedding(ids[offset:offset+nt].long(),embedding,8,4096,torch.bfloat16)
                residual=hc_expand(emb,4);values={'embedding':emb.clone(),'expanded':residual.clone()};previous=None
                for li,w in enumerate(layers):
                    if previous is None:
                        post,comb,x=mhc_pre_tilelang(residual,*w['attn'][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=w['attn'][3],norm_eps=1e-5)
                        r=residual
                    else:
                        branch,r0,p0,c0=previous
                        r,post,comb,x=mhc_fused_post_pre_tilelang(branch,r0,p0,c0,*w['attn'][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=w['attn'][3],norm_eps=1e-5)
                    projected=_fused_mul_mat_gguf(x.contiguous(),w['merged'],8)
                    qkv,b,fa,ga=projected.split([24576,64,128,128],-1)
                    g1=_fused_mul_mat_gguf(fa.contiguous(),w['fb'],8);g2=_fused_mul_mat_gguf(ga.contiguous(),w['gb'],8)
                    idx=torch.tensor([1],device='cuda',dtype=torch.int32);hist=histories[li]
                    if nt==1:conv=causal_conv1d_update(qkv.contiguous().clone(),hist,w['cw'],activation='silu',conv_state_indices=idx)
                    else:conv=causal_conv1d_fn(qkv.T,w['cw'],None,hist,torch.tensor([0,nt],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([True],device='cuda')).T.contiguous()
                    q,k,v=[z.reshape(1,nt,64,128) for z in conv.split(8192,-1)]
                    gate=(-w['a'][None,None,:,None])*(g1.reshape(1,nt,64,128).float()+w['bias'])
                    core,state=fused_recurrent_kda(q,k,v,gate,b.contiguous().reshape(1,nt,64),scale=float(np.float32(128**-.5)),initial_state=states[li],inplace_final_state=False,use_qk_l2norm_in_kernel=True,sigmoid_beta=True,a_log=torch.zeros_like(w['a']),g_bias=torch.zeros_like(w['bias']),compute_gate=True,lower_bound=-5.)
                    states[li]=state[-1:].clone()
                    gated=rms_norm_gated(core.clone(),g2.reshape(nt,64,128),w['norm'],None,activation='sigmoid',eps=1e-5)
                    branch=_fused_mul_mat_gguf(gated.reshape(nt,8192).contiguous(),w['out'],8)
                    fr,fp,fc,fx=mhc_fused_post_pre_tilelang(branch,r,post,comb,*w['ffn'][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=w['ffn'][3],norm_eps=1e-5)
                    gu=_fused_mul_mat_gguf(fx.contiguous(),w['gu'],8);act=torch.empty((nt,12288),device='cuda',dtype=torch.bfloat16)
                    torch.ops._C.silu_and_mul_with_clamp(act,gu,10.,1.,0.)
                    result=_fused_mul_mat_gguf(act,w['down'],8)
                    data=dict(input=x,entry_residual=r,entry_post=post,entry_comb=comb,projected=projected,g1=g1,g2=g2,conv=conv,core=core,gated=gated,attn_out=branch,state=states[li],history=hist[1].T.contiguous(),ffn_residual=fr,ffn_post=fp,ffn_comb=fc,ffn_input=fx,gate_up=gu,activated=act,out=result)
                    values.update({f'layer{li}-{k}':v.clone() for k,v in data.items()});previous=(result,fr,fp,fc)
                outputs.append(values);offset+=nt
            torch.cuda.synchronize();return outputs
        first,repeat=run(),run();assert all(torch.equal(x[k],y[k]) for x,y in zip(first,repeat) for k in x)
        out=args.out/f'{resumed}-{ci}';out.mkdir(exist_ok=True)
        result=[{k:save(out/f'{phase}-{k}.bin',v) for k,v in row.items()} for phase,row in enumerate(first)]
        records.append(dict(resumed=resumed,case=ci,steps=schedule,hashes=result));print(resumed,ci,schedule,'captured twice',flush=True)
sources=[_fused_mul_mat_gguf,_apply_gguf_embedding,SiluAndMulWithClamp,hc_expand,mhc_pre_tilelang,mhc_fused_post_pre_tilelang,causal_conv1d_fn,causal_conv1d_update,fused_recurrent_kda,rms_norm_gated]
(args.out/'reference.json').write_text(json.dumps(dict(records=records,weights=hashes,sources={str(Path(inspect.getfile(f))):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest() for f in sources}),indent=2)+'\n')
