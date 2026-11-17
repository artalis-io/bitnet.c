"""Independent original raw-GGUF operator profile, actual IDs through45layers.

All router inputs, weights, states and activations originate here. Full288
expert domains are retained; only original-selected raw rows are populated.
Vectors retain full288 IDs; matrices partition ONLY expert domains, preserving
all token/slot geometry. Checkpoint clamp10 and fused-recurrent KDA are explicit.
Sparse scoring/top-k use original operators. Original permutations are saved
unchanged; chronological mode independently applies a documented score/lower-ID
membership policy and chronological ordering. Neither is a whole-plugin/HF or
production-performance parity claim.
"""
from pathlib import Path
import argparse, hashlib, inspect, json
import numpy as np
import torch
import gguf
import vllm
from vllm import _custom_ops as custom
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf as linear
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding
from vllm_gguf_plugin import ops as ggops
from vllm.model_executor.layers.mhc import hc_expand, hc_contract
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang, mhc_fused_post_pre_tilelang, mhc_post_tilelang
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
from vllm.models.glm5next.common.attention import _fused_indexer_k_norm, _fused_indexer_weight_scale
from vllm.models.glm5next.common.sparse_indexer import _fill_causal_indices
from vllm.models.glm5next.nvidia.ops import kpool_compress as poolops
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.v1.worker.workspace import init_workspace_manager
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as attention
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--tokens', type=int, default=9, choices=[1,9])
parser.add_argument('--schedule', help='Comma-separated prompt chunks, each1..8191, total<=8191; followed by one greedy decode.')
parser.add_argument('--input-ids', type=Path, help='JSON list of actual prompt token IDs, without implicit BOS or template changes.')
parser.add_argument('--decode-frames', type=int, default=1, help='Greedy continuation forwards after the prompt (default1).')
parser.add_argument('--states', action='store_true', help='Capture every committed recurrent/cache state.')
parser.add_argument('--sparse-order', choices=['original','chronological'], default='original', help='Original raw order or explicit deterministic selected-set profile.')
parser.add_argument('--index-diagnostics', action='store_true')
parser.add_argument('--raw-attention-audit', action='store_true', help='Also preserve original-order attention on the SAME incoming hidden state.')
a = parser.parse_args()
input_ids = json.loads(a.input_ids.read_text()) if a.input_ids else None
if input_ids is not None:
    assert isinstance(input_ids,list) and input_ids and all(type(n) is int and 0 <= n < 154880 for n in input_ids)
schedule = [int(n) for n in a.schedule.split(',')] if a.schedule else [len(input_ids) if input_ids is not None else a.tokens]
assert schedule and all(1 <= n <= 8191 for n in schedule) and sum(schedule) <= 8191
assert input_ids is None or len(input_ids) == sum(schedule)
assert 1 <= a.decode_frames <= 256
frames = len(schedule) + a.decode_frames
capacity = max(64, ((sum(schedule)+a.decode_frames+63)//64)*64)
latent_blocks = (capacity+63)//64
index_blocks = ((capacity+3)//4+63)//64
a.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
if sum(schedule)+a.decode_frames > 2048:
    init_workspace_manager(torch.device('cuda'),num_ubatches=1)
decode_topk = get_indexer_topk('auto')
selection_audit = []
readers = [gguf.GGUFReader(str(a.model).replace('00001-of', f'{i:05d}-of')) for i in range(1,7)]
table = {t.name: t for reader in readers for t in reader.tensors}
weight_hashes = {}
def tensor(name, shape=None, typ=None):
    source = table[name]
    if typ is not None:
        assert int(source.tensor_type) == typ, name
    raw = np.array(source.data, copy=True)
    weight_hashes[name] = hashlib.sha256(raw.tobytes()).hexdigest()
    return torch.from_numpy(raw.reshape(shape) if shape else raw).cuda()
def save(path, value):
    data = value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    if path.exists():
        assert path.read_bytes() == data, path
    else:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
def mhc(layer, kind):
    prefix = f'blk.{layer}.'
    name = prefix + f'hc_{kind}_fn.weight'
    source = table[name]
    raw = np.array(source.data, copy=True)
    assert int(source.tensor_type) == 8
    weight_hashes[name] = hashlib.sha256(raw.tobytes()).hexdigest()
    fn = torch.from_numpy(gguf.quants.dequantize(raw,8).reshape(24,16384)).cuda()
    return [fn, tensor(prefix+f'hc_{kind}_scale.weight',(3,),0), tensor(prefix+f'hc_{kind}_base.weight',(24,),0), tensor(prefix+('attn_norm' if kind=='attn' else 'ffn_norm')+'.weight',(4096,),0).bfloat16()]
embedding = tensor('token_embd.weight',(154880,4352),8)
head = tensor('output.weight',(154880,4352),8)
gamma = tensor('output_norm.weight',(4096,),0).bfloat16()
layers = []
for layer in range(45):
    prefix = f'blk.{layer}.'
    w = dict(attn=mhc(layer,'attn'), ffn=mhc(layer,'ffn'))
    def src(name,shape=None,typ=None):
        return tensor(prefix+name,shape,typ)
    if layer % 4 != 3:
        names = dict(q='attn_q.weight',k='attn_k.weight',v='attn_v.weight',beta='ssm_beta.weight',fa='ssm_f_a.weight',ga='ssm_g_a.weight',fb='ssm_f_b.weight',gb='ssm_g_b.weight',out='attn_output.weight',cq='ssm_conv1d_q.weight',ck='ssm_conv1d_k.weight',cv='ssm_conv1d_v.weight',a='ssm_a',bias='ssm_dt.bias',norm='ssm_norm.weight')
        for key,name in names.items():
            w[key] = src(name,typ=8 if key in ['q','k','v','beta','fa','ga','fb','gb','out'] else 0)
        w['merged'] = torch.cat([w[key].reshape(-1,4352) for key in ['q','k','v','beta','fa','ga']],0).contiguous()
        for key in ['q','k','v','beta','fa','ga']:
            del w[key]
        for key,cols in [('fb',128),('gb',128),('out',8192)]:
            w[key] = w[key].reshape(-1,cols//32*34)
        w['cw'] = torch.cat([w[key].reshape(8192,4) for key in ['cq','ck','cv']],0).float()
        for key in ['cq','ck','cv']:
            del w[key]
        w['a'] = w['a'].reshape(64).float()
        w['bias'] = w['bias'].reshape(64,128).float()
        w['norm'] = w['norm'].reshape(128).float()
    else:
        for key,name,shape in [('qa','attn_q_a.weight',(1536,4352)),('qb','attn_q_b.weight',(16384,1632)),('kva','attn_kv_a_mqa.weight',(512,4352)),('out','attn_output.weight',(4096,17408)),('iq','indexer.attn_q_b.weight',(4096,1632)),('ik','indexer.attn_k.weight',(128,4352)),('ig','indexer_compressor_gate.weight',(128,4352))]:
            w[key] = src(name,shape,8)
        w['qn'] = src('attn_q_a_norm.weight',(1536,),0).bfloat16()
        w['kn'] = src('attn_kv_a_norm.weight',(512,),0).bfloat16()
        kb = src('attn_k_b.weight',(32768,272),8)
        vb = src('attn_v_b.weight',(16384,544),8)
        # Original quantized-weight getter's identity projection, retaining
        # Q8_1 identity quantization; direct canonical dequant is different.
        w['key'] = linear(torch.eye(256,device='cuda',dtype=torch.bfloat16),kb,8).T.contiguous().reshape(64,512,256)
        w['value'] = linear(torch.eye(512,device='cuda',dtype=torch.bfloat16),vb,8).T.contiguous().reshape(64,256,512)
        for key,name,shape in [('ih','indexer.proj.weight',(32,4096)),('inn','indexer.k_norm.weight',(128,)),('ibias','indexer.k_norm.bias',(128,)),('ape','indexer_compressor_ape.weight',(4,128))]:
            w[key] = src(name,shape,0)
    if layer < 3:
        w['gu'] = torch.cat([src(f'ffn_{role}.weight',(12288,4352),8) for role in ['gate','up']],0).contiguous()
        w['down'] = src('ffn_down.weight',(4096,13056),8)
    else:
        w['router'] = src('ffn_gate_inp.weight',(288,4096),0)
        w['route_bias'] = src('exp_probs_b.bias',(288,),0)
        w['sgu'] = torch.cat([src(f'ffn_{role}_shexp.weight',(2048,4352),8) for role in ['gate','up']],0).contiguous()
        w['sd'] = src('ffn_down_shexp.weight',(4096,2176),8)
        w['gate_type'] = int(table[prefix+'ffn_gate_exps.weight'].tensor_type)
        w['down_type'] = int(table[prefix+'ffn_down_exps.weight'].tensor_type)
        assert w['gate_type'] in [12,13] and w['down_type'] in [13,14]
        w['gate_row'] = 2304 if w['gate_type']==12 else 2816
        w['down_row'] = 1408 if w['down_type']==13 else 1680
    layers.append(w)
    print('independent weights prepared',layer,flush=True)
# Raw tensor domains contain all288 IDs. Each frame/layer overwrites all rows
# selected by its ORIGINAL router before either unchanged vector operator.
gate_pool = torch.empty(288*4096*2816,device='cuda',dtype=torch.uint8)
down_pool = torch.empty(288*4096*1680,device='cuda',dtype=torch.uint8)
workspace = torch.empty(394*1024*1024,device='cuda',dtype=torch.uint8)
original_ids = [0,154879,1,713,32767,100000,42,8192,12345,17,154878,256]
original_ids += [(j*7919+j*j+713)%154880 for j in range(12,sum(schedule))]
if input_ids is not None: original_ids = input_ids
source_hashes = {}
def experts(layer,w,ids):
    gu = gate_pool[:288*4096*w['gate_row']].view(288,4096,w['gate_row'])
    down = down_pool[:288*4096*w['down_row']].view(288,4096,w['down_row'])
    selected = sorted(set(ids.cpu().reshape(-1).tolist()))
    for role,dest,shape in [('gate',gu[:,:2048],(288,2048,w['gate_row'])),('up',gu[:,2048:],(288,2048,w['gate_row'])),('down',down,(288,4096,w['down_row']))]:
        name = f'blk.{layer}.ffn_{role}_exps.weight'
        raw = table[name].data.reshape(shape)
        for expert in selected:
            data = np.array(raw[expert],copy=True)
            digest = hashlib.sha256(data.tobytes()).hexdigest()
            key = f'{layer}-{role}-{expert}'
            assert source_hashes.setdefault(key,digest)==digest
            dest[expert].copy_(torch.from_numpy(data))
    return gu,down

def routed_matrix(inp,weight,ids,typ,rows,top_k,total):
    # Preserve all token/slot geometry and unchanged original matrix math;
    # partition ONLY the expert domain to avoid original expert-ID>255 bug.
    result = torch.zeros((total*top_k,rows),device='cuda',dtype=inp.dtype)
    for lo,hi in [(0,256),(256,288)]:
        mask = (ids>=lo)&(ids<hi)
        local = torch.where(mask,ids-lo,-torch.ones_like(ids)).contiguous()
        block = ggops.ggml_moe_get_block_size(typ)
        si,ei,padded = moe_align_block_size(local,block,hi-lo,ignore_invalid_experts=True)
        part = ggops.ggml_moe_a8(inp,weight[lo:hi].contiguous(),si,ei,padded,typ,rows,top_k,total)
        result = torch.where(mask.reshape(-1,1),part,result)
    return result

def select_indices(cache,fp8,head_weights,positions,position,label,phase,layer):
    nt = positions.numel()
    nk = (position+nt)//4
    ends = (positions+1)//4
    pools = torch.full((nt,512),-1,device='cuda',dtype=torch.int32)
    raw = torch.full((nt,2051),-1,device='cuda',dtype=torch.int32)
    if position+nt <= 2048:
        _fill_causal_indices(raw,positions.long())
        ids = torch.arange(512,device='cuda',dtype=torch.int32)[None,:].expand(nt,-1)
        ordered = torch.where(ids<ends[:,None],ids,-1).contiguous()
        return raw,dict(pools=ordered,selection=raw),raw
    assert not poolops.mqa_logits_use_mx_scales()
    keys = torch.empty((nk,128),device='cuda',dtype=torch.float8_e4m3fn)
    scale_bytes = torch.empty((nk,4),device='cuda',dtype=torch.uint8)
    table = torch.arange(index_blocks,device='cuda',dtype=torch.int32)[None,:]
    custom.cp_gather_indexer_k_quant_cache(cache['index'],keys,scale_bytes,table,torch.tensor([0,nk],device='cuda',dtype=torch.int32))
    scores0 = fp8_fp4_mqa_logits((fp8.reshape(nt,32,128),None),(keys,scale_bytes.view(torch.float32).reshape(nk)),head_weights,torch.zeros_like(ends),ends,nk)
    eligible = torch.arange(nk,device='cuda')[None,:] < ends[:,None]
    scores = torch.where(eligible,scores0[:,:nk],-torch.inf).contiguous()
    assert bool(torch.isfinite(scores[eligible]).all())
    if nt == 1:
        backend = decode_topk.resolve_backend(scores,512,nt)
        decode_topk(scores,ends[:,None],1,pools,512,nk)
    else:
        backend = 'prefill-per-row'
        torch.ops._C.top_k_per_row_prefill(scores,torch.zeros_like(ends),ends,pools,nt,nk,1,512)
    # Independent deterministic policy: score descending, lower logical ID at
    # ties. Validate ORIGINAL raw choices against exact top-k inequalities;
    # retain every raw permutation independently, rather than rewriting it.
    count = ends.clamp_max(512)
    rank = torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:512]
    expected = torch.where(torch.arange(512,device='cuda')[None,:]<count[:,None],rank,-1)
    sorted_raw = pools.sort(dim=-1).values
    assert bool(((pools>=0).sum(dim=-1)==count).all())
    assert not bool(((sorted_raw[:,1:]==sorted_raw[:,:-1])&(sorted_raw[:,1:]>=0)).any())
    assert bool(((pools<ends[:,None])|(pools<0)).all())
    cutoff = scores.gather(1,rank.gather(1,(count-1).clamp_min(0).long()[:,None]))
    chosen = scores.gather(1,pools.clamp_min(0).long())
    assert bool(((chosen>=cutoff)|(pools<0)).all())
    marker = torch.zeros((nt,nk+1),device='cuda',dtype=torch.bool)
    marker.scatter_(1,torch.where(pools>=0,pools,nk).long(),True)
    assert bool((marker[:,:nk]|~(eligible&(scores>cutoff))).all())
    ordered = torch.where(expected<0,2147483647,expected).sort(dim=-1).values
    ordered = torch.where(ordered==2147483647,-1,ordered).int()
    raw = poolops.expand_pools_and_append_tail(pools.long(),positions+1,4)
    canonical = poolops.expand_pools_and_append_tail(ordered.long(),positions+1,4)
    tied = (eligible&(scores==cutoff)).sum(dim=-1)>1
    mismatched = (pools.sort(dim=-1).values!=ordered.sort(dim=-1).values).any(dim=-1)
    assert not bool((mismatched&~tied).any())
    dest = a.out/f'raw-selector-{label}-{phase}'
    dest.mkdir(exist_ok=True)
    selection_audit.append(dict(label=label,phase=phase,layer=layer,backend=backend,tied_cutoff_rows=int(tied.sum()),raw_set_difference_rows=int(mismatched.sum()),raw_pools_sha256=save(dest/f'layer{layer}-pools.bin',pools),raw_selection_sha256=save(dest/f'layer{layer}-selection.bin',raw),canonical_selection_sha256=save(dest/f'layer{layer}-canonical.bin',canonical)))
    selected = raw if a.sparse_order=='original' else canonical
    return selected,dict(scores=scores,pools=ordered,selection=selected),raw

def mla_result(query,latent,selection,w,nt):
    physical = torch.full((nt,1,2176),-1,device='cuda',dtype=torch.int32)
    physical[:,0,:2051] = selection
    context,lse = attention(query=query[:,None],kv_cache=latent[:,None],workspace_buffer=workspace,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=physical,seq_lens=None,max_seq_len=2176,sparse_mla_top_k=2176,bmm1_scale=.0625,bmm2_scale=1.,kv_scale_format='arbitrary_fp32',return_lse=True)
    values = torch.empty((nt,64,256),device='cuda',dtype=torch.bfloat16)
    torch.bmm(context[:,0].transpose(0,1),w['value'].transpose(1,2),out=values.transpose(0,1))
    return linear(values.reshape(nt,16384),w['out'],8)

def run(label):
    histories = {l:torch.zeros((2,24576,3),device='cuda',dtype=torch.bfloat16) for l in range(45) if l%4!=3}
    states = {l:torch.zeros((1,64,128,128),device='cuda',dtype=torch.float32) for l in histories}
    caches = {l:dict(latent=torch.zeros((latent_blocks,64,656),device='cuda',dtype=torch.uint8),index=torch.zeros((index_blocks,64,132),device='cuda',dtype=torch.uint8),tail=torch.zeros((1,2,4,128),device='cuda',dtype=torch.bfloat16)) for l in range(3,45,4)}
    position = 0
    results = []
    token = None
    for phase in range(frames):
        prompt = phase < len(schedule)
        ids = torch.tensor(original_ids[position:position+schedule[phase]] if prompt else [token],device='cuda',dtype=torch.int32)
        nt = ids.numel()
        residual = hc_expand(_apply_gguf_embedding(ids.long(),embedding,8,4096,torch.bfloat16),4)
        previous = None
        layer_hashes = []
        for layer,w in enumerate(layers):
            selector_values = {}
            if previous is None:
                post,comb,x = mhc_pre_tilelang(residual,*w['attn'][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=w['attn'][3],norm_eps=1e-5)
                r = residual
            else:
                branch,r0,p0,c0 = previous
                r,post,comb,x = mhc_fused_post_pre_tilelang(branch,r0,p0,c0,*w['attn'][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=w['attn'][3],norm_eps=1e-5)
            if layer % 4 != 3:
                projected = linear(x.contiguous(),w['merged'],8)
                qkv,b,fa,ga = projected.split([24576,64,128,128],-1)
                g1 = linear(fa.contiguous(),w['fb'],8)
                g2 = linear(ga.contiguous(),w['gb'],8)
                idx = torch.tensor([1],device='cuda',dtype=torch.int32)
                hist = histories[layer]
                if nt == 1:
                    conv = causal_conv1d_update(qkv.contiguous().clone(),hist,w['cw'],activation='silu',conv_state_indices=idx)
                else:
                    conv = causal_conv1d_fn(qkv.T,w['cw'],None,hist,torch.tensor([0,nt],device='cuda',dtype=torch.int32),cache_indices=idx,has_initial_state=torch.tensor([True],device='cuda')).T.contiguous()
                q,k,v = [z.reshape(1,nt,64,128) for z in conv.split(8192,-1)]
                gate = (-w['a'][None,None,:,None])*(g1.reshape(1,nt,64,128).float()+w['bias'])
                core,state = fused_recurrent_kda(q,k,v,gate,b.contiguous().reshape(1,nt,64),scale=float(np.float32(128**-.5)),initial_state=states[layer],inplace_final_state=False,use_qk_l2norm_in_kernel=True,sigmoid_beta=True,a_log=torch.zeros_like(w['a']),g_bias=torch.zeros_like(w['bias']),compute_gate=True,lower_bound=-5.)
                states[layer] = state[-1:].clone()
                gated = rms_norm_gated(core.clone(),g2.reshape(nt,64,128),w['norm'],None,activation='sigmoid',eps=1e-5)
                branch = linear(gated.reshape(nt,8192).contiguous(),w['out'],8)
            else:
                cache = caches[layer]
                qr0 = linear(x,w['qa'],8)
                qr = torch.empty_like(qr0)
                custom.rms_norm(qr,qr0,w['qn'],1e-5)
                q = linear(qr,w['qb'],8).reshape(nt,64,256)
                query = torch.empty((nt,64,512),device='cuda',dtype=torch.bfloat16)
                torch.bmm(q.transpose(0,1),w['key'].transpose(1,2),out=query.transpose(0,1))
                kr = linear(x,w['kva'],8)
                kv = torch.empty_like(kr)
                custom.rms_norm(kv,kr,w['kn'],1e-5)
                pos = torch.arange(position,position+nt,device='cuda',dtype=torch.int32)
                custom.concat_and_cache_mla(kv,torch.empty((nt,0),device='cuda',dtype=torch.bfloat16),cache['latent'],pos.long(),'fp8_ds_mla',torch.ones(1,device='cuda'))
                ibq = linear(qr,w['iq'],8)
                fp8,scales = poolops.fwht128_quant_fp8(ibq.reshape(-1,128))
                ikn = _fused_indexer_k_norm(linear(x,w['ik'],8),w['inn'],w['ibias'],128,1e-6)
                ig = linear(x,w['ig'],8)
                ih = x.float()@w['ih'].T
                head_weights = _fused_indexer_weight_scale(ih,scales.reshape(nt,32,1),.015625)
                loc = torch.where((pos+1)%4==0,pos//4,-1)
                poolops.kpool_decode_update_and_maybe_write_cache_batched(cache['index'],cache['tail'],(pos%4)[None],ikn[None],ig[None],w['ape'],loc[None],pos[None],4)
                selection,selector_values,raw_selection = select_indices(cache,fp8,head_weights,pos,position,label,phase,layer)
                branch = mla_result(query,cache['latent'],selection,w,nt)
                if a.raw_attention_audit and position+nt>2048:
                    raw_branch = mla_result(query,cache['latent'],raw_selection,w,nt)
                    dest = a.out/f'raw-selector-{label}-{phase}'
                    raw_sha = save(dest/f'layer{layer}-attention.bin',raw_branch)
                    canonical_sha = save(dest/f'layer{layer}-selected-attention.bin',branch)
                    selection_audit[-1].update(raw_attention_sha256=raw_sha,selected_attention_sha256=canonical_sha,attention_bit_mismatch_words=int((raw_branch.view(torch.uint8)!=branch.view(torch.uint8)).reshape(-1,2).any(dim=-1).sum()))
            fr,fp,fc,fx = mhc_fused_post_pre_tilelang(branch,r,post,comb,*w['ffn'][:3],1e-5,1e-6,1e-6,2.,20,norm_weight=w['ffn'][3],norm_eps=1e-5)
            if layer < 3:
                gu = linear(fx,w['gu'],8)
                act = torch.empty((nt,12288),device='cuda',dtype=torch.bfloat16)
                torch.ops._C.silu_and_mul_with_clamp(act,gu,10.,1.,0.)
                result = linear(act,w['down'],8)
            else:
                logits = fx.float()@w['router'].T
                weights,route_ids = fused_grouped_topk(fx,logits,8,True,w['route_bias'],1,1,'sigmoid',2.5)
                sgu = linear(fx,w['sgu'],8)
                sact = torch.empty((nt,2048),device='cuda',dtype=torch.bfloat16)
                torch.ops._C.silu_and_mul_with_clamp(sact,sgu,10.,1.,0.)
                shared = linear(sact,w['sd'],8)
                raw_gu,raw_down = experts(layer,w,route_ids)
                rgu = ggops.ggml_moe_a8_vec(fx,raw_gu,route_ids,8,w['gate_type'],4096,nt) if nt<=64 else routed_matrix(fx,raw_gu,route_ids,w['gate_type'],4096,8,nt)
                ract = torch.empty((nt*8,2048),device='cuda',dtype=torch.bfloat16)
                torch.ops._C.silu_and_mul_with_clamp(ract,rgu,10.,1.,0.)
                rdown = ggops.ggml_moe_a8_vec(ract,raw_down,route_ids,1,w['down_type'],4096,nt*8) if nt<=64 else routed_matrix(ract,raw_down,route_ids,w['down_type'],4096,1,nt*8)
                weighted = rdown.reshape(nt,8,4096).clone().mul_(weights[:,:,None])
                routed = torch.empty_like(fx)
                ggops.moe_sum(weighted,routed)
                result = routed+shared
            previous = result,fr,fp,fc
            dest = a.out/f'{label}-{phase}'
            dest.mkdir(exist_ok=True)
            values = dict(out=result,residual=fr,post=fp,comb=fc)
            if a.index_diagnostics:
                values.update(selector_values)
            if a.states:
                if layer % 4 != 3:
                    values.update(history=histories[layer][1].T.contiguous(),state=states[layer])
                else:
                    values.update(latent=caches[layer]['latent'],index=caches[layer]['index'],tail=caches[layer]['tail'])
            layer_hashes.append({name:save(dest/f'layer{layer}-{name}.bin',value) for name,value in values.items()})
            print(label,'phase',phase,'layer',layer,'completed',flush=True)
        completed = mhc_post_tilelang(*previous)
        collapsed = hc_contract(completed,4)
        normalized = torch.empty_like(collapsed)
        custom.rms_norm(normalized,collapsed,gamma,1e-5)
        logits = linear(normalized[-1:],head,8)
        token = logits.float().argmax(dim=-1).item()
        record = dict(nt=nt,position=position,ids=ids.cpu().tolist(),greedy=token,layers=layer_hashes,logits=save(a.out/f'{label}-{phase}-logits.bin',logits))
        results.append(record)
        position += nt
        print(label,'phase',phase,'greedy',token,flush=True)
    torch.cuda.synchronize()
    return results
first = run('first')
repeat = run('repeat')
if a.sparse_order=='chronological' or sum(schedule)+a.decode_frames<=2048:
    assert first == repeat, 'independent original repeat mismatch'
manifest = dict(records=first,weights=weight_hashes,selected_expert_projections=source_hashes,original_extension_sha256=hashlib.sha256(Path(ggops._C_gguf.__file__).read_bytes()).hexdigest(),profile=__doc__,sources={str(Path(inspect.getfile(fn))):hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest() for fn in [linear,_apply_gguf_embedding,mhc_pre_tilelang,mhc_fused_post_pre_tilelang,mhc_post_tilelang,fused_recurrent_kda,rms_norm_gated,experts,routed_matrix,select_indices,mla_result,fp8_fp4_mqa_logits,attention]})
if a.schedule or a.states or a.input_ids or a.decode_frames != 1:
    manifest.update(prompt_schedule=schedule,capture_states=a.states,decode_frames=a.decode_frames,capacity=capacity,latent_blocks=latent_blocks,index_blocks=index_blocks,expert_matrix_profile='Original unchanged matrices, expert domain partitioned0..255/256..287; original token geometry preserved and outputs bit-selected without arithmetic merging.')
if a.input_ids:
    manifest.update(input_ids=original_ids,input_ids_sha256=hashlib.sha256(a.input_ids.read_bytes()).hexdigest())
manifest.update(vllm_stable_extension_sha256=hashlib.sha256((Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so').read_bytes()).hexdigest(),pool_ops_source_sha256=hashlib.sha256(Path(poolops.__file__).read_bytes()).hexdigest(),indexer_topk_source_sha256=hashlib.sha256(Path(inspect.getfile(inspect.unwrap(get_indexer_topk))).read_bytes()).hexdigest(),sm120_logits_source_sha256=hashlib.sha256((Path(vllm.__file__).parent/'third_party/deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh').read_bytes()).hexdigest(),repeat_records=repeat,sparse_order=a.sparse_order,index_diagnostics=a.index_diagnostics,selection_audit=selection_audit,raw_reference_repeat_identical=first==repeat)
data=(json.dumps(manifest,indent=2)+'\n').encode()
path=a.out/'reference.json'
if path.exists():
    assert path.read_bytes()==data
else:
    path.write_bytes(data)
