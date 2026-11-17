"""Actual 40-layer source forward: fixed IDs, compressed pools and F16 head.

Matched cuBLAS source environment; immutable argv[1] output directory.
No candidate engine code, synthetic branches or generated expected logits.
"""
from pathlib import Path
import hashlib
import os

forward_parent=Path(__file__).with_name('reference_native_block.py')
assert hashlib.sha256(forward_parent.read_bytes()).hexdigest()=='223c71b7c44d661f21df0e0c4cfa587e49fd713a1688d71d33d9e67c953281e8'
exec(compile(forward_parent.read_text().split("with safe_open(str(root/index['embed.weight'])")[0],str(forward_parent),'exec'))
cfg=json.loads((root/'config.json').read_text())['text_config']
assert cfg['num_hidden_layers']==40 and cfg['sliding_window']==128
ratios=cfg['compress_ratios'][:40];owners={l:max(s for s in cfg['kv_source_layer_ids']if s<=l)for l in range(40)if ratios[l]}
assert sorted(set(owners.values()))==[2,8,14,20]
rings={};pools={};crings={};position=0
def reset():
    rings.clear();pools.clear();crings.clear()
    for l in range(40):rings[l]=torch.zeros((768,512),dtype=torch.float16,device='cuda')
    for l in sorted(set(owners.values())):
        rate=ratios[l];pools[l]=[torch.zeros((1024//rate,n),dtype=torch.float16,device='cuda')for n in [448,64,128]]
        crings[l]=[torch.zeros((256+rate,512),dtype=torch.float16,device='cuda')for _ in range(2)]

def attention(layer,x):
    R=x.shape[0];pos=position;rate=ratios[layer]or 1
    freq=ans['freqs']['compress'if ratios[layer]else'main'];rope=ans['rope']
    qa=x@ans['matrix'](layer,'wq_a');qres=torch.empty_like(qa)
    assert not norm.reference(qa.data_ptr(),ans['param'](layer,'q_norm').data_ptr(),qres.data_ptr(),R,1280,16,16,0,1,1e-20,0.,1.,stream)
    q=(qres@ans['matrix'](layer,'wq_b')).reshape(R,64,512);kv=(x@ans['matrix'](layer,'wkv')).reshape(R,1,512)
    assert not rope.ref_rope(q.data_ptr(),freq.data_ptr(),None,None,R,64,64,512,448,64,0,pos,0,1e-20,stream)
    assert not rope.ref_rope(kv.data_ptr(),freq.data_ptr(),ans['param'](layer,'kv_norm').data_ptr(),None,R,1,512,512,0,64,448,pos,1,1e-20,stream)
    kv=kv.reshape(R,512);owner=owners.get(layer,-1);T=(pos+R)//rate if owner>=0 else 0
    if layer==owner:
        wcomp=ans['param'](layer,'compressor.wkv',True);wgate=ans['param'](layer,'compressor.wgate',True)if rate>1 else None
        cn=ans['param'](layer,'compressor.norm').half();key_w=ans['param'](layer,'indexer.wk').reshape(512,128);ink=ans['param'](layer,'indexer.k_norm')
        if rate>1 and R<=32:
            ck=torch.empty((R,512),dtype=torch.float16,device='cuda');cg=torch.empty_like(ck)
            assert not ans['dense'].ref_dense(ans['bc'],x.data_ptr(),wcomp.data_ptr(),ck.data_ptr(),R,5120,512,0)
            assert not ans['dense'].ref_dense(ans['bc'],x.data_ptr(),wgate.data_ptr(),cg.data_ptr(),R,5120,512,0)
        else:ck=x@wcomp;cg=x@wgate if rate>1 else ck
        emitted=T-pos//rate;lat=torch.empty((max(emitted,1),512),dtype=torch.float16,device='cuda');ape=torch.zeros((rate,512),device='cuda');rk,rg=crings[owner]
        arr=(ptr*9)(*(t.data_ptr()if t is not None else None for t in [ck,cg,rk,rg,ape,cn,ans['zero'],lat,None]))
        assert not ans['comp'].ref_compress(arr,R,pos,rate,512,64,256+rate,512,1,stream)
        if emitted:
            lat=lat[:emitted];keyraw=lat@key_w;keys=torch.empty_like(keyraw)
            assert not norm.reference(keyraw.data_ptr(),ink.data_ptr(),keys.data_ptr(),emitted,128,16,16,0,1,1e-20,0.,1.,stream)
            positions=(torch.arange(emitted,device='cuda',dtype=torch.int32)+pos//rate)*rate
            assert not rope.ref_rope(keys.data_ptr(),ans['freqs']['compress'].data_ptr(),None,positions.data_ptr(),emitted,1,64,128,64,64,0,0,0,1e-20,stream)
            assert not rope.ref_rope(lat.data_ptr(),ans['freqs']['compress'].data_ptr(),None,positions.data_ptr(),emitted,1,64,512,448,64,0,0,0,1e-20,stream)
            pools[owner][0][pos//rate:T]=lat[:,:448];pools[owner][1][pos//rate:T]=lat[:,448:];pools[owner][2][pos//rate:T]=keys
    pc,pr_,_=pools[owner if owner>=0 else 2]
    grouped=ans['g']['dsa_attn'](q,pc,pr_,ans['bt'],sinks=parameter(f'layers.{layer}.attn.attn_sink'),ring=rings[layer],kv_chunk=kv,
        win_len=128,win_floor=max(0,pos-127),ring_beg=0,indices=None,k_len=0,pool_len=T,q_pos0=pos,
        compress_rate=rate,derot_inv_freq=-freq,groups=8,out=torch.empty((8,R,4096),dtype=torch.float16,device='cuda'),page_size=256//rate)
    wa=ans['matrix'](layer,'wo_a');Linear=ans['Linear'];obj=ans['types'].SimpleNamespace(o_groups=8,woa_multi_ready=True,wo_a_multi=None,
        wo_a=[Linear(4096,1024)for _ in range(8)],wo_b=Linear(8192,5120),out_dtype=torch.float16)
    for i in range(8):obj.wo_a[i].weight=wa[:,i*1024:(i+1)*1024].contiguous()
    obj.wo_b.weight=ans['matrix'](layer,'wo_b');delta=ans['project'](obj,grouped.unsqueeze(1),{},torch.float32).reshape(R,5120)
    rings[layer][pos:pos+R]=kv
    return delta

with safe_open(str(root/index['embed.weight']),framework='pt',device='cpu')as reader:embedding=reader.get_tensor('embed.weight')
raw=torch.frombuffer(bytearray((hashed/'hash-1-1-513-text-tokens.i64le').read_bytes()),dtype=torch.int64)
raw=raw.clone();extended=os.environ.get('BN_REFERENCE_EXTENDED')=='1'
cases=[];frames=[]
specs=[('single',1,0,True),('prefill',7,0,True),('decode',1,7,False)]
if extended:specs+=[('chunk',7,8,False)]+[(f'greedy-{i}',1,15+i,False)for i in range(5)]
for name,R,pos,fresh in specs:
    if fresh:reset()
    if name.startswith('greedy-'):raw[pos]=frames[-1]['argmax_last']
    position=pos;tokens=raw[pos:pos+R]
    history=torch.cat((torch.full((3,),2,dtype=torch.int64),mapping[raw]))[pos:pos+R+3].reshape(1,-1)
    x=embedding[tokens].float().unsqueeze(1).expand(R,4,5120).contiguous().cuda();pre=None
    frame=dict(name=name,rows=R,position=pos,reset=fresh,tokens=save(name+'.tokens',tokens),cases=[])
    if name.startswith('greedy-'):frame['generated']=True
    for layer in range(40):
        tag=f'{name}.l{layer}';boundaries={}
        if layer in models:x=engram(layer,x,history,R)
        for kind in ['attn','ffn']:
            fn,base,scale=[parameter(f'layers.{layer}.hc_{kind}_'+s)for s in ['fn','base','scale']]
            part=torch.empty((R,hc.chunks(R,5120),25),device='cuda');post=torch.empty((R,4),device='cuda');comb=torch.empty((R,4,4),device='cuda');collapsed=torch.empty((R,5120),device='cuda');own=torch.empty((R,4),device='cuda');fn=fn.half()
            assert not hc.mix(x.data_ptr(),fn.data_ptr(),base.data_ptr(),scale.data_ptr(),part.data_ptr(),post.data_ptr(),comb.data_ptr(),collapsed.data_ptr(),pre.data_ptr()if pre is not None else None,own.data_ptr(),R,5120,1,stream)
            normalized=normalize(collapsed,parameter(f'layers.{layer}.{kind}_norm.weight'));boundaries[kind+'_normalized']=save(tag+'.'+kind+'.normalized',normalized)
            if kind=='attn':y=attention(layer,normalized)
            else:
                y,ids,rw=ffn(layer,normalized);boundaries['ids']=save(tag+'.ids',ids);boundaries['weights']=save(tag+'.weights',rw)
            boundaries[kind+'_delta']=save(tag+'.'+kind+'.delta',y)
            assert not hc.apply(x.data_ptr(),y.data_ptr(),post.data_ptr(),comb.data_ptr(),R,5120,0,stream)
            pre=own;boundaries[kind+'_streams']=save(tag+'.'+kind+'.streams',x)
        assert torch.isfinite(x).all(),(name,layer)
        frame['cases'].append(len(cases));cases.append(dict(frame=name,rows=R,layer=layer,tokens=frame['tokens'],outputs=boundaries));print('CAPTURED',name,layer,flush=True)
    collapsed=torch.matmul(pre.unsqueeze(-2),x).squeeze(-2);normalized=normalize(collapsed,parameter('norm.weight'))
    frame['collapsed']=save(name+'.head-collapse',collapsed);frame['normalized']=save(name+'.head-norm',normalized)
    m=parameters['head.weight'];b=(pr/(m['tag']+'.expected')).read_bytes();assert sha(b)==m['expected_sha256'];weight=torch.frombuffer(bytearray(b),dtype=torch.float16).reshape(5120,129280).cuda();del b
    obj=ans['types'].SimpleNamespace(weight=weight,out_features=129280,out_dtype=None,bias=None,_pinned_store=None)
    full=ans['linear'](obj,normalized,{});last=ans['linear'](obj,normalized[-1:].contiguous(),{})
    assert full.dtype==last.dtype==torch.float16 and torch.isfinite(full).all()and torch.isfinite(last).all()
    frame['logits']=save(name+'.logits',full);frame['last_logits']=save(name+'.last-logits',last)
    frame['argmax_all_last']=int(full[-1].argmax());frame['argmax_last']=int(last[-1].argmax());frames.append(frame)
    del obj,weight,full,last
libraries={}
for line in Path('/proc/self/maps').read_text().splitlines():
    f=line.split()[-1]
    if 'libcublas'in f and f.startswith('/')and f not in libraries:libraries[f]=sha(Path(f).read_bytes())
scope='Actual 40-layer fixed-ID full forward, all F16 logits and source last_tokens_only=1 logits; short fresh/resumed context, no long indexed selection or generated continuation'
if extended:scope='Actual 40-layer full forward, resumed/chunked short prefill and five autoregressive source greedy tokens; all F16 and last-only logits; no long indexed selection'
emit('reference.json',(json.dumps(dict(scope=scope,frames=frames,cases=cases,files=files,libraries=libraries),indent=2)+'\n').encode())
proof=dict(parent_sha256=sha(forward_parent.read_bytes()),experts=expert_provenance,attention_weights=ans['weights'],libraries=libraries,
    sources={str(f):sha(f.read_bytes())for f in [forward_parent,hcfile,attfile,decfile,src/'model/model_ls.py',src/'modules/linear.py',src/'modules/quant/fp16.py',src/'architecture/deepseek_v41.py']},
    binaries={str(Path(v._name)):sha(Path(v._name).read_bytes())for v in [hc,norm,ans['rope'],ans['comp'],*dn['libs'].values()]},
    head=dict(file=str(pr/(m['tag']+'.expected')),sha256=m['expected_sha256']))
emit('provenance.json',(json.dumps(proof,indent=2)+'\n').encode());print('CAPTURED full forward',len(cases),flush=True)
